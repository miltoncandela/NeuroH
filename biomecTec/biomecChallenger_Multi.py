import pyrealsense2 as rs
import cv2
from json import loads
from os.path import exists
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series
from statistics import mean
from multiprocessing import Process
from time import sleep

pipeline = rs.pipeline()
config = rs.config()

# Depth
# L515: 1024 × 768
# 640 x 480
depth_resolution = (1024, 768)
config.enable_stream(rs.stream.depth, depth_resolution[0], depth_resolution[1], rs.format.z16, 30)

# Start streaming
pipeline.start(config)


def obtain_depth(l_p):
    # p = list of tuples (x, y)

    l_p = resize(l_p, (cam_resolution, depth_resolution))

    # print(l_p)

    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()

    if not depth_frame:
        return False, 0

    # depth_image = cv2.resize(np.asanyarray(depth_frame.get_data()), dsize=desired_dim, interpolation=cv2.INTER_AREA)
    #depth_image = np.asanyarray(depth_frame.get_data())
    #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    #cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    #cv2.imshow('RealSense', depth_colormap)
    #cv2.waitKey(1)

    def mean_rectangle_dist(p, r=25):
        # r=100 Circulo de 100 pixeles de radio
        # area =
        # h = round(np.sqrt(r^2 + r^2))# hiptenusa del circulo para esquinas
        # rect_coords = []
        # Coordenadas del rectángulo
        # Esquina superior izquierda: (p[0] - r, p[1] + r)
        # Esquina superior derecha: (p[0] + r, p[1] + r)
        # Esquina inferior izquierda: (p[0] - r, p[1] - r)
        # Esquina inferior derecho: (p[0] + r, p[1] - r)

        distancias = []
        p_inis = [p - r if (p - r) > 0 else 0 for p in p]
        p_fins = [p[i] + r if (p[i] + r) < depth_resolution[i] else depth_resolution[i] for i in range(len(p))]

        for i in range(p_inis[0], p_fins[0]):  # From i_ini to i_fin (x-axis)
            for j in range(p_inis[1], p_fins[1]):  # From j_ini to j_fin (y-axis)
                distancia = depth_frame.get_distance(i, j)
                if distancia != 0:
                    distancias.append(distancia)

        if len(distancias) > 0:
            return mean(distancias)
        else:
            return 0

    l_dis = []
    for p in l_p:
        dis = mean_rectangle_dist(p)
        if dis == 0:
            return False, 0
        l_dis.append(dis)

    # return True, [depth_frame.get_distance(p[0], p[1]) for p in l_p]
    return True, l_dis


BODY_PARTS = {'MHip': 8, 'RHip': 9, 'RKnee': 10, 'RAnkle': 11, 'LHip' : 12, 'LKnee': 13, 'LAnkle': 14}


def obtain_keypoints(file_path):
    # frame_id = int

    file_data = [loads(line) for line in open(file_path)]

    try:
        if len(file_data[0]['people']) <= 0:
            return False, 0
            # key_points_data = np.zeros(len(BODY_PARTS))
        else:
            key_points = file_data[0]['people'][0]['pose_keypoints_2d']
            key_points_data = []
            for body_part in BODY_PARTS.keys():
                joint_coord = key_points[BODY_PARTS[body_part]*3], key_points[BODY_PARTS[body_part]*3 + 1], key_points[BODY_PARTS[body_part]*3 + 2]
                key_points_data.append(joint_coord)

            # for i in range(0, len(key_points), 3):
            #     joint_coord = key_points[i], key_points[i + 1], key_points[i + 2]
            #     key_points_data.append(joint_coord)
        frame = Series(data=key_points_data, index=BODY_PARTS.keys())
    except IndexError:
        return False, 0

    coord_x_vector, coord_y_vector = [], []
    # Additional modifications must be done on the Y axis, because the outputted JSON files from the OpenPose
    # software, are inverse on the Y axis. And so they must be flipped on this axis to see the plot as the video.
    for marker in BODY_PARTS:
        coord_x, coord_y, acc = str(frame[marker]).strip("()\s").replace(' ', '').split(',')
        if float(coord_x) == 0 and float(coord_y) == 0:
            return False, 0
        else:
            coord_x_vector.append(float(coord_x))
            coord_y_vector.append(cam_resolution[1] - float(coord_y) - 1)

    # print(coord_x_vector, coord_y_vector)
    l_p = [(coord_x_vector[i], coord_y_vector[i]) for i in range(len(BODY_PARTS.keys()))]
    return True, l_p


def calc_features(l_p, l_z):
    # l_p List of points x y
    # l_z List of z coordinate

    l_t_p = [] # List of transformed points
    for i in range(len(l_z)):
        l_t_p.append(np.array([l_p[i][0], l_p[i][1], l_z[i]]))

    k = min(BODY_PARTS.values())
    # k = 9

    # 'MHip': 8, 'RHip': 9, 'RKnee': 10, 'RAnkle': 11, 'LHip' : 12, 'LKnee': 13, 'LAnkle': 14

    a_D, b_D, c_D = l_t_p[10 - k], l_t_p[9 - k], l_t_p[11 - k]
    a_I, b_I, c_I = l_t_p[13 - k], l_t_p[12 - k], l_t_p[14 - k]

    def cosine_angle(coords):
        # coords is a list of [a, b, c]
        # a is an array of [x, y, z]
        # so coords is a list of arrays [[x, y, z], [x, y, z], [x, y, z]]

        # a = R = Pierna = Rodilla
        # b = P = Pie
        # c = C = Cadera

        a, b, c = coords

        # La = | R - C | = | a - c |
        # Lb = | R - P | = | a - b |
        # Lc = | C - P | = | c - b |

        La = a - c
        Lb = a - b
        Lc = c - b

        mags = [np.sqrt(L[0]**2 + L[1]**2 + L[2]**2) for L in [La, Lb, Lc]] # List of magnitudes

        # Cosine rule:
        # cos(A) = (b^2 + c^2 - a^2)/(2*b*c)

        upper = mags[0]**2 + mags[1]**2 - mags[2]**2
        bottom = 2*np.multiply(mags[0], mags[1])

        # Returns angle in degres

        return abs(np.arccos(upper/bottom)*(180/np.pi) - 180)

    gamma_D = cosine_angle([a_D, b_D, c_D])
    gamma_I = cosine_angle([a_I, b_I, c_I])

    # 'MHip': 8, 'RHip': 9, 'RKnee': 10, 'RAnkle': 11, 'LHip' : 12, 'LKnee': 13, 'LAnkle': 14

    rodilla_izq, rodilla_der, cadera_media = l_t_p[13 - k][0], l_t_p[10 - k][0], l_t_p[8 - k][0]
    dist_izq = abs(cadera_media[0] - rodilla_izq[0])  # cadera_media_x - rodilla_izq_x
    dist_der = abs(rodilla_der[0] - cadera_media[0])  # rodilla_der_x - cadera_media_x

    # normalizando a la resolución de la cámara, un valor mínimo de 0 es que la distancia sea igual (cada rodilla
    # ensimismada con la cadera media, mientras que un valor máximo de 100 es que cada rodilla se encuentre a una
    # distancia resolución_máxima_y/2 de la cadera media, entonces promedio máximo o 100 es resolución_máxima_y/2
    dist_K = (mean([dist_der, dist_izq])/(depth_resolution[1]/2)) * 100

    # normalizando la distancia de las rodillsa con "intertrochanteric distance"
    cadera_izq, cadera_der = l_t_p[12 - k][0], l_t_p[9 - k][0]
    dist_K = (rodilla_der[0] -)

    return round(gamma_D), round(gamma_I), round(dist_K)


def resize(l_p, dims):
    # l_p = list of x y
    # dims = tuple of tuples (actual_dim, desired_dim) XY

    actual_dim, desired_dim = dims[0], dims[1]
    return [(int(p[0]*desired_dim[0]/actual_dim[0]), int(p[1]*desired_dim[1]/actual_dim[1])) for p in l_p]


cam_resolution = (1920, 1080)


def plot_data(csv_name, refresh_rate=0.01):
    # type_plot could be angle | distance
    # csv_names = ['angleD', 'angleI', 'distaK']

    ph, = plt.plot(-10, -10, marker='o', color='red', linestyle='solid')
    ax = plt.gca()
    ax.set_xlim([0, 2])

    if csv_name[:3] == 'ang':
        ax.set_ylim([0, 90])
        ax.set_ylabel('Ángulo en grados (°)')
    elif csv_name[:3] == 'dis':
        ax.set_ylim([0, 100])
        ax.set_ylabel('Distancia normalizada')

    ax.set_xlabel('Observaciones válidas')
    title_dict = {'angleD': 'Ángulo de flexión de la rodilla derecha',
                  'angleI': 'Ángulo de flexión de la rodilla izquierda',
                  'distaK': 'Distancia de separación de la rodilla'}
    ax.set_title('OpenPose y L515: {}'.format(title_dict[csv_name]))

    valid_samples = 0

    def get_n_samples(name):
        text_name = 'data/samples_' + name + '.txt'
        file = open(text_name, 'r')
        n = file.read()
        file.close()
        return n

    while True:
        n_samples = int(get_n_samples(csv_name))
        if valid_samples != n_samples:
            file = open('data/' + csv_name + '.txt', 'r')
            data = [int(d) for d in file.read().split(', ')]
            file.close()

            ax.set_xlim([0, n_samples])
            ph.set_xdata(range(n_samples))
            ph.set_ydata(data)
            plt.pause(refresh_rate)
            valid_samples = n_samples
        sleep(1)


def get_and_write_data():
    frame_id = 1
    camino = 'C:/Users/Milton/Documents/openpose/output_jsons'
    datos_D, datos_I, datos_K = [], [], []
    skip_frames = 2
    folder = 'data'

    def write_data(l, name):
        file = open('{}/{}.txt'.format(folder, name), 'w')
        file.write(', '.join([str(d) for d in l]))
        file.close()

        file = open('{}/samples_{}.txt'.format(folder, name), 'w')
        file.write(str(len(l)))
        file.close()

    while True:
        n = [str(x) for x in [0] * len(str(100000000000 // (frame_id + 1)))]
        file = camino + '/' + ''.join(n) + str(frame_id) + '_keypoints.json'
        print(frame_id)
        if exists(file):
            c_k, l_p = obtain_keypoints(file)  # List of points x y
            if c_k:
                c_d, l_z = obtain_depth(l_p)  # List of z coordinate
                if c_d:
                    ang_D, ang_I, dis_K = calc_features(l_p, l_z)
                    datos_D.append(ang_D), datos_I.append(ang_I), datos_K.append(dis_K)
                    write_data(datos_D, 'angleD')
                    write_data(datos_I, 'angleI')
                    write_data(datos_K, 'distaK')
            frame_id += 1 + skip_frames
        else:
            while exists(file) is not True:
                pass


if __name__ == '__main__':
    refresh_rate = 0.01
    g_data = Process(target=get_and_write_data)
    p_angD = Process(target=plot_data, args=('angleD', refresh_rate))
    p_angI = Process(target=plot_data, args=('angleI', refresh_rate))
    # p_kneD = Process(target=plot_data, args=('distaK', refresh_rate))

    g_data.start()
    p_angD.start()
    p_angI.start()
    # p_kneD.start()

    g_data.join()
    p_angD.join()
    p_angI.join()
    # p_kneD.join()
