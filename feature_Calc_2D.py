# Author: Milton Candela (https://github.com/milkbacon)
# Date: October 2022

# The following code creates a filtered CSV, according to the importance of the features to the target variable using
# RandomForestRegressor, even though no sampling technique was employed to solve the class imbalance, this was due to
# the fact that a regression approach was taken rather than a classification. The dataset is saved as a CSV file inside
# the "processed" folder, with the top 100 features and their respective information variables.

import cv2
import numpy as np
import pandas as pd
from json import loads
import pyrealsense2 as rs
from os.path import exists
from statistics import mean
from datetime import datetime

'''
- altura (punto maximo a la camara), tomar en cuenta el de profundidad al normalizar
- del punto medio: pertenencia a la parte derecha e izquierda (lo distal al punto medio) para ritmo (frecuencia)
- profundidad para amplitud (0.5-5 m)
- pixeles ocupados (predicción persona y ratio con area total) [0-100]
- cantidad de movimiento (cantidad de pixeles que cambian de un momento a otro)

amount of movement (OpenFrameworks) https://forum.openframeworks.cc/t/amount-of-movement/126
'''

# si feature 1: requiere de la cabeza 0 [2D]
# si feature 2: requiere de cadera 8 o pecho 1 [2D]
# si feature 3: requiere de cadera 8 o pecho 1 [3D]
# no feature 4: predicción de persona con YOLOv4
# no feature 5: https://forum.openframeworks.cc/t/amount-of-movement/126


def calc_features(l_p):
    # l_p List of points x y
    # l_z List of z coordinate

    l_t_p = []  # List of transformed points
    for i in range(len(l_p)):
        l_t_p.append(np.array([l_p[i][0], l_p[i][1]]))

    # l_t_p = (parts, dimensions) = (dim(BODY_PARTS), [x, y, z])
    # Head = 0, Chest = 1, Hip = 2

    feature_1 = int(desired_dim[1] - l_t_p[0][1])  # altura de cabeza con punto máximo: resolución_y - cabeza_y
    feature_2 = int(abs(desired_dim[0]/2 - l_t_p[1][0]))  # punto distal del punto medio (izquierda o derecha): |resolución_x/2 - pecho_x|

    client.send_message("/height", feature_1)
    client.send_message("/filter", feature_2)

    #df = pd.DataFrame({'Height': feature_1, 'MDistance': feature_2})
    #df.index = datetime.strftime(datetime.now(), "%d/%m/%Y %H:%M:%S")
    #df.to_csv('data/OP_Feat.csv', mode='a', header=False)


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
        frame = pd.Series(data=key_points_data, index=BODY_PARTS.keys())
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
    l_p = resize(l_p, (cam_resolution, depth_resolution))
    return True, l_p


def resize(l_p, dims):
    # l_p = list of x y
    # dims = tuple of tuples (actual_dim, desired_dim) XY

    actual_dim, desired_dim = dims[0], dims[1]
    return [(int(p[0] * desired_dim[0] / actual_dim[0]), int(p[1] * desired_dim[1] / actual_dim[1])) for p in l_p]


# # # Initial settings # # #
# Initialize CSV
#df = pd.DataFrame({'Height': 0, 'MDistance': 0, 'Depth': 0})
#df.to_csv('data/OP_Feat.csv')

# Initialize resolutions
depth_resolution = (1024, 768)
cam_resolution = (1920, 1080)
desired_dim = depth_resolution

# Initialize variables
frame_id = 1
t_radius = 100
skip_frames = 0
refresh_rate = 0.01
BODY_PARTS = {'Head': 0, 'Chest': 1, 'Hip': 8}
camino = 'C:/Users/Milton/Documents/openpose/output_jsons'

from pythonosc import udp_client
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="10.22.235.48", help="The ip of the OSC server")
parser.add_argument("--port", type=int, default=5000, help="The port the OSC server is listening on")
args = parser.parse_args()

client = udp_client.SimpleUDPClient(args.ip, args.port)

while True:
    n = [str(x) for x in [0] * len(str(100000000000 // (frame_id + 1)))]
    file = camino + '/' + ''.join(n) + str(frame_id) + '_keypoints.json'
    print(frame_id)
    if exists(file):

        c_k, l_p = obtain_keypoints(file)  # List of points x y
        if c_k:
            calc_features(l_p)
            client.send_message("/amhere", 1)
        else:
            client.send_message("/amhere", 0)
        frame_id += 1 + skip_frames
    else:
        while exists(file) is not True:
            pass
