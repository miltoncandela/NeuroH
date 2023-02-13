# Author: Milton Candela (https://github.com/milkbacon)
# Date: September 2021

# The current code reads all JSON files created from OpenPose (https://github.com/CMU-Perceptual-Computing-Lab/openpose)
# and joins them into a single DataFrame, which is then outputted into a CSV. It is worth noting that when the Computer
# Vision (CV) software does not detect a human joint, it defaults its value to 0, on the other hand, if no human is
# detected, the JSON file would be blank for that frame of time, although it would be taken as zeros for each joint.

from pandas import Series
from json import loads
from os.path import exists
import matplotlib.pyplot as plt

BODY_PARTS = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist', 'MidHip', 'RHip',
              'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye', 'LEye', 'REar', 'LEar', 'LBigToe', 'LSmallToe',
              'LHeel', 'RBigToe', 'RSmallToe', 'RHeel']

def plot_coordinate(file_path, resolution):
    """
    This functions takes the name of the file as an input (Student, Taekwondo, Pers1), the original resolution of that
    video, and an optional refresh rate from which the animated plot would be refreshed. It is worth noting that
    matplotlib function is not being used, instead it creates a base plot on where it adds additional markers, the speed
    on which these markers are being added is controlled by the refresh_rate parameter.

    :param string file_path:
    :param tuple resolution:
    """

    file_data = [loads(line) for line in open(file_path)]
    try:
        if len(file_data[0]['people']) <= 0:
            return False, 0, 0
            # key_points_data = np.zeros(len(BODY_PARTS))
        else:
            key_points = file_data[0]['people'][0]['pose_keypoints_2d']
            key_points_data = []
            for i in range(0, len(key_points), 3):
                joint_coord = key_points[i], key_points[i + 1], key_points[i + 2]
                key_points_data.append(joint_coord)
        frame = Series(data=key_points_data, index=BODY_PARTS)
    except IndexError:
        return False, 0, 0

    # The next for loop iterates all the rows on the extracted DataFrame, this rows represent each frame extracted by
    # the OpenPose software. A vector is created because it is also required to repeat this process for each joint
    # in the DataFrame, and so the final vector, which contains all the XY positions of all markers, is plotted.

    coord_x_vector, coord_y_vector = [], []
    # Additional modifications must be done on the Y axis, because the outputted JSON files from the OpenPose
    # software, are inverse on the Y axis. And so they must be flipped on this axis to see the plot as the video.
    for marker in BODY_PARTS:
        coord_x, coord_y, acc = str(frame[marker]).strip("()\s").replace(' ', '').split(',')
        if (float(coord_x) != 0) and (float(coord_y) != 0):
            coord_x_vector.append(float(coord_x))
            coord_y_vector.append(resolution[1] - float(coord_y) - 1)

    return True, coord_x_vector, coord_y_vector


cam_resolution = (1920, 1080)

ph, = plt.plot(-10, -10, marker='o', color='red', linestyle='')
ax = plt.gca()
ax.set_xlim([0, cam_resolution[0]])
ax.set_ylim([0, cam_resolution[1]])
ax.set_xlabel('X coordinate (Pixels)')
ax.set_ylabel('Y coordinate (Pixels)')
ax.set_title('Real-time OpenPose 2D key points with webcam')

frame_id = 1
refresh_rate = 0.01
camino = 'C:/Users/Milton/Documents/openpose/output_jsons'
while True:
    n = [str(x) for x in [0] * len(str(100000000000 // (frame_id + 1)))]
    file = camino + '/' + ''.join(n) + str(frame_id) + '_keypoints.json'
    if exists(file):
        c, coord_x, coord_y = plot_coordinate(file, cam_resolution)
        if c:
            ph.set_xdata(coord_x)
            ph.set_ydata(coord_y)
            plt.pause(refresh_rate)
        frame_id += 1
    else:
        while exists(file) is not True:
            pass

# 1 = 000000000001, n = 11
# 10 = 000000000010, n = 10
# 100 = 000000000100, n = 9
# 1000 = 000000001000, n = 8
# 10000 = 000000010000, n = 7
# 100000 = 000000100000, n = 6
# 1000000 = 000001000000, n = 5
# 10000000 = 000010000000, n = 4
# 100000000 = 000100000000, n = 3
# 1000000000 = 001000000000, n = 2
# 10000000000 = 010000000000, n = 1
# 100000000000 = 100000000000, n = 0
