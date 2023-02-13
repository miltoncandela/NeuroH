# Author: Milton Candela (https://github.com/milkbacon)
# Date: October 2022

# The following code creates a filtered CSV, according to the importance of the features to the target variable using
# RandomForestRegressor, even though no sampling technique was employed to solve the class imbalance, this was due to
# the fact that a regression approach was taken rather than a classification. The dataset is saved as a CSV file inside
# the "processed" folder, with the top 100 features and their respective information variables.

import pandas as pd
from json import loads
import pyautogui as pt
from os.path import exists
from statistics import mean, stdev


def move_hand(p):
    # p list of (x, y)
    pt.moveTo(p[0], cam_resolution[1]-p[1])


def click_hand(l_x, l_y, z=3):
    # l_x list of x coordinates (n = t_frames)
    # l_y list of y coordiantes (n = t_frames)

    means = (mean(l_x), mean(l_y))
    devia = (stdev(l_x), stdev(l_y))

    for i in range(t_frames):
        if (((l_x[i]-means[0])/devia[0]) > z) or (((l_y[i]-means[1])/devia[1]) > z):
            return 0

    pt.click(l_x[-1], l_y[-1])



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
    p = (coord_x_vector[0], coord_y_vector[0])
    return True, p


# # # Initial settings # # #
# Initialize resolutions
cam_resolution = (1920, 1080)

# Initialize variables
frame_id = 1
t_frames = 60  # 60 frames = 2 seconds
refresh_rate = 0.01
BODY_PARTS = {'LHand': 7}
camino = 'C:/Users/Milton/Documents/openpose/output_jsons'
last_frames_x, last_frames_y = [], []

while True:
    n = [str(x) for x in [0] * len(str(100000000000 // (frame_id + 1)))]
    file = camino + '/' + ''.join(n) + str(frame_id) + '_keypoints.json'
    print(frame_id)
    if exists(file):
        c_k, p = obtain_keypoints(file)  # List of points x y
        if c_k:
            move_hand(p)
            #last_frames_x.append(p[0]), last_frames_y.append(p[1])
            #if len(last_frames_x) > t_frames:
            #    last_frames_x.pop(0), last_frames_y.pop(0)
            #    click_hand(last_frames_x, last_frames_y)
        frame_id += 1
    else:
        while exists(file) is not True:
            pass
