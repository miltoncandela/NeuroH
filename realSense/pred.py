## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

# USAGE
# python detect_mask_video.py

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import argparse
import imutils
import time
import os

import pyrealsense2 as rs
import numpy as np
import cv2
import math


def calculatedistance(point1, point2):
    return math.sqrt(
        math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2) + math.pow(
            point1[2] - point2[2], 2))


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # prediction
    return (locs, preds)


class Person:
    def __init__(self, xpixel, ypixel, id, color_intrin, depth_scale, x, y, w, h):
        self.xpixel = xpixel
        self.ypixel = ypixel
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.id = id
        self.drawed = False
        depth_pixel = [self.xpixel, self.ypixel]
        udist = depth_frame.get_distance(self.xpixel, self.ypixel)
        self.point = rs.rs2_deproject_pixel_to_point(color_intrin, depth_pixel,
                                                     udist * (depth_scale * 1000))  # Configure depth and color streams


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
                default="Models/Face",
                help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
                default="Models/Mask/mask_detector.model",
                help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
                                "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
faceNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
faceNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

##
depth_sensor = pipeline_profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
##

device_product_line = str(device.get_info(rs.camera_info.product_line))

#config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
#if device_product_line == 'L500':
#    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
#elif device_product_line == 'L515':
#    config.enable_stream(rs.stream.color, 1024, 768, rs.format.bgr8, 30)
#else:
#    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)


# Start streaming
pipeline.start(config)

# Load Yolo
print("LOADING YOLO")
net = cv2.dnn.readNet("Models/People/yolov4.cfg", "Models/People/yolov4.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# save all the names in file o the list classes
classes = []
with open("Models/People/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
# get layers of the network
layer_names = net.getLayerNames()
# Determine the output layer names from the YOLO model
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print("YOLO LOADED")

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # OPENCV
        height, width, channels = color_image.shape

        (locs, preds) = detect_and_predict_mask(color_image, faceNet, maskNet)
        print(preds)

        # USing blob function of opencv to preprocess image
        blob = cv2.dnn.blobFromImage(color_image, 1 / 255.0, (320, 320),
                                     swapRB=True, crop=False)
        # Detecting objects
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # We use NMS function in opencv to perform Non-maximum Suppression
        # we give it score threshold and nms threshold as arguments.
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        colorred = (0, 0, 255)
        colorblue = (0, 255, 0)
        colorgreen = (0, 255, 0)
        colorx = (120, 120, 0)
        persons = []
        currid = 1

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(color_image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(color_image, (startX, startY), (endX, endY), color, 2)

        distances = {}
        for thisperson in range(len(persons)):
            distances[str(persons[thisperson].id)] = []
        #
        for iperson in range(len(persons)):
            for jperson in range(len(persons)):
                person1 = persons[iperson]
                person2 = persons[jperson]
                if person1.id != person2.id:
                    distance = calculatedistance(person1.point, person2.point)
                    distances[str(person1.id)].append(distance)
                    distances[str(person2.id)].append(distance)
        counter = 0
        for iperson in range(len(persons)):
            x = persons[iperson].x
            y = persons[iperson].y
            w = persons[iperson].w
            h = persons[iperson].h
            for idist in distances[str(persons[iperson].id)]:
                if idist < float(1.0):
                    cv2.rectangle(color_image, (x, y), (x + w, y + h), colorred, 2)
                    persons[iperson].drawed = True
                    counter = counter + 1
                    break
            if persons[iperson].drawed == False:
                cv2.rectangle(color_image, (x, y), (x + w, y + h), colorgreen, 2)

        cv2.putText(color_image, 'Distance Violations:' + str(counter), (5, 25), font, 2, colorx, 3)

        # print(distances)
        # print(counter)
        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                             interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()