import numpy as np
import pyrealsense2 as rs
import cv2

from statistics import mean

pipeline = rs.pipeline()
config = rs.config()

# Depth
# L515: 1024 × 768
# 640 x 480
config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)

# RGB
# L515: 1920 x 1080
# L500: 960 x 540
# 640 x 480
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

print('LOADING YOLO')
folder = 'realSense'
net = cv2.dnn.readNet("{}/Models/People/yolov4.cfg".format(folder), "{}/Models/People/yolov4.weights".format(folder))
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

output_layers = [net.getLayerNames()[j - 1] for j in [i for i in net.getUnconnectedOutLayers()]]
print("YOLO LOADED")

desired_dim = (640, 480)
width, height = desired_dim[0], desired_dim[1]

while True:
    frames = pipeline.wait_for_frames()

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        continue

    color_image = cv2.resize(np.asanyarray(color_frame.get_data()), dsize=desired_dim, interpolation=cv2.INTER_AREA)
    depth_image = cv2.resize(np.asanyarray(depth_frame.get_data()), dsize=desired_dim, interpolation=cv2.INTER_AREA)

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    blob = cv2.dnn.blobFromImage(color_image, 1 / 255.0, (320, 320), swapRB=True, crop=False)

    # Detecting objects
    net.setInput(blob)
    outs = net.forward(output_layers)

    distancias = []
    for out in outs:
        for detection in out:
            scores = detection[:5]
            class_id = np.argmax(scores)
            if class_id == 0 and scores[class_id] > 0.9:  # 0 is the "person" class id
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                xmid = int(x + w/2)
                ymid = int(y + h/2)

                d = depth_frame.get_distance(xmid, ymid)
                if d != 0:
                    distancias.append(d)

    if len(distancias) != 0:
        cv2.putText(color_image, str(round(mean(distancias), 2)), (width - 100, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

    images = np.hstack((color_image, depth_colormap))
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', images)
    cv2.waitKey(1)

pipeline.stop()
