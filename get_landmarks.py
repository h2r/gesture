import os
from tkinter import RIGHT
from turtle import left, right
from xml.etree.ElementInclude import include
import cv2
import sys
import math
import pandas as pd
import numpy as np
import mediapipe as mp
# from tqdm import tqdm
import json
# from add_intersection import *

# this script maps skeleton tracking to our collected image data, and save the  
# world landmark coordinates in m to json
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
path = script_dir+'/data' # change file path to image path
# fnum = 9 # can change this to desired file number (just remember to change it in the other files too)

# human measurement defined base on mediapipe pose landmark
NOSE = 0
LEFT_EYE = 2
RIGHT_EYE = 5
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32


data = {"image":[]}

# map skeleton to image
# For static images:
BG_COLOR = (192, 192, 192) # gray
BOUNDARY_SEGMENTATION = False
PLOT_WORLD_LANDMARK = False

landmark_list = ['nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
                 'right_eye_inner', 'right_eye', 'right_eye_outer', 'left_ear',
                 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder',
                 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 
                 'right_wrist', 'left_pinky', 'right_pinky', 'left_index', 
                 'right_index', 'left_thumb', 'right_thumb', 'left_hip', 
                 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
                 'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index']
with mp_pose.Pose(
    static_image_mode=True,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
  for (i, file) in enumerate(os.listdir(path)):
    image_data = {
        "name": [],
        "landmark_3D": [],
        "landmark_2D_normalized": []
    }
    if '.png'in file:
        image = cv2.imread(path+"/"+file)
        h, w, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        lw = results.pose_world_landmarks.landmark
        ll = results.pose_landmarks.landmark
        image_data['name'] = [file]
        for i in range(len(landmark_list)):
           coordinate_3D = {"landmark_name": landmark_list[i],
                            "x": lw[i].x,
                            "y": lw[i].y,
                            "z": lw[i].z}
           coordinate_2D = {"landmark_name": landmark_list[i],
                            "x": ll[i].x,
                            "y": ll[i].y,
                            "z": ll[i].z}
           image_data['landmark_3D'].append(coordinate_3D)
           image_data['landmark_2D_normalized'].append(coordinate_2D)
           
        # Append the data using concat
        data["image"].append(image_data) 
    else:
        continue
# # Save the DataFrame to a JSON file
# output = script_dir+"/landmark_data.json"
# df.to_json(output, orient = "records")
with open(script_dir+'/landmark_data.json', 'w') as json_file:
    json.dump(data, json_file, indent=4) 
print("Finished exporting landmark data.")

        # print(l)
        # nose = [l[NOSE].x, l[NOSE].y, l[NOSE].z]
        # left_e = [l[LEFT_EYE].x, l[LEFT_EYE].y, l[LEFT_EYE].z]
        # right_e = [l[RIGHT_EYE].x, l[RIGHT_EYE].y, l[RIGHT_EYE].z]
        # mid_e = [(left_e[0]+right_e[0])/2, (left_e[1]+right_e[1])/2, (left_e[2]+right_e[2])/2]
        # left_w = [l[LEFT_WRIST].x, l[LEFT_WRIST].y, l[LEFT_WRIST].z]
        # right_w = [l[RIGHT_WRIST].x, l[RIGHT_WRIST].y, l[RIGHT_WRIST].z]
        # left_s = [l[LEFT_SHOULDER].x, l[LEFT_SHOULDER].y, l[LEFT_SHOULDER].z]
        # right_s = [l[RIGHT_SHOULDER].x, l[RIGHT_SHOULDER].y, l[RIGHT_SHOULDER].z]
        # left_elb = [l[LEFT_ELBOW].x, l[LEFT_ELBOW].y, l[LEFT_ELBOW].z]
        # right_elb = [l[RIGHT_ELBOW].x, l[RIGHT_ELBOW].y, l[RIGHT_ELBOW].z]
        # left_h = [l[LEFT_HEEL].x, l[LEFT_HEEL].y, l[LEFT_HEEL].z]
        # right_h = [l[RIGHT_HEEL].x, l[RIGHT_HEEL].y, l[RIGHT_HEEL].z]
        # left_f_idx = [l[LEFT_FOOT_INDEX].x, l[LEFT_FOOT_INDEX].y, l[LEFT_FOOT_INDEX].z]
        # right_f_idx = [l[RIGHT_FOOT_INDEX].x, l[RIGHT_FOOT_INDEX].y, l[RIGHT_FOOT_INDEX].z]
        # ground_offset = min(left_h[1],right_h[1], left_f_idx[1], right_f_idx[1] )


        # # Output image data to json
        # image_data = {}
        # image_data["name"] = file
        # image_data["world_landmark_nose"] = nose
        # image_data["world_landmark_left_eye"] = left_e
        # image_data["world_landmark_right_eye"] = right_e
        # image_data["world_landmark_mid_eye"] = mid_e
        # image_data["world_landmark_left_wrist"] = left_w
        # image_data["world_landmark_right_wrist"] = right_w
        # image_data["world_landmark_left_shoulder"] = left_s
        # image_data["world_landmark_right_shoulder"] = right_s
        # image_data["world_landmark_left_elbow"] = left_elb
        # image_data["world_landmark_right_elbow"] = right_elb

        # image_data["left_eye_left_wrist_intersection"] = left_eye_left_wrist[i]
        # image_data["right_eye_right_wrist_intersection"] = right_eye_right_wrist[i]
        # image_data["mid_eye_left_wrist_intersection"] = mid_eye_left_wrist[i]
        # image_data["mid_eye_right_wrist_intersection"] = mid_eye_right_wrist[i]
        # image_data["nose_left_wrist_intersection"] = nose_left_wrist[i]
        # image_data["nose_right_wrist_intersection"] = nose_right_wrist[i]
        # image_data["left_shoulder_left_wrist_intersection"] = left_shoulder_left_wrist[i]
        # image_data["right_shoulder_right_wrist_intersection"] = right_shoulder_right_wrist[i]
        # image_data["left_elbow_left_wrist_intersection"] = left_elbow_left_wrist[i]
        # image_data["right_elbow_right_wrist_intersection"] = right_elbow_right_wrist[i]

        # image_data["ground_offset"] = ground_offset
        # image_data["target"] = int(file[-5])
        
        # data["image"].append(image_data)
        

    # annotated_image = image.copy()
    # # Draw segmentation on the image.
    # # To improve segmentation around boundaries, consider applying a joint
    # # bilateral filter to "results.segmentation_mask" with "image".
    # if BOUNDARY_SEGMENTATION: 
    #     condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    #     bg_image = np.zeros(image.shape, dtype=np.uint8)
    #     bg_image[:] = BG_COLOR
    #     annotated_image = np.where(condition, annotated_image, bg_image)
    
    # # Draw pose landmarks on the image.
    # mp_drawing.draw_landmarks(
    #     annotated_image,
    #     results.pose_landmarks,
    #     mp_pose.POSE_CONNECTIONS,
    #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    # cv2.imshow('annotated', annotated_image)
    # cv2.waitKey(0)
    # if BOUNDARY_SEGMENTATION: 
    #     cv2.imwrite(path + '/seg/annotated_' + file + '.png', annotated_image)



