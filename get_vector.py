import os
from tkinter import RIGHT
from turtle import left, right
import cv2
import math
import pickle
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import open3d as o3d
import json

# this script maps skeleton tracking to our collected image data
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

path = '/Users/ivyhe/Desktop/pointing/data' # change file path to image path
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


# Intel RealSense D435 CAMERA INTRINSICS (in pixel units)
focal_x = 190.926941
focal_y = 190.926941
res_x = 320
res_y = 240
principal_x = 159.634918
principal_y = 119.397690
thresh = math.pi/18 #vector angle threshold for determining pointing observation

## helper Functions
# Check if p exists, if so, convert to x, y coordinate
def check(p, w, h):
  if p: # syntax that checks if list is empty...
    return [round(p.x * w), round(p.y * h)]
  else:
    return None
  
# Calculates a normalized vector from point a to point b.
def calculate_vector(a, b):
  if a and b:
    distance = [b[0]-a[0], b[1]-a[1]]
    norm = math.sqrt(distance[0] ** 2 + distance[1] ** 2)
    return [distance[0] / norm, distance[1] / norm]
  else:
    return None



def plane_line_intersection(la, lb, ground_offset):
  if la and lb:
    # the line passing through la and lb is la + lab*t, where t is a scalar parameter
    la = np.array(la)
    lb = np.array(lb)
    lab = lb-la # vector from point 1 to point 2
    #print(la, lb)

    # the plane passing through p0, p1, p2 is p0 + p01*u + p02*v, where u and v are scalar parameters
    # ground plane (y-0)
    
    # pcd = o3d.io.read_point_cloud("/Users/kylelee/Desktop/thao/research_env/initial_pointcloud.pcd")
    # plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
    #                                         ransac_n=3,
    #                                         num_iterations=1000)
    # [a, b, c, d] = plane_model
    # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    # default ground plane
    p0 = np.array([0,0-ground_offset,0]) # point 0 on plane
    p1 = np.array([0,0-ground_offset,1]) # point 1 on plane
    p2 = np.array([1,0-ground_offset,0]) # point 2 on plane

    p01 = p1-p0 # vector from point 0 to point 1
    p02 = p2-p0 # vector from point 0 to point 2

    # setting this up as a system of linear equations and solving for t,u,v
    A = np.array([-lab, p01, p02]).T # the matrix of coefficients
    b = np.array([la-p0]).T# the vector of constants
    try:
      tuv = np.matmul(np.linalg.inv(A),b) # solve the system of linear equations
      intersection =  la+lab*tuv[0] # the solution is the point of intersection
      # calculate the angle between the vector and plane
      n = np.cross(p01, p02)
      angle = math.pi/2 - np.arccos(abs(np.dot(n, lab)) / np.linalg.norm(n) * np.linalg.norm(lab))
      return [angle, intersection]
    except:
      return [None]
  else:
    return [None] 
  

data = {
   "image": [], 
   "world": []
}
# map skeleton to image
# For static images:
BG_COLOR = (192, 192, 192) # gray
BOUNDARY_SEGMENTATION = False
PLOT_WORLD_LANDMARK = False

with mp_pose.Pose(
    static_image_mode=True,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
  for file in os.listdir(path):
    if '.png'in file:
        print(file)
        image = cv2.imread(path+"/"+file)
        h, w, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        l = results.pose_world_landmarks.landmark
        nose = [l[NOSE].x, l[NOSE].y, l[NOSE].z]
        left_e = [l[LEFT_EYE].x, l[LEFT_EYE].y, l[LEFT_EYE].z]
        right_e = [l[RIGHT_EYE].x, l[RIGHT_EYE].y, l[RIGHT_EYE].z]
        mid_e = [(left_e[0]+right_e[0])/2, (left_e[1]+right_e[1])/2, (left_e[2]+right_e[2])/2]
        left_w = [l[LEFT_WRIST].x, l[LEFT_WRIST].y, l[LEFT_WRIST].z]
        right_w = [l[RIGHT_WRIST].x, l[RIGHT_WRIST].y, l[RIGHT_WRIST].z]
        left_s = [l[LEFT_SHOULDER].x, l[LEFT_SHOULDER].y, l[LEFT_SHOULDER].z]
        right_s = [l[RIGHT_SHOULDER].x, l[RIGHT_SHOULDER].y, l[RIGHT_SHOULDER].z]
        left_elb = [l[LEFT_ELBOW].x, l[LEFT_ELBOW].y, l[LEFT_ELBOW].z]
        right_elb = [l[RIGHT_ELBOW].x, l[RIGHT_ELBOW].y, l[RIGHT_ELBOW].z]
        left_h = [l[LEFT_HEEL].x, l[LEFT_HEEL].y, l[LEFT_HEEL].z]
        right_h = [l[RIGHT_HEEL].x, l[RIGHT_HEEL].y, l[RIGHT_HEEL].z]
        left_f_idx = [l[LEFT_FOOT_INDEX].x, l[LEFT_FOOT_INDEX].y, l[LEFT_FOOT_INDEX].z]
        right_f_idx = [l[RIGHT_FOOT_INDEX].x, l[RIGHT_FOOT_INDEX].y, l[RIGHT_FOOT_INDEX].z]
        ground_offset = min(left_h[1],right_h[1], left_f_idx[1], right_f_idx[1] )
        # Output image data to json
        image_data = {}
        image_data["name"] = file
        image_data["world_landmark_nose"] = nose
        image_data["world_landmark_left_eye"] = left_e
        image_data["world_landmark_right_eye"] = right_e
        image_data["world_landmark_mid_eye"] = mid_e
        image_data["world_landmark_left_wrist"] = left_w
        image_data["world_landmark_right_wrist"] = right_w
        image_data["world_landmark_left_shoulder"] = left_s
        image_data["world_landmark_right_shoulder"] = right_s
        image_data["world_landmark_left_elbow"] = left_elb
        image_data["world_landmark_right_elbow"] = right_elb
        image_data["ground_offset"] = ground_offset
        image_data["target"] = int(file[-5])
        data["image"].append(image_data)
    else:
        continue

    annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    if BOUNDARY_SEGMENTATION: 
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        annotated_image = np.where(condition, annotated_image, bg_image)
    
    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    # plot vectors
    # left eye to left wrist
    left_etw = plane_line_intersection(left_e, left_w, ground_offset)
    right_etw = plane_line_intersection(right_e, right_w, ground_offset)
    left_stw = plane_line_intersection(left_s, left_w, ground_offset)
    right_stw = plane_line_intersection(right_s, right_w, ground_offset)


    print(left_etw, right_etw, left_stw, right_stw) 
    cv2.imshow('annotated', annotated_image)
    cv2.waitKey(0)
    if BOUNDARY_SEGMENTATION: 
        cv2.imwrite(path + '/seg/annotated_' + file + '.png', annotated_image)
    # Plot pose world landmarks.
    if PLOT_WORLD_LANDMARK:
        mp_drawing.plot_landmarks(
            results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

# Write the data to a JSON file (create the file if it doesn't exist)
with open(path+'.json', 'w') as json_file:
    json.dump(data, json_file, indent=4) 

