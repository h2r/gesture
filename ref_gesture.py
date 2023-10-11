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
mp_pose = mp.solutions.pose
from initial_pointcloud import make_pointcloud

path = '/Users/kylelee/Desktop/thao2/'
fnum = 5 # can change this to desired file number (just remember to change it in the other files too)

LEFT_EYE = 2
RIGHT_EYE = 5
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16

# SPOT CAMERA INTRINSICS (in pixel units)
focal_x = 552.0291012161067
focal_y = 552.0291012161067
principal_x = 320.0
principal_y = 240.0
thresh = math.pi/18 #vector angle threshold for determining pointing observation

def check(p, w, h):
  if p: # syntax that checks if list is empty...
    return [round(p.x * w), round(p.y * h)]
  else:
    return None

def calculate_vector(a, b):
  if a and b:
    distance = [b[0]-a[0], b[1]-a[1]]
    norm = math.sqrt(distance[0] ** 2 + distance[1] ** 2)
    return [distance[0] / norm, distance[1] / norm]
  else:
    return None

def pixel_to_vision_frame(i,j,depth_img,rotation_matrix,position):
  '''
  Converts a pixel (i,j) in HxW image to 3d position in vision frame

  i,j: pixel location in image
  depth_img: HxW depth image
  rotaton_matrix: 3x3 rotation matrix of hand in vision frame
  position: 3x1 position vector of hand in vision frame
  '''

  #hand_tform_camera comes from line below, just a hardcoded version of it
  #rot2 = mesh_frame.get_rotation_matrix_from_xyz((0, np.pi/2, -np.pi/2))
  hand_tform_camera = np.array([[ 3.74939946e-33,6.12323400e-17,1.00000000e+00],
  [-1.00000000e+00,6.12323400e-17,0.00000000e+00],
  [-6.12323400e-17,-1.00000000e+00,6.12323400e-17]])

  #Intrinsics for RGB hand camera on spot
  CX = 320
  CY = 240
  FX= 552.0291012161067
  FY = 552.0291012161067


  z_RGB = depth_img[i,j]
  x_RGB = (j - CX) * z_RGB / FX
  y_RGB = (i - CY) * z_RGB / FY
  print("x_RGB, y_RGB, z_RGB:", x_RGB, y_RGB, z_RGB)

  bad_z = z_RGB == 0 #if z_RGB is 0, the depth was 0, which means we didn't get a real point. x,y,z will just be where robot hand was

  #first apply rot2 to move camera into hand frame, then apply rotation + transform of hand frame in vision frame
  transformed_xyz = np.matmul(rotation_matrix,np.matmul(hand_tform_camera,np.array([x_RGB,y_RGB,z_RGB]))) + position
  return(transformed_xyz,bad_z)

def plane_line_intersection(la, lb):
  if la and lb:
    # the line passing through la and lb is la + lab*t, where t is a scalar parameter
    la = np.array(la)
    lb = np.array(lb)
    lab = lb-la # vector from point 1 to point 2
    #print(la, lb)

    # the plane passing through p0, p1, p2 is p0 + p01*u + p02*v, where u and v are scalar parameters
    # ground plane (y-0)
    
    pcd = o3d.io.read_point_cloud("/Users/kylelee/Desktop/thao/research_env/initial_pointcloud.pcd")
    # plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
    #                                         ransac_n=3,
    #                                         num_iterations=1000)
    # [a, b, c, d] = plane_model
    # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    p0 = np.array([0,0,0]) # point 0 on plane
    p1 = np.array([0,0,1]) # point 1 on plane
    p2 = np.array([1,0,0]) # point 2 on plane
    # p0 = np.array([-d/a,0,0]) # point 0 on plane
    # p1 = np.array([0,-d/b,0]) # point 1 on plane
    # p2 = np.array([0,0,-d/c]) # point 2 on plane

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


# Skeleton tracking
# make_pointcloud()
# pcd = o3d.io.read_point_cloud("/Users/kylelee/Desktop/thao/research_env/initial_pointcloud.pcd")
# plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
#                                         ransac_n=3,
#                                         num_iterations=1000)
# pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
# good_points = []
# good_colors = []
# for file in os.listdir(path):
#   if 'color_' + str(fnum) + '.jpg' in file:
#     print(file)
#     idx = file.split('_')[1].split('.')[0]
#     image = cv2.imread(path+file)
#     h, w, _ = image.shape
#     # Convert the BGR image to RGB before processing.
#     results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     if not results.pose_landmarks:
#       continue
#     l = results.pose_landmarks.landmark
#     left_e = check(l[LEFT_EYE], w, h)
#     right_e = check(l[RIGHT_EYE], w, h)
#     left_w = check(l[LEFT_WRIST], w, h)
#     right_w = check(l[RIGHT_WRIST], w, h)
#     left_s = check(l[LEFT_SHOULDER], w, h)
#     right_s = check(l[RIGHT_SHOULDER], w, h)
#     left_elb = check(l[LEFT_ELBOW], w, h)
#     right_elb = check(l[RIGHT_ELBOW], w, h)
#     pixels = [left_e, right_e, left_w, right_w, left_s, right_s, left_elb, right_elb] # landmarks (2d coordinates)
#     print("Pixels:", pixels) # first coordinate corresponds to w, second is h (note w goes to 640, h goes to 480)

#     # # Transform landmark locations from image space to 3D space
#     # depth = pickle.load(open(path+'hand_depth_in_hand_color_frame'+idx,'rb'))
    
#     data_path="/Users/kylelee/Desktop/thao2/"
#     pose_data_fname="pose_data.pkl"
#     file_names = os.listdir(data_path)
#     num_files = int((len(file_names)-1)/ 3.0)

#     #for file_num in tqdm(range(num_files)):
#     file_num = fnum
#     points = [] # should this be on line 129?
#     pose_dir = pickle.load(open(f"{data_path}{pose_data_fname}","rb"))
#     rotation_matrix = pose_dir[file_num]['rotation_matrix']
#     position = pose_dir[file_num]['position']
#     depth_img = pickle.load(open(f"{data_path}hand_depth_in_hand_color_frame{str(file_num)}","rb"))
#     #depth_img = np.ones(np.shape(depth_img))
#     #print(depth_img)
#     for i in pixels:
#         try:
#           transformed_xyz, bad_z = pixel_to_vision_frame(i[1], i[0], depth_img, rotation_matrix, position)
#           if bad_z != True:
#             points.append(transformed_xyz.tolist())
#           else:
#             points.append(None)
#         except:
#           points.append(None)
#     print("Points:", points)
#     print()
#     # Calculate pointing vectors' intersection point with ground plane
#     left_target = plane_line_intersection(points[0], points[2]) # left eye to left wrist
#     vec_endpoints = (points[0], points[2])
#     if left_target[0]:
#       vec_start = farthest_from_point([vec_endpoints[0], vec_endpoints[1]], left_target[1])
#       vec_end = left_target[1]
#       print('left eye to wrist', vec_start, vec_end)
#       good_points.append((vec_start, vec_end.tolist(), 'blue'))
#       good_colors.append([0., 0., 0.])
#     else:
#       print('No left eye to wrist vector')
#     left_target = plane_line_intersection(points[4], points[2]) # left arm (shoulder to wrist)
#     vec_endpoints = (points[4], points[2])
#     if left_target[0]:
#       vec_start = farthest_from_point([vec_endpoints[0], vec_endpoints[1]], left_target[1])
#       vec_end = left_target[1]
#       print('left shoulder to wrist', vec_start, vec_end)
#       good_points.append((vec_start, vec_end.tolist(), 'gray'))
#       good_colors.append([0., 0., 0.])
#     else:
#       print('No left shoulder to wrist vector')
#     left_target = plane_line_intersection(points[6], points[2]) # left elbow to wrist
#     vec_endpoints = (points[6], points[2])
#     if left_target[0]:
#       vec_start = farthest_from_point([vec_endpoints[0], vec_endpoints[1]], left_target[1])
#       vec_end = left_target[1]
#       print('left elbow to wrist', vec_start, vec_end)
#       good_points.append((vec_start, vec_end.tolist(), 'red'))
#       good_colors.append([0., 0., 0.])
#     else:
#       print('No left elbow to wrist vector')

#     right_target = plane_line_intersection(points[1], points[3]) # right eye to wrist
#     vec_endpoints = (points[1], points[3])
#     if right_target[0]:
#       vec_start = farthest_from_point([vec_endpoints[0], vec_endpoints[1]], right_target[1])
#       vec_end = right_target[1]
#       print('right eye to wrist', vec_start, vec_end)
#       good_points.append((vec_start, vec_end.tolist(), 'blue'))
#       good_colors.append([0., 0., 0.])
#     else:
#       print('No right eye to wrist vector')
#     right_target = plane_line_intersection(points[5], points[3]) # right arm (shoulder to wrist)
#     vec_endpoints = (points[5], points[3])
#     if right_target[0]:
#       vec_start = farthest_from_point([vec_endpoints[0], vec_endpoints[1]], right_target[1])
#       vec_end = right_target[1]
#       print('right shoulder to wrist', vec_start, vec_end)
#       good_points.append((vec_start, vec_end.tolist(), 'gray'))
#       good_colors.append([0., 0., 0.])
#     else:
#       print('No right shoulder to wrist vector')
#     right_target = plane_line_intersection(points[7], points[3]) # right elbow to wrist
#     vec_endpoints = (points[7], points[3])
#     if right_target[0]:
#       vec_start = farthest_from_point([vec_endpoints[0], vec_endpoints[1]], right_target[1])
#       vec_end = right_target[1]
#       print('right elbow to wrist', vec_start, vec_end)
#       good_points.append((vec_start, vec_end.tolist(), 'red'))
#       good_colors.append([0., 0., 0.])
#     else:
#       print('No right elbow to wrist vector')
      
# print(good_points)
#print(good_colors)
pose.close()