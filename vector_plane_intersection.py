import json
import os
import pandas as pd
import math
import numpy as np
import sys 

# Specify the path to the JSON file
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
json_file_path = os.getcwd() + "/data.json"
data = pd.read_json(script_dir+'/landmark_data.json')

names = []
nose = []
left_eye = []
right_eye = []
mid_eye = []
left_wrist = []
right_wrist = []
left_shoulder = []
right_shoulder = []
left_elbow = []
right_elbow = []
offsets = []
targets = []

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

# load coordinates to df
for entry in data["image"]:
    names.append(entry['name'][0])
    c = entry['landmark_3D']
    nose.append([c[NOSE]['x'],c[NOSE]['y'],c[NOSE]['z']])
    left_eye_coord = [c[LEFT_EYE]['x'], c[LEFT_EYE]['y'],c[LEFT_EYE]['z']]
    right_eye_coord = [c[RIGHT_EYE]['x'], c[RIGHT_EYE]['y'],c[RIGHT_EYE]['z']]
    left_eye.append(left_eye_coord)
    right_eye.append(right_eye_coord)
    mid_eye.append(((np.array(left_eye_coord)+ np.array(right_eye_coord))/2).tolist())
    left_wrist.append([c[LEFT_WRIST]['x'], c[LEFT_WRIST]['y'],c[LEFT_WRIST]['z']])
    right_wrist.append([c[RIGHT_WRIST]['x'], c[RIGHT_WRIST]['y'],c[RIGHT_WRIST]['z']])
    left_shoulder.append([c[LEFT_SHOULDER]['x'], c[LEFT_SHOULDER]['y'],c[LEFT_SHOULDER]['z']])
    right_shoulder.append([c[RIGHT_SHOULDER]['x'], c[RIGHT_SHOULDER]['y'],c[RIGHT_SHOULDER]['z']])
    left_elbow.append([c[LEFT_ELBOW]['x'], c[LEFT_ELBOW]['y'],c[LEFT_ELBOW]['z']])
    right_elbow.append([c[RIGHT_ELBOW]['x'], c[RIGHT_ELBOW]['y'],c[RIGHT_ELBOW]['z']])
    ground = max(c[LEFT_HEEL]['y'], c[RIGHT_HEEL]['y'], c[LEFT_FOOT_INDEX]['y'], c[RIGHT_FOOT_INDEX]['y'])
    offsets.append(ground)
    print(names)

df = pd.DataFrame(list(zip(names, nose, left_eye, right_eye, mid_eye, left_wrist, right_wrist, left_shoulder, right_shoulder, left_elbow, right_elbow, offsets)),
               columns =['Names', 'Nose', 'Left eye', 'Right eye', 'Mid eye', 'Left wrist', 'Right wrist', 'Left shoulder', 'Right shoulder', 'Left elbow', 'Right elbow', 'Offsets'])

def plane_line_intersection(la, lb, y_offset):
  if la and lb:
    # the line passing through la and lb is la + lab*t, where t is a scalar parameter
    la = np.array(la)
    lb = np.array(lb)
    lab = lb-la # vector from point 1 to point 2

    # the plane passing through p0, p1, p2 is p0 + p01*u + p02*v, where u and v are scalar parameters
    # ground plane (y-0)

    p0 = np.array([0,0-y_offset,0]) # point 0 on plane
    p1 = np.array([0,0-y_offset,1]) # point 1 on plane
    p2 = np.array([1,0-y_offset,0]) # point 2 on plane

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
  
def calculate_vector(a, b):
  if a and b:
    distance = [b[0]-a[0], b[1]-a[1], b[2]-a[2]]
    norm = math.sqrt(distance[0] ** 2 + distance[1] ** 2 + distance[2] ** 2)
    return [distance[0] / norm, distance[1] / norm, distance[2] / norm]
  else:
    return None

# create vectors
# for reference
vector_list = ["left_eye-left_wrist", 
           "right_eye-right_wrist",
           "mid_eye-left_wrist",
           "mid_eye-right_wrist",
           "nose_to-left_wrist",
           "nose_to-right_wrist",
           "left_shoulder-left_wrist", 
           "right_shoulder-right_wrist", 
           "left_elbow-left_wrist",
           "right_elbow-right_wrist"]
output = {'image':[]}
for i, row in df.iterrows():
    # ground plane has y=0 and we shift the world coordinates up by offset
    #init dict structure
    vector_ground_data = {}

    # # left eye left wrist
    vector_ground_data["name"] = df['Names'][i]
    point_a1 = (df['Left eye'][i][0], df['Left eye'][i][1] - df['Offsets'][i], df['Left eye'][i][2])
    point_b1 = (df['Left wrist'][i][0], df['Left wrist'][i][1] - df['Offsets'][i], df['Left wrist'][i][2])
    intersect_point_1 = plane_line_intersection(point_a1, point_b1, 0)
    vector_1 = calculate_vector(point_a1, point_b1)

    # right eye right wrist
    point_a2 = (df['Right eye'][i][0], df['Right eye'][i][1] - df['Offsets'][i], df['Right eye'][i][2])
    point_b2 = (df['Right wrist'][i][0], df['Right wrist'][i][1] - df['Offsets'][i], df['Right wrist'][i][2])
    intersect_point_2 = plane_line_intersection(point_a2, point_b2, 0)
    vector_2 = calculate_vector(point_a2, point_b2)

    # mid eye left wrist
    point_a3 = (df['Mid eye'][i][0], df['Mid eye'][i][1] - df['Offsets'][i], df['Mid eye'][i][2])
    point_b3 = (df['Left wrist'][i][0], df['Left wrist'][i][1] - df['Offsets'][i], df['Left wrist'][i][2])
    intersect_point_3 = plane_line_intersection(point_a3, point_b3, 0)
    vector_3 = calculate_vector(point_a3, point_b3)

    # mid eye right wrist
    point_a4 = (df['Mid eye'][i][0], df['Mid eye'][i][1] - df['Offsets'][i], df['Mid eye'][i][2])
    point_b4 = (df['Right wrist'][i][0], df['Right wrist'][i][1] - df['Offsets'][i], df['Right wrist'][i][2])
    intersect_point_4 = plane_line_intersection(point_a4, point_b4, 0)
    vector_4 = calculate_vector(point_a4, point_b4)

    # nose to left wrist
    point_a5 = (df['Nose'][i][0], df['Nose'][i][1] - df['Offsets'][i], df['Nose'][i][2])
    point_b5 = (df['Left wrist'][i][0], df['Left wrist'][i][1] - df['Offsets'][i], df['Left wrist'][i][2])
    intersect_point_5 = plane_line_intersection(point_a5, point_b5, 0)
    vector_5= calculate_vector(point_a5, point_b5)

    # nose to right wrist
    point_a6 = (df['Nose'][i][0], df['Nose'][i][1] - df['Offsets'][i], df['Nose'][i][2])
    point_b6 = (df['Right wrist'][i][0], df['Right wrist'][i][1] + df['Offsets'][i], df['Right wrist'][i][2])
    intersect_point_6 = plane_line_intersection(point_a6, point_b6, 0)
    vector_6 = calculate_vector(point_a6, point_b6)

    # left shoulder left wrist
    point_a7 = (df['Left shoulder'][i][0], df['Left shoulder'][i][1] - df['Offsets'][i], df['Left shoulder'][i][2])
    point_b7 = (df['Left wrist'][i][0], df['Left wrist'][i][1] - df['Offsets'][i], df['Left wrist'][i][2])
    intersect_point_7 = plane_line_intersection(point_a7, point_b7, 0)
    vector_7 = calculate_vector(point_a7, point_b7)
    
    # right shoulder right wrist
    point_a8 = (df['Right shoulder'][i][0], df['Right shoulder'][i][1] - df['Offsets'][i], df['Right shoulder'][i][2])
    point_b8 = (df['Right wrist'][i][0], df['Right wrist'][i][1] - df['Offsets'][i], df['Right wrist'][i][2])
    intersect_point_8 = plane_line_intersection(point_a8, point_b8, 0)
    vector_8 = calculate_vector(point_a8, point_b8)

    # left elbow left wrist
    point_a9 = (df['Left elbow'][i][0], df['Left elbow'][i][1] - df['Offsets'][i], df['Left elbow'][i][2])
    point_b9 = (df['Left wrist'][i][0], df['Left wrist'][i][1] - df['Offsets'][i], df['Left wrist'][i][2])
    intersect_point_9 = plane_line_intersection(point_a9, point_b9, 0)
    vector_9 = calculate_vector(point_a9, point_b9)

    # right elbow right wrist 
    point_a10 = (df['Right elbow'][i][0], df['Right elbow'][i][1] - df['Offsets'][i], df['Right elbow'][i][2])
    point_b10 = (df['Right wrist'][i][0], df['Right wrist'][i][1] - df['Offsets'][i], df['Right wrist'][i][2])
    intersect_point_10 = plane_line_intersection(point_a10, point_b10, 0)
    vector_10 = calculate_vector(point_a10, point_b10)

    ground = {"left_eye-left_wrist": intersect_point_1[1].tolist(), 
           "right_eye-right_wrist": intersect_point_2[1].tolist(),
           "mid_eye-left_wrist": intersect_point_3[1].tolist(),
           "mid_eye-right_wrist": intersect_point_4[1].tolist(),
           "nose_to-left_wrist": intersect_point_5[1].tolist(),
           "nose_to-right_wrist": intersect_point_6[1].tolist(),
           "left_shoulder-left_wrist": intersect_point_7[1].tolist(), 
           "right_shoulder-right_wrist": intersect_point_8[1].tolist(), 
           "left_elbow-left_wrist": intersect_point_9[1].tolist(),
           "right_elbow-right_wrist": intersect_point_10[1].tolist()}
    vector = {"left_eye-left_wrist": vector_1, 
           "right_eye-right_wrist": vector_2,
           "mid_eye-left_wrist": vector_3,
           "mid_eye-right_wrist": vector_4,
           "nose_to-left_wrist": vector_5,
           "nose_to-right_wrist": vector_6,
           "left_shoulder-left_wrist": vector_7, 
           "right_shoulder-right_wrist": vector_8, 
           "left_elbow-left_wrist": vector_9,
           "right_elbow-right_wrist": vector_10}
    offset = df['Offsets'][i]
    vector_ground_data["ground"] = ground
    vector_ground_data["vector"] = vector
    vector_ground_data["offset"] = offset
    output["image"].append(vector_ground_data)
with open(script_dir+'/vector_and_intersection_data.json', 'w') as json_file:
    json.dump(output, json_file, indent=4) 
print("Finished exporting landmark data.")