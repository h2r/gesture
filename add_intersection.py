import pandas as pd
import math
import numpy as np

df = pd.read_json('data.json')
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

for entry in df["image"]:
    names.append(entry['name'])
    nose.append(entry['world_landmark_left_eye'])
    left_eye.append(entry['world_landmark_right_eye'])
    right_eye.append(entry['world_landmark_mid_eye'])
    mid_eye.append(entry['world_landmark_mid_eye'])
    left_wrist.append(entry['world_landmark_left_wrist'])
    right_wrist.append(entry['world_landmark_right_wrist'])
    left_shoulder.append(entry['world_landmark_left_shoulder'])
    right_shoulder.append(entry['world_landmark_right_shoulder'])
    left_elbow.append(entry['world_landmark_left_elbow'])
    right_elbow.append(entry['world_landmark_right_elbow'])
    offsets.append(entry['ground_offset'])
    targets.append(entry['target'])

df = pd.DataFrame(list(zip(names, nose, left_eye, right_eye, mid_eye, left_wrist, right_wrist, left_shoulder, right_shoulder, left_elbow, right_elbow, offsets, targets)),
               columns =['Names', 'Nose', 'Left eye', 'Right eye', 'Mid eye', 'Left wrist', 'Right wrist', 'Left shoulder', 'Right shoulder', 'Left elbow', 'Right elbow', 'Offsets', 'Targets'])

# left eye left wrist
# right eye right wrist
# mid eye left wrist
# mid eye right wrist
# nose to left wrist
# nose to right wrist
# left shoulder left wrist
# right shoulder right wrist
# left elbow left wrist
# right elbow right wrist 

# print(df)

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

# left_eye_to_left_wrist
intersection_points = []
for i, row in df.iterrows():
    # ground plane has y=0 and we shift the world coordinates up by offset
    point_a = (df['Left eye'][i][0], df['Left eye'][i][1] + df['Offsets'][i], df['Left eye'][i][2])
    point_b = (df['Left wrist'][i][0], df['Left wrist'][i][1] + df['Offsets'][i], df['Left wrist'][i][2])
    intersect_point = plane_line_intersection(point_a, point_b, 0)

    # point_a = df['Left eye'][i]
    # point_b = df['Left wrist'][i]
    # intersect_point = plane_line_intersection(point_a, point_b, df['Offsets'][i])
    intersection_points.append(intersect_point[1].tolist())
print(intersection_points)

# store normalized vector instead of angle (for example, vector representing left eye to left wrist)