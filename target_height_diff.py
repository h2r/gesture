# create a plane in vertical calculate vector intersection difference 
import json
import os
import pandas as pd
import math
import numpy as np
import sys 
import matplotlib.pyplot as plt
# Specify the path to the JSON file
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
json_file_path = os.getcwd() + "/data.json"
data = pd.read_json(script_dir+'/landmark_data.json')
old_target_img = ["1_left_n_1", "1_left_y_1","2_left_n_1", "2_left_y_1", 
                  "3_left_n_1", "3_left_y_1", "4_left_n_1",	"4_left_y_1",	
                  "4_right_n_1"]

# identify values that make sense
ideal_image_list = ["4_right_n_3.png", "4_right_n_2.png", "1_right_y_2.png",
                    "4_right_n_1.png", "3_right_n_1.png", "4_right_n_5.png", 
                    "4_right_n_4.png", "4_left_n_1.png", "3_left_n_1.png",
                    "4_left_n_2.png", "3_left_y_3.png", "2_left_y_5.png", 
                    "4_left_y_2.png", "4_left_y_1.png", "1_left_n_4.png", 
                    "4_right_y_5.png", "3_right_y_1.png"]
# Target 3D location, 1-4 from left to right
t1 = [1.05, 0, -1.02]
t2 = [.39, 0, -.72]
t3 = [-.39, 0, -.72]
t4 = [-1.05, 0, -1.02]
targets = [t1, t2, t3, t4]

t1_old = [1.05, 0, -0.56]
t2_old = [.39, 0, -.24]
t3_old = [-.39, 0, -.24]
t4_old = [-1.05, 0, -0.56]
old_targets = [t1_old, t2_old, t3_old, t4_old]

# get landmark location
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
target = []

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

for entry in data["image"]:
    name = entry['name'][0]
    # skip the image if there is no eye orientation
    if not(name in ideal_image_list):
       continue
    # if '_n_' in name:
    #    continue
    names.append(entry['name'][0])
    if name in old_target_img:
       t = old_targets[int(name[0])-1]
    else:
       t = targets[int(name[0])-1]
    target.append(t)
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

df = pd.DataFrame(list(zip(names, target, nose, left_eye, right_eye, mid_eye, left_wrist, right_wrist, left_shoulder, right_shoulder, left_elbow, right_elbow, offsets)),
               columns =['Names', 'Target', 'Nose', 'Left eye', 'Right eye', 'Mid eye', 'Left wrist', 'Right wrist', 'Left shoulder', 'Right shoulder', 'Left elbow', 'Right elbow', 'Offsets'])




# get landmark vector

# get target locations

# define vertical plane 

# find plane intersection
def plane_line_intersection(la, lb, target_location, y_offset):
  # print(la, lb, target_location, y_offset)
  if la and lb:
    # the line passing through la and lb is la + lab*t, where t is a scalar parameter
    la = np.array(la)
    lb = np.array(lb)
    lab = lb-la # vector from point 1 to point 2
    tx = target_location[0]
    ty = target_location[1]
    tz = target_location[2]
    # the plane passing through p0, p1, p2 is p0 + p01*u + p02*v, where u and v are scalar parameters
    # ground plane (y-0)
    # TODO: Change to vertical plane base on the target location
    p0 = np.array([tx,  ty+1,       tz]) # point 0 on plane
    p1 = np.array([tx,ty,         tz+1]) # point 1 on plane
    p2 = np.array([tx,  ty,         tz]) # point 2 on plane

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

output = {'image':[]}
vector_list = ["eye_wrist", 
           "ave_eye_wrist",
           "nose_wrist",
           "shoulder_wrist", 
           "elbow_wrist"]
eye_list = []
ave_eye_list = []
shoulder_list = []
nose_list = []
elbow_list = []
for i, row in df.iterrows():
    # ground plane has y=0 and we shift the world coordinates up by offset
    #init dict structure
    vector_ground_data = {}

    # # left eye left wrist
    name = df['Names'][i]
    print(name)
    vector_ground_data["name"] = name
    if "left" in name:
       # point with left arm
       eye = 'Left eye'
       shoulder = 'Left shoulder'
       elbow = 'Left elbow'
       wrist = 'Left wrist'
    else: 
       eye = 'Right eye'
       shoulder = 'Right shoulder'
       elbow = 'Right elbow'
       wrist = 'Right wrist'

    # eye
    point_a1 = (df[eye][i][0], df[eye][i][1] - df['Offsets'][i], df[eye][i][2])
    point_b1 = (df[wrist][i][0], df[wrist][i][1] - df['Offsets'][i], df[wrist][i][2])
    intersect_point_1 = plane_line_intersection(point_a1, point_b1, df['Target'][i], 0)
    print("eye:", intersect_point_1[1][1])
    # ave_eye
    point_a2 = (df['Mid eye'][i][0], df['Mid eye'][i][1] - df['Offsets'][i], df['Mid eye'][i][2])
    point_b2 = (df[wrist][i][0], df[wrist][i][1] - df['Offsets'][i], df[wrist][i][2])
    intersect_point_2 = plane_line_intersection(point_a2, point_b2, df['Target'][i], 0)
    print("average eye:", intersect_point_2[1][1])
    # nose
    point_a3 = (df['Nose'][i][0], df['Nose'][i][1] - df['Offsets'][i], df['Nose'][i][2])
    point_b3 = (df[wrist][i][0], df[wrist][i][1] - df['Offsets'][i], df[wrist][i][2])
    intersect_point_3 = plane_line_intersection(point_a3, point_b3, df['Target'][i], 0)
    print("nose vector:", intersect_point_3[1][1])
    # shoulder
    point_a4 = (df[shoulder][i][0], df[shoulder][i][1] - df['Offsets'][i], df[shoulder][i][2])
    point_b4 = (df[wrist][i][0], df[wrist][i][1] - df['Offsets'][i], df[wrist][i][2])
    intersect_point_4 = plane_line_intersection(point_a4, point_b4, df['Target'][i], 0)
    print("shoulder vector:", intersect_point_4[1][1])
    #elbow
    point_a5 = (df[elbow][i][0], df[elbow][i][1] - df['Offsets'][i], df[elbow][i][2])
    point_b5 = (df[wrist][i][0], df[wrist][i][1] - df['Offsets'][i], df[wrist][i][2])
    intersect_point_5 = plane_line_intersection(point_a5, point_b5, df['Target'][i], 0)
    print("elbow vector:", intersect_point_5[1][1])

    eye_list.append(intersect_point_1[1][1])
    ave_eye_list.append(intersect_point_2[1][1])
    nose_list.append(intersect_point_3[1][1])
    shoulder_list.append(intersect_point_4[1][1])
    elbow_list.append(intersect_point_5[1][1])


    """ground = {"eye": intersect_point_1[1].tolist(), 
              "ave_eye":intersect_point_2[1].tolist(), 
              "shoulder":intersect_point_3[1].tolist(),
              "nose": intersect_point_4[1].tolist(), 
              "elbow": intersect_point_5[1].tolist()}

    vector_ground_data["ground"] = ground
    output["image"].append(vector_ground_data)"""

plt.figure()
fig, ax = plt.subplots()
# # Creating a scatter plot for each list
# plt.scatter(range(len(eye_list)), eye_list, label='eye', marker='o')
# plt.scatter(range(len(ave_eye_list)), ave_eye_list, label='ave_eye', marker='s')
# plt.scatter(range(len(nose_list)), nose_list, label='nose', marker='^')
# plt.scatter(range(len(shoulder_list)), shoulder_list, label='shoulder', marker='x')
# plt.scatter(range(len(elbow_list)), elbow_list, label='elbow', marker='D')

# # Adding labels and a legend
# plt.yscale('log')
# plt.xlabel('image')
# plt.ylabel('Y values')
# plt.legend()
# plt.show()
def filter_outliers(dataset):
    std_dev = np.std(dataset)
    mean = np.mean(dataset)
    return [x for x in dataset if abs(x - mean) < 3 * std_dev]

# Create a box and whisker plot
data = [eye_list, ave_eye_list, nose_list, shoulder_list, elbow_list]
# Filter outliers and calculate the mean for each dataset
filtered_data = [filter_outliers(dataset) for dataset in data]
means = [round(np.mean(filtered_dataset),2) for filtered_dataset in filtered_data]
medians = [round(np.median(filtered_dataset),2) for filtered_dataset in filtered_data]
print("mean:", means)
print("median", medians)
plt.boxplot(filtered_data, showfliers=False)
plt.scatter([1, 2, 3, 4, 5], means)

# Add labels to the x-axis
plt.xticks([1, 2, 3, 4, 5], ['eye\n mean:%.2f \n median: %.2f'%(means[0], medians[0]), 
                             'ave_eye\n mean:%.2f \n median: %.2f'%(means[1], medians[1]), 
                             'nose\n mean:%.2f \n median: %.2f'%(means[2], medians[2]), 
                             'shoulder\n mean:%.2f \n median: %.2f'%(means[3], medians[3]), 
                             'elbow\n mean:%.2f \n median: %.2f'%(means[4], medians[4])])


# Add a title and labels to the axes
plt.title('y intersection of target and landmark to wrist vector')
plt.xlabel('landmark to wrist vector[m]')
plt.ylabel('height[ground as 0]')

# Show the plot
# Add minor gridlines on the y-axis
plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)
plt.subplots_adjust(bottom=0.2 )
plt.show()

"""with open(script_dir+'/target_intersection_height.json', 'w') as json_file:
    json.dump(output, json_file, indent=4) 
print("Finished exporting landmark data.")"""
# get vertical distribution & skewness
# loop through df, we only care about the one with looking in the direction, 
# scatter plot

# box & whisker plot