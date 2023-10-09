# plot 3D
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import numpy as np
import json

# Get the directory of the currently running script
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
# Target 3D location, 1-4 from left to right
t4 = [-1.05, 0, -.56]
t3 = [-.39, 0, -.24]
t1 = [1.05, 0, -.56]
t2 = [.39, 0, -.24]
targets = [t1, t2, t3, t4]

# open and import data from json
ld = pd.read_json(script_dir+"/landmark_data.json")
vaid = pd.read_json(script_dir+"/vector_and_intersection_data.json")
# pose connection pair
pose_connection = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
                              (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
                              (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                              (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                              (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                              (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                              (29, 31), (30, 32), (27, 31), (28, 32)]

vector_connection = [
                               (2, 15),  # left eye to left wrist
                               (5, 16),  # right eye to right wrist
                              (0,15), # nose to left wrist
                              (0,16), # nose to right wrist
                               (11, 15),  # left shoulder to left wrist
                              (12, 16),  # right shoulder to right wrist
                              (13, 15),  # left elbow to left wrist
                              (14, 16), # right elbow to right wrist
                              ]
# Color assignment
WHITE_COLOR = (224/255, 224/255, 224/255) # (shoulder to wrist)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255/255) #(elbow to wrist)
GREEN_COLOR = (0, 128/255, 0) #(nose to wrist)
BLUE_COLOR = (255/255, 0, 0) #(eye to wrist)

# left eye left wrist
# right eye right wrist
# nose to left wrist
# nose to right wrist
# left shoulder left wrist
# right shoulder right wrist
# left elbow left wrist
# right elbow right wrist 
color_list = [BLUE_COLOR, BLUE_COLOR,  GREEN_COLOR, GREEN_COLOR,
              WHITE_COLOR, WHITE_COLOR, RED_COLOR, RED_COLOR]
vector_list = ["left_eye", 
           "right_eye",
           "mid_eye_left",
           "mid_eye-right",
           "nose_to-left",
           "nose_to-right",
           "left_shoulder", 
           "right_shoulder", 
           "left_elbow",
           "right_elbow"]

output = {"image" : []}
for i in range(len(ld["image"])):
    image_output = {"name":[],
                    "target":[]}
    img = ld["image"][i]
    # # 3D plotting
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111, projection='3d')
    fig1 = plt.figure(figsize=(7, 5))
    fig2 = plt.figure(figsize=(10, 5))
    ax1 = fig1.add_subplot(111, projection='3d')
    # Get body coornidates for all joints
    lmk = pd.DataFrame(img["landmark_3D"])
    img_name = img["name"][0]
    target = int(img_name[-5])
    image_output['name'].append(img_name)
    image_output['target'].append(target)
    ground = pd.DataFrame(vaid["image"][i]["ground"])
    vector = pd.DataFrame(vaid["image"][i]["vector"])
    offset = vaid["image"][i]["offset"]
    # # plot joint coordinates, y, z flipped
    # for name in lmk['landmark_name']:
    #     x = lmk.loc[lmk['landmark_name']==name,'x'].values[0]
    #     y = lmk.loc[lmk['landmark_name']==name,'y'].values[0] - offset
    #     z = lmk.loc[lmk['landmark_name']==name,'z'].values[0]
    #     ax1.scatter(x, z, y, c='b', marker = 'o')
    # draw skeleton, y, z flipped
    for (start, end) in pose_connection:
        x = [lmk.iloc[start]["x"], lmk.iloc[end]["x"]]
        y = [lmk.iloc[start]["y"]-offset, lmk.iloc[end]["y"]-offset] 
        z = [lmk.iloc[start]["z"], lmk.iloc[end]["z"]]
        ax1.plot(x, z, y, color='blue')
    
    # plot target position
    for i in range(len(targets)):
        t = targets[i]
        x = t[0]
        y = t[1]
        z = t[2]
        if i+1 == target:
            c = 'green'
        else:
            c = 'grey'
        ax1.scatter(x, z, y, marker = 's', color = c, s = 100, label = str(i+1))


    # plot ground intersection & vector
    c_list = [BLUE_COLOR, BLUE_COLOR, 'm','m', GREEN_COLOR, GREEN_COLOR,
              WHITE_COLOR, WHITE_COLOR, RED_COLOR, RED_COLOR]
    i_temp = 0
    for name in ground:

        x = ground[name][0]
        y = ground[name][1]
        z = ground[name][2]
        c = c_list[i_temp]
        v = vector_list[i_temp]
        i_temp += 1
        ax1.scatter(x, z, y, color = c, marker = 'o')

    # plot connection vector
    for i in range(len(vector_connection)):
        start= vector_connection[i][0]
        end= vector_connection[i][1]
        x = [lmk.iloc[start]["x"], lmk.iloc[end]["x"]]
        y = [lmk.iloc[start]["y"]-offset, lmk.iloc[end]["y"]-offset] 
        z = [lmk.iloc[start]["z"], lmk.iloc[end]["z"]]
        c = color_list[i]
        ax1.plot(x, z, y, color=c)

    # plot pointing vectors


    # output target distance using different vectors

    # Set ax1is labels
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim(2, -2)  
    ax1.set_zlim( -1.8, 0.1)  
    ax1.set_ylim(-2, 2)  
    # Make the ax1is planes solid gray
    ax1.w_xaxis.pane.fill = True  # Disable filling the x-ax1is plane
    ax1.w_yaxis.pane.fill = True  # Disable filling the y-ax1is plane
    ax1.w_zaxis.pane.fill = True  # Disable filling the z-ax1is plane

    # Set the view to have y as vertical and z pointing out
    ax1.view_init(elev=-155, azim=85)

    # Set plot title
    ax1.set_title('3D Plot of %s'%img_name)

    # 2D distance plotting
    ax2 = fig2.add_subplot(121)
    ax3 = fig2.add_subplot(122)
    ax2.grid(True)
    ax2.set_axisbelow(True)
    ax3.grid(True)
    ax3.set_axisbelow(True)


    # plot target position
    for i in range(len(targets)):
        t = targets[i]
        x = t[0]
        y = t[2]
        if i+1 == target:
            c = 'green'
        else:
            c = 'grey'
        ax2.scatter(x, y, marker = 's', color = c, s = 100, label = str(i+1))
        

    # plot ground intersection & vector
    c_list = [BLUE_COLOR, BLUE_COLOR, 'm','m', GREEN_COLOR, GREEN_COLOR,
              WHITE_COLOR, WHITE_COLOR, RED_COLOR, RED_COLOR]
    i_temp = 0
    x_values = []
    y_values = []
    dist_values = []
    for name in ground:
        tx = targets[target-1][0]
        ty = targets[target-1][2]
        x = ground[name][0]
        y = ground[name][2]
        c = c_list[i_temp]
        v = vector_list[i_temp]
        i_temp += 1
        ax2.scatter(x, y, color = c, marker = 'o')
        x_values.append(abs(tx-x))
        y_values.append(abs(ty-x))
        dist_values.append(np.sqrt((tx-x)**2 + (ty-y)**2))
    image_output['vector_name'] = vector_list
    image_output['x_diff'] = x_values
    image_output['y_diff'] = y_values
    image_output['dist'] = dist_values
    
        
    ax2.set_title("x-y plane visualization")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    categories = vector_list
    # Create a bar graph
    bar_width = 0.24
    x = np.arange(len(categories))
    ax3.bar(x - bar_width, x_values, bar_width, label='x_diff', color='blue')
    ax3.bar(x, y_values, bar_width, label='y_diff', color='green')
    ax3.bar(x + bar_width, dist_values, bar_width, label='Distance', color='red')
    ax3.set_xticks(x, categories, rotation=90)
    ax3.set_title('distance distribution of vectors to target%i'%target)
    ax3.set_ylabel('distance to target[m]')

    # Show the plot
    plt.subplots_adjust(bottom = 0.2)
    fig1.savefig(script_dir+'/plot/%s_3D.png'%img_name)
    fig2.savefig(script_dir+'/plot/%s_target_%i.png'%(img_name,target))
    plt.legend()
    plt.show()

    # save output to json
    output["image"].append(image_output)

with open(script_dir+'/dist_data.json', 'w') as json_file:
    json.dump(output, json_file, indent=4) 
print("Finished exporting landmark data.")