import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mediapipe as mp
import os
import sys
import cv2
# Camera intrinsic parameters
fx = 190.926941  # Focal length in pixels along the x-axis
fy = 190.926941  # Focal length in pixels along the y-axis
cx = 159.634918   # Principal point's x-coordinate in pixels
cy = 119.39769   # Principal point's y-coordinate in pixels
landmark_list = ['nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
                 'right_eye_inner', 'right_eye', 'right_eye_outer', 'left_ear',
                 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder',
                 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 
                 'right_wrist', 'left_pinky', 'right_pinky', 'left_index', 
                 'right_index', 'left_thumb', 'right_thumb', 'left_hip', 
                 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
                 'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index']

# pose connection pair
pose_connection = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
                              (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
                              (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                              (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                              (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                              (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                              (29, 31), (30, 32), (27, 31), (28, 32)]

# Camera to human
camera_to_human_D435_old = [1.79, 1.26, 3.4]
camera_to_human_D435_new = [1.7, 1.26, 4.34]
camera_to_human_D455 = [1.73, 0.88, 2.54]
skew_offset = 5
# Define the dimensions of the 3D data
width = 320  # Change this to your data's width
height = 240  # Change this to your data's height
depth = 1  # Change this to your data's depth

img_list = ["1_left_n_2", "1_left_n_3", "1_left_y_2", "1_right_y_1",
           "1_right_y_3", "2_left_y_2","2_left_y_3", "2_left_y_4", "2_right_n_1", 
           "2_right_y_1", "3_right_n_1", "4_left_y_2", "4_right_n_2",
          "4_right_n_3", "4_right_n_4"]
# Initialize the MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=0)

# Define the path to the .raw file(Depth Image)
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
path = script_dir+'/data/'

with mp_pose.Pose(
    static_image_mode=True,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
    for (i, image_name) in enumerate(img_list):
    
        # Read the input image
        image_path = path+'%s.png'%(image_name)
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = pose.process(image_rgb)
        lmk = results.pose_landmarks.landmark
        # Assuming the data is in little-endian format (you may need to adjust this)
        width = 320  # Change this to your data's width
        height = 240  # Change this to your data's height
        file_path = path+"raw/%s.raw"%(image_name)
        with open(file_path, 'rb') as file:
            raw_data = file.read()
        depth_array = np.frombuffer(raw_data, dtype=np.uint16).reshape((height, width))

        # Normalize the depth values (adjust these values as needed)
        min_depth = depth_array.min()
        max_depth = depth_array.max()
        normalized_depth = ((depth_array - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)

        # Display the depth map using Matplotlib
        # Create a figure with two subplots
        fig, axes = plt.subplots(2, 1, figsize=(4, 7))
        axes[0].imshow(image_rgb)
        axes[1].imshow(normalized_depth, cmap='viridis')
        x_pixels = []
        y_pixels = []
        depth_list = []
        for i in range(len(landmark_list)):
             x = round(lmk[i].x * width+skew_offset)
             y = round(lmk[i].y * width+skew_offset)
             d = depth_array[x][y]
             x_pixels.append(x)
             y_pixels.append(y)
             depth_list.append(d)

        for (start, end) in pose_connection:
                x = [lmk[start].x * width+skew_offset, lmk[end].x * width+skew_offset]
                y = [lmk[start].y * height, lmk[end].y * height]
                
                axes[1].plot(x, y, color='blue')
        
        if "left" in image_name:
            wrist = [x_pixels[15], y_pixels[15], depth_list[15]]
            elbow = [x_pixels[13], y_pixels[13], depth_list[13]]
            shoulder = [x_pixels[11], y_pixels[11], depth_list[11]]
            eye = [x_pixels[2], y_pixels[2], depth_list[2]]
        else:
            wrist = [x_pixels[16], y_pixels[16], depth_list[16]]
            elbow = [x_pixels[14], y_pixels[14], depth_list[14]]
            shoulder = [x_pixels[12], y_pixels[12], depth_list[12]]
            eye = [x_pixels[5], y_pixels[5], depth_list[5]]
        nose = [x_pixels[0], y_pixels[0], depth_list[0]]
        ave_eye = [(x_pixels[2]+x_pixels[5])//2, (y_pixels[2]+y_pixels[5])//2, (depth_list[2]+depth_list[5])//2]
        


        # plot 3D model and target
        plt.title('%sDepth Map'%image_name)
        plt.show()
