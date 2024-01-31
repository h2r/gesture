import re
import ast
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --------- plane line intersection-----------
import numpy as np

filepath = '/Users/ivyhe/Downloads/color_2.jpg'

import cv2
import mediapipe as mp

def main():
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Initialize OpenCV video capture
    frame = cv2.imread(filepath)  # Use 0 for default webcam

    
        # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe Pose
    results = pose.process(rgb_frame)

        # Render the keypoints and connections on the image
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the output
    cv2.imshow('MediaPipe Pose Detection', frame)

  

    # Release the capture and destroy all OpenCV windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# def plane_line_intersection( LA, LB, A1, A2, A3):
#     # Calculate the normal vector of the plane
#     N = np.cross(A2 - A1, A3 - A1)

#     # Calculate the direction vector of the line
#     L = LB - LA

#     # Calculate the parameter t
#     t = np.dot(N, (A1 - LA)) / np.dot(N, L)

#     # Calculate the intersection point
#     intersection_point = LA + t * L

#     return intersection_point

# def plane_line_intersection2(la, lb, p0, p1, p2):

#     # the line passing through la and lb is la + lab*t, where t is a scalar parameter
#     la = np.array(la)
#     lb = np.array(lb)
#     lab = lb-la # vector from point 1 to point 2

#     # the plane passing through p0, p1, p2 is p0 + p01*u + p02*v, where u and v are scalar parameters
#     # ground plane (y-0)

#    #  p0 = np.array([0,0-y_offset,0]) # point 0 on plane
#    #  p1 = np.array([0,0-y_offset,1]) # point 1 on plane
#    #  p2 = np.array([1,0-y_offset,0]) # point 2 on plane

#     p01 = np.array(p1)-np.array(p0) # vector from point 0 to point 1
#     p02 = np.array(p2)-np.array(p0) # vector from point 0 to point 2

#     # setting this up as a system of linear equations and solving for t,u,v
#     A = np.array([-lab, p01, p02]).T # the matrix of coefficients
#     b = np.array([la-p0]).T# the vector of constants
    
#     tuv = np.matmul(np.linalg.inv(A),b) # solve the system of linear equations
#     intersection =  la+lab*tuv[0] # the solution is the point of intersection
#     # calculate the angle between the vector and plane
#     n = np.cross(p01, p02)
#     # angle = math.pi/2 - np.arccos(abs(np.dot(n, lab)) / np.linalg.norm(n) * np.linalg.norm(lab))
#     return intersection
   
    
# # Example points for the plane (A1, A2, A3) and the line (LA, LB)
# A1 = np.array([420, 230, 2.8])
# A2 = np.array([358, 219, 2.986])
# A3 = np.array([301, 222, 2.958])

# LA = np.array([327, 74, 3.062])
# LB = np.array([344, 142, 3.204])

# # Find the intersection point
# intersection_point1 = plane_line_intersection(A1, A2, A3, LA, LB)
# intersection_point2 = plane_line_intersection2(A1, A2, A3, LA, LB)

# print(f"Intersection point1: {intersection_point1}")
# print(f"Intersection point2: {intersection_point2}")


#----------------- PERPENDICULAR_VECTOR----------------
# def find_perpendicular_vector(point1, point2, point3):
#     # Vector from point1 to point2
#     vector1 = np.array(point2) - np.array(point1)

#     # Vector from point1 to point3
#     vector2 = np.array(point3) - np.array(point1)

#     # Cross product to find a vector perpendicular to the plane
#     perpendicular_vector = np.cross(vector1, vector2)

#     return perpendicular_vector


# # Example: Define three points to form a plane
# point1 = [2, 2, 0]
# point2 = [3, 2, 0]
# point3 = [2, 3, 0]

# # Find a vector perpendicular to the plane
# perpendicular_vector = find_perpendicular_vector(point1, point2, point3)
# print(perpendicular_vector)
# # Visualize the plane and perpendicular vector
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the plane defined by the three points
# points = np.array([point1, point2, point3, point1])
# ax.plot(points[:, 0], points[:, 1], points[:, 2], label='Plane')

# # Plot the perpendicular vector starting from the first point on the plane
# ax.quiver(point1[0], point1[1], point1[2], perpendicular_vector[0], perpendicular_vector[1], perpendicular_vector[2],
#           color='red', label='Perpendicular Vector', length=1, normalize=True)

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.legend()

# plt.show()


# # --------------_DISPLAY DEPTH (RAW) IMAGE---------------------
# # Replace 'your_raw_file.raw' with the actual path to your raw file
# file_path = '/Users/ivyhe/Desktop/pointing_trim/1125_t1_Depth.raw'
# color_path = "/Users/ivyhe/Desktop/pointing_trim/1125_t1_Color.png"
# depth_path = "/Users/ivyhe/Desktop/pointing_trim/1125_t1_Depth.png"
# # Specify the dimensions of the image (replace with your actual dimensions)
# width = 640
# height = 360

# # Assuming the depth data is stored as a 2D array of 16-bit unsigned integers (adjust dtype accordingly)
# depth_data = np.fromfile(file_path, dtype=np.uint16).reshape((height, width))
# print("should be around 2880")
# print(depth_data[236][242])
# # Plotting the depth data as an image
# plt.imshow(depth_data, cmap='viridis')  # You can choose a different colormap
# plt.colorbar(label='Depth Value')
# plt.title('Depth Data Visualization')
# plt.show()


# # Replace 'your_raw_file.raw' with the actual path to your raw file

# # Specify the dimensions of the image (replace with your actual dimensions)
# width = 640

# # Read the raw file to get the size of the array
# with open(file_path, 'rb') as file:
#     file.seek(0, 2)  # Move the cursor to the end of the file
#     file_size = file.tell()

# # Assuming the depth data is stored as a 1D array of 16-bit unsigned integers (adjust dtype accordingly)
# depth_data = np.fromfile(file_path, dtype=np.uint16)

# # Calculate the height based on the total size and width
# height = file_size // (width * depth_data.itemsize)

# # Reshape the array with the calculated dimensions
# depth_data = depth_data.reshape((height, width))

# # Create two sample images (replace with your own image data)
# image1 = cv2.imread(color_path)
# image2 = cv2.imread(depth_path)

# # Scale factor for downscaling
# scale_factor = 0.38

# # Resize the image using cv2's resize function
# resized_image = cv2.resize(image1, (0, 0), fx=scale_factor, fy=scale_factor)
# # Calculate padding dimensions
# pad_height = (image2.shape[0] - resized_image.shape[0]) // 2
# pad_width = (image2.shape[1] - resized_image.shape[1]) // 2

# # Pad the resized image with black edges using cv2's copyMakeBorder
# padded_image = cv2.copyMakeBorder(resized_image, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_CONSTANT, value=0)


# # Plotting side by side
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# # Plotting the depth data as an image
# axes[0].imshow(depth_data, cmap='viridis')  # You can choose a different colormap
# axes[0].set_title('Depth Data Visualization')
# axes[0].set_xlabel('Width')
# axes[0].set_ylabel('Height')

# # Plotting the first additional image
# axes[1].imshow(padded_image, cmap='gray')  # You can choose a different colormap
# axes[1].set_title('Image 1')
# axes[1].set_xlabel('Width')
# axes[1].set_ylabel('Height')

# # Plotting the second additional image
# axes[2].imshow(image2, cmap='gray')  # You can choose a different colormap
# axes[2].set_title('Image 2')
# axes[2].set_xlabel('Width')
# axes[2].set_ylabel('Height')

# plt.tight_layout()
# plt.show()

#------------------PARSE INTERSECTION POINT(POST PROCESSING)------------------

# def parse_intersection_points(file_path):
#     with open(file_path, 'r') as file:
#         lines = file.readlines()

#     data = []
#     current_data = None

#     for line in lines:
#         if line.startswith('--------------'):
#             if current_data:
#                 data.append(current_data)
#             current_data = {'Filename': ''}
#         elif line.startswith('thumb'):
#             current_data['Filename'] = line.strip()
#         elif 'INTERSECTION POINT' in line:
#             continue  # Skip this line
#         else:
#             # Parse the line containing the body part and coordinates
#             body_part, coordinates = re.match(r'(\w+) (\[.*\])', line).groups()
#             current_data[body_part] = coordinates[1:-1].split()

#     if current_data:
#         data.append(current_data)

#     return data

# # Example usage
# file_path = '/Users/ivyhe/Desktop/pointing/front_cam/front_left.txt'
# output_data = parse_intersection_points(file_path)


# import pandas as pd

# # Your data
# data =output_data

# # Create a DataFrame
# df = pd.DataFrame(data)

# # Split 'Nose', 'Eye', 'Shoulder', and 'Elbow' columns into 'x', 'y', 'z' columns
# df[['Nose_x', 'Nose_y', 'Nose_z']] = pd.DataFrame(df['nose'].tolist(), index=df.index)
# df[['Eye_x', 'Eye_y', 'Eye_z']] = pd.DataFrame(df['eye'].tolist(), index=df.index)
# df[['Shoulder_x', 'Shoulder_y', 'Shoulder_z']] = pd.DataFrame(df['shoulder'].tolist(), index=df.index)
# df[['Elbow_x', 'Elbow_y', 'Elbow_z']] = pd.DataFrame(df['elbow'].tolist(), index=df.index)

# # Drop the original 'Nose', 'Eye', 'Shoulder', and 'Elbow' columns
# df = df.drop(['nose', 'eye', 'shoulder', 'elbow'], axis=1)

# # Save to CSV
# df.to_csv('/Users/ivyhe/Desktop/pointing/front_cam/side_right.csv', index=False)

# # Display the updated DataFrame
# print(df)
