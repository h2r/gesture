import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

# Camera intrinsic parameters
fx = 190.926941  # Focal length in pixels along the x-axis
fy = 190.926941  # Focal length in pixels along the y-axis
cx = 159.634918   # Principal point's x-coordinate in pixels
cy = 119.39769   # Principal point's y-coordinate in pixels


# Camera to human
camera_to_human_D435_old = [1.79, 1.26, 3.4]
camera_to_human_D435_new = [1.7, 1.26, 4.34]
camera_to_human_D455 = [1.73, 0.88, 2.54]

# Define the dimensions of the 3D data
width = 320  # Change this to your data's width
height = 240  # Change this to your data's height
depth = 1  # Change this to your data's depth

# Define the path to the .raw file
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
path = script_dir+'/data/'
file_path = path+"1_Depth_t1.raw"

# Read binary data from the .raw file
with open(file_path, 'rb') as raw_file:
    data = np.fromfile(raw_file, dtype=np.uint16)  # Assuming 32-bit floating-point data

data = data.reshape((height, width))
# set a threshold
x_min = -1.0
x_max = 1.0
y_min = -1.0
y_max = 1.0
z_min = 2.5
z_max = 5.0

points_3d = np.zeros((height, width, 3), dtype=float)
for v in range(height):
    for u in range(width):
        # Depth value in meters
        depth = data[v, u]/1000
        if depth > z_min and depth < z_max:
            # Calculate 3D point
            X = (u - cx) * (depth / fx) 
            Y = (v - cy) * (depth / fy) - camera_to_human_D435_old[1]
            Z = depth - camera_to_human_D435_old[2]
            points_3d[v, u] = [X, Y, Z]
# Plot the 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract X, Y, and Z coordinates from the points_3d array
X, Y, Z = points_3d[:, :, 0], points_3d[:, :, 1], points_3d[:, :, 2]

# Create a 3D scatter plot
ax.scatter(X, Y, Z, c=Z, s = 1, cmap='viridis')  # You can use Z for color mapping

# Set the view to have y as vertical and z pointing out
ax.view_init(elev=-81, azim=-90)

# Set labels for the axes
ax.set_xlabel('X Axis (meters)')
ax.set_ylabel('Y Axis (meters)')
ax.set_zlabel('Z Axis (meters)')
# ax.set_xlim(x_min, x_max)
# ax.set_ylim(y_min, y_max)
# ax.set_zlim(z_min, z_max)
# Show the 3D plot
plt.show()

