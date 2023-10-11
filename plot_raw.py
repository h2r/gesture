import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

# plot the depth map 

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
x = np.arange(width)
y = np.arange(height)
x, y = np.meshgrid(x, y)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D surface
ax.plot_surface(x, y, data, cmap='viridis')  # Change the cmap to your preferred colormap

# Set labels for the axes
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Depth Value')
ax.view_init(elev=-90, azim=-90)
# Show the 3D plot
plt.show()

# # Reshape the data to match the 2D dimensions
# data = data.reshape((height, width))
# print(np.size(data, 1))
# print(np.size(data, 0))
# print(data[50][238])
# # Create a 2D plot
# plt.imshow(data, cmap='viridis')  # Change the cmap to your preferred colormap

# # Set labels for the axes
# plt.xlabel('X Axis')
# plt.ylabel('Y Axis')

# # Show the 2D plot
# plt.colorbar()  # Add a color bar for reference
# plt.show()
