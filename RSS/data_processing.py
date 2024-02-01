# import numpy as np
# from scipy.stats import multivariate_normal
# import pandas as pd

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# import numpy as np

# # Read CSV file into a DataFrame
# df = pd.read_csv('/Users/ivyhe/Downloads/Pointing Data Processing-human-to-human - eye_h.csv', skip_blank_lines=True).dropna(how = 'all')
# out_path = '/Users/ivyhe/Downloads/Probability-head.csv'
# columns = ['img_name', 'P(t1)', 'P(t2)', 'P(t3)', 'P(t4)','target_guess']

# table_data = pd.DataFrame(columns=columns)

# data = np.array([[]])
# # Loop through each row in the DataFrame
# for index in range(len(df)):
#     if index < 2:
#         continue
#     x_value = df.iloc[index]['guess_dist_x[m]']
#     y_value = df.iloc[index]['guess_dist_y[m]']
#     z_value = df.iloc[index]['guess_dist_z[m]']
#     if x_value is None or y_value is None or z_value is None:
#         continue
#     data = np.append(data, [x_value, y_value, z_value])

# # Perform actions with the values from the current row
# data = np.reshape(data, (len(data)//3, 3))
# # Remove NaN values from the data
# valid_indices = ~np.isnan(data).any(axis=1)
# data = data[valid_indices, :]

# print(data)
# # Assuming data is a 2D array with columns x, y, z

# # Calculate mean of each variable
# # need to group into target 1, 2
# mean_x = np.mean(data[:, 0])
# mean_y = np.mean(data[:, 1])
# mean_z = np.mean(data[:, 2])

# # Calculate covariance matrix
# cov_matrix = np.cov(data, rowvar=False)

# # Assuming mean_distances and covariance_matrix are known parameters
# mean_distances = np.array([mean_x, mean_y, mean_z])
# covariance_matrix = cov_matrix

# # Create the multivariate normal distribution
# mv_normal = multivariate_normal(mean=mean_distances, cov=covariance_matrix)

# # Given distances [x, y, z]

# # Loop through each row in the DataFrame
# for index, row in df.iterrows():
#     if index < 2:
#         continue
#     # Access values in each row using column names
#     t1_x = row['t1_d_x[m]']
#     t1_y = row['t1_d_y[m]']
#     t1_z = row['t1_d_z[m]']
#     t2_x = row['t2_d_x[m]']
#     t2_y = row['t2_d_y[m]']
#     t2_z = row['t2_d_z[m]']
#     t3_x = row['t3_d_x[m]']
#     t3_y = row['t3_d_y[m]']
#     t3_z = row['t3_d_z[m]']
#     t4_x = row['t4_d_x[m]']
#     t4_y = row['t4_d_y[m]']
#     t4_z = row['t4_d_z[m]']
#     out = [row['img_name']]
#     distances = np.array([[t1_x, t1_y, t1_z],
#                           [t2_x, t2_y, t2_z],
#                           [t3_x, t3_y, t3_z],
#                           [t4_x, t4_y, t4_z]])
#     # Perform actions with the values from the current row
#     # print(f"Processing row {index + 1}: x = {x_value}, y = {y_value}, z = {z_value}")


#     # Calculate the probability for each target
#     probability_per_target = mv_normal.pdf(distances)

#     # Normalize probabilities
#     normalized_probabilities = probability_per_target / np.sum(probability_per_target)
#     norm_prob = normalized_probabilities.tolist()
#     out.append(norm_prob[0])
#     out.append(norm_prob[1])
#     out.append(norm_prob[2])
#     out.append(norm_prob[3])
#     out.append(norm_prob.index(max(norm_prob)) + 1)
#     table_data = table_data.append(pd.Series(out, index=columns), ignore_index=True)

# # Create a grid of points for visualization
# x, y, z = np.meshgrid(np.linspace(min(data[:, 0]), max(data[:, 0]), 200),
#                       np.linspace(min(data[:, 1]), max(data[:, 1]), 200),
#                       np.linspace(min(data[:, 2]), max(data[:, 2]), 200))

# # Stack the grid points into a 3D array
# grid_points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

# # Evaluate the probability density at each grid point
# pdf_values = mv_normal.pdf(grid_points)

# # Reshape the PDF values to match the grid shape
# pdf_values = pdf_values.reshape(x.shape)        

# # Create a 2D contour plot
# plt.figure(figsize=(10, 8))
# contour = plt.contourf(x[:,:,0], y[:,:,0], pdf_values[:,:,0], cmap='viridis', alpha=0.7)

# # Scatter plot of the data points
# plt.scatter(data[:, 0], data[:, 1], c='r', marker='o', label='Data Points')

# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Multivariate Normal Distribution and Data Points')
# plt.colorbar(contour, label='Probability Density')

# plt.legend()
# plt.show()

# table_data.to_csv(out_path, index=False)
# print("New table saved")


import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd


import numpy as np

# Read CSV file into a DataFrame
in_names = ['/Users/ivyhe/Downloads/Pointing Data Processing-human-to-human - head_v.csv',
            '/Users/ivyhe/Downloads/Pointing Data Processing-human-to-human - elbow_v.csv',
             '/Users/ivyhe/Downloads/Pointing Data Processing-human-to-human - shoulder_v.csv',
              '/Users/ivyhe/Downloads/Pointing Data Processing-human-to-human - eye_v.csv',
               '/Users/ivyhe/Downloads/Pointing Data Processing-human-to-human - nose_v.csv',
            '/Users/ivyhe/Downloads/Pointing Data Processing-human-to-human - head_h.csv',
            '/Users/ivyhe/Downloads/Pointing Data Processing-human-to-human - elbow_h.csv',
             '/Users/ivyhe/Downloads/Pointing Data Processing-human-to-human - shoulder_h.csv',
              '/Users/ivyhe/Downloads/Pointing Data Processing-human-to-human - eye_h.csv',
               '/Users/ivyhe/Downloads/Pointing Data Processing-human-to-human - nose_h.csv',
            ]
out_names = ['head_v', 'elbow_v','shoulder_v', 'eye_v','nose_v',
             'head_h', 'elbow_h','shoulder_h', 'eye_h','nose_h',]
for name in out_names:
    df = pd.read_csv(f'/Users/ivyhe/Downloads/Pointing Data Processing-human-to-human - {name}.csv')
    out_path = f'/Users/ivyhe/Downloads/Probability-{name}.csv'
    columns = ['img_name', 'P(t1)', 'P(t2)', 'P(t3)', 'P(t4)',
            'target_guess1', 'target_guess2', 'target_guess3', 'target_guess4',
            'weighted_accuracy_score'
            ]

    table_data = pd.DataFrame(columns=columns)

    # Given distances [x, y, z]

    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        if index < 2:
            continue
        # Access values in each row using column names
        t1 = row['t1_d[m]']
        t2 = row['t2_d[m]']
        t3 = row['t3_d[m]']
        t4 = row['t4_d[m]']
        t_real = row['target_real']
        out = [row['img_name']]
        # Given Euclidean distances to the 4 targets
        distances = np.array([t1, t2, t3, t4])

        # Assuming mean and standard deviation for the distribution
        mean_distance = np.mean(distances)
        # Calculate the inverse distances as likelihood
        likelihood = 1 / distances

        # Normalize likelihood values
        normalized_likelihood = (likelihood / np.sum(likelihood)).tolist()
        probability = normalized_likelihood
        out.append(normalized_likelihood[0])
        out.append(normalized_likelihood[1])
        out.append(normalized_likelihood[2])
        out.append(normalized_likelihood[3])
        # Print the normalized likelihood values
        for i in range(len(distances)):
            print(f"Target {i + 1}: Normalized Likelihood = {normalized_likelihood[i]}")
            
        guess = normalized_likelihood.index(max(normalized_likelihood)) + 1
        g_list = [guess, 0, 0, 0]
        p_top = probability[guess-1]
        p_bot = probability[guess-1]
        for i in range(1, len(g_list)):
            if guess != t_real:
                normalized_likelihood[guess-1] = 0
                guess = normalized_likelihood.index(max(normalized_likelihood)) + 1  
                p_top = probability[guess-1]
                p_bot += probability[guess-1]
                g_list[i] = guess  
        out.append(g_list[0])
        out.append(g_list[1])
        out.append(g_list[2])
        out.append(g_list[3])
        weighted_accuracy = p_top / p_bot
        out.append(weighted_accuracy)
        
        table_data = table_data.append(pd.Series(out, index=columns), ignore_index=True)
        # Perform actions with the values from the current row
        # print(f"Processing row {index + 1}: x = {x_value}, y = {y_value}, z = {z_value}")
    table_data.to_csv(out_path, index=False)
    print("New table saved")