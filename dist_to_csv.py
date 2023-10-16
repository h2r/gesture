import pandas as pd

# Read the JSON file
data = pd.read_json("dist_data.json")


# Create an empty list to store DataFrames
dfs = []

for image_data in data["image"]:
    # Extract the image name
    image_name = image_data["name"][0]
    image_name = image_name[0:len(image_name)-4]
    
    # Create a DataFrame for the current image
    df = pd.DataFrame({
        # "vector_name": image_data["vector_name"],
        f"dist_{image_name}": image_data["dist"],
        f"x_{image_name}": image_data["x_diff"],
        f"y_{image_name}": image_data["y_diff"]
    })
    
    dfs.append(df)

# Concatenate all DataFrames into a single DataFrame
result_df = pd.concat(dfs, axis=1).T
result_df.to_csv('overall_diff.csv', index=True)
print(result_df)
