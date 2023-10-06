import json
import os

# Specify the path to the JSON file
json_file_path = os.getcwd() + "/data.json"

# Load data from the JSON file
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

print(data)