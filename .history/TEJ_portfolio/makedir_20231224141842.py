import os
import os

def makedirs(model)
# Specify the folder path
folder_path = f'data/{model}'  # Replace with the actual path

# Check if the directory exists
if not os.path.exists(folder_path):
    # Create the directory
    os.makedirs(folder_path)
    print(f"Folder created successfully at: {folder_path}")
else:
    print(f"Folder already exists at: {folder_path}")