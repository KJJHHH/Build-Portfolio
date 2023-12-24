import os

# Specify the folder name and path
folder_name = 'my_folder'
folder_path = '/data'  # Replace with the actual path

# Join the path and folder name
folder_full_path = os.path.join(folder_path, folder_name)

# Print for debugging
print(f"Attempting to create folder at: {folder_full_path}")

# Create the folder
try:
    os.makedirs(folder_full_path, exist_ok=True)
    print(f"Folder created successfully at: {folder_full_path}")
except Exception as e:
    print(f"Error creating folder: {e}")