import os
import os

def makedirs(folder):
    """
    folder: eg. f'result/{model}' | model: 'ols', 'wls'
    """
    # Check if the directory exists
    if not os.path.exists(folder):
        # Create the directory
        os.makedirs(folder)
        print(f"Folder created successfully at: {folder}")
    else:
        print(f"Folder already exists at: {folder}")