import os
from PIL import Image, ImageOps

def duplicate_and_invert_pngs(folder_path):
    """
    Reads a folder of .png files, duplicates each file with "_invert" added before ".png",
    and saves the inverted image.

    Args:
        folder_path (str): Path to the folder containing .png files.
    """
    # Ensure the folder path exists
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    # Get a list of .png files in the folder
    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

    if not png_files:
        print("No .png files found in the folder.")
        return

    # Loop through each file and create an inverted duplicate
    for file_name in png_files:
        original_path = os.path.join(folder_path, file_name)
        file_base, file_ext = os.path.splitext(file_name)
        inverted_name = f"{file_base}_invert{file_ext}"
        inverted_path = os.path.join(folder_path, inverted_name)

        # Open the original image and invert its colors
        with Image.open(original_path) as img:
            inverted_img = ImageOps.invert(img.convert("RGB"))  # Ensure conversion to RGB for inversion
            inverted_img.save(inverted_path)

        print(f"Created inverted image: {inverted_path}")

# Specify the folder path
folder_path = "/home/lin/Research/denoise/Restormer/dataset/train_clean"
duplicate_and_invert_pngs(folder_path)
