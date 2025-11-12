#!/usr/bin/env python3
import os
import glob
import argparse
from PIL import Image

def convert_to_grayscale(input_folder, output_folder=None):
    # If no output folder is specified, use the input folder
    if output_folder is None:
        output_folder = input_folder

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Search for JPG and JPEG files (case-insensitive)
    patterns = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG']
    jpg_files = []
    for pattern in patterns:
        jpg_files.extend(glob.glob(os.path.join(input_folder, pattern)))

    if not jpg_files:
        print(f"No JPG files found in {input_folder}")
        return

    # Process each image file
    for file in jpg_files:
        try:
            with Image.open(file) as img:
                # Convert the image to grayscale ('L' mode)
                gray_img = img.convert("L")
                
                # Remove any ICC profile metadata that may have come from the original
                if "icc_profile" in gray_img.info:
                    gray_img.info.pop("icc_profile")
                
                # Alternatively, you can force no ICC profile to be saved by:
                # gray_img.save(output_file, 'PNG', icc_profile=None)
                
                # Build the output file name (replace extension with .png)
                base_name = os.path.splitext(os.path.basename(file))[0]
                output_file = os.path.join(output_folder, base_name + '.png')
                
                # Save the grayscale image as a PNG file
                gray_img.save(output_file, 'PNG')
                print(f"Converted '{file}' -> '{output_file}'")
        except Exception as e:
            print(f"Error processing '{file}': {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert all JPG images in a folder to grayscale PNG images.'
    )
    parser.add_argument('--input_folder', type=str,
                        help='Path to the folder containing JPG images.')
    parser.add_argument('--output_folder', type=str, default=None,
                        help='(Optional) Folder where PNG images will be saved. Defaults to the input folder.')
    args = parser.parse_args()

    convert_to_grayscale(args.input_folder, args.output_folder)
