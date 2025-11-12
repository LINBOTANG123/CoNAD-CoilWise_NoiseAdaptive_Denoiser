import numpy as np
import h5py
import os
import cv2
import argparse

def load_mat_hdf5_and_save_pngs(input_mat, output_folder):
    """
    Load a MATLAB v7.3 .mat file (HDF5 format) containing MRI data with shape (90, N, 32, 146, 146),
    reorder dimensions to (146, 146, 32, 90), convert complex numbers to magnitude,
    process each of the N samples separately, and save the 90 combined slices as PNGs.

    Args:
        input_mat (str): Path to the .mat file.
        output_folder (str): Directory where PNG images will be saved.
    """

    # Open .mat file with h5py (HDF5 format)
    with h5py.File(input_mat, 'r') as f:
        # Print available keys to identify dataset
        print("Available keys in .mat file:", list(f.keys()))

        # Extract the dataset (you may need to adjust the key)
        if "image" in f:
            Im = np.array(f["image"])  # Read as numpy array
        elif "Im" in f:
            Im = np.array(f["Im"])
        else:
            raise KeyError("Could not find 'image' or 'Im' in the .mat file.")

    # Print the detected shape and dtype before reordering
    print(f"Original shape: {Im.shape}, dtype: {Im.dtype}")

    # Convert structured complex dtype to a proper magnitude array
    real_part = Im['real'].astype(np.float32)
    imag_part = Im['imag'].astype(np.float32)
    Im = np.sqrt(real_part**2 + imag_part**2)  # Convert to magnitude

    # Get the number of samples (N)
    num_samples = Im.shape[1]  # Second dimension represents the number of samples
    print(f"Number of samples: {num_samples}")

    # Process each sample separately
    for sample_idx in range(num_samples):
        print(f"Processing sample {sample_idx + 1}/{num_samples}...")

        # Extract the sample → Now shape: (90, 32, 146, 146)
        sample_data = Im[:, sample_idx, :, :, :]

        # Fix the shape: Currently (90, 32, 146, 146), needs to be (146, 146, 32, 90)
        sample_data = np.transpose(sample_data, (2, 3, 1, 0))  # Correct order!

        # Convert to float32 for compatibility
        sample_data = sample_data.astype(np.float32)

        # Verify new shape
        if sample_data.shape != (146, 146, 32, 90):
            raise ValueError(f"Unexpected data shape after transpose: {sample_data.shape}. Expected (146, 146, 32, 90).")

        # Create a subfolder for this sample
        sample_output_folder = os.path.join(output_folder, f"sample_{sample_idx+1}")
        os.makedirs(sample_output_folder, exist_ok=True)

        # Process each slice
        for slice_idx in range(90):
            # Extract 32-channel data for this slice → Shape: (146, 146, 32)
            slice_data = sample_data[..., slice_idx]

            # Combine 32 channels using the root-sum-of-squares (RSS) method
            combined_image = np.sqrt(np.sum(slice_data**2, axis=-1))  # Shape: (146, 146)

            # ✅ Fix Rotation Issue: Rotate the image by 90 degrees counterclockwise
            combined_image = np.rot90(combined_image, k=3)  # Rotate 270 degrees = -90 degrees

            # Normalize to 8-bit range (0-255) for PNG
            combined_image = cv2.normalize(combined_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Save as PNG
            output_filename = os.path.join(sample_output_folder, f"slice_{slice_idx+1:03d}.png")
            cv2.imwrite(output_filename, combined_image)

        print(f"✅ Sample {sample_idx+1} saved in '{sample_output_folder}'")

    print(f"🎉 All {num_samples} samples processed and saved in '{output_folder}'")

if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser(description="Convert a MATLAB v7.3 .mat file (with 32-channel MRI slices) to PNG images.")
    parser.add_argument("input_mat", type=str, help="Path to the input .mat file")
    parser.add_argument("output_folder", type=str, help="Folder to save PNG images")

    args = parser.parse_args()
    
    load_mat_hdf5_and_save_pngs(args.input_mat, args.output_folder)
