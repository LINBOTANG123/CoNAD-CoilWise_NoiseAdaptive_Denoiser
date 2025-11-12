import os
import cv2
import numpy as np
import random  # For random.uniform

def repeated_gaussian_smoothing(img, ksize=3, sigma=1.0, times=5):
    """
    Repeatedly applies Gaussian blur to an image.
    Args:
        img   (np.ndarray): 2D NumPy array (grayscale).
        ksize (int): Gaussian kernel size (e.g., 3 or 5).
        sigma (float): Standard deviation for Gaussian kernel.
        times (int): Number of times to apply the blur.
    Returns:
        np.ndarray: The smoothed image.
    """
    smoothed = img.copy()
    for _ in range(times):
        smoothed = cv2.GaussianBlur(smoothed, (ksize, ksize), sigma)
    return smoothed

def create_inverted_intensity_noise_dataset(
        input_folder, output_folder,
        noise_std_min=0.05, 
        noise_std_max=0.3,
        ksize=3, blur_sigma=7.0, times=300,
        add_noise_to_smoothed=False):
    """
    Creates a dataset by:
      1) Loading images (grayscale).
      2) Repeatedly smoothing each image.
      3) Normalizing the smoothed image to [0,1].
      4) Inverting intensities to give higher noise to lower-intensity regions.
      5) Randomly choose a global noise std in [noise_std_min, noise_std_max].
      6) Multiply the inverted map by global_noise_std => noise_map.
      7) Generate random Gaussian noise per voxel using noise_map[i,j] as std.
      8) Add the noise to the data (original or smoothed).
      9) Save the noise map and the noised image.

    Args:
        input_folder        (str): Path to the folder containing input images.
        output_folder       (str): Path to the folder where processed images will be saved.
        noise_std_min     (float): Minimum global noise std.
        noise_std_max     (float): Maximum global noise std.
        ksize              (int): Gaussian kernel size for smoothing.
        blur_sigma       (float): Gaussian kernel std for smoothing.
        times              (int): Number of times to apply smoothing.
        add_noise_to_smoothed (bool): If True, noise is added to the smoothed image;
                                       else to the original image.

    Returns:
        None
    """
    os.makedirs(output_folder, exist_ok=True)

    # Directories for saving noise maps and noised images
    noise_map_folder = os.path.join(output_folder, "noise_maps")
    noised_folder = os.path.join(output_folder, "noised_images")
    os.makedirs(noise_map_folder, exist_ok=True)
    os.makedirs(noised_folder, exist_ok=True)

    # Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    for filename in os.listdir(input_folder):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in image_extensions:
            continue  # Skip non-image files

        file_path = os.path.join(input_folder, filename)
        # Read as grayscale
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: unable to read {file_path}. Skipping.")
            continue

        # Convert to float32 [0, 1]
        img_float = img.astype(np.float32) / 255.0

        # Step 1: Repeated smoothing
        smoothed = repeated_gaussian_smoothing(img_float, ksize, blur_sigma, times)

        # Step 2: Normalize smoothed to [0,1]
        sm_min, sm_max = smoothed.min(), smoothed.max()
        if sm_max - sm_min > 1e-8:  # avoid div-by-zero
            smoothed = (smoothed - sm_min) / (sm_max - sm_min)
        else:
            smoothed = np.clip(smoothed, 0, 1)

        # Step 3: Invert intensities -> higher noise for lower intensities
        inverted_intensity_map = 1.0 - smoothed

        # Step 4: Randomly choose global noise std in [noise_std_min, noise_std_max]
        global_noise_std = random.uniform(noise_std_min, noise_std_max)

        # Step 5: Multiply inverted map by global_noise_std => noise_map
        noise_map = inverted_intensity_map * global_noise_std  # shape: same as image

        # Step 6: Generate random Gaussian noise with voxel-wise std
        noise = np.random.randn(*noise_map.shape).astype(np.float32) * noise_map

        # Step 7: Add noise to the data
        if add_noise_to_smoothed:
            base_img = smoothed
        else:
            base_img = img_float

        noised_img = base_img + noise
        noised_img = np.clip(noised_img, 0, 1)  # clamp

        # Convert images to 8-bit
        # For visualization, scale noise_map to [0, 255] by dividing 
        # by global_noise_std (avoid zero division).
        div_safe = global_noise_std if global_noise_std > 1e-8 else 1e-8
        noise_map_uint8 = (noise_map / div_safe * 255).astype(np.uint8)
        noised_uint8 = (noised_img * 255).astype(np.uint8)

        # Construct output paths
        base_name = os.path.splitext(filename)[0]
        noise_map_path = os.path.join(noise_map_folder, f"{base_name}_invert.png")
        noised_path = os.path.join(noised_folder, f"{base_name}_invert.png")

        # Save outputs
        cv2.imwrite(noise_map_path, noise_map_uint8)
        cv2.imwrite(noised_path, noised_uint8)

        print(f"Processed {filename} with global_noise_std={global_noise_std:.3f}")
        print(f"  -> Noise map saved to: {noise_map_path}")
        print(f"  -> Noised image saved to: {noised_path}")

def main():
    # Example usage
    input_folder = "/home/lin/Research/denoise/Restormer/dataset/train_clean"
    output_folder = "/home/lin/Research/denoise/Restormer/dataset/train_noise_invert"

    # Parameters
    noise_std_min = 0.05  # minimum global noise std
    noise_std_max = 0.35   # maximum global noise std
    ksize = 3             # Gaussian kernel size
    blur_sigma = 7.0      # Gaussian blur sigma
    times = 300           # number of times to blur
    add_noise_to_smoothed = False

    create_inverted_intensity_noise_dataset(
        input_folder=input_folder,
        output_folder=output_folder,
        noise_std_min=noise_std_min,
        noise_std_max=noise_std_max,
        ksize=ksize,
        blur_sigma=blur_sigma,
        times=times,
        add_noise_to_smoothed=add_noise_to_smoothed
    )

if __name__ == "__main__":
    main()
