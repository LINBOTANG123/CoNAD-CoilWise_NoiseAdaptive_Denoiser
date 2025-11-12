import os
import cv2
import numpy as np

def repeated_gaussian_smoothing(img, ksize=3, sigma=1.0, times=5):
    """
    Repeatedly applies Gaussian blur to an image.

    Args:
        img   (np.ndarray): 2D or 3D NumPy array (grayscale or color).
        ksize       (int): Gaussian kernel size (e.g., 3 or 5).
        sigma     (float): Standard deviation for Gaussian kernel.
        times       (int): Number of times to apply the blur.

    Returns:
        np.ndarray: The smoothed image.
    """
    smoothed = img.copy()
    for _ in range(times):
        smoothed = cv2.GaussianBlur(smoothed, (ksize, ksize), sigma)
    return smoothed

def create_synthetic_dataset(input_folder, smoothed_folder, noisy_folder,
                             ksize=3, sigma=1.0, times=5, noise_std=0.05):
    """
    Creates a synthetic dataset by repeatedly smoothing images, normalizing to [0,1],
    then adding Gaussian noise. Saves both smoothed and noisy images.

    Args:
        input_folder    (str): Path to the folder containing input images.
        smoothed_folder (str): Path to save the smoothed images.
        noisy_folder    (str): Path to save the noisy images.
        ksize           (int): Gaussian kernel size for smoothing.
        sigma         (float): Gaussian kernel standard deviation.
        times           (int): Number of times to apply Gaussian blur.
        noise_std     (float): Standard deviation for the added Gaussian noise.

    Returns:
        None: Processed images are saved to specified folders.
    """
    # Create output directories if they don't exist
    os.makedirs(smoothed_folder, exist_ok=True)
    os.makedirs(noisy_folder, exist_ok=True)

    # Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in image_extensions:
            continue  # Skip non-image files

        file_path = os.path.join(input_folder, filename)
        # Read as grayscale; comment out or adjust if you need color images
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: unable to read {file_path}. Skipping.")
            continue

        # Convert to float32 [0, 1]
        img_float = img.astype(np.float32) / 255.0

        # Repeated smoothing
        smoothed = repeated_gaussian_smoothing(img_float, ksize, sigma, times)

        # Save smoothed image (scaled back to 8-bit)
        smoothed_uint8 = (smoothed * 255).astype(np.uint8)
        smoothed_path = os.path.join(smoothed_folder, filename)
        cv2.imwrite(smoothed_path, smoothed_uint8)

        # Add Gaussian noise
        noise = np.random.normal(loc=0.0, scale=noise_std, size=smoothed.shape)
        noised_img = smoothed + noise

        # Clip to [0, 1]
        noised_img = np.clip(noised_img, 0, 1)

        # Save noisy image (scaled back to 8-bit)
        noised_uint8 = (noised_img * 255).astype(np.uint8)
        noisy_path = os.path.join(noisy_folder, filename)
        cv2.imwrite(noisy_path, noised_uint8)

        print(f"Processed {filename} -> Smoothed: {smoothed_path}, Noisy: {noisy_path}")

def main():
    # Example usage
    input_folder = "/home/lin/Research/denoise/Restormer/dataset/test_gray_imgs"
    smoothed_folder = "./synthetic_smoothed"
    noisy_folder = "./synthetic_noisy"

    # Parameters for smoothing and noise
    ksize = 3       # Gaussian kernel size
    sigma = 7.0     # Gaussian sigma
    times = 300       # Number of times to apply smoothing
    noise_std = 0.1  # Standard deviation of added noise

    create_synthetic_dataset(input_folder, smoothed_folder, noisy_folder,
                             ksize=ksize, sigma=sigma, times=times, noise_std=noise_std)

if __name__ == "__main__":
    main()
