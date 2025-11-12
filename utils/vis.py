import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_metrics(clean_folder, restored_folder, noised_folder, output_folder, noise_std=25):
    os.makedirs(output_folder, exist_ok=True)

    clean_files = sorted(os.listdir(clean_folder))
    restored_files = sorted(os.listdir(restored_folder))
    noised_files = sorted(os.listdir(noised_folder))

    assert len(clean_files) == len(restored_files) == len(noised_files), "Mismatch in number of files"

    psnr_values = []
    ssim_values = []

    metrics_file = os.path.join(output_folder, "metrics.txt")

    with open(metrics_file, "w") as f:
        f.write("File-wise PSNR and SSIM Metrics:\n")
        f.write("================================\n")

        for clean_file, restored_file, noised_file in zip(clean_files, restored_files, noised_files):
            clean_path = os.path.join(clean_folder, clean_file)
            restored_path = os.path.join(restored_folder, restored_file)
            noised_path = os.path.join(noised_folder, noised_file)

            clean_img = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            restored_img = cv2.imread(restored_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            noised_img = cv2.imread(noised_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

            # Calculate PSNR and SSIM
            psnr_value = psnr(clean_img, restored_img, data_range=1.0)
            ssim_value = ssim(clean_img, restored_img, data_range=1.0)
            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)

            # Save metrics to file
            f.write(f"{clean_file} - PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.3f}\n")

            # Calculate residual map
            residual = restored_img - noised_img

            # Save residual map
            residual_map_path = os.path.join(output_folder, f"{os.path.splitext(clean_file)[0]}_residual.png")
            residual_img = ((residual - residual.min()) / (residual.max() - residual.min()) * 255).astype(np.uint8)
            cv2.imwrite(residual_map_path, residual_img)

            # Plot and save residual distribution
            residual_flat = residual.flatten()
            plt.figure(figsize=(10, 6))
            plt.hist(residual_flat, bins=100, alpha=0.6, label="Residual Distribution", density=True)

            # Add Gaussian noise distribution
            noise = np.random.normal(0, noise_std / 255.0, size=100000)
            plt.hist(noise, bins=100, alpha=0.6, label=f"Gaussian Noise (std={noise_std})", density=True, color='orange')

            plt.xlabel("Value")
            plt.ylabel("Density")
            plt.legend()
            plt.title(f"Residual vs Noise Distribution - {clean_file}")
            plt.grid(True)
            dist_plot_path = os.path.join(output_folder, f"{os.path.splitext(clean_file)[0]}_distribution.png")
            plt.savefig(dist_plot_path)
            plt.close()

            print(f"Processed {clean_file} - PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.3f}")

        # Write average metrics to the file
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        f.write("\nAverage Metrics:\n")
        f.write("================\n")
        f.write(f"Average PSNR: {avg_psnr:.2f}\n")
        f.write(f"Average SSIM: {avg_ssim:.3f}\n")

    print(f"Metrics saved to {metrics_file}. Residual maps and distribution plots saved to {output_folder}.")

# Usage
clean_folder = "/home/lin/Research/denoise/Restormer/results/noise_brain_frac_channel_50/Gaussian_Gray_Denoising"
restored_folder = "/home/lin/Research/denoise/Restormer/results/noise_brain_frac_channel_50/Gaussian_Gray_Denoising"
noised_folder = "/home/lin/Research/denoise/Restormer/dataset/b0_images_val"
output_folder = "results/metrics_brain_frac100k_std50"

calculate_metrics(clean_folder, restored_folder, noised_folder, output_folder, noise_std=80)
