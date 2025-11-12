import os
import cv2
import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


############################################
# Utility Functions
############################################

def repeated_gaussian_smoothing(img, ksize=3, sigma=1.0, times=5):
    """
    Repeatedly applies Gaussian blur to a 2D float32 image in [0,1].
    Returns the smoothed image.
    """
    import cv2
    smoothed = img.copy()
    for _ in range(times):
        smoothed = cv2.GaussianBlur(smoothed, (ksize, ksize), sigma)
    return smoothed

def load_model(model_path, device='cuda'):
    """
    Load a Restormer-like model from a .pth file.
    Modify or replace to match your actual architecture.
    """
    from basicsr.models.archs.restormer_arch import Restormer
    net_g = Restormer(
        inp_channels=2,  # we feed grayscale + noise map -> 2 channels
        out_channels=1,
        dim=48,
        num_blocks=[4,6,6,8],
        num_refinement_blocks=4,
        heads=[1,2,4,8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='BiasFree',
        dual_pixel_task=False
    )
    net_g.to(device)
    ckpt = torch.load(model_path, map_location=device)
    net_g.load_state_dict(ckpt['params'], strict=True)
    net_g.eval()
    return net_g


############################################
# Inference / Main Logic
############################################

def run_inference(
    model_pth,
    input_path,
    output_folder,
    noise_std=0.1,
    invert=False,
    apply_smoothing=True,
    times=5,
    ksize=3,
    sigma=1.0,
    device='cuda'
):
    """
    1) Load .pth model
    2) For each clean grayscale image in input_path:
       - Possibly apply repeated smoothing, optionally invert intensities
       - Multiply by fixed noise_std => noise_map
       - Add noise to original => noised_img
       - Feed (noised_img + noise_map) => model => denoised
       - Compute residual = denoised - clean
       - Save noised_img, denoised, and residual distribution overlay plot
    """
    os.makedirs(output_folder, exist_ok=True)
    noised_dir = os.path.join(output_folder, "noised")
    denoised_dir = os.path.join(output_folder, "denoised")
    plots_dir = os.path.join(output_folder, "plots")
    os.makedirs(noised_dir, exist_ok=True)
    os.makedirs(denoised_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = load_model(model_pth, device=device)

    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    input_files = []
    if os.path.isfile(input_path) and input_path.lower().endswith(valid_exts):
        input_files = [input_path]
    elif os.path.isdir(input_path):
        for fn in os.listdir(input_path):
            if fn.lower().endswith(valid_exts):
                input_files.append(os.path.join(input_path, fn))
        input_files.sort()
    else:
        print(f"Error: {input_path} is neither a valid image nor folder.")
        return

    with torch.no_grad():
        for fpath in input_files:
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: failed to read {fpath}. Skipping.")
                continue

            # convert to float32 [0,1]
            clean_img = img.astype(np.float32) / 255.0

            # possibly apply repeated smoothing => smooth_map
            if apply_smoothing:
                smooth_map = repeated_gaussian_smoothing(clean_img, ksize=ksize, sigma=sigma, times=times)
                # re-normalize
                sm_min, sm_max = smooth_map.min(), smooth_map.max()
                if sm_max - sm_min > 1e-8:
                    smooth_map = (smooth_map - sm_min) / (sm_max - sm_min)
                else:
                    smooth_map = np.clip(smooth_map, 0, 1)
            else:
                smooth_map = clean_img.copy()

            # optionally invert
            if invert:
                smooth_map = 1.0 - smooth_map

            # create noise map
            noise_map = smooth_map * noise_std

            # add noise to the original clean image
            noise = np.random.randn(*noise_map.shape).astype(np.float32) * noise_map
            noised_img = clean_img + noise
            noised_img = np.clip(noised_img, 0, 1)

            # forward pass: lq -> model
            # shape => (2, H, W) => 1 channel noised + 1 channel noise_map
            inp_noised = torch.from_numpy(noised_img).unsqueeze(0)  # shape (1, H, W)
            inp_map    = torch.from_numpy(noise_map).unsqueeze(0)   # shape (1, H, W)
            inp = torch.cat([inp_noised, inp_map], dim=0).unsqueeze(0).to(device)  # (1, 2, H, W)

            _, _, h, w = inp.shape
            pad_h = (2 - h % 2) % 2
            pad_w = (2 - w % 2) % 2
            if pad_h > 0 or pad_w > 0:
                inp = F.pad(inp, (0, pad_w, 0, pad_h), mode='reflect')

            denoised = model(inp)

            if pad_h > 0 or pad_w > 0:
                denoised = denoised[..., :h, :w]

            denoised = torch.clamp(denoised, 0, 1)

            # Convert back to CPU numpy
            denoised_img = denoised.squeeze(0).squeeze(0).cpu().numpy()  # (H, W)

            residual = noised_img - denoised_img

            # Save noised, denoised, and residual images
            base_name = os.path.splitext(os.path.basename(fpath))[0]
            suffix = "_invert" if invert else "_normal"
            noised_path = os.path.join(noised_dir, f"{base_name}{suffix}_noised.png")
            denoised_path = os.path.join(denoised_dir, f"{base_name}{suffix}_denoised.png")
            residual_path = os.path.join(plots_dir, f"{base_name}{suffix}_residual.png")
            cv2.imwrite(noised_path, (noised_img * 255).astype(np.uint8))
            cv2.imwrite(denoised_path, (denoised_img * 255).astype(np.uint8))

            # Normalize residual for visualization and save
            res_min, res_max = residual.min(), residual.max()
            normalized_residual = (residual - res_min) / (res_max - res_min)
            cv2.imwrite(residual_path, (normalized_residual * 255).astype(np.uint8))

            # Create residual histogram vs noise distribution
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(residual.flatten(), bins=100, alpha=0.6, density=True, label="Residual Distribution")

            ax.hist(noise.flatten(), bins=100, alpha=0.6, density=True, label=f"Noise Distribution (std={noise_std})")

            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.set_title(f"Residual vs Noise - {base_name}{suffix}")
            ax.grid(True)
            ax.legend()
            plot_path = os.path.join(plots_dir, f"{base_name}{suffix}_residual_histogram.png")
            plt.savefig(plot_path)
            plt.close()

            print(f"Processed {fpath}: noise_std={noise_std}, invert={invert}")
            print(f"  -> Noised saved to: {noised_path}")
            print(f"  -> Denoised saved to: {denoised_path}")
            print(f"  -> Residual image saved to: {residual_path}")
            print(f"  -> Residual histogram saved to: {plot_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_pth', type=str, required=True,
                        help='Path to the trained .pth model file')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input image or folder containing images')
    parser.add_argument('--output_folder', type=str, default='./results_infer',
                        help='Directory to save results')
    parser.add_argument('--noise_std', type=float, default=0.1,
                        help='Fixed noise standard deviation to add')
    parser.add_argument('--invert', action='store_true',
                        help='Invert the intensity map for higher noise in dark areas')
    parser.add_argument('--apply_smoothing', action='store_true',
                        help='Apply repeated Gaussian smoothing to generate a smooth map')
    parser.add_argument('--times', type=int, default=5,
                        help='Number of times to apply smoothing')
    parser.add_argument('--ksize', type=int, default=3,
                        help='Gaussian kernel size')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='Gaussian sigma')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Computation device, e.g., cuda or cpu')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    run_inference(
        model_pth=args.model_pth,
        input_path=args.input_path,
        output_folder=args.output_folder,
        noise_std=args.noise_std,
        invert=args.invert,
        apply_smoothing=args.apply_smoothing,
        times=args.times,
        ksize=args.ksize,
        sigma=args.sigma,
        device=args.device
    )

if __name__ == "__main__":
    main()
