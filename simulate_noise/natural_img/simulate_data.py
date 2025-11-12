#!/usr/bin/env python3
import os
import glob
import argparse

import numpy as np
import cv2
from PIL import Image

# Fixed parameters (match your training setup)
SNR_LEVELS       = [5, 10, 15, 20]
CROP_SIZE        = 256
SMOOTH_KSIZE     = 3
SMOOTH_SIGMA     = 7.0
SMOOTH_TIMES     = 300
WHOLE_NOISE_STD  = 0.01    # global noise std used during training


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def repeated_gaussian_smoothing(img, ksize, sigma, times):
    out = img.copy()
    for _ in range(times):
        out = cv2.GaussianBlur(out, (ksize, ksize), sigma)
    return out


def center_crop(img, size):
    h, w = img.shape
    top  = (h - size) // 2
    left = (w - size) // 2
    return img[top:top+size, left:left+size]


def compute_sigma(img, snr_db):
    """
    Compute the target noise standard deviation for a given SNR:
        σ = sqrt( P_signal / 10^(SNR/10) )
    where P_signal = mean(img^2).
    """
    P_signal = np.mean(img**2)
    P_noise  = P_signal / (10**(snr_db / 10))
    return np.sqrt(P_noise)


def main():
    parser = argparse.ArgumentParser(
        description="Generate 256×256 noisy images + σ‐maps using training noise model"
    )
    parser.add_argument(
        '--clean_dir', required=True,
        help='Folder of clean .png/.jpg images'
    )
    parser.add_argument(
        '--output_dir', default='./synth_noisy',
        help='Where to write outputs'
    )
    args = parser.parse_args()

    # Gather all clean images
    img_paths = sorted(
        glob.glob(os.path.join(args.clean_dir, '*.png')) +
        glob.glob(os.path.join(args.clean_dir, '*.jpg'))
    )
    if not img_paths:
        raise RuntimeError(f"No .png/.jpg found in {args.clean_dir}")

    # Prepare output directories
    for snr in SNR_LEVELS:
        ensure_dir(os.path.join(args.output_dir, f'SNR_{snr}', 'noisy'))
        ensure_dir(os.path.join(args.output_dir, f'SNR_{snr}', 'noise_maps'))
    clean_crop_dir = os.path.join(args.output_dir, 'clean_crops')
    ensure_dir(clean_crop_dir)

    # Process each clean image
    for img_path in img_paths:
        name, _ = os.path.splitext(os.path.basename(img_path))
        clean = np.array(Image.open(img_path).convert('L'), dtype=np.float32) / 255.0
        h, w = clean.shape
        if h < CROP_SIZE or w < CROP_SIZE:
            print(f"Skipping '{name}' ({h}×{w} < {CROP_SIZE}×{CROP_SIZE})")
            continue

        # Center‐crop
        clean_crop = center_crop(clean, CROP_SIZE)

        # Save cropped clean image
        clean_img = (clean_crop * 255.0).round().astype(np.uint8)
        Image.fromarray(clean_img).save(
            os.path.join(clean_crop_dir, f'{name}.png')
        )

        # For each SNR level...
        for snr in SNR_LEVELS:
            # 1) Compute total σ from SNR
            sigma_total = compute_sigma(clean_crop, snr)

            # 2) Smooth the clean crop
            sm = repeated_gaussian_smoothing(
                clean_crop,
                ksize=SMOOTH_KSIZE,
                sigma=SMOOTH_SIGMA,
                times=SMOOTH_TIMES
            )
            sm = np.clip(sm, 0, 1)

            # 3) Define global‐noise σ (constant)
            global_std = WHOLE_NOISE_STD

            # 4) Calibrate spatial factor so that Var(spatial_noise)+Var(global_noise)=σ_total^2
            M = np.mean(sm**2)
            spatial_factor = np.sqrt(
                max(sigma_total**2 - global_std**2, 0.0) / M
            )

            # 5) Build σ‐maps
            spatial_map = sm * spatial_factor                   # per-pixel σ_spatial
            global_map  = np.full_like(clean_crop, global_std)  # per-pixel σ_global
            noise_map   = np.sqrt(spatial_map**2 + global_map**2)
            noise_map   = np.clip(noise_map, 0, 1)  # for 8-bit storage

            # 6) Draw noise realizations and add
            smoothing_noise = (np.random.randn(*clean_crop.shape).astype(np.float32)
                               * spatial_map)
            whole_noise = (np.random.randn(*clean_crop.shape).astype(np.float32)
                           * global_map)
            noisy = np.clip(clean_crop + smoothing_noise + whole_noise, 0, 1)

            # 7) Save noisy image
            noisy_img = (noisy * 255.0).round().astype(np.uint8)
            Image.fromarray(noisy_img).save(
                os.path.join(args.output_dir, f'SNR_{snr}', 'noisy', f'{name}.png')
            )

            # 8) Save σ‐map
            sigma_img = (noise_map * 255.0).round().astype(np.uint8)
            Image.fromarray(sigma_img).save(
                os.path.join(args.output_dir, f'SNR_{snr}', 'noise_maps', f'{name}.png')
            )

    print(f"Done! Generated noisy images, σ-maps, and clean crops under '{args.output_dir}'")

if __name__ == '__main__':
    main()
