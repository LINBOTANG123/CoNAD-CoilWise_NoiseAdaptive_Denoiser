#!/usr/bin/env python3
import os
import argparse
from glob import glob

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import torch
import lpips  # pip install lpips

def load_image(path):
    """Load image as float32 array in [0,1], RGB."""
    img = Image.open(path).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0
    return arr

def tensor_for_lpips(img_np):
    """Convert H×W×C [0,1] numpy → torch tensor 1×3×H×W in [−1,1]."""
    t = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0)
    return t * 2.0 - 1.0

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main(args):
    # prepare LPIPS model (AlexNet backbone)
    lpips_fn = lpips.LPIPS(net='alex').to('cuda' if torch.cuda.is_available() else 'cpu')
    
    denoised_paths = sorted(glob(os.path.join(args.denoised_dir, '*')))
    clean_dir   = args.clean_dir
    noisy_dir   = args.noisy_dir
    out_res_dir = args.residual_dir
    ensure_dir(out_res_dir)
    
    psnr_list, ssim_list, lpips_list = [], [], []
    
    for den_path in denoised_paths:
        name = os.path.basename(den_path)
        clean_path = os.path.join(clean_dir, name)
        noisy_path = os.path.join(noisy_dir, name)
        if not (os.path.isfile(clean_path) and os.path.isfile(noisy_path)):
            print(f"Skipping {name}: missing clean or noisy counterpart.")
            continue
        
        # load
        den = load_image(den_path)
        clean = load_image(clean_path)
        noisy = load_image(noisy_path)
        
        # PSNR & SSIM (range=1.0)
        p = peak_signal_noise_ratio(clean, den, data_range=1.0)
        s = structural_similarity(clean, den, data_range=1.0, multichannel=True)
        
        # LPIPS
        den_t   = tensor_for_lpips(den).to('cuda')
        clean_t = tensor_for_lpips(clean).to('cuda')
        l = lpips_fn(den_t, clean_t).item()
        
        psnr_list.append(p)
        ssim_list.append(s)
        lpips_list.append(l)
        
        print(f"{name} → PSNR: {p:.4f}, SSIM: {s:.4f}, LPIPS: {l:.4f}")
        
        # Residual
        resid = noisy - den
        r_min, r_max = resid.min(), resid.max()
        r_mean, r_std = resid.mean(), resid.std()
        print(f"  Residual stats → min: {r_min:.4e}, max: {r_max:.4e}, "
              f"mean: {r_mean:.4e}, std: {r_std:.4e}")
        
        # Save absolute residual
        abs_resid = np.clip(np.abs(resid), 0, 1)
        res_img = (abs_resid * 255).astype(np.uint8)
        res_pil = Image.fromarray(res_img)
        res_pil.save(os.path.join(out_res_dir, name))
    
    # overall averages
    if psnr_list:
        print("\n=== AVERAGES ===")
        print(f"Mean PSNR:  {np.mean(psnr_list):.4f}")
        print(f"Mean SSIM:  {np.mean(ssim_list):.4f}")
        print(f"Mean LPIPS: {np.mean(lpips_list):.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate denoised vs clean: PSNR, SSIM, LPIPS; compute & save residuals"
    )
    parser.add_argument('--denoised_dir', required=True)
    parser.add_argument('--clean_dir',    required=True)
    parser.add_argument('--noisy_dir',    required=True)
    parser.add_argument('--residual_dir', required=True,
                        help="where to save absolute residual PNGs")
    args = parser.parse_args()
    main(args)
