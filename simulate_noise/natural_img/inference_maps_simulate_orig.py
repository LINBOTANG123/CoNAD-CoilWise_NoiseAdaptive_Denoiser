#!/usr/bin/env python3
import os
import argparse
import glob

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
import pandas as pd

# --- 1) load_model (single-channel Restormer) ---
def load_model(model_path, device='cuda'):
    from basicsr.models.archs.restormer_arch import Restormer
    net = Restormer(
        inp_channels=1,
        out_channels=1,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='BiasFree',
        dual_pixel_task=False
    ).to(device)
    ckpt = torch.load(model_path, map_location=device)
    net.load_state_dict(ckpt['params'], strict=True)
    net.eval()
    return net

# --- 2) helper to pad/crop to multiples of 8 ---
def pad_to_multiple_of_8(x):
    _, _, h, w = x.size()
    ph = (8 - h % 8) % 8
    pw = (8 - w % 8) % 8
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode='reflect')
    return x, h, w


def main():
    parser = argparse.ArgumentParser(description="Single-channel denoising + metrics")
    parser.add_argument('--model_pth',   required=True, help='Restormer .pth checkpoint')
    parser.add_argument('--input_dir',   required=True, help='Folder of noisy .png/.jpg')
    parser.add_argument('--clean_dir',   required=True, help='Folder of clean GT images')
    parser.add_argument('--output_dir',  default='./results_single', help='Where to save outputs')
    parser.add_argument('--device',      default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # make output dirs
    denoised_dir = os.path.join(args.output_dir, 'denoised')
    os.makedirs(denoised_dir, exist_ok=True)

    # load LPIPS (alex backbone)
    loss_lpips = lpips.LPIPS(net='alex').to(args.device)

    # load single-channel Restormer
    model = load_model(args.model_pth, device=args.device)

    # gather image list
    img_paths = sorted(glob.glob(os.path.join(args.input_dir, '*.png')) +
                       glob.glob(os.path.join(args.input_dir, '*.jpg')))

    results = []
    for img_path in img_paths:
        name, ext = os.path.splitext(os.path.basename(img_path))
        # load noisy and clean
        noisy_path = img_path
        clean_path = None
        for e in ('.png', '.jpg', '.jpeg'):
            candidate = os.path.join(args.clean_dir, name + e)
            if os.path.isfile(candidate):
                clean_path = candidate
                break
        if clean_path is None:
            raise FileNotFoundError(f"No clean image found for '{name}' in {args.clean_dir}")

        noisy = np.array(Image.open(noisy_path).convert('L'), dtype=np.float32) / 255.0
        clean = np.array(Image.open(clean_path).convert('L'), dtype=np.float32) / 255.0

        H, W = noisy.shape
        # stack → [1,1,H,W]
        inp = torch.from_numpy(noisy[None, None, :, :]).to(args.device)

        inp, h0, w0 = pad_to_multiple_of_8(inp)
        with torch.no_grad():
            out = model(inp)
            out = out[..., :h0, :w0]
            # debias
            bias = (out - inp[..., :h0, :w0]).mean()
            out = out - bias

        deno = out.cpu().squeeze().numpy()  # [H,W]

        # save PNG
        deno_img = (np.clip(deno, 0, 1) * 255).round().astype(np.uint8)
        Image.fromarray(deno_img).save(os.path.join(denoised_dir, name + '.png'))

        # compute PSNR & SSIM
        psnr = peak_signal_noise_ratio(clean, deno, data_range=1.0)
        ssim = structural_similarity(clean, deno, data_range=1.0)

        # compute LPIPS
        def to_lpips_tensor(x):
            x3 = np.tile(x[None], (3, 1, 1))
            t = torch.from_numpy(x3).unsqueeze(0).to(args.device)
            return t * 2 - 1
        lpips_val = loss_lpips(to_lpips_tensor(deno), to_lpips_tensor(clean)).item()

        results.append({'filename': name,
                        'PSNR': psnr,
                        'SSIM': ssim,
                        'LPIPS': lpips_val})

    # aggregate to CSV
    df = pd.DataFrame(results)
    print(df.describe().loc[['mean', 'std']])
    df.to_csv(os.path.join(args.output_dir, 'metrics.csv'), index=False)
    print("Saved denoised images →", denoised_dir)
    print("Saved metrics CSV →", args.output_dir + '/metrics.csv')

if __name__ == '__main__':
    main()
