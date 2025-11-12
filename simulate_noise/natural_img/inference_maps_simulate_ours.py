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
import matplotlib.pyplot as plt

# --- 1) load_model (copied from your inference code) ---
def load_model(model_path, inp_channels, device='cuda'):
    from basicsr.models.archs.restormer_arch import Restormer
    net = Restormer(
        inp_channels=inp_channels,
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
    p = argparse.ArgumentParser(description="Synthetic‑noise denoising + metrics with stats")
    p.add_argument('--model_pth',   required=True, help='Restormer .pth checkpoint')
    p.add_argument('--input_dir',   required=True, help='Folder of noisy .png/.jpg')
    p.add_argument('--noise_dir',   required=True, help='Folder of corresponding noise maps')
    p.add_argument('--clean_dir',   required=True, help='Folder of clean GT images')
    p.add_argument('--output_dir',  default='./results_synth', help='Where to save outputs')
    p.add_argument('--device',      default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    # make output dirs
    denoised_dir = os.path.join(args.output_dir, 'denoised')
    os.makedirs(denoised_dir, exist_ok=True)
    histo_dir = os.path.join(args.output_dir, 'hists')
    os.makedirs(histo_dir, exist_ok=True)

    # (Optional) aggregate arrays across all images for one big overlay at the end
    agg_z_true, agg_z_resid = [], []

    # load LPIPS (alex backbone)
    loss_lpips = lpips.LPIPS(net='alex').to(args.device)

    # load your Restormer
    model = load_model(args.model_pth, inp_channels=2, device=args.device)

    # gather image list
    img_paths = sorted(glob.glob(os.path.join(args.input_dir, '*.png')) +
                       glob.glob(os.path.join(args.input_dir, '*.jpg')))

    results = []
    for img_path in img_paths:
        name, ext = os.path.splitext(os.path.basename(img_path))
        # load noisy image
        noisy = np.array(Image.open(img_path).convert('L'), dtype=np.float32) / 255.0
        # load noise map
        noise_map = np.array(
            Image.open(os.path.join(args.noise_dir, name + ext)).convert('L'),
            dtype=np.float32) / 255.0
        # robustly load clean image (.png/.jpg/.jpeg)
        clean_path = None
        for e in ('.png', '.jpg', '.jpeg'):
            candidate = os.path.join(args.clean_dir, name + e)
            if os.path.isfile(candidate):
                clean_path = candidate
                break
        if clean_path is None:
            raise FileNotFoundError(f"No clean image found for '{name}' in {args.clean_dir}")
        clean = np.array(Image.open(clean_path).convert('L'), dtype=np.float32) / 255.0

        # print stats
        print(f"[{name}] noisy   mean={noisy.mean():.4f}, std={noisy.std():.4f}")
        print(f"[{name}] noise_map mean={noise_map.mean():.4f}, std={noise_map.std():.4f}")

        H, W = noisy.shape
        # stack → [1,2,H,W]
        inp = torch.from_numpy(np.stack([noisy, noise_map], 0))[None].to(args.device)

        inp, h0, w0 = pad_to_multiple_of_8(inp)
        with torch.no_grad():
            out = model(inp)
            out = out[..., :h0, :w0]
            # debias (as in your script)
            # bias = (out - inp[:, 0:1, :h0, :w0]).mean()
            # out = out - bias

        deno = out.cpu().squeeze().numpy()  # [H,W]


        # --- Residuals vs true noise (raw) ---
        residual   = noisy - deno            # model residual
        true_noise = noisy - clean           # true injected noise

        # Mask out clipped pixels (optional, to avoid boundary contamination)
        noisy_u8 = np.array(Image.open(img_path).convert('L'), dtype=np.uint8)
        valid = (noisy_u8 > 0) & (noisy_u8 < 255)
        r_valid = residual[valid]
        n_valid = true_noise[valid]
        y_valid = noisy[valid]   # add noisy input values

        # Choose bin edges based on joint range
        vals_for_edges = np.concatenate([arr for arr in (r_valid, n_valid, y_valid) if arr.size])
        if vals_for_edges.size:
            lo = np.percentile(vals_for_edges, 0.1)
            hi = np.percentile(vals_for_edges, 99.9)
            edges = np.linspace(lo, hi, 201)
        else:
            edges = 201

        # --- Plot residual vs true noise ---
        plt.figure(figsize=(6,4))
        if n_valid.size:
            plt.hist(n_valid, bins=edges, density=True, alpha=0.5, label='true noise')
        if r_valid.size:
            plt.hist(r_valid, bins=edges, density=True, alpha=0.5, label='residual')
        plt.title(f"{name}: residual vs true noise")
        plt.xlabel("intensity"); plt.ylabel("density"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(histo_dir, f"hist_residual_vs_true_{name}.png"), dpi=150)
        plt.close()

        # --- Plot noisy input vs true noise (NEW) ---
        plt.figure(figsize=(6,4))
        if n_valid.size:
            plt.hist(n_valid, bins=edges, density=True, alpha=0.5, label='true noise')
        if y_valid.size:
            plt.hist(y_valid, bins=edges, density=True, alpha=0.5, label='noisy input')
        plt.title(f"{name}: noisy input vs true noise")
        plt.xlabel("intensity"); plt.ylabel("density"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(histo_dir, f"hist_noisy_vs_true_{name}.png"), dpi=150)
        plt.close()




        # save PNG
        deno_img = (np.clip(deno, 0, 1) * 255).round().astype(np.uint8)
        Image.fromarray(deno_img).save(os.path.join(denoised_dir, name + '.png'))

        # compute PSNR & SSIM
        psnr = peak_signal_noise_ratio(clean, deno, data_range=1.0)
        ssim = structural_similarity(clean, deno, data_range=1.0)

        # compute LPIPS (replicate channel → [1,3,H,W], map [0,1]→[-1,1])
        def to_lpips_tensor(x):
            x3 = np.tile(x[None], (3, 1, 1))
            t = torch.from_numpy(x3).unsqueeze(0).to(args.device)
            return (t * 2 - 1)
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
