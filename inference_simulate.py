#!/usr/bin/env python3
import os
import glob
import argparse

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips

from basicsr.models.archs.restormer_arch import Restormer

# -------------------
# Helpers
# -------------------
def load_nifti(path):
    return nib.load(path).get_fdata().astype(np.float32)

def pad_to_8(x):
    # x is a torch.Tensor of shape (1, C, H, W)
    _, _, h, w = x.shape
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode='reflect')
    return x, h, w

def build_model(model_pth: str, inp_ch: int, device: torch.device) -> torch.nn.Module:
    """
    Load Restormer and map checkpoint to `device` (CPU or GPU).
    """
    # 1) load checkpoint onto the chosen device
    ckpt = torch.load(model_pth, map_location=device)

    # 2) instantiate network and move to device
    net = Restormer(
        inp_channels=inp_ch,
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

    # 3) load weights
    net.load_state_dict(ckpt['params'], strict=True)
    net.eval()
    return net

# -------------------
# Main
# -------------------
def main():
    parser = argparse.ArgumentParser(
        description="Batch inference + metrics over simulated SNR data"
    )
    parser.add_argument(
        '--results_folder', required=True,
        help='Top-level folder containing <image>/snrXX/csm/'
    )
    parser.add_argument(
        '--model_pth', required=True,
        help='Restormer checkpoint (.pth)'
    )
    parser.add_argument(
        '--use_noise', action='store_true',
        help='If set, load noise_map_stack.nii as 2nd input channel'
    )
    args = parser.parse_args()

    # ---- choose device only if there's at least one GPU ----
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    inp_ch = 2 if args.use_noise else 1

    # build Restormer + LPIPS
    model    = build_model(args.model_pth, inp_ch, device)
    lpips_fn = lpips.LPIPS(net='alex').to(device)

    # discover all images and SNR levels
    sample_dirs = sorted(glob.glob(os.path.join(args.results_folder, '*')))
    snr_levels = set()
    for sd in sample_dirs:
        for sd_snr in glob.glob(os.path.join(sd, 'snr*')):
            try:
                snr_levels.add(int(os.path.basename(sd_snr).lstrip('snr')))
            except ValueError:
                pass
    snr_levels = sorted(snr_levels)

    summary = {}
    for snr in snr_levels:
        all_psnr, all_ssim, all_lpips, all_logres2 = [], [], [], []

        for sample in sample_dirs:
            csm_dir = os.path.join(sample, f'snr{snr}', 'csm')
            if not os.path.isdir(csm_dir):
                continue

            # locate the key files
            noisy_fp   = glob.glob(os.path.join(csm_dir, '*noisy_stack.nii'))[0]
            cleancc_fp = glob.glob(os.path.join(csm_dir, '*clean_cc.nii'))[0]
            noisy_stack = load_nifti(noisy_fp)      # (H, W, S, C)
            clean_cc    = load_nifti(cleancc_fp)    # (H, W, S)

            if args.use_noise:
                noise_fp = glob.glob(os.path.join(csm_dir, '*noise_map_stack.nii'))[0]
                noise_map_stack = load_nifti(noise_fp)  # (H, W, S, C)
            else:
                noise_map_stack = None

            H, W, S, C = noisy_stack.shape
            deno_cc = np.zeros((H, W, S), np.float32)

            # slice-by-slice inference
            for z in range(S):
                percoil = []
                for c in range(C):
                    img_c = noisy_stack[..., z, c]
                    arr   = img_c[np.newaxis, ...]
                    if args.use_noise:
                        nm = noise_map_stack[..., z, c]
                        arr = np.stack([arr[0], nm], axis=0)
                    inp = torch.from_numpy(arr[None]).to(device)  # (1, C, H, W)
                    inp, h, w = pad_to_8(inp)
                    with torch.no_grad():
                        out = model(inp)
                    out = out[..., :h, :w].cpu().squeeze().numpy()
                    percoil.append(out)
                stack_out = np.stack(percoil, axis=-1)
                deno_cc[..., z] = np.sqrt((stack_out ** 2).sum(axis=-1))

            # compute metrics per slice
            for z in range(S):
                gt   = clean_cc[..., z]
                pred = deno_cc[...,   z]

                all_psnr.append(peak_signal_noise_ratio(gt, pred, data_range=1.0))
                all_ssim.append(structural_similarity(gt, pred, data_range=1.0))

                # LPIPS: replicate to 3 channels in [-1,1]
                t_gt    = torch.from_numpy(gt[None,None]).to(device).repeat(1,3,1,1)*2-1
                t_pred  = torch.from_numpy(pred[None,None]).to(device).repeat(1,3,1,1)*2-1
                all_lpips.append(lpips_fn(t_gt, t_pred).item())

                res2 = (pred - gt)**2
                all_logres2.append(np.mean(np.log(res2 + 1e-12)))

        # aggregate for this SNR
        summary[snr] = {
            'PSNR'     : float(np.mean(all_psnr)),
            'SSIM'     : float(np.mean(all_ssim)),
            'LPIPS'    : float(np.mean(all_lpips)),
            'MeanLogR²': float(np.mean(all_logres2)),
            'samples'  : len(all_psnr)
        }

    # write CSV-style summary
    out_fp = os.path.join(args.results_folder, 'metrics_summary.txt')
    with open(out_fp, 'w') as f:
        f.write("SNR_dB,PSNR,SSIM,LPIPS,MeanLogRes²,samples\n")
        for snr in snr_levels:
            r = summary[snr]
            f.write(f"{snr},{r['PSNR']:.4f},{r['SSIM']:.4f},"
                    f"{r['LPIPS']:.4f},{r['MeanLogR²']:.6f},{r['samples']}\n")

    print("Done. Metrics summary saved to:", out_fp)


if __name__ == "__main__":
    main()
