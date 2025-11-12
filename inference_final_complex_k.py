#!/usr/bin/env python
import os
import argparse

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

from utils.mat_loader import load_mri_data
from utils.noise_loader import load_noise_data

NUM_COILS = 32  # adjust if needed


def load_model(model_path: str, inp_channels: int, device: str = 'cuda') -> torch.nn.Module:
    """Load the multi-coil Restormer model with dynamic input channels."""
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


def denoise_channel(model, img: np.ndarray, sigma: np.ndarray, scale: float, device: str, tag: str = ""):
    """
    Denoise one channel (real or imag).
    img:   (H,W) real-valued
    sigma: (H,W) std-map for this channel (or None)
    scale: scalar normalization factor
    """

    # ----- stats before scaling -----
    # print(f"[{tag} BEFORE scale] img: min={img.min():.4g} max={img.max():.4g} "
    #       f"mean={img.mean():.4g} std={img.std():.4g}")
    # if sigma is not None:
    #     print(f"[{tag} BEFORE scale] sigma: min={sigma.min():.4g} max={sigma.max():.4g} "
    #           f"mean={sigma.mean():.4g} std={sigma.std():.4g}")

    # ----- apply scaling -----
    if sigma is not None:
        arr = np.stack([(img * scale).astype(np.float32),
                        (sigma * scale).astype(np.float32)], axis=0)
    else:
        arr = (img * scale)[np.newaxis, ...].astype(np.float32)

    # ----- stats after scaling -----
    # print(f"[{tag} AFTER scale] img: min={arr[0].min():.4g} max={arr[0].max():.4g} "
    #       f"mean={arr[0].mean():.4g} std={arr[0].std():.4g}")
    # if sigma is not None:
    #     print(f"[{tag} AFTER scale] sigma: min={arr[1].min():.4g} max={arr[1].max():.4g} "
    #           f"mean={arr[1].mean():.4g} std={arr[1].std():.4g}")
        
    # if sigma is not None:
    #     ratio = arr[1].mean() / (arr[0].std() + 1e-12)
    #     print(f"[{tag} AFTER scale] ratio sigma.mean/img.std = {ratio:.3g}")

    # ----- model inference -----
    inp = torch.from_numpy(arr).unsqueeze(0).to(device)  # (1,C,H,W)
    _, _, h, w = inp.shape
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    if ph or pw:
        inp_p = F.pad(inp, (0, pw, 0, ph), mode='reflect')
    else:
        inp_p = inp

    with torch.no_grad():
        out = model(inp_p)[..., :h, :w]

    # descale back to original units
    return (out.squeeze().cpu().numpy().astype(np.float32)) / scale

def _robust_sigma_1d(x: np.ndarray, method: str='mad') -> float:
    x = np.asarray(x)
    if x.size == 0: return 0.0
    if method == 'mad':
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        return 1.4826 * mad
    return x.std(ddof=1) if x.size > 1 else 0.0

def _finite_median(a: np.ndarray) -> float:
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else 0.0

def main():
    p = argparse.ArgumentParser(description="Inference for multi-coil MRI denoising.")
    p.add_argument('--model_pth', required=True, help='Restormer checkpoint (.pth)')
    p.add_argument('--mri_mat', required=True, help='Path to the MRI .mat/.h5 or NIfTI')
    p.add_argument('--mri_key', default='image', help='Dataset key inside .mat/HDF5')
    p.add_argument('--mri_format',
                   choices=['Hwihun_phantom', 'b1000', 'C', 'gslider', 'gslider_2', 'simulate'],
                   default='b1000')
    p.add_argument('--noise_mat', required=False,
                   help='Path to noise .mat/.h5 or NIfTI (only if using noise)')
    p.add_argument('--noise_key', default='k_gc',
                   help='Key inside noise .mat/HDF5 (ignored for simulate)')
    p.add_argument('--noise_format',
                   choices=['Hwihun_phantom', 'b1000', 'C', 'gslider', 'gslider_2',
                            'simulate', 'gslider_v5'],
                   default='b1000')
    p.add_argument('--use_noise', action='store_true',
                   help="If set, load and feed a noise‐map as second channel")
    p.add_argument('--output_folder', default='./results_infer',
                   help='Where to save NIfTI outputs')
    p.add_argument('--num_samples', type=int, default=2,
                   help='How many DWI channels to process (for MAT formats)')
    p.add_argument('--dwi_index', type=int, default=None,
                   help='Specify which DWI sample index to denoise (0-based). If None, denoise all samples.')
    p.add_argument('--brain_mask', required=True,
                help='Brain mask NIfTI (.nii/.nii.gz), 1=brain, 0=background')
    p.add_argument('--sigma_estimator', choices=['mad','std'], default='mad',
                help='Background sigma estimator: mad (1.4826*MAD) or std.')


    args = p.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    # --- 1) Load MRI as COMPLEX image space ---
    mri_img = load_mri_data(
        args.mri_mat,
        key=args.mri_key,
        data_format=args.mri_format,
        num_samples_to_load=args.num_samples,
        output_space='complex_image'
    )
    H, W, C, S, N = mri_img.shape
    print("Loaded MRI (complex):", mri_img.shape)

    # --- 1.2) Load brain mask ---
    mask_img = nib.load(args.brain_mask)
    brain_mask = mask_img.get_fdata().astype(bool)  # (H,W,S)
    if brain_mask.shape != (H, W, S):
        raise ValueError(f"Brain mask shape {brain_mask.shape} != {(H,W,S)}")
    bg_mask = ~brain_mask  # background is where mask==0

    # --- 2) Optionally load Noise (complex) ---
    if args.use_noise:
        noise_cplx = load_noise_data(
            args.noise_mat,
            key=args.noise_key,
            data_format=args.noise_format,
            output_space='complex_image'
        )  # (H,W,C,S_noise)
        if noise_cplx.shape[:3] != (H, W, C):
            raise ValueError(f"Noise shape {noise_cplx.shape} does not match MRI {(H,W,C)}")

        # rotate noise map to match
        noise_cplx = np.rot90(noise_cplx, k=3, axes=(0, 1)).copy()

        ddof = 1 if noise_cplx.shape[-1] > 1 else 0
        sigma_real = noise_cplx.real.mean(axis=-1).astype(np.float32)  # (H,W,C)
        sigma_imag = noise_cplx.imag.mean(axis=-1).astype(np.float32)  # (H,W,C)

        # (Optional) smooth σ to avoid grid
        # from scipy.ndimage import gaussian_filter
        # SMOOTH_SIGMA = 1.0
        # for c in range(C):
        #     sigma_real[:, :, c] = gaussian_filter(sigma_real[:, :, c], sigma=SMOOTH_SIGMA, mode='nearest')
        #     sigma_imag[:, :, c] = gaussian_filter(sigma_imag[:, :, c], sigma=SMOOTH_SIGMA, mode='nearest')

        # Single small gain that preserves relative scale (from your background gammas ~0.0051)
        KAPPA = 1.0
        sigma_real *= KAPPA
        sigma_imag *= KAPPA


        # Save σ maps (real & imag components)
        nib.save(nib.Nifti1Image(sigma_real, np.eye(4)),
                os.path.join(args.output_folder, 'sigma_real_raw.nii'))
        nib.save(nib.Nifti1Image(sigma_imag, np.eye(4)),
                os.path.join(args.output_folder, 'sigma_imag_raw.nii'))

        # Coil-mean for a quick sanity check (H,W)
        sigma_real_mean = sigma_real.mean(axis=-1)
        sigma_imag_mean = sigma_imag.mean(axis=-1)
        nib.save(nib.Nifti1Image(sigma_real_mean, np.eye(4)),
                os.path.join(args.output_folder, 'sigma_real_mean.nii'))
        nib.save(nib.Nifti1Image(sigma_imag_mean, np.eye(4)),
                os.path.join(args.output_folder, 'sigma_imag_mean.nii'))

    else:
        sigma_real, sigma_imag = None, None

    # --- 3) Load model ---
    inp_ch = 2 if args.use_noise else 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.model_pth, inp_channels=inp_ch, device=device)

    # --- 4) Denoising Loop ---
    all_deno, all_res, all_orig = [], [], []
    sample_indices = [args.dwi_index] if args.dwi_index is not None else list(range(N))

    for sid in sample_indices:
        print(f"Sample {sid+1}/{N}")
        vol = mri_img[..., sid]                              # (H,W,C,S) complex
        orig_cc = np.sqrt((np.abs(vol) ** 2).sum(axis=2))    # (H,W,S)
        all_orig.append(orig_cc)

        if args.use_noise:
            # Precompute coil-wise σ stds (in original units, after smoothing & KAPPA)
            sig_r_std = sigma_real.std(axis=(0, 1)) + 1e-12  # (C,)
            sig_i_std = sigma_imag.std(axis=(0, 1)) + 1e-12  # (C,)
            TARGET_RATIO = 0.8  # keep σ a bit weaker than the image channel

            alpha_r = np.empty((S, C), dtype=np.float32)
            alpha_i = np.empty((S, C), dtype=np.float32)
            for c in range(C):
                for z in range(S):
                    img_cz = vol[:, :, c, z]
                    img_r_std = img_cz.real.std() + 1e-12
                    img_i_std = img_cz.imag.std() + 1e-12
                    # Clamp to avoid over/under-guidance
                    alpha_r[z, c] = np.clip(TARGET_RATIO * img_r_std / sig_r_std[c], 1e-3, 0.2)
                    alpha_i[z, c] = np.clip(TARGET_RATIO * img_i_std / sig_i_std[c], 1e-3, 0.2)


        # ---- Calibrate external sigma maps to MRI background scale (per COIL) ----
        # Only if you actually use external noise maps
        # if args.use_noise:
        #     sigma_real_cal = np.empty_like(sigma_real, dtype=np.float32)  # (H,W,C)
        #     sigma_imag_cal = np.empty_like(sigma_imag, dtype=np.float32)  # (H,W,C)

        #     for c in range(C):
        #         gammas_r, gammas_i = [], []
        #         for z in range(S):
        #             bg = bg_mask[:, :, z]            # background pixels for this slice
        #             img_c = vol[:, :, c, z]          # complex slice for this coil

        #             # Reference σ from MRI BACKGROUND (complex domain)
        #             sig_r_ref = _robust_sigma_1d(img_c.real[bg], method=args.sigma_estimator)
        #             sig_i_ref = _robust_sigma_1d(img_c.imag[bg], method=args.sigma_estimator)

        #             # Current σ-map scale (use BACKGROUND region for robustness)
        #             # Your sigma_real/sigma_imag are (H,W,C) — no slice dim — so compare on the same bg mask
        #             sig_r_cur = _finite_median(sigma_real[:, :, c][bg]) if sigma_real is not None else 0.0
        #             sig_i_cur = _finite_median(sigma_imag[:, :, c][bg]) if sigma_imag is not None else 0.0

        #             if sig_r_ref > 0 and sig_r_cur > 0:
        #                 gammas_r.append(sig_r_ref / sig_r_cur)
        #             if sig_i_ref > 0 and sig_i_cur > 0:
        #                 gammas_i.append(sig_i_ref / sig_i_cur)

        #         gamma_r = float(np.median(gammas_r)) if len(gammas_r) else 1.0
        #         gamma_i = float(np.median(gammas_i)) if len(gammas_i) else 1.0

        #         sigma_real_cal[:, :, c] = sigma_real[:, :, c] * gamma_r
        #         sigma_imag_cal[:, :, c] = sigma_imag[:, :, c] * gamma_i

        #         print(f"[σ-calib] coil {c:02d}: gamma_real={gamma_r:.3g}, gamma_imag={gamma_i:.3g}")

        #     # Swap in calibrated maps for this sample
        #     sigma_real = sigma_real_cal
        #     sigma_imag = sigma_imag_cal

        deno_real = np.empty((H, W, C, S), dtype=np.float32)
        deno_imag = np.empty((H, W, C, S), dtype=np.float32)

        for z in range(S):
            print(f"  Denoising slice {z+1}/{S}")
            for c in range(C):
                img_c = vol[:, :, c, z]  # (H,W) complex

                # compute scalers for this slice/coil
                S_real = 0.99 / (np.percentile(np.abs(img_c.real), 99.8) + 1e-12)
                S_imag = 0.99 / (np.percentile(np.abs(img_c.imag), 99.8) + 1e-12)

                deno_real[:, :, c, z] = denoise_channel(
                    model, img_c.real, None if sigma_real is None else sigma_real[:, :, c],
                    S_real, device, tag=f"REAL slice{z} coil{c}"
                )

                deno_imag[:, :, c, z] = denoise_channel(
                    model, img_c.imag, None if sigma_imag is None else sigma_imag[:, :, c],
                    S_imag, device, tag=f"IMAG slice{z} coil{c}"
                )

        # Per-coil magnitude after denoising
        deno_mag = np.sqrt(deno_real ** 2 + deno_imag ** 2)     # (H,W,C,S)
        vol_mag = np.abs(vol).astype(np.float32)                # (H,W,C,S)

        # Save per-coil results
        nib.save(nib.Nifti1Image(np.transpose(vol_mag,  (0, 1, 3, 2)), np.eye(4)),
                 os.path.join(args.output_folder, f"original_percoil_sample{sid+1}.nii"))
        nib.save(nib.Nifti1Image(np.transpose(deno_mag, (0, 1, 3, 2)), np.eye(4)),
                 os.path.join(args.output_folder, f"denoised_percoil_sample{sid+1}.nii"))

        # Coil-combined (RSS)
        deno_cc = np.sqrt((deno_mag ** 2).sum(axis=2))          # (H,W,S)
        residual_cc = orig_cc - deno_cc

        all_deno.append(deno_cc)
        all_res.append(residual_cc)

    # --- 5) Save combined outputs ---
    deno4d = np.stack(all_deno, axis=-1)
    res4d  = np.stack(all_res, axis=-1)
    orig4d = np.stack(all_orig, axis=-1)

    nib.save(nib.Nifti1Image(deno4d.astype(np.float32), np.eye(4)),
             os.path.join(args.output_folder, 'combined_denoised_all.nii'))
    nib.save(nib.Nifti1Image(res4d.astype(np.float32), np.eye(4)),
             os.path.join(args.output_folder, 'combined_residual_all.nii'))
    nib.save(nib.Nifti1Image(orig4d.astype(np.float32), np.eye(4)),
             os.path.join(args.output_folder, 'original_coilcombined_all.nii'))

    print("All outputs saved in", args.output_folder)


if __name__ == "__main__":
    main()
