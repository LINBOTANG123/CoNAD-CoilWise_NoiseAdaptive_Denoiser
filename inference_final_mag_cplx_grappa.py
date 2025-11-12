#!/usr/bin/env python
import os
import argparse

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

from utils.mat_loader import load_mri_data
from utils.noise_loader import load_noise_data, replicate_noise_map_with_sampling
from typing import Optional

NUM_COILS = 32  # adjust if needed

# ------------------------ Utilities (unchanged or adapted) ------------------------
def _rayleigh_sigma_mle(x: np.ndarray) -> float:
    x = np.asarray(x); x = x[np.isfinite(x)]
    if x.size == 0: return 0.0
    return float(np.sqrt(np.mean(x**2) / 2.0))  # MLE for Rayleigh scale

def auto_scale_sigma_mag(vol_mag: np.ndarray, sigma_mag: np.ndarray, bg_mask: np.ndarray,
                         clamp=(0.7, 2.5)) -> float:
    """
    Returns a single scalar α so that the noise map matches the observed background Rayleigh scale.
    α = median_z,c [ σ_bg(z,c) / median_bg( sigma_mag[:,:,c] ) ]
    """
    H, W, C, S = vol_mag.shape
    ratios = []
    for z in range(S):
        bg = bg_mask[:, :, z]
        if not np.any(bg):
            continue
        for c in range(C):
            sig_bg = _rayleigh_sigma_mle(vol_mag[:, :, c, z][bg])
            sm = sigma_mag[:, :, c][bg]
            sm = sm[np.isfinite(sm)]
            if sm.size == 0 or sig_bg <= 0:
                continue
            ratios.append(sig_bg / (np.median(sm) + 1e-12))
    if not ratios:
        return 1.0
    alpha = float(np.median(ratios))
    return float(np.clip(alpha, clamp[0], clamp[1]))

def report_noise_stats(noise_cplx_scaled: np.ndarray, brain_mask: np.ndarray = None):
    """
    Compute and print global noise statistics in scaled domain (complex).
    noise_cplx_scaled: (H,W,C,S_noise) complex
    """
    real_vals = noise_cplx_scaled.real
    imag_vals = noise_cplx_scaled.imag
    if brain_mask is not None:
        print("⚠️ Warning: brain_mask shape likely != noise slices; ignoring mask for noise stats.")
    r = real_vals.reshape(-1); i = imag_vals.reshape(-1)
    r = r[np.isfinite(r)]; i = i[np.isfinite(i)]
    print("\n=== [Noise stats in scaled domain] ===")
    print(f" REAL mean={r.mean():.3e}, std={r.std(ddof=1):.3e}")
    print(f" IMAG mean={i.mean():.3e}, std={i.std(ddof=1):.3e}")

def report_percoil_noise_stats(noise_cplx_scaled: np.ndarray, brain_mask: np.ndarray = None):
    """
    Compute per-coil noise statistics (mean, std) of real/imag parts in scaled domain.
    """
    H, W, C, S_noise = noise_cplx_scaled.shape
    mu_r, mu_i, std_r, std_i = [], [], [], []
    for c in range(C):
        real_vals = noise_cplx_scaled.real[:, :, c, :].ravel()
        imag_vals = noise_cplx_scaled.imag[:, :, c, :].ravel()
        real_vals = real_vals[np.isfinite(real_vals)]
        imag_vals = imag_vals[np.isfinite(imag_vals)]
        mu_r.append(real_vals.mean() if real_vals.size else 0.0)
        mu_i.append(imag_vals.mean() if imag_vals.size else 0.0)
        std_r.append(real_vals.std(ddof=1) if real_vals.size > 1 else 0.0)
        std_i.append(imag_vals.std(ddof=1) if imag_vals.size > 1 else 0.0)

    mu_r, mu_i, std_r, std_i = map(np.array, (mu_r, mu_i, std_r, std_i))
    print("\n=== [Per-coil noise stats in scaled domain] ===")
    print("Coil | μ_real | σ_real | μ_imag | σ_imag")
    for c in range(C):
        print(f"{c:3d} | {mu_r[c]: .3e} | {std_r[c]: .3e} | {mu_i[c]: .3e} | {std_i[c]: .3e}")

    print("\nSummary:")
    print(f" Mean σ_real={std_r.mean():.3e}, Mean σ_imag={std_i.mean():.3e}")
    print(f" Median σ_real={np.median(std_r):.3e}, Median σ_imag={np.median(std_i):.3e}")
    return std_r, std_i

def _trimmed_mean_std(vals: np.ndarray, trim_pct: float = 1.0):
    vals = np.asarray(vals); vals = vals[np.isfinite(vals)]
    if vals.size == 0: return 0.0, 0.0
    lo, hi = np.percentile(vals, [trim_pct, 100.0 - trim_pct])
    core = vals[(vals >= lo) & (vals <= hi)]
    if core.size == 0: core = vals
    return float(core.mean()), float(core.std(ddof=1) if core.size > 1 else core.std())

def denoise_complex(model, img_real: np.ndarray, img_imag: np.ndarray,
                    sigma_map: np.ndarray, scale: float, device: str, tag: str = ""):
    def _run_one(ch: np.ndarray, sig: Optional[np.ndarray], label: str):
        if sig is not None:
            arr = np.stack([(ch * scale).astype(np.float32),
                            (sig * scale).astype(np.float32)], axis=0)
        else:
            arr = (ch * scale)[np.newaxis, ...].astype(np.float32)
        inp = torch.from_numpy(arr).unsqueeze(0).to(device)  # (1,C,H,W)
        _, _, h, w = inp.shape
        ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
        inp_p = F.pad(inp, (0, pw, 0, ph), mode='reflect') if (ph or pw) else inp
        with torch.no_grad():
            out = model(inp_p)[..., :h, :w]
        return (out.squeeze().cpu().numpy().astype(np.float32)) / scale

    deno_r = _run_one(img_real, sigma_map, f"{tag} [REAL]")
    deno_i = _run_one(img_imag, sigma_map, f"{tag} [IMAG]")
    return deno_r, deno_i

def compute_mri_foreground_stats_scaled(mri_cplx_scaled: np.ndarray,
                                        brain_mask: np.ndarray,
                                        trim_pct: float = 1.0):
    """
    Foreground stats in the SAME scaled domain (complex + RSS magnitude).
    mri_cplx_scaled: (H,W,C,S,N) complex
    brain_mask:     (H,W,S) bool
    """
    H, W, C, S, N = mri_cplx_scaled.shape
    mask4d = np.broadcast_to(brain_mask[:, :, :, None], (H, W, S, N))
    mu_R, sig_R, mu_I, sig_I = [], [], [], []
    for c in range(C):
        r = mri_cplx_scaled.real[:, :, c, :, :][mask4d]
        i = mri_cplx_scaled.imag[:, :, c, :, :][mask4d]
        mu_r, sd_r = _trimmed_mean_std(r, trim_pct)
        mu_i, sd_i = _trimmed_mean_std(i, trim_pct)
        mu_R.append(mu_r); sig_R.append(sd_r)
        mu_I.append(mu_i); sig_I.append(sd_i)
    mu_R = np.array(mu_R); sig_R = np.array(sig_R)
    mu_I = np.array(mu_I); sig_I = np.array(sig_I)
    mag_cc = np.sqrt((np.abs(mri_cplx_scaled) ** 2).sum(axis=2))  # (H,W,S,N)
    mag_vals = mag_cc[mask4d]
    mu_mag, sig_mag = _trimmed_mean_std(mag_vals, trim_pct)

    def _rng(a): return float(np.percentile(a, 5)), float(np.percentile(a, 95))
    summary = {
        "mu_R_median":  float(np.median(mu_R)),
        "mu_I_median":  float(np.median(mu_I)),
        "sig_R_median": float(np.median(sig_R)),
        "sig_I_median": float(np.median(sig_I)),
        "mu_R_p5": _rng(mu_R)[0],  "mu_R_p95": _rng(mu_R)[1],
        "mu_I_p5": _rng(mu_I)[0],  "mu_I_p95": _rng(mu_I)[1],
        "sig_R_p5": _rng(sig_R)[0],"sig_R_p95": _rng(sig_R)[1],
        "sig_I_p5": _rng(sig_I)[0],"sig_I_p95": _rng(sig_I)[1],
        "mu_mag": mu_mag, "sig_mag": sig_mag
    }
    per_coil = {"mu_R": mu_R, "sig_R": sig_R, "mu_I": mu_I, "sig_I": sig_I}
    rss_mag  = {"mu_mag": mu_mag, "sig_mag": sig_mag}
    return per_coil, rss_mag, summary

def load_model(model_path: str, inp_channels: int, device: str = 'cuda') -> torch.nn.Module:
    """Load the Restormer model with dynamic input channels."""
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

def denoise_magnitude(model, img_mag: np.ndarray, sigma_mag: np.ndarray, scale: float, device: str, tag: str = ""):
    """
    Denoise one channel: magnitude-only.
    img_mag: (H,W) nonnegative float
    sigma_mag: (H,W) Rayleigh (per-coil |n|) sigma map for magnitude, or None
    scale: scalar normalization factor (same S_GLOBAL)
    """
    if sigma_mag is not None:
        arr = np.stack([(img_mag * scale).astype(np.float32),
                        (sigma_mag * scale).astype(np.float32)], axis=0)
    else:
        arr = (img_mag * scale)[np.newaxis, ...].astype(np.float32)

    inp = torch.from_numpy(arr).unsqueeze(0).to(device)  # (1,C,H,W)
    _, _, h, w = inp.shape
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    inp_p = F.pad(inp, (0, pw, 0, ph), mode='reflect') if (ph or pw) else inp

    with torch.no_grad():
        out = model(inp_p)[..., :h, :w]

    return (out.squeeze().cpu().numpy().astype(np.float32)) / scale

# -------------------------------------- Main --------------------------------------
def main():
    p = argparse.ArgumentParser(description="Inference for multi-coil MRI denoising (magnitude-only).")

    # Accept single file OR list of GRAPPA DWIs
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--mri_mat', help='Path to SINGLE MRI .mat/.h5 or NIfTI')
    group.add_argument('--mri_list', help='Text file with one GRAPPA .mat path per line (each = one DWI)')

    p.add_argument('--model_pth', required=True, help='Restormer checkpoint (.pth)')
    p.add_argument('--mri_key', default='image', help='Dataset key inside .mat/HDF5')
    p.add_argument('--mri_format',
                   choices=['Hwihun_phantom', 'b1000', 'C', 'gslider', 'gslider_2', 'simulate', 'gslider_2_all', 'grappa'],
                   default='b1000')

    p.add_argument('--noise_mat', required=False, help='Path to noise .mat/.h5 or NIfTI (only if using noise)')
    p.add_argument('--noise_key', default='k_gc', help='Key inside noise .mat/HDF5 (ignored for simulate)')
    p.add_argument('--noise_format',
                   choices=['Hwihun_phantom', 'b1000', 'C', 'gslider', 'gslider_2', 'simulate', 'gslider_v5', 'grappa'],
                   default='b1000')
    p.add_argument('--use_noise', action='store_true',
                   help="If set, load and feed a magnitude noise-map as 2nd channel")

    p.add_argument('--domain', choices=['mag', 'complex'], default='mag',
                   help='Denoise in |magnitude| domain or in complex domain (real & imag separately).')
    p.add_argument('--output_folder', default='./results_infer', help='Where to save NIfTI outputs')

    # For legacy single-file modes
    p.add_argument('--num_samples', type=int, default=2, help='How many DWI channels to process (for MAT formats)')
    p.add_argument('--dwi_index', type=int, default=None,
                   help='Specify which DWI sample index to denoise (0-based). If None, denoise all samples.')
    p.add_argument('--max_dirs', type=int, default=None,
                   help='If set, process only the first K diffusion directions; default: all')

    p.add_argument('--brain_mask', default=None,
                   help='Optional brain mask NIfTI (.nii/.nii.gz), 1=brain, 0=background. If not provided, use full FOV.')
    p.add_argument('--sigma_estimator', choices=['mad','std'], default='mad',
                   help='Background sigma estimator for calibration.')
    p.add_argument('--scale_txt', default=None,
               help='Optional path to persist the global scale S_GLOBAL. '
                    'If the file exists, load S_GLOBAL from it and skip recomputation. '
                    'If not, compute S_GLOBAL and save it here (defaults to <output_folder>/S_GLOBAL.txt).')

    args = p.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    # Decide where to read/write the scale
    scale_path = args.scale_txt or os.path.join(args.output_folder, 'S_GLOBAL.txt')

    # --- GRAPPA path discovery ---
    dwi_paths = None
    if args.mri_format == 'grappa':
        if not args.mri_list:
            raise ValueError("In 'grappa' mode, provide --mri_list (one .mat path per line).")
        with open(args.mri_list, 'r') as f:
            dwi_paths = [ln.strip() for ln in f if ln.strip()]
        if not dwi_paths:
            raise ValueError("No paths found in --mri_list.")
        if args.max_dirs is not None:
            dwi_paths = dwi_paths[:args.max_dirs]

    # --- Load shapes and brain mask; compute global S_GLOBAL ---
    if args.mri_format == 'grappa':
        # Probe first DWI to get shapes
        probe = load_mri_data(
            dwi_paths[0], key=args.mri_key, data_format='grappa',
            num_samples_to_load=None, output_space='complex_image'
        )  # (H,W,C,S,1)
        H, W, C, S, _ = probe.shape
        print("Probe MRI (complex):", probe.shape)

        # Brain mask
        if args.brain_mask is not None:
            mask_img = nib.load(args.brain_mask)
            brain_mask = mask_img.get_fdata().astype(bool)  # (H,W,S)
            if brain_mask.shape != (H, W, S):
                raise ValueError(f"Brain mask shape {brain_mask.shape} != {(H,W,S)}")
            bg_mask = ~brain_mask
        else:
            print("⚠️ No brain_mask provided → using full FOV as mask.")
            brain_mask = np.ones((H, W, S), dtype=bool)
            bg_mask = ~brain_mask

        # --- Load or compute S_GLOBAL ---
        if os.path.exists(scale_path):
            with open(scale_path, 'r') as f:
                S_GLOBAL = float(f.read().strip())
            print(f"[scale] Loaded S_GLOBAL={S_GLOBAL:.3e} from {scale_path}")
        else:
            # Welford across ALL DWIs to compute μ, σ of RSS magnitude
            count = 0
            mean = 0.0
            M2 = 0.0
            for pth in dwi_paths:
                vol = load_mri_data(
                    pth, key=args.mri_key, data_format='grappa',
                    num_samples_to_load=None, output_space='complex_image'
                )
                # Accept either 5D (H,W,C,S,1) or 4D (H,W,C,S)
                if vol.ndim == 5 and vol.shape[-1] == 1:
                    vol = vol[..., 0]
                elif vol.ndim != 4:
                    raise ValueError(f"Expected 4D (H,W,C,S) or 5D with singleton last dim, got {vol.shape}")

                mag_cc = np.sqrt((np.abs(vol) ** 2).sum(axis=2))  # (H,W,S)
                x = mag_cc.reshape(-1).astype(np.float64)
                x = x[np.isfinite(x)]
                for v in x:  # Welford
                    count += 1
                    delta = v - mean
                    mean += delta / count
                    delta2 = v - mean
                    M2 += delta * delta2
            mu = mean if count > 0 else 0.0
            sig = np.sqrt(M2 / (count - 1)) if count > 1 else 0.0
            target   = mu + 3.0 * sig
            S_GLOBAL = 0.99 / (target + 1e-12)
            print(f"[scale] Using S_GLOBAL={S_GLOBAL:.3e} from μ+3σ over ALL DWIs: μ={mu:.3e}, σ={sig:.3e}")

            # Save it for next time
            with open(scale_path, 'w') as f:
                f.write(f"{S_GLOBAL:.8e}\n")
            print(f"[scale] Saved S_GLOBAL to {scale_path}")

    else:
        # Legacy single-input path
        mri_img = load_mri_data(
            args.mri_mat,
            key=args.mri_key,
            data_format=args.mri_format,
            num_samples_to_load=args.num_samples,
            output_space='complex_image'
        )
        H, W, C, S, N = mri_img.shape
        print("Loaded MRI (complex):", mri_img.shape)
        mag_cc = np.sqrt((np.abs(mri_img) ** 2).sum(axis=2))  # (H,W,S,N)
        vals = mag_cc.reshape(-1).astype(np.float64)
        vals = vals[np.isfinite(vals)]
        mu   = vals.mean() if vals.size else 0.0
        sig  = (vals.std(ddof=1) if vals.size > 1 else 0.0)
        target   = mu + 3.0 * sig
        S_GLOBAL = 0.99 / (target + 1e-12)
        print(f"[scale] Using S_GLOBAL={S_GLOBAL:.3e} from μ+3σ: μ={mu:.3e}, σ={sig:.3e}")

        # Brain mask
        if args.brain_mask is not None:
            mask_img = nib.load(args.brain_mask)
            brain_mask = mask_img.get_fdata().astype(bool)  # (H,W,S)
            if brain_mask.shape != (H, W, S):
                raise ValueError(f"Brain mask shape {brain_mask.shape} != {(H,W,S)}")
            bg_mask = ~brain_mask
        else:
            print("⚠️ No brain_mask provided → using full FOV as mask.")
            brain_mask = np.ones((H, W, S), dtype=bool)
            bg_mask = ~brain_mask

    # --- 2) Optionally load Noise (complex), derive magnitude sigma maps (shared across DWIs) ---
    sigma_mag_rayleigh = None  # (H,W,C)
    noise_cplx = None
    if args.use_noise:
        noise_cplx = load_noise_data(
            args.noise_mat,
            key=args.noise_key,
            data_format=args.noise_format,
            output_space='complex_image'
        )  # (H,W,C,S_noise)

        if args.noise_format == 'b1000':
            noise_cplx = replicate_noise_map_with_sampling(noise_cplx, H, W, S)
        if noise_cplx.shape[:3] != (H, W, C):
            raise ValueError(f"Noise shape {noise_cplx.shape} does not match MRI {(H,W,C)}")

        # rotate noise map to match
        noise_cplx = np.rot90(noise_cplx, k=3, axes=(0, 1)).copy()

        # Stats in scaled domain (complex) for logging
        noise_cplx_scaled = noise_cplx * S_GLOBAL
        report_noise_stats(noise_cplx_scaled, brain_mask=brain_mask)
        sigma_r, sigma_i = report_percoil_noise_stats(noise_cplx_scaled, brain_mask=None)
        out_csv = os.path.join(args.output_folder, "percoil_noise_stats_complex.csv")
        header = "coil,sigma_real,sigma_imag"
        data_csv = np.stack([np.arange(len(sigma_r)), sigma_r, sigma_i], axis=1)
        np.savetxt(out_csv, data_csv, delimiter=",", header=header, comments="", fmt="%.6e")
        print(f"Saved per-coil noise stats (complex) → {out_csv}")

        # Build magnitude noise sigma per voxel/coil via Rayleigh MLE:
        nmag_sq_mean = (np.abs(noise_cplx)**2).mean(axis=-1).astype(np.float32)  # (H,W,C)
        sigma_mag_rayleigh = np.sqrt(np.maximum(nmag_sq_mean, 0.0) / 2.0).astype(np.float32)
        nib.save(nib.Nifti1Image(sigma_mag_rayleigh, np.eye(4)),
                 os.path.join(args.output_folder, 'sigma_mag_rayleigh.nii'))
        
        # --- Save per-coil noise magnitude (unmasked) ---
        noise_mag = np.abs(noise_cplx).astype(np.float32)  # (H, W, C, S_noise)
        nib.save(
            nib.Nifti1Image(np.transpose(noise_mag, (0, 1, 3, 2)), np.eye(4)),
            os.path.join(args.output_folder, "original_percoil_noise_mag.nii")
        )
        print("Saved per-coil noise magnitude → original_percoil_noise_mag.nii")

        # Also central-chi scale for RSS (optional output)
        noise_power_mean = (np.abs(noise_cplx)**2).sum(axis=2).mean(axis=-1).astype(np.float32)  # (H,W)
        sigma_rss_chi = np.sqrt(np.maximum(noise_power_mean, 0.0) / (2.0 * C)).astype(np.float32)
        nib.save(nib.Nifti1Image(sigma_rss_chi, np.eye(4)),
                 os.path.join(args.output_folder, 'sigma_rss_chi.nii'))

    # --- 3) Load model ---
    inp_ch = 2 if (args.use_noise and sigma_mag_rayleigh is not None) else 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.model_pth, inp_channels=inp_ch, device=device)

    # --- 4) Denoising Loop ---
    all_deno, all_res, all_orig = [], [], []

    if args.mri_format == 'grappa':
        N_total = len(dwi_paths)
        for di, pth in enumerate(dwi_paths):
            print(f"[DWI {di+1}/{N_total}] {pth}")
            vol = load_mri_data(
                pth, key=args.mri_key, data_format='grappa',
                num_samples_to_load=None, output_space='complex_image'
            )[..., 0]  # (H,W,C,S)

            vol_mag = np.abs(vol).astype(np.float32)                # (H,W,C,S)
            orig_cc = np.sqrt((vol_mag ** 2).sum(axis=2)) * brain_mask.astype(np.float32)  # (H,W,S)
            if di == 7:
                masked_vol_mag = np.empty_like(vol_mag)
                for z in range(vol_mag.shape[3]):
                    masked_vol_mag[:, :, :, z] = vol_mag[:, :, :, z] * brain_mask[:, :, z, None].astype(np.float32)

                # --- Save masked per-coil magnitude ---
                nib.save(
                    nib.Nifti1Image(np.transpose(masked_vol_mag, (0, 1, 3, 2)), np.eye(4)),
                    os.path.join(args.output_folder, f"original_percoil_mag_dwi{di+1}.nii")
                )
            
            all_orig.append(orig_cc)

            # Shared sigma map for all DWIs
            sigma_mag_to_use = None
            if args.use_noise and (sigma_mag_rayleigh is not None):
                alpha = 1.0  # keep as-is per your instruction
                print(f"[σ_mag auto-scale] alpha={alpha:.3f}")
                sigma_mag_to_use = alpha * sigma_mag_rayleigh  # (H,W,C)

            # Denoise per-slice, per-coil
            deno_mag = np.empty_like(vol_mag)  # (H,W,C,S)
            res_mag  = np.empty_like(deno_mag)

            for z in range(S):
                print(f"  Denoising slice {z+1}/{S}")
                mask2d = brain_mask[:, :, z].astype(np.float32)
                for c in range(C):
                    sig_map = None
                    if sigma_mag_to_use is not None:
                        sig_map = sigma_mag_to_use[:, :, c] * mask2d
                    if args.domain == 'mag':
                        img_mag_c = vol_mag[:, :, c, z]
                        out_mag = denoise_magnitude(model, img_mag_c, sig_map, S_GLOBAL, device, tag=f"MAG s{z} c{c}")
                        deno_mag[:, :, c, z] = mask2d * out_mag
                        res_mag[:, :, c, z]  = img_mag_c - out_mag
                    else:
                        img_real_c = vol.real[:, :, c, z]
                        img_imag_c = vol.imag[:, :, c, z]
                        deno_r, deno_i = denoise_complex(model, img_real_c, img_imag_c, sig_map, S_GLOBAL, device, tag=f"CPLX s{z} c{c}")
                        out_mag = np.sqrt(deno_r**2 + deno_i**2).astype(np.float32)
                        deno_mag[:, :, c, z] = mask2d * out_mag

            # Coil-combined RSS after denoising
            deno_cc = np.sqrt((deno_mag ** 2).sum(axis=2))          # (H,W,S)
            residual_cc = orig_cc - deno_cc

            dwi_tag = f"dwi{di+1}"

            if di == 7:
                nib.save(nib.Nifti1Image(np.transpose(deno_mag, (0, 1, 3, 2)), np.eye(4)),
                        os.path.join(args.output_folder, f"denoised_percoil_mag_{dwi_tag}.nii"))
                nib.save(nib.Nifti1Image(np.transpose(res_mag, (0, 1, 3, 2)), np.eye(4)),
                        os.path.join(args.output_folder, f"residual_percoil_mag_{dwi_tag}.nii"))

            # Save per-DWI outputs (H,W,S)
            # nib.save(nib.Nifti1Image(deno_cc.astype(np.float32), np.eye(4)),
            #          os.path.join(args.output_folder, f"denoised_dwi_{di+1:03d}.nii"))
            # nib.save(nib.Nifti1Image(residual_cc.astype(np.float32), np.eye(4)),
            #          os.path.join(args.output_folder, f"residual_dwi_{di+1:03d}.nii"))

            all_deno.append(deno_cc)
            all_res.append(residual_cc)

        # --- Combined outputs (coil-combined RSS stacks) ---
        deno4d = np.stack(all_deno, axis=-1)   # (H,W,S,DWI)
        res4d  = np.stack(all_res, axis=-1)    # (H,W,S,DWI)
        orig4d = np.stack(all_orig, axis=-1)   # (H,W,S,DWI)

        nib.save(nib.Nifti1Image(deno4d.astype(np.float32), np.eye(4)),
                 os.path.join(args.output_folder, 'combined_denoised_all.nii'))
        nib.save(nib.Nifti1Image(res4d.astype(np.float32), np.eye(4)),
                 os.path.join(args.output_folder, 'combined_residual_all.nii'))
        nib.save(nib.Nifti1Image(orig4d.astype(np.float32), np.eye(4)),
                 os.path.join(args.output_folder, 'original_coilcombined_all.nii'))

    else:
        # --- Legacy single-input path (no histograms) ---
        N = mri_img.shape[-1]
        if args.max_dirs is not None:
            N_use = min(N, args.max_dirs)
        else:
            N_use = N

        for sid in range(N_use):
            print(f"Sample {sid+1}/{N}")
            vol = mri_img[..., sid]                           # (H,W,C,S) complex
            vol_mag = np.abs(vol).astype(np.float32)          # (H,W,C,S)

            orig_cc = np.sqrt((vol_mag ** 2).sum(axis=2)) * brain_mask.astype(np.float32)     # (H,W,S)
            all_orig.append(orig_cc)

            sigma_mag_to_use = None
            if args.use_noise and (sigma_mag_rayleigh is not None):
                alpha = 1.0
                print(f"[σ_mag auto-scale] alpha={alpha:.3f}")
                sigma_mag_to_use = alpha * sigma_mag_rayleigh

            H, W, C, S = vol_mag.shape
            deno_mag = np.empty((H, W, C, S), dtype=np.float32)

            for z in range(S):
                print(f"  Denoising slice {z+1}/{S}")
                mask2d = brain_mask[:, :, z].astype(np.float32)
                for c in range(C):
                    sig_map = None
                    if sigma_mag_to_use is not None:
                        sig_map = sigma_mag_to_use[:, :, c] * mask2d

                    if args.domain == 'mag':
                        img_mag_c = vol_mag[:, :, c, z]
                        out_mag = denoise_magnitude(model, img_mag_c, sig_map, S_GLOBAL, device, tag=f"MAG slice{z} coil{c}")
                        deno_mag[:, :, c, z] = mask2d * out_mag
                    else:
                        img_real_c = vol.real[:, :, c, z]
                        img_imag_c = vol.imag[:, :, c, z]
                        deno_r, deno_i = denoise_complex(model, img_real_c, img_imag_c, sig_map, S_GLOBAL, device, tag=f"CPLX slice{z} coil{c}")
                        out_mag = np.sqrt(deno_r**2 + deno_i**2).astype(np.float32)
                        deno_mag[:, :, c, z] = mask2d * out_mag

            deno_cc = np.sqrt((deno_mag ** 2).sum(axis=2))          # (H,W,S)
            residual_cc = orig_cc - deno_cc

            all_deno.append(deno_cc)
            all_res.append(residual_cc)

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
