#!/usr/bin/env python
import os
import argparse

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

from utils.mat_loader import load_mri_data
from utils.noise_loader import load_noise_data, replicate_noise_map_with_sampling
import matplotlib.pyplot as plt
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
            # observed noise in background (per-coil magnitude → Rayleigh)
            sig_bg = _rayleigh_sigma_mle(vol_mag[:, :, c, z][bg])
            # map value in the same background region
            sm = sigma_mag[:, :, c][bg]
            sm = sm[np.isfinite(sm)]
            if sm.size == 0 or sig_bg <= 0: 
                continue
            ratios.append(sig_bg / (np.median(sm) + 1e-12))
    if not ratios:
        return 1.0  # fall back to raw if no background available
    alpha = float(np.median(ratios))
    return float(np.clip(alpha, clamp[0], clamp[1]))

def report_noise_stats(noise_cplx_scaled: np.ndarray, brain_mask: np.ndarray = None):
    """
    Compute and print global noise statistics in scaled domain (complex).
    noise_cplx_scaled: (H,W,C,S_noise) complex
    """
    H, W, C, S_noise = noise_cplx_scaled.shape

    real_vals = noise_cplx_scaled.real
    imag_vals = noise_cplx_scaled.imag

    if brain_mask is not None:
        print("⚠️ Warning: brain_mask shape likely != noise slices; ignoring mask for noise stats.")
    # Flatten across all dims
    r = real_vals.reshape(-1)
    i = imag_vals.reshape(-1)

    r = r[np.isfinite(r)]
    i = i[np.isfinite(i)]

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
    vals = np.asarray(vals)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 0.0
    lo, hi = np.percentile(vals, [trim_pct, 100.0 - trim_pct])
    core = vals[(vals >= lo) & (vals <= hi)]
    if core.size == 0:
        core = vals
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

    def _rng(a):
        return float(np.percentile(a, 5)), float(np.percentile(a, 95))
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


def save_mri_foreground_stats(out_dir: str, per_coil: dict, rss_mag: dict, summary: dict):
    os.makedirs(out_dir, exist_ok=True)
    coils = np.arange(len(per_coil["mu_R"]))
    data = np.stack([coils,
                     per_coil["mu_R"], per_coil["sig_R"],
                     per_coil["mu_I"], per_coil["sig_I"]], axis=1)
    header = "coil,mu_R,sig_R,mu_I,sig_I"
    np.savetxt(os.path.join(out_dir, "foreground_percoil_complex_stats.csv"),
               data, delimiter=",", header=header, comments="", fmt="%.6e")
    with open(os.path.join(out_dir, "foreground_rss_magnitude_stats.csv"), "w") as f:
        f.write("mu_mag,sig_mag\n")
        f.write(f"{rss_mag['mu_mag']:.6e},{rss_mag['sig_mag']:.6e}\n")
    keys = ["mu_R_median","mu_I_median","sig_R_median","sig_I_median",
            "mu_R_p5","mu_R_p95","mu_I_p5","mu_I_p95",
            "sig_R_p5","sig_R_p95","sig_I_p5","sig_I_p95",
            "mu_mag","sig_mag"]
    with open(os.path.join(out_dir, "foreground_summary.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        f.write(",".join([f"{summary[k]:.6e}" for k in keys]) + "\n")


def _robust_sigma_1d(x: np.ndarray, method: str='mad') -> float:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
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
    if ph or pw:
        inp_p = F.pad(inp, (0, pw, 0, ph), mode='reflect')
    else:
        inp_p = inp

    with torch.no_grad():
        out = model(inp_p)[..., :h, :w]

    return (out.squeeze().cpu().numpy().astype(np.float32)) / scale


# -------------------------------------- Main --------------------------------------

def main():
    p = argparse.ArgumentParser(description="Inference for multi-coil MRI denoising (magnitude-only).")
    p.add_argument('--model_pth', required=True, help='Restormer checkpoint (.pth)')
    p.add_argument('--mri_mat', required=True, help='Path to the MRI .mat/.h5 or NIfTI')
    p.add_argument('--mri_key', default='image', help='Dataset key inside .mat/HDF5')
    p.add_argument('--mri_format',
                   choices=['Hwihun_phantom', 'b1000', 'C', 'gslider', 'gslider_2', 'simulate', 'gslider_2_all'],
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
                   help="If set, load and feed a magnitude noise-map as 2nd channel")
    p.add_argument('--domain', choices=['mag', 'complex'], default='mag',
               help='Denoise in |magnitude| domain or in complex domain (real & imag separately).')
    p.add_argument('--output_folder', default='./results_infer',
                   help='Where to save NIfTI outputs')
    p.add_argument('--num_samples', type=int, default=2,
                   help='How many DWI channels to process (for MAT formats)')
    p.add_argument('--dwi_index', type=int, default=None,
                   help='Specify which DWI sample index to denoise (0-based). If None, denoise all samples.')
    p.add_argument('--brain_mask', default=None,
                   help='Optional brain mask NIfTI (.nii/.nii.gz), 1=brain, 0=background. '
                        'If not provided, use full FOV.')
    p.add_argument('--sigma_estimator', choices=['mad','std'], default='mad',
                   help='Background sigma estimator for calibration.')
    p.add_argument('--max_dirs', type=int, default=None,
               help='If set, only process the first K diffusion directions; default: all')

    args = p.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    diag_dir = os.path.join(args.output_folder, "residuals")
    os.makedirs(diag_dir, exist_ok=True)

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

    # Compute coil-combined RSS magnitude
    mag_cc = np.sqrt((np.abs(mri_img) ** 2).sum(axis=2))  # (H,W,S,N)

    # Robust global scale using mean + 3*std over all voxels (same as your revision)
    vals = mag_cc.reshape(-1).astype(np.float64)
    vals = vals[np.isfinite(vals)]
    mu   = vals.mean() if vals.size else 0.0
    sig  = (vals.std(ddof=1) if vals.size > 1 else 0.0)

    target   = mu + 3.0 * sig     # map μ+3σ -> 0.99
    S_GLOBAL = 0.99 / (target + 1e-12)
    print(f"[scale] Using S_GLOBAL={S_GLOBAL:.3e} from μ+3σ: μ={mu:.3e}, σ={sig:.3e}")

    # --- Basic stats (for info) ---
    def _pr_stats(tag, arr):
        arr = arr.astype(np.float32)
        print(f"[{tag}] min={arr.min():.6g}  max={arr.max():.6g}  "
              f"mean={arr.mean():.6g}  std={arr.std(ddof=1):.6g}")
    _pr_stats("MRI |complex|",   np.abs(mri_img))
    _pr_stats("MRI RSS (over C)", mag_cc)

    # --- 1.2) Brain mask ---
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

    # --- Foreground stats (scaled domain) for consistency with your pipeline ---
    mri_scaled = mri_img * S_GLOBAL  # (H,W,C,S,N) complex

    # --- 2) Optionally load Noise (complex), derive magnitude sigma maps ---
    sigma_mag_rayleigh = None  # (H,W,C) Rayleigh sigma for |n| per coil
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
        data = np.stack([np.arange(len(sigma_r)), sigma_r, sigma_i], axis=1)
        np.savetxt(out_csv, data, delimiter=",", header=header, comments="", fmt="%.6e")
        print(f"Saved per-coil noise stats (complex) → {out_csv}")

        # Build magnitude noise sigma per voxel/coil via Rayleigh MLE:
        # sigma_mag = sqrt( mean(|n|^2)/2 ) over noise samples
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

    # --- 4) Denoising Loop (magnitude-only) ---
    all_deno, all_res, all_orig = [], [], []
    sample_indices = [args.dwi_index] if args.dwi_index is not None else list(range(N))

    if 'max_dirs' in args and args.max_dirs is not None:
        N_use = min(N, args.max_dirs)
    else:
        N_use = N

    for sid in range(N_use):
        print(f"Sample {sid+1}/{N}")
        if args.dwi_index is not None and sid != args.dwi_index:
            continue  # only process the requested DWI

        vol = mri_img[..., sid]  # (H,W,C,S) complex
        vol_mag = np.abs(vol).astype(np.float32)
        # --- Apply brain mask to each slice ---
        if sid == 5:
            masked_vol_mag = np.empty_like(vol_mag)
            for z in range(vol_mag.shape[3]):
                masked_vol_mag[:, :, :, z] = vol_mag[:, :, :, z] * brain_mask[:, :, z, None].astype(np.float32)

            # --- Save masked per-coil magnitude ---
            nib.save(
                nib.Nifti1Image(np.transpose(masked_vol_mag, (0, 1, 3, 2)), np.eye(4)),
                os.path.join(args.output_folder, f"original_percoil_mag_dwi{sid+1}.nii")
            )

        orig_cc = np.sqrt((vol_mag ** 2).sum(axis=2)) * brain_mask.astype(np.float32)
        all_orig.append(orig_cc)

        # ---- use sigma map ----
        if args.use_noise and (sigma_mag_rayleigh is not None):
            alpha = 1.0
            print(f"[σ_mag auto-scale] alpha={alpha:.3f}")
            sigma_mag_to_use = alpha * sigma_mag_rayleigh
        else:
            sigma_mag_to_use = None

        H, W, C, S = vol_mag.shape
        deno_mag = np.empty((H, W, C, S), dtype=np.float32)
        res_mag  = np.empty_like(deno_mag)

        # NEW: for complex residuals (real & imag)
        res_real = np.empty_like(vol.real, dtype=np.float32)
        res_imag = np.empty_like(vol.imag, dtype=np.float32)

        for z in range(S):
            print(f"  Denoising slice {z+1}/{S}")
            mask2d = brain_mask[:, :, z].astype(np.float32)

            for c in range(C):
                sig_map = sigma_mag_to_use[:, :, c] * mask2d if sigma_mag_to_use is not None else None

                if args.domain == 'mag':
                    # --- Magnitude-only denoising ---
                    img_mag_c = vol_mag[:, :, c, z]
                    out_mag = denoise_magnitude(model, img_mag_c, sig_map, S_GLOBAL, device)
                    deno_mag[:, :, c, z] = mask2d * out_mag
                    res_mag[:, :, c, z]  = img_mag_c - out_mag

                else:
                    # --- Complex denoising per coil ---
                    img_real_c = vol.real[:, :, c, z]
                    img_imag_c = vol.imag[:, :, c, z]
                    deno_r, deno_i = denoise_complex(model, img_real_c, img_imag_c, sig_map, S_GLOBAL, device)
                    res_real[:, :, c, z] = img_real_c - deno_r
                    res_imag[:, :, c, z] = img_imag_c - deno_i

                    out_mag = np.sqrt(deno_r**2 + deno_i**2).astype(np.float32)
                    deno_mag[:, :, c, z] = mask2d * out_mag
                    res_mag[:, :, c, z]  = np.sqrt(res_real[:, :, c, z]**2 + res_imag[:, :, c, z]**2)

        # Coil-combined (RSS)
        deno_cc = np.sqrt((deno_mag ** 2).sum(axis=2))
        residual_cc = orig_cc - deno_cc

        # --- SAVE all outputs for this DWI ---
        dwi_tag = f"dwi{sid+1}"

        # (1) per-coil magnitude denoised and residuals
        if sid == 5:
            nib.save(nib.Nifti1Image(np.transpose(deno_mag, (0, 1, 3, 2)), np.eye(4)),
                    os.path.join(args.output_folder, f"denoised_percoil_mag_{dwi_tag}.nii"))
            nib.save(nib.Nifti1Image(np.transpose(res_mag, (0, 1, 3, 2)), np.eye(4)),
                    os.path.join(args.output_folder, f"residual_percoil_mag_{dwi_tag}.nii"))

        # Combined RSS results (for completeness)
        # nib.save(nib.Nifti1Image(deno_cc.astype(np.float32), np.eye(4)),
        #         os.path.join(args.output_folder, f"combined_denoised_{dwi_tag}.nii"))
        # nib.save(nib.Nifti1Image(residual_cc.astype(np.float32), np.eye(4)),
        #         os.path.join(args.output_folder, f"combined_residual_{dwi_tag}.nii"))

        all_deno.append(deno_cc)
        all_res.append(residual_cc)


    # --- 5) Save combined outputs (coil-combined RSS) ---
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
