#!/usr/bin/env python
import os
import argparse

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

from utils.mat_loader import load_mri_data
from utils.noise_loader import load_noise_data
import matplotlib.pyplot as plt

NUM_COILS = 32  # adjust if needed

def report_noise_stats(noise_cplx_scaled: np.ndarray, brain_mask: np.ndarray = None):
    """
    Compute and print global noise statistics in scaled domain.
    noise_cplx_scaled: (H,W,C,S_noise) complex
    brain_mask: optional (H,W,S) bool (will be broadcast across coils)
    """
    H, W, C, S_noise = noise_cplx_scaled.shape

    real_vals = noise_cplx_scaled.real
    imag_vals = noise_cplx_scaled.imag

    if brain_mask is not None:
        # Expand brain_mask to (H,W,S_noise), then broadcast across coils
        mask3d = np.broadcast_to(brain_mask[..., None], (H, W, brain_mask.shape[2], 1))
        # Note: your brain_mask has S slices, but noise has S_noise samples (not same!)
        # Safer to just ignore brain_mask for pure noise, unless you know they align.
        print("⚠️ Warning: brain_mask shape != noise slices, ignoring mask for noise stats.")
        mask = np.ones_like(real_vals[..., 0], dtype=bool)  # fallback
    else:
        mask = np.ones_like(real_vals[..., 0], dtype=bool)

    # Flatten across all dims
    r = real_vals[mask, :].ravel()
    i = imag_vals[mask, :].ravel()

    print("\n=== [Noise stats in scaled domain] ===")
    print(f" REAL mean={r.mean():.3e}, std={r.std(ddof=1):.3e}")
    print(f" IMAG mean={i.mean():.3e}, std={i.std(ddof=1):.3e}")



def report_percoil_noise_stats(noise_cplx_scaled: np.ndarray, brain_mask: np.ndarray = None):
    """
    Compute per-coil noise statistics (mean, std) of real/imag parts
    in the scaled domain (after applying S_GLOBAL).
    
    noise_cplx_scaled: (H,W,C,S_noise) complex ndarray
    brain_mask: optional (H,W,S) boolean mask. If provided, stats are only taken
                from those spatial locations (background or brain).
    """
    H, W, C, S_noise = noise_cplx_scaled.shape
    mu_r, mu_i, std_r, std_i = [], [], [], []

    for c in range(C):
        real_vals = noise_cplx_scaled.real[:, :, c, :].ravel()
        imag_vals = noise_cplx_scaled.imag[:, :, c, :].ravel()

        # If you want to mask: only apply spatial mask, ignore slice mismatch
        if brain_mask is not None:
            H, W, S = brain_mask.shape
            mask2d = brain_mask.reshape(H*W*S)
            real_vals = real_vals[:mask2d.size][mask2d]
            imag_vals = imag_vals[:mask2d.size][mask2d]

        mu_r.append(real_vals.mean())
        mu_i.append(imag_vals.mean())
        std_r.append(real_vals.std(ddof=1))
        std_i.append(imag_vals.std(ddof=1))

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
    """
    Robust mean/std by trimming both tails by `trim_pct` percent.
    vals: 1D array
    """
    vals = np.asarray(vals)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 0.0
    lo, hi = np.percentile(vals, [trim_pct, 100.0 - trim_pct])
    core = vals[(vals >= lo) & (vals <= hi)]
    if core.size == 0:
        core = vals
    return float(core.mean()), float(core.std(ddof=1) if core.size > 1 else core.std())

def compute_mri_foreground_stats_scaled(mri_cplx_scaled: np.ndarray,
                                        brain_mask: np.ndarray,
                                        trim_pct: float = 1.0):
    """
    Compute foreground stats in the SAME scaled domain you feed the network.
    mri_cplx_scaled: (H,W,C,S,N) complex
    brain_mask:     (H,W,S) bool
    Returns:
      per_coil: dict with arrays mu_R[C], sig_R[C], mu_I[C], sig_I[C]
      rss_mag:  dict with mu_mag, sig_mag (pooled over coils)
      summary:  dict of medians and 5–95% ranges for the complex channels
    """
    H, W, C, S, N = mri_cplx_scaled.shape
    # Mask to (H,W,S,N) then used for each coil slice-set
    mask4d = np.broadcast_to(brain_mask[:, :, :, None], (H, W, S, N))

    mu_R, sig_R, mu_I, sig_I = [], [], [], []

    # Per-coil complex stats (aggregate over all slices & volumes within brain)
    for c in range(C):
        r = mri_cplx_scaled.real[:, :, c, :, :][mask4d]
        i = mri_cplx_scaled.imag[:, :, c, :, :][mask4d]
        mu_r, sd_r = _trimmed_mean_std(r, trim_pct)
        mu_i, sd_i = _trimmed_mean_std(i, trim_pct)
        mu_R.append(mu_r); sig_R.append(sd_r)
        mu_I.append(mu_i); sig_I.append(sd_i)

    mu_R = np.array(mu_R); sig_R = np.array(sig_R)
    mu_I = np.array(mu_I); sig_I = np.array(sig_I)

    # RSS magnitude stats (pooled)
    mag_cc = np.sqrt((np.abs(mri_cplx_scaled) ** 2).sum(axis=2))  # (H,W,S,N)
    mag_vals = mag_cc[mask4d]
    mu_mag, sig_mag = _trimmed_mean_std(mag_vals, trim_pct)

    # Summaries (medians + 5–95% ranges) across coils
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
    """Save per-coil CSV + summary CSV for training use."""
    os.makedirs(out_dir, exist_ok=True)
    # Per-coil CSV
    coils = np.arange(len(per_coil["mu_R"]))
    data = np.stack([coils,
                     per_coil["mu_R"], per_coil["sig_R"],
                     per_coil["mu_I"], per_coil["sig_I"]], axis=1)
    header = "coil,mu_R,sig_R,mu_I,sig_I"
    np.savetxt(os.path.join(out_dir, "foreground_percoil_complex_stats.csv"),
               data, delimiter=",", header=header, comments="", fmt="%.6e")
    # RSS magnitude CSV (single row)
    with open(os.path.join(out_dir, "foreground_rss_magnitude_stats.csv"), "w") as f:
        f.write("mu_mag,sig_mag\n")
        f.write(f"{rss_mag['mu_mag']:.6e},{rss_mag['sig_mag']:.6e}\n")
    # Summary CSV
    keys = ["mu_R_median","mu_I_median","sig_R_median","sig_I_median",
            "mu_R_p5","mu_R_p95","mu_I_p5","mu_I_p95",
            "sig_R_p5","sig_R_p95","sig_I_p5","sig_I_p95",
            "mu_mag","sig_mag"]
    with open(os.path.join(out_dir, "foreground_summary.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        f.write(",".join([f"{summary[k]:.6e}" for k in keys]) + "\n")


def save_residual_hist_and_stats(
    vol_cplx: np.ndarray,      # (H,W,C,S) complex noisy/original (current sample)
    deno_real: np.ndarray,     # (H,W,C,S) float32
    deno_imag: np.ndarray,     # (H,W,C,S) float32
    brain_mask: np.ndarray,    # (H,W,S) bool
    out_dir: str,
    sample_idx: int,
    noise_cplx: np.ndarray = None,  # (H,W,C,S_noise) complex, optional
    bins: int = 100,
    coils_to_plot: list = None,
    slices_to_plot: list = None
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    H, W, C, S = vol_cplx.shape
    mask_flat = brain_mask.reshape(-1)

    # ---- Residuals ----
    res_r = np.real(vol_cplx) - deno_real
    res_i = np.imag(vol_cplx) - deno_imag

    # Default coils/slices
    if coils_to_plot is None:
        coils_to_plot = sorted(set([0, C//2, C-1])) if C >= 3 else list(range(C))
    if slices_to_plot is None:
        if S >= 3:
            slices_to_plot = [max(0, S//2-1), S//2, min(S-1, S//2+1)]
        else:
            slices_to_plot = list(range(S))

    # ---- Loop over slices/coils ----
    for z in slices_to_plot:
        m = brain_mask[:, :, z].reshape(-1)

        for c in coils_to_plot:
            r = res_r[:, :, c, z].reshape(-1)[m]
            i = res_i[:, :, c, z].reshape(-1)[m]
            noisy_r = np.real(vol_cplx[:, :, c, z]).reshape(-1)[m]
            noisy_i = np.imag(vol_cplx[:, :, c, z]).reshape(-1)[m]

            if noise_cplx is not None:
                S_noise = noise_cplx.shape[-1]
                nr = noise_cplx.real[:, :, c, :].reshape(-1, S_noise)
                ni = noise_cplx.imag[:, :, c, :].reshape(-1, S_noise)
                nr_m = nr[m, :].ravel()
                ni_m = ni[m, :].ravel()
            else:
                nr_m, ni_m = np.array([]), np.array([])

            # --- Per-coil noisy vs noise ---
            if noisy_r.size or nr_m.size:
                edges = np.linspace(
                    np.quantile(np.concatenate([noisy_r, nr_m]), 0.001),
                    np.quantile(np.concatenate([noisy_r, nr_m]), 0.999),
                    bins+1
                ) if (noisy_r.size and nr_m.size) else bins
                plt.figure()
                if noisy_r.size: plt.hist(noisy_r, bins=edges, density=True, alpha=0.6, label="Noisy MRI (brain)")
                if nr_m.size:    plt.hist(nr_m,    bins=edges, density=True, alpha=0.6, label="Pure noise (brain)")
                plt.title(f"REAL noisy vs noise | sample {sample_idx+1} | coil {c} | slice {z}")
                plt.xlabel("Value"); plt.ylabel("Density"); plt.legend()
                plt.savefig(os.path.join(out_dir, f"hist_noisy_vs_noise_real_s{sample_idx+1}_c{c:02d}_z{z:03d}.png"), dpi=150)
                plt.close()

            if noisy_i.size or ni_m.size:
                edges = np.linspace(
                    np.quantile(np.concatenate([noisy_i, ni_m]), 0.001),
                    np.quantile(np.concatenate([noisy_i, ni_m]), 0.999),
                    bins+1
                ) if (noisy_i.size and ni_m.size) else bins
                plt.figure()
                if noisy_i.size: plt.hist(noisy_i, bins=edges, density=True, alpha=0.6, label="Noisy MRI (brain)")
                if ni_m.size:    plt.hist(ni_m,    bins=edges, density=True, alpha=0.6, label="Pure noise (brain)")
                plt.title(f"IMAG noisy vs noise | sample {sample_idx+1} | coil {c} | slice {z}")
                plt.xlabel("Value"); plt.ylabel("Density"); plt.legend()
                plt.savefig(os.path.join(out_dir, f"hist_noisy_vs_noise_imag_s{sample_idx+1}_c{c:02d}_z{z:03d}.png"), dpi=150)
                plt.close()

            # --- Per-coil residual vs noise ---
            if r.size or nr_m.size:
                edges = np.linspace(
                    np.quantile(np.concatenate([r, nr_m]), 0.001),
                    np.quantile(np.concatenate([r, nr_m]), 0.999),
                    bins+1
                ) if (r.size and nr_m.size) else bins
                plt.figure()
                if r.size:    plt.hist(r,    bins=edges, density=True, alpha=0.6, label="Residual (brain)")
                if nr_m.size: plt.hist(nr_m, bins=edges, density=True, alpha=0.6, label="Pure noise (brain)")
                plt.title(f"REAL residual vs noise | sample {sample_idx+1} | coil {c} | slice {z}")
                plt.xlabel("Value"); plt.ylabel("Density"); plt.legend()
                plt.savefig(os.path.join(out_dir, f"hist_residual_vs_noise_real_s{sample_idx+1}_c{c:02d}_z{z:03d}.png"), dpi=150)
                plt.close()

            if i.size or ni_m.size:
                edges = np.linspace(
                    np.quantile(np.concatenate([i, ni_m]), 0.001),
                    np.quantile(np.concatenate([i, ni_m]), 0.999),
                    bins+1
                ) if (i.size and ni_m.size) else bins
                plt.figure()
                if i.size:    plt.hist(i,    bins=edges, density=True, alpha=0.6, label="Residual (brain)")
                if ni_m.size: plt.hist(ni_m, bins=edges, density=True, alpha=0.6, label="Pure noise (brain)")
                plt.title(f"IMAG residual vs noise | sample {sample_idx+1} | coil {c} | slice {z}")
                plt.xlabel("Value"); plt.ylabel("Density"); plt.legend()
                plt.savefig(os.path.join(out_dir, f"hist_residual_vs_noise_imag_s{sample_idx+1}_c{c:02d}_z{z:03d}.png"), dpi=150)
                plt.close()

        # ---------- Coil-combined overlays ----------
        noisy_cc_slice = np.sqrt((np.abs(vol_cplx[:, :, :, z])**2).sum(axis=2))
        deno_cc_slice  = np.sqrt((deno_real[:, :, :, z]**2 + deno_imag[:, :, :, z]**2).sum(axis=2))
        residual_cc    = noisy_cc_slice - deno_cc_slice

        if noise_cplx is not None:
            noise_cc_all  = np.sqrt((np.abs(noise_cplx)**2).sum(axis=2))  # (H,W,S_noise)
            noise_cc_pool = noise_cc_all.reshape(H*W, -1)[brain_mask[:, :, z].reshape(-1), :].ravel()
        else:
            noise_cc_pool = np.array([])

        # Residual vs noise (coil-combined)
        if residual_cc.size or noise_cc_pool.size:
            edges = np.linspace(
                np.quantile(np.concatenate([residual_cc[brain_mask[:,:,z]], noise_cc_pool]), 0.001),
                np.quantile(np.concatenate([residual_cc[brain_mask[:,:,z]], noise_cc_pool]), 0.999),
                bins+1
            ) if (residual_cc.size and noise_cc_pool.size) else bins
            plt.figure()
            if residual_cc.size: plt.hist(residual_cc[brain_mask[:,:,z]], bins=edges, density=True, alpha=0.6, label="Residual (RSS)")
            if noise_cc_pool.size: plt.hist(noise_cc_pool, bins=edges, density=True, alpha=0.6, label="Pure noise (RSS)")
            plt.title(f"Coil-combined residual vs noise | sample {sample_idx+1} | slice {z}")
            plt.xlabel("Value"); plt.ylabel("Density"); plt.legend()
            plt.savefig(os.path.join(out_dir, f"hist_cc_residual_vs_noise_s{sample_idx+1}_z{z:03d}.png"), dpi=150)
            plt.close()

        # Noisy vs noise (coil-combined) 🔥 NEW
        if noisy_cc_slice.size or noise_cc_pool.size:
            edges = np.linspace(
                np.quantile(np.concatenate([noisy_cc_slice[brain_mask[:,:,z]], noise_cc_pool]), 0.001),
                np.quantile(np.concatenate([noisy_cc_slice[brain_mask[:,:,z]], noise_cc_pool]), 0.999),
                bins+1
            ) if (noisy_cc_slice.size and noise_cc_pool.size) else bins
            plt.figure()
            if noisy_cc_slice.size: plt.hist(noisy_cc_slice[brain_mask[:,:,z]], bins=edges, density=True, alpha=0.6, label="Noisy MRI (RSS)")
            if noise_cc_pool.size:  plt.hist(noise_cc_pool, bins=edges, density=True, alpha=0.6, label="Pure noise (RSS)")
            plt.title(f"Coil-combined noisy vs noise | sample {sample_idx+1} | slice {z}")
            plt.xlabel("Value"); plt.ylabel("Density"); plt.legend()
            plt.savefig(os.path.join(out_dir, f"hist_cc_noisy_vs_noise_s{sample_idx+1}_z{z:03d}.png"), dpi=150)
            plt.close()



def check_complex_residual_zero_mean(
    vol_cplx: np.ndarray,          # (H,W,C,S) complex  - noisy/original
    deno_real: np.ndarray,         # (H,W,C,S) float32 - denoised real
    deno_imag: np.ndarray,         # (H,W,C,S) float32 - denoised imag
    brain_mask: np.ndarray,        # (H,W,S) bool
    out_dir: str,
    sample_idx: int
):
    """
    Compute complex residuals r = noisy - denoised (per-coil, pre-magnitude/RSS)
    and report mean/std in real & imag across masked voxels. Also reports
    fraction of voxels where |denoised| > |noisy| after RSS.

    Saves a CSV of the per-coil stats into out_dir.
    """
    H, W, C, S = vol_cplx.shape
    assert deno_real.shape == (H, W, C, S)
    assert deno_imag.shape == (H, W, C, S)
    assert brain_mask.shape == (H, W, S)

    # Complex residual per-coil
    res_real = np.real(vol_cplx) - deno_real   # (H,W,C,S)
    res_imag = np.imag(vol_cplx) - deno_imag   # (H,W,C,S)

    # Reorder to (H,W,S,C) so mask (H,W,S) can index rows cleanly
    res_real_hwsc = np.transpose(res_real, (0, 1, 3, 2))  # (H,W,S,C)
    res_imag_hwsc = np.transpose(res_imag, (0, 1, 3, 2))  # (H,W,S,C)

    # Flatten spatial dims and apply mask
    res_real_flat = res_real_hwsc.reshape(-1, C)          # (H*W*S, C)
    res_imag_flat = res_imag_hwsc.reshape(-1, C)          # (H*W*S, C)
    mask_flat     = brain_mask.reshape(-1)                # (H*W*S,)

    res_r_m = res_real_flat[mask_flat]                    # (#vox, C)
    res_i_m = res_imag_flat[mask_flat]                    # (#vox, C)

    # Means and (unbiased) stds across masked voxels, per-coil
    mu_r  = res_r_m.mean(axis=0)
    mu_i  = res_i_m.mean(axis=0)
    std_r = res_r_m.std(axis=0, ddof=1) if res_r_m.shape[0] > 1 else res_r_m.std(axis=0)
    std_i = res_i_m.std(axis=0, ddof=1) if res_i_m.shape[0] > 1 else res_i_m.std(axis=0)

    # Combine for a relative bias summary |μ|/σ per coil
    bias_mag   = np.sqrt(mu_r**2 + mu_i**2)
    noise_sc   = np.sqrt(std_r**2 + std_i**2)
    rel_bias   = bias_mag / (noise_sc + 1e-12)

    # Optional: fraction where |den|>|noisy| after RSS (masked)
    noisy_mag = np.sqrt(np.sum(np.abs(vol_cplx)**2, axis=2))          # (H,W,S)
    den_mag   = np.sqrt(deno_real**2 + deno_imag**2).sum(axis=2)**0.5 # (H,W,S)
    frac_up   = np.mean((den_mag - noisy_mag)[brain_mask] > 0)

    # Print a concise report
    print("\n=== Complex per-coil residual zero-mean check (pre-magnitude/RSS) ===")
    print(f"Masked voxels: n = {int(mask_flat.sum())}; Coils = {C}")
    print("Coil |   μ_real (μ/σ)   |   μ_imag (μ/σ)  ")
    for c in range(C):
        br = mu_r[c] / (std_r[c] + 1e-12)
        bi = mu_i[c] / (std_i[c] + 1e-12)
        print(f"{c:>4d} | {mu_r[c]: .3e} ({br: .2f}) | {mu_i[c]: .3e} ({bi: .2f})")
    print("\nSummary over coils:")
    print(f" median |μ|/σ: {np.median(rel_bias):.3f}")
    print(f"   mean |μ|/σ: {np.mean(rel_bias):.3f}")
    print(f"    max |μ|/σ: {np.max(rel_bias):.3f}")
    print(f"Fraction of masked voxels with |denoised| > |noisy| (RSS): {frac_up:.4f}")

    # Save CSV
    csv_path = os.path.join(out_dir, f"complex_residual_stats_sample{sample_idx+1}.csv")
    header = "coil,mu_real,mu_imag,std_real,std_imag,abs_mu_over_sigma"
    rows = []
    for c in range(C):
        rows.append([c, mu_r[c], mu_i[c], std_r[c], std_i[c],
                     np.sqrt(mu_r[c]**2 + mu_i[c]**2) / (np.sqrt(std_r[c]**2 + std_i[c]**2) + 1e-12)])
    np.savetxt(csv_path, np.array(rows), delimiter=",", header=header, comments="", fmt="%.6e")
    print(f"Saved per-coil complex residual stats → {csv_path}")


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
    p_cap  = np.percentile(mag_cc, 99.8)
    S_GLOBAL = 0.99 / (p_cap + 1e-12)
    print(f"[scale] Using S_GLOBAL={S_GLOBAL:.3e} from RSS|mag| 99.8%")
    

    # --- MRI input basic stats (over all voxels) ---
    mri_real = mri_img.real
    mri_imag = mri_img.imag
    mri_mag  = np.abs(mri_img)                               # (H,W,C,S,N)
    mri_cc   = np.sqrt((np.abs(mri_img)**2).sum(axis=2))     # RSS coil-combined → (H,W,S,N)

    def _pr_stats(tag, arr):
        arr = arr.astype(np.float32)
        print(f"[{tag}] min={arr.min():.6g}  max={arr.max():.6g}  "
            f"mean={arr.mean():.6g}  std={arr.std(ddof=1):.6g}")

    _pr_stats("MRI real",        mri_real)
    _pr_stats("MRI imag",        mri_imag)
    _pr_stats("MRI |complex|",   mri_mag)
    _pr_stats("MRI RSS (over C)", mri_cc)


    # --- 1.2) Load brain mask ---
    mask_img = nib.load(args.brain_mask)
    brain_mask = mask_img.get_fdata().astype(bool)  # (H,W,S)
    if brain_mask.shape != (H, W, S):
        raise ValueError(f"Brain mask shape {brain_mask.shape} != {(H,W,S)}")
    bg_mask = ~brain_mask  # background is where mask==0

    # --- Foreground stats (scaled domain) for rescaling naturals -------------------
    # Work on scaled MRI so stats match the network's units
    mri_scaled = mri_img * S_GLOBAL  # (H,W,C,S,N) complex
    per_coil_fg, rss_mag_fg, fg_summary = compute_mri_foreground_stats_scaled(
        mri_scaled, brain_mask=brain_mask, trim_pct=1.0  # 1% trim on each tail
    )

    # Quick printout
    print("\n=== [MRI foreground stats in scaled domain] ===")
    print(f" RSS magnitude: mu={rss_mag_fg['mu_mag']:.3e}, sigma={rss_mag_fg['sig_mag']:.3e}")
    print(f" Complex (per coil) medians: "
        f"mu_R={fg_summary['mu_R_median']:.3e}, mu_I={fg_summary['mu_I_median']:.3e}, "
        f"sig_R={fg_summary['sig_R_median']:.3e}, sig_I={fg_summary['sig_I_median']:.3e}")
    print(f" sig_R range (p5–p95): {fg_summary['sig_R_p5']:.3e}–{fg_summary['sig_R_p95']:.3e}")
    print(f" sig_I range (p5–p95): {fg_summary['sig_I_p5']:.3e}–{fg_summary['sig_I_p95']:.3e}")

    # Save CSVs for training pipeline to consume
    save_mri_foreground_stats(args.output_folder, per_coil_fg, rss_mag_fg, fg_summary)
    print(f"Saved MRI foreground stats to {args.output_folder}")


    # --- 2) Optionally load Noise (complex) ---
    noise_cplx = None
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

        # --- Pure-noise basic stats (over all noise samples) ---
        noise_cplx_scaled = noise_cplx * S_GLOBAL
        report_noise_stats(noise_cplx_scaled, brain_mask=brain_mask)
        sigma_r, sigma_i = report_percoil_noise_stats(noise_cplx_scaled, brain_mask=None)
        out_csv = os.path.join(args.output_folder, "percoil_noise_stats.csv")
        header = "coil,sigma_real,sigma_imag"
        data = np.stack([np.arange(len(sigma_r)), sigma_r, sigma_i], axis=1)
        np.savetxt(out_csv, data, delimiter=",", header=header, comments="", fmt="%.6e")
        print(f"Saved per-coil noise stats → {out_csv}")


        # --- Pure-noise basic stats (over all noise samples) ---
        noise_real = noise_cplx.real
        noise_imag = noise_cplx.imag
        noise_mag  = np.abs(noise_cplx)                          # (H,W,C,S_noise)
        noise_cc   = np.sqrt((np.abs(noise_cplx)**2).sum(axis=2))# RSS coil-combined → (H,W,S_noise)

        def _pr_stats(tag, arr):
            arr = arr.astype(np.float32)
            print(f"[{tag}] min={arr.min():.6g}  max={arr.max():.6g}  "
                f"mean={arr.mean():.6g}  std={arr.std(ddof=1):.6g}")

        _pr_stats("NOISE real",        noise_real)
        _pr_stats("NOISE imag",        noise_imag)
        _pr_stats("NOISE |complex|",   noise_mag)
        _pr_stats("NOISE RSS (over C)", noise_cc)


        ddof = 1 if noise_cplx.shape[-1] > 1 else 0
        sigma_real = noise_cplx.real.std(axis=-1, ddof=ddof).astype(np.float32)  # (H,W,C)
        sigma_imag = noise_cplx.imag.std(axis=-1, ddof=ddof).astype(np.float32)  # (H,W,C)

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

        # ---- Calibrate external sigma maps to MRI background scale (per COIL) ----
        # Only if you actually use external noise maps
        if args.use_noise:
            sigma_real_cal = np.empty_like(sigma_real, dtype=np.float32)  # (H,W,C)
            sigma_imag_cal = np.empty_like(sigma_imag, dtype=np.float32)  # (H,W,C)

            for c in range(C):
                gammas_r, gammas_i = [], []
                for z in range(S):
                    bg = bg_mask[:, :, z]            # background pixels for this slice
                    img_c = vol[:, :, c, z]          # complex slice for this coil

                    # Reference σ from MRI BACKGROUND (complex domain)
                    sig_r_ref = _robust_sigma_1d(img_c.real[bg], method=args.sigma_estimator)
                    sig_i_ref = _robust_sigma_1d(img_c.imag[bg], method=args.sigma_estimator)

                    # Current σ-map scale (use BACKGROUND region for robustness)
                    # Your sigma_real/sigma_imag are (H,W,C) — no slice dim — so compare on the same bg mask
                    sig_r_cur = _finite_median(sigma_real[:, :, c][bg]) if sigma_real is not None else 0.0
                    sig_i_cur = _finite_median(sigma_imag[:, :, c][bg]) if sigma_imag is not None else 0.0

                    if sig_r_ref > 0 and sig_r_cur > 0:
                        gammas_r.append(sig_r_ref / sig_r_cur)
                    if sig_i_ref > 0 and sig_i_cur > 0:
                        gammas_i.append(sig_i_ref / sig_i_cur)

                gamma_r = float(np.median(gammas_r)) if len(gammas_r) else 1.0
                gamma_i = float(np.median(gammas_i)) if len(gammas_i) else 1.0

                sigma_real_cal[:, :, c] = sigma_real[:, :, c] * gamma_r
                sigma_imag_cal[:, :, c] = sigma_imag[:, :, c] * gamma_i

                print(f"[σ-calib] coil {c:02d}: gamma_real={gamma_r:.3g}, gamma_imag={gamma_i:.3g}")

            # Swap in calibrated maps for this sample
            sigma_real = sigma_real_cal
            sigma_imag = sigma_imag_cal

        deno_real = np.empty((H, W, C, S), dtype=np.float32)
        deno_imag = np.empty((H, W, C, S), dtype=np.float32)

        for z in range(S):
            print(f"  Denoising slice {z+1}/{S}")
            for c in range(C):

                img_c  = vol[:, :, c, z]                               # (H,W) complex
                mask2d = brain_mask[:, :, z].astype(np.float32)        # (H,W) 1=brain, 0=bg

                # Mask σ so guidance is zero outside brain
                sig_r = None if sigma_real is None else (sigma_real[:, :, c] * mask2d)
                sig_i = None if sigma_imag is None else (sigma_imag[:, :, c] * mask2d)

                # Run model
                out_r = denoise_channel(
                    model, img_c.real,
                    sig_r, S_GLOBAL, device, tag=f"REAL slice{z} coil{c}"
                )
                out_i = denoise_channel(
                    model, img_c.imag,
                    sig_i, S_GLOBAL, device, tag=f"IMAG slice{z} coil{c}"
                )

                deno_real[:, :, c, z] = mask2d * out_r
                deno_imag[:, :, c, z] = mask2d * out_i

        # Per-coil magnitude after denoising
        deno_mag = np.sqrt(deno_real ** 2 + deno_imag ** 2)     # (H,W,C,S)
        vol_mag = np.abs(vol).astype(np.float32)                # (H,W,C,S)


        # --- 4.5) Complex residual zero-mean diagnostic (per-coil, pre-magnitude/RSS) ---
        check_complex_residual_zero_mean(
            vol_cplx=vol,                    # (H,W,C,S) complex noisy
            deno_real=deno_real,             # (H,W,C,S) float
            deno_imag=deno_imag,             # (H,W,C,S) float
            brain_mask=brain_mask,           # (H,W,S) bool
            out_dir=args.output_folder,
            sample_idx=sid
        )

        # --- Residual histograms + residual stats CSV + pure-noise hist/stats (if available)
        save_residual_hist_and_stats(
            vol_cplx=vol * S_GLOBAL,
            deno_real=deno_real * S_GLOBAL,
            deno_imag=deno_imag * S_GLOBAL,
            brain_mask=brain_mask,
            out_dir=diag_dir,
            sample_idx=sid,
            noise_cplx=noise_cplx * S_GLOBAL  # will be None if not using noise
        )

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
