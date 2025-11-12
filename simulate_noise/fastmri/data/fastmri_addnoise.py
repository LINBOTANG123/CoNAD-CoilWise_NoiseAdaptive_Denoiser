#!/usr/bin/env python3
"""
Make noisy multi-coil inputs from fastMRI .h5 (brain multicoil) by
adding Gaussian noise directly in image space.

Inputs (.h5): expects dataset "kspace" with shape (S, C, H, W) complex64.
Outputs (NIfTI):
  - <case>_noisy.nii.gz      shape (H, W, C, S, 1)  float32
  - <case>_noise_std.nii.gz  shape (H, W, C, S)     float32 (constant sigma)
  - <case>_clean.nii.gz      shape (H, W, C, S, 1)  float32  (clean, scaled if normalization enabled)

By default, clean coil magnitudes are scaled so the p99.8 is ~1.0,
then Gaussian noise with std = --sigma (in those units) is added.
"""

import os
import glob
import argparse
from typing import Tuple, List

import numpy as np
import h5py
import nibabel as nib


# ----------------------------- FFT utils ----------------------------- #
def ifft2c(kspace: np.ndarray) -> np.ndarray:
    """
    Centered 2D inverse FFT with orthonormal scaling over last two axes.
    kspace: (..., H, W) complex -> complex image with same leading dims.
    """
    x = np.fft.ifftshift(kspace)
    x = np.fft.ifft2(x)
    x = np.fft.fftshift(x)
    return x


# ---------------------------- I/O helpers ---------------------------- #
def find_h5_files(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, "*.h5")))
    elif input_path.endswith(".h5") and os.path.isfile(input_path):
        files = [input_path]
    else:
        files = []
    if not files:
        raise FileNotFoundError("No .h5 files found under '{}'".format(input_path))
    return files


def load_kspace_h5(path: str) -> np.ndarray:
    """
    Load fastMRI k-space from .h5.
    Returns complex64 array of shape (S, C, H, W).
    """
    with h5py.File(path, 'r') as f:
        if 'kspace' not in f:
            raise KeyError("'kspace' dataset not found in {}".format(path))
        ks = f['kspace'][()]  # complex
    if ks.ndim != 4:
        raise ValueError("Unexpected kspace shape {} in {}, expected (S, C, H, W)".format(ks.shape, path))
    return ks.astype(np.complex64, copy=False)


def save_nii(data: np.ndarray, path: str) -> None:
    """Save array as NIfTI with identity affine."""
    img = nib.Nifti1Image(data.astype(np.float32, copy=False), np.eye(4))
    nib.save(img, path)


# ---------------------------- Conversions ---------------------------- #
def mag_per_coil_from_kspace(ks: np.ndarray) -> np.ndarray:
    """
    ks: (S, C, H, W) complex64
    returns magnitude per coil: (H, W, C, S), float32
    """
    img = ifft2c(ks)                  # (S, C, H, W) complex
    mag = np.abs(img).astype(np.float32)
    return np.transpose(mag, (2, 3, 1, 0))  # -> (H, W, C, S)


def normalize_volume(vol: np.ndarray, cap_percentile: float) -> Tuple[np.ndarray, float]:
    """
    vol: (H, W, C, S), float
    Scale so that the given percentile maps to 1.0; clip to [0, 1].
    Returns (scaled_vol, scale_factor).
    """
    p = float(np.percentile(vol, cap_percentile))
    sf = 1.0 / (p + 1e-12)
    vol_scaled = np.clip(vol * sf, 0.0, 1.0)
    return vol_scaled, sf


# ------------------------------- Main -------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Add Gaussian noise in image space to fastMRI brain multicoil .h5")
    ap.add_argument("--input", required=True, help="Path to a .h5 file or directory containing .h5 files")
    ap.add_argument("--output", required=True, help="Output directory for NIfTI files")
    ap.add_argument("--sigma", type=float, required=True,
                    help="Gaussian noise std to add (if normalized, this is in [0,1] units)")
    ap.add_argument("--cap_percentile", type=float, default=99.8,
                    help="Percentile used to scale clean per-coil magnitudes to ~[0,1]")
    ap.add_argument("--no_normalize", action="store_true",
                    help="Disable normalization; sigma is applied in raw intensity units")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    if args.seed is not None:
        np.random.seed(args.seed)

    files = find_h5_files(args.input)
    print("Found {} file(s). Writing to: {}".format(len(files), args.output))

    for fpath in files:
        base = os.path.splitext(os.path.basename(fpath))[0]
        out_noisy = os.path.join(args.output, "{}_noisy.nii.gz".format(base))
        out_noise = os.path.join(args.output, "{}_noise_std.nii.gz".format(base))
        out_clean = os.path.join(args.output, "{}_clean.nii.gz".format(base))

        print("\nProcessing: {}".format(fpath))
        ks = load_kspace_h5(fpath)             # (S, C, H, W)
        S, C, H, W = ks.shape
        print("  kspace shape: (S={}, C={}, H={}, W={})".format(S, C, H, W))

        clean = mag_per_coil_from_kspace(ks)   # (H, W, C, S)
        print("  clean per-coil mag: {} | min={:.4g} max={:.4g} mean={:.4g}".format(
            clean.shape, float(clean.min()), float(clean.max()), float(clean.mean())))

        if args.no_normalize:
            scaled = clean
            scale_factor = 1.0
        else:
            scaled, scale_factor = normalize_volume(clean, args.cap_percentile)

        print("  scale_factor: {:.3e} (cap p{})".format(scale_factor, args.cap_percentile))
        print("  after scaling:  min={:.4g} max={:.4g} mean={:.4g}".format(
            float(scaled.min()), float(scaled.max()), float(scaled.mean())))

        # Save the (clean) per-coil magnitudes with SAME SHAPE as noisy: (H,W,C,S,1)
        clean_5d = scaled[..., None]  # (H, W, C, S, 1)

        # Gaussian noise in image space (iid), constant sigma everywhere
        sigma = float(args.sigma)
        noise = np.random.normal(loc=0.0, scale=sigma, size=scaled.shape).astype(np.float32)
        noisy = scaled + noise

        # keep non-negative magnitudes; if normalized, clamp to [0,1]
        if args.no_normalize:
            noisy = np.clip(noisy, 0.0, None)
        else:
            noisy = np.clip(noisy, 0.0, 1.0)

        # Noise std map: constant sigma everywhere (H, W, C, S)
        noise_std_map = np.full(scaled.shape, sigma, dtype=np.float32)

        # Expand noisy to (H, W, C, S, 1) for your simulate pipeline (N=1)
        noisy_5d = noisy[..., None]

        print("  noisy stats:     min={:.4g} max={:.4g} mean={:.4g}".format(
            float(noisy_5d.min()), float(noisy_5d.max()), float(noisy_5d.mean())))
        print("  noise map stats: min={:.4g} max={:.4g}".format(
            float(noise_std_map.min()), float(noise_std_map.max())))

        # Write files
        save_nii(clean_5d, out_clean)
        save_nii(noisy_5d, out_noisy)
        save_nii(noise_std_map, out_noise)

        print("  wrote: {}".format(out_clean))
        print("  wrote: {}".format(out_noisy))
        print("  wrote: {}".format(out_noise))


if __name__ == "__main__":
    main()
