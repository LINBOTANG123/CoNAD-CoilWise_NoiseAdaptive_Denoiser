#!/usr/bin/env python
import os
import argparse
from typing import Optional

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import h5py
import scipy.io as sio


# ------------------------ FFT helpers ------------------------

def process_kspace_to_img(kspace_2d: np.ndarray) -> np.ndarray:
    """
    2D k-space -> magnitude image (float32)
      1) ifftshift
      2) 2D IFFT
      3) fftshift
      4) magnitude
    """
    shifted = np.fft.ifftshift(kspace_2d)
    img_complex = np.fft.ifft2(shifted)
    img_complex = np.fft.fftshift(img_complex)
    return np.abs(img_complex).astype(np.float32)

def process_kspace_to_img_complex(kspace_2d: np.ndarray) -> np.ndarray:
    """
    2D k-space -> complex image (complex64)
      1) ifftshift
      2) 2D IFFT
      3) fftshift
    """
    shifted = np.fft.ifftshift(kspace_2d)
    img_complex = np.fft.ifft2(shifted)
    img_complex = np.fft.fftshift(img_complex)
    return img_complex.astype(np.complex64)


# ------------------------ I/O helpers ------------------------

def load_array(path: str, key: str):
    """Load a dataset from .mat or .h5 by key and return np.ndarray."""
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as f:
            if key not in f:
                raise KeyError(f"Key '{key}' not found in {path}. Keys: {list(f.keys())}")
            arr = f[key][()]
    else:
        mat = sio.loadmat(path)
        if key not in mat:
            raise KeyError(f"Key '{key}' not found in {path}. Keys: {list(mat.keys())}")
        arr = mat[key]
    # If MATLAB struct with real/imag fields
    if hasattr(arr, "dtype") and arr.dtype.kind == "V":
        fields = arr.dtype.names or []
        if set(["real", "imag"]).issubset(fields):
            arr = arr["real"] + 1j * arr["imag"]
        else:
            arr = arr[fields[0]]
    return np.array(arr)


# ------------------------ Model ------------------------

def load_model(model_path: str, device: str = 'cuda') -> torch.nn.Module:
    """Load Restormer configured for 2-channel input (img, sigma)."""
    from basicsr.models.archs.restormer_arch import Restormer
    net = Restormer(
        inp_channels=2,          # fixed: img + sigma
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
    ckpt = torch.load(model_path, map_memory=True, map_location=device) if hasattr(torch, "load") else torch.load(model_path, map_location=device)
    net.load_state_dict(ckpt['params'], strict=True)
    net.eval()
    return net


def denoise_magnitude(model, img_mag: np.ndarray, sigma_mag: np.ndarray,
                      scale: float, device: str) -> np.ndarray:
    """
    Denoise a single (H,W) magnitude image using a (H,W) sigma map as 2nd channel.
    Returns the denoised magnitude (H,W) in original scale.
    """
    arr = np.stack([(img_mag * scale).astype(np.float32),
                    (sigma_mag * scale).astype(np.float32)], axis=0)
    inp = torch.from_numpy(arr).unsqueeze(0).to(device)  # (1,2,H,W)
    _, _, h, w = inp.shape
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    inp_p = F.pad(inp, (0, pw, 0, ph), mode='reflect') if (ph or pw) else inp
    with torch.no_grad():
        out = model(inp_p)[..., :h, :w]
    return (out.squeeze().cpu().numpy().astype(np.float32)) / scale


# ------------------------ Main ------------------------

def main():
    p = argparse.ArgumentParser(
        description="Inference for multi-coil MRI denoising (magnitude-only) with 2-channel input (img + sigma). "
                    "Loads arrays directly by key with shapes: MRI (H,W,C,S,N), Noise (H,W,C,S_noise). "
                    "Can convert MRI and/or Noise from k-space to image space."
    )
    # Files + keys
    p.add_argument('--model_pth', required=True, help='Restormer checkpoint (.pth), trained for 2-channel input')
    p.add_argument('--mri_file', required=True, help='.mat or .h5 file containing MRI data')
    p.add_argument('--mri_key', required=True, help='Key for MRI array (H,W,coils,slices,dwis)')
    p.add_argument('--noise_file', required=True, help='.mat or .h5 file containing noise data')
    p.add_argument('--noise_key', required=True, help='Key for noise array (H,W,coils,noise_slices)')

    # Flags: whether MRI / Noise are in k-space
    p.add_argument('--mri_is_kspace', action='store_true', help='Set if MRI array is k-space and needs IFFT.')
    p.add_argument('--noise_is_kspace', action='store_true', help='Set if noise array is k-space and needs IFFT.')

    # Optional brain mask
    p.add_argument('--brain_mask', default=None,
                   help='Optional brain mask NIfTI (.nii/.nii.gz). Must be (H,W,S). If absent, full FOV is used.')

    # DWI selection and output
    p.add_argument('--dwi_index', type=int, default=None,
                   help='If set, only denoise this DWI index (0-based).')
    p.add_argument('--max_dirs', type=int, default=None,
                   help='If set, only process first K diffusion directions')
    p.add_argument('--output_folder', default='./results_infer', help='Output directory')

    args = p.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- 1) Load arrays ----
    mri_arr = load_array(args.mri_file, args.mri_key)         # (H,W,C,S,N)
    noise_arr = load_array(args.noise_file, args.noise_key)   # (H,W,C,S_noise)

    # Sanity shapes
    if mri_arr.ndim != 5:
        raise ValueError(f"MRI array must be 5D (H,W,coils,slices,dwis). Got {mri_arr.shape}")
    if noise_arr.ndim != 4:
        raise ValueError(f"Noise array must be 4D (H,W,coils,noise_slices). Got {noise_arr.shape}")

    H, W, C, S, N = mri_arr.shape
    Hn, Wn, Cn, Sn = noise_arr.shape
    if (H, W, C) != (Hn, Wn, Cn):
        raise ValueError(f"Spatial/coil mismatch: MRI (HWC)={(H,W,C)} vs Noise (HWC)={(Hn,Wn,Cn)}")

    # ---- 2) Convert to image space if needed ----
    # MRI: want complex image (to later compute magnitudes per coil/slice)
    if args.mri_is_kspace:
        mri_img = np.empty((H, W, C, S, N), dtype=np.complex64)
        for c in range(C):
            for s in range(S):
                for n in range(N):
                    mri_img[:, :, c, s, n] = process_kspace_to_img_complex(mri_arr[:, :, c, s, n])
    else:
        # If already image space: accept complex or float; promote to complex if needed
        if np.iscomplexobj(mri_arr):
            mri_img = mri_arr.astype(np.complex64)
        else:
            mri_img = mri_arr.astype(np.float32).astype(np.complex64)

    # Noise: need complex image samples to estimate Rayleigh sigma
    if args.noise_is_kspace:
        noise_img = np.empty((H, W, C, Sn), dtype=np.complex64)
        for c in range(C):
            for s in range(Sn):
                noise_img[:, :, c, s] = process_kspace_to_img_complex(noise_arr[:, :, c, s])
    else:
        if np.iscomplexobj(noise_arr):
            noise_img = noise_arr.astype(np.complex64)
        else:
            # If noise is magnitude only, we cannot compute Rayleigh sigma from |n| properly.
            # We still promote to complex with zero imaginary part, but this will *underestimate* σ.
            # Prefer providing complex noise or k-space noise.
            noise_img = noise_arr.astype(np.float32).astype(np.complex64)

    print("Loaded MRI (complex image):", mri_img.shape)
    print("Loaded Noise (complex image):", noise_img.shape)

    # ---- 3) Build coil-combined RSS and scaling ----
    mag_cc = np.sqrt((np.abs(mri_img) ** 2).sum(axis=2))  # (H,W,S,N)
    vals = mag_cc.reshape(-1).astype(np.float64)
    vals = vals[np.isfinite(vals)]
    mu = vals.mean() if vals.size else 0.0
    sig = (vals.std(ddof=1) if vals.size > 1 else 0.0)
    target = mu + 3.0 * sig
    S_GLOBAL = 0.99 / (target + 1e-12)
    print(f"[scale] S_GLOBAL={S_GLOBAL:.3e} (μ+3σ mapping)")

    # ---- 4) Build per-voxel Rayleigh σ map from complex noise ----
    # σ_rayleigh = sqrt(mean(|n|^2)/2) across noise samples
    nmag_sq_mean = (np.abs(noise_img) ** 2).mean(axis=-1).astype(np.float32)  # (H,W,C)
    sigma_mag = np.sqrt(np.maximum(nmag_sq_mean, 0.0) / 2.0).astype(np.float32)

    nib.save(nib.Nifti1Image(sigma_mag, np.eye(4)),
             os.path.join(args.output_folder, 'sigma_mag_rayleigh.nii'))
    print("Saved sigma_mag_rayleigh.nii")

    # ---- 5) Optional brain mask ----
    if args.brain_mask is not None:
        mask_img = nib.load(args.brain_mask)
        brain_mask = mask_img.get_fdata().astype(bool)  # (H,W,S)
        if brain_mask.shape != (H, W, S):
            raise ValueError(f"Brain mask shape {brain_mask.shape} != {(H,W,S)}")
    else:
        print("No brain_mask provided → using full FOV.")
        brain_mask = np.ones((H, W, S), dtype=bool)

    # ---- 6) Load model ----
    model = load_model(args.model_pth, device=device)

    # ---- 7) Denoising loop (per coil, per slice), 2-channel input (mag, sigma) ----
    all_deno, all_res, all_orig = [], [], []

    # Which DWIs to process
    if args.dwi_index is not None:
        dwi_indices = [args.dwi_index]
    else:
        dwi_indices = list(range(N))
        if args.max_dirs is not None:
            dwi_indices = dwi_indices[:args.max_dirs]

    for sid in dwi_indices:
        print(f"Denoising DWI {sid+1}/{N}")
        vol = mri_img[..., sid]  # (H,W,C,S) complex
        vol_mag = np.abs(vol).astype(np.float32)  # (H,W,C,S)

        H_, W_, C_, S_ = vol_mag.shape
        deno_mag = np.empty((H_, W_, C_, S_), dtype=np.float32)
        res_mag  = np.empty_like(deno_mag)

        for z in range(S_):
            mask2d = brain_mask[:, :, z].astype(np.float32)
            for c in range(C_):
                img_mag_c = vol_mag[:, :, c, z]
                sig_map_c = sigma_mag[:, :, c]

                # Mask inputs (brain region only)
                img_in = img_mag_c * mask2d
                sig_in = sig_map_c * mask2d

                out_mag = denoise_magnitude(model, img_in, sig_in, S_GLOBAL, device)
                out_mag *= mask2d  # keep output masked

                deno_mag[:, :, c, z] = out_mag
                res_mag[:, :, c, z]  = img_mag_c - out_mag

        # Coil-combined (RSS)
        orig_cc = np.sqrt((vol_mag ** 2).sum(axis=2)) * brain_mask.astype(np.float32)
        deno_cc = np.sqrt((deno_mag ** 2).sum(axis=2))
        residual_cc = orig_cc - deno_cc

        all_orig.append(orig_cc)
        all_deno.append(deno_cc)
        all_res.append(residual_cc)

    # ---- 8) Save combined outputs ----
    orig4d = np.stack(all_orig, axis=-1) if all_orig else None
    deno4d = np.stack(all_deno, axis=-1) if all_deno else None
    res4d  = np.stack(all_res,  axis=-1) if all_res  else None

    if orig4d is not None:
        nib.save(nib.Nifti1Image(orig4d.astype(np.float32), np.eye(4)),
                 os.path.join(args.output_folder, 'original_coilcombined_all.nii'))
    if deno4d is not None:
        nib.save(nib.Nifti1Image(deno4d.astype(np.float32), np.eye(4)),
                 os.path.join(args.output_folder, 'combined_denoised_all.nii'))
    if res4d is not None:
        nib.save(nib.Nifti1Image(res4d.astype(np.float32), np.eye(4)),
                 os.path.join(args.output_folder, 'combined_residual_all.nii'))

    print("All outputs saved in", args.output_folder)


if __name__ == "__main__":
    main()
