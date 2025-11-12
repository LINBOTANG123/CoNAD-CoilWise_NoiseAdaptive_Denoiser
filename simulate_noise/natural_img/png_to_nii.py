#!/usr/bin/env python3
"""
png2nii.py — Convert PNG image(s) to NIfTI (.nii)

Examples:
  # Single PNG -> 2D NIfTI
  python png2nii.py --input image.png --out image.nii

  # Folder of PNGs -> 3D volume (stack along Z)
  python png2nii.py --input ./slices --out volume.nii

  # Custom voxel size (mm), keep RGB channels, float32
  python png2nii.py --input ./slices --out volume_rgb.nii --voxel-size 0.5 0.5 1.2 --keep-channels --dtype float32
"""

import argparse
import os
import re
from glob import glob
from typing import List, Tuple

import numpy as np
from PIL import Image
import nibabel as nib


def natural_key(s: str):
    """Sort key that treats numbers in filenames naturally: slice2.png < slice10.png."""
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r'\d+|\D+', s)]


def load_png(path: str, keep_channels: bool, to_dtype: str) -> np.ndarray:
    img = Image.open(path)
    if keep_channels:
        arr = np.array(img)  # shape (H,W,C) for RGB/RGBA or (H,W) for grayscale
        # If RGBA, drop alpha
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
    else:
        # Force single channel (grayscale)
        arr = np.array(img.convert('L'))  # shape (H,W)
    # Cast dtype
    if to_dtype == 'uint8':
        arr = arr.astype(np.uint8, copy=False)
    elif to_dtype == 'uint16':
        # If original <= 255, upscale to use full 16-bit range (optional but sensible)
        if arr.dtype != np.uint16:
            arr = (arr.astype(np.uint16) * 257)  # 255 -> 65535
    elif to_dtype == 'float32':
        # Normalize 0..255 to 0..1 if integer
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unsupported dtype: {to_dtype}")
    return arr


def stack_images(images: List[np.ndarray], axis: int) -> np.ndarray:
    """Stack list of arrays along a new axis."""
    # Ensure consistent spatial shape and channel count
    ref_shape = images[0].shape
    for i, a in enumerate(images):
        if a.shape != ref_shape:
            raise ValueError(f"Image {i} shape {a.shape} does not match first image shape {ref_shape}")
    return np.stack(images, axis=axis)


def build_affine(voxel_size: Tuple[float, float, float], ndim: int) -> np.ndarray:
    """
    Build a simple diagonal affine using voxel sizes.
    For 2D images, uses (vx, vy, 1.0) with a dummy third axis.
    For 3D images, uses (vx, vy, vz).
    """
    if ndim == 2:
        vx, vy = voxel_size[:2]
        vz = 1.0
    else:
        vx, vy, vz = voxel_size
    aff = np.eye(4, dtype=np.float32)
    aff[0, 0] = vx
    aff[1, 1] = vy
    aff[2, 2] = vz
    return aff


def parse_args():
    p = argparse.ArgumentParser(description="Convert PNG image(s) to NIfTI (.nii)")
    p.add_argument("--input", required=True,
                   help="Path to a PNG file or a directory containing PNG files")
    p.add_argument("--out", required=True, help="Output .nii filename")
    p.add_argument("--dtype", choices=["uint8", "uint16", "float32"], default="uint8",
                   help="Output image dtype")
    p.add_argument("--keep-channels", action="store_true",
                   help="Keep RGB channels instead of converting to grayscale")
    p.add_argument("--voxel-size", nargs="+", type=float, default=None,
                   help="Voxel size in mm. 2D: vx vy ; 3D: vx vy vz. Defaults: 1 1 (2D) or 1 1 1 (3D).")
    p.add_argument("--axis", type=int, default=2,
                   help="Axis to stack slices when input is a directory. "
                        "For grayscale: 0/1/2 valid; for RGB: 0/1/3 recommended (stack after channels). "
                        "Default: 2 (Z-axis).")
    p.add_argument("--pattern", default="*.png",
                   help="Glob pattern for PNGs inside directory (default: *.png)")
    p.add_argument("--transpose", nargs="+", type=int, default=None,
                   help="Optional axis permutation after stacking, e.g. 0 2 1. Use with care.")
    return p.parse_args()


def main():
    args = parse_args()

    if os.path.isdir(args.input):
        files = glob(os.path.join(args.input, args.pattern))
        if not files:
            raise FileNotFoundError(f"No files matched {args.pattern} in {args.input}")
        files.sort(key=natural_key)

        # Load first to decide channel behavior
        first = load_png(files[0], keep_channels=args.keep_channels, to_dtype=args.dtype)
        imgs = [first]
        for f in files[1:]:
            imgs.append(load_png(f, keep_channels=args.keep_channels, to_dtype=args.dtype))

        # Decide stacking axis
        if args.keep_channels:
            # For RGB data, typical approach: stack along a new axis AFTER channels
            # data is (H,W,C); stack axis 3 gives (H,W,C,N)
            axis = args.axis
            if axis < 0:
                axis = (first.ndim + 1) + axis
            if axis > first.ndim:
                raise ValueError(f"Invalid axis {args.axis} for RGB data with base ndim {first.ndim}")
            vol = stack_images(imgs, axis=axis)
        else:
            # Grayscale: (H,W) -> stack along new axis (default 2) gives (H,W,N)
            vol = stack_images(imgs, axis=args.axis)

        if args.transpose is not None:
            vol = np.transpose(vol, axes=tuple(args.transpose))

        # Pick voxel size defaults
        if args.voxel_size is None:
            if vol.ndim == 2:
                voxel_size = (1.0, 1.0)
            elif vol.ndim == 3:
                voxel_size = (1.0, 1.0, 1.0)
            elif vol.ndim == 4:
                # e.g., (H,W,C,N) or similar; use 1mm isotropic for spatial dims
                voxel_size = (1.0, 1.0, 1.0)
            else:
                raise ValueError(f"Unsupported ndim: {vol.ndim}")
        else:
            voxel_size = tuple(args.voxel_size)

        # Affine uses first 3 dims as spatial
        affine = build_affine(voxel_size, ndim=3 if vol.ndim >= 3 else 2)

    else:
        # Single PNG
        vol = load_png(args.input, keep_channels=args.keep_channels, to_dtype=args.dtype)
        if args.transpose is not None:
            vol = np.transpose(vol, axes=tuple(args.transpose))
        if args.voxel_size is None:
            voxel_size = (1.0, 1.0)
        else:
            voxel_size = tuple(args.voxel_size)
        affine = build_affine(voxel_size, ndim=2)

    nii = nib.Nifti1Image(vol, affine)
    nib.save(nii, args.out)
    print(f"Saved NIfTI: {args.out}")
    print(f"Shape: {vol.shape}, dtype: {vol.dtype}")


if __name__ == "__main__":
    main()
