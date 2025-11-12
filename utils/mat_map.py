import numpy as np
import h5py
import os
import cv2
import argparse
import pdb

def patchwise_kspace_noise(kspace_slice, patch_size=32, step=16, noise_threshold=1e-4):
    """
    Estimates a spatially varying (non-uniform) noise map in k-space using a patch-based approach.

    Args:
        kspace_slice (np.array): Complex k-space data for a single slice, shape: [coils, H, W].
        patch_size   (int)     : Size of the local patch in k-space.
        step         (int)     : Step size (stride) for moving the patch.
        noise_threshold (float): If the mean magnitude in a patch is below this threshold, 
                                 we consider it "noise-only" (no strong signal).

    Returns:
        noise_map (np.array): A 2D float32 array of shape [H, W], representing 
                              the local noise standard deviation in k-space.
    """
    num_coils, H, W = kspace_slice.shape

    # We'll accumulate patch-based noise estimates and an occurrence count per pixel
    noise_map  = np.zeros((H, W), dtype=np.float32)
    counts_map = np.zeros((H, W), dtype=np.float32)

    for top in range(0, H - patch_size + 1, step):
        for left in range(0, W - patch_size + 1, step):
            # Extract patch: shape = [coils, patch_size, patch_size]
            patch_data = kspace_slice[:, top:top+patch_size, left:left+patch_size]

            # Magnitude of patch
            patch_mag = np.abs(patch_data)

            # If this patch is mostly noise (low magnitude),
            # we estimate the local std as the noise level:
            if patch_mag.mean() < noise_threshold:
                patch_std = np.std(patch_mag)  # std across coils & space

                # Accumulate the patch std into the noise map
                noise_map[top:top+patch_size, left:left+patch_size] += patch_std
                counts_map[top:top+patch_size, left:left+patch_size] += 1.0

    # Average out the values in regions that have multiple patch overlaps
    mask = (counts_map > 0)
    noise_map[mask] /= counts_map[mask]

    return noise_map


def process_kspace_nonuniform_noise(input_mat, output_folder,
                                    patch_size=32, step=16, noise_threshold=1e-4):
    """
    Load k-space from .mat, estimate a non-uniform noise map (per slice) using patch-based approach,
    and save both the noise map and the original k-space magnitude as PNG.

    Args:
        input_mat    (str)  : Path to the .mat file (with key "image").
        output_folder(str)  : Folder to save results.
        patch_size   (int)  : Patch size for local noise estimation.
        step         (int)  : Stride for moving the patch.
        noise_threshold(float): Threshold to consider a patch "noise-only."
    """
    os.makedirs(output_folder, exist_ok=True)

    # 1) Load data
    with h5py.File(input_mat, 'r') as f:
        print("Available keys:", list(f.keys()))
        if "image" in f:
            kspace = np.array(f["image"])
        else:
            raise KeyError("Could not find 'image' key in the .mat file.")

    print(f"k-space shape: {kspace.shape}, dtype: {kspace.dtype}")

    # 2) Convert from structured complex if needed
    if kspace.dtype.names:  # check if dtype has ('real', 'imag')
        kspace = kspace['real'] + 1j*kspace['imag']

    # Expect shape: [num_slices, num_samples, num_coils, H, W]
    if len(kspace.shape) != 5:
        raise ValueError(f"Expected 5D k-space, got {kspace.shape}")

    num_slices, num_samples, num_coils, H, W = kspace.shape

    # 3) Process each slice
    for slice_idx in range(num_slices):
        print(f"Processing slice {slice_idx+1}/{num_slices}")

        # Take first sample only
        kspace_slice = kspace[slice_idx, 0, :, :, :]  # shape = [coils, H, W]

        # A) Compute local noise map
        noise_map = patchwise_kspace_noise(kspace_slice,
                                           patch_size=patch_size,
                                           step=step,
                                           noise_threshold=noise_threshold)

        # B) Save noise map as PNG

        noise_map_normalized = cv2.normalize(noise_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        noise_map_file = os.path.join(output_folder, f"noise_map_slice_{slice_idx+1:03d}.png")
        cv2.imwrite(noise_map_file, noise_map_normalized)

        # C) Also save the original k-space magnitude (for reference)
        kspace_magnitude = np.abs(kspace_slice).mean(axis=0)  # average coil dimension
        kspace_magnitude_norm = cv2.normalize(kspace_magnitude, None, 0, 255,
                                              cv2.NORM_MINMAX).astype(np.uint8)
        kspace_magnitude_file = os.path.join(output_folder, f"kspace_slice_{slice_idx+1:03d}.png")
        cv2.imwrite(kspace_magnitude_file, kspace_magnitude_norm)

        print(f"  Saved: {noise_map_file}")
        print(f"  Saved: {kspace_magnitude_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Non-uniform patch-based noise estimation in k-space.")
    parser.add_argument("input_mat", type=str, help="Path to .mat file with 'image' k-space data")
    parser.add_argument("output_folder", type=str, help="Folder to save results")
    parser.add_argument("--patch_size", type=int, default=32, help="Patch size for local noise estimation (default=32)")
    parser.add_argument("--step", type=int, default=16, help="Stride for sliding window (default=16)")
    parser.add_argument("--noise_threshold", type=float, default=1e-10,
                        help="Mean magnitude threshold to treat a patch as noise-only (default=1e-4)")
    args = parser.parse_args()

    process_kspace_nonuniform_noise(args.input_mat,
                                    args.output_folder,
                                    patch_size=args.patch_size,
                                    step=args.step,
                                    noise_threshold=args.noise_threshold)
