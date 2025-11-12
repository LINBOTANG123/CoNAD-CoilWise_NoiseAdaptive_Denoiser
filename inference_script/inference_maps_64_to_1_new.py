#!/usr/bin/env python
import os
import re
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import scipy.io as sio
import cv2
import scipy.ndimage as ndimage

###########################
# MRI & Noise Loading Functions
###########################

def load_mri_kspace_and_transform_raw(file_path, key_kspace="image"):
    """
    Load MRI k-space data from a MATLAB .mat file using scipy.io.loadmat.
    Expected shape in the file: (num_slices, num_samples, num_coils, H, W)
    This function transposes the data to shape (H, W, num_coils, num_slices, num_samples)
    and returns the magnitude image for each k-space slice.
    (Here we assume the data are already reconstructed, so no FFT is applied.)
    """
    data = sio.loadmat(file_path)
    print("Available keys in MRI .mat file:", list(data.keys()))
    if key_kspace not in data:
        raise KeyError(f"Key '{key_kspace}' not found in {file_path}")
    mri_kspace = data[key_kspace]
    # Expected shape: (num_slices, num_samples, num_coils, H, W)
    if mri_kspace.ndim != 5:
        raise ValueError("Expected MRI k-space shape: (num_slices, num_samples, num_coils, H, W). Got shape: " + str(mri_kspace.shape))
    # Transpose to (H, W, num_coils, num_slices, num_samples)
    mri_kspace = np.transpose(mri_kspace, (3, 4, 2, 0, 1))
    H, W, num_coils, num_slices, num_samples = mri_kspace.shape
    print(f"Loaded MRI k-space with shape: {mri_kspace.shape} (H, W, num_coils, num_slices, num_samples)")
    
    # Here we assume that the k-space data is already in image space,
    # so we simply take the absolute value.
    mri_imgspace = np.abs(mri_kspace).astype(np.float32)
    return mri_imgspace

def load_pure_noise_kspace_and_transform_raw(file_path, key_kspace="k_gc"):
    """
    Load pure noise k-space data from a .mat file using scipy.io.loadmat.
    Expected shape: (H, W, num_coils, num_slices)
    Returns a numpy array of magnitude images with shape (H, W, num_coils, num_slices).
    """
    data = sio.loadmat(file_path)
    print("Available keys in pure noise .mat file:", list(data.keys()))
    if key_kspace not in data:
        raise KeyError(f"Key '{key_kspace}' not found in {file_path}")
    noise_kspace = data[key_kspace]
    if noise_kspace.ndim != 4:
        raise ValueError("Expected noise k-space shape: (H, W, num_coils, num_slices). Got shape: " + str(noise_kspace.shape))
    H, W, num_coils, num_slices = noise_kspace.shape
    print(f"Loaded pure noise k-space with shape: {noise_kspace.shape}")
    noise_imgspace = np.abs(noise_kspace).astype(np.float32)
    return noise_imgspace

###########################
# Noise Map Expansion
###########################

def replicate_noise_map(noise_imgspace, target_H, target_W, target_slices):
    """
    Given noise_imgspace of shape (H_noise, W_noise, num_coils, noise_slices),
    for each coil:
      - Pad the noise map (for each available slice) with edge-padding to expand from (H_noise, W_noise)
        to (target_H, target_W).
      - For slices beyond the available noise_slices, fill with the average noise map (computed from available slices).
    Returns an array of shape (target_H, target_W, num_coils, target_slices).
    """
    H_noise, W_noise, num_coils, noise_slices = noise_imgspace.shape
    expanded_noise = np.zeros((target_H, target_W, num_coils, target_slices), dtype=noise_imgspace.dtype)
    
    for coil in range(num_coils):
        coil_maps = []
        # Process each available noise slice.
        for s in range(noise_slices):
            noise_map = noise_imgspace[:, :, coil, s]
            pad_h = target_H - H_noise
            pad_w = target_W - W_noise
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            padded_map = np.pad(noise_map, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')
            # Crop to exactly target dimensions.
            padded_map = padded_map[:target_H, :target_W]
            coil_maps.append(padded_map)
        coil_maps = np.stack(coil_maps, axis=2)  # shape: (target_H, target_W, noise_slices)
        # Compute average map for this coil.
        avg_map = np.mean(coil_maps, axis=2)  # (target_H, target_W)
        # Build final map: if target_slices <= noise_slices, take first target_slices.
        final_maps = np.zeros((target_H, target_W, target_slices), dtype=coil_maps.dtype)
        if target_slices <= noise_slices:
            final_maps = coil_maps[:, :, :target_slices]
        else:
            final_maps[:, :, :noise_slices] = coil_maps
            for s in range(noise_slices, target_slices):
                final_maps[:, :, s] = avg_map
        expanded_noise[:, :, coil, :] = final_maps
    return expanded_noise

def load_model(model_path, device='cuda'):
    from basicsr.models.archs.restormer_arch import Restormer
    # Model configured for 64-channel input (32 noisy coils + 32 noise maps) and 1-channel output (RSS combined)
    net_g = Restormer(
        inp_channels=64,
        out_channels=1,  # 1-channel output (coil-combined RSS)
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='BiasFree',
        dual_pixel_task=False
    )
    net_g.to(device)
    ckpt = torch.load(model_path, map_location=device)
    net_g.load_state_dict(ckpt['params'], strict=True)
    net_g.eval()
    return net_g

###########################
# Inference Pipeline
###########################

def run_inference_from_mat(model_pth, mri_mat_file, noise_mat_file, output_folder, device='cuda',
                           data_scale_factor=1):
    """
    Perform inference with a Restormer-like model using MRI and noise MAT files.
    Each raw MRI image is multiplied by data_scale_factor.
    The noise maps are loaded, scaled, and then expanded using edge-padding for spatial dimensions,
    and for missing slices the average noise is used.
    For each MRI slice:
      - 32 coil images are combined to compute an RSS (root-sum-of-squares) image.
      - A 2-channel input per coil is formed by stacking the scaled MRI image (one channel)
        and the corresponding expanded noise map (one channel), resulting in a 64-channel input.
      - The model outputs a single-channel RSS combined denoised image.
      - The residual is computed as the difference between the original RSS and the denoised output.
    The outputs are saved as PNG images per slice and assembled into 3D NIfTI volumes.
    """
    os.makedirs(output_folder, exist_ok=True)
    png_out_dir = os.path.join(output_folder, "png_outputs")
    os.makedirs(png_out_dir, exist_ok=True)
    nii_dir = os.path.join(output_folder, "nii")
    os.makedirs(nii_dir, exist_ok=True)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = load_model(model_pth, device=device)
    
    # Load noise maps.
    noise_imgspace = load_pure_noise_kspace_and_transform_raw(noise_mat_file, key_kspace="k_gc")
    # noise_imgspace shape: (H_noise, W_noise, num_coils, noise_slices)
    H_noise, W_noise, num_coils, noise_slices = noise_imgspace.shape
    print(f"Loaded pure noise k-space with shape: {noise_imgspace.shape}")
    
    # Load MRI images.
    mri_imgspace = load_mri_kspace_and_transform_raw(mri_mat_file, key_kspace="k_gc")
    # mri_imgspace shape: (H, W, num_coils, mri_slices, num_samples)
    H, W, num_coils_mri, mri_slices, num_samples = mri_imgspace.shape
    print(f"Processing MRI data with shape: {mri_imgspace.shape}")
    
    # Expand noise maps to match MRI dimensions using replication (edge padding and average for missing slices)
    expanded_noise_maps = replicate_noise_map(noise_imgspace, target_H=H, target_W=W, target_slices=mri_slices)
    # expanded_noise_maps has shape: (H, W, num_coils, mri_slices)
    
    # Rearrange noise maps for inference loop: shape (mri_slices, num_coils, H, W)
    noise_maps_all = np.transpose(expanded_noise_maps, (3, 2, 0, 1))
    
    # Save average noise maps (averaged over slices) for record.
    avg_noise = np.mean(expanded_noise_maps, axis=3)  # (H, W, num_coils)
    avg_noise = np.transpose(avg_noise, (1, 2, 0))       # (H, W, num_coils)
    noise_nii_path = os.path.join(output_folder, "noise_maps.nii")
    nib.save(nib.Nifti1Image(avg_noise, affine=np.eye(4)), noise_nii_path)
    print(f"Saved noise maps NIfTI: {noise_nii_path}")
    
    original_slices = []   # List of arrays (H, W) per slice (RSS combined)
    denoised_slices = []   # List of arrays (H, W) per slice.
    residual_slices = []   # List of arrays (H, W) per slice.
    
    sample_idx = 0  # Process one sample.
    
    for slice_idx in range(mri_slices):
        mri_slice = []
        for coil in range(num_coils_mri):
            img = mri_imgspace[:, :, coil, slice_idx, sample_idx] * data_scale_factor
            mri_slice.append(img)
        mri_slice = np.stack(mri_slice, axis=0)  # shape: (num_coils, H, W)
        
        # Compute the original RSS image.
        original_rss = np.sqrt(np.sum(np.square(mri_slice), axis=0))
        original_slices.append(original_rss)
        
        # Extract the noise map for the current slice (shape: (num_coils, H, W)).
        noise_maps_slice = noise_maps_all[slice_idx]
        
        # Form the 64-channel input: concatenate the MRI slice (32 channels) with the noise maps (32 channels).
        input_multi_coil = np.concatenate([mri_slice, noise_maps_slice], axis=0)  # (64, H, W)
        inp_tensor = torch.from_numpy(input_multi_coil).unsqueeze(0).to(device)
        
        # Pad if necessary.
        _, _, h, w = inp_tensor.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            inp_tensor = F.pad(inp_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        
        with torch.no_grad():
            output = model(inp_tensor)
        if pad_h > 0 or pad_w > 0:
            output = output[..., :h, :w]
        output = torch.clamp(output, 0, 1)
        # Expecting output shape: (1, 1, H, W)
        output = output[0, 0].detach().cpu().numpy()
        denoised_slices.append(output)
        
        residual = original_rss - output
        residual_slices.append(residual)
        
        base_name = f"slice_{slice_idx:03d}"
        cv2.imwrite(os.path.join(png_out_dir, f"{base_name}_original.png"),
                    (original_rss*255).clip(0,255).astype(np.uint8))
        cv2.imwrite(os.path.join(png_out_dir, f"{base_name}_denoised.png"),
                    (output*255).clip(0,255).astype(np.uint8))
        res_norm = (residual - residual.min()) / (residual.max()-residual.min() + 1e-8)
        cv2.imwrite(os.path.join(png_out_dir, f"{base_name}_residual.png"),
                    (res_norm*255).clip(0,255).astype(np.uint8))
        
        print(f"Processed slice {slice_idx}.")
    
    original_vol = np.stack(original_slices, axis=0)
    denoised_vol = np.stack(denoised_slices, axis=0)
    residual_vol = np.stack(residual_slices, axis=0)
    
    original_vol = np.transpose(original_vol, (1, 2, 0)) / data_scale_factor
    denoised_vol = np.transpose(denoised_vol, (1, 2, 0)) / data_scale_factor
    residual_vol = np.transpose(residual_vol, (1, 2, 0)) / data_scale_factor
    
    orig_nii_path = os.path.join(nii_dir, "original_rss.nii")
    denoised_nii_path = os.path.join(nii_dir, "denoised_rss.nii")
    residual_nii_path = os.path.join(nii_dir, "residual_rss.nii")
    nib.save(nib.Nifti1Image(original_vol, affine=np.eye(4)), orig_nii_path)
    nib.save(nib.Nifti1Image(denoised_vol, affine=np.eye(4)), denoised_nii_path)
    nib.save(nib.Nifti1Image(residual_vol, affine=np.eye(4)), residual_nii_path)
    print(f"Saved original RSS volume as {orig_nii_path}")
    print(f"Saved denoised RSS volume as {denoised_nii_path}")
    print(f"Saved residual RSS volume as {residual_nii_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Inference from MAT inputs (64-ch input, 1-ch output: RSS combined)")
    parser.add_argument('--model_pth', type=str, required=True,
                        help='Path to the trained .pth model file')
    parser.add_argument('--mri_mat', type=str, required=True,
                        help='Path to the MRI k-space .mat file')
    parser.add_argument('--noise_mat', type=str, required=True,
                        help='Path to the pure noise k-space .mat file')
    parser.add_argument('--output_folder', type=str, default='./results_infer',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Computation device, e.g., cuda or cpu')
    parser.add_argument('--data_scale_factor', type=float, default=6e4,
                        help='Scaling factor applied to both MRI images and noise maps')
    return parser.parse_args()

def main():
    args = parse_args()
    run_inference_from_mat(args.model_pth, args.mri_mat, args.noise_mat, args.output_folder,
                           device=args.device, data_scale_factor=args.data_scale_factor)

if __name__ == "__main__":
    main()
