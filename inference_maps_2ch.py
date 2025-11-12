#!/usr/bin/env python
import os
import re
import cv2
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import scipy.io as sio

def parse_coil_slice(filename):
    """
    Extract slice and coil indices from filenames like:
       slice_001_coil_001.png
    Returns (slice_idx, coil_idx) as integers.
    """
    match = re.search(r"slice_(\d+)_coil_(\d+)\.png", filename, re.IGNORECASE)
    if not match:
        return None, None
    slice_idx = int(match.group(1))
    coil_idx = int(match.group(2))
    return slice_idx, coil_idx

def load_model(model_path, device='cuda'):
    from basicsr.models.archs.restormer_arch import Restormer
    # Model configured for 2-channel input and 1-channel output.
    net_g = Restormer(
        inp_channels=2,   # Channel 1: MRI image, Channel 2: Noise map
        out_channels=1,   # Output: denoised image for the coil
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

def load_mri_kspace_and_transform_raw(file_path, key_kspace="k_gc"):
    """
    Load MRI k-space data from a .mat file.
    Expected shape: (H, W, num_coils, num_slices, num_samples)
    Returns raw magnitude images as a numpy array of shape (H, W, num_coils, num_slices, num_samples).
    """
    data = sio.loadmat(file_path)
    print("Available keys in MRI .mat file:", data.keys())
    if key_kspace not in data:
        raise KeyError(f"Key '{key_kspace}' not found in {file_path}")
    mri_kspace = data[key_kspace]
    if mri_kspace.ndim != 5:
        raise ValueError("Expected MRI k-space shape: (H, W, num_coils, num_slices, num_samples).")
    H, W, num_coils, num_slices, num_samples = mri_kspace.shape
    print(f"Loaded MRI k-space with shape: {mri_kspace.shape}")
    mri_imgspace = np.zeros((H, W, num_coils, num_slices, num_samples), dtype=np.float32)
    for sample_idx in range(num_samples):
        for slice_idx in range(num_slices):
            for coil_idx in range(num_coils):
                kspace_2d = mri_kspace[:, :, coil_idx, slice_idx, sample_idx]
                shifted = np.fft.ifftshift(kspace_2d)
                img_complex = np.fft.ifft2(shifted)
                img_complex = np.fft.fftshift(img_complex)
                img_mag = np.abs(img_complex).astype(np.float32)
                mri_imgspace[:, :, coil_idx, slice_idx, sample_idx] = img_mag
    return mri_imgspace

def load_pure_noise_kspace_and_transform_raw(file_path, key_kspace="k_gc"):
    """
    Load pure noise k-space data from a .mat file.
    Expected shape: (H, W, num_coils, num_slices)
    Returns raw magnitude images as a numpy array of shape (H, W, num_coils, num_slices).
    """
    data = sio.loadmat(file_path)
    print("Available keys in pure noise .mat file:", data.keys())
    if key_kspace not in data:
        raise KeyError(f"Key '{key_kspace}' not found in {file_path}")
    noise_kspace = data[key_kspace]
    if noise_kspace.ndim != 4:
        raise ValueError("Expected noise k-space shape: (H, W, num_coils, num_slices).")
    H, W, num_coils, num_slices = noise_kspace.shape
    print(f"Loaded pure noise k-space with shape: {noise_kspace.shape}")
    noise_imgspace = np.zeros((H, W, num_coils, num_slices), dtype=np.float32)
    for coil_idx in range(num_coils):
        for slice_idx in range(num_slices):
            kspace_2d = noise_kspace[:, :, coil_idx, slice_idx]
            shifted = np.fft.ifftshift(kspace_2d)
            img_complex = np.fft.ifft2(shifted)
            img_complex = np.fft.fftshift(img_complex)
            img_mag = np.abs(img_complex).astype(np.float32)
            noise_imgspace[:, :, coil_idx, slice_idx] = img_mag
    return noise_imgspace

def load_noise_map_from_mat(noise_mat_file, key_kspace="k_gc", data_scale_factor=1):
    """
    Load pure noise k-space, perform coil-wise iFFT and average across slices for each coil.
    Then, scale the result by data_scale_factor.
    Returns a dictionary mapping coil index (1-indexed) to the scaled noise map.
    """
    noise_imgspace = load_pure_noise_kspace_and_transform_raw(noise_mat_file, key_kspace)
    H, W, num_coils, num_slices = noise_imgspace.shape
    noise_map_dict = {}
    for coil_idx in range(num_coils):
        avg_noise_map = np.mean(noise_imgspace[:, :, coil_idx, :], axis=2)
        avg_noise_map = avg_noise_map * data_scale_factor
        noise_map_dict[coil_idx+1] = avg_noise_map
    return noise_map_dict

def run_inference_from_mat(model_pth, mri_mat_file, noise_mat_file, output_folder, device='cuda',
                           data_scale_factor=1):
    """
    Perform inference with a Restormer-like model using MRI and noise MAT files.
    For each slice and for each coil, the raw MRI image is multiplied by data_scale_factor
    (to bring it to a similar range as natural images), and the corresponding noise map is loaded
    (already scaled by data_scale_factor). A 2-channel input is formed per coil and passed to
    a model (configured with inp_channels=2, out_channels=1). The output is the denoised image
    for that coil.
    
    The script saves per-slice, per-coil PNGs for the original MRI image, the denoised output, 
    and the residual (original - denoised). It also assembles these into 4D NIfTI volumes with 
    dimensions (H, W, num_slices, num_coils).
    """
    os.makedirs(output_folder, exist_ok=True)
    png_out_dir = os.path.join(output_folder, "png_outputs")
    os.makedirs(png_out_dir, exist_ok=True)
    nii_dir = os.path.join(output_folder, "nii")
    os.makedirs(nii_dir, exist_ok=True)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = load_model(model_pth, device=device)
    
    # Load noise maps (scaled)
    noise_map_dict = load_noise_map_from_mat(noise_mat_file, key_kspace="k_gc", data_scale_factor=data_scale_factor)
    if not noise_map_dict:
        print("Error: No noise maps loaded from", noise_mat_file)
        return
    sorted_coil_keys = sorted(noise_map_dict.keys())
    noise_maps = np.stack([noise_map_dict[k] for k in sorted_coil_keys], axis=-1)
    noise_maps = np.transpose(noise_maps, (2, 0, 1))  # shape: (num_coils, H, W)
    noise_nii_path = os.path.join(output_folder, "noise_maps.nii")
    nib.save(nib.Nifti1Image(np.transpose(noise_maps, (1, 2, 0)), affine=np.eye(4)), noise_nii_path)
    print(f"Saved noise maps NIfTI: {noise_nii_path}")
    
    # Load MRI images.
    mri_imgspace = load_mri_kspace_and_transform_raw(mri_mat_file, key_kspace="k_gc")
    H, W, num_coils, num_slices, num_samples = mri_imgspace.shape
    print(f"Processing MRI data with shape: {mri_imgspace.shape}")
    
    # Process one sample (sample_idx = 0)
    sample_idx = 1
    
    # Lists to collect per-slice data.
    original_slices = []   # (num_slices, num_coils, H, W)
    denoised_slices = []   # (num_slices, num_coils, H, W)
    residual_slices = []   # (num_slices, num_coils, H, W)
    
    for slice_idx in range(num_slices):
        mri_slice = []
        for coil in range(num_coils):
            # Multiply raw MRI image by data_scale_factor.
            img = mri_imgspace[:, :, coil, slice_idx, sample_idx] * data_scale_factor
            mri_slice.append(img)
        mri_slice = np.stack(mri_slice, axis=0)  # shape: (num_coils, H, W)
        original_slices.append(mri_slice)
        
        denoised_coils = []
        residual_coils = []
        for coil in range(num_coils):
            # For each coil, get the corresponding MRI image and noise map.
            mri_img = mri_slice[coil]  # shape: (H, W)
            noise_map = noise_maps[coil]  # shape: (H, W)
            # Form a 2-channel input.
            input_2ch = np.stack([mri_img, noise_map], axis=0)  # shape: (2, H, W)
            inp_tensor = torch.from_numpy(input_2ch).unsqueeze(0).to(device)  # shape: (1, 2, H, W)
            # Pad if needed.
            _, _, h, w = inp_tensor.shape
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            if pad_h > 0 or pad_w > 0:
                inp_tensor = F.pad(inp_tensor, (0, pad_w, 0, pad_h), mode='reflect')
            with torch.no_grad():
                output = model(inp_tensor)
            if pad_h > 0 or pad_w > 0:
                output = output[..., :h, :w]
            # output = torch.clamp(output, 0, 1)
            # Output shape: (1, 1, H, W)
            output_img = output[0, 0].detach().cpu().numpy()  # (H, W)
            denoised_coils.append(output_img)
            residual_img = mri_img - output_img
            residual_coils.append(residual_img)
            
            # Save PNGs for this coil.
            base_name = f"slice_{slice_idx:03d}_coil_{coil+1:03d}"
            cv2.imwrite(os.path.join(png_out_dir, f"{base_name}_original.png"),
                        (mri_img).clip(0,1)*255)
            cv2.imwrite(os.path.join(png_out_dir, f"{base_name}_denoised.png"),
                        (output_img).clip(0,1)*255)
            # For residual, perform min-max normalization for visualization.
            res_norm = (residual_img - residual_img.min()) / (residual_img.max() - residual_img.min() + 1e-8)
            cv2.imwrite(os.path.join(png_out_dir, f"{base_name}_residual.png"),
                        (res_norm*255).clip(0,255).astype(np.uint8))
        
        denoised_coils = np.stack(denoised_coils, axis=0)   # (num_coils, H, W)
        residual_coils = np.stack(residual_coils, axis=0)   # (num_coils, H, W)
        denoised_slices.append(denoised_coils)
        residual_slices.append(residual_coils)
        print(f"Processed slice {slice_idx}.")
    
    # Assemble 4D volumes.
    original_vol = np.stack(original_slices, axis=0)   # (num_slices, num_coils, H, W)
    denoised_vol = np.stack(denoised_slices, axis=0)   # (num_slices, num_coils, H, W)
    residual_vol = np.stack(residual_slices, axis=0)     # (num_slices, num_coils, H, W)
    # Transpose to (H, W, num_slices, num_coils)
    original_vol = np.transpose(original_vol, (2, 3, 0, 1)) / data_scale_factor
    denoised_vol = np.transpose(denoised_vol, (2, 3, 0, 1)) / data_scale_factor
    residual_vol = np.transpose(residual_vol, (2, 3, 0, 1)) / data_scale_factor
    
    orig_nii_path = os.path.join(nii_dir, "original_4d.nii")
    denoised_nii_path = os.path.join(nii_dir, "denoised_4d.nii")
    residual_nii_path = os.path.join(nii_dir, "residual_4d.nii")
    nib.save(nib.Nifti1Image(original_vol, affine=np.eye(4)), orig_nii_path)
    nib.save(nib.Nifti1Image(denoised_vol, affine=np.eye(4)), denoised_nii_path)
    nib.save(nib.Nifti1Image(residual_vol, affine=np.eye(4)), residual_nii_path)
    print(f"Saved 4D original volume as {orig_nii_path}")
    print(f"Saved 4D denoised volume as {denoised_nii_path}")
    print(f"Saved 4D residual volume as {residual_nii_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Inference from MAT inputs (2-ch input, 1-ch output per coil)")
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
    parser.add_argument('--data_scale_factor', type=float, default=5e4,
                        help='Scaling factor applied to MRI images and noise maps')
    return parser.parse_args()

def main():
    args = parse_args()
    run_inference_from_mat(args.model_pth, args.mri_mat, args.noise_mat, args.output_folder,
                           device=args.device, data_scale_factor=args.data_scale_factor)

if __name__ == "__main__":
    main()
