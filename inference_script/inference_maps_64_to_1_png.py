#!/usr/bin/env python
import os
import re
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import cv2

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

def normalize_to_unit_interval(img):
    """
    Normalize a numpy array to [0,1] using min-max normalization.
    """
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val - min_val == 0:
        return np.zeros_like(img)
    return (img - min_val) / (max_val - min_val)

def load_model(model_path, device='cuda'):
    from basicsr.models.archs.restormer_arch import Restormer
    # Model configured for 64-channel input (32 MRI coils + 32 noise maps) and 1-channel output (RSS combined)
    net_g = Restormer(
        inp_channels=64,
        out_channels=1,
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

def load_png_volume(folder):
    """
    Load all PNG images from a folder and arrange them into a volume.
    Assumes filenames follow the pattern 'slice_###_coil_###.png'.
    Returns a numpy array of shape (H, W, num_coils, num_slices).
    """
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.png')])
    if len(files)==0:
        raise ValueError(f"No PNG files found in {folder}")
    max_slice = 0
    max_coil = 0
    for f in files:
        s, c = parse_coil_slice(f)
        if s is None:
            continue
        if s > max_slice:
            max_slice = s
        if c > max_coil:
            max_coil = c
    sample_img = cv2.imread(os.path.join(folder, files[0]), cv2.IMREAD_GRAYSCALE)
    H, W = sample_img.shape
    volume = np.zeros((H, W, max_coil, max_slice), dtype=np.float32)
    for f in files:
        s, c = parse_coil_slice(f)
        if s is None:
            continue
        img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0
        volume[:, :, c-1, s-1] = img
    return volume

def coil_combine_rss(volume):
    """
    Compute the Root Sum-of-Squares (RSS) of a multi-coil volume.
    Expects volume shape (H, W, num_coils) and returns (H, W).
    """
    rss = np.sqrt(np.sum(np.square(volume), axis=2))
    return rss

def run_inference_from_png(model_pth, mri_folder, noise_folder, output_folder, device='cuda'):
    """
    Perform inference using a Restormer-like model on inputs read from PNG files.
    The MRI folder and noise folder should each contain PNGs following the naming
    convention 'slice_###_coil_###.png'.
    """
    os.makedirs(output_folder, exist_ok=True)
    png_out_dir = os.path.join(output_folder, "png_outputs")
    os.makedirs(png_out_dir, exist_ok=True)
    nii_dir = os.path.join(output_folder, "nii")
    os.makedirs(nii_dir, exist_ok=True)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = load_model(model_pth, device=device)
    
    print("Loading MRI images from:", mri_folder)
    mri_volume = load_png_volume(mri_folder)  # (H, W, num_coils, num_slices)
    # Add a sample dimension -> (H, W, num_coils, num_slices, 1)
    mri_volume = mri_volume[..., np.newaxis]
    
    print("Loading noise images from:", noise_folder)
    noise_volume = load_png_volume(noise_folder)  # (H, W, num_coils, num_slices)
    
    H, W, num_coils, num_slices = mri_volume.shape[:4]
    print(f"MRI volume shape: {(H, W, num_coils, num_slices)}")
    print(f"Noise volume shape: {(H, W, num_coils, num_slices)}")
    
    # Compute noise maps: average over slices for each coil.
    noise_map_dict = {}
    for coil in range(num_coils):
        avg_noise_map = np.mean(noise_volume[:, :, coil, :], axis=2)  # (H, W)
        noise_map_dict[coil+1] = avg_noise_map
    sorted_keys = sorted(noise_map_dict.keys())
    noise_maps = np.stack([noise_map_dict[k] for k in sorted_keys], axis=-1)
    noise_maps = np.transpose(noise_maps, (2, 0, 1))  # (num_coils, H, W)
    
    noise_nii_path = os.path.join(output_folder, "noise_maps.nii")
    nib.save(nib.Nifti1Image(np.transpose(noise_maps, (1,2,0)), affine=np.eye(4)), noise_nii_path)
    print(f"Saved noise maps NIfTI: {noise_nii_path}")
    
    original_slices = []   # To store original RSS images (per slice)
    denoised_slices = []   # To store denoised outputs (per slice)
    residual_slices = []   # To store residuals (per slice)
    
    sample_idx = 0  # Only one sample
    
    for slice_idx in range(num_slices):
        # Build a multi-coil MRI slice (initial shape: (num_coils, H, W))
        mri_slice = []
        for coil in range(num_coils):
            img = mri_volume[:, :, coil, slice_idx, sample_idx]
            mri_slice.append(img)
        mri_slice = np.stack(mri_slice, axis=0)  # (num_coils, H, W)
        # Transpose so that coil axis is last: (H, W, num_coils)
        mri_slice = np.transpose(mri_slice, (1, 2, 0))
        
        # Compute original RSS image for the slice: (H, W)
        original_rss = coil_combine_rss(mri_slice)
        original_slices.append(original_rss)
        
        # Form 64-channel input by concatenating MRI slice and noise maps.
        # For the model, we need the MRI slice in original order (num_coils, H, W), so we re-transpose.
        mri_slice_for_input = np.transpose(mri_slice, (2, 0, 1))  # (num_coils, H, W)
        input_multi_coil = np.concatenate([mri_slice_for_input, noise_maps], axis=0)  # (64, H, W)
        inp_tensor = torch.from_numpy(input_multi_coil).unsqueeze(0).to(device)
        
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
        output = torch.clamp(output, 0, 1)
        # Expected output shape: (1, 1, H, W)
        output = output[0, 0].detach().cpu().numpy()
        denoised_slices.append(output)
        
        # Compute residual.
        residual = original_rss - output
        residual_slices.append(residual)
        
        # Save per-slice PNGs.
        base_name = f"slice_{slice_idx:03d}"
        cv2.imwrite(os.path.join(png_out_dir, f"{base_name}_original.png"),
                    (original_rss*255).clip(0,255).astype(np.uint8))
        cv2.imwrite(os.path.join(png_out_dir, f"{base_name}_denoised.png"),
                    (output*255).clip(0,255).astype(np.uint8))
        res_norm = (residual - residual.min()) / (residual.max()-residual.min() + 1e-8)
        cv2.imwrite(os.path.join(png_out_dir, f"{base_name}_residual.png"),
                    (res_norm*255).clip(0,255).astype(np.uint8))
        
        print(f"Processed slice {slice_idx}.")
    
    # Assemble 3D volumes.
    original_vol = np.stack(original_slices, axis=0).transpose(1, 2, 0)
    denoised_vol = np.stack(denoised_slices, axis=0).transpose(1, 2, 0)
    residual_vol = np.stack(residual_slices, axis=0).transpose(1, 2, 0)
    
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
    parser = argparse.ArgumentParser(
        description="Inference from PNG inputs with two subfolders (MRI images and noise images)."
    )
    parser.add_argument('--model_pth', type=str, required=True,
                        help='Path to the trained .pth model file')
    parser.add_argument('--mri_folder', type=str, required=True,
                        help='Path to the folder containing MRI PNG images')
    parser.add_argument('--noise_folder', type=str, required=True,
                        help='Path to the folder containing noise PNG images')
    parser.add_argument('--output_folder', type=str, default='./results_infer',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Computation device, e.g., cuda or cpu')
    return parser.parse_args()

def main():
    args = parse_args()
    run_inference_from_png(args.model_pth, args.mri_folder, args.noise_folder, args.output_folder,
                           device=args.device)

if __name__ == "__main__":
    main()
