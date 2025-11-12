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
    For PNGs read via cv2 (8-bit), simply dividing by 255 is enough.
    """
    # Here we assume PNGs are 8-bit so that values are in [0,255].
    return img.astype(np.float32) / 255.0

def load_model(model_path, device='cuda'):
    from basicsr.models.archs.restormer_arch import Restormer
    # Model configured for 64-channel input (32 noisy coils + 32 noise maps) 
    # and 32-channel output (one denoised output per coil)
    net_g = Restormer(
        inp_channels=64,
        out_channels=32,
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
    if not files:
        raise ValueError(f"No PNG files found in {folder}")
    
    # Determine maximum slice and coil indices from filenames.
    max_slice, max_coil = 0, 0
    for f in files:
        s, c = parse_coil_slice(f)
        if s is None:
            continue
        max_slice = max(max_slice, s)
        max_coil = max(max_coil, c)
    
    # Read one sample to get image dimensions.
    sample_img = cv2.imread(os.path.join(folder, files[0]), cv2.IMREAD_GRAYSCALE)
    H, W = sample_img.shape
    volume = np.zeros((H, W, max_coil, max_slice), dtype=np.float32)
    for f in files:
        s, c = parse_coil_slice(f)
        if s is None:
            continue
        img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
        # Normalize the PNG image (assumed 8-bit).
        img_norm = normalize_to_unit_interval(img)
        volume[:, :, c-1, s-1] = img_norm
    return volume

def coil_combine_rss(volume):
    """
    Compute the Root Sum-of-Squares (RSS) of a multi-coil volume.
    Expects volume shape (H, W, num_coils) and returns (H, W).
    """
    return np.sqrt(np.sum(np.square(volume), axis=2))

def run_inference_from_png(model_pth, mri_folder, noise_folder, output_folder, device='cuda'):
    """
    Perform inference using a Restormer-like model on inputs read from PNG files.
    
    The MRI folder contains noisy MRI PNGs and the noise folder contains pure noise PNGs.
    Both are expected to follow the naming convention 'slice_###_coil_###.png'.
    
    Steps:
      1. Load the MRI volume (shape: (H, W, num_coils, num_slices))
      2. Load the noise volume (shape: (H, W, num_coils, num_slices))
      3. Compute a noise map for each coil by averaging over slices.
      4. For each slice:
         a. Extract the multi-coil MRI slice.
         b. Compute its RSS (this is the "original" coil-combined image).
         c. Form a 64-channel input by concatenating the MRI slice (32 channels) and the noise maps (32 channels).
         d. Run inference through the model (output shape: (1, 32, H, W)).
         e. Compute per-coil residuals (MRI slice minus model output).
         f. Save per-slice, per-coil PNGs.
      5. Assemble all slices into 4D NIfTI volumes.
    """
    os.makedirs(output_folder, exist_ok=True)
    png_out_dir = os.path.join(output_folder, "png_outputs")
    os.makedirs(png_out_dir, exist_ok=True)
    nii_dir = os.path.join(output_folder, "nii")
    os.makedirs(nii_dir, exist_ok=True)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = load_model(model_pth, device=device)
    
    # Load MRI and noise volumes.
    print("Loading MRI images from:", mri_folder)
    mri_volume = load_png_volume(mri_folder)  # (H, W, num_coils, num_slices)
    print("Loading noise images from:", noise_folder)
    noise_volume = load_png_volume(noise_folder)  # (H, W, num_coils, num_slices)
    
    H, W, num_coils, num_slices = mri_volume.shape
    print(f"MRI volume shape: {(H, W, num_coils, num_slices)}")
    print(f"Noise volume shape: {(H, W, num_coils, num_slices)}")
    
    # Compute noise maps: average over slices for each coil.
    noise_map_dict = {}
    for coil in range(num_coils):
        avg_noise_map = np.mean(noise_volume[:, :, coil, :], axis=2)  # (H, W)
        noise_map_dict[coil+1] = avg_noise_map
    sorted_keys = sorted(noise_map_dict.keys())
    noise_maps = np.stack([noise_map_dict[k] for k in sorted_keys], axis=-1)  # (H, W, num_coils)
    # Transpose to (num_coils, H, W)
    noise_maps = np.transpose(noise_maps, (2, 0, 1))
    
    # Save noise maps as a NIfTI volume for record.
    noise_nii_path = os.path.join(output_folder, "noise_maps.nii")
    nib.save(nib.Nifti1Image(np.transpose(noise_maps, (1, 2, 0)), affine=np.eye(4)), noise_nii_path)
    print(f"Saved noise maps NIfTI: {noise_nii_path}")
    
    original_slices = []   # List to store per-slice original MRI (multi-coil) images (shape: (num_coils, H, W))
    denoised_slices = []   # List to store per-slice denoised outputs (shape: (num_coils, H, W))
    residual_slices = []   # List to store per-slice residuals (shape: (num_coils, H, W))
    
    for slice_idx in range(num_slices):
        # Build multi-coil MRI slice: shape (num_coils, H, W)
        mri_slice = []
        for coil in range(num_coils):
            mri_slice.append(mri_volume[:, :, coil, slice_idx])
        mri_slice = np.stack(mri_slice, axis=0)
        original_slices.append(mri_slice)
        
        # Compute the original coil-combined image for visualization.
        original_rss = coil_combine_rss(np.transpose(mri_slice, (1, 2, 0)))  # (H, W)
        
        # Form the 64-channel input.
        # The MRI slice is already (num_coils, H, W); noise_maps is (num_coils, H, W).
        input_multi_coil = np.concatenate([mri_slice, noise_maps], axis=0)  # (64, H, W)
        inp_tensor = torch.from_numpy(input_multi_coil).unsqueeze(0).to(device)
        
        # Ensure dimensions are multiples of 8.
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
        # Expected output shape: (1, 32, H, W)
        output = output[0].detach().cpu().numpy()  # now shape (32, H, W)
        denoised_slices.append(output)
        
        # Compute per-coil residuals: (num_coils, H, W)
        residual = mri_slice - output
        residual_slices.append(residual)
        
        # Save per-slice, per-coil PNG outputs.
        for coil in range(num_coils):
            base_name = f"slice_{slice_idx:03d}_coil_{coil+1:03d}"
            cv2.imwrite(os.path.join(png_out_dir, f"{base_name}_original.png"),
                        (mri_slice[coil]*255).clip(0,255).astype(np.uint8))
            cv2.imwrite(os.path.join(png_out_dir, f"{base_name}_denoised.png"),
                        (output[coil]*255).clip(0,255).astype(np.uint8))
            # Normalize residual for visualization.
            res = residual[coil]
            res_norm = (res - res.min()) / (res.max()-res.min() + 1e-8)
            cv2.imwrite(os.path.join(png_out_dir, f"{base_name}_residual.png"),
                        (res_norm*255).clip(0,255).astype(np.uint8))
        
        print(f"Processed slice {slice_idx}.")
    
    # Assemble 4D volumes: desired shape (H, W, num_slices, num_coils)
    original_vol = np.stack(original_slices, axis=0)   # (num_slices, num_coils, H, W)
    denoised_vol = np.stack(denoised_slices, axis=0)   # (num_slices, num_coils, H, W)
    residual_vol = np.stack(residual_slices, axis=0)     # (num_slices, num_coils, H, W)
    original_vol = np.transpose(original_vol, (2, 3, 0, 1))  # (H, W, num_slices, num_coils)
    denoised_vol = np.transpose(denoised_vol, (2, 3, 0, 1))
    residual_vol = np.transpose(residual_vol, (2, 3, 0, 1))
    
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
    parser = argparse.ArgumentParser(
        description="Inference from PNG inputs for a 64-channel input and 32-channel output model."
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
    parser.add_argument('--data_scale_factor', type=float, default=1.0,
                        help='Scaling factor applied to the MRI images if needed (should be 1.0 if images are normalized)')
    return parser.parse_args()

def main():
    args = parse_args()
    run_inference_from_png(args.model_pth, args.mri_folder, args.noise_folder, args.output_folder,
                           device=args.device)

if __name__ == "__main__":
    main()
