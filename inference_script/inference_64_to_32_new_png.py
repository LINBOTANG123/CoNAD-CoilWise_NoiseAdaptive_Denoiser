import os
import argparse
import numpy as np
import cv2
import scipy.io as sio
import nibabel as nib
import torch
import torch.nn.functional as F
import pdb

from basicsr.models.archs.restormer_arch import Restormer

# Number of coil groups
NUM_COILS = 32

def interleave_channels(x):
    """
    Rearranges a tensor of shape [B, 64, H, W] where the first 32 channels are coil images 
    and the next 32 channels are noise maps, into interleaved order:
    [coil1, noise1, coil2, noise2, ..., coil32, noise32].
    """
    B, C, H, W = x.shape
    assert C == 64, "Expected 64 channels (32 coil images + 32 noise maps)"
    coils = x[:, :32, :, :]
    noise = x[:, 32:, :, :]
    x_int = torch.stack([coils, noise], dim=2)  # shape: [B, 32, 2, H, W]
    x_int = x_int.view(B, 64, H, W)
    return x_int

def process_noise_kspace(kspace_2d):
    """
    Process a 2D noise k-space slice:
      - ifftshift, 2D IFFT, fftshift, and magnitude extraction.
    Returns a float32 magnitude image.
    """
    shifted = np.fft.ifftshift(kspace_2d)
    img_complex = np.fft.ifft2(shifted)
    img_complex = np.fft.fftshift(img_complex)
    img_mag = np.abs(img_complex).astype(np.float32)
    return img_mag

def replicate_noise_map(noise_imgspace, target_H, target_W, target_slices):
    """
    Given noise_imgspace of shape (H_noise, W_noise, num_coils, noise_slices),
    for each coil:
      - Pad each noise map (using edge-padding) to (target_H, target_W).
      - For slices beyond available noise_slices, fill with the average noise map.
    Returns an array of shape (target_H, target_W, num_coils, target_slices).
    """
    H_noise, W_noise, num_coils, noise_slices = noise_imgspace.shape
    expanded_noise = np.zeros((target_H, target_W, num_coils, target_slices), dtype=noise_imgspace.dtype)
    for coil in range(num_coils):
        coil_maps = []
        for s in range(noise_slices):
            noise_map = noise_imgspace[:, :, coil, s]
            pad_h = target_H - H_noise
            pad_w = target_W - W_noise
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            padded_map = np.pad(noise_map, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')
            padded_map = padded_map[:target_H, :target_W]
            coil_maps.append(padded_map)
        coil_maps = np.stack(coil_maps, axis=2)
        avg_map = np.mean(coil_maps, axis=2)
        final_maps = np.zeros((target_H, target_W, target_slices), dtype=coil_maps.dtype)
        if target_slices <= noise_slices:
            final_maps = coil_maps[:, :, :target_slices]
        else:
            final_maps[:, :, :noise_slices] = coil_maps
            for s in range(noise_slices, target_slices):
                final_maps[:, :, s] = avg_map
        expanded_noise[:, :, coil, :] = final_maps
    return expanded_noise

def load_mri_data(mat_path, key='k_gc'):
    """
    Load MRI noisy data from a .mat file.
    Expected shape: (num_slices, num_samples, num_coils, H, W).
    Select sample index 0 and rearrange to (H, W, num_coils, mri_slices).
    """
    data = sio.loadmat(mat_path)
    if key not in data:
        raise KeyError(f"Key '{key}' not found in {mat_path}")
    mri_data = data[key]  # shape: (num_slices, num_samples, num_coils, H, W)
    num_slices, num_samples, num_coils, H, W = mri_data.shape
    print("MRI data shape:", mri_data.shape)
    # Select sample index 0 and rearrange dimensions to (H, W, num_coils, num_slices)
    mri_img = mri_data[:, 0, :, :, :]  # shape: (num_slices, num_coils, H, W)
    mri_img = np.transpose(mri_img, (2, 3, 1, 0))  # -> (H, W, num_coils, num_slices)
    mri_img = np.abs(mri_img).astype(np.float32)
    print("After processing, MRI image range: min =", mri_img.min(), "max =", mri_img.max())
    return mri_img

def load_noise_data(mat_path, key='k_gc'):
    """
    Load pure noise k-space data from a .mat file.
    Expected shape: (H, W, num_coils, noise_slices).
    Process each 2D slice with IFFT to produce magnitude images.
    """
    data = sio.loadmat(mat_path)
    if key not in data:
        raise KeyError(f"Key '{key}' not found in {mat_path}")
    noise_kspace = data[key]
    H, W, num_coils, noise_slices = noise_kspace.shape
    print("Noise k-space shape:", noise_kspace.shape)
    noise_imgspace = np.zeros((H, W, num_coils, noise_slices), dtype=np.float32)
    for coil in range(num_coils):
        for s in range(noise_slices):
            kspace_slice = noise_kspace[:, :, coil, s]
            noise_imgspace[:, :, coil, s] = process_noise_kspace(kspace_slice)
    print("After processing, noise image range: min =", noise_imgspace.min(), "max =", noise_imgspace.max())
    return noise_imgspace

def load_model(model_path, device='cuda'):
    """
    Load the multi-coil Restormer model.
    The model is configured with:
      - inp_channels=64 (32 coil images + 32 noise maps)
      - out_channels=32 (one denoised image per coil)
    """
    net_g = Restormer(
        inp_channels=64,
        out_channels=32,
        dim=48,
        num_blocks=[4,6,6,8],
        num_refinement_blocks=4,
        heads=[1,2,4,8],
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

def normalize_for_png(image):
    """
    Normalize a single image array (2D) to the [0,1] range based on its min and max.
    """
    im_min = image.min()
    im_max = image.max()
    if im_max > im_min:
        norm_image = (image - im_min) / (im_max - im_min)
    else:
        norm_image = image.copy()
    return norm_image

def main():
    parser = argparse.ArgumentParser(description="Inference for multi-coil MRI denoising.")
    parser.add_argument('--model_pth', type=str, required=True, help='Path to the trained model checkpoint (.pth)')
    parser.add_argument('--mri_mat', type=str, required=True, help='Path to the MRI .mat file (noisy data)')
    parser.add_argument('--noise_mat', type=str, required=True, help='Path to the pure noise .mat file')
    parser.add_argument('--output_folder', type=str, default='./results_infer', help='Folder to save outputs')
    parser.add_argument('--data_scale_factor', type=float, default=1, help='Scaling factor for data (applied to both MRI and noise maps)')
    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    ### Load and save MRI data as PNG images ###
    # Load MRI data; shape: (H, W, num_coils, mri_slices)
    mri_img = load_mri_data(args.mri_mat, key='k_gc')
    H, W, num_coils, mri_slices = mri_img.shape
    print("MRI image shape (one sample):", mri_img.shape)
    
    # Save original MRI data as PNGs (using 8-bit encoding after normalization)
    mri_png_folder = os.path.join(args.output_folder, "mri_png")
    os.makedirs(mri_png_folder, exist_ok=True)
    for s in range(mri_slices):
        for coil in range(num_coils):
            image = mri_img[:, :, coil, s]
            image_norm = normalize_for_png(image)
            # Scale to 0-255 and cast to uint8 for 8-bit PNG
            image_uint8 = (image_norm * 255).astype(np.uint8)
            filename = os.path.join(mri_png_folder, f"mri_slice{s:03d}_coil{coil:02d}.png")
            cv2.imwrite(filename, image_uint8)
    print(f"Saved MRI PNGs to {mri_png_folder}")
    
    # Read MRI data back from PNGs (values will be in [0,1] after dividing by 255)
    mri_img_from_png = np.zeros_like(mri_img)
    for s in range(mri_slices):
        for coil in range(num_coils):
            filename = os.path.join(mri_png_folder, f"mri_slice{s:03d}_coil{coil:02d}.png")
            image_uint8 = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            if image_uint8 is None:
                raise FileNotFoundError(f"Could not read {filename}")
            image_float = image_uint8.astype(np.float32) / 255.0
            mri_img_from_png[:, :, coil, s] = image_float
    # Use the re-read MRI images for further processing
    mri_img = mri_img_from_png

    # Save the original MRI data (before scaling) as a 4D NIfTI volume.
    # Rearrange from (H, W, num_coils, mri_slices) to (H, W, mri_slices, num_coils)
    original_vol = np.transpose(mri_img, (0, 1, 3, 2))
    original_nii_path = os.path.join(args.output_folder, "original_4d.nii")
    nib.save(nib.Nifti1Image(original_vol.astype(np.float32), affine=np.eye(4)), original_nii_path)
    print(f"Saved original volume as {original_nii_path}")

    ### Load and save Noise data as PNG images ###
    # Load noise data; shape: (H_noise, W_noise, num_coils, noise_slices)
    noise_imgspace = load_noise_data(args.noise_mat, key='k_gc')
    H_noise, W_noise, _, noise_slices = noise_imgspace.shape
    print("Noise k-space processed shape:", noise_imgspace.shape)
    
    # Save noise images (calculated noise maps) as PNGs (using 8-bit encoding after normalization)
    noise_png_folder = os.path.join(args.output_folder, "noise_png")
    os.makedirs(noise_png_folder, exist_ok=True)
    for coil in range(noise_imgspace.shape[2]):
        for s in range(noise_imgspace.shape[3]):
            image = noise_imgspace[:, :, coil, s]
            image_norm = normalize_for_png(image)
            image_uint8 = (image_norm * 255).astype(np.uint8)
            filename = os.path.join(noise_png_folder, f"noise_coil{coil:02d}_slice{s:03d}.png")
            cv2.imwrite(filename, image_uint8)
    print(f"Saved Noise PNGs to {noise_png_folder}")
    
    # Read noise data back from PNGs
    noise_imgspace_from_png = np.zeros_like(noise_imgspace)
    for coil in range(noise_imgspace.shape[2]):
        for s in range(noise_imgspace.shape[3]):
            filename = os.path.join(noise_png_folder, f"noise_coil{coil:02d}_slice{s:03d}.png")
            image_uint8 = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            if image_uint8 is None:
                raise FileNotFoundError(f"Could not read {filename}")
            image_float = image_uint8.astype(np.float32) / 255.0
            noise_imgspace_from_png[:, :, coil, s] = image_float
    # Use the re-read noise images for further processing
    noise_imgspace = noise_imgspace_from_png

    # Replicate noise maps to match MRI spatial dimensions and number of slices.
    noise_maps = replicate_noise_map(noise_imgspace, H, W, mri_slices)
    print("Replicated noise maps shape:", noise_maps.shape)

    # Apply data scaling.
    scale = args.data_scale_factor
    mri_img_scaled = mri_img * scale
    noise_maps_scaled = noise_maps * scale

    # Build model inputs: for each MRI slice, create a tensor of shape [64, H, W]
    # (first 32 channels: coil images, next 32 channels: noise maps)
    input_slices = []
    for s in range(mri_slices):
        coil_imgs = mri_img_scaled[:, :, :, s]      # (H, W, num_coils)
        noise_slice = noise_maps_scaled[:, :, :, s]   # (H, W, num_coils)
        coil_imgs = np.transpose(coil_imgs, (2, 0, 1))      # (num_coils, H, W)
        noise_slice = np.transpose(noise_slice, (2, 0, 1))    # (num_coils, H, W)
        slice_input = np.concatenate([coil_imgs, noise_slice], axis=0)  # (64, H, W)
        input_slices.append(torch.from_numpy(slice_input))
    
    # Create folder for denoised PNGs
    denoised_png_folder = os.path.join(args.output_folder, "denoised_png")
    os.makedirs(denoised_png_folder, exist_ok=True)

    # Process slices one-by-one to avoid GPU OOM.
    denoised_list = []
    residual_list = []
    model = load_model(args.model_pth, device='cuda')
    
    for s in range(mri_slices):
        slice_input = input_slices[s].unsqueeze(0).float().cuda()  # shape: [1, 64, H, W]
        # Pad slice so that H and W are divisible by 8.
        _, _, H_in, W_in = slice_input.shape
        pad_h = (8 - H_in % 8) % 8
        pad_w = (8 - W_in % 8) % 8
        if pad_h or pad_w:
            slice_input = F.pad(slice_input, (0, pad_w, 0, pad_h), mode='reflect')
        with torch.no_grad():
            pred = model(slice_input)  # Expected shape: [1, 32, H_out, W_out]
        pred = pred[:, :, :H_in, :W_in]  # Crop to original H and W.
        # Compute residual (difference between denoised output and original coil images)
        original_coil = slice_input[:, :32, :H_in, :W_in]
        residual = pred - original_coil
        denoised_list.append(pred.cpu())
        residual_list.append(residual.cpu())
        print(f"Processed slice {s+1}/{mri_slices}")
        
        # Save the denoised output for each coil as PNG images.
        pred_np = pred.cpu().numpy()  # shape: [1, 32, H, W]
        for coil in range(pred_np.shape[1]):
            image = pred_np[0, coil, :, :]
            image_norm = normalize_for_png(image)
            image_uint8 = (image_norm * 255).astype(np.uint8)
            filename = os.path.join(denoised_png_folder, f"denoised_slice{s:03d}_coil{coil:02d}.png")
            cv2.imwrite(filename, image_uint8)
    
    # Stack results along the slice dimension.
    denoised_vol = torch.cat(denoised_list, dim=0).numpy()   # shape: [mri_slices, 32, H, W]
    residual_vol = torch.cat(residual_list, dim=0).numpy()     # shape: [mri_slices, 32, H, W]
    # Rearrange to (H, W, mri_slices, 32)
    denoised_vol = np.transpose(denoised_vol, (2, 3, 0, 1))
    residual_vol = np.transpose(residual_vol, (2, 3, 0, 1))
    
    # Revert scaling for output volumes.
    denoised_vol = denoised_vol / scale
    residual_vol = residual_vol / scale

    # Save volumes as NIfTI files.
    denoised_nii_path = os.path.join(args.output_folder, "denoised_4d.nii")
    residual_nii_path = os.path.join(args.output_folder, "residual_4d.nii")
    nib.save(nib.Nifti1Image(denoised_vol.astype(np.float32), affine=np.eye(4)), denoised_nii_path)
    nib.save(nib.Nifti1Image(residual_vol.astype(np.float32), affine=np.eye(4)), residual_nii_path)
    print(f"Saved denoised volume as {denoised_nii_path}")
    print(f"Saved residual volume as {residual_nii_path}")

if __name__ == "__main__":
    main()
