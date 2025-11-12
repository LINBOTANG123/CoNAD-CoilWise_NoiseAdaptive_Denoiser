import os
import argparse
import numpy as np
import scipy.io as sio
import nibabel as nib
import torch
import torch.nn.functional as F

NUM_COILS = 32  # Total number of coils

def process_kspace_to_img(kspace_2d):
    """
    Process a 2D k-space slice:
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

def load_mri_data(mat_path, key='k_gc', sample=0):
    """
    Load MRI noisy k-space data from a .mat file.
    Expected shape: (146, 146, 32, 40, 4) with dimensions
      (x, y, num_coils, num_slices, sample).
    This function takes the sample at index 2 and converts each
    2D k-space slice to image space, returning data of shape:
      (146, 146, num_coils, num_slices)
    """
    data = sio.loadmat(mat_path)
    if key not in data:
        raise KeyError(f"Key '{key}' not found in {mat_path}")
    mri_data = data[key]  # shape: (146, 146, 32, 40, 4)
    print("ORIGINAL SHAPE: ", mri_data.shape)
    # Select sample index 2 (adjust as needed)
    mri_data = mri_data[:, :, :, :, sample]  # resulting shape: (146, 146, 32, 40)
    H, W, num_coils, mri_slices = mri_data.shape
    mri_img = np.zeros((H, W, num_coils, mri_slices), dtype=np.float32)
    for coil in range(num_coils):
        for s in range(mri_slices):
            kspace_slice = mri_data[:, :, coil, s]
            mri_img[:, :, coil, s] = process_kspace_to_img(kspace_slice)
    return mri_img

def load_noise_data(mat_path, key='k_gc', sample=0):
    """
    Load pure noise k-space data from a .mat file.
    Expected shape: (146, 146, 32, 40, 4) with dimensions
      (x, y, num_coils, noise_slices, sample).
    Takes the sample at index 2 and processes each 2D slice with IFFT 
    to produce magnitude images, returning an array of shape:
      (146, 146, num_coils, noise_slices)
    """
    data = sio.loadmat(mat_path)
    if key not in data:
        raise KeyError(f"Key '{key}' not found in {mat_path}")
    noise_kspace = data[key]  # shape: (146, 146, 32, 40, 4)
    # Select sample index
    noise_kspace = noise_kspace[:, :, :, :, sample]  # resulting shape: (146,146,32,40)
    H, W, num_coils, noise_slices = noise_kspace.shape
    noise_imgspace = np.zeros((H, W, num_coils, noise_slices), dtype=np.float32)
    for coil in range(num_coils):
        for s in range(noise_slices):
            kspace_slice = noise_kspace[:, :, coil, s]
            noise_imgspace[:, :, coil, s] = process_kspace_to_img(kspace_slice)
    return noise_imgspace

def load_model(model_path, device='cuda'):
    """
    Load the multi-coil Restormer model.
    Now configured for a 2->1 mapping:
      - inp_channels=2 (a single coil image and its noise map)
      - out_channels=1 (denoised target coil image)
    """
    from basicsr.models.archs.restormer_arch import Restormer
    net_g = Restormer(
        inp_channels=2,
        out_channels=1,
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

def main():
    parser = argparse.ArgumentParser(description="Inference for multi-coil MRI denoising (2->1 mapping).")
    parser.add_argument('--model_pth', type=str, required=True, help='Path to the trained model checkpoint (.pth)')
    parser.add_argument('--mri_mat', type=str, required=True, help='Path to the MRI .mat file (noisy data)')
    parser.add_argument('--noise_mat', type=str, required=True, help='Path to the pure noise .mat file')
    parser.add_argument('--output_folder', type=str, default='./results_infer', help='Folder to save outputs')
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--data_scale_factor', type=float, default=5e4, help='Scaling factor for data (if not normalizing)')
    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    # Load MRI data; expected shape after processing: (146,146,32,40)
    mri_img = load_mri_data(args.mri_mat, key='Kimage_first_full', sample=args.sample)
    H, W, num_coils, mri_slices = mri_img.shape
    print("MRI image shape (sample index 2):", mri_img.shape)

    # Save the original MRI data (before scaling) as a 4D NIfTI volume.
    original_vol = np.transpose(mri_img, (0, 1, 3, 2))
    original_nii_path = os.path.join(args.output_folder, "original_4d.nii")
    nib.save(nib.Nifti1Image(original_vol.astype(np.float32), affine=np.eye(4)), original_nii_path)
    print(f"Saved original volume as {original_nii_path}")

    # Load noise data; expected shape after processing: (146,146,32,40)
    noise_imgspace = load_noise_data(args.noise_mat, key='Kimage_first_full', sample=args.sample)
    H_noise, W_noise, _, noise_slices = noise_imgspace.shape
    print("Processed noise image space shape:", noise_imgspace.shape)

    # Replicate noise maps to match MRI spatial dimensions and number of slices.
    noise_maps = replicate_noise_map(noise_imgspace, H, W, mri_slices)
    print("Replicated noise maps shape:", noise_maps.shape)

    # Apply data scaling.
    scale = args.data_scale_factor
    mri_img_scaled = mri_img * scale
    noise_maps_scaled = noise_maps * scale

    # Process each coil for every slice using 2-channel inputs.
    # For each coil, the input is:
    #   Channel 0: The coil's MRI image.
    #   Channel 1: The corresponding noise map.
    # The model outputs the denoised image for that coil.
    denoised_slices = []
    residual_slices = []
    model = load_model(args.model_pth, device='cuda')

    for s in range(mri_slices):
        denoised_coils = []
        residual_coils = []
        for c in range(num_coils):
            # Extract coil c for slice s: shape (H, W)
            coil_img = mri_img_scaled[:, :, c, s]
            coil_noise = noise_maps_scaled[:, :, c, s]
            # Build 2-channel input: shape (2, H, W)
            input_array = np.stack([coil_img, coil_noise], axis=0)
            input_tensor = torch.from_numpy(input_array).unsqueeze(0).float().cuda()  # shape: [1, 2, H, W]

            # Pad if necessary so that H and W are divisible by 8.
            _, _, H_in, W_in = input_tensor.shape
            pad_h = (8 - H_in % 8) % 8
            pad_w = (8 - W_in % 8) % 8
            if pad_h or pad_w:
                input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='reflect')

            with torch.no_grad():
                pred = model(input_tensor)  # Expected shape: [1, 1, H_out, W_out]
            pred = pred[:, :, :H_in, :W_in]  # Crop to original H, W (if padded)
            # Compute residual: difference between predicted and original coil image (channel 0).
            target_tensor = input_tensor[:, 0:1, :H_in, :W_in]
            residual = pred - target_tensor
            denoised_coils.append(pred.cpu().squeeze(0).squeeze(0))  # shape: (H, W)
            residual_coils.append(residual.cpu().squeeze(0).squeeze(0))
        # Stack coils for this slice -> shape: (H, W, num_coils)
        denoised_slice = np.stack(denoised_coils, axis=2)
        residual_slice = np.stack(residual_coils, axis=2)
        denoised_slices.append(denoised_slice)
        residual_slices.append(residual_slice)
        print(f"Processed slice {s+1}/{mri_slices}")

    # Stack all slices -> final shape: (H, W, mri_slices, num_coils)
    denoised_vol = np.stack(denoised_slices, axis=2)
    residual_vol = np.stack(residual_slices, axis=2)

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
