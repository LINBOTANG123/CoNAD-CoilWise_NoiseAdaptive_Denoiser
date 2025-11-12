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

def load_mri_data(mat_path, key='Kimage_first_full', sample=0):
    """
    Load MRI noisy k-space data from a .mat file.
    Expected shape: (146, 146, 32, 40, samples) with dimensions
      (x, y, num_coils, num_slices, samples).
    This function takes the specified sample and converts each
    2D k-space slice to image space, returning data of shape:
      (146, 146, num_coils, num_slices)
    """
    data = sio.loadmat(mat_path)
    if key not in data:
        raise KeyError(f"Key '{key}' not found in {mat_path}")
    mri_data = data[key]  # shape: (146, 146, 32, 40, samples)
    print("ORIGINAL MRI SHAPE:", mri_data.shape)
    # Select the specified sample
    mri_data = mri_data[:, :, :, :, sample]  # resulting shape: (146, 146, 32, 40)
    H, W, num_coils, mri_slices = mri_data.shape
    mri_img = np.zeros((H, W, num_coils, mri_slices), dtype=np.float32)
    for coil in range(num_coils):
        for s in range(mri_slices):
            kspace_slice = mri_data[:, :, coil, s]
            mri_img[:, :, coil, s] = process_kspace_to_img(kspace_slice)
    return mri_img

def load_noise_data(mat_path, key='Kimage_first_full', sample=0):
    """
    Load pure noise k-space data from a .mat file.
    Expected shape: (146, 146, 32, 40, samples) with dimensions
      (x, y, num_coils, noise_slices, samples).
    Takes the specified sample and processes each 2D slice with IFFT 
    to produce magnitude images, returning an array of shape:
      (146, 146, num_coils, noise_slices)
    """
    data = sio.loadmat(mat_path)
    if key not in data:
        raise KeyError(f"Key '{key}' not found in {mat_path}")
    noise_kspace = data[key]  # shape: (146, 146, 32, 40, samples)
    # Select the specified sample
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
    Configured for a 2->1 mapping:
      - inp_channels=2 (a single coil image and its noise map)
      - out_channels=1 (denoised target coil image)
    """
    from basicsr.models.archs.restormer_arch import Restormer
    net_g = Restormer(
        inp_channels=2,
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

def coil_combine(images):
    """
    Combine coil images using a sum-of-squares approach.
    images: numpy array of shape (..., num_coils)
    Returns combined image of shape (...) by computing sqrt(sum(images**2)).
    """
    return np.sqrt(np.sum(np.square(images), axis=-1))

def main():
    parser = argparse.ArgumentParser(description="Multi-sample MRI denoising with coil combination.")
    parser.add_argument('--model_pth', type=str, required=True, help='Path to the trained model checkpoint (.pth)')
    parser.add_argument('--mri_mat', type=str, required=True, help='Path to the MRI .mat file (noisy data)')
    parser.add_argument('--noise_mat', type=str, required=True, help='Path to the pure noise .mat file')
    parser.add_argument('--output_folder', type=str, default='./results_infer', help='Folder to save outputs')
    parser.add_argument('--data_scale_factor', type=float, default=6e4, help='Scaling factor for data (if not normalizing)')
    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    # Determine number of samples from the .mat file
    mri_all = sio.loadmat(args.mri_mat)['Kimage_first_full']
    num_samples = mri_all.shape[4]
    print(f"Found {num_samples} samples.")

    # Lists to store coil-combined outputs for each sample
    original_combined_all = []
    denoised_combined_all = []
    residual_combined_all = []

    model = load_model(args.model_pth, device='cuda')

    for sample_idx in range(num_samples):
        print(f"\n=== Processing sample {sample_idx} ===")
        # Load and process MRI data (shape: H x W x num_coils x mri_slices)
        mri_img = load_mri_data(args.mri_mat, key='Kimage_first_full', sample=sample_idx)
        H, W, num_coils, mri_slices = mri_img.shape
        print("MRI image shape:", mri_img.shape)
        
        # For coil-combination, we first work with individual coil images.
        # Load noise data and replicate noise maps.
        noise_imgspace = load_noise_data(args.noise_mat, key='Kimage_first_full', sample=sample_idx)
        print("Noise image space shape:", noise_imgspace.shape)
        noise_maps = replicate_noise_map(noise_imgspace, H, W, mri_slices)
        print("Replicated noise maps shape:", noise_maps.shape)
        
        scale = args.data_scale_factor
        mri_img_scaled = mri_img * scale
        noise_maps_scaled = noise_maps * scale

        # Prepare lists for denoised and residual coil images (per slice)
        denoised_slices = []
        residual_slices = []
        original_slices = []  # for coil-combined original image

        for s in range(mri_slices):
            # Lists to collect coil outputs for the current slice
            denoised_coils = []
            original_coils = []
            for c in range(num_coils):
                # Original coil image and corresponding noise map (scaled)
                coil_img = mri_img_scaled[:, :, c, s]
                coil_noise = noise_maps_scaled[:, :, c, s]
                input_array = np.stack([coil_img, coil_noise], axis=0)
                input_tensor = torch.from_numpy(input_array).unsqueeze(0).float().cuda()  # [1, 2, H, W]

                # Pad if necessary so that H and W are divisible by 8.
                _, _, H_in, W_in = input_tensor.shape
                pad_h = (8 - H_in % 8) % 8
                pad_w = (8 - W_in % 8) % 8
                if pad_h or pad_w:
                    input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='reflect')

                with torch.no_grad():
                    pred = model(input_tensor)  # [1, 1, H_out, W_out]
                pred = pred[:, :, :H_in, :W_in]  # crop if padded

                # Append model output and original coil image (from scaled data)
                denoised_coils.append(pred.cpu().squeeze(0).squeeze(0))
                original_coils.append(coil_img)
            
            # Stack coil images along last axis; shape becomes (H, W, num_coils)
            denoised_coils = np.stack(denoised_coils, axis=-1)
            original_coils = np.stack(original_coils, axis=-1)
            # Coil combine using sum-of-squares
            denoised_combined = coil_combine(denoised_coils) / scale  # revert scaling
            original_combined = coil_combine(original_coils) / scale
            # Residual is the difference between denoised and original combined images
            residual_combined = original_combined - denoised_combined

            denoised_slices.append(denoised_combined)
            residual_slices.append(residual_combined)
            original_slices.append(original_combined)
            print(f"Processed slice {s+1}/{mri_slices}")

        # Stack slices for this sample (resulting shape: H x W x mri_slices)
        denoised_vol = np.stack(denoised_slices, axis=2)
        residual_vol = np.stack(residual_slices, axis=2)
        original_vol = np.stack(original_slices, axis=2)

        # Append coil-combined volumes for this sample
        denoised_combined_all.append(denoised_vol)
        residual_combined_all.append(residual_vol)
        original_combined_all.append(original_vol)

    # Merge all samples along a new 4th dimension (resulting shape: H x W x mri_slices x num_samples)
    denoised_4d = np.stack(denoised_combined_all, axis=3)
    residual_4d = np.stack(residual_combined_all, axis=3)
    original_4d = np.stack(original_combined_all, axis=3)

    # Save the 4D volumes as NIfTI files.
    original_nii_path = os.path.join(args.output_folder, "original_4d.nii")
    denoised_nii_path = os.path.join(args.output_folder, "denoised_4d.nii")
    residual_nii_path = os.path.join(args.output_folder, "residual_4d.nii")
    
    nib.save(nib.Nifti1Image(original_4d.astype(np.float32), affine=np.eye(4)), original_nii_path)
    nib.save(nib.Nifti1Image(denoised_4d.astype(np.float32), affine=np.eye(4)), denoised_nii_path)
    nib.save(nib.Nifti1Image(residual_4d.astype(np.float32), affine=np.eye(4)), residual_nii_path)

    print("\nAll samples processed, coil combined, and merged into 4D volumes:")
    print(f"Original (coil-combined): {original_nii_path}")
    print(f"Denoised (coil-combined): {denoised_nii_path}")
    print(f"Residual (combined): {residual_nii_path}")

if __name__ == "__main__":
    main()
