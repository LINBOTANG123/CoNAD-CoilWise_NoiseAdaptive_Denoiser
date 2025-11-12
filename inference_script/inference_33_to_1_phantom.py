import os
import argparse
import numpy as np
import scipy.io as sio
import nibabel as nib
import torch
import torch.nn.functional as F

# Number of coil groups remains 32, but we now use a target coil approach.
NUM_COILS = 32

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

def load_mri_data(mat_path, key='k_gc'):
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
    # Select sample index 2 (adjust as needed)
    mri_data = mri_data[:, :, :, :, 2]  # resulting shape: (146, 146, 32, 40)
    H, W, num_coils, mri_slices = mri_data.shape
    mri_img = np.zeros((H, W, num_coils, mri_slices), dtype=np.float32)
    for coil in range(num_coils):
        for s in range(mri_slices):
            kspace_slice = mri_data[:, :, coil, s]
            mri_img[:, :, coil, s] = process_kspace_to_img(kspace_slice)
    return mri_img

def load_noise_data(mat_path, key='k_gc'):
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
    # Select sample index 2
    noise_kspace = noise_kspace[:, :, :, :, 2]  # resulting shape: (146,146,32,40)
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
    Now configured with:
      - inp_channels=33 (target coil image, target coil noise, and 31 remaining coil images)
      - out_channels=1 (denoised target coil image)
      - heads=[1,2,4,8] to match the checkpoint.
    """
    from basicsr.models.archs.restormer_arch import Restormer
    net_g = Restormer(
        inp_channels=33,
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
    parser = argparse.ArgumentParser(description="Inference for multi-coil MRI denoising with 33-channel input and 1-channel output.")
    parser.add_argument('--model_pth', type=str, required=True, help='Path to the trained model checkpoint (.pth)')
    parser.add_argument('--mri_mat', type=str, required=True, help='Path to the MRI .mat file (noisy data)')
    parser.add_argument('--noise_mat', type=str, required=True, help='Path to the pure noise .mat file')
    parser.add_argument('--output_folder', type=str, default='./results_infer', help='Folder to save outputs')
    parser.add_argument('--data_scale_factor', type=float, default=1e4, help='Scaling factor for data (if not normalizing)')
    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    # Load MRI data; expected shape: (146,146,32,40)
    mri_img = load_mri_data(args.mri_mat, key='Kimage_first_full')
    H, W, num_coils, mri_slices = mri_img.shape
    print("MRI image shape (sample index 2):", mri_img.shape)

    # Save the original MRI data (before scaling) as a 4D NIfTI volume.
    # Rearranging from (H, W, num_coils, mri_slices) to (H, W, mri_slices, num_coils)
    original_vol = np.transpose(mri_img, (0, 1, 3, 2))
    original_nii_path = os.path.join(args.output_folder, "original_4d.nii")
    nib.save(nib.Nifti1Image(original_vol.astype(np.float32), affine=np.eye(4)), original_nii_path)
    print(f"Saved original volume as {original_nii_path}")

    # Load noise data; expected shape: (146,146,32,40)
    noise_imgspace = load_noise_data(args.noise_mat, key='Kimage_first_full')
    H_noise, W_noise, _, noise_slices = noise_imgspace.shape
    print("Processed noise image space shape:", noise_imgspace.shape)

    # Replicate noise maps to match MRI spatial dimensions and number of slices.
    noise_maps = replicate_noise_map(noise_imgspace, H, W, mri_slices)
    print("Replicated noise maps shape:", noise_maps.shape)

    # Apply data scaling.
    scale = args.data_scale_factor
    mri_img_scaled = mri_img * scale
    noise_maps_scaled = noise_maps * scale

    # Build model inputs: for each MRI slice, create a tensor of shape [33, H, W]
    # Input channels are built as follows:
    #   1. Target coil (index 0) noisy MRI image.
    #   2. Target coil (index 0) noise map.
    #   3. The remaining 31 noisy MRI coil images (from indices 1 to 31).
    input_slices = []
    for s in range(mri_slices):
        # Extract the slice for all coils.
        slice_mri = mri_img_scaled[:, :, :, s]      # shape: (H, W, 32)
        slice_noise = noise_maps_scaled[:, :, :, s]   # shape: (H, W, 32)
        # Target coil is index 0.
        target_mri = np.expand_dims(slice_mri[:, :, 0], axis=0)    # shape: (1, H, W)
        target_noise = np.expand_dims(slice_noise[:, :, 0], axis=0)  # shape: (1, H, W)
        # Remaining 31 coil images (channels 1 to 31) from MRI.
        remaining_coils = np.transpose(slice_mri[:, :, 1:], (2, 0, 1))  # shape: (31, H, W)
        # Concatenate to form a 33-channel input.
        slice_input = np.concatenate([target_mri, target_noise, remaining_coils], axis=0)  # (33, H, W)
        input_slices.append(torch.from_numpy(slice_input))
    
    # Process slices one-by-one to avoid GPU OOM.
    denoised_list = []
    residual_list = []
    model = load_model(args.model_pth, device='cuda')
    
    for s in range(mri_slices):
        slice_input = input_slices[s].unsqueeze(0).float().cuda()  # shape: [1, 33, H, W]
        # Pad slice so that H and W are divisible by 8.
        _, _, H_in, W_in = slice_input.shape
        pad_h = (8 - H_in % 8) % 8
        pad_w = (8 - W_in % 8) % 8
        if pad_h or pad_w:
            slice_input = F.pad(slice_input, (0, pad_w, 0, pad_h), mode='reflect')
        with torch.no_grad():
            pred = model(slice_input)  # Expected shape: [1, 1, H_out, W_out]
        pred = pred[:, :, :H_in, :W_in]  # Crop to original H and W.
        # Compute the residual using the target coil input (channel 0 of the MRI image).
        target_input = slice_input[:, 0:1, :H_in, :W_in]  # shape: [1, 1, H, W]
        residual = pred - target_input
        denoised_list.append(pred.cpu())
        residual_list.append(residual.cpu())
        print(f"Processed slice {s+1}/{mri_slices}")

    # Stack results along the slice dimension.
    denoised_vol = torch.cat(denoised_list, dim=0).numpy()   # shape: [mri_slices, 1, H, W]
    residual_vol = torch.cat(residual_list, dim=0).numpy()     # shape: [mri_slices, 1, H, W]
    # Rearrange to (H, W, mri_slices, 1)
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
