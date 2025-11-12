#!/usr/bin/env python
import os
import argparse
import numpy as np
import scipy.io as sio
import nibabel as nib
import torch
import torch.nn.functional as F
import h5py
# Import the MRI loader util.
from utils.mat_loader import load_mri_data

NUM_COILS = 32  # Total number of coils

def process_kspace_to_img(kspace_2d):
    """
    Process a 2D k-space slice for MRI data:
      - Simply return the magnitude of the input.
      (No IFFT is performed for the image input.)
    """
    return np.abs(kspace_2d).astype(np.float32)

def process_noise_kspace_to_img(kspace_2d):
    """
    Process a 2D k-space slice of pure noise:
      - ifftshift, 2D IFFT, fftshift, and magnitude extraction.
    Returns a float32 magnitude image.
    """
    shifted = np.fft.ifftshift(kspace_2d)
    img_complex = np.fft.ifft2(shifted)
    img_complex = np.fft.fftshift(img_complex)
    img_mag = np.abs(img_complex).astype(np.float32)
    return img_mag

def replicate_noise_map_with_sampling(noise_imgspace, target_H, target_W, target_slices):
    """
    Replicate noise maps to match the target spatial and slice dimensions,
    taking into account that the original noise exhibits a column-wise pattern,
    where the middle columns have higher noise. This version embeds the original
    noise map in the center and pads the left/right sides as before, and for 
    the row padding (upper and lower), it samples values in a column-wise manner,
    so that each column's statistics are preserved.
    
    Input:
        noise_imgspace: np.ndarray of shape (H_noise, W_noise, num_coils, noise_slices)
    
    Output:
        expanded_noise: np.ndarray of shape (target_H, target_W, num_coils, target_slices)
    """
    H_noise, W_noise, num_coils, noise_slices = noise_imgspace.shape
    expanded_noise = np.zeros((target_H, target_W, num_coils, target_slices), dtype=np.float32)
    
    num_edge = 10  # Number of rows/columns to use for computing edge statistics

    # Loop over each coil.
    for coil in range(num_coils):
        # Get noise for this coil: shape (H_noise, W_noise, noise_slices)
        coil_noise = noise_imgspace[:, :, coil, :]
        
        # Global statistics for slice replication (if needed)
        global_samples = coil_noise.reshape(-1)
        global_mu = np.mean(global_samples)
        global_sigma = np.std(global_samples)
        
        padded_slices = []
        # Process each slice individually.
        for s in range(noise_slices):
            noise_map = coil_noise[:, :, s]  # shape: (H_noise, W_noise)

            # --- Row Padding (Upper and Lower) with Column-wise Stats ---
            pad_h = target_H - H_noise
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top

            # Pad top rows: for each column, sample based on that column's top edge.
            if pad_top > 0:
                top_pad = np.empty((pad_top, noise_map.shape[1]), dtype=np.float32)
                for j in range(noise_map.shape[1]):
                    n_edge = min(num_edge, noise_map.shape[0])
                    col_top = noise_map[:n_edge, j]
                    mu = np.mean(col_top)
                    sigma = np.std(col_top)
                    top_pad[:, j] = np.random.normal(mu, sigma, size=(pad_top,))
                noise_map = np.vstack([top_pad, noise_map])
            
            # Pad bottom rows: for each column, sample based on that column's bottom edge.
            if pad_bottom > 0:
                bottom_pad = np.empty((pad_bottom, noise_map.shape[1]), dtype=np.float32)
                for j in range(noise_map.shape[1]):
                    n_edge = min(num_edge, noise_map.shape[0])
                    col_bottom = noise_map[-n_edge:, j]
                    mu = np.mean(col_bottom)
                    sigma = np.std(col_bottom)
                    bottom_pad[:, j] = np.random.normal(mu, sigma, size=(pad_bottom,))
                noise_map = np.vstack([noise_map, bottom_pad])
            # Now, noise_map has shape (target_H, W_noise)

            # --- Column Padding (Left and Right) ---
            pad_w = target_W - W_noise
            if pad_w > 0:
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                
                left_edge = noise_map[:, :min(num_edge, noise_map.shape[1])]
                left_mu = np.mean(left_edge, axis=1)  # per row
                left_sigma = np.std(left_edge, axis=1)
                left_pad = np.empty((noise_map.shape[0], pad_left), dtype=np.float32)
                for i in range(noise_map.shape[0]):
                    left_pad[i, :] = np.random.normal(left_mu[i], left_sigma[i], size=(pad_left,))
                
                right_edge = noise_map[:, -min(num_edge, noise_map.shape[1]):]
                right_mu = np.mean(right_edge, axis=1)
                right_sigma = np.std(right_edge, axis=1)
                right_pad = np.empty((noise_map.shape[0], pad_right), dtype=np.float32)
                for i in range(noise_map.shape[0]):
                    right_pad[i, :] = np.random.normal(right_mu[i], right_sigma[i], size=(pad_right,))
                
                new_noise_map = np.hstack([left_pad, noise_map, right_pad])
            elif pad_w < 0:
                crop_start = (W_noise - target_W) // 2
                new_noise_map = noise_map[:, crop_start:crop_start+target_W]
            else:
                new_noise_map = noise_map
            
            new_noise_map = new_noise_map[:target_H, :target_W]
            padded_slices.append(new_noise_map)
        
        padded_slices = np.stack(padded_slices, axis=2)  # shape: (target_H, target_W, noise_slices)
        
        if target_slices <= noise_slices:
            final_maps = padded_slices[:, :, :target_slices]
        else:
            final_maps = np.zeros((target_H, target_W, target_slices), dtype=np.float32)
            final_maps[:, :, :noise_slices] = padded_slices
            remaining = target_slices - noise_slices
            extra_slices = np.random.normal(global_mu, global_sigma, size=(target_H, target_W, remaining))
            final_maps[:, :, noise_slices:] = extra_slices
        
        expanded_noise[:, :, coil, :] = final_maps

    return expanded_noise

def load_noise_data(mat_path, key='image'):
    """
    Load pure noise data from a file.

    If the file is in HDF5 format (e.g. MATLAB v7.3), it is loaded using h5py;
    otherwise, it is loaded using scipy.io.loadmat.

    Expected shape: (H, W, num_coils, noise_slices).
    Each slice is processed using the IFFT-based transformation.
    Returns a 4D array: (H, W, num_coils, noise_slices).
    """
    if h5py.is_hdf5(mat_path):
        with h5py.File(mat_path, 'r') as f:
            if key not in f:
                raise KeyError(f"Key '{key}' not found in {mat_path}")
            noise_data = f[key][()]
    else:
        data = sio.loadmat(mat_path)
        if key not in data:
            raise KeyError(f"Key '{key}' not found in {mat_path}")
        noise_data = data[key]

    if noise_data.dtype.kind == 'V':
        fields = noise_data.dtype.names
        if fields is not None:
            if set(['real', 'imag']).issubset(set(fields)):
                noise_data = noise_data['real'] + 1j * noise_data['imag']
            else:
                noise_data = noise_data[fields[0]]
        else:
            raise ValueError("Loaded noise data has void type without fields.")

    H, W, num_coils, noise_slices = noise_data.shape
    noise_imgspace = np.zeros((H, W, num_coils, noise_slices), dtype=np.float32)
    for coil in range(num_coils):
        for s in range(noise_slices):
            noise_imgspace[:, :, coil, s] = process_noise_kspace_to_img(noise_data[:, :, coil, s])
    return noise_imgspace

def load_model(model_path, device='cuda'):
    """
    Load the multi-coil Restormer model.
    Configured for a 2->1 mapping:
      - inp_channels=2 (coil image and its noise map)
      - out_channels=1 (denoised coil image)
    """
    from basicsr.models.archs.restormer_arch import Restormer
    net_g = Restormer(
        inp_channels=1,
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

def main():
    parser = argparse.ArgumentParser(description="Inference for multi-coil MRI denoising (2->1 mapping).")
    parser.add_argument('--model_pth', type=str, required=True, help='Path to the trained model checkpoint (.pth)')
    parser.add_argument('--mri_mat', type=str, required=True, help='Path to the MRI file (.mat or .h5)')
    parser.add_argument('--output_folder', type=str, default='./results_infer', help='Folder to save outputs')
    parser.add_argument('--data_scale_factor', type=float, default=7e5, help='Scaling factor for data (if not normalizing)')
    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    # Load all MRI samples using the separate utility. 
    # For a file with shape (32, 62, 90, 146, 146) (i.e. (num_coils, num_samples, num_slices, H, W)),
    # we load only the first 10 samples and assume data_format 'A'.
    mri_img = load_mri_data(args.mri_mat, key='image', data_format='b1000', num_samples_to_load=1)
    H, W, num_coils, mri_slices, num_samples = mri_img.shape
    print("MRI image shape:", mri_img.shape)

    # Replicate noise maps to match MRI spatial dimensions and mri_slices.
    # noise_maps shape: (H, W, num_coils, mri_slices)

    # Apply data scaling.
    scale = args.data_scale_factor
    mri_img_scaled = mri_img * scale   # Shape: (H, W, num_coils, mri_slices, num_samples)


    # Prepare lists for concatenated outputs from all samples.
    all_combined_denoised = []
    all_combined_residual = []
    all_original_coil_combined = []

    # Load the model.
    model = load_model(args.model_pth, device='cuda')

    # HARDCODE TO PROCESS FIRST 5 SAMPLES
    for sample in range(1):
        print(f"Processing sample {sample+1}/{num_samples}...")
        # For the original coil-combined input image, use the scaled MRI input for the sample.
        mri_sample = mri_img_scaled[..., sample]  # shape: (H, W, num_coils, mri_slices)
        combined_original = np.sqrt(np.sum(mri_sample**2, axis=2)) / scale  # (H, W, mri_slices)
        all_original_coil_combined.append(combined_original)

        # Now perform denoising on a per-coil basis.
        denoised_slices = []
        residual_slices = []
        for s in range(mri_slices):
            denoised_coils = []
            residual_coils = []
            for c in range(num_coils):
                coil_img = mri_sample[:, :, c, s]
                input_array = np.stack([coil_img], axis=0)  # (1, H, W)
                input_tensor = torch.from_numpy(input_array).unsqueeze(0).float().cuda()  # (1,1,H,W)

                _, _, H_in, W_in = input_tensor.shape
                pad_h = (8 - H_in % 8) % 8
                pad_w = (8 - W_in % 8) % 8
                if pad_h or pad_w:
                    input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='reflect')
                with torch.no_grad():
                    pred = model(input_tensor)  # (1,1,H_out,W_out)
                pred = pred[:, :, :H_in, :W_in]  # Crop back if needed
                target_tensor = input_tensor[:, 0:1, :H_in, :W_in]
                residual = pred - target_tensor
                denoised_coils.append(pred.cpu().squeeze(0).squeeze(0))   # (H, W)
                residual_coils.append(residual.cpu().squeeze(0).squeeze(0))
            denoised_slice = np.stack(denoised_coils, axis=2)   # (H, W, num_coils)
            residual_slice = np.stack(residual_coils, axis=2)   # (H, W, num_coils)
            denoised_slices.append(denoised_slice)
            residual_slices.append(residual_slice)
            print(f"  Processed slice {s+1}/{mri_slices}")
        # Stack slices for this sample (pre coil-combination)
        denoised_vol_sample = np.stack(denoised_slices, axis=2)   # (H, W, mri_slices, num_coils)
        residual_vol_sample = np.stack(residual_slices, axis=2)   # (H, W, mri_slices, num_coils)
        # Coil combine using root-sum-of-squares over coils to produce a 3D volume: (H, W, mri_slices)
        combined_denoised = np.sqrt(np.sum(denoised_vol_sample**2, axis=3))
        combined_residual = np.sqrt(np.sum(residual_vol_sample**2, axis=3))
        
        final_combined_denoised = combined_denoised / scale
        final_combined_residual = combined_residual / scale

        # Append the concatenated coil-combined results to the lists.
        all_combined_denoised.append(final_combined_denoised)
        all_combined_residual.append(final_combined_residual)

    # After processing all samples, concatenate the coil-combined results to a 4D tensor:
    # New tensor shape: (H, W, mri_slices, num_samples)
    combined_deno_all = np.stack(all_combined_denoised, axis=-1)
    combined_res_all = np.stack(all_combined_residual, axis=-1)
    original_combined_all = np.stack(all_original_coil_combined, axis=-1)
    
    # Save the concatenated 4D tensors.
    combined_deno_all_filename = os.path.join(args.output_folder, "combined_denoised_all.nii")
    combined_res_all_filename = os.path.join(args.output_folder, "combined_residual_all.nii")
    original_combined_all_filename = os.path.join(args.output_folder, "original_coilcombined_all.nii")
    
    nib.save(nib.Nifti1Image(combined_deno_all.astype(np.float32), affine=np.eye(4)), combined_deno_all_filename)
    nib.save(nib.Nifti1Image(combined_res_all.astype(np.float32), affine=np.eye(4)), combined_res_all_filename)
    nib.save(nib.Nifti1Image(original_combined_all.astype(np.float32), affine=np.eye(4)), original_combined_all_filename)
    
    print(f"Saved concatenated coil-combined denoised tensor as {combined_deno_all_filename}")
    print(f"Saved concatenated coil-combined residual tensor as {combined_res_all_filename}")
    print(f"Saved concatenated original input coil-combined tensor as {original_combined_all_filename}")

if __name__ == "__main__":
    main()
