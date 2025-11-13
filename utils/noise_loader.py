import numpy as np
import scipy.io as sio
import h5py
import nibabel as nib

def process_noise_kspace_to_img(kspace_2d):
    """
    IFFT-based conversion of a 2D k-space noise slice into a magnitude image.
    Steps:
      1) ifftshift
      2) 2D IFFT
      3) fftshift
      4) magnitude extraction
    Returns:
        np.float32 magnitude image of same shape as input.
    """
    shifted = np.fft.ifftshift(kspace_2d)
    img_complex = np.fft.ifft2(shifted)
    img_complex = np.fft.fftshift(img_complex)
    return np.abs(img_complex).astype(np.float32)


def process_noise_kspace_to_img_complex(kspace_2d):
    """
    IFFT of a 2D k-space NOISE slice to COMPLEX image space.
    Returns complex64; no magnitude taken.
    """
    shifted = np.fft.ifftshift(kspace_2d)
    img_complex = np.fft.ifft2(shifted)
    img_complex = np.fft.fftshift(img_complex)
    return img_complex.astype(np.complex64)


def load_noise_data(mat_path, key='k_gc', data_format='b1000', output_space='magnitude'):
    """
    Load pure-noise k-space from a .mat or HDF5 file and convert to image-space.

    Supports multiple formats:
    - 'Hwihun_phantom', 'b1000', 'C':
        Expect a dataset with shape (H, W, num_coils, noise_slices) under key 'k_gc'.
    - 'gslider':
        Expect a dataset with shape (num_coils, noise_slices, H, W) under key 'image'.

    Parameters
    ----------
    ...
    output_space : {'magnitude', 'complex_image'}
        'magnitude' (default): previous behavior (float32 magnitude images).
        'complex_image'      : return complex64 image-space noise (no magnitude).
    """
    if data_format == 'simulate':
        img = nib.load(mat_path)
        arr = img.get_fdata().astype(np.float32)    # shape (X,Y,Z,coils)
        # reorder to (X, Y, coils, Z)
        return np.transpose(arr, (0, 1, 3, 2))

    if h5py.is_hdf5(mat_path):
        with h5py.File(mat_path, 'r') as f:
            print("HDF5 keys:", list(f.keys()))
            if key not in f:
                raise KeyError(f"Key '{key}' not found in {mat_path}")
            raw = f[key][()]
    else:
        mat = sio.loadmat(mat_path)
        if key not in mat:
            raise KeyError(f"Key '{key}' not found in {mat_path}")
        raw = mat[key]

    # 2) Handle MATLAB complex void dtype
    if raw.dtype.kind == 'V':
        fields = raw.dtype.names or []
        if set(['real', 'imag']).issubset(fields):
            raw = raw['real'] + 1j * raw['imag']
        else:
            raw = raw[fields[0]]

    # 3) Reorder per format
    if data_format in ['Hwihun_phantom','b1000']:
        # raw might be 4D (H,W,coils,slices) or
        #         5D (slices, samples, coils, H, W)
        if raw.ndim == 4:
            # already (H, W, coils, slices)
            data = raw
        elif raw.ndim == 5:
            S, N, C, H, W = raw.shape
            if N != 1:
                raise ValueError(f"Expected single-sample noise, got N={N}")
            # drop the samples axis, then permute
            tmp = raw[:,0,...]               # (slices, coils, H, W)
            data = np.transpose(tmp, (2, 3, 1, 0))  # → (H, W, coils, slices)
        else:
            raise ValueError(f"Unexpected noise shape {raw.shape} for format {data_format}")
    elif data_format == 'gslider':
        # Expect raw shape: (num_coils, noise_slices, H, W)
        print("Raw noise shape: ", raw.shape)
        if raw.ndim != 4:
            raise ValueError(f"Expected 4D noise for 'gslider', got {raw.shape}")
        num_coils, noise_slices, H, W = raw.shape
        # Transpose to (H, W, coils, slices)
        data = np.transpose(raw, (2, 3, 0, 1))
    elif data_format == 'gslider_v5':
        # Expect raw shape: (num_coils, noise_slices, H, W)
        print("Raw noise shape: ", raw.shape)
        if raw.ndim != 4:
            raise ValueError(f"Expected 4D noise for 'gslider', got {raw.shape}")
        num_coils, noise_slices, H, W = raw.shape
        # Transpose to (H, W, coils, slices)
        data = np.transpose(raw, (2, 3, 1, 0))
        return data
    elif data_format == 'grappa':
        # Expect raw shape: (slices, coils, H, W) = (S, C, H, W)
        if raw.ndim != 4:
            raise ValueError(f"Expected 4D grappa noise (S,C,H,W), got {raw.shape}")
        S, C, H, W = raw.shape
        # Transpose to (H, W, coils, slices)
        data = np.transpose(raw, (2, 3, 0, 1))  # -> (H, W, S, C)

    else:
        raise ValueError(f"Unrecognized data_format: {data_format}")

    # 4) Validate output dims
    if data.ndim != 4:
        raise ValueError(f"After reorder, expected 4D array, got {data.shape}")
    H, W, num_coils, noise_slices = data.shape

    if output_space == 'magnitude':
        noise_imgspace = np.empty((H, W, num_coils, noise_slices), dtype=np.float32)
        for c in range(num_coils):
            for s in range(noise_slices):
                # If data is already image-domain, this will still work if you want magnitude;
                # if it's k-space, this does the IFFT to magnitude as before.
                noise_imgspace[:, :, c, s] = process_noise_kspace_to_img(data[:, :, c, s])
        return noise_imgspace
    elif output_space == 'complex_kimage':  # 'complex_image'
        # If data is k-space -> IFFT to complex. If already image-domain complex,
        # bypass IFFT by detecting and returning directly.
        if np.iscomplexobj(data):
            print("WRONG KIMAGE")
            return data.astype(np.complex64)
        print("CORRECT KIMAGE")
        noise_imgspace = np.empty((H, W, num_coils, noise_slices), dtype=np.complex64)
        for c in range(num_coils):
            for s in range(noise_slices):
                noise_imgspace[:, :, c, s] = process_noise_kspace_to_img_complex(data[:, :, c, s])
        return noise_imgspace
    elif output_space == 'complex_image':
        if np.iscomplexobj(data):
            return data.astype(np.complex64)


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

