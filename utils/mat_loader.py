#!/usr/bin/env python

import numpy as np
import h5py
import scipy.io as sio


# -------------------- IFFT helpers --------------------

def process_kspace_to_img(kspace_2d: np.ndarray) -> np.ndarray:
    """
    2D k-space -> image magnitude:
      ifftshift -> 2D IFFT -> fftshift -> |.| (float32)
    """
    shifted = np.fft.ifftshift(kspace_2d)
    img_complex = np.fft.ifft2(shifted)
    img_complex = np.fft.fftshift(img_complex)
    return np.abs(img_complex).astype(np.float32)

def process_kspace_to_img_complex(kspace_2d: np.ndarray) -> np.ndarray:
    """
    2D k-space -> image complex:
      ifftshift -> 2D IFFT -> fftshift (complex64)
    """
    shifted = np.fft.ifftshift(kspace_2d)
    img_complex = np.fft.ifft2(shifted)
    img_complex = np.fft.fftshift(img_complex)
    return img_complex.astype(np.complex64)


# -------------------- Loader --------------------

def load_mri_data(
    file_path,
    key='image',
    data_format='Siemens',         # <- RENAMED options: 'Siemens' (was gslider_2_all), 'GE' (was grappa)
    num_samples_to_load=None,      # For Siemens: max # of DWIs to keep
    output_space='magnitude'       # 'magnitude' or 'complex_image'
):
    """
    Load MRI data and return array shaped (H, W, coils, slices, samples).

    Parameters
    ----------
    file_path : str
    key       : str
        Dataset key in .mat/.h5.
    data_format : {'Siemens', 'GE'}
        'Siemens': complex image-domain data (H,W,coils,slices,samples) after transpose.
        'GE'     : k-space data (S,C,H,W) -> IFFT to image domain via helpers.
    num_samples_to_load : int or None
        If provided (Siemens), truncate # of DWIs.
    output_space : {'magnitude', 'complex_image'}
        - 'magnitude'     : return float32 |image|
        - 'complex_image' : return complex64 image

    Returns
    -------
    np.ndarray
        Shape (H, W, coils, slices, samples); dtype depends on output_space.
    """

    # ---- 1) Load container (HDF5 or MAT) ----
    if h5py.is_hdf5(file_path):
        with h5py.File(file_path, 'r') as f:
            if key not in f:
                raise KeyError(f"Key '{key}' not found in {file_path}; keys={list(f.keys())}")
            raw_data = f[key][()]
    else:
        mat = sio.loadmat(file_path)
        if key not in mat:
            raise KeyError(f"Key '{key}' not found in {file_path}; keys={list(mat.keys())}")
        raw_data = mat[key]

    # ---- 2) MATLAB complex struct handling ----
    if raw_data.dtype.kind == 'V':  # void/struct
        fields = raw_data.dtype.names or []
        if set(['real', 'imag']).issubset(fields):
            raw_data = raw_data['real'] + 1j * raw_data['imag']
        else:
            raw_data = raw_data[fields[0]]

    # ---- 3) Format-specific reordering and processing ----
    if data_format == 'Siemens':
        """
        Expected raw_data shape (num_coil, n_dwi, num_slices, H, W)
        (This matches the older 'gslider_2_all' code path.)

        We transpose to (H, W, coils, slices, samples=dwi).
        Siemens is already in IMAGE SPACE (complex), so:
          - 'magnitude'     → return |data|
          - 'complex_image' → return complex image if complex, else raise.
        """
        if raw_data.ndim != 5:
            raise ValueError(f"Siemens expects 5D (C, D, S, H, W); got {raw_data.shape}")

        num_coil, n_dwi, num_slices, H, W = raw_data.shape

        # Optionally truncate # of DWIs
        sel = raw_data[:, :num_samples_to_load, ...] if num_samples_to_load is not None else raw_data

        # (C, D, S, H, W) -> (H, W, C, S, D)
        data = np.transpose(sel, (3, 4, 0, 2, 1))  # (H, W, coils, slices, samples)

        if output_space == 'magnitude':
            # If data is already complex image-domain, just take |.|:
            return np.abs(data).astype(np.float32)
        else:  # 'complex_image'
            if np.iscomplexobj(data):
                return data.astype(np.complex64)
            else:
                # If not complex, treat it as already magnitude float:
                # Promote to complex with zero imaginary part for consistency.
                return data.astype(np.float32).astype(np.complex64)

    elif data_format == 'GE':
        """
        Expected raw_data shape (slices, coils, H, W) in K-SPACE (grappa-like).
        We reorder to (H, W, coils, slices, samples=1) and run IFFT per (H,W) to image.
        """
        if raw_data.ndim != 4:
            raise ValueError(f"GE expects 4D (S, C, H, W) k-space; got {raw_data.shape}")

        S, C, H, W = raw_data.shape
        # (S, C, H, W) -> (H, W, C, S, 1)
        data_k = np.transpose(raw_data, (2, 3, 1, 0))  # (H, W, C, S)
        data_k = data_k[..., np.newaxis]               # add samples axis

        # Now convert k-space -> image domain according to output_space
        if output_space == 'magnitude':
            out = np.empty_like(data_k, dtype=np.float32)
            for c in range(C):
                for s in range(S):
                    for n in range(1):  # samples=1
                        out[:, :, c, s, n] = process_kspace_to_img(data_k[:, :, c, s, n])
            return out
        else:  # 'complex_image'
            out = np.empty_like(data_k, dtype=np.complex64)
            for c in range(C):
                for s in range(S):
                    for n in range(1):
                        out[:, :, c, s, n] = process_kspace_to_img_complex(data_k[:, :, c, s, n])
            return out

    else:
        raise ValueError(f"Unrecognized data_format: {data_format} (use 'Siemens' or 'GE')")
