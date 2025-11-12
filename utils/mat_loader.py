#!/usr/bin/env python

import numpy as np
import h5py
import scipy.io as sio


def process_kspace_to_img(kspace_2d):
    """
    Process a 2D k-space slice:
      - ifftshift, 2D IFFT, fftshift, and magnitude extraction.
    Returns a float32 magnitude image.
    """
    # shift zero-frequency component to center
    shifted = np.fft.ifftshift(kspace_2d)
    # 2D inverse FFT to image domain
    img_complex = np.fft.ifft2(shifted)
    # shift back
    img_complex = np.fft.fftshift(img_complex)
    # magnitude
    img_mag = np.abs(img_complex).astype(np.float32)
    return img_mag

def process_kspace_to_img_complex(kspace_2d):
    """
    Process a 2D k-space slice into a COMPLEX image:
      - ifftshift, 2D IFFT, fftshift
    Returns a complex64 image (no magnitude taken).
    """
    shifted = np.fft.ifftshift(kspace_2d)
    img_complex = np.fft.ifft2(shifted)
    img_complex = np.fft.fftshift(img_complex)
    return img_complex.astype(np.complex64)


def load_mri_data(
    file_path,
    key='image',
    data_format='b1000',
    num_samples_to_load=None,
    gslider_index=0,
    output_space='magnitude'  # NEW: 'magnitude' (default) or 'complex_image'
):
    """
    Load MRI k-space or related data and convert to images.

    Parameters
    ----------
    ...
    output_space : {'magnitude', 'complex_image'}
        'magnitude' (default): return float32 magnitudes as before.
        'complex_image'      : return complex64 image-domain data (real+imag),
                               no magnitude taken.
    """
    if data_format == 'simulate':
        import nibabel as nib
        img = nib.load(file_path)
        arr = img.get_fdata().astype(np.float32)       # shape: (X, Y, Z, coils)
        # reorder to (X, Y, coils, Z)
        data = np.transpose(arr, (0, 1, 3, 2))
        # add singleton samples axis → (X, Y, coils, Z, 1)
        return data[..., np.newaxis]
    
    if h5py.is_hdf5(file_path):
        with h5py.File(file_path, 'r') as f:
            print("HDF5 keys:", list(f.keys()))
            if key not in f:
                raise KeyError(f"Key '{key}' not found in {file_path}")
            raw_data = f[key][()]
    else:
        mat = sio.loadmat(file_path)
        if key not in mat:
            raise KeyError(f"Key '{key}' not found in {file_path}")
        raw_data = mat[key]

    # 2) Handle MATLAB complex void dtype
    if raw_data.dtype.kind == 'V':
        fields = raw_data.dtype.names or []
        if set(['real','imag']).issubset(fields):
            raw_data = raw_data['real'] + 1j * raw_data['imag']
        else:
            raw_data = raw_data[fields[0]]

    # 3) Reorder per format
    if data_format == 'Hwihun_phantom':
        # raw shape: (coils, samples, slices, H, W)
        if num_samples_to_load is not None:
            raw = raw_data[:, :num_samples_to_load, ...]
        # → (H, W, coils, slices, samples)
        data = np.transpose(raw, (3,4,0,2,1))

    elif data_format == 'b1000':
        # raw_data shape: (slices, samples, coils, H, W)
        if num_samples_to_load is not None:
            raw = raw_data[:, :num_samples_to_load, ...]
        else:
            raw = raw_data
        # permute to (H, W, coils, slices, samples)
        data = np.transpose(raw, (3,4,2,0,1))

        # **SPECIAL-CASE**: already in image space → just magnitude & cast
        # data is (H, W, coils, slices, samples)
        return np.abs(data).astype(np.float32)
    elif data_format == 'grappa':
        # Expect raw_data: (slices, coils, H, W) = (S, C, H, W) in **k-space**
        if raw_data.ndim != 4:
            raise ValueError(f"Expected 4D grappa data (S,C,H,W), got {raw_data.shape}")
        S, C, H, W = raw_data.shape

        # Reorder to (H, W, coils, slices)
        data = np.transpose(raw_data, (2, 3, 1, 0))  # (H, W, C, S)

        # Add singleton samples axis → (H, W, C, S, 1) so the existing IFFT loop works
        data = data[..., np.newaxis]
        return data

    elif data_format == 'C':
        data = raw_data
    elif data_format == 'gslider':
        if raw_data.ndim != 6:
            raise ValueError(f"Expected 6D gslider data, got {raw_data.shape}")
        n_dwi, n_gs, mri_slices, num_coils, H, W = raw_data.shape
        if not (0 <= gslider_index < n_gs):
            raise ValueError(f"gslider_index out of range [0,{n_gs}), got {gslider_index}")
        sel = raw_data[:, gslider_index, :, :, :, :]
        if num_samples_to_load is not None:
            sel = sel[:num_samples_to_load, ...]
        data = np.transpose(sel, (3, 4, 2, 1, 0))  # (H, W, coils, slices, dwi)  # (H, W, coils, slices, dwi)
    elif data_format == 'gslider_2':
        if raw_data.ndim != 5:
            raise ValueError(f"Expected 5D gslider data, got {raw_data.shape}")
        print("RAW Gslider shape: ", raw_data.shape)
        num_coil, n_dwi, num_slices, H, W = raw_data.shape
        sel = raw_data[:, :num_samples_to_load, ...] if num_samples_to_load is not None else raw_data
        data = np.transpose(sel, (3, 4, 0, 2, 1))  # (H, W, coils, slices, dwi)
        print("Final loaded gslider_2 shape: ", data.shape)

        if output_space == 'magnitude':
            return np.abs(data).astype(np.float32)
        else:  # 'complex_image'
            if np.iscomplexobj(data):
                print("get complex mri input")
                return data.astype(np.complex64)
<<<<<<< HEAD
    elif data_format == 'gslider_2_all':
        if raw_data.ndim != 5:
            raise ValueError(f"Expected 5D gslider data, got {raw_data.shape}")
        print("RAW Gslider shape: ", raw_data.shape)
        num_coil, n_dwi, num_slices, H, W = raw_data.shape
        data = np.transpose(raw_data, (3, 4, 2, 1, 0))  # (H, W, coils, slices, dwi)
        data = data[:,:,:,:,:] if num_samples_to_load is not None else raw_data
        print("Final selected gslider_2_all shape: ", data.shape)

        if output_space == 'magnitude':
            return np.abs(data).astype(np.float32)
        else:  # 'complex_image'
            if np.iscomplexobj(data):
                print("get complex mri input")
                return data.astype(np.complex64)
=======
>>>>>>> 28fd6abf92a8710f7b390afc3fbff0332a951f35
    else:
        raise ValueError(f"Unrecognized data_format: {data_format}")

    if data.ndim != 5:
        raise ValueError(f"After reorder expected 5D, got {data.shape}")

    H, W, num_coils, mri_slices, num_samples = data.shape

    # 4) Process each 2D k-space with IFFT to magnitude
    if output_space == 'magnitude':
        mri_img = np.empty((H, W, num_coils, mri_slices, num_samples), dtype=np.float32)
        for c in range(num_coils):
            for s in range(mri_slices):
                for n in range(num_samples):
                    mri_img[:, :, c, s, n] = process_kspace_to_img(data[:, :, c, s, n])
        return mri_img
    else:  # 'complex_image'
        mri_img = np.empty((H, W, num_coils, mri_slices, num_samples), dtype=np.complex64)
        for c in range(num_coils):
            for s in range(mri_slices):
                for n in range(num_samples):
                    mri_img[:, :, c, s, n] = process_kspace_to_img_complex(data[:, :, c, s, n])
        return mri_img