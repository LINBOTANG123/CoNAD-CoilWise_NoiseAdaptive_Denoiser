#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
import h5py


def _to_numpy_complex(arr: np.ndarray) -> np.ndarray:
    """
    Handle MATLAB v7.3 complex storage where dtype is a compound with fields 'real'/'imag'.
    Otherwise return the array as-is.
    """
    if np.iscomplexobj(arr):
        return arr
    if hasattr(arr.dtype, "names") and arr.dtype.names and {"real", "imag"} <= set(arr.dtype.names):
        return arr["real"] + 1j * arr["imag"]
    return arr


def load_mat_array(file_path: str, key: str) -> np.ndarray:
    """
    Load `key` from a .mat that could be classic MAT (<=v7) or HDF5 (v7.3).
    Returns a NumPy ndarray.
    """
    if h5py.is_hdf5(file_path):
        with h5py.File(file_path, "r") as f:
            print("HDF5 keys:", list(f.keys()))
            if key not in f:
                raise KeyError(f"Key '{key}' not found in {file_path}")
            data = f[key][()]
    else:
        mat = sio.loadmat(file_path, squeeze_me=False, struct_as_record=False)
        print("MAT keys:", [k for k in mat.keys() if not k.startswith("__")])
        if key not in mat:
            raise KeyError(f"Key '{key}' not found in {file_path}")
        data = mat[key]

    data = _to_numpy_complex(np.asarray(data))
    return data


def main():
    ap = argparse.ArgumentParser(description="Split all DWIs from MAT and save separately as (coils, dwi=1, slices, H, W).")
    ap.add_argument("--in_mat", required=True, help="Path to input .mat")
    ap.add_argument("--key", required=True, help="Variable name inside the .mat to load/save")
    ap.add_argument("--out_dir", default=None, help="Output directory (default: <in>_split/)")
    ap.add_argument("--prefix", default="b1500", help="Prefix for output files (default: b1500)")
    args = ap.parse_args()

    in_path = Path(args.in_mat)
    out_dir = Path(args.out_dir) if args.out_dir else in_path.with_name(in_path.stem + "_split")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load
    arr = load_mat_array(str(in_path), args.key)
    print("Loaded array shape:", arr.shape)

    # Expecting (dwi, slices, coils, H, W)
    if arr.ndim != 5:
        raise ValueError(f"Expected 5D array, got {arr.ndim}D with shape {arr.shape}")

    dwi, slices, coils, H, W = arr.shape
    print(f"Interpreted dims -> DWI:{dwi}, Slices:{slices}, Coils:{coils}, H:{H}, W:{W}")

    # 2) Loop over all DWI directions
    for d in range(dwi):
        one_dwi = arr[d:d+1, ...]  # (1, slices, coils, H, W)
        out_arr = np.transpose(one_dwi, (2, 0, 1, 3, 4))  # (coils, 1, slices, H, W)
        out_name = f"{args.prefix}_dwi{d}.mat"
        out_path = out_dir / out_name
        sio.savemat(out_path, {args.key: out_arr})
        print(f"Saved {out_path} with key '{args.key}' shape {out_arr.shape}")


if __name__ == "__main__":
    main()
