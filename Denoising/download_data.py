#!/usr/bin/env python3
"""
Restormer denoising data downloader (SIDD/DND + Gaussian sets) without external CLIs.
Requires: pip install gdown
"""

import os
import argparse
import shutil
import zipfile
from pathlib import Path

try:
    import gdown
except ImportError:
    raise SystemExit("Missing dependency: gdown. Install with `pip install gdown` in your env.")

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='train, test or train-test')
parser.add_argument('--dataset', type=str, default='SIDD', help='all or SIDD or DND (for --data test, --noise real)')
parser.add_argument('--noise', type=str, required=True, help='real or gaussian')
args = parser.parse_args()

# -------- Google Drive file IDs --------
SIDD_train = '1UHjWZzLPGweA9ZczmV8lFSRcIxqiOVJw'
SIDD_val   = '1Fw6Ey1R-nCHN9WEpxv0MnMqxij-ECQYJ'
SIDD_test  = '11vfqV-lqousZTuAit1Qkqghiv_taY0KZ'
DND_test   = '1CYCDhaVxYYcXhSfEVDUwkvJDtGxeQ10G'

BSD400     = '1idKFDkAHJGAFDn1OyXZxsTbOSBx9GS8N'
DIV2K      = '13wLWWXvFkuYYVZMMAYiMVdSA7iVEf2fM'
Flickr2K   = '1J8xjFCrVzeYccD-LF08H7HiIsmi8l2Wn'
WaterlooED = '19_mCE_GXfmE5yYsm-HEzuZQqmwMjPpJr'
gaussian_test = '1mwMLt-niNqcQpfN_ZduG9j4k6P_ZkOl0'

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def download(id_str: str, out_path: Path):
    """Download a file by Google Drive ID to out_path using gdown; returns out_path."""
    print(f"Downloading to {out_path} (id={id_str}) ...")
    ensure_dir(out_path.parent)
    # gdown returns the output path or None
    res = gdown.download(id=id_str, output=str(out_path), quiet=False)
    if res is None or not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"Download failed or empty file: {out_path}")
    return out_path

def extract_zip(zip_path: Path, extract_dir: Path):
    """Extract zip_path into extract_dir after validation."""
    if not zipfile.is_zipfile(zip_path):
        # Sometimes Drive returns an HTML file for quota/auth issues
        head = zip_path.read_bytes()[:200].decode(errors="ignore")
        raise RuntimeError(f"{zip_path} is not a valid zip. First bytes: {head[:120]!r}")
    print(f"Extracting {zip_path} -> {extract_dir}")
    ensure_dir(extract_dir)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(str(extract_dir))

def download_and_extract(id_str: str, out_zip: Path, extract_dir: Path, rename_from: str = None, rename_to: Path = None, keep_zip=False):
    zip_path = download(id_str, out_zip)
    extract_zip(zip_path, extract_dir)
    if rename_from and rename_to:
        src = extract_dir / rename_from
        dst = rename_to
        if dst.exists():
            shutil.rmtree(dst)
        src.rename(dst)
    if not keep_zip:
        zip_path.unlink(missing_ok=True)

def handle_real_train():
    print('SIDD Training Data!')
    dldir = Path('Datasets/Downloads')
    ensure_dir(dldir)
    # Train split
    download_and_extract(
        SIDD_train,
        out_zip=dldir / 'train.zip',
        extract_dir=dldir,
        rename_from='train',  # expected folder name inside the zip
        rename_to=dldir / 'SIDD'
    )
    # Val split
    print('SIDD Validation Data!')
    download_and_extract(
        SIDD_val,
        out_zip=Path('Datasets/val.zip'),
        extract_dir=Path('Datasets')
    )

def handle_real_test(dataset_choice: str):
    if dataset_choice in ('all', 'SIDD'):
        print('SIDD Testing Data!')
        download_and_extract(
            SIDD_test,
            out_zip=Path('Datasets/test.zip'),
            extract_dir=Path('Datasets')
        )
    if dataset_choice in ('all', 'DND'):
        print('DND Testing Data!')
        download_and_extract(
            DND_test,
            out_zip=Path('Datasets/test.zip'),   # same name reused, zip contains different top folder
            extract_dir=Path('Datasets')
        )

def handle_gaussian_train():
    dldir = Path('Datasets/Downloads')
    ensure_dir(dldir)

    print('WaterlooED Training Data!')
    download_and_extract(
        WaterlooED,
        out_zip=dldir / 'WaterlooED.zip',
        extract_dir=dldir
    )

    print('DIV2K Training Data!')
    download_and_extract(
        DIV2K,
        out_zip=dldir / 'DIV2K.zip',
        extract_dir=dldir
    )

    print('BSD400 Training Data!')
    download_and_extract(
        BSD400,
        out_zip=dldir / 'BSD400.zip',
        extract_dir=dldir
    )

    print('Flickr2K Training Data!')
    download_and_extract(
        Flickr2K,
        out_zip=dldir / 'Flickr2K.zip',
        extract_dir=dldir
    )

def handle_gaussian_test():
    print('Gaussian Denoising Testing Data!')
    download_and_extract(
        gaussian_test,
        out_zip=Path('Datasets/test.zip'),
        extract_dir=Path('Datasets')
    )

def main():
    data_parts = args.data.split('-')
    if args.noise == 'real':
        if 'train' in data_parts:
            handle_real_train()
        if 'test' in data_parts:
            handle_real_test(args.dataset)
    elif args.noise == 'gaussian':
        if 'train' in data_parts:
            handle_gaussian_train()
        if 'test' in data_parts:
            handle_gaussian_test()
    else:
        raise ValueError("--noise must be 'real' or 'gaussian'")
    print('Download completed successfully!')

if __name__ == "__main__":
    main()
