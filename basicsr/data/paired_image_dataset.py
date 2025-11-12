from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_DP_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP

import random
import numpy as np
import torch
import cv2
from basicsr.utils.misc import scandir
import matplotlib.pyplot as plt
import scipy.io as sio
from random import randrange, uniform
import os
import pdb
import scipy.io as sio
import os
import time

class Dataset_PairedImage(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedImage, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)
            
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)

import os
import re
import cv2
import random
import numpy as np
import torch
import torch.nn.functional as F
import scipy.io as sio

# You need to have this function defined (or import it from your module)
def repeated_gaussian_smoothing(img, ksize=3, sigma=1.0, times=5):
    """
    Repeatedly applies Gaussian blur to a 2D image (float32, [0,1]).
    """
    smoothed = img.copy()
    for _ in range(times):
        smoothed = cv2.GaussianBlur(smoothed, (ksize, ksize), sigma)
    return smoothed

# Dummy functions for imfrombytes and random_augmentation.
# Replace these with your actual implementations.
def imfrombytes(img_bytes, flag='grayscale', float32=True):
    import cv2
    import numpy as np
    # Decode image from bytes (assuming grayscale)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if float32:
        img = img.astype(np.float32) / 255.0
    return img

def random_augmentation(img):
    # For simplicity, just return the image.
    return (img,)

class Dataset_OnlineGaussianDenoising(torch.utils.data.Dataset):
    """
    Online Gaussian denoising dataset for a single-coil MRI simulation.
    
    For each GT image sample, the pipeline is:
      1) Crop the image to the sensitivity map’s spatial dimensions (e.g., 146x146).
      2) Optionally apply data augmentation.
      3) Use a coil sensitivity map to simulate a clean coil image (target coil, here index 0).
      4) Generate a noise map via repeated Gaussian smoothing (with optional inversion) on the clean image.
      5) Add spatially varying noise and a small whole-image noise to obtain a noisy coil image.
      6) Form the input as 2 channels:
             - Channel 0: Noisy coil image.
             - Channel 1: Corresponding noise map.
         The target is the clean coil image.
    """
    def __init__(self, opt):
        super(Dataset_OnlineGaussianDenoising, self).__init__()
        self.opt = opt
        self.phase = opt.get('phase', 'train')
        self.in_ch = opt.get('in_ch', 1)  # assumed grayscale
        self.gt_size = opt.get('gt_size', 146)
        self.geometric_augs = opt.get('geometric_augs', True) if self.phase == 'train' else False

        # --- NEW: Option A scaling into MRI band ---
        # Alpha rescales your [0,1] natural images into the MRI numeric band (you said "remember Option A").
        # Set, e.g., alpha=0.2 to roughly match MRI spread.
        self.alpha = float(opt.get('alpha', 1.0))  # 1.0 keeps old behavior

        # --- NEW: Coil-noise matching controls ---
        # Path to CSV with columns: coil,mu_R,sig_R,mu_I,sig_I
        self.mri_stats_csv = opt.get('mri_stats_csv', None)
        # Choose which sigma to match:
        #   'real_only' -> use sig_R[k]
        #   'effective_complex' -> use sqrt((sig_R[k]^2 + sig_I[k]^2)/2)
        self.sigma_mode = opt.get('sigma_mode', 'real_only')
        # If True, use the SAME coil index for CSM and for noise statistics; else sample another
        self.use_same_coil_for_noise = bool(opt.get('use_same_coil_for_noise', True))

        # Whether to clip after adding noise (MRI complex data is signed; default False)
        self.clip_after_noise = bool(opt.get('clip_after_noise', False))
        # If clipping, clip to this symmetric band
        self.clip_band = float(opt.get('clip_band', 1.0))  # clip to [-clip_band, +clip_band]

        # Smoothing parameters.
        self.smooth_times = opt.get('smooth_times', 5)
        self.smooth_ksize = opt.get('smooth_ksize', 3)
        self.smooth_sigma = opt.get('smooth_sigma', 1.0)

        # Noise standard deviation range. (We will normalize to match coil sigma; these shape the spatial map.)
        self.noise_std_min = opt.get('noise_std_min', 0.50)
        self.noise_std_max = opt.get('noise_std_max', 0.70)

        # Whole-image noise constant (part of the spatial std map before normalization).
        self.whole_noise_std = opt.get('whole_noise_std', 0.01)

        # Probability to invert intensities for building spatial map.
        self.random_invert_prob = opt.get('random_invert_prob', 0.5)

        # Multi-coil simulation parameters.
        self.num_coils = opt.get('num_coils', 32)
        self.use_csm = opt.get('use_csm', True)  # Only using CSM in this pipeline.
        print("USE CSM: ", self.use_csm)

        # File I/O.
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']
        self.file_client = None

        # ---- Stats logging (log once per run) ----
        # Where to write the one-time stats log (configurable via opt)
        self.stats_log_path = opt.get(
            'stats_log_path',
            os.path.join(os.path.dirname(self.gt_folder), 'scaling_noise_stats.log')
        )
        # If the file already exists, assume someone has logged it; else we will log once.
        self._stats_logged = os.path.exists(self.stats_log_path)

        # Gather GT image paths.
        self.paths = sorted(list(self._scandir(self.gt_folder)))

        # Load the coil sensitivity map.
        self.coil_sens_path = opt.get('coil_sens_path', None)
        if self.coil_sens_path is None:
            raise ValueError("Please provide the path to the coil sensitivity map in 'coil_sens_path'.")
        mat = sio.loadmat(self.coil_sens_path)
        self.sens_gre = mat['sens_gre']  # expected shape: (gt_size, gt_size, num_coils, num_versions)
        self.sens_gre = np.squeeze(self.sens_gre)
        if self.sens_gre.ndim != 4:
            raise ValueError("The coil sensitivity map should have 4 dimensions (H, W, num_coils, num_versions).")
        H_sens, W_sens, num_coils, num_versions = self.sens_gre.shape
        if H_sens != self.gt_size or W_sens != self.gt_size:
            raise ValueError("Sensitivity map spatial dimensions do not match gt_size.")
        self.num_coils = num_coils
        self.num_versions = num_versions

        # --- NEW: Load pure-noise coil statistics (sigma_real/sigma_imag) ---
        if self.mri_stats_csv is not None:
            data = np.genfromtxt(self.mri_stats_csv, delimiter=',', names=True)

            if data.dtype.names is None:
                raise ValueError(
                    f"mri_stats_csv '{self.mri_stats_csv}' must include a header with named columns "
                    "(expected: coil,sigma_real,sigma_imag)."
                )

            # Normalize column names (handle minor variations)
            names = {n.lower(): n for n in data.dtype.names}
            required = ['coil', 'sigma_real', 'sigma_imag']
            for col in required:
                if col not in names:
                    raise ValueError(
                        f"CSV missing required column '{col}'. Found columns: {list(data.dtype.names)}"
                    )

            # Extract arrays
            coil_idx   = np.asarray(data[names['coil']], dtype=np.float32)
            sigma_real = np.asarray(data[names['sigma_real']], dtype=np.float32)
            sigma_imag = np.asarray(data[names['sigma_imag']], dtype=np.float32)

            # Truncate / align to CSM coil count if needed
            if sigma_real.size != self.num_coils or sigma_imag.size != self.num_coils:
                print(f"[WARN] CSV coil count ({sigma_real.size}) != CSM coil count ({self.num_coils}); "
                    f"proceeding with min length.")
                minC = int(min(sigma_real.size, sigma_imag.size, self.num_coils))
                coil_idx   = coil_idx[:minC]
                sigma_real = sigma_real[:minC]
                sigma_imag = sigma_imag[:minC]
                self.num_coils = minC  # keep consistency with CSM usage

            # Store as "R/I" for downstream code compatibility
            self.sig_R = sigma_real.astype(np.float32)
            self.sig_I = sigma_imag.astype(np.float32)

            # Choose which sigma to match during noise injection
            if self.sigma_mode == 'effective_complex':
                # RMS combine real/imag (good if approximating complex with a single real channel)
                self.sig_eff = np.sqrt((self.sig_R**2 + self.sig_I**2) / 2.0).astype(np.float32)
            elif self.sigma_mode == 'real_only':
                self.sig_eff = self.sig_R.astype(np.float32)
            else:
                raise ValueError(f"Unsupported sigma_mode '{self.sigma_mode}'. "
                                f"Use 'real_only' or 'effective_complex'.")
        else:
            print("[INFO] mri_stats_csv not provided; coil sigma matching will be skipped (keeps old behavior).")
            self.sig_R = self.sig_I = self.sig_eff = None



    def __getitem__(self, index):
        if self.file_client is None:
            from basicsr.utils import FileClient  # or your FileClient module
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # --- Load GT image ---
        gt_path = self.paths[index % len(self.paths)]
        img_bytes = self.file_client.get(gt_path, 'gt')
        if self.in_ch == 3:
            raise ValueError("Multi-coil MRI simulation only supports grayscale (in_ch=1).")
        else:
            img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)
            img_gt = np.expand_dims(img_gt, axis=2)  # shape: (H, W, 1)
        img_gt = self._crop_to_csm_size(img_gt, self.gt_size)
        if self.phase == 'train' and self.geometric_augs:
            img_gt, = random_augmentation(img_gt)
        # Normalize to [0,1]
        if img_gt.max() > 1.0:
            img_gt = img_gt / 255.0
        else:
            img_gt = np.clip(img_gt, 0, 1.0)
        img_gt = np.squeeze(img_gt, axis=2)  # shape: (H, W)

        # --- REMOVE/REPLACE intensity offset; we want consistent scaling into MRI band ---
        # *** Option A scaling: bring natural images into MRI numeric band ***
        if self.alpha != 1.0:
            img_gt = img_gt * self.alpha
        # NOTE: do not clip back to [0,1] after scaling; we want signed/narrow MRI-like range later.
        H, W = img_gt.shape

        # --- Generate clean coil image for target coil using the CSM ---
        # Use a random version from the sensitivity map.
        if self.use_csm:
            # pick a sensitivity‐map version
            slice_idx = random.randint(0, self.num_versions - 1)
            csm_slice = self.sens_gre[:, :, :, slice_idx]    # H×W×coils
            csm_slice = np.transpose(csm_slice, (2,0,1))    # coils×H×W
            target_coil = random.randint(0, self.num_coils - 1)
            clean_coil   = img_gt * np.abs(csm_slice[target_coil])
        else:
            clean_coil = img_gt

        # --- Simulate noise for the target coil ---
        coil_smoothed = repeated_gaussian_smoothing(
            clean_coil,
            ksize=self.smooth_ksize,
            sigma=self.smooth_sigma,
            times=self.smooth_times
        )
        coil_smoothed = np.clip(coil_smoothed, 0, None)  # nonnegative map for std shaping
        if random.random() < self.random_invert_prob:
            # keep nonnegative by inverting around its max (or 1.0 if you prefer)
            # Here we invert with max=coil_smoothed.max() to avoid negatives:
            inv_base = coil_smoothed.max() if coil_smoothed.size > 0 else 1.0
            coil_smoothed = np.clip(inv_base - coil_smoothed, 0, None)

        # Spatial heteroscedasticity before normalization
        noise_std = random.uniform(self.noise_std_min, self.noise_std_max)
        spatial_map = coil_smoothed * noise_std  # shape (H,W), >=0

        # Whole-volume (global) std baseline
        global_std = float(self.whole_noise_std)

        # Base std map before normalization
        std_map_base = np.sqrt(spatial_map**2 + global_std**2).astype(np.float32)  # (H,W)

        # --- NEW: Normalize std map RMS to coil sigma ---
        if self.sig_eff is not None:
            if self.use_same_coil_for_noise:
                k_noise = target_coil
            else:
                k_noise = random.randint(0, self.num_coils - 1)
            sigma_coil = float(self.sig_eff[k_noise])  # scalar target sigma for this sample

            # RMS of base map
            rms_base = float(np.sqrt(np.mean(std_map_base**2)) + 1e-12)
            scale_std = sigma_coil / rms_base
            std_map = (std_map_base * scale_std).astype(np.float32)
        else:
            # No CSV provided: keep your old behavior
            std_map = std_map_base

        # --- Draw actual noise realization (zero-mean) ---
        smoothing_noise = np.random.randn(H, W).astype(np.float32) * std_map
        # If you want to keep a tiny independent whole_noise term, it's already embedded in std_map_base; no need to add twice.
        true_noise = smoothing_noise  # (H,W), signed

        # Noisy input (NO clipping by default)
        pre = clean_coil + true_noise
        if self.clip_after_noise:
            low, high = -self.clip_band, self.clip_band
            noisy_coil = np.clip(pre, low, high).astype(np.float32)
        else:
            noisy_coil = pre.astype(np.float32)

        # ---- One-time logging of scaled intensity + scaled noise stats ----
        if not self._stats_logged:
            try:
                os.makedirs(os.path.dirname(self.stats_log_path), exist_ok=True)

                # Clean (scaled) intensity stats (after alpha + CSM)
                intensity_mean = float(np.mean(clean_coil))
                intensity_std  = float(np.std(clean_coil))

                # Scaled noise stats (the actual injected noise)
                noise_mean = float(np.mean(true_noise))
                noise_std  = float(np.std(true_noise))

                # Optional context for reproducibility
                ts = time.strftime('%Y-%m-%d %H:%M:%S')
                coil_info = f"coil_idx={int(target_coil)}"
                if self.sig_eff is not None:
                    # If you used a separate noise coil index, include it; else reuse target_coil.
                    try:
                        coil_info += f", coil_idx_noise={int(k_noise)}, sigma_coil={float(sigma_coil):.6g}"
                    except NameError:
                        coil_info += f", coil_idx_noise={int(target_coil)}, sigma_coil={float(self.sig_eff[target_coil]):.6g}"

                with open(self.stats_log_path, 'a') as f:
                    f.write(
                        "[One-time stats]\n"
                        f"timestamp: {ts}\n"
                        f"{coil_info}\n"
                        f"alpha: {float(self.alpha):.6g}\n"
                        f"clean_intensity_mean: {intensity_mean:.6g}\n"
                        f"clean_intensity_std:  {intensity_std:.6g}\n"
                        f"noise_mean:           {noise_mean:.6g}\n"
                        f"noise_std:            {noise_std:.6g}\n"
                        "----\n"
                    )

                # Mark as logged so we won’t log again
                self._stats_logged = True
            except Exception as e:
                # Non-fatal: continue training even if logging fails
                print(f"[WARN] Failed to write one-time stats log at '{self.stats_log_path}': {e}")


        # --- Form Input and GT ---
        lq = np.stack([noisy_coil, std_map], axis=0)  # (2,H,W): [noisy, std_map]
        gt_final = clean_coil

        # --- Convert to torch tensors ---
        lq_tensor = torch.from_numpy(lq).float()
        gt_tensor = torch.from_numpy(gt_final).float().unsqueeze(0)

        ret = {
            'lq': lq_tensor,                    # (2,H,W): [noisy, std_map actually used]
            'gt': gt_tensor,                    # (1,H,W): clean coil (Option A scaled, CSM applied)
            'lq_path': gt_path,
            'gt_path': gt_path,
            'std_map': torch.from_numpy(std_map).float().unsqueeze(0),   # (1,H,W)
            'true_noise': torch.from_numpy(true_noise).float().unsqueeze(0),  # (1,H,W), signed
            'coil_idx': int(target_coil),
            'alpha': float(self.alpha),
            'clip_after_noise': bool(self.clip_after_noise),
        }

        # Include noise coil & sigma when available
        if self.sig_eff is not None:
            ret['sigma_mode'] = self.sigma_mode
            if self.use_same_coil_for_noise:
                ret['coil_idx_noise'] = int(target_coil)
                ret['sigma_coil'] = float(self.sig_eff[target_coil])
            else:
                ret['coil_idx_noise'] = int(k_noise)
                ret['sigma_coil'] = float(self.sig_eff[k_noise])

        return ret

    
    def __len__(self):
        return len(self.paths)

    # ----------------- Utility Functions -----------------
    def _scandir(self, folder):
        """Return full paths to files in the folder."""
        for entry in sorted(os.listdir(folder)):
            full_path = os.path.join(folder, entry)
            if os.path.isfile(full_path):
                yield full_path

    def _crop_to_csm_size(self, img, target_size):
        """
        Crop the image to the target_size.
        If the image is larger than target_size, a random crop is taken.
        If it is smaller, padding is applied.
        """
        h, w, c = img.shape
        if h < target_size or w < target_size:
            pad_h = max(0, target_size - h)
            pad_w = max(0, target_size - w)
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            h, w, _ = img.shape
        rnd_h = random.randint(0, h - target_size)
        rnd_w = random.randint(0, w - target_size)
        return img[rnd_h:rnd_h+target_size, rnd_w:rnd_w+target_size, :]

    def _random_crop(self, img, patch_size):
        """Crop a patch of size (patch_size, patch_size) from the image."""
        h, w, c = img.shape
        if h == patch_size and w == patch_size:
            return img
        rnd_h = random.randint(0, h - patch_size)
        rnd_w = random.randint(0, w - patch_size)
        return img[rnd_h:rnd_h+patch_size, rnd_w:rnd_w+patch_size, :]

    def _apply_smoothing_per_channel(self, img):
        """
        Apply repeated Gaussian smoothing to each channel of the image.
        """
        h, w, c = img.shape
        out = np.zeros_like(img)
        for ch in range(c):
            out[..., ch] = repeated_gaussian_smoothing(
                img[..., ch],
                ksize=self.smooth_ksize,
                sigma=self.smooth_sigma,
                times=self.smooth_times
            )
        return out

class Dataset_GaussianDenoising(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_GaussianDenoising, self).__init__()
        self.opt = opt

        if self.opt['phase'] == 'train':
            self.sigma_type  = opt['sigma_type']
            self.sigma_range = opt['sigma_range']
            assert self.sigma_type in ['constant', 'random', 'choice']
        else:
            self.sigma_test = opt['sigma_test']
        self.in_ch = opt['in_ch']

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None        

        self.gt_folder = opt['dataroot_gt']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            self.paths = paths_from_lmdb(self.gt_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [
                    osp.join(self.gt_folder,
                             line.split(' ')[0]) for line in fin
                ]
        else:
            self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        # gt_path = self.paths[index]['gt_path']
        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path, 'gt')

        if self.in_ch == 3:
            try:
                img_gt = imfrombytes(img_bytes, float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        else:
            try:
                img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = np.expand_dims(img_gt, axis=2)
        img_lq = img_gt.copy()


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            # flip, rotation
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                                        bgr2rgb=False,
                                        float32=True)


            if self.sigma_type == 'constant':
                sigma_value = self.sigma_range
            elif self.sigma_type == 'random':
                sigma_value = random.uniform(self.sigma_range[0], self.sigma_range[1])
            elif self.sigma_type == 'choice':
                sigma_value = random.choice(self.sigma_range)

            noise_level = torch.FloatTensor([sigma_value])/255.0
            # noise_level_map = torch.ones((1, img_lq.size(1), img_lq.size(2))).mul_(noise_level).float()
            noise_level_map = noise_level.expand(1, img_lq.size(1), img_lq.size(2))  # Shape: (1, H, W)
            noise = torch.randn_like(img_lq).mul_(noise_level).float()
            # noise = torch.randn(img_lq.size()).mul_(noise_level).float()
            img_lq.add_(noise)

        else:            
            np.random.seed(seed=0)
            img_lq += np.random.normal(0, self.sigma_test/255.0, img_lq.shape)
            # noise_level_map = torch.ones((1, img_lq.shape[0], img_lq.shape[1])).mul_(self.sigma_test/255.0).float()

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                            bgr2rgb=False,
                            float32=True)

            noise_level = torch.FloatTensor([self.sigma_test]) / 255.0
            noise_level_map = noise_level.expand(1, img_lq.size(1), img_lq.size(2))  # Shape: (1, H, W)

        # Concatenate the noise level map with the LQ image
        # TODO: channel enable here
        img_lq = torch.cat([img_lq, noise_level_map], dim=0)  # Shape: (2, H, W) for grayscale, (4, H, W) for RGB
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': gt_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)

class Dataset_DefocusDeblur_DualPixel_16bit(data.Dataset):
    def __init__(self, opt):
        super(Dataset_DefocusDeblur_DualPixel_16bit, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lqL_folder, self.lqR_folder = opt['dataroot_gt'], opt['dataroot_lqL'], opt['dataroot_lqR']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.paths = paired_DP_paths_from_folder(
            [self.lqL_folder, self.lqR_folder, self.gt_folder], ['lqL', 'lqR', 'gt'],
            self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lqL_path = self.paths[index]['lqL_path']
        img_bytes = self.file_client.get(lqL_path, 'lqL')
        try:
            img_lqL = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("lqL path {} not working".format(lqL_path))

        lqR_path = self.paths[index]['lqR_path']
        img_bytes = self.file_client.get(lqR_path, 'lqR')
        try:
            img_lqR = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("lqR path {} not working".format(lqR_path))


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_lqL, img_lqR, img_gt = padding_DP(img_lqL, img_lqR, img_gt, gt_size)

            # random crop
            img_lqL, img_lqR, img_gt = paired_random_crop_DP(img_lqL, img_lqR, img_gt, gt_size, scale, gt_path)
            
            # flip, rotation            
            if self.geometric_augs:
                img_lqL, img_lqR, img_gt = random_augmentation(img_lqL, img_lqR, img_gt)
        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lqL, img_lqR, img_gt = img2tensor([img_lqL, img_lqR, img_gt],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lqL, self.mean, self.std, inplace=True)
            normalize(img_lqR, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        img_lq = torch.cat([img_lqL, img_lqR], 0)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lqL_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)
