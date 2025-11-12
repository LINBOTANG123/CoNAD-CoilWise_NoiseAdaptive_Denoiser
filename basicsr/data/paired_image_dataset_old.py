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
import os
import pdb

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

def repeated_gaussian_smoothing(img, ksize=3, sigma=1.0, times=5):
    """
    Repeatedly applies Gaussian blur to a 2D image (float32, [0,1]).
    """
    smoothed = img.copy()
    for _ in range(times):
        smoothed = cv2.GaussianBlur(smoothed, (ksize, ksize), sigma)
    return smoothed


class Dataset_OnlineGaussianDenoising(data.Dataset):
    """
    An on-the-fly denoising dataset that simulates multi-coil MRI data.
    For each GT image, the pipeline is:
      1) Crop the image to the sensitivity map’s spatial dimensions (e.g., 146x146)
         to ensure alignment with the coil sensitivity map.
      2) Optionally, apply a final random crop (data augmentation) if desired.
      3) Apply repeated Gaussian smoothing (and possible inversion) to generate a noise map.
      4) Randomly choose a global noise std.
      5) Randomly select one version (slice) from the coil-sensitivity map,
         which has shape (146,146,32,90), to simulate 32 coils.
      6) For each coil, simulate the coil image by multiplying the GT image
         (or its magnitude) with the sensitivity, then add noise.
      7) Concatenate the 32 noisy coil images with the noise map (extra channel) so that
         the network input has 33 channels.
    """

    def __init__(self, opt):
        """
        opt dict must contain:
          'dataroot_gt' (str): path to GT folder.
          'phase' (str): 'train' or 'val'.
          'in_ch' (int): 1=grayscale, 3=RGB (here we assume grayscale for MRI).
          'gt_size' (int): desired spatial size, recommended to match sensitivity map spatial dims (e.g., 146).
          'geometric_augs' (bool): whether to apply flip/rotate (only for training).
          'noise_std_min' (float): min global noise level.
          'noise_std_max' (float): max global noise level.
          'smooth_times', 'smooth_ksize', 'smooth_sigma': parameters for smoothing.
          'random_invert_prob' (float): probability to invert intensities.
          'coil_sens_path' (str): path to the .mat file containing the coil sensitivity map (key 'sens_gre').
          ... (others as needed)
        """
        super(Dataset_OnlineGaussianDenoising, self).__init__()
        self.opt = opt
        self.phase = opt.get('phase', 'train')
        self.in_ch = opt.get('in_ch', 1)  # assume MRI is grayscale
        self.gt_size = opt.get('gt_size', 146)  # should match sensitivity map's spatial dims
        self.geometric_augs = opt.get('geometric_augs', True) if self.phase == 'train' else False

        # Smoothing parameters
        self.smooth_times = opt.get('smooth_times', 5)
        self.smooth_ksize = opt.get('smooth_ksize', 3)
        self.smooth_sigma = opt.get('smooth_sigma', 1.0)

        # Noise std range
        self.noise_std_min = opt.get('noise_std_min', 0.50)
        self.noise_std_max = opt.get('noise_std_max', 0.70)

        # Probability of inverting intensities
        self.random_invert_prob = opt.get('random_invert_prob', 0.5)

        # File I/O
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']
        self.file_client = None

        # Gather GT image paths
        self.paths = sorted(list(self._scandir(self.gt_folder)))

        # --- Load the coil sensitivity map ---
        # The .mat file is expected to have key 'sens_gre'
        self.coil_sens_path = opt.get('coil_sens_path', None)
        if self.coil_sens_path is None:
            raise ValueError("Please provide the path to the coil sensitivity map in 'coil_sens_path'.")
        mat = sio.loadmat(self.coil_sens_path)
        self.sens_gre = mat['sens_gre']  # expected shape: (146, 146, 32, 90)
        self.sens_gre = np.squeeze(self.sens_gre)
        if self.sens_gre.ndim != 4:
            raise ValueError("The coil sensitivity map should have 4 dimensions (H, W, num_coils, num_versions).")
        H_sens, W_sens, num_coils, num_versions = self.sens_gre.shape
        if H_sens != self.gt_size or W_sens != self.gt_size:
            raise ValueError("Sensitivity map spatial dimensions do not match gt_size.")
        self.num_coils = num_coils
        self.num_versions = num_versions

    def __getitem__(self, index):
        if self.file_client is None:
            from basicsr.utils import FileClient  # or your FileClient module
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        index = index % len(self.paths)
        gt_path = self.paths[index]

        # 1. Load GT image (grayscale)
        img_bytes = self.file_client.get(gt_path, 'gt')
        if self.in_ch == 3:
            raise ValueError("Multi-coil MRI simulation only supports grayscale (in_ch=1).")
        else:
            img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)
            # Expand dims to H x W x 1
            img_gt = np.expand_dims(img_gt, axis=2)

        # 2. First, crop the GT image to the sensitivity map size (e.g., 146x146)
        img_gt = self._crop_to_csm_size(img_gt, self.gt_size)

        # 3. (Optional) Apply further data augmentation if in training mode.
        # This _random_crop can be used if you want a smaller patch than the full sensitivity map.
        if self.phase == 'train':
            # img_gt = self._random_crop(img_gt, self.gt_size)  # or use a different patch size if desired.
            if self.geometric_augs:
                img_gt, = random_augmentation(img_gt,)

        # 4. Normalize GT image to [0,1]
        
        if img_gt.max() > 1.0:
            img_gt = img_gt / 255.0
        else:
            img_gt = np.clip(img_gt, 0, 1.0)

        # 5. Generate the smoothed version and noise map from GT.
        smoothed = self._apply_smoothing_per_channel(img_gt)  # shape: (H, W, 1)
        smoothed = np.squeeze(smoothed, axis=2)  # shape: (H, W)
        sm_min, sm_max = smoothed.min(), smoothed.max()
        if sm_max - sm_min > 1e-8:
            smoothed = (smoothed - sm_min) / (sm_max - sm_min)
        else:
            smoothed = np.clip(smoothed, 0, 1)
        do_invert = (random.random() < self.random_invert_prob)
        if do_invert:
            smoothed = 1.0 - smoothed
        # global_noise_std = random.uniform(self.noise_std_min, self.noise_std_max)
        # noise_map = smoothed * global_noise_std  # shape: (H, W)

        # 6. Squeeze GT image to (H, W)
        img_gt = np.squeeze(img_gt, axis=2)  # shape: (H, W)
        H, W = img_gt.shape

        # 7. Randomly select one version (slice) from the sensitivity map.
        slice_idx = random.randint(0, self.num_versions - 1)
        # Extract slice: shape becomes (H, W, num_coils)
        csm_slice = self.sens_gre[:, :, :, slice_idx]
        # Transpose to get shape (num_coils, H, W)
        csm_slice = np.transpose(csm_slice, (2, 0, 1))

        # 8. Simulate multi-coil data.
        # For each coil, multiply the GT image with the magnitude of the sensitivity.
        coil_clean = np.zeros((self.num_coils, H, W), dtype=np.float32)
        coil_noisy = np.zeros((self.num_coils, H, W), dtype=np.float32)
        noise_map_array = np.zeros((self.num_coils, H, W), dtype=np.float32)
        noise_std_list = []  # to store the noise std used for each coil

        for i in range(self.num_coils):
            # Compute clean coil image
            coil_clean[i] = img_gt * np.abs(csm_slice[i])
            # Sample an independent noise std for this coil
            coil_noise_std = random.uniform(self.noise_std_min, self.noise_std_max)
            noise_std_list.append(coil_noise_std)
            # Compute per-coil noise map using the smoothed image as a base
            noise_map_coil = smoothed * coil_noise_std
            noise_map_array[i] = noise_map_coil
            # Generate independent noise for this coil
            noise = np.random.randn(H, W).astype(np.float32) * noise_map_coil
            coil_noisy[i] = coil_clean[i] + noise
            coil_noisy[i] = np.clip(coil_noisy[i], 0, 1)
            
        # # ----------------------------------------------
        # # Save the first 3 samples in folder './example_imgs'
        # if index < 3:
        #     save_dir = './example_imgs'
        #     os.makedirs(save_dir, exist_ok=True)
            
        #     # Save original GT image
        #     orig_to_save = (img_gt * 255).clip(0, 255).astype(np.uint8)
        #     cv2.imwrite(os.path.join(save_dir, f'original_{index}.png'), orig_to_save)
            
        #     # Save 32 clean coil images (after applying the CSM)
        #     for coil in range(self.num_coils):
        #         clean_to_save = (coil_clean[coil] * 255).clip(0, 255).astype(np.uint8)
        #         cv2.imwrite(os.path.join(save_dir, f'clean_{index}_coil_{coil}.png'), clean_to_save)
            
        #     # Save 32 noisy coil images
        #     for coil in range(self.num_coils):
        #         noisy_to_save = (coil_noisy[coil] * 255).clip(0, 255).astype(np.uint8)
        #         cv2.imwrite(os.path.join(save_dir, f'noisy_{index}_coil_{coil}.png'), noisy_to_save)
            
        #     # Save 32 coil sensitivity maps (magnitude)
        #     for coil in range(self.num_coils):
        #         csm_to_save = (np.abs(csm_slice[coil]) * 255).clip(0, 255).astype(np.uint8)
        #         cv2.imwrite(os.path.join(save_dir, f'csm_{index}_coil_{coil}.png'), csm_to_save)
            
        #     # Save 32 per-coil noise maps
        #     for coil in range(self.num_coils):
        #         noise_map_to_save = (noise_map_array[coil] * 255).clip(0, 255).astype(np.uint8)
        #         cv2.imwrite(os.path.join(save_dir, f'noise_map_{index}_coil_{coil}.png'), noise_map_to_save)
        # # ----------------------------------------------

        # 9. Concatenate the 32 noisy coil images with the 32 per-coil noise maps.
        # This results in an input with (num_coils + num_coils) channels = 64 channels.
        lq = np.concatenate([coil_noisy, noise_map_array], axis=0)  # shape: (2*num_coils, H, W)
        gt = coil_clean  # shape: (num_coils, H, W)

        # 10. Convert to torch tensors.
        lq_tensor = torch.from_numpy(lq).float()
        gt_tensor = torch.from_numpy(gt).float()

        return {
            'lq': lq_tensor,         # Input: 32 noisy coil images + 32 noise map (64 channels)
            'gt': gt_tensor,         # Target: 32 clean coil images
            'lq_path': gt_path,
            'gt_path': gt_path,
            'global_noise_std': noise_std_list,
            'inverted': do_invert,
        }

    def __len__(self):
        return len(self.paths)

    ########################
    # Utility Functions
    ########################

    def _scandir(self, folder):
        """Return full paths to files in the folder."""
        for entry in sorted(os.listdir(folder)):
            full_path = os.path.join(folder, entry)
            if os.path.isfile(full_path):
                yield full_path

    def _crop_to_csm_size(self, img, target_size):
        """
        Crop the image to the target_size (assumed to match the sensitivity map spatial dims).
        If the image is larger than target_size, a random crop is taken.
        If it is smaller, padding is applied.
        """
        h, w, c = img.shape
        if h < target_size or w < target_size:
            pad_h = max(0, target_size - h)
            pad_w = max(0, target_size - w)
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            h, w, _ = img.shape
        # Now, take a random crop of size (target_size, target_size)
        rnd_h = random.randint(0, h - target_size)
        rnd_w = random.randint(0, w - target_size)
        return img[rnd_h:rnd_h+target_size, rnd_w:rnd_w+target_size, :]

    def _random_crop(self, img, patch_size):
        """Further crop a patch of size (patch_size, patch_size) from the image."""
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
