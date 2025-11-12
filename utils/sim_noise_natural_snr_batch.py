import os
import glob
import cv2
import numpy as np
import scipy.io as sio
import nibabel as nib

# ---- CONFIG ----
input_folder   = '/home/lin/Research/denoise/data/test_natural'
coil_sens_path = '/home/lin/Research/denoise/data/csm.mat'
base_output    = 'output_simulated_snr_all'

CROP_SIZE = 146
SNR_LIST  = [10, 15, 20]  # in dB

# ---- Helpers ----
def load_grayscale_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    print(f"Loaded image {os.path.basename(path)} shape:", img.shape)
    return img

def crop_center(img, size):
    h, w = img.shape
    sx, sy = (w - size) // 2, (h - size) // 2
    return img[sy:sy + size, sx:sx + size]

def compute_rms(img):
    return np.sqrt(np.mean(img ** 2))

def snr_to_noise_std(signal_rms, snr_db):
    return signal_rms / (10 ** (snr_db / 20))

# ---- Load coil sensitivity maps once ----
mat   = sio.loadmat(coil_sens_path)
sens  = np.squeeze(mat['sens_gre'])  # (H, W, coils, slices)
_, _, n_coils, n_slices = sens.shape
assert sens.shape[:2] == (CROP_SIZE, CROP_SIZE), "CSM dims must match CROP_SIZE"

# ---- Process all PNGs ----
png_paths = sorted(glob.glob(os.path.join(input_folder, '*.jpg')))
print(f"Found {len(png_paths)} PNG(s) in {input_folder}")

for image_path in png_paths:
    # get base name
    base = os.path.splitext(os.path.basename(image_path))[0]

    # Load + preprocess image
    img_gt = load_grayscale_image(image_path)
    img_gt = np.rot90(img_gt, k=3)
    img_gt = crop_center(img_gt, CROP_SIZE)

    # RMS of ground-truth center crop
    clean_rms_reference = compute_rms(img_gt)
    print(f"{base}: Base image RMS: {clean_rms_reference:.4f}")

    # Simulate for each SNR
    for snr_db in SNR_LIST:
        print(f"\n--- {base}: Generating for SNR = {snr_db} dB ---")

        # set up output folder: base_output/<base>/snrXX/csm
        csm_dir = os.path.join(base_output, base, f'snr{snr_db}', 'csm')
        os.makedirs(csm_dir, exist_ok=True)

        # 1) build clean coil stack
        clean_stack = np.zeros((CROP_SIZE, CROP_SIZE, n_slices, n_coils), dtype=np.float32)
        for s in range(n_slices):
            for c in range(n_coils):
                csm = np.abs(sens[:, :, c, s])
                clean_stack[:, :, s, c] = img_gt * csm

        # 2) coil combine + compute RMS
        clean_cc     = np.sqrt((clean_stack ** 2).sum(axis=-1))
        clean_cc_rms = compute_rms(clean_cc)
        print(f"  Coil-combined RMS: {clean_cc_rms:.4f}")

        # 3) required noise std
        noise_std = snr_to_noise_std(clean_cc_rms, snr_db)
        print(f"  Target noise STD: {noise_std:.4f}")

        # 4) generate noise on coil images
        noise_stack     = np.random.randn(*clean_stack.shape).astype(np.float32) * noise_std
        noise_map_stack = np.full(clean_stack.shape, noise_std, dtype=np.float32)

        # 5) add noise
        noisy_stack = np.clip(clean_stack + noise_stack, 0, 1)

        # 6) coil‐combined outputs
        noisy_cc = np.sqrt((noisy_stack ** 2).sum(axis=-1))
        noise_cc = np.sqrt((noise_stack ** 2).sum(axis=-1))

        # 7) save NIfTIs
        # nib.save(nib.Nifti1Image(clean_stack,     np.eye(4)),
        #          os.path.join(csm_dir, f'{base}_clean_stack.nii'))
        nib.save(nib.Nifti1Image(noisy_stack,     np.eye(4)),
                 os.path.join(csm_dir, f'{base}_noisy_stack.nii'))
        nib.save(nib.Nifti1Image(noise_map_stack, np.eye(4)),
                 os.path.join(csm_dir, f'{base}_noise_map_stack.nii'))
        nib.save(nib.Nifti1Image(clean_cc,        np.eye(4)),
                 os.path.join(csm_dir, f'{base}_clean_cc.nii'))
        nib.save(nib.Nifti1Image(noisy_cc,        np.eye(4)),
                 os.path.join(csm_dir, f'{base}_noisy_cc.nii'))
        # nib.save(nib.Nifti1Image(noise_cc,        np.eye(4)),
        #          os.path.join(csm_dir, f'{base}_noise_map_cc.nii'))

        # 8) verify actual SNR
        actual_noise_cc_rms = compute_rms(noise_cc)
        actual_snr_db       = 20 * np.log10(clean_cc_rms / (actual_noise_cc_rms + 1e-8))
        print(f"  Actual SNR: {actual_snr_db:.2f} dB")
