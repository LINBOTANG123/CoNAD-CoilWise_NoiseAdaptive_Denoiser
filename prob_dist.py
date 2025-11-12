#!/usr/bin/env python
# -------------------------------------------------------------
#  Histogram + metrics: |residual| vs. σ-map  (brain mask only)
#  - Fills holes in mask
#  - (Optional) Repeats noise slices (34 -> 170) if repeat_factor > 1
#  - Keeps PDF bins with density ≥ pdf_floor
#  - Saves masked residual volumes
#  - ALSO saves masked denoised volumes if provided
# -------------------------------------------------------------
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.stats import entropy, wasserstein_distance, ks_2samp
from utils.noise_loader import load_noise_data

# ---------- USER SETTINGS ----------
residual_paths = {
    'Ours'   : '/n/netscratch/zickler_lab/Lab/linbo/denoising_project/Restormer/results_gslider_b1500_scale_no_debais_CW_complex_globalscale_noiseimage/combined_residual_all.nii',
    'MPPCA'  : '/n/netscratch/zickler_lab/Lab/linbo/denoising_project/results/gslider_human_aug_11/mppca/mppca_residual_new.nii',
    'NORDIC' : '/n/netscratch/zickler_lab/Lab/linbo/denoising_project/results/gslider_human_aug_11/nordic/nordic_residual.nii.gz',
}

# (Optional) Provide denoised images to also mask & save. Set to None to skip.
# Keys should match residual_paths labels (but they don't have to—any label is fine).
denoised_paths = {
    'Ours'   : '/n/netscratch/zickler_lab/Lab/linbo/denoising_project/Restormer/results_gslider_b1500_scale_no_debais_CW_complex_globalscale_noiseimage/combined_denoised_all.nii',  # e.g., '/path/to/ours_denoised.nii.gz'
    'MPPCA'  : '/n/netscratch/zickler_lab/Lab/linbo/denoising_project/results/gslider_human_aug_11/mppca/mppca_denoised.nii',
    'NORDIC' : '/n/netscratch/zickler_lab/Lab/linbo/denoising_project/results/gslider_human_aug_11/nordic/nordic_denoised.nii.gz',
    'ORIG' : '/n/netscratch/zickler_lab/Lab/linbo/denoising_project/Restormer/results_gslider_b1500_scale_no_debais_CW_complex_globalscale_noiseimage_onlybrain/original_coilcombined_all.nii'
}

noise_mat_path = '/n/netscratch/zickler_lab/Lab/linbo/denoising_project/dataset/new_new_human_gslider/pf1_noise_v4.mat'
noise_key      = 'image'
noise_format   = 'gslider'
mask_path      = '/n/netscratch/zickler_lab/Lab/linbo/denoising_project/dataset/new_new_human_gslider/Segmentation-Segment_1-label.nii'

nbins          = 200
pdf_floor      = 1e-15
masked_out_dir = './masked_residuals'
repeat_factor  = 1  # set >1 to repeat noise slices along the sample dimension

# -------------------------------------------------------------
def apply_mask_and_save(nifti_path, mask, out_dir, label, suffix='masked'):
    """Apply mask to a NIfTI volume, save float32 output, return saved path."""
    img_obj = nib.load(nifti_path)
    data    = img_obj.get_fdata().squeeze()
    if data.shape != mask.shape:
        raise ValueError(f"{label}: image shape {data.shape} != mask {mask.shape}")
    out_vol = np.zeros_like(data, dtype=np.float32)
    out_vol[mask] = np.asarray(np.abs(data[mask]), dtype=np.float32)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{label.replace(" ", "_")}_{suffix}.nii.gz')
    nib.save(nib.Nifti1Image(out_vol, img_obj.affine), out_path)
    return out_path

def main():
    # 1) LOAD + FIX MASK (fill holes)
    mask_img = nib.load(mask_path)
    mask_raw = mask_img.get_fdata() > 0
    mask     = ndi.binary_fill_holes(mask_raw).astype(bool)

    # 2) LOAD RESIDUALS & SAVE MASKED VOLUMES
    os.makedirs(masked_out_dir, exist_ok=True)
    res_vals = {}

    print('\n=== Residuals (brain mask applied) ===')
    for lab, pth in residual_paths.items():
        img_obj = nib.load(pth)
        data    = img_obj.get_fdata().squeeze()
        if data.shape != mask.shape:
            raise ValueError(f"{lab} shape {data.shape} != mask {mask.shape}")
        vals = np.abs(data)[mask]
        vals = vals[np.isfinite(vals)]
        res_vals[lab] = vals
        print(f'{lab:22s}  mean={vals.mean():.3e}  σ={vals.std():.3e}')

        # Save masked residual volume
        out_path = apply_mask_and_save(pth, mask, masked_out_dir, lab, suffix='residual_masked')
        print(f'  -> saved masked residual: {out_path}')

    # 2b) (OPTIONAL) LOAD DENOISED IMAGES, APPLY MASK, SAVE
    print('\n=== Denoised volumes (mask + save) ===')
    for lab, pth in denoised_paths.items():
        if pth is None or (not os.path.exists(pth)):
            print(f'{lab:22s}  (skip: not provided or not found)')
            continue
        try:
            out_path = apply_mask_and_save(pth, mask, masked_out_dir, lab, suffix='denoised_masked')
            print(f'{lab:22s}  -> saved masked denoised: {out_path}')
        except Exception as e:
            print(f'{lab:22s}  ERROR masking denoised: {e}')

    # 3) LOAD NOISE & (OPTIONALLY) REPEAT SLICES
    #    noise_4d: (H, W, C, 34) complex noise samples
    noise_4d = load_noise_data(
        noise_mat_path, key=noise_key, data_format=noise_format, output_space='complex_image'
    )
    # Magnitude first, then RSS across coils (axis=2)
    noise_mag = np.abs(noise_4d)
    noise_rss = np.sqrt(np.sum(noise_mag**2, axis=2))  # (H, W, 34)

    if repeat_factor and repeat_factor > 1:
        noise_rss = np.tile(noise_rss, reps=(1, 1, repeat_factor))  # (H, W, 34*repeat_factor)

    # Extract masked noise values across all samples
    # noise_rss[mask] -> shape (N_masked, #samples) -> flatten finite entries
    noise_raw = noise_rss[mask]
    noise_raw = noise_raw[np.isfinite(noise_raw)]
    noise_vals = noise_raw.ravel()
    res_vals['Pure Noise'] = noise_vals
    print(f'\n[Pure Noise] mean={noise_vals.mean():.3e}  σ={noise_vals.std():.3e}')

    # 4) COMMON BINS
    all_vals = np.concatenate(list(res_vals.values()))
    bins = np.linspace(0, all_vals.max(), nbins + 1)
    cent = 0.5 * (bins[:-1] + bins[1:])

    # 5) HISTOGRAMS + FLOOR
    hist = {lab: np.histogram(v, bins=bins, density=True)[0] + 1e-12
            for lab, v in res_vals.items()}
    keep = hist['Pure Noise'] >= pdf_floor
    trim_centers = cent[keep]
    for lab in hist:
        hist[lab] = hist[lab][keep]

    # 6) METRICS
    print(f'\n=== Metrics vs. σ-map, bins with PDF ≥ {pdf_floor:g} ===')
    noise_hist = hist['Pure Noise']
    for lab, v in res_vals.items():
        if lab == 'Pure Noise':
            continue
        h    = hist[lab]
        kl   = entropy(h, noise_hist)
        w1   = wasserstein_distance(v, noise_vals)
        ks_p = ks_2samp(v, noise_vals).pvalue
        print(f'{lab:22s}  KL={kl:.4f}  W₁={w1:.3e}  KS-p={ks_p:.3e}')

    # 7) PLOT
    plt.figure(figsize=(12, 8))
    for lab in hist:
        if lab == 'Pure Noise':
            plt.plot(trim_centers, hist[lab], marker='o', ms=2, ls='--', label=lab)
        elif lab == 'NORDIC':
            plt.plot(trim_centers, hist[lab], marker='o', ms=2, ls='-',  label=lab)
        else:
            plt.plot(trim_centers, hist[lab], marker='o', ms=2, ls='-',  label=lab)

    plt.ylim(pdf_floor, None)
    plt.xlabel('Intensity Value')
    plt.ylabel('PDF')
    plt.title('Residual Histogram (brain only)')
    plt.grid(True, linestyle=':', lw=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('residual_vs_noise_pdf_brainonly.png', dpi=300)
    plt.show()
    print('\nSaved → residual_vs_noise_pdf_brainonly.png')

if __name__ == '__main__':
    main()
