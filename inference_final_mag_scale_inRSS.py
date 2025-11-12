#!/usr/bin/env python
import os
import argparse

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import skimage.exposure as ex
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
import pdb
from utils.mat_loader    import load_mri_data
from utils.noise_loader import load_noise_data, replicate_noise_map_with_sampling

NUM_COILS = 32  # adjust if needed

# === PLOT HELPERS (model-scale, masked) =======================================
def _masked_vals(arr, mask=None):
    if mask is None:
        return arr.reshape(-1)
    return arr[mask].reshape(-1)

def plot_hist_overlay(data_a, data_b, mask, title, label_a, label_b, out_png,
                      bins=100, xlim=None, logy=False):
    """
    Overlays normalized histograms of two arrays on model scale.
    data_a/data_b shape can be (...), mask is broadcastable boolean of same spatial dims.
    """
    import matplotlib.pyplot as plt
    va = _masked_vals(data_a, mask)
    vb = _masked_vals(data_b, mask)
    va = va[np.isfinite(va)]
    vb = vb[np.isfinite(vb)]

    plt.figure(figsize=(5,4), dpi=140)
    plt.hist(va, bins=bins, density=True, alpha=0.5, label=label_a)
    plt.hist(vb, bins=bins, density=True, alpha=0.5, label=label_b)
    if xlim is not None:
        plt.xlim(xlim)
    if logy:
        plt.yscale('log')
    plt.title(title)
    plt.xlabel('Intensity (model scale)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def load_model(model_path: str, inp_channels: int, device: str='cuda') -> torch.nn.Module:
    """
    Load the multi-coil Restormer model with dynamic input channels.
    """
    from basicsr.models.archs.restormer_arch import Restormer

    net = Restormer(
        inp_channels=inp_channels,
        out_channels=1,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='BiasFree',
        dual_pixel_task=False
    ).to(device)

    ckpt = torch.load(model_path, map_location=device)
    net.load_state_dict(ckpt['params'], strict=True)
    net.eval()
    return net

def main():
    p = argparse.ArgumentParser(
        description="Inference for multi-coil MRI denoising."
    )
    p.add_argument('--model_pth',   required=True,
                   help='Restormer checkpoint (.pth)')
    p.add_argument('--mri_mat',     required=True,
                   help='Path to the MRI .mat/.h5 or NIfTI')
    p.add_argument('--mri_key',     default='image',
                   help='Dataset key inside .mat/HDF5 (ignored for simulate)')
    p.add_argument('--mri_format',
                   choices=['Hwihun_phantom','b1000','C','gslider', 'gslider_2', 'simulate'],
                   default='b1000',
                   help="MRI data format (use 'simulate' for NIfTI input)")
    p.add_argument('--noise_mat',   required=False,
                   help='Path to noise .mat/.h5 or NIfTI (only if using noise)')
    p.add_argument('--noise_key',   default='k_gc',
                   help='Key inside noise .mat/HDF5 (ignored for simulate)')
    p.add_argument('--noise_format',
                   choices=['Hwihun_phantom','b1000','C','gslider', 'gslider_2','simulate', 'gslider_v5'],
                   default='b1000',
                   help="Noise data format (only if --use_noise)")
    p.add_argument('--use_noise', action='store_true',
                   help="If set, load and feed a noise‐map as second channel")
    p.add_argument('--output_folder', default='./results_infer',
                   help='Where to save NIfTI outputs')
    p.add_argument('--num_samples', type=int, default=2,
                   help='How many DWI channels to process (for MAT formats)')
    p.add_argument('--dwi_index', type=int, default=None,
               help='Specify which DWI sample index to denoise (0-based). If None, denoise all samples.')
    p.add_argument('--mask_nifti', required=False,
                help='Optional brain mask NIfTI (.nii or .nii.gz) to restrict denoising to brain tissue')


    args = p.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    # --- 1) Load MRI (always) ---
    mri_img = load_mri_data(
        args.mri_mat,
        key=args.mri_key,
        data_format=args.mri_format,
        num_samples_to_load=args.num_samples
    )
    H, W, C, S, N = mri_img.shape
    print("Loaded MRI:", mri_img.shape)
    print("MRI min: ", mri_img.min(), "max: ", mri_img.max(), "mean: ", np.mean(mri_img), "std: ", np.std(mri_img))

    # --- 2) Optional brain mask + RSS-based scaling (NO CLIP) --------------------
    if args.mask_nifti:
        mask = nib.load(args.mask_nifti).get_fdata().astype(bool)  # (H,W,S)
        if mask.shape != (H, W, S):
            raise ValueError(f"Mask shape {mask.shape} != {(H,W,S)}")
        # apply mask to per-coil magnitudes
        mri_img *= mask[:, :, None, :, None]

        # RSS magnitude across coils: (H, W, S, N)
        mag_cc = np.sqrt((mri_img**2).sum(axis=2))
        brain_vals = mag_cc[mask[:, :, :, None]]
        p_cap = np.percentile(brain_vals, 99.8) if brain_vals.size else np.percentile(mag_cc, 99.8)
    else:
        mask = None
        # RSS magnitude across coils: (H, W, S, N)
        mag_cc = np.sqrt((mri_img**2).sum(axis=2))
        nz = mag_cc > 0
        p_cap = np.percentile(mag_cc[nz], 99.8) if nz.any() else np.percentile(mag_cc, 99.8)

    scale_factor = 0.99 / (p_cap + 1e-12)
    print(f"Dynamic scale factor (RSS|mag| 99.8%): {scale_factor:.3e}")

    # IMPORTANT: no clipping (preserve dynamic range)
    mri_scaled = mri_img * scale_factor

    scale_factor = 0.99 / (p_cap + 1e-12)
    # scale_factor = 3e6
    # scale_factor = 5e5
    print(f"Dynamic scale factor: {scale_factor:.3e}")

    mri_scaled = np.clip(mri_img * scale_factor, 0, 1)


    # --- 2) Optionally load Noise (2nd channel) ---
    if args.use_noise:
        if not args.noise_mat:
            raise ValueError("`--noise_mat` must be provided when `--use_noise`.")
        noise_img = load_noise_data(
            args.noise_mat,
            key=args.noise_key,
            data_format=args.noise_format
        )
        print("Loaded noise:", noise_img.shape)
        # --- Adjust noise slices if fewer than MRI slices ---
        if args.noise_format == 'gslider' or args.noise_format == 'gslider_v5':
            _, _, _, noise_slices = noise_img.shape
            noise_maps = noise_img
        if args.noise_format == 'b1000':
            noise_maps = replicate_noise_map_with_sampling(noise_img, H, W, S)

        # CHANGE: rotate noise map Clockwise 90 degrees
        noise_maps = np.rot90(noise_maps, k=3, axes=(0, 1)).copy()
        noise_scaled = np.clip(noise_maps * scale_factor, 0, 1).astype(np.float32)

        # collapse to 2D per coil
        print("NOISE MAP min: ", noise_maps.min(), "max: ", noise_maps.max(),  "mean: ", np.mean(noise_maps), "std", np.std(noise_maps))

        # apply mask to noise
        if mask is not None:
            noise_maps *= mask[:, :, None, :]
            
        # --- 2b) Brain-masked raw stats ---------------------------------------------
        if mask is not None:
            # MRI stats
            mask_full = np.broadcast_to(mask[:, :, None, :, None], mri_img.shape)  # (H,W,C,S,N)
            valid_mri = mri_img[mask_full] 
            print(f"[RAW brain]  MRI   min={valid_mri.min():.4g} "
                f"max={valid_mri.max():.4g} "
                f"mean={valid_mri.mean():.4g} "
                f"std={valid_mri.std():.4g} "
                f"p10={np.percentile(valid_mri,10):.4g} "
                f"p90={np.percentile(valid_mri,90):.4g}")

            if args.use_noise:
                mask_noise = np.broadcast_to(mask[:, :, None, :], noise_maps.shape)    # (H,W,C,S)
                valid_noise = noise_maps[mask_noise]
                print(f"[RAW brain]  Noise min={valid_noise.min():.4g} "
                    f"max={valid_noise.max():.4g} "
                    f"mean={valid_noise.mean():.4g} "
                    f"std={valid_noise.std():.4g} "
                    f"p10={np.percentile(valid_noise,10):.4g} "
                    f"p90={np.percentile(valid_noise,90):.4g}")

        # CHANGE to NOISE STD
        # noise_norm = noise_maps.mean(axis=-1)  # (H, W, coils)

        S = noise_maps.shape[-1]
        ddof = 1 if S > 1 else 0  # 1 = unbiased (n-1), 0 = biased (n)
        noise_norm = noise_maps.std(axis=-1, ddof=ddof).astype(np.float32)  # (H, W, coils)

        # Smooth per-coil std map to match training smoothness
        from scipy.ndimage import gaussian_filter
        SMOOTH_SIGMA = 1.0  # tweak 0.8–1.5 if desired
        for c in range(noise_norm.shape[-1]):
            noise_norm[:, :, c] = gaussian_filter(noise_norm[:, :, c],
                                                  sigma=SMOOTH_SIGMA, mode='nearest')

        # Scale to model’s input units and (optionally) clip
        noise_norm = noise_norm * scale_factor
        noise_norm = np.clip(noise_norm, 0, 1)
        # ------------------------------------------------------------

        # save for inspection (optional)
        nib.save(nib.Nifti1Image((noise_norm/scale_factor).astype(np.float32), np.eye(4)),
                os.path.join(args.output_folder, 'noise_map_scaled.nii'))
        nib.save(nib.Nifti1Image(noise_maps.astype(np.float32), np.eye(4)),
                os.path.join(args.output_folder, 'noise_map_raw.nii'))


    # --- 3) Load model with dynamic input channels ---
    inp_ch = 2 if args.use_noise else 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.model_pth, inp_channels=inp_ch, device=device)

    # --- 4) Denoising Loop ---
    all_deno  = []
    all_res   = []
    all_orig  = []

    sample_indices = [args.dwi_index] if args.dwi_index is not None else list(range(N))

    for sid in sample_indices:
        print(f"Sample {sid+1}/{S}")
        vol = mri_scaled[..., sid]         # (H, W, coils, slices)
        orig = np.sqrt((vol**2).sum(axis=2))
        all_orig.append(orig)

        deno_slices = []
        res_slices  = []

        for z in range(S):
            print(f"  Denoising slice {z+1}/{S} for sample {sid+1}/{N}")
            deno_coils = []
            res_coils  = []
            for c in range(NUM_COILS):
                img_c = vol[:, :, c, z]
                if args.use_noise:
                    noise_c = noise_norm[:, :, c]
                    arr = np.stack([img_c, noise_c], axis=0)
                else:
                    arr = img_c[np.newaxis, ...]  # single‐channel

                inp = torch.tensor(arr,
                                   dtype=torch.float32,
                                   device=device).unsqueeze(0)  # (1, C, H, W)

                # pad to multiples of 8
                _, _, h, w = inp.shape
                ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
                if ph or pw:
                    inp = F.pad(inp, (0, pw, 0, ph), mode='reflect')

                with torch.no_grad():
                    out = model(inp)
                out    = out[..., :h, :w]

                target = inp[:, 0:1, :h, :w]
                res    = out - target

                deno_coils.append(out.cpu().squeeze().numpy())
                res_coils.append(res.cpu().squeeze().numpy())

            deno_slices.append(np.stack(deno_coils, axis=2))
            res_slices.append(np.stack(res_coils, axis=2))

        deno_vol = np.stack(deno_slices, axis=2)
        res_vol  = np.stack(res_slices, axis=2)
        # Save per-coil original and denoised results (H, W, Coils, Slices)
        # --- Save per-coil volumes (brain-masked, correct axis order) -------------
        # Put both as (H, W, S, C) before saving/masking
        vol_4d   = np.transpose(vol,      (0, 1, 3, 2))   # (H,W,S,C)
        deno_4d  = deno_vol   # (H,W,S,C)

        if mask is not None:
            brain_mask_4d = mask[..., None]               # (H,W,S,1)
            vol_denorm   = (vol_4d  / scale_factor) * brain_mask_4d
            deno_denorm  = (deno_4d / scale_factor) * brain_mask_4d
        else:
            vol_denorm   =  vol_4d  / scale_factor
            deno_denorm  =  deno_4d / scale_factor

        orig_path = os.path.join(args.output_folder, f"original_percoil_sample{sid+1}.nii")
        deno_path = os.path.join(args.output_folder, f"denoised_percoil_sample{sid+1}.nii")

        nib.save(nib.Nifti1Image(vol_denorm.astype(np.float32),  np.eye(4)), orig_path)
        nib.save(nib.Nifti1Image(deno_denorm.astype(np.float32), np.eye(4)), deno_path)

        # all_deno.append(np.sqrt((deno_vol**2).sum(axis=3)))
        # all_res.append(np.sqrt((res_vol**2).sum(axis=3)))

        # Coil-combine (RSS) over COILS = axis=2
        deno_cc = np.sqrt((deno_vol**2).sum(axis=3))   # (H, W, S)
        orig_cc = orig                                 # already sqrt((vol**2).sum(axis=2))

        # Signed residual at coil-combined level (still in SCALED units)
        residual_cc = orig_cc - deno_cc
        # --- PLOTTING: coil-level and coil-combined overlays (model scale) ------------
        # Build masks for plotting (broadcasted)
        mask_3d = mask.astype(bool) if mask is not None else None          # (H,W,S)

        # 1) COIL-LEVEL overlay: residual vs pure noise (aggregate across slices & coils)
        #    res_vol: (H, W, C, S)  [model scale]
        #    noise_scaled: (H, W, C, S)  [model scale]
        if args.use_noise and noise_scaled is not None:
            # Match shapes: both are (H,W,C,S). Mask needs to be (H,W,S) -> broadcast.
            plot_hist_overlay(
                data_a=res_vol,                      # residual per coil
                data_b=noise_scaled,                 # pure noise per coil
                mask=mask_3d[..., None],             # broadcast to (H,W,S,1) then to (H,W,C,S)
                title=f'Coil-level: Residual vs Noise (sample {sid})',
                label_a='Residual (coil)',
                label_b='Noise (coil)',
                out_png=os.path.join(args.output_folder, f'overlay_coil_res_vs_noise_sample{sid+1}.png'),
                bins=120, xlim=None, logy=True
            )

        # 2) COIL-COMBINED overlay: residual vs pure noise (RSS over coils)
        #    residual_cc: (H, W, S)  [model scale]
        #    Build noise coil-combined: RSS across coils at each slice
        if args.use_noise and noise_scaled is not None:
            noise_cc = np.sqrt((noise_scaled**2).sum(axis=2))               # (H,W,S)
            plot_hist_overlay(
                data_a=residual_cc,
                data_b=noise_cc,
                mask=mask_3d,
                title=f'Coil-combined: Residual vs Noise (sample {sid})',
                label_a='Residual (coil-combined)',
                label_b='Noise (coil-combined)',
                out_png=os.path.join(args.output_folder, f'overlay_cc_res_vs_noise_sample{sid+1}.png'),
                bins=120, xlim=None, logy=True
            )


        all_deno.append(deno_cc)
        all_res.append(residual_cc)

    if mask is not None:
        brain_mask_3d = mask.astype(np.float32)       # (H,W,S)
        for i in range(len(all_orig)):
            all_orig[i] *= brain_mask_3d
            all_deno[i] *= brain_mask_3d
            all_res[i]  *= brain_mask_3d
    # --- 5) Assemble & Save ---
    deno4d = np.stack(all_deno, axis=-1)
    res4d  = np.stack(all_res, axis=-1)
    orig4d = np.stack(all_orig, axis=-1)

    deno4d_denorm = deno4d / scale_factor
    res4d_denorm  = res4d / scale_factor
    orig4d_denorm = orig4d / scale_factor

    nib.save(nib.Nifti1Image(deno4d_denorm.astype(np.float32), np.eye(4)), 
            os.path.join(args.output_folder, 'combined_denoised_all.nii'))
    nib.save(nib.Nifti1Image(res4d_denorm.astype(np.float32),  np.eye(4)), 
            os.path.join(args.output_folder, 'combined_residual_all.nii'))
    nib.save(nib.Nifti1Image(orig4d_denorm.astype(np.float32), np.eye(4)), 
            os.path.join(args.output_folder, 'original_coilcombined_all.nii'))

    print("All outputs saved in", args.output_folder)


if __name__ == "__main__":
    main()
