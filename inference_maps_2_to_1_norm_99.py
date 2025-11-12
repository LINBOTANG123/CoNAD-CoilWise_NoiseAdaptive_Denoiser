import os
import argparse

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

from utils.mat_loader    import load_mri_data
from utils.noise_loader import load_noise_data, replicate_noise_map_with_sampling

NUM_COILS = 32  # adjust if needed

def load_model(model_path: str, device: str = 'cuda') -> torch.nn.Module:
    """
    Load the multi-coil Restormer model (2→1 mapping).
    """
    from basicsr.models.archs.restormer_arch import Restormer

    net = Restormer(
        inp_channels=2,           # coil image + noise map
        out_channels=1,           # denoised coil image
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
        description="Inference for multi-coil MRI denoising (2→1 mapping)."
    )
    p.add_argument('--model_pth',   required=True,
                   help='Restormer checkpoint (.pth)')
    p.add_argument('--mri_mat',     required=True,
                   help='.mat or .h5 with raw MRI k-space')
    p.add_argument('--noise_mat',   required=True,
                   help='.mat or .h5 with pure-noise k-space')
    p.add_argument('--output_folder', default='./results_infer',
                   help='Where to save NIfTI outputs')
    p.add_argument('--num_samples', type=int, default=1,
                   help='How many DWI channels to process')
    args = p.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    # 1) Load MRI data
    mri_img = load_mri_data(
        args.mri_mat,
        key='image',
        data_format='simulate',
        num_samples_to_load=args.num_samples
    )
    H, W, _, mri_slices, num_samples = mri_img.shape
    print("Loaded MRI:", mri_img.shape)

    # 2) Compute 1st and 99th percentiles for normalization
    p_low, p_high = np.percentile(mri_img, [1, 99])
    print(f"MRI 1st percentile: {p_low:.3e}, 99th percentile: {p_high:.3e}")

    # 3) Normalize MRI data to [0,1] using percentiles
    eps = 1e-12
    mri_norm = (mri_img - p_low) / (p_high - p_low + eps)
    mri_norm = np.clip(mri_norm, 0, 1)

    # 4) Load & expand noise maps
    noise_img = load_noise_data(
        args.noise_mat,
        key='k_gc',
        data_format='simulate'
    )
    print("Loaded noise:", noise_img.shape)

    noise_maps = replicate_noise_map_with_sampling(
        noise_img, H, W, mri_slices
    )
    noise_map2D = noise_maps.mean(axis=-1)  # (H, W, coils)

    # 5) Normalize noise using same percentiles
    noise_norm = (noise_map2D - p_low) / (p_high - p_low + eps)
    noise_norm = np.clip(noise_norm, 0, 1)

    # Save per-coil noise map
    nib.save(
        nib.Nifti1Image(noise_norm.astype(np.float32), np.eye(4)),
        os.path.join(args.output_folder, 'noise_map.nii')
    )

    # 6) Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.model_pth, device=device)

    # 7) Denoise loop
    all_deno  = []
    all_res   = []
    all_orig  = []

    for sid in range(num_samples):
        print(f"Sample {sid+1}/{num_samples}")
        vol = mri_norm[..., sid]         # (H, W, coils, slices)
        orig = np.sqrt((vol**2).sum(axis=2))
        all_orig.append(orig)

        deno_slices = []
        res_slices  = []

        for z in range(mri_slices):
            deno_coils = []
            res_coils  = []
            for c in range(NUM_COILS):
                img_c = vol[:, :, c, z]
                noise_c = noise_norm[:, :, c]
                inp = torch.from_numpy(
                    np.stack([img_c, noise_c], axis=0)
                ).unsqueeze(0).to(device, dtype=torch.float32)

                # pad to multiples of 8
                _, _, h, w = inp.shape
                ph, pw = (8 - h%8)%8, (8 - w%8)%8
                if ph or pw:
                    inp = F.pad(inp, (0,pw,0,ph), mode='reflect')

                with torch.no_grad():
                    out = model(inp)
                # crop to original height/width
                out    = out[..., :h, :w]
                target = inp[:, 0:1, :h, :w]
                res    = out - target

                deno_coils.append(out.cpu().squeeze().numpy())
                res_coils.append(res.cpu().squeeze().numpy())

            deno_slices.append(np.stack(deno_coils, axis=2))
            res_slices.append(np.stack(res_coils, axis=2))

        deno_vol = np.stack(deno_slices, axis=2)
        res_vol  = np.stack(res_slices, axis=2)

        all_deno.append(np.sqrt((deno_vol**2).sum(axis=3)))
        all_res.append(np.sqrt((res_vol**2).sum(axis=3)))

    # 8) Assemble & un-normalize
    deno4d = np.stack(all_deno, axis=-1)  # (H, W, slices, samples)
    res4d  = np.stack(all_res, axis=-1)
    orig4d = np.stack(all_orig, axis=-1)


    nib.save(
        nib.Nifti1Image(deno4d.astype(np.float32), np.eye(4)),
        os.path.join(args.output_folder, 'combined_denoised_all.nii')
    )
    nib.save(
        nib.Nifti1Image(res4d.astype(np.float32), np.eye(4)),
        os.path.join(args.output_folder, 'combined_residual_all.nii')
    )
    nib.save(
        nib.Nifti1Image(orig4d.astype(np.float32), np.eye(4)),
        os.path.join(args.output_folder, 'original_coilcombined_all.nii')
    )

    print("All outputs saved in", args.output_folder)

if __name__ == "__main__":
    main()
