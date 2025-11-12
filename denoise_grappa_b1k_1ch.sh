python inference_final_mag_cplx_grappa.py \
  --model_pth /home/lin/Research/denoise/Restormer/Denoising/pretrained_models/gaussian_gray_denoising_blind.pth \
  --mri_list /home/lin/Downloads/grappa_b2k/paths_b2k.txt \
  --mri_format grappa \
  --mri_key img_coil_single_dir \
  --output_folder ./results_grappa_b2k_1ch_mask \
  --brain_mask /home/lin/Downloads/grappa_b1k/grappa_mask.nii \
  --scale_txt /home/lin/Research/denoise/Restormer/results_grappa_b2k_1ch_mask/S_GLOBAL.txt