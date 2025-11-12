# Paths
MODEL_PTH="/home/lin/Research/denoise/Restormer/Denoising/pretrained_models/gaussian_gray_denoising_blind.pth"
MRI_MAT="/home/lin/Research/denoise/data/new_new_human_data/img_b3k.mat"
OUTPUT_DIR="results_b3000_best_22k_all30_1ch_restomer"

# Other params
NUM_SAMPLES=31
MRI_FORMAT="gslider_2_all"

# Run inference
python inference_final_mag_cplx.py \
  --model_pth  "$MODEL_PTH" \
  --mri_mat    "$MRI_MAT" \
  --mri_key "img_coil_all" \
  --domain mag \
  --output_folder "$OUTPUT_DIR" \
  --num_samples  "$NUM_SAMPLES" \
  --mri_format   "$MRI_FORMAT" \
  --brain_mask "/home/lin/Research/denoise/data/new_new_human_data/new_brain_mask_wo_skull.nii"