# Paths
MODEL_PTH="/home/lin/Research/denoise/Restormer/weights/net_g_220000_best_morenoise.pth"
MRI_MAT="/home/lin/Research/denoise/data/new_new_human_data/img_b3k.mat"
NOISE_MAT="/home/lin/Research/denoise/data/new_new_human_data/pf1_noise_v4.mat"
OUTPUT_DIR="results_b3000_best_22k_all30"

# Other params
NUM_SAMPLES=31
MRI_FORMAT="gslider_2_all"
NOISE_FORMAT="gslider"

# Run inference
python inference_final_mag_cplx.py \
  --model_pth  "$MODEL_PTH" \
  --mri_mat    "$MRI_MAT" \
  --mri_key "img_coil_all" \
  --noise_mat  "$NOISE_MAT" \
  --noise_key "image" \
  --domain mag \
  --output_folder "$OUTPUT_DIR" \
  --num_samples  "$NUM_SAMPLES" \
  --mri_format   "$MRI_FORMAT" \
  --use_noise \
  --noise_format "$NOISE_FORMAT" \
  --brain_mask "/home/lin/Research/denoise/data/new_new_human_data/new_brain_mask_wo_skull.nii"