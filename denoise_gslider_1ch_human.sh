# Paths
MODEL_PTH="/home/lin/Research/denoise/Restormer/Denoising/pretrained_models/gaussian_gray_denoising_blind.pth"
MRI_MAT="/home/lin/Research/denoise/data/new_human/HR005.mat"
NOISE_MAT="/home/lin/Research/denoise/data/new_human/pf1_noise_v4.mat"
OUTPUT_DIR="results_gslider_human_1ch"

# Other params
NUM_SAMPLES=1
MRI_FORMAT="gslider_2"

# Run inference
python inference_final.py \
  --model_pth  "$MODEL_PTH" \
  --mri_mat    "$MRI_MAT" \
  --mri_key "Img_Super" \
  --output_folder "$OUTPUT_DIR" \
  --num_samples  "$NUM_SAMPLES" \
  --mri_format   "$MRI_FORMAT" \
  --dwi_index 0