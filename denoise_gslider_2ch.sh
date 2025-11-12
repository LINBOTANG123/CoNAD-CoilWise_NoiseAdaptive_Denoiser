# Paths
MODEL_PTH="/home/lin/Research/denoise/Restormer/weights/lessnoise_208000.pth"
MRI_MAT="/home/lin/Research/denoise/data/newdata_gslider/Img_SuperRes.mat"
NOISE_MAT="/home/lin/Research/denoise/data/newdata_gslider/pf1_noise_v4.mat"
OUTPUT_DIR="results_gslider_new_2ch_kimg"

# Other params
NUM_SAMPLES=2
MRI_FORMAT="gslider_2"
NOISE_FORMAT="gslider"

# Run inference
python inference_final.py \
  --model_pth  "$MODEL_PTH" \
  --mri_mat    "$MRI_MAT" \
  --mri_key "Img_Super" \
  --noise_mat  "$NOISE_MAT" \
  --noise_key "kimage" \
  --output_folder "$OUTPUT_DIR" \
  --num_samples  "$NUM_SAMPLES" \
  --mri_format   "$MRI_FORMAT" \
  --use_noise \
  --noise_format "$NOISE_FORMAT" \
  --dwi_index 1