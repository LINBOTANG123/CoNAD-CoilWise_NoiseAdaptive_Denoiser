# Paths
MODEL_PTH="/home/lin/Research/denoise/Restormer/weights/new_model_net_g_220000.pth"
MRI_MAT="/home/lin/Research/denoise/data/new_human/HR005.mat"
NOISE_MAT="/home/lin/Research/denoise/data/new_human/pf1_noise_v4.mat"
OUTPUT_DIR="results_gslider_mask_norm_1.0"

# Other params
NUM_SAMPLES=1
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
  --dwi_index 0 \
  --mask_nifti /home/lin/Research/denoise/results/gslider_new_human/new_3D_slicer_mask_by_hand.nii \
