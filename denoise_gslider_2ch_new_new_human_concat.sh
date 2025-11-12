# Paths
MODEL_PTH="/n/netscratch/zickler_lab/Lab/linbo/denoising_project/Restormer/experiments/train_scratch_csm_newattn_concat/models/net_g_220000.pth"
MRI_MAT="/n/netscratch/zickler_lab/Lab/linbo/denoising_project/dataset/new_new_human_gslider/b1500_dwi5.mat"
NOISE_MAT="/n/netscratch/zickler_lab/Lab/linbo/denoising_project/dataset/new_new_human_gslider/pf1_noise_v4.mat"
OUTPUT_DIR="results_gslider_b1500_concat_CW_220k"

# Other params
NUM_SAMPLES=1
MRI_FORMAT="gslider_2"
NOISE_FORMAT="gslider"

# Run inference
python inference_final.py \
  --model_pth  "$MODEL_PTH" \
  --mri_mat    "$MRI_MAT" \
  --mri_key "img_coil_all" \
  --noise_mat  "$NOISE_MAT" \
  --noise_key "kimage" \
  --output_folder "$OUTPUT_DIR" \
  --num_samples  "$NUM_SAMPLES" \
  --mri_format   "$MRI_FORMAT" \
  --use_noise \
  --noise_format "$NOISE_FORMAT" \
  --dwi_index 0