# Paths
MODEL_PTH="/home/lin/Research/denoise/Restormer/weights/lessnoise_208000.pth"
MRI_MAT="/home/lin/Research/denoise/data/b1000/b1000_noisy.mat"
NOISE_NII="/home/lin/Research/denoise/data/pulseq_ep2d_noise_kspace.mat"
OUTPUT_DIR="results_b1000"

# Other params
NUM_SAMPLES=1
MRI_FORMAT="b1000"
NOISE_FORMAT="b1000"

# Run inference
python inference_final.py \
  --model_pth  "$MODEL_PTH" \
  --mri_mat    "$MRI_MAT" \
  --mri_key "k_gc" \
  --noise_mat  "$NOISE_NII" \
  --output_folder "$OUTPUT_DIR" \
  --num_samples  "$NUM_SAMPLES" \
  --mri_format   "$MRI_FORMAT" \
  --use_noise \
  --noise_format "$NOISE_FORMAT"