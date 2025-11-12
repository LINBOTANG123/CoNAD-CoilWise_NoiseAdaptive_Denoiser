# Paths
MODEL_PTH="/home/lin/Research/denoise/Restormer/Denoising/pretrained_models/gaussian_gray_denoising_sigma50.pth"
MRI_NII="/home/lin/Research/denoise/util/output_simulated_snr_data1/snr20/csm/0802x2_noisy_stack.nii"
NOISE_NII="/home/lin/Research/denoise/util/output_simulated_snr_data1/snr20/csm/0802x2_noise_map_stack.nii"
OUTPUT_DIR="results_simulate_snr20_1ch"

# Other params
NUM_SAMPLES=1
MRI_FORMAT="simulate"
NOISE_FORMAT="simulate"

# Run inference
python inference_final.py \
  --model_pth  "$MODEL_PTH" \
  --mri_mat    "$MRI_NII" \
  --output_folder "$OUTPUT_DIR" \
  --num_samples  "$NUM_SAMPLES" \
  --mri_format   "$MRI_FORMAT" \
  --noise_format "$NOISE_FORMAT"