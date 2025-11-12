#!/bin/bash
#SBATCH --job-name=denoise_restormer
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/denoise_%j.out
#SBATCH --error=logs/denoise_%j.err

# Load conda and activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate restomer

# Ensure logs folder exists
mkdir -p logs

# (Optional) Change to the directory from which sbatch was called

# Run training
python basicsr/train.py \
  -opt Denoising/Options/GaussianGrayDenoising_RestormerSigma_smooth_ft_csm.yml \
  --launcher none
