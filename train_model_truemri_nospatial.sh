#!/bin/bash
#SBATCH --job-name=csm
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/denoise_%j.out
#SBATCH --error=logs/denoise_%j.err

unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Load conda and activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate restomer

# Ensure logs folder exists
mkdir -p logs

# (Optional) Change to the directory from which sbatch was called

# Run training
python basicsr/train.py \
  -opt Denoising/Options/denoising_MRI_no_spatial.yml \
  --launcher none
