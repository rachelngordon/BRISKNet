#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=logs/grasp_timing_vol.err
#SBATCH --output=logs/grasp_timing_vol.out
#SBATCH --exclude=''
#SBATCH --gpus-per-node=1
#SBATCH --job-name=grasp_timing_vol
#SBATCH --mem-per-gpu=80000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --partition=general
#SBATCH --time=700

# Load Micromamba
source /home/rachelgordon/micromamba/etc/profile.d/mamba.sh

# Activate your Micromamba environment
micromamba activate recon_mri

# Run the training script with srun
python time_raw_inference_scan.py \
  --spokes-per-frame-list 8 \
  --num-samples 15 \
  --save_example_images \
  --example_out_dir timing_examples
  --device cuda:0
