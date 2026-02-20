#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=logs/grasp_recon_var_frames_lamda0.0001.err
#SBATCH --output=logs/grasp_recon_var_frames_lamda0.0001.out
#SBATCH --exclude=''
#SBATCH --gpus-per-node=1
#SBATCH --job-name=grasp_recon_var_frames_lamda0.0001
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
python grasp_recon_var_frames.py --lamda 0.0001
