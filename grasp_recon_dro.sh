#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=logs/grasp_recon_test_set.err
#SBATCH --output=logs/grasp_recon_test_set.out
#SBATCH --exclude=''
#SBATCH --gpus-per-node=1
#SBATCH --job-name=grasp_recon_test_set
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
python grasp_recon_var_frames.py --dro-root /net/scratch2/rachelgordon/dro_test_set --lamda 0.001
