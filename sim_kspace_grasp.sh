#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=logs/sim_kspace_grasp.err
#SBATCH --output=logs/sim_kspace_grasp.out
#SBATCH --exclude=''
#SBATCH --gpus-per-node=1
#SBATCH --job-name=sim_kspace_grasp
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
python3 simulate_kspace_grasp_var_frames.py   --dro-root /net/scratch2/rachelgordon/dro_var_frames --traj-method get_traj --kspace-noise-std 0.05 --kspace-noise-seed 1234