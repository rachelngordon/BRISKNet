#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=1
#SBATCH --error=logs/zero_pad_data.err
#SBATCH --output=logs/zero_pad_data.out
#SBATCH --exclude=''
#SBATCH --gpus-per-node=1
#SBATCH --job-name=zero_pad_data
#SBATCH --mem-per-gpu=50000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --partition=gpuq
#SBATCH --time=1440


# Load Micromamba
source /gpfs/data/karczmar-lab/workspaces/rachelgordon/micromamba/etc/profile.d/micromamba.sh

# Activate your Micromamba environment
micromamba activate recon_mri

# Run the training script with srun
bash process_all_data.sh /net/scratch2/rachelgordon/fastMRI_breast_data/fastMRI_breast_IDS_ /net/scratch2/rachelgordon/zf_data_192_slices 192