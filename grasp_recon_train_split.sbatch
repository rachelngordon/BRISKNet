#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=80000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=general
#SBATCH --time=700
#SBATCH --requeue
#SBATCH --job-name=grasp_train_spf
#SBATCH --output=/net/scratch2/annawoodard/grasp_logs/%x_%j.out
#SBATCH --error=/net/scratch2/annawoodard/grasp_logs/%x_%j.err
#SBATCH --open-mode=append

set -euo pipefail

SPF="${SPF:?Set SPF via sbatch --export=ALL,SPF=<2|36>}"
TARGET_ROOT="${TARGET_ROOT:-/net/scratch2/annawoodard/grasp_targets_fastmri_train_packed}"
DATA_DIR="${DATA_DIR:-/net/scratch2/rachelgordon/zf_data_192_slices/zf_kspace}"
CSMAPS_DIR="${CSMAPS_DIR:-/net/scratch2/rachelgordon/zf_data_192_slices/cs_maps}"
SPLIT_FILE="${SPLIT_FILE:-/home/annawoodard/radial-breast-ddei/data/data_split.json}"
ENV_NAME="${ENV_NAME:-brisknet}"
RECON_MAX_ITER="${RECON_MAX_ITER:-10}"
ENV_PYTHON="${ENV_PYTHON:-/net/projects/annawoodard/micromamba/envs/${ENV_NAME}/bin/python}"
SLICE_PRIORITY_ORDER="${SLICE_PRIORITY_ORDER:-middle_first}"
PRIORITY_SLICES_PER_EXAM="${PRIORITY_SLICES_PER_EXAM:-24}"
SHARD_OFFSET="${SHARD_OFFSET:-0}"

mkdir -p /net/scratch2/annawoodard/grasp_logs
mkdir -p "${TARGET_ROOT}"

cd /home/annawoodard/radial-breast-ddei

if [ -z "${SLURM_NTASKS:-}" ]; then
  echo "ERROR: SLURM_NTASKS is not set. Submit with explicit --ntasks-per-node." >&2
  exit 1
fi

TOTAL_WORKERS="${SLURM_NTASKS}"
if [ "${TOTAL_WORKERS}" -lt 1 ]; then
  echo "ERROR: TOTAL_WORKERS must be >= 1 (got ${TOTAL_WORKERS})." >&2
  exit 1
fi

NUM_SHARDS="${NUM_SHARDS:-${TOTAL_WORKERS}}"
if [ "${NUM_SHARDS}" -lt 1 ]; then
  echo "ERROR: NUM_SHARDS must be >= 1 (got ${NUM_SHARDS})." >&2
  exit 1
fi
if [ "${SHARD_OFFSET}" -lt 0 ]; then
  echo "ERROR: SHARD_OFFSET must be >= 0 (got ${SHARD_OFFSET})." >&2
  exit 1
fi
if [ $((SHARD_OFFSET + TOTAL_WORKERS)) -gt "${NUM_SHARDS}" ]; then
  echo "ERROR: shard range [${SHARD_OFFSET}, $((SHARD_OFFSET + TOTAL_WORKERS - 1))] exceeds NUM_SHARDS=${NUM_SHARDS}." >&2
  exit 1
fi

echo "[GRASP-SLURM] job_id=${SLURM_JOB_ID} nodes=${SLURM_JOB_NUM_NODES} workers=${TOTAL_WORKERS} spf=${SPF} num_shards=${NUM_SHARDS} shard_offset=${SHARD_OFFSET}"

# Export variables consumed inside the srun-launched bash subshell.
export SPF TARGET_ROOT DATA_DIR CSMAPS_DIR SPLIT_FILE ENV_NAME RECON_MAX_ITER
export ENV_PYTHON SLICE_PRIORITY_ORDER PRIORITY_SLICES_PER_EXAM NUM_SHARDS SHARD_OFFSET

srun --ntasks="${TOTAL_WORKERS}" --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=1 --kill-on-bad-exit=1 \
  bash -lc '
    set -euo pipefail
    SHARD_INDEX="$((SHARD_OFFSET + SLURM_PROCID))"
    echo "[GRASP-SLURM] node=${SLURMD_NODENAME} procid=${SLURM_PROCID} localid=${SLURM_LOCALID} shard=${SHARD_INDEX}/${NUM_SHARDS} offset=${SHARD_OFFSET}"
    "${ENV_PYTHON}" generate_grasp_targets_split.py \
      --data-dir "${DATA_DIR}" \
      --target-root "${TARGET_ROOT}" \
      --split-file "${SPLIT_FILE}" \
      --splits train \
      --spokes-per-frame "${SPF}" \
      --total-spokes 288 \
      --slice-mode all \
      --slice-priority-order "${SLICE_PRIORITY_ORDER}" \
      --priority-slices-per-exam "${PRIORITY_SLICES_PER_EXAM}" \
      --csmaps-dir "${CSMAPS_DIR}" \
      --save-dtype complex64 \
      --recon-max-iter "${RECON_MAX_ITER}" \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${SHARD_INDEX}"
  '
