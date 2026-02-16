#!/bin/bash

#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --partition=general
#SBATCH --time=700
#SBATCH --requeue
#SBATCH --job-name=grasp_pack
#SBATCH --output=/net/scratch2/annawoodard/grasp_logs/%x_%j.out
#SBATCH --error=/net/scratch2/annawoodard/grasp_logs/%x_%j.err
#SBATCH --open-mode=append

set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-/net/scratch2/annawoodard/grasp_targets_fastmri_train}"
TARGET_ROOT="${TARGET_ROOT:-/net/scratch2/annawoodard/grasp_targets_fastmri_train_packed}"
DATA_DIR="${DATA_DIR:-/net/scratch2/rachelgordon/zf_data_192_slices/zf_kspace}"
ENV_NAME="${ENV_NAME:-brisknet}"
ENV_PYTHON="${ENV_PYTHON:-/net/projects/annawoodard/micromamba/envs/${ENV_NAME}/bin/python}"
SPF_LIST="${SPF_LIST:-36 2}"
DELETE_SOURCE="${DELETE_SOURCE:-1}"
SHARD_OFFSET="${SHARD_OFFSET:-0}"

mkdir -p /net/scratch2/annawoodard/grasp_logs
mkdir -p "${TARGET_ROOT}"

cd /home/annawoodard/radial-breast-ddei

if [ -z "${SLURM_NTASKS:-}" ]; then
  echo "ERROR: SLURM_NTASKS is not set." >&2
  exit 1
fi

TOTAL_WORKERS="${SLURM_NTASKS}"
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

echo "[PACK-SLURM] job_id=${SLURM_JOB_ID} workers=${TOTAL_WORKERS} num_shards=${NUM_SHARDS} shard_offset=${SHARD_OFFSET}"

export SOURCE_ROOT TARGET_ROOT DATA_DIR ENV_PYTHON SPF_LIST DELETE_SOURCE NUM_SHARDS SHARD_OFFSET

srun --ntasks="${TOTAL_WORKERS}" --cpus-per-task="${SLURM_CPUS_PER_TASK}" --kill-on-bad-exit=1 \
  bash -lc '
    set -euo pipefail
    SHARD_INDEX="$((SHARD_OFFSET + SLURM_PROCID))"
    DELETE_FLAG=""
    if [ "${DELETE_SOURCE}" = "1" ]; then
      DELETE_FLAG="--delete-source"
    fi
    echo "[PACK-SLURM] node=${SLURMD_NODENAME} procid=${SLURM_PROCID} shard=${SHARD_INDEX}/${NUM_SHARDS}"
    "${ENV_PYTHON}" pack_grasp_targets_to_packed.py \
      --source-root "${SOURCE_ROOT}" \
      --target-root "${TARGET_ROOT}" \
      --data-dir "${DATA_DIR}" \
      --spokes-per-frame ${SPF_LIST} \
      ${DELETE_FLAG} \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${SHARD_INDEX}"
  '
