#!/bin/bash
set -euo pipefail

# Submission helper for full train-split GRASP target generation.
#
# Usage:
#   bash grasp_recon.sh
#   bash grasp_recon.sh 36 2
#   NODES=2 GPUS_PER_NODE=4 bash grasp_recon.sh 36 2
#   JOBS_PER_SPF=2 NODES=2 GPUS_PER_NODE=4 bash grasp_recon.sh 36
#
# This submits one multi-node/multi-GPU job per SPF via grasp_recon_train_split.sbatch.
# Each worker reconstructs one shard with shard-index = SHARD_OFFSET + SLURM_PROCID.

SPFS=("$@")
if [ "${#SPFS[@]}" -eq 0 ]; then
  SPFS=(36 2)
fi

NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
PARTITION="${PARTITION:-general}"
TIME_MIN="${TIME_MIN:-700}"
MEM_PER_GPU="${MEM_PER_GPU:-80000}"
TARGET_ROOT="${TARGET_ROOT:-/net/scratch2/annawoodard/grasp_targets_fastmri_train_packed}"
ENV_NAME="${ENV_NAME:-brisknet}"
RECON_MAX_ITER="${RECON_MAX_ITER:-10}"
JOBS_PER_SPF="${JOBS_PER_SPF:-1}"
WORKERS_PER_JOB="$((NODES * GPUS_PER_NODE))"
NUM_SHARDS="${NUM_SHARDS:-$((WORKERS_PER_JOB * JOBS_PER_SPF))}"
SLICE_PRIORITY_ORDER="${SLICE_PRIORITY_ORDER:-middle_first}"
PRIORITY_SLICES_PER_EXAM="${PRIORITY_SLICES_PER_EXAM:-24}"
CONSTRAINT="${CONSTRAINT:-}"
EXCLUDE="${EXCLUDE:-}"
QOS="${QOS:-}"

if [ "${NODES}" -lt 1 ] || [ "${GPUS_PER_NODE}" -lt 1 ] || [ "${WORKERS_PER_JOB}" -lt 1 ]; then
  echo "ERROR: NODES and GPUS_PER_NODE must be >= 1." >&2
  exit 1
fi
if [ "${JOBS_PER_SPF}" -lt 1 ]; then
  echo "ERROR: JOBS_PER_SPF must be >= 1." >&2
  exit 1
fi
if [ "${NUM_SHARDS}" -ne $((WORKERS_PER_JOB * JOBS_PER_SPF)) ]; then
  echo "ERROR: NUM_SHARDS (${NUM_SHARDS}) must equal NODES*GPUS_PER_NODE*JOBS_PER_SPF ($((WORKERS_PER_JOB * JOBS_PER_SPF)))." >&2
  exit 1
fi

for SPF in "${SPFS[@]}"; do
  for ((JOB_IDX=0; JOB_IDX<JOBS_PER_SPF; JOB_IDX++)); do
    SHARD_OFFSET="$((JOB_IDX * WORKERS_PER_JOB))"
    JOB_NAME="grasp_train_spf${SPF}_${NODES}n${GPUS_PER_NODE}g_j${JOB_IDX}"
    SBATCH_ARGS=(
      --job-name "${JOB_NAME}"
      --nodes "${NODES}"
      --ntasks-per-node "${GPUS_PER_NODE}"
      --gpus-per-task 1
      --cpus-per-task "${CPUS_PER_TASK}"
      --partition "${PARTITION}"
      --time "${TIME_MIN}"
      --mem-per-gpu "${MEM_PER_GPU}"
      --export "ALL,SPF=${SPF},NUM_SHARDS=${NUM_SHARDS},SHARD_OFFSET=${SHARD_OFFSET},TARGET_ROOT=${TARGET_ROOT},ENV_NAME=${ENV_NAME},RECON_MAX_ITER=${RECON_MAX_ITER},SLICE_PRIORITY_ORDER=${SLICE_PRIORITY_ORDER},PRIORITY_SLICES_PER_EXAM=${PRIORITY_SLICES_PER_EXAM}"
    )
    if [ -n "${CONSTRAINT}" ]; then
      SBATCH_ARGS+=(--constraint "${CONSTRAINT}")
    fi
    if [ -n "${EXCLUDE}" ]; then
      SBATCH_ARGS+=(--exclude "${EXCLUDE}")
    fi
    if [ -n "${QOS}" ]; then
      SBATCH_ARGS+=(--qos "${QOS}")
    fi
    sbatch \
      "${SBATCH_ARGS[@]}" \
      grasp_recon_train_split.sbatch
  done
done
