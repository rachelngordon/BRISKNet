# SLURM/NCCL "no GPUs found" incidents

Date updated: 2026-02-16
Owner: annawoodard
Purpose: share reproducible evidence with sysadmin.

## Symptom
Multiple distributed jobs fail immediately at torch distributed init with:

- `ValueError: ProcessGroupNCCL is only supported with GPUs, no GPUs found!`

This occurs even though the job requested GPUs via SLURM (`gres/gpu`) and was scheduled to GPU nodes.

## Confirmed failing jobs (recent)

1. Job `714035` (`m36_bk_absmae_nomc_w3_long_2n4g`)
- State: `FAILED`
- Nodes: `k002,k003`
- Log: `/net/projects2/annawoodard/experiments/m36_bk_absmae_nomc_w3_long_2n4g/submitit_logs/714035_0_log.err`
- Error lines include repeated `ProcessGroupNCCL ... no GPUs found`.

2. Job `714037` (`m36_bk_absmse_mc010_lr5e5_2n4g`)
- State: `FAILED`
- Nodes: `k002,k003`
- Log: `/net/projects2/annawoodard/experiments/m36_bk_absmse_mc010_lr5e5_2n4g/submitit_logs/714037_0_log.err`
- Same NCCL no-GPU signature.

3. Job `714038` (`m36_bk_absmse_nomc_freq005_2n4g`)
- State: `CANCELLED` (after early failure/requeue handling)
- Nodes: `k002,k003`
- Log: `/net/projects2/annawoodard/experiments/m36_bk_absmse_nomc_freq005_2n4g/submitit_logs/714038_0_log.err`
- Same NCCL no-GPU signature.

Older related evidence:
- Job `705900` (`mamba_temporal_2spf_debug_1n2g_postopt_anygpu`) also showed the same error in its submitit log.

## Pattern observed
- Failures correlate strongly with `k002/k003` (`l40s`) placements.
- Equivalent code path succeeds on A40/A100/H100/H200 nodes (many completed runs in same environment).
- This points to node/runtime GPU visibility/configuration, not model code.