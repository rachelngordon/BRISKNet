# Breast DCE-MRI Reconstruction Agent

This project trains a reconstruction agent for highly undersampled breast DCE-MRI using the Dynamic Diffeomorphic Equivariant Imaging (DDEI) framework. The goal is to recover diagnostic-quality temporal dynamics for cancer treatment, diagnosis, and risk prediction without requiring ground-truth images during training.

## Development Principles
- This is research code: prefer fail-fast behavior over defensive fallback chains.
- Do not add broad catch-and-continue logic to hide configuration/environment problems.
- When a required dependency/path/config is wrong, raise a clear error and fix the root cause.
- Prefer architecture-driven behavior over redundant configuration knobs; if a feature is architecture-specific, hard-disable it for other architectures instead of requiring users to set a redundant value (for example, adjoint loss is LSFP-only and disabled for Mamba).
- Backward compatibility is not a priority; prefer simpler, clearer code over preserving old config/workflow behavior.
- Prefer vectorized tensor operations over Python loops where practical.
- Use reparameterization/architecture changes before adding extra regularization losses unless a regularizer has a clear, tested purpose.

## Core Ideas
- **Unsupervised objective**: Combine a measurement consistency (MC) loss in k-space with an equivariant imaging (EI) loss in image space to enforce physics fidelity and artifact removal without paired labels (`mc.py`, `ei.py`).
- **Supervision source**: Training uses only measured/ground-truth k-space data as supervision; there are no ground-truth image targets during training.
- **Backbone**: LSFPNet unrolls the Low-rank + Sparsity with Framelet transform and primal–dual fixed-point optimization; includes learnable cascades, Film-style modulation, and optional low-k DC (`lsfpnet.py`, `lsfpnet_encoding.py`, `radial_lsfp.py`).
- **Flexible transforms**: EI loss supports spatial transforms (rotation, warp, subsample) and optional noise/augmentation scheduling via YAML config.

## Data & Splits
- **Dataset**: fastMRI breast DCE-MRI; 300 radial k-space scans (288 spokes, 640 samples/spoke). 83 z-partitions are zero-padded to 192 slices then FFTed to image space. The data is located at /net/scratch2/rachelgordon/zf_data_192_slices/zf_kspace. 
The sensitivity maps are in /net/scratch2/rachelgordon/zf_data_192_slices/, each within a separate directory for the patient id. Tumor segmentations for each non-DRO malignant scan are in /net/scratch2/rachelgordon/zf_data_192_slices/tumor_segmentations. The reconstructions stored in subdirectories for each patient in /net/scratch2/rachelgordon/zf_data_192_slices/ are of shape (2, 320, 320) and are complex, so they should be converted to real for plotting by np.abs(img[0]+1j*img[1])
- **Splits**: 258 train / 15 val / remainder test (`data/data_split.json`).
- **Slice strategy**: One slice/partition per scan per epoch, randomly resampled each epoch (`dataloader.py` supports `num_random_slices`).
- **Temporal setup**: 8 spokes per frame → 36 timepoints. Training draws a random 24-frame window per scan; evaluation reconstructs with a sliding window of 24 frames with 12-frame overlap.

## Training Pipeline
- **Entry point**: `train_zf.py --config <yaml> --exp_name <run_id>`. Multi-GPU uses NCCL; logs, configs, checkpoints, evals stored under `output/<exp_name>/`.
- **Checkpoint config**: use `experiment.resume_checkpoint` for full-state resume (optimizer/epoch/curves), and `experiment.init_checkpoint` for weights-only warm start. `experiment.pretrained_checkpoint` is supported as a deprecated alias for `init_checkpoint`.
- **Config example**: `configs/config_ei_no_noise_encode_both.yaml` (LSFPNet, EI on, MC weight 10, adjoint loss weight 1, Adam lr 5e-4, batch size 1, curriculum optional).
- **Loss terms**:
  - MC loss on forward NUFFT outputs (`MCLoss`).
  - EI loss on transformed reconstructions (`EILoss`); warmup and duration schedule via `model.losses.ei_loss`.
  - Optional adjoint loss and normalization (spatial, temporal, or both).
- **Transforms & physics**: NUFFT operator (`radial_lsfp.MCNUFFT`) built from k-space metadata; optional time warps, rotations, subsampling (`transform.py`).

## Evaluation
- **Metrics**: SSIM, PSNR, MSE, LPIPS (with complex-to-magnitude handling in `loss_metrics.py`), k-space MSE/MAE, and Pearson correlation of tumor enhancement curve. Validation/test use DRO-simulated ground truth for reference.
- **Sliding evaluation**: Uses chunked inference (`utils.sliding_window_inference`) with configurable chunk size/overlap (`evaluation.chunk_size`, `evaluation.chunk_overlap`).
- **Data**: Evaluation is conducted only on IDs within the DRO dataset (data/DROSubID_vs_fastMRIbreastID.csv), and is conducted for both DRO (/net/scratch2/rachelgordon/dro_dataset contains directories for DROs with different temporal resolutions, indicated by number of timeframes, which each contain the DRO image/mask, GRASP reconstruction, and simulated k-space) with ground truth and non-DRO using only raw k-space (/net/scratch2/rachelgordon/zf_data_192_slices/zf_kspace) and generated tumor segmentations (/net/scratch2/rachelgordon/zf_data_192_slices/tumor_segmentations). Both evaluations compare deep learning and GRASP reconstructions with either non-DRO or DRO data.

## Running Experiments
- **Single run**: `python3 train_zf.py --config configs/config_ei_no_noise_encode_both.yaml --exp_name <name>`.
- **Grid search**: `grid_search_batch.py` splits hyperparameter sweeps across batches; default grid searches `model.losses.adj_loss.weight` and `model.losses.ei_loss.weight`. Update `base_config_file` to an existing YAML before running. SLURM example in `grid_search.sh`.
- **Outputs**: Each run writes `eval_results/eval_metrics.csv` for downstream parsing (`parse_results` in `grid_search_batch.py`).

## Experiment Design & Analysis
- Every sweep should have a brief causal hypothesis: what changed, why it should help, and what failure mode it targets.
- Prefer learning-oriented follow-ups over blind sweeps; propose the smallest next experiment that answers the open question.
- When metrics shift meaningfully or collapse appears, include a short "why" analysis tied to data mix, loss schedules, optimization settings, and sampling behavior.
- For any recommendation, include tradeoffs (what is gained, what is sacrificed, and viable alternatives).

## Lab Notebook & Logging
- Maintain a canonical experiment notebook at `experiment_log.md`.
- When new results or failures appear, immediately append:
  - run name/path, job id, config path, key hypothesis,
  - best/last metrics with epoch,
  - outcome vs hypothesis,
  - concrete next action.
- When submitting or canceling runs, log that action in `experiment_log.md` in the same turn.
- If a run crashes and is resubmitted, record root cause and fix; low-value failed run dirs may be deleted after logging the lesson.
- If a run completes with usable validation, add a takeaway block (what happened, what was learned, what to do next).

## Monitoring & Triage
- For status updates: check `squeue`, then reconcile with `sacct` for recently exited jobs.
- Treat stale logs as a triage signal, not an automatic resubmit trigger; confirm state/error first.
- When a run leaves the queue, capture: job id/state, error signature (if any), likely cause, fix, and resubmit details.
- If asked to "babysit" jobs, default to an active loop: monitor, diagnose failures, apply minimal fix, resubmit, and update `experiment_log.md`.
- Do not promise continuous monitoring after returning a message in a normal turn; continuous monitoring requires staying in-turn.

## Config Discipline
- New knobs must be intentional and expected to be ablated soon; avoid adding knobs for stable architectural behavior.
- Prefer hardcoding stable behavior over proliferating optional flags.
- Do not retroactively edit configs for completed runs; create new configs for new behavior.
- Keep reproducibility explicit: seeds and effective schedule settings should be logged for each run.

## Submitting Jobs (SLURM via submit.py)
- Use `submit.py` for distributed launches with `torchrun` + `submitit`.
- Run `submit.py` from an environment that has `submitit` installed (e.g., `brisknet`).
- Required flags: `--config`, `--exp-name`, `--nodes`, `--gpus-per-node`, `--micromamba-path`, `--env-name`.
- Slot policy:
  - Standard slots: assume 8 non-burst SLURM job slots are available; keep them utilized with the highest-value active experiments unless explicitly directed otherwise.
  - Burst slots: assume 2 additional `qos=burst` slots are available; treat burst as preemptible and use it for debugging/probes/sanity checks rather than long critical runs.
  - When proposing experiment plans, default to filling all standard slots first, then optionally use burst for short-turnaround diagnostics.
- Typical 1 node / 4 GPU submission:
  - `micromamba run -n brisknet python submit.py --config configs/config_sampling_2spf_rebin_v3_mamba_debug.yaml --exp-name mamba_2spf_rebin_v3_debug_1n4g --job-name mamba_2spf_dbg_1n4g --nodes 1 --gpus-per-node 4 --cpus-per-task 8 --partition general --timeout-min 240 --micromamba-path /net/projects/annawoodard/micromamba/etc/profile.d/micromamba.sh --env-name brisknet`
- Dry-run (print resolved settings only):
  - append `--dry-run` to the command above.
- Monitor and control:
  - `squeue -j <job_id>`
  - `scancel <job_id>`
  - logs are written under `<experiment.output_dir>/<exp-name>/submitit_logs/`.

## Notes & Tips
- Keep `data.root_dir` and `experiment.cluster` aligned with your cluster paths; `cluster_paths.py` can rewrite paths per environment.
- For curriculum learning, adjust `training.curriculum_learning.phases` to introduce higher accelerations gradually.
- When extending transforms, ensure they operate on rearranged video tensors (see `EILoss` and `transform.py`) and preserve complex channel conventions.
