# Experiment Notebook: Beating GRASP

- Last updated: 2026-02-12
- Source overlay: `/net/projects2/annawoodard/experiments/mamba_36spf_overlay/overlay_metric_tables.txt`
- Scope in this entry: SPF=36 sweep family in `mamba_36spf_overlay` (this is a staging step toward 2spf).

## Objective
- Primary: beat GRASP on final temporal fidelity and final reconstruction quality.
- Secondary: get there with a training process that is stable and not painfully slow.

## GRASP Baseline (from checkpoints used in these runs)
- SSIM: `0.9795`
- PSNR: `48.58`
- LPIPS: `~0.0021`
- Enhancement curve corr: `0.9993`

Current best BRISKNet in this sweep (`707694` at ep35): SSIM `0.872`, PSNR `39.2`, LPIPS `0.0497`, corr `0.997`.

## Snapshot Findings
- Best overall run so far is `mamba_36spf_sweepB_early_ei_archfix_lr5em4_minlf6_wu5_ep160_archtune_dconv4_2n4g` (`707694`): strongest PSNR/LPIPS and high corr.
- Many “promising” runs did not progress because they were manually canceled or cut off early (often <= ep20), not because they clearly failed scientifically.
- Most Sweep-B runs have `rebin_loss.enable: false`; rebin was not active there.
- In rebin-enabled Sweep-C runs, `train_rebin_loss` stayed zero in observed epochs because rebin warmup starts late (`warmup=20` or `30`) and jobs did not run long enough beyond that transition.

## Why EI Curves Look Confusing
- `train_ei_loss` raw values are tiny (`~1e-9`) after warmup; this is expected from scale + averaging.
- `weighted_train_ei_loss` rises after EI turns on because EI weight ramps in (raw EI can stay small while weighted term grows).
- `val_ei_loss` is noisy and can increase while PSNR/SSIM improve; EI augmentation objective is not perfectly aligned with pixel metrics.
- The burst debug run (`708510`) is not comparable: EI warmup `0`, EI duration `4`, and produced very large weighted EI early.

## Promising But Stalled Early
| experiment | job | best PSNR | best SSIM | best corr | last eval epoch | slurm state | note |
|---|---:|---:|---:|---:|---:|---|---|
| `mamba_36spf_sweepC_early_ei_rebin_archfix_2n4g` | 708500 | 37.70 | 0.873 | 0.995 | 20 | CANCELLED by 22734 | Rebin-enabled branch. |
| `mamba_36spf_sweepB_early_ei_archfix_lr2em4_2n4g` | 706829 | 37.50 | 0.799 | 0.995 | 10 | CANCELLED by 22734 | Early-EI branch (rebin disabled). |
| `mamba_36spf_sweepB_early_ei_archfix_lr3em4_2n4g` | 706830 | 37.80 | 0.798 | 0.995 | 10 | CANCELLED by 22734 | Early-EI branch (rebin disabled). |
| `mamba_36spf_sweepB_early_ei_archfix_lr5em4_minlf6_wu5_ep160_2n4g` | 707628 | 37.70 | 0.778 | 0.995 | 10 | CANCELLED by 22734 | Early-EI branch (rebin disabled). |

## Hypotheses And Outcomes (Grouped)
### H1: MC-first stabilization helps anatomy before EI
- Tested in: `sweepA_mc_archfix` (MC-heavy baseline).
- Outcome: stable long run to ep100, good trend, but still far from GRASP ceiling.
- Decision: keep MC-first idea, but use it as a phase, not the whole training recipe.

### H2: Turning on EI early can improve temporal metrics faster
- Tested in: Sweep-B early-EI runs (many LR/min-lr-factor variants).
- Outcome: early eval jumps can look strong; some runs reach high early SSIM/PSNR quickly, but several are volatile and under-trained (short trajectories).
- Decision: keep EI, but reduce variance by running fewer configs deeper instead of many shallow runs.

### H3: Rebin consistency (teacher, offset, temporal-diff, dynamic mask) will improve kinetics
- Tested in: Sweep-C rebin variants.
- Outcome so far inconclusive because most runs ended near/before rebin warmup transition; train rebin loss is zero in logged epochs.
- Decision: run one sweep-C config long enough past rebin warmup before judging.

### H4: Arch tuning (`d_state=32`, dconv4) gives quality boost
- Tested in: `archtune_dconv4` and related archtune runs.
- Outcome: strongest observed run currently (`707694`).
- Decision: use this as the main backbone for next long run.

## Experiment Ledger (Current Overlay Set)
| run | job | running_now | best PSNR | best SSIM | best corr | best LPIPS | last eval ep | EI warmup/duration | rebin enabled (warmup,weight) | note |
|---|---:|---|---:|---:|---:|---:|---:|---|---|---|
| `mamba_36spf_sweepB_early_ei_archfix_lr5em4_minlf6_wu5_ep160_archtune_dconv4_2n4g` | 707694 | yes | 39.20 | 0.872 | 0.997 | 0.0497 | 35 | 8/20 | False (20,400) | Early-EI branch (rebin disabled). |
| `mamba_36spf_sweepA_mc_archfix_2n4g` | 706808 | no | 39.10 | 0.832 | 0.996 | 0.0401 | 100 | 8/20 | False (20,400) | MC-only baseline (no rebin). |
| `mamba_36spf_sweepB_early_ei_archfix_lr6em4_minlf6_wu5_ep160_2n4g` | 708441 | yes | 38.80 | 0.840 | 0.995 | 0.0518 | 10 | 8/20 | False (20,400) | Early-EI branch (rebin disabled). |
| `mamba_36spf_sweepB_early_ei_archfix_lr3em4_minlf6_wu5_ep160_eiw12_d40_2n4g` | 708442 | yes | 38.40 | 0.786 | 0.995 | 0.0478 | 20 | 12/40 | False (20,400) | Early-EI branch (rebin disabled). |
| `mamba_36spf_sweepB_early_ei_archfix_lr3em4_minlf8_wu5_ep160_2n4g` | 708439 | yes | 38.00 | 0.797 | 0.995 | 0.0623 | 10 | 8/20 | False (20,400) | Early-EI branch (rebin disabled). |
| `mamba_36spf_sweepB_early_ei_archfix_lr3em4_2n4g` | 706830 | no | 37.80 | 0.798 | 0.995 | 0.0641 | 10 | 8/20 | False (20,400) | Early-EI branch (rebin disabled). |
| `mamba_36spf_sweepB_early_ei_archfix_lr3em4_minlf6_wu5_ep160_2n4g` | 708440 | yes | 37.80 | 0.774 | 0.995 | 0.0614 | 10 | 8/20 | False (20,400) | Early-EI branch (rebin disabled). |
| `mamba_36spf_sweepB_early_ei_archfix_lr5em4_minlf6_wu5_ep160_2n4g` | 707628 | no | 37.70 | 0.778 | 0.995 | 0.0612 | 10 | 8/20 | False (20,400) | Early-EI branch (rebin disabled). |
| `mamba_36spf_sweepC_early_ei_rebin_archfix_2n4g` | 708500 | no | 37.70 | 0.873 | 0.995 | 0.0809 | 20 | 8/20 | True (20,400) | Rebin-enabled branch. |
| `mamba_36spf_sweepB_early_ei_archfix_lr2em4_2n4g` | 706829 | no | 37.50 | 0.799 | 0.995 | 0.0704 | 10 | 8/20 | False (20,400) | Early-EI branch (rebin disabled). |
| `mamba_36spf_sweepB_early_ei_archfix_lr5em4_minlf6_wu5_ep160_archtune_fullgrad_2n4g` | 708513 | yes | 37.50 | 0.813 | 0.996 | 0.0600 | 10 | 8/20 | False (20,400) | Early-EI branch (rebin disabled). |
| `mamba_36spf_sweepB_early_ei_archfix_lr5em4_minlf6_wu5_ep160_archtune_2n4g` | 708443 | no | 37.40 | 0.811 | 0.994 | 0.0756 | 5 | 8/20 | False (20,400) | Early-EI branch (rebin disabled). |
| `mamba_36spf_sweepC_ei_rebin_stabilized_combo_2n4g` | 708438 | yes | 37.40 | 0.737 | 0.994 | 0.0641 | 14 | 8/25 | True (30,120) | Rebin-enabled branch. |
| `mamba_36spf_sweepB_early_ei_archfix_lr15em4_minlf4_2n4g` | 707357 | no | 37.30 | 0.782 | 0.994 | 0.0764 | 10 | 8/20 | False (20,400) | Early-EI branch (rebin disabled). |
| `mamba_36spf_sweepB_early_ei_archfix_lr15em4_minlf6_2n4g` | 707358 | no | 37.20 | 0.772 | 0.995 | 0.0759 | 10 | 8/20 | False (20,400) | Early-EI branch (rebin disabled). |
| `mamba_36spf_sweepC_early_ei_rebin_archfix_fullgrad_2n4g` | 708514 | yes | 36.80 | 0.745 | 0.994 | 0.0836 | 10 | 8/20 | True (20,400) | Rebin-enabled branch. |
| `mamba_36spf_sweepB_early_ei_archfix_lr4em4_minlf6_wu5_ep160_2n4g` | 707462 | no | 36.50 | 0.740 | 0.995 | 0.0709 | 10 | 8/20 | False (20,400) | Early-EI branch (rebin disabled). |
| `mamba_36spf_sweepB_early_ei_archfix_lr15em4_2n4g` | 706828 | no | 35.70 | 0.709 | 0.993 | 0.0964 | 5 | 8/20 | False (20,400) | Early-EI branch (rebin disabled). |
| `mamba_36spf_temporal_pretrain_2n4g` | 706273 | no | 35.60 | 0.748 | 0.992 | 0.0974 | 25 | 24/40 | False (60,400) | Temporal pretrain attempt. |
| `mamba_36spf_sweepB_early_ei_archfix_2n4g` | 706809 | no | 35.30 | 0.678 | 0.993 | 0.1030 | 5 | 8/20 | False (20,400) | Early-EI branch (rebin disabled). |
| `mamba_36spf_sweepB_early_ei_archfix_lr5em5_2n4g` | 706827 | no | 34.10 | 0.601 | 0.993 | 0.1230 | 5 | 8/20 | False (20,400) | Early-EI branch (rebin disabled). |
| `mamba_36spf_fullgrad_ei_debugsafe_nocpei_1n1g_burst` | 708510 | no | 33.50 | 0.617 | 0.993 | 0.1860 | 4 | 0/4 | False (20,400) | Debug burst (warmup=0, duration=4). |
| `mamba_36spf_sweepB_early_ei_archfix_lr3em5_2n4g` | 706826 | no | 32.90 | 0.540 | 0.993 | 0.1480 | 5 | 8/20 | False (20,400) | Early-EI branch (rebin disabled). |

## Immediate Path To Beat GRASP (from this evidence)
1. Stop breadth search: keep only 1-2 high-ceiling runs (`707694` family and one rebin-enabled Sweep-C variant) and run to >=80 epochs.
2. Ensure rebin is actually active in the chosen main run (`rebin_loss.enable: true`) and starts early enough to be observable (not only near end).
3. Keep deterministic EI validation on; compare at fixed epochs (5, 10, 15, 20, …) against GRASP baseline.
4. Track transition checkpoints (`pre_ei`, `pre_rebin`) and do quick inference snapshots to catch regressions right after loss turn-on.
5. If PSNR rises while SSIM drops repeatedly after EI-on, lower EI aggressiveness (longer EI warmup or lower EI weight ramp) before trying bigger architecture changes.

## Open Questions
- Is the final target to beat GRASP on all global metrics, or primarily lesion-region temporal fidelity?
- Should 36spf stage be optimized for anatomy only, then transfer aggressively to 2spf, instead of forcing early temporal constraints?

## Next Update Template
- Date:
- New runs:
- Best metrics (PSNR/SSIM/LPIPS/corr):
- What changed in config:
- Decision for next run:

## 2026-02-12 Deadline Matrix (7 New Slots + Keep 1 LSFP)

### Slot policy used
- Kept existing LSFP run queued: `707436` (`sampling_2spf_rebin_v3_ft120_4n4g_warmfix`).
- Canceled all active Mamba runs to avoid fragmented progress and free slots:
  - `708440 708439 708438 707694 708514 708513 708442 708441`
- Submitted 7 new focused runs (below) to refill remaining slots.

### Hypothesis-driven matrix

| slot | experiment | job id | config | hypothesis |
|---|---|---:|---|---|
| keep | LSFP continuation | 707436 | existing run | Keep strongest LSFP track alive as paper-safe baseline while Mamba matrix runs. |
| 1 | m2 transfer main + rebin | 708702 | `configs/deadline_20260212/m2_deadline_xfer_main_rebin.yaml` | Warm-start from best 36spf Mamba + standard EI/rebin schedule should improve 2spf convergence speed and final temporal fidelity. |
| 2 | m2 transfer long MC then rebin | 708703 | `configs/deadline_20260212/m2_deadline_xfer_long_mc_rebin_late.yaml` | Longer MC-only phase should improve anatomy before EI/rebin, reducing early instability and flattening. |
| 3 | m2 transfer lower EI + rebin | 708704 | `configs/deadline_20260212/m2_deadline_xfer_low_ei_rebin.yaml` | Lower EI weight should reduce over-constraint and improve SSIM/LPIPS while preserving curve metrics. |
| 4 | m2 transfer no-rebin control | 708705 | `configs/deadline_20260212/m2_deadline_xfer_no_rebin_control.yaml` | Control arm to isolate whether rebin contributes net gain or hurt at 2spf. |
| 5 | m36 MC refresh from best | 708706 | `configs/deadline_20260212/m36_deadline_mc_refresh_from_best.yaml` | MC-only continuation from best 36spf checkpoint should close spatial quality gap without EI-induced artifacts. |
| 6 | m36 late EI from best | 708707 | `configs/deadline_20260212/m36_deadline_late_ei_from_best.yaml` | Delaying EI should keep MC recovery then improve temporal metrics with less damage to anatomy. |
| 7 | m36 late EI + late rebin | 708708 | `configs/deadline_20260212/m36_deadline_late_ei_plus_rebin_from_best.yaml` | Rebin added only after EI stabilization should improve curve fidelity without early collapse. |

### Shared initialization choice
- All 7 new runs use best 36spf checkpoint as warm start (`experiment.init_checkpoint`):
  - `/net/projects2/annawoodard/experiments/mamba_36spf_sweepB_early_ei_archfix_lr5em4_minlf6_wu5_ep160_archtune_dconv4_2n4g/mamba_36spf_sweepB_early_ei_archfix_lr5em4_minlf6_wu5_ep160_archtune_dconv4_2n4g_best_model.pth`

### Immediate decision rule (first checkpoint gate)
- After first eval windows (epochs 10/15/20), prioritize continuation of runs that improve curve correlation and LPIPS together; de-prioritize any run that only improves one metric while collapsing the other.

## 2026-02-12 2-SPF Inference Timing Probe (Burst)
- Action: submitted quick burst probe job to measure comparable 2-SPF inference timing for Mamba vs LSFP and reuse GRASP timing reference.
- Job:
  - `709983` (`probe_2spf_inf_burst`)
  - Queue: `general` + `qos=burst`
  - Resources: `1 node x 1 GPU`, `cpus_per_task=8`, `timeout=240 min`
- Launcher:
  - `submit.py` with entry script `probe_inference_2spf.py`
  - Config anchor: `configs/config_sampling_2spf_rebin_v3_mamba_temporal_speed12_arrival_only_2n4g_ablation.yaml`
- Probe targets:
  - Mamba exp dir: `/net/projects2/annawoodard/experiments/sampling_2spf_rebin_v3_mamba_temporal_2n4g_arrival_only_ablation`
  - LSFP exp dir: `/net/projects2/annawoodard/experiments/sampling_2spf_rebin_v3_ft120_4n4g_warmfix`
- Output summary path:
  - `/net/projects2/annawoodard/experiments/probe_2spf_inference_burst/probe_summary.json`

### 2026-02-12 update: preempted run + successful n=4 rerun
- `709983` (`probe_2spf_inf_burst`) was preempted mid-run (burst), then requeued; canceled and replaced with smaller quick probe.
- Replacement:
  - `709999` (`probe_2spf_n4_burst`), completed.
  - Summary JSON: `/net/projects2/annawoodard/experiments/probe_2spf_inference_burst_n4/probe_summary.json`
- 2-SPF inference timing (recon-only):
  - Mamba (`sampling_2spf_rebin_v3_mamba_temporal_2n4g_arrival_only_ablation`): `34.183 ± 3.571 s/sample` (n=4)
  - LSFP (`sampling_2spf_rebin_v3_ft120_4n4g_warmfix`): `41.136 ± 1.586 s/sample` (n=4)
  - GRASP reference (from `val_inference_logs.json`): `221.717 ± 3.063 s/sample` (n=15)

## 2026-02-13 Update: Latest 36-SPF Matrix (6-8h Runtime)

### Runs analyzed
- `m36_dline_timeenc_mixmc002_ctrl_2n4g` (job `709973`)
- `m36_dline_timeenc_mixmc005_ctrl_2n4g` (job `709974`)
- `m36_dline_timeenc_mixmc002_earlyei_norebin_2n4g` (job `709975`)
- `m36_dline_timeenc_mixmc002_earlyei_rebin_2n4g` (job `709976`)

### Source of truth used
- Overlay tables/plots: `/net/projects2/annawoodard/experiments/mamba_overlay_36spf/overlay_metric_tables.txt` (corrected overlay source).
- Checkpoint curves from each run's `<exp_name>_model.pth` (not just overlay event files).
- Note: overlay from event logs can look stale after requeue/multi-attempt runs; checkpoint curves contain the latest eval points.

### Current snapshot vs GRASP baseline
- GRASP baseline in these checkpoints: SSIM `0.9795`, PSNR `48.580`, LPIPS `0.00211`, curve corr `0.99933`.

| run | ckpt epoch | best PSNR (epoch) | best SSIM (epoch) | best LPIPS (epoch) | best corr (epoch) | latest PSNR | latest SSIM | latest LPIPS | latest corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `m36_dline_timeenc_mixmc002_ctrl_2n4g` | 31 | 41.435 (e20) | 0.9264 (e10) | 0.0297 (e25) | 0.9984 (e25) | 36.339 (e30) | 0.5595 (e30) | 0.0412 (e30) | 0.9982 (e30) |
| `m36_dline_timeenc_mixmc005_ctrl_2n4g` | 29 | 41.369 (e20) | 0.9308 (e10) | 0.0298 (e25) | 0.9984 (e25) | 40.492 (e25) | 0.7931 (e25) | 0.0298 (e25) | 0.9984 (e25) |
| `m36_dline_timeenc_mixmc002_earlyei_norebin_2n4g` | 28 | 41.086 (e5) | 0.9153 (e5) | 0.0344 (e5) | 0.9978 (e5) | 8.112 (e25) | 0.2871 (e25) | 0.6132 (e25) | 0.9953 (e25) |
| `m36_dline_timeenc_mixmc002_earlyei_rebin_2n4g` | 17 | 41.072 (e5) | 0.9191 (e5) | 0.0348 (e5) | 0.9978 (e5) | 9.146 (e15) | 0.1222 (e15) | 0.6125 (e15) | 0.8907 (e15) |

### What we learned
- Early EI (`warmup=5`, `weight=6000`) is unstable in this setup:
  - Both early-EI runs collapse immediately after EI turns on.
  - Rebin + early EI is worst (large PSNR/SSIM failure by e10-e15).
- The control runs are much healthier through e20, then degrade after EI activation:
  - `mixmc002_ctrl`: good through e20, then strong collapse by e30.
  - `mixmc005_ctrl`: also degrades after e20, but less severely so far.
- Rebin is **not** the cause in the control runs yet:
  - `rebin warmup=70`, so it has not started during e25-e30 collapse.
  - This points to EI schedule/strength, not rebin.
- Mixed MC with slightly more MSE (`mse_weight=0.05`) appears more robust than `0.02` in this matrix.

### Gap to GRASP (best observed so far in this matrix)
- Best PSNR gap: `41.435 vs 48.580` (about `-7.15 dB`).
- Best SSIM gap: `0.9308 vs 0.9795` (about `-0.0487`).
- LPIPS remains far from GRASP despite improvements.
- Temporal corr is already close numerically, so main gap is still image quality.

### Updated hypotheses
1. Main failure mode is EI overpowering MC once EI turns on, not lack of rebin.
2. 36-SPF should use longer MC-dominant phase and gentler EI ramp; this is likely the highest-ROI path before returning to 2-SPF.
3. For now, rebin should stay delayed/off at 36-SPF until EI is stable.

### Recommended immediate 36-SPF next steps
1. Keep running `mixmc005_ctrl` as the lead run (currently most stable after EI-on).
2. Start a continuation branch from best pre-collapse checkpoint (`e20`) with gentler EI:
   - EI warmup `>=35`
   - EI weight `~1000-2000` (not 6000-8000)
   - EI duration `~80-100`
   - keep EI metric `MSE` (early-EI `MAE` variants were unstable here).
3. Add EI activation gate in config (enable EI only after MC reaches threshold), then compare against fixed warmup.
4. Keep rebin disabled or very late in 36-SPF while tuning EI stability; revisit rebin once spatial quality is retained past e30.

## 2026-02-13 Planned Matrix: 36-SPF Distillation + Shuffle Buffer

### Motivation for this matrix
- Recent evidence points to EI instability as the main collapse driver; rebin is not yet the bottleneck at 36-SPF.
- We added an exam-aware shuffle buffer sampler to reduce repeated file churn and improve data throughput.
- Next question: can distillation pressure improve convergence while keeping EI stable?

### Important note
- Current `teacher_distill` in code is **model-to-model checkpoint distillation**, not direct GRASP image-target supervision.
- This matrix uses that existing distillation path (fast to run now) while keeping architecture fixed.

### Shared settings (all runs below)
- SPF: `36` (8 frames).
- Shuffle buffer enabled:
  - `dataloader.shuffle_buffer_enable: true`
  - `dataloader.shuffle_buffer_active_exams: 12`
  - `dataloader.shuffle_buffer_slices_per_exam: 8`
  - `dataloader.shuffle_buffer_replace_fraction: 0.25`
- Student init checkpoint:
  - `/net/projects2/annawoodard/experiments/m36_dline_timeenc_mixmc005_ctrl_2n4g/m36_dline_timeenc_mixmc005_ctrl_2n4g_best_model.pth`
- Teacher checkpoint (for distill-enabled runs):
  - `/net/projects2/annawoodard/experiments/mamba_36spf_sweepB_early_ei_archfix_lr5em4_minlf6_wu5_ep160_archtune_dconv4_2n4g/mamba_36spf_sweepB_early_ei_archfix_lr5em4_minlf6_wu5_ep160_archtune_dconv4_2n4g_best_model.pth`
- EI schedule default: `warmup=35`, `duration_steps=80`, `weight=1500`.
- Rebin disabled in this stage (`rebin_loss.enable: false`).

### Config set (prepared, not yet submitted)
| run family | config path | hypothesis |
|---|---|---|
| no-distill control | `configs/deadline_20260213/m36_distill_buf_control_no_distill.yaml` | Baseline for buffer + gentle-EI without distillation. |
| distill temporal-diff w=0.10 | `configs/deadline_20260213/m36_distill_buf_tdiff_w010_ei.yaml` | Weak distill should improve stability with minimal bias. |
| distill temporal-diff w=0.25 | `configs/deadline_20260213/m36_distill_buf_tdiff_w025_ei.yaml` | Main candidate: balanced distill pressure + gentle EI. |
| distill temporal-diff w=0.50 | `configs/deadline_20260213/m36_distill_buf_tdiff_w050_ei.yaml` | Strong distill stress-test for faster convergence vs overconstraint risk. |
| distill absolute w=0.25 | `configs/deadline_20260213/m36_distill_buf_abs_w025_ei.yaml` | Tests whether direct frame alignment outperforms temporal-diff mode. |
| distill percent-enhancement w=0.25 | `configs/deadline_20260213/m36_distill_buf_penh_w025_ei.yaml` | Bias distill toward enhancement dynamics rather than absolute intensity. |
| distill temporal-diff w=0.25 + EI gate | `configs/deadline_20260213/m36_distill_buf_tdiff_w025_ei_gate.yaml` | Gate EI on `train_mc_loss` to reduce post-EI collapse. |
| distill temporal-diff w=0.25, EI off | `configs/deadline_20260213/m36_distill_buf_tdiff_w025_noei.yaml` | Isolate whether distillation alone can improve spatial/temporal metrics before EI. |

### Decision rule after first 3 eval points
- Keep top 2 runs by combined trend on:
  - DRO PSNR (up),
  - DRO SSIM (up),
  - DRO LPIPS (down),
  - DRO curve corr (non-degrading),
  - non-DRO raw dynamic DCE MAE/MSE (down if available).
- Kill runs showing monotonic collapse immediately after EI activation.

## 2026-02-13 Update: Supervised GRASP Distill Implemented (No Checkpoint Distill)

### What changed in code
- Added supervised GRASP distillation path in training:
  - New loss block: `model.losses.supervised_distill` (MAE/MSE + temporal modes + schedule).
  - Distill target comes from precomputed GRASP recon files per slice (not teacher model forward).
  - Distill computed only on samples with available GRASP target (`grasp_target_valid` mask).
- Added training-time logging for supervised distill:
  - `Loss/Supervised_Distill_Weight`
  - `Loss/Train_Supervised_Distill`
  - `Loss/Train_Weighted_Supervised_Distill`
  - `Loss/Supervised_Distill_Valid_Fraction`
- Added dataset-side supervised target indexing in `ZFSliceDataset`:
  - filters to exams that have matching GRASP targets for current SPF/timeframes.
  - supports GRASP target shape normalization to model output layout.
  - enforces one target slice per active exam per sampling window when enabled (`supervised_distill_force_target_slice=true`).

### Distill data availability note
- For train split at 36 SPF: GRASP targets currently available for `67/258` train exams.
- Distill-enabled runs therefore train on that matched subset only (explicit, intentional).

### 36-SPF supervised-distill matrix (prepared, not submitted yet)
- Config directory: `configs/deadline_20260213_supdist/`

| config | purpose |
|---|---|
| `m36_supdist_control_w000_noei.yaml` | Control with supervised distill pipeline enabled but zero distill weight. |
| `m36_supdist_tdiff_w020_noei.yaml` | Main run: temporal-diff supervised distill, no EI. |
| `m36_supdist_tdiff_w050_noei.yaml` | Stronger distill pressure ablation. |
| `m36_supdist_abs_w020_noei.yaml` | Distill temporal mode ablation: absolute. |
| `m36_supdist_penh_w020_noei.yaml` | Distill temporal mode ablation: percent enhancement. |
| `m36_supdist_tdiff_w020_ei.yaml` | Add EI on top of main distill setting (late gentle EI schedule). |
| `m36_supdist_tdiff_w020_eigate.yaml` | EI-gated variant to reduce EI-triggered collapse risk. |

### Shared matrix settings
- Shuffle buffer enabled:
  - `active_exams=48`
  - `slices_per_exam=4`
  - `replace_fraction=0.2`
  - `supervised_distill_force_target_slice=true`
- Rebin disabled for this stage.
- Start from best known 36-SPF checkpoint:
  - `/net/projects2/annawoodard/experiments/m36_dline_timeenc_mixmc005_ctrl_2n4g/m36_dline_timeenc_mixmc005_ctrl_2n4g_best_model.pth`
- Configs were normalized to iteration-native keys used by current trainer:
  - `training.max_steps`
  - `data.eval_every_steps`
  - `training.save_every_steps`
  - `training.plot_every_steps`
  - `training.lr_schedule.warmup_max_steps`
- Smoke checks completed:
  - Python syntax check passed for `dataloader.py` and `train_zf.py`.
  - Dataset-level supervised-distill sample load in `brisknet` env returned expected tuple:
    `(kspace, csmap, N_samples, spf, N_time, grasp_target, grasp_target_valid)`.
  - Full-train split coverage check at 36spf found `67` matched exams with GRASP targets, valid for `active_exams=48`.

## 2026-02-13 Submission: 8-Run Supervised Distill Matrix (4x2-node + 4x3-node)

### Added config
- Added 8th ablation:
  - `configs/deadline_20260213_supdist/m36_supdist_tdiff_w010_noei.yaml`
  - change vs main: `model.losses.supervised_distill.weight: 0.10`

### Submission command baseline
- Environment: `brisknet`
- Submission script: `submit.py`
- Common resources:
  - `gpus-per-node=4`
  - `cpus-per-task=8`
  - `partition=general`
  - `timeout-min=720`
  - `requeue=true`

### 2-node jobs (4)
| job id | exp/job/output name | config |
|---|---|---|
| `711747` | `m36_supdist_control_w000_noei_2n4g` | `configs/deadline_20260213_supdist/m36_supdist_control_w000_noei.yaml` |
| `711748` | `m36_supdist_tdiff_w020_noei_2n4g` | `configs/deadline_20260213_supdist/m36_supdist_tdiff_w020_noei.yaml` |
| `711749` | `m36_supdist_abs_w020_noei_2n4g` | `configs/deadline_20260213_supdist/m36_supdist_abs_w020_noei.yaml` |
| `711750` | `m36_supdist_penh_w020_noei_2n4g` | `configs/deadline_20260213_supdist/m36_supdist_penh_w020_noei.yaml` |

### 3-node jobs (4)
| job id | exp/job/output name | config |
|---|---|---|
| `711751` | `m36_supdist_tdiff_w010_noei_3n4g` | `configs/deadline_20260213_supdist/m36_supdist_tdiff_w010_noei.yaml` |
| `711752` | `m36_supdist_tdiff_w050_noei_3n4g` | `configs/deadline_20260213_supdist/m36_supdist_tdiff_w050_noei.yaml` |
| `711753` | `m36_supdist_tdiff_w020_ei_3n4g` | `configs/deadline_20260213_supdist/m36_supdist_tdiff_w020_ei.yaml` |
| `711754` | `m36_supdist_tdiff_w020_eigate_3n4g` | `configs/deadline_20260213_supdist/m36_supdist_tdiff_w020_eigate.yaml` |

### Queue state at submission
- All 8 are currently `PENDING (Resources)`.

## 2026-02-13 Crash Follow-up: Supervised Distill NaN Targets

### Failure observed
- Initial supervised-distill launches failed early (example: `711748`) with:
  - `RuntimeError: total_loss is NaN`
  - failure occurred at step 1 immediately after supervised distill became active.
- Control run with `supervised_distill.weight=0` (`711747`) completed, indicating failure was in distill target path, not MC baseline.

### Root cause
- Some precomputed GRASP target files used for supervised distill contain non-finite values (all-NaN arrays).
- These non-finite targets propagated into supervised distill loss and made `total_loss` NaN.

### Fix applied
- Updated `dataloader.py`:
  - During supervised-distill index build, load each candidate GRASP target and discard any file with non-finite values.
  - Added load-time finite guard in `_load_supervised_distill_target` returning `grasp_target_valid=False` if a target is non-finite.
- Validation after patch (36spf train split):
  - `invalid_target_files=7`
  - `dropped_exams_no_finite_targets=7`
  - matched exams reduced from `67` to `60`
  - indexed targets now verified finite (`indexed_nonfinite_files=0`).

### Resubmitted jobs (post-fix)
| old failed job | new job | exp/job/output name |
|---|---:|---|
| `711748` | `711822` | `m36_supdist_tdiff_w020_noei_2n4g` |
| `711749` | `711823` | `m36_supdist_abs_w020_noei_2n4g` |
| `711750` | `711824` | `m36_supdist_penh_w020_noei_2n4g` |
| `711751` | `711825` | `m36_supdist_tdiff_w010_noei_3n4g` |
| `711752` | `711826` | `m36_supdist_tdiff_w050_noei_3n4g` |
| `711753` | `711827` | `m36_supdist_tdiff_w020_ei_3n4g` |
| `711754` | `711828` | `m36_supdist_tdiff_w020_eigate_3n4g` |

### Current status
- Resubmitted jobs are queued (`PENDING (Resources)`).

## 2026-02-13 Completed Results: 711822 and 711823

### Runs
- `711822` / `m36_supdist_tdiff_w020_noei_2n4g` (COMPLETED)
- `711823` / `m36_supdist_abs_w020_noei_2n4g` (COMPLETED)

### Final metrics (step 120)

| run | PSNR | SSIM | LPIPS | Curve Corr | DC MAE | DC MSE |
|---|---:|---:|---:|---:|---:|---:|
| `m36_supdist_tdiff_w020_noei_2n4g` | `41.5128` | `0.9165` | `0.03079` | `0.998317` | `0.07214` | `0.01344` |
| `m36_supdist_abs_w020_noei_2n4g` | `41.6456` | `0.9365` | `0.03098` | `0.998357` | `0.07224` | `0.01340` |
| GRASP baseline (same eval set) | `48.5799` | `0.9795` | `0.0021` | `0.999326` | `0.05679` | `0.01027` |

### Best points during run

| run | best PSNR (step) | best SSIM (step) | best LPIPS (step) | best Corr (step) |
|---|---:|---:|---:|---:|
| `m36_supdist_tdiff_w020_noei_2n4g` | `41.6889` (40) | `0.9400` (40) | `0.02992` (55) | `0.998451` (20) |
| `m36_supdist_abs_w020_noei_2n4g` | `41.6695` (115) | `0.9397` (35) | `0.03018` (40) | `0.998447` (20) |

### Distillation activation diagnostics
- `supervised_distill.weight` schedule ramped to `0.2` by step `20`, stayed until step `45`, then set to `0` after `stop_step`.
- On rank-0 logs, `supervised_distill_valid_fraction` was non-zero on only `7/120` steps, and supervised-distill train loss was tiny (`~1e-5`) when non-zero.
- Practical read: these two runs behaved mostly like strong MC fine-tuning with only sparse/weak supervised-distill pressure.

### What we learned
1. `absolute` target mode slightly outperformed `temporal_difference` at 36spf in this pair (higher final PSNR/SSIM, similar temporal corr).
2. Even with low effective distill pressure, both runs reached stable high-quality recon in low-40s PSNR and ~0.94 peak SSIM, so the training stack is stable post-NaN fix.
3. Major gap to GRASP remains large (about `-6.9 dB` PSNR and LPIPS much worse), so this is still not enough for paper target.
4. Current supervised-distill signal is too sparse/weak to be a strong driver; next experiments should increase effective target-hit rate or remove the early `stop_step=45` constraint if stable.

### Logging note
- `eval_results/eval_metrics.csv` currently has a column/header mismatch (values present but shifted labels); checkpoint curves/log prints were used as source of truth for this summary.

### Timing interpretation (why these finished fast)
- `711822` runtime (attempt wall): `3922.6s` (`65.4 min`), mean train iteration time from log: `14.23s`.
- `711823` runtime (attempt wall): `2206.8s` (`36.8 min`), mean train iteration time from log: `4.72s`.
- Major source of runtime difference between the two completed runs is node/GPU mix:
  - `711822` ran on `g006,g007` (`A40` nodes),
  - `711823` ran on `j002-ds,k003` (`A100 + L40S` nodes).
- Compared with prior 36spf control family (`m36_dline_timeenc_mixmc005_ctrl_2n4g`), eval inference time/sample is similar order (`~1.85-1.96s` there vs `~1.79s` in `711823`), so the apparent speedup is mostly training runtime placement + short step budget.

### Are these fully trained?
- They are fully trained only with respect to configured budget (`max_steps=120` reached).
- They are not “fully converged” toward GRASP target:
  - substantial metric gap remains,
  - `abs` run still improved PSNR across the last eval window (`+0.136` over last 5 eval points),
  - supervised distill signal was sparse and ended early (`stop_step=45`), so these runs mostly behaved like MC-focused continuation.

## 2026-02-14 New Matrix: Long 36spf Supervised-Distill Push

### Queue reset
- Canceled all active/pending jobs before relaunching matrix:
  - `711824` (`m36_supdist_penh_w020_noei_2n4g`)
  - `711827` (`m36_supdist_tdiff_w020_ei_3n4g`)
  - `711828` (`m36_supdist_tdiff_w020_eigate_3n4g`)
  - `711885` (`burst-sleep`)

### Goal
- Convert the short successful pilot (`m36_supdist_abs_w020_noei_2n4g`) into a long-run matrix with stronger, persistent supervised distill and faster effective optimization throughput.

### Common changes across all new configs
- Base config: `configs/deadline_20260213_supdist/m36_supdist_abs_w020_noei.yaml`
- New config folder: `configs/deadline_20260214_supdist_long/`
- Warm-start checkpoint:
  - `/net/projects2/annawoodard/experiments/m36_supdist_abs_w020_noei_2n4g/m36_supdist_abs_w020_noei_2n4g_best_model.pth`
- Common training/data edits:
  - `training.max_steps: 900` (was 120)
  - `data.eval_every_steps: 10` (was 5)
  - `training.save_every_steps: 10` (was 5)
  - `training.plot_every_steps: 40` (was 20)
  - `training.lr_schedule.min_lr_factor: 0.1` (was 0.5)
  - `dataloader.slice_sampling_mode: uniform`
  - `dataloader.slice_sampling_uniform_fraction: 1.0`
  - `dataloader.shuffle_buffer_active_exams: 96`
  - `dataloader.shuffle_buffer_slices_per_exam: 1`
  - `dataloader.shuffle_buffer_replace_fraction: 0.25`
  - `model.losses.supervised_distill.duration_steps: 240`
  - `model.losses.supervised_distill.stop_step: null` (keep distill active)

### Submitted jobs
- Submission resources: `2 nodes x 4 gpus/node`, `partition=general`, `timeout=700`, env=`brisknet`.

| job id | exp/job/output name | config | key variant |
|---|---|---|---|
| `712023` | `m36_supdistlong_abs_w020_mix005_2n4g` | `configs/deadline_20260214_supdist_long/m36_supdistlong_abs_w020_mix005_2n4g.yaml` | absolute distill `w=0.20`, MC `MIXED(0.05)` |
| `712024` | `m36_supdistlong_abs_w035_mix005_2n4g` | `configs/deadline_20260214_supdist_long/m36_supdistlong_abs_w035_mix005_2n4g.yaml` | absolute distill `w=0.35`, MC `MIXED(0.05)` |
| `712025` | `m36_supdistlong_abs_w050_mix005_2n4g` | `configs/deadline_20260214_supdist_long/m36_supdistlong_abs_w050_mix005_2n4g.yaml` | absolute distill `w=0.50`, MC `MIXED(0.05)` |
| `712026` | `m36_supdistlong_abs_w035_mix010_2n4g` | `configs/deadline_20260214_supdist_long/m36_supdistlong_abs_w035_mix010_2n4g.yaml` | absolute distill `w=0.35`, MC `MIXED(0.10)` |
| `712027` | `m36_supdistlong_abs_w035_mse_2n4g` | `configs/deadline_20260214_supdist_long/m36_supdistlong_abs_w035_mse_2n4g.yaml` | absolute distill `w=0.35`, MC `MSE` |
| `712028` | `m36_supdistlong_penh_w035_mix005_2n4g` | `configs/deadline_20260214_supdist_long/m36_supdistlong_penh_w035_mix005_2n4g.yaml` | percent-enhancement distill `w=0.35`, MC `MIXED(0.05)` |
| `712029` | `m36_supdistlong_abs_w035_lateei_2n4g` | `configs/deadline_20260214_supdist_long/m36_supdistlong_abs_w035_lateei_2n4g.yaml` | absolute distill `w=0.35` + late EI (`warmup=300`, `duration=120`, `w=600`) |

### Immediate hypotheses
1. Keeping supervised-distill active for the full run (no early stop) plus target-slice-heavy sampling should materially increase effective distill signal and reduce GRASP gap.
2. Increasing distill weight to `0.35-0.50` should improve DRO perceptual/structural metrics faster than prior `w=0.20` short run.
3. Adding more MC MSE pressure (`mix010` / pure `MSE`) should preferentially improve PSNR while potentially trading some SSIM/LPIPS.
4. Percent-enhancement target mode may improve temporal fidelity metrics even if raw PSNR is slightly lower.
5. Late EI variant tests whether adding invariance after anatomy stabilizes helps temporal metrics without early destabilization.

## 2026-02-13 Matrix Failure + Immediate Resubmit

### Why queue went empty
- The 7 long-run jobs (`712023`-`712029`) all failed at startup (within ~20s), not completed training.
- Root cause in every run:
  - `ValueError: shuffle_buffer_active_exams=96 exceeds available exams=60`
- `available exams=60` comes from supervised-distill filtering after invalid GRASP-target removal.

### Applied fix and schedule correction
- Updated all `configs/deadline_20260214_supdist_long/*.yaml`:
  - `dataloader.shuffle_buffer_active_exams: 48` (from 96)
  - `data.eval_every_steps: 100` (from 10)
  - `training.save_every_steps: 100` (from 10)
  - `training.plot_every_steps: 200` (from 40)
  - `model.losses.supervised_distill.duration_steps: 80` (from 240)
  - `model.losses.supervised_distill.stop_step: 220` (from null)

### Resubmitted jobs
- New job IDs:
  - `712044` `m36_supdistlong_abs_w020_mix005_2n4g`
  - `712045` `m36_supdistlong_abs_w035_mix005_2n4g`
  - `712046` `m36_supdistlong_abs_w050_mix005_2n4g`
  - `712047` `m36_supdistlong_abs_w035_mix010_2n4g`
  - `712049` `m36_supdistlong_abs_w035_mse_2n4g`
  - `712050` `m36_supdistlong_penh_w035_mix005_2n4g`
  - `712051` `m36_supdistlong_abs_w035_lateei_2n4g`

## 2026-02-13 Clarification: Why only 60 supervised-distill exams

### Direct count (matches dataloader behavior)
- Train split size in `data/data_split.json`: `258` IDs.
- Files selected by dataloader substring match in `zf_kspace/*.h5`: `258` exams.
- Exams with matching GRASP target pattern for distill (`grasp_recon_36spf_8frames_slice*.npy`): `67`.
- Exams with at least one finite target: `60`.
- Invalid/non-finite target files: `7`.

### Interpretation
- The `60` is not caused by enhancement-weighted slice sampling.
- It is caused by supervised-distill target availability + finite-value filtering.
- With `supervised_distill.enable=true`, current dataloader intentionally restricts trainable exams to those with valid GRASP targets.

## 2026-02-13 Resampling policy update per request

### Requested change
- Use uniform sampling (no enhancement weighting) with `4 slices/exam`.

### Applied to all `configs/deadline_20260214_supdist_long/*.yaml`
- `dataloader.slice_sampling_mode: uniform`
- `dataloader.slice_sampling_uniform_fraction: 1.0`
- `dataloader.shuffle_buffer_active_exams: 60`
- `dataloader.shuffle_buffer_slices_per_exam: 4`
- `dataloader.shuffle_buffer_replace_fraction: 0.0`

### Eval cadence correction for step-native loop
- `data.eval_every_steps: 200`
- `training.save_every_steps: 200`
- `training.plot_every_steps: 400`

### Re-submitted jobs after this change
- `712053` `m36_supdistlong_abs_w020_mix005_2n4g`
- `712054` `m36_supdistlong_abs_w035_mix005_2n4g`
- `712055` `m36_supdistlong_abs_w050_mix005_2n4g`
- `712056` `m36_supdistlong_abs_w035_mix010_2n4g`
- `712057` `m36_supdistlong_abs_w035_mse_2n4g`
- `712058` `m36_supdistlong_penh_w035_mix005_2n4g`
- `712059` `m36_supdistlong_abs_w035_lateei_2n4g`

## 2026-02-13 Distill-on-all-exams fix + relaunch

### Issue discovered
- Even after switching to uniform sampling, supervised distill still reduced training population because `dataloader.py` filtered `self.file_list` down to exams with targets.
- This conflicted with desired behavior (train on all exams, distill only where target exists).

### Code fix applied
- `dataloader.py`:
  - `_build_supervised_distill_index` no longer overwrites `self.file_list`.
  - Keeps full train file list and builds target index only for exams that have finite GRASP targets.
  - Logs `target_coverage={matched}/{total}`.
  - `_resolve_distill_slice_pool` no longer raises when an exam has no target slices; returns full slice range.
- Verified by local dataset init:
  - `file_list_len=258`, `target_index_len=60`, `slice_index_map_len=240`.

### Clarified target coverage
- Train split (`258` exams): `60` with finite 36spf/8frame GRASP targets.
- Val split (`15` exams): `15` with finite targets.
- Across all 300 h5 exams: `91` have finite targets.

### Config updates for current matrix
- All `configs/deadline_20260214_supdist_long/*.yaml` now use:
  - uniform sampling, `4 slices/exam`
  - `shuffle_buffer_active_exams=60`
  - `shuffle_buffer_replace_fraction=0.2`
  - `eval_every_steps=200`, `save_every_steps=200`, `plot_every_steps=400`
  - supervised distill `stop_step=765` (`85%` of `max_steps=900`)

### Fresh submissions after fixes
- `712062` `m36_supdistlong_abs_w020_mix005_2n4g`
- `712063` `m36_supdistlong_abs_w035_mix005_2n4g`
- `712064` `m36_supdistlong_abs_w050_mix005_2n4g`
- `712065` `m36_supdistlong_abs_w035_mix010_2n4g`
- `712066` `m36_supdistlong_abs_w035_mse_2n4g`
- `712067` `m36_supdistlong_penh_w035_mix005_2n4g`
- `712068` `m36_supdistlong_abs_w035_lateei_2n4g`

## 2026-02-14 GRASP target coverage + regeneration path

### Coverage check (fastMRI split-aware)
- `data_split.json` sizes:
  - train: `258`
  - val: `15`
  - val_dro: `15`
  - test_dro: `25`
- Existing GRASP 36spf/8f files under `/net/scratch2/rachelgordon/zf_data_192_slices`:
  - all exams with files: `98`
  - all exams with finite files: `91`
- Matched by fastMRI split IDs (substring match to `_2` names):
  - train: `67` with files, `60` finite
  - val: `15` with files, `15` finite
  - val_dro / test_dro: `0` (expected; separate DRO dataset)

### New script for safe full-target generation
- Added `generate_grasp_targets_split.py`.
- Key properties:
  - split-scoped processing via `--split-file` + `--splits` (prevents accidental train/test mixing),
  - writes target filenames exactly as expected by dataloader,
  - uses precomputed ESPIRiT cs-maps (`--csmaps-dir`) for consistency.

### Note
- On local `brisknet` env, `sigpy.mri.app.HighDimensionalRecon` is unavailable, so this script requires the project SigPy build used for GRASP reconstruction jobs.

## 2026-02-13 GRASP train-target processing launch (annawoodard scratch)

### Goal
- Generate missing fastMRI train-split GRASP targets for supervised distillation using:
  - `spf=36` (`8 frames`)
  - `spf=2` (`144 frames`)
- Save outputs under:
  - `/net/scratch2/annawoodard/grasp_targets_fastmri_train`
- Save dtype:
  - `complex64`

### Implementation updates
- Added `generate_grasp_targets_split.py` options:
  - `--save-dtype {complex64,complex128}`
  - `--num-shards`, `--shard-index` for array parallelism
  - `--recon-max-iter`
- Added `install_sigpy_highdim_shim.py` to install a `HighDimensionalRecon` compatibility shim into `brisknet`.
- Added Slurm array recipe:
  - `grasp_recon_train_split.sbatch`
- Updated helper submission script:
  - `grasp_recon.sh` now submits split-aware train processing jobs.

### Submitted jobs
- `712095` (`grasp_train_spf36`, array shards `0-3`)
- `712096` (`grasp_train_spf2`, array shards `0-3`)

### Notes
- The first `spf=2` shard runs immediately (`712096_0`); remaining shards queue behind user slot limits.
- No output files had landed yet at initial check (jobs were still in first recons).

## 2026-02-14 m36_supdist Overlay Readout (why it looks ceilinged)

### Scope analyzed
- Overlay tables:
  - `/net/projects2/annawoodard/experiments/m36_supdist_overlay/overlay_metric_tables.txt`
- Run family:
  - `/net/projects2/annawoodard/experiments/m36_s*`
- Cross-checked with checkpoint curves and submitit logs for distill activity.

### What the sweep is actually showing
- Short runs (many eval points) all cluster tightly in DRO image metrics:
  - best PSNR about `41.48-41.69`
  - best SSIM about `0.92-0.94`
  - best LPIPS about `0.029-0.031`
- Non-DRO k-space metrics are nearly invariant across runs:
  - `eval_raw_dc_mae` span: `~1.84e-07`
  - `eval_raw_dc_mse` span: `~2.15e-11`
  - relative-L2 low/mid/high are almost flat run-to-run.
- Relative to control (`m36_supdist_control_w000_noei_2n4g`), distill variants improved PSNR by only about `+0.19` to `+0.22 dB` at best.
- GRASP gap remains very large:
  - best DL PSNR `~41.7` vs GRASP baseline `~48.58`
  - best DL SSIM `~0.95` vs GRASP `~0.979`
  - best DL LPIPS `~0.029-0.032` vs GRASP `~0.0021`
  - temporal gaps are still substantial (`curve_mae`, `early_corr`, `early_mae`).

### Distillation did run, but sparsely
- Distill config was enabled in these runs (`model.losses.supervised_distill.enable: true`) except control weight `0.0`.
- However, effective supervised-distill signal was sparse in short runs:
  - many steps logged `Training Supervised Distill Loss: 0.000000 (valid_fraction=0.000)`
  - only a small subset of steps had `valid_fraction=1.000`.
- Checkpoints confirm this sparsity:
  - typical nonzero distill-valid steps: `~6-7` total in early runs.
  - with `stop_step=45`, distill was turned off for the majority of training.
- Also important: training loop is step-native with one train batch per step (`Step x/y Training: 0/1` in logs + explicit `break` after one batch), so when a step misses a target slice, distill contributes nothing that step.

### Why the "hard ceiling" happened
1. Distill supervision density was too low:
  - one batch per step + sparse target hits + early distill stop.
2. Objective remains strongly MC-dominated:
  - when distill is absent on a step, optimization follows MC-only behavior.
3. Many long-run variants currently have only one eval point in overlay (`step200`), so they are not informative about trajectory yet; they only show current endpoint clustering, not full convergence behavior.

### Interpretation: is there still hope for GRASP-supervised distill?
- Yes, still plausible, but the current runs are not a strong test of the hypothesis because distill was under-applied in practice.
- These results are evidence against "small, sparse, early-stopped distill" as currently configured, not evidence against distillation in principle.
- Practical implication: if we want to judge distillation fairly, we need much higher per-step distill hit-rate and a longer active window before concluding it cannot help.

## 2026-02-14 36spf strong-distill dense matrix (new sweep)

### Cleanup to free slots
- Canceled old low-value supdist-long family jobs:
  - `712062` `m36_supdistlong_abs_w020_mix005_2n4g`
  - `712063` `m36_supdistlong_abs_w035_mix005_2n4g`
  - `712064` `m36_supdistlong_abs_w050_mix005_2n4g`
  - `712065` `m36_supdistlong_abs_w035_mix010_2n4g`
  - `712066` `m36_supdistlong_abs_w035_mse_2n4g`
  - `712067` `m36_supdistlong_penh_w035_mix005_2n4g`
  - `712068` `m36_supdistlong_abs_w035_lateei_2n4g`

### New configs
- Added under `configs/deadline_20260214_supdist_dense/`:
  - `m36_supdistdense_tdiff_w080_lr2em4_s1_2n4g.yaml`
  - `m36_supdistdense_tdiff_w080_lr3em4_s1_2n4g.yaml`
  - `m36_supdistdense_abs_w060_mix010_lr2em4_s1_2n4g.yaml`

### Common matrix intent
- Keep 36spf fixed and push stronger, denser supervised distill:
  - `supervised_distill.enable: true`
  - `warmup: 0`
  - `duration_steps: 80`
  - `stop_step: null`
- Increase effective target-hit frequency:
  - `dataloader.shuffle_buffer_active_exams: 60`
  - `dataloader.shuffle_buffer_slices_per_exam: 1`
  - `dataloader.shuffle_buffer_replace_fraction: 0.1`
  - `dataloader.supervised_distill_force_target_slice: true`
- Temporary missing-target tolerance while GRASP regen is in progress:
  - `dataloader.supervised_distill_require_any_target: false`
- Runtime for faster feedback:
  - `training.max_steps: 360`
  - `data.eval_every_steps: 20`
  - `training.save_every_steps: 20`
  - `training.plot_every_steps: 80`
  - `training.lr_schedule.warmup_max_steps: 5`
  - `training.lr_schedule.min_lr_factor: 0.2`

### Submitted jobs (2 nodes x 4 GPUs, general, 700 min)
- `712165` `m36_supdistdense_tdiff_w080_lr2em4_s1_2n4g`
- `712166` `m36_supdistdense_tdiff_w080_lr3em4_s1_2n4g`
- `712167` `m36_supdistdense_abs_w060_mix010_lr2em4_s1_2n4g`

### Variant deltas
- `712165`: temporal-difference distill, weight `0.8`, lr `2e-4`.
- `712166`: temporal-difference distill, weight `0.8`, lr `3e-4`.
- `712167`: absolute distill, weight `0.6`, lr `2e-4`, MC `MIXED` with `mse_weight=0.1`.

## 2026-02-14 Mid-run check: strong-distill matrix + GRASP generation

### m36 strong-distill status
- `712165` (`m36_supdistdense_tdiff_w080_lr2em4_s1_2n4g`): `COMPLETED`.
  - Best PSNR: `42.3359` at step `340`.
  - Final (step `360`): SSIM `0.9591`, PSNR `42.3209`, LPIPS `0.0271`.
- `712166` (`m36_supdistdense_tdiff_w080_lr3em4_s1_2n4g`): `RUNNING` (latest logged step around `150/360` at check time).
  - Current best so far: step `20` PSNR `41.0892`.
- `712167` (`m36_supdistdense_abs_w060_mix010_lr2em4_s1_2n4g`): `PENDING` (`QOSMaxJobsPerUserLimit`).

### Distillation pressure evidence
- Distill is active but intermittent due target-availability at sampled step:
  - `712165`: logged distill points `18`, nonzero-valid points `7` (`mean valid_fraction=0.389`).
  - `712166` (so far): logged points `7`, nonzero-valid points `3` (`mean valid_fraction=0.429`).
- When valid, weighted distill contribution is material relative to MC:
  - observed `(distill_loss * weight) / mc_loss` about `0.13` to `0.88` on nonzero-valid checkpoints.

### GRASP generation status (train split campaign)
- Active jobs:
  - 36spf shards: `712155_0`, `712155_1`, `712155_2` running.
  - 2spf shards: `712156_0`, `712156_1`, `712156_2`, `712156_3` running.
- Failed shard:
  - `712155_3` failed with nonzero exit.
  - Root cause: cs-map dimensionality mismatch in one slice:
    - `/net/scratch2/rachelgordon/zf_data_192_slices/cs_maps/fastMRI_breast_067_2_cs_maps/cs_map_slice_000.npy`
    - shape `(1, 1, 16, 320, 320)` while generator currently expects ndim=4 `[C,1,H,W]`.

### GRASP output coverage snapshot
- Target root: `/net/scratch2/annawoodard/grasp_targets_fastmri_train`
- Train expected files per SPF: `258 exams * 192 slices = 49,536`.
- Current counts:
  - 36spf files: `7,620` (`15.4%`).
  - 2spf files: `986` (`2.0%`).

### Decision / next action
- Missing outputs are not auto-backfilled after shard failure; once running shards finish, explicit catch-up submission is required.
- For 36spf shard-3 recovery, generator should be patched to normalize known cs-map shape variants before requeueing failed shard.

### Follow-up diagnosis: why only one shard failed so far
- Confirmed failing exam assignment:
  - `fastMRI_breast_067` index in train split = `47`, so with `idx % 4` sharding it belongs to shard `3` only.
- Additional finding:
  - The malformed cs-map pattern is not unique to this patient.
  - Train split scan of `cs_map_slice_000.npy` found `26/258` exams with shape `(1, 1, 16, 320, 320)` instead of expected 4D `[C,1,H,W]`.
  - These problematic exams are distributed across all shards (`idx % 4`), so other shards are also at risk once they hit affected exams.
- Implication:
  - Resubmitting only shard `712155_3` without a loader normalization fix will likely fail again.
  - Robust fix should accept both shapes by squeezing/reshaping singleton leading dimension before validating coil/time axes.

### Fix applied: cs-map shape normalization + targeted shard retry
- Updated `generate_grasp_targets_split.py` `_load_espirit_csmap` to normalize known cs-map variants into canonical `[C,1,H,W]`:
  - accepted as-is: `[C,1,H,W]`
  - normalized: `[1,1,C,H,W] -> [C,1,H,W]`
  - normalized: `[1,C,1,H,W] -> [C,1,H,W]`
  - normalized: `[1,C,H,W] -> [C,1,H,W]`
- Still fail-fast for unknown layouts:
  - unsupported 4D/5D shapes continue to raise explicit `ValueError`.
- Local sanity check passed on problematic and normal files:
  - `fastMRI_breast_067_2` slice `0` now resolves to `(16,1,320,320)`.
  - `fastMRI_breast_067_2` slice `1` remains `(16,1,320,320)`.
- Resubmitted only failed 36spf shard:
  - `712318_3` (`grasp_train_spf36_retry`) is now running.
- Duplicate-work behavior confirmed unchanged:
  - generator skips existing per-slice outputs (`if out_path.exists() and not overwrite: continue`),
  - only `pending_slices` are reconstructed, so retry jobs backfill missing outputs rather than recomputing finished ones.

## 2026-02-14 GRASP coverage snapshot + additional retry

### Current coverage snapshot (train split)
- Expected files per SPF: `49,536` (`258 exams * 192 slices`).
- Current totals:
  - 36spf: `8,175 / 49,536` (`16.50%`)
  - 2spf: `1,118 / 49,536` (`2.26%`)
- Per-shard coverage:
  - 36spf:
    - shard0: `2,064 / 12,480` (`16.54%`)
    - shard1: `2,078 / 12,480` (`16.65%`)
    - shard2: `1,920 / 12,288` (`15.62%`)
    - shard3: `2,113 / 12,288` (`17.20%`)
  - 2spf:
    - shard0: `505 / 12,480` (`4.05%`)
    - shard1: `192 / 12,480` (`1.54%`)
    - shard2: `210 / 12,288` (`1.71%`)
    - shard3: `211 / 12,288` (`1.72%`)

### Job status and proactive recovery
- Running 36spf original shards: `712155_0`, `712155_1`.
- Previously failed 36spf shards: `712155_2`, `712155_3` (same pre-patch cs-map issue).
- Patched retries:
  - `712318_3` running.
  - `712322_2` submitted and pending (QoS slot limit).
- Running 2spf shards: `712156_0`, `712156_1`, `712156_2`, `712156_3`.

## 2026-02-14 GRASP crash triage (recent submissions)

### Queue outcome at triage time
- No GRASP jobs were running.
- Recent GRASP jobs had exited with two separate failure modes:
  - immediate launch failures in recent 2-node jobs (`712657`, `712658`, `712659`, `712660`),
  - runtime failures in patched retry shards (`712318_3`, `712322_2`).

### Failure mode A: immediate launch failure (`712657`-`712660`)
- Error signature in logs:
  - `/usr/bin/bash: line 5: ENV_PYTHON: unbound variable`
- Root cause:
  - `grasp_recon_train_split.sbatch` runs worker commands via `srun ... bash -lc '...'`.
  - The inner shell depends on variables like `ENV_PYTHON`, but these were shell-local and not exported.
  - Under `set -u`, worker shell aborted immediately.
- Fix applied:
  - Updated `grasp_recon_train_split.sbatch` to export all inner-shell variables before `srun`:
    - `ENV_PYTHON`, `DATA_DIR`, `TARGET_ROOT`, `CSMAPS_DIR`, `SPLIT_FILE`, `SPF`,
      `RECON_MAX_ITER`, `NUM_SHARDS`, `SHARD_OFFSET`, `SLICE_PRIORITY_ORDER`,
      `PRIORITY_SLICES_PER_EXAM`, `ENV_NAME`.

### Failure mode B: runtime numerical failure in retry shards
- `712318_3` and `712322_2` failed with:
  - `RuntimeError: Non-finite recon values encountered.`
- This indicates we moved past the earlier cs-map shape mismatch and now hit a numerical stability issue in GRASP reconstruction for at least one slice.

### Historical context for older runs
- Earlier 36spf array jobs (`712155_0..3`) failed on pre-fix cs-map shape mismatch (`ndim=5` at `slice_000`).
- Earlier 2spf array jobs (`712156_0..3`) ended by walltime (`TIMEOUT`), not code crash.

### Decision / next action
- Next restart should use the exported-env fix (now in sbatch).
- Before large resubmission, run one short shard debug with enhanced per-slice failure context to identify the exact patient/slice causing non-finite recon and decide whether to:
  - hard-fail and prefilter those slices, or
  - allow skip-with-log for known pathological slices.

## 2026-02-14 Relaunch execution (post-triage)

### Script fixes in place
- `grasp_recon_train_split.sbatch` updated to export worker-needed env vars before `srun`.

### Submission actions taken
- Submitted targeted non-finite debug shard:
  - `712676` `grasp_nf_debug_s2` (1 node x 1 GPU, shard `2/4`, `spf=36`, `max_exams=12`).
- Initially submitted unintended default-size GRASP jobs (1n1g):
  - `712675` `grasp_train_spf36_1n1g_j0`
  - `712677` `grasp_train_spf2_1n1g_j0`
- Immediately canceled unintended jobs and relaunched intended campaign size:
  - `712678` `grasp_train_spf36_2n4g_j0`
  - `712679` `grasp_train_spf36_2n4g_j1`
  - `712680` `grasp_train_spf2_2n4g_j0`
  - `712681` `grasp_train_spf2_2n4g_j1`

### Queue snapshot after relaunch
- `712676` running.
- `712678`, `712679`, `712680`, `712681` pending (`Priority`).

### Coverage snapshot at relaunch check
- 36spf files in target root: `8788`.

## 2026-02-15 Dense 36spf results (final) + plateau readout

### Completed jobs
- `712165` `m36_supdistdense_tdiff_w080_lr2em4_s1_2n4g` (`COMPLETED`)
- `712166` `m36_supdistdense_tdiff_w080_lr3em4_s1_2n4g` (`COMPLETED`)
- `712167` `m36_supdistdense_abs_w060_mix010_lr2em4_s1_2n4g` (`COMPLETED`)

### Best-per-run summary (from best checkpoints)
- `712166` (tdiff, w=0.8, lr=3e-4):
  - SSIM `0.9645`, PSNR `42.5134`, LPIPS `0.02542`, MSE `0.10041`
  - Temporal: `rho_early=0.6098`, `mae_early=0.9102`, `t_arr=6.1378`, `iAUC10=15.5322`, `t_peak=13.0442`
- `712165` (tdiff, w=0.8, lr=2e-4):
  - SSIM `0.9645`, PSNR `42.3359`, LPIPS `0.02753`, MSE `0.10440`
  - Temporal: `rho_early=0.5906`, `mae_early=0.9107`, `t_arr=6.0332` (best `t_arr`), `iAUC10=15.8840`, `t_peak=13.3436`
- `712167` (abs distill + mixed MC):
  - SSIM `0.9660` (best SSIM), PSNR `42.3453`, LPIPS `0.02583`, MSE `0.10421`
  - Temporal: `rho_early=0.5759`, `mae_early=0.9415`, `t_arr=6.1667`, `iAUC10=16.3789`, `t_peak=13.6891`

### Key takeaways
1. `tdiff` distillation remains the strongest direction:
  - best PSNR/LPIPS and best early-temporal fidelity are from `712166`.
2. `abs + mixed-MC` improved SSIM but hurt temporal metrics:
  - indicates optimization is drifting toward smooth spatial similarity while timing fidelity degrades.
3. Plateau/regression concern is real for temporal metrics:
  - image metrics (PSNR/LPIPS/SSIM) continue improving late,
  - but `rho_early` / `mae_early` / `t_arr` / `iAUC10` often peak earlier and then worsen.
4. Distill sparsity is still a bottleneck:
  - recent dense runs still showed supervised-distill valid fraction around `~0.39` (many zero-valid steps).
  - with step-native training (`1` batch/step), missing target hits directly reduce distill pressure.

### GRASP train-target generation status (latest)
- Queue status at check: no running jobs.
- Coverage under `/net/scratch2/annawoodard/grasp_targets_fastmri_train`:
  - expected per SPF: `49,536` files (`258` train exams x `192` slices)
  - 36spf: `8,788 / 49,536` (`17.74%`), exams touched: `47`
  - 2spf: `1,539 / 49,536` (`3.11%`), exams touched: `10`
- Recent failure modes:
  - 36spf retries failed on `Non-finite recon values encountered` (after cs-map shape issue was addressed).
  - 2spf shards primarily hit walltime timeout.

### New processing strategy implemented for diversity-first target generation
- Updated `generate_grasp_targets_split.py` to support two-phase slice scheduling:
  - phase 1: process `N` prioritized slices/exam first,
  - phase 2: fill remaining slices.
- Added priority order options:
  - `middle_first` (default): center-ish slices first,
  - `sequential`.
- Submission defaults updated in wrappers:
  - `SLICE_PRIORITY_ORDER=middle_first`
  - `PRIORITY_SLICES_PER_EXAM=24`
- Rationale:
  - if jobs stop early (timeout/failure), we still gain broad exam coverage rather than fully finishing only early exams in each shard.

### How this helps beat GRASP
1. Increase distill supervision density first:
  - middle-first `N` improves target diversity quickly across train exams, which should raise nonzero distill-valid step frequency.
2. Keep `tdiff` distill as primary loss shape:
  - best current direction for both image and early temporal metrics.
3. Shift model selection criteria:
  - do not select only by PSNR/SSIM; include early-timing metrics (`rho_early`, `mae_early`, `t_arr`, `iAUC10`) to avoid late-stage temporal collapse.
4. Once target coverage is substantially higher, run a two-stage schedule:
  - stage A: strong distill to pull close to GRASP manifold,
  - stage B: taper distill and emphasize task losses (MC + EI/rebin) to surpass teacher in final model behavior.

## 2026-02-15 GRASP generation relaunch (2 jobs per SPF, 2n4g each)

### Action
- Submitted GRASP train-target generation using the new multi-node/multi-worker sharding launcher with two disjoint jobs per SPF.
- Launch settings:
  - `JOBS_PER_SPF=2`
  - `NODES=2`
  - `GPUS_PER_NODE=4`
  - `PARTITION=general`
  - command: `JOBS_PER_SPF=2 NODES=2 GPUS_PER_NODE=4 PARTITION=general bash grasp_recon.sh 36 2`

### Submitted jobs
- `712657` `grasp_train_spf36_2n4g_j0`
- `712658` `grasp_train_spf36_2n4g_j1`
- `712659` `grasp_train_spf2_2n4g_j0`
- `712660` `grasp_train_spf2_2n4g_j1`

### Expected shard layout
- Per SPF: total shards = `NODES * GPUS_PER_NODE * JOBS_PER_SPF = 16`
- Job `j0` covers shard offset `[0..7]`
- Job `j1` covers shard offset `[8..15]`
- Non-overlapping shard assignment is enforced via `SHARD_OFFSET` + `SLURM_PROCID`.

### Status at submit time
- All four jobs entered queue in `PENDING (Resources)` state.

## 2026-02-15 Update: Temporal Distill LOO Matrix Submitted (Items 1/2/3/5)

### Goal
- Evaluate impact of new supervised-distill improvements for temporal fidelity at SPF=36:
  - multi-lag temporal-difference distill
  - arrival-focused temporal weighting
  - baseline-subtracted -> temporal-difference curriculum
  - temporal-frequency loss

### Distill data status used for this matrix
- `grasp_target_root`: `/net/scratch2/annawoodard/grasp_targets_fastmri_train`
- `dataloader.max_subjects: null` (no subject cap)
- Current target coverage in train split (SPF=36):
  - matched train exams: `258`
  - exams with at least one 36spf target: `47` (`18.2%`)
  - total available 36spf target slices on matched exams: `8788`
- Note: training still uses all exams; distill is applied only when `grasp_target_valid=true`.

### Shared run settings
- Nodes/GPUs: `2 nodes x 4 GPUs/node`
- Env: `brisknet`
- EI: off (`use_ei_loss: false`) for clean supervised-distill ablation
- Distill schedule:
  - `weight: 0.8`
  - `warmup: 0`
  - `duration_steps: 80`
  - `stop_step: 306` (~85% of 360-step run)

### Submitted runs
| ID | exp/job name | queue | hypothesis |
|---:|---|---|---|
| 712667 | `m36_supdist_loo1_full_2n4g` | general | Full stack (multi-lag + arrival + curriculum + freq) is strongest. |
| 712668 | `m36_supdist_loo2_nomultilag_2n4g` | general | Remove multi-lag to test whether long-lag diffs matter. |
| 712669 | `m36_supdist_loo3_noarrival_2n4g` | general | Remove arrival weighting to test arrival-window emphasis. |
| 712670 | `m36_supdist_loo4_nocurriculum_2n4g` | general | Remove baseline-subtracted curriculum and use fixed temporal-diff mode. |
| 712673 | `m36_supdist_loo5_nofreq_2n4g_burst` | burst | Remove frequency term to isolate its net value. |
| 712674 | `m36_supdist_loo6_freqw010_2n4g_burst` | burst | Frequency-term strength sensitivity (`weight=0.10` vs `0.05`). |

### Burst queue note
- Burst QoS max walltime is `04:00:00`; burst runs are submitted with `timeout=230 min`.

### 2026-02-15 resubmission fix
- Initial submission crashed immediately due config parse mismatch:
  - `dataloader.max_subjects: null` triggered `TypeError` in trainer (`if max_subjects < 300`).
- Fix applied:
  - set `dataloader.max_subjects: 999999` in all six LOO configs (effective no-cap under current code path).
- Resubmitted jobs:
  - `712682` `m36_supdist_loo1_full_2n4g`
  - `712683` `m36_supdist_loo2_nomultilag_2n4g`
  - `712684` `m36_supdist_loo3_noarrival_2n4g`
  - `712685` `m36_supdist_loo4_nocurriculum_2n4g`
  - `712686` `m36_supdist_loo5_nofreq_2n4g_burst`
  - `712687` `m36_supdist_loo6_freqw010_2n4g_burst`

### 2026-02-15 Failure analysis (LOO matrix)
- All six LOO submissions failed due cluster hardware, not model config.
- Root cause in submitit rank-0 stderr (example):
  - `/net/projects2/annawoodard/experiments/m36_supdist_loo1_full_2n4g/submitit_logs/712682_0_log.err`
  - error: `RuntimeError: CUDA error: uncorrectable ECC error encountered`
  - first observed failure: host `k003`, rank `1 (local_rank: 1)`.
- Same ECC fault repeated for:
  - `712682`, `712683`, `712684`, `712685`, `712686`, `712687`
- Action for next resubmit:
  - exclude bad node: `--exclude k003`
  - keep same configs/job names and queue split.

### 2026-02-15 Resubmission after ECC node failure
- Resubmitted the same 6 LOO runs with `--exclude k003` to avoid repeated ECC failures.
- New job IDs:
  - `712878` `m36_supdist_loo1_full_2n4g` (general)
  - `712879` `m36_supdist_loo2_nomultilag_2n4g` (general)
  - `712880` `m36_supdist_loo3_noarrival_2n4g` (general)
  - `712881` `m36_supdist_loo4_nocurriculum_2n4g` (general)
  - `712882` `m36_supdist_loo5_nofreq_2n4g_burst` (burst)
  - `712883` `m36_supdist_loo6_freqw010_2n4g_burst` (burst)

### 2026-02-15 GRASP layout change (metadata reduction)
- Changed GRASP target storage from per-slice `.npy` files to packed per-patient/per-SPF files:
  - new path: `<target_root>/<patient>/grasp_recon_<spf>spf_<frames>frames.h5`
  - datasets: `recon` `[num_slices, H, W, T]`, `valid` `[num_slices]`
- Generator now:
  - skips and logs per-slice failures instead of aborting shard
  - writes detailed failure JSONL and summary under `<target_root>/_grasp_failures/`
  - writes `<...>.skipped_slices.tsv` for triage
- Dataloader updated to index/load both packed H5 targets and legacy per-slice `.npy` targets.
- Pending jobs using patched generator:
  - `712896` (`grasp_train_spf2_2n4g_j1`)
  - `712897` (`grasp_train_spf2_2n4g_j0`)

### 2026-02-15 GRASP relaunch to packed root + migration job
- Canceled stale pending SPF=2 jobs:
  - `712896`, `712897`
- Fixed `generate_grasp_targets_split.py`:
  - repaired syntax/indentation regression in phase loop
  - made lock handling `try/finally` safe (always releases lock)
  - switched to `fcntl.flock` file locking to avoid stale lockfiles and multi-writer collisions
  - standardized recon layout to canonical `H,W,T` before writing packed H5
- Updated defaults to packed root:
  - `grasp_recon.sh` and `grasp_recon_train_split.sbatch` now default `TARGET_ROOT=/net/scratch2/annawoodard/grasp_targets_fastmri_train_packed`
- Submitted fresh generation jobs (2 jobs/SPF, 2 nodes x 4 GPUs/node, exclude `k003`):
  - `712898` `grasp_train_spf36_2n4g_j0`
  - `712899` `grasp_train_spf36_2n4g_j1`
  - `712900` `grasp_train_spf2_2n4g_j0`
  - `712901` `grasp_train_spf2_2n4g_j1`
- Added migration tooling:
  - `pack_grasp_targets_to_packed.py`
  - `grasp_pack_legacy_to_packed.sbatch`
  - migration writes packed H5 with per-file lock, shard support, and optional delete-on-success.
- Submitted migration job (currently **non-destructive** to avoid breaking running matrix still reading old root):
  - `712902` `grasp_pack_train_to_packed` (2 nodes, 8 tasks/node, `DELETE_SOURCE=0`)

### 2026-02-15 Overnight data-first queue pivot
- Goal: maximize GRASP target coverage overnight.
- Canceled matrix jobs to free slots:
  - `712878`, `712879`, `712880`, `712881`, `712882`, `712883`
- Canceled earlier 2-per-SPF GRASP pending jobs:
  - `712898`, `712899`, `712900`, `712901`
- Kept migration running:
  - `712902` `grasp_pack_train_to_packed` (RUNNING)
- Submitted fresh GRASP generation at 4 jobs/SPF (8 jobs total), each `2 nodes x 4 GPUs/node`, packed root, `--exclude k003`:
  - SPF 36: `712907`, `712908`, `712909`, `712910`
  - SPF 2: `712911`, `712912`, `712913`, `712914`
- Migration mode switch:
  - canceled copy-only migration `712902` (`DELETE_SOURCE=0`)
  - resubmitted delete-after-pack migration as `712915` (`grasp_pack_train_to_packed_del`, `DELETE_SOURCE=1`)

### 2026-02-15 36spf matrix relaunch after distill data expansion
- Updated 36spf LOO configs to use packed distill target root:
  - `data.grasp_target_root: /net/scratch2/annawoodard/grasp_targets_fastmri_train_packed`
- Submitted 5 runs (2 nodes x 4 GPUs/node, `exclude=k003`, env=`brisknet`):
  - General:
    - `713113` `m36_supdist_loo1_full_2n4g`
    - `713114` `m36_supdist_loo2_nomultilag_2n4g`
    - `713115` `m36_supdist_loo3_noarrival_2n4g`
  - Burst:
    - `713116` `m36_supdist_loo5_nofreq_2n4g_burst` (`timeout=230`)
    - `713117` `m36_supdist_loo6_freqw010_2n4g_burst` (`timeout=230`)
- Immediate launcher failure on first attempt:
  - jobs `713113`..`713117` failed because `submit.py` was given `--micromamba-path /tmp/codex_mamba_init.sh`, which does not exist on compute nodes.
  - error: `FileNotFoundError: Micromamba init script not found: /tmp/codex_mamba_init.sh`
- Fix:
  - created persistent shared init script: `/home/annawoodard/.codex_mamba_init.sh`
  - resubmitted same 5 runs:
    - General: `713118` (loo1 full), `713119` (loo2 nomultilag), `713120` (loo3 noarrival)
    - Burst: `713121` (loo5 nofreq), `713122` (loo6 freqw010)
- Follow-up crash diagnosis:
  - `713120`, `713121`, `713122` failed at startup due packed HDF5 lock contention while GRASP writers were still active.
  - Error signature: `BlockingIOError: unable to lock file, errno = 11` from `h5py.File(..., "r")` in `dataloader.py::_build_supervised_distill_index`.
- Code fix:
  - updated packed target read paths in `dataloader.py` to open with `locking=False`:
    - `_build_supervised_distill_index` packed H5 scan
    - `_load_supervised_distill_target` packed H5 slice read
- Resubmitted failed trio after patch:
  - `713131` (loo3 noarrival, general)
  - `713132` (loo5 nofreq, burst)
  - `713133` (loo6 freqw010, burst)

### 2026-02-15 Distill mismatch cleanup + clean resubmits
- Issue observed:
  - `m36_supdist_loo1_full_2n4g` and `m36_supdist_loo2_nomultilag_2n4g` were running with `valid_fraction=0.000` (effectively no supervised distill active).
  - Logs showed they were still tied to stale non-packed target context from older run state.
- Action taken:
  - Canceled non-distill / stale runs and pending stale variants:
    - canceled: `713118`, `713119`, `713132`, `713133`
  - Deleted stale output dirs to force clean state:
    - `/net/projects2/annawoodard/experiments/m36_supdist_loo1_full_2n4g`
    - `/net/projects2/annawoodard/experiments/m36_supdist_loo2_nomultilag_2n4g`
    - `/net/projects2/annawoodard/experiments/m36_supdist_loo5_nofreq_2n4g_burst`
    - `/net/projects2/annawoodard/experiments/m36_supdist_loo6_freqw010_2n4g_burst`
- Clean resubmits (same experiment names, fresh dirs):
  - `713160` `m36_supdist_loo1_full_2n4g`
  - `713161` `m36_supdist_loo2_nomultilag_2n4g`
  - `713162` `m36_supdist_loo5_nofreq_2n4g_burst`
  - `713163` `m36_supdist_loo6_freqw010_2n4g_burst`
- Verification:
  - `713160` startup log now shows packed-target distill coverage:
    - `[SupervisedDistill] target_coverage=258/258 exams, target_slices=45624, spf=36, frames=8`
  - checkpoint behavior: clean start (`No existing checkpoint found; starting new run.`)

### 2026-02-15 Warm-start pivot for ablations (keep one clean scratch proof)
- Strategy agreed:
  - keep one clean scratch run as proof-of-method (`m36_supdist_loo3_noarrival_2n4g`, job `713131`)
  - warm-start all other active ablations to accelerate iteration under deadline.
- Warm-start source checkpoint selected:
  - `/net/projects2/annawoodard/experiments/m36_supdistdense_tdiff_w080_lr3em4_s1_2n4g/m36_supdistdense_tdiff_w080_lr3em4_s1_2n4g_best_model.pth`
  - source run best PSNR in `run_state.json`: `42.5133`.
- Config updates:
  - set `experiment.init_checkpoint` to the path above in:
    - `configs/deadline_20260215_supdist_temporal_loo/m36_supdist_loo1_full_2n4g.yaml`
    - `configs/deadline_20260215_supdist_temporal_loo/m36_supdist_loo2_nomultilag_2n4g.yaml`
    - `configs/deadline_20260215_supdist_temporal_loo/m36_supdist_loo5_nofreq_2n4g_burst.yaml`
    - `configs/deadline_20260215_supdist_temporal_loo/m36_supdist_loo6_freqw010_2n4g_burst.yaml`
- Recycle for clean run dirs:
  - canceled stale jobs: `713160`, `713161`, `713162`, `713163`
  - deleted stale dirs for those 4 experiments to avoid residual state
  - resubmitted fresh jobs:
    - `713166` `m36_supdist_loo1_full_2n4g`
    - `713167` `m36_supdist_loo2_nomultilag_2n4g`
    - `713168` `m36_supdist_loo5_nofreq_2n4g_burst`
    - `713169` `m36_supdist_loo6_freqw010_2n4g_burst`
- Startup verification (`713166`):
  - warm-start confirmed:
    - `[Checkpoint] No run checkpoint found; warm-starting from experiment.init_checkpoint: ...m36_supdistdense_tdiff_w080_lr3em4_s1_2n4g_best_model.pth`
  - distill target indexing confirmed:
    - `[SupervisedDistill] target_coverage=258/258 exams, target_slices=45699, spf=36, frames=8`

### 2026-02-15 Queue status check (post warm-start resubmit)
- User question: why only 8 jobs visible.
- Current queue snapshot:
  - RUNNING: `713166`, `713167`, `712911`, `712912`, `712913`, `712914`
  - PENDING: `713168`, `713169`
- Missing two are not crashes:
  - `712909` (`grasp_train_spf36_2n4g_j2`) -> `COMPLETED`, exit `0:0`
  - `713131` (`m36_supdist_loo3_noarrival_2n4g`) -> `COMPLETED`, exit `0:0`
- No new crash among currently active jobs.

### 2026-02-15 Completed-run analysis: m36_supdist_loo3_noarrival_2n4g
- Run: `m36_supdist_loo3_noarrival_2n4g` (`713131`)
- Status: `COMPLETED`
- Config: `configs/deadline_20260215_supdist_temporal_loo/m36_supdist_loo3_noarrival_2n4g.yaml`
- Key setup notes:
  - `init_checkpoint: null` (clean scratch)
  - supervised distill enabled with packed targets (`target_coverage=258/258`)
  - no arrival weighting (`arrival_weighting.enable: false`)
  - distill stop at step 306
- Best metrics:
  - SSIM: `0.8911` @ step `260`
  - PSNR: `35.1912` @ step `360`
  - LPIPS: `0.1408` @ step `360`
  - CurveCorr: `0.9952` @ step `40` (final `0.9947`)
- Final metrics (step 360):
  - SSIM `0.8555`, PSNR `35.1912`, LPIPS `0.1408`, CurveCorr `0.9947`
  - Temporal: `rho_full=0.9910`, `rho_early=0.5410`, `MAE_full=1.626`, `MAE_early=2.488`, `t_arr_err=9.563s`, `t_peak_err=15.45s`
- Runtime:
  - wall time (attempt 3): `7205s` (~2.00h), cumulative `7241s`
  - mean iteration time: `7.13s` (n=360)
  - eval inference time/sample: mean `3.158s` (n=15 per eval)
- Overlay artifacts written:
  - `/net/projects2/annawoodard/experiments/m36_loo3_analysis_overlay_20260215/eval_metrics_overlay.png`
  - `/net/projects2/annawoodard/experiments/m36_loo3_analysis_overlay_20260215/eval_temporal_metrics_overlay.png`
  - `/net/projects2/annawoodard/experiments/m36_loo3_analysis_overlay_20260215/training_metrics_overlay.png`
  - `/net/projects2/annawoodard/experiments/m36_loo3_analysis_overlay_20260215/overlay_metric_tables.txt`
- Comparison takeaway vs prior strong dense-distill runs (`m36_supdistdense_*`):
  - This run is much worse on DRO image quality (PSNR ~35 vs ~42.3-42.5, LPIPS ~0.141 vs ~0.026).
  - Temporal metrics are also much worse (e.g., early correlation ~0.54 vs ~0.60; t_peak error ~15.5s vs ~13.7s).
  - Non-DRO raw k-space MAE/MSE remain "good" (beats GRASP), reinforcing that these metrics alone are weak indicators of DRO-domain quality.
  - Main confounder: this ablation was scratch while dense baselines were warm-started; cannot attribute gap solely to `noarrival`.
- Behavior around distill stop:
  - Distill active through step 306 (`valid_fraction=1.0`), then zero by schedule.
  - After distill-off, SSIM declines (~0.88 -> ~0.86), PSNR flat/slight up, LPIPS slight improvement; temporal metrics mostly flat/poor.
- Decision:
  - Do not treat this as evidence against arrival-weighting yet.
  - Evaluate `noarrival` only in warm-start matched A/B (currently queued as warm-start runs).

### 2026-02-15 Warm-start lineage + distill plateau diagnosis
- Question: "Do we remember how warm-start checkpoints were obtained, and does warm-start explain distill plateau?"
- Warm-start lineage (from `run_state.json` chain):
  1. `m36_supdistdense_tdiff_w080_lr3em4_s1_2n4g` (best PSNR 42.513) 
     <- init from
  2. `m36_supdist_abs_w020_noei_2n4g` (best PSNR 41.670)
     <- init from
  3. `m36_dline_timeenc_mixmc005_ctrl_2n4g` (best PSNR 41.369)
     <- init from
  4. `m36_deadline_late_ei_rebin_frombest_2n4g` (best PSNR 41.048)
     <- init from
  5. `mamba_36spf_sweepB_early_ei_archfix_lr5em4_minlf6_wu5_ep160_archtune_dconv4_2n4g` (best PSNR 39.188)
- So checkpoint provenance is recoverable and explicit.

- Distill pressure findings:
  - `m36_supdist_loo3_noarrival_2n4g` (scratch):
    - distill active 306/360 steps (`vf_mean=0.85`, stop at step 306)
    - weighted distill(+freq) / weighted MC ratio when active: mean ~0.655 (substantial)
  - `m36_supdistdense_tdiff_w080_lr3em4_s1_2n4g` (warm chain):
    - distill valid on 81/360 steps (`vf_mean=0.225`) in that historical run
    - ratio when active also substantial: mean ~0.684
- Interpretation:
  - Warm-start explains large baseline quality gap vs scratch.
  - Plateau is not only warm-start-related; it is also consistent with distill-target coverage/schedule and objective mismatch with temporal metrics on DRO.
  - In prior dense runs, distill was active only intermittently (low `valid_fraction`), so improvements were largely inherited from lineage + MC, not persistent full-run dense distill.

- Additional trend signals:
  - `dense_tdiff` PSNR still rising late (slope ~+0.0031 per step), while early-correlation drifts down late.
  - `loo3` after distill stop (step>306): PSNR roughly flat, SSIM declines, temporal metrics stay poor/flat.
- Decision implication:
  - Need warm-start + high, consistent distill coverage in the same run (current requeued jobs aim for this) to answer whether dense distill truly moves temporal fidelity, independent of lineage.

### 2026-02-15 Plateau-break matrix submission (early-stopping-on by default)
- User request:
  - submit a new matrix to break the 36spf plateau,
  - include one higher-capacity run,
  - enforce early stopping for new runs moving forward.

- Code update relevant to this matrix:
  - `train_zf.py` now supports config-native
    - `training.auto_actions.early_stopping`
    - `training.auto_actions.lr_restart_on_plateau`

- Previous LOO jobs handled:
  - `713166` `m36_supdist_loo1_full_2n4g` -> `COMPLETED` (`0:0`, elapsed `01:53:34`)
  - `713167` `m36_supdist_loo2_nomultilag_2n4g` -> `CANCELLED by 22734`
  - `713168` `m36_supdist_loo5_nofreq_2n4g_burst` -> `CANCELLED by 22734`
  - `713169` `m36_supdist_loo6_freqw010_2n4g_burst` -> `CANCELLED by 22734`

- New config set created:
  - `configs/deadline_20260215_plateau_break_es/m36_plateau_full_esr_2n4g.yaml`
  - `configs/deadline_20260215_plateau_break_es/m36_plateau_distillrelease_esr_2n4g.yaml`
  - `configs/deadline_20260215_plateau_break_es/m36_plateau_mixmc010_esr_2n4g.yaml`
  - `configs/deadline_20260215_plateau_break_es/m36_plateau_lateei_esr_2n4g.yaml`
  - `configs/deadline_20260215_plateau_break_es/m36_plateau_highcap160x8x48_esr_2n4g.yaml`

- Shared settings across all five:
  - warm-start checkpoint: `/net/projects2/annawoodard/experiments/m36_supdistdense_tdiff_w080_lr3em4_s1_2n4g/m36_supdistdense_tdiff_w080_lr3em4_s1_2n4g_best_model.pth`
  - `training.auto_actions.early_stopping.enabled: true`
  - `training.auto_actions.lr_restart_on_plateau.enabled: true`
  - 2 nodes x 4 GPUs/node, general queue, brisknet env

- Matrix hypotheses:
  1. `m36_plateau_full_esr_2n4g` (`713243`):
     - long baseline with persistent distill (`stop_step=440`, `max_steps=520`) + restart-on-plateau should beat the prior ~step-150 flattening.
  2. `m36_plateau_distillrelease_esr_2n4g` (`713246`):
     - earlier distill release (`stop_step=320`) should allow post-teacher adaptation and potentially improve DRO PSNR late.
  3. `m36_plateau_mixmc010_esr_2n4g` (`713242`):
     - stronger MSE pressure (`mc mse_weight=0.10`, lr `2.5e-4`) should push PSNR trajectory upward if current plateau is MC-objective-limited.
  4. `m36_plateau_lateei_esr_2n4g` (`713244`):
     - delayed EI (`warmup=360`) + delayed rebin (`warmup=420`) tests whether adding temporal regularization only after strong MC/distill fit improves temporal metrics without early degradation.
  5. `m36_plateau_highcap160x8x48_esr_2n4g` (`713245`):
     - conservative higher-capacity speculative slot (`hidden_dim=160`, `num_blocks=8`, `d_state=48`) to test under-capacity as a plateau driver.

- Submission status at submit time:
  - `713242`, `713243`, `713244`, `713245`, `713246` all `PENDING (Priority)`.
- Startup verification (`713242`, `m36_plateau_mixmc010_esr_2n4g`):
  - warm-start confirmed from `m36_supdistdense_tdiff_w080_lr3em4_s1_2n4g_best_model.pth`
  - distill coverage confirmed: `target_coverage=258/258`
  - auto-actions confirmed in log:
    - `[AutoActions] early_stopping enabled: rules=2, require=all, min_step=180`
    - `[AutoActions] lr_restart_on_plateau enabled: metric=dro_psnr ...`

### 2026-02-15 LOO1 vs LOO3 overlay analysis (requested)
- Overlay bundle generated:
  - `/net/projects2/annawoodard/experiments/m36_loo13_overlay_20260215/eval_metrics_overlay.png`
  - `/net/projects2/annawoodard/experiments/m36_loo13_overlay_20260215/eval_temporal_metrics_overlay.png`
  - `/net/projects2/annawoodard/experiments/m36_loo13_overlay_20260215/training_metrics_overlay.png`
  - `/net/projects2/annawoodard/experiments/m36_loo13_overlay_20260215/overlay_metric_tables.txt`

- Runs compared:
  - `m36_supdist_loo1_full_2n4g` (`713166`, warm-started)
  - `m36_supdist_loo3_noarrival_2n4g` (`713131`, from scratch)

- Key observed values from overlay tables:
  - `loo1` best/final regime:
    - SSIM peaks ~`0.965`; PSNR reaches ~`42.7`; LPIPS best ~`0.023`; `rho_early` peaks ~`0.64`.
    - `t_arr` error reaches low ~`6.16s`; `t_peak` error ~`13.0s` best.
  - `loo3` best/final regime:
    - SSIM peaks ~`0.891` and final ~`0.855`; PSNR final ~`35.2`; LPIPS ~`0.141`; `rho_early` ~`0.54`.
    - `t_arr` error degrades late to ~`9-10s`; `t_peak` error ~`15.4s`.

- Interpretation:
  - This specific comparison is confounded by init condition (`loo1` warm-start vs `loo3` scratch), so it does **not** isolate arrival-weighting effect.
  - Non-DRO k-space raw metrics are relatively close across the two runs while DRO and temporal metrics are very far apart, reinforcing that raw non-DRO metrics alone are weak for selecting 36spf temporal-fidelity settings.
  - Practical takeaway: use matched warm-start + matched scheduler for LOO attribution.

### 2026-02-15 LOO follow-up submission with early-stopping/restart (fair warm-start)
- New config directory:
  - `configs/deadline_20260215_supdist_temporal_loo_es/`
- Changes applied to all six LOO configs:
  - unified warm-start checkpoint:
    - `/net/projects2/annawoodard/experiments/m36_supdistdense_tdiff_w080_lr3em4_s1_2n4g/m36_supdistdense_tdiff_w080_lr3em4_s1_2n4g_best_model.pth`
  - added:
    - `training.auto_actions.early_stopping`
    - `training.auto_actions.lr_restart_on_plateau`

- Submitted jobs (exp name = output dir = slurm name):
  - General (2 nodes x 4 gpus):
    - `713280` `m36_supdist_loo1_full_esr_2n4g`
    - `713283` `m36_supdist_loo2_nomultilag_esr_2n4g`
    - `713294` `m36_supdist_loo3_noarrival_esr_2n4g`
    - `713299` `m36_supdist_loo4_nocurriculum_esr_2n4g`
  - Burst (1 node x 4 gpus):
    - `713302` `m36_supdist_loo5_nofreq_esr_1n4g_burst`
    - `713303` `m36_supdist_loo6_freqw010_esr_1n4g_burst`

- Note:
  - First burst submission attempt (`713300`, `713301`) was blocked by burst walltime limit and was replaced with `240 min` jobs (`713302`, `713303`).

### 2026-02-16 Overlay refresh + plateau interpretation (mixmc010 vs dense_tdiff)
- Refreshed overlay bundles after plotting updates:
  - `/net/projects2/annawoodard/experiments/m36_overlay_mixmc010_vs_tdiff080_20260215/`
  - `/net/projects2/annawoodard/experiments/m36_loo13_overlay_20260215/`
- `overlay_metrics.py` updates used for this refresh:
  - eval overlay now includes LR in `eval_metrics_overlay.png`
  - training overlay now includes distill and weighted-loss traces:
    - supervised distill losses (+weighted +freq),
    - teacher distill losses (+weighted),
    - distill valid fraction, distill weights, curriculum alpha,
    - weighted MC/EI/Adj/Rebin losses

- Comparison takeaway (`m36_plateau_mixmc010_esr_2n4g` vs `m36_supdistdense_tdiff_w080_lr3em4_s1_2n4g`):
  - Not a pure "needs one LR kick" story.
  - LR restarts happened in `mixmc010` and gave temporary PSNR recovery (best PSNR ~`42.40` at step `320`), but late image-quality collapse still occurred by step `480` (PSNR ~`41.64`, SSIM ~`0.901`).
  - Temporal early-correlation improved late in `mixmc010` (`rho_early` up to ~`0.659`), while SSIM/PSNR dropped, indicating objective tradeoff rather than simple optimization stall.
  - Practical implication:
    - LR restart is useful but insufficient alone.
    - Need guardrails around post-distill phase (release timing, EI/rebin timing/weights, and stopping rules tied to DRO image metrics and temporal metrics jointly).

### 2026-02-16 Running m36 overlay snapshot (user-requested)
- Overlay regenerated from currently running m36 jobs:
  - `/net/projects2/annawoodard/experiments/m36_overlay_running_now_latest/eval_metrics_overlay.png`
  - `/net/projects2/annawoodard/experiments/m36_overlay_running_now_latest/eval_temporal_metrics_overlay.png`
  - `/net/projects2/annawoodard/experiments/m36_overlay_running_now_latest/training_metrics_overlay.png`
  - `/net/projects2/annawoodard/experiments/m36_overlay_running_now_latest/overlay_metric_tables.txt`

- Included in overlay (has checkpoints):
  - `m36_plateau_full_esr_2n4g` (`713243`)
  - `m36_plateau_lateei_esr_2n4g` (`713244`)
  - `m36_plateau_distillrelease_esr_2n4g` (`713246`)
  - `m36_plateau_highcap160x8x48_esr_2n4g` (`713245`)
- Not yet included (no checkpoint written yet):
  - `m36_supdist_loo1_full_esr_2n4g` (`713280`)

- Current quantitative snapshot vs GRASP baseline (best observed in each run so far):
  - `m36_plateau_full_esr_2n4g`:
    - DRO PSNR `42.34` (GRASP `48.58`), SSIM `0.955` (`0.979`), LPIPS `0.026` (`0.0021`)
    - temporal: `rho_early=0.616` (`0.736`), `t_arr=6.22s` (`4.73s`), `t_peak=12.85s` (`9.66s`)
  - `m36_plateau_lateei_esr_2n4g`:
    - DRO PSNR `42.27`, SSIM `0.944`, LPIPS `0.026`
    - temporal: `rho_early=0.611`, `t_arr=6.31s`, `t_peak=13.16s`
  - `m36_plateau_distillrelease_esr_2n4g`:
    - Early good point at step 20 then degraded by step 60 (SSIM `0.939 -> 0.888`, PSNR `42.11 -> 41.56`)
  - `m36_plateau_highcap160x8x48_esr_2n4g`:
    - still far behind at this stage (best PSNR `35.21`, SSIM `0.823`, `rho_early=0.553`)

- Takeaways:
  - Full and late-EI tracks are currently strongest and close to each other; neither is yet close to GRASP on temporal timing metrics.
  - `distillrelease` looks unstable when releasing teacher pressure early (at least in this early window).
  - The high-capacity run is not competitive in early trajectory; likely needs either different optimization or should be treated as speculative only.
  - Non-DRO raw k-space metrics remain nearly flat across these runs and are not useful for ranking progress at this stage.

### 2026-02-16 Why `m36_plateau_full_esr_2n4g` / `m36_plateau_lateei_esr_2n4g` did not stop early
- Runtime confirmation from logs:
  - `m36_plateau_full_esr_2n4g` log shows:
    - `[AutoActions] early_stopping enabled: rules=2, require=all, min_step=180`
    - stop criteria were threshold-based (`dro_psnr < 41.0` AND `dro_ssim < 0.92`).
  - `m36_plateau_lateei_esr_2n4g` log shows:
    - `[AutoActions] early_stopping enabled: rules=3, require=all, min_step=180`
    - stop criteria were threshold-based (`dro_psnr < 41.0` AND `dro_ssim < 0.90` AND `rho_early < 0.56`).
- Therefore these runs do not early-stop when plateaued near PSNR~42 and SSIM~0.94; they only stop if they drop below those floor thresholds.
- Both runs did trigger one LR restart (`full` at step 220, `lateei` at step 200), but no sustained improvement followed.
- Decision implication:
  - Treat these as exhausted for current objective; keep best checkpoint and branch to a stronger distill-focused continuation recipe rather than waiting for threshold-based stop.

### 2026-02-16 Cancel + running-overlay refresh (user-requested)
- Actions:
  - Cancelled plateaued jobs:
    - `713243` `m36_plateau_full_esr_2n4g`
    - `713244` `m36_plateau_lateei_esr_2n4g`
  - Remaining running m36 jobs at refresh time:
    - `713246` `m36_plateau_distillrelease_esr_2n4g`
    - `713280` `m36_supdist_loo1_full_esr_2n4g`
    - `713283` `m36_supdist_loo2_nomultilag_esr_2n4g`
    - `713294` `m36_supdist_loo3_noarrival_esr_2n4g`

- New overlay artifacts:
  - `/net/projects2/annawoodard/experiments/m36_overlay_running_postcancel/eval_metrics_overlay.png`
  - `/net/projects2/annawoodard/experiments/m36_overlay_running_postcancel/eval_temporal_metrics_overlay.png`
  - `/net/projects2/annawoodard/experiments/m36_overlay_running_postcancel/training_metrics_overlay.png`
  - `/net/projects2/annawoodard/experiments/m36_overlay_running_postcancel/overlay_metric_tables.txt`

- Quantitative snapshot (best so far, vs GRASP):
  - `m36_plateau_distillrelease_esr_2n4g` (step 160):
    - PSNR `42.22` (gap `-6.36`), SSIM `0.946` (gap `-0.033`), `rho_early 0.618` (gap `-0.118`), `t_arr 6.25s` (gap `+1.52s`)
  - `m36_supdist_loo1_full_esr_2n4g` (step 140):
    - PSNR `42.16` (gap `-6.42`), SSIM `0.943` (gap `-0.037`), `rho_early 0.603` (gap `-0.133`), `t_arr 6.45s` (gap `+1.72s`)
  - `m36_supdist_loo2_nomultilag_esr_2n4g` (step 60):
    - PSNR `41.91` (gap `-6.67`), SSIM `0.924` (gap `-0.056`), `rho_early 0.595` (gap `-0.141`), `t_arr 6.49s` (gap `+1.76s`)
  - `m36_supdist_loo3_noarrival_esr_2n4g` (step 40):
    - PSNR `42.02` (gap `-6.56`), SSIM `0.938` (gap `-0.042`), `rho_early 0.566` (gap `-0.170`), `t_arr 6.34s` (gap `+1.62s`)

- Takeaways:
  - Current survivors are clustered in the same basin; none is closing the temporal gap quickly enough yet.
  - `loo3_noarrival` is currently weakest on `rho_early`, suggesting arrival weighting is helping early-temporal behavior.
  - `loo2_nomultilag` is trailing both image and temporal metrics so far, suggesting multi-lag temporal-diff still looks beneficial.
  - `distillrelease` is currently best among active runs but still far from GRASP on temporal timing metrics.
- Refresh note:
  - Re-ran `m36_overlay_running_postcancel` including currently running `m36_supdist_loo4_nocurriculum_esr_2n4g`.
  - It was skipped by `overlay_metrics.py` because no checkpoint exists yet.

### 2026-02-16 Plateau diagnosis from training overlays/checkpoints
- Checked current running m36 checkpoints (`m36_plateau_distillrelease_esr_2n4g`, `m36_supdist_loo1_full_esr_2n4g`, `m36_supdist_loo2_nomultilag_esr_2n4g`, `m36_supdist_loo3_noarrival_esr_2n4g`).
- Observed trend:
  - `train_mc_losses` roughly flat around `~1.3e-5` (no sustained downward trend).
  - `train_supervised_distill_losses` roughly flat around `~1.1e-5` (also no sustained downward trend).
  - `weighted_train_supervised_distill_losses` increases mainly because `supervised_distill_weight_history` ramps from near `0` to `0.8` by ~step 80.
  - `val_mc_losses` also mostly flat (`~0.072-0.075`).
  - Eval PSNR/SSIM flatten near `~42 / ~0.93-0.95`.
- Interpretation:
  - These runs are in a true optimization plateau (not just noisy train-loss plots).
  - Early SSIM dip aligns with aggressive distill-weight ramp; objective transitions are dominating early behavior.

### 2026-02-16 Plateau-attack action: cancel low-performing LOO + submit PSNR-first distill run
- User request:
  - cancel low-performing LOO runs,
  - submit a new run with plateau-break plan,
  - record takeaways.

- Cancellations executed:
  - `713283` `m36_supdist_loo2_nomultilag_esr_2n4g` (low PSNR/SSIM trend among active LOO)
  - `713294` `m36_supdist_loo3_noarrival_esr_2n4g` (low early-temporal trend among active LOO)
  - `713302` `m36_supdist_loo5_nofreq_esr_1n4g_burst` (pending low-priority LOO)
  - `713303` `m36_supdist_loo6_freqw010_esr_1n4g_burst` (pending low-priority LOO)
  - `713509` `m36_meetgrasp_distillonly_abs_w120_2n4g` (replaced by improved config variant)

- New config created:
  - `configs/deadline_20260216_meet_grasp/m36_meetgrasp_distillonly_abs_w120_mcmse020_flatlr_2n4g.yaml`
- Key changes vs previous meet-GRASP config:
  - stronger MC MSE pressure: `model.losses.mc_loss.mse_weight: 0.2` (from `0.05`)
  - stronger but slower-ramped distill: `weight: 1.5`, `duration_steps: 120`, absolute mode, EI/rebin still off
  - flatter LR schedule: `training.lr_schedule.min_lr_factor: 1.0`
  - true patience stop active: `training.early_stopping.enabled: true` with `min_step=220`, `patience_evals=5`, `min_delta=0.02`
  - longer budget to allow escape if learning resumes: `training.max_steps: 640`

- Submission:
  - job id `713540`
  - exp/job: `m36_meetgrasp_distillonly_abs_w120_mcmse020_flatlr_2n4g`
  - 2 nodes x 4 GPUs, general queue
  - logs: `/net/projects2/annawoodard/experiments/m36_meetgrasp_distillonly_abs_w120_mcmse020_flatlr_2n4g/submitit_logs`

- Consolidated takeaways recorded for this decision:
  - Active runs show true optimization plateau: MC/distill raw losses are flat while PSNR stagnates near ~42.
  - Early SSIM drop aligns with aggressive distill-weight ramp (objective transition effect), not with MC collapse.
  - Prior stop rules were failure-threshold guards, not plateau detection; switched to simple patience-based early stopping for this new run.
  - Multi-lag and arrival weighting looked directionally useful from active LOO ranking; weakest LOO variants were canceled to free slots for a direct PSNR-first push.

### 2026-02-16 Monitoring refresh + running overlay (post new plateau-attack run)
- Current standard-slot occupancy snapshot:
  - Running: 4 (`713246`, `713540`, `713469`, `713299`)
  - Pending: 0
  - Occupancy: `4/8` standard slots currently filled.

- Fresh overlay generated for currently running m36 jobs:
  - `/net/projects2/annawoodard/experiments/m36_overlay_running_20260216_latest/eval_metrics_overlay.png`
  - `/net/projects2/annawoodard/experiments/m36_overlay_running_20260216_latest/eval_temporal_metrics_overlay.png`
  - `/net/projects2/annawoodard/experiments/m36_overlay_running_20260216_latest/training_metrics_overlay.png`
  - `/net/projects2/annawoodard/experiments/m36_overlay_running_20260216_latest/overlay_metric_tables.txt`

- Running m36 quantitative snapshot:
  - `m36_meetgrasp_distillonly_abs_w120_mcmse020_flatlr_2n4g` (`713540`):
    - step `160/640` (25.0%)
    - best PSNR `43.01`, SSIM `0.9682`, LPIPS `0.0201`
    - temporal still weak: best `rho_early=0.6058`, best `t_arr=6.60s`
  - `m36_supdist_loo4_nocurriculum_esr_2n4g` (`713299`):
    - step `260/360` (72.2%)
    - best PSNR `42.59`, SSIM `0.9672`
    - temporal: best `rho_early=0.6128`, best `t_arr=6.14s`
  - `m36_plateau_distillrelease_esr_2n4g` (`713246`):
    - step `420/520` (80.8%)
    - best PSNR `42.47` (at step 320), but current degraded to `41.46`
    - temporal mixed: `rho_early` improved late (best `0.6682`) but iAUC error worsened badly late (current `17.49`).

- Consolidated interpretation:
  - New PSNR-first run is the best image-quality trajectory so far in active set.
  - Temporal metrics are still the bottleneck; PSNR gains are not automatically translating to `rho_early`/timing gains.
  - Distill-release run now looks unstable post-release and appears low value to continue versus new config lineage.
- Recommended immediate decisions (pending user confirmation):
  - Kill candidate: `m36_plateau_distillrelease_esr_2n4g` (late degradation after distill release, low remaining value).
  - Keep: `m36_meetgrasp_distillonly_abs_w120_mcmse020_flatlr_2n4g` (best active PSNR/SSIM trajectory).
  - Keep (near-term): `m36_supdist_loo4_nocurriculum_esr_2n4g` until next 2-3 evals to confirm final trend.
  - Keep data pipeline job: `grasp_train_spf2_2n4g_j2`.
- Suggested next fills to restore standard-slot occupancy to 8:
  - derivatives of `m36_meetgrasp_distillonly_abs_w120_mcmse020_flatlr_2n4g` focused on temporal improvement while preserving PSNR lead.

### 2026-02-16 Student-vs-GRASP metric plumbing + disambiguated labeling (code + overlays)
- Goal:
  - Remove DRO/GRASP ambiguity in logs/plots and make student-vs-GRASP progress first-class for model selection.
- Code updates:
  - `eval.py`
    - Added explicit `student_vs_grasp_*` image metrics (`ssim/psnr/mse/lpips`) into `temporal_metrics` for DRO and non-DRO eval branches.
    - Added explicit `student_vs_grasp_*` temporal metrics for malignant/benign ROI (e.g., `student_vs_grasp_all_curve_corr`, `student_vs_grasp_all_early_corr`, timing errors).
  - `train_zf.py`
    - Added checkpoint curve storage/loading/logging for student-vs-GRASP image + temporal metrics.
    - Added TensorBoard logging for student-vs-GRASP metrics.
    - Renamed console reporting to explicit references:
      - `Student-vs-DRO ...`
      - `Student-vs-GRASP ...`
      - `GRASP-vs-DRO baseline ...`
      - `GRASP-vs-NonDRO ...`
    - Added run-local plot output `eval_temporal_metrics_student_vs_grasp.png`.
    - Updated eval CSV row labels to explicit references (`Student_vs_DRO`, `GRASP_vs_DRO_baseline`).
  - `overlay_metrics.py`
    - Renamed panel titles to explicit reference framing (`Student vs DRO ...`, `Student vs GRASP ...`).
    - Added student-vs-GRASP eval panels (SSIM/PSNR/MSE/LPIPS).
    - Split temporal overlays into:
      - `eval_temporal_metrics_overlay.png` (student-vs-DRO, with GRASP-vs-DRO baselines)
      - `eval_temporal_metrics_student_vs_grasp_overlay.png` (student-vs-GRASP)
    - Baseline legend labels now differentiate DRO vs non-DRO references.
- Validation:
  - `python -m py_compile eval.py train_zf.py overlay_metrics.py` passed.
  - Smoke overlay run passed and wrote new overlay artifacts under `/tmp/overlay_metric_smoke/`.

### 2026-02-16 Distill-first meet-GRASP matrix submission (user-requested)
- Strategy:
  - Keep EI/rebin off, maximize supervised distill pressure while retaining MC as constraint.
  - Use the current best warm-start checkpoint and vary distill pressure/temporal mode/MC MSE mix.
- New config directory:
  - `configs/deadline_20260216_meet_grasp_matrix/`
- Submitted jobs (2 nodes x 4 GPUs, general):
  - `713639` `m36_meetgrasp_abs_w220_mcmse010_flatlr_2n4g`
    - config: `configs/deadline_20260216_meet_grasp_matrix/m36_meetgrasp_abs_w220_mcmse010_flatlr_2n4g.yaml`
  - `713640` `m36_meetgrasp_tdiff_w150_mcmse020_flatlr_2n4g`
    - config: `configs/deadline_20260216_meet_grasp_matrix/m36_meetgrasp_tdiff_w150_mcmse020_flatlr_2n4g.yaml`
  - `713641` `m36_meetgrasp_abs_w180_mcmse020_flatlr_2n4g`
    - config: `configs/deadline_20260216_meet_grasp_matrix/m36_meetgrasp_abs_w180_mcmse020_flatlr_2n4g.yaml`
  - `713642` `m36_meetgrasp_abs_w150_mcmse000_flatlr_2n4g`
    - config: `configs/deadline_20260216_meet_grasp_matrix/m36_meetgrasp_abs_w150_mcmse000_flatlr_2n4g.yaml`
  - `713643` `m36_meetgrasp_curric_abs2tdiff_w180_mcmse020_flatlr_2n4g`
    - config: `configs/deadline_20260216_meet_grasp_matrix/m36_meetgrasp_curric_abs2tdiff_w180_mcmse020_flatlr_2n4g.yaml`
  - `713645` `m36_meetgrasp_tdiff_w180_arrw300_mcmse020_flatlr_2n4g`
    - config: `configs/deadline_20260216_meet_grasp_matrix/m36_meetgrasp_tdiff_w180_arrw300_mcmse020_flatlr_2n4g.yaml`
- Queue occupancy after submission:
  - Standard slots: `8/8` (`RUNNING+PENDING`).
  - Active queue set also includes existing: `713540` and `713469`.

### 2026-02-15 Recent m36 overlay refresh + high-capacity status check
- User ask:
  - verify whether larger-capacity model has results or is still queued,
  - generate overlays for recent experiments.
- Queue/history status:
  - Larger-capacity run `m36_plateau_highcap160x8x48_esr_2n4g` (`713245`) is **COMPLETED** (not queued).
  - No active/pending high-capacity m36 run currently in queue.
- Overlay generated:
  - output dir: `/net/projects2/annawoodard/experiments/m36_overlay_recent_20260215_205713/`
  - files:
    - `eval_metrics_overlay.png`
    - `eval_temporal_metrics_overlay.png`
    - `eval_temporal_metrics_student_vs_grasp_overlay.png`
    - `training_metrics_overlay.png`
    - `overlay_metric_tables.txt`
  - inputs used:
    - running: `m36_meetgrasp_distillonly_abs_w120_mcmse020_flatlr_2n4g`, `m36_meetgrasp_abs_w220_mcmse010_flatlr_2n4g`, `m36_meetgrasp_tdiff_w150_mcmse020_flatlr_2n4g`
    - completed context: `m36_plateau_highcap160x8x48_esr_2n4g`
    - pending/new runs without checkpoints were skipped automatically.
- Snapshot from `overlay_metric_tables.txt`:
  - High-capacity (`713245`) best observed:
    - Student-vs-DRO PSNR ~`35.6`, SSIM ~`0.861`, LPIPS ~`0.130`, `rho_early` ~`0.581`.
  - Current stronger run (`713540`) best observed:
    - Student-vs-DRO PSNR ~`43.3`, SSIM ~`0.970`, LPIPS ~`0.0182`.
  - Newly started runs (`713639`, `713640`) have early points only (up to step 60).
- Takeaway:
  - Existing high-capacity attempt underperformed substantially and is not a current queue consumer.
  - Near-term decision signal should come from the new meet-GRASP matrix once additional eval steps land.

### 2026-02-15 High-capacity continuation swap (user-requested)
- User request:
  - continue high-capacity trajectory (or warm-start from high-capacity checkpoint) and run longer with best current strategy.
- Action:
  - canceled lower-priority pending run: `713642` `m36_meetgrasp_abs_w150_mcmse000_flatlr_2n4g`.
  - submitted high-capacity warm-start continuation:
    - job `713715`
    - exp/job name: `m36_meetgrasp_highcap160x8x48_abs_w240_mcmse010_flatlr_2n4g`
    - config: `configs/deadline_20260216_meet_grasp_matrix/m36_meetgrasp_highcap160x8x48_abs_w240_mcmse010_flatlr_2n4g.yaml`
    - resources: 2 nodes x 4 gpus/node, partition `general`.
- Continuation design details:
  - warm start source:
    - `/net/projects2/annawoodard/experiments/m36_plateau_highcap160x8x48_esr_2n4g/m36_plateau_highcap160x8x48_esr_2n4g_best_model.pth`
  - architecture retained high-capacity:
    - `hidden_dim=160`, `num_blocks=8`, `d_state=48`, `use_checkpoint=true`.
  - recipe aligned to stronger current insights:
    - EI/rebin off,
    - supervised distill `absolute`, `weight=2.0`, `duration_steps=240`, `stop_step=720`,
    - MC mixed with `mse_weight=0.1`,
    - flatter LR schedule (`min_lr_factor=1.0`),
    - longer budget `max_steps=840` with patience stop.
- Queue occupancy after swap:
  - standard slots: `8/8` (`running+pending`).

### 2026-02-16 Multiobjective supervised distill implementation + burst probe submissions
- User request:
  - implement multiobjective distill to improve PSNR/image quality without sacrificing temporal behavior,
  - submit runs using the new objective.

- Code changes:
  - `train_zf.py`
    - fully wired `model.losses.supervised_distill.multiobjective` into runtime training (previously parse-only):
      - image + temporal distill components are computed separately and combined as:
        - `L_sd = w_img * L_img + (temporal_scale * w_temp * L_temp) [+ optional frequency term]`
      - supports temporal multi-lag + arrival-weighting for the temporal component.
      - added adaptive temporal scale controller (optional) with EMA guardrail on image loss.
      - added fail-fast validation:
        - multiobjective and curriculum are mutually exclusive.
        - adaptive mode requires positive image weight.
      - added checkpoint persistence/resume for multiobjective state:
        - `supervised_multiobj_temporal_scale`, `supervised_multiobj_image_ema`, `supervised_multiobj_image_best`, and history/step arrays.
      - added TensorBoard + console logging for:
        - image/temporal distill component losses,
        - weighted component losses,
        - temporal scale + adaptive EMA/best diagnostics.
  - `overlay_metrics.py`
    - added training overlay panels for:
      - `train_supervised_distill_image_losses`
      - `weighted_train_supervised_distill_image_losses`
      - `train_supervised_distill_temporal_losses`
      - `weighted_train_supervised_distill_temporal_losses`
      - `supervised_multiobj_temporal_scale_history`
    - added train-axis support for `supervised_multiobj_temporal_scale_steps`.

- Validation:
  - `python -m py_compile train_zf.py overlay_metrics.py` passed.

- New configs:
  - `configs/deadline_20260216_meet_grasp_matrix/m36_meetgrasp_multiobj_adapt_absimg_tdiff_1n4g_burst.yaml`
    - multiobjective enabled, adaptive temporal scale enabled.
  - `configs/deadline_20260216_meet_grasp_matrix/m36_meetgrasp_multiobj_fixed_absimg_tdiff_1n4g_burst.yaml`
    - multiobjective enabled, fixed temporal scale (adaptive disabled).

- Submissions (burst QoS):
  - `713875` `m36_meetgrasp_multiobj_adapt_absimg_tdiff_1n4g_burst`
    - 1 node x 4 GPUs, timeout 240 min.
  - `713876` `m36_meetgrasp_multiobj_fixed_absimg_tdiff_1n4g_burst`
    - 1 node x 4 GPUs, timeout 240 min.

- Notes:
  - Initial submissions `713873/713874` were canceled and resubmitted as `713875/713876` due burst walltime limit (`QOSMaxWallDurationPerJobLimit` with 480 min).
  - Standard-slot occupancy remains filled (`8/8` running+pending); burst probes are additional.

- Hypothesis:
  - Multiobjective supervision will hold/improve image-space optimization pressure (PSNR proxy) while preserving temporal-difference learning, and adaptive temporal scaling should avoid over-penalizing image quality early.
- Decision trigger:
  - compare first 2-3 evals of `713875` vs `713876`; keep whichever gives better PSNR trend at matched/better early-temporal metrics.

### 2026-02-16 Monitoring refresh + consolidated recent overlay (meet-GRASP set)
- User request:
  - update monitoring,
  - assess current run status,
  - regenerate overlays for recent experiments,
  - summarize takeaways and highest-ROI next step.

- Queue snapshot (at refresh):
  - `RUNNING`: `713715` (`m36_meetgrasp_highcap160x8x48_abs_w240_mcmse010_flatlr_2n4g`)
  - `PENDING`: none
  - Burst probes from prior turn:
    - `713876` (`m36_meetgrasp_multiobj_fixed_absimg_tdiff_1n4g_burst`) `COMPLETED`
    - `713875` (`m36_meetgrasp_multiobj_adapt_absimg_tdiff_1n4g_burst`) `FAILED`
      - root cause: resumed same exp dir with completed schedule; fail-fast error `ValueError("Full training max_steps already complete.")`.

- Consolidated overlay generated:
  - output dir: `/net/projects2/annawoodard/experiments/m36_meetgrasp_overlay_20260216_status/`
  - files:
    - `eval_metrics_overlay.png`
    - `eval_temporal_metrics_overlay.png`
    - `eval_temporal_metrics_student_vs_grasp_overlay.png`
    - `training_metrics_overlay.png`
    - `overlay_metric_tables.txt`

- Best metric summary from recent `m36_meetgrasp_*` checkpoints:
  - `m36_meetgrasp_abs_w220_mcmse010_flatlr_2n4g`
    - best PSNR `43.572@660`, SSIM `0.9712@600`, LPIPS `0.0162@660`
    - temporal: `rho_early 0.6373@640`, `t_arr 6.338s@560`
    - student-vs-GRASP: PSNR `43.867@660`, `rho_early 0.7166@640`
  - `m36_meetgrasp_abs_w180_mcmse020_flatlr_2n4g`
    - best PSNR `43.540@600`, SSIM `0.9721@600`, LPIPS `0.0165@560`
    - temporal: `rho_early 0.6451@580`, `t_arr 6.316s@560`
    - student-vs-GRASP: PSNR `43.899@600`, `rho_early 0.7070@580`
  - `m36_meetgrasp_distillonly_abs_w120_mcmse020_flatlr_2n4g`
    - best PSNR `43.404@520`, SSIM `0.9709@420`, LPIPS `0.0170@500`
    - temporal: `rho_early 0.6372@380`, `t_arr 6.336s@400`
  - `m36_meetgrasp_multiobj_fixed_absimg_tdiff_1n4g_burst`
    - best PSNR `42.659@580`, SSIM `0.9495@400`, LPIPS `0.0228@580`
    - temporal: `rho_early 0.6272@300`, `t_arr 5.810s@540`
  - `m36_meetgrasp_highcap160x8x48_abs_w240_mcmse010_flatlr_2n4g` (running)
    - best so far PSNR `38.111@720`, SSIM `0.9384@640`, LPIPS `0.0803@720`
    - temporal weak (`rho_early 0.5596@80`) and still far below 128-dim runs.

- GRASP-vs-DRO baseline (from checkpoints):
  - PSNR `48.58`, SSIM `0.9795`, LPIPS `0.0021`, curve corr `0.9993`, `rho_early 0.7360`, `t_arr 4.729s`.

- Additional runtime observation from running high-capacity log:
  - at step 800, supervised distill weight is `0` (distill stop reached), metrics are poor (`PSNR 37.65`, `rho_early 0.5226`).
  - This run is effectively in post-distill drift mode and not competitive.

- Takeaways:
  - Best image-quality regime remains absolute distill with 128-dim model (`abs_w220` / `abs_w180`).
  - Temporal-diff heavy variants and current multiobjective probes did not beat the best absolute-distill image trajectory.
  - High-capacity warm-start trajectory is materially worse and currently low ROI.
  - Distill-release (weight going to 0 before end) appears harmful for this regime; keeping distill active through end is safer.

- Highest-ROI next action recommendation:
  - Stop `713715` high-capacity run and allocate slots to fresh 128-dim warm-start continuations from `m36_meetgrasp_abs_w220_mcmse010_flatlr_2n4g_best_model.pth` with:
    - no distill release (`stop_step >= max_steps` or unset),
    - long/fixed high distill pressure,
    - only small MC constraint,
    - one clean multiobjective variant with a new exp name (no resume collision).
  - Rationale: maximizes probability of closing remaining gap quickly while avoiding proven low-yield branches.

### 2026-02-16 Follow-up: plateau interpretation + new 8-slot break-plateau matrix launched
- User feedback:
  - do not abandon high-capacity too early if it has non-plateau trend,
  - verify whether multiobjective was truly active,
  - require substantive objective changes (not only “keep distill on”).

- Additional analysis completed:
  - Distill coverage at 36spf is **not** the bottleneck: `supervised_distill_valid_fraction_history` is `1.0` for all recent m36 meet-GRASP runs.
  - Multiobjective was active in multiobjective runs:
    - `m36_meetgrasp_multiobj_adapt_absimg_tdiff_1n4g_burst`: non-empty scale history, temporal scale ranged `1.224 -> 0.25`.
    - `m36_meetgrasp_multiobj_fixed_absimg_tdiff_1n4g_burst`: scale fixed at `1.0`.
  - Interpretation: adaptive multiobjective likely over-suppressed temporal term (collapsed to min scale), so failure is more likely settings/balance than a broken implementation.

- High-capacity decision:
  - Canceled prior high-cap run `713715` and replaced with a high-cap continuation variant using:
    - warm start from high-cap checkpoint,
    - MSE distill,
    - no distill release (`stop_step=2000`),
    - longer horizon (`max_steps=1600`).

- New config matrix created (all in `configs/deadline_20260216_break_plateau_matrix/`):
  - `m36_bk_absmse_nomc_w1_long_2n4g.yaml`
  - `m36_bk_absmse_mc005_w1_long_2n4g.yaml`
  - `m36_bk_absmae_nomc_w3_long_2n4g.yaml`
  - `m36_bk_absmse_nomc_freq005_2n4g.yaml`
  - `m36_bk_mobjfix_nomc_imgmse_tdw04_2n4g.yaml`
  - `m36_bk_mobjada_nomc_imgmse_tdw06_2n4g.yaml`
  - `m36_bk_absmse_mc010_lr5e5_2n4g.yaml`
  - `m36_bk_highcap_mse_norelease_2n4g.yaml`

- Submission actions:
  - initial submissions: `714035`..`714042`.
  - observed hardware failures on `k002/k003` (no GPU detected or ECC errors):
    - `714035` failed (`ProcessGroupNCCL ... no GPUs found`).
    - `714037` failed (`CUDA error: uncorrectable ECC error encountered`).
  - resubmitted failed jobs with GPU constraints (`a100|h100|h200`):
    - `714043` replacement for `m36_bk_absmae_nomc_w3_long_2n4g`
    - `714044` replacement for `m36_bk_absmse_mc010_lr5e5_2n4g`
  - proactively canceled and resubmitted unconstrained pending/running jobs with same constraint:
    - `714045` `m36_bk_absmse_nomc_freq005_2n4g`
    - `714046` `m36_bk_absmse_nomc_w1_long_2n4g`
    - `714047` `m36_bk_highcap_mse_norelease_2n4g`
    - `714048` `m36_bk_mobjada_nomc_imgmse_tdw06_2n4g`
    - `714049` `m36_bk_mobjfix_nomc_imgmse_tdw04_2n4g`

- Current queue occupancy after recovery:
  - standard slots: `8/8` (`RUNNING+PENDING`).
  - running at log time: `714036` (healthy early training logs), plus remaining jobs pending by priority/resources.

- Next decision trigger:
  - first eval at step 20 for at least 3 variants, then compare:
    - PSNR/SSIM trend vs best prior (`43.57 / 0.971`),
    - temporal `rho_early` and `t_arr` behavior,
    - whether multiobjective variants avoid early image collapse under new settings.

### 2026-02-16 Break-plateau campaign refresh (latest overlay + live runs)
- User request:
  - check the last campaign, refresh overlays, summarize what we learned.

- Fresh overlay generated:
  - `/net/projects2/annawoodard/experiments/m36_bk_overlay_20260216_180725/`
  - files:
    - `eval_metrics_overlay.png`
    - `eval_temporal_metrics_overlay.png`
    - `eval_temporal_metrics_student_vs_grasp_overlay.png`
    - `training_metrics_overlay.png`
    - `overlay_metric_tables.txt`

- Current active jobs (at refresh):
  - `714043` `m36_bk_absmae_nomc_w3_long_2n4g` (running)
  - `714404` `m36_bk_highcap_mse_norelease_1n4g_fastq` (running)
  - `714406` `m36_bk_mobjfix_nomc_imgmse_tdw04_1n4g_fastq` (running)

- Completed jobs from this campaign:
  - `714401` `m36_bk_absmse_mc010_lr5e5_1n4g_fastq`
  - `714402` `m36_bk_absmse_nomc_freq005_1n4g_fastq`
  - `714403` `m36_bk_absmse_nomc_w1_long_1n4g_fastq`
  - `714405` `m36_bk_mobjada_nomc_imgmse_tdw06_1n4g_fastq`

- Best-performing run so far in this campaign: `714043`.
  - Latest eval (`step1020`, from log):
    - Student-vs-DRO: SSIM `0.9742`, PSNR `44.2091`, LPIPS `0.0123`, MSE `0.0681`
    - Student-vs-GRASP: SSIM `0.9880`, PSNR `44.7587`, LPIPS `0.0131`, MSE `0.0573`
    - Student-vs-GRASP temporal: `rho_early 0.7188`, `t_arr error 9.0069s`, `t_peak error 9.6251s`
  - Comparison to prior best baseline (`713639`, `m36_meetgrasp_abs_w220_mcmse010_flatlr_2n4g`):
    - image metrics improved (PSNR ~`43.6 -> 44.2`, LPIPS ~`0.0162 -> 0.0123`)
    - early timing still problematic (`t_arr` remains much larger than desired and oscillatory).

- Other campaign outcomes:
  - `714402` and `714403` are near-identical trajectories; frequency weighting tweak showed little effect in this regime.
  - `714406` remains clearly below top trajectory (best checkpoint still at step 20, no competitive recovery yet).
  - `714404` (high-capacity) remains far below in image and temporal metrics (PSNR ~`38.8` at step 1000); not competitive currently.

- Key takeaways:
  - Absolute supervised distill with stronger weight (`714043`) gives the best image-quality gains so far.
  - Plateau was partially breakable for image metrics, but timing metrics (especially arrival-time error) remain unstable and weak.
  - Capacity increase alone did not help in this setup; high-cap run underperforms despite longer training.
  - Multiobjective variant needs rework/tuning before it can compete with the simple strong absolute-distill recipe.
