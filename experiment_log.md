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
