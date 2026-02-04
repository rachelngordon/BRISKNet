# PR Notes (eval-debug branch)

## Why this PR exists

We were seeing suspiciously high DRO PSNR (e.g., ~50 dB for GRASP at 36 spokes/frame) and inconsistent BRISKNet-vs-GRASP evaluation. This PR focuses on making inference/eval **consistent, inspectable, and debuggable**, and on adding **foreground-masked metrics** so background pixels don’t dominate.

## Big changes

### Inference
- Added `--output_root` so inference results can be written to a different location than the experiment directory (useful when `--exp_dir` is read-only).
- Default checkpoint selection now prefers the best checkpoint (`--use_best_checkpoint` default; `--use_last_checkpoint` available).
- Added/debug-friendly table formatting in the terminal summary and renamed method label `DL` → `BRISKNet`.
- Default DRO inference settings now use ESPIRiT simulation + ESPIRiT coil maps + noise `0.05` unless overridden:
  - `--dro_sim_source espirit --dro_csmaps_source espirit --dro_noise_level 0.05`
- Added optional debug artifacts:
  - `--save_debug_arrays` writes per-sample `debug_arrays_mag.npz` containing aligned magnitude arrays (GT/BRISKNet/GRASP and optional ZF).
  - `--compute_zf_baseline` computes the adjoint/ZF baseline for reference.
- Added “first sample” debug prints (paths, shapes, magnitude ranges) when debug flags are enabled.

### Evaluation
- Added foreground-masked metrics (`PSNR_FG`, `MSE_FG`, `fg_fraction`) using DRO tissue masks when available, otherwise a GT support heuristic.
- Added best-fit complex scalar DC diagnostics in k-space (`*_dc_mse_bestfit`, `*_dc_scale_abs`) and compute them unconditionally.
- Guarded Pearson correlation against constant-input edge cases and made `tight_layout` warning-safe.
- Fixed a GRASP best-fit image-gain reporting bug due to axis order mismatches.
- Standardized plot labels (`GRASP` instead of `GRASP Recon`) and changed the time-series comparison figure to use a shared per-frame intensity window (so global scaling differences aren’t hidden by per-image auto-windowing).

### Training/Eval loop consistency
- Aligned `train_zf.py` DRO evaluation defaults with inference defaults so train-time eval matches `run_inference.py` when configs omit these knobs.

### Safety / warnings
- Switched checkpoint loads to prefer `torch.load(..., weights_only=True)` when supported (with fallback), reducing future warning noise and tightening security posture.

## Main conclusions so far

- Whole-image PSNR can look “too high” on DRO because:
  - the intensity range used for PSNR is large, and
  - background dominates the image area.
  Foreground-masked metrics (`PSNR_FG`) are a better sanity check.
- The large jump in BRISKNet quality after using:
  - `--dro_sim_source espirit --dro_csmaps_source espirit --dro_noise_level 0.05`
  suggests earlier failures were largely **forward-model / coil-map domain mismatch**, not an inherent “GRASP always wins at 36 spf” phenomenon.
- K-space best-fit gains near 1.0 indicate we are not looking at a trivial global gain bug when the ESPIRiT settings match.

## How to reproduce (example)

```bash
python run_inference.py \
  --exp_dir /home/rachelgordon/mri_recon/radial-breast-ddei/output/mc_36spf_baseline \
  --output_root /home/annawoodard/radial-breast-ddei/output \
  --disable_ssdu --num_samples 15 \
  --save_debug_arrays
```
