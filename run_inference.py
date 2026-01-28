import argparse
import json
import math
import os
import statistics
import time
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import rearrange
from radial_lsfp import from_torch_complex

from cluster_paths import apply_cluster_paths
from dataloader import SimulatedDataset
from eval import (
    eval_grasp,
    eval_sample,
    compute_ssdu_kspace_nmse,
    compute_ssdu_kspace_nmse_grasp,
    _load_tumor_mask,
    _load_slice_map,
    _resolve_plot_label,
)
from lsfpnet_encoding import ArtifactRemovalLSFPNet, LSFPNet
from radial_lsfp import MCNUFFT
from utils import (
    GRASPRecon_from_ktraj,
    prep_nufft,
    remove_module_prefix,
    set_seed,
    sliding_window_inference,
)


def _resolve_eval_params(config: dict, spokes: int, frames: int, phase_idx: int) -> Tuple[int, int]:
    """Pick evaluation spokes/frame and num_frames using overrides or curriculum."""
    if spokes and frames:
        return spokes, frames

    curriculum_cfg = config.get("training", {}).get("curriculum_learning", {})
    phases = curriculum_cfg.get("phases", [])
    if curriculum_cfg.get("enabled") and phases:
        # Default to the last phase unless the user specifies otherwise.
        phase_idx = len(phases) - 1 if phase_idx is None else phase_idx
        phase_idx = max(0, min(phase_idx, len(phases) - 1))
        phase = phases[phase_idx]
        return phase["eval_spokes_per_frame"], phase["eval_num_frames"]

    data_cfg = config["data"]
    return data_cfg["eval_spokes"], data_cfg["eval_timeframes"]


def _build_model(config: dict, device, block_dir: str):
    """Create the LSFP model and load weights."""
    initial_lambdas = {
        "lambda_L": config["model"]["lambda_L"],
        "lambda_S": config["model"]["lambda_S"],
        "lambda_spatial_L": config["model"]["lambda_spatial_L"],
        "lambda_spatial_S": config["model"]["lambda_spatial_S"],
        "gamma": config["model"]["gamma"],
        "lambda_step": config["model"]["lambda_step"],
    }

    lsfp_backbone = LSFPNet(
        LayerNo=config["model"]["num_layers"],
        lambdas=initial_lambdas,
        channels=config["model"]["channels"],
        style_dim=config["model"]["style_dim"],
        svd_mode=config["model"]["svd_mode"],
        use_lowk_dc=config["model"]["use_lowk_dc"],
        lowk_frac=config["model"]["lowk_frac"],
        lowk_alpha=config["model"]["lowk_alpha"],
        film_bounded=config["model"]["film_bounded"],
        film_gain=config["model"]["film_gain"],
        film_identity_init=config["model"]["film_identity_init"],
        svd_noise_std=config["model"]["svd_noise_std"],
        film_L=config["model"]["film_L"],
        kernel_size_L=config["model"].get("kernel_size_L", 3),
        kernel_size_S=config["model"].get("kernel_size_S", 3),
        activation_checkpointing=config["model"].get("activation_checkpointing", False),
        checkpoint_use_reentrant=config["model"].get("checkpoint_use_reentrant", False),
    )

    if config["model"]["encode_acceleration"] and config["model"]["encode_time_index"]:
        channels = 2
    else:
        channels = 1

    model = ArtifactRemovalLSFPNet(lsfp_backbone, block_dir, channels=channels).to(device)
    model.eval()
    return model


def _load_weights(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(remove_module_prefix(state_dict))
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on validation samples.")
    parser.add_argument("--exp_dir", required=True, help="Experiment directory location.")
    parser.add_argument("--config", help="Path to config.yaml (defaults to output/<exp>/config.yaml).")
    parser.add_argument("--checkpoint", help="Path to model checkpoint (defaults to output/<exp>/<exp>_model.pth).")
    parser.add_argument("--num_samples", type=int, help="Number of validation samples to evaluate (default: config value).")
    parser.add_argument("--device", default=None, help="Torch device to use (default: config training.device).")
    parser.add_argument("--eval_spokes", type=int, help="Override spokes per frame for inference.")
    parser.add_argument("--eval_frames", type=int, help="Override number of frames for inference.")
    parser.add_argument("--phase_index", type=int, help="Curriculum phase index to use for eval params (default: last).")
    parser.add_argument("--disable_ssdu", action="store_true", help="Skip SSDU NMSE computation to speed up inference.")
    parser.add_argument("--diagnostics", action="store_true", help="Enable diagnostic plots per sample.")
    parser.add_argument("--diag_topk", type=int, default=16, help="Top-K pixels to plot in diagnostic curves.")
    parser.add_argument("--diag_num_frames", type=int, default=6, help="Number of frames for heatmap diagnostics.")
    parser.add_argument(
        "--diag_ref",
        default="gt",
        choices=("gt", "grasp", "raw_grasp", "raw_recon"),
        help="Reference image source for diagnostics.",
    )
    parser.add_argument(
        "--diag_normalize",
        action="store_true",
        help="Normalize pixel curves by baseline (t=0) reference intensity.",
    )
    parser.add_argument("--seed", type=int, default=12, help="Random seed.")
    return parser.parse_args()


def _to_numpy_img(x: torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if torch.is_complex(x):
            x = torch.abs(x)
        x = x.float().numpy()
    else:
        x = np.asarray(x)
        if np.iscomplexobj(x):
            x = np.abs(x)
        x = x.astype(np.float32, copy=False)
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == 2:
        x = x[None, ...]
    return x.astype(np.float32, copy=False)


def _normalize_mask(mask, T: int, H: int, W: int):
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = np.asarray(mask)
    if mask_np.ndim == 3 and mask_np.shape[0] == 1:
        mask_np = mask_np[0]
    if mask_np.ndim == 2:
        mask_stack = np.broadcast_to(mask_np, (T, H, W))
    elif mask_np.ndim == 3 and mask_np.shape[0] == T:
        mask_stack = mask_np
    else:
        return None
    return mask_stack > 0


def _robust_limits(values: np.ndarray, low_q: float, high_q: float):
    finite_vals = values[np.isfinite(values)]
    if finite_vals.size == 0:
        return 0.0, 1.0
    return (float(np.percentile(finite_vals, low_q)), float(np.percentile(finite_vals, high_q)))


def _select_timepoints(ref_roi_mean: np.ndarray, num_frames: int):
    T = ref_roi_mean.shape[0]
    if T == 0:
        return []
    num_frames = max(1, min(num_frames, T))
    evenly = np.linspace(0, T - 1, num_frames).round().astype(int).tolist()
    peak = int(np.nanargmax(ref_roi_mean)) if np.isfinite(ref_roi_mean).any() else 0
    selected = sorted(set(evenly + [peak]))
    return selected


def _overlay_plot(background, overlay, mask, vmin, vmax, cmap, alpha, outpath, title, cbar_label=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(background, cmap="gray")
    overlay_masked = np.ma.array(overlay, mask=~mask)
    im = ax.imshow(overlay_masked, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _contour_mask(ax, mask):
    ax.contour(mask.astype(float), levels=[0.5], colors="yellow", linewidths=0.8)


def _mask_bbox(mask: np.ndarray, pad: int, H: int, W: int):
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return (0, H, 0, W)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, H)
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, W)
    return (y0, y1, x0, x1)


def save_diagnostics(
    sample_dir: str,
    brisk,
    reference,
    mask,
    args,
    diag_subdir: str = "diagnostics",
    ref_label: str | None = None,
    brisk_label: str | None = None,
):
    diag_dir = os.path.join(sample_dir, diag_subdir)
    os.makedirs(diag_dir, exist_ok=True)

    brisk_np = _to_numpy_img(brisk).squeeze()
    brisk_np = np.abs(brisk_np[0] + 1j * brisk_np[1])
    brisk_np = rearrange(brisk_np, 'h w t -> t h w')

    ref_np = _to_numpy_img(reference).squeeze()
    ref_np = np.abs(ref_np[0] + 1j * ref_np[1])

    if ref_np.shape[0] == 320:
        ref_np = rearrange(ref_np, 'h t w -> t h w')

    if brisk_np.shape != ref_np.shape:
        raise ValueError(f"Diagnostics shape mismatch: brisk {brisk_np.shape} vs ref {ref_np.shape}")

    T, H, W = ref_np.shape
    mask_stack = _normalize_mask(mask, T, H, W)
    if mask_stack is None:
        raise ValueError(f"Diagnostics unsupported mask shape: {np.asarray(mask).shape}")

    mask_union = np.any(mask_stack, axis=0)
    if not np.any(mask_union):
        raise ValueError("Diagnostics tumor mask is empty.")

    ref_roi_mean = np.array(
        [
            np.nanmean(ref_np[t][mask_stack[t]])
            if np.any(mask_stack[t]) else np.nan
            for t in range(T)
        ],
        dtype=np.float32,
    )
    baseline_t = 0
    if not np.isfinite(ref_roi_mean[baseline_t]):
        baseline_t = int(np.nanargmax(ref_roi_mean)) if np.isfinite(ref_roi_mean).any() else 0

    background = ref_np[baseline_t]
    pad = max(4, int(0.05 * max(H, W)))
    y0, y1, x0, x1 = _mask_bbox(mask_union, pad, H, W)

    ref_vals = ref_np[mask_stack]
    vmin_ref, vmax_ref = _robust_limits(ref_vals, 1, 99)

    diff = brisk_np - ref_np
    diff_vals = diff[mask_stack]
    diff_abs = np.abs(diff_vals[np.isfinite(diff_vals)])
    diff_lim = float(np.percentile(diff_abs, 99)) if diff_abs.size else 1.0
    diff_lim = max(diff_lim, 1e-6)

    timepoints = _select_timepoints(ref_roi_mean, args.diag_num_frames)
    if ref_label is None:
        ref_label = {
            "gt": "Ground Truth",
            "grasp": "GRASP",
            "raw_grasp": "Raw GRASP",
            "raw_recon": "Raw Recon",
        }.get(args.diag_ref, "Reference")
    if brisk_label is None:
        brisk_label = "BRISKNet"
    if not timepoints:
        return
    nrows = len(timepoints)
    fig, axes = plt.subplots(nrows, 3, figsize=(12, 4 * nrows), squeeze=False)
    for row_idx, t in enumerate(timepoints):
        mask_t = mask_stack[t][y0:y1, x0:x1]
        bg_crop = background[y0:y1, x0:x1]
        ref_crop = ref_np[t][y0:y1, x0:x1]
        brisk_crop = brisk_np[t][y0:y1, x0:x1]
        diff_crop = diff[t][y0:y1, x0:x1]
        overlays = (ref_crop, brisk_crop, diff_crop)
        cmaps = ("magma", "magma", "coolwarm")
        vmins = (vmin_ref, vmin_ref, -diff_lim)
        vmaxs = (vmax_ref, vmax_ref, diff_lim)
        titles = (
            f"{ref_label} tumor intensity (t={t})",
            f"{brisk_label} tumor intensity (t={t})",
            f"{brisk_label} − {ref_label} (t={t})",
        )
        cbar_labels = (
            f"{ref_label} intensity (a.u.)",
            f"{brisk_label} intensity (a.u.)",
            f"{brisk_label} − {ref_label} (a.u.)",
        )
        for col_idx in range(3):
            ax = axes[row_idx][col_idx]
            ax.imshow(bg_crop, cmap="gray")
            overlay_masked = np.ma.array(overlays[col_idx], mask=~mask_t)
            im = ax.imshow(overlay_masked, cmap=cmaps[col_idx], vmin=vmins[col_idx], vmax=vmaxs[col_idx], alpha=0.6)
            ax.set_title(titles[col_idx], fontsize=9)
            ax.axis("off")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(cbar_labels[col_idx], fontsize=8)
            cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(diag_dir, "heatmaps_triptych_grid.png"), dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)

    exp_name = args.exp_dir.split('/')[-1]

    # Resolve config/checkpoint paths and load config.
    config_path = args.config or os.path.join(args.exp_dir, "config.yaml")
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        best_ckpt_path = os.path.join(args.exp_dir, f"{exp_name}_best_model.pth")
        if os.path.exists(best_ckpt_path):
            ckpt_path = best_ckpt_path
        else:
            ckpt_path = os.path.join(args.exp_dir, f"{exp_name}_model.pth")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = apply_cluster_paths(config)

    device = torch.device(args.device or config["training"]["device"])
    rescale = config.get("evaluation", {}).get("rescale", True)
    raw_grasp_slice_idx = config.get("evaluation", {}).get("raw_grasp_slice_idx", 95)
    cluster = config.get("experiment", {}).get("cluster", "Randi")

    # Where to save inference outputs.
    output_dir = os.path.join(config["experiment"]["output_dir"], exp_name)
    inference_dir = os.path.join(output_dir, f"inference_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(inference_dir, exist_ok=True)

    # Dataset setup.
    with open(config["data"]["split_file"], "r") as fp:
        splits = json.load(fp)

    val_ids = splits.get("val_dro") or splits.get("val") or []

    N_spokes_eval, N_time_eval = _resolve_eval_params(
        config, spokes=args.eval_spokes, frames=args.eval_frames, phase_idx=args.phase_index
    )

    data_dir = config["data"]["root_dir"]
    model_type = config["model"]["name"]
    traj_method = config.get("data", {}).get("traj_method", "trajGR")

    val_dataset = SimulatedDataset(
        # root_dir=config["evaluation"]["simulated_dataset_path"],
        root_dir="/net/scratch2/rachelgordon/dro_dataset_frontpad",
        raw_kspace_path=data_dir,
        model_type=model_type,
        patient_ids=val_ids,
        dataset_key=config["data"]["dataset_key"],
        spokes_per_frame=N_spokes_eval,
        num_frames=N_time_eval,
        traj_method=traj_method,
        grasp_slice_idx=raw_grasp_slice_idx,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["dataloader"]["batch_size"],
        shuffle=False,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=True,
    )

    num_samples = args.num_samples or config.get("evaluation", {}).get("num_samples", len(val_dataset))
    num_samples = min(num_samples, len(val_dataset))

    # Prep physics for inference.
    N_samples = config["data"]["samples"]
    H, W = config["data"]["height"], config["data"]["width"]
    N_full = H * math.pi / 2

    eval_ktraj, eval_dcomp, eval_nufft_ob, eval_adjnufft_ob = prep_nufft(
        N_samples, N_spokes_eval, N_time_eval, traj_method=traj_method
    )
    eval_ktraj = eval_ktraj.to(device)
    eval_dcomp = eval_dcomp.to(device)
    eval_nufft_ob = eval_nufft_ob.to(device)
    eval_adjnufft_ob = eval_adjnufft_ob.to(device)
    eval_physics = MCNUFFT(eval_nufft_ob, eval_adjnufft_ob, eval_ktraj, eval_dcomp)

    eval_chunk_size = config.get("evaluation", {}).get("chunk_size", N_time_eval)
    eval_chunk_overlap = config.get("evaluation", {}).get("chunk_overlap", 0)
    compute_ssdu = config.get("evaluation", {}).get("compute_ssdu", True)
    if args.disable_ssdu:
        compute_ssdu = False

    ssdu_k_folds = config.get("evaluation", {}).get("ssdu_k_folds", 4)
    ssdu_grasp_k_folds = config.get("evaluation", {}).get("ssdu_grasp_k_folds", ssdu_k_folds)
    ssdu_weighting = config.get("evaluation", {}).get("ssdu_weighting", "sqrt_dcomp")

    # Build and load model.
    block_dir = os.path.join(output_dir, "block_outputs")
    os.makedirs(block_dir, exist_ok=True)
    model = _build_model(config, device, block_dir)
    model = _load_weights(model, ckpt_path)

    acceleration_val = torch.tensor([N_full / int(eval_ktraj.shape[1] / config["data"]["samples"])], dtype=torch.float, device=device)

    results = []
    raw_results = []
    grasp_results = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader, total=num_samples, desc="Inference on validation")):
            if idx >= num_samples:
                break

            (
                dro_kspace,
                csmap,
                ground_truth,
                dro_grasp_img,
                mask,
                grasp_path,
                raw_kspace,
                raw_grasp_img,
                raw_csmaps,
            ) = batch

            csmap = csmap.squeeze(0).to(device)
            ground_truth = ground_truth.to(device)
            dro_grasp_img = dro_grasp_img.to(device)
            dro_kspace = dro_kspace.squeeze(0).to(device)
            raw_kspace = raw_kspace.squeeze(0).to(device)
            raw_grasp_img = raw_grasp_img.to(device)
            raw_csmaps = raw_csmaps.squeeze(0).to(device)

            acceleration_encoding = acceleration_val if config["model"]["encode_acceleration"] else None
            start_timepoint_index = torch.tensor([0], dtype=torch.float, device=device) if config["model"]["encode_time_index"] else None

            if N_time_eval > eval_chunk_size:
                x_recon, _ = sliding_window_inference(
                    H,
                    W,
                    N_time_eval,
                    eval_ktraj,
                    eval_dcomp,
                    eval_nufft_ob,
                    eval_adjnufft_ob,
                    eval_chunk_size,
                    eval_chunk_overlap,
                    dro_kspace,
                    csmap,
                    acceleration_encoding,
                    start_timepoint_index,
                    model,
                    epoch="inference",
                    device=device,
                    norm=config["model"]["norm"],
                )
                raw_x_recon, _ = sliding_window_inference(
                    H,
                    W,
                    N_time_eval,
                    eval_ktraj,
                    eval_dcomp,
                    eval_nufft_ob,
                    eval_adjnufft_ob,
                    eval_chunk_size,
                    eval_chunk_overlap,
                    raw_kspace,
                    raw_csmaps,
                    acceleration_encoding,
                    start_timepoint_index,
                    model,
                    epoch="inference",
                    device=device,
                    norm=config["model"]["norm"],
                )
            else:
                x_recon, *_ = model(
                    dro_kspace, eval_physics, csmap, acceleration_encoding, start_timepoint_index, epoch="inference", norm="none"#config["model"]["norm"]
                )
                raw_x_recon, *_ = model(
                    raw_kspace, eval_physics, raw_csmaps, acceleration_encoding, start_timepoint_index, epoch="inference", norm="none"#config["model"]["norm"]
                )

            sample_dir = os.path.join(inference_dir, f"sample_{idx:02d}")
            os.makedirs(sample_dir, exist_ok=True)
            label = f"sample{idx:02d}"

            if args.diagnostics:
                ref_map = {
                    "gt": ground_truth,
                    "grasp": dro_grasp_img,
                    "raw_grasp": raw_grasp_img,
                    "raw_recon": raw_x_recon,
                }
                ref_choice = ref_map.get(args.diag_ref)
                if ref_choice is None:
                    raise ValueError(f"Unknown diag_ref '{args.diag_ref}'.")
                diag_mask = None
                if isinstance(mask, dict):
                    if "malignant" in mask:
                        diag_mask = mask["malignant"]
                        mask_np = diag_mask.detach().cpu().numpy() if isinstance(diag_mask, torch.Tensor) else np.asarray(diag_mask)
                        if not np.any(mask_np):
                            diag_mask = None
                else:
                    diag_mask = mask
                if diag_mask is not None:
                    save_diagnostics(
                        sample_dir=sample_dir,
                        brisk=x_recon,
                        reference=ref_choice,
                        mask=diag_mask,
                        args=args,
                    )

                _, patient_id = _resolve_plot_label(label, grasp_path)
                slice_map = _load_slice_map()
                resolved_slice_idx = slice_map.get(patient_id, raw_grasp_slice_idx)
                raw_tumor_mask = None
                if resolved_slice_idx is not None and resolved_slice_idx >= 0:
                    raw_tumor_mask = _load_tumor_mask(cluster, patient_id, slice_idx=resolved_slice_idx)
                if raw_tumor_mask is not None and np.any(raw_tumor_mask):
                    save_diagnostics(
                        sample_dir=sample_dir,
                        brisk=raw_x_recon,
                        reference=raw_grasp_img,
                        mask=raw_tumor_mask,
                        args=args,
                        diag_subdir="diagnostics_raw",
                        ref_label="Raw GRASP",
                        brisk_label="BRISKNet (raw)",
                    )

            ssdu_result = {}
            ssdu_grasp_result = {}
            if compute_ssdu:
                ssdu_chunk_size = eval_chunk_size if N_time_eval > eval_chunk_size else None
                ssdu_result = compute_ssdu_kspace_nmse(
                    model,
                    raw_kspace,
                    raw_csmaps,
                    eval_ktraj,
                    eval_dcomp,
                    eval_nufft_ob,
                    eval_adjnufft_ob,
                    spokes_per_frame=int(N_spokes_eval),
                    K_folds=ssdu_k_folds,
                    baseline_weighting=ssdu_weighting,
                    device=device,
                    acceleration_encoding=acceleration_encoding,
                    start_timepoint_index=start_timepoint_index,
                    norm=config["model"]["norm"],
                    epoch="inference",
                    chunk_size=ssdu_chunk_size,
                    chunk_overlap=eval_chunk_overlap,
                )
                ssdu_grasp_result = compute_ssdu_kspace_nmse_grasp(
                    lambda y_used, ktraj_used, dcomp_used, csmap, samples_per_spoke: GRASPRecon_from_ktraj(
                        csmap,
                        y_used,
                        ktraj_used,
                        samples_per_spoke,
                        device=None,
                    ),
                    raw_kspace,
                    raw_csmaps,
                    eval_ktraj,
                    eval_dcomp,
                    eval_nufft_ob,
                    eval_adjnufft_ob,
                    spokes_per_frame=int(N_spokes_eval),
                    K_folds=ssdu_grasp_k_folds,
                    orientation_transform="raw_grasp",
                    baseline_weighting=ssdu_weighting,
                    device=device,
                )

            dro_metrics = eval_sample(
                dro_kspace,
                csmap,
                ground_truth,
                x_recon,
                eval_physics,
                mask,
                dro_grasp_img,
                acceleration_val,
                int(N_spokes_eval),
                sample_dir,
                label,
                device,
                cluster,
                dro_eval=True,
                grasp_path=grasp_path,
                rescale=rescale,
            )

            grasp_metrics = eval_grasp(
                dro_kspace,
                csmap,
                ground_truth,
                dro_grasp_img,
                eval_physics,
                device,
                sample_dir,
                dro_eval=True,
            )

            raw_dc_mse, raw_dc_mae, _ = eval_sample(
                raw_kspace,
                raw_csmaps,
                ground_truth,
                raw_x_recon,
                eval_physics,
                mask,
                raw_grasp_img,
                acceleration_val,
                int(N_spokes_eval),
                sample_dir,
                f"{label}_raw",
                device,
                cluster,
                dro_eval=False,
                grasp_path=grasp_path,
                raw_slice_idx=raw_grasp_slice_idx,
                rescale=rescale,
            )
            raw_grasp_dc_mse, raw_grasp_dc_mae = eval_grasp(
                raw_kspace,
                raw_csmaps,
                ground_truth,
                raw_grasp_img,
                eval_physics,
                device,
                sample_dir,
                dro_eval=False,
            )

            ssim, psnr, mse, lpips, dc_mse, dc_mae, recon_corr, grasp_corr, temporal_metrics = dro_metrics
            grasp_ssim, grasp_psnr, grasp_mse, grasp_lpips, grasp_dc_mse, grasp_dc_mae = grasp_metrics

            results.append(
                dict(
                    sample=label,
                    ssim=ssim,
                    psnr=psnr,
                    mse=mse,
                    lpips=lpips,
                    dc_mse=dc_mse,
                    dc_mae=dc_mae,
                    recon_corr=recon_corr,
                    grasp_corr=grasp_corr,
                    **(temporal_metrics or {}),
                )
            )
            grasp_results.append(
                dict(
                    sample=label,
                    ssim=grasp_ssim,
                    psnr=grasp_psnr,
                    mse=grasp_mse,
                    lpips=grasp_lpips,
                    dc_mse=grasp_dc_mse,
                    dc_mae=grasp_dc_mae,
                )
            )
            raw_results.append(
                dict(
                    sample=label,
                    raw_dc_mse=raw_dc_mse,
                    raw_dc_mae=raw_dc_mae,
                    raw_grasp_dc_mse=raw_grasp_dc_mse,
                    raw_grasp_dc_mae=raw_grasp_dc_mae,
                    raw_ssdu_nmse=ssdu_result.get("ssdu_nmse_mean"),
                    raw_grasp_ssdu_nmse=ssdu_grasp_result.get("ssdu_nmse_mean"),
                )
            )

    # Save metrics.
    metrics_path = os.path.join(inference_dir, "metrics.csv")
    with open(metrics_path, "w") as f:
        headers = [
            "sample",
            "dl_ssim",
            "dl_psnr",
            "dl_mse",
            "dl_lpips",
            "dl_dc_mse",
            "dl_dc_mae",
            "dl_recon_corr",
            "grasp_corr",
            "grasp_ssim",
            "grasp_psnr",
            "grasp_mse",
            "grasp_lpips",
            "grasp_dc_mse",
            "grasp_dc_mae",
            "raw_dc_mse",
            "raw_dc_mae",
            "raw_grasp_dc_mse",
            "raw_grasp_dc_mae",
            "raw_ssdu_nmse",
            "raw_grasp_ssdu_nmse",
        ]
        f.write(",".join(headers) + "\n")
        for dro_row, grasp_row, raw_row in zip(results, grasp_results, raw_results):
            row = [
                dro_row["sample"],
                f"{dro_row['ssim']:.6f}",
                f"{dro_row['psnr']:.6f}",
                f"{dro_row['mse']:.6f}",
                f"{dro_row['lpips']:.6f}",
                f"{dro_row['dc_mse']:.6f}",
                f"{dro_row['dc_mae']:.6f}",
                "" if dro_row["recon_corr"] is None else f"{dro_row['recon_corr']:.6f}",
                "" if dro_row["grasp_corr"] is None else f"{dro_row['grasp_corr']:.6f}",
                f"{grasp_row['ssim']:.6f}",
                f"{grasp_row['psnr']:.6f}",
                f"{grasp_row['mse']:.6f}",
                f"{grasp_row['lpips']:.6f}",
                f"{grasp_row['dc_mse']:.6f}",
                f"{grasp_row['dc_mae']:.6f}",
                f"{raw_row['raw_dc_mse']:.6f}",
                f"{raw_row['raw_dc_mae']:.6f}",
                f"{raw_row['raw_grasp_dc_mse']:.6f}",
                f"{raw_row['raw_grasp_dc_mae']:.6f}",
                "" if raw_row.get("raw_ssdu_nmse") is None else f"{raw_row['raw_ssdu_nmse']:.6f}",
                "" if raw_row.get("raw_grasp_ssdu_nmse") is None else f"{raw_row['raw_grasp_ssdu_nmse']:.6f}",
            ]
            f.write(",".join(row) + "\n")

    metric_names = [
        "curve_corr",
        "curve_mae",
        "early_corr",
        "early_mae",
        "ttae_sec",
        "wash_in_slope_err",
        "iauc10_err",
        "peak_err",
        "ttpeak_err_sec",
    ]
    for label, prefix in (("malignant", ""), ("benign", "benign_")):
        for subset in ("all", "top10", "top20"):
            keys = [
                f"{prefix}{model}_{subset}_{metric}"
                for model in ("dl", "grasp")
                for metric in metric_names
            ]
            temporal_metrics_path = os.path.join(
                inference_dir, f"metrics_temporal_{label}_{subset}.csv"
            )
            with open(temporal_metrics_path, "w") as f:
                f.write(",".join(["sample"] + keys) + "\n")
                for dro_row in results:
                    row = [dro_row["sample"]]
                    for key in keys:
                        value = dro_row.get(key, np.nan)
                        if value is None or (isinstance(value, float) and np.isnan(value)):
                            row.append("")
                        else:
                            row.append(f"{value:.6f}")
                    f.write(",".join(row) + "\n")

    def _mean_std(values, key):
        vals = [v.get(key) for v in values if v.get(key) is not None and np.isfinite(v.get(key))]
        if not vals:
            return None, None
        mean = sum(vals) / len(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return mean, std

    dl_summary = {
        "ssim": _mean_std(results, "ssim"),
        "psnr": _mean_std(results, "psnr"),
        "mse": _mean_std(results, "mse"),
        "lpips": _mean_std(results, "lpips"),
        "dc_mse": _mean_std(results, "dc_mse"),
        "dc_mae": _mean_std(results, "dc_mae"),
        "recon_corr": _mean_std(results, "recon_corr"),
        "grasp_corr": _mean_std(results, "grasp_corr"),
    }

    grasp_summary = {
        "ssim": _mean_std(grasp_results, "ssim"),
        "psnr": _mean_std(grasp_results, "psnr"),
        "mse": _mean_std(grasp_results, "mse"),
        "lpips": _mean_std(grasp_results, "lpips"),
        "dc_mse": _mean_std(grasp_results, "dc_mse"),
        "dc_mae": _mean_std(grasp_results, "dc_mae"),
    }

    def _format_mean_std(mean, std):
        if mean is None:
            return ""
        return f"{mean:.4f} ± {std:.4f}"

    def _format_mean_std_precise(mean, std):
        if mean is None:
            return ""
        return f"{mean:.4e} ± {std:.4e}"

    recon_corr_str = _format_mean_std(*dl_summary["recon_corr"])
    grasp_corr_str = _format_mean_std(*dl_summary["grasp_corr"])

    raw_summary = {
        "raw_dc_mse": _mean_std(raw_results, "raw_dc_mse"),
        "raw_dc_mae": _mean_std(raw_results, "raw_dc_mae"),
        "raw_grasp_dc_mse": _mean_std(raw_results, "raw_grasp_dc_mse"),
        "raw_grasp_dc_mae": _mean_std(raw_results, "raw_grasp_dc_mae"),
        "raw_ssdu_nmse": _mean_std(raw_results, "raw_ssdu_nmse"),
        "raw_grasp_ssdu_nmse": _mean_std(raw_results, "raw_grasp_ssdu_nmse"),
    }

    print("=== Inference Summary (averaged over samples) ===")
    print("Image Metrics")
    print(
        "DL   -> "
        f"SSIM: {_format_mean_std(*dl_summary['ssim'])}, "
        f"PSNR: {_format_mean_std(*dl_summary['psnr'])}, "
        f"MSE: {_format_mean_std(*dl_summary['mse'])}, "
        f"LPIPS: {_format_mean_std(*dl_summary['lpips'])}, "
        f"EC Corr (DL): {recon_corr_str}, "
        f"EC Corr (GRASP): {grasp_corr_str}"
    )
    print(
        "GRASP-> "
        f"SSIM: {_format_mean_std(*grasp_summary['ssim'])}, "
        f"PSNR: {_format_mean_std(*grasp_summary['psnr'])}, "
        f"MSE: {_format_mean_std(*grasp_summary['mse'])}, "
        f"LPIPS: {_format_mean_std(*grasp_summary['lpips'])}"
    )
    print("K-space Metrics")
    print(
        "DRO  -> "
        f"DL DC_MSE: {_format_mean_std(*dl_summary['dc_mse'])}, "
        f"DL DC_MAE: {_format_mean_std(*dl_summary['dc_mae'])}, "
        f"GRASP DC_MSE: {_format_mean_std(*grasp_summary['dc_mse'])}, "
        f"GRASP DC_MAE: {_format_mean_std(*grasp_summary['dc_mae'])}"
    )
    print(
        "RAW  -> "
        f"DL DC_MSE: {_format_mean_std_precise(*raw_summary['raw_dc_mse'])}, "
        f"DL DC_MAE: {_format_mean_std_precise(*raw_summary['raw_dc_mae'])}, "
        f"GRASP DC_MSE: {_format_mean_std_precise(*raw_summary['raw_grasp_dc_mse'])}, "
        f"GRASP DC_MAE: {_format_mean_std_precise(*raw_summary['raw_grasp_dc_mae'])}, "
        f"DL SSDU NMSE: {_format_mean_std_precise(*raw_summary['raw_ssdu_nmse'])}, "
        f"GRASP SSDU NMSE: {_format_mean_std_precise(*raw_summary['raw_grasp_ssdu_nmse'])}"
    )
    def _has_any_metric(keys):
        return any(
            v.get(key) is not None and np.isfinite(v.get(key))
            for v in results
            for key in keys
        )

    def _print_temporal_table(label, prefix):
        if not _has_any_metric([
            f"{prefix}{model}_{subset}_{metric}"
            for model in ("dl", "grasp")
            for subset in ("all", "top10", "top20")
            for metric in metric_names
        ]):
            return
        print(label)
        header = f"{'Subset':<8} {'Metric':<22} {'DL':<16} {'GRASP':<16}"
        print(header)
        print("-" * len(header))
        for subset in ("all", "top10", "top20"):
            for metric in metric_names:
                dl_key = f"{prefix}dl_{subset}_{metric}"
                grasp_key = f"{prefix}grasp_{subset}_{metric}"
                dl_val = _format_mean_std(*_mean_std(results, dl_key))
                grasp_val = _format_mean_std(*_mean_std(results, grasp_key))
                print(f"{subset:<8} {metric:<22} {dl_val:<16} {grasp_val:<16}")

    print("----- Temporal Fidelity Metrics (mean ± std) -----")
    _print_temporal_table("Malignant", prefix="")
    _print_temporal_table("Benign", prefix="benign_")
    print(f"Inference complete. Results saved to {inference_dir}")


if __name__ == "__main__":
    main()
