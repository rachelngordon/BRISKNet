#!/usr/bin/env python3
"""Plot BriskNet vs GRASP center comparisons for DRO data. Run: python3 -m inference.plot_brisknet_grasp_center_compare --help"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Dict, Tuple

import numpy as np
import torch
import yaml
import matplotlib
import h5py
from einops import rearrange

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from skimage.measure import find_contours, label

REPO_ROOT = Path(__file__).resolve().parents[1]
JOB_SCRIPTS_DIR = REPO_ROOT / "job-scripts"
if str(JOB_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(JOB_SCRIPTS_DIR))

from cluster_paths import apply_cluster_paths
from inference.eval import _resolve_baseline_frames, _load_tumor_mask
from model.radial import MCNUFFT
from inference.run_inference_new_dro import (
    NewDROMatDataset,
    _build_model,
    _grasp_np_to_torch,
    _load_weights,
    _prep_nufft_from_dro_traj,
    _resolve_eval_params,
)
from utils import prep_nufft, sliding_window_inference


PLOT_FONT_SIZES = {
    "title": 18,
    "label": 18,
    "tick": 16,
    "legend": 13,
    "image_title": 16,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot BRISKNet vs GRASP center frames + tumor ROI enhancement curves."
    )
    parser.add_argument("--exp_low", required=True, help="BRISKNet exp dir for low acceleration.")
    parser.add_argument("--exp_high", required=True, help="BRISKNet exp dir for high acceleration.")
    parser.add_argument("--config_low", help="Override config.yaml for low exp.")
    parser.add_argument("--config_high", help="Override config.yaml for high exp.")
    parser.add_argument("--ckpt_low", help="Override checkpoint for low exp.")
    parser.add_argument("--ckpt_high", help="Override checkpoint for high exp.")
    parser.add_argument(
        "--use_best_checkpoint",
        action="store_true",
        default=False,
        help="Use <exp>_best_model.pth if available (else fall back).",
    )
    parser.add_argument("--device", help="Torch device override (e.g., cuda:0).")
    parser.add_argument(
        "--dro_root",
        default="/net/scratch2/rachelgordon/dro_var_frames",
        help="DRO root (default: /net/scratch2/rachelgordon/dro_var_frames).",
    )
    parser.add_argument(
        "--dro_espirit_csmaps_dir",
        help="Override ESPIRiT csmaps dir (default: <dro_root>/csmaps_espirit).",
    )
    parser.add_argument(
        "--dro_espirit_grasp_dir",
        help=(
            "Override ESPIRiT GRASP dir (default: <dro_root>/espirit_grasp_recons_lam<lambda>)."
        ),
    )
    parser.add_argument(
        "--grasp_lambda",
        type=float,
        default=0.001,
        help="GRASP lambda value used to select ESPIRiT recon dir (default: 0.001).",
    )
    parser.add_argument(
        "--sample_id",
        help="Explicit DRO sample id (e.g., sample_09). Must exist for both frame counts.",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Index into the sorted intersection of sample ids (default: 0).",
    )
    parser.add_argument(
        "--split_key",
        help=(
            "Optional split key from data_split.json to filter DRO samples "
            "(e.g., val_dro, test_dro). If set, selection is restricted to that split."
        ),
    )
    parser.add_argument(
        "--split_file",
        help="Override split file path (default: config_low data.split_file or data/data_split.json).",
    )
    parser.add_argument(
        "--phase_index",
        type=int,
        help="Curriculum phase index override (default: last phase).",
    )
    parser.add_argument("--eval_spokes_low", type=int, help="Override low exp eval spokes.")
    parser.add_argument("--eval_frames_low", type=int, help="Override low exp eval frames.")
    parser.add_argument("--eval_spokes_high", type=int, help="Override high exp eval spokes.")
    parser.add_argument("--eval_frames_high", type=int, help="Override high exp eval frames.")
    parser.add_argument(
        "--baseline_mode",
        default="seconds",
        choices=("seconds", "fraction"),
        help="Baseline window selection mode.",
    )
    parser.add_argument(
        "--baseline_seconds",
        type=float,
        default=20.0,
        help="Baseline duration in seconds when baseline_mode=seconds.",
    )
    parser.add_argument(
        "--baseline_fraction",
        type=float,
        default=0.1,
        help="Baseline fraction when baseline_mode=fraction.",
    )
    parser.add_argument("--baseline_min_frames", type=int, default=1)
    parser.add_argument("--baseline_max_frames", type=int, default=10)
    parser.add_argument(
        "--total_scan_seconds",
        type=float,
        default=150.0,
        help="Total scan duration for time axis.",
    )
    parser.add_argument(
        "--dro_curve_mode",
        choices=("low", "high", "both", "off"),
        default="low",
        help="Which DRO GT curve(s) to plot: low, high, both, or off (default: high).",
    )
    parser.add_argument(
        "--dro_curve_from",
        choices=("low", "high"),
        default=None,
        help="Deprecated. Use --dro_curve_mode instead.",
    )
    parser.add_argument(
        "--out",
        default="brisknet_grasp_center_frame_compare.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--use_raw",
        action="store_true",
        help="Use raw fastMRI data instead of DRO (plots BRISKNet vs raw GRASP).",
    )
    parser.add_argument(
        "--raw_slice_idx",
        type=int,
        default=None,
        help="Override raw slice index (defaults to slice map or config eval raw_grasp_slice_idx).",
    )
    parser.add_argument(
        "--invert_colormap",
        action="store_true",
        help="Invert grayscale colormap for image panels.",
    )
    parser.add_argument(
        "--curves_only",
        action="store_true",
        help="Only plot enhancement curves (omit image panels and extra whitespace).",
    )
    return parser.parse_args()


def _load_config(exp_dir: str, config_override: str | None) -> dict:
    config_path = config_override or os.path.join(exp_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return apply_cluster_paths(config)


def _resolve_checkpoint(exp_dir: str, ckpt_override: str | None, use_best: bool) -> str:
    if ckpt_override:
        return ckpt_override
    exp_name = os.path.basename(exp_dir.rstrip("/"))
    if use_best:
        best_path = os.path.join(exp_dir, f"{exp_name}_best_model.pth")
        if os.path.exists(best_path):
            return best_path
    return os.path.join(exp_dir, f"{exp_name}_model.pth")


def _is_temporal_mamba(config: dict) -> bool:
    model_type = str(config.get("model", {}).get("name", "")).strip().lower()
    mamba_variant = str(config.get("model", {}).get("mamba", {}).get("variant", "")).strip().lower()
    return model_type in {
        "mambatemporal",
        "mamba_temporal",
        "temporalmamba",
    } or (
        model_type in {"mambarecon", "mamba_recon", "mamba"}
        and mamba_variant in {"temporal", "temporal_1d", "radial_temporal"}
    )


def _robust_window_multi(images, p_low=1, p_high=99.5):
    flat = []
    for img in images:
        if img is None:
            continue
        arr = np.asarray(img)
        flat.append(arr.ravel())
    if not flat:
        return 0.0, 1.0
    stacked = np.concatenate(flat)
    finite_vals = stacked[np.isfinite(stacked)]
    if finite_vals.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite_vals, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def _ensure_thw(arr: np.ndarray, num_frames: int) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got {arr.shape}")
    if arr.shape[0] == num_frames:
        return arr
    if arr.shape[1] == num_frames:
        return np.transpose(arr, (1, 0, 2))
    if arr.shape[2] == num_frames:
        return np.transpose(arr, (2, 0, 1))
    return arr


def _pick_tumor_mask(masks: Dict[str, np.ndarray]) -> Tuple[np.ndarray, str]:
    if not masks:
        return None, "none"
    if "malignant" in masks and np.any(masks["malignant"]):
        return masks["malignant"], "malignant"
    if "benign" in masks and np.any(masks["benign"]):
        return masks["benign"], "benign"
    union = None
    for v in masks.values():
        if v is None:
            continue
        if union is None:
            union = v.astype(bool)
        else:
            union = union | v.astype(bool)
    if union is not None and np.any(union):
        return union, "union"
    return None, "none"


def _baseline_subtract(curve: np.ndarray, n_baseline: int) -> np.ndarray:
    if curve.size == 0:
        return curve
    n_baseline = max(1, min(int(n_baseline), curve.size))
    baseline = np.nanmean(curve[:n_baseline])
    return curve - baseline


def _compute_roi_curve(mag_thw: np.ndarray, mask_hw: np.ndarray) -> np.ndarray:
    if mag_thw.ndim != 3:
        raise ValueError(f"Expected mag stack (T,H,W); got {mag_thw.shape}")
    T, H, W = mag_thw.shape
    if mask_hw.shape != (H, W):
        if mask_hw.T.shape == (H, W):
            mask_hw = mask_hw.T
        else:
            raise ValueError(f"Mask shape {mask_hw.shape} does not match image {(H, W)}")
    return np.array([mag_thw[t][mask_hw].mean() for t in range(T)])


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labeled = label(mask.astype(np.uint8), connectivity=1)
    if labeled.max() == 0:
        return mask
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest = counts.argmax()
    return labeled == largest


def _compute_crop_bbox(
    img_hw: np.ndarray,
    tumor_mask: np.ndarray | None,
    margin_frac: float = 0.08,
    min_size: int = 96,
) -> Tuple[int, int, int, int]:
    H, W = img_hw.shape
    mask = None
    if tumor_mask is not None and np.any(tumor_mask):
        mask = tumor_mask
    else:
        finite = np.asarray(img_hw)
        finite = finite[np.isfinite(finite)]
        finite = finite[finite > 0]
        if finite.size > 0:
            thresh = np.percentile(finite, 10)
            breast_mask = img_hw > thresh
            if breast_mask.any():
                mask = _largest_component(breast_mask)

    if mask is None or not np.any(mask):
        return 0, H, 0, W

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    cy = 0.5 * (y0 + y1)
    cx = 0.5 * (x0 + x1)
    size = max(y1 - y0, x1 - x0)
    size = max(size, min_size)
    size = int(round(size * (1 + 2 * margin_frac)))

    y0 = int(round(cy - size / 2))
    y1 = y0 + size
    x0 = int(round(cx - size / 2))
    x1 = x0 + size

    if y0 < 0:
        y1 = min(H, y1 - y0)
        y0 = 0
    if x0 < 0:
        x1 = min(W, x1 - x0)
        x0 = 0
    if y1 > H:
        y0 = max(0, y0 - (y1 - H))
        y1 = H
    if x1 > W:
        x0 = max(0, x0 - (x1 - W))
        x1 = W

    return y0, y1, x0, x1


def _run_single_inference(
    config: dict,
    ckpt_path: str,
    dataset: NewDROMatDataset,
    sample_id: str,
    device: torch.device,
    eval_spokes: int,
    eval_frames: int,
):
    H, W = config["data"]["height"], config["data"]["width"]
    samples = int(config["data"]["samples"])
    eval_chunk_size = config.get("evaluation", {}).get("chunk_size", eval_frames)
    eval_chunk_overlap = config.get("evaluation", {}).get("chunk_overlap", 0)

    # Trajectory + physics from DRO k-space file
    kspace_path = dataset._resolve_kspace_mat_path(sample_id)
    eval_ktraj, eval_dcomp, eval_nufft_ob, eval_adjnufft_ob, traj_samples = _prep_nufft_from_dro_traj(
        kspace_path,
        spokes_per_frame=int(eval_spokes),
        num_frames=int(eval_frames),
        expected_samples=samples,
        traj_method=config.get("data", {}).get("traj_method", "get_traj"),
    )
    if traj_samples != samples:
        raise ValueError(f"Traj samples ({traj_samples}) != config samples ({samples})")

    eval_ktraj = eval_ktraj.to(device)
    eval_dcomp = eval_dcomp.to(device)
    eval_nufft_ob = eval_nufft_ob.to(device)
    eval_adjnufft_ob = eval_adjnufft_ob.to(device)
    eval_physics = MCNUFFT(eval_nufft_ob, eval_adjnufft_ob, eval_ktraj, eval_dcomp)

    # Model
    block_dir = os.path.join(os.path.dirname(ckpt_path), "block_outputs")
    os.makedirs(block_dir, exist_ok=True)
    model = _build_model(config, device, block_dir)
    model = _load_weights(model, ckpt_path)

    # Load DRO sample (no raw)
    sim_img, smap, mask_dict = dataset._load_dro_mat(sample_id)
    if dataset.dro_csmaps_source == "espirit":
        smap = dataset._load_espirit_csmaps(sample_id, expected_coils=smap.shape[0])
    kspace = dataset._load_kspace_mat(sample_id)
    grasp_np, grasp_path = dataset._load_grasp_recon(sample_id)

    # Torch inputs
    dro_kspace = torch.from_numpy(kspace).to(torch.complex64).to(device)
    csmap = torch.from_numpy(smap).to(torch.complex64).to(device)

    # Acceleration encoding
    N_full = H * math.pi / 2
    accel = torch.tensor(
        [N_full / int(eval_ktraj.shape[1] / samples)], dtype=torch.float, device=device
    )
    acceleration_encoding = accel if config["model"].get("encode_acceleration", False) else None
    start_timepoint_index = (
        torch.tensor([0], dtype=torch.float, device=device)
        if config["model"].get("encode_time_index", False)
        else None
    )

    model_type_is_temporal_mamba = _is_temporal_mamba(config)
    eval_uses_sliding = bool(eval_frames > eval_chunk_size and not model_type_is_temporal_mamba)

    csmap = csmap.unsqueeze(0)
    
    with torch.no_grad():
        if eval_uses_sliding:
            x_recon, _ = sliding_window_inference(
                H,
                W,
                eval_frames,
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
                collect_adj_loss=False,
            )
        else:
            x_recon, *_ = model(
                dro_kspace,
                eval_physics,
                csmap,
                acceleration_encoding,
                start_timepoint_index,
                epoch="inference",
                norm=config["model"]["norm"],
            )

    # Convert simImg to torch for consistent rotation
    gt_torch = torch.from_numpy(sim_img).to(torch.float32)
    gt_torch = torch.stack([gt_torch, torch.zeros_like(gt_torch)], dim=0)  # (2,T,H,W)
    gt_torch = gt_torch.unsqueeze(0)  # (1,2,T,H,W)

    # GRASP to torch (2,H,T,W)
    grasp_torch = _grasp_np_to_torch(grasp_np).to(device)  # (2,H,T,W)
    grasp_torch = grasp_torch.unsqueeze(0)  # (1,2,H,T,W)

    # Rotate to match eval orientation
    x_recon = torch.rot90(x_recon, k=3, dims=[2, 3])
    gt_torch = torch.rot90(gt_torch, k=3, dims=[-2, -1])
    grasp_torch = torch.rot90(grasp_torch, k=3, dims=[2, 4])

    # Rotate masks
    masks_rot = {}
    for k, v in (mask_dict or {}).items():
        if v is None:
            continue
        masks_rot[k] = np.rot90(v.astype(bool), k=3)

    return {
        "x_recon": x_recon,
        "gt": gt_torch,
        "grasp": grasp_torch,
        "masks": masks_rot,
        "grasp_path": grasp_path,
        "eval_spokes": int(eval_spokes),
        "eval_frames": int(eval_frames),
    }


def _load_raw_csmaps(csmap_path: str) -> torch.Tensor:
    raw_csmaps = np.load(csmap_path)
    cs_t = torch.from_numpy(raw_csmaps)
    if cs_t.ndim == 4:
        cs_t = rearrange(cs_t, "c b h w -> b c h w")
    elif cs_t.ndim == 3:
        # Accept (C,H,W) or (H,W,C).
        if cs_t.shape[0] < cs_t.shape[-1]:
            cs_t = cs_t.unsqueeze(0)
        else:
            cs_t = cs_t.permute(2, 0, 1).unsqueeze(0)
    else:
        raise ValueError(f"Unexpected csmap shape {cs_t.shape} in {csmap_path}")
    cs_t = cs_t.to(torch.complex64)
    cs_t = torch.rot90(cs_t, k=2, dims=[-2, -1])
    return cs_t


def _load_raw_grasp(raw_grasp_path: str) -> torch.Tensor:
    raw_grasp = np.load(raw_grasp_path).squeeze()
    raw_grasp_t = torch.from_numpy(raw_grasp).permute(2, 0, 1)  # (T,H,W)
    raw_grasp_t = torch.stack([raw_grasp_t.real, raw_grasp_t.imag], dim=0)  # (2,T,H,W)
    raw_grasp_t = torch.flip(raw_grasp_t, dims=[-3])
    raw_grasp_t = torch.rot90(raw_grasp_t, k=1, dims=[-3, -1])
    return raw_grasp_t.unsqueeze(0)  # (1,2,T,H,W)


def _load_raw_bundle(
    dataset: NewDROMatDataset,
    sample_id: str,
    raw_kspace_root: str,
    dataset_key: str,
    eval_spokes: int,
    eval_frames: int,
    cluster: str,
    raw_slice_idx: int | None,
) -> Dict[str, object]:
    fastmri_id = dataset.get_fastmri_id(sample_id)
    patient_id = f"fastMRI_breast_{fastmri_id:03d}_2"
    slice_idx = dataset.slice_map.get(patient_id, None)
    if slice_idx is None or slice_idx < 0:
        slice_idx = raw_slice_idx if raw_slice_idx is not None else 95

    raw_kspace_path = os.path.join(raw_kspace_root, f"{patient_id}.h5")
    raw_csmap_path = os.path.join(
        os.path.dirname(raw_kspace_root),
        f"cs_maps/{patient_id}_cs_maps/cs_map_slice_{slice_idx:03d}.npy",
    )
    raw_grasp_path = os.path.join(
        os.path.dirname(raw_kspace_root),
        f"{patient_id}/grasp_recon_{eval_spokes}spf_{eval_frames}frames_slice{slice_idx}.npy",
    )

    with h5py.File(raw_kspace_path, "r") as f:
        if dataset_key not in f:
            raise KeyError(f"{raw_kspace_path} missing key '{dataset_key}'")
        raw_kspace_slice = np.asarray(f[dataset_key][slice_idx])

    N_spokes_prep = int(eval_spokes) * int(eval_frames)
    if raw_kspace_slice.shape[1] < N_spokes_prep:
        raise ValueError(
            f"Raw k-space spokes ({raw_kspace_slice.shape[1]}) < required ({N_spokes_prep}) "
            f"for spf={eval_spokes}, frames={eval_frames}."
        )
    ksp_redu = raw_kspace_slice[:, :N_spokes_prep, :]
    ksp_prep = np.swapaxes(ksp_redu, 0, 1)
    ksp_prep_shape = ksp_prep.shape
    ksp_prep = np.reshape(
        ksp_prep,
        [int(eval_frames), int(eval_spokes)] + list(ksp_prep_shape[1:]),
    )
    ksp_prep = torch.flip(torch.from_numpy(ksp_prep), dims=[-1])
    raw_kspace = rearrange(ksp_prep, "t sp c sam -> c (sp sam) t").to(torch.complex64)

    raw_csmaps = _load_raw_csmaps(raw_csmap_path)
    raw_grasp = _load_raw_grasp(raw_grasp_path)

    raw_tumor_mask = _load_tumor_mask(cluster, patient_id, slice_idx=slice_idx)
    masks = {}
    if raw_tumor_mask is not None:
        masks["malignant"] = raw_tumor_mask.astype(bool)

    return {
        "raw_kspace": raw_kspace,
        "raw_csmaps": raw_csmaps,
        "raw_grasp": raw_grasp,
        "masks": masks,
        "patient_id": patient_id,
        "slice_idx": int(slice_idx),
    }


def _run_single_inference_raw(
    config: dict,
    ckpt_path: str,
    dataset: NewDROMatDataset,
    sample_id: str,
    device: torch.device,
    eval_spokes: int,
    eval_frames: int,
    raw_slice_idx: int | None,
):
    samples = int(config["data"]["samples"])
    eval_chunk_size = config.get("evaluation", {}).get("chunk_size", eval_frames)
    eval_chunk_overlap = config.get("evaluation", {}).get("chunk_overlap", 0)

    cluster = config.get("experiment", {}).get("cluster", "Randi")
    raw_kspace_root = config["data"]["root_dir"]
    raw_bundle = _load_raw_bundle(
        dataset,
        sample_id,
        raw_kspace_root,
        config["data"]["dataset_key"],
        eval_spokes,
        eval_frames,
        cluster,
        raw_slice_idx,
    )

    raw_kspace = raw_bundle["raw_kspace"].to(device)
    raw_csmaps = raw_bundle["raw_csmaps"].to(device)

    N_samples = raw_kspace.shape[1] // int(eval_spokes)
    if N_samples <= 0:
        raise ValueError("Invalid samples/spoke computed from raw k-space.")

    raw_ktraj, raw_dcomp, raw_nufft_ob, raw_adjnufft_ob = prep_nufft(
        int(N_samples), int(eval_spokes), int(eval_frames), traj_method=config.get("data", {}).get("traj_method", "get_traj")
    )
    raw_ktraj = raw_ktraj.to(device)
    raw_dcomp = raw_dcomp.to(device)
    raw_nufft_ob = raw_nufft_ob.to(device)
    raw_adjnufft_ob = raw_adjnufft_ob.to(device)
    raw_physics = MCNUFFT(raw_nufft_ob, raw_adjnufft_ob, raw_ktraj, raw_dcomp)

    # Model
    block_dir = os.path.join(os.path.dirname(ckpt_path), "block_outputs")
    os.makedirs(block_dir, exist_ok=True)
    model = _build_model(config, device, block_dir)
    model = _load_weights(model, ckpt_path)

    acceleration_encoding = None
    if config["model"].get("encode_acceleration", False):
        N_full = raw_csmaps.shape[-2] * math.pi / 2
        accel = torch.tensor(
            [N_full / int(eval_spokes)], dtype=torch.float, device=device
        )
        acceleration_encoding = accel
    start_timepoint_index = (
        torch.tensor([0], dtype=torch.float, device=device)
        if config["model"].get("encode_time_index", False)
        else None
    )

    model_type_is_temporal_mamba = _is_temporal_mamba(config)
    eval_uses_sliding = bool(eval_frames > eval_chunk_size and not model_type_is_temporal_mamba)

    with torch.no_grad():
        if eval_uses_sliding:
            x_recon, _ = sliding_window_inference(
                raw_csmaps.shape[-2],
                raw_csmaps.shape[-1],
                eval_frames,
                raw_ktraj,
                raw_dcomp,
                raw_nufft_ob,
                raw_adjnufft_ob,
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
                collect_adj_loss=False,
            )
        else:
            x_recon, *_ = model(
                raw_kspace,
                raw_physics,
                raw_csmaps,
                acceleration_encoding,
                start_timepoint_index,
                epoch="inference",
                norm=config["model"]["norm"],
            )

    return {
        "x_recon": x_recon,
        "grasp": raw_bundle["raw_grasp"].to(device),
        "masks": raw_bundle["masks"],
        "eval_spokes": int(eval_spokes),
        "eval_frames": int(eval_frames),
        "patient_id": raw_bundle["patient_id"],
        "slice_idx": raw_bundle["slice_idx"],
    }


def main():
    args = parse_args()

    config_low = _load_config(args.exp_low, args.config_low)
    config_high = _load_config(args.exp_high, args.config_high)

    spf_low, frames_low = _resolve_eval_params(
        config_low, args.eval_spokes_low, args.eval_frames_low, args.phase_index
    )
    spf_high, frames_high = _resolve_eval_params(
        config_high, args.eval_spokes_high, args.eval_frames_high, args.phase_index
    )

    ckpt_low = _resolve_checkpoint(args.exp_low, args.ckpt_low, args.use_best_checkpoint)
    ckpt_high = _resolve_checkpoint(args.exp_high, args.ckpt_high, args.use_best_checkpoint)

    device_str = args.device or config_low.get("training", {}).get("device", "cuda")
    device = torch.device(device_str)

    dro_root = args.dro_root
    espirit_csmaps_dir = args.dro_espirit_csmaps_dir or os.path.join(dro_root, "csmaps_espirit")
    espirit_grasp_dir = args.dro_espirit_grasp_dir or os.path.join(
        dro_root, f"espirit_grasp_recons_lam{args.grasp_lambda:g}"
    )

    dataset_low = NewDROMatDataset(
        root_dir=dro_root,
        raw_kspace_path=config_low["data"]["root_dir"],
        model_type=config_low["model"]["name"],
        patient_ids=[],
        dataset_key=config_low["data"]["dataset_key"],
        spokes_per_frame=spf_low,
        num_frames=frames_low,
        dro_csmaps_source="espirit",
        espirit_csmaps_dir=espirit_csmaps_dir,
        espirit_grasp_recons_dir=espirit_grasp_dir,
        skip_raw_eval_if_invalid_slice=True,
    )
    dataset_high = NewDROMatDataset(
        root_dir=dro_root,
        raw_kspace_path=config_high["data"]["root_dir"],
        model_type=config_high["model"]["name"],
        patient_ids=[],
        dataset_key=config_high["data"]["dataset_key"],
        spokes_per_frame=spf_high,
        num_frames=frames_high,
        dro_csmaps_source="espirit",
        espirit_csmaps_dir=espirit_csmaps_dir,
        espirit_grasp_recons_dir=espirit_grasp_dir,
        skip_raw_eval_if_invalid_slice=True,
    )

    common_ids = sorted(set(dataset_low.sample_ids) & set(dataset_high.sample_ids))
    if not common_ids:
        raise ValueError("No overlapping DRO sample ids between low/high frame counts.")
    if args.split_key:
        split_file = (
            args.split_file
            or config_low.get("data", {}).get("split_file")
            or "data/data_split.json"
        )
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        with open(split_file, "r", encoding="utf-8") as f:
            splits = json.load(f)
        split_ids = splits.get(args.split_key) or []
        if not split_ids:
            raise ValueError(
                f"Split key '{args.split_key}' not found or empty in {split_file}."
            )
        common_ids = [sid for sid in common_ids if sid in split_ids]
        if not common_ids:
            raise ValueError(
                f"No overlapping DRO sample ids after filtering by '{args.split_key}'."
            )
    if args.sample_id:
        if args.split_key and args.sample_id not in common_ids:
            raise ValueError(
                f"Sample id {args.sample_id} not in split '{args.split_key}'."
            )
        if args.sample_id not in common_ids:
            raise ValueError(f"Sample id {args.sample_id} not found in both datasets.")
        sample_id = args.sample_id
    else:
        sample_idx = max(0, min(args.sample_idx, len(common_ids) - 1))
        sample_id = common_ids[sample_idx]

    print(f"Using sample_id: {sample_id}")
    print(f"Low: spf={spf_low}, frames={frames_low}, ckpt={ckpt_low}")
    print(f"High: spf={spf_high}, frames={frames_high}, ckpt={ckpt_high}")

    if args.use_raw:
        raw_slice_idx = args.raw_slice_idx
        if raw_slice_idx is None:
            raw_slice_idx = (
                config_low.get("evaluation", {}).get("raw_grasp_slice_idx")
                or config_high.get("evaluation", {}).get("raw_grasp_slice_idx")
                or 95
            )
        low_out = _run_single_inference_raw(
            config_low, ckpt_low, dataset_low, sample_id, device, spf_low, frames_low, raw_slice_idx
        )
        high_out = _run_single_inference_raw(
            config_high, ckpt_high, dataset_high, sample_id, device, spf_high, frames_high, raw_slice_idx
        )
        if args.dro_curve_from is not None:
            print("[Warn] --dro_curve_from ignored when --use_raw is set.")
    else:
        low_out = _run_single_inference(
            config_low, ckpt_low, dataset_low, sample_id, device, spf_low, frames_low
        )
        high_out = _run_single_inference(
            config_high, ckpt_high, dataset_high, sample_id, device, spf_high, frames_high
        )

    # Tumor mask (prefer malignant)
    tumor_mask, mask_key = _pick_tumor_mask(low_out["masks"])
    if tumor_mask is None or not np.any(tumor_mask):
        raise ValueError("No tumor ROI found in mask.")
    print(f"Using ROI mask: {mask_key}")

    # Convert to magnitude stacks (T,H,W)
    def mag_to_thw_brisk(x_recon: torch.Tensor, num_frames: int) -> np.ndarray:
        mag = torch.sqrt(x_recon[:, 0, ...] ** 2 + x_recon[:, 1, ...] ** 2).squeeze(0)
        mag = mag.permute(2, 0, 1).detach().cpu().numpy()
        return _ensure_thw(mag, num_frames)

    def mag_to_thw_gt(gt: torch.Tensor, num_frames: int) -> np.ndarray:
        mag = torch.sqrt(gt[:, 0, ...] ** 2 + gt[:, 1, ...] ** 2).squeeze(0)
        mag = mag.detach().cpu().numpy()
        return _ensure_thw(mag, num_frames)

    def mag_to_thw_grasp(grasp: torch.Tensor, num_frames: int) -> np.ndarray:
        mag = torch.sqrt(grasp[:, 0, ...] ** 2 + grasp[:, 1, ...] ** 2).squeeze(0)
        mag = mag.detach().cpu().numpy()
        if mag.ndim == 3 and mag.shape[1] == num_frames and mag.shape[0] != num_frames:
            # Match run_inference_new_dro debug canonicalization: (H, T, W) -> (T, H, W)
            mag = np.transpose(mag, (1, 0, 2))
        return _ensure_thw(mag, num_frames)

    brisk_low = mag_to_thw_brisk(low_out["x_recon"], frames_low)
    grasp_low = mag_to_thw_grasp(low_out["grasp"], frames_low)
    brisk_high = mag_to_thw_brisk(high_out["x_recon"], frames_high)
    grasp_high = mag_to_thw_grasp(high_out["grasp"], frames_high)
    gt_low = None if args.use_raw else mag_to_thw_gt(low_out["gt"], frames_low)
    gt_high = None if args.use_raw else mag_to_thw_gt(high_out["gt"], frames_high)

    # Time axes
    time_low = np.linspace(0, args.total_scan_seconds, frames_low)
    time_high = np.linspace(0, args.total_scan_seconds, frames_high)

    # Baseline frames
    nbase_low = _resolve_baseline_frames(
        num_frames=frames_low,
        time_points=time_low,
        baseline_mode=args.baseline_mode,
        baseline_seconds=args.baseline_seconds,
        baseline_fraction=args.baseline_fraction,
        baseline_min_frames=args.baseline_min_frames,
        baseline_max_frames=args.baseline_max_frames,
    )
    nbase_high = _resolve_baseline_frames(
        num_frames=frames_high,
        time_points=time_high,
        baseline_mode=args.baseline_mode,
        baseline_seconds=args.baseline_seconds,
        baseline_fraction=args.baseline_fraction,
        baseline_min_frames=args.baseline_min_frames,
        baseline_max_frames=args.baseline_max_frames,
    )

    # ROI curves (baseline-subtracted)
    curve_brisk_low = _baseline_subtract(_compute_roi_curve(brisk_low, tumor_mask), nbase_low)
    curve_grasp_low = _baseline_subtract(_compute_roi_curve(grasp_low, tumor_mask), nbase_low)
    curve_brisk_high = _baseline_subtract(_compute_roi_curve(brisk_high, tumor_mask), nbase_high)
    curve_grasp_high = _baseline_subtract(_compute_roi_curve(grasp_high, tumor_mask), nbase_high)
    dro_mode = "off" if args.use_raw else args.dro_curve_mode
    if args.dro_curve_from is not None:
        dro_mode = args.dro_curve_from

    curve_dro_low = None
    curve_dro_high = None
    if dro_mode in ("low", "both") and gt_low is not None:
        curve_dro_low = _baseline_subtract(_compute_roi_curve(gt_low, tumor_mask), nbase_low)
    if dro_mode in ("high", "both") and gt_high is not None:
        curve_dro_high = _baseline_subtract(_compute_roi_curve(gt_high, tumor_mask), nbase_high)

    # Center frames
    center_low = frames_low // 2
    center_high = frames_high // 2
    img_brisk_low = brisk_low[center_low]
    img_grasp_low = grasp_low[center_low]
    img_brisk_high = brisk_high[center_high]
    img_grasp_high = grasp_high[center_high]
    img_ref = img_grasp_low if args.use_raw else gt_low[center_low]

    # Crop to zoomed tumor/breast region.
    y0, y1, x0, x1 = _compute_crop_bbox(img_ref, tumor_mask)
    img_brisk_low = img_brisk_low[y0:y1, x0:x1]
    img_grasp_low = img_grasp_low[y0:y1, x0:x1]
    img_brisk_high = img_brisk_high[y0:y1, x0:x1]
    img_grasp_high = img_grasp_high[y0:y1, x0:x1]
    tumor_mask_crop = tumor_mask[y0:y1, x0:x1]

    vmin, vmax = _robust_window_multi(
        [img_brisk_low, img_grasp_low, img_brisk_high, img_grasp_high], p_low=1, p_high=99.5
    )
    contours = find_contours(tumor_mask_crop.astype(float), 0.5) if tumor_mask_crop.any() else []

    # Plot
    if args.curves_only:
        fig = plt.figure(figsize=(9, 4.5))
        fig.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.18)
        ax_curve = fig.add_subplot(1, 1, 1)
    else:
        fig = plt.figure(figsize=(14, 7))
        gs = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[1, 1.1])
        fig.subplots_adjust(left=0.03, right=0.995, top=0.96, bottom=0.08, wspace=0.02, hspace=0.08)

        titles = [
            f"BRISKNet {spf_low} SPF",
            f"GRASP {spf_low} SPF",
            f"BRISKNet {spf_high} SPF",
            f"GRASP {spf_high} SPF",
        ]
        images = [img_brisk_low, img_grasp_low, img_brisk_high, img_grasp_high]

        cmap = "gray_r" if args.invert_colormap else "gray"
        for i in range(4):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(images[i], cmap=cmap, vmin=vmin, vmax=vmax)
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=1.0, color="red")
            ax.set_title(titles[i], fontsize=PLOT_FONT_SIZES["image_title"])
            ax.axis("off")

        ax_curve = fig.add_subplot(gs[1, :])
    ax_curve.plot(
        time_low,
        curve_brisk_low,
        color="tab:red",
        linewidth=2,
        marker="o",
        markersize=4,
        label=f"BRISKNet {spf_low} SPF",
    )
    ax_curve.plot(
        time_low,
        curve_grasp_low,
        color="tab:blue",
        linewidth=2,
        linestyle="--",
        marker="o",
        markersize=4,
        label=f"GRASP {spf_low} SPF",
    )
    ax_curve.plot(
        time_high,
        curve_brisk_high,
        color="tab:orange",
        linewidth=2,
        marker="o",
        markersize=4,
        label=f"BRISKNet {spf_high} SPF",
    )
    ax_curve.plot(
        time_high,
        curve_grasp_high,
        color="tab:green",
        linewidth=2,
        linestyle="--",
        marker="o",
        markersize=4,
        label=f"GRASP {spf_high} SPF",
    )
    if curve_dro_low is not None and curve_dro_high is not None:
        ax_curve.plot(
            time_low,
            curve_dro_low,
            color="black",
            linewidth=2,
            linestyle=":",
            marker="o",
            markersize=4,
            label=f"DRO (GT) {spf_low} SPF",
        )
        ax_curve.plot(
            time_high,
            curve_dro_high,
            color="dimgray",
            linewidth=2,
            linestyle=":",
            marker="o",
            markersize=4,
            label=f"DRO (GT) {spf_high} SPF",
        )
    elif curve_dro_low is not None:
        ax_curve.plot(
            time_low,
            curve_dro_low,
            color="black",
            linewidth=2,
            linestyle=":",
            marker="o",
            markersize=4,
            label=f"DRO (GT) {spf_low} SPF",
        )
    elif curve_dro_high is not None:
        ax_curve.plot(
            time_high,
            curve_dro_high,
            color="black",
            linewidth=2,
            linestyle=":",
            marker="o",
            markersize=4,
            label=f"DRO (GT) {spf_high} SPF",
        )

    if not args.curves_only:
        # Mark the timepoints shown in the image panels.
        ax_curve.plot(
            time_low[center_low],
            curve_brisk_low[center_low],
            marker="*",
            markersize=14,
            color="tab:red",
            markeredgecolor="none",
            label="_nolegend_",
            zorder=5,
        )
        ax_curve.plot(
            time_low[center_low],
            curve_grasp_low[center_low],
            marker="*",
            markersize=14,
            color="tab:blue",
            markeredgecolor="none",
            label="_nolegend_",
            zorder=5,
        )
        ax_curve.plot(
            time_high[center_high],
            curve_brisk_high[center_high],
            marker="*",
            markersize=14,
            color="tab:orange",
            markeredgecolor="none",
            label="_nolegend_",
            zorder=5,
        )
        ax_curve.plot(
            time_high[center_high],
            curve_grasp_high[center_high],
            marker="*",
            markersize=14,
            color="tab:green",
            markeredgecolor="none",
            label="_nolegend_",
            zorder=5,
        )

    ax_curve.set_xlabel("Time (s)", fontsize=PLOT_FONT_SIZES["label"])
    ax_curve.set_ylabel("Baseline-Subtracted Signal", fontsize=PLOT_FONT_SIZES["label"])
    ax_curve.grid(True, linestyle="--", alpha=0.5)
    ax_curve.tick_params(axis="both", which="major", labelsize=PLOT_FONT_SIZES["tick"])
    handles, labels = ax_curve.get_legend_handles_labels()
    if not args.curves_only:
        star_handle = Line2D(
            [0],
            [0],
            marker="*",
            linestyle="None",
            color="k",
            markeredgecolor="none",
            markersize=12,
            label="Center Frame",
        )
        handles.append(star_handle)
        labels.append("Center Frame")
    legend_size = PLOT_FONT_SIZES["legend"]
    if args.curves_only:
        legend_size = max(9, int(PLOT_FONT_SIZES["legend"] * 0.8)) + 1
    ax_curve.legend(
        handles,
        labels,
        loc="upper left",
        ncol=3 if not args.curves_only else 2,
        fontsize=legend_size,
        frameon=True,
    )

    fig.savefig(args.out, dpi=200, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
