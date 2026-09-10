"""Diagnose GRASP-to-torchkbnufft orientation consistency on one raw case.

Run:
    python3 inference/diagnose_grasp_orientation.py \
        --config configs/config_8spf_sampling_arrshift.yaml \
        --split-key test_dro \
        --sample-index 0 \
        --device cuda:0

This script is intentionally isolated from the inference pipeline. It reconstructs
GRASP from the used spokes in one SSDU fold, sweeps the eight spatial dihedral
transforms, and measures consistency on both used and held-out spokes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import torch
import yaml
from einops import rearrange

from cluster_paths import apply_cluster_paths
from model.radial import MCNUFFT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep the eight spatial orientations of a fold-specific GRASP "
            "reconstruction and rank them by k-space correlation/NMSE."
        )
    )
    parser.add_argument(
        "--exp-dir",
        "--exp_dir",
        help="Experiment directory; config defaults to <exp-dir>/config.yaml.",
    )
    parser.add_argument("--config", help="Config YAML path.")
    parser.add_argument(
        "--patient-id",
        help=(
            "Raw patient ID, e.g. fastMRI_breast_010_2. If omitted, resolve it "
            "from --split-key and --sample-index."
        ),
    )
    parser.add_argument(
        "--split-key",
        default="test_dro",
        help="Config split key used when --patient-id is omitted (default: test_dro).",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Zero-based index within --split-key (default: 0).",
    )
    parser.add_argument(
        "--mapping-csv",
        default=str(REPO_ROOT / "data/split/DROSubID_vs_fastMRIbreastID.csv"),
        help="DRO-to-fastMRI mapping CSV.",
    )
    parser.add_argument(
        "--slice-map-csv",
        default=str(REPO_ROOT / "data/split/largest_tumor_slices.csv"),
        help="Patient-to-slice mapping CSV.",
    )
    parser.add_argument(
        "--slice-index",
        type=int,
        help="Raw slice index. Defaults to the slice map, then config value, then 95.",
    )
    parser.add_argument(
        "--raw-kspace-root",
        help="Override config data.root_dir (directory containing patient HDF5 files).",
    )
    parser.add_argument("--dataset-key", help="Override config data.dataset_key.")
    parser.add_argument(
        "--csmap-path",
        help="Override the inferred ESPIRiT sensitivity-map .npy path.",
    )
    parser.add_argument(
        "--spokes-per-frame",
        "--eval-spokes",
        type=int,
        help="Override evaluation spokes per frame.",
    )
    parser.add_argument(
        "--phase-index",
        type=int,
        help="Curriculum phase used to resolve spokes/frame (default: last phase).",
    )
    parser.add_argument(
        "--total-spokes",
        type=int,
        help="Number of raw spokes to use (default: config data.total_spokes or 288).",
    )
    parser.add_argument(
        "--traj-method",
        choices=("trajGR", "get_traj"),
        help="Override config data.traj_method.",
    )
    parser.add_argument(
        "--k-folds",
        type=int,
        default=2,
        help="Number of interleaved SSDU folds (default: 2).",
    )
    parser.add_argument(
        "--fold-index",
        type=int,
        default=0,
        help="Zero-based fold to diagnose (default: 0).",
    )
    parser.add_argument(
        "--all-folds",
        action="store_true",
        help="Run every valid fold instead of only --fold-index.",
    )
    parser.add_argument(
        "--weighting",
        choices=("sqrt_dcomp", "none"),
        default="sqrt_dcomp",
        help="Metric weighting, matching SSDU inference by default.",
    )
    parser.add_argument("--grasp-lambda", type=float, default=0.001)
    parser.add_argument("--grasp-max-iter", type=int, default=10)
    parser.add_argument("--grasp-rho", type=float, default=0.1)
    parser.add_argument(
        "--device",
        help="Torch device (default: config training.device).",
    )
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument(
        "--output-json",
        help="Optional path for machine-readable per-transform results.",
    )
    args = parser.parse_args()

    if args.config is None and args.exp_dir is None:
        parser.error("one of --config or --exp-dir is required")
    if args.k_folds < 2:
        parser.error("--k-folds must be at least 2")
    if args.grasp_lambda < 0:
        parser.error("--grasp-lambda must be non-negative")
    if args.grasp_max_iter < 1:
        parser.error("--grasp-max-iter must be at least 1")
    if args.grasp_rho <= 0:
        parser.error("--grasp-rho must be positive")
    return args


def _load_config(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    config_path = (
        Path(args.config)
        if args.config
        else Path(args.exp_dir).expanduser() / "config.yaml"
    )
    config_path = config_path.expanduser().resolve()
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    return apply_cluster_paths(config), config_path


def _resolve_eval_spokes(
    config: dict[str, Any],
    override: int | None,
    phase_index: int | None,
) -> int:
    if override is not None:
        return int(override)

    curriculum = config.get("training", {}).get("curriculum_learning", {})
    phases = curriculum.get("phases", [])
    if curriculum.get("enabled") and phases:
        idx = len(phases) - 1 if phase_index is None else int(phase_index)
        idx = max(0, min(idx, len(phases) - 1))
        return int(phases[idx]["eval_spokes_per_frame"])
    return int(config["data"]["eval_spokes"])


def _load_dro_mapping(path: Path) -> dict[int, int]:
    mapping: dict[int, int] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                mapping[int(row["DRO"])] = int(row["fastMRIbreast"])
            except (KeyError, TypeError, ValueError):
                continue
    if not mapping:
        raise ValueError(f"No usable DRO mappings found in {path}.")
    return mapping


def _normalize_patient_id(patient_id: str) -> str:
    patient_id = patient_id.removesuffix(".h5")
    if re.fullmatch(r"fastMRI_breast_\d{3}", patient_id):
        return f"{patient_id}_2"
    return patient_id


def _resolve_patient_id(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[str, str]:
    if args.patient_id:
        patient_id = _normalize_patient_id(args.patient_id)
        return patient_id, f"--patient-id {patient_id}"

    split_path = Path(config["data"]["split_file"]).expanduser()
    with split_path.open() as stream:
        splits = json.load(stream)
    samples = splits.get(args.split_key)
    if not samples:
        raise KeyError(f"Split '{args.split_key}' is absent or empty in {split_path}.")
    if args.sample_index < 0 or args.sample_index >= len(samples):
        raise IndexError(
            f"--sample-index {args.sample_index} is outside split "
            f"'{args.split_key}' with {len(samples)} entries."
        )

    sample_id = str(samples[args.sample_index])
    if sample_id.startswith("fastMRI_breast_"):
        return _normalize_patient_id(sample_id), sample_id

    match = re.search(r"_sub(\d+)$", sample_id)
    if match is None:
        raise ValueError(
            f"Cannot map split entry '{sample_id}' to a raw patient ID. "
            "Pass --patient-id explicitly."
        )
    dro_id = int(match.group(1))
    mapping = _load_dro_mapping(Path(args.mapping_csv).expanduser())
    if dro_id not in mapping:
        raise KeyError(f"DRO subject {dro_id} is absent from {args.mapping_csv}.")
    return f"fastMRI_breast_{mapping[dro_id]:03d}_2", sample_id


def _load_slice_map(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    mapping: dict[str, int] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            patient_id = row.get("fastMRI_breast_id")
            slice_idx = row.get("largest_slice_idx")
            if patient_id is None or slice_idx is None:
                continue
            try:
                mapping[patient_id.removesuffix(".nii")] = int(slice_idx)
            except ValueError:
                continue
    return mapping


def _resolve_slice_index(
    args: argparse.Namespace,
    config: dict[str, Any],
    patient_id: str,
) -> tuple[int, str]:
    if args.slice_index is not None:
        return int(args.slice_index), "command line"

    slice_map = _load_slice_map(Path(args.slice_map_csv).expanduser())
    if patient_id in slice_map:
        return int(slice_map[patient_id]), str(args.slice_map_csv)

    fallback = int(config.get("evaluation", {}).get("raw_grasp_slice_idx", 95))
    return fallback, "config/fallback"


def _load_raw_case(
    raw_root: Path,
    dataset_key: str,
    patient_id: str,
    slice_index: int,
    total_spokes: int,
    spokes_per_frame: int,
    csmap_path_override: str | None,
) -> tuple[torch.Tensor, torch.Tensor, int, Path, Path]:
    kspace_path = raw_root / f"{patient_id}.h5"
    if not kspace_path.is_file():
        raise FileNotFoundError(f"Raw k-space file not found: {kspace_path}")

    with h5py.File(kspace_path, "r") as h5_file:
        if dataset_key not in h5_file:
            raise KeyError(f"{kspace_path} does not contain dataset '{dataset_key}'.")
        dataset = h5_file[dataset_key]
        if slice_index < 0 or slice_index >= dataset.shape[0]:
            raise IndexError(
                f"Slice {slice_index} is outside {kspace_path}:{dataset_key} "
                f"with {dataset.shape[0]} slices."
            )
        raw_kspace = torch.as_tensor(dataset[slice_index])

    if raw_kspace.ndim != 3:
        raise ValueError(
            f"Expected raw k-space shape (coils, spokes, samples), got "
            f"{tuple(raw_kspace.shape)}."
        )
    if raw_kspace.shape[1] < total_spokes:
        raise ValueError(
            f"Requested {total_spokes} spokes, but {kspace_path} has "
            f"{raw_kspace.shape[1]}."
        )
    if total_spokes % spokes_per_frame != 0:
        raise ValueError(
            f"total_spokes={total_spokes} is not divisible by "
            f"spokes_per_frame={spokes_per_frame}."
        )

    frames = total_spokes // spokes_per_frame
    coils, _, samples_per_spoke = raw_kspace.shape
    kspace = raw_kspace[:, :total_spokes, :].permute(1, 0, 2)
    kspace = kspace.reshape(
        frames,
        spokes_per_frame,
        coils,
        samples_per_spoke,
    )
    kspace = torch.flip(kspace, dims=[-1])
    kspace = rearrange(kspace, "t sp c sam -> c (sp sam) t").to(torch.complex64)

    csmap_path = (
        Path(csmap_path_override).expanduser()
        if csmap_path_override
        else raw_root.parent
        / "cs_maps"
        / f"{patient_id}_cs_maps"
        / f"cs_map_slice_{slice_index:03d}.npy"
    )
    if not csmap_path.is_file():
        raise FileNotFoundError(f"Coil sensitivity map not found: {csmap_path}")

    csmaps_np = np.load(csmap_path)
    csmaps = torch.from_numpy(csmaps_np)
    if csmaps.ndim == 4 and csmaps.shape[1] == 1:
        csmaps = rearrange(csmaps, "c b h w -> b c h w")
    elif csmaps.ndim == 3:
        csmaps = csmaps.unsqueeze(0)
    else:
        raise ValueError(
            f"Expected csmaps shape (coils,1,H,W) or (coils,H,W), got "
            f"{tuple(csmaps.shape)}."
        )
    csmaps = torch.rot90(csmaps.to(torch.complex64), k=2, dims=(-2, -1))

    if csmaps.shape[1] != coils:
        raise ValueError(
            f"Coil count mismatch: k-space has {coils}, csmaps have {csmaps.shape[1]}."
        )
    expected_size = samples_per_spoke // 2
    if tuple(csmaps.shape[-2:]) != (expected_size, expected_size):
        raise ValueError(
            "Current project NUFFT construction assumes square images with size "
            f"samples_per_spoke/2={expected_size}, but csmaps have spatial shape "
            f"{tuple(csmaps.shape[-2:])}."
        )
    return kspace, csmaps, samples_per_spoke, kspace_path, csmap_path


def _build_fold_indices(
    spokes_per_frame: int,
    samples_per_spoke: int,
    k_folds: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    effective_folds = min(int(k_folds), int(spokes_per_frame))
    if effective_folds < 2:
        raise ValueError("At least two SSDU folds are required.")

    samples_per_frame = spokes_per_frame * samples_per_spoke
    sample_offsets = torch.arange(samples_per_spoke, device=device)
    folds: list[tuple[torch.Tensor, torch.Tensor]] = []
    for fold_index in range(effective_folds):
        held_spokes = torch.arange(
            fold_index,
            spokes_per_frame,
            effective_folds,
            device=device,
        )
        if spokes_per_frame - held_spokes.numel() < 2:
            continue
        held_indices = (
            held_spokes[:, None] * samples_per_spoke + sample_offsets[None, :]
        ).reshape(-1)
        held_mask = torch.zeros(
            samples_per_frame,
            dtype=torch.bool,
            device=device,
        )
        held_mask[held_indices] = True
        used_indices = (~held_mask).nonzero(as_tuple=False).squeeze(-1)
        folds.append((held_indices, used_indices))
    if not folds:
        raise ValueError("No valid folds leave at least two reconstruction spokes.")
    return folds


def _orientation_variants(
    image_thw: torch.Tensor,
) -> list[tuple[str, torch.Tensor]]:
    variants: list[tuple[str, torch.Tensor]] = []
    for flipped in (False, True):
        base = torch.flip(image_thw, dims=(-2,)) if flipped else image_thw
        prefix = "flip_h_" if flipped else ""
        for k in range(4):
            name = f"{prefix}rot{k * 90}"
            variants.append((name, torch.rot90(base, k=k, dims=(-2, -1))))
    return variants


def _consistency_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    dcomp: torch.Tensor,
    weighting: str,
) -> dict[str, float]:
    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction/target shape mismatch: "
            f"{tuple(prediction.shape)} vs {tuple(target.shape)}."
        )
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("Prediction or target contains non-finite values.")

    if weighting == "sqrt_dcomp":
        weight = torch.sqrt(torch.abs(dcomp)).unsqueeze(0)
    else:
        weight = torch.ones((), device=prediction.device)

    weighted_prediction = (weight * prediction).reshape(-1).to(torch.complex128)
    weighted_target = (weight * target).reshape(-1).to(torch.complex128)
    prediction_energy = torch.sum(torch.abs(weighted_prediction) ** 2).real
    target_energy = torch.sum(torch.abs(weighted_target) ** 2).real
    if prediction_energy <= 0 or target_energy <= 0:
        raise ValueError("Prediction or target has zero weighted energy.")

    inner_product = torch.sum(torch.conj(weighted_prediction) * weighted_target)
    correlation = torch.abs(inner_product) / torch.sqrt(
        prediction_energy * target_energy
    )
    correlation = torch.clamp(correlation, min=0.0, max=1.0)
    scale = inner_product / prediction_energy
    unscaled_nmse = (
        torch.sum(torch.abs(weighted_prediction - weighted_target) ** 2)
        / target_energy
    )
    matched_nmse = (
        torch.sum(torch.abs(scale * weighted_prediction - weighted_target) ** 2)
        / target_energy
    )
    theoretical_nmse = 1.0 - correlation**2

    return {
        "correlation": float(correlation.real.item()),
        "nmse_unscaled": float(unscaled_nmse.real.item()),
        "nmse_scale_matched": float(matched_nmse.real.item()),
        "nmse_from_correlation": float(theoretical_nmse.real.item()),
        "scale_abs": float(torch.abs(scale).item()),
        "scale_phase_rad": float(torch.angle(scale).item()),
        "prediction_norm": float(torch.sqrt(prediction_energy).item()),
        "target_norm": float(torch.sqrt(target_energy).item()),
    }


def _sigpy_device(torch_device: torch.device) -> Any:
    import sigpy as sp

    if torch_device.type == "cuda":
        return sp.Device(torch_device.index if torch_device.index is not None else 0)
    return sp.Device(-1)


def _grasp_reconstruct(
    csmaps: torch.Tensor,
    kspace: torch.Tensor,
    ktraj: torch.Tensor,
    samples_per_spoke: int,
    device: Any,
    lamda: float,
    max_iter: int,
    rho: float,
) -> np.ndarray:
    from sigpy.mri import app

    if kspace.ndim == 3:
        if kspace.shape[1] % samples_per_spoke != 0:
            raise ValueError(
                "GRASP k-space length is not divisible by samples_per_spoke."
            )
        spokes = kspace.shape[1] // samples_per_spoke
        kspace = kspace.reshape(
            kspace.shape[0],
            spokes,
            samples_per_spoke,
            kspace.shape[2],
        )
    elif kspace.ndim != 4:
        raise ValueError(f"Unsupported GRASP k-space shape: {tuple(kspace.shape)}")
    kspace_np = (
        kspace.permute(3, 0, 1, 2)
        .unsqueeze(1)
        .unsqueeze(3)
        .detach()
        .cpu()
        .numpy()
    )

    if csmaps.ndim == 3:
        csmaps = csmaps.unsqueeze(0)
    csmaps_np = (
        rearrange(csmaps, "b c h w -> c b h w").detach().cpu().numpy()
    )

    from utils import _ktraj_to_sigpy_coord

    trajectory_np = _ktraj_to_sigpy_coord(
        ktraj,
        samples_per_spoke,
        image_shape=tuple(int(size) for size in csmaps.shape[-2:]),
    )

    reconstruction = app.HighDimensionalRecon(
        kspace_np,
        csmaps_np,
        combine_echo=False,
        lamda=lamda,
        coord=trajectory_np,
        regu="TV",
        regu_axes=[0],
        max_iter=max_iter,
        solver="ADMM",
        rho=rho,
        device=device,
        show_pbar=False,
        verbose=False,
    ).run()
    if hasattr(reconstruction, "get"):
        reconstruction = reconstruction.get()
    return np.squeeze(np.asarray(reconstruction))


def _configure_numba_cache() -> None:
    cache_root = Path(tempfile.gettempdir()) / f"brisknet-{os.getuid()}"
    cache_root.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("NUMBA_CACHE_DIR"):
        numba_cache = cache_root / "numba"
        numba_cache.mkdir(parents=True, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = str(numba_cache)
    if not os.environ.get("MPLCONFIGDIR"):
        matplotlib_cache = cache_root / "matplotlib"
        matplotlib_cache.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)


@torch.inference_mode()
def _run_fold(
    fold_index: int,
    held_indices: torch.Tensor,
    used_indices: torch.Tensor,
    kspace: torch.Tensor,
    csmaps: torch.Tensor,
    ktraj: torch.Tensor,
    dcomp: torch.Tensor,
    nufft_ob: torch.nn.Module,
    adjnufft_ob: torch.nn.Module,
    samples_per_spoke: int,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    y_used = kspace[:, used_indices, :]
    y_held = kspace[:, held_indices, :]
    ktraj_used = ktraj[:, used_indices, :]
    ktraj_held = ktraj[:, held_indices, :]
    dcomp_used = dcomp[used_indices, :]
    dcomp_held = dcomp[held_indices, :]

    print(
        f"\nReconstructing fold {fold_index}: "
        f"{used_indices.numel() // samples_per_spoke} used spokes/frame, "
        f"{held_indices.numel() // samples_per_spoke} held spokes/frame"
    )
    image_np = _grasp_reconstruct(
        csmaps,
        y_used,
        ktraj_used,
        samples_per_spoke,
        device=_sigpy_device(device),
        lamda=args.grasp_lambda,
        max_iter=args.grasp_max_iter,
        rho=args.grasp_rho,
    )
    image_thw = torch.as_tensor(image_np)
    if not torch.is_complex(image_thw):
        raise ValueError("GRASP returned a non-complex reconstruction.")
    if image_thw.ndim == 2 and kspace.shape[-1] == 1:
        image_thw = image_thw.unsqueeze(0)
    if image_thw.ndim != 3 or image_thw.shape[0] != kspace.shape[-1]:
        raise ValueError(
            "Expected GRASP output in production shape (T,H,W), got "
            f"{tuple(image_thw.shape)} for T={kspace.shape[-1]}."
        )
    image_thw = image_thw.to(device=device, dtype=torch.complex64)

    physics_used = MCNUFFT(nufft_ob, adjnufft_ob, ktraj_used, dcomp_used)
    physics_held = MCNUFFT(nufft_ob, adjnufft_ob, ktraj_held, dcomp_held)
    expected_spatial_shape = tuple(csmaps.shape[-2:])

    results: list[dict[str, Any]] = []
    for transform_name, transformed_thw in _orientation_variants(image_thw):
        spatial_shape = tuple(transformed_thw.shape[-2:])
        if spatial_shape != expected_spatial_shape:
            results.append(
                {
                    "fold": fold_index,
                    "transform": transform_name,
                    "status": "skipped_shape_mismatch",
                    "image_shape": list(spatial_shape),
                    "expected_shape": list(expected_spatial_shape),
                }
            )
            continue

        image_hwt = transformed_thw.permute(1, 2, 0).contiguous()
        y_hat_used = physics_used(False, image_hwt, csmaps.to(image_hwt.dtype))
        y_hat_held = physics_held(False, image_hwt, csmaps.to(image_hwt.dtype))
        results.append(
            {
                "fold": fold_index,
                "transform": transform_name,
                "status": "ok",
                "used": _consistency_metrics(
                    y_hat_used,
                    y_used,
                    dcomp_used,
                    args.weighting,
                ),
                "held": _consistency_metrics(
                    y_hat_held,
                    y_held,
                    dcomp_held,
                    args.weighting,
                ),
            }
        )
    return results


def _print_results(results: list[dict[str, Any]]) -> None:
    valid = [row for row in results if row["status"] == "ok"]
    if not valid:
        raise RuntimeError("Every transform was skipped due to shape mismatch.")

    print(
        "\nTransform              used rho  used NMSE*  held rho  held NMSE*  "
        "|alpha held|"
    )
    print("-" * 81)
    for row in sorted(valid, key=lambda item: item["used"]["correlation"], reverse=True):
        print(
            f"{row['transform']:<22}"
            f"{row['used']['correlation']:>9.5f}"
            f"{row['used']['nmse_scale_matched']:>12.5f}"
            f"{row['held']['correlation']:>10.5f}"
            f"{row['held']['nmse_scale_matched']:>12.5f}"
            f"{row['held']['scale_abs']:>14.5g}"
        )
    print("NMSE* is after the optimal weighted complex scalar.")

    best = max(valid, key=lambda item: item["used"]["correlation"])
    identity = next(
        (row for row in valid if row["transform"] == "rot0"),
        None,
    )
    max_identity_error = max(
        abs(
            row[subset]["nmse_scale_matched"]
            - row[subset]["nmse_from_correlation"]
        )
        for row in valid
        for subset in ("used", "held")
    )

    print(
        f"\nBest self-consistency transform: {best['transform']} "
        f"(used rho={best['used']['correlation']:.6f}, "
        f"held rho={best['held']['correlation']:.6f})"
    )
    print(
        "Scale/correlation identity check: "
        f"max |NMSE* - (1-rho^2)| = {max_identity_error:.3e}"
    )
    detail_rows = [("identity", identity)] if identity is not None else []
    if identity is None or best["transform"] != identity["transform"]:
        detail_rows.append(("best", best))
    for label, row in detail_rows:
        for subset in ("used", "held"):
            metrics = row[subset]
            norm_ratio = metrics["prediction_norm"] / metrics["target_norm"]
            print(
                f"{label} {row['transform']} {subset}: "
                f"rho={metrics['correlation']:.6f}, "
                f"||A(x)||/||y||={norm_ratio:.6g}, "
                f"NMSE={metrics['nmse_unscaled']:.6f} -> "
                f"NMSE*={metrics['nmse_scale_matched']:.6f}, "
                f"|alpha|={metrics['scale_abs']:.6g}"
            )

    if identity is None:
        print("Diagnosis: identity was unavailable because of a spatial shape mismatch.")
    else:
        improvement = (
            best["used"]["correlation"] - identity["used"]["correlation"]
        )
        if best["used"]["correlation"] < 0.1:
            print(
                "Diagnosis: all tested orientations remain weakly correlated; "
                "orientation alone is unlikely to explain the mismatch. Check "
                "trajectory, sample ordering, coil maps, and Fourier conventions."
            )
        elif best["transform"] != "rot0" and improvement > 0.05:
            print(
                "Diagnosis: a non-identity transform materially improves "
                "self-consistency, supporting an image-orientation mismatch."
            )
        elif improvement <= 0.01:
            print(
                "Diagnosis: identity is effectively tied with the best transform; "
                "this test does not support a dominant orientation mismatch."
            )
        else:
            print(
                "Diagnosis: orientation has a measurable but not decisive effect; "
                "repeat across folds/cases and inspect trajectory conventions."
            )

    if max_identity_error > 1e-5:
        print(
            "Warning: the scale-matched NMSE does not agree with 1-rho^2 at "
            "numerical tolerance; inspect the weighting/scaling implementation."
        )


def _aggregate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        if row["status"] == "ok":
            grouped.setdefault(row["transform"], []).append(row)

    aggregate: list[dict[str, Any]] = []
    for transform, rows in grouped.items():
        aggregate.append(
            {
                "transform": transform,
                "folds": len(rows),
                "used_correlation_mean": float(
                    np.mean([row["used"]["correlation"] for row in rows])
                ),
                "used_nmse_scale_matched_mean": float(
                    np.mean([row["used"]["nmse_scale_matched"] for row in rows])
                ),
                "held_correlation_mean": float(
                    np.mean([row["held"]["correlation"] for row in rows])
                ),
                "held_nmse_scale_matched_mean": float(
                    np.mean([row["held"]["nmse_scale_matched"] for row in rows])
                ),
            }
        )
    return sorted(
        aggregate,
        key=lambda row: row["used_correlation_mean"],
        reverse=True,
    )


def main() -> None:
    args = parse_args()
    _configure_numba_cache()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config, config_path = _load_config(args)
    patient_id, sample_source = _resolve_patient_id(args, config)
    slice_index, slice_source = _resolve_slice_index(args, config, patient_id)
    spokes_per_frame = _resolve_eval_spokes(
        config,
        args.spokes_per_frame,
        args.phase_index,
    )
    total_spokes = int(
        args.total_spokes
        if args.total_spokes is not None
        else config.get("data", {}).get("total_spokes", 288)
    )
    raw_root = Path(
        args.raw_kspace_root or config["data"]["root_dir"]
    ).expanduser()
    dataset_key = args.dataset_key or config["data"]["dataset_key"]
    traj_method = (
        args.traj_method
        or config.get("data", {}).get("traj_method", "get_traj")
    )
    device = torch.device(
        args.device or config.get("training", {}).get("device", "cuda")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device '{device}', but CUDA is unavailable. Pass --device cpu."
        )

    print("=== GRASP orientation diagnostic ===")
    print(f"Config: {config_path}")
    print(f"Sample source: {sample_source}")
    print(f"Patient: {patient_id}")
    print(f"Slice: {slice_index} ({slice_source})")
    print(f"Spokes/frame: {spokes_per_frame}")
    print(f"Total spokes / frames: {total_spokes} / {total_spokes // spokes_per_frame}")
    print(f"Trajectory: {traj_method}")
    print(f"Weighting: {args.weighting}")
    print(f"Device: {device}")

    kspace, csmaps, samples_per_spoke, kspace_path, csmap_path = _load_raw_case(
        raw_root=raw_root,
        dataset_key=dataset_key,
        patient_id=patient_id,
        slice_index=slice_index,
        total_spokes=total_spokes,
        spokes_per_frame=spokes_per_frame,
        csmap_path_override=args.csmap_path,
    )
    print(f"K-space: {kspace_path} -> {tuple(kspace.shape)}")
    print(f"CS maps: {csmap_path} -> {tuple(csmaps.shape)}")

    frames = total_spokes // spokes_per_frame
    from utils import prep_nufft

    ktraj, dcomp, nufft_ob, adjnufft_ob = prep_nufft(
        samples_per_spoke,
        spokes_per_frame,
        frames,
        traj_method=traj_method,
    )
    kspace = kspace.to(device)
    csmaps = csmaps.to(device)
    ktraj = ktraj.to(device)
    dcomp = dcomp.to(device)
    nufft_ob = nufft_ob.to(device)
    adjnufft_ob = adjnufft_ob.to(device)

    folds = _build_fold_indices(
        spokes_per_frame,
        samples_per_spoke,
        args.k_folds,
        device,
    )
    if args.all_folds:
        selected_folds = list(enumerate(folds))
    else:
        if args.fold_index < 0 or args.fold_index >= len(folds):
            raise IndexError(
                f"--fold-index {args.fold_index} is outside {len(folds)} valid folds."
            )
        selected_folds = [(args.fold_index, folds[args.fold_index])]

    all_results: list[dict[str, Any]] = []
    for fold_index, (held_indices, used_indices) in selected_folds:
        fold_results = _run_fold(
            fold_index,
            held_indices,
            used_indices,
            kspace,
            csmaps,
            ktraj,
            dcomp,
            nufft_ob,
            adjnufft_ob,
            samples_per_spoke,
            args,
            device,
        )
        _print_results(fold_results)
        all_results.extend(fold_results)

    aggregate = _aggregate_results(all_results)
    if len(selected_folds) > 1:
        print("\n=== Mean across folds ===")
        print("Transform              used rho  used NMSE*  held rho  held NMSE*")
        print("-" * 69)
        for row in aggregate:
            print(
                f"{row['transform']:<22}"
                f"{row['used_correlation_mean']:>9.5f}"
                f"{row['used_nmse_scale_matched_mean']:>12.5f}"
                f"{row['held_correlation_mean']:>10.5f}"
                f"{row['held_nmse_scale_matched_mean']:>12.5f}"
            )

    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": str(config_path),
            "patient_id": patient_id,
            "sample_source": sample_source,
            "slice_index": slice_index,
            "spokes_per_frame": spokes_per_frame,
            "total_spokes": total_spokes,
            "frames": frames,
            "samples_per_spoke": samples_per_spoke,
            "k_folds": args.k_folds,
            "weighting": args.weighting,
            "grasp_lambda": args.grasp_lambda,
            "grasp_max_iter": args.grasp_max_iter,
            "grasp_rho": args.grasp_rho,
            "results": all_results,
            "aggregate": aggregate,
        }
        with output_path.open("w") as stream:
            json.dump(payload, stream, indent=2)
        print(f"\nWrote results: {output_path}")


if __name__ == "__main__":
    main()
