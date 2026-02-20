"""Time BRISKNet vs GRASP reconstructions on raw fastMRI validation slices.

This script:
  - Loads the 15 validation IDs from data/data_split.json ("val").
  - Extracts a single fixed slice (center by default) from raw k-space.
  - Time-bins k-space to specified spokes-per-frame (same as training).
  - Times BRISKNet inference and GRASP (sigpy HighDimensionalRecon) per sample.
  - Prints a LaTeX table with acceleration, seconds/frame, and mean±std times.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
import yaml
from einops import rearrange

import sigpy as sp
from sigpy.mri import app

from model_factory import build_recon_model
from radial_lsfp import MCNUFFT
from utils import prep_nufft, remove_module_prefix, to_torch_complex, get_traj, sliding_window_inference


def _parse_int_list(value: str) -> List[int]:
    if not value:
        return []
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _load_split_ids(split_file: str, split_key: str) -> List[str]:
    with open(split_file, "r", encoding="utf-8") as f:
        splits = json.load(f)
    if split_key not in splits:
        raise KeyError(f"Split key '{split_key}' not found in {split_file}.")
    ids = splits[split_key]
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"Split '{split_key}' in {split_file} is empty or not a list.")
    return ids


def _find_kspace_file(root_dir: str, patient_id: str) -> str:
    pattern = os.path.join(root_dir, f"*{patient_id}*.h5")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No k-space file matching {pattern}")
    if len(matches) > 1:
        sample = "\n".join(matches[:5])
        raise RuntimeError(
            f"Multiple k-space files matched {patient_id} (showing up to 5):\n{sample}"
        )
    return matches[0]


def _load_kspace_slice(kspace_path: str, dataset_key: str, slice_idx: int) -> np.ndarray:
    with h5py.File(kspace_path, "r") as f:
        if dataset_key not in f:
            raise KeyError(f"Dataset key '{dataset_key}' not found in {kspace_path}")
        data = f[dataset_key]
        if slice_idx < 0 or slice_idx >= data.shape[0]:
            raise IndexError(
                f"slice_idx {slice_idx} out of bounds for {kspace_path} with {data.shape[0]} slices"
            )
        kspace_slice = data[slice_idx]
    return np.asarray(kspace_slice)


def _load_csmap(csmap_root: str, patient_id: str, slice_idx: int, flip_kspace: bool) -> torch.Tensor:
    csmap_path = os.path.join(
        csmap_root, f"{patient_id}_2_cs_maps", f"cs_map_slice_{slice_idx:03d}.npy"
    )
    if not os.path.exists(csmap_path):
        raise FileNotFoundError(f"CSMAP file not found: {csmap_path}")
    csmap = np.load(csmap_path).squeeze()
    csmap_t = torch.from_numpy(csmap)
    if not torch.is_complex(csmap_t):
        csmap_t = csmap_t.to(torch.complex64)
    if csmap_t.ndim != 3:
        raise ValueError(
            f"Expected csmap shape (C,H,W) after squeeze, got {tuple(csmap_t.shape)}"
        )
    csmap_t = csmap_t.unsqueeze(0)  # (1, C, H, W)
    if flip_kspace:
        csmap_t = torch.rot90(csmap_t, k=2, dims=[-2, -1])
    return csmap_t


def _time_bin_kspace_train(
    kspace_slice: np.ndarray, spokes_per_frame: int, flip_kspace: bool
) -> Tuple[torch.Tensor, int, int]:
    """Time-bin k-space following ZFSliceDataset (training) logic."""
    if kspace_slice.ndim != 3:
        raise ValueError(
            f"Expected kspace slice shape (C,Spokes,Samples), got {kspace_slice.shape}"
        )
    n_coils, n_spokes, n_samples = kspace_slice.shape
    if n_spokes % spokes_per_frame != 0:
        raise ValueError(
            f"Total spokes ({n_spokes}) not divisible by spokes_per_frame ({spokes_per_frame})."
        )
    n_time = n_spokes // spokes_per_frame
    n_spokes_prep = n_time * spokes_per_frame

    ksp_redu = kspace_slice[:, :n_spokes_prep, :]
    ksp_prep = np.swapaxes(ksp_redu, 0, 1)  # (spokes, coils, samples)
    ksp_prep = ksp_prep.reshape(n_time, spokes_per_frame, n_coils, n_samples)
    ksp_prep = np.transpose(ksp_prep, (0, 2, 1, 3))  # (T, C, Sp, Sam)

    real_part = torch.from_numpy(ksp_prep.real)
    imag_part = torch.from_numpy(ksp_prep.imag)
    kspace_final = torch.stack([real_part, imag_part], dim=0).float()

    if flip_kspace:
        kspace_final = torch.flip(kspace_final, dims=[-1])

    return kspace_final, n_time, n_samples


def _torch_load_checkpoint(path: str, map_location: str = "cpu") -> Dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


def _sync_torch(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _sync_sigpy(device: sp.Device) -> None:
    if hasattr(device, "synchronize"):
        device.synchronize()


def _resolve_device(device_str: str | None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sigpy_device_from_torch(device: torch.device) -> sp.Device:
    if device.type != "cuda":
        return sp.Device(-1)
    index = device.index if device.index is not None else 0
    return sp.Device(index)


def _print_device_info(device: torch.device) -> None:
    if device.type != "cuda":
        print(f"[Device] Using {device.type.upper()}.")
        return
    index = device.index if device.index is not None else 0
    try:
        name = torch.cuda.get_device_name(index)
        props = torch.cuda.get_device_properties(index)
        total_gb = props.total_memory / (1024**3)
        cap = f"{props.major}.{props.minor}"
        print(f"[Device] CUDA:{index} {name} | CC {cap} | {total_gb:.1f} GB")
    except Exception as exc:
        print(f"[Device] CUDA:{index} (failed to query properties: {exc})")


def _load_config(config_path: str) -> Dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_model(config: Dict, device: torch.device) -> torch.nn.Module:
    model = build_recon_model(config, device=device, block_dir=None)
    model.eval()
    return model


def _load_weights(model: torch.nn.Module, checkpoint_path: str) -> None:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = _torch_load_checkpoint(checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(remove_module_prefix(state))


def _format_float(val: float | None, decimals: int = 3) -> str:
    if val is None:
        return ""
    return f"{val:.{decimals}f}"

def _format_mean_std(mean: float | None, std: float | None, decimals: int = 3) -> str:
    if mean is None:
        return ""
    std_val = 0.0 if std is None else float(std)
    return f"${mean:.{decimals}f} \\pm {std_val:.{decimals}f}$"


def _std_or_zero(values: List[float]) -> float:
    if len(values) > 1:
        return float(np.std(values, ddof=1))
    return 0.0


def _latex_table(rows: List[Dict]) -> str:
    lines = []
    lines.append("\\begin{table}")
    lines.append("\\caption{Timing results.}\\label{tab1}")
    lines.append("\\begin{tabular}{|l|l|l|l|l|l|l|}")
    lines.append("\\hline")
    lines.append(
        "SPF & AF & Sec/frame & BRISKNet solve (s) & BRISKNet end-to-end (s) "
        "& GRASP solve (s) & GRASP end-to-end (s) \\\\"
    )
    lines.append("\\hline")
    for row in rows:
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} \\\\".format(
                int(row["spokes_per_frame"]),
                _format_float(row["acceleration"], 2),
                _format_float(row["seconds_per_frame"], 2),
                _format_mean_std(row["brisknet_time"], row["brisknet_std"], 3),
                _format_mean_std(row["brisknet_e2e_time"], row["brisknet_e2e_std"], 3),
                _format_mean_std(row["grasp_time"], row["grasp_std"], 3),
                _format_mean_std(row["grasp_e2e_time"], row["grasp_e2e_std"], 3),
            )
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Time BRISKNet vs GRASP reconstructions on raw validation slices."
    )
    parser.add_argument(
        "--split-file",
        default="data/data_split.json",
        help="Path to data split JSON.",
    )
    parser.add_argument(
        "--split-key",
        default="val",
        help="Split key to use from the split file (default: val).",
    )
    parser.add_argument(
        "--kspace-root",
        default="/net/scratch2/rachelgordon/zf_data_192_slices/zf_kspace",
        help="Root directory containing raw k-space .h5 files.",
    )
    parser.add_argument(
        "--csmap-root",
        default="/net/scratch2/rachelgordon/zf_data_192_slices/cs_maps",
        help="Root directory containing cs_maps/<patient>_cs_maps/.",
    )
    parser.add_argument(
        "--dataset-key",
        default="kspace",
        help="Dataset key inside the .h5 files (default: kspace).",
    )
    parser.add_argument(
        "--slice-idx",
        type=int,
        default=96,
        help="Fixed slice index (0-based) to reconstruct (default: 96).",
    )
    parser.add_argument(
        "--spokes-per-frame-list",
        default="2,4,8,16,24,36",
        help="Comma-separated spokes/frame list.",
    )
    parser.add_argument(
        "--model-template",
        default=(
            "/net/projects2/annawoodard/rachelgordon/experiments/"
            "ei_diffeo_{spf}spf_slice_sampling/"
            "ei_diffeo_{spf}spf_slice_sampling_model.pth"
        ),
        help="Checkpoint path template with {spf} placeholder.",
    )
    parser.add_argument(
        "--config-template",
        default="configs/config_sampling_{spf}spf.yaml",
        help="Config path template with {spf} placeholder.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (default: cuda if available).",
    )
    parser.add_argument("--lamda", type=float, default=0.001, help="GRASP TV weight.")
    parser.add_argument("--max-iter", type=int, default=10, help="GRASP max iterations.")
    parser.add_argument("--rho", type=float, default=0.1, help="GRASP ADMM rho.")
    parser.add_argument(
        "--total-scan-seconds",
        type=float,
        default=150.0,
        help="Total scan duration in seconds (for seconds/frame).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit number of validation samples (default: all).",
    )
    args = parser.parse_args()

    spokes_list = _parse_int_list(args.spokes_per_frame_list)
    if not spokes_list:
        raise ValueError("No spokes_per_frame values provided.")

    val_ids = _load_split_ids(args.split_file, args.split_key)
    if args.num_samples:
        val_ids = val_ids[: args.num_samples]

    device = _resolve_device(args.device)
    _print_device_info(device)
    sigpy_device = _sigpy_device_from_torch(device)

    results = []

    for spf in spokes_list:
        config_path = args.config_template.format(spf=spf)
        model_path = args.model_template.format(spf=spf)

        config = _load_config(config_path)
        model = _build_model(config, device)
        _load_weights(model, model_path)

        flip_kspace = bool(config.get("data", {}).get("flip_kspace", False))
        norm_mode = config.get("model", {}).get("norm", "both")
        encode_acc = bool(config.get("model", {}).get("encode_acceleration", False))
        encode_time = bool(config.get("model", {}).get("encode_time_index", False))
        eval_chunk_size = int(config.get("evaluation", {}).get("chunk_size", 0) or 0)
        eval_chunk_overlap = int(config.get("evaluation", {}).get("chunk_overlap", 0) or 0)
        traj_method = config.get("data", {}).get("traj_method", "get_traj")

        brisk_times: List[float] = []
        brisk_e2e_times: List[float] = []
        grasp_times: List[float] = []
        grasp_e2e_times: List[float] = []
        per_spf_setup_time = 0.0
        num_samples_target = max(1, len(val_ids))

        # Prepared per spf once we know N_samples/N_time.
        ktraj = None
        dcomp = None
        nufft_ob = None
        adjnufft_ob = None
        n_time_ref = None
        n_samples_ref = None
        acceleration_encoding = None
        start_timepoint_index = None
        coord = None

        for patient_id in val_ids:
            preprocess_start = time.perf_counter()
            kspace_path = _find_kspace_file(args.kspace_root, patient_id)
            kspace_slice = _load_kspace_slice(kspace_path, args.dataset_key, args.slice_idx)
            kspace_binned, n_time, n_samples = _time_bin_kspace_train(
                kspace_slice, spf, flip_kspace
            )

            if n_time_ref is None:
                n_time_ref = n_time
                n_samples_ref = n_samples
            else:
                if n_time != n_time_ref or n_samples != n_samples_ref:
                    raise ValueError(
                        f"Inconsistent k-space shape for {patient_id}: "
                        f"time {n_time} vs {n_time_ref}, samples {n_samples} vs {n_samples_ref}."
                    )

            csmap = _load_csmap(args.csmap_root, patient_id, args.slice_idx, flip_kspace)

            # Convert to complex k-space (coils, samples, time)
            kspace_cplx = to_torch_complex(kspace_binned.unsqueeze(0)).squeeze(0)
            kspace_cplx = rearrange(kspace_cplx, "t co sp sam -> co (sp sam) t")
            preprocess_time = time.perf_counter() - preprocess_start

            if ktraj is None:
                setup_start = time.perf_counter()
                ktraj, dcomp, nufft_ob, adjnufft_ob = prep_nufft(
                    n_samples, spf, n_time, traj_method=traj_method
                )
                if device.type == "cuda":
                    ktraj = ktraj.to(device)
                    dcomp = dcomp.to(device)
                    nufft_ob = nufft_ob.to(device)
                    adjnufft_ob = adjnufft_ob.to(device)
                else:
                    ktraj = ktraj.to(device)
                    dcomp = dcomp.to(device)
                per_spf_setup_time = time.perf_counter() - setup_start

                H = csmap.shape[-2]
                n_full = H * math.pi / 2.0
                acceleration = torch.tensor([n_full / float(spf)], dtype=torch.float, device=device)
                acceleration_encoding = acceleration if encode_acc else None
                start_timepoint_index = (
                    torch.tensor([0], dtype=torch.float, device=device) if encode_time else None
                )

                coord = get_traj(
                    N_spokes=spf,
                    N_time=n_time,
                    base_res=int(n_samples // 2),
                    gind=1,
                )
                coord = np.asarray(coord, dtype=np.float32)
                if coord.ndim == 3:
                    coord = coord[None, ...]

            # --- BRISKNet timing ---
            transfer_start = time.perf_counter()
            kspace_dev = kspace_cplx.to(device)
            csmap_dev = csmap.to(device).to(kspace_dev.dtype)
            _sync_torch(device)
            transfer_time = time.perf_counter() - transfer_start

            use_sliding = eval_chunk_size > 0 and eval_chunk_size < n_time
            physics_time = 0.0
            with torch.no_grad():
                if use_sliding:
                    _sync_torch(device)
                    t0 = time.perf_counter()
                    _ = sliding_window_inference(
                        csmap_dev.shape[-2],
                        csmap_dev.shape[-1],
                        n_time,
                        ktraj,
                        dcomp,
                        nufft_ob,
                        adjnufft_ob,
                        eval_chunk_size,
                        eval_chunk_overlap,
                        kspace_dev,
                        csmap_dev,
                        acceleration_encoding,
                        start_timepoint_index,
                        model,
                        epoch="inference",
                        device=device,
                        norm=norm_mode,
                        collect_adj_loss=False,
                    )
                    _sync_torch(device)
                    brisk_times.append(time.perf_counter() - t0)
                else:
                    physics_start = time.perf_counter()
                    physics = MCNUFFT(nufft_ob, adjnufft_ob, ktraj, dcomp)
                    physics_time = time.perf_counter() - physics_start
                    _sync_torch(device)
                    t0 = time.perf_counter()
                    _ = model(
                        kspace_dev,
                        physics,
                        csmap_dev,
                        acceleration_encoding,
                        start_timepoint_index,
                        epoch="inference",
                        norm=norm_mode,
                    )
                    _sync_torch(device)
                    brisk_times.append(time.perf_counter() - t0)
            brisk_e2e_times.append(
                preprocess_time
                + transfer_time
                + physics_time
                + brisk_times[-1]
                + (per_spf_setup_time / float(num_samples_target))
            )

            # --- GRASP timing ---
            # Prepare kspace/csmaps for sigpy (preprocessing outside timed block).
            grasp_total_start = time.perf_counter()
            kspace_np = rearrange(
                kspace_cplx.cpu(), "c (sp sam) t -> t c sp sam", sam=n_samples
            ).unsqueeze(1).unsqueeze(3).numpy()
            csmaps_np = rearrange(csmap, "b c h w -> c b h w").cpu().numpy()
            coord_np = coord

            # Move arrays to sigpy device before timing.
            kspace_sp = sp.to_device(kspace_np, sigpy_device)
            csmaps_sp = sp.to_device(csmaps_np, sigpy_device)
            coord_sp = sp.to_device(coord_np, sigpy_device)

            recon_op = app.HighDimensionalRecon(
                kspace_sp,
                csmaps_sp,
                combine_echo=False,
                lamda=args.lamda,
                coord=coord_sp,
                regu="TV",
                regu_axes=[0],
                max_iter=args.max_iter,
                solver="ADMM",
                rho=args.rho,
                device=sigpy_device,
                show_pbar=False,
                verbose=False,
            )

            _sync_sigpy(sigpy_device)
            t0 = time.perf_counter()
            recon_out = recon_op.run()
            _sync_sigpy(sigpy_device)
            grasp_times.append(time.perf_counter() - t0)
            grasp_e2e_times.append(preprocess_time + (time.perf_counter() - grasp_total_start))
            # Free recon object to keep memory stable.
            del recon_out
            del recon_op

        if n_time_ref is None:
            raise RuntimeError(f"No samples processed for spf={spf}.")

        seconds_per_frame = (
            float(args.total_scan_seconds) / float(n_time_ref - 1)
            if n_time_ref > 1
            else float(args.total_scan_seconds)
        )
        acceleration = (csmap.shape[-2] * math.pi / 2.0) / float(spf)

        results.append(
            {
                "spokes_per_frame": spf,
                "acceleration": acceleration,
                "seconds_per_frame": seconds_per_frame,
                "brisknet_time": float(np.mean(brisk_times)),
                "brisknet_std": _std_or_zero(brisk_times),
                "brisknet_e2e_time": float(np.mean(brisk_e2e_times)),
                "brisknet_e2e_std": _std_or_zero(brisk_e2e_times),
                "grasp_time": float(np.mean(grasp_times)),
                "grasp_std": _std_or_zero(grasp_times),
                "grasp_e2e_time": float(np.mean(grasp_e2e_times)),
                "grasp_e2e_std": _std_or_zero(grasp_e2e_times),
                "num_samples": len(brisk_times),
            }
        )

    print(_latex_table(results))


if __name__ == "__main__":
    main()
