"""Time BRISKNet vs GRASP reconstructions on raw fastMRI validation scans (all slices).

This script:
  - Loads validation IDs from data/data_split.json ("val" by default).
  - For each scan, reconstructs all slices in the raw k-space volume.
  - Time-bins k-space to specified spokes-per-frame (same as training).
  - Times BRISKNet and GRASP end-to-end per scan, and reports mean±std across samples.
  - Prints preprocessing/setup timing summaries with brief descriptions.
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def _parse_float_list(value: str) -> List[float]:
    if not value:
        return []
    return [float(v.strip()) for v in value.split(",") if v.strip()]


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


def _robust_window(values: List[np.ndarray], p_low: float = 1.0, p_high: float = 99.5) -> Tuple[float, float]:
    flat = []
    for arr in values:
        if arr is None:
            continue
        a = np.asarray(arr)
        flat.append(a.ravel())
    if not flat:
        return 0.0, 1.0
    stacked = np.concatenate(flat)
    stacked = stacked[np.isfinite(stacked)]
    if stacked.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(stacked, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def _ensure_thw(arr: np.ndarray, n_time: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array for THW, got {arr.shape}")
    if arr.shape[0] == n_time:
        return arr
    if arr.shape[1] == n_time:
        return np.transpose(arr, (1, 0, 2))
    if arr.shape[2] == n_time:
        return np.transpose(arr, (2, 0, 1))
    return arr


def _brisk_to_thw(x_recon: torch.Tensor, n_time: int) -> np.ndarray:
    if not torch.is_tensor(x_recon):
        raise ValueError("Expected torch tensor for BRISKNet output.")
    t = x_recon.detach().cpu()
    # Drop batch dim if present.
    if t.ndim >= 5 and t.shape[0] == 1:
        t = t[0]
    # Move channel (real/imag) dim to front.
    chan_dim = None
    for i, s in enumerate(t.shape):
        if s == 2:
            chan_dim = i
            break
    if chan_dim is None:
        raise ValueError(f"Could not find complex channel dim in shape {tuple(t.shape)}")
    t = t.movedim(chan_dim, 0)  # (2, ...)
    mag = torch.sqrt(t[0] ** 2 + t[1] ** 2).numpy()
    mag = np.squeeze(mag)
    if mag.ndim != 3:
        raise ValueError(f"Expected 3D magnitude for BRISKNet, got {mag.shape}")
    return _ensure_thw(mag, n_time)


def _grasp_to_thw(recon_out, n_time: int) -> np.ndarray:
    try:
        recon_cpu = sp.to_device(recon_out, sp.cpu_device)
        arr = np.asarray(recon_cpu)
    except Exception:
        arr = np.asarray(recon_out)
    arr = np.squeeze(arr)
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D magnitude for GRASP, got {arr.shape}")
    return _ensure_thw(arr, n_time)


def _save_example_images(
    brisk_thw: np.ndarray,
    grasp_thw: np.ndarray,
    out_dir: str,
    tag: str,
    frame_idx: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    T = brisk_thw.shape[0]
    idx = int(frame_idx)
    if idx < 0 or idx >= T:
        idx = T // 2
    brisk_img = brisk_thw[idx]
    grasp_img = grasp_thw[idx]
    vmin, vmax = _robust_window([brisk_img, grasp_img])
    plt.imsave(os.path.join(out_dir, f"{tag}_brisknet_frame{idx:03d}.png"), brisk_img, cmap="gray", vmin=vmin, vmax=vmax)
    plt.imsave(os.path.join(out_dir, f"{tag}_grasp_frame{idx:03d}.png"), grasp_img, cmap="gray", vmin=vmin, vmax=vmax)
def _latex_table(rows: List[Dict]) -> str:
    lines = []
    lines.append("\\begin{table}")
    lines.append("\\caption{Timing results (end-to-end per scan, all slices).}\\label{tab1}")
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


def _derive_spf_from_accel(accel_list: List[float], H: int) -> List[int]:
    spf_list: List[int] = []
    for acc in accel_list:
        spf = int(round((H * math.pi / 2.0) / float(acc)))
        if spf <= 0:
            raise ValueError(f"Derived spokes/frame <= 0 for acceleration {acc}.")
        recon_acc = (H * math.pi / 2.0) / float(spf)
        rel_err = abs(recon_acc - acc) / float(acc)
        if rel_err > 0.05:
            raise ValueError(
                f"Acceleration {acc} maps to SPF {spf} with rel error {rel_err:.3f}. "
                "Use --spokes-per-frame-list or adjust acceleration list."
            )
        spf_list.append(spf)
    return spf_list


def _load_progress(progress_path: str) -> Dict:
    if not os.path.exists(progress_path):
        return {"meta": {}, "spfs": {}}
    with open(progress_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_progress(progress_path: str, progress: Dict) -> None:
    tmp_path = f"{progress_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, sort_keys=True)
    os.replace(tmp_path, progress_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Time BRISKNet vs GRASP reconstructions on raw validation scans (all slices)."
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
        "--spokes-per-frame-list",
        default="2,4,8,16,24,36",
        help="Comma-separated spokes/frame list (optional if --acceleration-list is provided).",
    )
    parser.add_argument(
        "--acceleration-list",
        default="",
        help="Comma-separated acceleration list (AF).",
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
        default=15,
        help="Limit number of validation samples (default: 15).",
    )
    parser.add_argument(
        "--save_example_images",
        action="store_true",
        help="Save example BRISKNet/GRASP frame images for one slice per SPF.",
    )
    parser.add_argument(
        "--example_patient_id",
        default=None,
        help="Validation patient id to use for example images (default: first).",
    )
    parser.add_argument(
        "--example_slice_idx",
        type=int,
        default=-1,
        help="Slice index for example images (-1 uses center slice).",
    )
    parser.add_argument(
        "--example_frame_idx",
        type=int,
        default=-1,
        help="Frame index for example images (-1 uses center frame).",
    )
    parser.add_argument(
        "--example_out_dir",
        default="timing_examples",
        help="Output directory for example images.",
    )
    parser.add_argument(
        "--resume-path",
        default="time_raw_inference_scan_progress.json",
        help="Path to resume JSON file (default: time_raw_inference_scan_progress.json).",
    )
    args = parser.parse_args()

    spf_list = _parse_int_list(args.spokes_per_frame_list)
    accel_list = _parse_float_list(args.acceleration_list)
    if spf_list and accel_list:
        print("[Warn] Both --spokes-per-frame-list and --acceleration-list provided. "
              "Using spokes-per-frame list.")
        accel_list = []

    val_ids = _load_split_ids(args.split_file, args.split_key)
    if args.num_samples:
        val_ids = val_ids[: args.num_samples]

    device = _resolve_device(args.device)
    _print_device_info(device)
    sigpy_device = _sigpy_device_from_torch(device)

    progress = _load_progress(args.resume_path)
    progress.setdefault("spfs", {})

    # Derive SPF list from acceleration if needed (requires H).
    if not spf_list:
        if not accel_list:
            raise ValueError("Provide either --spokes-per-frame-list or --acceleration-list.")
        first_id = val_ids[0]
        csmap0 = _load_csmap(args.csmap_root, first_id, slice_idx=0, flip_kspace=False)
        H = csmap0.shape[-2]
        spf_list = _derive_spf_from_accel(accel_list, H)

    progress_meta = {
        "split_file": args.split_file,
        "split_key": args.split_key,
        "kspace_root": args.kspace_root,
        "csmap_root": args.csmap_root,
        "dataset_key": args.dataset_key,
        "model_template": args.model_template,
        "config_template": args.config_template,
        "num_samples": args.num_samples,
        "total_scan_seconds": args.total_scan_seconds,
        "spokes_per_frame_list": spf_list,
        "acceleration_list": accel_list,
    }
    if progress.get("meta") and progress.get("meta") != progress_meta:
        print("[Warn] Resume file metadata differs from current run arguments.")
    progress["meta"] = progress_meta

    results = []

    for spf in spf_list:
        print(spf)
        spf_key = str(spf)
        spf_state = progress.get("spfs", {}).get(spf_key, {})
        spf_patients = spf_state.get("patients", {})
        completed_ids = set(spf_patients.keys())
        pending_ids = [pid for pid in val_ids if pid not in completed_ids]

        if pending_ids:
            print(
                f"[Resume] SPF {spf}: {len(completed_ids)}/{len(val_ids)} completed; "
                f"processing {len(pending_ids)} remaining."
            )
        else:
            print(f"[Resume] SPF {spf}: all {len(val_ids)} samples already completed.")

        config_path = args.config_template.format(spf=spf)
        model_path = args.model_template.format(spf=spf)

        # Load model/config only if we have pending work.
        if pending_ids:
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
        else:
            # Minimal defaults for summary printing if no new work.
            config = {}
            model = None
            flip_kspace = bool(spf_state.get("flip_kspace", False))
            norm_mode = spf_state.get("norm_mode", "both")
            encode_acc = bool(spf_state.get("encode_acc", False))
            encode_time = bool(spf_state.get("encode_time", False))
            eval_chunk_size = int(spf_state.get("eval_chunk_size", 0) or 0)
            eval_chunk_overlap = int(spf_state.get("eval_chunk_overlap", 0) or 0)
            traj_method = spf_state.get("traj_method", "get_traj")

        brisk_times: List[float] = []
        brisk_e2e_times: List[float] = []
        grasp_times: List[float] = []
        grasp_e2e_times: List[float] = []
        brisk_pre_times: List[float] = []
        brisk_xfer_times: List[float] = []
        grasp_pre_times: List[float] = []

        num_samples_target = max(1, len(val_ids))

        # Prepared per spf once we know N_samples/N_time.
        ktraj = None
        dcomp = None
        nufft_ob = None
        adjnufft_ob = None
        n_time_ref = spf_state.get("n_time_ref")
        n_samples_ref = spf_state.get("n_samples_ref")
        acceleration_encoding = None
        start_timepoint_index = None
        coord = None
        physics = None

        brisk_setup_time = float(spf_state.get("brisk_setup_time", 0.0))
        grasp_setup_time = float(spf_state.get("grasp_setup_time", 0.0))
        record_brisk_setup = brisk_setup_time == 0.0
        record_grasp_setup = grasp_setup_time == 0.0

        use_sliding = False

        example_saved = False
        example_patient_id = args.example_patient_id or (val_ids[0] if val_ids else None)

        # Use pending IDs to avoid reprocessing.
        for patient_id in pending_ids:
            kspace_path = _find_kspace_file(args.kspace_root, patient_id)
            with h5py.File(kspace_path, "r") as f:
                if args.dataset_key not in f:
                    raise KeyError(f"Dataset key '{args.dataset_key}' not found in {kspace_path}")
                data = f[args.dataset_key]
                num_slices = data.shape[0]
                slice_target = args.example_slice_idx
                if slice_target is None or int(slice_target) < 0:
                    slice_target = num_slices // 2

                if num_slices != 192:
                    print(
                        f"[Warn] {patient_id}: expected 192 slices, found {num_slices}. "
                        "Using dataset slice count."
                    )

                # Per-scan accumulators
                brisk_pre_total = 0.0
                brisk_xfer_total = 0.0
                brisk_solve_total = 0.0
                grasp_pre_total = 0.0
                grasp_solve_total = 0.0
                grasp_acq_slices = []

                for slice_idx in range(num_slices):
                    want_example = (
                        args.save_example_images
                        and (not example_saved)
                        and (example_patient_id == patient_id)
                        and (int(slice_idx) == int(slice_target))
                    )
                    preprocess_start = time.perf_counter()
                    kspace_slice = np.asarray(data[slice_idx])
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

                    csmap = _load_csmap(args.csmap_root, patient_id, slice_idx, flip_kspace)

                    # Convert to complex k-space (coils, samples, time)
                    kspace_cplx = to_torch_complex(kspace_binned.unsqueeze(0)).squeeze(0)
                    kspace_cplx = rearrange(kspace_cplx, "t co sp sam -> co (sp sam) t")
                    brisk_pre_total += time.perf_counter() - preprocess_start

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
                        H = csmap.shape[-2]
                        n_full = H * math.pi / 2.0
                        acceleration = torch.tensor([n_full / float(spf)], dtype=torch.float, device=device)
                        acceleration_encoding = acceleration if encode_acc else None
                        start_timepoint_index = (
                            torch.tensor([0], dtype=torch.float, device=device) if encode_time else None
                        )
                        if record_brisk_setup:
                            brisk_setup_time += time.perf_counter() - setup_start

                        coord_start = time.perf_counter()
                        coord = get_traj(
                            N_spokes=spf,
                            N_time=n_time,
                            base_res=int(n_samples // 2),
                            gind=1,
                        )
                        coord = np.asarray(coord, dtype=np.float32)
                        if coord.ndim == 3:
                            coord = coord[None, ...]
                        if record_grasp_setup:
                            grasp_setup_time += time.perf_counter() - coord_start

                        use_sliding = eval_chunk_size > 0 and eval_chunk_size < n_time
                        if not use_sliding:
                            physics_start = time.perf_counter()
                            physics = MCNUFFT(nufft_ob, adjnufft_ob, ktraj, dcomp)
                            if record_brisk_setup:
                                brisk_setup_time += time.perf_counter() - physics_start

                        # Store static per-SPF metadata for resume.
                        spf_state["n_time_ref"] = n_time
                        spf_state["n_samples_ref"] = n_samples
                        spf_state["H"] = int(H)
                        spf_state["flip_kspace"] = flip_kspace
                        spf_state["norm_mode"] = norm_mode
                        spf_state["encode_acc"] = encode_acc
                        spf_state["encode_time"] = encode_time
                        spf_state["eval_chunk_size"] = eval_chunk_size
                        spf_state["eval_chunk_overlap"] = eval_chunk_overlap
                        spf_state["traj_method"] = traj_method

                    # --- BRISKNet timing ---
                    transfer_start = time.perf_counter()
                    kspace_dev = kspace_cplx.to(device)
                    csmap_dev = csmap.to(device).to(kspace_dev.dtype)
                    _sync_torch(device)
                    brisk_xfer_total += time.perf_counter() - transfer_start

                    brisk_example = None
                    with torch.no_grad():
                        if use_sliding:
                            _sync_torch(device)
                            t0 = time.perf_counter()
                            if want_example:
                                brisk_example, _ = sliding_window_inference(
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
                            else:
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
                            brisk_solve_total += time.perf_counter() - t0
                        else:
                            _sync_torch(device)
                            t0 = time.perf_counter()
                            out = model(
                                kspace_dev,
                                physics,
                                csmap_dev,
                                acceleration_encoding,
                                start_timepoint_index,
                                epoch="inference",
                                norm=norm_mode,
                            )
                            if want_example:
                                brisk_example = out[0] if isinstance(out, (tuple, list)) else out
                            _sync_torch(device)
                            brisk_solve_total += time.perf_counter() - t0

                    # --- GRASP timing (match dce_recon slice prep/combination) ---
                    grasp_pre_start = time.perf_counter()
                    # k1 shape: (T, 1, C, 1, Sp, Sam) to match dce_recon.py
                    ksp_redu = kspace_slice[:, : (n_time * spf), :]
                    ksp_prep = np.swapaxes(ksp_redu, 0, 1)  # (Sp, C, Sam)
                    ksp_prep = ksp_prep.reshape(n_time, spf, kspace_slice.shape[0], n_samples)
                    ksp_prep = np.transpose(ksp_prep, (0, 2, 1, 3))  # (T, C, Sp, Sam)
                    kspace_np = ksp_prep[:, None, :, None, :, :]  # (T,1,C,1,Sp,Sam)

                    csmaps_np = rearrange(csmap, "b c h w -> c b h w").cpu().numpy()
                    coord_np = coord

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
                    grasp_pre_total += time.perf_counter() - grasp_pre_start

                    t0 = time.perf_counter()
                    recon_out = recon_op.run()
                    _sync_sigpy(sigpy_device)
                    grasp_solve_total += time.perf_counter() - t0

                    # Collect slice recon for dce_recon-style combining.
                    grasp_acq_slices.append(recon_out)

                    if want_example and brisk_example is not None:
                        try:
                            brisk_thw = _brisk_to_thw(brisk_example, n_time)
                            grasp_thw = _grasp_to_thw(recon_out, n_time)
                            tag = f"{patient_id}_spf{spf}_slice{slice_idx:03d}"
                            _save_example_images(
                                brisk_thw,
                                grasp_thw,
                                args.example_out_dir,
                                tag,
                                args.example_frame_idx,
                            )
                            example_saved = True
                            print(f"[Example] Saved images for {tag} to {args.example_out_dir}")
                        except Exception as exc:
                            print(f"[Example] Failed to save example images: {exc}")

                    # Free recon objects to keep memory stable.
                    del recon_op

                # Combine slices like dce_recon.py
                if grasp_acq_slices:
                    combine_start = time.perf_counter()
                    try:
                        import cupy as cp  # optional, mirrors dce_recon behavior
                        grasp_acq_slices = sp.to_device(grasp_acq_slices, sigpy_device)
                        grasp_acq_slices = cp.array(grasp_acq_slices)
                        grasp_acq_slices = cp.asnumpy(grasp_acq_slices)
                    except Exception:
                        grasp_acq_slices = np.asarray(grasp_acq_slices)
                    # Default dce_recon behavior: magnitude + squeeze
                    try:
                        grasp_acq_slices = np.abs(grasp_acq_slices).squeeze(axis=(2, 3, 4))
                    except Exception:
                        grasp_acq_slices = np.abs(np.squeeze(grasp_acq_slices))
                    grasp_solve_total += time.perf_counter() - combine_start

                # Per-scan totals
                spf_patients[patient_id] = {
                    "num_slices": num_slices,
                    "brisk_pre": brisk_pre_total,
                    "brisk_xfer": brisk_xfer_total,
                    "brisk_solve": brisk_solve_total,
                    "grasp_pre": grasp_pre_total,
                    "grasp_solve": grasp_solve_total,
                }

                spf_state["patients"] = spf_patients
                spf_state["brisk_setup_time"] = brisk_setup_time
                spf_state["grasp_setup_time"] = grasp_setup_time
                progress["spfs"][spf_key] = spf_state
                _save_progress(args.resume_path, progress)

        # Aggregate from resume state (including any newly added patients).
        if not spf_patients:
            raise RuntimeError(f"No samples processed for spf={spf}.")

        # Ensure n_time_ref exists for seconds/frame.
        if n_time_ref is None:
            n_time_ref = spf_state.get("n_time_ref")
        if n_time_ref is None:
            raise RuntimeError(f"No samples processed for spf={spf}.")

        seconds_per_frame = (
            float(args.total_scan_seconds) / float(n_time_ref - 1)
            if n_time_ref and n_time_ref > 1
            else float(args.total_scan_seconds)
        )
        H = spf_state.get("H")
        if H is None:
            # Fallback: use first patient and slice 0 to infer H.
            first_id = val_ids[0]
            csmap0 = _load_csmap(args.csmap_root, first_id, slice_idx=0, flip_kspace=flip_kspace)
            H = int(csmap0.shape[-2])
            spf_state["H"] = H
        acceleration = (float(H) * math.pi / 2.0) / float(spf)

        # Pull per-scan metrics from stored patients.
        for pid in val_ids:
            if pid not in spf_patients:
                continue
            p = spf_patients[pid]
            brisk_times.append(p["brisk_solve"])
            brisk_pre_times.append(p["brisk_pre"])
            brisk_xfer_times.append(p["brisk_xfer"])
            brisk_e2e_times.append(
                p["brisk_pre"]
                + p["brisk_xfer"]
                + p["brisk_solve"]
                + (brisk_setup_time / float(num_samples_target))
            )
            grasp_times.append(p["grasp_solve"])
            grasp_pre_times.append(p["grasp_pre"])
            grasp_e2e_times.append(
                p["grasp_pre"]
                + p["grasp_solve"]
                + (grasp_setup_time / float(num_samples_target))
            )

        print(f"\n[SPF={spf}] Preprocessing/setup notes:")
        print(
            "  BRISKNet preprocess includes: load k-space slice, time-bin to (T,C,Sp,Sam), "
            "load csmap, convert to complex, rearrange to (C,Sp*Sam,T)."
        )
        print(
            "  GRASP preprocess includes: rearrange k-space/csmaps to sigpy layout, "
            "move to sigpy device, build HighDimensionalRecon operator."
        )
        print(
            f"  BRISKNet preprocess mean per scan: {np.mean(brisk_pre_times):.3f}s "
            f"(std { _std_or_zero(brisk_pre_times):.3f}s)"
        )
        print(
            f"  BRISKNet transfer mean per scan: {np.mean(brisk_xfer_times):.3f}s "
            f"(std { _std_or_zero(brisk_xfer_times):.3f}s)"
        )
        print(
            f"  BRISKNet setup (per SPF): {brisk_setup_time:.3f}s "
            f"(amortized per scan: {brisk_setup_time / float(num_samples_target):.3f}s)"
        )
        print(
            f"  GRASP preprocess mean per scan: {np.mean(grasp_pre_times):.3f}s "
            f"(std { _std_or_zero(grasp_pre_times):.3f}s)"
        )
        print(
            f"  GRASP setup (per SPF): {grasp_setup_time:.3f}s "
            f"(amortized per scan: {grasp_setup_time / float(num_samples_target):.3f}s)"
        )
        if len(brisk_times) < len(val_ids):
            print(
                f"  [Warn] SPF {spf}: only {len(brisk_times)}/{len(val_ids)} samples completed. "
                "Table reflects completed subset."
            )

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

    print("\n" + _latex_table(results))


if __name__ == "__main__":
    main()
