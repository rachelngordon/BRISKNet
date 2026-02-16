#!/usr/bin/env python3
"""Generate GRASP distillation targets with explicit split control.

This script is designed for supervised distillation target generation on fastMRI
data. It enforces split-scoped processing (e.g., train only) to avoid
train/val/test leakage and writes targets in the format expected by
`RadialKspaceDataset`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import time
import traceback
from typing import Iterable
import uuid

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GRASP targets for selected split(s).")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing raw fastMRI k-space .h5 files.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        required=True,
        help="Output root (patient subdirs will be created here).",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        required=True,
        help="Path to data_split.json.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        required=True,
        help="Split names from split-file to process (e.g., train val).",
    )
    parser.add_argument(
        "--spokes-per-frame",
        type=int,
        nargs="+",
        default=[36],
        help="Spokes-per-frame values to reconstruct.",
    )
    parser.add_argument(
        "--total-spokes",
        type=int,
        default=288,
        help="Total spokes used to derive number of frames.",
    )
    parser.add_argument(
        "--center-partition",
        type=int,
        default=31,
        help="Center partition passed to process_kspace.",
    )
    parser.add_argument(
        "--images-per-slab",
        type=int,
        default=1,
        help="Images-per-slab passed to process_kspace.",
    )
    parser.add_argument(
        "--slice-mode",
        choices=("all", "center"),
        default="all",
        help="Slices to reconstruct per exam.",
    )
    parser.add_argument(
        "--center-slice",
        type=int,
        default=95,
        help="Center slice index when --slice-mode=center.",
    )
    parser.add_argument(
        "--slice-priority-order",
        choices=("middle_first", "sequential"),
        default="middle_first",
        help=(
            "Ordering used within each exam for pending slices. "
            "'middle_first' prioritizes center-ish anatomy before edges."
        ),
    )
    parser.add_argument(
        "--priority-slices-per-exam",
        type=int,
        default=0,
        help=(
            "If >0, run two phases: first process this many prioritized slices/exam, "
            "then fill the remaining slices."
        ),
    )
    parser.add_argument(
        "--csmaps-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing precomputed ESPIRiT maps: "
            "<csmaps-dir>/<patient>_cs_maps/cs_map_slice_XXX.npy"
        ),
    )
    parser.add_argument(
        "--max-exams",
        type=int,
        default=0,
        help="Optional cap for debugging (0 means no cap).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing target files.",
    )
    parser.add_argument(
        "--save-dtype",
        choices=("complex64", "complex128"),
        default="complex64",
        help="Complex dtype used when saving GRASP targets.",
    )
    parser.add_argument(
        "--recon-max-iter",
        type=int,
        default=10,
        help="Maximum iterations for HighDimensionalRecon.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split resolved exams into this many shards (for job arrays).",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index to process when --num-shards > 1.",
    )
    return parser.parse_args()


def _load_split_ids(split_file: Path, split_names: Iterable[str]) -> list[str]:
    with split_file.open("r", encoding="utf-8") as f:
        split_data = json.load(f)
    ids: list[str] = []
    for name in split_names:
        if name not in split_data:
            raise KeyError(f"Split '{name}' not found in {split_file}.")
        ids.extend(split_data[name])
    seen = set()
    unique_ids = []
    for pid in ids:
        if pid not in seen:
            unique_ids.append(pid)
            seen.add(pid)
    if not unique_ids:
        raise RuntimeError(f"No patient IDs resolved from splits={list(split_names)}.")
    return unique_ids


def _resolve_h5_path(data_dir: Path, split_patient_id: str) -> Path:
    candidates = [
        data_dir / f"{split_patient_id}_2.h5",
        data_dir / f"{split_patient_id}.h5",
    ]
    existing = [p for p in candidates if p.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"Expected exactly one h5 file for patient '{split_patient_id}', found {len(existing)}. "
            f"Tried: {[str(c) for c in candidates]}"
        )
    return existing[0]


def _slice_indices_for_mode(h5_path: Path, mode: str, center_slice: int) -> list[int]:
    import h5py

    with h5py.File(h5_path, "r") as f:
        if "kspace" not in f:
            raise KeyError(f"'kspace' dataset missing in {h5_path}")
        n_slices = int(f["kspace"].shape[0])
    if mode == "all":
        return list(range(n_slices))
    if mode == "center":
        if not (0 <= center_slice < n_slices):
            raise ValueError(f"center_slice={center_slice} out of bounds for {h5_path} (n_slices={n_slices})")
        return [int(center_slice)]
    raise ValueError(f"Unsupported slice-mode: {mode}")


def _ordered_slices(
    pending_slices: list[int],
    all_slice_indices: list[int],
    priority_order: str,
) -> list[int]:
    if not pending_slices:
        return []
    if priority_order == "sequential":
        return sorted(pending_slices)
    if priority_order == "middle_first":
        if all_slice_indices:
            center = (min(all_slice_indices) + max(all_slice_indices)) / 2.0
        else:
            center = (min(pending_slices) + max(pending_slices)) / 2.0
        return sorted(pending_slices, key=lambda s: (abs(float(s) - center), int(s)))
    raise ValueError(f"Unsupported slice-priority-order: {priority_order}")


def _split_priority_and_remaining(
    pending_slices: list[int],
    all_slice_indices: list[int],
    priority_order: str,
    priority_slices_per_exam: int,
) -> tuple[list[int], list[int]]:
    ordered = _ordered_slices(pending_slices, all_slice_indices, priority_order)
    if priority_slices_per_exam <= 0:
        return [], ordered
    n_priority = min(int(priority_slices_per_exam), len(ordered))
    return ordered[:n_priority], ordered[n_priority:]


def _load_espirit_csmap(csmaps_dir: Path, patient_id: str, slice_idx: int) -> np.ndarray:
    def _normalize_csmap_shape(arr: np.ndarray, path: Path) -> np.ndarray:
        # Canonical shape expected by HighDimensionalRecon is [C, 1, H, W].
        if arr.ndim == 4:
            if arr.shape[1] == 1:
                return arr
            # Some exports store [1, C, H, W] instead of [C, 1, H, W].
            if arr.shape[0] == 1 and arr.shape[1] > 1:
                return np.transpose(arr, (1, 0, 2, 3))
            raise ValueError(f"Unsupported 4D cs-map shape {arr.shape} at {path}")

        if arr.ndim == 5:
            # Known variant: [1, 1, C, H, W] -> [C, 1, H, W]
            if arr.shape[0] == 1 and arr.shape[1] == 1 and arr.shape[2] > 1:
                return np.transpose(arr[0], (1, 0, 2, 3))
            # Known variant: [1, C, 1, H, W] -> [C, 1, H, W]
            if arr.shape[0] == 1 and arr.shape[2] == 1 and arr.shape[1] > 1:
                return arr[0]
            raise ValueError(f"Unsupported 5D cs-map shape {arr.shape} at {path}")

        raise ValueError(f"Expected cs-map ndim in {{4,5}}, got {arr.ndim} at {path}")

    csmap_path = csmaps_dir / f"{patient_id}_cs_maps" / f"cs_map_slice_{slice_idx:03d}.npy"
    if not csmap_path.is_file():
        raise FileNotFoundError(f"Missing cs-map: {csmap_path}")
    csmap = _normalize_csmap_shape(np.load(csmap_path), csmap_path)
    if csmap.shape[1] != 1:
        raise ValueError(f"Expected cs-map shape [C,1,H,W], got {csmap.shape} at {csmap_path}")
    if not np.isfinite(csmap).all():
        raise ValueError(f"Non-finite cs-map values found: {csmap_path}")
    return csmap


def _recon_single_slice(
    ksp_slice: np.ndarray,
    traj: np.ndarray,
    csmap: np.ndarray,
    device,
    max_iter: int,
    context: str = "",
) -> np.ndarray:
    import sigpy as sp
    from sigpy.mri import app
    if not hasattr(app, "HighDimensionalRecon"):
        raise RuntimeError(
            "sigpy.mri.app.HighDimensionalRecon is unavailable in this environment. "
            "Use the project SigPy build that includes GRASP reconstruction."
        )

    recon = app.HighDimensionalRecon(
        ksp_slice,
        csmap,
        combine_echo=False,
        lamda=0.001,
        coord=traj,
        regu="TV",
        regu_axes=[0],
        max_iter=max_iter,
        solver="ADMM",
        rho=0.1,
        device=device,
        show_pbar=False,
        verbose=False,
    ).run()
    recon_np = np.asarray(sp.to_device(recon, sp.cpu_device))
    recon_np = np.squeeze(recon_np)
    if recon_np.ndim != 3:
        suffix = f" ({context})" if context else ""
        raise RuntimeError(f"Unexpected recon shape after squeeze: {recon_np.shape}{suffix}")
    if not np.isfinite(recon_np).all():
        suffix = f" ({context})" if context else ""
        raise RuntimeError(f"Non-finite recon values encountered{suffix}.")
    return recon_np


def _is_valid_saved_target(path: Path, expected_frames: int) -> bool:
    """Return True if an on-disk target file is readable and shape-consistent."""
    try:
        arr = np.load(path)
    except Exception:
        return False
    if not np.isfinite(arr).all():
        return False
    if not np.iscomplexobj(arr):
        return False
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        return False
    # Accept any of the layouts consumed by RadialKspaceDataset:
    # (H, W, T), (T, H, W), (H, T, W)
    return expected_frames in arr.shape


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    """Atomically replace target file to survive interruptions safely."""
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}.npy")
    try:
        np.save(tmp_path, arr)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass


def _packed_h5_path(patient_out_dir: Path, spf: int, n_frames: int) -> Path:
    return patient_out_dir / f"grasp_recon_{spf}spf_{n_frames}frames.h5"


def _valid_slices_from_packed_h5(h5_path: Path, expected_frames: int) -> set[int]:
    import h5py

    if not h5_path.is_file():
        return set()

    with h5py.File(h5_path, "r") as f:
        if "valid" not in f or "recon" not in f:
            return set()
        valid_ds = f["valid"]
        recon_ds = f["recon"]
        if valid_ds.ndim != 1:
            raise RuntimeError(f"Packed GRASP file has invalid 'valid' ndim: {h5_path}")
        if recon_ds.ndim != 4:
            raise RuntimeError(f"Packed GRASP file has invalid 'recon' ndim: {h5_path}")
        if int(recon_ds.shape[-1]) != int(expected_frames):
            raise RuntimeError(
                f"Packed GRASP file frame mismatch at {h5_path}: "
                f"expected {expected_frames}, got {recon_ds.shape[-1]}"
            )
        valid_mask = np.asarray(valid_ds[:], dtype=bool)
        return set(np.where(valid_mask)[0].tolist())


def _open_packed_h5_for_write(
    h5_path: Path,
    n_slices: int,
    h: int,
    w: int,
    t: int,
    save_dtype: np.dtype,
):
    import h5py

    mode = "r+" if h5_path.is_file() else "w"
    f = h5py.File(h5_path, mode)
    if "recon" not in f or "valid" not in f:
        if mode == "r+":
            f.close()
            raise RuntimeError(
                f"Packed GRASP file exists but missing datasets ('recon'/'valid'): {h5_path}"
            )
        recon_ds = f.create_dataset(
            "recon",
            shape=(int(n_slices), int(h), int(w), int(t)),
            dtype=save_dtype,
            chunks=(1, int(h), int(w), int(t)),
        )
        valid_ds = f.create_dataset(
            "valid",
            shape=(int(n_slices),),
            dtype=np.bool_,
            chunks=(min(256, int(n_slices)),),
        )
        valid_ds[:] = False
        f.attrs["format_version"] = 1
        f.attrs["layout"] = "packed_per_patient_spf"
        f.attrs["num_frames"] = int(t)
        return f, recon_ds, valid_ds

    recon_ds = f["recon"]
    valid_ds = f["valid"]
    expected_shape = (int(n_slices), int(h), int(w), int(t))
    if tuple(recon_ds.shape) != expected_shape:
        f.close()
        raise RuntimeError(
            f"Packed GRASP shape mismatch at {h5_path}: expected {expected_shape}, got {tuple(recon_ds.shape)}"
        )
    if valid_ds.shape != (int(n_slices),):
        f.close()
        raise RuntimeError(
            f"Packed GRASP valid-shape mismatch at {h5_path}: expected {(int(n_slices),)}, got {tuple(valid_ds.shape)}"
        )
    return f, recon_ds, valid_ds


def _array_debug_summary(arr: np.ndarray | None) -> dict[str, object]:
    if arr is None:
        return {}
    arr_np = np.asarray(arr)
    summary: dict[str, object] = {
        "shape": list(arr_np.shape),
        "dtype": str(arr_np.dtype),
        "numel": int(arr_np.size),
        "all_finite": bool(np.isfinite(arr_np).all()),
    }
    if arr_np.size:
        abs_arr = np.abs(arr_np)
        finite_abs = abs_arr[np.isfinite(abs_arr)]
        if finite_abs.size:
            summary["abs_min"] = float(np.min(finite_abs))
            summary["abs_max"] = float(np.max(finite_abs))
            summary["abs_mean"] = float(np.mean(finite_abs))
    return summary


def _to_hwt(arr: np.ndarray, expected_frames: int, context: str = "") -> np.ndarray:
    arr_np = np.squeeze(np.asarray(arr))
    if arr_np.ndim != 3:
        suffix = f" ({context})" if context else ""
        raise RuntimeError(f"Expected 3D recon, got {arr_np.shape}{suffix}")
    if int(arr_np.shape[-1]) == int(expected_frames):
        return arr_np
    if int(arr_np.shape[0]) == int(expected_frames):
        return np.transpose(arr_np, (1, 2, 0))
    if int(arr_np.shape[1]) == int(expected_frames):
        return np.transpose(arr_np, (0, 2, 1))
    suffix = f" ({context})" if context else ""
    raise RuntimeError(
        f"Could not map recon shape {arr_np.shape} to HWT with expected_frames={expected_frames}{suffix}"
    )


def _acquire_lock(lock_path: Path, timeout_sec: int = 120, poll_sec: float = 1.0) -> int:
    import fcntl

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o664)
    deadline = time.time() + float(timeout_sec)
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.ftruncate(fd, 0)
            os.write(fd, f"{socket.gethostname()} pid={os.getpid()} time={time.time()}\n".encode("utf-8"))
            os.fsync(fd)
            return fd
        except BlockingIOError:
            if time.time() >= deadline:
                os.close(fd)
                raise TimeoutError(f"Timeout acquiring lock: {lock_path}")
            time.sleep(float(poll_sec))


def _release_lock(lock_fd: int, lock_path: Path) -> None:
    import fcntl

    try:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    finally:
        return


def main() -> None:
    args = _parse_args()
    import sigpy as sp
    import torch
    from raw_kspace_eval import process_kspace

    if not args.data_dir.is_dir():
        raise NotADirectoryError(f"--data-dir not found: {args.data_dir}")
    if not args.target_root.is_dir():
        raise NotADirectoryError(f"--target-root not found: {args.target_root}")
    if not args.csmaps_dir.is_dir():
        raise NotADirectoryError(f"--csmaps-dir not found: {args.csmaps_dir}")
    if args.total_spokes <= 0:
        raise ValueError("--total-spokes must be > 0.")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1.")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards.")
    save_dtype = np.complex64 if args.save_dtype == "complex64" else np.complex128

    split_patient_ids = _load_split_ids(args.split_file, args.splits)
    if args.num_shards > 1:
        split_patient_ids = [
            pid for idx, pid in enumerate(split_patient_ids) if (idx % args.num_shards) == args.shard_index
        ]
    if args.max_exams > 0:
        split_patient_ids = split_patient_ids[: args.max_exams]

    device = sp.Device(0 if torch.cuda.is_available() else -1)
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "nojid")
    slurm_procid = os.environ.get("SLURM_PROCID", "noproc")
    hostname = socket.gethostname()
    failure_dir = args.target_root / "_grasp_failures"
    failure_dir.mkdir(parents=True, exist_ok=True)
    failure_log_path = failure_dir / (
        f"skipped_shard{args.shard_index:04d}_of_{args.num_shards:04d}"
        f"_job{slurm_job_id}_proc{slurm_procid}_{hostname}_pid{os.getpid()}.jsonl"
    )
    failure_summary_path = failure_log_path.with_suffix(".summary.json")
    skipped_slices_path = failure_log_path.with_suffix(".skipped_slices.tsv")

    print(
        f"[GRASP] exams={len(split_patient_ids)}, splits={args.splits}, "
        f"spokes_per_frame={args.spokes_per_frame}, slice_mode={args.slice_mode}, "
        f"slice_priority_order={args.slice_priority_order}, "
        f"priority_slices_per_exam={args.priority_slices_per_exam}, "
        f"shard={args.shard_index}/{args.num_shards - 1}, "
        f"failure_log={failure_log_path}"
    )

    patient_specs: list[dict[str, object]] = []
    for split_pid in split_patient_ids:
        h5_path = _resolve_h5_path(args.data_dir, split_pid)
        patient_id = h5_path.stem
        slice_indices = _slice_indices_for_mode(h5_path, args.slice_mode, args.center_slice)
        patient_specs.append(
            {
                "h5_path": h5_path,
                "patient_id": patient_id,
                "slice_indices": slice_indices,
                "out_dir": args.target_root / patient_id,
            }
        )

    # Build per-exam/per-spf worklists once, then execute in two phases:
    # phase 1 covers N priority slices per exam (for diversity),
    # phase 2 fills remaining slices.
    worklists: dict[tuple[str, int], dict[str, object]] = {}
    saved = 0
    skipped_existing_valid = 0
    stale_existing_recomputed = 0
    skipped_exam_failures = 0
    skipped_slice_failures = 0
    skipped_slice_rows: list[tuple[str, int, int, str, str, str]] = []
    failure_log_fh = failure_log_path.open("a", encoding="utf-8")
    for spec in patient_specs:
        patient_id = str(spec["patient_id"])
        patient_out_dir = Path(spec["out_dir"])
        slice_indices = list(spec["slice_indices"])
        patient_out_dir.mkdir(parents=True, exist_ok=True)
        for spf in args.spokes_per_frame:
            if args.total_spokes % spf != 0:
                raise ValueError(f"total_spokes={args.total_spokes} not divisible by spf={spf}")
            n_frames = args.total_spokes // spf
            packed_h5_path = _packed_h5_path(patient_out_dir, int(spf), int(n_frames))
            valid_from_h5 = set()
            if not args.overwrite and packed_h5_path.exists():
                valid_from_h5 = _valid_slices_from_packed_h5(
                    packed_h5_path,
                    expected_frames=int(n_frames),
                )

            pending_slices = []
            for sidx in slice_indices:
                if not args.overwrite and int(sidx) in valid_from_h5:
                    skipped_existing_valid += 1
                    continue

                # Legacy fallback for old per-slice npy layout.
                legacy_npy_path = patient_out_dir / f"grasp_recon_{spf}spf_{n_frames}frames_slice{sidx}.npy"
                if legacy_npy_path.exists() and not args.overwrite:
                    if _is_valid_saved_target(legacy_npy_path, expected_frames=int(n_frames)):
                        skipped_existing_valid += 1
                        continue
                    stale_existing_recomputed += 1
                pending_slices.append(sidx)

            priority_slices, remaining_slices = _split_priority_and_remaining(
                pending_slices=pending_slices,
                all_slice_indices=slice_indices,
                priority_order=str(args.slice_priority_order),
                priority_slices_per_exam=int(args.priority_slices_per_exam),
            )
            worklists[(patient_id, int(spf))] = {
                "n_frames": int(n_frames),
                "packed_h5_path": str(packed_h5_path),
                "priority_slices": priority_slices,
                "remaining_slices": remaining_slices,
            }

    phase_specs: list[tuple[str, str]] = []
    if int(args.priority_slices_per_exam) > 0:
        phase_specs.append(("priority", "priority_slices"))
    phase_specs.append(("remaining", "remaining_slices"))

    try:
        for phase_name, phase_key in phase_specs:
            total_phase_slices = sum(
                len(list(worklists[(str(spec["patient_id"]), int(spf))][phase_key]))
                for spec in patient_specs
                for spf in args.spokes_per_frame
            )
            if total_phase_slices == 0:
                continue
            print(f"[GRASP] phase={phase_name}, total_slices={total_phase_slices}")

            for i, spec in enumerate(patient_specs, start=1):
                h5_path = Path(spec["h5_path"])
                patient_id = str(spec["patient_id"])

                for spf in args.spokes_per_frame:
                    work = worklists[(patient_id, int(spf))]
                    target_slices = list(work[phase_key])
                    if not target_slices:
                        continue

                    n_frames = int(work["n_frames"])
                    packed_h5_path = Path(str(work["packed_h5_path"]))
                    lock_path = packed_h5_path.with_suffix(packed_h5_path.suffix + ".lock")
                    print(
                        f"[{phase_name}] [{i}/{len(patient_specs)}] "
                        f"patient={patient_id}, spf={spf}, slices={len(target_slices)}"
                    )

                    lock_fd = None
                    h5_out = None
                    try:
                        try:
                            lock_fd = _acquire_lock(lock_path, timeout_sec=7200, poll_sec=1.0)
                        except Exception as exc:
                            skipped_exam_failures += 1
                            failure_record = {
                                "kind": "exam_lock_failure",
                                "phase": phase_name,
                                "patient_id": patient_id,
                                "h5_path": str(h5_path),
                                "slice_idx": None,
                                "spf": int(spf),
                                "n_frames": int(n_frames),
                                "lock_path": str(lock_path),
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                                "traceback": traceback.format_exc(),
                            }
                            failure_log_fh.write(json.dumps(failure_record, sort_keys=True) + "\n")
                            failure_log_fh.flush()
                            print(
                                f"[GRASP][WARN] lock failure for patient={patient_id}, spf={int(spf)} "
                                f"phase={phase_name}: {type(exc).__name__}: {exc}"
                            )
                            continue

                        try:
                            zf_kspace, binned_kspace, traj = process_kspace(
                                str(h5_path),
                                device=device,
                                spokes_per_frame=spf,
                                images_per_slab=args.images_per_slab,
                                center_partition=args.center_partition,
                            )
                        except Exception as exc:
                            skipped_exam_failures += 1
                            failure_record = {
                                "kind": "exam_failure",
                                "phase": phase_name,
                                "patient_id": patient_id,
                                "h5_path": str(h5_path),
                                "slice_idx": None,
                                "spf": int(spf),
                                "n_frames": int(n_frames),
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                                "traceback": traceback.format_exc(),
                            }
                            failure_log_fh.write(json.dumps(failure_record, sort_keys=True) + "\n")
                            failure_log_fh.flush()
                            print(
                                f"[GRASP][WARN] skipping patient={patient_id}, spf={int(spf)} "
                                f"phase={phase_name} due to {type(exc).__name__}: {exc}"
                            )
                            continue

                        n_available_slices = int(zf_kspace.shape[0])
                        recon_ds = None
                        valid_ds = None
                        h5_needs_init = True
                        for sidx in target_slices:
                            context = f"patient={patient_id}, slice={int(sidx)}, spf={int(spf)}"
                            csmap = None
                            ksp_slice = None
                            out_path = packed_h5_path
                            try:
                                if not (0 <= int(sidx) < n_available_slices):
                                    raise IndexError(
                                        f"Slice {sidx} out of bounds for {h5_path} (available={n_available_slices})"
                                    )
                                csmap = _load_espirit_csmap(args.csmaps_dir, patient_id, int(sidx))
                                ksp_slice = binned_kspace[int(sidx)]
                                recon = _recon_single_slice(
                                    ksp_slice=ksp_slice,
                                    traj=traj,
                                    csmap=csmap,
                                    device=device,
                                    max_iter=int(args.recon_max_iter),
                                    context=context,
                                )
                                recon_hwt = _to_hwt(recon, expected_frames=int(n_frames), context=context)
                                if h5_needs_init:
                                    h5_out, recon_ds, valid_ds = _open_packed_h5_for_write(
                                        h5_path=packed_h5_path,
                                        n_slices=n_available_slices,
                                        h=int(recon_hwt.shape[0]),
                                        w=int(recon_hwt.shape[1]),
                                        t=int(recon_hwt.shape[2]),
                                        save_dtype=save_dtype,
                                    )
                                    h5_needs_init = False
                                recon_ds[int(sidx), ...] = recon_hwt.astype(save_dtype, copy=False)
                                valid_ds[int(sidx)] = True
                                h5_out.flush()
                                saved += 1
                            except Exception as exc:
                                skipped_slice_failures += 1
                                skipped_slice_rows.append(
                                    (
                                        patient_id,
                                        int(sidx),
                                        int(spf),
                                        phase_name,
                                        type(exc).__name__,
                                        str(exc),
                                    )
                                )
                                failure_record = {
                                    "kind": "slice_failure",
                                    "phase": phase_name,
                                    "patient_id": patient_id,
                                    "h5_path": str(h5_path),
                                    "slice_idx": int(sidx),
                                    "spf": int(spf),
                                    "n_frames": int(n_frames),
                                    "out_path": str(out_path),
                                    "error_type": type(exc).__name__,
                                    "error_message": str(exc),
                                    "traceback": traceback.format_exc(),
                                    "ksp_slice": _array_debug_summary(ksp_slice),
                                    "csmap": _array_debug_summary(csmap),
                                    "traj": _array_debug_summary(traj),
                                }
                                failure_log_fh.write(json.dumps(failure_record, sort_keys=True) + "\n")
                                failure_log_fh.flush()
                                print(
                                    f"[GRASP][WARN] skipped {context} due to {type(exc).__name__}: {exc}"
                                )
                                continue
                    finally:
                        if h5_out is not None:
                            h5_out.close()
                        if lock_fd is not None:
                            _release_lock(lock_fd, lock_path)
    finally:
        failure_log_fh.close()

    if skipped_slice_rows:
        with skipped_slices_path.open("w", encoding="utf-8") as f:
            f.write("patient_id\tslice_idx\tspf\tphase\terror_type\terror_message\n")
            for row in skipped_slice_rows:
                patient_id, slice_idx, spf, phase_name, error_type, error_message = row
                msg = error_message.replace("\t", " ").replace("\n", " ")
                f.write(
                    f"{patient_id}\t{slice_idx}\t{spf}\t{phase_name}\t{error_type}\t{msg}\n"
                )

    summary_payload = {
        "saved": int(saved),
        "skipped_existing_valid": int(skipped_existing_valid),
        "stale_existing_recomputed": int(stale_existing_recomputed),
        "skipped_exam_failures": int(skipped_exam_failures),
        "skipped_slice_failures": int(skipped_slice_failures),
        "failure_log_path": str(failure_log_path),
        "skipped_slices_path": str(skipped_slices_path) if skipped_slice_rows else None,
        "save_dtype": args.save_dtype,
        "target_root": str(args.target_root),
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
    }
    with failure_summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, sort_keys=True)

    print(
        f"[GRASP] done. saved={saved}, skipped_existing_valid={skipped_existing_valid}, "
        f"stale_existing_recomputed={stale_existing_recomputed}, "
        f"skipped_exam_failures={skipped_exam_failures}, "
        f"skipped_slice_failures={skipped_slice_failures}, "
        f"save_dtype={args.save_dtype}, target_root={args.target_root}, "
        f"failure_log={failure_log_path}, summary={failure_summary_path}"
    )


if __name__ == "__main__":
    main()
