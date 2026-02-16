#!/usr/bin/env python3
"""Migrate legacy per-slice GRASP .npy targets into packed per-patient HDF5 files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import socket
import time
import traceback

import h5py
import numpy as np


SLICE_RE = re.compile(r"slice(\d+)\.npy$")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pack legacy GRASP slice files into HDF5.")
    p.add_argument("--source-root", type=Path, required=True, help="Legacy root with per-slice .npy files.")
    p.add_argument("--target-root", type=Path, required=True, help="Packed output root.")
    p.add_argument("--data-dir", type=Path, required=True, help="Raw k-space root to resolve n_slices per exam.")
    p.add_argument("--spokes-per-frame", type=int, nargs="+", default=[36, 2], help="SPF values to migrate.")
    p.add_argument("--total-spokes", type=int, default=288, help="Total spokes used to derive frame count.")
    p.add_argument("--save-dtype", choices=("complex64", "complex128"), default="complex64")
    p.add_argument("--delete-source", action="store_true", help="Delete legacy file only after successful pack.")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--max-exams", type=int, default=0, help="Debug cap after sharding (0 means no cap).")
    return p.parse_args()


def _resolve_h5_path(data_dir: Path, patient_id: str) -> Path:
    candidates = [
        data_dir / f"{patient_id}_2.h5",
        data_dir / f"{patient_id}.h5",
    ]
    existing = [p for p in candidates if p.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"Expected exactly one raw h5 for patient '{patient_id}', found {len(existing)}. "
            f"Tried: {[str(c) for c in candidates]}"
        )
    return existing[0]


def _num_slices_from_raw_h5(h5_path: Path) -> int:
    with h5py.File(h5_path, "r") as f:
        if "kspace" not in f:
            raise KeyError(f"'kspace' missing in {h5_path}")
        return int(f["kspace"].shape[0])


def _to_hwt(arr: np.ndarray, expected_frames: int, context: str = "") -> np.ndarray:
    arr_np = np.squeeze(np.asarray(arr))
    if arr_np.ndim != 3:
        suffix = f" ({context})" if context else ""
        raise RuntimeError(f"Expected 3D array, got {arr_np.shape}{suffix}")
    if int(arr_np.shape[-1]) == int(expected_frames):
        return arr_np
    if int(arr_np.shape[0]) == int(expected_frames):
        return np.transpose(arr_np, (1, 2, 0))
    if int(arr_np.shape[1]) == int(expected_frames):
        return np.transpose(arr_np, (0, 2, 1))
    suffix = f" ({context})" if context else ""
    raise RuntimeError(
        f"Could not map shape {arr_np.shape} to HWT with expected_frames={expected_frames}{suffix}"
    )


def _acquire_lock(lock_path: Path, timeout_sec: int = 7200, poll_sec: float = 1.0) -> int:
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


def _release_lock(lock_fd: int) -> None:
    import fcntl

    try:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
    finally:
        os.close(lock_fd)


def _open_or_create_packed(
    packed_h5_path: Path,
    n_slices: int,
    h: int,
    w: int,
    t: int,
    save_dtype: np.dtype,
):
    mode = "r+" if packed_h5_path.is_file() else "w"
    f = h5py.File(packed_h5_path, mode)
    if "recon" not in f or "valid" not in f:
        if mode == "r+":
            f.close()
            raise RuntimeError(
                f"Packed target exists but missing recon/valid datasets: {packed_h5_path}"
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
            f"Packed shape mismatch at {packed_h5_path}: expected {expected_shape}, got {tuple(recon_ds.shape)}"
        )
    if tuple(valid_ds.shape) != (int(n_slices),):
        f.close()
        raise RuntimeError(
            f"Packed valid shape mismatch at {packed_h5_path}: expected {(int(n_slices),)}, got {tuple(valid_ds.shape)}"
        )
    return f, recon_ds, valid_ds


def _iter_patient_dirs(source_root: Path) -> list[Path]:
    return sorted([p for p in source_root.iterdir() if p.is_dir()])


def main() -> None:
    args = _parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError(f"--shard-index must be in [0, {args.num_shards - 1}]")
    if not args.source_root.is_dir():
        raise NotADirectoryError(args.source_root)
    args.target_root.mkdir(parents=True, exist_ok=True)

    save_dtype = np.complex64 if args.save_dtype == "complex64" else np.complex128
    migration_log_dir = args.target_root / "_migration_logs"
    migration_log_dir.mkdir(parents=True, exist_ok=True)

    patient_dirs_all = _iter_patient_dirs(args.source_root)
    patient_dirs = [p for i, p in enumerate(patient_dirs_all) if (i % args.num_shards) == args.shard_index]
    if args.max_exams > 0:
        patient_dirs = patient_dirs[: int(args.max_exams)]

    summary = {
        "source_root": str(args.source_root),
        "target_root": str(args.target_root),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "patients_total": int(len(patient_dirs_all)),
        "patients_in_shard": int(len(patient_dirs)),
        "files_migrated": 0,
        "files_deleted": 0,
        "files_skipped_existing": 0,
        "files_invalid": 0,
        "exam_failures": 0,
    }
    failures_path = migration_log_dir / f"failures_shard{int(args.shard_index):03d}.jsonl"
    summary_path = migration_log_dir / f"summary_shard{int(args.shard_index):03d}.json"

    with failures_path.open("a", encoding="utf-8") as failure_fh:
        for pidx, patient_dir in enumerate(patient_dirs, start=1):
            patient_id = patient_dir.name
            raw_h5_path = _resolve_h5_path(args.data_dir, patient_id)
            n_slices = _num_slices_from_raw_h5(raw_h5_path)

            for spf in args.spokes_per_frame:
                if args.total_spokes % int(spf) != 0:
                    raise ValueError(
                        f"total_spokes={args.total_spokes} is not divisible by spf={int(spf)}"
                    )
                n_frames = int(args.total_spokes // int(spf))
                pattern = f"grasp_recon_{int(spf)}spf_{int(n_frames)}frames_slice*.npy"
                source_files = sorted(patient_dir.glob(pattern))
                if not source_files:
                    continue

                target_patient_dir = args.target_root / patient_id
                target_patient_dir.mkdir(parents=True, exist_ok=True)
                packed_h5_path = target_patient_dir / f"grasp_recon_{int(spf)}spf_{int(n_frames)}frames.h5"
                lock_path = packed_h5_path.with_suffix(packed_h5_path.suffix + ".lock")

                print(
                    f"[PACK] [{pidx}/{len(patient_dirs)}] patient={patient_id} spf={int(spf)} "
                    f"legacy_files={len(source_files)}"
                )

                lock_fd = None
                h5_out = None
                try:
                    lock_fd = _acquire_lock(lock_path, timeout_sec=7200, poll_sec=1.0)
                    recon_ds = None
                    valid_ds = None
                    for source_path in source_files:
                        m = SLICE_RE.search(source_path.name)
                        if m is None:
                            summary["files_invalid"] += 1
                            continue
                        sidx = int(m.group(1))
                        if not (0 <= sidx < n_slices):
                            summary["files_invalid"] += 1
                            continue

                        if h5_out is not None and bool(valid_ds[sidx]):
                            summary["files_skipped_existing"] += 1
                            if args.delete_source:
                                source_path.unlink(missing_ok=True)
                                summary["files_deleted"] += 1
                            continue

                        arr = np.load(source_path)
                        if not np.isfinite(arr).all() or not np.iscomplexobj(arr):
                            summary["files_invalid"] += 1
                            continue
                        arr_hwt = _to_hwt(
                            arr,
                            expected_frames=n_frames,
                            context=f"patient={patient_id},spf={int(spf)},slice={sidx}",
                        )

                        if h5_out is None:
                            h5_out, recon_ds, valid_ds = _open_or_create_packed(
                                packed_h5_path=packed_h5_path,
                                n_slices=n_slices,
                                h=int(arr_hwt.shape[0]),
                                w=int(arr_hwt.shape[1]),
                                t=int(arr_hwt.shape[2]),
                                save_dtype=save_dtype,
                            )
                        recon_ds[sidx, ...] = arr_hwt.astype(save_dtype, copy=False)
                        valid_ds[sidx] = True
                        h5_out.flush()
                        summary["files_migrated"] += 1

                        if args.delete_source:
                            source_path.unlink(missing_ok=True)
                            summary["files_deleted"] += 1
                except Exception as exc:
                    summary["exam_failures"] += 1
                    failure_record = {
                        "kind": "pack_failure",
                        "patient_id": patient_id,
                        "spf": int(spf),
                        "n_frames": int(n_frames),
                        "source_root": str(args.source_root),
                        "target_root": str(args.target_root),
                        "packed_h5_path": str(packed_h5_path),
                        "raw_h5_path": str(raw_h5_path),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    failure_fh.write(json.dumps(failure_record, sort_keys=True) + "\n")
                    failure_fh.flush()
                    print(
                        f"[PACK][WARN] failed patient={patient_id} spf={int(spf)} "
                        f"due to {type(exc).__name__}: {exc}"
                    )
                finally:
                    if h5_out is not None:
                        h5_out.close()
                    if lock_fd is not None:
                        _release_lock(lock_fd)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(
        "[PACK] done "
        f"files_migrated={summary['files_migrated']} "
        f"files_deleted={summary['files_deleted']} "
        f"files_skipped_existing={summary['files_skipped_existing']} "
        f"files_invalid={summary['files_invalid']} "
        f"exam_failures={summary['exam_failures']}"
    )


if __name__ == "__main__":
    main()
