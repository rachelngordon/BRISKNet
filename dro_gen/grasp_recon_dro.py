#!/usr/bin/env python3
"""Run a GRASP reconstruction for a DRO k-space .mat file using ESPIRiT csmaps. Run: python3 dro_gen/grasp_recon_dro.py --help

Example:
  python3 dro_gen/grasp_recon_dro.py --sample-id sample_001 --spokes-per-frame 36
"""
import argparse
import json
import os
import re

import h5py
import h5py.h5t as h5t
import numpy as np
import sigpy as sp
from sigpy.mri import app
import torch

KSPACE_KEY = "kspace"
TRAJ_KEY = "traj"


def _read_h5_float(dset: h5py.Dataset) -> np.ndarray:
    type_id = dset.id.get_type()
    if type_id.get_class() != h5t.FLOAT:
        raise TypeError(f"Expected float dataset; got class={type_id.get_class()}")
    size = type_id.get_size()
    if size == 4:
        np_dtype = np.float32
    elif size == 8:
        np_dtype = np.float64
    else:
        raise TypeError(f"Unsupported float size for HDF5 dtype: {size}")
    arr = np.empty(dset.shape, dtype=np_dtype)
    dset.read_direct(arr)
    return arr


def _read_h5_complex(dset: h5py.Dataset) -> np.ndarray:
    type_id = dset.id.get_type()
    type_class = type_id.get_class()
    if type_class == h5t.COMPOUND:
        memtype = h5t.create(h5t.COMPOUND, type_id.get_size())
        names = []
        formats = []
        offsets = []
        for idx in range(type_id.get_nmembers()):
            name = type_id.get_member_name(idx)
            name_str = name.decode() if isinstance(name, (bytes, bytearray)) else str(name)
            member = type_id.get_member_type(idx)
            if member.get_class() != h5t.FLOAT:
                raise TypeError("Unsupported compound member type in complex dataset.")
            size = member.get_size()
            if size == 4:
                np_fmt = np.float32
                native = h5t.NATIVE_FLOAT
            elif size == 8:
                np_fmt = np.float64
                native = h5t.NATIVE_DOUBLE
            else:
                raise TypeError(f"Unsupported float size for HDF5 dtype: {size}")
            memtype.insert(name, type_id.get_member_offset(idx), native)
            names.append(name_str)
            formats.append(np_fmt)
            offsets.append(type_id.get_member_offset(idx))
        np_dtype = np.dtype(
            {
                "names": names,
                "formats": formats,
                "offsets": offsets,
                "itemsize": type_id.get_size(),
            }
        )
        arr = np.empty(dset.shape, dtype=np_dtype)
        dset.id.read(h5py.h5s.ALL, h5py.h5s.ALL, arr, memtype)
        if "real" in arr.dtype.names and "imag" in arr.dtype.names:
            return arr["real"] + 1j * arr["imag"]
        if len(arr.dtype.names) >= 2:
            return arr[arr.dtype.names[0]] + 1j * arr[arr.dtype.names[1]]
        raise ValueError("Compound dataset did not contain real/imag fields.")
    if type_class == h5t.FLOAT:
        real = _read_h5_float(dset)
        return real.astype(np.complex64)
    raise TypeError(f"Unsupported HDF5 dtype class for complex read: {type_class}")


def _load_kspace_and_traj(kspace_path: str):
    with h5py.File(kspace_path, "r") as f:
        if KSPACE_KEY not in f:
            raise KeyError(f"{kspace_path} missing required key '{KSPACE_KEY}'.")
        if TRAJ_KEY not in f:
            raise KeyError(f"{kspace_path} missing required key '{TRAJ_KEY}'.")
        kspace = _read_h5_complex(f[KSPACE_KEY]).astype(np.complex64)
        traj = _read_h5_complex(f[TRAJ_KEY]).astype(np.complex64)
    return kspace, traj


def _normalize_traj(traj: np.ndarray) -> np.ndarray:
    if traj.ndim != 2:
        raise ValueError(f"Unexpected traj shape {traj.shape}; expected 2D array.")
    return traj


def _kspace_matches_dims(kspace: np.ndarray, spokes: int, samples: int) -> bool:
    spoke_axes = [i for i, s in enumerate(kspace.shape) if s == spokes]
    sample_axes = [i for i, s in enumerate(kspace.shape) if s == samples]
    return len(spoke_axes) == 1 and len(sample_axes) == 1


def _infer_expected_samples_from_csmaps(csmaps: np.ndarray) -> int | None:
    if csmaps.ndim != 3:
        return None
    dims = sorted(csmaps.shape)
    spatial = dims[-2:]
    if spatial[0] == spatial[1]:
        return int(spatial[0] * 2)
    return None


def _select_traj_orientation(
    traj: np.ndarray,
    kspace: np.ndarray,
    spf: int | None,
    expected_samples: int | None,
) -> tuple[np.ndarray, bool]:
    traj = _normalize_traj(traj)
    candidates = [(traj, False)]
    if traj.shape[0] != traj.shape[1]:
        candidates.append((traj.T, True))

    scored = []
    for cand, transposed in candidates:
        spokes, samples = cand.shape
        if not _kspace_matches_dims(kspace, spokes, samples):
            continue
        score = 0
        if spf is not None and spokes % spf != 0:
            score += 10
        if expected_samples is not None and samples != expected_samples:
            score += 5
        scored.append((score, transposed, cand))

    if not scored:
        raise ValueError(
            f"Could not align traj with kspace; traj shape {traj.shape}, kspace shape {kspace.shape}."
        )

    scored.sort(key=lambda x: (x[0], 0 if not x[1] else 1))
    score, transposed, best = scored[0]
    if transposed:
        print("[traj] Using transposed traj to match kspace/spokes-per-frame.")
    return best, transposed


def _align_kspace_to_traj(kspace: np.ndarray, total_spokes: int, samples: int) -> np.ndarray:
    if kspace.ndim != 3:
        raise ValueError(f"Unexpected kspace shape {kspace.shape}; expected 3D array.")

    spoke_axes = [i for i, s in enumerate(kspace.shape) if s == total_spokes]
    sample_axes = [i for i, s in enumerate(kspace.shape) if s == samples]
    if len(spoke_axes) != 1 or len(sample_axes) != 1:
        raise ValueError(
            f"Could not align kspace shape {kspace.shape} to traj "
            f"(spokes={total_spokes}, samples={samples})."
        )

    spoke_axis = spoke_axes[0]
    sample_axis = sample_axes[0]
    coil_axis = [i for i in range(3) if i not in (spoke_axis, sample_axis)][0]
    kspace = np.moveaxis(kspace, (coil_axis, spoke_axis, sample_axis), (0, 1, 2))
    return kspace


def _standardize_csmaps(csmaps: np.ndarray, n_coils: int) -> np.ndarray:
    if csmaps.ndim != 3:
        raise ValueError(f"Unexpected csmaps shape {csmaps.shape}; expected (coils, H, W).")
    if csmaps.shape[0] != n_coils and csmaps.shape[-1] == n_coils:
        csmaps = np.transpose(csmaps, (2, 0, 1))
    if csmaps.shape[0] != n_coils:
        raise ValueError(
            f"CSMAP coil count mismatch: expected {n_coils}, got {csmaps.shape[0]}."
        )
    return csmaps.astype(np.complex64)


def _load_csmaps(csmap_path: str, n_coils: int) -> np.ndarray:
    csmaps = np.load(csmap_path)
    return _standardize_csmaps(csmaps, n_coils)


def _scale_traj_if_needed(traj: np.ndarray, samples: int) -> np.ndarray:
    max_abs = float(np.max(np.abs(traj)))
    base_res = samples // 2
    if max_abs <= 1.0:
        traj = traj * base_res
        print(f"[traj] Scaling normalized traj by base_res={base_res} (max_abs={max_abs:.3g}).")
    return traj


def _build_coord(traj: np.ndarray, num_frames: int, spf: int, samples: int) -> np.ndarray:
    traj = np.ascontiguousarray(traj.reshape(num_frames, spf, samples))
    return np.stack([traj.imag, traj.real], axis=-1).astype(np.float32)


def _build_kspace(
    kspace: np.ndarray, n_coils: int, num_frames: int, spf: int, samples: int
) -> np.ndarray:
    kspace = np.ascontiguousarray(kspace.reshape(n_coils, num_frames, spf, samples))
    kspace = np.transpose(kspace, (1, 0, 2, 3))
    return kspace[:, None, :, None, :, :]


def _parse_spf_from_name(path: str) -> int:
    match = re.search(r"_kspace_(\d+)spf", os.path.basename(path))
    if match is None:
        raise ValueError(f"Could not parse spokes/frame from filename: {path}")
    return int(match.group(1))


def _parse_sample_id_from_name(path: str) -> str:
    fname = os.path.basename(path)
    if "_kspace_" not in fname:
        raise ValueError(f"Could not parse sample id from filename: {path}")
    return fname.split("_kspace_")[0]


def _resolve_sample_id(sample_id: str) -> str:
    if sample_id is None:
        return sample_id
    if not sample_id.startswith("sample_"):
        return f"sample_{sample_id}"
    return sample_id


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a single GRASP recon for DRO k-space using ESPIRiT CSMAPS."
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="DRO sample id (e.g., sample_005_sub5).",
    )
    parser.add_argument(
        "--spokes-per-frame",
        type=int,
        default=None,
        help="Spokes per frame (e.g., 2).",
    )
    parser.add_argument(
        "--total-spokes",
        type=int,
        default=288,
        help="Total spokes (spf * frames).",
    )
    parser.add_argument(
        "--spokes-per-frame-list",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16, 24, 36],
        help="Spokes per frame list for loop mode.",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default="/net/scratch2/rachelgordon/dro_var_frames_kspace",
        help="Root directory containing DRO kspace .mat files.",
    )
    parser.add_argument(
        "--csmaps-dir",
        type=str,
        default="/net/scratch2/rachelgordon/dro_var_frames_kspace/csmaps_espirit",
        help="Directory containing ESPIRiT CSMAPS .npy files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/net/scratch2/rachelgordon/dro_var_frames_kspace/espirit_grasp_recons",
        help="Output directory for GRASP reconstructions.",
    )
    parser.add_argument(
        "--kspace-path",
        type=str,
        default=None,
        help="Optional explicit path to kspace .mat file (overrides root/sample/spf).",
    )
    parser.add_argument(
        "--csmaps-path",
        type=str,
        default=None,
        help="Optional explicit path to csmaps .npy file (overrides csmaps-dir/sample-id).",
    )
    parser.add_argument("--lamda", type=float, default=0.001, help="GRASP TV regularization weight.")
    parser.add_argument("--max-iter", type=int, default=10, help="GRASP max iterations.")
    parser.add_argument("--rho", type=float, default=0.1, help="GRASP ADMM rho.")
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="SigPy device id (e.g., 0 for GPU, -1 for CPU). Defaults to auto.",
    )
    parser.add_argument(
        "--loop-val-dro",
        action="store_true",
        help="Loop over val_dro samples in data/data_split.json.",
    )
    parser.add_argument(
        "--split-file",
        type=str,
        default="data/data_split.json",
        help="Path to data split JSON.",
    )
    parser.add_argument(
        "--split-key",
        type=str,
        default="val_dro",
        help="Split key to use from data split JSON.",
    )
    return parser.parse_args()


def _load_split_ids(split_file: str, split_key: str):
    with open(split_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if split_key not in data:
        raise KeyError(f"Split key '{split_key}' not found in {split_file}.")
    if not isinstance(data[split_key], list):
        raise ValueError(f"Split '{split_key}' in {split_file} is not a list.")
    return data[split_key]


def _resolve_device(device_id: int | None) -> sp.Device:
    if device_id is None:
        return sp.Device(0 if torch.cuda.is_available() else -1)
    return sp.Device(device_id)


def _run_recon(sample_id: str, spf: int, num_frames: int, kspace_path: str, csmaps_path: str, args):
    if not os.path.exists(kspace_path):
        raise FileNotFoundError(f"K-space file not found: {kspace_path}")
    if not os.path.exists(csmaps_path):
        raise FileNotFoundError(f"CSMAP file not found: {csmaps_path}")

    print(f"[input] kspace: {kspace_path}")
    print(f"[input] csmaps: {csmaps_path}")
    print(f"[input] sample_id={sample_id}, spokes_per_frame={spf}")

    kspace, traj = _load_kspace_and_traj(kspace_path)
    csmaps_raw = np.load(csmaps_path)
    expected_samples = _infer_expected_samples_from_csmaps(csmaps_raw)
    traj, _ = _select_traj_orientation(traj, kspace, spf, expected_samples)
    total_spokes, samples = traj.shape

    kspace = _align_kspace_to_traj(kspace, total_spokes, samples)
    n_coils = kspace.shape[0]
    print(f"[shape] kspace (coils, spokes, samples): {kspace.shape}")
    print(f"[shape] traj (spokes, samples): {traj.shape}")

    if total_spokes % spf != 0:
        raise ValueError(
            f"Total spokes ({total_spokes}) not divisible by spokes/frame ({spf})."
        )
    num_frames = total_spokes // spf
    print(f"[info] num_frames={num_frames}")

    traj = _scale_traj_if_needed(traj, samples)
    coord = _build_coord(traj, num_frames, spf, samples)
    kspace = _build_kspace(kspace, n_coils, num_frames, spf, samples)

    csmaps = _standardize_csmaps(csmaps_raw, n_coils)
    csmaps = csmaps[:, None, :, :]

    print(f"[shape] kspace for recon: {kspace.shape}")
    print(f"[shape] csmaps for recon: {csmaps.shape}")
    print(f"[shape] coord for recon: {coord.shape}")

    device = _resolve_device(args.device)

    recon = app.HighDimensionalRecon(
        kspace,
        csmaps,
        combine_echo=False,
        lamda=args.lamda,
        coord=coord,
        regu="TV",
        regu_axes=[0],
        max_iter=args.max_iter,
        solver="ADMM",
        rho=args.rho,
        device=device,
        show_pbar=False,
        verbose=False,
    ).run()

    recon_np = np.squeeze(recon.get())
    if recon_np.ndim == 3 and recon_np.shape[0] == num_frames and recon_np.shape[-1] != num_frames:
        recon_np = np.transpose(recon_np, (1, 2, 0))

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"grasp_{sample_id}_{spf}spf_{num_frames}frames.npy")
    np.save(output_path, recon_np)
    print(f"[done] saved GRASP recon to {output_path}")


def main():
    args = parse_args()

    if args.loop_val_dro:
        if args.kspace_path is not None or args.csmaps_path is not None:
            raise ValueError("Do not pass --kspace-path/--csmaps-path when using --loop-val-dro.")
        sample_ids = _load_split_ids(args.split_file, args.split_key)
        for raw_id in sample_ids:
            sample_id = _resolve_sample_id(raw_id)
            for spf in args.spokes_per_frame_list:
                num_frames = int(args.total_spokes / spf)
                kspace_path = os.path.join(
                    args.root_dir, f"{sample_id}_kspace_{spf}spf_{num_frames}frames.mat"
                )
                csmaps_path = os.path.join(
                    args.csmaps_dir, f"csmaps_{sample_id}.npy"
                )
                _run_recon(sample_id, spf, num_frames, kspace_path, csmaps_path, args)
        return

    if args.kspace_path is None:
        if args.sample_id is None or args.spokes_per_frame is None:
            raise ValueError("Provide --sample-id and --spokes-per-frame, or pass --kspace-path.")
        sample_id = _resolve_sample_id(args.sample_id)
        spf = args.spokes_per_frame
        num_frames = int(args.total_spokes / spf)
        kspace_path = os.path.join(args.root_dir, f"{sample_id}_kspace_{spf}spf_{num_frames}frames.mat")
    else:
        kspace_path = args.kspace_path
        sample_id = _resolve_sample_id(args.sample_id) or _parse_sample_id_from_name(kspace_path)
        spf = args.spokes_per_frame or _parse_spf_from_name(kspace_path)

    if args.csmaps_path is None:
        csmaps_path = os.path.join(args.csmaps_dir, f"csmaps_{sample_id}.npy")
    else:
        csmaps_path = args.csmaps_path

    _run_recon(sample_id, spf, num_frames, kspace_path, csmaps_path, args)


if __name__ == "__main__":
    main()
