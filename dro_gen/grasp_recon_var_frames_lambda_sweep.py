#!/usr/bin/env python3
"""
Run GRASP reconstructions for a single sample/spf across a list of lambda values,
plot per-lambda enhancement panels, and save an overlay of enhancement curves
(optionally including the DRO curve).
"""
from __future__ import annotations

import argparse
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import sigpy as sp
from sigpy.mri import app
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


SPF_TO_FRAMES = {
    36: 8,
    24: 12,
    16: 18,
    8: 36,
    4: 72,
    2: 144,
}


def trajGR(Nkx, Nspokes):
    ga = np.pi * ((1 - np.sqrt(5)) / 2)
    kx = np.zeros(shape=(Nkx, Nspokes))
    ky = np.zeros(shape=(Nkx, Nspokes))
    ky[:, 0] = np.linspace(-np.pi, np.pi, Nkx)
    for i in range(1, Nspokes):
        kx[:, i] = np.cos(ga) * kx[:, i - 1] - np.sin(ga) * ky[:, i - 1]
        ky[:, i] = np.sin(ga) * kx[:, i - 1] + np.cos(ga) * ky[:, i - 1]
    ky = np.transpose(ky)
    kx = np.transpose(kx)
    ktraj = np.stack((ky.flatten(), kx.flatten()), axis=0)
    return ktraj


def get_traj(N_spokes=13, N_time=1, base_res=320, gind=1):
    n_tot_spokes = N_spokes * N_time
    n_samples = base_res * 2
    base_lin = np.arange(n_samples).reshape(1, -1) - base_res

    tau = 0.5 * (1 + 5**0.5)
    base_rad = np.pi / (gind + tau - 1)
    base_rot = np.arange(n_tot_spokes).reshape(-1, 1) * base_rad

    traj = np.zeros((n_tot_spokes, n_samples, 2))
    traj[..., 0] = np.cos(base_rot) @ base_lin
    traj[..., 1] = np.sin(base_rot) @ base_lin
    traj = traj / 2
    traj = traj.reshape(N_time, N_spokes, n_samples, 2)
    return np.squeeze(traj)


def _build_recon_traj(spf: int, frames: int, nsample: int, traj_method: str) -> np.ndarray:
    if traj_method == "get_traj":
        return get_traj(N_spokes=spf, N_time=frames)
    if traj_method == "trajGR":
        ktraj = trajGR(nsample, spf * frames)  # (2, total_spokes * samples)
        ktraj = ktraj.reshape(2, spf * frames, nsample)
        traj = np.transpose(ktraj, (1, 2, 0))  # (total_spokes, samples, 2)
        traj = traj.reshape(frames, spf, nsample, 2)
        return traj
    raise ValueError(f"Unknown traj_method: {traj_method}")


def _run_grasp(
    kspace_np: np.ndarray,
    csmaps_np: np.ndarray,
    spf: int,
    frames: int,
    traj_method: str,
    lamda: float,
    max_iter: int,
    rho: float,
    device: sp.Device,
) -> np.ndarray:
    if kspace_np.ndim != 3:
        raise ValueError(f"Expected kspace shape (C, M, T), got {kspace_np.shape}")
    n_coils, m, t = kspace_np.shape
    if t != frames:
        raise ValueError(f"kspace frames ({t}) != expected frames ({frames})")
    if m % spf != 0:
        raise ValueError(f"kspace samples ({m}) not divisible by spf ({spf})")
    nsample = m // spf

    if csmaps_np.ndim != 3:
        raise ValueError(f"Expected csmaps shape (C, H, W), got {csmaps_np.shape}")
    if csmaps_np.shape[0] != n_coils and csmaps_np.shape[-1] == n_coils:
        csmaps_np = np.transpose(csmaps_np, (2, 0, 1))
    if csmaps_np.shape[0] != n_coils:
        raise ValueError(
            f"CSMAP coil count mismatch: expected {n_coils}, got {csmaps_np.shape[0]}"
        )

    kspace_tcsp = kspace_np.reshape(n_coils, spf, nsample, t)
    kspace_tcsp = np.transpose(kspace_tcsp, (3, 0, 1, 2))  # (T, C, spf, sam)
    kspace_tcsp = kspace_tcsp[:, None, :, None, :, :]

    csmaps = csmaps_np[:, None, :, :]
    traj = _build_recon_traj(spf, frames, nsample, traj_method)

    recon = app.HighDimensionalRecon(
        kspace_tcsp,
        csmaps,
        combine_echo=False,
        lamda=lamda,
        coord=traj,
        regu="TV",
        regu_axes=[0],
        max_iter=max_iter,
        solver="ADMM",
        rho=rho,
        device=device,
        show_pbar=False,
        verbose=False,
    ).run()

    recon_np = np.squeeze(recon.get())
    if recon_np.ndim == 3 and recon_np.shape[0] == frames and recon_np.shape[-1] != frames:
        recon_np = np.transpose(recon_np, (1, 2, 0))
    return recon_np


def _parse_sigpy_device(device: str) -> sp.Device:
    if device == "cuda":
        return sp.Device(0)
    if device == "cpu":
        return sp.Device(-1)
    raise ValueError(f"Unknown sigpy device '{device}'. Use 'cpu' or 'cuda'.")


def _normalize_suffix(suffix: str) -> str:
    if not suffix:
        return ""
    if not suffix.startswith("_"):
        suffix = "_" + suffix
    return suffix


def _parse_float_list(value: str) -> List[float]:
    if not value:
        return []
    parts = [v.strip() for v in value.split(",") if v.strip()]
    return [float(v) for v in parts]


def _resolve_frames(spf: int, frames: int | None) -> int:
    if frames is not None:
        return int(frames)
    if spf in SPF_TO_FRAMES:
        return SPF_TO_FRAMES[spf]
    raise ValueError(
        f"frames not provided and SPF {spf} not in mapping. "
        "Pass --frames explicitly."
    )


def _first_frame_magnitude(recon: np.ndarray, frames: int) -> np.ndarray:
    if recon.ndim != 3:
        raise ValueError(f"Expected recon shape (H,W,T) or (T,H,W), got {recon.shape}")
    if recon.shape[-1] == frames:
        frame = recon[..., 0]
    elif recon.shape[0] == frames:
        frame = recon[0, ...]
    else:
        frame = recon[..., 0]
    return np.abs(frame)


def _read_h5_mask_array(dset: h5py.Dataset) -> np.ndarray:
    arr = np.asarray(dset)
    if arr.dtype.kind == "V":
        raise TypeError("Expected numeric dataset for mask, got compound.")
    return arr


def _read_h5_complex(dset: h5py.Dataset) -> np.ndarray:
    type_id = dset.id.get_type()
    type_class = type_id.get_class()
    if type_class != h5py.h5t.COMPOUND:
        raise TypeError("Expected compound dataset for complex values.")
    memtype = h5py.h5t.create(h5py.h5t.COMPOUND, type_id.get_size())
    names = []
    formats = []
    offsets = []
    for idx in range(type_id.get_nmembers()):
        name = type_id.get_member_name(idx)
        name_str = name.decode() if isinstance(name, (bytes, bytearray)) else str(name)
        member = type_id.get_member_type(idx)
        if member.get_class() != h5py.h5t.FLOAT:
            raise TypeError("Unsupported compound member type in complex dataset.")
        np_fmt = np.float32 if member.get_size() == 4 else np.float64
        memtype.insert(
            name,
            type_id.get_member_offset(idx),
            h5py.h5t.NATIVE_FLOAT if np_fmt == np.float32 else h5py.h5t.NATIVE_DOUBLE,
        )
        names.append(name_str)
        formats.append(np_fmt)
        offsets.append(type_id.get_member_offset(idx))
    arr = np.empty(
        dset.shape,
        dtype=np.dtype(
            {"names": names, "formats": formats, "offsets": offsets, "itemsize": type_id.get_size()}
        ),
    )
    dset.read_direct(arr)
    name_map = {n.lower(): n for n in names}
    real_key = name_map.get("real", names[0])
    imag_key = name_map.get("imag", names[-1])
    return arr[real_key] + 1j * arr[imag_key]


def _read_h5_array_any(dset: h5py.Dataset) -> np.ndarray:
    if dset.dtype.kind == "V":
        return _read_h5_complex(dset)
    arr = np.asarray(dset)
    if arr.ndim == 4 and arr.shape[-1] == 2 and not np.iscomplexobj(arr):
        arr = arr[..., 0] + 1j * arr[..., 1]
    return arr


def _find_h5_dataset(h5: h5py.File, key: Optional[str]) -> tuple[np.ndarray, str]:
    if key:
        if key not in h5:
            raise KeyError(f"Key '{key}' not found in {h5.filename}")
        return _read_h5_array_any(h5[key]), key

    if "simImg" in h5:
        return _read_h5_array_any(h5["simImg"]), "simImg"

    candidates = []

    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
            candidates.append(name)

    h5.visititems(_visit)
    if not candidates:
        raise ValueError(f"No 3D dataset found in {h5.filename}. Use --dro-key to specify.")
    return _read_h5_array_any(h5[candidates[0]]), candidates[0]


def _collect_mask_dict_from_h5(h5: h5py.File) -> Dict[str, np.ndarray]:
    mask_dict: Dict[str, np.ndarray] = {}
    if "mask" not in h5:
        return mask_dict
    mask_group = h5["mask"]
    if not isinstance(mask_group, h5py.Group):
        return mask_dict
    for key in mask_group.keys():
        try:
            mask_arr = _read_h5_mask_array(mask_group[key])
        except Exception:
            continue
        mask_dict[key] = mask_arr
    return mask_dict


def _load_mask_from_dro_mat(dro_mat_path: str) -> Dict[str, np.ndarray]:
    with h5py.File(dro_mat_path, "r") as h5:
        return _collect_mask_dict_from_h5(h5)


def _load_mask_from_npz(npz_path: str) -> Dict[str, np.ndarray]:
    mask_dict: Dict[str, np.ndarray] = {}
    with np.load(npz_path, allow_pickle=True) as npz:
        # If masks are stored as a dict-like object under "mask"
        if "mask" in npz:
            mask_obj = npz["mask"]
            if isinstance(mask_obj, np.ndarray) and mask_obj.dtype == object:
                mask_obj = mask_obj.item()
            if isinstance(mask_obj, dict):
                for key, val in mask_obj.items():
                    mask_dict[key] = np.asarray(val)
        # Also allow direct tissue-name keys
        for key in npz.files:
            if key not in mask_dict:
                try:
                    mask_dict[key] = np.asarray(npz[key])
                except Exception:
                    continue
    return mask_dict


def _normalize_mask(mask: np.ndarray, h: int, w: int) -> Optional[np.ndarray]:
    mask_np = np.asarray(mask)
    if mask_np.ndim == 2:
        if mask_np.shape == (h, w):
            return mask_np.astype(bool)
        if mask_np.shape == (w, h):
            return mask_np.T.astype(bool)
        return None
    if mask_np.ndim == 3:
        # If mask has time/slice dimension, collapse it.
        if mask_np.shape[0] == h and mask_np.shape[1] == w:
            return np.any(mask_np, axis=2).astype(bool)
        if mask_np.shape[-2:] == (h, w):
            return np.any(mask_np, axis=0).astype(bool)
        if mask_np.shape[1:] == (h, w):
            return np.any(mask_np, axis=0).astype(bool)
    return None


def _select_mask(
    mask_dict: Dict[str, np.ndarray],
    mask_key: str,
    h: int,
    w: int,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    preferred = ["malignant", "benign", "glandular", "muscle", "full"]
    if mask_key and mask_key != "auto":
        if mask_key in mask_dict:
            mask = _normalize_mask(mask_dict[mask_key], h, w)
            if mask is not None and np.any(mask):
                return mask, mask_key
        return None, None

    for key in preferred:
        if key in mask_dict:
            mask = _normalize_mask(mask_dict[key], h, w)
            if mask is not None and np.any(mask):
                return mask, key
    return None, None


def _infer_foreground_mask(stack: np.ndarray) -> Optional[np.ndarray]:
    if stack.ndim != 3:
        return None
    proj = np.max(stack, axis=2)
    max_val = float(np.max(proj))
    if not np.isfinite(max_val) or max_val <= 0:
        return None
    thr = 0.05 * max_val
    mask = proj > thr
    if not np.any(mask):
        return None
    return mask.astype(bool)


def _ensure_hwt(stack: np.ndarray, frames: int) -> np.ndarray:
    if stack.ndim != 3:
        raise ValueError(f"Expected 3D stack, got {stack.shape}")
    if stack.shape[-1] == frames:
        return stack
    if stack.shape[0] == frames:
        return np.transpose(stack, (1, 2, 0))
    return stack


def _to_complex(arr: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(arr):
        return arr
    if arr.ndim >= 3 and arr.shape[-1] == 2:
        return arr[..., 0] + 1j * arr[..., 1]
    if arr.ndim >= 3 and arr.shape[0] == 2:
        return arr[0] + 1j * arr[1]
    return arr


def _to_magnitude_stack(stack: np.ndarray, frames: int) -> np.ndarray:
    complex_stack = _to_complex(stack)
    if complex_stack.ndim != 3:
        raise ValueError(f"Expected 3D stack, got {complex_stack.shape}")
    stack_hwt = _ensure_hwt(complex_stack, frames)
    return np.abs(stack_hwt)


def _default_frames_to_show(num_frames: int) -> List[int]:
    interval = max(1, round(num_frames / 4))
    frames = [0, interval, min(2 * interval, num_frames - 1), num_frames - 1]
    return [min(max(0, idx), num_frames - 1) for idx in frames]


def _compute_mean_curve(stack_hwt: np.ndarray, mask_hw: np.ndarray) -> np.ndarray:
    curve = []
    for t in range(stack_hwt.shape[2]):
        vals = stack_hwt[:, :, t][mask_hw]
        curve.append(float(np.mean(vals)) if vals.size else float("nan"))
    return np.asarray(curve, dtype=np.float64)


def _plot_enhancement_panel(
    stack_hwt: np.ndarray,
    mask_hw: Optional[np.ndarray],
    time_points: np.ndarray,
    frames_to_show: List[int],
    out_path: str,
    title: str,
) -> None:
    num_frames = stack_hwt.shape[2]
    if len(frames_to_show) != 4:
        raise ValueError("Expected exactly 4 frames to show.")

    fig = plt.figure(figsize=(20, 8.5))
    fig.suptitle(title, fontsize=22, y=0.98)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.16, wspace=0.16)

    ax_curve = fig.add_subplot(gs[:, 0:2])
    ax_imgs = [
        fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3]),
        fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[1, 3])
    ]

    if mask_hw is None:
        mask_hw = np.ones(stack_hwt.shape[:2], dtype=bool)

    curve = _compute_mean_curve(stack_hwt, mask_hw)
    ax_curve.plot(time_points, curve, "o-", linewidth=2, markersize=5, label="Mean ROI Signal")
    highlight_times = [time_points[i] for i in frames_to_show]
    highlight_vals = [curve[i] for i in frames_to_show]
    ax_curve.plot(highlight_times, highlight_vals, "r*", markersize=16, zorder=10)
    ax_curve.set_title("Enhancement Curve", fontsize=16, pad=8)
    ax_curve.set_xlabel("Frame", fontsize=14)
    ax_curve.set_ylabel("Mean Signal", fontsize=14)
    ax_curve.grid(True, linestyle="--", alpha=0.5)
    ax_curve.legend(fontsize=12)

    vmin, vmax = np.percentile(stack_hwt, [1, 99.5])
    if vmax <= vmin:
        vmax = vmin + 1e-6

    for ax, frame_idx in zip(ax_imgs, frames_to_show):
        image = stack_hwt[:, :, frame_idx]
        ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
        if mask_hw is not None:
            ax.contour(mask_hw.astype(float), levels=[0.5], colors="red", linewidths=1.2)
        ax.set_title(f"Frame {frame_idx}", fontsize=14)
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_overlay_curves(
    curves: List[np.ndarray],
    lamdas: List[float],
    time_points: np.ndarray,
    out_path: str,
    title: str,
    dro_curve: Optional[np.ndarray] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(curves))) if curves else []
    for curve, lamda, color in zip(curves, lamdas, colors):
        ax.plot(time_points, curve, linewidth=2, label=f"lambda={lamda:g}", color=color)
    if dro_curve is not None:
        ax.plot(time_points, dro_curve, "k--", linewidth=2.5, label="DRO")
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("Mean Signal", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="GRASP lambda sweep for a single sample/spf."
    )
    parser.add_argument(
        "--dro-root",
        default="/net/scratch2/rachelgordon/dro_var_frames",
        help="Root directory containing kspace and csmaps_espirit.",
    )
    parser.add_argument(
        "--sample-id",
        required=True,
        help="Sample ID (e.g., sample_032_sub32).",
    )
    parser.add_argument(
        "--spf",
        type=int,
        required=True,
        help="Spokes per frame.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Override number of frames (defaults from SPF mapping).",
    )
    parser.add_argument(
        "--lamdas",
        required=True,
        help="Comma-separated lambda values (e.g., 0.0005,0.001,0.002).",
    )
    parser.add_argument(
        "--csmaps-dir",
        default=None,
        help="Directory containing csmaps_<sample_id>.npy (default: <dro_root>/csmaps_espirit).",
    )
    parser.add_argument(
        "--sigpy-device",
        choices=("cpu", "cuda"),
        default="cuda",
        help="Device for GRASP recon.",
    )
    parser.add_argument(
        "--traj-method",
        default="get_traj",
        choices=("trajGR", "get_traj"),
        help="Trajectory method to use for GRASP recon.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=10,
        help="GRASP max iterations.",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.1,
        help="GRASP ADMM rho.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Optional suffix to append to output filenames (before .npy).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to save recon outputs (default: <dro_root>/espirit_grasp_recons).",
    )
    parser.add_argument(
        "--panel-out",
        default=None,
        help="PNG output path for the panel (default: <out_dir>/grasp_panel_<sample>_<spf>spf.png).",
    )
    parser.add_argument(
        "--mask-path",
        default=None,
        help="Optional path to mask file (.mat/.npz/.npy). If omitted, tries to load masks from the DRO .mat.",
    )
    parser.add_argument(
        "--mask-key",
        default="malignant",
        help="Mask key to use (e.g., malignant, benign, glandular, muscle, full, or auto).",
    )
    parser.add_argument(
        "--dro-mat",
        default=None,
        help="Optional DRO .mat file path for the enhancement curve (default: <dro_root>/<sample_id>_dro_<frames>frames.mat).",
    )
    parser.add_argument(
        "--dro-key",
        default=None,
        help="Dataset key inside the DRO .mat to use (default: simImg or first 3D dataset).",
    )
    parser.add_argument(
        "--skip-dro",
        action="store_true",
        help="Skip loading and plotting the DRO enhancement curve in the overlay plot.",
    )
    parser.add_argument(
        "--time-points",
        default=None,
        help="Comma-separated time points for frames (default: 0..T-1).",
    )
    parser.add_argument(
        "--frames-to-show",
        default=None,
        help="Comma-separated list of 4 frame indices to show in per-lambda plots.",
    )
    parser.add_argument(
        "--overlay-out",
        default=None,
        help="PNG output path for overlay curve plot (default: <out_dir>/grasp_curve_overlay_<sample>_<spf>spf_<frames>frames.png).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sigpy_device = _parse_sigpy_device(args.sigpy_device)
    suffix = _normalize_suffix(args.suffix)
    frames = _resolve_frames(args.spf, args.frames)
    lamdas = _parse_float_list(args.lamdas)
    if not lamdas:
        raise ValueError("No lambda values provided.")
    dro_mat_path = args.dro_mat or os.path.join(
        args.dro_root,
        f"{args.sample_id}_dro_{frames}frames.mat",
    )

    csmaps_dir = args.csmaps_dir or os.path.join(args.dro_root, "csmaps_espirit")
    out_dir = args.out_dir or os.path.join(args.dro_root, "espirit_grasp_recons")
    os.makedirs(out_dir, exist_ok=True)

    kspace_path = os.path.join(
        args.dro_root,
        f"{args.sample_id}_kspace_{args.spf}spf_{frames}frames{suffix}.npy",
    )
    csmaps_path = os.path.join(csmaps_dir, f"csmaps_{args.sample_id}{suffix}.npy")
    if not os.path.exists(kspace_path):
        raise FileNotFoundError(f"Missing kspace file: {kspace_path}")
    if not os.path.exists(csmaps_path):
        raise FileNotFoundError(f"Missing csmaps file: {csmaps_path}")

    kspace_np = np.load(kspace_path)
    csmaps_np = np.load(csmaps_path)

    # Optional mask loading
    mask_dict: Dict[str, np.ndarray] = {}
    mask_label: Optional[str] = None
    if args.mask_path:
        if args.mask_path.endswith(".mat"):
            mask_dict = _load_mask_from_dro_mat(args.mask_path)
        elif args.mask_path.endswith(".npz"):
            mask_dict = _load_mask_from_npz(args.mask_path)
        elif args.mask_path.endswith(".npy"):
            mask_arr = np.load(args.mask_path)
            mask_dict = {"mask": mask_arr}
        else:
            raise ValueError(f"Unsupported mask path: {args.mask_path}")
    else:
        if os.path.exists(dro_mat_path):
            mask_dict = _load_mask_from_dro_mat(dro_mat_path)

    time_points = (
        np.asarray(_parse_float_list(args.time_points), dtype=np.float64)
        if args.time_points
        else np.arange(frames, dtype=np.float64)
    )
    if time_points.size != frames:
        raise ValueError(
            f"time_points length ({time_points.size}) must match frames ({frames})."
        )

    frames_to_show = (
        [int(v) for v in _parse_float_list(args.frames_to_show)]
        if args.frames_to_show
        else _default_frames_to_show(frames)
    )
    if len(frames_to_show) != 4:
        raise ValueError("frames_to_show must contain exactly 4 indices.")

    recon_images = []
    curves = []
    mask_hw: Optional[np.ndarray] = None
    for lamda in lamdas:
        recon_np = _run_grasp(
            kspace_np=kspace_np,
            csmaps_np=csmaps_np,
            spf=args.spf,
            frames=frames,
            traj_method=args.traj_method,
            lamda=lamda,
            max_iter=args.max_iter,
            rho=args.rho,
            device=sigpy_device,
        )
        recon_mag = np.abs(recon_np)
        recon_hwt = _ensure_hwt(recon_mag, frames)
        if mask_hw is None:
            mask_hw, mask_label = _select_mask(
                mask_dict, args.mask_key, recon_hwt.shape[0], recon_hwt.shape[1]
            )
            if mask_hw is None and args.mask_key == "auto":
                mask_hw = _infer_foreground_mask(recon_hwt)
                if mask_hw is not None:
                    mask_label = "foreground"
            if mask_hw is None:
                available = ", ".join(sorted(mask_dict.keys())) if mask_dict else "none"
                raise ValueError(
                    "Failed to load ROI mask. "
                    f"Requested '{args.mask_key}', available keys: {available}. "
                    "Provide --mask-path or set --mask-key=auto to infer a foreground mask."
                )

        recon_path = os.path.join(
            out_dir,
            f"grasp_{args.sample_id}_{args.spf}spf_{frames}frames_lam{lamda:g}{suffix}.npy",
        )
        np.save(recon_path, recon_np)
        recon_images.append(_first_frame_magnitude(recon_mag, frames))

        curve = _compute_mean_curve(recon_hwt, mask_hw)
        curves.append(curve)

        curve_out = os.path.join(
            out_dir,
            f"grasp_curve_{args.sample_id}_{args.spf}spf_{frames}frames_lam{lamda:g}{suffix}.png",
        )
        plot_title = (
            f"Enhancement Curve (lambda={lamda:g}, SPF={args.spf}, frames={frames})"
            + (f" [{mask_label}]" if mask_label else "")
        )
        _plot_enhancement_panel(
            recon_hwt,
            mask_hw,
            time_points,
            frames_to_show,
            curve_out,
            plot_title,
        )

    # Plot panel
    n = len(lamdas)
    ncols = min(n, 4)
    nrows = int(math.ceil(n / ncols))
    vmax = np.percentile(np.concatenate([img.ravel() for img in recon_images]), 99.5)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(nrows, ncols)
    for idx, (lamda, img) in enumerate(zip(lamdas, recon_images)):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]
        ax.imshow(img, cmap="gray", vmin=0, vmax=vmax)
        ax.set_title(f"lambda={lamda:g}")
        ax.axis("off")
    for idx in range(len(lamdas), nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r, c].axis("off")

    panel_out = args.panel_out or os.path.join(
        out_dir, f"grasp_panel_{args.sample_id}_{args.spf}spf.png"
    )
    fig.tight_layout()
    fig.savefig(panel_out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved panel to {panel_out}")

    dro_curve = None
    if curves and not args.skip_dro:
        if not os.path.exists(dro_mat_path):
            print(f"Warning: DRO .mat file not found: {dro_mat_path}. Skipping DRO curve.")
        else:
            with h5py.File(dro_mat_path, "r") as h5:
                dro_stack, _ = _find_h5_dataset(h5, args.dro_key)
            dro_mag = _to_magnitude_stack(dro_stack, frames)
            if dro_mag.shape[2] != frames:
                raise ValueError(
                    f"DRO stack frames ({dro_mag.shape[2]}) do not match expected frames ({frames})."
                )
            if dro_mag.shape[:2] != (mask_hw.shape[0], mask_hw.shape[1]):
                raise ValueError(
                    f"DRO image shape {dro_mag.shape[:2]} does not match mask shape {mask_hw.shape}."
                )
            dro_curve = _compute_mean_curve(dro_mag, mask_hw)
            if dro_curve.size != time_points.size:
                raise ValueError(
                    "DRO curve length does not match time_points. "
                    f"{dro_curve.size} vs {time_points.size}."
                )

    if curves:
        overlay_out = args.overlay_out or os.path.join(
            out_dir, f"grasp_curve_overlay_{args.sample_id}_{args.spf}spf_{frames}frames{suffix}.png"
        )
        overlay_title = (
            f"Enhancement Curves Overlay (SPF={args.spf}, frames={frames})"
            + (f" [{mask_label}]" if mask_label else "")
        )
        _plot_overlay_curves(curves, lamdas, time_points, overlay_out, overlay_title, dro_curve=dro_curve)
        print(f"Saved overlay curves to {overlay_out}")


if __name__ == "__main__":
    main()
