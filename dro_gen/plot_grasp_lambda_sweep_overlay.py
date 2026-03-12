#!/usr/bin/env python3
"""Plot enhancement-curve overlays for precomputed GRASP reconstructions.

Example:
  python plot_grasp_lambda_sweep_overlay.py --sample-id sample_001 --spf 36 --lamdas 0.001,0.002
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt


SPF_TO_FRAMES = {
    36: 8,
    24: 12,
    16: 18,
    8: 36,
    4: 72,
    2: 144,
}


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


def _resolve_baseline_frames(
    num_frames: int,
    time_points: Optional[np.ndarray] = None,
    baseline_mode: str = "fraction",
    baseline_seconds: float = 20.0,
    baseline_fraction: float = 0.1,
    baseline_min_frames: int = 4,
    baseline_max_frames: Optional[int] = 10,
) -> int:
    if num_frames <= 0:
        return 0

    mode = (baseline_mode or "fraction").lower()
    if mode == "seconds":
        dt = None
        if time_points is not None and len(time_points) > 1:
            dt = float(time_points[1] - time_points[0])
        if not dt or dt <= 0:
            dt = float(baseline_seconds) if baseline_seconds > 0 else 1.0
        frames = int(np.ceil(baseline_seconds / dt)) if baseline_seconds > 0 else 0
        if baseline_min_frames is not None:
            frames = max(frames, baseline_min_frames)
        frames = min(frames, num_frames)
        return max(1, frames)

    if mode == "fraction":
        frames = int(round(baseline_fraction * num_frames))
        if baseline_min_frames is not None:
            frames = max(frames, baseline_min_frames)
        if baseline_max_frames is not None:
            frames = min(frames, baseline_max_frames)
        frames = min(frames, num_frames)
        return max(1, frames)

    raise ValueError(f"Unknown baseline_mode: {baseline_mode!r}")


def _baseline_subtract_curve(curve: np.ndarray, n_baseline: int) -> np.ndarray:
    if curve.size == 0 or n_baseline <= 0:
        return curve
    baseline = np.nanmean(curve[:n_baseline])
    return curve - baseline


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


def _read_h5_array(dset: h5py.Dataset) -> np.ndarray:
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
        return _read_h5_array(h5[key]), key

    if "simImg" in h5:
        return _read_h5_array(h5["simImg"]), "simImg"

    candidates = []

    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
            candidates.append(name)

    h5.visititems(_visit)
    if not candidates:
        raise ValueError(f"No 3D dataset found in {h5.filename}. Use --dro-key to specify.")
    return _read_h5_array(h5[candidates[0]]), candidates[0]


def _collect_mask_dict_from_h5(h5: h5py.File) -> Dict[str, np.ndarray]:
    mask_dict: Dict[str, np.ndarray] = {}
    if "mask" not in h5:
        return mask_dict
    mask_group = h5["mask"]
    if not isinstance(mask_group, h5py.Group):
        return mask_dict
    for key in mask_group.keys():
        try:
            mask_arr = _read_h5_array(mask_group[key])
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
        if "mask" in npz:
            mask_obj = npz["mask"]
            if isinstance(mask_obj, np.ndarray) and mask_obj.dtype == object:
                mask_obj = mask_obj.item()
            if isinstance(mask_obj, dict):
                for key, val in mask_obj.items():
                    mask_dict[key] = np.asarray(val)
        for key in npz.files:
            if key not in mask_dict:
                try:
                    mask_dict[key] = np.asarray(npz[key])
                except Exception:
                    continue
    return mask_dict


def _load_mask_dict(mask_path: Optional[str], dro_mat_path: str) -> Dict[str, np.ndarray]:
    if mask_path:
        if mask_path.endswith(".mat"):
            return _load_mask_from_dro_mat(mask_path)
        if mask_path.endswith(".npz"):
            return _load_mask_from_npz(mask_path)
        if mask_path.endswith(".npy"):
            return {"mask": np.load(mask_path)}
        raise ValueError(f"Unsupported mask path: {mask_path}")

    if os.path.exists(dro_mat_path):
        return _load_mask_from_dro_mat(dro_mat_path)
    return {}


def _normalize_mask(mask: np.ndarray, h: int, w: int) -> Optional[np.ndarray]:
    mask_np = np.asarray(mask)
    if mask_np.ndim == 2:
        if mask_np.shape == (h, w):
            return mask_np.astype(bool)
        if mask_np.shape == (w, h):
            return mask_np.T.astype(bool)
        return None
    if mask_np.ndim == 3:
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
    raise ValueError(
        f"Expected stack with frames on axis 0 or -1 (frames={frames}), got {stack.shape}."
    )


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


def _compute_mean_curve(stack_hwt: np.ndarray, mask_hw: np.ndarray) -> np.ndarray:
    curve = []
    for t in range(stack_hwt.shape[2]):
        vals = stack_hwt[:, :, t][mask_hw]
        curve.append(float(np.mean(vals)) if vals.size else float("nan"))
    return np.asarray(curve, dtype=np.float64)


def _plot_overlay_curves(
    curves: List[np.ndarray],
    lamdas: List[float],
    time_points: np.ndarray,
    out_path: str,
    title: str,
    dro_curve: Optional[np.ndarray] = None,
    ylabel: str = "Mean Signal",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(curves))) if curves else []
    for curve, lamda, color in zip(curves, lamdas, colors):
        ax.plot(time_points, curve, linewidth=2, label=f"lambda={lamda:g}", color=color)
    if dro_curve is not None:
        ax.plot(time_points, dro_curve, "k--", linewidth=2.5, label="DRO")
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot GRASP lambda-sweep enhancement curves from precomputed recon files."
    )
    parser.add_argument(
        "--dro-root",
        default="/net/scratch2/rachelgordon/dro_var_frames",
        help="Root directory containing DRO .mat files and GRASP recon outputs.",
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
        "--recon-dir",
        default=None,
        help="Directory containing GRASP recon .npy files (default: <dro_root>/espirit_grasp_recons).",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Optional suffix appended in filenames (before .npy).",
    )
    parser.add_argument(
        "--dro-mat",
        default=None,
        help="Optional DRO .mat file path (default: <dro_root>/<sample_id>_dro_<frames>frames.mat).",
    )
    parser.add_argument(
        "--dro-key",
        default=None,
        help="Dataset key inside the DRO .mat to use (default: simImg or first 3D dataset).",
    )
    parser.add_argument(
        "--skip-dro",
        action="store_true",
        help="Skip loading and plotting the DRO enhancement curve.",
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
        "--time-points",
        default=None,
        help="Comma-separated time points for frames (default: 0..T-1).",
    )
    parser.add_argument(
        "--baseline-subtract",
        action="store_true",
        help="Baseline-subtract mean curves before plotting (same defaults as inference).",
    )
    parser.add_argument(
        "--baseline_mode",
        default="fraction",
        choices=("seconds", "fraction"),
        help="Baseline window selection mode (seconds or fraction).",
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
        help="Baseline fraction of frames when baseline_mode=fraction.",
    )
    parser.add_argument(
        "--baseline_min_frames",
        type=int,
        default=4,
        help="Minimum baseline frames to use.",
    )
    parser.add_argument(
        "--baseline_max_frames",
        type=int,
        default=10,
        help="Maximum baseline frames when baseline_mode=fraction.",
    )
    parser.add_argument(
        "--out-path",
        default=None,
        help="PNG output path (default: <recon_dir>/grasp_curve_overlay_<sample>_<spf>spf_<frames>frames_with_dro.png).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = _resolve_frames(args.spf, args.frames)
    lamdas = _parse_float_list(args.lamdas)
    if not lamdas:
        raise ValueError("No lambda values provided.")
    suffix = _normalize_suffix(args.suffix)

    recon_dir = args.recon_dir or os.path.join(args.dro_root, "espirit_grasp_recons")
    if not os.path.isdir(recon_dir):
        raise FileNotFoundError(f"Recon directory not found: {recon_dir}")

    dro_mat_path = args.dro_mat or os.path.join(
        args.dro_root, f"{args.sample_id}_dro_{frames}frames.mat"
    )

    mask_dict = _load_mask_dict(args.mask_path, dro_mat_path)

    time_points = (
        np.asarray(_parse_float_list(args.time_points), dtype=np.float64)
        if args.time_points
        else np.arange(frames, dtype=np.float64)
    )
    if time_points.size != frames:
        raise ValueError(
            f"time_points length ({time_points.size}) must match frames ({frames})."
        )

    n_baseline = None
    if args.baseline_subtract:
        n_baseline = _resolve_baseline_frames(
            num_frames=frames,
            time_points=time_points,
            baseline_mode=args.baseline_mode,
            baseline_seconds=args.baseline_seconds,
            baseline_fraction=args.baseline_fraction,
            baseline_min_frames=args.baseline_min_frames,
            baseline_max_frames=args.baseline_max_frames,
        )

    curves: List[np.ndarray] = []
    mask_hw: Optional[np.ndarray] = None
    mask_label: Optional[str] = None

    for lamda in lamdas:
        recon_path = os.path.join(
            recon_dir,
            f"grasp_{args.sample_id}_{args.spf}spf_{frames}frames_lam{lamda:g}{suffix}.npy",
        )
        if not os.path.exists(recon_path):
            raise FileNotFoundError(
                f"Missing recon file for lambda={lamda:g}: {recon_path}"
            )
        recon_np = np.load(recon_path)
        recon_mag = _to_magnitude_stack(recon_np, frames)

        if mask_hw is None:
            mask_hw, mask_label = _select_mask(
                mask_dict, args.mask_key, recon_mag.shape[0], recon_mag.shape[1]
            )
            if mask_hw is None and args.mask_key == "auto":
                mask_hw = _infer_foreground_mask(recon_mag)
                if mask_hw is not None:
                    mask_label = "foreground"
            if mask_hw is None:
                available = ", ".join(sorted(mask_dict.keys())) if mask_dict else "none"
                raise ValueError(
                    "Failed to load ROI mask. "
                    f"Requested '{args.mask_key}', available keys: {available}. "
                    "Provide --mask-path or set --mask-key=auto to infer a foreground mask."
                )

        curve = _compute_mean_curve(recon_mag, mask_hw)
        if args.baseline_subtract and n_baseline is not None:
            curve = _baseline_subtract_curve(curve, n_baseline)
        curves.append(curve)

    dro_curve = None
    if not args.skip_dro:
        if not os.path.exists(dro_mat_path):
            raise FileNotFoundError(
                f"DRO .mat file not found: {dro_mat_path}. "
                "Provide --dro-mat or use --skip-dro."
            )
        with h5py.File(dro_mat_path, "r") as h5:
            dro_stack, _ = _find_h5_dataset(h5, args.dro_key)
        dro_mag = _to_magnitude_stack(dro_stack, frames)
        if dro_mag.shape[:2] != (mask_hw.shape[0], mask_hw.shape[1]):
            raise ValueError(
                f"DRO image shape {dro_mag.shape[:2]} does not match mask shape {mask_hw.shape}."
            )
        dro_curve = _compute_mean_curve(dro_mag, mask_hw)
        if args.baseline_subtract and n_baseline is not None:
            dro_curve = _baseline_subtract_curve(dro_curve, n_baseline)

    out_path = args.out_path or os.path.join(
        recon_dir,
        f"grasp_curve_overlay_{args.sample_id}_{args.spf}spf_{frames}frames_with_dro{suffix}.png",
    )
    title = (
        f"Enhancement Curves Overlay (SPF={args.spf}, frames={frames})"
        + (f" [{mask_label}]" if mask_label else "")
    )
    ylabel = "Baseline-Subtracted Signal" if args.baseline_subtract else "Mean Signal"
    _plot_overlay_curves(
        curves,
        lamdas,
        time_points,
        out_path,
        title,
        dro_curve=dro_curve,
        ylabel=ylabel,
    )
    print(f"Saved overlay curves to {out_path}")


if __name__ == "__main__":
    main()
