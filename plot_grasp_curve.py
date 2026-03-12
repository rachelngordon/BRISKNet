"""Plot GRASP enhancement curves for a DRO sample. Run: python3 plot_grasp_curve.py --help"""

import argparse
import glob
import os
from pathlib import Path

import h5py
import h5py.h5t as h5t
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=str,
        default="/net/scratch2/rachelgordon/dro_var_frames/espirit_grasp_recons_lam0.001",
    )
    p.add_argument(
        "--dro_root",
        type=str,
        default="/net/scratch2/rachelgordon/dro_var_frames",
        help="DRO root containing sample_*_dro_*frames.mat files.",
    )
    p.add_argument("--sample", type=int, default=5)
    p.add_argument("--sub", type=int, default=5)
    p.add_argument("--spf", type=int, default=16)
    p.add_argument("--lam", type=float, default=0.001)
    p.add_argument(
        "--mask_region",
        type=str,
        default="auto",
        help="Mask region for curve/bbox: auto, malignant, benign, or union.",
    )
    p.add_argument(
        "--roi",
        type=str,
        default="tumor",
        help="ROI for curve: 'tumor', 'center', or 'x,y' pixel coordinates",
    )
    p.add_argument(
        "--frame_idxs",
        type=str,
        default="auto",
        help="Comma-separated frame indices to display, or 'auto' for quarter-spaced frames.",
    )
    p.add_argument(
        "--total_scan_seconds",
        type=float,
        default=150.0,
        help="Total scan duration for time axis (seconds).",
    )
    p.add_argument("--save", type=str, default="grasp_curve.png")
    return p.parse_args()


def build_filename(sample, sub, spf):
    num_frames = 288 // spf
    return f"grasp_sample_{sample:03d}_sub{sub}_{spf}spf_{num_frames}frames.npy", num_frames


def to_t_hw(vol, num_frames=None):
    # vol expected shape: (T, H, W) or (H, W, T) or (H, T, W)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {vol.shape}")

    t_axis = None
    if num_frames is not None and num_frames in vol.shape:
        # If multiple axes match, pick the smallest index (rare but deterministic).
        t_axis = [i for i, s in enumerate(vol.shape) if s == num_frames][0]
    else:
        # Heuristic: time is the smallest axis if it's clearly smaller than the others.
        sizes = list(vol.shape)
        t_axis = int(np.argmin(sizes))
        # If all axes are similar, default to last axis (common in recon exports).
        if max(sizes) - min(sizes) < 4:
            t_axis = 2

    if t_axis != 0:
        vol = np.moveaxis(vol, t_axis, 0)
    return vol


def _h5py_float_dtype(type_id: h5t.TypeID) -> np.dtype:
    size = type_id.get_size()
    if size == 4:
        return np.float32
    if size == 8:
        return np.float64
    raise TypeError(f"Unsupported float size for HDF5 dtype: {size}")


def _read_h5_float(dset: h5py.Dataset) -> np.ndarray:
    type_id = dset.id.get_type()
    if type_id.get_class() != h5t.FLOAT:
        raise TypeError(f"Expected float dataset; got class={type_id.get_class()}")
    np_dtype = _h5py_float_dtype(type_id)
    arr = np.empty(dset.shape, dtype=np_dtype)
    dset.read_direct(arr)
    return arr


def _read_h5_numeric(dset: h5py.Dataset) -> np.ndarray:
    type_id = dset.id.get_type()
    type_class = type_id.get_class()
    if type_class == h5t.FLOAT:
        return _read_h5_float(dset)
    if type_class != h5t.INTEGER:
        raise TypeError(f"Expected numeric dataset; got class={type_class}")
    size = type_id.get_size()
    signed = type_id.get_sign() == h5t.SGN_2
    if size == 1:
        np_dtype = np.int8 if signed else np.uint8
    elif size == 2:
        np_dtype = np.int16 if signed else np.uint16
    elif size == 4:
        np_dtype = np.int32 if signed else np.uint32
    elif size == 8:
        np_dtype = np.int64 if signed else np.uint64
    else:
        raise TypeError(f"Unsupported integer size for HDF5 dtype: {size}")
    arr = np.empty(dset.shape, dtype=np_dtype)
    dset.read_direct(arr)
    return arr


def _resolve_dro_mat_path(dro_root: str, sample_id: str, num_frames: int) -> str:
    candidates = [
        os.path.join(dro_root, f"{sample_id}_dro_{num_frames}frames.mat"),
        os.path.join(dro_root, f"{sample_id}_dro.mat"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    alt_pattern = os.path.join(dro_root, f"{sample_id}_dro_*frames.mat")
    alt_files = glob.glob(alt_pattern)
    if alt_files:
        frames = sorted(
            {
                int(os.path.basename(fp).split("_dro_")[1].split("frames")[0])
                for fp in alt_files
                if "_dro_" in os.path.basename(fp)
            }
        )
        frames_str = ", ".join(str(f) for f in frames) if frames else "unknown"
        raise FileNotFoundError(
            f"Missing DRO file for {sample_id} with {num_frames} frames. "
            f"Available frames: {frames_str}."
        )
    raise FileNotFoundError(f"Missing DRO file: {candidates[0]}")


def _load_dro_masks(dro_root: str, sample_id: str, num_frames: int) -> dict:
    dro_path = _resolve_dro_mat_path(dro_root, sample_id, num_frames)
    masks = {}
    tissues = [
        "glandular",
        "benign",
        "malignant",
        "muscle",
        "skin",
        "liver",
        "heart",
        "vascular",
    ]
    with h5py.File(dro_path, "r") as f:
        if "mask" in f and isinstance(f["mask"], h5py.Group):
            mask_group = f["mask"]
            for tissue in tissues:
                if tissue in mask_group:
                    try:
                        arr = _read_h5_numeric(mask_group[tissue])
                    except Exception:
                        # Fall back to direct read if the HDF5 dtype is already supported.
                        arr = mask_group[tissue][()]
                    masks[tissue] = arr > 0
    return masks


def _pick_tumor_mask(masks: dict, region: str = "auto"):
    if not masks:
        return None, "none"
    region = (region or "auto").lower()
    if region == "malignant" and "malignant" in masks and masks["malignant"].any():
        return masks["malignant"], "malignant"
    if region == "benign" and "benign" in masks and masks["benign"].any():
        return masks["benign"], "benign"
    if region == "union":
        union = None
        for v in masks.values():
            if v is None:
                continue
            union = v.astype(bool) if union is None else (union | v.astype(bool))
        if union is not None and union.any():
            return union, "union"
        return None, "none"
    # auto
    if "malignant" in masks and masks["malignant"].any():
        return masks["malignant"], "malignant"
    if "benign" in masks and masks["benign"].any():
        return masks["benign"], "benign"
    union = None
    for v in masks.values():
        if v is None:
            continue
        union = v.astype(bool) if union is None else (union | v.astype(bool))
    if union is not None and union.any():
        return union, "union"
    return None, "none"


def _robust_window(img_stack: np.ndarray, p_low: float = 1, p_high: float = 99.5):
    lo, hi = np.percentile(img_stack, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)




def main():
    args = parse_args()

    fname, num_frames = build_filename(args.sample, args.sub, args.spf)
    path = Path(args.root) / fname
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    vol = np.load(path).squeeze()
    vol = to_t_hw(vol, num_frames=num_frames)
    if np.iscomplexobj(vol):
        vol = np.abs(vol)

    # Rotate to match run_inference_new_dro orientation (k=3 on H/W),
    # then flip horizontally to align with DRO display.
    vol = np.rot90(vol, k=3, axes=(1, 2))
    vol = np.flip(vol, axis=2)

    # Convert to (H, W, T) for plotting.
    img_stack = np.transpose(vol, (1, 2, 0))
    T, H, W = vol.shape

    sample_id = f"sample_{args.sample:03d}_sub{args.sub}"
    try:
        masks = _load_dro_masks(args.dro_root, sample_id, num_frames)
    except Exception as exc:
        print(f"Warning: failed to load DRO masks for {sample_id}: {exc}")
        masks = {}
    tumor_mask, mask_key = _pick_tumor_mask(masks, args.mask_region)
    if tumor_mask is not None and tumor_mask.T.shape == (H, W) and tumor_mask.shape != (H, W):
        tumor_mask = tumor_mask.T
    if tumor_mask is not None:
        tumor_mask = np.rot90(tumor_mask, k=3)
        tumor_mask = np.flip(tumor_mask, axis=1)

    if args.frame_idxs.strip().lower() == "auto":
        interval = max(1, int(round(T / 4)))
        frame_idxs = [0, min(interval, T - 1), min(2 * interval, T - 1), T - 1]
    else:
        frame_idxs = [int(i) for i in args.frame_idxs.split(",")]
        frame_idxs = [max(0, min(fi, T - 1)) for fi in frame_idxs]

    time_points = np.linspace(0, args.total_scan_seconds, T)

    roi_mode = args.roi.lower()
    if roi_mode == "tumor":
        if tumor_mask is not None and tumor_mask.any():
            mean_curve = np.array([img_stack[:, :, t][tumor_mask].mean() for t in range(T)])
            roi_label = f"{mask_key} mask"
        else:
            y, x = H // 2, W // 2
            mean_curve = img_stack[y, x, :]
            roi_label = f"center ({y}, {x}) (tumor mask unavailable)"
    elif roi_mode == "center":
        y, x = H // 2, W // 2
        mean_curve = img_stack[y, x, :]
        roi_label = f"center ({y}, {x})"
    else:
        x_str, y_str = args.roi.split(",")
        x, y = int(x_str), int(y_str)
        mean_curve = img_stack[y, x, :]
        roi_label = f"({y}, {x})"

    vmin, vmax = _robust_window(img_stack, 1, 99.5)
    contour_mask = None
    if tumor_mask is not None and tumor_mask.any():
        contour_mask = tumor_mask.astype(float)

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.2])

    # Curve
    ax_curve = fig.add_subplot(gs[0, :])
    ax_curve.plot(time_points, mean_curve, lw=2, color="tab:blue")
    ax_curve.plot(time_points, mean_curve, "o", markersize=4, color="tab:blue")
    ax_curve.plot(time_points[frame_idxs[:4]], mean_curve[frame_idxs[:4]], "r*", markersize=12)
    ax_curve.set_title(f"GRASP Enhancement Curve | {args.spf} spf ({num_frames} frames)")
    ax_curve.set_xlabel("Time (s)")
    ax_curve.set_ylabel("Signal")

    # Frames
    for i, fi in enumerate(frame_idxs[:4]):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(img_stack[:, :, fi], cmap="gray", vmin=vmin, vmax=vmax)
        if contour_mask is not None:
            ax.contour(contour_mask, levels=[0.5], colors="red", linewidths=1.5)
        ax.set_title(f"Frame {fi}")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(args.save, dpi=200)
    print(f"Saved: {args.save}")


if __name__ == "__main__":
    main()
