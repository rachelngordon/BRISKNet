#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_dro_id(map_csv: Path, fastmri_id: int) -> int:
    with map_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["fastMRIbreast"]) == fastmri_id:
                return int(row["DRO"])
    raise ValueError(f"No DRO mapping for fastMRI breast id {fastmri_id} in {map_csv}")


def find_dro_sample_dir(dro_root: Path, dro_id: int) -> Path:
    candidates = sorted(dro_root.glob(f"sample_{dro_id:03d}_sub{dro_id}"))
    if not candidates:
        raise FileNotFoundError(f"No sample directory matching sub{dro_id} in {dro_root}")
    return candidates[0]


def load_csmaps(path: Path) -> np.ndarray:
    csmaps = np.load(path)
    if csmaps.ndim == 4:
        csmaps = np.squeeze(csmaps)
    if csmaps.ndim != 3:
        raise ValueError(f"Expected 3D csmaps at {path}, got shape {csmaps.shape}")

    # Normalize to (coils, height, width).
    if csmaps.shape[0] == 16:
        return csmaps
    if csmaps.shape[-1] == 16:
        return np.moveaxis(csmaps, -1, 0)
    if csmaps.shape[1] == 16:
        return np.moveaxis(csmaps, 1, 0)
    raise ValueError(f"Could not infer coil dimension for {path} with shape {csmaps.shape}")


def parse_slice_idx(path: Path) -> int:
    match = re.search(r"cs_map_slice_(\d+)\.npy", path.name)
    if not match:
        raise ValueError(f"Could not parse slice index from {path.name}")
    return int(match.group(1))


def row_limits(arr: np.ndarray) -> tuple[float, float]:
    mag = np.abs(arr)
    vmin, vmax = np.percentile(mag, (2, 98))
    if vmin == vmax:
        vmax = vmin + 1e-6
    return vmin, vmax


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot DRO csmaps and raw fastMRI csmaps per slice for visual matching."
    )
    parser.add_argument("--fastmri-id", type=int, default=141, help="fastMRI breast id (integer).")
    parser.add_argument(
        "--dro-map-csv",
        type=Path,
        default=Path("data/DROSubID_vs_fastMRIbreastID.csv"),
        help="Path to DRO mapping CSV.",
    )
    parser.add_argument(
        "--dro-root",
        type=Path,
        default=Path("/net/scratch2/rachelgordon/dro_dataset_frontpad/dro_18frames"),
        help="Root directory for DRO samples.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/net/scratch2/rachelgordon/zf_data_192_slices"),
        help="Root directory for raw fastMRI data and cs_maps.",
    )
    parser.add_argument(
        "--raw-suffix",
        type=str,
        default="_2",
        help="Suffix for raw fastMRI patient IDs (e.g., _2).",
    )
    parser.add_argument(
        "--num-slices",
        type=int,
        default=8,
        help="Number of raw z-slices to plot.",
    )
    parser.add_argument(
        "--slice-start",
        type=int,
        default=0,
        help="Starting index into the sorted slice list.",
    )
    parser.add_argument(
        "--rotate-raw-k2",
        action="store_true",
        help="Rotate raw csmaps by k=2 to match dataloader orientation.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("dro_vs_raw_csmaps_141.png"),
        help="Output PNG path.",
    )
    args = parser.parse_args()

    dro_id = load_dro_id(args.dro_map_csv, args.fastmri_id)
    dro_sample_dir = find_dro_sample_dir(args.dro_root, dro_id)
    dro_csmaps = load_csmaps(dro_sample_dir / "csmaps.npy")

    patient_base = f"fastMRI_breast_{args.fastmri_id:03d}"
    patient_id = f"{patient_base}{args.raw_suffix}"
    raw_csmap_dir = args.raw_root / "cs_maps" / f"{patient_id}_cs_maps"
    raw_csmap_paths = sorted(raw_csmap_dir.glob("cs_map_slice_*.npy"))
    if not raw_csmap_paths:
        raise FileNotFoundError(f"No raw csmaps found in {raw_csmap_dir}")

    start = args.slice_start
    end = start + args.num_slices
    selected_paths = raw_csmap_paths[start:end]
    if len(selected_paths) < args.num_slices:
        raise ValueError(
            f"Requested {args.num_slices} slices starting at {start}, "
            f"but only found {len(selected_paths)}."
        )

    n_rows = 1 + args.num_slices
    n_cols = 16
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.6 * n_cols, 1.6 * n_rows))
    axes = np.atleast_2d(axes)

    dro_vmin, dro_vmax = row_limits(dro_csmaps)
    for coil in range(n_cols):
        ax = axes[0, coil]
        ax.axis("off")
        ax.imshow(np.abs(dro_csmaps[coil]), cmap="viridis", vmin=dro_vmin, vmax=dro_vmax)
        ax.set_title(f"C{coil}", fontsize=8)
    axes[0, 0].set_ylabel("DRO", rotation=0, labelpad=36, va="center")

    for row_idx, raw_path in enumerate(selected_paths, start=1):
        raw_csmaps = load_csmaps(raw_path)
        if args.rotate_raw_k2:
            raw_csmaps = np.rot90(raw_csmaps, k=2, axes=(-2, -1))

        vmin, vmax = row_limits(raw_csmaps)
        for coil in range(n_cols):
            ax = axes[row_idx, coil]
            ax.axis("off")
            ax.imshow(np.abs(raw_csmaps[coil]), cmap="viridis", vmin=vmin, vmax=vmax)

        slice_idx = parse_slice_idx(raw_path)
        axes[row_idx, 0].set_ylabel(f"slice {slice_idx}", rotation=0, labelpad=36, va="center")

    fig.tight_layout(rect=[0.06, 0.02, 1, 0.98])
    fig.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
