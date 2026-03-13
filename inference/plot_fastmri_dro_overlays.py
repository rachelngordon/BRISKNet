#!/usr/bin/env python3
"""Plot fastMRI DCE overlays with tumor masks across time. Run: python3 -m inference.plot_fastmri_dro_overlays --help"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from einops import rearrange


def parse_fastmri_id(raw_id: str) -> str:
    raw_id = str(raw_id).strip()
    if raw_id.startswith("fastMRI_breast_"):
        return raw_id if raw_id.endswith("_2") else f"{raw_id}_2"
    return f"fastMRI_breast_{int(raw_id):03d}_2"


def load_slice_index(slice_csv: Path, fastmri_breast_id: str) -> int:
    with open(slice_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("fastMRI_breast_id") == fastmri_breast_id:
                return int(row["largest_slice_idx"])
    raise ValueError(f"No slice index for {fastmri_breast_id} in {slice_csv}")


def load_fastmri_mask(mask_root: Path, fastmri_breast_id: str, slice_idx: int) -> np.ndarray:
    for suffix in (".nii.gz", ".nii"):
        mask_path = mask_root / f"{fastmri_breast_id}{suffix}"
        if mask_path.exists():
            nii = nib.load(str(mask_path))
            data = nii.get_fdata()
            if data.ndim != 3:
                raise ValueError(f"Expected 3D mask in {mask_path}, got shape {data.shape}")
            return (data[..., slice_idx] > 0).astype(np.uint8)
    raise FileNotFoundError(f"No mask found for {fastmri_breast_id} in {mask_root}")


def load_fastmri_mask_volume(mask_root: Path, fastmri_breast_id: str) -> np.ndarray:
    for suffix in (".nii.gz", ".nii"):
        mask_path = mask_root / f"{fastmri_breast_id}{suffix}"
        if mask_path.exists():
            nii = nib.load(str(mask_path))
            data = nii.get_fdata()
            if data.ndim != 3:
                raise ValueError(f"Expected 3D mask in {mask_path}, got shape {data.shape}")
            return (data > 0).astype(np.uint8)
    raise FileNotFoundError(f"No mask found for {fastmri_breast_id} in {mask_root}")


def to_magnitude(img: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(img):
        return np.abs(img)
    if img.ndim == 4 and img.shape[-1] == 2:
        return np.sqrt(img[..., 0] ** 2 + img[..., 1] ** 2)
    return img.astype(np.float32)


def ensure_time_last(img: np.ndarray) -> np.ndarray:
    if img.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape {img.shape}")
    time_axis = int(np.argmin(img.shape))
    return np.moveaxis(img, time_axis, -1)


def orient_slice_2d(img2d: np.ndarray) -> np.ndarray:
    img2d = np.rot90(img2d, k=2, axes=(-2, -1))
    img2d = np.flip(img2d, axis=-1)
    return img2d


def save_time_series_overlay(img: np.ndarray, mask2d: np.ndarray, out_path: Path, title: str) -> None:
    img = to_magnitude(img)
    img = rearrange(img.squeeze(), 't h w -> h w t')
    img = ensure_time_last(img)

    t_frames = img.shape[-1]
    ncols = 6
    nrows = int(math.ceil(t_frames / ncols))

    vmin, vmax = np.percentile(img, (2, 98))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 2.6))
    axes = np.atleast_1d(axes).ravel()

    for t in range(t_frames):
        ax = axes[t]
        ax.imshow(img[..., t], cmap="gray", vmin=vmin, vmax=vmax)
        if mask2d is not None:
            overlay = np.ma.masked_where(mask2d == 0, mask2d)
            ax.imshow(overlay, cmap="autumn", alpha=0.6, vmin=0, vmax=1)
            ax.contour(mask2d, levels=[0.5], colors="yellow", linewidths=0.8)
        ax.set_title(f"t={t}")
        ax.axis("off")

    for ax in axes[t_frames:]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def align_grasp_volume_to_mask(img: np.ndarray, mask_vol: np.ndarray) -> np.ndarray:
    if img.ndim != 4:
        raise ValueError(f"Expected 4D GRASP volume, got shape {img.shape}")

    m_h, m_w, m_z = mask_vol.shape
    shape = img.shape

    z_axes = [i for i, s in enumerate(shape) if s == m_z]
    if not z_axes:
        raise ValueError(f"No axis matches mask z size {m_z} in GRASP volume {shape}")
    z_axis = z_axes[0]

    if m_h == m_w:
        hw_axes = [i for i, s in enumerate(shape) if s == m_h and i != z_axis]
        if len(hw_axes) < 2:
            raise ValueError(f"Could not find H/W axes matching {m_h} in {shape}")
        h_axis, w_axis = hw_axes[:2]
    else:
        h_axis = next((i for i, s in enumerate(shape) if s == m_h and i != z_axis), None)
        w_axis = next((i for i, s in enumerate(shape) if s == m_w and i not in (z_axis, h_axis)), None)
        if h_axis is None or w_axis is None:
            raise ValueError(f"Could not match H/W axes ({m_h}, {m_w}) in {shape}")

    time_axis = next(i for i in range(4) if i not in (z_axis, h_axis, w_axis))
    return np.moveaxis(img, (z_axis, h_axis, w_axis, time_axis), (0, 1, 2, 3))


def save_z_slices_overlay(
    vol_zhwt: np.ndarray,
    mask_vol: np.ndarray,
    timepoint: int,
    out_path: Path,
    title: str,
) -> None:
    vol_zhwt = to_magnitude(vol_zhwt)
    if timepoint < 0 or timepoint >= vol_zhwt.shape[-1]:
        raise ValueError(f"timepoint {timepoint} out of range for {vol_zhwt.shape[-1]} frames")

    tumor_slices = [z for z in range(mask_vol.shape[-1]) if mask_vol[..., z].any()]
    if not tumor_slices:
        raise ValueError("No tumor-positive slices found in mask volume")

    ncols = 6
    nrows = int(math.ceil(len(tumor_slices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 2.6))
    axes = np.atleast_1d(axes).ravel()

    slice_imgs = []
    for z in tumor_slices:
        img2d = vol_zhwt[z, :, :, timepoint]
        mask2d = mask_vol[:, :, z]
        # img2d = orient_slice_2d(img2d)
        # mask2d = orient_slice_2d(mask2d)
        slice_imgs.append((img2d, mask2d))

    all_imgs = np.stack([s[0] for s in slice_imgs], axis=0)
    vmin, vmax = np.percentile(all_imgs, (2, 98))

    for i, z in enumerate(tumor_slices):
        ax = axes[i]
        img2d, mask2d = slice_imgs[i]
        ax.imshow(img2d, cmap="gray", vmin=vmin, vmax=vmax)
        overlay = np.ma.masked_where(mask2d == 0, mask2d)
        ax.imshow(overlay, cmap="autumn", alpha=0.6, vmin=0, vmax=1)
        ax.contour(mask2d, levels=[0.5], colors="yellow", linewidths=0.8)
        ax.set_title(f"z={z}")
        ax.axis("off")

    for ax in axes[len(tumor_slices):]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_single_timepoint_comparison(
    grasp_img: np.ndarray,
    grasp_mask2d: np.ndarray,
    dro_img: np.ndarray,
    dro_mask2d: np.ndarray,
    timepoint: int,
    out_path: Path,
    title: str,
) -> None:
    grasp_img = to_magnitude(grasp_img)
    grasp_img = rearrange(grasp_img.squeeze(), 't h w -> h w t')
    grasp_img = ensure_time_last(grasp_img)

    dro_img = to_magnitude(dro_img)
    dro_img = rearrange(dro_img.squeeze(), 't h w -> h w t')
    dro_img = ensure_time_last(dro_img)

    if timepoint < 0 or timepoint >= grasp_img.shape[-1]:
        raise ValueError(f"GRASP timepoint {timepoint} out of range for {grasp_img.shape[-1]} frames")
    if timepoint < 0 or timepoint >= dro_img.shape[-1]:
        raise ValueError(f"DRO timepoint {timepoint} out of range for {dro_img.shape[-1]} frames")

    vmin_g, vmax_g = np.percentile(grasp_img[..., timepoint], (2, 98))
    vmin_d, vmax_d = np.percentile(dro_img[..., timepoint], (2, 98))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax_g, ax_d = axes

    ax_g.imshow(grasp_img[..., timepoint], cmap="gray", vmin=vmin_g, vmax=vmax_g)
    if grasp_mask2d is not None:
        overlay = np.ma.masked_where(grasp_mask2d == 0, grasp_mask2d)
        ax_g.imshow(overlay, cmap="autumn", alpha=0.6, vmin=0, vmax=1)
        ax_g.contour(grasp_mask2d, levels=[0.5], colors="yellow", linewidths=0.8)
    ax_g.set_title("GRASP")
    ax_g.axis("off")

    ax_d.imshow(dro_img[..., timepoint], cmap="gray", vmin=vmin_d, vmax=vmax_d)
    if dro_mask2d is not None:
        overlay = np.ma.masked_where(dro_mask2d == 0, dro_mask2d)
        ax_d.imshow(overlay, cmap="autumn", alpha=0.6, vmin=0, vmax=1)
        ax_d.contour(dro_mask2d, levels=[0.5], colors="yellow", linewidths=0.8)
    ax_d.set_title("DRO")
    ax_d.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def find_dro_sample_dir(dro_root: Path, dro_id: int) -> Path:
    candidates = sorted(dro_root.glob(f"sample_{dro_id:03d}_sub{dro_id}"))
    if not candidates:
        raise FileNotFoundError(f"No sample directory matching sub{dro_id} in {dro_root}")
    return candidates[0]


def load_dro_data(dro_npz: Path) -> tuple[np.ndarray, np.ndarray]:
    dro = np.load(dro_npz, allow_pickle=True)
    print(f"DRO loaded from {dro_npz}")

    if "simImg" in dro:
        img = dro["simImg"]
    elif "ground_truth_images" in dro:
        img = dro["ground_truth_images"]
    else:
        raise KeyError(f"No simImg or ground_truth_images in {dro_npz}")

    mask = None
    if "mask" in dro:
        mask_obj = dro["mask"]
        if isinstance(mask_obj, np.ndarray) and mask_obj.dtype == object:
            mask_obj = mask_obj.item()
        if isinstance(mask_obj, dict):
            mask = mask_obj.get("malignant")
    if mask is None and "malignant" in dro:
        mask = dro["malignant"]

    if mask is None:
        raise KeyError(f"No malignant mask in {dro_npz}")

    return img, mask


def load_dro_id(dro_map_csv: Path, fastmri_breast_id: str) -> int:
    fastmri_numeric = int(fastmri_breast_id.split("_")[2])
    with open(dro_map_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["fastMRIbreast"]) == fastmri_numeric:
                return int(row["DRO"])
    raise ValueError(f"No DRO mapping for {fastmri_breast_id} in {dro_map_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GRASP and DRO time series with tumor overlays.")
    parser.add_argument("fastmri_id", help="fastMRI id (e.g., 141 or fastMRI_breast_141_2)")
    parser.add_argument("--slice_csv", default="data/largest_tumor_slices.csv")
    parser.add_argument("--grasp_root", default="/net/scratch2/rachelgordon/zf_data_192_slices")
    parser.add_argument("--tumor_seg_root", default="/net/scratch2/rachelgordon/zf_data_192_slices/tumor_segmentations_lcr")
    parser.add_argument("--spokes_per_frame", type=int, default=16)
    parser.add_argument("--num_frames", type=int, default=18)
    parser.add_argument("--dro_map_csv", default="data/DROSubID_vs_fastMRIbreastID.csv")
    parser.add_argument("--dro_root", default="/net/scratch2/rachelgordon/dro_dataset_frontpad/dro_18frames")
    parser.add_argument("--z_spokes_per_frame", type=int, default=36)
    parser.add_argument("--z_timepoint", type=int, default=0)
    parser.add_argument("--compare_timepoint", type=int, default=0)
    parser.add_argument("--out_dir", default="output/overlay_plots")
    args = parser.parse_args()

    fastmri_breast_id = parse_fastmri_id(args.fastmri_id)
    slice_idx = load_slice_index(Path(args.slice_csv), fastmri_breast_id)

    grasp_dir = Path(args.grasp_root) / fastmri_breast_id
    grasp_name = f"grasp_recon_{args.spokes_per_frame}spf_{args.num_frames}frames_slice{slice_idx}.npy"
    grasp_path = grasp_dir / grasp_name
    if not grasp_path.exists():
        raise FileNotFoundError(f"GRASP recon not found: {grasp_path}")

    fastmri_mask = load_fastmri_mask(Path(args.tumor_seg_root), fastmri_breast_id, slice_idx)
    grasp_img = np.load(grasp_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grasp_out = out_dir / f"{fastmri_breast_id}_grasp_slice{slice_idx}.png"

    print(grasp_img.shape)

    grasp_img = np.rot90(grasp_img, k=2, axes=[-2, -1])
    # grasp_img = np.flip(grasp_img, axis=3)
    # fastmri_mask = np.rot90(fastmri_mask, k=2, axes=[-2, -1])
    # fastmri_mask = np.flip(fastmri_mask, axis=1)

    save_time_series_overlay(
        grasp_img,
        fastmri_mask,
        grasp_out,
        title=f"GRASP {fastmri_breast_id} slice {slice_idx}",
    )

    grasp_vol_name = f"grasp_recon_{args.z_spokes_per_frame}spf.npy"
    grasp_vol_path = grasp_dir / grasp_vol_name
    if not grasp_vol_path.exists():
        raise FileNotFoundError(f"GRASP volume not found: {grasp_vol_path}")

    mask_vol = load_fastmri_mask_volume(Path(args.tumor_seg_root), fastmri_breast_id)
    grasp_vol = np.load(grasp_vol_path)
    print(grasp_vol.shape)
    grasp_vol = align_grasp_volume_to_mask(grasp_vol, mask_vol)

    z_out = out_dir / f"{fastmri_breast_id}_grasp_z_slices_t{args.z_timepoint}.png"
    save_z_slices_overlay(
        grasp_vol,
        mask_vol,
        args.z_timepoint,
        z_out,
        title=f"GRASP z-slices {fastmri_breast_id} t={args.z_timepoint}",
    )

    dro_id = load_dro_id(Path(args.dro_map_csv), fastmri_breast_id)

    dro_root = Path(args.dro_root)
    sample_dir = find_dro_sample_dir(dro_root, dro_id)
    dro_npz = sample_dir / "dro_ground_truth.npz"
    if not dro_npz.exists():
        raise FileNotFoundError(f"DRO ground truth not found: {dro_npz}")

    dro_img, dro_mask = load_dro_data(dro_npz)

    dro_img = np.rot90(dro_img, k=1, axes=[0, 1])
    dro_img = np.flip(dro_img, axis=0)
    # dro_mask = np.rot90(dro_img, k=1, axes=[0, 1])

    dro_out = out_dir / f"{fastmri_breast_id}_dro_sub{dro_id}.png"
    save_time_series_overlay(
        dro_img,
        dro_mask,
        dro_out,
        title=f"DRO sub{dro_id} for {fastmri_breast_id}",
    )

    compare_out = out_dir / f"{fastmri_breast_id}_grasp_vs_dro_t{args.compare_timepoint}.png"
    save_single_timepoint_comparison(
        grasp_img,
        fastmri_mask,
        dro_img,
        dro_mask,
        args.compare_timepoint,
        compare_out,
        title=f"GRASP vs DRO {fastmri_breast_id} t={args.compare_timepoint}",
    )

    print(f"Saved GRASP overlay to {grasp_out}")
    print(f"Saved GRASP z-slices overlay to {z_out}")
    print(f"Saved DRO overlay to {dro_out}")
    print(f"Saved GRASP vs DRO overlay to {compare_out}")


if __name__ == "__main__":
    main()
