#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


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


def gaussian_blur(csmaps: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return csmaps

    radius = max(1, int(round(3 * sigma)))
    grid = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (grid / sigma) ** 2)
    kernel /= kernel.sum()

    blurred = np.empty_like(csmaps)
    for coil in range(csmaps.shape[0]):
        temp = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="same"),
            0,
            csmaps[coil],
        )
        blurred[coil] = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="same"),
            1,
            temp,
        )
    return blurred


def downsample_csmaps(csmaps: np.ndarray, factor: int, method: str, sigma: float) -> np.ndarray:
    if factor <= 1:
        return csmaps

    coils, height, width = csmaps.shape
    new_height = height // factor
    new_width = width // factor
    if new_height < 1 or new_width < 1:
        raise ValueError(
            f"Downsample factor {factor} too large for shape {csmaps.shape}"
        )

    crop_height = new_height * factor
    crop_width = new_width * factor
    if crop_height != height or crop_width != width:
        csmaps = csmaps[:, :crop_height, :crop_width]

    if method == "average":
        # Average pooling over spatial blocks to avoid aliasing.
        csmaps = csmaps.reshape(coils, new_height, factor, new_width, factor)
        return csmaps.mean(axis=(2, 4))

    if method == "gaussian":
        blurred = gaussian_blur(csmaps, sigma)
        return blurred[:, ::factor, ::factor]

    raise ValueError(f"Unknown downsample method: {method}")


def resize_2d(image: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
    height, width = image.shape
    if (height, width) == (new_height, new_width):
        return image

    y = np.linspace(0, height - 1, new_height)
    x = np.linspace(0, width - 1, new_width)
    x0 = np.floor(x).astype(int)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y0 = np.floor(y).astype(int)
    y1 = np.clip(y0 + 1, 0, height - 1)

    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)

    top = (1 - wx)[None, :] * image[y0][:, x0] + wx[None, :] * image[y0][:, x1]
    bottom = (1 - wx)[None, :] * image[y1][:, x0] + wx[None, :] * image[y1][:, x1]
    return (1 - wy)[:, None] * top + wy[:, None] * bottom


def center_crop_resize(csmaps: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0 or scale > 1:
        raise ValueError("FOV scale must be in (0, 1].")
    if scale == 1:
        return csmaps

    coils, height, width = csmaps.shape
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    if new_height > height or new_width > width:
        raise ValueError("FOV scale results in larger crop than input.")

    start_h = (height - new_height) // 2
    start_w = (width - new_width) // 2
    cropped = csmaps[:, start_h : start_h + new_height, start_w : start_w + new_width]

    resized = np.empty((coils, height, width), dtype=cropped.dtype)
    for coil in range(coils):
        resized[coil] = resize_2d(cropped[coil], height, width)
    return resized


def save_csmaps_for_inference(csmaps: np.ndarray, out_path: Path) -> None:
    if csmaps.ndim != 3:
        raise ValueError(f"Expected 3D csmaps to save, got {csmaps.shape}")
    csmaps_to_save = csmaps[:, None, :, :]
    np.save(out_path, csmaps_to_save)


def resize_csmaps(csmaps: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    coils, _, _ = csmaps.shape
    resized = np.empty((coils, target_height, target_width), dtype=csmaps.dtype)
    for coil in range(coils):
        resized[coil] = resize_2d(csmaps[coil], target_height, target_width)
    return resized


def row_limits(arr: np.ndarray) -> tuple[float, float]:
    mag = np.abs(arr)
    vmin, vmax = np.percentile(mag, (2, 98))
    if vmin == vmax:
        vmax = vmin + 1e-6
    return vmin, vmax


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot DRO csmaps vs downsampled fastMRI csmaps."
    )
    parser.add_argument(
        "--dro-path",
        type=Path,
        default=Path(
            "/net/scratch2/rachelgordon/dro_dataset_frontpad/dro_18frames/sample_005_sub5/csmaps.npy"
        ),
        help="Path to DRO csmaps.npy.",
    )
    parser.add_argument(
        "--fastmri-path",
        type=Path,
        default=Path(
            "/net/scratch2/rachelgordon/zf_data_192_slices/cs_maps/fastMRI_breast_142_2_cs_maps/cs_map_slice_125.npy"
        ),
        help="Path to fastMRI cs_map_slice_*.npy.",
    )
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=2,
        help="Integer downsample factor for fastMRI csmaps.",
    )
    parser.add_argument(
        "--downsample-method",
        type=str,
        choices=("average", "gaussian"),
        default="average",
        help="Downsampling method: average pooling or Gaussian blur + decimate.",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma for blur (in pixels) when using gaussian method.",
    )
    parser.add_argument(
        "--fov-scale",
        type=float,
        default=1.0,
        help=(
            "Apply a center crop to fastMRI csmaps by this scale, then resize "
            "back to original size to simulate smaller FOV without padding."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("dro_vs_fastmri_downsampled_csmaps.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--save-csmaps-out",
        type=Path,
        default=Path("fastmri_csmaps_for_inference.npy"),
        help="Output .npy path for fastMRI csmaps in run_inference_single_fastmri format.",
    )
    args = parser.parse_args()

    dro_csmaps = load_csmaps(args.dro_path)
    fastmri_csmaps = load_csmaps(args.fastmri_path)

    fastmri_csmaps = np.rot90(fastmri_csmaps, k=2, axes=(-2, -1))

    fastmri_csmaps = center_crop_resize(fastmri_csmaps, args.fov_scale)
    fastmri_csmaps_ds = downsample_csmaps(
        fastmri_csmaps,
        args.downsample_factor,
        args.downsample_method,
        args.gaussian_sigma,
    )

    n_cols = dro_csmaps.shape[0]
    if fastmri_csmaps_ds.shape[0] != n_cols:
        raise ValueError(
            "Coil count mismatch: "
            f"DRO has {n_cols}, fastMRI has {fastmri_csmaps_ds.shape[0]}"
        )

    fig, axes = plt.subplots(2, n_cols, figsize=(1.6 * n_cols, 3.2))
    axes = np.atleast_2d(axes)

    dro_vmin, dro_vmax = row_limits(dro_csmaps)
    for coil in range(n_cols):
        ax = axes[0, coil]
        ax.axis("off")
        ax.imshow(np.abs(dro_csmaps[coil]), cmap="viridis", vmin=dro_vmin, vmax=dro_vmax)
        ax.set_title(f"C{coil}", fontsize=8)
    axes[0, 0].set_ylabel("DRO", rotation=0, labelpad=36, va="center")

    fast_vmin, fast_vmax = row_limits(fastmri_csmaps_ds)
    for coil in range(n_cols):
        ax = axes[1, coil]
        ax.axis("off")
        ax.imshow(
            np.abs(fastmri_csmaps_ds[coil]),
            cmap="viridis",
            vmin=fast_vmin,
            vmax=fast_vmax,
        )
    axes[1, 0].set_ylabel(
        f"fastMRI fov*{args.fov_scale:.2f} /{args.downsample_factor}",
        rotation=0,
        labelpad=36,
        va="center",
    )

    fig.tight_layout(rect=[0.06, 0.02, 1, 0.98])
    fig.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")
    fastmri_csmaps_save = resize_csmaps(fastmri_csmaps_ds, 320, 320)
    save_csmaps_for_inference(fastmri_csmaps_save, args.save_csmaps_out)
    print(f"Saved fastMRI csmaps for inference to {args.save_csmaps_out}")


if __name__ == "__main__":
    main()
