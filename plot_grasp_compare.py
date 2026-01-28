# save as plot_grasp_compare.py
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import nibabel as nib
from skimage.measure import find_contours

path_2spf = "/net/scratch2/rachelgordon/zf_data_192_slices/fastMRI_breast_141_2/grasp_recon_2spf_144frames_slice66.npy"
path_36spf = "/net/scratch2/rachelgordon/zf_data_192_slices/fastMRI_breast_141_2/grasp_recon_36spf_8frames_slice66.npy"

TUMOR_SEG_ROOT = os.environ.get(
    "TUMOR_SEG_ROOT",
    "/net/scratch2/rachelgordon/zf_data_192_slices/tumor_segmentations_lcr",
)

def robust_window_multi(images, p_low=1, p_high=99.5):
    flat = []
    for img in images:
        if img is None:
            continue
        arr = np.asarray(img)
        flat.append(arr.ravel())
    if not flat:
        return 0.0, 1.0
    stacked = np.concatenate(flat)
    finite_vals = stacked[np.isfinite(stacked)]
    if finite_vals.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite_vals, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def _parse_frames_from_path(path):
    match = re.search(r"_(\d+)frames", os.path.basename(path))
    return int(match.group(1)) if match else None


def _parse_patient_and_slice(path):
    patient_id = os.path.basename(os.path.dirname(path))
    match = re.search(r"slice(\d+)", os.path.basename(path))
    slice_idx = int(match.group(1)) if match else None
    return patient_id, slice_idx


def _load_tumor_mask(patient_id, slice_idx, seg_root=TUMOR_SEG_ROOT):
    if patient_id is None:
        raise ValueError("Patient id is required to load tumor segmentation.")
    seg_path = os.path.join(seg_root, f"{patient_id}.nii.gz")
    if not os.path.exists(seg_path):
        raise FileNotFoundError(f"Tumor segmentation not found at {seg_path}")
    seg_vol = nib.load(seg_path).get_fdata()
    if seg_vol.ndim == 3:
        num_slices = seg_vol.shape[-1]
        if slice_idx is None or slice_idx < 0 or slice_idx >= num_slices:
            slice_sums = seg_vol.sum(axis=tuple(range(seg_vol.ndim - 1)))
            slice_idx = int(np.argmax(slice_sums))
        tumor_mask = seg_vol[..., int(slice_idx)]
    else:
        tumor_mask = seg_vol
    return tumor_mask.astype(bool), slice_idx


def _ensure_thw(arr, num_frames=None):
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array after squeeze, got shape {arr.shape}")
    if num_frames is None:
        return arr
    if arr.shape[0] == num_frames:
        return arr
    if arr.shape[1] == num_frames:
        return np.moveaxis(arr, 1, 0)
    if arr.shape[2] == num_frames:
        return np.moveaxis(arr, 2, 0)
    return arr


def _apply_raw_grasp_orientation(arr_thw):
    """Match SimulatedDataset raw GRASP orientation (flip + rot90 k=1)."""
    arr_wth = np.moveaxis(arr_thw, 2, 0)  # W, T, H (permute 2, 0, 1)
    arr_wth = np.flip(arr_wth, axis=0)
    arr_wth = np.rot90(arr_wth, k=1, axes=(0, 2))
    # Match eval's rearrange('h t w -> h w t') on the oriented stack.
    arr_hwt = np.transpose(arr_wth, (0, 2, 1))
    return arr_hwt


def load_center_frame_mag(path):
    arr = np.load(path)
    arr = np.squeeze(arr)
    num_frames = _parse_frames_from_path(path)
    arr_thw = _ensure_thw(arr, num_frames=num_frames)
    arr_hwt = _apply_raw_grasp_orientation(arr_thw)
    mag_stack = np.abs(arr_hwt)
    center_idx = int(mag_stack.shape[2] / 2)
    frame = mag_stack[:, :, center_idx]
    return frame, center_idx


def load_mag_stack(path):
    arr = np.load(path)
    arr = np.squeeze(arr)
    num_frames = _parse_frames_from_path(path)
    arr_thw = _ensure_thw(arr, num_frames=num_frames)
    arr_hwt = _apply_raw_grasp_orientation(arr_thw)
    return np.abs(arr_hwt)


img_2spf_stack = load_mag_stack(path_2spf)
img_36spf_stack = load_mag_stack(path_36spf)

img_2spf, center_idx_2spf = load_center_frame_mag(path_2spf)
img_36spf, center_idx_36spf = load_center_frame_mag(path_36spf)

vmin, vmax = robust_window_multi([img_2spf, img_36spf], p_low=1, p_high=99.5)

patient_id_2spf, slice_idx_2spf = _parse_patient_and_slice(path_2spf)
patient_id_36spf, slice_idx_36spf = _parse_patient_and_slice(path_36spf)
if patient_id_2spf != patient_id_36spf:
    raise ValueError(f"Patient mismatch: {patient_id_2spf} vs {patient_id_36spf}")
if slice_idx_2spf != slice_idx_36spf:
    print(f"Warning: slice mismatch: {slice_idx_2spf} vs {slice_idx_36spf}")

tumor_mask, resolved_slice = _load_tumor_mask(patient_id_2spf, slice_idx_2spf)
if tumor_mask.shape != img_2spf_stack.shape[:2]:
    if tumor_mask.shape[::-1] == img_2spf_stack.shape[:2]:
        tumor_mask = tumor_mask.T
    else:
        raise ValueError(
            f"Tumor mask shape {tumor_mask.shape} does not match image shape {img_2spf_stack.shape[:2]}"
        )
if not tumor_mask.any():
    raise ValueError(f"Tumor mask is empty for {patient_id_2spf} slice {resolved_slice}.")

contours = find_contours(tumor_mask.astype(float), 0.5) if tumor_mask.any() else []

def _plot_roi_contours(ax):
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color="red")

curve_2spf = np.array(
    [img_2spf_stack[:, :, t][tumor_mask].mean() for t in range(img_2spf_stack.shape[2])]
)
curve_36spf = np.array(
    [img_36spf_stack[:, :, t][tumor_mask].mean() for t in range(img_36spf_stack.shape[2])]
)
time_2spf = np.linspace(0, 150, img_2spf_stack.shape[2])
time_36spf = np.linspace(0, 150, img_36spf_stack.shape[2])

fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.1], hspace=0.28, wspace=0.08)
ax_img_2 = fig.add_subplot(gs[0, 0])
ax_img_36 = fig.add_subplot(gs[0, 1])
ax_curve = fig.add_subplot(gs[1, :])

ax_img_2.imshow(img_2spf, cmap="gray", vmin=vmin, vmax=vmax)
_plot_roi_contours(ax_img_2)
ax_img_2.set_title(f"2spf 144frames center frame ({center_idx_2spf})")
ax_img_2.axis("off")

ax_img_36.imshow(img_36spf, cmap="gray", vmin=vmin, vmax=vmax)
_plot_roi_contours(ax_img_36)
ax_img_36.set_title(f"36spf 8frames center frame ({center_idx_36spf})")
ax_img_36.axis("off")

ax_curve.plot(time_2spf, curve_2spf, "o-", linewidth=2, label="2spf tumor mean")
ax_curve.plot(time_36spf, curve_36spf, "s--", linewidth=2, label="36spf tumor mean")
ax_curve.set_title("Tumor ROI enhancement curves")
ax_curve.set_xlabel("Time (s)")
ax_curve.set_ylabel("Mean signal intensity")
ax_curve.grid(True, linestyle="--", alpha=0.5)
ax_curve.legend()

plt.tight_layout()
out_path = "grasp_center_frame_compare.png"
plt.savefig(out_path, dpi=200)
print(f"Saved {out_path}")
