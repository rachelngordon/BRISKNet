"""Visualize largest tumor component overlays for a sample slice. Run: python3 -m inference.select_tumor_region"""

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import label

SEG_ROOT = Path("/net/scratch2/rachelgordon/zf_data_192_slices/tumor_segmentations")
DATA_ROOT = SEG_ROOT.parent

def load_largest_component(mask_path: Path):
    nii = nib.load(str(mask_path))
    data = nii.get_fdata()
    bin_mask = data > 0  # binarize

    labeled, n = label(bin_mask)
    if n == 0:
        return np.zeros_like(bin_mask, dtype=bool), nii.affine, nii.header

    counts = np.bincount(labeled.ravel())[1:]  # skip background
    largest_label = counts.argmax() + 1
    largest = labeled == largest_label
    return largest, nii.affine, nii.header

def load_all_largest():
    out = {}
    for p in sorted(SEG_ROOT.glob("*.nii.gz")):
        mask, affine, header = load_largest_component(p)
        patient_id = p.name.replace(".nii.gz", "")
        out[patient_id] = {"mask": mask, "affine": affine, "header": header}
    return out

if __name__ == "__main__":
    masks = load_all_largest()
    print(f"Loaded {len(masks)} masks; first few IDs: {list(masks)[:5]}")

    # --- Example plot: original mask vs largest component overlaid on the source slice ---
    example_path = sorted(SEG_ROOT.glob("*.nii.gz"))[0]
    patient_id = example_path.name.replace(".nii.gz", "")  # e.g., fastMRI_breast_010_2

    # Load masks
    raw_mask = nib.load(str(example_path)).get_fdata()
    largest_mask, _, _ = load_largest_component(example_path)
    raw_bin = raw_mask > 0

    # Pick the slice with maximal tumor area along the last axis using the largest component
    slice_sums = largest_mask.sum(axis=tuple(range(largest_mask.ndim - 1)))
    slice_idx = int(slice_sums.argmax())

    # Load matching image slice
    img_path = DATA_ROOT / patient_id / f"slice_{slice_idx:03d}_frame_000.nii"
    img_arr = nib.load(str(img_path)).get_fdata()
    if img_arr.ndim >= 3 and img_arr.shape[0] == 2:
        img_complex = img_arr[0] + 1j * img_arr[1]
        img = np.abs(img_complex)
    else:
        img = np.squeeze(img_arr)

    # Prepare overlays
    raw_slice = raw_bin[..., slice_idx]
    largest_slice = largest_mask[..., slice_idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img, cmap="gray")
    axes[0].imshow(np.ma.masked_where(~raw_slice, raw_slice), cmap="autumn", alpha=0.4)
    axes[0].set_title(f"{patient_id} slice {slice_idx:03d} — original mask")
    axes[0].axis("off")

    axes[1].imshow(img, cmap="gray")
    axes[1].imshow(np.ma.masked_where(~largest_slice, largest_slice), cmap="autumn", alpha=0.4)
    axes[1].set_title(f"{patient_id} slice {slice_idx:03d} — largest component")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig('example_tumor_selected_region.png')
