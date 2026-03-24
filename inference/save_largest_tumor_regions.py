"""Save largest tumor components and write slice indices CSV. Run: python3 -m inference.save_largest_tumor_regions"""

from pathlib import Path

import csv
import nibabel as nib
import numpy as np
from scipy.ndimage import label

REPO_ROOT = Path(__file__).resolve().parents[1]

# Source and destination directories
# SEG_ROOT = Path("/net/scratch2/rachelgordon/zf_data_192_slices/tumor_segmentations")
SEG_ROOT = Path("/ess/scratch/scratch1/rachelgordon/zf_data_192_slices/tumor_segmentations")
OUT_ROOT = SEG_ROOT.parent / "tumor_segmentations_lcr"
SLICE_CSV = REPO_ROOT / "data" / "largest_tumor_slices.csv"


def largest_component(mask_path: Path) -> tuple[np.ndarray, np.ndarray, nib.nifti1.Nifti1Header]:
    nii = nib.load(str(mask_path))
    data = nii.get_fdata()
    bin_mask = data > 0

    labeled, n = label(bin_mask)
    if n == 0:
        return np.zeros_like(bin_mask, dtype=np.uint8), nii.affine, nii.header

    counts = np.bincount(labeled.ravel())[1:]  # skip background
    largest_label = counts.argmax() + 1
    largest = (labeled == largest_label).astype(np.uint8)
    return largest, nii.affine, nii.header


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    SLICE_CSV.parent.mkdir(parents=True, exist_ok=True)

    masks = sorted(SEG_ROOT.glob("*.nii.gz"))
    print(f"Found {len(masks)} segmentation files in {SEG_ROOT}")

    rows = [("fastMRI_breast_id", "largest_slice_idx")]

    for p in masks:
        patient_id = p.name.replace(".nii.gz", "")
        if patient_id.endswith(".nii"):
            patient_id = patient_id.replace(".nii", "")

        largest, affine, header = largest_component(p)
        out_path = OUT_ROOT / p.name
        nib.save(nib.Nifti1Image(largest, affine, header), str(out_path))
        print(f"Saved largest component to {out_path}")

        # Find the slice (last axis) with the largest tumor area within the largest component
        slice_idx = -1
        if largest.any():
            slice_sums = largest.sum(axis=tuple(range(largest.ndim - 1)))
            slice_idx = int(np.argmax(slice_sums))

        rows.append((patient_id, slice_idx))

    with open(SLICE_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Wrote largest-slice indices to {SLICE_CSV}")


if __name__ == "__main__":
    main()
