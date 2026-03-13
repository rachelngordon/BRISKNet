"""Run GRASP recon for the largest-tumor slices listed in a CSV. Run: python3 -m inference.grasp_recon_largest_slices --help"""

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import sigpy as sp
import torch

from inference.raw_kspace_eval import process_kspace, raw_grasp_recon


def normalize_patient_id(pid: str) -> str:
    """Drop any trailing .nii/.nii.gz from the patient id."""
    if pid.endswith(".nii.gz"):
        pid = pid[:-7]
    if pid.endswith(".nii"):
        pid = pid[:-4]
    return pid


def load_slice_map(csv_path: Path) -> List[Tuple[str, int]]:
    """Return list of (patient_id, slice_idx) for valid slices (slice_idx >=0)."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("fastMRI_breast_id")
            idx = row.get("largest_slice_idx")
            if pid is None or idx is None:
                continue
            pid = normalize_patient_id(pid)
            try:
                slice_idx = int(idx)
            except ValueError:
                continue
            if slice_idx < 0:
                continue  # skip empty masks
            rows.append((pid, slice_idx))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run GRASP recon for slices listed in largest_tumor_slices.csv")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/largest_tumor_slices.csv",
        help="CSV with fastMRI_breast_id and largest_slice_idx",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/net/scratch2/rachelgordon/zf_data_192_slices/zf_kspace",
        help="Directory containing raw k-space .h5 files",
    )
    parser.add_argument(
        "--total_spokes",
        type=int,
        default=288,
        help="Total spokes per scan (used to derive num_frames)",
    )
    parser.add_argument(
        "--spokes_per_frame",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16, 24, 36],
        help="List of spokes/frame to reconstruct",
    )
    parser.add_argument(
        "--images_per_slab",
        type=int,
        default=1,
        help="Number of slices to reconstruct per slab (matches process_kspace arg)",
    )
    parser.add_argument(
        "--center_partition",
        type=int,
        default=31,
        help="Center partition index passed to process_kspace",
    )
    args = parser.parse_args()

    slice_map = load_slice_map(Path(args.csv_path))
    if not slice_map:
        print(f"No valid slice entries found in {args.csv_path}")
        return

    device = sp.Device(0 if torch.cuda.is_available() else -1)
    output_root = Path(args.data_dir).resolve().parent

    for spf in args.spokes_per_frame:
        num_frames = int(args.total_spokes / spf)
        print(f"\n=== Reconstructing with {spf} spokes/frame ({num_frames} frames) ===")

        for patient_id, slice_idx in slice_map:
            raw_kspace_path = Path(args.data_dir) / f"{patient_id}.h5"
            patient_dir = output_root / patient_id
            grasp_img_path = patient_dir / f"grasp_recon_{spf}spf_{num_frames}frames_slice{slice_idx}.npy"

            if not raw_kspace_path.exists():
                print(f"Skipping {patient_id}: k-space not found at {raw_kspace_path}")
                continue
            if grasp_img_path.exists():
                print(f"Skipping {patient_id}, slice {slice_idx}: recon already exists at {grasp_img_path}")
                continue

            print(f"Patient {patient_id}, slice {slice_idx}")
            zf_kspace, binned_kspace, traj = process_kspace(
                str(raw_kspace_path),
                device=device,
                spokes_per_frame=spf,
                images_per_slab=args.images_per_slab,
                center_partition=args.center_partition,
            )

            if slice_idx >= zf_kspace.shape[0]:
                print(f"  Slice {slice_idx} out of bounds for {patient_id} (max {zf_kspace.shape[0]-1}); skipping.")
                continue

            zf_kspace_slice = np.expand_dims(zf_kspace[slice_idx], axis=0)
            binned_kspace_slice = np.expand_dims(binned_kspace[slice_idx], axis=0)

            patient_dir.mkdir(parents=True, exist_ok=True)

            grasp_img_slices = raw_grasp_recon(
                zf_kspace_slice,
                binned_kspace_slice,
                traj,
                N_slices=args.images_per_slab,
                spokes_per_frame=spf,
                device=device,
            )
            np.save(grasp_img_path, grasp_img_slices)
            print(f"  Saved GRASP recon to {grasp_img_path}")


if __name__ == "__main__":
    main()
