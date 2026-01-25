#!/usr/bin/env python3
import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp

from process_data.nifti.dce_recon import get_traj as get_traj_dce
from utils import trajGR


def _as_complex(arr: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(arr):
        return arr
    if arr.shape[-1] == 2:
        return arr[..., 0] + 1j * arr[..., 1]
    raise ValueError(f"Unsupported kspace dtype/shape: {arr.dtype}, shape={arr.shape}")


def _rss(img: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.abs(img) ** 2, axis=0))


def _adjoint_nufft(kspace: np.ndarray, traj: np.ndarray) -> np.ndarray:
    # kspace: (C, Spokes, Samples)
    # traj: (Spokes, Samples, 2)
    dcf = np.sqrt(traj[..., 0] ** 2 + traj[..., 1] ** 2)
    ishape = [kspace.shape[0], kspace.shape[2] // 2, kspace.shape[2] // 2]
    F = sp.linop.NUFFT(ishape, traj)
    img = F.H(kspace * dcf)
    return _rss(img)


def main():
    parser = argparse.ArgumentParser(description="Compare adjoint images from two trajectories.")
    parser.add_argument("--h5", required=True, help="Path to h5 file in zf_kspace.")
    parser.add_argument("--slice-idx", type=int, default=0, help="Slice index in the h5.")
    parser.add_argument("--frame-idx", type=int, default=0, help="Time frame index to use.")
    parser.add_argument("--out", default="adjoint_compare.png", help="Output image path.")
    args = parser.parse_args()

    if not os.path.isfile(args.h5):
        raise FileNotFoundError(args.h5)

    with h5py.File(args.h5, "r") as f:
        if "kspace" not in f:
            raise KeyError("Expected dataset 'kspace' in h5 file.")
        kspace_slice = f["kspace"][args.slice_idx]

    # Expected shape: (T, C, Spokes, Samples)
    # if kspace_slice.ndim != 4:
    #     raise ValueError(f"Expected kspace shape (T,C,Spokes,Samples), got {kspace_slice.shape}")

    kspace_frame = _as_complex(kspace_slice)
    n_spokes = kspace_frame.shape[1]
    n_samples = kspace_frame.shape[2]
    base_res = n_samples // 2

    class _Args:
        spokes_per_frame = n_spokes

    # Trajectory 1: get_traj from dce_recon.py
    traj_dce = get_traj_dce(_Args(), csmaps=False, N_spokes=n_spokes, N_time=1, base_res=base_res, gind=1)
    print("get_traj output shape: ", traj_dce.shape)
    if traj_dce.ndim == 4:
        traj_dce = traj_dce[0]

    # Trajectory 2: trajGR from utils.py
    traj_gr = trajGR(n_samples, n_spokes)
    print("trajGR output shape: ", traj_gr.shape)
    traj_gr = traj_gr.reshape(2, n_spokes, n_samples).transpose(1, 2, 0)

    img_dce = _adjoint_nufft(kspace_frame, traj_dce)
    img_gr = _adjoint_nufft(kspace_frame, traj_gr)

    vmin = np.percentile(np.concatenate([img_dce.ravel(), img_gr.ravel()]), 1)
    vmax = np.percentile(np.concatenate([img_dce.ravel(), img_gr.ravel()]), 99.5)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_dce, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title("Adjoint (get_traj)")
    axes[0].axis("off")

    axes[1].imshow(img_gr, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].set_title("Adjoint (trajGR)")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
