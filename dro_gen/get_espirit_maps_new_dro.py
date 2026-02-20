#!/usr/bin/env python3
import argparse
import glob
import os

import h5py
import h5py.h5t as h5t
import numpy as np
import sigpy as sp
from sigpy.mri import app
import matplotlib.pyplot as plt


def _read_h5_float(dset: h5py.Dataset) -> np.ndarray:
    type_id = dset.id.get_type()
    if type_id.get_class() != h5t.FLOAT:
        raise TypeError(f"Expected float dataset; got class={type_id.get_class()}")
    size = type_id.get_size()
    if size == 4:
        np_dtype = np.float32
    elif size == 8:
        np_dtype = np.float64
    else:
        raise TypeError(f"Unsupported float size for HDF5 dtype: {size}")
    arr = np.empty(dset.shape, dtype=np_dtype)
    dset.read_direct(arr)
    return arr


def _read_h5_complex(dset: h5py.Dataset) -> np.ndarray:
    type_id = dset.id.get_type()
    type_class = type_id.get_class()
    if type_class == h5t.COMPOUND:
        memtype = h5t.create(h5t.COMPOUND, type_id.get_size())
        names = []
        formats = []
        offsets = []
        for idx in range(type_id.get_nmembers()):
            name = type_id.get_member_name(idx)
            name_str = name.decode() if isinstance(name, (bytes, bytearray)) else str(name)
            member = type_id.get_member_type(idx)
            if member.get_class() != h5t.FLOAT:
                raise TypeError("Unsupported compound member type in complex dataset.")
            size = member.get_size()
            if size == 4:
                np_fmt = np.float32
                native = h5t.NATIVE_FLOAT
            elif size == 8:
                np_fmt = np.float64
                native = h5t.NATIVE_DOUBLE
            else:
                raise TypeError(f"Unsupported float size for HDF5 dtype: {size}")
            memtype.insert(name, type_id.get_member_offset(idx), native)
            names.append(name_str)
            formats.append(np_fmt)
            offsets.append(type_id.get_member_offset(idx))
        np_dtype = np.dtype(
            {
                "names": names,
                "formats": formats,
                "offsets": offsets,
                "itemsize": type_id.get_size(),
            }
        )
        arr = np.empty(dset.shape, dtype=np_dtype)
        dset.id.read(h5py.h5s.ALL, h5py.h5s.ALL, arr, memtype)
        if "real" in arr.dtype.names and "imag" in arr.dtype.names:
            return arr["real"] + 1j * arr["imag"]
        if len(arr.dtype.names) >= 2:
            return arr[arr.dtype.names[0]] + 1j * arr[arr.dtype.names[1]]
        raise ValueError("Compound dataset did not contain real/imag fields.")
    if type_class == h5t.FLOAT:
        real = _read_h5_float(dset)
        return real.astype(np.complex64)
    raise TypeError(f"Unsupported HDF5 dtype class for complex read: {type_class}")


def _load_kspace_and_traj(kspace_path: str):
    with h5py.File(kspace_path, "r") as f:
        if "kspace" not in f:
            raise KeyError(f"{kspace_path} missing required key 'kspace'.")
        if "traj" not in f:
            raise KeyError(f"{kspace_path} missing required key 'traj'.")
        kspace = _read_h5_complex(f["kspace"]).astype(np.complex64)
        traj = _read_h5_complex(f["traj"]).astype(np.complex64)
    return kspace, traj


def _load_dro_csmaps(root_dir: str, sample_id: str, num_frames: int):
    dro_path = os.path.join(root_dir, f"{sample_id}_dro_{num_frames}frames.mat")
    with h5py.File(dro_path, "r") as f:
        if "smap" not in f:
            raise KeyError(f"{dro_path} missing required key 'smap'.")
        smap = _read_h5_complex(f["smap"]).astype(np.complex64)
    if smap.ndim != 3:
        raise ValueError(f"Unexpected smap shape {smap.shape} in {dro_path}.")
    if smap.shape[0] != 16 and smap.shape[-1] == 16:
        smap = np.transpose(smap, (2, 0, 1))
    return smap


def _plot_csmaps_compare(original_maps: np.ndarray, espirit_maps: np.ndarray, out_path: str):
    original_mag = np.abs(original_maps)
    espirit_mag = np.abs(espirit_maps)
    n_coils = original_mag.shape[0]
    vmax = np.percentile(
        np.concatenate([original_mag.ravel(), espirit_mag.ravel()]), 99.5
    )
    fig, axes = plt.subplots(2, n_coils, figsize=(n_coils * 1.2, 4))
    for coil_idx in range(n_coils):
        axes[0, coil_idx].imshow(
            original_mag[coil_idx], cmap="gray", vmin=0, vmax=vmax
        )
        axes[0, coil_idx].axis("off")
        axes[0, coil_idx].set_title(f"C{coil_idx:02d}", fontsize=8)

        axes[1, coil_idx].imshow(
            espirit_mag[coil_idx], cmap="gray", vmin=0, vmax=vmax
        )
        axes[1, coil_idx].axis("off")

    axes[0, 0].set_ylabel("DRO", fontsize=10)
    axes[1, 0].set_ylabel("ESPIRiT", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _prep_coord(traj: np.ndarray, total_spokes: int, samples: int):
    if traj.shape == (samples, total_spokes):
        traj = traj.T
    if traj.shape != (total_spokes, samples):
        raise ValueError(
            f"Unexpected traj shape {traj.shape}; expected ({total_spokes}, {samples})."
        )
    max_abs = float(np.max(np.abs(traj)))
    base_res = samples // 2
    if max_abs <= 1.0:
        traj = traj * base_res
        print(f"[traj] Scaling normalized traj by base_res={base_res} (max_abs={max_abs:.3g}).")
    # coord = np.stack([traj.real, traj.imag], axis=-1)  # (spokes, samples, 2)
    coord = np.stack([traj.imag, traj.real], axis=-1)

    return coord


def _get_espirit_maps(kspace: np.ndarray, coord: np.ndarray, device: sp.Device):
    n_coils, total_spokes, samples = kspace.shape
    base_res = samples // 2
    ishape = [n_coils, base_res, base_res]

    dcf = np.sqrt(coord[..., 0] ** 2 + coord[..., 1] ** 2)

    F = sp.linop.NUFFT(ishape, coord)
    cim = F.H(kspace * dcf)
    cim = sp.fft(cim, axes=(-2, -1))

    mps = app.EspiritCalib(cim, device=device).run()
    mps = sp.to_device(mps, sp.Device(-1))
    return mps


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ESPIRiT maps for new DRO kspace.")
    parser.add_argument(
        "--root_dir",
        default="/net/scratch2/rachelgordon/dro_var_frames_kspace",
        help="Root directory containing DRO .mat files.",
    )
    parser.add_argument(
        "--spokes_per_frame",
        type=int,
        default=36,
        help="Spokes per frame to select kspace files (e.g., 36).",
    )
    parser.add_argument(
        "--total_spokes",
        type=int,
        default=288,
        help="Total spokes for calculating number of frames.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device for ESPIRiT calibration.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = sp.Device(0) if args.device == "cuda" else sp.Device(-1)

    out_dir = os.path.join(args.root_dir, "csmaps_espirit")
    os.makedirs(out_dir, exist_ok=True)

    num_frames = int(args.total_spokes / args.spokes_per_frame)

    pattern = os.path.join(args.root_dir, f"sample_*_kspace_{args.spokes_per_frame}spf_{num_frames}frames.mat")
    kspace_files = sorted(glob.glob(pattern))
    if not kspace_files:
        raise FileNotFoundError(f"No kspace files found for pattern: {pattern}")

    for kspace_path in kspace_files:
        fname = os.path.basename(kspace_path)
        sample_id = fname.split("_kspace_")[0]

        kspace, traj = _load_kspace_and_traj(kspace_path)
        if kspace.ndim != 3:
            raise ValueError(f"Unexpected kspace shape {kspace.shape} in {kspace_path}.")
        n_coils, total_spokes, samples = kspace.shape

        coord = _prep_coord(traj, total_spokes=total_spokes, samples=samples)

        espirit_maps = _get_espirit_maps(kspace, coord, device=device)
        print(f"{sample_id}: ESPIRiT maps shape: {espirit_maps.shape}")

        out_path = os.path.join(out_dir, f"csmaps_{sample_id}.npy")
        np.save(out_path, espirit_maps)
        print(f"Saved ESPIRiT maps to {out_path}")

        dro_csmaps = _load_dro_csmaps(args.root_dir, sample_id, num_frames)
        plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"csmaps_compare_{sample_id}.png")
        _plot_csmaps_compare(dro_csmaps, espirit_maps, plot_path)
        print(f"Saved comparison plot to {plot_path}")


if __name__ == "__main__":
    main()
