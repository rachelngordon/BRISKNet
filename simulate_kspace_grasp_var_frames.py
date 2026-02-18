#!/usr/bin/env python3
"""
Loop over DRO variable-frame samples, simulate k-space, compute/load ESPIRiT csmaps,
and run GRASP reconstructions. Designed to be resumable (skips existing outputs).
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Optional

import numpy as np
import torch
import torchkbnufft as tkbn
import sigpy as sp
from sigpy.mri import app

try:
    import h5py
except ImportError as exc:
    raise ImportError("h5py is required to read .mat (v7.3) files.") from exc

from radial_lsfp import MCNUFFT


FRAME_TO_SPF = {
    8: 36,
    12: 24,
    18: 16,
    36: 8,
    72: 4,
    144: 2,
}
FRAME_ORDER = [8, 12, 18, 36, 72, 144]


def trajGR(Nkx, Nspokes):
    ga = np.pi * ((1 - np.sqrt(5)) / 2)
    kx = np.zeros(shape=(Nkx, Nspokes))
    ky = np.zeros(shape=(Nkx, Nspokes))
    ky[:, 0] = np.linspace(-np.pi, np.pi, Nkx)
    for i in range(1, Nspokes):
        kx[:, i] = np.cos(ga) * kx[:, i - 1] - np.sin(ga) * ky[:, i - 1]
        ky[:, i] = np.sin(ga) * kx[:, i - 1] + np.cos(ga) * ky[:, i - 1]
    ky = np.transpose(ky)
    kx = np.transpose(kx)
    ktraj = np.stack((ky.flatten(), kx.flatten()), axis=0)
    return ktraj


def get_traj(N_spokes=13, N_time=1, base_res=320, gind=1):
    n_tot_spokes = N_spokes * N_time
    n_samples = base_res * 2
    base_lin = np.arange(n_samples).reshape(1, -1) - base_res

    tau = 0.5 * (1 + 5**0.5)
    base_rad = np.pi / (gind + tau - 1)
    base_rot = np.arange(n_tot_spokes).reshape(-1, 1) * base_rad

    traj = np.zeros((n_tot_spokes, n_samples, 2))
    traj[..., 0] = np.cos(base_rot) @ base_lin
    traj[..., 1] = np.sin(base_rot) @ base_lin
    traj = traj / 2
    traj = traj.reshape(N_time, N_spokes, n_samples, 2)
    return np.squeeze(traj)


def _ktraj_and_dcomp_from_get_traj(Nsample, Nspokes, Ng, im_size):
    base_res = int(Nsample // 2)
    traj = get_traj(
        N_spokes=Nspokes,
        N_time=Ng,
        base_res=base_res,
        gind=1,
    )
    traj = np.asarray(traj)
    if traj.ndim == 3:
        traj = traj[None, ...]

    traj_flat = traj.reshape(Ng, Nspokes * Nsample, 2)
    ktraj = np.transpose(traj_flat, (2, 1, 0))  # (2, M, T)
    ktraj = torch.tensor(ktraj, dtype=torch.float)
    ktraj = ktraj * (2 * np.pi / base_res)

    dcomps = []
    for t in range(Ng):
        d = tkbn.calc_density_compensation_function(
            ktraj=ktraj[:, :, t], im_size=im_size
        ).squeeze()
        dcomps.append(d)

    dcomp = torch.stack(dcomps, dim=-1)  # (M, T)
    dcomp = dcomp.to(torch.complex64)
    return ktraj, dcomp


def prep_nufft(Nsample, Nspokes, Ng, traj_method="trajGR"):
    overSmaple = 2
    im_size = (int(Nsample / overSmaple), int(Nsample / overSmaple))
    grid_size = (Nsample, Nsample)

    if traj_method == "trajGR":
        ktraj = trajGR(Nsample, Nspokes * Ng)
        ktraj = torch.tensor(ktraj, dtype=torch.float)
        dcomp = tkbn.calc_density_compensation_function(ktraj=ktraj, im_size=im_size)
        dcomp = dcomp.squeeze()

        ktraju = np.zeros([2, Nspokes * Nsample, Ng], dtype=float)
        dcompu = np.zeros([Nspokes * Nsample, Ng], dtype=complex)

        for ii in range(0, Ng):
            ktraju[:, :, ii] = ktraj[:, (ii * Nspokes * Nsample):((ii + 1) * Nspokes * Nsample)]
            dcompu[:, ii] = dcomp[(ii * Nspokes * Nsample):((ii + 1) * Nspokes * Nsample)]

        ktraju = torch.tensor(ktraju, dtype=torch.float)
        dcompu = torch.tensor(dcompu, dtype=torch.complex64)
    elif traj_method == "get_traj":
        ktraju, dcompu = _ktraj_and_dcomp_from_get_traj(Nsample, Nspokes, Ng, im_size)
    else:
        raise ValueError(f"Unknown traj_method: {traj_method}")

    nufft_ob = tkbn.KbNufft(im_size=im_size, grid_size=grid_size)
    adjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size, grid_size=grid_size)
    return ktraju, dcompu, nufft_ob, adjnufft_ob


def _read_h5_complex(dset: h5py.Dataset) -> np.ndarray:
    type_id = dset.id.get_type()
    type_class = type_id.get_class()
    if type_class != h5py.h5t.COMPOUND:
        raise TypeError("Expected compound dataset for complex values.")
    memtype = h5py.h5t.create(h5py.h5t.COMPOUND, type_id.get_size())
    names = []
    formats = []
    offsets = []
    for idx in range(type_id.get_nmembers()):
        name = type_id.get_member_name(idx)
        name_str = name.decode() if isinstance(name, (bytes, bytearray)) else str(name)
        member = type_id.get_member_type(idx)
        if member.get_class() != h5py.h5t.FLOAT:
            raise TypeError("Unsupported compound member type in complex dataset.")
        np_fmt = np.float32 if member.get_size() == 4 else np.float64
        memtype.insert(
            name,
            type_id.get_member_offset(idx),
            h5py.h5t.NATIVE_FLOAT if np_fmt == np.float32 else h5py.h5t.NATIVE_DOUBLE,
        )
        names.append(name_str)
        formats.append(np_fmt)
        offsets.append(type_id.get_member_offset(idx))
    arr = np.empty(
        dset.shape,
        dtype=np.dtype(
            {"names": names, "formats": formats, "offsets": offsets, "itemsize": type_id.get_size()}
        ),
    )
    dset.read_direct(arr)
    name_map = {n.lower(): n for n in names}
    real_key = name_map.get("real", names[0])
    imag_key = name_map.get("imag", names[-1])
    return arr[real_key] + 1j * arr[imag_key]


def _read_h5_array(dset: h5py.Dataset) -> np.ndarray:
    if dset.dtype.kind == "V":
        return _read_h5_complex(dset)
    arr = np.asarray(dset)
    if arr.ndim == 4 and arr.shape[-1] == 2 and not np.iscomplexobj(arr):
        arr = arr[..., 0] + 1j * arr[..., 1]
    return arr


def _find_h5_dataset(h5: h5py.File, key: Optional[str]) -> tuple[np.ndarray, str]:
    if key:
        if key not in h5:
            raise KeyError(f"Key '{key}' not found in {h5.filename}")
        return _read_h5_array(h5[key]), key

    if "simImg" in h5:
        return _read_h5_array(h5["simImg"]), "simImg"

    candidates = []

    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
            candidates.append(name)

    h5.visititems(_visit)
    if not candidates:
        raise ValueError(f"No 3D dataset found in {h5.filename}. Use --key to specify.")
    return _read_h5_array(h5[candidates[0]]), candidates[0]


def _load_smaps(h5: h5py.File) -> np.ndarray:
    if "smap" not in h5:
        raise KeyError(f"smap key not found in {h5.filename}")
    smap = h5["smap"]
    try:
        smaps = _read_h5_complex(smap)
    except TypeError:
        smaps = np.asarray(smap)
        if smaps.ndim == 4 and smaps.shape[-1] == 2:
            smaps = smaps[..., 0] + 1j * smaps[..., 1]
    return smaps


def _ensure_hwt(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")
    if arr.shape[0] == h and arr.shape[1] == w:
        return arr
    if arr.shape[1] == h and arr.shape[2] == w:
        return np.transpose(arr, (1, 2, 0))
    return arr


def get_coil(ksp: np.ndarray, spokes_per_frame: int, device=sp.Device(-1)) -> np.ndarray:
    N_coils, N_spokes, N_samples = ksp.shape
    base_res = N_samples // 2
    ishape = [N_coils] + [base_res] * 2

    traj = get_traj(N_spokes=N_spokes, N_time=1, base_res=base_res, gind=1)
    dcf = (traj[..., 0] ** 2 + traj[..., 1] ** 2) ** 0.5

    F = sp.linop.NUFFT(ishape, traj)
    cim = F.H(ksp * dcf)
    cim = sp.fft(cim, axes=(-2, -1))

    mps = app.EspiritCalib(cim, device=device).run()
    mps = sp.to_device(mps, sp.Device(-1))
    return np.asarray(mps)


def _normalize_suffix(suffix: str) -> str:
    if not suffix:
        return ""
    if not suffix.startswith("_"):
        suffix = "_" + suffix
    return suffix


def _find_mat_path(sample_dir: str, frames: int) -> str:
    pattern = os.path.join(sample_dir, f"*dro_{frames}frames.mat")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No .mat found for {frames} frames in {sample_dir}")
    if len(matches) > 1:
        raise ValueError(f"Multiple .mat files found for {frames} frames in {sample_dir}: {matches}")
    return matches[0]


def _load_dro(mat_path: str, key: Optional[str]) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(mat_path, "r") as f:
        img, img_key = _find_h5_dataset(f, key)
        smaps = _load_smaps(f)
    if smaps.ndim != 3:
        raise ValueError(f"Expected smap shape (C,H,W) or (H,W,C), got {smaps.shape}")
    if smaps.shape[0] <= smaps.shape[1] and smaps.shape[0] <= smaps.shape[2]:
        smaps_chw = smaps
    else:
        smaps_chw = np.transpose(smaps, (2, 0, 1))

    h = int(smaps_chw.shape[1])
    w = int(smaps_chw.shape[2])
    img = _ensure_hwt(img, h, w)
    return img, smaps_chw


def _build_recon_traj(spf: int, frames: int, nsample: int, traj_method: str) -> np.ndarray:
    if traj_method == "get_traj":
        return get_traj(N_spokes=spf, N_time=frames)
    if traj_method == "trajGR":
        ktraj = trajGR(nsample, spf * frames)  # (2, total_spokes * samples)
        ktraj = ktraj.reshape(2, spf * frames, nsample)
        traj = np.transpose(ktraj, (1, 2, 0))  # (total_spokes, samples, 2)
        traj = traj.reshape(frames, spf, nsample, 2)
        return traj
    raise ValueError(f"Unknown traj_method: {traj_method}")


def _simulate_kspace(
    img: np.ndarray,
    smaps_chw: np.ndarray,
    spf: int,
    frames: int,
    device: torch.device,
    traj_method: str,
    noise_std: float,
    noise_seed: Optional[int],
) -> np.ndarray:
    h, w, t = img.shape
    if h != w:
        raise ValueError(f"Expected square frames, got H={h}, W={w}")
    if t != frames:
        raise ValueError(f"Image frames ({t}) != expected frames ({frames})")
    nsample = h * 2

    ktraj, dcomp, nufft_ob, adjnufft_ob = prep_nufft(
        nsample, spf, frames, traj_method=traj_method
    )
    physics = MCNUFFT(
        nufft_ob.to(device),
        adjnufft_ob.to(device),
        ktraj.to(device),
        dcomp.to(device),
    )

    sim_img = torch.tensor(img, dtype=torch.complex64, device=device)
    csmaps = torch.tensor(np.expand_dims(smaps_chw, axis=0), dtype=torch.complex64, device=device)

    with torch.no_grad():
        kspace = physics(False, sim_img, csmaps)
        if noise_std < 0:
            raise ValueError("--kspace-noise-std must be >= 0.")
        if noise_std > 0:
            if noise_seed is not None:
                torch.manual_seed(noise_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(noise_seed)
            noise = torch.randn(kspace.shape, device=kspace.device, dtype=kspace.real.dtype)
            kspace = kspace + noise_std * noise

    return kspace.detach().cpu().numpy()


def _compute_csmaps(kspace_np: np.ndarray, spf: int, frames: int, device: sp.Device) -> np.ndarray:
    if kspace_np.ndim != 3:
        raise ValueError(f"Expected kspace shape (C, M, T), got {kspace_np.shape}")
    n_coils, m, t = kspace_np.shape
    if t != frames:
        raise ValueError(f"kspace frames ({t}) != expected frames ({frames})")
    if m % spf != 0:
        raise ValueError(f"kspace samples ({m}) not divisible by spf ({spf})")
    nsample = m // spf

    kspace_all = kspace_np.reshape(n_coils, spf, nsample, t)
    kspace_all = np.transpose(kspace_all, (0, 3, 1, 2))  # (C, T, spf, sam)
    kspace_all = kspace_all.reshape(n_coils, t * spf, nsample)
    csmaps = get_coil(kspace_all, t * spf, device=device)
    return np.asarray(csmaps).astype(np.complex64)


def _run_grasp(
    kspace_np: np.ndarray,
    csmaps_np: np.ndarray,
    spf: int,
    frames: int,
    traj_method: str,
    lamda: float,
    max_iter: int,
    rho: float,
    device: sp.Device,
) -> np.ndarray:
    if kspace_np.ndim != 3:
        raise ValueError(f"Expected kspace shape (C, M, T), got {kspace_np.shape}")
    n_coils, m, t = kspace_np.shape
    if t != frames:
        raise ValueError(f"kspace frames ({t}) != expected frames ({frames})")
    if m % spf != 0:
        raise ValueError(f"kspace samples ({m}) not divisible by spf ({spf})")
    nsample = m // spf

    if csmaps_np.ndim != 3:
        raise ValueError(f"Expected csmaps shape (C, H, W), got {csmaps_np.shape}")
    if csmaps_np.shape[0] != n_coils and csmaps_np.shape[-1] == n_coils:
        csmaps_np = np.transpose(csmaps_np, (2, 0, 1))
    if csmaps_np.shape[0] != n_coils:
        raise ValueError(
            f"CSMAP coil count mismatch: expected {n_coils}, got {csmaps_np.shape[0]}"
        )

    kspace_tcsp = kspace_np.reshape(n_coils, spf, nsample, t)
    kspace_tcsp = np.transpose(kspace_tcsp, (3, 0, 1, 2))  # (T, C, spf, sam)
    kspace_tcsp = kspace_tcsp[:, None, :, None, :, :]

    csmaps = csmaps_np[:, None, :, :]
    traj = _build_recon_traj(spf, frames, nsample, traj_method)

    recon = app.HighDimensionalRecon(
        kspace_tcsp,
        csmaps,
        combine_echo=False,
        lamda=lamda,
        coord=traj,
        regu="TV",
        regu_axes=[0],
        max_iter=max_iter,
        solver="ADMM",
        rho=rho,
        device=device,
        show_pbar=False,
        verbose=False,
    ).run()

    recon_np = np.squeeze(recon.get())
    if recon_np.ndim == 3 and recon_np.shape[0] == frames and recon_np.shape[-1] != frames:
        recon_np = np.transpose(recon_np, (1, 2, 0))
    return recon_np


def _parse_sigpy_device(device: str) -> sp.Device:
    if device == "cuda":
        return sp.Device(0)
    if device == "cpu":
        return sp.Device(-1)
    raise ValueError(f"Unknown sigpy device '{device}'. Use 'cpu' or 'cuda'.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate k-space and GRASP reconstructions for DRO variable-frame .mat files."
    )
    parser.add_argument(
        "--dro-root",
        default="/net/scratch2/rachelgordon/dro_var_frames",
        help="Root directory containing sub*/sample_*_dro_*frames.mat files.",
    )
    parser.add_argument(
        "--key",
        default=None,
        help="Dataset key inside the .mat file (optional).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for k-space simulation (e.g., cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--sigpy-device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for ESPIRiT calibration and GRASP recon.",
    )
    parser.add_argument(
        "--traj-method",
        default="get_traj",
        choices=("trajGR", "get_traj"),
        help="Trajectory method to use for k-space simulation.",
    )
    parser.add_argument(
        "--kspace-noise-std",
        type=float,
        default=0.0,
        help="Std of additive real Gaussian noise in k-space (0 disables).",
    )
    parser.add_argument(
        "--kspace-noise-seed",
        type=int,
        default=None,
        help="Optional RNG seed for k-space noise.",
    )
    parser.add_argument(
        "--lamda",
        type=float,
        default=0.001,
        help="GRASP TV weight.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=10,
        help="GRASP max iterations.",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.1,
        help="GRASP ADMM rho.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Optional suffix to append to output filenames (before .npy).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    sigpy_device = _parse_sigpy_device(args.sigpy_device)
    suffix = _normalize_suffix(args.suffix)

    sample_dirs = sorted(
        [
            os.path.join(args.dro_root, d)
            for d in os.listdir(args.dro_root)
            if d.startswith("sub") and os.path.isdir(os.path.join(args.dro_root, d))
        ],
        key=lambda p: int(re.search(r"sub(\d+)", os.path.basename(p)).group(1))
        if re.search(r"sub(\d+)", os.path.basename(p))
        else os.path.basename(p),
    )
    if not sample_dirs:
        raise FileNotFoundError(f"No sub* directories found under {args.dro_root}")

    for sample_dir in sample_dirs:
        sample_name = os.path.basename(sample_dir)
        for frames in FRAME_ORDER:
            spf = FRAME_TO_SPF[frames]
            mat_path = _find_mat_path(sample_dir, frames)

            kspace_path = os.path.join(
                sample_dir,
                f"simulated_kspace_spf{spf}_frames{frames}{suffix}.npy",
            )
            csmaps_path = os.path.join(
                sample_dir,
                f"csmaps_espirit_spf{spf}_frames{frames}{suffix}.npy",
            )
            recon_path = os.path.join(
                sample_dir,
                f"grasp_spf{spf}_frames{frames}{suffix}.npy",
            )

            need_kspace = not os.path.exists(kspace_path)
            need_csmaps = not os.path.exists(csmaps_path)
            need_recon = not os.path.exists(recon_path)

            if not (need_kspace or need_csmaps or need_recon):
                print(f"[skip] {sample_name} frames={frames} (all outputs exist)")
                continue

            print(
                f"[run] {sample_name} frames={frames} spf={spf} "
                f"(kspace={need_kspace}, csmaps={need_csmaps}, recon={need_recon})"
            )

            kspace_np = None
            if need_kspace:
                img, smaps_chw = _load_dro(mat_path, args.key)
                kspace_np = _simulate_kspace(
                    img=img,
                    smaps_chw=smaps_chw,
                    spf=spf,
                    frames=frames,
                    device=device,
                    traj_method=args.traj_method,
                    noise_std=args.kspace_noise_std,
                    noise_seed=args.kspace_noise_seed,
                )
                np.save(kspace_path, kspace_np)
                print(f"  saved kspace: {kspace_path}")
            else:
                kspace_np = np.load(kspace_path)

            csmaps_np = None
            if need_csmaps:
                csmaps_np = _compute_csmaps(
                    kspace_np=kspace_np,
                    spf=spf,
                    frames=frames,
                    device=sigpy_device,
                )
                np.save(csmaps_path, csmaps_np)
                print(f"  saved csmaps: {csmaps_path}")
            else:
                csmaps_np = np.load(csmaps_path)

            if need_recon:
                recon_np = _run_grasp(
                    kspace_np=kspace_np,
                    csmaps_np=csmaps_np,
                    spf=spf,
                    frames=frames,
                    traj_method=args.traj_method,
                    lamda=args.lamda,
                    max_iter=args.max_iter,
                    rho=args.rho,
                    device=sigpy_device,
                )
                np.save(recon_path, recon_np)
                print(f"  saved recon: {recon_path}")


if __name__ == "__main__":
    main()
