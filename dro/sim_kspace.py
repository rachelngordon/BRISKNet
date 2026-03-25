#!/usr/bin/env python3
"""Loop over DRO variable-frame samples, simulate k-space, and compute/load ESPIRiT csmaps.

Run:
  python3 dro/sim_kspace.py --help
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

from model.radial import MCNUFFT


FRAME_TO_SPF = {
    8: 36,
    12: 24,
    18: 16,
    36: 8,
    72: 4,
    144: 2,
}
FRAME_ORDER = [8, 12, 18, 36, 72, 144]
CSMAPS_FRAMES = 8
_SAMPLE_ID_RE = re.compile(r"sample_(\d+)")


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


def _collect_dro_files(dro_root: str) -> dict[str, dict[int, str]]:
    pattern = os.path.join(dro_root, "sample_*_dro_*frames.mat")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No DRO .mat files found under {dro_root} matching {pattern}"
        )
    sample_map: dict[str, dict[int, str]] = {}
    for path in matches:
        base = os.path.basename(path)
        match = re.match(r"^(sample_\d+_sub\d+)_dro_(\d+)frames\.mat$", base)
        if not match:
            continue
        sample_id, frames_str = match.groups()
        frames = int(frames_str)
        sample_map.setdefault(sample_id, {})[frames] = path
    if not sample_map:
        raise FileNotFoundError(
            f"No DRO .mat files matched expected naming in {dro_root}."
        )
    return sample_map


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


def _sample_sort_key(sample_id: str) -> tuple[int, int | str]:
    match = _SAMPLE_ID_RE.search(sample_id)
    if match:
        return (0, int(match.group(1)))
    return (1, sample_id)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate k-space and ESPIRiT csmaps for DRO variable-frame .mat files."
    )
    parser.add_argument(
        "--dro-root",
        default="/net/scratch2/rachelgordon/dro_var_frames",
        help="Root directory containing sample_*_dro_*frames.mat files.",
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
        help="Device for ESPIRiT calibration.",
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

    sample_map = _collect_dro_files(args.dro_root)
    sample_ids = sorted(sample_map, key=_sample_sort_key)

    csmaps_dir = os.path.join(args.dro_root, "csmaps_espirit")
    os.makedirs(csmaps_dir, exist_ok=True)

    for sample_id in sample_ids:
        available_frames = sample_map[sample_id]
        ordered_frames = [f for f in FRAME_ORDER if f in available_frames]
        if not ordered_frames:
            ordered_frames = sorted(available_frames)
        csmaps_path = os.path.join(csmaps_dir, f"csmaps_{sample_id}{suffix}.npy")
        csmaps_np = None
        kspace_cache: dict[int, np.ndarray] = {}

        if not os.path.exists(csmaps_path):
            if CSMAPS_FRAMES not in available_frames:
                raise FileNotFoundError(
                    f"{sample_id} missing {CSMAPS_FRAMES}-frame DRO file required for ESPIRiT csmaps."
                )
            spf_csmaps = FRAME_TO_SPF[CSMAPS_FRAMES]
            mat_path = available_frames[CSMAPS_FRAMES]
            kspace_csmaps_path = os.path.join(
                args.dro_root,
                f"{sample_id}_kspace_{spf_csmaps}spf_{CSMAPS_FRAMES}frames{suffix}.npy",
            )
            if os.path.exists(kspace_csmaps_path):
                kspace_csmaps_np = np.load(kspace_csmaps_path)
            else:
                img, smaps_chw = _load_dro(mat_path, args.key)
                kspace_csmaps_np = _simulate_kspace(
                    img=img,
                    smaps_chw=smaps_chw,
                    spf=spf_csmaps,
                    frames=CSMAPS_FRAMES,
                    device=device,
                    traj_method=args.traj_method,
                    noise_std=args.kspace_noise_std,
                    noise_seed=args.kspace_noise_seed,
                )
                np.save(kspace_csmaps_path, kspace_csmaps_np)
                print(f"  saved kspace (for csmaps): {kspace_csmaps_path}")
            kspace_cache[CSMAPS_FRAMES] = kspace_csmaps_np
            csmaps_np = _compute_csmaps(
                kspace_np=kspace_csmaps_np,
                spf=spf_csmaps,
                frames=CSMAPS_FRAMES,
                device=sigpy_device,
            )
            np.save(csmaps_path, csmaps_np)
            print(f"  saved csmaps: {csmaps_path}")
        else:
            csmaps_np = np.load(csmaps_path)
        for frames in ordered_frames:
            if frames not in FRAME_TO_SPF:
                print(f"[skip] {sample_id} frames={frames}: no SPF mapping available.")
                continue
            spf = FRAME_TO_SPF[frames]
            mat_path = available_frames[frames]

            kspace_path = os.path.join(
                args.dro_root,
                f"{sample_id}_kspace_{spf}spf_{frames}frames{suffix}.npy",
            )

            need_kspace = not os.path.exists(kspace_path)
            if not need_kspace:
                print(f"[skip] {sample_id} frames={frames} (kspace exists)")
                continue

            print(f"[run] {sample_id} frames={frames} spf={spf} (kspace={need_kspace})")

            kspace_np = kspace_cache.get(frames)
            if kspace_np is None and need_kspace:
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
                kspace_cache[frames] = kspace_np
            elif kspace_np is None:
                kspace_np = np.load(kspace_path)

            # GRASP recon now handled in a separate script.


if __name__ == "__main__":
    main()
