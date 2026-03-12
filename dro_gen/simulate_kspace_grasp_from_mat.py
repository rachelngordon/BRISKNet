#!/usr/bin/env python3
"""Simulate k-space from a DRO .mat and reconstruct with GRASP.

Example:
  python simulate_kspace_grasp_from_mat.py --mat-path /path/to/sample_dro_8frames.mat --spokes-per-frame 36
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import torch
import torchkbnufft as tkbn
import sigpy as sp
import matplotlib.pyplot as plt
import torch.nn as nn
from einops import rearrange

try:
    import h5py
except ImportError as exc:
    raise ImportError("h5py is required to read .mat (v7.3) files.") from exc

from sigpy.mri import app


dtype = torch.complex64


class MCNUFFT(nn.Module):
    def __init__(self, nufft_ob, adjnufft_ob, ktraj, dcomp):
        super(MCNUFFT, self).__init__()
        self.nufft_ob = nufft_ob
        self.adjnufft_ob = adjnufft_ob
        self.ktraj = torch.squeeze(ktraj)
        self.dcomp = torch.squeeze(dcomp)

    def forward(self, inv, data, smaps):
        data = torch.squeeze(data)  # delete redundant dimension
        Nx = smaps.shape[2]
        Ny = smaps.shape[3]

        if inv:  # adjoint nufft

            smaps = smaps.to(dtype)

            if len(data.shape) > 2:  # multi-frame

                x = torch.zeros([Nx, Ny, data.shape[2]], dtype=dtype)

                for ii in range(0, data.shape[2]):
                    kd = data[:, :, ii]
                    k = self.ktraj[:, :, ii]
                    d = self.dcomp[:, ii]

                    kd = kd.unsqueeze(0)
                    d = d.unsqueeze(0).unsqueeze(0)

                    x_temp = self.adjnufft_ob(kd * d, k, smaps=smaps)
                    x[:, :, ii] = torch.squeeze(x_temp) / np.sqrt(Nx * Ny)

            else:  # single frame

                kd = data.unsqueeze(0)
                d = self.dcomp.unsqueeze(0).unsqueeze(0)
                x = self.adjnufft_ob(kd * d, self.ktraj, smaps=smaps)
                x = torch.squeeze(x) / np.sqrt(Nx * Ny)

        else:  # forward nufft

            if len(data.shape) > 2:  # multi-frame

                x = torch.zeros([smaps.shape[1], self.ktraj.shape[1], data.shape[-1]], dtype=dtype)

                for ii in range(0, data.shape[-1]):
                    image = data[:, :, ii]
                    k = self.ktraj[:, :, ii]

                    image = image.unsqueeze(0).unsqueeze(0)
                    x_temp = self.nufft_ob(image, k, smaps=smaps)
                    x[:, :, ii] = torch.squeeze(x_temp) / np.sqrt(Nx * Ny)

            else:  # single frame

                image = data.unsqueeze(0).unsqueeze(0)
                x = self.nufft_ob(image, self.ktraj, smaps=smaps)
                x = torch.squeeze(x) / np.sqrt(Nx * Ny)

        return x


def trajGR(Nkx, Nspokes):
    # golden-angle radial sampling trajectory
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
    oversample = 2
    im_size = (int(Nsample / oversample), int(Nsample / oversample))
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
        memtype.insert(name, type_id.get_member_offset(idx), h5py.h5t.NATIVE_FLOAT if np_fmt == np.float32 else h5py.h5t.NATIVE_DOUBLE)
        names.append(name_str)
        formats.append(np_fmt)
        offsets.append(type_id.get_member_offset(idx))
    arr = np.empty(dset.shape, dtype=np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": type_id.get_size()}))
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
        # Fallback for real/imag stored in last axis.
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
    dcf = (traj[..., 0]**2 + traj[..., 1]**2)**0.5

    F = sp.linop.NUFFT(ishape, traj)
    cim = F.H(ksp * dcf)
    cim = sp.fft(cim, axes=(-2, -1))

    mps = app.EspiritCalib(cim, device=device).run()
    mps = sp.to_device(mps, sp.cpu_device)
    return np.asarray(mps)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate k-space with torchkbnufft and run GRASP recon on a .mat image."
    )
    parser.add_argument(
        "--mat-path",
        default="/net/scratch2/rachelgordon/test_dro_144frames.mat",
        help="Path to .mat file containing (T,H,W) or (H,W,T) image array.",
    )
    parser.add_argument(
        "--key",
        default=None,
        help="Dataset key inside the .mat file (optional).",
    )
    parser.add_argument(
        "--spokes-per-frame",
        type=int,
        required=True,
        help="Number of spokes per frame.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Use only the first N frames (default: all frames).",
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
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for k-space simulation (e.g., cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--traj-method",
        default="get_traj",
        choices=("trajGR", "get_traj"),
        help="Trajectory method to use for k-space simulation.",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Output directory for PNG (and optional npy).",
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
        "--save-npy",
        action="store_true",
        help="Also save simulated k-space and recon as .npy files.",
    )
    args = parser.parse_args()

    with h5py.File(args.mat_path, "r") as f:
        img, img_key = _find_h5_dataset(f, args.key)
        smaps = _load_smaps(f)
    print(f"Using image dataset: {img_key}")

    if smaps.ndim != 3:
        raise ValueError(f"Expected smap shape (C,H,W) or (H,W,C), got {smaps.shape}")
    if smaps.shape[0] <= smaps.shape[1] and smaps.shape[0] <= smaps.shape[2]:
        smaps_chw = smaps
    else:
        smaps_chw = np.transpose(smaps, (2, 0, 1))

    h = int(smaps_chw.shape[1])
    w = int(smaps_chw.shape[2])

    img = _ensure_hwt(img, h, w)
    if args.num_frames is not None:
        img = img[..., : args.num_frames]

    h, w, t = img.shape
    if h != w:
        raise ValueError(f"Expected square frames, got H={h}, W={w}")

    device = torch.device(args.device)
    simImg_torch = torch.tensor(img).to(torch.cfloat).to(device)
    csmaps = torch.tensor(np.expand_dims(smaps_chw, axis=0), device=device)
    csmaps = csmaps.to(torch.complex64)
    print(simImg_torch.dtype)
    print(csmaps.dtype)

    nsample = h * 2
    ktraj, dcomp, nufft_ob, adjnufft_ob = prep_nufft(
        nsample, args.spokes_per_frame, t, traj_method=args.traj_method
    )

    physics = MCNUFFT(
        nufft_ob.to(device),
        adjnufft_ob.to(device),
        ktraj.to(device),
        dcomp.to(device),
    )

    kspace = physics(False, simImg_torch, csmaps)
    if args.kspace_noise_std < 0:
        raise ValueError("--kspace-noise-std must be >= 0.")
    if args.kspace_noise_std > 0:
        if args.kspace_noise_seed is not None:
            torch.manual_seed(args.kspace_noise_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.kspace_noise_seed)
        noise = torch.randn(
            kspace.shape, device=kspace.device, dtype=kspace.real.dtype
        )
        kspace = kspace + args.kspace_noise_std * noise
        print(f"Added k-space noise (std={args.kspace_noise_std}).")
    print(f"Simulated k-space shape: {tuple(kspace.shape)}")

    kspace_all = kspace.detach().cpu().numpy()
    kspace_all = kspace_all.reshape(
        kspace_all.shape[0], args.spokes_per_frame, nsample, t
    )
    kspace_all = np.transpose(kspace_all, (0, 3, 1, 2))  # (C, T, spf, sam)
    kspace_all = kspace_all.reshape(
        kspace_all.shape[0], t * args.spokes_per_frame, nsample
    )
    print(f"ESPIRiT kspace shape (C, spokes, sam): {kspace_all.shape}")
    print(f"ESPIRiT spokes used: {t * args.spokes_per_frame}")
    sigpy_device = sp.Device(0 if torch.cuda.is_available() else -1)
    csmaps_est = get_coil(kspace_all, t * args.spokes_per_frame, device=sigpy_device)
    csmaps_est = csmaps_est[:, None, :, :]

    kspace_np = (
        rearrange(kspace, "c (sp sam) t -> t c sp sam", sam=nsample)
        .unsqueeze(1)
        .unsqueeze(3)
        .cpu()
        .numpy()
    )
    print(f"kspace rearranged shape: {kspace_np.shape}")

    csmaps_np = csmaps_est
    print(f"csmaps shape: {csmaps_np.shape}")

    traj = get_traj(N_spokes=args.spokes_per_frame, N_time=t)
    print(f"traj shape: {traj.shape}")

    recon = app.HighDimensionalRecon(
        kspace_np,
        csmaps_np,
        combine_echo=False,
        lamda=args.lamda,
        coord=traj,
        regu="TV",
        regu_axes=[0],
        max_iter=args.max_iter,
        solver="ADMM",
        rho=args.rho,
        device=sigpy_device,
        show_pbar=False,
        verbose=False,
    ).run()

    recon_np = np.squeeze(recon.get())
    print(f"GRASP recon shape: {recon_np.shape}")

    if recon_np.ndim == 3 and recon_np.shape[0] == t:
        recon_thw = recon_np
    elif recon_np.ndim == 3 and recon_np.shape[-1] == t:
        recon_thw = np.transpose(recon_np, (2, 0, 1))
    else:
        recon_thw = recon_np
    first_frame = np.abs(recon_thw[0])

    os.makedirs(args.out_dir, exist_ok=True)
    out_png = os.path.join(
        args.out_dir,
        f"grasp_first_frame_spf{args.spokes_per_frame}_t{t}.png",
    )
    plt.figure(figsize=(4, 4))
    plt.imshow(first_frame, cmap="gray_r")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved PNG: {out_png}")

    if args.save_npy:
        out_kspace = os.path.join(
            args.out_dir, f"sim_kspace_spf{args.spokes_per_frame}_t{t}.npy"
        )
        out_recon = os.path.join(
            args.out_dir, f"grasp_recon_spf{args.spokes_per_frame}_t{t}.npy"
        )
        np.save(out_kspace, kspace.detach().cpu().numpy())
        np.save(out_recon, recon_np)
        print(f"Saved kspace: {out_kspace}")
        print(f"Saved recon: {out_recon}")


if __name__ == "__main__":
    main()
