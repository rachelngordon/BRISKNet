#!/usr/bin/env python3
"""
Run GRASP reconstructions for DRO variable-frame samples using precomputed k-space
and ESPIRiT csmaps.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import sigpy as sp
from sigpy.mri import app


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GRASP reconstructions using precomputed k-space and ESPIRiT csmaps."
    )
    parser.add_argument(
        "--dro-root",
        default="/net/scratch2/rachelgordon/dro_var_frames",
        help="Root directory containing sample_*_dro_*frames.mat files and kspace .npy.",
    )
    parser.add_argument(
        "--csmaps-dir",
        default=None,
        help="Directory containing csmaps_<sample_id>.npy (default: <dro_root>/csmaps_espirit).",
    )
    parser.add_argument(
        "--sigpy-device",
        choices=("cpu", "cuda"),
        default="cuda",
        help="Device for GRASP recon.",
    )
    parser.add_argument(
        "--traj-method",
        default="get_traj",
        choices=("trajGR", "get_traj"),
        help="Trajectory method to use for GRASP recon.",
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
    sigpy_device = _parse_sigpy_device(args.sigpy_device)
    suffix = _normalize_suffix(args.suffix)
    csmaps_dir = args.csmaps_dir or os.path.join(args.dro_root, "csmaps_espirit")
    recon_dir = os.path.join(args.dro_root, f"espirit_grasp_recons_lam{args.lamda:g}")
    os.makedirs(recon_dir, exist_ok=True)

    sample_map = _collect_dro_files(args.dro_root)
    sample_ids = sorted(
        sample_map,
        key=lambda s: (
            int(re.search(r"sample_(\d+)", s).group(1))
            if re.search(r"sample_(\d+)", s)
            else s
        ),
    )

    for sample_id in sample_ids:
        csmaps_path = os.path.join(csmaps_dir, f"csmaps_{sample_id}{suffix}.npy")
        if not os.path.exists(csmaps_path):
            raise FileNotFoundError(f"Missing csmaps file: {csmaps_path}")
        csmaps_np = np.load(csmaps_path)

        available_frames = sample_map[sample_id]
        ordered_frames = [f for f in FRAME_ORDER if f in available_frames]
        if not ordered_frames:
            ordered_frames = sorted(available_frames)

        for frames in ordered_frames:
            if frames not in FRAME_TO_SPF:
                print(f"[skip] {sample_id} frames={frames}: no SPF mapping available.")
                continue
            spf = FRAME_TO_SPF[frames]

            kspace_path = os.path.join(
                args.dro_root,
                f"{sample_id}_kspace_{spf}spf_{frames}frames{suffix}.npy",
            )
            recon_path = os.path.join(
                recon_dir,
                f"grasp_{sample_id}_{spf}spf_{frames}frames{suffix}.npy",
            )

            if os.path.exists(recon_path):
                print(f"[skip] {sample_id} frames={frames} (recon exists)")
                continue
            if not os.path.exists(kspace_path):
                print(f"[skip] {sample_id} frames={frames} (missing kspace)")
                continue

            print(f"[run] {sample_id} frames={frames} spf={spf}")
            kspace_np = np.load(kspace_path)
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
