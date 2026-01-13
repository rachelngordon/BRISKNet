#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from einops import rearrange
import matplotlib.pyplot as plt

from radial_lsfp import MCNUFFT, to_torch_complex
import torchkbnufft as tkbn

def trajGR(Nkx, Nspokes, flip_traj):
    '''
    function for generating golden-angle radial sampling trajectory
    :param Nkx: spoke length
    :param Nspokes: number of spokes
    :return: ktraj: golden-angle radial sampling trajectory
    '''
    # ga = np.deg2rad(180 / ((np.sqrt(5) + 1) / 2))
    ga = np.pi * ((1 - np.sqrt(5)) / 2)
    kx = np.zeros(shape=(Nkx, Nspokes))
    ky = np.zeros(shape=(Nkx, Nspokes))
    ky[:, 0] = np.linspace(-np.pi, np.pi, Nkx)
    for i in range(1, Nspokes):
        kx[:, i] = np.cos(ga) * kx[:, i - 1] - np.sin(ga) * ky[:, i - 1]
        ky[:, i] = np.sin(ga) * kx[:, i - 1] + np.cos(ga) * ky[:, i - 1]
    ky = np.transpose(ky)
    kx = np.transpose(kx)

    if flip_traj:
        ky = np.flip(ky, axis=[-1])
        kx = np.flip(kx, axis=[-1])

    ktraj = np.stack((ky.flatten(), kx.flatten()), axis=0)

    # print(f"------ k-space trajectory shape: {ktraj.shape} ------")

    return ktraj


def prep_nufft(Nsample, Nspokes, Ng, flip_traj):

    overSmaple = 2
    im_size = (int(Nsample/overSmaple), int(Nsample/overSmaple))
    grid_size = (Nsample, Nsample)

    ktraj = trajGR(Nsample, Nspokes * Ng, flip_traj)

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

    nufft_ob = tkbn.KbNufft(im_size=im_size, grid_size=grid_size)  # forward nufft
    adjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size, grid_size=grid_size)  # backward nufft

    return ktraju, dcompu, nufft_ob, adjnufft_ob



def load_csmap(root_dir: str, patient_id: str, slice_idx: int) -> np.ndarray:
    csmap_root = os.path.join(os.path.dirname(root_dir), "cs_maps")
    csmap_path = os.path.join(
        csmap_root, f"{patient_id}_cs_maps", f"cs_map_slice_{slice_idx:03d}.npy"
    )
    if not os.path.exists(csmap_path):
        raise FileNotFoundError(f"CSMap not found: {csmap_path}")
    csmap = np.load(csmap_path)
    return csmap.squeeze()


def load_kspace_slice(kspace_path: str, dataset_key: str, slice_idx: int) -> np.ndarray:
    with h5py.File(kspace_path, "r") as f:
        if dataset_key not in f:
            raise KeyError(f"Dataset key '{dataset_key}' not found in {kspace_path}")
        kspace_slice = f[dataset_key][slice_idx]
    return np.asarray(kspace_slice)


def prep_kspace(
    kspace_slice: np.ndarray, spokes_per_frame: int, do_flip: bool
) -> tuple[torch.Tensor, int, int]:
    n_coils, n_spokes, n_samples = kspace_slice.shape
    n_time = n_spokes // spokes_per_frame
    n_spokes_prep = n_time * spokes_per_frame

    ksp_redu = kspace_slice[:, :n_spokes_prep, :]
    ksp_prep = np.swapaxes(ksp_redu, 0, 1)  # (spokes, coils, samples)
    ksp_prep = np.reshape(
        ksp_prep, [n_time, spokes_per_frame, n_coils, n_samples]
    )  # (t, sp, c, sam)
    ksp_prep = rearrange(ksp_prep, "t sp c sam -> t c sp sam")

    real_part = ksp_prep.real
    imag_part = ksp_prep.imag
    kspace_final = torch.stack(
        [torch.from_numpy(real_part), torch.from_numpy(imag_part)], dim=0
    ).float()

    if do_flip:
        kspace_final = torch.flip(kspace_final, dims=[-1])

    return kspace_final, n_samples, n_time


def run_zf_recon(kspace_final: torch.Tensor, csmap: torch.Tensor, device: str, flip_traj: bool):
    kspace_complex = to_torch_complex(kspace_final.unsqueeze(0)).squeeze(0)
    kspace_complex = rearrange(kspace_complex, "t c sp sam -> c (sp sam) t")
    kspace_complex = kspace_complex.to(device)

    csmap = csmap.to(device).to(kspace_complex.dtype)
    n_samples = kspace_final.shape[-1]
    spokes_per_frame = kspace_final.shape[-2]
    n_time = kspace_final.shape[1]

    ktraj, dcomp, nufft_ob, adjnufft_ob = prep_nufft(
        n_samples, spokes_per_frame, n_time, flip_traj
    )

    if flip_traj:
        print("ktraj: ", ktraj.shape)
        print("dcomp: ", dcomp.shape)

        kspace_final = torch.flip(kspace_final, dims=[-1])

    physics = MCNUFFT(
        nufft_ob.to(device),
        adjnufft_ob.to(device),
        ktraj.to(device),
        dcomp.to(device),
    )

    with torch.no_grad():
        x_zf = physics(inv=True, data=kspace_complex, smaps=csmap.unsqueeze(0))

    return x_zf


def main():
    parser = argparse.ArgumentParser(
        description="Run ZF recon with/without k-space flip from the dataloader."
    )
    parser.add_argument("--root-dir", type=str, default="/net/scratch2/rachelgordon/zf_data_192_slices/zf_kspace")
    parser.add_argument("--patient-id", type=str, required=True)
    parser.add_argument("--slice-idx", type=int, default=41)
    parser.add_argument("--dataset-key", type=str, default="kspace")
    parser.add_argument("--spokes-per-frame", type=int, default=36)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="output/zf_flip_compare")
    parser.add_argument("--flip-traj", type=bool, default=False)
    args = parser.parse_args()

    kspace_path = os.path.join(args.root_dir, f"{args.patient_id}.h5")
    if not os.path.exists(kspace_path):
        raise FileNotFoundError(f"k-space file not found: {kspace_path}")

    kspace_slice = load_kspace_slice(kspace_path, args.dataset_key, args.slice_idx)
    csmap_np = load_csmap(args.root_dir, args.patient_id, args.slice_idx)

    csmap_t = torch.from_numpy(csmap_np)
    csmap_noflip = torch.rot90(csmap_t, k=2, dims=[-2, -1])
    csmap_flip = csmap_t

    kspace_flip, _, _ = prep_kspace(kspace_slice, args.spokes_per_frame, do_flip=True)
    kspace_noflip, _, _ = prep_kspace(kspace_slice, args.spokes_per_frame, do_flip=False)

    x_zf_flip = run_zf_recon(kspace_flip, csmap_flip, args.device, args.flip_traj)
    x_zf_noflip = run_zf_recon(kspace_noflip, csmap_noflip, args.device, args.flip_traj)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mag_flip = torch.abs(x_zf_flip).cpu().numpy()
    mag_noflip = torch.abs(x_zf_noflip).cpu().numpy()
    diff = np.abs(mag_flip - mag_noflip)

    def save_png(img: np.ndarray, out_path: Path):
        plt.figure(figsize=(5, 5))
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close()

    t0 = 0
    save_png(mag_flip[..., t0], out_dir / "zf_with_flip_t0.png")
    save_png(mag_noflip[..., t0], out_dir / "zf_without_flip_t0.png")
    save_png(diff[..., t0], out_dir / "zf_diff_t0.png")

    if mag_flip.shape[-1] > 1:
        save_png(mag_flip.mean(axis=-1), out_dir / "zf_with_flip_mean.png")
        save_png(mag_noflip.mean(axis=-1), out_dir / "zf_without_flip_mean.png")
        save_png(diff.mean(axis=-1), out_dir / "zf_diff_mean.png")

    print(f"Saved outputs to: {out_dir}")
    print(f"Mean abs magnitude diff: {np.mean(diff):.6e}")


if __name__ == "__main__":
    main()
