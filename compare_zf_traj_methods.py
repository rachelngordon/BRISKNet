#!/usr/bin/env python3
import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from einops import rearrange
from torch.utils.data import DataLoader

from cluster_paths import apply_cluster_paths
from dataloader import ZFSliceDataset
from process_data.nifti.dce_recon import get_traj as get_traj_dce
from radial_lsfp import MCNUFFT
from utils import set_seed, to_torch_complex, trajGR
import torchkbnufft as tkbn


def _build_train_dataset(config, data_dir, cluster, flip_kspace):
    split_file = config["data"]["split_file"]
    max_subjects = config["dataloader"]["max_subjects"]
    N_time = config["data"]["timeframes"]
    N_coils = config["data"]["coils"]
    train_spokes_per_frame = config["data"]["train_spokes_per_frame"]
    curriculum_enabled = config.get("training", {}).get("curriculum_learning", {}).get("enabled", False)
    curriculum_phases = config.get("training", {}).get("curriculum_learning", {}).get("phases", [])

    initial_train_spokes_range = [8, 16, 24, 36]
    if curriculum_enabled and curriculum_phases:
        initial_train_spokes_range = curriculum_phases[0]["train_spokes_range"]

    with open(split_file, "r") as fp:
        splits = json.load(fp)

    if max_subjects < 300:
        max_train = int(max_subjects * (1 - config["data"]["val_split_ratio"]))
        train_patient_ids = splits["train"][:max_train]
    else:
        train_patient_ids = splits["train"]

    if config["dataloader"]["slice_range_start"] == "None" or config["dataloader"]["slice_range_end"] == "None":
        slice_idx = config["dataloader"]["slice_idx"]
    else:
        slice_idx = range(
            config["dataloader"]["slice_range_start"],
            config["dataloader"]["slice_range_end"],
        )

    return ZFSliceDataset(
        root_dir=data_dir,
        patient_ids=train_patient_ids,
        dataset_key=config["data"]["dataset_key"],
        file_pattern="*.h5",
        slice_idx=slice_idx,
        num_random_slices=config["dataloader"].get("num_random_slices", None),
        N_time=N_time,
        N_coils=N_coils,
        spf_aug=config["data"]["spf_aug"],
        spokes_per_frame=train_spokes_per_frame,
        weight_accelerations=config["data"]["weight_accelerations"],
        initial_spokes_range=initial_train_spokes_range,
        cluster=cluster,
        flip_kspace=flip_kspace,
    )


# def _ktraj_and_dcomp_from_get_traj(spokes_per_frame, samples_per_spoke, num_frames):
#     base_res = int(samples_per_spoke // 2)

#     class _Args:
#         pass

#     args = _Args()
#     args.spokes_per_frame = spokes_per_frame

#     traj = get_traj_dce(args, csmaps=False, N_spokes=spokes_per_frame, N_time=num_frames, base_res=base_res, gind=1)
#     traj = np.asarray(traj)
#     if traj.ndim == 3:
#         traj = traj[None, ...]

#     expected_shape = (num_frames, spokes_per_frame, samples_per_spoke, 2)
#     if traj.shape != expected_shape:
#         raise ValueError(f"get_traj returned {traj.shape}, expected {expected_shape}")

#     dcf = np.sqrt(traj[..., 0] ** 2 + traj[..., 1] ** 2)  # (T, Sp, Sam)

#     traj_flat = traj.reshape(num_frames, spokes_per_frame * samples_per_spoke, 2)
#     ktraj = np.transpose(traj_flat, (2, 1, 0))  # (2, M, T)
#     dcomp = dcf.reshape(num_frames, spokes_per_frame * samples_per_spoke).T  # (M, T)

#     ktraj = torch.tensor(ktraj, dtype=torch.float)
#     dcomp = torch.tensor(dcomp, dtype=torch.complex64)
#     return ktraj, dcomp

def _ktraj_and_dcomp_from_get_traj(spokes_per_frame, samples_per_spoke, num_frames, im_size):
    base_res = int(samples_per_spoke // 2)

    class _Args: pass
    args = _Args()
    args.spokes_per_frame = spokes_per_frame

    traj = get_traj_dce(args, csmaps=False, N_spokes=spokes_per_frame,
                       N_time=num_frames, base_res=base_res, gind=1)
    traj = np.asarray(traj)
    if traj.ndim == 3:
        traj = traj[None, ...]

    # traj: (T, Sp, Sam, 2)
    traj_flat = traj.reshape(num_frames, spokes_per_frame * samples_per_spoke, 2)
    ktraj = np.transpose(traj_flat, (2, 1, 0))  # (2, M, T)
    ktraj = torch.tensor(ktraj, dtype=torch.float)

    # normalized -> radians for torchkbnufft
    # ktraj = ktraj * (2 * np.pi)
    ktraj = ktraj * (2*np.pi / base_res)


    # tkbn DCF expects (2, M) per frame
    dcomps = []
    for t in range(num_frames):
        d = tkbn.calc_density_compensation_function(
            ktraj=ktraj[:, :, t], im_size=im_size
        ).squeeze()
        dcomps.append(d)

    dcomp = torch.stack(dcomps, dim=-1)  # (M, T), real
    return ktraj, dcomp



def _prep_nufft_local(Nsample, Nspokes, Ng):
    overSmaple = 2
    im_size = (int(Nsample / overSmaple), int(Nsample / overSmaple))
    grid_size = (Nsample, Nsample)

    ktraj = trajGR(Nsample, Nspokes * Ng)
    ktraj = torch.tensor(ktraj, dtype=torch.float)
    dcomp = tkbn.calc_density_compensation_function(ktraj=ktraj, im_size=im_size).squeeze()

    ktraju = np.zeros([2, Nspokes * Nsample, Ng], dtype=float)
    dcompu = np.zeros([Nspokes * Nsample, Ng], dtype=complex)

    for ii in range(0, Ng):
        start = ii * Nspokes * Nsample
        end = (ii + 1) * Nspokes * Nsample
        ktraju[:, :, ii] = ktraj[:, start:end]
        dcompu[:, ii] = dcomp[start:end]

    ktraju = torch.tensor(ktraju, dtype=torch.float)
    dcompu = torch.tensor(dcompu, dtype=torch.complex64)

    nufft_ob = tkbn.KbNufft(im_size=im_size, grid_size=grid_size)
    adjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size, grid_size=grid_size)

    return ktraju, dcompu, nufft_ob, adjnufft_ob


def _compute_zf_image(measured_kspace, csmap, nufft_ob, adjnufft_ob, ktraj, dcomp):
    physics = MCNUFFT(nufft_ob, adjnufft_ob, ktraj, dcomp)
    zf = physics(inv=True, data=measured_kspace, smaps=csmap)
    return zf


def _plot_traj_compare(ktraj_get, ktraj_gr, out_path):
    # ktraj_*: (2, M, T)
    traj_get_xy = ktraj_get[:, :, 0].T
    traj_gr_xy = ktraj_gr[:, :, 0].T

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(traj_get_xy[:, 0], traj_get_xy[:, 1], ",", alpha=0.6)
    axes[0].set_title("Trajectory (get_traj)")
    axes[0].set_aspect("equal", "box")
    axes[0].axis("off")

    axes[1].plot(traj_gr_xy[:, 0], traj_gr_xy[:, 1], ",", alpha=0.6)
    axes[1].set_title("Trajectory (trajGR)")
    axes[1].set_aspect("equal", "box")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare ZF recon using get_traj vs trajGR.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.")
    parser.add_argument("--device", type=str, default=None, help="Override device.")
    parser.add_argument("--seed", type=int, default=12, help="Random seed.")
    parser.add_argument("--frame-idx", type=int, default=0, help="Frame index to plot.")
    parser.add_argument("--out", type=str, default="zf_traj_compare.png", help="Output image path.")
    parser.add_argument("--traj-out", type=str, default="traj_compare.png", help="Output trajectory image path.")
    args = parser.parse_args()

    set_seed(args.seed)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config = apply_cluster_paths(config)
    device = torch.device(args.device or config["training"]["device"])

    data_dir = config["data"]["root_dir"]
    cluster = config.get("experiment", {}).get("cluster", "Randi")
    flip_kspace = config.get("data", {}).get("flip_kspace", False)

    dataset = _build_train_dataset(config, data_dir, cluster, flip_kspace)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    batch = next(iter(loader))
    measured_kspace, csmap, N_samples, N_spokes, N_time = batch

    measured_kspace = to_torch_complex(measured_kspace).squeeze(0)
    measured_kspace = rearrange(measured_kspace, "t co sp sam -> co (sp sam) t")
    measured_kspace = measured_kspace.to(device)

    csmap = csmap.to(device).to(measured_kspace.dtype)

    # trajGR path
    ktraj_gr, dcomp_gr, nufft_ob, adjnufft_ob = _prep_nufft_local(
        int(N_samples), int(N_spokes), int(N_time)
    )
    ktraj_gr = ktraj_gr.to(device)
    dcomp_gr = dcomp_gr.to(device)
    nufft_ob = nufft_ob.to(device)
    adjnufft_ob = adjnufft_ob.to(device)

    # get_traj path (DCF consistent with process_data/nifti/dce_recon.py)
    # ktraj_gt, dcomp_gt = _ktraj_and_dcomp_from_get_traj(
    #     int(N_spokes), int(N_samples), int(N_time)
    # )
    im_size = (int(N_samples / 2), int(N_samples / 2))  # same as your _prep_nufft_local
    ktraj_gt, dcomp_gt = _ktraj_and_dcomp_from_get_traj(int(N_spokes), int(N_samples), int(N_time), im_size)

    ktraj_gt = ktraj_gt.to(device)
    dcomp_gt = dcomp_gt.to(device)


    zf_get_traj = _compute_zf_image(
        measured_kspace, csmap, nufft_ob, adjnufft_ob, ktraj_gt, dcomp_gt
    )
    zf_traj_gr = _compute_zf_image(
        measured_kspace, csmap, nufft_ob, adjnufft_ob, ktraj_gr, dcomp_gr
    )

    frame_idx = max(0, min(int(args.frame_idx), zf_get_traj.shape[-1] - 1))
    img_get = torch.abs(zf_get_traj[..., frame_idx]).detach().cpu().numpy()
    img_gr = torch.abs(zf_traj_gr[..., frame_idx]).detach().cpu().numpy()

    vmin = np.percentile(np.concatenate([img_get.ravel(), img_gr.ravel()]), 1)
    vmax = np.percentile(np.concatenate([img_get.ravel(), img_gr.ravel()]), 99.5)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_get, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title("ZF (get_traj + DCF)")
    axes[0].axis("off")

    axes[1].imshow(img_gr, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].set_title("ZF (trajGR)")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")

    _plot_traj_compare(
        ktraj_gt.detach().cpu().numpy(),
        ktraj_gr.detach().cpu().numpy(),
        args.traj_out,
    )


if __name__ == "__main__":
    main()
