import argparse
import csv
import glob
import math
import os
import time
import warnings
from typing import Tuple

import h5py
import numpy as np
import torch
import yaml
from einops import rearrange

from cluster_paths import apply_cluster_paths
from dataloader import SLICE_MAP_PATH, load_slice_map
from eval import (
    compute_ssdu_kspace_nmse,
    compute_ssdu_kspace_nmse_grasp,
    eval_grasp,
    eval_sample,
)
from model_factory import build_recon_model
from radial_lsfp import MCNUFFT
from utils import (
    GRASPRecon_from_ktraj,
    prep_nufft,
    remove_module_prefix,
    set_seed,
    sliding_window_inference,
)

# Silence torchmetrics/torch FutureWarning about torch.load(weights_only=...) defaults.
warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch\.load` with `weights_only=False`.*",
    category=FutureWarning,
)


def _torch_load_checkpoint(path: str, map_location="cpu"):
    """Load a checkpoint in the safest available way across torch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        return torch.load(path, map_location=map_location)


## COMMANDS
# downsample: python run_inference_single_fastmri.py --exp_dir output/ei_warp_large_Lkernel --fastmri_id 141 --use_downsampled_csmaps --disable_ssdu
# raw: python run_inference_single_fastmri.py --exp_dir output/ei_warp_large_Lkernel --fastmri_id 141 --raw_csmap_path /net/scratch2/rachelgordon/zf_data_192_slices/cs_maps/fastMRI_breast_141_2_cs_maps/cs_map_slice_125.npy --disable_ssdu

def _resolve_eval_params(config: dict, spokes: int, frames: int, phase_idx: int) -> Tuple[int, int]:
    """Pick evaluation spokes/frame and num_frames using overrides or curriculum."""
    if spokes and frames:
        return spokes, frames

    curriculum_cfg = config.get("training", {}).get("curriculum_learning", {})
    phases = curriculum_cfg.get("phases", [])
    if curriculum_cfg.get("enabled") and phases:
        # Default to the last phase unless the user specifies otherwise.
        phase_idx = len(phases) - 1 if phase_idx is None else phase_idx
        phase_idx = max(0, min(phase_idx, len(phases) - 1))
        phase = phases[phase_idx]
        return phase["eval_spokes_per_frame"], phase["eval_num_frames"]

    data_cfg = config["data"]
    return data_cfg["eval_spokes"], data_cfg["eval_timeframes"]


def _build_model(config: dict, device, block_dir: str):
    """Create model from config."""
    model = build_recon_model(config, device=device, block_dir=block_dir)
    model.eval()
    return model


def _load_weights(model, ckpt_path: str):
    ckpt = _torch_load_checkpoint(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(remove_module_prefix(state_dict))
    return model


def _load_dro_mapping(csv_path: str) -> dict:
    mapping = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dro_id = row.get("DRO")
            fastmri_id = row.get("fastMRIbreast")
            if not dro_id or not fastmri_id:
                continue
            mapping[int(fastmri_id)] = int(dro_id)
    return mapping


def _find_sample_dir(root_dir: str, num_frames: int, dro_id: int) -> str:
    dro_dir = os.path.join(root_dir, f"dro_{num_frames}frames")
    candidates = sorted(glob.glob(os.path.join(dro_dir, f"sample_{dro_id:03d}_*")))
    if not candidates:
        raise FileNotFoundError(
            f"No DRO sample found for DRO {dro_id:03d} in {dro_dir}."
        )
    if len(candidates) > 1:
        print(f"Found multiple DRO samples for {dro_id:03d}; using {candidates[0]}.")
    return candidates[0]


def _load_raw_csmaps(csmap_path: str) -> torch.Tensor:
    raw_csmaps = np.load(csmap_path)
    raw_csmaps_torch = torch.from_numpy(raw_csmaps)
    raw_csmaps_torch = rearrange(raw_csmaps_torch, "c b h w -> b c h w")
    raw_csmaps_torch = torch.rot90(raw_csmaps_torch, k=2, dims=[-2, -1])
    return raw_csmaps_torch


def _ensure_recon_layout(x: torch.Tensor, num_frames: int) -> torch.Tensor:
    if x.ndim == 5 and x.shape[-2] == num_frames and x.shape[-1] != num_frames:
        return x.transpose(-1, -2)
    return x


def _ensure_grasp_layout(grasp: torch.Tensor, num_frames: int, height: int, width: int) -> torch.Tensor:
    if grasp.ndim == 4 and grasp.shape[0] == 2:
        if grasp.shape[1:] == (num_frames, height, width):
            grasp = grasp.permute(0, 2, 1, 3)
        elif grasp.shape[1:] == (height, width, num_frames):
            grasp = grasp.permute(0, 1, 3, 2)
        grasp = grasp.unsqueeze(0)
    elif grasp.ndim == 5 and grasp.shape[1] == 2:
        if grasp.shape[2:] == (num_frames, height, width):
            grasp = grasp.permute(0, 1, 3, 2, 4)
        elif grasp.shape[2:] == (height, width, num_frames):
            grasp = grasp.permute(0, 1, 2, 4, 3)
    return grasp


def _load_sample_data(
    sample_dir: str,
    spokes_per_frame: int,
    num_frames: int,
    raw_kspace_root: str,
    dataset_key: str,
    fastmri_id: int,
    slice_map: dict,
    default_raw_slice_idx: int,
):
    csmaps = np.load(os.path.join(sample_dir, "csmaps.npy"))
    dro = np.load(os.path.join(sample_dir, "dro_ground_truth.npz"))
    grasp_path = os.path.join(sample_dir, f"grasp_spf{spokes_per_frame}_frames{num_frames}.npy")
    grasp_recon = np.load(grasp_path)

    grasp_recon_torch = torch.from_numpy(grasp_recon).permute(2, 0, 1)
    grasp_recon_torch = torch.stack([grasp_recon_torch.real, grasp_recon_torch.imag], dim=0)
    grasp_recon_torch = torch.flip(grasp_recon_torch, dims=[-3])
    grasp_recon_torch = torch.rot90(grasp_recon_torch, k=3, dims=[-3, -1])

    kspace_path = os.path.join(
        sample_dir, f"simulated_kspace_spf{spokes_per_frame}_frames{num_frames}.npy"
    )
    if not os.path.exists(kspace_path):
        raise FileNotFoundError(f"Missing simulated k-space: {kspace_path}")
    kspace_complex = np.load(kspace_path, allow_pickle=True)
    kspace_torch = torch.from_numpy(kspace_complex)

    patient_id = f"fastMRI_breast_{fastmri_id:03d}_2"
    slice_idx = slice_map.get(patient_id, default_raw_slice_idx)
    if slice_idx is None or slice_idx < 0:
        slice_idx = default_raw_slice_idx

    raw_grasp_path = os.path.join(
        os.path.dirname(raw_kspace_root),
        f"{patient_id}/grasp_recon_{spokes_per_frame}spf_{num_frames}frames_slice{slice_idx}.npy",
    )
    raw_kspace_path = os.path.join(raw_kspace_root, f"{patient_id}.h5")
    raw_csmap_path = os.path.join(
        os.path.dirname(raw_kspace_root),
        f"cs_maps/{patient_id}_cs_maps/cs_map_slice_{slice_idx:03d}.npy",
    )

    raw_csmaps = np.load(raw_csmap_path)
    raw_grasp_recon = np.load(raw_grasp_path).squeeze()

    raw_grasp_recon = torch.from_numpy(raw_grasp_recon).permute(2, 0, 1)
    raw_grasp_recon = torch.stack([raw_grasp_recon.real, raw_grasp_recon.imag], dim=0)
    raw_grasp_recon = torch.flip(raw_grasp_recon, dims=[-3])
    raw_grasp_recon = torch.rot90(raw_grasp_recon, k=1, dims=[-3, -1])

    with h5py.File(raw_kspace_path, "r") as f:
        raw_kspace_slice = torch.tensor(f[dataset_key][slice_idx])

    N_spokes_prep = num_frames * spokes_per_frame
    ksp_redu = raw_kspace_slice[:, :N_spokes_prep, :]
    ksp_prep = np.swapaxes(ksp_redu, 0, 1)
    ksp_prep_shape = ksp_prep.shape
    ksp_prep = np.reshape(
        ksp_prep, [num_frames, spokes_per_frame] + list(ksp_prep_shape[1:])
    )
    ksp_prep = torch.flip(ksp_prep, dims=[-1])
    raw_kspace_slice = rearrange(ksp_prep, "t sp c sam -> c (sp sam) t").to(kspace_torch.dtype)

    ground_truth_complex = dro["ground_truth_images"]
    tissue_names = [
        "glandular",
        "benign",
        "malignant",
        "muscle",
        "skin",
        "liver",
        "heart",
        "vascular",
    ]
    mask_dictionary = {}
    for tissue_name in tissue_names:
        if tissue_name in dro:
            mask_dictionary[tissue_name] = dro[tissue_name]

    ground_truth_torch = torch.from_numpy(ground_truth_complex).permute(2, 0, 1)
    ground_truth_torch = torch.stack([ground_truth_torch.real, ground_truth_torch.imag], dim=0)

    csmaps_torch = torch.from_numpy(csmaps).permute(2, 0, 1).unsqueeze(0)
    raw_csmaps_torch = torch.from_numpy(raw_csmaps)
    raw_csmaps_torch = rearrange(raw_csmaps_torch, "c b h w -> b c h w").to(csmaps_torch.dtype)
    raw_csmaps_torch = torch.rot90(raw_csmaps_torch, k=2, dims=[-2, -1])

    mask_torch = {}
    for key, val in mask_dictionary.items():
        mask_torch[key] = torch.from_numpy(val) if isinstance(val, np.ndarray) else val

    return (
        kspace_torch,
        csmaps_torch,
        ground_truth_torch,
        grasp_recon_torch,
        mask_torch,
        grasp_path,
        raw_kspace_slice,
        raw_grasp_recon,
        raw_csmaps_torch,
        slice_idx,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference for one fastMRI ID.")
    parser.add_argument("--exp_dir", required=True, help="Experiment directory location.")
    parser.add_argument("--fastmri_id", type=int, required=True, help="fastMRI breast ID (e.g., 141).")
    parser.add_argument(
        "--raw_csmap_path",
        help="Path to raw csmap .npy to use for DRO reconstruction.",
    )
    parser.add_argument(
        "--use_downsampled_csmaps",
        action="store_true",
        help=(
            "Use fastmri_csmaps_for_inference.npy from the current working "
            "directory instead of raw csmap paths."
        ),
    )
    parser.add_argument(
        "--raw_csmap_dir",
        help="Directory containing per-patient cs_maps/<patient>_cs_maps/ for optional construction.",
    )
    parser.add_argument(
        "--raw_csmap_slice",
        type=int,
        help="Slice index to construct a raw csmap path when --raw_csmap_dir is set.",
    )
    parser.add_argument("--config", help="Path to config.yaml (defaults to output/<exp>/config.yaml).")
    parser.add_argument("--checkpoint", help="Path to model checkpoint (defaults to output/<exp>/<exp>_model.pth).")
    parser.add_argument("--device", default=None, help="Torch device to use (default: config training.device).")
    parser.add_argument("--eval_spokes", type=int, help="Override spokes per frame for inference.")
    parser.add_argument("--eval_frames", type=int, help="Override number of frames for inference.")
    parser.add_argument("--phase_index", type=int, help="Curriculum phase index to use for eval params (default: last).")
    parser.add_argument("--disable_ssdu", action="store_true", help="Skip SSDU NMSE computation to speed up inference.")
    parser.add_argument("--raw_slice_idx", type=int, help="Override raw slice index for raw k-space/grasp.")
    parser.add_argument(
        "--sim_root",
        help="Override simulated dataset root (defaults to evaluation.simulated_dataset_path or /net/.../dro_dataset_frontpad).",
    )
    parser.add_argument("--seed", type=int, default=12, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    exp_name = args.exp_dir.split("/")[-1]
    config_path = args.config or os.path.join(args.exp_dir, "config.yaml")
    ckpt_path = args.checkpoint or os.path.join(args.exp_dir, f"{exp_name}_model.pth")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = apply_cluster_paths(config)

    device = torch.device(args.device or config["training"]["device"])
    rescale = config.get("evaluation", {}).get("rescale", True)
    raw_grasp_slice_idx = config.get("evaluation", {}).get("raw_grasp_slice_idx", 95)
    ei_cfg = config.get("model", {}).get("losses", {}).get("ei_loss", {})
    arrival_method = (ei_cfg.get("arrival_method", "threshold") or "threshold").lower()
    arrival_fraction = float(ei_cfg.get("arrival_fraction", 0.1))
    arrival_k = float(ei_cfg.get("arrival_shift_baseline_k", 2.0))
    if args.raw_slice_idx is not None:
        raw_grasp_slice_idx = args.raw_slice_idx

    simulated_root = (
        args.sim_root
        or config.get("evaluation", {}).get("simulated_dataset_path")
        or "/net/scratch2/rachelgordon/dro_dataset_frontpad"
    )

    dro_mapping = _load_dro_mapping(
        os.path.join(os.path.dirname(__file__), "data", "DROSubID_vs_fastMRIbreastID.csv")
    )
    if args.fastmri_id not in dro_mapping:
        raise ValueError(f"fastMRI ID {args.fastmri_id} not found in DRO mapping.")
    dro_id = dro_mapping[args.fastmri_id]

    N_spokes_eval, N_time_eval = _resolve_eval_params(
        config, spokes=args.eval_spokes, frames=args.eval_frames, phase_idx=args.phase_index
    )

    sample_dir = _find_sample_dir(simulated_root, N_time_eval, dro_id)

    data_dir = config["data"]["root_dir"]
    dataset_key = config["data"]["dataset_key"]
    cluster = config.get("experiment", {}).get("cluster", "Randi")

    slice_map = load_slice_map(SLICE_MAP_PATH)

    (
        dro_kspace,
        simulated_csmaps,
        ground_truth,
        dro_grasp_img,
        mask,
        grasp_path,
        raw_kspace,
        raw_grasp_img,
        raw_csmaps,
        raw_slice_idx,
    ) = _load_sample_data(
        sample_dir,
        N_spokes_eval,
        N_time_eval,
        data_dir,
        dataset_key,
        args.fastmri_id,
        slice_map,
        raw_grasp_slice_idx,
    )

    if args.use_downsampled_csmaps:
        dro_csmap_path = os.path.join(os.getcwd(), "fastmri_csmaps_for_inference.npy")
        if not os.path.exists(dro_csmap_path):
            raise FileNotFoundError(
                f"Downsampled csmaps not found at {dro_csmap_path}."
            )
    elif args.raw_csmap_path:
        dro_csmap_path = args.raw_csmap_path
    elif args.raw_csmap_dir and args.raw_csmap_slice is not None:
        patient_id = f"fastMRI_breast_{args.fastmri_id:03d}_2"
        dro_csmap_path = os.path.join(
            args.raw_csmap_dir,
            f"{patient_id}_cs_maps/cs_map_slice_{args.raw_csmap_slice:03d}.npy",
        )
    else:
        raise ValueError("Provide --raw_csmap_path or --raw_csmap_dir with --raw_csmap_slice.")

    dro_csmaps_override = _load_raw_csmaps(dro_csmap_path)

    output_dir = os.path.join(config["experiment"]["output_dir"], exp_name)
    inference_dir = os.path.join(
        output_dir, f"inference_fastmri_{args.fastmri_id:03d}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(inference_dir, exist_ok=True)

    N_samples = config["data"]["samples"]
    H, W = config["data"]["height"], config["data"]["width"]
    N_full = H * math.pi / 2

    eval_ktraj, eval_dcomp, eval_nufft_ob, eval_adjnufft_ob = prep_nufft(
        N_samples, N_spokes_eval, N_time_eval
    )
    eval_ktraj = eval_ktraj.to(device)
    eval_dcomp = eval_dcomp.to(device)
    eval_nufft_ob = eval_nufft_ob.to(device)
    eval_adjnufft_ob = eval_adjnufft_ob.to(device)
    eval_physics = MCNUFFT(eval_nufft_ob, eval_adjnufft_ob, eval_ktraj, eval_dcomp)

    eval_chunk_size = config.get("evaluation", {}).get("chunk_size", N_time_eval)
    eval_chunk_overlap = config.get("evaluation", {}).get("chunk_overlap", 0)
    compute_ssdu = config.get("evaluation", {}).get("compute_ssdu", True)
    if args.disable_ssdu:
        compute_ssdu = False

    ssdu_k_folds = config.get("evaluation", {}).get("ssdu_k_folds", 4)
    ssdu_grasp_k_folds = config.get("evaluation", {}).get("ssdu_grasp_k_folds", ssdu_k_folds)
    ssdu_weighting = config.get("evaluation", {}).get("ssdu_weighting", "sqrt_dcomp")

    block_dir = os.path.join(output_dir, "block_outputs")
    os.makedirs(block_dir, exist_ok=True)
    model = _build_model(config, device, block_dir)
    model = _load_weights(model, ckpt_path)

    acceleration_val = torch.tensor(
        [N_full / int(eval_ktraj.shape[1] / config["data"]["samples"])],
        dtype=torch.float,
        device=device,
    )

    dro_kspace = dro_kspace.to(device)
    simulated_csmaps = simulated_csmaps.squeeze(0).to(device)
    ground_truth = ground_truth.to(device)
    dro_grasp_img = dro_grasp_img.to(device)
    raw_kspace = raw_kspace.to(device)
    raw_grasp_img = raw_grasp_img.to(device)
    raw_csmaps = raw_csmaps.squeeze(0).to(device)
    dro_csmaps_override = dro_csmaps_override.squeeze(0).to(device).to(simulated_csmaps.dtype)

    acceleration_encoding = acceleration_val if config["model"]["encode_acceleration"] else None
    start_timepoint_index = (
        torch.tensor([0], dtype=torch.float, device=device)
        if config["model"]["encode_time_index"]
        else None
    )

    dro_csmaps_override = dro_csmaps_override.unsqueeze(0)
    raw_csmaps = raw_csmaps.unsqueeze(0)

    with torch.no_grad():
        if N_time_eval > eval_chunk_size:
            x_recon, _ = sliding_window_inference(
                H,
                W,
                N_time_eval,
                eval_ktraj,
                eval_dcomp,
                eval_nufft_ob,
                eval_adjnufft_ob,
                eval_chunk_size,
                eval_chunk_overlap,
                dro_kspace,
                dro_csmaps_override,
                acceleration_encoding,
                start_timepoint_index,
                model,
                epoch="inference",
                device=device,
                norm=config["model"]["norm"],
            )
            raw_x_recon, _ = sliding_window_inference(
                H,
                W,
                N_time_eval,
                eval_ktraj,
                eval_dcomp,
                eval_nufft_ob,
                eval_adjnufft_ob,
                eval_chunk_size,
                eval_chunk_overlap,
                raw_kspace,
                raw_csmaps,
                acceleration_encoding,
                start_timepoint_index,
                model,
                epoch="inference",
                device=device,
                norm=config["model"]["norm"],
            )
        else:

            print("dro_kspace: ", dro_kspace.shape)
            print("dro_csmaps_override: ", dro_csmaps_override.shape)
            x_recon, *_ = model(
                dro_kspace,
                eval_physics,
                dro_csmaps_override,
                acceleration_encoding,
                start_timepoint_index,
                epoch="inference",
                norm=config["model"]["norm"],
            )
            raw_x_recon, *_ = model(
                raw_kspace,
                eval_physics,
                raw_csmaps,
                acceleration_encoding,
                start_timepoint_index,
                epoch="inference",
                norm=config["model"]["norm"],
            )

    x_recon = _ensure_recon_layout(x_recon, N_time_eval)
    raw_x_recon = _ensure_recon_layout(raw_x_recon, N_time_eval)
    dro_grasp_img = _ensure_grasp_layout(dro_grasp_img, N_time_eval, H, W)
    raw_grasp_img = _ensure_grasp_layout(raw_grasp_img, N_time_eval, H, W)

    ssdu_result = {}
    ssdu_grasp_result = {}
    if compute_ssdu:
        ssdu_chunk_size = eval_chunk_size if N_time_eval > eval_chunk_size else None
        ssdu_result = compute_ssdu_kspace_nmse(
            model,
            raw_kspace,
            raw_csmaps,
            eval_ktraj,
            eval_dcomp,
            eval_nufft_ob,
            eval_adjnufft_ob,
            spokes_per_frame=int(N_spokes_eval),
            K_folds=ssdu_k_folds,
            baseline_weighting=ssdu_weighting,
            device=device,
            acceleration_encoding=acceleration_encoding,
            start_timepoint_index=start_timepoint_index,
            norm=config["model"]["norm"],
            epoch="inference",
            chunk_size=ssdu_chunk_size,
            chunk_overlap=eval_chunk_overlap,
        )
        ssdu_grasp_result = compute_ssdu_kspace_nmse_grasp(
            lambda y_used, ktraj_used, dcomp_used, csmap, samples_per_spoke: GRASPRecon_from_ktraj(
                csmap,
                y_used,
                ktraj_used,
                samples_per_spoke,
                device=None,
            ),
            raw_kspace,
            raw_csmaps,
            eval_ktraj,
            eval_dcomp,
            eval_nufft_ob,
            eval_adjnufft_ob,
            spokes_per_frame=int(N_spokes_eval),
            K_folds=ssdu_grasp_k_folds,
            orientation_transform="raw_grasp",
            baseline_weighting=ssdu_weighting,
            device=device,
        )

    label = f"fastmri_{args.fastmri_id:03d}"

    
    ground_truth = ground_truth.unsqueeze(0)
    simulated_csmaps = simulated_csmaps.unsqueeze(0)
    # dro_grasp_img = rearrange(dro_grasp_img, 'c h t w -> c h w t').unsqueeze(0)
    print("ground_truth: ", ground_truth.shape)
    print("dro_grasp_img: ", dro_grasp_img.shape)
    print("dro_csmaps_override: ", dro_csmaps_override.shape)
    print("simulated_csmaps: ", simulated_csmaps.shape)

    dro_metrics = eval_sample(
        dro_kspace,
        dro_csmaps_override,
        ground_truth,
        x_recon,
        eval_physics,
        mask,
        dro_grasp_img,
        acceleration_val,
        int(N_spokes_eval),
        inference_dir,
        label,
        device,
        cluster,
        dro_eval=True,
        grasp_path=grasp_path,
        rescale=rescale,
        filename_suffix="raw_csmaps",
        arrival_k=arrival_k,
        arrival_method=arrival_method,
        arrival_fraction=arrival_fraction,
    )

    grasp_metrics = eval_grasp(
        dro_kspace,
        simulated_csmaps,
        ground_truth,
        dro_grasp_img,
        eval_physics,
        device,
        inference_dir,
        dro_eval=True,
    )

    raw_dc_mse, raw_dc_mae, _ = eval_sample(
        raw_kspace,
        raw_csmaps,
        ground_truth,
        raw_x_recon,
        eval_physics,
        mask,
        raw_grasp_img,
        acceleration_val,
        int(N_spokes_eval),
        inference_dir,
        f"{label}_raw",
        device,
        cluster,
        dro_eval=False,
        grasp_path=grasp_path,
        raw_slice_idx=raw_slice_idx,
        rescale=rescale,
        arrival_k=arrival_k,
        arrival_method=arrival_method,
        arrival_fraction=arrival_fraction,
    )
    raw_grasp_dc_mse, raw_grasp_dc_mae = eval_grasp(
        raw_kspace,
        raw_csmaps,
        ground_truth,
        raw_grasp_img,
        eval_physics,
        device,
        inference_dir,
        dro_eval=False,
    )

    ssim, psnr, mse, lpips, dc_mse, dc_mae, recon_corr, grasp_corr, _ = dro_metrics
    grasp_ssim, grasp_psnr, grasp_mse, grasp_lpips, grasp_dc_mse, grasp_dc_mae = grasp_metrics

    metrics_path = os.path.join(inference_dir, "metrics.csv")
    with open(metrics_path, "w") as f:
        headers = [
            "sample",
            "dl_ssim",
            "dl_psnr",
            "dl_mse",
            "dl_lpips",
            "dl_dc_mse",
            "dl_dc_mae",
            "dl_recon_corr",
            "grasp_corr",
            "grasp_ssim",
            "grasp_psnr",
            "grasp_mse",
            "grasp_lpips",
            "grasp_dc_mse",
            "grasp_dc_mae",
            "raw_dc_mse",
            "raw_dc_mae",
            "raw_grasp_dc_mse",
            "raw_grasp_dc_mae",
            "raw_ssdu_nmse",
            "raw_grasp_ssdu_nmse",
        ]
        row = [
            label,
            f"{ssim:.6f}",
            f"{psnr:.6f}",
            f"{mse:.6f}",
            f"{lpips:.6f}",
            f"{dc_mse:.6f}",
            f"{dc_mae:.6f}",
            "" if recon_corr is None else f"{recon_corr:.6f}",
            "" if grasp_corr is None else f"{grasp_corr:.6f}",
            f"{grasp_ssim:.6f}",
            f"{grasp_psnr:.6f}",
            f"{grasp_mse:.6f}",
            f"{grasp_lpips:.6f}",
            f"{grasp_dc_mse:.6f}",
            f"{grasp_dc_mae:.6f}",
            f"{raw_dc_mse:.6f}",
            f"{raw_dc_mae:.6f}",
            f"{raw_grasp_dc_mse:.6f}",
            f"{raw_grasp_dc_mae:.6f}",
            "" if ssdu_result.get("ssdu_nmse_mean") is None else f"{ssdu_result['ssdu_nmse_mean']:.6f}",
            "" if ssdu_grasp_result.get("ssdu_nmse_mean") is None else f"{ssdu_grasp_result['ssdu_nmse_mean']:.6f}",
        ]
        f.write(",".join(headers) + "\n")
        f.write(",".join(row) + "\n")

    print(f"Saved DRO spatial comparison to {inference_dir}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
