import argparse
import json
import math
import os
import statistics
import time
from typing import Tuple

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from cluster_paths import apply_cluster_paths
from dataloader import SimulatedDataset
from eval import eval_grasp, eval_sample
from lsfpnet_encoding import ArtifactRemovalLSFPNet, LSFPNet
from radial_lsfp import MCNUFFT
from utils import (
    prep_nufft,
    remove_module_prefix,
    set_seed,
    sliding_window_inference,
)


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
    """Create the LSFP model and load weights."""
    initial_lambdas = {
        "lambda_L": config["model"]["lambda_L"],
        "lambda_S": config["model"]["lambda_S"],
        "lambda_spatial_L": config["model"]["lambda_spatial_L"],
        "lambda_spatial_S": config["model"]["lambda_spatial_S"],
        "gamma": config["model"]["gamma"],
        "lambda_step": config["model"]["lambda_step"],
    }

    lsfp_backbone = LSFPNet(
        LayerNo=config["model"]["num_layers"],
        lambdas=initial_lambdas,
        channels=config["model"]["channels"],
        style_dim=config["model"]["style_dim"],
        svd_mode=config["model"]["svd_mode"],
        use_lowk_dc=config["model"]["use_lowk_dc"],
        lowk_frac=config["model"]["lowk_frac"],
        lowk_alpha=config["model"]["lowk_alpha"],
        film_bounded=config["model"]["film_bounded"],
        film_gain=config["model"]["film_gain"],
        film_identity_init=config["model"]["film_identity_init"],
        svd_noise_std=config["model"]["svd_noise_std"],
        film_L=config["model"]["film_L"],
        kernel_size_L=config["model"].get("kernel_size_L", 3),
        kernel_size_S=config["model"].get("kernel_size_S", 3),
        activation_checkpointing=config["model"].get("activation_checkpointing", False),
        checkpoint_use_reentrant=config["model"].get("checkpoint_use_reentrant", False),
    )

    if config["model"]["encode_acceleration"] and config["model"]["encode_time_index"]:
        channels = 2
    else:
        channels = 1

    model = ArtifactRemovalLSFPNet(lsfp_backbone, block_dir, channels=channels).to(device)
    model.eval()
    return model


def _load_weights(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(remove_module_prefix(state_dict))
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on validation samples.")
    parser.add_argument("--exp_dir", required=True, help="Experiment directory location.")
    parser.add_argument("--config", help="Path to config.yaml (defaults to output/<exp>/config.yaml).")
    parser.add_argument("--checkpoint", help="Path to model checkpoint (defaults to output/<exp>/<exp>_model.pth).")
    parser.add_argument("--num_samples", type=int, help="Number of validation samples to evaluate (default: config value).")
    parser.add_argument("--device", default=None, help="Torch device to use (default: config training.device).")
    parser.add_argument("--eval_spokes", type=int, help="Override spokes per frame for inference.")
    parser.add_argument("--eval_frames", type=int, help="Override number of frames for inference.")
    parser.add_argument("--phase_index", type=int, help="Curriculum phase index to use for eval params (default: last).")
    parser.add_argument("--seed", type=int, default=12, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    exp_name = args.exp_dir.split('/')[-1]

    # Resolve config/checkpoint paths and load config.
    config_path = args.config or os.path.join(args.exp_dir, "config.yaml")
    ckpt_path = args.checkpoint or os.path.join(args.exp_dir, f"{exp_name}_model.pth")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = apply_cluster_paths(config)

    device = torch.device(args.device or config["training"]["device"])
    rescale = config.get("evaluation", {}).get("rescale", True)
    raw_grasp_slice_idx = config.get("evaluation", {}).get("raw_grasp_slice_idx", 95)
    cluster = config.get("experiment", {}).get("cluster", "Randi")

    # Where to save inference outputs.
    output_dir = os.path.join(config["experiment"]["output_dir"], exp_name)
    inference_dir = os.path.join(output_dir, f"inference_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(inference_dir, exist_ok=True)

    # Dataset setup.
    with open(config["data"]["split_file"], "r") as fp:
        splits = json.load(fp)

    val_ids = splits.get("val_dro") or splits.get("val") or []

    N_spokes_eval, N_time_eval = _resolve_eval_params(
        config, spokes=args.eval_spokes, frames=args.eval_frames, phase_idx=args.phase_index
    )

    data_dir = config["data"]["root_dir"]
    model_type = config["model"]["name"]

    val_dataset = SimulatedDataset(
        root_dir=config["evaluation"]["simulated_dataset_path"],
        raw_kspace_path=data_dir,
        model_type=model_type,
        patient_ids=val_ids,
        dataset_key=config["data"]["dataset_key"],
        spokes_per_frame=N_spokes_eval,
        num_frames=N_time_eval,
        grasp_slice_idx=raw_grasp_slice_idx,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["dataloader"]["batch_size"],
        shuffle=False,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=True,
    )

    num_samples = args.num_samples or config.get("evaluation", {}).get("num_samples", len(val_dataset))
    num_samples = min(num_samples, len(val_dataset))

    # Prep physics for inference.
    N_samples = config["data"]["samples"]
    H, W = config["data"]["height"], config["data"]["width"]
    N_full = H * math.pi / 2

    eval_ktraj, eval_dcomp, eval_nufft_ob, eval_adjnufft_ob = prep_nufft(N_samples, N_spokes_eval, N_time_eval)
    eval_ktraj = eval_ktraj.to(device)
    eval_dcomp = eval_dcomp.to(device)
    eval_nufft_ob = eval_nufft_ob.to(device)
    eval_adjnufft_ob = eval_adjnufft_ob.to(device)
    eval_physics = MCNUFFT(eval_nufft_ob, eval_adjnufft_ob, eval_ktraj, eval_dcomp)

    eval_chunk_size = config.get("evaluation", {}).get("chunk_size", N_time_eval)
    eval_chunk_overlap = config.get("evaluation", {}).get("chunk_overlap", 0)

    # Build and load model.
    block_dir = os.path.join(output_dir, "block_outputs")
    os.makedirs(block_dir, exist_ok=True)
    model = _build_model(config, device, block_dir)
    model = _load_weights(model, ckpt_path)

    acceleration_val = torch.tensor([N_full / int(eval_ktraj.shape[1] / config["data"]["samples"])], dtype=torch.float, device=device)

    results = []
    raw_results = []
    grasp_results = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader, total=num_samples, desc="Inference on validation")):
            if idx >= num_samples:
                break

            (
                dro_kspace,
                csmap,
                ground_truth,
                dro_grasp_img,
                mask,
                grasp_path,
                raw_kspace,
                raw_grasp_img,
                raw_csmaps,
            ) = batch

            csmap = csmap.squeeze(0).to(device)
            ground_truth = ground_truth.to(device)
            dro_grasp_img = dro_grasp_img.to(device)
            dro_kspace = dro_kspace.squeeze(0).to(device)
            raw_kspace = raw_kspace.squeeze(0).to(device)
            raw_grasp_img = raw_grasp_img.to(device)
            raw_csmaps = raw_csmaps.squeeze(0).to(device)

            acceleration_encoding = acceleration_val if config["model"]["encode_acceleration"] else None
            start_timepoint_index = torch.tensor([0], dtype=torch.float, device=device) if config["model"]["encode_time_index"] else None

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
                    csmap,
                    acceleration_encoding,
                    start_timepoint_index,
                    model,
                    epoch="inference",
                    device=device,
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
                )
            else:
                x_recon, *_ = model(
                    dro_kspace, eval_physics, csmap, acceleration_encoding, start_timepoint_index, epoch="inference", norm=config["model"]["norm"]
                )
                raw_x_recon, *_ = model(
                    raw_kspace, eval_physics, raw_csmaps, acceleration_encoding, start_timepoint_index, epoch="inference", norm=config["model"]["norm"]
                )

            sample_dir = os.path.join(inference_dir, f"sample_{idx:02d}")
            os.makedirs(sample_dir, exist_ok=True)
            label = f"sample{idx:02d}"

            dro_metrics = eval_sample(
                dro_kspace,
                csmap,
                ground_truth,
                x_recon,
                eval_physics,
                mask,
                dro_grasp_img,
                acceleration_val,
                int(N_spokes_eval),
                sample_dir,
                label,
                device,
                cluster,
                dro_eval=True,
                grasp_path=grasp_path,
                rescale=rescale,
            )

            grasp_metrics = eval_grasp(
                dro_kspace,
                csmap,
                ground_truth,
                dro_grasp_img,
                eval_physics,
                device,
                sample_dir,
                dro_eval=True,
            )

            raw_dc_mse, raw_dc_mae = eval_sample(
                raw_kspace,
                raw_csmaps,
                ground_truth,
                raw_x_recon,
                eval_physics,
                mask,
                raw_grasp_img,
                acceleration_val,
                int(N_spokes_eval),
                sample_dir,
                f"{label}_raw",
                device,
                cluster,
                dro_eval=False,
                grasp_path=grasp_path,
                raw_slice_idx=raw_grasp_slice_idx,
                rescale=rescale,
            )

            ssim, psnr, mse, lpips, dc_mse, dc_mae, recon_corr, grasp_corr = dro_metrics
            grasp_ssim, grasp_psnr, grasp_mse, grasp_lpips, grasp_dc_mse, grasp_dc_mae = grasp_metrics

            results.append(
                dict(
                    sample=label,
                    ssim=ssim,
                    psnr=psnr,
                    mse=mse,
                    lpips=lpips,
                    dc_mse=dc_mse,
                    dc_mae=dc_mae,
                    recon_corr=recon_corr,
                    grasp_corr=grasp_corr,
                )
            )
            grasp_results.append(
                dict(
                    sample=label,
                    ssim=grasp_ssim,
                    psnr=grasp_psnr,
                    mse=grasp_mse,
                    lpips=grasp_lpips,
                    dc_mse=grasp_dc_mse,
                    dc_mae=grasp_dc_mae,
                )
            )
            raw_results.append(dict(sample=label, raw_dc_mse=raw_dc_mse, raw_dc_mae=raw_dc_mae))

    # Save metrics.
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
        ]
        f.write(",".join(headers) + "\n")
        for dro_row, grasp_row, raw_row in zip(results, grasp_results, raw_results):
            row = [
                dro_row["sample"],
                f"{dro_row['ssim']:.6f}",
                f"{dro_row['psnr']:.6f}",
                f"{dro_row['mse']:.6f}",
                f"{dro_row['lpips']:.6f}",
                f"{dro_row['dc_mse']:.6f}",
                f"{dro_row['dc_mae']:.6f}",
                "" if dro_row["recon_corr"] is None else f"{dro_row['recon_corr']:.6f}",
                "" if dro_row["grasp_corr"] is None else f"{dro_row['grasp_corr']:.6f}",
                f"{grasp_row['ssim']:.6f}",
                f"{grasp_row['psnr']:.6f}",
                f"{grasp_row['mse']:.6f}",
                f"{grasp_row['lpips']:.6f}",
                f"{grasp_row['dc_mse']:.6f}",
                f"{grasp_row['dc_mae']:.6f}",
                f"{raw_row['raw_dc_mse']:.6f}",
                f"{raw_row['raw_dc_mae']:.6f}",
            ]
            f.write(",".join(row) + "\n")

    def _mean_std(values, key):
        vals = [v[key] for v in values if v[key] is not None]
        if not vals:
            return None, None
        mean = sum(vals) / len(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return mean, std

    dl_summary = {
        "ssim": _mean_std(results, "ssim"),
        "psnr": _mean_std(results, "psnr"),
        "mse": _mean_std(results, "mse"),
        "lpips": _mean_std(results, "lpips"),
        "dc_mse": _mean_std(results, "dc_mse"),
        "dc_mae": _mean_std(results, "dc_mae"),
        "recon_corr": _mean_std(results, "recon_corr"),
        "grasp_corr": _mean_std(results, "grasp_corr"),
    }

    grasp_summary = {
        "ssim": _mean_std(grasp_results, "ssim"),
        "psnr": _mean_std(grasp_results, "psnr"),
        "mse": _mean_std(grasp_results, "mse"),
        "lpips": _mean_std(grasp_results, "lpips"),
        "dc_mse": _mean_std(grasp_results, "dc_mse"),
        "dc_mae": _mean_std(grasp_results, "dc_mae"),
    }

    def _format_mean_std(mean, std):
        if mean is None:
            return ""
        return f"{mean:.4f} ± {std:.4f}"

    recon_corr_str = _format_mean_std(*dl_summary["recon_corr"])
    grasp_corr_str = _format_mean_std(*dl_summary["grasp_corr"])

    print("=== Inference Summary (averaged over samples) ===")
    print(
        "DL   -> "
        f"SSIM: {_format_mean_std(*dl_summary['ssim'])}, "
        f"PSNR: {_format_mean_std(*dl_summary['psnr'])}, "
        f"MSE: {_format_mean_std(*dl_summary['mse'])}, "
        f"LPIPS: {_format_mean_std(*dl_summary['lpips'])}, "
        f"DC_MSE: {_format_mean_std(*dl_summary['dc_mse'])}, "
        f"DC_MAE: {_format_mean_std(*dl_summary['dc_mae'])}, "
        f"EC Corr (DL): {recon_corr_str}, "
        f"EC Corr (GRASP): {grasp_corr_str}"
    )
    print(
        "GRASP-> "
        f"SSIM: {_format_mean_std(*grasp_summary['ssim'])}, "
        f"PSNR: {_format_mean_std(*grasp_summary['psnr'])}, "
        f"MSE: {_format_mean_std(*grasp_summary['mse'])}, "
        f"LPIPS: {_format_mean_std(*grasp_summary['lpips'])}, "
        f"DC_MSE: {_format_mean_std(*grasp_summary['dc_mse'])}, "
        f"DC_MAE: {_format_mean_std(*grasp_summary['dc_mae'])}"
    )
    print(f"Inference complete. Results saved to {inference_dir}")


if __name__ == "__main__":
    main()
