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
    generate_sliding_window_indices,
    prep_nufft,
    remove_module_prefix,
    set_seed,
)


def _temporal_window(length: int, kind: str = "hann", device=None, dtype=torch.float32):
    """Create a 1D temporal window for overlap-add stitching."""
    if length <= 1:
        w = torch.ones(1, device=device, dtype=dtype)
    elif kind == "hann":
        w = torch.hann_window(length, periodic=False, dtype=dtype, device=device)
        w = torch.clamp(w, min=1e-3)
    elif kind == "box":
        w = torch.ones(length, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported window kind: {kind}")
    w = w * (length / torch.sum(w))
    return w.view(1, 1, 1, 1, length)


def _pad_temporal(x: torch.Tensor, pad_left: int, pad_right: int, mode: str) -> torch.Tensor:
    """Pad the temporal (last) dimension with reflect or replicate padding."""
    if pad_left == 0 and pad_right == 0:
        return x
    if pad_left < 0 or pad_right < 0:
        raise ValueError("pad_left and pad_right must be non-negative.")
    t = x.shape[-1]
    if t == 0:
        raise ValueError("Cannot pad an empty temporal dimension.")
    if mode == "reflect":
        if pad_left >= t or pad_right >= t:
            raise ValueError("Reflect padding requires pad size < number of frames.")
        parts = []
        if pad_left:
            parts.append(x[..., 1:pad_left + 1].flip(-1))
        parts.append(x)
        if pad_right:
            parts.append(x[..., -pad_right - 1:-1].flip(-1))
        return torch.cat(parts, dim=-1)
    if mode == "replicate":
        parts = []
        if pad_left:
            parts.append(x[..., :1].expand(*x.shape[:-1], pad_left))
        parts.append(x)
        if pad_right:
            parts.append(x[..., -1:].expand(*x.shape[:-1], pad_right))
        return torch.cat(parts, dim=-1)
    raise ValueError(f"Unsupported pad mode: {mode}")


@torch.no_grad()
def sliding_window_inference_padded(
    H, W, N_frames,
    ktraj, dcomp, nufft_ob, adjnufft_ob,
    chunk_size, chunk_overlap,
    kspace, csmap,
    acceleration_encoding,
    start_timepoint_index,
    model, epoch, device,
    pad_size: int = 0,
    pad_mode: str = "reflect",
    window_kind: str = "hann",
    offset: int = 0,
):
    """
    Sliding window inference with temporal padding to improve edge-frame context.
    """
    pad_left = pad_right = int(pad_size)
    if pad_left:
        kspace = _pad_temporal(kspace, pad_left, pad_right, pad_mode)
        ktraj = _pad_temporal(ktraj, pad_left, pad_right, pad_mode)
        dcomp = _pad_temporal(dcomp, pad_left, pad_right, pad_mode)
    N_frames_pad = N_frames + pad_left + pad_right

    step_size = max(1, chunk_size - chunk_overlap)
    offset = int(offset)
    if offset:
        offset = offset % step_size

    chunk_indices = generate_sliding_window_indices(N_frames_pad, chunk_size, chunk_overlap)
    if offset:
        chunk_indices = [(s + offset, e + offset) for s, e in chunk_indices]
        chunk_indices = [(s, e) for s, e in chunk_indices if s < N_frames_pad]
        chunk_indices = [(s, min(e, N_frames_pad)) for s, e in chunk_indices if e > s]

    stitched_recon = torch.zeros(1, 2, H, W, N_frames_pad, device=device, dtype=torch.float32)
    weight_sum = torch.zeros(1, 1, 1, 1, N_frames_pad, device=device, dtype=torch.float32)

    csmap = csmap.to(device)
    adj_losses = []

    for i, (start_idx, end_idx) in enumerate(chunk_indices):
        print(f"Processing chunk {i+1}/{len(chunk_indices)}: frames {start_idx}-{end_idx}")

        kspace_chunk = kspace[..., start_idx:end_idx].to(device)
        ktraj_chunk = ktraj[..., start_idx:end_idx].to(device)
        dcomp_chunk = dcomp[..., start_idx:end_idx].to(device)

        physics_chunk = MCNUFFT(nufft_ob, adjnufft_ob, ktraj_chunk, dcomp_chunk)

        sti = None
        if start_timepoint_index is not None:
            sti_val = start_idx - pad_left
            sti_val = max(0, min(N_frames - 1, sti_val))
            sti = torch.tensor([sti_val], dtype=torch.float32, device=device)

        x_recon_chunk, adj_loss, *_ = model(
            kspace_chunk, physics_chunk, csmap, acceleration_encoding, sti, epoch=epoch, norm="both"
        )
        adj_losses.append(adj_loss.item())

        T_chunk = end_idx - start_idx
        w = _temporal_window(T_chunk, kind=window_kind, device=device, dtype=x_recon_chunk.dtype)

        stitched_recon[..., start_idx:end_idx] += x_recon_chunk * w
        weight_sum[..., start_idx:end_idx] += w

        del kspace_chunk, ktraj_chunk, dcomp_chunk, physics_chunk, x_recon_chunk, w
        torch.cuda.empty_cache()

    stitched_recon = stitched_recon / torch.clamp(weight_sum, min=1e-8)
    if pad_left:
        stitched_recon = stitched_recon[..., pad_left:pad_left + N_frames]

    mean_adj_loss = float(torch.tensor(adj_losses).mean().item()) if adj_losses else 0.0
    return stitched_recon, mean_adj_loss


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
    parser = argparse.ArgumentParser(description="Run inference on validation samples with temporal padding.")
    parser.add_argument("--exp_dir", required=True, help="Experiment directory location.")
    parser.add_argument("--config", help="Path to config.yaml (defaults to output/<exp>/config.yaml).")
    parser.add_argument("--checkpoint", help="Path to model checkpoint (defaults to output/<exp>/<exp>_model.pth).")
    parser.add_argument("--num_samples", type=int, help="Number of validation samples to evaluate (default: config value).")
    parser.add_argument("--device", default=None, help="Torch device to use (default: config training.device).")
    parser.add_argument("--eval_spokes", type=int, help="Override spokes per frame for inference.")
    parser.add_argument("--eval_frames", type=int, help="Override number of frames for inference.")
    parser.add_argument("--phase_index", type=int, help="Curriculum phase index to use for eval params (default: last).")
    parser.add_argument("--pad_size", type=int, default=None, help="Temporal padding size for sliding window inference.")
    parser.add_argument("--pad_mode", choices=["reflect", "replicate"], default="reflect", help="Temporal padding mode.")
    parser.add_argument("--double_pass", action="store_true", help="Enable a second offset pass and average results.")
    parser.add_argument("--offset", type=int, default=None, help="Temporal offset for the second pass (default: half step).")
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
    eval_pad_size = args.pad_size
    if eval_pad_size is None:
        eval_pad_size = config.get("evaluation", {}).get("pad_size", eval_chunk_overlap)
    eval_step = max(1, eval_chunk_size - eval_chunk_overlap)
    eval_offset = args.offset if args.offset is not None else (eval_step // 2)

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
                x_recon, _ = sliding_window_inference_padded(
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
                    pad_size=eval_pad_size,
                    pad_mode=args.pad_mode,
                )
                raw_x_recon, _ = sliding_window_inference_padded(
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
                    pad_size=eval_pad_size,
                    pad_mode=args.pad_mode,
                )
                if args.double_pass:
                    x_recon_2, _ = sliding_window_inference_padded(
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
                        pad_size=eval_pad_size,
                        pad_mode=args.pad_mode,
                        offset=eval_offset,
                    )
                    raw_x_recon_2, _ = sliding_window_inference_padded(
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
                        pad_size=eval_pad_size,
                        pad_mode=args.pad_mode,
                        offset=eval_offset,
                    )
                    x_recon = 0.5 * (x_recon + x_recon_2)
                    raw_x_recon = 0.5 * (raw_x_recon + raw_x_recon_2)
            else:
                x_recon, *_ = model(
                    dro_kspace, eval_physics, csmap, acceleration_encoding, start_timepoint_index, epoch="inference", norm=config["model"]["norm"]
                )
                raw_x_recon, *_ = model(
                    raw_kspace, eval_physics, raw_csmaps, acceleration_encoding, start_timepoint_index, epoch="inference", norm=config["model"]["norm"]
                )

            # Align raw recon orientation to match training eval.
            raw_x_recon = torch.rot90(raw_x_recon, k=2, dims=[-3, -2])

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
