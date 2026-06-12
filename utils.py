"""Shared utilities for training, inference, and evaluation. Run: imported by other scripts (not intended to run directly)."""

import csv
import os
import random
import subprocess
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
import torch
import torchkbnufft as tkbn
from einops import rearrange
from sigpy.mri import app
from torch.nn.parallel import DistributedDataParallel as DDP

from model.radial import MCNUFFT
from model.transform import estimate_bolus_arrival_index


def _torch_load_checkpoint(path: str, map_location="cpu"):
    """Load a checkpoint in the safest available way across torch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        # Torch >=2.6 defaults to weights_only=True. Fall back to full checkpoint
        # load for trusted local experiment files.
        return torch.load(path, map_location=map_location, weights_only=False)


def save_csmap_png(csmap, output_dir, tag, max_coils=16, cmap="viridis"):
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(csmap, torch.Tensor):
        data = csmap.detach().cpu()
        if data.dim() >= 4:
            data = data[0]
        if data.dim() == 2:
            data = data.unsqueeze(0)
        if data.dim() == 3 and data.shape[0] > 64 and data.shape[-1] <= 32:
            data = data.movedim(-1, 0)
        mag = torch.abs(data) if data.is_complex() else torch.abs(data)
        mag = mag.numpy()
    else:
        data = np.asarray(csmap)
        if data.ndim >= 4:
            data = data[0]
        if data.ndim == 2:
            data = data[None, ...]
        if data.ndim == 3 and data.shape[0] > 64 and data.shape[-1] <= 32:
            data = np.moveaxis(data, -1, 0)
        mag = np.abs(data)

    num_coils = min(mag.shape[0], max_coils)
    ncols = int(np.ceil(np.sqrt(num_coils)))
    nrows = int(np.ceil(num_coils / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
    vmax = np.max(mag[:num_coils]) if num_coils > 0 else 1.0

    for idx in range(nrows * ncols):
        ax = axes[idx // ncols][idx % ncols]
        if idx < num_coils:
            ax.imshow(mag[idx], cmap=cmap, vmin=0.0, vmax=vmax)
            ax.set_title(f"Coil {idx + 1}")
        ax.axis("off")

    fig.suptitle(f"{tag} CSMaps (|S|)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{tag}_csmaps.png"), dpi=150)
    plt.close(fig)

def log_gradient_stats(model, epoch, iteration, output_dir, log_filename="gradient_stats.csv"):
    """
    Computes, prints, and logs the L2 norm of gradients for each parameter and the total gradient norm.
    
    Args:
        model (torch.nn.Module): The model being trained.
        epoch (int): The current epoch.
        iteration (int): The current global iteration/step count.
        output_dir (str): The main experiment output directory.
        log_filename (str): The CSV filename for storing detailed logs.
    """
    total_norm = 0.0
    param_norms = []
    
    # Iterate over all named parameters
    for name, p in model.named_parameters():
        if p.grad is not None and p.requires_grad:
            # Calculate the L2 norm of the gradient for this parameter
            param_norm = p.grad.data.norm(2)
            # Handle potential inf/nan values gracefully
            if not torch.isfinite(param_norm):
                param_norm_item = float('inf')
            else:
                param_norm_item = param_norm.item()
            
            param_norms.append((name, param_norm_item))
            total_norm += param_norm_item ** 2
            
    total_norm = total_norm ** 0.5
    
    # --- Logging to Console ---
    print(f"--- Gradient Stats (Epoch {epoch}, Iter {iteration}) ---")
    
    # Use :.4e for exponential notation with 4 digits of precision
    print(f"Total Gradient Norm: {total_norm:.4e}")
    
    # Sort parameters by gradient norm (descending) to see the largest ones
    # Use a lambda that is safe for potential 'inf' values
    param_norms.sort(key=lambda x: x[1], reverse=True)
    
    print("Top 5 layers with largest gradients:")
    for name, norm in param_norms[:5]:
        # Use :.4e here as well
        print(f"  - {name}: {norm:.4e}")
        
    print("Top 5 layers with smallest gradients:")
    # To print the smallest non-zero gradients, we filter out zeros
    non_zero_norms = [p for p in param_norms if p[1] > 0]
    for name, norm in non_zero_norms[-5:]:
        # Use :.4e here as well
        print(f"  - {name}: {norm:.4e}")
    print("-------------------------------------------------")

    # --- Logging to CSV File for later analysis ---
    # No changes needed here, as CSV should ideally store full precision numbers.
    # The exponential formatting is mainly for console readability.
    log_path = os.path.join(output_dir, log_filename)
    file_exists = os.path.isfile(log_path)
    
    with open(log_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['epoch', 'iteration', 'total_norm', 'param_name', 'param_norm'])
            
        # The writer will handle inf/nan correctly
        writer.writerow([epoch, iteration, total_norm, '---TOTAL---', total_norm])
        for name, norm in param_norms:
            writer.writerow([epoch, iteration, total_norm, name, norm])


def log_lsfpnet_component_grads(model, epoch, iteration, output_dir, log_filename="lsfpnet_component_grads.csv"):
    """
    Aggregate gradient norms for key LSFPNet components and log them.

    Components:
      - low_rank_component: params controlling low-rank proximal branch (lambdas, FiLM heads)
      - sparse_dynamic_component: params for sparse branch lambdas/FiLM
      - forward_cnn_L/backward_cnn_L: convs enforcing sparsity for L
      - forward_cnn_S/backward_cnn_S: convs enforcing sparsity for S
    """
    if isinstance(model, DDP):
        model = model.module

    component_stats = {}

    def _accumulate(component_name: str, grad_norm: float):
        if component_name not in component_stats:
            component_stats[component_name] = {"sum_sq": 0.0, "count": 0}
        component_stats[component_name]["sum_sq"] += grad_norm ** 2
        component_stats[component_name]["count"] += 1

    def _component_from_name(name: str):
        lname = name.lower()
        if "forward_l" in lname:
            return "forward_cnn_L"
        if "backward_l" in lname:
            return "backward_cnn_L"
        if "forward_s" in lname:
            return "forward_cnn_S"
        if "backward_s" in lname:
            return "backward_cnn_S"
        if "lambda_l" in lname or "spatial_l" in lname or "style_injector_l" in lname:
            return "low_rank_component"
        if "lambda_s" in lname or "spatial_s" in lname or "style_injector_s" in lname:
            return "sparse_dynamic_component"
        return None

    for name, param in model.named_parameters():
        if param.grad is None or not param.requires_grad:
            continue
        component = _component_from_name(name)
        if component is None:
            continue
        grad_norm = param.grad.data.norm(2)
        if torch.isfinite(grad_norm):
            _accumulate(component, grad_norm.item())

    if not component_stats:
        return

    print(f"--- LSFPNet Component Gradients (Epoch {epoch}, Iter {iteration}) ---")
    for comp, stats in sorted(component_stats.items()):
        comp_norm = (stats["sum_sq"] ** 0.5)
        print(f"  - {comp}: {comp_norm:.4e} ({stats['count']} params)")

    log_path = os.path.join(output_dir, log_filename)
    file_exists = os.path.isfile(log_path)
    with open(log_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["epoch", "iteration", "component", "grad_norm", "param_count"])
        for comp, stats in sorted(component_stats.items()):
            comp_norm = (stats["sum_sq"] ** 0.5)
            writer.writerow([epoch, iteration, comp, comp_norm, stats["count"]])



def trajGR(Nkx, Nspokes):
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

    ktraj = np.stack((ky.flatten(), kx.flatten()), axis=0)

    # print(f"------ k-space trajectory shape: {ktraj.shape} ------")

    return ktraj


def get_traj(N_spokes=13, N_time=1, base_res=320, gind=1):

    N_tot_spokes = N_spokes * N_time

    N_samples = base_res * 2

    base_lin = np.arange(N_samples).reshape(1, -1) - base_res

    tau = 0.5 * (1 + 5**0.5)
    base_rad = np.pi / (gind + tau - 1)

    base_rot = np.arange(N_tot_spokes).reshape(-1, 1) * base_rad

    traj = np.zeros((N_tot_spokes, N_samples, 2))
    traj[..., 0] = np.cos(base_rot) @ base_lin
    traj[..., 1] = np.sin(base_rot) @ base_lin

    traj = traj / 2

    traj = traj.reshape(N_time, N_spokes, N_samples, 2)

    return np.squeeze(traj)
    

def _ktraj_and_dcomp_from_get_traj(Nsample, Nspokes, Ng, im_size):
    base_res = int(Nsample // 2)

    class _Args:
        pass

    args = _Args()
    args.spokes_per_frame = Nspokes

    from data.dce_recon import get_traj as get_traj_dce

    traj = get_traj_dce(
        args,
        csmaps=False,
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


################### prepare NUFFT ################
def prep_nufft(Nsample, Nspokes, Ng, traj_method="trajGR"):

    overSmaple = 2
    im_size = (int(Nsample/overSmaple), int(Nsample/overSmaple))
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

    nufft_ob = tkbn.KbNufft(im_size=im_size, grid_size=grid_size)  # forward nufft
    adjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size, grid_size=grid_size)  # backward nufft

    return ktraju, dcompu, nufft_ob, adjnufft_ob


def _calculate_top_percentile_curve(dynamic_slice: torch.Tensor, percentile: float) -> list[float]:
    """Helper function to calculate the enhancement curve for a single dynamic slice."""
    
    if dynamic_slice.dim() != 5 or dynamic_slice.shape[0] != 1 or dynamic_slice.shape[1] != 2:
        raise ValueError(f"Expected input shape (1, 2, T, H, W), but got {dynamic_slice.shape}")

    # Calculate magnitude: sqrt(real^2 + imag^2) and remove batch/channel dims
    magnitude_video = torch.sqrt(dynamic_slice[:, 0, ...] ** 2 + dynamic_slice[:, 1, ...] ** 2).squeeze(0)
    
    num_time_frames = magnitude_video.shape[0]
    top_percentile_means = []
    
    q = percentile / 100.0

    for t in range(num_time_frames):
        frame_t = magnitude_video[t, :, :]
        
        if frame_t.max() == 0:
            top_percentile_means.append(0)
            continue
            
        threshold = torch.quantile(frame_t.flatten(), q)
        bright_pixels = frame_t[frame_t > threshold]
        
        mean_val = torch.mean(bright_pixels) if bright_pixels.numel() > 0 else threshold
        top_percentile_means.append(mean_val.item())
        
    return top_percentile_means


def plot_enhancement_curve(
    model_output: torch.Tensor,
    percentile: float = 99.0,
    title: str = "Enhancement Curve Comparison",
    output_filename: str = None,
    show_arrival: bool = False,
    arrival_percentile: float = 0.95,
    arrival_baseline_k: float = 2.0,
    arrival_method: str = "threshold",
    arrival_fraction: float = 0.1,
    arrival_pre_contrast_baseline: str = "n_frames",
    arrival_baseline_seconds: float = 20.0,
    arrival_total_seconds: float = 150.0,
):
    """
    Calculates and plots the enhancement curves for a model output and a benchmark
    image on the same graph for direct comparison.

    Args:
        model_output (torch.Tensor): The model's reconstructed dynamic slice.
                                     Shape (1, 2, T, H, W).
        benchmark_image (torch.Tensor): The ground truth or benchmark dynamic slice.
                                        Shape (1, 2, T, H, W).
        percentile (float, optional): The percentile for defining the brightest pixels.
                                      Defaults to 99.0.
        title (str, optional): The title for the plot. Defaults to "Enhancement Curve Comparison".
        output_filename (str, optional): If provided, saves the plot to this file path.
                                         Defaults to None (displays plot).
    """

    # --- 1. Input Validation ---
    if not 0 < percentile < 100:
        raise ValueError("Percentile must be between 0 and 100.")

    # --- 2. Calculate Curves for Both Images ---
    model_curve = _calculate_top_percentile_curve(model_output.detach(), percentile)
    # benchmark_curve = _calculate_top_percentile_curve(benchmark_image.detach(), percentile)
    
    # Ensure time axis is consistent
    num_time_frames = model_output.shape[2]
    time_axis = np.arange(num_time_frames)

    # --- 3. Plotting ---
    plt.figure(figsize=(12, 7))
    
    # Plot model output curve
    plt.plot(time_axis, model_curve, label='Model Output', marker='o', linestyle='-', color='tab:blue')

    if show_arrival:
        arrival_idx = estimate_bolus_arrival_index(
            model_output.detach(),
            percentile=arrival_percentile,
            baseline_k=arrival_baseline_k,
            arrival_method=arrival_method,
            arrival_fraction=arrival_fraction,
            pre_contrast_baseline=arrival_pre_contrast_baseline,
            baseline_seconds=arrival_baseline_seconds,
            total_seconds=arrival_total_seconds,
        )
        if arrival_idx is not None:
            arrival_time = time_axis[arrival_idx]
            plt.axvline(arrival_time, color='tab:red', linestyle='--', linewidth=1.5, label=f'Arrival t={arrival_time}')
            plt.scatter([arrival_time], [model_curve[arrival_idx]], color='tab:red', zorder=5)

    # Plot benchmark curve
    # plt.plot(time_axis, benchmark_curve, label='GRASP Benchmark', marker='x', linestyle='--', color='tab:orange')
    
    plt.title(title, fontsize=16)
    plt.xlabel("Time Frame", fontsize=12)
    plt.ylabel(f"Mean Signal of Top {100-percentile:.1f}% Pixels", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if output_filename:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        plt.savefig(output_filename)
    else:
        plt.show()
        
    plt.close()


def _ensure_b2hwt(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure a complex 2-channel video tensor is shaped (B, 2, H, W, T).

    Accepts:
      - (B, 2, H, W, T)  [preferred in this repo]
      - (B, 2, T, H, W)  [some plotting utilities]
      - (2, H, W, T) / (2, T, H, W) [no batch]
    """
    if not torch.is_tensor(x):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")

    if x.dim() == 4:
        x = x.unsqueeze(0)

    if x.dim() != 5 or x.shape[1] != 2:
        raise ValueError(f"Expected shape (B, 2, ..., ..., ...), got {tuple(x.shape)}")

    # Decide which of dims {2,3,4} is time by assuming time is the smallest.
    d2, d3, d4 = int(x.shape[2]), int(x.shape[3]), int(x.shape[4])
    dims = np.array([d2, d3, d4], dtype=np.int64)
    time_axis = int(np.argmin(dims))

    if time_axis == 0:
        # (B,2,T,H,W) -> (B,2,H,W,T)
        return rearrange(x, "b c t h w -> b c h w t")
    if time_axis == 1:
        # (B,2,H,T,W) -> (B,2,H,W,T)
        return rearrange(x, "b c h t w -> b c h w t")
    return x


def _downsample_time_mean_b2hwt(x: torch.Tensor, factor: int) -> torch.Tensor:
    if factor <= 1:
        return x
    B, C, H, W, T = x.shape
    T_lo = T // factor
    if T_lo <= 0:
        return x[:, :, :, :, :0]
    x_cropped = x[:, :, :, :, : T_lo * factor]
    x_grouped = x_cropped.view(B, C, H, W, T_lo, factor)
    return x_grouped.mean(dim=-1)


def _masked_mean_curve_mag_b2hwt(x: torch.Tensor, mask_hw: torch.Tensor) -> torch.Tensor:
    x = _ensure_b2hwt(x)
    if mask_hw.dim() != 2:
        raise ValueError(f"Expected mask with shape (H,W), got {tuple(mask_hw.shape)}")
    if mask_hw.shape[0] != x.shape[2] or mask_hw.shape[1] != x.shape[3]:
        raise ValueError(
            f"Mask shape {tuple(mask_hw.shape)} does not match spatial dims {(int(x.shape[2]), int(x.shape[3]))}"
        )

    # mag: (B,H,W,T)
    mag = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
    mag0 = mag[0].reshape(-1, mag.shape[-1])
    mask_flat = mask_hw.reshape(-1).to(dtype=torch.bool, device=mag0.device)
    if int(mask_flat.sum().item()) <= 0:
        return torch.zeros((mag0.shape[-1],), dtype=mag0.dtype, device=mag0.device)
    return mag0[mask_flat].mean(dim=0)


def plot_rebin_consistency_diagnostic(
    x_hi: torch.Tensor,
    x_lo: torch.Tensor,
    factor: int,
    output_filename: str,
    *,
    x_hi_down: Optional[torch.Tensor] = None,
    mask_fraction: float = 0.01,
    min_pixels: int = 256,
    baseline_frames: int = 4,
    title: Optional[str] = None,
):
    """
    Save a simple, training-safe diagnostic plot for the rebin branch:
      - dynamic mask from temporal std of |x_hi|
      - baseline-subtracted masked mean curves for x_hi, Avg_factor(x_hi), and x_lo

    This is meant to run once at startup (or on a checkpoint) to sanity-check whether
    the rebin branch is learning dynamics or collapsing to a flat curve.
    """
    factor = max(1, int(factor))
    if factor <= 1:
        raise ValueError("factor must be > 1 for a rebin diagnostic.")

    if not (0.0 < float(mask_fraction) <= 1.0):
        raise ValueError("mask_fraction must be in (0, 1].")

    x_hi_b2hwt = _ensure_b2hwt(x_hi.detach())
    x_lo_b2hwt = _ensure_b2hwt(x_lo.detach())
    if x_hi_down is None:
        x_hi_down_b2hwt = _downsample_time_mean_b2hwt(x_hi_b2hwt, factor)
    else:
        x_hi_down_b2hwt = _ensure_b2hwt(x_hi_down.detach())

    if x_hi_down_b2hwt.shape[-1] != x_lo_b2hwt.shape[-1]:
        raise ValueError(
            "x_hi_down and x_lo must have the same number of frames. "
            f"Got {int(x_hi_down_b2hwt.shape[-1])} vs {int(x_lo_b2hwt.shape[-1])}."
        )

    # Compute a dynamic-region mask from temporal std of |x_hi|.
    mag_hi = torch.sqrt(x_hi_b2hwt[:, 0] ** 2 + x_hi_b2hwt[:, 1] ** 2)  # (B,H,W,T)
    std_map = mag_hi[0].std(dim=-1)  # (H,W)
    std_flat = std_map.reshape(-1)
    numel = int(std_flat.numel())
    # min_pixels is deprecated for diagnostics; fraction controls mask size.
    target_pixels = int(round(float(mask_fraction) * numel))
    target_pixels = min(max(target_pixels, 1), numel)
    topk_idx = torch.topk(std_flat, k=target_pixels, largest=True, sorted=False).indices
    mask_flat = torch.zeros_like(std_flat, dtype=torch.bool)
    mask_flat[topk_idx] = True
    mask_hw = mask_flat.view_as(std_map)

    # Curves.
    curve_hi = _masked_mean_curve_mag_b2hwt(x_hi_b2hwt, mask_hw)
    curve_hi_down = _masked_mean_curve_mag_b2hwt(x_hi_down_b2hwt, mask_hw)
    curve_lo = _masked_mean_curve_mag_b2hwt(x_lo_b2hwt, mask_hw)

    T_hi = int(curve_hi.shape[0])
    T_lo = int(curve_lo.shape[0])
    baseline_hi = max(1, min(int(baseline_frames), T_hi))
    baseline_lo = max(1, min(int(np.ceil(baseline_hi / float(factor))), T_lo))

    curve_hi = (curve_hi - curve_hi[:baseline_hi].mean()).cpu().numpy()
    curve_hi_down = (curve_hi_down - curve_hi_down[:baseline_lo].mean()).cpu().numpy()
    curve_lo = (curve_lo - curve_lo[:baseline_lo].mean()).cpu().numpy()

    t_hi = np.arange(T_hi)
    t_lo = np.arange(T_lo) * factor + (factor - 1) / 2.0

    std_np = std_map.detach().cpu().numpy()
    mask_np = mask_hw.detach().cpu().numpy().astype(np.float32)

    os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.3]})

    axes[0].imshow(std_np, cmap="magma")
    axes[0].imshow(mask_np, cmap="Reds", alpha=0.35, vmin=0.0, vmax=1.0)
    axes[0].set_title(f"Temporal std(|x_hi|), mask={target_pixels} px")
    axes[0].axis("off")

    axes[1].plot(t_hi, curve_hi, label=f"x_hi (T={T_hi})", color="tab:blue", linewidth=2)
    axes[1].plot(
        t_lo,
        curve_hi_down,
        label=f"Avg_{factor}(x_hi) (T={T_lo})",
        color="tab:orange",
        linewidth=2,
        marker="o",
        markersize=3,
    )
    axes[1].plot(
        t_lo,
        curve_lo,
        label=f"x_lo (rebinned) (T={T_lo})",
        color="tab:green",
        linewidth=2,
        marker="o",
        markersize=3,
    )
    axes[1].axhline(0.0, color="k", linewidth=1, alpha=0.3)
    axes[1].set_xlabel("High-temporal frame index")
    axes[1].set_ylabel("Masked mean |x| (baseline-subtracted)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    if title is None:
        title = f"Rebin diagnostic (factor={factor}, mask_fraction={mask_fraction:g})"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_filename, dpi=200)
    plt.close(fig)

    return {
        "factor": factor,
        "mask_fraction": float(mask_fraction),
        "mask_pixels": int(target_pixels),
        "baseline_hi_frames": int(baseline_hi),
        "baseline_lo_bins": int(baseline_lo),
        "curve_hi": curve_hi,
        "curve_hi_down": curve_hi_down,
        "curve_lo": curve_lo,
    }

    

def get_cosine_ei_weight(
    current_epoch,
    warmup_epochs,
    schedule_duration,
    target_weight
):
    """
    Calculates the EI loss weight for the current epoch using a cosine schedule.

    This implements a curriculum learning strategy:
    1. For `warmup_epochs`, the weight is 0 (MC loss only).
    2. Over the next `schedule_duration` epochs, the weight smoothly ramps
       up from 0 to `target_weight` following a cosine curve.
    3. After the schedule is complete, the weight stays at `target_weight`.

    Args:
        current_epoch (int): The current training epoch (starting from 1).
        warmup_epochs (int): Number of epochs to train with only MC loss.
        schedule_duration (int): Number of epochs for the ramp-up.
        target_weight (float): The final EI loss weight to reach.

    Returns:
        float: The EI loss weight for the current epoch.
    """
    # Phase 1: Warm-up phase (MC loss only)
    if current_epoch <= warmup_epochs:
        return 0.0

    # Calculate progress within the scheduling phase
    schedule_progress_epoch = current_epoch - warmup_epochs

    # Phase 3: Schedule is complete, hold at target weight
    if schedule_progress_epoch >= schedule_duration:
        return target_weight

    # Phase 2: Cosine ramp-up phase
    # This creates a value that goes from 0 to 1 along a cosine curve.
    cosine_multiplier = 0.5 * (1 - np.cos(np.pi * schedule_progress_epoch / schedule_duration))
    
    return target_weight * cosine_multiplier





def _select_plot_time_indices(n_timeframes: int, max_points: int = 12):
    n_timeframes = int(n_timeframes)
    if n_timeframes <= max_points:
        return list(range(n_timeframes))

    # Prefer an evenly divisible selection (stride = integer) close to max_points.
    for n_show in range(min(max_points, n_timeframes), 1, -1):
        if n_timeframes % n_show == 0:
            stride = n_timeframes // n_show
            return list(range(0, n_timeframes, stride))

    # Fallback: approximately even spacing if divisibility is not available.
    idx = np.linspace(0, n_timeframes - 1, num=max_points, dtype=int)
    return np.unique(idx).tolist()


def plot_reconstruction_sample(x_recon, title, filename, output_dir, grasp_img=None, batch_idx=0, transform=False):
    """
    Plot reconstruction sample showing magnitude images across timeframes.

    Args:
        x_recon: Reconstructed image tensor of shape (B, C, T, H, W)
        title: Title for the plot
        filename: Filename for saving (without extension)
        output_dir: Directory to save the plot
        batch_idx: Which batch element to plot (default: 0)
    """
    os.makedirs(output_dir, exist_ok=True)

    # compute magnitude from complex reconstruction
    if x_recon.shape[1] == 2:
        x_recon_mag = torch.sqrt(x_recon[:, 0, ...] ** 2 + x_recon[:, 1, ...] ** 2)
    else:
        x_recon_mag = x_recon


    n_timeframes = int(x_recon_mag.shape[-1])
    time_indices = _select_plot_time_indices(n_timeframes=n_timeframes, max_points=12)
    n_plot = len(time_indices)

    if grasp_img is not None:
        grasp_img_mag = torch.sqrt(grasp_img[:, 0, ...] ** 2 + grasp_img[:, 1, ...] ** 2)

        # if grasp_img_mag.shape[-1] == 320 and grasp_img_mag.shape[-2] == 320:
        #     n_timeframes = grasp_img_mag.shape[1]
        # elif grasp_img_mag.shape[-1] == 320 and grasp_img_mag.shape[1] == 320:
        #     n_timeframes = grasp_img_mag.shape[-2]
        # else:
        #     n_timeframes = grasp_img_mag.shape[-1]

        fig, axes = plt.subplots(
            nrows=2,
            ncols=n_plot,
            figsize=(n_plot * 3, 8),
            squeeze=False,
        )

        if transform:
            axes[0, 0].set_ylabel("Transformed Image", fontsize=14, labelpad=10)
            axes[1, 0].set_ylabel("Model Output", fontsize=14, labelpad=10)

            os.makedirs(os.path.join(output_dir, "transforms"), exist_ok=True)

        else:
            
            axes[0, 0].set_ylabel("Model Output", fontsize=14, labelpad=10)
            axes[1, 0].set_ylabel("GRASP Benchmark", fontsize=14, labelpad=10)


    else:
        fig, axes = plt.subplots(
            nrows=1,
            ncols=n_plot,
            figsize=(n_plot * 3, 4),
            squeeze=False,
        )

    for col, t in enumerate(time_indices):
        
        if x_recon_mag.shape[1] == n_timeframes:
            img = x_recon_mag[batch_idx, t, :, :].cpu().detach().numpy()
        else:
            img = x_recon_mag[batch_idx, ..., t].cpu().detach().numpy()

        if grasp_img is not None:
            if grasp_img_mag.shape[1] == n_timeframes:
                grasp_img_frame = grasp_img_mag[batch_idx, t, :, :].cpu().detach().numpy()
            elif grasp_img_mag.shape[-1] == n_timeframes:
                grasp_img_frame = grasp_img_mag[batch_idx, :, :, t].cpu().detach().numpy()
            else:
                grasp_img_frame = grasp_img_mag[batch_idx, :, t, :].cpu().detach().numpy()


            ax1 = axes[0, col]
        else:
            ax1 = axes[0, col]

        ax1.imshow(img, cmap="gray_r")
        ax1.set_title(f"t = {t}")
        ax1.set_xticks([])
        ax1.set_yticks([])

        if grasp_img is not None:
            ax2 = axes[1, col]
            ax2.imshow(grasp_img_frame, cmap="gray_r")
            ax2.set_title(f"t = {t}")
            ax2.set_xticks([])
            ax2.set_yticks([])
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f"{filename}.png"))
    plt.close(fig)


def get_git_commit():
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )
        return commit_hash
    except Exception as e:
        print(f"Error retrieving Git commit: {e}")
        return "unknown"
    

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')  # Remove 'module.' prefix
        new_state_dict[new_key] = v
    return new_state_dict


def save_checkpoint(model, optimizer, epoch,
                    train_curves, val_curves, eval_curves, ei_weight, step0_train_ei_loss, epoch_train_mc_loss, avg_grasp_ssim, avg_grasp_psnr, avg_grasp_mse, avg_grasp_lpips, avg_grasp_dc_mse, avg_grasp_dc_mae, avg_grasp_curve_corr, avg_grasp_raw_dc_mae, avg_grasp_raw_dc_mse, filename):
    
    # If the model is a DDP model, we need to access the underlying model
    # via the .module attribute to save a clean state_dict.
    model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "ei_weight": ei_weight,
        "step0_train_ei_loss": step0_train_ei_loss,
        "epoch_train_mc_loss": epoch_train_mc_loss,
        "avg_grasp_ssim": avg_grasp_ssim,
        "avg_grasp_psnr": avg_grasp_psnr,
        "avg_grasp_mse": avg_grasp_mse,
        "avg_grasp_lpips": avg_grasp_lpips,
        "avg_grasp_dc_mse": avg_grasp_dc_mse,
        "avg_grasp_dc_mae": avg_grasp_dc_mae,
        "avg_grasp_curve_corr": avg_grasp_curve_corr,
        "avg_grasp_raw_dc_mae": avg_grasp_raw_dc_mae,
        "avg_grasp_raw_dc_mse": avg_grasp_raw_dc_mse,
        **train_curves,   # unpack the dicts
        **val_curves,
        **eval_curves,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch} to {filename}")


def load_checkpoint(model, optimizer, filename):
    ckpt = _torch_load_checkpoint(filename, map_location="cpu")

    model_to_load = model.module if isinstance(model, DDP) else model

    model_to_load.load_state_dict(remove_module_prefix(ckpt["model_state_dict"]))
    optimizer.load_state_dict(remove_module_prefix(ckpt["optimizer_state_dict"]))

    # curves come back as Python lists (or start empty if key not found)
    train_curves = {
        "train_mc_losses": ckpt.get("train_mc_losses", []),
        "train_ei_losses": ckpt.get("train_ei_losses", []),
        "train_adj_losses": ckpt.get("train_adj_losses", []),
        "weighted_train_mc_losses": ckpt.get("weighted_train_mc_losses", []),
        "weighted_train_ei_losses": ckpt.get("weighted_train_ei_losses", []),
        "weighted_train_adj_losses": ckpt.get("weighted_train_adj_losses", []),
        "train_rebin_losses": ckpt.get("train_rebin_losses", []),
        "weighted_train_rebin_losses": ckpt.get("weighted_train_rebin_losses", []),
        "lr_history": ckpt.get("lr_history", []),
        "lr_epochs": ckpt.get("lr_epochs", []),
        "ei_weight_history": ckpt.get("ei_weight_history", []),
        "ei_weight_epochs": ckpt.get("ei_weight_epochs", []),
        "ei_gradnorm_ratio_history": ckpt.get("ei_gradnorm_ratio_history", []),
        "ei_gradnorm_ratio_epochs": ckpt.get("ei_gradnorm_ratio_epochs", []),
        "ei_gradnorm_ratio_ema": ckpt.get("ei_gradnorm_ratio_ema"),
        "ei_gradnorm_samples": ckpt.get("ei_gradnorm_samples", 0),
        "ei_gradnorm_locked": ckpt.get("ei_gradnorm_locked", False),
        "ei_target_weight_base": ckpt.get(
            "ei_target_weight_base",
            ckpt.get("ei_weight", 0.0),
        ),
        "ei_target_weight_effective": ckpt.get(
            "ei_target_weight_effective",
            ckpt.get("ei_weight", 0.0),
        ),
        "rebin_weight_history": ckpt.get("rebin_weight_history", []),
        "rebin_weight_epochs": ckpt.get("rebin_weight_epochs", []),
    }
    val_curves = {
        "val_mc_losses": ckpt.get("val_mc_losses", []),
        "val_ei_losses": ckpt.get("val_ei_losses", []),
        "val_adj_losses": ckpt.get("val_adj_losses", []),
        "val_raw_ssdu_losses": ckpt.get("val_raw_ssdu_losses", []),
    }

    eval_curves = {
        "eval_ssims": ckpt.get("eval_ssims", []),
        "eval_psnrs": ckpt.get("eval_psnrs", []),
        "eval_mses": ckpt.get("eval_mses", []),
        "eval_lpipses": ckpt.get("eval_lpipses", []),
        "eval_dc_mses": ckpt.get("eval_dc_mses", []),
        "eval_dc_maes": ckpt.get("eval_dc_maes", []),
        "eval_raw_dc_mses": ckpt.get("eval_raw_dc_mses", []),
        "eval_raw_dc_maes": ckpt.get("eval_raw_dc_maes", []),
        "eval_curve_corrs": ckpt.get("eval_curve_corrs", []),
        "eval_temporal_epochs": ckpt.get("eval_temporal_epochs", []),
        "eval_curve_maes": ckpt.get("eval_curve_maes", []),
        "eval_ttae_secs": ckpt.get("eval_ttae_secs", []),
        "eval_iauc10_errs": ckpt.get("eval_iauc10_errs", []),
        "eval_peak_errs": ckpt.get("eval_peak_errs", []),
        "eval_dl_dc_mae_bestfits": ckpt.get("eval_dl_dc_mae_bestfits", []),
        "eval_raw_ssdu_nmses": ckpt.get("eval_raw_ssdu_nmses", []),
        "avg_grasp_curve_mae": ckpt.get("avg_grasp_curve_mae", float("nan")),
        "avg_grasp_ttae_sec": ckpt.get("avg_grasp_ttae_sec", float("nan")),
        "avg_grasp_iauc10_err": ckpt.get("avg_grasp_iauc10_err", float("nan")),
        "avg_grasp_peak_err": ckpt.get("avg_grasp_peak_err", float("nan")),
        "avg_grasp_dc_mae_bestfit": ckpt.get("avg_grasp_dc_mae_bestfit", float("nan")),
        "avg_grasp_raw_ssdu_nmse": ckpt.get("avg_grasp_raw_ssdu_nmse", float("nan")),
        "eval_spf_curves": ckpt.get("eval_spf_curves", {}),
        "best_psnr": ckpt.get("best_psnr"),
        "best_epoch": ckpt.get("best_epoch"),
    }

    return model, optimizer, ckpt.get("epoch", 1), ckpt.get("ei_weight"), ckpt.get("step0_train_ei_loss"), ckpt.get("epoch_train_mc_loss"), train_curves, val_curves, eval_curves, ckpt.get("avg_grasp_ssim"), ckpt.get("avg_grasp_psnr"), ckpt.get("avg_grasp_mse"), ckpt.get("avg_grasp_lpips"), ckpt.get("avg_grasp_dc_mse"), ckpt.get("avg_grasp_dc_mae"), ckpt.get("avg_grasp_curve_corr"), ckpt.get("avg_grasp_raw_dc_mae"), ckpt.get("avg_grasp_raw_dc_mse")


def load_pretrained_weights(model, filename, skip_prefixes=None):
    """
    Load model weights for warm-starting without optimizer/epoch state.

    This is forgiving about missing/unexpected keys and shape mismatches,
    which is useful when enabling new components (e.g., encodings).
    """
    ckpt = _torch_load_checkpoint(filename, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    state_dict = remove_module_prefix(state_dict)

    model_to_load = model.module if isinstance(model, DDP) else model
    model_state = model_to_load.state_dict()

    skip_prefixes = tuple(skip_prefixes or [])
    filtered_state = {}
    skipped_keys = []
    mismatched_keys = []

    for key, value in state_dict.items():
        if skip_prefixes and key.startswith(skip_prefixes):
            skipped_keys.append(key)
            continue
        if key not in model_state:
            skipped_keys.append(key)
            continue
        if model_state[key].shape != value.shape:
            mismatched_keys.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        filtered_state[key] = value

    incompatible = model_to_load.load_state_dict(filtered_state, strict=False)

    info = {
        "checkpoint_keys": len(state_dict),
        "loaded_keys": len(filtered_state),
        "skipped_keys": len(skipped_keys),
        "skipped_sample": skipped_keys[:10],
        "mismatched_keys": mismatched_keys,
        "missing_keys": len(incompatible.missing_keys),
        "unexpected_keys": len(incompatible.unexpected_keys),
        "checkpoint_epoch": ckpt.get("epoch"),
    }
    return model, info

def to_torch_complex(x: torch.Tensor):
    """(B, 2, ...) real -> (B, ...) complex"""
    assert x.shape[1] == 2, (
        f"Input tensor must have 2 channels (real, imag), but got shape {x.shape}"
    )
    return torch.view_as_complex(rearrange(x, "b c ... -> b ... c").contiguous())







def _ktraj_to_sigpy_coord(ktraj: torch.Tensor, samples_per_spoke: int) -> np.ndarray:
    if not torch.is_tensor(ktraj):
        ktraj = torch.tensor(ktraj)
    if ktraj.ndim != 3 or ktraj.shape[0] != 2:
        raise ValueError(f"GRASP expects ktraj with shape (2, M, T), got {ktraj.shape}")
    M, T = ktraj.shape[1], ktraj.shape[2]
    if M % samples_per_spoke != 0:
        raise ValueError("GRASP ktraj length is not divisible by samples_per_spoke.")
    spokes = M // samples_per_spoke
    ktraj = ktraj.reshape(2, spokes, samples_per_spoke, T).permute(3, 1, 2, 0)
    return ktraj.cpu().numpy()


def GRASPRecon_from_ktraj(
    csmaps: torch.Tensor,
    kspace: torch.Tensor,
    ktraj: torch.Tensor,
    samples_per_spoke: int,
    device: Optional[sp.Device] = None,
    lamda: float = 0.001,
    max_iter: int = 10,
    rho: float = 0.1,
):
    if device is None:
        device = sp.Device(0 if torch.cuda.is_available() else -1)

    if kspace.dim() == 3:
        if kspace.shape[1] % samples_per_spoke != 0:
            raise ValueError("GRASP kspace length is not divisible by samples_per_spoke.")
        spokes = kspace.shape[1] // samples_per_spoke
        kspace = kspace.reshape(kspace.shape[0], spokes, samples_per_spoke, kspace.shape[2])
    elif kspace.dim() != 4:
        raise ValueError(f"Unsupported GRASP kspace shape: {kspace.shape}")

    kspace = kspace.permute(3, 0, 1, 2).unsqueeze(1).unsqueeze(3).cpu().numpy()

    if csmaps.dim() == 3:
        csmaps = csmaps.unsqueeze(0)
    csmaps = rearrange(csmaps, 'b c h w -> c b h w').cpu().numpy()

    traj = _ktraj_to_sigpy_coord(ktraj, samples_per_spoke)

    recon = app.HighDimensionalRecon(
        kspace,
        csmaps,
        combine_echo=False,
        lamda=lamda,
        coord=traj,
        regu='TV',
        regu_axes=[0],
        max_iter=max_iter,
        solver='ADMM',
        rho=rho,
        device=device,
        show_pbar=False,
        verbose=False,
    ).run()

    return np.squeeze(recon.get())


def GRASPRecon(csmaps, kspace, spokes_per_frame, num_frames, grasp_path):

    traj = get_traj(N_spokes=spokes_per_frame, N_time=num_frames)
    device = sp.Device(0 if torch.cuda.is_available() else -1)

    if kspace.dim() == 3:
        kspace = rearrange(kspace, 'c (sp sam) t -> t c sp sam', sam=640).unsqueeze(1).unsqueeze(3).cpu().numpy()
    elif kspace.dim() == 4:
        kspace = kspace.unsqueeze(1).unsqueeze(3).cpu().numpy()
        
    csmaps = rearrange(csmaps, 'b c h w -> c b h w').cpu().numpy()

    # reconstruct image
    R1 = app.HighDimensionalRecon(kspace, csmaps,
                            combine_echo=False,
                            lamda=0.001,
                            coord=traj,
                            regu='TV', regu_axes=[0],
                            max_iter=10,
                            solver='ADMM', rho=0.1,
                            device=device,
                            show_pbar=False,
                            verbose=False).run()

    R1 = np.squeeze(R1.get())

    np.save(grasp_path, R1)
    print(f"GRASP with {spokes_per_frame} spokes/frame and {num_frames} timeframes saved to {grasp_path}")

    return R1


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # For CUDA reproducibility
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def generate_sliding_window_indices(N_frames, chunk_size, overlap_size):
    """
    Generates start and end indices for a sliding window reconstruction.
    """
    if chunk_size <= 0 or N_frames <= 0:
        raise ValueError("chunk_size and N_frames must be positive.")
    if overlap_size >= chunk_size:
        raise ValueError("overlap_size must be less than chunk_size.")
    if overlap_size < 0:
        raise ValueError("overlap_size cannot be negative.")

    chunks = []
    step_size = chunk_size - overlap_size
    if step_size <= 0:
        step_size = 1  # ultra-conservative safety

    # Walk forward
    start = 0
    while start + chunk_size < N_frames:
        chunks.append((start, start + chunk_size))
        start += step_size

    # Ensure we always cover the tail
    tail_start = max(0, N_frames - chunk_size)
    if not chunks or chunks[-1][0] != tail_start:
        chunks.append((tail_start, N_frames))

    # Dedup (rare but safe)
    uniq = []
    seen = set()
    for s, e in chunks:
        if (s, e) not in seen:
            uniq.append((s, e))
            seen.add((s, e))
    return uniq


def _temporal_window(length: int, kind: str = "hann", device=None, dtype=torch.float32):
    """
    Create a 1D temporal window of size `length`.
    Using Hann is standard for overlap-add; if overlap=0, it's equivalent to a box.
    """
    if length <= 1:
        w = torch.ones(1, device=device, dtype=dtype)
    elif kind == "hann":
        # torch.hann_window produces length-L Hann in [0,1] with zeros at edges
        w = torch.hann_window(length, periodic=False, dtype=dtype, device=device)
        # Avoid exact zeros at boundaries to keep denominator well-conditioned
        w = torch.clamp(w, min=1e-3)
    elif kind == "box":
        w = torch.ones(length, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported window kind: {kind}")
    # Normalize window so that average weight ~1 (aids interpretability of adj_loss mean)
    w = w * (length / torch.sum(w))
    return w.view(1, 1, 1, 1, length)  # broadcastable to (B,C,H,W,T_chunk)


@torch.no_grad()
def sliding_window_inference(
    H, W, N_frames,
    ktraj, dcomp, nufft_ob, adjnufft_ob,
    chunk_size, chunk_overlap,
    kspace, csmap,
    acceleration_encoding,
    start_timepoint_index,  # ignored here; recomputed per chunk if time-encoding is enabled
    model, epoch, device,
    norm: str = "both",
    window_kind: str = "hann",
    collect_adj_loss: bool = True,
):
    """
    Edge-safe temporal stitching using Hann overlap-add:
      - Each chunk is multiplied by a temporal window (Hann by default).
      - Overlapping chunks are summed and normalized by the accumulated window weights.
    """
    chunk_indices = generate_sliding_window_indices(N_frames, chunk_size, chunk_overlap)

    # Pre-allocate stitched recon and weight accumulator
    # Shape target: (1, 2, H, W, T)
    stitched_recon = torch.zeros(1, 2, H, W, N_frames, device=device, dtype=torch.float32)
    weight_sum     = torch.zeros(1, 1, 1, 1, N_frames, device=device, dtype=torch.float32)

    csmap = csmap.to(device)
    # Keep track of adjoint-loss across chunks only when requested.
    adj_losses = [] if collect_adj_loss else None

    for i, (start_idx, end_idx) in enumerate(chunk_indices):
        print(f"Processing chunk {i+1}/{len(chunk_indices)}: frames {start_idx}-{end_idx}")

        # Slice per-chunk signals/operators (time last)
        kspace_chunk = kspace[..., start_idx:end_idx].to(device)
        ktraj_chunk  = ktraj[...,  start_idx:end_idx].to(device)
        dcomp_chunk  = dcomp[...,  start_idx:end_idx].to(device)

        # Per-chunk physics
        physics_chunk = MCNUFFT(nufft_ob, adjnufft_ob, ktraj_chunk, dcomp_chunk)

        # Time index encoding per chunk (if enabled by your model config)
        # We recreate start_timepoint_index here so the backbone sees absolute frame index
        sti = None
        if start_timepoint_index is not None:
            sti = torch.tensor([start_idx], dtype=torch.float32, device=device)

        # Forward pass
        model_out = model(
            kspace_chunk,
            physics_chunk,
            csmap,
            acceleration_encoding,
            sti,
            epoch=epoch,
            norm=norm,
            total_frames=N_frames,
        )
        x_recon_chunk = model_out[0]
        # x_recon_chunk: (1, 2, H, W, T_chunk)
        if collect_adj_loss:
            adj_losses.append(model_out[1].item())

        # Build temporal window for this chunk and overlap-add
        T_chunk = end_idx - start_idx
        w = _temporal_window(T_chunk, kind=window_kind, device=device, dtype=x_recon_chunk.dtype)

        stitched_recon[..., start_idx:end_idx] += x_recon_chunk * w
        weight_sum[...,     start_idx:end_idx] += w

        # (Optional) free chunk tensors early
        del kspace_chunk, ktraj_chunk, dcomp_chunk, physics_chunk, x_recon_chunk, w

    # Normalize by accumulated weights (safe divide)
    stitched_recon = stitched_recon / torch.clamp(weight_sum, min=1e-8)

    if collect_adj_loss and adj_losses is not None and len(adj_losses):
        mean_adj_loss = float(np.mean(adj_losses))
    else:
        mean_adj_loss = 0.0
    return stitched_recon, mean_adj_loss
