import os
import subprocess
import matplotlib.pyplot as plt
import torch
import numpy as np
from einops import rearrange
import torchkbnufft as tkbn
import csv
import sigpy as sp
from sigpy.mri import app
from radial_lsfp import MCNUFFT
import random
from torch.nn.parallel import DistributedDataParallel as DDP

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

    return ktraj

################### prepare NUFFT ################
def prep_nufft(Nsample, Nspokes, Ng):

    overSmaple = 2
    im_size = (int(Nsample/overSmaple), int(Nsample/overSmaple))
    grid_size = (Nsample, Nsample)

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
    output_filename: str = None
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


    n_timeframes = x_recon_mag.shape[-1]

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
            ncols=n_timeframes,
            figsize=(n_timeframes * 3, 8),
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
            ncols=n_timeframes,
            figsize=(n_timeframes * 3, 4),
            squeeze=True,
        )

    
    for t in range(n_timeframes):
        
        if x_recon_mag.shape[1] == n_timeframes:
            img = x_recon_mag[batch_idx, t, :, :].cpu().detach().numpy()
        else:
            img = x_recon_mag[batch_idx, ..., t].cpu().detach().numpy()

        if grasp_img is not None:
            if grasp_img_mag.shape[1] == n_timeframes:
                grasp_img = grasp_img_mag[batch_idx, t, :, :].cpu().detach().numpy()
            elif grasp_img_mag.shape[-1] == n_timeframes:
                grasp_img = grasp_img_mag[batch_idx, :, :, t].cpu().detach().numpy()
            else:
                grasp_img = grasp_img_mag[batch_idx, :, t, :].cpu().detach().numpy()


            ax1 = axes[0, t]
        else:
            ax1 = axes[t]

        ax1.imshow(img, cmap="gray")
        ax1.set_title(f"t = {t}")
        ax1.set_xticks([])
        ax1.set_yticks([])

        if grasp_img is not None:
            ax2 = axes[1, t]
            ax2.imshow(grasp_img, cmap="gray")
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
    ckpt = torch.load(filename, map_location="cpu")

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
    }
    val_curves = {
        "val_mc_losses": ckpt.get("val_mc_losses", []),
        "val_ei_losses": ckpt.get("val_ei_losses", []),
        "val_adj_losses": ckpt.get("val_adj_losses", []),
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
    }

    return model, optimizer, ckpt.get("epoch", 1), ckpt.get("ei_weight"), ckpt.get("step0_train_ei_loss"), ckpt.get("epoch_train_mc_loss"), train_curves, val_curves, eval_curves, ckpt.get("avg_grasp_ssim"), ckpt.get("avg_grasp_psnr"), ckpt.get("avg_grasp_mse"), ckpt.get("avg_grasp_lpips"), ckpt.get("avg_grasp_dc_mse"), ckpt.get("avg_grasp_dc_mae"), ckpt.get("avg_grasp_curve_corr"), ckpt.get("avg_grasp_raw_dc_mae"), ckpt.get("avg_grasp_raw_dc_mse")

def to_torch_complex(x: torch.Tensor):
    """(B, 2, ...) real -> (B, ...) complex"""
    assert x.shape[1] == 2, (
        f"Input tensor must have 2 channels (real, imag), but got shape {x.shape}"
    )
    return torch.view_as_complex(rearrange(x, "b c ... -> b ... c").contiguous())



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
    print(f"GRASP Recon with {spokes_per_frame} spokes/frame and {num_frames} timeframes saved to {grasp_path}")

    return R1



# def generate_sliding_window_indices(N_frames, chunk_size, overlap_size):
#     """
#     Generates start and end indices for a sliding window reconstruction.

#     Args:
#         N_frames (int): Total number of time frames in the dynamic MRI.
#         chunk_size (int): The number of frames in each chunk.
#         overlap_size (int): The number of frames that consecutive chunks will overlap.

#     Returns:
#         list of tuple: A list where each tuple contains (start_index, end_index)
#                        for a chunk.
#     """
#     if chunk_size <= 0 or N_frames <= 0:
#         raise ValueError("chunk_size and N_frames must be positive.")
#     if overlap_size >= chunk_size:
#         raise ValueError("overlap_size must be less than chunk_size.")
#     if overlap_size < 0:
#         raise ValueError("overlap_size cannot be negative.")

#     chunks = []
#     step_size = chunk_size - overlap_size
#     current_start = 0

#     while True:
#         current_end = current_start + chunk_size

#         # If the current chunk goes beyond the total N_frames,
#         # we adjust it to be the last possible full chunk.
#         # This handles the tail end of the sequence.
#         if current_end > N_frames:
#             # If the current_start is already past the beginning of a full chunk,
#             # or if it's the very first chunk and N_frames < chunk_size,
#             # we simply take the last `chunk_size` frames.
#             if N_frames - chunk_size >= 0:
#                 current_start = N_frames - chunk_size
#                 current_end = N_frames
#             else:
#                 # If N_frames is less than chunk_size, just take all frames as one chunk
#                 current_start = 0
#                 current_end = N_frames
#             chunks.append((current_start, current_end))
#             break # We've covered the end of the sequence

#         chunks.append((current_start, current_end))

#         # If we've already reached or passed the end of the sequence, stop
#         if current_end == N_frames:
#             break

#         current_start += step_size

#     # Remove duplicates if the last chunk was already added in a previous iteration
#     unique_chunks = []
#     seen = set()
#     for chunk in chunks:
#         if chunk not in seen:
#             unique_chunks.append(chunk)
#             seen.add(chunk)

#     return unique_chunks




# def sliding_window_inference(H, W, N_frames, ktraj, dcomp, nufft_ob, adjnufft_ob, chunk_size, chunk_overlap, kspace, csmap, acceleration_encoding, start_timepoint_index, model, epoch, device):

#     chunk_indices = generate_sliding_window_indices(N_frames, chunk_size, chunk_overlap)

#     # define physics object for chunk of timeframes
#     # ktraj_chunk, dcomp_chunk, nufft_ob_chunk, adjnufft_ob_chunk = prep_nufft(N_samples, N_spokes, chunk_size)
#     # physics_chunk = MCNUFFT(nufft_ob_chunk.to(device), adjnufft_ob_chunk.to(device), ktraj_chunk.to(device), dcomp_chunk.to(device))

#     # Define tensor for stitched reconstruction
#     stitched_recon = torch.zeros(1, 2, H, W, N_frames).to(device)

#     # Also keep track of how many times each frame is contributed to (for averaging)
#     frame_contribution_count = torch.zeros(H, W, N_frames).to(device)

#     csmap = csmap.to(device)


#     for i, (start_idx, end_idx) in enumerate(chunk_indices):

#         print(f"Processing chunk {i+1}: frames {start_idx}-{end_idx}")

#         # select chunk from k-space
#         kspace_chunk = kspace[..., start_idx:end_idx].to(device)
#         ktraj_chunk = ktraj[..., start_idx:end_idx].to(device)
#         dcomp_chunk = dcomp[..., start_idx:end_idx].to(device)

#         physics_chunk = MCNUFFT(nufft_ob, adjnufft_ob, ktraj_chunk, dcomp_chunk)

#         if start_timepoint_index is not None:
#             start_timepoint_index = torch.tensor([start_idx], dtype=torch.float, device=device)

#         # print("Time encoding: ", start_timepoint_index.item())

#         # generate reconstruction
#         x_recon_chunk, adj_loss, *_ = model(
#             kspace_chunk.to(device), physics_chunk, csmap, acceleration_encoding, start_timepoint_index, epoch=epoch, norm="both"
#         )

#         # Add the reconstructed chunk to the stitched_recon
#         stitched_recon[..., start_idx:end_idx] += x_recon_chunk
#         frame_contribution_count[..., start_idx:end_idx] += 1


#     # Average the overlapping regions
#     stitched_recon /= frame_contribution_count # This performs element-wise division

#     return stitched_recon, adj_loss.item()


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
    window_kind: str = "hann"
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
    # Keep track of adjoint-loss across chunks (report mean)
    adj_losses = []

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
        x_recon_chunk, adj_loss, *_ = model(
            kspace_chunk, physics_chunk, csmap, acceleration_encoding, sti, epoch=epoch, norm="both"
        )
        # x_recon_chunk: (1, 2, H, W, T_chunk)
        adj_losses.append(adj_loss.item())

        # Build temporal window for this chunk and overlap-add
        T_chunk = end_idx - start_idx
        w = _temporal_window(T_chunk, kind=window_kind, device=device, dtype=x_recon_chunk.dtype)

        stitched_recon[..., start_idx:end_idx] += x_recon_chunk * w
        weight_sum[...,     start_idx:end_idx] += w

        # (Optional) free chunk tensors early
        del kspace_chunk, ktraj_chunk, dcomp_chunk, physics_chunk, x_recon_chunk, w
        torch.cuda.empty_cache()

    # Normalize by accumulated weights (safe divide)
    stitched_recon = stitched_recon / torch.clamp(weight_sum, min=1e-8)

    mean_adj_loss = float(np.mean(adj_losses)) if len(adj_losses) else 0.0
    return stitched_recon, mean_adj_loss
