import os
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torchmetrics
import time
from dataloader import SimulatedDataset
from lsfpnet import to_torch_complex, from_torch_complex
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import PchipInterpolator
from tqdm import tqdm # A library for a nice progress bar
from scipy.stats import mannwhitneyu
from skimage.metrics import structural_similarity as ssim_map_func
import matplotlib.gridspec as gridspec
from skimage.measure import find_contours
from typing import List, Dict
from scipy.stats import pearsonr
import nibabel as nib
import pandas as pd
from functools import lru_cache
from cluster_paths import _swap_base


TUMOR_SEG_ROOT = os.environ.get("TUMOR_SEG_ROOT", "/net/scratch2/rachelgordon/zf_data_192_slices/tumor_segmentations_lcr")
SLICE_MAP_PATH = Path(__file__).resolve().parent / "data" / "largest_tumor_slices.csv"

# Plot styling 
PLOT_FONT_SIZES = {
    "suptitle": 28,
    "title": 24,
    "label": 20,
    "tick": 18,
    "legend": 16,
}
PLOT_LAYOUT = {
    "pad": 0.5,
    "w_pad": 0.5,
    "h_pad": 0.5,
}
PLOT_ADJUST = {
    "left": 0.02,
    "right": 0.98,
    "bottom": 0.03,
    "top": 0.9,
    "wspace": 0.1,
    "hspace": 0.1,
}

# ==========================================================
# EVALUATION FUNCTIONS
# ==========================================================

def normalize_for_lpips(image, data_range):
    """Normalizes an image tensor to the [-1, 1] range for LPIPS."""
    min_val, max_val = data_range
    # Scale to [0, 1]
    image_0_1 = (image - min_val) / (max_val - min_val)
    # Scale to [-1, 1]
    image_minus1_1 = 2 * image_0_1 - 1
    return image_minus1_1


def calc_image_metrics(input, reference, data_range, device):
    """
    Calculates image metrics for a given input and reference image.
    """

    # --- Initialize Metrics ---
    # We will compute metrics frame by frame. data_range is important for PSNR.
    ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=data_range).to(device)
    psnr = torchmetrics.image.PeakSignalNoiseRatio(data_range=data_range).to(device)
    mse = torchmetrics.MeanSquaredError().to(device)
    lpips_metric = torchmetrics.image.LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=False).to(device)

    ssim = ssim(input, reference)
    psnr = psnr(input, reference)
    mse = mse(input, reference)

    # --- Handle 5D Volumetric Data by averaging over slices ---
    if input.dim() == 5:
        # Input shape: [N, C, D, H, W]
        num_slices = input.shape[2]
        
        lpips_scores = []

        for i in range(num_slices):
            # Extract the i-th slice from both tensors
            # Resulting shape is [N, C, H, W] which is a valid 4D tensor
            input_slice = input[:, :, i, :, :]
            reference_slice = reference[:, :, i, :, :]

            # --- Prepare the slice for LPIPS ---
            input_lpips = normalize_for_lpips(input_slice.clone(), data_range)
            reference_lpips = normalize_for_lpips(reference_slice.clone(), data_range)

            # LPIPS expects 3 channels. Since the slice is now 4D, this repeat will work.
            if input_lpips.shape[1] == 1:
                input_lpips = input_lpips.repeat(1, 3, 1, 1)
                reference_lpips = reference_lpips.repeat(1, 3, 1, 1)

            input_lpips = input_lpips.to(reference_lpips.dtype)
            
            lpips_scores.append(lpips_metric(input_lpips, reference_lpips).item())

        # Average the scores from all slices
        final_lpips = sum(lpips_scores) / len(lpips_scores)


    return ssim.item(), psnr.item(), mse.item(), final_lpips
    


## Evaluate Data Consistency in k-space

def calc_dc(input, reference, device):
    """
    Calculates data consistency MSE for a given input and reference k-space tensor.
    """

    mse = torchmetrics.MeanSquaredError().to(device)
    mae = torchmetrics.MeanAbsoluteError().to(device)

    input = from_torch_complex(input).to(device)
    reference = from_torch_complex(reference).to(device)

    mse = mse(input, reference)
    mae = mae(input, reference)

    return mse.item(), mae.item()


def _get_patient_id_from_grasp_path(grasp_path: str, mapping_csv: str = "data/DROSubID_vs_fastMRIbreastID.csv") -> str:
    """Maps a DRO sample path back to the fastMRI patient id."""
    if grasp_path is None:
        return None

    # DataLoader batches lists of strings when batch_size>0; unwrap singletons.
    if isinstance(grasp_path, (list, tuple)):
        if len(grasp_path) == 0:
            return None
        grasp_path = grasp_path[0]

    sample_dir = os.path.basename(os.path.dirname(grasp_path))
    try:
        dro_id = int(sample_dir.split("_")[1])
    except (IndexError, ValueError):
        print(f"Could not parse DRO id from grasp path: {grasp_path}")
        return None

    if not os.path.exists(mapping_csv):
        print(f"Mapping CSV not found at {mapping_csv}; cannot fetch patient id.")
        return None

    id_map = pd.read_csv(mapping_csv)
    match = id_map[id_map["DRO"] == dro_id]
    if match.empty:
        print(f"No fastMRI id found for DRO id {dro_id} in {mapping_csv}.")
        return None

    fastmri_id = int(match["fastMRIbreast"].iloc[0])
    return f"fastMRI_breast_{fastmri_id:03d}_2"


def _load_tumor_mask(cluster: str, patient_id: str, slice_idx: int = None, seg_root: str = TUMOR_SEG_ROOT) -> np.ndarray:
    """Loads the tumor segmentation for a raw scan and selects the desired slice."""

    if cluster == "Randi":
        seg_root = _swap_base(seg_root, cluster, path_type="data")
        
    if patient_id is None:
        return None

    seg_path = os.path.join(seg_root, f"{patient_id}.nii.gz")
    if not os.path.exists(seg_path):
        print(f"Tumor segmentation not found at {seg_path}")
        return None

    seg_vol = nib.load(seg_path).get_fdata()

    if seg_vol.ndim == 3:
        num_slices = seg_vol.shape[-1]
        if slice_idx is None or slice_idx < 0 or slice_idx >= num_slices:
            slice_sums = seg_vol.sum(axis=tuple(range(seg_vol.ndim - 1)))
            slice_idx = int(np.argmax(slice_sums))
        tumor_mask = seg_vol[..., int(slice_idx)]
    else:
        tumor_mask = seg_vol

    return tumor_mask.astype(bool)


@lru_cache(maxsize=1)
def _load_slice_map(slice_map_path: Path = SLICE_MAP_PATH) -> Dict[str, int]:
    """Load patient -> slice index map for non-DRO eval; cache for reuse."""
    if not slice_map_path.exists():
        print(f"Slice map not found at {slice_map_path}; falling back to configured slice indices.")
        return {}

    mapping = {}
    with open(slice_map_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("fastMRI_breast_id")
            idx = row.get("largest_slice_idx")
            if pid is None or idx is None:
                continue
            pid = pid.replace(".nii", "")
            try:
                mapping[pid] = int(idx)
            except ValueError:
                continue
    return mapping


def _resolve_plot_label(label: str, grasp_path: str):
    """Return a plot label and patient id, preferring the fastMRI id mapped from the DRO grasp path."""
    patient_id = _get_patient_id_from_grasp_path(grasp_path)
    return patient_id or label, patient_id


# ==========================================================
# PLOTTING FUNCTIONS
# ==========================================================

def plot_spatial_quality(
    recon_img: np.ndarray,
    gt_img: np.ndarray,
    grasp_img: np.ndarray,
    time_frame_index: int,
    filename: str,
    grasp_comparison_filename: str,
    data_range: float, 
    acceleration: float,
    spokes_per_frame: int, 
    plot_dro: bool = True,
):
    """
    Generates a comparison plot for a single time frame in a 2x4 grid.
    Each row includes: Ground Truth, Reconstruction, Error Map, and SSIM Map.

    Args:
        recon_img (np.ndarray): Your model's reconstructed image for this frame.
        gt_img (np.ndarray): The ground truth image for this frame.
        grasp_img (np.ndarray): The GRASP reconstruction image for this frame.
        time_frame_index (int): The index of the time frame for titling.
        filename (str): The path to save the output plot.
    """
    if plot_dro:
        # Calculate error maps
        error_map_dl = recon_img - gt_img
        error_map_grasp = grasp_img - gt_img

        # Calculate SSIM maps
        ssim_dl, ssim_map_dl = ssim_map_func(gt_img, recon_img, data_range=data_range, full=True)
        ssim_grasp, ssim_map_grasp = ssim_map_func(gt_img, grasp_img, data_range=data_range, full=True)

        # Create a 2x4 plot grid with dedicated colorbar columns so image axes stay the same size.
        fig = plt.figure(figsize=(24, 12))
        gs = gridspec.GridSpec(
            2,
            6,
            figure=fig,
            width_ratios=[1, 1, 1, 0.05, 1, 0.05],
            wspace=0.16,
            hspace=0.16,
        )
        axes = np.empty((2, 4), dtype=object)
        axes[0, 0] = fig.add_subplot(gs[0, 0])
        axes[0, 1] = fig.add_subplot(gs[0, 1])
        axes[0, 2] = fig.add_subplot(gs[0, 2])
        cax_err_dl = fig.add_subplot(gs[0, 3])
        axes[0, 3] = fig.add_subplot(gs[0, 4])
        cax_ssim_dl = fig.add_subplot(gs[0, 5])
        axes[1, 0] = fig.add_subplot(gs[1, 0])
        axes[1, 1] = fig.add_subplot(gs[1, 1])
        axes[1, 2] = fig.add_subplot(gs[1, 2])
        cax_err_grasp = fig.add_subplot(gs[1, 3])
        axes[1, 3] = fig.add_subplot(gs[1, 4])
        cax_ssim_grasp = fig.add_subplot(gs[1, 5])
        fig.suptitle(
            f"Spatial Quality Comparison at Time Frame {time_frame_index} with AF {acceleration} and SPF {spokes_per_frame}",
            fontsize=PLOT_FONT_SIZES["suptitle"],
            y=0.95,
        )

        # --- Top Row: DL Reconstruction Comparison ---
        axes[0, 0].imshow(gt_img, cmap='gray')
        axes[0, 0].set_title("Ground Truth", fontsize=PLOT_FONT_SIZES["title"])

        axes[0, 1].imshow(recon_img, cmap='gray')
        axes[0, 1].set_title("DL Reconstruction", fontsize=PLOT_FONT_SIZES["title"])

        im_err_dl = axes[0, 2].imshow(error_map_dl, cmap='coolwarm', vmin=-0.5, vmax=0.5)
        axes[0, 2].set_title("DL Error Map", fontsize=PLOT_FONT_SIZES["title"])
        cb_err_dl = fig.colorbar(im_err_dl, cax=cax_err_dl)
        cb_err_dl.ax.tick_params(labelsize=PLOT_FONT_SIZES["tick"])

        im_ssim_dl = axes[0, 3].imshow(ssim_map_dl, cmap='viridis', vmin=0, vmax=1)
        axes[0, 3].set_title(
            f"DL SSIM Map (SSIM: {round(ssim_dl, 3)})",
            fontsize=PLOT_FONT_SIZES["title"],
        )
        cb_ssim_dl = fig.colorbar(im_ssim_dl, cax=cax_ssim_dl)
        cb_ssim_dl.ax.tick_params(labelsize=PLOT_FONT_SIZES["tick"])

        # --- Bottom Row: GRASP Reconstruction Comparison ---
        axes[1, 0].imshow(gt_img, cmap='gray')
        axes[1, 0].set_title("Ground Truth", fontsize=PLOT_FONT_SIZES["title"])

        axes[1, 1].imshow(grasp_img, cmap='gray')
        axes[1, 1].set_title("GRASP Reconstruction", fontsize=PLOT_FONT_SIZES["title"])

        im_err_grasp = axes[1, 2].imshow(error_map_grasp, cmap='coolwarm', vmin=-0.5, vmax=0.5)
        axes[1, 2].set_title("GRASP Error Map", fontsize=PLOT_FONT_SIZES["title"])
        cb_err_grasp = fig.colorbar(im_err_grasp, cax=cax_err_grasp)
        cb_err_grasp.ax.tick_params(labelsize=PLOT_FONT_SIZES["tick"])

        im_ssim_grasp = axes[1, 3].imshow(ssim_map_grasp, cmap='viridis', vmin=0, vmax=1)
        axes[1, 3].set_title(
            f"GRASP SSIM Map (SSIM: {round(ssim_grasp, 3)})",
            fontsize=PLOT_FONT_SIZES["title"],
        )
        cb_ssim_grasp = fig.colorbar(im_ssim_grasp, cax=cax_ssim_grasp)
        cb_ssim_grasp.ax.tick_params(labelsize=PLOT_FONT_SIZES["tick"])
        
        # Turn off axes for all plots
        for ax in axes.flat:
            ax.axis('off')

        fig.subplots_adjust(**{**PLOT_ADJUST, "top": 0.86})
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.02)

        # Save a separate first-row-only figure without a suptitle.
        top_fig = plt.figure(figsize=(24, 6))
        top_gs = gridspec.GridSpec(
            1,
            6,
            figure=top_fig,
            width_ratios=[1, 1, 1, 0.07, 1, 0.07],
            wspace=0.24,
        )
        top_axes = np.empty(4, dtype=object)
        top_axes[0] = top_fig.add_subplot(top_gs[0, 0])
        top_axes[1] = top_fig.add_subplot(top_gs[0, 1])
        top_axes[2] = top_fig.add_subplot(top_gs[0, 2])
        top_cax_err = top_fig.add_subplot(top_gs[0, 3])
        top_axes[3] = top_fig.add_subplot(top_gs[0, 4])
        top_cax_ssim = top_fig.add_subplot(top_gs[0, 5])

        top_axes[0].imshow(gt_img, cmap='gray')
        top_axes[0].set_title("Ground Truth", fontsize=PLOT_FONT_SIZES["title"])
        top_axes[1].imshow(recon_img, cmap='gray')
        top_axes[1].set_title("DL Reconstruction", fontsize=PLOT_FONT_SIZES["title"])
        top_axes[2].imshow(error_map_dl, cmap='coolwarm', vmin=-0.5, vmax=0.5)
        top_axes[2].set_title("DL Error Map", fontsize=PLOT_FONT_SIZES["title"])
        top_cb_err = top_fig.colorbar(top_axes[2].images[0], cax=top_cax_err)
        top_cb_err.ax.tick_params(labelsize=PLOT_FONT_SIZES["tick"])
        top_axes[3].imshow(ssim_map_dl, cmap='viridis', vmin=0, vmax=1)
        top_axes[3].set_title(
            f"DL SSIM Map (SSIM: {round(ssim_dl, 3)})",
            fontsize=PLOT_FONT_SIZES["title"],
        )
        top_cb_ssim = top_fig.colorbar(top_axes[3].images[0], cax=top_cax_ssim)
        top_cb_ssim.ax.tick_params(labelsize=PLOT_FONT_SIZES["tick"])

        for ax in top_axes.flat:
            ax.axis('off')

        base_name, ext = os.path.splitext(filename)
        top_row_filename = f"{base_name}_top_row{ext}"
        top_fig.subplots_adjust(**{**PLOT_ADJUST, "top": 0.95, "bottom": 0.04})
        top_fig.savefig(top_row_filename, bbox_inches='tight', pad_inches=0.02)
        plt.close(top_fig)
        plt.close()


    # Plot the Difference Between GRASP and DL Recon

    # Calculate error map
    error_map = recon_img - grasp_img

    vmin = np.percentile(error_map, 1)
    vmax = np.percentile(error_map, 99)

    # Calculate SSIM maps
    ssim, ssim_map = ssim_map_func(grasp_img, recon_img, data_range=data_range, full=True)

    vmin_ssim = np.percentile(ssim_map, 5)
    vmax_ssim = np.percentile(ssim_map, 95)

    # Create a 1x4 plot grid with dedicated colorbar columns so image axes stay the same size.
    fig = plt.figure(figsize=(24, 6))
    gs = gridspec.GridSpec(
        1,
        6,
        figure=fig,
        width_ratios=[1, 1, 1, 0.05, 1, 0.05],
        wspace=0.16,
    )
    axes = np.empty(4, dtype=object)
    axes[0] = fig.add_subplot(gs[0, 0])
    axes[1] = fig.add_subplot(gs[0, 1])
    axes[2] = fig.add_subplot(gs[0, 2])
    cax_err_dl = fig.add_subplot(gs[0, 3])
    axes[3] = fig.add_subplot(gs[0, 4])
    cax_ssim_dl = fig.add_subplot(gs[0, 5])
    fig.suptitle(
        f"DL vs GRASP Comparison at Time Frame {time_frame_index} with AF {acceleration} and SPF {spokes_per_frame}",
        fontsize=PLOT_FONT_SIZES["suptitle"],
        y=0.995,
    )

    # --- Top Row: DL Reconstruction Comparison ---
    axes[0].imshow(grasp_img, cmap='gray')
    axes[0].set_title("GRASP Reconstruction", fontsize=PLOT_FONT_SIZES["title"])

    axes[1].imshow(recon_img, cmap='gray')
    axes[1].set_title("DL Reconstruction", fontsize=PLOT_FONT_SIZES["title"])

    im_err_dl = axes[2].imshow(error_map, cmap='coolwarm', vmin=vmin, vmax=vmax)
    axes[2].set_title("Error Map", fontsize=PLOT_FONT_SIZES["title"])
    cb_err_dl = fig.colorbar(im_err_dl, cax=cax_err_dl)
    cb_err_dl.ax.tick_params(labelsize=PLOT_FONT_SIZES["tick"])

    im_ssim_dl = axes[3].imshow(ssim_map, cmap='viridis', vmin=vmin_ssim, vmax=vmax_ssim)
    axes[3].set_title(
        f"SSIM Map (SSIM: {round(ssim, 3)})",
        fontsize=PLOT_FONT_SIZES["title"],
    )
    cb_ssim_dl = fig.colorbar(im_ssim_dl, cax=cax_ssim_dl)
    cb_ssim_dl.ax.tick_params(labelsize=PLOT_FONT_SIZES["tick"])
    
    # Turn off axes for all plots
    for ax in axes.flat:
        ax.axis('off')

    fig.subplots_adjust(**{**PLOT_ADJUST, "top": 0.82})
    plt.savefig(grasp_comparison_filename, bbox_inches='tight', pad_inches=0.02)
    plt.close()





def plot_temporal_curves(
    gt_img_stack: np.ndarray,
    recon_img_stack: np.ndarray,
    grasp_img_stack: np.ndarray,
    masks: dict,
    time_points: np.ndarray,
    filename: str, 
    acceleration: float,
    spokes_per_frame: int, 
    plot_dro: bool = True,
):
    """
    Plots the mean signal intensity vs. time for different tissue regions.
    This is CRITICAL for debugging PK model fitting.

    Args:
        gt_img_stack (np.ndarray): Time series of ground truth images (H, W, T).
        recon_img_stack (np.ndarray): Time series of your model's images (H, W, T).
        grasp_img_stack (np.ndarray): Time series of GRASP images (H, W, T).
        masks (dict): Dictionary of boolean NumPy masks for different regions.
        time_points (np.ndarray): The time vector for the x-axis.
        filename (str): The path to save the output plot.
    """
    regions = [r for r in ['malignant', 'glandular', 'muscle'] if r in masks and masks[r].any()]

    if not regions:
        print("No relevant regions found in mask to plot temporal curves.")
        return

    fig, axes = plt.subplots(1, len(regions), figsize=(7 * len(regions), 5))
    if len(regions) == 1: axes = [axes] # Ensure axes is always a list
    fig.suptitle(
        f"Mean Signal vs. Time (AF = {acceleration}, SPF = {spokes_per_frame})",
        fontsize=PLOT_FONT_SIZES["suptitle"],
    )

    region_corrs = {}

    for i, region in enumerate(regions):
        mask = masks[region]

        # Calculate mean signal in the masked region for each time point
        gt_curve = [gt_img_stack[:, :, t][mask].mean() for t in range(gt_img_stack.shape[2])]
        recon_curve = [recon_img_stack[:, :, t][mask].mean() for t in range(recon_img_stack.shape[2])]
        grasp_curve = [grasp_img_stack[:, :, t][mask].mean() for t in range(grasp_img_stack.shape[2])]

        # compute the pearson correlation coefficients
        recon_correlation, _ = pearsonr(recon_curve, gt_curve)
        grasp_correlation, _ = pearsonr(grasp_curve, gt_curve)

        recon_correlation = float(recon_correlation)
        grasp_correlation = float(grasp_correlation)

        region_corrs[region] = {"DL": recon_correlation, "GRASP":  grasp_correlation}


        # if region == 'malignant':
        #     recon_correlation, _ = pearsonr(recon_curve, gt_curve)
        #     grasp_correlation, _ = pearsonr(grasp_curve, gt_curve)


        # Plot
        if plot_dro:
            axes[i].plot(time_points, gt_curve, 'k-', label='Ground Truth', linewidth=2, marker='o')

        axes[i].plot(time_points, recon_curve, 'r--', label='DL Recon', marker='o')
        axes[i].plot(time_points, grasp_curve, 'b:', label='GRASP Recon', marker='o')
        
        if plot_dro:
            axes[i].set_title(
                f"{region.capitalize()} (DL: {recon_correlation:.2f}, GRASP: {grasp_correlation:.2f})",
                fontsize=PLOT_FONT_SIZES["title"],
            )
        else: 
            axes[i].set_title(f"{region.capitalize()}", fontsize=PLOT_FONT_SIZES["title"])
        axes[i].set_xlabel("Time (s)", fontsize=PLOT_FONT_SIZES["label"])
        axes[i].tick_params(axis='both', which='major', labelsize=PLOT_FONT_SIZES["tick"])
        axes[i].grid(True)
        axes[i].legend(fontsize=PLOT_FONT_SIZES["legend"])

    axes[0].set_ylabel("Mean Signal Intensity", fontsize=PLOT_FONT_SIZES["label"])
    plt.tight_layout(rect=[0, 0.02, 1, 0.94], **PLOT_LAYOUT)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.02)
    plt.close()

    return region_corrs



def plot_single_temporal_curve(
    img_stack: np.ndarray,
    masks: Dict[str, np.ndarray],
    time_points: np.ndarray,
    num_frames: int,
    filename: str,
    acceleration: float,
    spokes_per_frame: int,
    # New arguments required for this specific plot style:
    frames_to_show: List[int] = None,
):
    """
    Generates a comprehensive analysis plot for a single sample, showing the
    Tumor Contrast Enhancement Curve (CEC) and corresponding image frames with
    the tumor Region of Interest (ROI) highlighted.

    This function is modified to produce a detailed analysis plot for the
    'malignant' tissue type, using the ground truth data.

    Args:
        gt_img_stack (np.ndarray): Time series of ground truth images (H, W, T).
        recon_img_stack (np.ndarray): Unused in this plot, kept for signature compatibility.
        grasp_img_stack (np.ndarray): Unused in this plot, kept for signature compatibility.
        masks (dict): Dictionary of boolean NumPy masks. Expects a 'malignant' key.
        time_points (np.ndarray): The time vector for the x-axis (e.g., frame numbers).
        filename (str): The path to save the output plot.
        sample_name (str): The name of the sample for the main plot title.
        frames_to_show (List[int]): A list of 4 frame indices to display in the
                                    image grid and highlight on the curve.
                                    If None, defaults to [0, 6, 13, 20].
    """
    # This function now specifically targets the 'malignant' tumor region.
    region_key = 'malignant'
    if region_key not in masks or not masks[region_key].any():
        print(f"'{region_key}' mask not found or is empty. Skipping plot generation.")
        return

    tumor_mask = masks[region_key]

    if frames_to_show is None:
        # frames_to_show = [0, 6, 20, 21] # Default frames from the example
        interval = round(num_frames / 4)
        frames_to_show = [0, interval, 2*interval, num_frames-1]
        # frames_to_show = [0, 6, 13, 20] # Default frames from the example
    if len(frames_to_show) != 4:
        raise ValueError(f"This function is designed to show exactly 4 frames, but {len(frames_to_show)} were provided.")

    # --- 1. Setup Figure and Layout ---
    fig = plt.figure(figsize=(20, 8.5))
    fig.suptitle(
        f"Tumor Enhancement Over Time (AF = {acceleration}, SPF = {spokes_per_frame})",
        fontsize=PLOT_FONT_SIZES["suptitle"],
        y=0.985,
    )
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.16, wspace=0.16)

    ax_curve = fig.add_subplot(gs[:, 0:2])
    ax_imgs = [
        fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3]),
        fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[1, 3])
    ]

    # fig.suptitle(f"Enhancement Curve", fontsize=22, y=0.98)

    # --- 2. Plot Tumor Enhancement Curve (Left Panel) ---
    mean_curve = [img_stack[:, :, t][tumor_mask].mean() for t in range(img_stack.shape[2])]

    # Plot the full curve with markers
    ax_curve.plot(time_points, mean_curve, 'o-', label='Mean Tumor Signal', linewidth=2, markersize=6)

    # Highlight specific points on the curve with red stars
    highlight_times = [time_points[i] for i in frames_to_show]
    highlight_vals = [mean_curve[i] for i in frames_to_show]
    ax_curve.plot(highlight_times, highlight_vals, 'r*', markersize=18, zorder=10) # zorder to ensure stars are on top

    # Formatting the curve plot
    ax_curve.set_title("Tumor Contrast Enhancement Curve (CEC)", fontsize=PLOT_FONT_SIZES["title"], pad=8)
    ax_curve.set_xlabel("Time Frame", fontsize=PLOT_FONT_SIZES["label"])
    ax_curve.set_ylabel("Mean Signal Intensity", fontsize=PLOT_FONT_SIZES["label"])
    ax_curve.legend(fontsize=PLOT_FONT_SIZES["legend"])
    ax_curve.grid(True, linestyle='--')
    ax_curve.tick_params(axis='both', which='major', labelsize=PLOT_FONT_SIZES["tick"])

    # --- 3. Plot Image Frames with ROI (Right Panel) ---
    # Find contours of the tumor mask to draw an outline
    contours = find_contours(tumor_mask, 0.5)

    # Use consistent intensity scaling for all image frames
    # vmin, vmax = np.percentile(img_stack, [1, 99.5])

    for i, frame_idx in enumerate(frames_to_show):
        ax = ax_imgs[i]
        image = img_stack[:, :, frame_idx]

        # Display the image
        ax.imshow(image, cmap='gray')#, vmin=vmin, vmax=vmax)

        # Overlay the tumor ROI outline
        for contour in contours:
            # contour is (row, col), plot needs (x, y)
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='red')

        # Formatting for each image subplot
        ax.set_title(f"Frame {frame_idx}", fontsize=PLOT_FONT_SIZES["title"])
        ax.axis('off')

    # --- 4. Finalize and Save ---
    plt.tight_layout(rect=[0, 0, 1, 0.9], **PLOT_LAYOUT)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.02, dpi=150)
    plt.close(fig)



def plot_time_series(
    recon_img_stack: np.ndarray,
    grasp_img_stack: np.ndarray,
    filename: str,
    acceleration: float,
    spokes_per_frame: int, 
):
    """
    Plots the middle 5 time points for Ground Truth, DL Recon, and GRASP.

    Args:
        gt_img_stack (np.ndarray): Time series of ground truth images (H, W, T).
        recon_img_stack (np.ndarray): Time series of your model's images (H, W, T).
        grasp_img_stack (np.ndarray): Time series of GRASP images (H, W, T).
        filename (str): The path to save the output plot.
    """
    num_frames = recon_img_stack.shape[2]
    
    # Select 5 time points: start, 1/4, 1/2, 3/4, end
    indices = np.linspace(0, num_frames - 1, 5, dtype=int)
    
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle(
        f"Temporal Series Comparison (AF = {acceleration}, SPF = {spokes_per_frame})",
        fontsize=PLOT_FONT_SIZES["suptitle"],
        y=0.995,
    )

    # --- Row 1: Ground Truth ---
    # for i, frame_idx in enumerate(indices):
    #     img = gt_img_stack[:, :, frame_idx]
    #     axes[0, i].imshow(img, cmap='gray')
    #     axes[0, i].set_title(f"GT: Frame {frame_idx}")
    #     axes[0, i].axis('off')

    # --- Row 2: DL Reconstruction ---
    for i, frame_idx in enumerate(indices):
        img = recon_img_stack[:, :, frame_idx]
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f"DL: Frame {frame_idx}", fontsize=PLOT_FONT_SIZES["title"])
        axes[0, i].axis('off')

    # --- Row 3: GRASP Reconstruction ---
    for i, frame_idx in enumerate(indices):
        img = grasp_img_stack[:, :, frame_idx]
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].set_title(f"GRASP: Frame {frame_idx}", fontsize=PLOT_FONT_SIZES["title"])
        axes[1, i].axis('off')

    fig.subplots_adjust(**PLOT_ADJUST)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.02)
    plt.close()





# ==========================================================
# EVALUATION 
# ==========================================================
def eval_grasp(kspace, csmap, ground_truth, grasp_recon, physics, device, output_dir, dro_eval=True):


    # ==========================================================
    # EVALUATE DATA CONSISTENCY
    # ==========================================================

    # Forward Simulation
    grasp_recon_complex = rearrange(to_torch_complex(grasp_recon).squeeze(), 'h t w -> h w t')
    kspace = kspace.squeeze()

    grasp_kspace = physics(False, grasp_recon_complex.to(csmap.dtype), csmap)


    # Compute MSE
    dc_mse_grasp, dc_mae_grasp = calc_dc(grasp_kspace, kspace, device)


    # ==========================================================
    # EVALUATE SPATIAL IMAGE QUALITY
    # ==========================================================
    if dro_eval:

        grasp_recon_np = grasp_recon.cpu().numpy()
        ground_truth_np = ground_truth.cpu().numpy()

        c = np.dot(grasp_recon_np.flatten(), ground_truth_np.flatten()) / np.dot(grasp_recon_np.flatten(), grasp_recon_np.flatten())

        grasp_recon = torch.tensor(c * grasp_recon_np, device=device)


        # Convert complex images to magnitude
        gt_mag = torch.sqrt(ground_truth[:, 0, ...]**2 + ground_truth[:, 1, ...]**2)
        grasp_mag = torch.sqrt(grasp_recon[:, 0, ...]**2 + grasp_recon[:, 1, ...]**2)

        # add batch dimension (input shape: B, C, T, H, W)
        grasp_mag = rearrange(grasp_mag, 'c h t w -> c t h w').unsqueeze(0)
        gt_mag = rearrange(gt_mag, 'c t h w -> c t h w').unsqueeze(0)

        # calculate data range from ground truth
        # data_range = gt_mag.max() - gt_mag.min()
        min_val = torch.min(gt_mag).item()
        max_val = torch.max(gt_mag).item()
        data_range = (min_val, max_val)

        ssim_grasp, psnr_grasp, mse_grasp, lpips_grasp = calc_image_metrics(grasp_mag.contiguous(), gt_mag.contiguous(), data_range, device)


        return ssim_grasp, psnr_grasp, mse_grasp, lpips_grasp, dc_mse_grasp, dc_mae_grasp

    else:
        return dc_mse_grasp, dc_mae_grasp



def eval_sample(kspace, csmap, ground_truth, x_recon, physics, mask, grasp_img, acceleration, spokes_per_frame, output_dir, label, device, cluster, dro_eval=True, grasp_path=None, raw_slice_idx=None, rescale=True, filename_suffix=""):

    acceleration = round(acceleration.item(), 1)
    plot_label, patient_id = _resolve_plot_label(label, grasp_path)
    suffix = f"_{filename_suffix}" if filename_suffix else ""

    # ==========================================================
    # EVALUATE DATA CONSISTENCY
    # ==========================================================


    # Forward Simulation
    x_recon_complex = to_torch_complex(x_recon).squeeze()
    kspace = kspace.squeeze()


    recon_kspace = physics(False, x_recon_complex, csmap)


    # Compute MSE
    dc_mse, dc_mae = calc_dc(recon_kspace, kspace, device)


    # RESCALE

    # calculate the single optimal scaling factor 'c'
    x_recon_np = x_recon.cpu().numpy()
    ground_truth_np = ground_truth.cpu().numpy()
    grasp_recon_np = grasp_img.cpu().numpy()

    if rescale:
        c = np.dot(x_recon_np.flatten(), ground_truth_np.flatten()) / np.dot(x_recon_np.flatten(), x_recon_np.flatten())
        recon_complex_scaled = torch.tensor(c * x_recon_np, device=device)
        
        c_grasp = np.dot(grasp_recon_np.flatten(), ground_truth_np.flatten()) / np.dot(grasp_recon_np.flatten(), grasp_recon_np.flatten())
        grasp_img = torch.tensor(c_grasp * grasp_recon_np, device=device)

    else:
        recon_complex_scaled = torch.tensor(x_recon_np, device=device)
        grasp_img = torch.tensor(grasp_recon_np, device=device)


    # Convert complex images to magnitude
    recon_mag_scaled = torch.sqrt(recon_complex_scaled[:, 0, ...]**2 + recon_complex_scaled[:, 1, ...]**2)
    gt_mag = torch.sqrt(ground_truth[:, 0, ...]**2 + ground_truth[:, 1, ...]**2)

    # add batch dimension (input shape: B, C, T, H, W)
    recon_mag_scaled = rearrange(recon_mag_scaled, 'c h w t -> c t h w').unsqueeze(0)
    gt_mag = rearrange(gt_mag, 'c t h w -> c t h w').unsqueeze(0)

    # calculate data range from ground truth
    # data_range = gt_mag.max() - gt_mag.min()
    min_val = torch.min(gt_mag).item()
    max_val = torch.max(gt_mag).item()
    data_range = (min_val, max_val)



    if dro_eval:

        # ==========================================================
        # EVALUATE SPATIAL IMAGE QUALITY
        # ==========================================================

        ssim, psnr, mse, lpips = calc_image_metrics(recon_mag_scaled.contiguous(), gt_mag.contiguous(), data_range, device)


        # ==========================================================
        # VISUALIZATION
        # ==========================================================

        grasp_recon_complex_np = rearrange(to_torch_complex(grasp_img).squeeze(), 'h t w -> h w t').cpu().numpy()
        # grasp_recon_complex_np = to_torch_complex(grasp_img).squeeze().cpu().numpy()
        grasp_mag_np = np.abs(grasp_recon_complex_np)

        x_recon_complex_np = to_torch_complex(recon_complex_scaled).squeeze().cpu().numpy()

        gt_squeezed = ground_truth.squeeze()  # Shape: (C, T, H, W) -> (2, 22, 320, 320)
        gt_rearranged = rearrange(gt_squeezed, 'c t h w -> t c h w') # Shape: (22, 320, 320, 2)
        gt_complex_tensor = to_torch_complex(gt_rearranged) # Shape: (22, 320, 320)
        gt_final_tensor = rearrange(gt_complex_tensor, 't h w -> h w t') # Shape: (320, 320, 22)
        gt_complex_np = gt_final_tensor.cpu().numpy()

        recon_mag_np = np.abs(x_recon_complex_np)
        gt_mag_np = np.abs(gt_complex_np)
        
        masks_np = {key: val.cpu().numpy().squeeze().astype(bool) for key, val in mask.items()}

        num_frames = recon_mag_np.shape[2]

        aif_time_points = np.linspace(0, 150, num_frames)

        if 'malignant' in mask and mask['malignant'].any() and plot_label is not None:
            
            # --- Plot Spatial Quality at a Peak Enhancement Frame ---
            # Find a frame around peak enhancement (e.g., 1/3 of the way through)
            peak_frame = num_frames // 3
            data_range = gt_mag_np[:, :, peak_frame].max() - gt_mag_np[:, :, peak_frame].min()
            plot_spatial_quality(
                recon_img=recon_mag_np[:, :, peak_frame],
                gt_img=gt_mag_np[:, :, peak_frame],
                grasp_img=grasp_mag_np[:, :, peak_frame],
                time_frame_index=peak_frame,
                filename=os.path.join(output_dir, f"spatial_quality_{plot_label}{suffix}.png"),
                grasp_comparison_filename=os.path.join(output_dir, f"grasp_comparison_{plot_label}{suffix}.png"),
                data_range=data_range,
                acceleration=acceleration,
                spokes_per_frame=spokes_per_frame,
                plot_dro=True,
            )

            # --- Plot Temporal Curves for Key Regions ---
            # This is the most important plot for debugging your PK results!
            region_corrs = plot_temporal_curves(
                gt_img_stack=gt_mag_np,
                recon_img_stack=recon_mag_np,
                grasp_img_stack=grasp_mag_np,
                masks=masks_np,
                time_points=aif_time_points,
                filename=os.path.join(output_dir, f"temporal_curves_{plot_label}{suffix}.png"),
                acceleration=acceleration,
                spokes_per_frame=spokes_per_frame,
                plot_dro=True
            )

            plot_single_temporal_curve(
                img_stack=recon_mag_np,
                masks=masks_np,
                time_points=aif_time_points,
                num_frames=num_frames,
                filename=os.path.join(output_dir, f"recon_temporal_curve_{plot_label}{suffix}.png"),
                acceleration=acceleration,
                spokes_per_frame=spokes_per_frame,
            )

            plot_time_series(
                recon_img_stack=recon_mag_np,
                grasp_img_stack=grasp_mag_np,
                filename=os.path.join(output_dir, f"time_points_{plot_label}{suffix}.png"),
                acceleration=acceleration,
                spokes_per_frame=spokes_per_frame,
            )

            print("Diagnostic plots saved.")

        else:
            region_corrs = {'malignant': {'DL': None, 'GRASP': None}}
        
        return ssim, psnr, mse, lpips, dc_mse, dc_mae, region_corrs['malignant']['DL'], region_corrs['malignant']['GRASP']
    

    else:

        # ==========================================================
        # VISUALIZATION
        # ==========================================================
        grasp_recon_complex_np = rearrange(to_torch_complex(grasp_img).squeeze(), 'h t w -> h w t').cpu().numpy()
        grasp_mag_np = np.abs(grasp_recon_complex_np)

        x_recon_complex_np = to_torch_complex(recon_complex_scaled).squeeze().cpu().numpy()

        gt_squeezed = ground_truth.squeeze()  # Shape: (C, T, H, W) -> (2, 22, 320, 320)
        gt_rearranged = rearrange(gt_squeezed, 'c t h w -> t c h w') # Shape: (22, 320, 320, 2)
        gt_complex_tensor = to_torch_complex(gt_rearranged) # Shape: (22, 320, 320)
        gt_final_tensor = rearrange(gt_complex_tensor, 't h w -> h w t') # Shape: (320, 320, 22)
        gt_complex_np = gt_final_tensor.cpu().numpy()

        recon_mag_np = np.abs(x_recon_complex_np)
        gt_mag_np = np.abs(gt_complex_np)

        # For raw data, replace the DRO mask with the correct tumor segmentation when available.
        dro_has_malignant = 'malignant' in mask and mask['malignant'].any()
        slice_map = _load_slice_map()
        resolved_slice_idx = slice_map.get(patient_id, raw_slice_idx)
        raw_tumor_mask = None
        if resolved_slice_idx is not None and resolved_slice_idx >= 0:
            raw_tumor_mask = _load_tumor_mask(cluster, patient_id, slice_idx=resolved_slice_idx)

        if raw_tumor_mask is not None and raw_tumor_mask.any():
            mask = {'malignant': torch.from_numpy(raw_tumor_mask.astype(np.bool_))}
        else:
            # Treat as non-malignant if mask is missing or empty.
            if dro_has_malignant and resolved_slice_idx is not None and resolved_slice_idx >= 0:
                print(f"Warning: malignant DRO label but empty/missing tumor mask for {patient_id} (slice {resolved_slice_idx}); skipping temporal plots.")
            mask = {}

        masks_np = {key: val.cpu().numpy().squeeze().astype(bool) for key, val in mask.items() if key == 'malignant'}

        num_frames = recon_mag_np.shape[2]

        aif_time_points = np.linspace(0, 150, num_frames)

        if 'malignant' in mask and mask['malignant'].any() and plot_label is not None:
            
            # --- Plot Spatial Quality at a Peak Enhancement Frame ---
            # Find a frame around peak enhancement (e.g., 1/3 of the way through)
            peak_frame = num_frames // 3
            data_range = gt_mag_np[:, :, peak_frame].max() - gt_mag_np[:, :, peak_frame].min()
            plot_spatial_quality(
                recon_img=recon_mag_np[:, :, peak_frame],
                gt_img=gt_mag_np[:, :, peak_frame],
                grasp_img=grasp_mag_np[:, :, peak_frame],
                time_frame_index=peak_frame,
                filename=os.path.join(output_dir, f"non_dro_spatial_quality_{plot_label}{suffix}.png"),
                grasp_comparison_filename=os.path.join(output_dir, f"non_dro_grasp_comparison_{plot_label}{suffix}.png"),
                data_range=data_range,
                acceleration=acceleration,
                spokes_per_frame=spokes_per_frame,
                plot_dro=False,
            )

            # --- Plot Temporal Curves for Key Regions ---
            # This is the most important plot for debugging your PK results!
            _ = plot_temporal_curves(
                gt_img_stack=gt_mag_np,
                recon_img_stack=recon_mag_np,
                grasp_img_stack=grasp_mag_np,
                masks=masks_np,
                time_points=aif_time_points,
                filename=os.path.join(output_dir, f"non_dro_temporal_curves_{plot_label}{suffix}.png"),
                acceleration=acceleration,
                spokes_per_frame=spokes_per_frame,
                plot_dro=False
            )

            plot_single_temporal_curve(
                img_stack=recon_mag_np,
                masks=masks_np,
                time_points=aif_time_points,
                num_frames=num_frames,
                filename=os.path.join(output_dir, f"non_dro_recon_temporal_curve_{plot_label}{suffix}.png"),
                acceleration=acceleration,
                spokes_per_frame=spokes_per_frame,
            )

            plot_time_series(
                recon_img_stack=recon_mag_np,
                grasp_img_stack=grasp_mag_np,
                filename=os.path.join(output_dir, f"non_dro_time_points_{plot_label}{suffix}.png"),
                acceleration=acceleration,
                spokes_per_frame=spokes_per_frame,
            )

            print("Diagnostic plots saved.")


        return dc_mse, dc_mae





def eval_sample_no_grasp(kspace, csmap, ground_truth, x_recon, physics, mask, acceleration, spokes_per_frame, output_dir, label, device):

    acceleration = round(acceleration.item(), 1)

    # ground_truth = ground_truth.to(device) # Shape: (1, 2, T, H, W)
    # grasp_recon = grasp_recon.to(device) # Shape: (1, 2, H, T, W)

    # ==========================================================
    # EVALUATE DATA CONSISTENCY
    # ==========================================================


    # Forward Simulation
    x_recon_complex = to_torch_complex(x_recon).squeeze()
    kspace = kspace.squeeze()


    recon_kspace = physics(False, x_recon_complex, csmap)


    # Compute MSE
    dc_mse, dc_mae = calc_dc(recon_kspace, kspace, device)


    # ==========================================================
    # EVALUATE SPATIAL IMAGE QUALITY
    # ==========================================================

    # calculate the single optimal scaling factor 'c'
    x_recon_np = x_recon.cpu().numpy()
    ground_truth_np = ground_truth.cpu().numpy()


    c = np.dot(x_recon_np.flatten(), ground_truth_np.flatten()) / np.dot(x_recon_np.flatten(), x_recon_np.flatten())

    recon_complex_scaled = torch.tensor(c * x_recon_np, device=device)


    # Convert complex images to magnitude
    recon_mag_scaled = torch.sqrt(recon_complex_scaled[:, 0, ...]**2 + recon_complex_scaled[:, 1, ...]**2)
    gt_mag = torch.sqrt(ground_truth[:, 0, ...]**2 + ground_truth[:, 1, ...]**2)

    # add batch dimension (input shape: B, C, T, H, W)
    recon_mag_scaled = rearrange(recon_mag_scaled, 'c h w t -> c t h w').unsqueeze(0)
    gt_mag = rearrange(gt_mag, 'c t h w -> c t h w').unsqueeze(0)

    # calculate data range from ground truth
    # data_range = gt_mag.max() - gt_mag.min()
    min_val = torch.min(gt_mag).item()
    max_val = torch.max(gt_mag).item()
    data_range = (min_val, max_val)

    ssim, psnr, mse, lpips = calc_image_metrics(recon_mag_scaled.contiguous(), gt_mag.contiguous(), data_range, device)


    # ssims = []
    # psnrs = []
    # mses = []

    # for t in range(recon_mag_scaled.shape[-1]): # Iterate over time frames


    #     frame_recon = recon_mag_scaled[..., t]
    #     frame_gt = gt_mag[:, t, :, :]

    #     # calculate data range from ground truth
    #     data_range = frame_gt.max() - frame_gt.min()


    #     # Add channel dimension for torchmetrics: (B, H, W) -> (B, 1, H, W)
    #     frame_recon = frame_recon.unsqueeze(1)
    #     frame_gt = frame_gt.unsqueeze(1)
        
    #     # Calculate Spatial Image Quality Metrics
    #     filename=os.path.join(output_dir, f"recon_metric_inputs.png")
    #     ssim, psnr, mse = calc_image_metrics(frame_recon, frame_gt, data_range, device, filename)
    #     ssims.append(ssim)
    #     psnrs.append(psnr)
    #     mses.append(mse)


    # ==========================================================
    # VISUALIZATION
    # ==========================================================


    x_recon_complex_np = to_torch_complex(recon_complex_scaled).squeeze().cpu().numpy()

    gt_squeezed = ground_truth.squeeze()  # Shape: (C, T, H, W) -> (2, 22, 320, 320)
    gt_rearranged = rearrange(gt_squeezed, 'c t h w -> t c h w') # Shape: (22, 320, 320, 2)
    gt_complex_tensor = to_torch_complex(gt_rearranged) # Shape: (22, 320, 320)
    gt_final_tensor = rearrange(gt_complex_tensor, 't h w -> h w t') # Shape: (320, 320, 22)
    gt_complex_np = gt_final_tensor.cpu().numpy()

    recon_mag_np = np.abs(x_recon_complex_np)
    gt_mag_np = np.abs(gt_complex_np)
    
    masks_np = {key: val.cpu().numpy().squeeze().astype(bool) for key, val in mask.items()}

    num_frames = recon_mag_np.shape[2]

    aif_time_points = np.linspace(0, 150, num_frames)

    # ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=data_range).to(device)
    # recon_mag_scaled = rearrange(recon_mag_scaled.squeeze(), 't h w -> h w t')
    # test_ssim = ssim(recon_mag_scaled.unsqueeze(0), torch.tensor(recon_mag_np, device=recon_mag_scaled.device).unsqueeze(0))
    # print(f"---- Debugging step: SSIM between ssim input and plot input: {test_ssim}")

    

    print("\nGenerating diagnostic plots...")
    if 'malignant' in mask and mask['malignant'].any() and label is not None:
        
        # --- Plot Spatial Quality at a Peak Enhancement Frame ---
        # Find a frame around peak enhancement (e.g., 1/3 of the way through)
        peak_frame = num_frames // 3
        data_range = gt_mag_np[:, :, peak_frame].max() - gt_mag_np[:, :, peak_frame].min()

        plot_single_temporal_curve(
            img_stack=recon_mag_np,
            masks=masks_np,
            time_points=aif_time_points,
            num_frames=num_frames,
            filename=os.path.join(output_dir, f"recon_temporal_curve_{label}.png"),
            acceleration=acceleration,
            spokes_per_frame=spokes_per_frame,
        )

        print("Diagnostic plots saved.")


    
    
    return ssim, psnr, mse, lpips, dc_mse, dc_mae
