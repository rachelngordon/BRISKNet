import os
import csv
import warnings
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
from radial_lsfp import MCNUFFT
import numpy as np
from scipy.optimize import curve_fit
from transform import estimate_bolus_arrival_index
from scipy.interpolate import PchipInterpolator
from tqdm import tqdm # A library for a nice progress bar
from scipy.stats import mannwhitneyu
from skimage.metrics import structural_similarity as ssim_map_func
import matplotlib.gridspec as gridspec
from skimage.measure import find_contours
from typing import List, Dict, Optional, Tuple, Callable
from scipy.stats import pearsonr
import nibabel as nib
import pandas as pd
from functools import lru_cache
from cluster_paths import _swap_base
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import binary_fill_holes, label as nd_label
from skimage.filters import threshold_otsu



TUMOR_SEG_ROOT = os.environ.get("TUMOR_SEG_ROOT", "/net/scratch2/rachelgordon/zf_data_192_slices/tumor_segmentations_lcr")
TUMOR_SEG_WARN = os.environ.get("TUMOR_SEG_WARN", "1") not in ("0", "false", "False")
_MISSING_TUMOR_SEGS = set()
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

def _safe_pearsonr(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return float("nan")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    corr, _ = pearsonr(x, y)
    return float(corr)


def _safe_tight_layout(fig, **kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=r".*Axes that are not compatible with tight_layout.*",
        )
        fig.tight_layout(**kwargs)


def robust_window(img, p_low=1, p_high=99):
    lo, hi = np.percentile(img, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def robust_window_multi(images, p_low=1, p_high=99.5):
    """Compute a shared display window across multiple images."""
    flat = []
    for img in images:
        if img is None:
            continue
        arr = np.asarray(img).ravel()
        if arr.size:
            flat.append(arr)
    if not flat:
        return 0.0, 1.0
    stacked = np.concatenate(flat)
    lo, hi = np.percentile(stacked, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def _infer_foreground_mask_from_stack(img_stack: np.ndarray) -> tuple[np.ndarray | None, float | None]:
    """Infer a coarse foreground/tissue mask from a (H, W, T) magnitude stack."""
    if img_stack is None:
        return None, None
    stack = np.asarray(img_stack)
    if stack.ndim != 3:
        return None, None
    if not np.isfinite(stack).any():
        return None, None

    # Use a max-projection to capture the body/breast across time.
    proj = np.nanmax(stack, axis=2)
    proj = np.asarray(proj, dtype=np.float32)
    proj[np.isnan(proj)] = 0.0
    max_val = float(np.max(proj))
    if not np.isfinite(max_val) or max_val <= 0:
        return None, None

    # Otsu threshold tends to separate background/noise from anatomy reasonably well.
    try:
        vals = proj[np.isfinite(proj)]
        thr = float(threshold_otsu(vals))
    except Exception:
        thr = 0.05 * max_val
    thr = max(0.0, min(thr, 0.9 * max_val))
    mask = proj > thr

    # Clean up: fill holes + keep largest connected component (reduces speckle in background).
    try:
        mask = binary_fill_holes(mask)
        labeled, n = nd_label(mask)
        if n > 1:
            counts = np.bincount(labeled.ravel())
            counts[0] = 0
            largest = int(np.argmax(counts))
            mask = labeled == largest
    except Exception:
        pass

    frac = float(mask.mean()) if mask.size else None
    if frac is None or frac <= 0 or frac >= 0.999:
        return None, None
    return mask.astype(bool), frac


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
    min_val, max_val = data_range

    # --- Initialize Metrics ---
    # We will compute metrics frame by frame. data_range is important for PSNR.
    ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=(max_val-min_val)).to(device)
    psnr = torchmetrics.image.PeakSignalNoiseRatio(data_range=(max_val-min_val)).to(device)
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


def _infer_kspace_readout_axis(
    ref_abs: torch.Tensor,
    expected_readout_samples: Optional[int] = None,
) -> Optional[int]:
    if ref_abs.ndim == 0:
        return None

    candidate_axes = [ax for ax, size in enumerate(ref_abs.shape) if int(size) > 1]
    if not candidate_axes:
        return None

    if expected_readout_samples is not None:
        try:
            expected_n = int(expected_readout_samples)
        except (TypeError, ValueError):
            expected_n = -1
        if expected_n > 1:
            matching = [ax for ax in candidate_axes if int(ref_abs.shape[ax]) == expected_n]
            if len(matching) == 1:
                return matching[0]
            if matching:
                candidate_axes = matching

    best_axis = None
    best_score = -float("inf")
    for ax in candidate_axes:
        n = int(ref_abs.shape[ax])
        if n < 8:
            continue
        reduce_dims = tuple(i for i in range(ref_abs.ndim) if i != ax)
        profile = ref_abs.mean(dim=reduce_dims) if reduce_dims else ref_abs
        if profile.numel() == 0:
            continue
        profile = profile.detach().float()

        center = (n - 1) / 2.0
        peak_idx = int(torch.argmax(profile).item())
        dist_norm = abs(peak_idx - center) / max(center, 1.0)

        center_idx = int(round(center))
        lo = max(0, center_idx - 1)
        hi = min(n, center_idx + 2)
        center_mean = float(profile[lo:hi].mean().item()) if hi > lo else 0.0
        edge_k = max(1, n // 10)
        edge_vals = torch.cat([profile[:edge_k], profile[-edge_k:]], dim=0)
        edge_mean = float(edge_vals.mean().item()) if edge_vals.numel() > 0 else 0.0

        contrast = center_mean / (edge_mean + 1e-12)
        score = contrast - 2.0 * dist_norm
        if score > best_score:
            best_score = score
            best_axis = ax

    if best_axis is None:
        # Fall back to largest axis if heuristic fails.
        best_axis = max(candidate_axes, key=lambda ax: int(ref_abs.shape[ax]))
    return int(best_axis)


def compute_kspace_residual_metrics(
    pred_kspace: torch.Tensor,
    ref_kspace: torch.Tensor,
    expected_readout_samples: Optional[int] = None,
) -> Dict[str, float]:
    """Compute normalized k-space residual metrics on full, low/mid/high radial bands."""
    if (not torch.is_complex(pred_kspace)) or (not torch.is_complex(ref_kspace)):
        return {}
    if pred_kspace.shape != ref_kspace.shape:
        return {}

    pred = pred_kspace
    ref = ref_kspace
    finite_mask = (
        torch.isfinite(pred.real)
        & torch.isfinite(pred.imag)
        & torch.isfinite(ref.real)
        & torch.isfinite(ref.imag)
    )
    if not bool(torch.any(finite_mask)):
        return {}

    diff = pred - ref
    diff_sq = torch.abs(diff) ** 2
    ref_sq = torch.abs(ref) ** 2

    finite_f = finite_mask.to(diff_sq.dtype)
    diff_sq = diff_sq * finite_f
    ref_sq = ref_sq * finite_f

    total_diff = float(diff_sq.sum().item())
    total_ref = float(ref_sq.sum().item())
    rel_l2 = float(np.sqrt(total_diff / (total_ref + 1e-12)))

    metrics: Dict[str, float] = {"rel_l2": rel_l2}

    readout_axis = _infer_kspace_readout_axis(
        ref_abs=torch.sqrt(ref_sq + 1e-12),
        expected_readout_samples=expected_readout_samples,
    )
    if readout_axis is None:
        return metrics

    n = int(ref.shape[readout_axis])
    if n <= 1:
        return metrics

    idx = torch.arange(n, device=ref.device, dtype=diff_sq.dtype)
    center = (n - 1) / 2.0
    denom = center if center > 0 else 1.0
    radius = torch.abs(idx - center) / denom
    radius = torch.clamp(radius, 0.0, 1.0)

    shape = [1] * ref.ndim
    shape[readout_axis] = n
    radius = radius.view(shape)

    bands = (
        ("low", 0.0, 1.0 / 3.0),
        ("mid", 1.0 / 3.0, 2.0 / 3.0),
        ("high", 2.0 / 3.0, 1.000001),
    )
    for band_name, lo, hi in bands:
        band_mask = (radius >= lo) & (radius < hi)
        band_mask = band_mask.to(diff_sq.dtype) * finite_f
        band_diff = float((diff_sq * band_mask).sum().item())
        band_ref = float((ref_sq * band_mask).sum().item())
        if band_ref <= 0:
            metrics[f"rel_l2_{band_name}"] = float("nan")
        else:
            metrics[f"rel_l2_{band_name}"] = float(np.sqrt(band_diff / (band_ref + 1e-12)))

    metrics["readout_axis"] = int(readout_axis)
    return metrics


def _best_fit_complex_scale(pred: torch.Tensor, ref: torch.Tensor) -> complex | None:
    """Least-squares complex scalar c minimizing ||c*pred - ref||_2.

    Returns Python complex, or None if pred energy is ~0.
    """
    pred_flat = pred.reshape(-1)
    ref_flat = ref.reshape(-1)
    denom = torch.sum(torch.conj(pred_flat) * pred_flat)
    if torch.abs(denom) < 1e-12:
        return None
    numer = torch.sum(torch.conj(pred_flat) * ref_flat)
    return (numer / denom).item()


def calc_dc_bestfit(pred: torch.Tensor, ref: torch.Tensor, device):
    """Compute DC metrics after applying a best-fit complex scalar gain."""
    c = _best_fit_complex_scale(pred, ref)
    if c is None:
        return None, None, None
    c_t = torch.as_tensor(c, dtype=pred.dtype, device=pred.device)
    pred_scaled = pred * c_t
    dc_mse, dc_mae = calc_dc(pred_scaled, ref, device)
    return dc_mse, dc_mae, c


def _standardize_kspace_for_ssdu(kspace: torch.Tensor, spokes_per_frame: int) -> Tuple[torch.Tensor, int]:
    kspace = kspace.squeeze()
    if kspace.ndim == 5:
        if kspace.shape[0] != 1:
            raise ValueError(f"SSDU expects batch size 1, got {kspace.shape}")
        kspace = kspace.squeeze(0)

    if kspace.ndim == 4:
        if kspace.shape[1] == spokes_per_frame:
            kspace_std = kspace
        elif kspace.shape[0] == spokes_per_frame:
            kspace_std = kspace.permute(1, 0, 2, 3)
        elif kspace.shape[2] == spokes_per_frame:
            if kspace.shape[3] > kspace.shape[1]:
                kspace_std = kspace.permute(1, 2, 3, 0)
            else:
                kspace_std = kspace.permute(0, 2, 1, 3)
        elif kspace.shape[3] == spokes_per_frame:
            kspace_std = kspace.permute(1, 2, 3, 0)
        else:
            raise ValueError(f"Unsupported 4D k-space shape for SSDU: {kspace.shape}")
    elif kspace.ndim == 3:
        if kspace.shape[1] % spokes_per_frame == 0:
            samples_per_spoke = kspace.shape[1] // spokes_per_frame
            kspace_std = kspace.reshape(kspace.shape[0], spokes_per_frame, samples_per_spoke, kspace.shape[2])
        elif kspace.shape[2] % spokes_per_frame == 0:
            samples_per_spoke = kspace.shape[2] // spokes_per_frame
            kspace_std = kspace.permute(1, 2, 0).reshape(kspace.shape[1], spokes_per_frame, samples_per_spoke, kspace.shape[0])
        else:
            raise ValueError(f"Unsupported 3D k-space shape for SSDU: {kspace.shape}")
    else:
        raise ValueError(f"Unsupported k-space shape for SSDU: {kspace.shape}")

    samples_per_spoke = kspace_std.shape[2]
    return kspace_std, samples_per_spoke


def _standardize_ktraj_for_ssdu(ktraj: torch.Tensor, M: int, T: int) -> torch.Tensor:
    if ktraj.ndim != 3:
        raise ValueError(f"SSDU expects 3D ktraj, got {ktraj.shape}")

    if ktraj.shape[0] == 2:
        if ktraj.shape[1] == M and ktraj.shape[2] == T:
            return ktraj
        if ktraj.shape[1] == T and ktraj.shape[2] == M:
            return ktraj.permute(0, 2, 1)
    elif ktraj.shape[2] == 2:
        if ktraj.shape[0] == M and ktraj.shape[1] == T:
            return ktraj.permute(2, 0, 1)
        if ktraj.shape[0] == T and ktraj.shape[1] == M:
            return ktraj.permute(2, 1, 0)

    raise ValueError(f"Unsupported ktraj shape for SSDU: {ktraj.shape}")


def _standardize_dcomp_for_ssdu(dcomp: torch.Tensor, M: int, T: int) -> torch.Tensor:
    if dcomp.ndim == 2:
        if dcomp.shape[0] == M and dcomp.shape[1] == T:
            return dcomp
        if dcomp.shape[0] == T and dcomp.shape[1] == M:
            return dcomp.permute(1, 0)
    elif dcomp.ndim == 1 and dcomp.shape[0] == M:
        return dcomp.unsqueeze(1).expand(M, T)

    raise ValueError(f"Unsupported dcomp shape for SSDU: {dcomp.shape}")


def _build_ssdu_fold_indices(
    spokes_per_frame: int,
    samples_per_spoke: int,
    K_folds: int,
    device: torch.device,
    allow_single_spoke: bool,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    effective_K = min(K_folds, spokes_per_frame)
    if effective_K < 2:
        return []

    M = spokes_per_frame * samples_per_spoke
    fold_indices = []
    for fold_idx in range(effective_K):
        held_spokes = torch.arange(fold_idx, spokes_per_frame, effective_K, device=device)
        used_spokes = spokes_per_frame - held_spokes.numel()
        if used_spokes < 2 and not allow_single_spoke:
            continue

        held_idx = (held_spokes[:, None] * samples_per_spoke + torch.arange(samples_per_spoke, device=device)[None, :]).reshape(-1)
        held_mask = torch.zeros(M, dtype=torch.bool, device=device)
        held_mask[held_idx] = True
        used_idx = (~held_mask).nonzero(as_tuple=False).squeeze(-1)
        fold_indices.append((held_idx, used_idx))

    return fold_indices


def _apply_grasp_orientation(
    img: torch.Tensor,
    orientation_transform: Optional[object],
) -> torch.Tensor:
    if orientation_transform is None or orientation_transform == "none":
        return img
    if callable(orientation_transform):
        return orientation_transform(img)
    if not isinstance(orientation_transform, str):
        raise ValueError("SSDU GRASP orientation_transform must be a callable or string.")

    img_complex = img
    if img_complex.ndim == 3 and img_complex.shape[0] != img_complex.shape[1]:
        img_thw = img_complex
    elif img_complex.ndim == 3:
        img_thw = img_complex.permute(2, 0, 1)
    else:
        raise ValueError(f"Unsupported GRASP image shape for orientation: {img_complex.shape}")

    img_ri = torch.stack([img_thw.real, img_thw.imag], dim=0)
    img_ri = torch.flip(img_ri, dims=[1])

    if orientation_transform == "raw_grasp":
        img_ri = torch.rot90(img_ri, k=1, dims=[1, 3])
    elif orientation_transform == "dro_grasp":
        img_ri = torch.rot90(img_ri, k=3, dims=[1, 3])
    else:
        raise ValueError(f"Unsupported GRASP orientation_transform: {orientation_transform}")

    img_out = img_ri[0] + 1j * img_ri[1]
    return img_out.permute(1, 2, 0)


@torch.no_grad()
def compute_ssdu_kspace_nmse(
    model,
    kspace: torch.Tensor,
    csmap: torch.Tensor,
    ktraj: torch.Tensor,
    dcomp: torch.Tensor,
    nufft_ob,
    adjnufft_ob,
    spokes_per_frame: int,
    K_folds: int = 4,
    baseline_weighting: str = "sqrt_dcomp",
    device: Optional[torch.device] = None,
    acceleration_encoding: Optional[torch.Tensor] = None,
    start_timepoint_index: Optional[torch.Tensor] = None,
    norm: str = "both",
    epoch: str = "inference",
    chunk_size: Optional[int] = None,
    chunk_overlap: int = 0,
    allow_single_spoke: bool = False,
) -> Dict[str, object]:
    if device is None:
        device = kspace.device

    if csmap.ndim == 3:
        csmap = csmap.unsqueeze(0)
    csmap = csmap.to(device)

    kspace_std, samples_per_spoke = _standardize_kspace_for_ssdu(kspace, spokes_per_frame)
    C, Sp, Samp, T = kspace_std.shape
    kspace_flat = kspace_std.reshape(C, Sp * Samp, T).to(device)

    M = kspace_flat.shape[1]
    ktraj_std = _standardize_ktraj_for_ssdu(ktraj.to(device), M, T)
    dcomp_std = _standardize_dcomp_for_ssdu(dcomp.to(device), M, T)

    fold_indices = _build_ssdu_fold_indices(
        spokes_per_frame,
        samples_per_spoke,
        K_folds,
        device,
        allow_single_spoke,
    )
    if not fold_indices:
        return {"ssdu_nmse_mean": float("nan"), "ssdu_nmse_folds": [], "ssdu_nmse_per_frame": None}

    fold_nmse = []
    fold_nmse_per_frame = []

    for held_idx, used_idx in fold_indices:
        y_used = kspace_flat[:, used_idx, :]
        y_held = kspace_flat[:, held_idx, :]

        ktraj_used = ktraj_std[:, used_idx, :]
        ktraj_held = ktraj_std[:, held_idx, :]
        dcomp_used = dcomp_std[used_idx, :]
        dcomp_held = dcomp_std[held_idx, :]

        physics_used = MCNUFFT(nufft_ob, adjnufft_ob, ktraj_used, dcomp_used)
        physics_held = MCNUFFT(nufft_ob, adjnufft_ob, ktraj_held, dcomp_held)

        if chunk_size is not None and T > chunk_size:
            from utils import sliding_window_inference

            H, W = csmap.shape[-2], csmap.shape[-1]
            x_hat, _ = sliding_window_inference(
                H,
                W,
                T,
                ktraj_used,
                dcomp_used,
                nufft_ob,
                adjnufft_ob,
                chunk_size,
                chunk_overlap,
                y_used,
                csmap,
                acceleration_encoding,
                start_timepoint_index,
                model,
                epoch=epoch,
                device=device,
                norm=norm,
            )
        else:
            x_hat, *_ = model(
                y_used,
                physics_used,
                csmap,
                acceleration_encoding,
                start_timepoint_index,
                epoch=epoch,
                norm=norm,
            )

        x_hat_complex = to_torch_complex(x_hat).squeeze(0)
        y_hat_held = physics_held(False, x_hat_complex, csmap)

        if baseline_weighting == "sqrt_dcomp":
            weight = torch.sqrt(torch.abs(dcomp_held)).unsqueeze(0)
        else:
            weight = 1.0

        diff = y_hat_held - y_held
        if isinstance(weight, torch.Tensor):
            num = torch.sum(torch.abs(weight * diff) ** 2)
            den = torch.sum(torch.abs(weight * y_held) ** 2)
            num_t = torch.sum(torch.abs(weight * diff) ** 2, dim=(0, 1))
            den_t = torch.sum(torch.abs(weight * y_held) ** 2, dim=(0, 1))
        else:
            num = torch.sum(torch.abs(diff) ** 2)
            den = torch.sum(torch.abs(y_held) ** 2)
            num_t = torch.sum(torch.abs(diff) ** 2, dim=(0, 1))
            den_t = torch.sum(torch.abs(y_held) ** 2, dim=(0, 1))

        nmse_fold = (num / (den + 1e-8)).item()
        nmse_per_frame = (num_t / (den_t + 1e-8)).detach().cpu()

        fold_nmse.append(nmse_fold)
        fold_nmse_per_frame.append(nmse_per_frame)

    if not fold_nmse:
        return {"ssdu_nmse_mean": float("nan"), "ssdu_nmse_folds": [], "ssdu_nmse_per_frame": None}

    ssdu_nmse_mean = float(np.mean(fold_nmse))
    ssdu_nmse_per_frame = None
    if fold_nmse_per_frame:
        ssdu_nmse_per_frame = torch.stack(fold_nmse_per_frame, dim=0).mean(dim=0).cpu().numpy()

    return {
        "ssdu_nmse_mean": ssdu_nmse_mean,
        "ssdu_nmse_folds": fold_nmse,
        "ssdu_nmse_per_frame": ssdu_nmse_per_frame,
    }


@torch.no_grad()
def compute_ssdu_kspace_nmse_grasp(
    grasp_recon_fn: Callable[..., torch.Tensor],
    kspace: torch.Tensor,
    csmap: torch.Tensor,
    ktraj: torch.Tensor,
    dcomp: torch.Tensor,
    nufft_ob,
    adjnufft_ob,
    spokes_per_frame: int,
    K_folds: int = 2,
    orientation_transform: Optional[object] = "raw_grasp",
    baseline_weighting: str = "sqrt_dcomp",
    device: Optional[torch.device] = None,
    allow_single_spoke: bool = False,
) -> Dict[str, object]:
    if device is None:
        device = kspace.device

    if csmap.ndim == 3:
        csmap = csmap.unsqueeze(0)
    csmap = csmap.to(device)

    kspace_std, samples_per_spoke = _standardize_kspace_for_ssdu(kspace, spokes_per_frame)
    C, Sp, Samp, T = kspace_std.shape
    kspace_flat = kspace_std.reshape(C, Sp * Samp, T).to(device)

    M = kspace_flat.shape[1]
    ktraj_std = _standardize_ktraj_for_ssdu(ktraj.to(device), M, T)
    dcomp_std = _standardize_dcomp_for_ssdu(dcomp.to(device), M, T)

    fold_indices = _build_ssdu_fold_indices(
        spokes_per_frame,
        samples_per_spoke,
        K_folds,
        device,
        allow_single_spoke,
    )
    if not fold_indices:
        return {"ssdu_nmse_mean": float("nan"), "ssdu_nmse_folds": [], "ssdu_nmse_per_frame": None}

    fold_nmse = []
    fold_nmse_per_frame = []

    for held_idx, used_idx in fold_indices:
        y_used = kspace_flat[:, used_idx, :]
        y_held = kspace_flat[:, held_idx, :]

        ktraj_used = ktraj_std[:, used_idx, :]
        ktraj_held = ktraj_std[:, held_idx, :]
        dcomp_used = dcomp_std[used_idx, :]
        dcomp_held = dcomp_std[held_idx, :]

        x_grasp = grasp_recon_fn(
            y_used,
            ktraj_used,
            dcomp_used,
            csmap,
            samples_per_spoke=samples_per_spoke,
        )
        if not torch.is_tensor(x_grasp):
            x_grasp = torch.tensor(x_grasp)
        if not torch.is_complex(x_grasp):
            raise ValueError("SSDU GRASP requires complex-valued image output.")

        x_grasp = x_grasp.to(device)
        x_grasp = rearrange(x_grasp, 't h w -> h w t')

        # x_grasp = _apply_grasp_orientation(x_grasp, orientation_transform)

        physics_held = MCNUFFT(nufft_ob, adjnufft_ob, ktraj_held, dcomp_held)
        if not hasattr(physics_held, "forward"):
            raise AttributeError("SSDU GRASP requires MCNUFFT to implement forward().")
        
        y_hat_held = physics_held(False, x_grasp, csmap.to(x_grasp.dtype))

        if baseline_weighting == "sqrt_dcomp":
            weight = torch.sqrt(torch.abs(dcomp_held)).unsqueeze(0)
        else:
            weight = 1.0

        diff = y_hat_held - y_held
        if isinstance(weight, torch.Tensor):
            num = torch.sum(torch.abs(weight * diff) ** 2)
            den = torch.sum(torch.abs(weight * y_held) ** 2)
            num_t = torch.sum(torch.abs(weight * diff) ** 2, dim=(0, 1))
            den_t = torch.sum(torch.abs(weight * y_held) ** 2, dim=(0, 1))
        else:
            num = torch.sum(torch.abs(diff) ** 2)
            den = torch.sum(torch.abs(y_held) ** 2)
            num_t = torch.sum(torch.abs(diff) ** 2, dim=(0, 1))
            den_t = torch.sum(torch.abs(y_held) ** 2, dim=(0, 1))

        nmse_fold = (num / (den + 1e-8)).item()
        nmse_per_frame = (num_t / (den_t + 1e-8)).detach().cpu()

        fold_nmse.append(nmse_fold)
        fold_nmse_per_frame.append(nmse_per_frame)

    if not fold_nmse:
        return {"ssdu_nmse_mean": float("nan"), "ssdu_nmse_folds": [], "ssdu_nmse_per_frame": None}

    ssdu_nmse_mean = float(np.mean(fold_nmse))
    ssdu_nmse_per_frame = None
    if fold_nmse_per_frame:
        ssdu_nmse_per_frame = torch.stack(fold_nmse_per_frame, dim=0).mean(dim=0).cpu().numpy()

    return {
        "ssdu_nmse_mean": ssdu_nmse_mean,
        "ssdu_nmse_folds": fold_nmse,
        "ssdu_nmse_per_frame": ssdu_nmse_per_frame,
    }


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
        if TUMOR_SEG_WARN and seg_path not in _MISSING_TUMOR_SEGS:
            print(f"Tumor segmentation not found at {seg_path}")
            _MISSING_TUMOR_SEGS.add(seg_path)
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
    tumor_mask: np.ndarray | None = None,
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

    contours = None
    if tumor_mask is not None and np.any(tumor_mask):
        contours = find_contours(tumor_mask, 0.5)

    def _overlay_contours(ax):
        if not contours:
            return
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='red')

    # Compute per-image windows to maximize contrast in each panel.
    vmin_gt, vmax_gt = robust_window(gt_img, p_low=1, p_high=99.5)
    vmin_recon, vmax_recon = robust_window(recon_img, p_low=1, p_high=99.5)
    vmin_grasp, vmax_grasp = robust_window(grasp_img, p_low=1, p_high=99.5)
    # Use fixed window for error maps (original scaling).

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
        # cax_err_dl = fig.add_subplot(gs[0, 3])
        axes[0, 3] = fig.add_subplot(gs[0, 4])
        # cax_ssim_dl = fig.add_subplot(gs[0, 5])
        axes[1, 0] = fig.add_subplot(gs[1, 0])
        axes[1, 1] = fig.add_subplot(gs[1, 1])
        axes[1, 2] = fig.add_subplot(gs[1, 2])
        # cax_err_grasp = fig.add_subplot(gs[1, 3])
        axes[1, 3] = fig.add_subplot(gs[1, 4])
        # cax_ssim_grasp = fig.add_subplot(gs[1, 5])
        fig.suptitle(
            f"Spatial Quality Comparison at Time Frame {time_frame_index} with AF {acceleration} and SPF {spokes_per_frame}",
            fontsize=PLOT_FONT_SIZES["suptitle"],
            y=0.95,
        )

        # --- Top Row: BRISKNet Comparison ---

        axes[0, 0].imshow(gt_img, cmap='gray', vmin=vmin_gt, vmax=vmax_gt)
        _overlay_contours(axes[0, 0])
        axes[0, 0].set_title(r"$|\mathrm{DRO}|$", fontsize=PLOT_FONT_SIZES["title"])

        axes[0, 1].imshow(recon_img, cmap='gray', vmin=vmin_recon, vmax=vmax_recon)
        _overlay_contours(axes[0, 1])
        axes[0, 1].set_title(r"$|\mathrm{BRISKNet}_{\mathrm{pred}}|$", fontsize=PLOT_FONT_SIZES["title"])

        im_err_dl = axes[0, 2].imshow(error_map_dl, cmap='coolwarm', vmin=-0.5, vmax=0.5)
        axes[0, 2].set_title(r"$|\mathrm{BRISKNet}_{\mathrm{pred}}| - |\mathrm{DRO}|$", fontsize=PLOT_FONT_SIZES["title"])

        div = make_axes_locatable(axes[0, 2])
        cax_err = div.append_axes("right", size="4%", pad=0.04)
        cb_err = fig.colorbar(im_err_dl, cax=cax_err)
        cb_err.ax.tick_params(labelsize=PLOT_FONT_SIZES["tick"])
        cb_err.set_label(r"$|\mathrm{BRISKNet}| - |\mathrm{DRO}|$", fontsize=PLOT_FONT_SIZES["tick"])


        # cb_err_dl = fig.colorbar(im_err_dl, cax=cax_err_dl)
        # cb_err_dl.ax.tick_params(labelsize=PLOT_FONT_SIZES["tick"])
        # cb_err_dl.set_label(
        #     r"$|\mathrm{BRISKNet}| - |\mathrm{DRO}|$",
        #     fontsize=PLOT_FONT_SIZES["tick"]
        # )


        im_ssim_dl = axes[0, 3].imshow(ssim_map_dl, cmap='viridis', vmin=0, vmax=1)
        axes[0, 3].set_title(
            rf"$\mathrm{{SSIM}}_{{\mathrm{{BRISKNet}}}}$ ({ssim_dl:.3f})",
            fontsize=PLOT_FONT_SIZES["title"],
        )

        div = make_axes_locatable(axes[0, 3])
        cax_ssim = div.append_axes("right", size="4%", pad=0.04)
        cb_ssim = fig.colorbar(im_ssim_dl, cax=cax_ssim)
        cb_ssim.ax.tick_params(labelsize=PLOT_FONT_SIZES["tick"])
        cb_ssim.set_label(r"$\mathrm{SSIM}$", fontsize=PLOT_FONT_SIZES["tick"])

        # cb_ssim_dl = fig.colorbar(im_ssim_dl, cax=cax_ssim_dl)
        # cb_ssim_dl.ax.tick_params(labelsize=PLOT_FONT_SIZES["tick"])
        # cb_ssim_dl.set_label(
        #     r"$\mathrm{SSIM}$",
        #     fontsize=PLOT_FONT_SIZES["tick"]
        # )


        # --- Bottom Row: GRASP Reconstruction Comparison ---
        axes[1, 0].imshow(gt_img, cmap='gray', vmin=vmin_gt, vmax=vmax_gt)
        _overlay_contours(axes[1, 0])
        axes[1, 0].set_title(r"$|\mathrm{DRO}|$", fontsize=PLOT_FONT_SIZES["title"])

        axes[1, 1].imshow(grasp_img, cmap='gray', vmin=vmin_grasp, vmax=vmax_grasp)
        _overlay_contours(axes[1, 1])
        axes[1, 1].set_title(r"$|\mathrm{GRASP}|$", fontsize=PLOT_FONT_SIZES["title"])

        im_err_grasp = axes[1, 2].imshow(error_map_grasp, cmap='coolwarm', vmin=-0.5, vmax=0.5)
        axes[1, 2].set_title(r"$|\mathrm{GRASP}| - |\mathrm{DRO}|$", fontsize=PLOT_FONT_SIZES["title"])

        div = make_axes_locatable(axes[1, 2])
        cax_err_grasp = div.append_axes("right", size="4%", pad=0.04)
        cb_err_grasp = fig.colorbar(im_err_grasp, cax=cax_err_grasp)
        cb_err_grasp.ax.tick_params(labelsize=PLOT_FONT_SIZES["tick"])
        cb_err_grasp.set_label(r"$|\mathrm{GRASP}| - |\mathrm{DRO}|$", fontsize=PLOT_FONT_SIZES["tick"])

        # cb_err_grasp = fig.colorbar(im_err_grasp, cax=cax_err_grasp)
        # cb_err_grasp.ax.tick_params(labelsize=PLOT_FONT_SIZES["tick"])

        im_ssim_grasp = axes[1, 3].imshow(ssim_map_grasp, cmap='viridis', vmin=0, vmax=1)
        axes[1, 3].set_title(
            rf"$\mathrm{{SSIM}}_{{\mathrm{{GRASP}}}}$ ({ssim_grasp:.3f})",
            fontsize=PLOT_FONT_SIZES["title"],
        )

        div = make_axes_locatable(axes[1, 3])
        cax_ssim_grasp = div.append_axes("right", size="4%", pad=0.04)
        cb_ssim_grasp = fig.colorbar(im_ssim_grasp, cax=cax_ssim_grasp)
        cb_ssim_grasp.ax.tick_params(labelsize=PLOT_FONT_SIZES["tick"])
        cb_ssim_grasp.set_label(r"$\mathrm{SSIM}$", fontsize=PLOT_FONT_SIZES["tick"])

        # cb_ssim_grasp = fig.colorbar(im_ssim_grasp, cax=cax_ssim_grasp)
        # cb_ssim_grasp.ax.tick_params(labelsize=PLOT_FONT_SIZES["tick"])
        
        # Turn off axes for all plots
        for ax in axes.flat:
            ax.axis('off')

        fig.subplots_adjust(**{**PLOT_ADJUST, "top": 0.86})
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.02)

        # Save a separate first-row-only figure without a suptitle.
        top_fig = plt.figure(figsize=(24, 6))
        top_gs = gridspec.GridSpec(
            1,
            4,
            figure=top_fig,
            width_ratios=[1, 1, 1, 1],
            wspace=0.24,
        )
        top_axes = np.empty(4, dtype=object)
        top_axes[0] = top_fig.add_subplot(top_gs[0, 0])
        top_axes[1] = top_fig.add_subplot(top_gs[0, 1])
        top_axes[2] = top_fig.add_subplot(top_gs[0, 2])
        top_axes[3] = top_fig.add_subplot(top_gs[0, 3])

        top_axes[0].imshow(gt_img, cmap='gray', vmin=vmin_gt, vmax=vmax_gt)
        _overlay_contours(top_axes[0])
        top_axes[0].set_title(r"$|\mathrm{DRO}|$", fontsize=PLOT_FONT_SIZES["title"])
        top_axes[1].imshow(recon_img, cmap='gray', vmin=vmin_recon, vmax=vmax_recon)
        _overlay_contours(top_axes[1])
        top_axes[1].set_title(r"$|\mathrm{BRISKNet}_{\mathrm{pred}}|$", fontsize=PLOT_FONT_SIZES["title"])
        top_axes[2].imshow(error_map_dl, cmap='coolwarm', vmin=-0.5, vmax=0.5)
        top_axes[2].set_title(r"$|\mathrm{BRISKNet}_{\mathrm{pred}}| - |\mathrm{DRO}|$", fontsize=PLOT_FONT_SIZES["title"])
        div = make_axes_locatable(top_axes[2])
        top_cax_err = div.append_axes("right", size="4%", pad=0.04)
        top_cb_err = top_fig.colorbar(top_axes[2].images[0], cax=top_cax_err)
        top_cb_err.ax.tick_params(labelsize=PLOT_FONT_SIZES["tick"])
        top_cb_err.set_label(r"$|\mathrm{BRISKNet}| - |\mathrm{DRO}|$", fontsize=PLOT_FONT_SIZES["tick"])
        top_axes[3].imshow(ssim_map_dl, cmap='viridis', vmin=0, vmax=1)
        top_axes[3].set_title(
            rf"$\mathrm{{SSIM}}_{{\mathrm{{BRISKNet}}}}$ ({ssim_dl:.3f})",
            fontsize=PLOT_FONT_SIZES["title"],
        )
        div = make_axes_locatable(top_axes[3])
        top_cax_ssim = div.append_axes("right", size="4%", pad=0.04)
        top_cb_ssim = top_fig.colorbar(top_axes[3].images[0], cax=top_cax_ssim)
        top_cb_ssim.ax.tick_params(labelsize=PLOT_FONT_SIZES["tick"])
        top_cb_ssim.set_label(r"$\mathrm{SSIM}$", fontsize=PLOT_FONT_SIZES["tick"])

        for ax in top_axes.flat:
            ax.axis('off')

        base_name, ext = os.path.splitext(filename)
        top_row_filename = f"{base_name}_top_row{ext}"
        top_fig.subplots_adjust(**{**PLOT_ADJUST, "top": 0.95, "bottom": 0.04})
        top_fig.savefig(top_row_filename, bbox_inches='tight', pad_inches=0.02)
        plt.close(top_fig)

        # Save a separate bottom-row-only figure (GRASP row) without a suptitle.
        bottom_fig = plt.figure(figsize=(24, 6))
        bottom_gs = gridspec.GridSpec(
            1,
            4,
            figure=bottom_fig,
            width_ratios=[1, 1, 1, 1],
            wspace=0.24,
        )
        bottom_axes = np.empty(4, dtype=object)
        bottom_axes[0] = bottom_fig.add_subplot(bottom_gs[0, 0])
        bottom_axes[1] = bottom_fig.add_subplot(bottom_gs[0, 1])
        bottom_axes[2] = bottom_fig.add_subplot(bottom_gs[0, 2])
        bottom_axes[3] = bottom_fig.add_subplot(bottom_gs[0, 3])

        bottom_axes[0].imshow(gt_img, cmap='gray', vmin=vmin_gt, vmax=vmax_gt)
        _overlay_contours(bottom_axes[0])
        bottom_axes[0].set_title(r"$|\mathrm{DRO}|$", fontsize=PLOT_FONT_SIZES["title"])
        bottom_axes[1].imshow(grasp_img, cmap='gray', vmin=vmin_grasp, vmax=vmax_grasp)
        _overlay_contours(bottom_axes[1])
        bottom_axes[1].set_title(r"$|\mathrm{GRASP}|$", fontsize=PLOT_FONT_SIZES["title"])
        bottom_axes[2].imshow(error_map_grasp, cmap='coolwarm', vmin=-0.5, vmax=0.5)
        bottom_axes[2].set_title(r"$|\mathrm{GRASP}| - |\mathrm{DRO}|$", fontsize=PLOT_FONT_SIZES["title"])
        div = make_axes_locatable(bottom_axes[2])
        bottom_cax_err = div.append_axes("right", size="4%", pad=0.04)
        bottom_cb_err = bottom_fig.colorbar(bottom_axes[2].images[0], cax=bottom_cax_err)
        bottom_cb_err.ax.tick_params(labelsize=PLOT_FONT_SIZES["tick"])
        bottom_cb_err.set_label(r"$|\mathrm{GRASP}| - |\mathrm{DRO}|$", fontsize=PLOT_FONT_SIZES["tick"])
        bottom_axes[3].imshow(ssim_map_grasp, cmap='viridis', vmin=0, vmax=1)
        bottom_axes[3].set_title(
            rf"$\mathrm{{SSIM}}_{{\mathrm{{GRASP}}}}$ ({ssim_grasp:.3f})",
            fontsize=PLOT_FONT_SIZES["title"],
        )
        div = make_axes_locatable(bottom_axes[3])
        bottom_cax_ssim = div.append_axes("right", size="4%", pad=0.04)
        bottom_cb_ssim = bottom_fig.colorbar(bottom_axes[3].images[0], cax=bottom_cax_ssim)
        bottom_cb_ssim.ax.tick_params(labelsize=PLOT_FONT_SIZES["tick"])
        bottom_cb_ssim.set_label(r"$\mathrm{SSIM}$", fontsize=PLOT_FONT_SIZES["tick"])

        for ax in bottom_axes.flat:
            ax.axis('off')

        bottom_row_filename = f"{base_name}_bottom_row{ext}"
        bottom_fig.subplots_adjust(**{**PLOT_ADJUST, "top": 0.95, "bottom": 0.04})
        bottom_fig.savefig(bottom_row_filename, bbox_inches='tight', pad_inches=0.02)
        plt.close(bottom_fig)
        plt.close()


    # Plot the Difference Between GRASP and BRISKNet

    # Calculate error map
    error_map = recon_img - grasp_img

    # Calculate SSIM maps
    ssim, ssim_map = ssim_map_func(grasp_img, recon_img, data_range=data_range, full=True)

    # Create a 1x4 plot grid with dedicated colorbar columns so image axes stay the same size.
    fig = plt.figure(figsize=(24, 6))
    gs = gridspec.GridSpec(
        1,
        4,
        figure=fig,
        width_ratios=[1, 1, 1, 1],
        wspace=0.16,
    )
    axes = np.empty(4, dtype=object)
    axes[0] = fig.add_subplot(gs[0, 0])
    axes[1] = fig.add_subplot(gs[0, 1])
    axes[2] = fig.add_subplot(gs[0, 2])
    axes[3] = fig.add_subplot(gs[0, 3])
    fig.suptitle(
        f"BRISKNet vs GRASP Comparison at Time Frame {time_frame_index} with AF {acceleration} and SPF {spokes_per_frame}",
        fontsize=PLOT_FONT_SIZES["suptitle"],
        y=0.995,
    )

    # --- Top Row: BRISKNet Comparison ---

    axes[0].imshow(grasp_img, cmap='gray', vmin=vmin_grasp, vmax=vmax_grasp)
    _overlay_contours(axes[0])
    axes[0].set_title(r"$|\mathrm{GRASP}|$", fontsize=PLOT_FONT_SIZES["title"])

    axes[1].imshow(recon_img, cmap='gray', vmin=vmin_recon, vmax=vmax_recon)
    _overlay_contours(axes[1])
    axes[1].set_title(r"$|\mathrm{BRISKNet}_{\mathrm{pred}}|$", fontsize=PLOT_FONT_SIZES["title"])

    im_err_dl = axes[2].imshow(error_map, cmap='coolwarm', vmin=-0.5, vmax=0.5)
    axes[2].set_title(r"$|\mathrm{BRISKNet}_{\mathrm{pred}}| - |\mathrm{GRASP}|$", fontsize=PLOT_FONT_SIZES["title"])
    div = make_axes_locatable(axes[2])
    cax_err_dl = div.append_axes("right", size="4%", pad=0.04)
    cb_err_dl = fig.colorbar(im_err_dl, cax=cax_err_dl)
    cb_err_dl.ax.tick_params(labelsize=PLOT_FONT_SIZES["tick"])
    # cb_err_dl.set_label(r"$|\mathrm{BRISKNet}| - |\mathrm{DRO}|$", fontsize=PLOT_FONT_SIZES["tick"])

    im_ssim_dl = axes[3].imshow(ssim_map, cmap='viridis', vmin=0, vmax=1)
    axes[3].set_title(
        rf"$\mathrm{{SSIM}}_{{\mathrm{{BRISKNet}}}}$ vs $\mathrm{{GRASP}}$ ({ssim:.3f})",
        fontsize=PLOT_FONT_SIZES["title"],
    )
    div = make_axes_locatable(axes[3])
    cax_ssim_dl = div.append_axes("right", size="4%", pad=0.04)
    cb_ssim_dl = fig.colorbar(im_ssim_dl, cax=cax_ssim_dl)
    cb_ssim_dl.ax.tick_params(labelsize=PLOT_FONT_SIZES["tick"])
    # cb_ssim_dl.set_label(r"$\mathrm{SSIM}$", fontsize=PLOT_FONT_SIZES["tick"])
    
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
    region_label_map: Optional[Dict[str, str]] = None,
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

    regions = [r for r in ['malignant', 'benign', 'glandular', 'muscle', 'full'] if r in masks and masks[r].any()]

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
    arrival_idx = None
    if show_arrival:
        arrival_idx = _arrival_index_from_mag_stack(
            recon_img_stack,
            arrival_percentile,
            arrival_baseline_k,
            arrival_method,
            arrival_fraction,
            arrival_pre_contrast_baseline,
            arrival_baseline_seconds,
            arrival_total_seconds,
        )

    for i, region in enumerate(regions):
        mask = masks[region]

        # Calculate mean signal in the masked region for each time point
        gt_curve = [gt_img_stack[:, :, t][mask].mean() for t in range(gt_img_stack.shape[2])]
        recon_curve = [recon_img_stack[:, :, t][mask].mean() for t in range(recon_img_stack.shape[2])]
        grasp_curve = [grasp_img_stack[:, :, t][mask].mean() for t in range(grasp_img_stack.shape[2])]

        # compute the pearson correlation coefficients (guard against constant curves)
        recon_correlation = _safe_pearsonr(recon_curve, gt_curve)
        grasp_correlation = _safe_pearsonr(grasp_curve, gt_curve)

        region_corrs[region] = {"DL": recon_correlation, "GRASP":  grasp_correlation}


        # if region == 'malignant':
        #     recon_correlation, _ = pearsonr(recon_curve, gt_curve)
        #     grasp_correlation, _ = pearsonr(grasp_curve, gt_curve)


        # Plot
        if plot_dro:
            axes[i].plot(time_points, gt_curve, 'k-', label='DRO', linewidth=2, marker='o')

        axes[i].plot(time_points, recon_curve, 'r--', label='BRISKNet', marker='o')
        axes[i].plot(time_points, grasp_curve, 'b:', label='GRASP', marker='o')
        
        display_region = region_label_map.get(region, region) if region_label_map else region
        if plot_dro:
            axes[i].set_title(
                f"{display_region.capitalize()} (BRISKNet: {recon_correlation:.2f}, GRASP: {grasp_correlation:.2f})",
                fontsize=PLOT_FONT_SIZES["title"],
            )
        else:
            axes[i].set_title(f"{display_region.capitalize()}", fontsize=PLOT_FONT_SIZES["title"])
        axes[i].set_xlabel("Time (s)", fontsize=PLOT_FONT_SIZES["label"])
        axes[i].tick_params(axis='both', which='major', labelsize=PLOT_FONT_SIZES["tick"])
        axes[i].grid(True)
        if arrival_idx is not None and arrival_idx < len(time_points):
            arrival_time = time_points[arrival_idx]
            axes[i].axvline(
                arrival_time,
                color='tab:red',
                linestyle='--',
                linewidth=1.5,
                label='Arrival' if i == 0 else None,
            )
        axes[i].legend(fontsize=PLOT_FONT_SIZES["legend"])

    axes[0].set_ylabel("Mean Signal Intensity", fontsize=PLOT_FONT_SIZES["label"])
    _safe_tight_layout(fig, rect=[0, 0.02, 1, 0.94], **PLOT_LAYOUT)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

    return region_corrs


def _resolve_baseline_frames(
    num_frames: int,
    time_points: Optional[np.ndarray] = None,
    baseline_mode: str = "fraction",
    baseline_seconds: float = 20.0,
    baseline_fraction: float = 0.1,
    baseline_min_frames: int = 4,
    baseline_max_frames: Optional[int] = 10,
) -> int:
    if num_frames <= 0:
        return 0

    mode = (baseline_mode or "fraction").lower()
    if mode == "seconds":
        dt = None
        if time_points is not None and len(time_points) > 1:
            dt = float(time_points[1] - time_points[0])
        if not dt or dt <= 0:
            dt = float(baseline_seconds) if baseline_seconds > 0 else 1.0
        frames = int(np.ceil(baseline_seconds / dt)) if baseline_seconds > 0 else 0
        if baseline_min_frames is not None:
            frames = max(frames, baseline_min_frames)
        frames = min(frames, num_frames)
        return max(1, frames)

    if mode == "fraction":
        frames = int(round(baseline_fraction * num_frames))
        if baseline_min_frames is not None:
            frames = max(frames, baseline_min_frames)
        if baseline_max_frames is not None:
            frames = min(frames, baseline_max_frames)
        frames = min(frames, num_frames)
        return max(1, frames)

    raise ValueError(f"Unknown baseline_mode: {baseline_mode!r}")


def _arrival_index_from_mag_stack(
    mag_stack: np.ndarray,
    percentile: float,
    baseline_k: float,
    arrival_method: str,
    arrival_fraction: float,
    pre_contrast_baseline: str,
    baseline_seconds: float,
    total_seconds: float,
) -> Optional[int]:
    if mag_stack is None or mag_stack.ndim != 3:
        return None
    try:
        mag = torch.from_numpy(mag_stack).float()
    except Exception:
        return None
    mag = mag.permute(2, 0, 1).unsqueeze(0)  # (1, T, H, W)
    zeros = torch.zeros_like(mag)
    x = torch.stack([mag, zeros], dim=1)  # (1, 2, T, H, W)
    try:
        idx = estimate_bolus_arrival_index(
            x,
            percentile=percentile,
            baseline_k=baseline_k,
            arrival_method=arrival_method,
            arrival_fraction=arrival_fraction,
            pre_contrast_baseline=pre_contrast_baseline,
            baseline_seconds=baseline_seconds,
            total_seconds=total_seconds,
        )
    except Exception:
        return None
    return int(idx)


def plot_temporal_curves_normalized(
    gt_img_stack: np.ndarray,
    recon_img_stack: np.ndarray,
    grasp_img_stack: np.ndarray,
    masks: dict,
    time_points: np.ndarray,
    filename: str,
    acceleration: float,
    spokes_per_frame: int,
    plot_dro: bool = True,
    baseline_mode: str = "fraction",
    baseline_seconds: float = 20.0,
    baseline_fraction: float = 0.1,
    baseline_min_frames: int = 4,
    baseline_max_frames: Optional[int] = 10,
    region_label_map: Optional[Dict[str, str]] = None,
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
    Plots baseline-subtracted mean signal vs. time for different tissue regions.
    """
    regions = [r for r in ['malignant', 'benign', 'glandular', 'muscle', 'full'] if r in masks and masks[r].any()]

    if not regions:
        print("No relevant regions found in mask to plot normalized temporal curves.")
        return

    num_frames = gt_img_stack.shape[2]
    n_baseline = _resolve_baseline_frames(
        num_frames=num_frames,
        time_points=time_points,
        baseline_mode=baseline_mode,
        baseline_seconds=baseline_seconds,
        baseline_fraction=baseline_fraction,
        baseline_min_frames=baseline_min_frames,
        baseline_max_frames=baseline_max_frames,
    )

    fig, axes = plt.subplots(1, len(regions), figsize=(7 * len(regions), 5))
    if len(regions) == 1:
        axes = [axes]
    fig.suptitle(
        f"Baseline-Subtracted Signal vs. Time (AF = {acceleration}, SPF = {spokes_per_frame})",
        fontsize=PLOT_FONT_SIZES["suptitle"],
    )

    arrival_idx = None
    if show_arrival:
        arrival_idx = _arrival_index_from_mag_stack(
            recon_img_stack,
            arrival_percentile,
            arrival_baseline_k,
            arrival_method,
            arrival_fraction,
            arrival_pre_contrast_baseline,
            arrival_baseline_seconds,
            arrival_total_seconds,
        )

    for i, region in enumerate(regions):
        mask = masks[region]

        gt_curve = np.array([gt_img_stack[:, :, t][mask].mean() for t in range(num_frames)])
        recon_curve = np.array([recon_img_stack[:, :, t][mask].mean() for t in range(num_frames)])
        grasp_curve = np.array([grasp_img_stack[:, :, t][mask].mean() for t in range(num_frames)])

        # Baseline-subtracted enhancement curves.
        gt_curve = gt_curve - np.nanmean(gt_curve[:n_baseline])
        recon_curve = recon_curve - np.nanmean(recon_curve[:n_baseline])
        grasp_curve = grasp_curve - np.nanmean(grasp_curve[:n_baseline])

        if plot_dro:
            axes[i].plot(time_points, gt_curve, 'k-', label='DRO', linewidth=2, marker='o')
        axes[i].plot(time_points, recon_curve, 'r--', label='BRISKNet', marker='o')
        axes[i].plot(time_points, grasp_curve, 'b:', label='GRASP', marker='o')
        display_region = region_label_map.get(region, region) if region_label_map else region
        axes[i].set_title(f"{display_region.capitalize()}", fontsize=PLOT_FONT_SIZES["title"])
        axes[i].set_xlabel("Time (s)", fontsize=PLOT_FONT_SIZES["label"])
        axes[i].tick_params(axis='both', which='major', labelsize=PLOT_FONT_SIZES["tick"])
        axes[i].grid(True)
        if arrival_idx is not None and arrival_idx < len(time_points):
            arrival_time = time_points[arrival_idx]
            axes[i].axvline(
                arrival_time,
                color='tab:red',
                linestyle='--',
                linewidth=1.5,
                label='Arrival' if i == 0 else None,
            )
        axes[i].legend(fontsize=PLOT_FONT_SIZES["legend"])

    axes[0].set_ylabel("Baseline-Subtracted Signal", fontsize=PLOT_FONT_SIZES["label"])
    _safe_tight_layout(fig, rect=[0, 0.02, 1, 0.94], **PLOT_LAYOUT)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)



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
    region_key: str | None = None,
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
    if region_key is None:
        if 'malignant' in masks and masks['malignant'].any():
            region_key = 'malignant'
        elif 'benign' in masks and masks['benign'].any():
            region_key = 'benign'
        elif 'full' in masks and masks['full'].any():
            region_key = 'full'

    if region_key is None or region_key not in masks or not masks[region_key].any():
        print("No valid ROI mask found for temporal curve plot. Skipping plot generation.")
        return

    tumor_mask = masks[region_key]

    if frames_to_show is None:
        interval = round(num_frames / 4)
        frames_to_show = [0, interval, 2 * interval, num_frames - 1]
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

    # --- 2. Plot Tumor Enhancement Curve (Left Panel) ---
    mean_curve = [img_stack[:, :, t][tumor_mask].mean() for t in range(img_stack.shape[2])]
    ax_curve.plot(time_points, mean_curve, 'o-', label='Mean Tumor Signal', linewidth=2, markersize=6)
    if show_arrival:
        arrival_idx = _arrival_index_from_mag_stack(
            img_stack,
            arrival_percentile,
            arrival_baseline_k,
            arrival_method,
            arrival_fraction,
            arrival_pre_contrast_baseline,
            arrival_baseline_seconds,
            arrival_total_seconds,
        )
        if arrival_idx is not None and arrival_idx < len(time_points):
            arrival_time = time_points[arrival_idx]
            ax_curve.axvline(arrival_time, color='tab:red', linestyle='--', linewidth=1.5, label='Arrival')
            ax_curve.plot(arrival_time, mean_curve[arrival_idx], 'ro', markersize=8, zorder=10)

    highlight_times = [time_points[i] for i in frames_to_show]
    highlight_vals = [mean_curve[i] for i in frames_to_show]
    ax_curve.plot(highlight_times, highlight_vals, 'r*', markersize=18, zorder=10)

    ax_curve.set_title("Tumor Contrast Enhancement Curve (CEC)", fontsize=PLOT_FONT_SIZES["title"], pad=8)
    ax_curve.set_xlabel("Time Frame", fontsize=PLOT_FONT_SIZES["label"])
    ax_curve.set_ylabel("Mean Signal Intensity", fontsize=PLOT_FONT_SIZES["label"])
    ax_curve.legend(fontsize=PLOT_FONT_SIZES["legend"])
    ax_curve.grid(True, linestyle='--')
    ax_curve.tick_params(axis='both', which='major', labelsize=PLOT_FONT_SIZES["tick"])

    # --- 3. Plot Image Frames with ROI (Right Panel) ---
    contours = find_contours(tumor_mask, 0.5)
    vmin_window, vmax_window = robust_window(img_stack, 1, 99.5)

    for i, frame_idx in enumerate(frames_to_show):
        ax = ax_imgs[i]
        image = img_stack[:, :, frame_idx]
        ax.imshow(image, cmap='gray', vmin=vmin_window, vmax=vmax_window)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='red')
        ax.set_title(f"Frame {frame_idx}", fontsize=PLOT_FONT_SIZES["title"])
        ax.axis('off')

    # --- 4. Finalize and Save ---
    _safe_tight_layout(fig, rect=[0, 0, 1, 0.9], **PLOT_LAYOUT)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.02, dpi=150)
    plt.close(fig)



def compute_temporal_metrics(
    gt_mag_np: np.ndarray,
    recon_mag_np: np.ndarray,
    tumor_mask: np.ndarray,
    time_points: np.ndarray,
    baseline_mode: str = "fraction",
    baseline_seconds: float = 20.0,
    baseline_fraction: float = 0.1,
    baseline_min_frames: int = 4,
    baseline_max_frames: Optional[int] = 10,
    arrival_k: float = 3.0,
    arrival_method: str = "threshold",
    arrival_fraction: float = 0.1,
    early_seconds: float = 35.0,
    early_min_frames: int = 4,
    early_max_frames: Optional[int] = 8,
) -> Dict[str, float]:
    if tumor_mask is None or not tumor_mask.any():
        return {}

    num_frames = gt_mag_np.shape[2]
    n_baseline = _resolve_baseline_frames(
        num_frames=num_frames,
        time_points=time_points,
        baseline_mode=baseline_mode,
        baseline_seconds=baseline_seconds,
        baseline_fraction=baseline_fraction,
        baseline_min_frames=baseline_min_frames,
        baseline_max_frames=baseline_max_frames,
    )
    dt = float(time_points[1] - time_points[0]) if num_frames > 1 else 1.0
    n_early = int(np.ceil(early_seconds / dt)) if early_seconds > 0 else 0
    if early_min_frames is not None:
        n_early = max(n_early, early_min_frames)
    if early_max_frames is not None:
        n_early = min(n_early, early_max_frames)
    n_early = max(1, n_early)

    gt_flat = gt_mag_np[tumor_mask].reshape(-1, num_frames)
    recon_flat = recon_mag_np[tumor_mask].reshape(-1, num_frames)

    if gt_flat.size == 0:
        return {}

    def normalize_baseline(curves: np.ndarray, n_base: int) -> np.ndarray:
        # Baseline-subtracted enhancement to focus metrics on temporal dynamics.
        if curves.size == 0 or n_base <= 0:
            return curves
        baseline = np.nanmean(curves[:, :n_base], axis=1, keepdims=True)
        baseline = np.nan_to_num(baseline, nan=0.0)
        return curves - baseline

    baseline_gt = gt_flat[:, :n_baseline].mean(axis=1)

    peak_gt = gt_flat.max(axis=1)
    peak_enh = peak_gt - baseline_gt

    valid_mask = peak_enh > 0
    metric_names = [
        "curve_corr",
        "curve_mae",
        "early_corr",
        "early_mae",
        "ttae_sec",
        "wash_in_slope_err",
        "iauc10_err",
        "peak_err",
        "ttpeak_err_sec",
    ]
    if not np.any(valid_mask):
        return {f"{subset}_{metric}": np.nan for subset in ("all", "top10", "top20") for metric in metric_names}

    valid_indices = np.where(valid_mask)[0]
    sorted_indices = valid_indices[np.argsort(peak_enh[valid_mask])[::-1]]

    def subset_indices(frac: float) -> np.ndarray:
        if sorted_indices.size == 0:
            return np.array([], dtype=int)
        count = max(1, int(np.ceil(frac * sorted_indices.size)))
        return sorted_indices[:count]

    subsets = {
        "all": sorted_indices,
        "top10": subset_indices(0.10),
        "top20": subset_indices(0.20),
    }

    mean_curve = gt_flat.mean(axis=0)
    smoothed = mean_curve.copy()
    if num_frames >= 3:
        smoothed[1:-1] = (mean_curve[:-2] + mean_curve[1:-1] + mean_curve[2:]) / 3.0

    mu0 = smoothed[:n_baseline].mean()
    sigma0 = smoothed[:n_baseline].std()
    method = (arrival_method or "threshold").lower()
    if method in ("fraction", "fraction_of_peak", "fop"):
        peak0 = mean_curve.max()
        frac = max(0.0, min(1.0, float(arrival_fraction)))
        thr0 = mu0 + frac * (peak0 - mu0)
    else:
        thr0 = mu0 + arrival_k * sigma0
    above0 = smoothed > thr0
    t_arr_idx = int(np.argmax(above0)) if np.any(above0) else 0
    t_peak_idx = int(np.argmax(mean_curve))

    early_start = t_arr_idx
    early_end = min(t_arr_idx + n_early, t_peak_idx)
    if early_end < early_start:
        early_end = early_start
    if (early_end - early_start + 1) < 3 and num_frames >= 3:
        early_end = min(early_start + 2, num_frames - 1)
    early_slice = slice(early_start, early_end + 1)

    def mean_pearson(a: np.ndarray, b: np.ndarray) -> float:
        a_mean = a.mean(axis=1, keepdims=True)
        b_mean = b.mean(axis=1, keepdims=True)
        a_diff = a - a_mean
        b_diff = b - b_mean
        num = np.sum(a_diff * b_diff, axis=1)
        den = np.sqrt(np.sum(a_diff ** 2, axis=1) * np.sum(b_diff ** 2, axis=1))
        corr = np.divide(num, den, out=np.full_like(num, np.nan, dtype=np.float64), where=den > 0)
        return float(np.nanmean(corr)) if corr.size else np.nan

    def arrival_indices(curves: np.ndarray, baseline_mu: np.ndarray, baseline_sigma: np.ndarray) -> np.ndarray:
        method_local = (arrival_method or "threshold").lower()
        if method_local in ("fraction", "fraction_of_peak", "fop"):
            peak = curves.max(axis=1)
            frac = max(0.0, min(1.0, float(arrival_fraction)))
            thr = baseline_mu + frac * (peak - baseline_mu)
        else:
            thr = baseline_mu + arrival_k * baseline_sigma
        above = curves > thr[:, None]
        has_arrival = np.any(above, axis=1)
        idx = np.argmax(above, axis=1)
        return np.where(has_arrival, idx, -1)

    def compute_iauc10(curves: np.ndarray, baseline: np.ndarray, arrivals: np.ndarray) -> np.ndarray:
        areas = np.full(curves.shape[0], np.nan, dtype=float)
        dt_local = float(time_points[1] - time_points[0]) if num_frames > 1 else 0.0
        for i, t_idx in enumerate(arrivals):
            if t_idx < 0:
                continue
            t_arr_time = time_points[t_idx]
            t_end = t_arr_time + 10.0
            end_idx = np.searchsorted(time_points, t_end, side="right") - 1
            end_idx = int(min(max(end_idx, t_idx), num_frames - 1))
            if end_idx <= t_idx:
                if dt_local > 10.0 and t_idx < (num_frames - 1):
                    end_idx = t_idx + 1
                else:
                    continue
            y = curves[i, t_idx:end_idx + 1] - baseline[i]
            areas[i] = float(np.trapz(y, time_points[t_idx:end_idx + 1]))
        return areas

    metrics = {}
    for subset_name, idx in subsets.items():
        if idx.size == 0:
            for metric_name in metric_names:
                metrics[f"{subset_name}_{metric_name}"] = np.nan
            continue

        gt_curves = gt_flat[idx]
        recon_curves = recon_flat[idx]

        gt_norm = normalize_baseline(gt_curves, n_baseline)
        recon_norm = normalize_baseline(recon_curves, n_baseline)

        metrics[f"{subset_name}_curve_corr"] = mean_pearson(recon_norm, gt_norm)
        metrics[f"{subset_name}_curve_mae"] = float(np.mean(np.abs(recon_norm - gt_norm)))
        metrics[f"{subset_name}_early_corr"] = mean_pearson(recon_norm[:, early_slice], gt_norm[:, early_slice])
        metrics[f"{subset_name}_early_mae"] = float(np.mean(np.abs(recon_norm[:, early_slice] - gt_norm[:, early_slice])))

        gt_baseline_mu = gt_curves[:, :n_baseline].mean(axis=1)
        gt_baseline_sigma = gt_curves[:, :n_baseline].std(axis=1)
        recon_baseline_mu = recon_curves[:, :n_baseline].mean(axis=1)
        recon_baseline_sigma = recon_curves[:, :n_baseline].std(axis=1)

        gt_arr = arrival_indices(gt_curves, gt_baseline_mu, gt_baseline_sigma)
        recon_arr = arrival_indices(recon_curves, recon_baseline_mu, recon_baseline_sigma)

        valid_arr = (gt_arr >= 0) & (recon_arr >= 0)
        if np.any(valid_arr):
            gt_arr_time = time_points[gt_arr[valid_arr]]
            recon_arr_time = time_points[recon_arr[valid_arr]]
            metrics[f"{subset_name}_ttae_sec"] = float(np.mean(np.abs(recon_arr_time - gt_arr_time)))
        else:
            metrics[f"{subset_name}_ttae_sec"] = np.nan

        gt_peak_idx = np.argmax(gt_curves, axis=1)
        recon_peak_idx = np.argmax(recon_curves, axis=1)

        valid_slope = (gt_arr >= 0) & (recon_arr >= 0)
        if np.any(valid_slope):
            gt_valid = valid_slope & (gt_peak_idx > gt_arr)
            recon_valid = valid_slope & (recon_peak_idx > recon_arr)
            valid = gt_valid & recon_valid
            if np.any(valid):
                gt_time_delta = time_points[gt_peak_idx[valid]] - time_points[gt_arr[valid]]
                recon_time_delta = time_points[recon_peak_idx[valid]] - time_points[recon_arr[valid]]
                gt_slope = (gt_norm[valid, gt_peak_idx[valid]] - gt_norm[valid, gt_arr[valid]]) / gt_time_delta
                recon_slope = (recon_norm[valid, recon_peak_idx[valid]] - recon_norm[valid, recon_arr[valid]]) / recon_time_delta
                metrics[f"{subset_name}_wash_in_slope_err"] = float(np.mean(np.abs(recon_slope - gt_slope)))
            else:
                metrics[f"{subset_name}_wash_in_slope_err"] = np.nan
        else:
            metrics[f"{subset_name}_wash_in_slope_err"] = np.nan

        gt_iauc = compute_iauc10(gt_norm, np.zeros_like(gt_baseline_mu), gt_arr)
        recon_iauc = compute_iauc10(recon_norm, np.zeros_like(recon_baseline_mu), recon_arr)
        valid_iauc = np.isfinite(gt_iauc) & np.isfinite(recon_iauc)
        metrics[f"{subset_name}_iauc10_err"] = float(np.mean(np.abs(recon_iauc[valid_iauc] - gt_iauc[valid_iauc]))) if np.any(valid_iauc) else np.nan

        gt_peak_val = gt_norm.max(axis=1)
        recon_peak_val = recon_norm.max(axis=1)
        metrics[f"{subset_name}_peak_err"] = float(np.mean(np.abs(recon_peak_val - gt_peak_val)))

        gt_peak_time = time_points[gt_peak_idx]
        recon_peak_time = time_points[recon_peak_idx]
        metrics[f"{subset_name}_ttpeak_err_sec"] = float(np.mean(np.abs(recon_peak_time - gt_peak_time)))

    return metrics


def compute_dynamic_weighted_dce_error(
    reference_mag_np: np.ndarray,
    recon_mag_np: np.ndarray,
    time_points: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    baseline_mode: str = "fraction",
    baseline_seconds: float = 20.0,
    baseline_fraction: float = 0.1,
    baseline_min_frames: int = 4,
    baseline_max_frames: Optional[int] = 10,
    top_fraction: float = 0.02,
    weight_power: float = 1.0,
) -> Dict[str, float]:
    """Dynamic-weighted DCE error for non-DRO comparisons."""
    ref = np.asarray(reference_mag_np, dtype=np.float64)
    rec = np.asarray(recon_mag_np, dtype=np.float64)
    if ref.shape != rec.shape or ref.ndim != 3:
        return {}

    h, w, t = ref.shape
    if t <= 0:
        return {}

    if roi_mask is None:
        mask = np.ones((h, w), dtype=bool)
    else:
        mask = np.asarray(roi_mask).squeeze().astype(bool)
        if mask.shape != (h, w):
            return {}
    if not mask.any():
        return {}

    n_baseline = _resolve_baseline_frames(
        num_frames=t,
        time_points=time_points,
        baseline_mode=baseline_mode,
        baseline_seconds=baseline_seconds,
        baseline_fraction=baseline_fraction,
        baseline_min_frames=baseline_min_frames,
        baseline_max_frames=baseline_max_frames,
    )

    ref_curves = ref[mask].reshape(-1, t)
    rec_curves = rec[mask].reshape(-1, t)
    if ref_curves.size == 0:
        return {}

    ref_base = ref_curves[:, :n_baseline].mean(axis=1, keepdims=True)
    rec_base = rec_curves[:, :n_baseline].mean(axis=1, keepdims=True)
    ref_dce = ref_curves - ref_base
    rec_dce = rec_curves - rec_base

    dyn_scores = np.std(ref_dce, axis=1)
    n_pixels = int(dyn_scores.shape[0])
    frac = float(top_fraction)
    if not np.isfinite(frac):
        frac = 0.02
    frac = min(max(frac, 0.0), 1.0)
    k = max(1, int(np.ceil(frac * n_pixels)))
    k = min(k, n_pixels)

    if k == n_pixels:
        keep = np.arange(n_pixels, dtype=np.int64)
    else:
        keep = np.argpartition(dyn_scores, -k)[-k:]

    err = rec_dce[keep] - ref_dce[keep]
    per_pixel_mae = np.mean(np.abs(err), axis=1)
    per_pixel_mse = np.mean(err ** 2, axis=1)

    weights = dyn_scores[keep].astype(np.float64)
    power = float(weight_power)
    if not np.isfinite(power) or power <= 0.0:
        power = 1.0
    weights = np.power(np.clip(weights, a_min=0.0, a_max=None), power)
    if not np.any(weights > 0):
        weights = np.ones_like(weights)
    weights = weights / (weights.sum() + 1e-12)

    return {
        "weighted_mae": float(np.sum(weights * per_pixel_mae)),
        "weighted_mse": float(np.sum(weights * per_pixel_mse)),
        "topk_mae": float(np.mean(per_pixel_mae)),
        "topk_mse": float(np.mean(per_pixel_mse)),
        "selected_pixels": int(k),
        "selected_fraction": float(k / max(1, n_pixels)),
    }


def plot_time_series(
    recon_img_stack: np.ndarray,
    grasp_img_stack: np.ndarray,
    filename: str,
    acceleration: float,
    spokes_per_frame: int, 
    tumor_mask: np.ndarray | None = None,
):
    """
    Plots the middle 5 time points for Ground Truth, BRISKNet, and GRASP.

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

    contours = None
    if tumor_mask is not None and np.any(tumor_mask):
        contours = find_contours(tumor_mask, 0.5)

    def _overlay_contours(ax):
        if not contours:
            return
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='red')

    # Use a shared window per time frame so intensity differences are comparable between rows.
    for i, frame_idx in enumerate(indices):
        recon_img = recon_img_stack[:, :, frame_idx]
        grasp_img = grasp_img_stack[:, :, frame_idx]
        vmin_frame, vmax_frame = robust_window_multi([recon_img, grasp_img], p_low=1, p_high=99.5)
        axes[0, i].imshow(recon_img, cmap='gray', vmin=vmin_frame, vmax=vmax_frame)
        _overlay_contours(axes[0, i])
        axes[0, i].set_title(f"BRISKNet: Frame {frame_idx}", fontsize=PLOT_FONT_SIZES["title"])
        axes[0, i].axis('off')

    # --- Row 3: GRASP Reconstruction ---
    for i, frame_idx in enumerate(indices):
        recon_img = recon_img_stack[:, :, frame_idx]
        grasp_img = grasp_img_stack[:, :, frame_idx]
        vmin_frame, vmax_frame = robust_window_multi([recon_img, grasp_img], p_low=1, p_high=99.5)
        axes[1, i].imshow(grasp_img, cmap='gray', vmin=vmin_frame, vmax=vmax_frame)
        _overlay_contours(axes[1, i])
        axes[1, i].set_title(f"GRASP: Frame {frame_idx}", fontsize=PLOT_FONT_SIZES["title"])
        axes[1, i].axis('off')

    fig.subplots_adjust(**PLOT_ADJUST)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.02)
    plt.close()





# ==========================================================
# EVALUATION 
# ==========================================================
def eval_grasp(
    kspace,
    csmap,
    ground_truth,
    grasp_recon,
    physics,
    device,
    output_dir,
    rescale,
    dro_eval=True,
    report_bestfit_dc: bool = False,
    return_aux: bool = False,
):


    # ==========================================================
    # EVALUATE DATA CONSISTENCY
    # ==========================================================

    # Forward Simulation
    grasp_recon_complex = rearrange(to_torch_complex(grasp_recon).squeeze(), 'h t w -> h w t')
    kspace = kspace.squeeze()

    grasp_kspace = physics(False, grasp_recon_complex.to(csmap.dtype), csmap)


    # Compute MSE
    dc_mse_grasp, dc_mae_grasp = calc_dc(grasp_kspace, kspace, device)
    aux = {}
    dc_mse_bestfit, dc_mae_bestfit, dc_scale = calc_dc_bestfit(grasp_kspace, kspace, device)
    if dc_mse_bestfit is not None and dc_scale is not None:
        aux.update(
            {
                "grasp_dc_mse_bestfit": dc_mse_bestfit,
                "grasp_dc_mae_bestfit": dc_mae_bestfit,
                "grasp_dc_scale_abs": float(abs(dc_scale)),
                "grasp_dc_scale_phase": float(np.angle(dc_scale)),
            }
        )


    # ==========================================================
    # EVALUATE SPATIAL IMAGE QUALITY
    # ==========================================================
    if dro_eval:

        grasp_recon_np = grasp_recon.cpu().numpy()
        ground_truth_np = ground_truth.cpu().numpy()

        if rescale:
            c = np.dot(grasp_recon_np.flatten(), ground_truth_np.flatten()) / np.dot(grasp_recon_np.flatten(), grasp_recon_np.flatten())
            grasp_recon = torch.tensor(c * grasp_recon_np, device=device)
        else:
            grasp_recon = torch.tensor(grasp_recon_np, device=device)


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


        if return_aux:
            return ssim_grasp, psnr_grasp, mse_grasp, lpips_grasp, dc_mse_grasp, dc_mae_grasp, aux
        return ssim_grasp, psnr_grasp, mse_grasp, lpips_grasp, dc_mse_grasp, dc_mae_grasp

    else:
        if return_aux:
            return dc_mse_grasp, dc_mae_grasp, aux
        return dc_mse_grasp, dc_mae_grasp


def eval_zf(
    kspace,
    csmap,
    ground_truth,
    physics,
    mask,
    device,
    rescale: bool,
    zf_complex_override: torch.Tensor | None = None,
    report_bestfit_dc: bool = False,
    return_aux: bool = False,
):
    """Adjoint (density-compensated) baseline + metrics against DRO ground truth."""
    kspace = kspace.squeeze()
    zf_complex = zf_complex_override if zf_complex_override is not None else physics(True, kspace, csmap)
    zf = torch.stack([zf_complex.real, zf_complex.imag], dim=0).unsqueeze(0)

    aux = {}

    zf_kspace = physics(False, zf_complex.to(csmap.dtype), csmap)
    dc_mse_zf, dc_mae_zf = calc_dc(zf_kspace, kspace, device)
    dc_mse_bestfit, dc_mae_bestfit, dc_scale = calc_dc_bestfit(zf_kspace, kspace, device)
    if dc_mse_bestfit is not None and dc_scale is not None:
        aux.update(
            {
                "zf_dc_mse_bestfit": dc_mse_bestfit,
                "zf_dc_mae_bestfit": dc_mae_bestfit,
                "zf_dc_scale_abs": float(abs(dc_scale)),
                "zf_dc_scale_phase": float(np.angle(dc_scale)),
            }
        )

    # Best-fit gain against GT (real scalar, matching eval_sample convention).
    zf_np = zf.cpu().numpy()
    gt_np = ground_truth.cpu().numpy()
    zf_scale_np = zf_np
    if zf_np.ndim == 5 and gt_np.ndim == 5:
        if (
            zf_np.shape[:2] == gt_np.shape[:2]
            and zf_np.shape[2] == gt_np.shape[3]
            and zf_np.shape[3] == gt_np.shape[4]
            and zf_np.shape[4] == gt_np.shape[2]
        ):
            zf_scale_np = np.transpose(zf_np, (0, 1, 4, 2, 3))
    denom = float(np.dot(zf_scale_np.flatten(), zf_scale_np.flatten()))
    c = float(np.dot(zf_scale_np.flatten(), gt_np.flatten()) / (denom + 1e-12)) if denom > 0 else 1.0
    aux["zf_img_scale"] = c
    if rescale:
        zf = torch.tensor(c * zf_np, device=device)

    # Convert to magnitude (treat time as depth slices).
    gt_mag = torch.sqrt(ground_truth[:, 0, ...] ** 2 + ground_truth[:, 1, ...] ** 2)  # (B,T,H,W)
    zf_mag = torch.sqrt(zf[:, 0, ...] ** 2 + zf[:, 1, ...] ** 2)  # (B,H,W,T)
    zf_mag = zf_mag.permute(0, 3, 1, 2).unsqueeze(1)  # (B,1,T,H,W)
    gt_mag = gt_mag.unsqueeze(1)  # (B,1,T,H,W)

    min_val = torch.min(gt_mag).item()
    max_val = torch.max(gt_mag).item()
    data_range = (min_val, max_val)
    ssim_zf, psnr_zf, mse_zf, lpips_zf = calc_image_metrics(zf_mag.contiguous(), gt_mag.contiguous(), data_range, device)

    # Foreground-masked metrics (same union mask as DRO tissues if available).
    masks_np = {key: val.cpu().numpy().squeeze().astype(bool) for key, val in (mask or {}).items()}
    foreground_mask = None
    foreground_fraction = None
    if masks_np:
        union = np.zeros(gt_mag.shape[-2:], dtype=bool)
        for mask_arr in masks_np.values():
            if mask_arr is None:
                continue
            mask_arr = np.asarray(mask_arr).squeeze().astype(bool)
            if mask_arr.shape != union.shape:
                continue
            union |= mask_arr
        if union.any() and union.mean() < 0.999:
            foreground_mask = union
            foreground_fraction = float(union.mean())
    if foreground_mask is None:
        gt_mag_np = gt_mag.squeeze().detach().cpu().numpy()  # (T,H,W)
        max_gt = float(np.max(gt_mag_np))
        if max_gt > 0:
            support = np.max(gt_mag_np, axis=0) > (1e-3 * max_gt)
            if support.any():
                foreground_mask = support
                foreground_fraction = float(support.mean())

    if foreground_mask is not None:
        gt_mag_np = gt_mag.squeeze().detach().cpu().numpy()  # (T,H,W)
        zf_mag_np = zf_mag.squeeze().detach().cpu().numpy()  # (T,H,W)
        # Convert to (H,W,T) for convenient masking.
        gt_mag_np_hw = np.transpose(gt_mag_np, (1, 2, 0))
        zf_mag_np_hw = np.transpose(zf_mag_np, (1, 2, 0))
        dr = float(gt_mag_np_hw.max() - gt_mag_np_hw.min())
        zf_mse_fg = float(np.mean((zf_mag_np_hw - gt_mag_np_hw)[foreground_mask] ** 2))
        zf_psnr_fg = None
        if dr > 0 and zf_mse_fg > 0:
            zf_psnr_fg = float(20.0 * np.log10(dr) - 10.0 * np.log10(zf_mse_fg))
        aux.update(
            {
                "zf_psnr_fg": zf_psnr_fg,
                "zf_mse_fg": zf_mse_fg,
                "fg_fraction": foreground_fraction,
            }
        )

    if return_aux:
        return ssim_zf, psnr_zf, mse_zf, lpips_zf, dc_mse_zf, dc_mae_zf, aux
    return ssim_zf, psnr_zf, mse_zf, lpips_zf, dc_mse_zf, dc_mae_zf



def eval_sample(
    kspace,
    csmap,
    ground_truth,
    x_recon,
    physics,
    mask,
    grasp_img,
    acceleration,
    spokes_per_frame,
    output_dir,
    label,
    device,
    cluster,
    dro_eval=True,
    grasp_path=None,
    raw_slice_idx=None,
    rescale=True,
    report_bestfit_dc: bool = False,
    filename_suffix="",
    baseline_mode: str = "fraction",
    baseline_seconds: float = 20.0,
    baseline_fraction: float = 0.1,
    baseline_min_frames: int = 4,
    baseline_max_frames: Optional[int] = 10,
    arrival_k: float = 3.0,
    arrival_method: str = "threshold",
    arrival_fraction: float = 0.1,
    early_seconds: float = 35.0,
    early_min_frames: int = 4,
    early_max_frames: Optional[int] = 8,
    total_scan_seconds: float = 150.0,
    plot_arrival: bool = False,
    arrival_percentile: float = 0.95,
    arrival_baseline_k: float = 2.0,
    arrival_method_plot: str | None = None,
    arrival_fraction_plot: float | None = None,
    arrival_pre_contrast_baseline: str = "n_frames",
    arrival_baseline_seconds: float = 20.0,
    arrival_total_seconds: float = 150.0,
    dynamic_dce_enabled: bool = True,
    dynamic_dce_top_fraction: float = 0.02,
    dynamic_dce_weight_power: float = 1.0,
    dynamic_dce_reference: str = "grasp",
    kspace_readout_samples: Optional[int] = None,
):

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

    extra_metrics = {}
    dc_mse_bestfit, dc_mae_bestfit, dc_scale = calc_dc_bestfit(recon_kspace, kspace, device)
    if dc_mse_bestfit is not None and dc_scale is not None:
        extra_metrics.update(
            {
                "dl_dc_mse_bestfit": dc_mse_bestfit,
                "dl_dc_mae_bestfit": dc_mae_bestfit,
                "dl_dc_scale_abs": float(abs(dc_scale)),
                "dl_dc_scale_phase": float(np.angle(dc_scale)),
            }
        )
    if not dro_eval:
        dl_kspace_metrics = compute_kspace_residual_metrics(
            pred_kspace=recon_kspace,
            ref_kspace=kspace,
            expected_readout_samples=kspace_readout_samples,
        )
        if dl_kspace_metrics:
            extra_metrics.update(
                {
                    "dl_raw_dc_rel_l2": dl_kspace_metrics.get("rel_l2"),
                    "dl_raw_dc_rel_l2_low": dl_kspace_metrics.get("rel_l2_low"),
                    "dl_raw_dc_rel_l2_mid": dl_kspace_metrics.get("rel_l2_mid"),
                    "dl_raw_dc_rel_l2_high": dl_kspace_metrics.get("rel_l2_high"),
                }
            )

        grasp_recon_complex_dc = rearrange(to_torch_complex(grasp_img).squeeze(), 'h t w -> h w t')
        grasp_kspace_dc = physics(False, grasp_recon_complex_dc.to(csmap.dtype), csmap)
        grasp_kspace_metrics = compute_kspace_residual_metrics(
            pred_kspace=grasp_kspace_dc,
            ref_kspace=kspace,
            expected_readout_samples=kspace_readout_samples,
        )
        if grasp_kspace_metrics:
            extra_metrics.update(
                {
                    "grasp_raw_dc_rel_l2": grasp_kspace_metrics.get("rel_l2"),
                    "grasp_raw_dc_rel_l2_low": grasp_kspace_metrics.get("rel_l2_low"),
                    "grasp_raw_dc_rel_l2_mid": grasp_kspace_metrics.get("rel_l2_mid"),
                    "grasp_raw_dc_rel_l2_high": grasp_kspace_metrics.get("rel_l2_high"),
                }
            )

    # RESCALE

    # calculate the single optimal scaling factor 'c'
    x_recon_np = x_recon.cpu().numpy()
    ground_truth_np = ground_truth.cpu().numpy()
    grasp_recon_np = grasp_img.cpu().numpy()

    x_recon_scale_np = x_recon_np
    if x_recon_np.ndim == 5 and ground_truth_np.ndim == 5:
        # Align (B,2,H,W,T) -> (B,2,T,H,W) for best-fit scalar computation.
        if (
            x_recon_np.shape[:2] == ground_truth_np.shape[:2]
            and x_recon_np.shape[2] == ground_truth_np.shape[3]
            and x_recon_np.shape[3] == ground_truth_np.shape[4]
            and x_recon_np.shape[4] == ground_truth_np.shape[2]
        ):
            x_recon_scale_np = np.transpose(x_recon_np, (0, 1, 4, 2, 3))

    denom = float(np.dot(x_recon_scale_np.flatten(), x_recon_scale_np.flatten()))
    c = float(np.dot(x_recon_scale_np.flatten(), ground_truth_np.flatten()) / (denom + 1e-12)) if denom > 0 else 1.0
    grasp_recon_scale_np = grasp_recon_np
    if (
        grasp_recon_np.ndim == 5
        and ground_truth_np.ndim == 5
        and grasp_recon_np.shape[:2] == ground_truth_np.shape[:2]
    ):
        # Align GRASP ordering to match GT: (B,2,T,H,W). In this codebase, GRASP tensors are
        # sometimes stored as (B,2,H,T,W) or (B,2,H,W,T).
        T = int(ground_truth_np.shape[2])
        if grasp_recon_np.shape[2] == T:
            grasp_recon_scale_np = grasp_recon_np
        elif grasp_recon_np.shape[3] == T:
            grasp_recon_scale_np = np.transpose(grasp_recon_np, (0, 1, 3, 2, 4))
        elif grasp_recon_np.shape[4] == T:
            grasp_recon_scale_np = np.transpose(grasp_recon_np, (0, 1, 4, 2, 3))

    denom_grasp = float(np.dot(grasp_recon_scale_np.flatten(), grasp_recon_scale_np.flatten()))
    c_grasp = (
        float(np.dot(grasp_recon_scale_np.flatten(), ground_truth_np.flatten()) / (denom_grasp + 1e-12))
        if denom_grasp > 0
        else 1.0
    )

    extra_metrics.update(
        {
            "dl_img_scale": c,
            "grasp_img_scale": c_grasp,
        }
    )

    if rescale:
        recon_complex_scaled = torch.tensor(c * x_recon_np, device=device)
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
    # data_range = max_val - min_val



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
        foreground_mask = None
        foreground_fraction = None
        if masks_np:
            union = np.zeros(recon_mag_np.shape[:2], dtype=bool)
            for mask_arr in masks_np.values():
                if mask_arr is None:
                    continue
                mask_arr = np.asarray(mask_arr).squeeze().astype(bool)
                if mask_arr.shape != union.shape:
                    continue
                union |= mask_arr
            if union.any() and union.mean() < 0.999:
                foreground_mask = union
                foreground_fraction = float(union.mean())

        if foreground_mask is None:
            max_val = float(np.max(gt_mag_np))
            if max_val > 0:
                support = np.max(gt_mag_np, axis=2) > (1e-3 * max_val)
                if support.any():
                    foreground_mask = support
                    foreground_fraction = float(support.mean())
        roi_source = None
        region_label_map = None
        if "malignant" in masks_np and masks_np["malignant"].any():
            roi_source = "malignant"
        elif "benign" in masks_np and masks_np["benign"].any():
            roi_source = "benign"
            print(f"Using benign ROI for plots (no malignant mask) for {patient_id or plot_label}.")
        else:
            fg_mask, fg_fraction = _infer_foreground_mask_from_stack(grasp_mag_np)
            if fg_mask is not None:
                masks_np = {"foreground": fg_mask}
                roi_source = "foreground"
                region_label_map = {"foreground": "Foreground"}
                print(
                    "No ROI mask found; plotting foreground mean curve "
                    f"(mask fraction {fg_fraction:.3f}) for {patient_id or plot_label}."
                )
            else:
                masks_np = {"full": np.ones(recon_mag_np.shape[:2], dtype=bool)}
                roi_source = "full"
                region_label_map = {"full": "Full"}
                print(f"No ROI mask found; plotting whole-image mean curve for {patient_id or plot_label}.")

        num_frames = recon_mag_np.shape[2]

        aif_time_points = np.linspace(0, total_scan_seconds, num_frames)

        arrival_method_plot = (arrival_method_plot or arrival_method or "threshold").lower()
        if arrival_fraction_plot is None:
            arrival_fraction_plot = arrival_fraction

        temporal_metrics = {}
        if 'malignant' in masks_np and masks_np['malignant'].any():
            dl_metrics = compute_temporal_metrics(
                gt_mag_np,
                recon_mag_np,
                masks_np['malignant'],
                aif_time_points,
                baseline_mode=baseline_mode,
                baseline_seconds=baseline_seconds,
                baseline_fraction=baseline_fraction,
                baseline_min_frames=baseline_min_frames,
                baseline_max_frames=baseline_max_frames,
                arrival_k=arrival_k,
                arrival_method=arrival_method,
                arrival_fraction=arrival_fraction,
                early_seconds=early_seconds,
                early_min_frames=early_min_frames,
                early_max_frames=early_max_frames,
            )
            grasp_metrics = compute_temporal_metrics(
                gt_mag_np,
                grasp_mag_np,
                masks_np['malignant'],
                aif_time_points,
                baseline_mode=baseline_mode,
                baseline_seconds=baseline_seconds,
                baseline_fraction=baseline_fraction,
                baseline_min_frames=baseline_min_frames,
                baseline_max_frames=baseline_max_frames,
                arrival_k=arrival_k,
                arrival_method=arrival_method,
                arrival_fraction=arrival_fraction,
                early_seconds=early_seconds,
                early_min_frames=early_min_frames,
                early_max_frames=early_max_frames,
            )
            temporal_metrics = {f"dl_{key}": val for key, val in dl_metrics.items()}
            temporal_metrics.update({f"grasp_{key}": val for key, val in grasp_metrics.items()})
        if 'benign' in masks_np and masks_np.get('benign', None) is not None and masks_np['benign'].any():
            dl_metrics = compute_temporal_metrics(
                gt_mag_np,
                recon_mag_np,
                masks_np['benign'],
                aif_time_points,
                baseline_mode=baseline_mode,
                baseline_seconds=baseline_seconds,
                baseline_fraction=baseline_fraction,
                baseline_min_frames=baseline_min_frames,
                baseline_max_frames=baseline_max_frames,
                arrival_k=arrival_k,
                arrival_method=arrival_method,
                arrival_fraction=arrival_fraction,
                early_seconds=early_seconds,
                early_min_frames=early_min_frames,
                early_max_frames=early_max_frames,
            )
            grasp_metrics = compute_temporal_metrics(
                gt_mag_np,
                grasp_mag_np,
                masks_np['benign'],
                aif_time_points,
                baseline_mode=baseline_mode,
                baseline_seconds=baseline_seconds,
                baseline_fraction=baseline_fraction,
                baseline_min_frames=baseline_min_frames,
                baseline_max_frames=baseline_max_frames,
                arrival_k=arrival_k,
                arrival_method=arrival_method,
                arrival_fraction=arrival_fraction,
                early_seconds=early_seconds,
                early_min_frames=early_min_frames,
                early_max_frames=early_max_frames,
            )
            temporal_metrics.update({f"benign_dl_{key}": val for key, val in dl_metrics.items()})
            temporal_metrics.update({f"benign_grasp_{key}": val for key, val in grasp_metrics.items()})

        if foreground_mask is not None:
            data_range = float(gt_mag_np.max() - gt_mag_np.min())
            dl_mse_fg = float(np.mean((recon_mag_np - gt_mag_np)[foreground_mask] ** 2))
            grasp_mse_fg = float(np.mean((grasp_mag_np - gt_mag_np)[foreground_mask] ** 2))
            dl_psnr_fg = None
            grasp_psnr_fg = None
            if data_range > 0 and dl_mse_fg > 0:
                dl_psnr_fg = float(20.0 * np.log10(data_range) - 10.0 * np.log10(dl_mse_fg))
            if data_range > 0 and grasp_mse_fg > 0:
                grasp_psnr_fg = float(20.0 * np.log10(data_range) - 10.0 * np.log10(grasp_mse_fg))
            temporal_metrics.update(
                {
                    "dl_psnr_fg": dl_psnr_fg,
                    "dl_mse_fg": dl_mse_fg,
                    "grasp_psnr_fg": grasp_psnr_fg,
                    "grasp_mse_fg": grasp_mse_fg,
                    "fg_fraction": foreground_fraction,
                }
            )
        if extra_metrics:
            temporal_metrics.update(extra_metrics)

        primary_region = None
        if "malignant" in masks_np and masks_np["malignant"].any():
            primary_region = "malignant"
        elif "benign" in masks_np and masks_np["benign"].any():
            primary_region = "benign"
        elif "foreground" in masks_np and masks_np["foreground"].any():
            primary_region = "foreground"
        elif "full" in masks_np and masks_np["full"].any():
            primary_region = "full"

        if dynamic_dce_enabled:
            ref_mode = str(dynamic_dce_reference or "grasp").strip().lower()
            if ref_mode == "grasp":
                ref_mag_np = grasp_mag_np
            elif ref_mode in {"gt", "ground_truth", "dro"}:
                ref_mag_np = gt_mag_np
            else:
                raise ValueError(
                    f"Unsupported dynamic_dce_reference='{dynamic_dce_reference}'. "
                    "Use one of: grasp, gt."
                )
            dynamic_roi = masks_np.get(primary_region) if primary_region in masks_np else None
            dyn_metrics = compute_dynamic_weighted_dce_error(
                reference_mag_np=ref_mag_np,
                recon_mag_np=recon_mag_np,
                time_points=aif_time_points,
                roi_mask=dynamic_roi,
                baseline_mode=baseline_mode,
                baseline_seconds=baseline_seconds,
                baseline_fraction=baseline_fraction,
                baseline_min_frames=baseline_min_frames,
                baseline_max_frames=baseline_max_frames,
                top_fraction=dynamic_dce_top_fraction,
                weight_power=dynamic_dce_weight_power,
            )
            if dyn_metrics:
                temporal_metrics["dl_raw_dyn_dce_mae"] = dyn_metrics["weighted_mae"]
                temporal_metrics["dl_raw_dyn_dce_mse"] = dyn_metrics["weighted_mse"]
                temporal_metrics["dl_raw_dyn_dce_topk_mae"] = dyn_metrics["topk_mae"]
                temporal_metrics["dl_raw_dyn_dce_topk_mse"] = dyn_metrics["topk_mse"]
                temporal_metrics["dl_raw_dyn_dce_selected_pixels"] = dyn_metrics["selected_pixels"]
                temporal_metrics["dl_raw_dyn_dce_selected_fraction"] = dyn_metrics["selected_fraction"]

                grasp_dyn_metrics = compute_dynamic_weighted_dce_error(
                    reference_mag_np=ref_mag_np,
                    recon_mag_np=grasp_mag_np,
                    time_points=aif_time_points,
                    roi_mask=dynamic_roi,
                    baseline_mode=baseline_mode,
                    baseline_seconds=baseline_seconds,
                    baseline_fraction=baseline_fraction,
                    baseline_min_frames=baseline_min_frames,
                    baseline_max_frames=baseline_max_frames,
                    top_fraction=dynamic_dce_top_fraction,
                    weight_power=dynamic_dce_weight_power,
                )
                if grasp_dyn_metrics:
                    temporal_metrics["grasp_raw_dyn_dce_mae"] = grasp_dyn_metrics["weighted_mae"]
                    temporal_metrics["grasp_raw_dyn_dce_mse"] = grasp_dyn_metrics["weighted_mse"]

        tumor_mask_for_plot = (
            None if primary_region in (None, "full", "foreground") else masks_np.get(primary_region)
        )
        if primary_region is not None and plot_label is not None:
            
            # --- Plot Spatial Quality at the central timepoint ---
            peak_frame = num_frames // 2
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
                tumor_mask=tumor_mask_for_plot,
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
                plot_dro=True,
                region_label_map=region_label_map,
                show_arrival=plot_arrival,
                arrival_percentile=arrival_percentile,
                arrival_baseline_k=arrival_baseline_k,
                arrival_method=arrival_method_plot,
                arrival_fraction=arrival_fraction_plot,
                arrival_pre_contrast_baseline=arrival_pre_contrast_baseline,
                arrival_baseline_seconds=arrival_baseline_seconds,
                arrival_total_seconds=arrival_total_seconds,
            )
            plot_temporal_curves_normalized(
                gt_img_stack=gt_mag_np,
                recon_img_stack=recon_mag_np,
                grasp_img_stack=grasp_mag_np,
                masks=masks_np,
                time_points=aif_time_points,
                filename=os.path.join(output_dir, f"temporal_curves_normalized_{plot_label}{suffix}.png"),
                acceleration=acceleration,
                spokes_per_frame=spokes_per_frame,
                baseline_mode=baseline_mode,
                baseline_seconds=baseline_seconds,
                baseline_fraction=baseline_fraction,
                baseline_min_frames=baseline_min_frames,
                baseline_max_frames=baseline_max_frames,
                region_label_map=region_label_map,
                show_arrival=plot_arrival,
                arrival_percentile=arrival_percentile,
                arrival_baseline_k=arrival_baseline_k,
                arrival_method=arrival_method_plot,
                arrival_fraction=arrival_fraction_plot,
                arrival_pre_contrast_baseline=arrival_pre_contrast_baseline,
                arrival_baseline_seconds=arrival_baseline_seconds,
                arrival_total_seconds=arrival_total_seconds,
            )

            if primary_region in ("malignant", "benign"):
                plot_single_temporal_curve(
                    img_stack=recon_mag_np,
                    masks=masks_np,
                    time_points=aif_time_points,
                    num_frames=num_frames,
                    filename=os.path.join(output_dir, f"recon_temporal_curve_{plot_label}{suffix}.png"),
                    acceleration=acceleration,
                    spokes_per_frame=spokes_per_frame,
                    region_key=primary_region,
                    show_arrival=plot_arrival,
                    arrival_percentile=arrival_percentile,
                    arrival_baseline_k=arrival_baseline_k,
                    arrival_method=arrival_method_plot,
                    arrival_fraction=arrival_fraction_plot,
                    arrival_pre_contrast_baseline=arrival_pre_contrast_baseline,
                    arrival_baseline_seconds=arrival_baseline_seconds,
                    arrival_total_seconds=arrival_total_seconds,
                )

            plot_time_series(
                recon_img_stack=recon_mag_np,
                grasp_img_stack=grasp_mag_np,
                filename=os.path.join(output_dir, f"time_points_{plot_label}{suffix}.png"),
                acceleration=acceleration,
                spokes_per_frame=spokes_per_frame,
                tumor_mask=tumor_mask_for_plot,
            )

            print("Diagnostic plots saved.")
        else:
            region_corrs = {}

        recon_corr = region_corrs.get(primary_region, {}).get("DL") if primary_region else None
        grasp_corr = region_corrs.get(primary_region, {}).get("GRASP") if primary_region else None

        return ssim, psnr, mse, lpips, dc_mse, dc_mae, recon_corr, grasp_corr, temporal_metrics
    

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

        masks_np = {key: val.cpu().numpy().squeeze().astype(bool) for key, val in mask.items() if key in ('malignant', 'benign')}
        roi_source = None
        region_label_map = None
        if "malignant" in masks_np and masks_np["malignant"].any():
            roi_source = "malignant"
        elif "benign" in masks_np and masks_np["benign"].any():
            roi_source = "benign"
            print(f"Using benign ROI for plots (no malignant mask) for {patient_id or plot_label}.")
        else:
            masks_np = {"full": np.ones(recon_mag_np.shape[:2], dtype=bool)}
            roi_source = "full"
            region_label_map = {"full": "Full"}
            print(f"No ROI mask found; plotting whole-image mean curve for {patient_id or plot_label}.")

        num_frames = recon_mag_np.shape[2]

        aif_time_points = np.linspace(0, total_scan_seconds, num_frames)

        temporal_metrics = {}
        if 'malignant' in masks_np and masks_np['malignant'].any():
            dl_metrics = compute_temporal_metrics(
                gt_mag_np,
                recon_mag_np,
                masks_np['malignant'],
                aif_time_points,
                baseline_mode=baseline_mode,
                baseline_seconds=baseline_seconds,
                baseline_fraction=baseline_fraction,
                baseline_min_frames=baseline_min_frames,
                baseline_max_frames=baseline_max_frames,
                arrival_k=arrival_k,
                arrival_method=arrival_method,
                arrival_fraction=arrival_fraction,
                early_seconds=early_seconds,
                early_min_frames=early_min_frames,
                early_max_frames=early_max_frames,
            )
            grasp_metrics = compute_temporal_metrics(
                gt_mag_np,
                grasp_mag_np,
                masks_np['malignant'],
                aif_time_points,
                baseline_mode=baseline_mode,
                baseline_seconds=baseline_seconds,
                baseline_fraction=baseline_fraction,
                baseline_min_frames=baseline_min_frames,
                baseline_max_frames=baseline_max_frames,
                arrival_k=arrival_k,
                arrival_method=arrival_method,
                arrival_fraction=arrival_fraction,
                early_seconds=early_seconds,
                early_min_frames=early_min_frames,
                early_max_frames=early_max_frames,
            )
            temporal_metrics = {f"dl_{key}": val for key, val in dl_metrics.items()}
            temporal_metrics.update({f"grasp_{key}": val for key, val in grasp_metrics.items()})
        if 'benign' in masks_np and masks_np.get('benign', None) is not None and masks_np['benign'].any():
            dl_metrics = compute_temporal_metrics(
                gt_mag_np,
                recon_mag_np,
                masks_np['benign'],
                aif_time_points,
                baseline_mode=baseline_mode,
                baseline_seconds=baseline_seconds,
                baseline_fraction=baseline_fraction,
                baseline_min_frames=baseline_min_frames,
                baseline_max_frames=baseline_max_frames,
                arrival_k=arrival_k,
                arrival_method=arrival_method,
                arrival_fraction=arrival_fraction,
                early_seconds=early_seconds,
                early_min_frames=early_min_frames,
                early_max_frames=early_max_frames,
            )
            grasp_metrics = compute_temporal_metrics(
                gt_mag_np,
                grasp_mag_np,
                masks_np['benign'],
                aif_time_points,
                baseline_mode=baseline_mode,
                baseline_seconds=baseline_seconds,
                baseline_fraction=baseline_fraction,
                baseline_min_frames=baseline_min_frames,
                baseline_max_frames=baseline_max_frames,
                arrival_k=arrival_k,
                arrival_method=arrival_method,
                arrival_fraction=arrival_fraction,
                early_seconds=early_seconds,
                early_min_frames=early_min_frames,
                early_max_frames=early_max_frames,
            )
            temporal_metrics.update({f"benign_dl_{key}": val for key, val in dl_metrics.items()})
            temporal_metrics.update({f"benign_grasp_{key}": val for key, val in grasp_metrics.items()})
        if extra_metrics:
            temporal_metrics.update(extra_metrics)

        primary_region = None
        if "malignant" in masks_np and masks_np["malignant"].any():
            primary_region = "malignant"
        elif "benign" in masks_np and masks_np["benign"].any():
            primary_region = "benign"
        elif "foreground" in masks_np and masks_np["foreground"].any():
            primary_region = "foreground"
        elif "full" in masks_np and masks_np["full"].any():
            primary_region = "full"

        tumor_mask_for_plot = (
            None if primary_region in (None, "full", "foreground") else masks_np.get(primary_region)
        )
        if primary_region is not None and plot_label is not None:
            
            # --- Plot Spatial Quality at the central timepoint ---
            peak_frame = num_frames // 2
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
                tumor_mask=tumor_mask_for_plot,
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
                plot_dro=False,
                region_label_map=region_label_map,
                show_arrival=plot_arrival,
                arrival_percentile=arrival_percentile,
                arrival_baseline_k=arrival_baseline_k,
                arrival_method=arrival_method_plot,
                arrival_fraction=arrival_fraction_plot,
                arrival_pre_contrast_baseline=arrival_pre_contrast_baseline,
                arrival_baseline_seconds=arrival_baseline_seconds,
                arrival_total_seconds=arrival_total_seconds,
            )
            plot_temporal_curves_normalized(
                gt_img_stack=gt_mag_np,
                recon_img_stack=recon_mag_np,
                grasp_img_stack=grasp_mag_np,
                masks=masks_np,
                time_points=aif_time_points,
                filename=os.path.join(output_dir, f"non_dro_temporal_curves_normalized_{plot_label}{suffix}.png"),
                acceleration=acceleration,
                spokes_per_frame=spokes_per_frame,
                plot_dro=False,
                baseline_mode=baseline_mode,
                baseline_seconds=baseline_seconds,
                baseline_fraction=baseline_fraction,
                baseline_min_frames=baseline_min_frames,
                baseline_max_frames=baseline_max_frames,
                region_label_map=region_label_map,
                show_arrival=plot_arrival,
                arrival_percentile=arrival_percentile,
                arrival_baseline_k=arrival_baseline_k,
                arrival_method=arrival_method_plot,
                arrival_fraction=arrival_fraction_plot,
                arrival_pre_contrast_baseline=arrival_pre_contrast_baseline,
                arrival_baseline_seconds=arrival_baseline_seconds,
                arrival_total_seconds=arrival_total_seconds,
            )

            if primary_region in ("malignant", "benign"):
                plot_single_temporal_curve(
                    img_stack=recon_mag_np,
                    masks=masks_np,
                    time_points=aif_time_points,
                    num_frames=num_frames,
                    filename=os.path.join(output_dir, f"non_dro_recon_temporal_curve_{plot_label}{suffix}.png"),
                    acceleration=acceleration,
                    spokes_per_frame=spokes_per_frame,
                    region_key=primary_region,
                    show_arrival=plot_arrival,
                    arrival_percentile=arrival_percentile,
                    arrival_baseline_k=arrival_baseline_k,
                    arrival_method=arrival_method_plot,
                    arrival_fraction=arrival_fraction_plot,
                    arrival_pre_contrast_baseline=arrival_pre_contrast_baseline,
                    arrival_baseline_seconds=arrival_baseline_seconds,
                    arrival_total_seconds=arrival_total_seconds,
                )

            plot_time_series(
                recon_img_stack=recon_mag_np,
                grasp_img_stack=grasp_mag_np,
                filename=os.path.join(output_dir, f"non_dro_time_points_{plot_label}{suffix}.png"),
                acceleration=acceleration,
                spokes_per_frame=spokes_per_frame,
                tumor_mask=tumor_mask_for_plot,
            )

            print("Diagnostic plots saved.")


        return dc_mse, dc_mae, temporal_metrics





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
        
        # --- Plot Spatial Quality at the central timepoint ---
        peak_frame = num_frames // 2
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
