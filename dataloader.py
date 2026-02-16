import glob
import os
import csv
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
from einops import rearrange
import random
import sigpy as sp
from utils import prep_nufft
from radial_lsfp import MCNUFFT
import time
from typing import Union, List, Optional
import re
import pandas as pd
from tqdm import tqdm
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO_ROOT = Path(__file__).resolve().parent
SLICE_MAP_PATH = REPO_ROOT / "data" / "largest_tumor_slices.csv"

def _get_distributed_rank() -> int:
    """Best-effort rank detection for DDP/SLURM/PMI setups."""
    for key in ("RANK", "SLURM_PROCID", "PMI_RANK", "LOCAL_RANK"):
        val = os.environ.get(key)
        if val is None:
            continue
        try:
            return int(val)
        except ValueError:
            continue
    return 0


def _should_log_once(dataset_obj: object, flag_attr: str) -> bool:
    """
    Log a noisy dataloader message at most once per dataset instance, and only
    from rank 0 / worker 0 (when applicable).
    """
    if getattr(dataset_obj, flag_attr, False):
        return False

    # Avoid multi-GPU spam.
    if _get_distributed_rank() != 0:
        return False

    # Avoid multi-worker spam.
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None and worker_info.id != 0:
        return False

    setattr(dataset_obj, flag_attr, True)
    return True


def _slice_sampling_cache_stats(dataset_obj: object) -> Optional[dict]:
    volume_map = getattr(dataset_obj, "volume_map", None)
    if not volume_map:
        return None

    cache_hits_mem = 0
    cache_hits_disk = 0
    cache_misses = 0
    cache_dir = getattr(dataset_obj, "slice_sampling_cache_dir", None)
    cache_workers = int(getattr(dataset_obj, "slice_sampling_cache_workers", 0) or 0)

    for file_path, num_slices, num_spokes, num_samples in volume_map:
        cached = getattr(dataset_obj, "_slice_score_cache", {}).get(file_path)
        if cached is not None and len(cached.get("background", [])) == int(num_slices):
            cache_hits_mem += 1
            continue
        disk_cache = None
        if hasattr(dataset_obj, "_load_slice_scores_from_disk"):
            disk_cache = dataset_obj._load_slice_scores_from_disk(
                file_path, num_slices, num_spokes, num_samples
            )
        if disk_cache is not None:
            cache_hits_disk += 1
            if hasattr(dataset_obj, "_slice_score_cache"):
                dataset_obj._slice_score_cache[file_path] = disk_cache
        else:
            cache_misses += 1

    total = cache_hits_mem + cache_hits_disk + cache_misses
    will_compute = cache_misses > 0
    return {
        "total": total,
        "hits_mem": cache_hits_mem,
        "hits_disk": cache_hits_disk,
        "misses": cache_misses,
        "will_compute": will_compute,
        "cache_dir": cache_dir,
        "cache_workers": cache_workers,
    }


def _summarize_scores(values: np.ndarray) -> Optional[dict]:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def _safe_percent(value: float) -> float:
    try:
        return 100.0 * float(value)
    except (TypeError, ValueError):
        return float("nan")


def log_slice_sampling_startup_report(
    dataset_obj: object,
    label: str = "train",
    output_dir: Optional[str] = None,
    plot_histograms: bool = True,
) -> None:
    num_random_slices = getattr(dataset_obj, "num_random_slices", None)
    volume_map = getattr(dataset_obj, "volume_map", None)
    if num_random_slices is None or not volume_map:
        print(f"[{label}] Slice sampling sanity report skipped (random sampling not enabled).")
        return

    try:
        num_random_slices = int(num_random_slices)
    except (TypeError, ValueError):
        print(f"[{label}] Slice sampling sanity report skipped (invalid num_random_slices).")
        return

    if num_random_slices <= 0:
        print(f"[{label}] Slice sampling sanity report skipped (num_random_slices <= 0).")
        return

    slice_sampling_mode = str(getattr(dataset_obj, "slice_sampling_mode", "uniform")).lower()
    try:
        uniform_fraction = float(getattr(dataset_obj, "slice_sampling_uniform_fraction", 1.0))
    except (TypeError, ValueError):
        uniform_fraction = 1.0
    try:
        filter_quantile = float(getattr(dataset_obj, "slice_sampling_filter_quantile", 0.0))
    except (TypeError, ValueError):
        filter_quantile = 0.0
    no_replacement = bool(getattr(dataset_obj, "slice_sampling_no_replacement", False))

    uniform_fraction = min(max(uniform_fraction, 0.0), 1.0)
    n_uniform = int(round(num_random_slices * uniform_fraction))
    n_uniform = min(n_uniform, num_random_slices)
    n_weighted = num_random_slices - n_uniform

    print(f"[{label}] Slice sampling sanity report:")
    print(
        "  settings: "
        f"mode={slice_sampling_mode}, num_random_slices={num_random_slices}, "
        f"n_uniform={n_uniform}, n_weighted={n_weighted}, "
        f"uniform_fraction={uniform_fraction:.3f}, filter_quantile={filter_quantile:.3f}, "
        f"no_replacement={no_replacement}"
    )

    if slice_sampling_mode == "uniform" or uniform_fraction >= 1.0:
        print("  cache: uniform sampling (no slice score cache used).")
        return

    if slice_sampling_mode not in ("background", "nonenhancing"):
        print(f"  cache: unsupported mode '{slice_sampling_mode}' (no stats computed).")
        return

    score_key = "background" if slice_sampling_mode == "background" else "enhancement"
    score_desc = (
        "mean center-kspace magnitude"
        if slice_sampling_mode == "background"
        else "time-std enhancement proxy"
    )

    cache_stats = _slice_sampling_cache_stats(dataset_obj)
    if cache_stats is None:
        print("  cache: unavailable (missing volume map).")
        return

    print(
        "  cache: "
        f"hits(mem={cache_stats['hits_mem']}, disk={cache_stats['hits_disk']}), "
        f"misses={cache_stats['misses']}, total={cache_stats['total']}, "
        f"will_compute={cache_stats['will_compute']}, "
        f"cache_dir={cache_stats['cache_dir']}, cache_workers={cache_stats['cache_workers']}"
    )

    if hasattr(dataset_obj, "_warm_slice_score_cache"):
        dataset_obj._warm_slice_score_cache()

    all_scores = []
    eligible_scores = []
    filtered_scores = []
    cutoff_values = []
    eligible_fracs = []

    volume_names = []
    m_all_values = []
    m_weighted_values = []
    m_uniform_values = []
    zero_score_counts = []
    weighted_positive_counts = []

    expected_uniform_eligible = []
    expected_weighted_eligible = []
    expected_uniform_ineligible = []
    expected_weighted_ineligible = []
    expected_eligible_fraction = []
    expected_ineligible_fraction = []
    sample_counts = []

    total_slices = 0
    total_eligible = 0
    total_filtered = 0

    for file_path, num_slices, num_spokes, num_samples in volume_map:
        if num_slices is None or int(num_slices) <= 0:
            continue
        num_slices = int(num_slices)
        scores = dataset_obj._get_slice_scores(
            file_path, num_slices, num_spokes, num_samples
        )
        weights = np.asarray(scores.get(score_key, []), dtype=np.float64)
        if weights.shape[0] != num_slices:
            continue

        if filter_quantile > 0:
            cutoff = float(np.quantile(weights, filter_quantile))
            eligible_mask = weights >= cutoff
        else:
            cutoff = float(np.min(weights)) if weights.size else 0.0
            eligible_mask = np.ones_like(weights, dtype=bool)

        weights_after_filter = np.where(eligible_mask, weights, 0.0)
        zero_score_count = int(np.count_nonzero(weights == 0))
        weighted_positive_count = int(np.count_nonzero(weights_after_filter > 0))

        eligible_count = int(np.count_nonzero(eligible_mask))
        filtered_count = int(num_slices - eligible_count)
        eligible_frac = eligible_count / num_slices if num_slices else 0.0

        volume_names.append(os.path.basename(file_path))
        m_all_values.append(num_slices)
        m_uniform_values.append(num_slices)
        m_weighted_values.append(weighted_positive_count)
        zero_score_counts.append(zero_score_count)
        weighted_positive_counts.append(weighted_positive_count)

        eligible_fracs.append(eligible_frac)
        all_scores.append(weights)
        eligible_scores.append(weights[eligible_mask])
        filtered_scores.append(weights[~eligible_mask])
        if filter_quantile > 0:
            cutoff_values.append(cutoff)

        total_slices += num_slices
        total_eligible += eligible_count
        total_filtered += filtered_count

        sample_count = min(num_random_slices, num_slices)
        if num_slices <= num_random_slices:
            exp_uniform_eligible = float(eligible_count)
            exp_uniform_ineligible = float(filtered_count)
            exp_weighted_eligible = 0.0
            exp_weighted_ineligible = 0.0
        else:
            p = eligible_frac
            exp_uniform_eligible = float(n_uniform * p)
            exp_uniform_ineligible = float(n_uniform * (1.0 - p))
            expected_remaining_eligible = max(eligible_count - exp_uniform_eligible, 0.0)
            exp_weighted_eligible = float(min(n_weighted, expected_remaining_eligible))
            exp_weighted_ineligible = float(n_weighted - exp_weighted_eligible)

        exp_eligible = exp_uniform_eligible + exp_weighted_eligible
        exp_ineligible = exp_uniform_ineligible + exp_weighted_ineligible

        expected_uniform_eligible.append(exp_uniform_eligible)
        expected_uniform_ineligible.append(exp_uniform_ineligible)
        expected_weighted_eligible.append(exp_weighted_eligible)
        expected_weighted_ineligible.append(exp_weighted_ineligible)
        expected_eligible_fraction.append(exp_eligible / sample_count if sample_count else 0.0)
        expected_ineligible_fraction.append(exp_ineligible / sample_count if sample_count else 0.0)
        sample_counts.append(sample_count)

    if not all_scores:
        print("  stats: no slice scores available.")
        return

    all_scores_arr = np.concatenate(all_scores)
    eligible_scores_arr = np.concatenate(eligible_scores) if eligible_scores else np.array([])
    filtered_scores_arr = np.concatenate(filtered_scores) if filtered_scores else np.array([])

    eligible_fracs_arr = np.asarray(eligible_fracs, dtype=np.float64)
    m_all_arr = np.asarray(m_all_values, dtype=np.float64)
    m_weighted_arr = np.asarray(m_weighted_values, dtype=np.float64)
    m_uniform_arr = np.asarray(m_uniform_values, dtype=np.float64)
    zero_score_arr = np.asarray(zero_score_counts, dtype=np.float64)
    weighted_positive_arr = np.asarray(weighted_positive_counts, dtype=np.float64)
    sample_counts_arr = np.asarray(sample_counts, dtype=np.float64)

    total_samples = float(sample_counts_arr.sum()) if sample_counts_arr.size else 0.0
    if total_samples > 0:
        weighted_uniform_eligible = float(np.sum(expected_uniform_eligible))
        weighted_weighted_eligible = float(np.sum(expected_weighted_eligible))
        weighted_uniform_ineligible = float(np.sum(expected_uniform_ineligible))
        weighted_weighted_ineligible = float(np.sum(expected_weighted_ineligible))

        eligible_share = (weighted_uniform_eligible + weighted_weighted_eligible) / total_samples
        ineligible_share = (weighted_uniform_ineligible + weighted_weighted_ineligible) / total_samples
        weighted_eligible_share = weighted_weighted_eligible / total_samples
        uniform_eligible_share = weighted_uniform_eligible / total_samples
        weighted_ineligible_share = weighted_weighted_ineligible / total_samples
    else:
        eligible_share = ineligible_share = weighted_eligible_share = uniform_eligible_share = weighted_ineligible_share = 0.0

    print(
        "  eligible fraction by volume (after per-volume quantile): "
        f"mean {_safe_percent(eligible_fracs_arr.mean()):.1f}% "
        f"(min {_safe_percent(eligible_fracs_arr.min()):.1f}%, "
        f"max {_safe_percent(eligible_fracs_arr.max()):.1f}%, "
        f"overall {_safe_percent(total_eligible / total_slices if total_slices else 0.0):.1f}%)"
    )
    print(
        "  expected sample mix (overall, weighted by samples): "
        f"weighted-eligible {_safe_percent(weighted_eligible_share):.1f}%, "
        f"uniform-eligible {_safe_percent(uniform_eligible_share):.1f}%, "
        f"below-cutoff {_safe_percent(ineligible_share):.1f}%"
        + (f" (weighted fallback {_safe_percent(weighted_ineligible_share):.1f}%)" if weighted_ineligible_share > 0 else "")
    )

    if m_all_arr.size:
        fraction_weighted_arr = np.divide(
            m_weighted_arr, m_all_arr, out=np.zeros_like(m_weighted_arr), where=m_all_arr > 0
        )
    else:
        fraction_weighted_arr = np.array([])

    if n_weighted > 0:
        epochs_to_exhaust_weighted = np.floor(m_weighted_arr / float(n_weighted))
    else:
        epochs_to_exhaust_weighted = np.full_like(m_weighted_arr, np.nan, dtype=np.float64)

    if num_random_slices > 0:
        epochs_to_exhaust_total = np.floor(m_all_arr / float(num_random_slices))
    else:
        epochs_to_exhaust_total = np.full_like(m_all_arr, np.nan, dtype=np.float64)

    def _summarize_epochs(name: str, arr: np.ndarray) -> Optional[dict]:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            print(f"  {name}: n/a")
            return None
        summary = {
            "min": float(np.min(finite)),
            "median": float(np.median(finite)),
            "p10": float(np.percentile(finite, 10)),
        }
        print(
            f"  {name}: min {summary['min']:.0f}, "
            f"median {summary['median']:.0f}, p10 {summary['p10']:.0f}"
        )
        return summary

    print("  coverage/exhaustion estimates (epochs):")
    if not no_replacement:
        print("    note: no-replacement is disabled; exhaustion estimates are analytical only.")
    print("    pool basis: all_slices (current implementation)")
    print("    M_uniform definition: M_all (pool uses all slices)")
    weighted_summary = _summarize_epochs("epochs_to_exhaust_weighted", epochs_to_exhaust_weighted)
    total_summary = _summarize_epochs("epochs_to_exhaust_total", epochs_to_exhaust_total)

    first_failure_candidates = []
    first_failure_labels = []
    if weighted_summary is not None:
        first_failure_candidates.append(weighted_summary["min"])
        first_failure_labels.append("weighted")
    if total_summary is not None:
        first_failure_candidates.append(total_summary["min"])
        first_failure_labels.append("total")

    if first_failure_candidates:
        earliest = min(first_failure_candidates)
        label = first_failure_labels[int(np.argmin(first_failure_candidates))]
        print(f"  likely first failure epoch: {earliest:.0f} ({label})")

    print("  per-volume coverage stats:")
    for name, m_all, zero_count, weighted_pos, cutoff in zip(
        volume_names, m_all_values, zero_score_counts, weighted_positive_counts,
        cutoff_values if cutoff_values else [np.nan] * len(volume_names),
    ):
        frac_weighted = (weighted_pos / m_all) if m_all else 0.0
        cutoff_str = f"{cutoff:.6g}" if np.isfinite(cutoff) else "n/a"
        print(
            f"    {name}: M_all={m_all}, M_uniform={m_all}, score==0={zero_count}, "
            f"weight_after_filter>0={weighted_pos}, "
            f"frac_weighted={frac_weighted:.3f}, cutoff={cutoff_str}"
        )

    def _print_summary(name: str, arr: np.ndarray) -> None:
        summary = _summarize_scores(arr)
        if summary is None:
            print(f"  {name}: empty")
            return
        print(
            f"  {name}: "
            f"n={summary['count']}, min={summary['min']:.6g}, "
            f"median={summary['median']:.6g}, p95={summary['p95']:.6g}, "
            f"p99={summary['p99']:.6g}, max={summary['max']:.6g}, mean={summary['mean']:.6g}"
        )

    print(f"  score={score_desc}")
    _print_summary("scores(all)", all_scores_arr)
    _print_summary("scores(eligible)", eligible_scores_arr)
    _print_summary("scores(filtered-out)", filtered_scores_arr)

    if not plot_histograms or output_dir is None:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  histograms: skipped (matplotlib import failed: {exc})")
        return

    hist_dir = os.path.join(output_dir, "sampling_coverage")
    os.makedirs(hist_dir, exist_ok=True)

    hist_path = os.path.join(hist_dir, f"{label}_slice_sampling_hist.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    if all_scores_arr.size > 0:
        min_val = float(np.min(all_scores_arr))
        max_val = float(np.max(all_scores_arr))
    else:
        min_val, max_val = 0.0, 1.0
    if max_val <= min_val:
        max_val = min_val + 1e-6
    bins = np.linspace(min_val, max_val, 80)

    axes[0].hist(all_scores_arr, bins=bins, color="#4C78A8", alpha=0.8)
    axes[0].set_title("Enhancement score (all slices)")
    axes[0].set_xlabel("Score")
    axes[0].set_ylabel("Count")

    if filtered_scores_arr.size > 0 or eligible_scores_arr.size > 0:
        axes[1].hist(
            [filtered_scores_arr, eligible_scores_arr],
            bins=bins,
            stacked=True,
            color=["#F58518", "#54A24B"],
            label=["filtered-out (below cutoff)", "eligible (>= cutoff)"],
        )
    else:
        axes[1].hist(all_scores_arr, bins=bins, color="#4C78A8", alpha=0.8)
    axes[1].set_title("Eligible vs filtered-out (per-volume quantile)")
    axes[1].set_xlabel("Score")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    fig.suptitle(f"{label} slice sampling scores ({score_desc})", fontsize=11)
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)

    if m_all_arr.size:
        size_path = os.path.join(hist_dir, f"{label}_slice_sampling_pool_sizes.png")
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        axes[0].hist(m_weighted_arr, bins=30, color="#4C78A8", alpha=0.8)
        axes[0].set_title("M_weighted (weights>0)")
        axes[0].set_xlabel("Slices")
        axes[0].set_ylabel("Count")
        axes[1].hist(fraction_weighted_arr, bins=30, color="#54A24B", alpha=0.8)
        axes[1].set_title("M_weighted / M_all")
        axes[1].set_xlabel("Fraction")
        axes[1].set_ylabel("Count")
        if cutoff_values:
            axes[2].hist(cutoff_values, bins=30, color="#E45756", alpha=0.8)
            axes[2].set_title(f"Cutoff values (q={filter_quantile:.2f})")
            axes[2].set_xlabel("Cutoff")
            axes[2].set_ylabel("Count")
        else:
            axes[2].set_axis_off()
            axes[2].text(0.5, 0.5, "No cutoff values", ha="center", va="center")
        fig.savefig(size_path, dpi=150)
        plt.close(fig)

        finite_epochs = []
        if np.isfinite(epochs_to_exhaust_weighted).any():
            finite_epochs.append(np.nanmax(epochs_to_exhaust_weighted))
        if np.isfinite(epochs_to_exhaust_total).any():
            finite_epochs.append(np.nanmax(epochs_to_exhaust_total))
        max_est = int(max(finite_epochs)) if finite_epochs else 10
        max_epochs = int(min(max(10, max_est * 2), 500))
        epochs = np.arange(0, max_epochs + 1)

        sort_idx = np.argsort(m_weighted_arr)
        worst_idx = int(sort_idx[0])
        median_idx = int(sort_idx[len(sort_idx) // 2])
        best_idx = int(sort_idx[-1])
        curve_specs = [
            ("Worst (min M_weighted)", worst_idx),
            ("Median", median_idx),
            ("Best (max M_weighted)", best_idx),
        ]

        coverage_path = os.path.join(hist_dir, f"{label}_slice_sampling_coverage_curve.png")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        for ax, (title, idx) in zip(axes, curve_specs):
            m_all = float(m_all_arr[idx])
            m_weighted = float(m_weighted_arr[idx])
            unique_uniform = np.minimum(m_all, epochs * float(n_uniform))
            unique_weighted = np.minimum(m_weighted, epochs * float(n_weighted))
            unique_total = np.minimum(m_all, unique_uniform + unique_weighted)
            ax.plot(epochs, unique_uniform, label="uniform")
            ax.plot(epochs, unique_weighted, label="weighted")
            ax.plot(epochs, unique_total, label="total")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Expected unique slices")
            ax.legend()
        fig.savefig(coverage_path, dpi=150)
        plt.close(fig)

    if filter_quantile > 0 and cutoff_values:
        cutoff_path = os.path.join(hist_dir, f"{label}_slice_sampling_cutoffs.png")
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
        ax.hist(cutoff_values, bins=50, color="#E45756", alpha=0.8)
        ax.set_title(f"Per-volume cutoff values (q={filter_quantile:.2f})")
        ax.set_xlabel("Cutoff score")
        ax.set_ylabel("Count")
        fig.savefig(cutoff_path, dpi=150)
        plt.close(fig)

    print(f"  histograms: saved to {hist_dir}")


def _compute_slice_sampling_scores(
    file_path: str,
    dataset_key: str,
    spokes_per_frame: Optional[int],
    n_time: Optional[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-slice background and enhancement proxy scores from center k-space."""
    with h5py.File(file_path, "r") as f:
        if dataset_key not in f:
            raise KeyError(f"Dataset key '{dataset_key}' not found in file {file_path}")
        ds = f[dataset_key]
        if ds.ndim != 4:
            raise ValueError(f"Expected k-space dataset with 4 dims, got shape {ds.shape} in {file_path}")
        num_slices, _, num_spokes, num_samples = ds.shape
        center_idx = num_samples // 2
        center = ds[:, :, :, center_idx]  # (slices, coils, spokes)
        mag = np.abs(center).mean(axis=1)  # (slices, spokes)

        spf = spokes_per_frame
        if spf is None or spf <= 0:
            if n_time is None or n_time <= 0:
                raise ValueError("spokes_per_frame or n_time must be provided to compute slice scores.")
            spf = max(1, num_spokes // n_time)

        t_frames = num_spokes // spf
        if t_frames <= 0:
            return np.zeros(num_slices), np.zeros(num_slices)

        mag = mag[:, : t_frames * spf].reshape(num_slices, t_frames, spf)
        time_curve = mag.mean(axis=2)  # (slices, T)

        background_score = time_curve.mean(axis=1)
        enhancement_score = time_curve.std(axis=1)
        return background_score, enhancement_score


def _write_slice_sampling_cache(
    cache_path: str,
    background: np.ndarray,
    enhancement: np.ndarray,
    metadata: dict,
) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_base = f"{cache_path}.tmp"
    np.savez_compressed(
        tmp_base,
        background=background.astype(np.float32),
        enhancement=enhancement.astype(np.float32),
        **metadata,
    )
    os.replace(f"{tmp_base}.npz", cache_path)


def _compute_and_cache_slice_scores(args: tuple) -> str:
    (
        file_path,
        dataset_key,
        spokes_per_frame,
        n_time,
        cache_path,
        metadata,
    ) = args
    background, enhancement = _compute_slice_sampling_scores(
        file_path=file_path,
        dataset_key=dataset_key,
        spokes_per_frame=spokes_per_frame,
        n_time=n_time,
    )
    _write_slice_sampling_cache(cache_path, background, enhancement, metadata)
    return file_path


def load_slice_map(csv_path: Path) -> dict:
    """Load patient -> slice index map; return empty dict if missing."""
    if not csv_path.exists():
        print(f"Slice map not found at {csv_path}; falling back to configured slice indices.")
        return {}

    mapping = {}
    with open(csv_path, newline="") as f:
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


class ZFSliceDataset(Dataset):
    """
    A Dataset that:
      - Looks for all .h5/.hdf5 files under `root_dir`.
      - Each file is assumed to contain a dataset at `dataset_key`, with shape (... Z),
        where Z is the number of slices/partitions.
      - Can either use a fixed set of slices or randomly sample N slices per volume
        at the start of each epoch (optionally without replacement across epochs).
      - Returns each slice as a torch.Tensor.
    """

    def __init__(
        self,
        root_dir,
        patient_ids,
        dataset_key="kspace",
        file_pattern="*.h5",
        slice_idx: Optional[Union[int, range]] = 41,
        num_random_slices: Optional[int] = None,  # New parameter for random sampling
        slice_sampling_mode: str = "uniform",
        slice_sampling_uniform_fraction: float = 1.0,
        slice_sampling_filter_quantile: float = 0.2,
        slice_sampling_no_replacement: bool = False,
        slice_sampling_cache_dir: Optional[str] = None,
        slice_sampling_cache_workers: int = 0,
        slice_sampling_cache_rank: Optional[int] = None,
        slice_sampling_cache_rank_only: Optional[int] = None,
        N_time=8,
        N_coils=16,
        spf_aug=False,
        spokes_per_frame=None,
        weight_accelerations=False, 
        initial_spokes_range=[8, 16, 24, 36],
        cluster="Randi",
        flip_kspace=True,
    ):
        """
        Args:
            root_dir (str): Path to the folder containing all HDF5 k-space files.
            patient_ids (list): List of patient IDs to filter the files.
            dataset_key (str): The key/path inside each .h5 file to the k-space dataset.
            file_pattern (str): Glob pattern to match your HDF5 files.
            slice_idx (int, range, optional): A fixed slice index or range of indices to use.
                                              This is ignored if num_random_slices is set.
            num_random_slices (int, optional): If provided, the dataset will randomly sample
                                               this many slices from each volume at the beginning
                                               of each epoch.
        """
        super().__init__()

        self.root_dir = root_dir
        self.dataset_key = dataset_key
        self.slice_idx = slice_idx
        self.num_random_slices = num_random_slices
        self.N_time = N_time
        self.N_coils = N_coils
        self.spf_aug = spf_aug
        self.weight_acc = weight_accelerations
        self.cluster=cluster
        self.flip_kspace=flip_kspace
        self.spokes_per_frame = spokes_per_frame
        self.slice_sampling_mode = str(slice_sampling_mode).lower()
        self.slice_sampling_uniform_fraction = float(slice_sampling_uniform_fraction)
        self.slice_sampling_filter_quantile = float(slice_sampling_filter_quantile)
        self.slice_sampling_no_replacement = bool(slice_sampling_no_replacement)
        self.slice_sampling_cache_dir = slice_sampling_cache_dir
        self.slice_sampling_cache_workers = int(slice_sampling_cache_workers)
        self.slice_sampling_cache_rank = slice_sampling_cache_rank
        self.slice_sampling_cache_rank_only = slice_sampling_cache_rank_only
        self._slice_score_cache = {}
        self._slice_remaining = {}
        self._slice_score_cache_ready = False
        self._slice_score_cache_ready = False

        # Find all matching HDF5 files under root_dir
        all_files = sorted(glob.glob(os.path.join(root_dir, file_pattern)))
        print("Number of files in root directory: ", len(all_files))

        if len(all_files) == 0:
            raise RuntimeError(
                f"No files found in {root_dir} matching pattern {file_pattern}"
            )

        # filter file list by patient ID substring
        filtered = []
        for fp in all_files:
            fname = os.path.basename(fp)
            if any(pid in fname for pid in patient_ids):
                filtered.append(fp)

        self.file_list = filtered

        if len(self.file_list) == 0:
            raise RuntimeError("No files matched the provided patient_ids filter.")

        # Logic for random slice sampling
        if self.num_random_slices is not None:
            print(f"Initializing in random slice sampling mode with N={self.num_random_slices} slices per volume.")
            self.volume_map = []
            for fp in self.file_list:
                with h5py.File(fp, "r") as f:
                    if self.dataset_key not in f:
                        raise KeyError(f"Dataset key '{self.dataset_key}' not found in file {fp}")
                    ds_shape = f[self.dataset_key].shape
                    num_slices, _, num_spokes, num_samples = ds_shape
                    self.volume_map.append((fp, num_slices, num_spokes, num_samples))
                    if self.slice_sampling_no_replacement:
                        self._slice_remaining[fp] = list(range(num_slices))
            
            # Perform the initial random sampling for the first epoch
            self.resample_slices()
        
        # Original logic for fixed slices, executed only if not in random mode
        else:
            print(f"Initializing in fixed slice mode with slice_idx={self.slice_idx}.")
            self.slice_index_map = []
            for fp in self.file_list:
                with h5py.File(fp, "r") as f:
                    if self.dataset_key not in f:
                        raise KeyError(f"Dataset key '{self.dataset_key}' not found in file {fp}")
                    ds = f[self.dataset_key]
                    num_slices = ds.shape[0]

                slices_to_add = []
                if isinstance(self.slice_idx, int):
                    if self.slice_idx < num_slices:
                        slices_to_add = [self.slice_idx]
                    else:
                        print(f"Warning: slice_idx {self.slice_idx} is out of bounds for {fp} "
                              f"(size {num_slices}). Skipping this file for this slice.")
                elif isinstance(self.slice_idx, range):
                    slices_to_add = [s for s in self.slice_idx if s < num_slices]
                    if len(slices_to_add) < len(self.slice_idx):
                        print(f"Warning: Some requested slices were out of bounds for {fp}. "
                              f"Using only the valid slice indices from the provided range.")
                else:
                    raise TypeError(f"slice_idx must be an int, range, or None, but got {type(self.slice_idx)}")

                for z in slices_to_add:
                    self.slice_index_map.append((fp, z))

        print(f"Dataset initialized with {len(self.slice_index_map)} total slice examples.")

        # NOTE: removed ultra-high accelerations until curriculum learning is implemented
        self.spokes_range = initial_spokes_range
        self.update_spokes_weights()
    
    def update_spokes_weights(self):

        if self.weight_acc:
            self.spf_weights = [1.0 / spf for spf in self.spokes_range]
        else:
            self.spf_weights = [1.0 for _ in self.spokes_range]


    def resample_slices(self):
        """
        Resamples N unique slices from each volume. This should be called at the
        beginning of each training epoch to ensure the model sees different data.
        """
        if self.num_random_slices is None:
            # If not in random sampling mode, do nothing.
            return
        self._warm_slice_score_cache()

        self.slice_index_map = []
        for file_path, num_slices, num_spokes, num_samples in self.volume_map:
            if num_slices <= 0:
                continue
            if num_slices <= self.num_random_slices:
                # If the volume has fewer than N slices, take all of them.
                print(f"Warning: Volume {os.path.basename(file_path)} has only {num_slices} slices, "
                      f"which is less than the requested {self.num_random_slices}. Using all available slices.")
                selected_slices = list(range(num_slices))
                if self.slice_sampling_no_replacement:
                    self._slice_remaining[file_path] = []
            else:
                if self.slice_sampling_no_replacement:
                    pool = self._slice_remaining.get(file_path)
                    if not pool:
                        pool = list(range(num_slices))
                    if len(pool) >= self.num_random_slices:
                        selected_slices = self._sample_from_pool(
                            file_path, pool, num_slices, num_spokes, num_samples, num_to_sample=self.num_random_slices
                        )
                        self._slice_remaining[file_path] = [idx for idx in pool if idx not in selected_slices]
                    else:
                        # Finish the current cycle first, then reset the pool.
                        if self.slice_sampling_mode != "uniform" and self.slice_sampling_uniform_fraction < 1.0:
                            print(
                                "[SliceSampling] Reset no-replacement pool for "
                                f"{os.path.basename(file_path)} (mode={self.slice_sampling_mode}, "
                                f"pool_remaining={len(pool)}, num_random_slices={self.num_random_slices})."
                            )
                        selected_slices = self._sample_from_pool(
                            file_path, pool, num_slices, num_spokes, num_samples, num_to_sample=len(pool)
                        )
                        selected_set = set(selected_slices)
                        reset_pool = [idx for idx in range(num_slices) if idx not in selected_set]
                        need = self.num_random_slices - len(selected_slices)
                        if need > 0:
                            extra = self._sample_from_pool(
                                file_path, reset_pool, num_slices, num_spokes, num_samples, num_to_sample=need
                            )
                            selected_slices = list(selected_slices) + list(extra)
                            extra_set = set(extra)
                            self._slice_remaining[file_path] = [idx for idx in reset_pool if idx not in extra_set]
                        else:
                            self._slice_remaining[file_path] = reset_pool
                elif self.slice_sampling_mode == "uniform" or self.slice_sampling_uniform_fraction >= 1.0:
                    selected_slices = random.sample(range(num_slices), self.num_random_slices)
                else:
                    selected_slices = self._sample_slices_weighted(
                        file_path, list(range(num_slices)), num_slices, num_spokes, num_samples
                    )

            for z in selected_slices:
                self.slice_index_map.append((file_path, z))

    def _warm_slice_score_cache(self) -> None:
        if self.slice_sampling_mode == "uniform" or self.slice_sampling_uniform_fraction >= 1.0:
            self._slice_score_cache_ready = True
            return
        if self._slice_score_cache_ready:
            return
        if not hasattr(self, "volume_map"):
            return
        if (
            self.slice_sampling_cache_rank_only is not None
            and self.slice_sampling_cache_rank is not None
            and self.slice_sampling_cache_rank != self.slice_sampling_cache_rank_only
        ):
            return
        iterator = self.volume_map
        if self.slice_sampling_cache_dir and self.slice_sampling_cache_workers and self.slice_sampling_cache_workers > 1:
            cache_jobs = []
            for file_path, num_slices, num_spokes, num_samples in iterator:
                cached = self._slice_score_cache.get(file_path)
                if cached is not None and len(cached.get("background", [])) == num_slices:
                    continue
                if self._load_slice_scores_from_disk(file_path, num_slices, num_spokes, num_samples) is not None:
                    continue
                cache_path = self._cache_path(file_path)
                if cache_path is None:
                    continue
                metadata = self._build_cache_metadata(
                    file_path, num_slices, num_spokes, num_samples
                )
                cache_jobs.append(
                    (
                        file_path,
                        self.dataset_key,
                        self.spokes_per_frame,
                        self.N_time,
                        cache_path,
                        metadata,
                    )
                )
            if cache_jobs:
                with ProcessPoolExecutor(max_workers=self.slice_sampling_cache_workers) as executor:
                    futures = [executor.submit(_compute_and_cache_slice_scores, job) for job in cache_jobs]
                    for _ in tqdm(as_completed(futures), total=len(futures), desc="Precomputing slice sampling scores", unit="vol"):
                        pass
        else:
            if len(iterator) >= 10:
                iterator = tqdm(iterator, desc="Precomputing slice sampling scores", unit="vol")
            for file_path, num_slices, num_spokes, num_samples in iterator:
                cached = self._slice_score_cache.get(file_path)
                if cached is not None and len(cached.get("background", [])) == num_slices:
                    continue
                self._get_slice_scores(file_path, num_slices, num_spokes, num_samples)
        self._slice_score_cache_ready = True

    def _cache_path(self, file_path: str) -> Optional[str]:
        if not self.slice_sampling_cache_dir:
            return None
        base = os.path.basename(file_path)
        digest = hashlib.md5(file_path.encode("utf-8")).hexdigest()[:8]
        return os.path.join(self.slice_sampling_cache_dir, f"{base}.{digest}.npz")

    def _build_cache_metadata(
        self,
        file_path: str,
        num_slices: int,
        num_spokes: int,
        num_samples: int,
    ) -> dict:
        return {
            "version": np.array(1, dtype=np.int64),
            "dataset_key": np.array(self.dataset_key),
            "file_mtime": np.array(int(os.path.getmtime(file_path)), dtype=np.int64),
            "num_slices": np.array(num_slices, dtype=np.int64),
            "num_spokes": np.array(num_spokes, dtype=np.int64),
            "num_samples": np.array(num_samples, dtype=np.int64),
        }

    def _load_slice_scores_from_disk(
        self,
        file_path: str,
        num_slices: int,
        num_spokes: int,
        num_samples: int,
    ) -> Optional[dict]:
        cache_path = self._cache_path(file_path)
        if cache_path is None or not os.path.exists(cache_path):
            return None
        try:
            with np.load(cache_path) as data:
                if int(data["version"]) != 1:
                    return None
                if str(data["dataset_key"]) != str(self.dataset_key):
                    return None
                if int(data["file_mtime"]) != int(os.path.getmtime(file_path)):
                    return None
                if int(data["num_slices"]) != int(num_slices):
                    return None
                if int(data["num_spokes"]) != int(num_spokes):
                    return None
                if int(data["num_samples"]) != int(num_samples):
                    return None
                return {
                    "background": data["background"],
                    "enhancement": data["enhancement"],
                }
        except Exception:
            return None

    def _save_slice_scores_to_disk(
        self,
        file_path: str,
        scores: dict,
        num_slices: int,
        num_spokes: int,
        num_samples: int,
    ) -> None:
        if (
            self.slice_sampling_cache_rank_only is not None
            and self.slice_sampling_cache_rank is not None
            and self.slice_sampling_cache_rank != self.slice_sampling_cache_rank_only
        ):
            return
        cache_path = self._cache_path(file_path)
        if cache_path is None:
            return
        metadata = self._build_cache_metadata(file_path, num_slices, num_spokes, num_samples)
        _write_slice_sampling_cache(cache_path, scores["background"], scores["enhancement"], metadata)

    def _get_slice_scores(
        self,
        file_path: str,
        num_slices: int,
        num_spokes: int,
        num_samples: int,
    ) -> dict:
        cached = self._slice_score_cache.get(file_path)
        if cached is not None and len(cached.get("background", [])) == num_slices:
            return cached
        cached = self._load_slice_scores_from_disk(file_path, num_slices, num_spokes, num_samples)
        if cached is not None:
            self._slice_score_cache[file_path] = cached
            return cached
        background, enhancement = _compute_slice_sampling_scores(
            file_path=file_path,
            dataset_key=self.dataset_key,
            spokes_per_frame=self.spokes_per_frame,
            n_time=self.N_time,
        )
        cached = {"background": background, "enhancement": enhancement}
        self._slice_score_cache[file_path] = cached
        self._save_slice_scores_to_disk(file_path, cached, num_slices, num_spokes, num_samples)
        return cached

    def _sample_from_pool(
        self,
        file_path: str,
        pool: list[int],
        num_slices: int,
        num_spokes: int,
        num_samples: int,
        num_to_sample: Optional[int] = None,
    ) -> list[int]:
        if num_to_sample is None:
            num_to_sample = self.num_random_slices
        num_to_sample = min(num_to_sample, len(pool))
        if self.slice_sampling_mode == "uniform" or self.slice_sampling_uniform_fraction >= 1.0:
            return random.sample(pool, num_to_sample)
        return self._sample_slices_weighted(
            file_path, pool, num_slices, num_spokes, num_samples, num_to_sample=num_to_sample
        )

    def _sample_slices_weighted(
        self,
        file_path: str,
        pool: list[int],
        num_slices: int,
        num_spokes: int,
        num_samples: int,
        num_to_sample: Optional[int] = None,
    ) -> list[int]:
        if num_to_sample is None:
            num_to_sample = self.num_random_slices
        num_to_sample = min(num_to_sample, len(pool))
        if num_to_sample <= 0:
            return []
        uniform_fraction = min(max(self.slice_sampling_uniform_fraction, 0.0), 1.0)
        n_uniform = int(round(num_to_sample * uniform_fraction))
        n_uniform = min(n_uniform, num_to_sample)
        n_weighted = num_to_sample - n_uniform

        selected = set()
        if n_uniform > 0 and pool:
            selected.update(random.sample(pool, min(n_uniform, len(pool))))

        if n_weighted <= 0:
            return list(selected)

        scores = self._get_slice_scores(file_path, num_slices, num_spokes, num_samples)
        if self.slice_sampling_mode == "background":
            weights = np.array(scores["background"], dtype=np.float64)
        elif self.slice_sampling_mode == "nonenhancing":
            weights = np.array(scores["enhancement"], dtype=np.float64)
        else:
            remaining = [idx for idx in pool if idx not in selected]
            return list(selected) + random.sample(remaining, min(n_weighted, len(remaining)))

        if self.slice_sampling_filter_quantile > 0:
            cutoff = np.quantile(weights, self.slice_sampling_filter_quantile)
            weights = np.where(weights >= cutoff, weights, 0.0)

        remaining = [idx for idx in pool if idx not in selected]
        if not remaining:
            return list(selected)

        remaining_weights = np.array([weights[idx] for idx in remaining], dtype=np.float64)
        n_weighted_target = min(n_weighted, len(remaining))
        if remaining_weights.sum() <= 0:
            selected.update(random.sample(remaining, n_weighted_target))
            return list(selected)

        nonzero_mask = remaining_weights > 0
        nonzero_count = int(np.count_nonzero(nonzero_mask))
        if nonzero_count < n_weighted_target:
            nonzero_pool = [remaining[i] for i in np.where(nonzero_mask)[0]]
            zero_pool = [remaining[i] for i in np.where(~nonzero_mask)[0]]
            selected.update([int(x) for x in nonzero_pool])
            need = n_weighted_target - nonzero_count
            if need > 0 and zero_pool:
                selected.update(random.sample(zero_pool, need))
            return list(selected)

        probs = remaining_weights / remaining_weights.sum()
        picks = np.random.choice(remaining, size=n_weighted_target, replace=False, p=probs)
        selected.update([int(x) for x in picks])
        return list(selected)

    def load_dynamic_img(self, patient_id, slice):
        # This method remains unchanged
        H = W = 320
        data = np.empty((2, self.N_time, H, W), dtype=np.float32)

        dirname = os.path.dirname(self.root_dir)
        
        for t in range(self.N_time):
            if self.cluster == "Randi":
                img_path = os.path.join(dirname, f'{patient_id}/slice_{slice:03d}_frame_{t:03d}.nii')
            elif self.cluster == "DSI":
                img_path = os.path.join(dirname, f'{patient_id}/slice_{slice:03d}_frame_{t:03d}.nii')
            else:
                raise ValueError("Undefined cluster name.")
            img = nib.load(img_path)
            img_data = img.get_fdata()

            if img_data.shape != (2, H, W):
                raise ValueError(f"{img_path} has shape {img_data.shape}; expected (2, {H}, {W})")

            data[:, t] = img_data.astype(np.float32)
            
        return torch.from_numpy(data)

    def load_csmaps(self, patient_id, slice):
        # This method remains unchanged
        ground_truth_dir = os.path.join(os.path.dirname(self.root_dir), 'cs_maps')
        csmap_path = os.path.join(ground_truth_dir, patient_id + '_cs_maps', f'cs_map_slice_{slice:03d}.npy')
        csmap = np.load(csmap_path)
        return csmap.squeeze()
    

    def __len__(self):
        return len(self.slice_index_map)

    def __getitem__(self, idx):
        # This method remains unchanged as it relies on self.slice_index_map
        file_path, current_slice_idx = self.slice_index_map[idx]

        current_slice_idx = int(current_slice_idx)
        patient_id = file_path.split('/')[-1].strip('.h5')

        csmap = self.load_csmaps(patient_id, current_slice_idx)

        with h5py.File(file_path, "r") as f:
            kspace_slice = torch.tensor(f[self.dataset_key][current_slice_idx])


        # if aug, select random spokes per frame
        if self.spf_aug:
            if _should_log_once(self, "_logged_spf_aug_msg"):
                print("setting random spokes per frame...")
            spokes_per_frame = random.choices(self.spokes_range, self.spf_weights, k=1)[0]
        else:
            spokes_per_frame = self.spokes_per_frame
            if _should_log_once(self, "_logged_fixed_spf_msg"):
                print(f"training with fixed spokes per frame ({spokes_per_frame})")

    
        # bin k-space according to desired spokes per frame (desired final shape (T, C, Sp, Sam))
        N_coils, N_spokes, N_samples = kspace_slice.shape
        N_time = N_spokes // spokes_per_frame
        N_spokes_prep = N_time * spokes_per_frame

        ksp_redu = kspace_slice[:, :N_spokes_prep, :] # (16, 288, 640)
        ksp_prep = np.swapaxes(ksp_redu, 0, 1) # (288, 16, 640)
        ksp_prep_shape = ksp_prep.shape
        ksp_prep = np.reshape(ksp_prep, [N_time, spokes_per_frame] + list(ksp_prep_shape[1:]))
        ksp_prep = rearrange(ksp_prep, 't sp c sam -> t c sp sam')

        real_part = ksp_prep.real
        imag_part = ksp_prep.imag
        kspace_final = torch.stack([real_part, imag_part], dim=0).float()

        if self.flip_kspace:
            kspace_final = torch.flip(kspace_final, dims=[-1])

            csmap_tensor = torch.from_numpy(csmap)
            csmap_tensor = torch.rot90(csmap_tensor, k=2, dims=[-2, -1])
            csmap = csmap_tensor.numpy()

        return kspace_final, csmap, N_samples, spokes_per_frame, N_time



class SliceDataset(Dataset):
    """
    A Dataset that:
      - Looks for all .h5/.hdf5 files under `root_dir`.
      - Each file is assumed to contain a dataset at `dataset_key`, with shape (... Z),
        where Z is the number of slices/partitions.
      - Can either use a fixed set of slices or randomly sample N slices per volume
        at the start of each epoch (optionally without replacement across epochs).
      - Returns each slice as a torch.Tensor.
    """

    def __init__(
        self,
        root_dir,
        patient_ids,
        dataset_key="kspace",
        file_pattern="*.h5",
        slice_idx: Optional[Union[int, range]] = 41,
        num_random_slices: Optional[int] = None,  # New parameter for random sampling
        slice_sampling_mode: str = "uniform",
        slice_sampling_uniform_fraction: float = 1.0,
        slice_sampling_filter_quantile: float = 0.2,
        slice_sampling_no_replacement: bool = False,
        slice_sampling_cache_dir: Optional[str] = None,
        slice_sampling_cache_workers: int = 0,
        slice_sampling_cache_rank: Optional[int] = None,
        slice_sampling_cache_rank_only: Optional[int] = None,
        N_time=8,
        N_coils=16,
        spf_aug=False,
        spokes_per_frame=None,
        weight_accelerations=False, 
        initial_spokes_range=[8, 16, 24, 36],
        interpolate_kspace=False,
        slices_to_interpolate=192,
        cluster="Randi"
    ):
        """
        Args:
            root_dir (str): Path to the folder containing all HDF5 k-space files.
            patient_ids (list): List of patient IDs to filter the files.
            dataset_key (str): The key/path inside each .h5 file to the k-space dataset.
            file_pattern (str): Glob pattern to match your HDF5 files.
            slice_idx (int, range, optional): A fixed slice index or range of indices to use.
                                              This is ignored if num_random_slices is set.
            num_random_slices (int, optional): If provided, the dataset will randomly sample
                                               this many slices from each volume at the beginning
                                               of each epoch.
        """
        super().__init__()

        self.root_dir = root_dir
        self.dataset_key = dataset_key
        self.slice_idx = slice_idx
        self.num_random_slices = num_random_slices
        self.N_time = N_time
        self.N_coils = N_coils
        self.spf_aug = spf_aug
        self.weight_acc = weight_accelerations
        self.cluster=cluster
        self.spokes_per_frame = spokes_per_frame
        self.slice_sampling_mode = str(slice_sampling_mode).lower()
        self.slice_sampling_uniform_fraction = float(slice_sampling_uniform_fraction)
        self.slice_sampling_filter_quantile = float(slice_sampling_filter_quantile)
        self.slice_sampling_no_replacement = bool(slice_sampling_no_replacement)
        self.slice_sampling_cache_dir = slice_sampling_cache_dir
        self.slice_sampling_cache_workers = int(slice_sampling_cache_workers)
        self.slice_sampling_cache_rank = slice_sampling_cache_rank
        self.slice_sampling_cache_rank_only = slice_sampling_cache_rank_only
        self._slice_score_cache = {}
        self._slice_remaining = {}
        self._slice_score_cache_ready = False

        # Find all matching HDF5 files under root_dir
        all_files = sorted(glob.glob(os.path.join(root_dir, file_pattern)))
        print("Number of files in root directory: ", len(all_files))

        if len(all_files) == 0:
            raise RuntimeError(
                f"No files found in {root_dir} matching pattern {file_pattern}"
            )

        # filter file list by patient ID substring
        filtered = []
        for fp in all_files:
            fname = os.path.basename(fp)
            if any(pid in fname for pid in patient_ids):
                filtered.append(fp)

        self.file_list = filtered

        if len(self.file_list) == 0:
            raise RuntimeError("No files matched the provided patient_ids filter.")

        # Logic for random slice sampling
        if self.num_random_slices is not None:
            print(f"Initializing in random slice sampling mode with N={self.num_random_slices} slices per volume.")
            self.volume_map = []
            for fp in self.file_list:
                with h5py.File(fp, "r") as f:
                    if self.dataset_key not in f:
                        raise KeyError(f"Dataset key '{self.dataset_key}' not found in file {fp}")
                    ds_shape = f[self.dataset_key].shape
                    num_slices, _, num_spokes, num_samples = ds_shape
                    self.volume_map.append((fp, num_slices, num_spokes, num_samples))
                    if self.slice_sampling_no_replacement:
                        self._slice_remaining[fp] = list(range(num_slices))
            
            # Perform the initial random sampling for the first epoch
            self.resample_slices()
        
        # Original logic for fixed slices, executed only if not in random mode
        else:
            print(f"Initializing in fixed slice mode with slice_idx={self.slice_idx}.")
            self.slice_index_map = []
            for fp in self.file_list:
                with h5py.File(fp, "r") as f:
                    if self.dataset_key not in f:
                        raise KeyError(f"Dataset key '{self.dataset_key}' not found in file {fp}")
                    ds = f[self.dataset_key]
                    num_slices = ds.shape[0]

                slices_to_add = []
                if isinstance(self.slice_idx, int):
                    if self.slice_idx < num_slices:
                        slices_to_add = [self.slice_idx]
                    else:
                        print(f"Warning: slice_idx {self.slice_idx} is out of bounds for {fp} "
                              f"(size {num_slices}). Skipping this file for this slice.")
                elif isinstance(self.slice_idx, range):
                    slices_to_add = [s for s in self.slice_idx if s < num_slices]
                    if len(slices_to_add) < len(self.slice_idx):
                        print(f"Warning: Some requested slices were out of bounds for {fp}. "
                              f"Using only the valid slice indices from the provided range.")
                else:
                    raise TypeError(f"slice_idx must be an int, range, or None, but got {type(self.slice_idx)}")

                for z in slices_to_add:
                    self.slice_index_map.append((fp, z))

        print(f"Dataset initialized with {len(self.slice_index_map)} total slice examples.")

        # NOTE: removed ultra-high accelerations until curriculum learning is implemented
        # self.spokes_range = [2, 4, 8, 16, 24, 36]
        # self.spokes_range = [8, 16, 24, 36]
        self.spokes_range = initial_spokes_range
        self.update_spokes_weights()
    
    def update_spokes_weights(self):

        if self.weight_acc:
            self.spf_weights = [1.0 / spf for spf in self.spokes_range]
        else:
            self.spf_weights = [1.0 for spf in self.spokes_range]


    def resample_slices(self):
        """
        Resamples N unique slices from each volume. This should be called at the
        beginning of each training epoch to ensure the model sees different data.
        """
        if self.num_random_slices is None:
            # If not in random sampling mode, do nothing.
            return
        self._warm_slice_score_cache()

        self.slice_index_map = []
        for file_path, num_slices, num_spokes, num_samples in self.volume_map:
            if num_slices <= 0:
                continue
            if num_slices <= self.num_random_slices:
                # If the volume has fewer than N slices, take all of them.
                print(f"Warning: Volume {os.path.basename(file_path)} has only {num_slices} slices, "
                      f"which is less than the requested {self.num_random_slices}. Using all available slices.")
                selected_slices = list(range(num_slices))
                if self.slice_sampling_no_replacement:
                    self._slice_remaining[file_path] = []
            else:
                if self.slice_sampling_no_replacement:
                    pool = self._slice_remaining.get(file_path)
                    if not pool:
                        pool = list(range(num_slices))
                    if len(pool) >= self.num_random_slices:
                        selected_slices = self._sample_from_pool(
                            file_path, pool, num_slices, num_spokes, num_samples, num_to_sample=self.num_random_slices
                        )
                        self._slice_remaining[file_path] = [idx for idx in pool if idx not in selected_slices]
                    else:
                        # Finish the current cycle first, then reset the pool.
                        if self.slice_sampling_mode != "uniform" and self.slice_sampling_uniform_fraction < 1.0:
                            print(
                                "[SliceSampling] Reset no-replacement pool for "
                                f"{os.path.basename(file_path)} (mode={self.slice_sampling_mode}, "
                                f"pool_remaining={len(pool)}, num_random_slices={self.num_random_slices})."
                            )
                        selected_slices = self._sample_from_pool(
                            file_path, pool, num_slices, num_spokes, num_samples, num_to_sample=len(pool)
                        )
                        selected_set = set(selected_slices)
                        reset_pool = [idx for idx in range(num_slices) if idx not in selected_set]
                        need = self.num_random_slices - len(selected_slices)
                        if need > 0:
                            extra = self._sample_from_pool(
                                file_path, reset_pool, num_slices, num_spokes, num_samples, num_to_sample=need
                            )
                            selected_slices = list(selected_slices) + list(extra)
                            extra_set = set(extra)
                            self._slice_remaining[file_path] = [idx for idx in reset_pool if idx not in extra_set]
                        else:
                            self._slice_remaining[file_path] = reset_pool
                elif self.slice_sampling_mode == "uniform" or self.slice_sampling_uniform_fraction >= 1.0:
                    selected_slices = random.sample(range(num_slices), self.num_random_slices)
                else:
                    selected_slices = self._sample_slices_weighted(
                        file_path, list(range(num_slices)), num_slices, num_spokes, num_samples
                    )

            for z in selected_slices:
                self.slice_index_map.append((file_path, z))

    def _warm_slice_score_cache(self) -> None:
        if self.slice_sampling_mode == "uniform" or self.slice_sampling_uniform_fraction >= 1.0:
            self._slice_score_cache_ready = True
            return
        if self._slice_score_cache_ready:
            return
        if not hasattr(self, "volume_map"):
            return
        if (
            self.slice_sampling_cache_rank_only is not None
            and self.slice_sampling_cache_rank is not None
            and self.slice_sampling_cache_rank != self.slice_sampling_cache_rank_only
        ):
            return
        iterator = self.volume_map
        if self.slice_sampling_cache_dir and self.slice_sampling_cache_workers and self.slice_sampling_cache_workers > 1:
            cache_jobs = []
            for file_path, num_slices, num_spokes, num_samples in iterator:
                cached = self._slice_score_cache.get(file_path)
                if cached is not None and len(cached.get("background", [])) == num_slices:
                    continue
                if self._load_slice_scores_from_disk(file_path, num_slices, num_spokes, num_samples) is not None:
                    continue
                cache_path = self._cache_path(file_path)
                if cache_path is None:
                    continue
                metadata = self._build_cache_metadata(
                    file_path, num_slices, num_spokes, num_samples
                )
                cache_jobs.append(
                    (
                        file_path,
                        self.dataset_key,
                        self.spokes_per_frame,
                        self.N_time,
                        cache_path,
                        metadata,
                    )
                )
            if cache_jobs:
                with ProcessPoolExecutor(max_workers=self.slice_sampling_cache_workers) as executor:
                    futures = [executor.submit(_compute_and_cache_slice_scores, job) for job in cache_jobs]
                    for _ in tqdm(as_completed(futures), total=len(futures), desc="Precomputing slice sampling scores", unit="vol"):
                        pass
        else:
            if len(iterator) >= 10:
                iterator = tqdm(iterator, desc="Precomputing slice sampling scores", unit="vol")
            for file_path, num_slices, num_spokes, num_samples in iterator:
                cached = self._slice_score_cache.get(file_path)
                if cached is not None and len(cached.get("background", [])) == num_slices:
                    continue
                self._get_slice_scores(file_path, num_slices, num_spokes, num_samples)
        self._slice_score_cache_ready = True

    def _cache_path(self, file_path: str) -> Optional[str]:
        if not self.slice_sampling_cache_dir:
            return None
        base = os.path.basename(file_path)
        digest = hashlib.md5(file_path.encode("utf-8")).hexdigest()[:8]
        return os.path.join(self.slice_sampling_cache_dir, f"{base}.{digest}.npz")

    def _build_cache_metadata(
        self,
        file_path: str,
        num_slices: int,
        num_spokes: int,
        num_samples: int,
    ) -> dict:
        return {
            "version": np.array(1, dtype=np.int64),
            "dataset_key": np.array(self.dataset_key),
            "file_mtime": np.array(int(os.path.getmtime(file_path)), dtype=np.int64),
            "num_slices": np.array(num_slices, dtype=np.int64),
            "num_spokes": np.array(num_spokes, dtype=np.int64),
            "num_samples": np.array(num_samples, dtype=np.int64),
        }

    def _load_slice_scores_from_disk(
        self,
        file_path: str,
        num_slices: int,
        num_spokes: int,
        num_samples: int,
    ) -> Optional[dict]:
        cache_path = self._cache_path(file_path)
        if cache_path is None or not os.path.exists(cache_path):
            return None
        try:
            with np.load(cache_path) as data:
                if int(data["version"]) != 1:
                    return None
                if str(data["dataset_key"]) != str(self.dataset_key):
                    return None
                if int(data["file_mtime"]) != int(os.path.getmtime(file_path)):
                    return None
                if int(data["num_slices"]) != int(num_slices):
                    return None
                if int(data["num_spokes"]) != int(num_spokes):
                    return None
                if int(data["num_samples"]) != int(num_samples):
                    return None
                return {
                    "background": data["background"],
                    "enhancement": data["enhancement"],
                }
        except Exception:
            return None

    def _save_slice_scores_to_disk(
        self,
        file_path: str,
        scores: dict,
        num_slices: int,
        num_spokes: int,
        num_samples: int,
    ) -> None:
        if (
            self.slice_sampling_cache_rank_only is not None
            and self.slice_sampling_cache_rank is not None
            and self.slice_sampling_cache_rank != self.slice_sampling_cache_rank_only
        ):
            return
        cache_path = self._cache_path(file_path)
        if cache_path is None:
            return
        metadata = self._build_cache_metadata(file_path, num_slices, num_spokes, num_samples)
        _write_slice_sampling_cache(cache_path, scores["background"], scores["enhancement"], metadata)

    def _get_slice_scores(
        self,
        file_path: str,
        num_slices: int,
        num_spokes: int,
        num_samples: int,
    ) -> dict:
        cached = self._slice_score_cache.get(file_path)
        if cached is not None and len(cached.get("background", [])) == num_slices:
            return cached
        cached = self._load_slice_scores_from_disk(file_path, num_slices, num_spokes, num_samples)
        if cached is not None:
            self._slice_score_cache[file_path] = cached
            return cached
        background, enhancement = _compute_slice_sampling_scores(
            file_path=file_path,
            dataset_key=self.dataset_key,
            spokes_per_frame=self.spokes_per_frame,
            n_time=self.N_time,
        )
        cached = {"background": background, "enhancement": enhancement}
        self._slice_score_cache[file_path] = cached
        self._save_slice_scores_to_disk(file_path, cached, num_slices, num_spokes, num_samples)
        return cached

    def _sample_from_pool(
        self,
        file_path: str,
        pool: list[int],
        num_slices: int,
        num_spokes: int,
        num_samples: int,
        num_to_sample: Optional[int] = None,
    ) -> list[int]:
        if num_to_sample is None:
            num_to_sample = self.num_random_slices
        num_to_sample = min(num_to_sample, len(pool))
        if self.slice_sampling_mode == "uniform" or self.slice_sampling_uniform_fraction >= 1.0:
            return random.sample(pool, num_to_sample)
        return self._sample_slices_weighted(
            file_path, pool, num_slices, num_spokes, num_samples, num_to_sample=num_to_sample
        )

    def _sample_slices_weighted(
        self,
        file_path: str,
        pool: list[int],
        num_slices: int,
        num_spokes: int,
        num_samples: int,
        num_to_sample: Optional[int] = None,
    ) -> list[int]:
        if num_to_sample is None:
            num_to_sample = self.num_random_slices
        num_to_sample = min(num_to_sample, len(pool))
        if num_to_sample <= 0:
            return []
        uniform_fraction = min(max(self.slice_sampling_uniform_fraction, 0.0), 1.0)
        n_uniform = int(round(num_to_sample * uniform_fraction))
        n_uniform = min(n_uniform, num_to_sample)
        n_weighted = num_to_sample - n_uniform

        selected = set()
        if n_uniform > 0 and pool:
            selected.update(random.sample(pool, min(n_uniform, len(pool))))

        if n_weighted <= 0:
            return list(selected)

        scores = self._get_slice_scores(file_path, num_slices, num_spokes, num_samples)
        if self.slice_sampling_mode == "background":
            weights = np.array(scores["background"], dtype=np.float64)
        elif self.slice_sampling_mode == "nonenhancing":
            weights = np.array(scores["enhancement"], dtype=np.float64)
        else:
            remaining = [idx for idx in pool if idx not in selected]
            return list(selected) + random.sample(remaining, min(n_weighted, len(remaining)))

        if self.slice_sampling_filter_quantile > 0:
            cutoff = np.quantile(weights, self.slice_sampling_filter_quantile)
            weights = np.where(weights >= cutoff, weights, 0.0)

        remaining = [idx for idx in pool if idx not in selected]
        if not remaining:
            return list(selected)

        remaining_weights = np.array([weights[idx] for idx in remaining], dtype=np.float64)
        n_weighted_target = min(n_weighted, len(remaining))
        if remaining_weights.sum() <= 0:
            selected.update(random.sample(remaining, n_weighted_target))
            return list(selected)

        nonzero_mask = remaining_weights > 0
        nonzero_count = int(np.count_nonzero(nonzero_mask))
        if nonzero_count < n_weighted_target:
            nonzero_pool = [remaining[i] for i in np.where(nonzero_mask)[0]]
            zero_pool = [remaining[i] for i in np.where(~nonzero_mask)[0]]
            selected.update([int(x) for x in nonzero_pool])
            need = n_weighted_target - nonzero_count
            if need > 0 and zero_pool:
                selected.update(random.sample(zero_pool, need))
            return list(selected)

        probs = remaining_weights / remaining_weights.sum()
        picks = np.random.choice(remaining, size=n_weighted_target, replace=False, p=probs)
        selected.update([int(x) for x in picks])
        return list(selected)

    def load_dynamic_img(self, patient_id, slice):
        # This method remains unchanged
        H = W = 320
        data = np.empty((2, self.N_time, H, W), dtype=np.float32)
        
        for t in range(self.N_time):
            if self.cluster == "Randi":
                img_path = f'/ess/scratch/scratch1/rachelgordon/dce-{self.N_time}tf/{patient_id}/slice_{slice:03d}_frame_{t:03d}.nii'
            elif self.cluster == "DSI":
                img_path = f'/net/scratch2/rachelgordon/dce-{self.N_time}tf/{patient_id}/slice_{slice:03d}_frame_{t:03d}.nii'
            else:
                raise ValueError("Undefined cluster name.")
            img = nib.load(img_path)
            img_data = img.get_fdata()

            if img_data.shape != (2, H, W):
                raise ValueError(f"{img_path} has shape {img_data.shape}; expected (2, {H}, {W})")

            data[:, t] = img_data.astype(np.float32)
            
        return torch.from_numpy(data)

    def load_csmaps(self, patient_id, slice):
        # This method remains unchanged
        ground_truth_dir = os.path.join(os.path.dirname(self.root_dir), 'cs_maps')
        csmap_path = os.path.join(ground_truth_dir, patient_id + '_cs_maps', f'cs_map_slice_{slice:03d}.npy')
        csmap = np.load(csmap_path)
        return csmap.squeeze()
    


    def __len__(self):
        return len(self.slice_index_map)

    def __getitem__(self, idx):
        # This method remains unchanged as it relies on self.slice_index_map
        file_path, current_slice_idx = self.slice_index_map[idx]
        current_slice_idx = int(current_slice_idx)
        patient_id = file_path.split('/')[-1].strip('.h5')

        # grasp_img = self.load_dynamic_img(patient_id, current_slice_idx)


        csmap = self.load_csmaps(patient_id, current_slice_idx)



        with h5py.File(file_path, "r") as f:

            kspace_slice = torch.tensor(f[self.dataset_key][current_slice_idx])

        if _should_log_once(self, "_logged_kspace_shape_msg"):
            print("loaded kspace shape: ", kspace_slice.shape) # torch.Size([8, 16, 36, 640])




        if self.spf_aug or self.spokes_per_frame:
            total_spokes = kspace_slice.shape[0] * kspace_slice.shape[2]
            N_samples = kspace_slice.shape[-1]
            kspace = rearrange(kspace_slice, 't c sp sam -> t sp c sam')
            kspace_flat = kspace.contiguous().view(total_spokes, self.N_coils, N_samples)
            # kspace_flat = kspace.contiguous().reshape(total_spokes, self.N_coils, N_samples)

            if self.spf_aug:
                if _should_log_once(self, "_logged_spf_aug_msg"):
                    print("setting random spokes per frame...")
                spokes_per_frame = random.choices(self.spokes_range, self.spf_weights, k=1)[0]
            else:
                spokes_per_frame = self.spokes_per_frame
                if _should_log_once(self, "_logged_fixed_spf_msg"):
                    print(f"training with fixed spokes per frame ({spokes_per_frame})")

            N_time = total_spokes // spokes_per_frame
            kspace_binned = kspace_flat.view(N_time, spokes_per_frame, self.N_coils, N_samples)
            kspace_slice = rearrange(kspace_binned, 't sp c sam -> t c sp sam')
        else:
            N_time = self.N_time
            N_samples = kspace_slice.shape[-1]
            spokes_per_frame = kspace_slice.shape[-2]

        real_part = kspace_slice.real
        imag_part = kspace_slice.imag
        kspace_final = torch.stack([real_part, imag_part], dim=0).float()

        # kspace_final = torch.flip(kspace_final, dims=[-1])

        # csmap = torch.from_numpy(csmap)
        # csmap_tensor = torch.rot90(csmap, k=2, dims=[-2, -1])
        # csmap = csmap_tensor.numpy()

        return kspace_final, csmap, N_samples, spokes_per_frame, N_time
    

class SimulatedDataset(Dataset):
    """
    Dataset for loading the simulated data generated by your script.
    It loads the simulated k-space, coil sensitivity maps, and the
    ground truth dynamic image (DRO).
    """
    def __init__(
        self,
        root_dir,
        raw_kspace_path,
        model_type,
        patient_ids,
        dataset_key,
        grasp_slice_idx=95,
        spokes_per_frame=36,
        num_frames=8,
        traj_method="trajGR",
        noise_level=0,
        dro_csmaps_source="original",
        espirit_csmaps_dir=None,
        dro_sim_source="original",
        skip_raw_eval_if_invalid_slice: bool = False,
    ):

        self.root_dir = root_dir
        self.raw_kspace_path = raw_kspace_path
        self.patient_ids = patient_ids
        self.model_type = model_type
        self.spokes_per_frame = spokes_per_frame
        self.num_frames = num_frames
        self.grasp_slice_idx = grasp_slice_idx
        self.dataset_key = dataset_key
        self.traj_method = traj_method
        self.skip_raw_eval_if_invalid_slice = bool(skip_raw_eval_if_invalid_slice)
        self.noise_level_value, self.noise_level_label = self._parse_noise_level(noise_level)
        if self.noise_level_value > 0 and self.traj_method != "get_traj":
            print(f"SimulatedDataset: noise_level={self.noise_level_label} ignored because traj_method={self.traj_method}.")
        self.dro_csmaps_source = dro_csmaps_source
        self.espirit_csmaps_dir = espirit_csmaps_dir
        if self.dro_csmaps_source not in ("original", "espirit"):
            raise ValueError(
                f"Unsupported dro_csmaps_source '{self.dro_csmaps_source}'. "
                "Expected 'original' or 'espirit'."
            )
        self.dro_sim_source = dro_sim_source
        if self.dro_sim_source not in ("original", "espirit"):
            raise ValueError(
                f"Unsupported dro_sim_source '{self.dro_sim_source}'. "
                "Expected 'original' or 'espirit'."
            )
        if self.dro_sim_source == "espirit":
            if self.traj_method != "get_traj":
                print(
                    "SimulatedDataset: dro_sim_source=espirit expects traj_method='get_traj' "
                    f"for _correct_traj filenames (got {self.traj_method})."
                )
            if abs(self.noise_level_value - 0.05) > 1e-8:
                print(
                    "SimulatedDataset: dro_sim_source=espirit expects noise_level=0.05 "
                    f"for _correct_traj_n0.05 filenames (got {self.noise_level_label})."
                )
        self.slice_map = load_slice_map(SLICE_MAP_PATH)
        self._update_sample_paths()


        self.TISSUE_NAMES = [
            'glandular', 'benign', 'malignant', 'muscle',
            'skin', 'liver', 'heart', 'vascular'
        ]

    def _update_sample_paths(self):
        self.dro_dir = os.path.join(self.root_dir, f'dro_{self.num_frames}frames')

        # Find all sample directories, e.g., 'sample_001_sub1', 'sample_002_sub2', etc.
        self.sample_paths = sorted(glob.glob(os.path.join(self.dro_dir, 'sample_*')))
        if not self.sample_paths:
            raise FileNotFoundError(f"No sample directories found in {self.dro_dir}. "
                                    "Please check the path to your simulated dataset.")
        
        # filter file list by patient ID substring
        filtered = []
        for fp in self.sample_paths:
            fname = os.path.basename(fp)
            # Check if any patient_id appears in the filename
            if any(pid in fname for pid in self.patient_ids):
                filtered.append(fp)

        self.sample_paths = filtered

        print(f"Found {len(self.sample_paths)} simulated samples in {self.dro_dir} for {self.num_frames} frames.")

    @staticmethod
    def _parse_noise_level(noise_level):
        if noise_level is None:
            return 0.0, None
        if isinstance(noise_level, str):
            label = noise_level.strip()
            if label == "":
                return 0.0, None
            try:
                value = float(label)
            except ValueError as exc:
                raise ValueError(f"noise_level must be numeric; got {noise_level!r}") from exc
            if value <= 0:
                return 0.0, None
            return value, label
        value = float(noise_level)
        if value <= 0:
            return 0.0, None
        return value, str(noise_level)

    def _traj_suffix(self):
        if self.traj_method != "get_traj":
            suffix = ".npy"
        elif self.noise_level_value > 0:
            suffix = f"_correct_traj_n{self.noise_level_label}.npy"
        else:
            suffix = "_correct_traj.npy"
        if self.dro_sim_source == "espirit" and suffix.endswith(".npy"):
            suffix = suffix[:-4] + "_espirit.npy"
        return suffix

    def _load_dro_csmaps(self, sample_dir):
        if self.dro_csmaps_source == "original":
            csmap_path = os.path.join(sample_dir, "csmaps.npy")
        else:
            esp_root = self.espirit_csmaps_dir or os.path.join(self.root_dir, "csmaps_espirit")
            sample_name = os.path.basename(sample_dir)
            csmap_path = os.path.join(esp_root, f"csmaps_{sample_name}.npy")
        return np.load(csmap_path)

    def get_fastMRI_id(self, sample_dir):

        sample_file = os.path.basename(sample_dir)
        id_map = pd.read_csv('data/DROSubID_vs_fastMRIbreastID.csv')

        dro_id = int(sample_file.split("_")[1])
        dro_row = id_map[id_map["DRO"] == dro_id]
        fastmri_id = int(dro_row["fastMRIbreast"].iloc[0])

        return fastmri_id
  
            
    def __len__(self):
        return len(self.sample_paths)
    


    def __getitem__(self, idx):
        sample_dir = self.sample_paths[idx]


        # Load the data from .npy files
        csmaps = self._load_dro_csmaps(sample_dir)
        dro = np.load(os.path.join(sample_dir, 'dro_ground_truth.npz'))
        traj_suffix = self._traj_suffix()
        grasp_path = os.path.join(
            sample_dir,
            f'grasp_spf{self.spokes_per_frame}_frames{self.num_frames}{traj_suffix}'
        )

        grasp_recon = np.load(grasp_path)

        # GRASP Recon: (H, W, T) -> (2, T, H, W) [real/imag, time, h, w]
        grasp_recon_torch = torch.from_numpy(grasp_recon).permute(2, 0, 1) # T, H, W
        grasp_recon_torch = torch.stack([grasp_recon_torch.real, grasp_recon_torch.imag], dim=0)

        grasp_recon_torch = torch.flip(grasp_recon_torch, dims=[-3])
        grasp_recon_torch = torch.rot90(grasp_recon_torch, k=3, dims=[-3,-1])

        kspace_path = os.path.join(
            sample_dir,
            f'simulated_kspace_spf{self.spokes_per_frame}_frames{self.num_frames}{traj_suffix}'
        )

        if os.path.exists(kspace_path):
            kspace_complex = np.load(kspace_path, allow_pickle=True)
            kspace_torch = torch.from_numpy(kspace_complex)
        else:
            kspace_torch = kspace_path

        # CSMaps: (H, W, C) -> (1, C, H, W) [batch, coils, h, w]
        if self.dro_csmaps_source == "original":
            csmaps_torch = torch.from_numpy(csmaps).permute(2, 0, 1).unsqueeze(0)
        else:
            csmaps_torch = torch.from_numpy(csmaps).unsqueeze(0).to(torch.complex64)


        # load raw k-space and GRASP recon
        fastmri_id = self.get_fastMRI_id(sample_dir)
        patient_id = f"fastMRI_breast_{fastmri_id:03d}_2"
        slice_idx = self.slice_map.get(patient_id, None)
        raw_slice_valid = slice_idx is not None and slice_idx >= 0
        if (not raw_slice_valid) and self.skip_raw_eval_if_invalid_slice:
            if _should_log_once(self, "_logged_skip_raw_eval"):
                print(
                    "[SimulatedDataset] Skipping raw eval for samples with invalid "
                    f"largest_tumor_slices.csv entries (e.g., {patient_id})."
                )
            raw_grasp_recon = torch.full_like(grasp_recon_torch, float("nan"))
            raw_kspace_slice = torch.full_like(kspace_torch, float("nan"))
            raw_csmaps_torch = torch.full_like(csmaps_torch, float("nan")).numpy()
        else:
            if slice_idx is None or slice_idx < 0:
                slice_idx = self.grasp_slice_idx

            raw_grasp_path = os.path.join(
                os.path.dirname(self.raw_kspace_path),
                f'{patient_id}/grasp_recon_{self.spokes_per_frame}spf_{self.num_frames}frames_slice{slice_idx}.npy'
            )
            raw_kspace_path = os.path.join(self.raw_kspace_path, f'{patient_id}.h5')
            raw_csmap_path = os.path.join(
                os.path.dirname(self.raw_kspace_path),
                f'cs_maps/{patient_id}_cs_maps/cs_map_slice_{slice_idx:03d}.npy'
            )
            
            raw_csmaps = np.load(raw_csmap_path)
            # raw_csmaps = rearrange(raw_csmaps, 'c b h w -> b c h w')

            raw_grasp_recon = np.load(raw_grasp_path).squeeze()


            # GRASP Recon: (H, W, T) -> (2, T, H, W) [real/imag, time, h, w]
            raw_grasp_recon = torch.from_numpy(raw_grasp_recon).permute(2, 0, 1) # T, H, W
            raw_grasp_recon = torch.stack([raw_grasp_recon.real, raw_grasp_recon.imag], dim=0)

            raw_grasp_recon = torch.flip(raw_grasp_recon, dims=[-3])
            raw_grasp_recon = torch.rot90(raw_grasp_recon, k=1, dims=[-3,-1])


            with h5py.File(raw_kspace_path, "r") as f:
                raw_kspace_slice = torch.tensor(f[self.dataset_key][slice_idx])

            # time-bin k-space
            N_spokes_prep = self.num_frames * self.spokes_per_frame

            ksp_redu = raw_kspace_slice[:, :N_spokes_prep, :] # (16, 288, 640)
            ksp_prep = np.swapaxes(ksp_redu, 0, 1) # (288, 16, 640)
            ksp_prep_shape = ksp_prep.shape
            ksp_prep = np.reshape(ksp_prep, [self.num_frames, self.spokes_per_frame] + list(ksp_prep_shape[1:]))

            ksp_prep = torch.flip(ksp_prep, dims=[-1])

            raw_kspace_slice = rearrange(ksp_prep, 't sp c sam -> c (sp sam) t').to(kspace_torch.dtype)


        ground_truth_complex = dro['ground_truth_images']

        parMap = dro['parMap']
        aif = dro['aif']
        S0 = dro['S0']
        T10 = dro['T10']
        # mask = dro['mask']

        # ==========================================================
        # --- RECONSTRUCT THE MASK DICTIONARY ---
        # ==========================================================
        mask_dictionary_rebuilt = {}
        for tissue_name in self.TISSUE_NAMES:
            # Check if the key for this tissue (e.g., 'malignant') exists in the file
            if tissue_name in dro:
                # Load the boolean array and add it to the dictionary
                mask_dictionary_rebuilt[tissue_name] = dro[tissue_name]
        
        # 'mask' is now the dictionary of boolean arrays, just like your functions expect
        mask = mask_dictionary_rebuilt


        # --- Convert to PyTorch Tensors ---
        # Ground truth: (H, W, T) -> (2, T, H, W) [real/imag, time, h, w]
        ground_truth_torch = torch.from_numpy(ground_truth_complex).permute(2, 0, 1) # T, H, W
        ground_truth_torch = torch.stack([ground_truth_torch.real, ground_truth_torch.imag], dim=0)

        if raw_slice_valid or (not self.skip_raw_eval_if_invalid_slice):
            raw_csmaps_torch = torch.from_numpy(raw_csmaps)#.permute(2, 0, 1).unsqueeze(0)
            raw_csmaps_torch = rearrange(raw_csmaps_torch, 'c b h w -> b c h w').to(csmaps_torch.dtype)

            raw_csmaps_torch = torch.rot90(raw_csmaps_torch, k=2, dims=[-2, -1])
            raw_csmaps_torch = raw_csmaps_torch.numpy()

        return kspace_torch, csmaps_torch, ground_truth_torch, grasp_recon_torch, mask, grasp_path, raw_kspace_slice, raw_grasp_recon, raw_csmaps_torch #, parMap, aif, S0, T10, mask
    
    


class SimulatedSPFDataset(Dataset):
    """
    Dataset for loading the simulated data generated by your script.
    It loads the simulated k-space, coil sensitivity maps, and the
    ground truth dynamic image (DRO).
    """
    def __init__(
        self,
        root_dir,
        raw_kspace_path,
        model_type,
        patient_ids,
        dataset_key,
        grasp_slice_idx=95,
        skip_raw_eval_if_invalid_slice: bool = False,
    ):
        self.model_type = model_type
        self.root_dir = root_dir
        self.patient_ids = patient_ids

        self.raw_kspace_path = raw_kspace_path
        self.grasp_slice_idx = grasp_slice_idx
        self.dataset_key = dataset_key
        self.slice_map = load_slice_map(SLICE_MAP_PATH)
        self.skip_raw_eval_if_invalid_slice = bool(skip_raw_eval_if_invalid_slice)

        # set default parameters to be changed before each call
        self.spokes_per_frame = 16
        self.num_frames = 18

        # Initialize sample paths based on default parameters
        self._update_sample_paths()
        

        self.TISSUE_NAMES = [
            'glandular', 'benign', 'malignant', 'muscle',
            'skin', 'liver', 'heart', 'vascular'
        ]

    def _update_sample_paths(self):
        self.dro_dir = os.path.join(self.root_dir, f'dro_{self.num_frames}frames')

        # Find all sample directories, e.g., 'sample_001_sub1', 'sample_002_sub2', etc.
        self.sample_paths = sorted(glob.glob(os.path.join(self.dro_dir, 'sample_*')))
        if not self.sample_paths:
            raise FileNotFoundError(f"No sample directories found in {self.dro_dir}. "
                                    "Please check the path to your simulated dataset.")
        
        # filter file list by patient ID substring
        filtered = []
        for fp in self.sample_paths:
            fname = os.path.basename(fp)
            # Check if any patient_id appears in the filename
            if any(pid in fname for pid in self.patient_ids):
                filtered.append(fp)

        self.sample_paths = filtered

        print(f"Found {len(self.sample_paths)} simulated samples in {self.dro_dir} for {self.num_frames} frames.")


    def get_fastMRI_id(self, sample_dir):

        sample_file = os.path.basename(sample_dir)
        id_map = pd.read_csv('data/DROSubID_vs_fastMRIbreastID.csv')

        dro_id = int(sample_file.split("_")[1])
        dro_row = id_map[id_map["DRO"] == dro_id]
        fastmri_id = int(dro_row["fastMRIbreast"].iloc[0])

        return fastmri_id
    
    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        sample_dir = self.sample_paths[idx]

        print(f"  Testing {self.spokes_per_frame} spokes/frame with {self.num_frames} frames.")


        # Load the data from .npy files
        csmaps = np.load(os.path.join(sample_dir, 'csmaps.npy'))
        dro = np.load(os.path.join(sample_dir, 'dro_ground_truth.npz'))

        grasp_path = os.path.join(sample_dir, f'grasp_spf{self.spokes_per_frame}_frames{self.num_frames}.npy')
        
        if os.path.exists(grasp_path):
            grasp_recon = np.load(grasp_path)

            # GRASP Recon: (H, W, T) -> (2, T, H, W) [real/imag, time, h, w]
            grasp_recon_torch = torch.from_numpy(grasp_recon).permute(2, 0, 1) # T, H, W
            grasp_recon_torch = torch.stack([grasp_recon_torch.real, grasp_recon_torch.imag], dim=0)

            grasp_recon_torch = torch.flip(grasp_recon_torch, dims=[-3])
            grasp_recon_torch = torch.rot90(grasp_recon_torch, k=3, dims=[-3,-1])

        else:
            grasp_recon_torch = 0


        ground_truth_complex = dro['ground_truth_images']

        # SELECT TIME WINDOW
        # ground_truth_complex = ground_truth_complex[..., self.window]

        smap_torch = rearrange(torch.tensor(csmaps), 'h w c -> c h w').unsqueeze(0)
        simImg_torch = torch.tensor(ground_truth_complex).to(torch.cfloat)



        parMap = dro['parMap']
        aif = dro['aif']
        S0 = dro['S0']
        T10 = dro['T10']
        # mask = dro['mask']

        # ==========================================================
        # --- RECONSTRUCT THE MASK DICTIONARY ---
        # ==========================================================
        mask_dictionary_rebuilt = {}
        for tissue_name in self.TISSUE_NAMES:
            # Check if the key for this tissue (e.g., 'malignant') exists in the file
            if tissue_name in dro:
                # Load the boolean array and add it to the dictionary
                mask_dictionary_rebuilt[tissue_name] = dro[tissue_name]
        
        # 'mask' is now the dictionary of boolean arrays, just like your functions expect
        mask = mask_dictionary_rebuilt


        # --- Convert to PyTorch Tensors ---
        # Ground truth: (H, W, T) -> (2, T, H, W) [real/imag, time, h, w]
        ground_truth_torch = torch.from_numpy(ground_truth_complex).permute(2, 0, 1) # T, H, W
        ground_truth_torch = torch.stack([ground_truth_torch.real, ground_truth_torch.imag], dim=0)


        # load raw k-space and GRASP recon
        fastmri_id = self.get_fastMRI_id(sample_dir)
        patient_id = f"fastMRI_breast_{fastmri_id:03d}_2"
        slice_idx = self.slice_map.get(patient_id, None)
        raw_slice_valid = slice_idx is not None and slice_idx >= 0
        if (not raw_slice_valid) and self.skip_raw_eval_if_invalid_slice:
            if _should_log_once(self, "_logged_skip_raw_eval"):
                print(
                    "[SimulatedSPFDataset] Skipping raw eval for samples with invalid "
                    f"largest_tumor_slices.csv entries (e.g., {patient_id})."
                )
            raw_grasp_recon = torch.full_like(grasp_recon_torch, float("nan"))
            raw_kspace_slice = torch.full_like(simImg_torch, float("nan"))
            raw_csmaps_torch = torch.full_like(smap_torch, float("nan")).numpy()
        else:
            if slice_idx is None or slice_idx < 0:
                slice_idx = self.grasp_slice_idx

            raw_grasp_path = os.path.join(
                os.path.dirname(self.raw_kspace_path),
                f'{patient_id}/grasp_recon_{self.spokes_per_frame}spf_{self.num_frames}frames_slice{slice_idx}.npy'
            )
            raw_kspace_path = os.path.join(self.raw_kspace_path, f'{patient_id}.h5')
            raw_csmap_path = os.path.join(
                os.path.dirname(self.raw_kspace_path),
                f'cs_maps/{patient_id}_cs_maps/cs_map_slice_{slice_idx:03d}.npy'
            )
            
            raw_csmaps = np.load(raw_csmap_path)
            # raw_csmaps = rearrange(raw_csmaps, 'c b h w -> b c h w')

            raw_grasp_recon = np.load(raw_grasp_path).squeeze()


            # GRASP Recon: (H, W, T) -> (2, T, H, W) [real/imag, time, h, w]
            raw_grasp_recon = torch.from_numpy(raw_grasp_recon).permute(2, 0, 1) # T, H, W
            raw_grasp_recon = torch.stack([raw_grasp_recon.real, raw_grasp_recon.imag], dim=0)

            raw_grasp_recon = torch.flip(raw_grasp_recon, dims=[-3])
            raw_grasp_recon = torch.rot90(raw_grasp_recon, k=3, dims=[-3,-1])


            with h5py.File(raw_kspace_path, "r") as f:
                raw_kspace_slice = torch.tensor(f[self.dataset_key][slice_idx])

            # time-bin k-space
            N_spokes_prep = self.num_frames * self.spokes_per_frame

            ksp_redu = raw_kspace_slice[:, :N_spokes_prep, :] # (16, 288, 640)
            ksp_prep = np.swapaxes(ksp_redu, 0, 1) # (288, 16, 640)
            ksp_prep_shape = ksp_prep.shape
            ksp_prep = np.reshape(ksp_prep, [self.num_frames, self.spokes_per_frame] + list(ksp_prep_shape[1:]))
            
            ksp_prep = torch.flip(ksp_prep, dims=[-1])

            raw_kspace_slice = rearrange(ksp_prep, 't sp c sam -> c (sp sam) t').to(smap_torch.dtype)

            raw_csmaps_torch = torch.from_numpy(raw_csmaps)#.permute(2, 0, 1).unsqueeze(0)
            raw_csmaps_torch = rearrange(raw_csmaps_torch, 'c b h w -> b c h w').to(smap_torch.dtype)

            raw_csmaps_torch = torch.rot90(raw_csmaps_torch, k=2, dims=[-2, -1])
            raw_csmaps_torch = raw_csmaps_torch.numpy()


        return smap_torch, simImg_torch, grasp_recon_torch, mask, grasp_path, raw_kspace_slice, raw_csmaps_torch, raw_grasp_recon #, parMap, aif, S0, T10, mask
