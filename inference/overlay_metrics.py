#!/usr/bin/env python3
"""Overlay evaluation metrics from multiple experiments. Run: python3 -m inference.overlay_metrics --help"""
import argparse
import glob
import json
import math
import os
import re
import subprocess
from typing import Dict, List, Optional, Tuple

import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


EVAL_PANEL_LAYOUT = [
    ("eval_ssims", "DRO SSIM", "SSIM", "avg_grasp_ssim", (0, 0)),
    ("eval_psnrs", "DRO PSNR", "PSNR", "avg_grasp_psnr", (0, 1)),
    ("eval_mses", "DRO Image MSE", "MSE", "avg_grasp_mse", (0, 2)),
    ("eval_raw_dc_maes", "Non-DRO k-space MAE", "MAE", "avg_grasp_raw_dc_mae", (0, 3)),
    ("eval_lpipses", "DRO LPIPS", "LPIPS", "avg_grasp_lpips", (1, 0)),
    ("eval_curve_corrs", "DRO Curve Correlation", "Pearson Correlation Coefficient", "avg_grasp_curve_corr", (1, 1)),
    ("eval_dl_dc_mae_bestfits", "DRO k-space MAE (best-fit gain)", "MAE", "avg_grasp_dc_mae_bestfit", (1, 2)),
    ("eval_raw_ssdu_nmses", "Non-DRO SSDU NMSE", "NMSE", "avg_grasp_raw_ssdu_nmse", (1, 3)),
]

TEMPORAL_METRICS = [
    ("eval_curve_maes", "Curve MAE", "MAE"),
    ("eval_ttae_secs", "Time to Arrival Error", "Seconds"),
    ("eval_iauc10_errs", "IAUC10 Error", "Error"),
    ("eval_peak_errs", "Peak Enhancement Error", "Error"),
]

TEMPORAL_GRASP_BASELINE_KEYS = {
    "eval_curve_maes": "avg_grasp_curve_mae",
    "eval_ttae_secs": "avg_grasp_ttae_sec",
    "eval_iauc10_errs": "avg_grasp_iauc10_err",
    "eval_peak_errs": "avg_grasp_peak_err",
}

TRAIN_METRICS = [
    ("train_mc_losses", "Train MC Loss", "Loss"),
    ("train_ei_losses", "Train EI Loss", "Loss"),
    ("weighted_train_ei_losses", "Train Weighted EI Loss", "Loss"),
    ("train_rebin_losses", "Train Rebin Loss", "Loss"),
    ("val_mc_losses", "Val MC Loss", "Loss"),
    ("val_ei_losses", "Val EI Loss", "Loss"),
    ("val_adj_losses", "Val Adj Loss", "Loss"),
    ("lr_history", "Learning Rate", "LR"),
]

GRASP_LINE_STYLE = {
    "color": "tab:red",
    "linestyle": "--",
    "alpha": 0.8,
    "linewidth": 1.8,
}


def _as_list(value) -> List[float]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _as_scalar(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().cpu().flatten()[0].item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_finite(value: Optional[float]) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def _has_finite(values: List[float]) -> bool:
    for v in values:
        try:
            if math.isfinite(float(v)):
                return True
        except (TypeError, ValueError):
            continue
    return False


def _load_checkpoint(path: str) -> Dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        # Torch>=2.6 defaults to weights_only=True and may reject numpy scalar
        # objects stored in older checkpoints. Fall back to full load.
        return torch.load(path, map_location="cpu", weights_only=False)


def _resolve_checkpoint_from_run_dir(run_dir: str, prefer_best: bool) -> str:
    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    patterns = ["*_best_model.pth", "*_model.pth"] if prefer_best else ["*_model.pth", "*_best_model.pth"]
    candidates: List[str] = []
    for pat in patterns:
        candidates = sorted(glob.glob(os.path.join(run_dir, pat)))
        if candidates:
            break
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint matching '*_model.pth' or '*_best_model.pth' found in {run_dir}"
        )

    base = os.path.basename(os.path.normpath(run_dir))
    exact_priority = [
        os.path.join(run_dir, f"{base}_model.pth"),
        os.path.join(run_dir, f"{base}_best_model.pth"),
    ]
    for exact in exact_priority:
        if exact in candidates:
            return exact
    return max(candidates, key=os.path.getmtime)


def _load_data_cfg_from_run_dir(run_dir: str) -> Dict:
    config_path = os.path.join(os.path.abspath(os.path.expanduser(run_dir)), "config.yaml")
    if not os.path.isfile(config_path):
        return {}
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to read run-dir configs.") from exc
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        return {}
    data_cfg = cfg.get("data", {})
    if not isinstance(data_cfg, dict):
        return {}
    return data_cfg


def _infer_eval_frequency_from_run_dir(run_dir: str) -> int:
    data_cfg = _load_data_cfg_from_run_dir(run_dir)
    raw = data_cfg.get("eval_frequency", 1)
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


def _infer_spf_from_run_dir(run_dir: str) -> Optional[int]:
    data_cfg = _load_data_cfg_from_run_dir(run_dir)
    for raw in (
        data_cfg.get("eval_spokes"),
        data_cfg.get("train_spokes_per_frame"),
        data_cfg.get("fpg"),
    ):
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def _infer_spf_from_ckpt_path(ckpt_path: str) -> Optional[int]:
    run_dir = os.path.dirname(os.path.abspath(os.path.expanduser(ckpt_path)))
    return _infer_spf_from_run_dir(run_dir)


def _infer_slurm_job_id_from_run_dir(run_dir: str) -> Optional[str]:
    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    submitit_dir = os.path.join(run_dir, "submitit_logs")
    if not os.path.isdir(submitit_dir):
        return None

    job_ids: List[int] = []
    for name in os.listdir(submitit_dir):
        match = re.match(r"^(\d+)(?:[_\.].*)?$", name)
        if not match:
            continue
        try:
            job_ids.append(int(match.group(1)))
        except ValueError:
            continue
    if not job_ids:
        return None
    return str(max(job_ids))


def _infer_slurm_job_id_from_ckpt_path(ckpt_path: str) -> Optional[str]:
    run_dir = os.path.dirname(os.path.abspath(os.path.expanduser(ckpt_path)))
    return _infer_slurm_job_id_from_run_dir(run_dir)


def _infer_run_state_status_from_run_dir(run_dir: str) -> Optional[str]:
    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    run_state_path = os.path.join(run_dir, "run_state.json")
    if not os.path.isfile(run_state_path):
        return None
    try:
        with open(run_state_path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None

    attempt_history = data.get("attempt_history")
    if isinstance(attempt_history, list) and len(attempt_history) > 0:
        last = attempt_history[-1]
        if isinstance(last, dict):
            status = last.get("status")
            if isinstance(status, str) and status.strip():
                return status.strip().lower()

    status = data.get("status")
    if isinstance(status, str) and status.strip():
        return status.strip().lower()
    return None


def _infer_run_state_status_from_ckpt_path(ckpt_path: str) -> Optional[str]:
    run_dir = os.path.dirname(os.path.abspath(os.path.expanduser(ckpt_path)))
    return _infer_run_state_status_from_run_dir(run_dir)


def _query_slurm_states(job_ids: List[str]) -> Optional[Dict[str, str]]:
    requested = sorted({str(j).strip() for j in job_ids if str(j).strip() and str(j).strip() != "-"})
    if not requested:
        return {}

    try:
        proc = subprocess.run(
            ["squeue", "-h", "-o", "%A %T", "-j", ",".join(requested)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None

    if proc.returncode != 0:
        return None

    state_rank = {
        "RUNNING": 3,
        "COMPLETING": 2,
        "PENDING": 1,
    }

    def _rank(state: str) -> int:
        return state_rank.get(state.upper(), 0)

    states: Dict[str, str] = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        raw_job_id, raw_state = parts
        base_job_id = raw_job_id.split("_", 1)[0]
        if base_job_id not in requested:
            continue
        state = raw_state.strip().upper()
        prev = states.get(base_job_id)
        if prev is None or _rank(state) > _rank(prev):
            states[base_job_id] = state
    return states


def _label_with_spf(label: str, spf: Optional[int]) -> str:
    if spf is None:
        return label
    return f"{label} (SPF={spf})"


def _default_label(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _expand_input_paths(paths: List[str], kind: str) -> List[str]:
    """
    Expand user-provided path arguments, supporting quoted glob patterns.
    Preserves input order and removes duplicates.
    """
    expanded: List[str] = []
    seen = set()
    for raw in paths:
        path = os.path.abspath(os.path.expanduser(raw))
        matches: List[str]
        if glob.has_magic(path):
            matches = sorted(glob.glob(path))
            if not matches:
                print(f"Warning: no {kind} paths matched pattern '{raw}'.")
                continue
        else:
            matches = [path]
        for match in matches:
            norm = os.path.abspath(os.path.expanduser(match))
            if norm in seen:
                continue
            seen.add(norm)
            expanded.append(norm)
    return expanded


def _build_eval_epoch_axis(n: int, eval_frequency: int) -> List[int]:
    return [(i + 1) * eval_frequency for i in range(n)]


def _build_train_epoch_axis(n: int) -> List[int]:
    return list(range(1, n + 1))


def _resolve_eval_axis(
    curves: Dict[str, List[float]],
    eval_frequency: int,
    n: int,
) -> List[int]:
    temporal_epochs = _as_list(curves.get("eval_temporal_epochs", []))
    if len(temporal_epochs) >= n and n > 0:
        return [int(round(float(v))) for v in temporal_epochs[:n]]
    return _build_eval_epoch_axis(n, eval_frequency)


def _resolve_train_axis(
    curves: Dict[str, List[float]],
    key: str,
    eval_frequency: int,
    n: int,
) -> List[int]:
    if key == "lr_history":
        lr_epochs = _as_list(curves.get("lr_epochs", []))
        if len(lr_epochs) >= n and n > 0:
            return [int(round(float(v))) for v in lr_epochs[:n]]
        return _build_train_epoch_axis(n)
    if key.startswith("val_"):
        return _resolve_eval_axis(curves, eval_frequency, n)
    return _build_train_epoch_axis(n)


def _resolve_temporal_axes(
    curves: Dict[str, List[float]],
    eval_frequency: int,
) -> Optional[Tuple[List[int], Dict[str, List[float]]]]:
    temporal_epochs = _as_list(curves.get("eval_temporal_epochs", []))
    metric_lists = [curves.get(k, []) for k, _, _ in TEMPORAL_METRICS]

    if temporal_epochs and len(temporal_epochs) == len(metric_lists[0]):
        epochs = [int(round(float(v))) for v in temporal_epochs]
    else:
        epochs = _build_eval_epoch_axis(len(metric_lists[0]), eval_frequency)

    min_len = min([len(epochs)] + [len(v) for v in metric_lists])
    if min_len == 0:
        return None

    epochs = epochs[:min_len]
    trimmed = {
        key: values[:min_len]
        for (key, _, _), values in zip(TEMPORAL_METRICS, metric_lists)
    }
    return epochs, trimmed


def _resolve_metric_series(
    exp: Dict,
    key: str,
    kind: str,
) -> Tuple[List[int], List[float]]:
    curves = exp["curves"]
    eval_frequency = exp["eval_frequency"]

    if kind == "temporal":
        resolved = _resolve_temporal_axes(curves, eval_frequency)
        if resolved is None:
            return [], []
        epochs, temporal_curves = resolved
        values = temporal_curves.get(key, [])
        n = min(len(epochs), len(values))
        return epochs[:n], values[:n]

    if kind == "eval":
        values = curves.get(key, [])
        n = len(values)
        return _resolve_eval_axis(curves, eval_frequency, n), values

    if kind == "train":
        values = curves.get(key, [])
        n = len(values)
        return _resolve_train_axis(curves, key, eval_frequency, n), values

    raise ValueError(f"Unsupported metric kind: {kind}")


def _resolve_grasp_baselines(
    experiments: List[Dict],
    baseline_keys: List[str],
) -> Dict[str, Optional[float]]:
    # Some checkpoints (e.g. before first eval) persist placeholder GRASP
    # baselines as 0/nan. Ignore those bundles so we do not anchor overlays to 0.
    def _has_non_placeholder_grasp_bundle(exp: Dict) -> bool:
        baselines = exp.get("baselines", {}) or {}
        indicator_keys = (
            "avg_grasp_ssim",
            "avg_grasp_psnr",
            "avg_grasp_mse",
            "avg_grasp_curve_corr",
            "avg_grasp_lpips",
            "avg_grasp_dc_mae",
        )
        finite_vals = []
        for key in indicator_keys:
            val = baselines.get(key)
            if _is_finite(val):
                finite_vals.append(float(val))
        if not finite_vals:
            return False
        return any(v > 0.0 for v in finite_vals)

    usable_experiments = [exp for exp in experiments if _has_non_placeholder_grasp_bundle(exp)]
    if usable_experiments and len(usable_experiments) < len(experiments):
        skipped = [exp["label"] for exp in experiments if exp not in usable_experiments]
        print(
            "Info: ignoring placeholder GRASP baselines (all 0/nan) from: "
            + ", ".join(skipped)
        )

    baselines: Dict[str, Optional[float]] = {}
    for key in baseline_keys:
        values = []
        labels = []
        for exp in usable_experiments:
            val = exp["baselines"].get(key)
            if _is_finite(val):
                values.append(float(val))
                labels.append(exp["label"])
        if not values:
            baselines[key] = None
            continue
        min_v = min(values)
        max_v = max(values)
        if not math.isclose(min_v, max_v, rel_tol=1e-4, abs_tol=1e-6):
            print(
                "Warning: GRASP baseline "
                f"{key} varies across checkpoints (min {min_v:.6g}, max {max_v:.6g}). "
                f"Using {values[0]:.6g} from {labels[0]}."
            )
        baselines[key] = values[0]
    return baselines


def _add_legend_entry(
    legend_entries: Dict[str, object],
    label: str,
    handle,
) -> None:
    if label in legend_entries:
        return
    legend_entries[label] = handle


def _experiment_has_any_data(exp: Dict) -> bool:
    metric_keys = [key for key, *_ in EVAL_PANEL_LAYOUT]
    metric_keys.extend([key for key, _, _ in TEMPORAL_METRICS])
    metric_keys.extend([key for key, _, _ in TRAIN_METRICS])
    for key in metric_keys:
        if _has_finite(exp["curves"].get(key, [])):
            return True
    return False


def _plot_six_panel(
    out_path: str,
    experiments: List[Dict],
    show_baseline: bool,
    grasp_baselines: Dict[str, Optional[float]],
    title: str,
    dpi: int,
) -> bool:
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 4, figsize=(32, 10))
    fig.suptitle(title, fontsize=20)

    if len(experiments) <= 10:
        palette = sns.color_palette("tab10", n_colors=len(experiments))
    else:
        palette = sns.color_palette("husl", n_colors=len(experiments))

    plotted_panels = 0
    legend_entries: Dict[str, object] = {}
    for key, panel_title, ylabel, baseline_key, (row, col) in EVAL_PANEL_LAYOUT:
        ax = axes[row][col]
        plotted = False
        for color, exp in zip(palette, experiments):
            x, values = _resolve_metric_series(exp, key=key, kind="eval")
            if not _has_finite(values):
                continue
            n = min(len(x), len(values))
            if n == 0:
                continue
            x = x[:n]
            values = values[:n]
            if len(values) == 1:
                line = ax.plot(
                    x,
                    values,
                    label=exp["label"],
                    color=color,
                    linestyle="None",
                    marker="o",
                    markersize=5,
                )[0]
            else:
                line = ax.plot(x, values, label=exp["label"], color=color, linewidth=1.8)[0]
            _add_legend_entry(legend_entries, exp["label"], line)
            plotted = True

        if not plotted:
            ax.axis("off")
            continue
        if show_baseline:
            baseline = grasp_baselines.get(baseline_key)
            if _is_finite(baseline):
                baseline_line = ax.axhline(y=baseline, label="GRASP", **GRASP_LINE_STYLE)
                _add_legend_entry(legend_entries, "GRASP", baseline_line)
        plotted_panels += 1
        ax.set_title(panel_title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
    if plotted_panels == 0:
        plt.close(fig)
        return False

    labels = list(legend_entries.keys())
    handles = [legend_entries[label] for label in labels]
    if handles:
        legend_ax = fig.add_axes([0.69, 0.08, 0.30, 0.84])
        legend_ax.axis("off")
        legend_ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=9)

    fig.subplots_adjust(left=0.05, right=0.67, bottom=0.10, top=0.88, wspace=0.30, hspace=0.35)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_temporal(
    out_path: str,
    experiments: List[Dict],
    show_baseline: bool,
    grasp_baselines: Dict[str, Optional[float]],
    title: str,
    dpi: int,
) -> bool:
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    fig.suptitle(title, fontsize=20)

    if len(experiments) <= 10:
        palette = sns.color_palette("tab10", n_colors=len(experiments))
    else:
        palette = sns.color_palette("husl", n_colors=len(experiments))

    plotted_panels = 0
    legend_entries: Dict[str, object] = {}
    for idx, (key, panel_title, ylabel) in enumerate(TEMPORAL_METRICS):
        ax = axes[idx // 2][idx % 2]
        plotted = False
        for color, exp in zip(palette, experiments):
            epochs, values = _resolve_metric_series(exp, key=key, kind="temporal")
            if not _has_finite(values):
                continue
            n = min(len(epochs), len(values))
            if n == 0:
                continue
            epochs = epochs[:n]
            values = values[:n]
            if len(values) == 1:
                line = ax.plot(
                    epochs,
                    values,
                    label=exp["label"],
                    color=color,
                    linestyle="None",
                    marker="o",
                    markersize=5,
                )[0]
            else:
                line = ax.plot(epochs, values, label=exp["label"], color=color, linewidth=1.8)[0]
            _add_legend_entry(legend_entries, exp["label"], line)
            plotted = True

        if not plotted:
            ax.axis("off")
            continue
        if show_baseline:
            baseline_key = TEMPORAL_GRASP_BASELINE_KEYS.get(key)
            if baseline_key:
                baseline = grasp_baselines.get(baseline_key)
                if _is_finite(baseline):
                    baseline_line = ax.axhline(y=baseline, label="GRASP", **GRASP_LINE_STYLE)
                    _add_legend_entry(legend_entries, "GRASP", baseline_line)
        plotted_panels += 1
        ax.set_title(panel_title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
    if plotted_panels == 0:
        plt.close(fig)
        return False

    labels = list(legend_entries.keys())
    handles = [legend_entries[label] for label in labels]
    if handles:
        legend_ax = fig.add_axes([0.69, 0.08, 0.30, 0.84])
        legend_ax.axis("off")
        legend_ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=9)

    fig.subplots_adjust(left=0.07, right=0.67, bottom=0.10, top=0.88, wspace=0.28, hspace=0.32)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_training(
    out_path: str,
    experiments: List[Dict],
    title: str,
    dpi: int,
) -> bool:
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 4, figsize=(32, 10))
    fig.suptitle(title, fontsize=20)

    if len(experiments) <= 10:
        palette = sns.color_palette("tab10", n_colors=len(experiments))
    else:
        palette = sns.color_palette("husl", n_colors=len(experiments))

    plotted_panels = 0
    legend_entries: Dict[str, object] = {}
    for idx, (key, panel_title, ylabel) in enumerate(TRAIN_METRICS):
        row, col = divmod(idx, 4)
        ax = axes[row][col]
        plotted = False
        for color, exp in zip(palette, experiments):
            epochs, values = _resolve_metric_series(exp, key=key, kind="train")
            if not _has_finite(values):
                continue
            n = min(len(epochs), len(values))
            if n == 0:
                continue
            epochs = epochs[:n]
            values = values[:n]
            if len(values) == 1:
                line = ax.plot(
                    epochs,
                    values,
                    label=exp["label"],
                    color=color,
                    linestyle="None",
                    marker="o",
                    markersize=5,
                )[0]
            else:
                line = ax.plot(epochs, values, label=exp["label"], color=color, linewidth=1.8)[0]
            _add_legend_entry(legend_entries, exp["label"], line)
            plotted = True

        if not plotted:
            ax.axis("off")
            continue
        plotted_panels += 1
        ax.set_title(panel_title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)

    if plotted_panels == 0:
        plt.close(fig)
        return False

    labels = list(legend_entries.keys())
    handles = [legend_entries[label] for label in labels]
    if handles:
        legend_ax = fig.add_axes([0.69, 0.08, 0.30, 0.84])
        legend_ax.axis("off")
        legend_ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=9)

    fig.subplots_adjust(left=0.05, right=0.67, bottom=0.10, top=0.88, wspace=0.30, hspace=0.35)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def _metric_table_specs() -> List[Tuple[str, str, str]]:
    specs: List[Tuple[str, str, str]] = []
    for key, title, _, _, _ in EVAL_PANEL_LAYOUT:
        specs.append(("eval", key, title))
    for key, title, _ in TEMPORAL_METRICS:
        specs.append(("temporal", key, title))
    for key, title, _ in TRAIN_METRICS:
        specs.append(("train", key, title))
    return specs


def _series_for_table(
    exp: Dict,
    kind: str,
    key: str,
) -> Tuple[List[int], List[float]]:
    if kind == "eval":
        return _resolve_metric_series(exp, key=key, kind="eval")
    if kind == "temporal":
        return _resolve_metric_series(exp, key=key, kind="temporal")
    if kind == "train":
        return _resolve_metric_series(exp, key=key, kind="train")
    raise ValueError(f"Unsupported table metric kind: {kind}")


def _metric_sort_goal(metric_key: str) -> str:
    maximize_keys = {
        "eval_ssims",
        "eval_psnrs",
        "eval_curve_corrs",
        "lr_history",
    }
    return "max" if metric_key in maximize_keys else "min"


def _format_table_cell(value: float) -> str:
    return f"{float(value):.3g}"


def _write_padded_table(
    f,
    columns: List[str],
    rows: List[List[str]],
) -> None:
    widths = [len(col) for col in columns]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _render_row(items: List[str]) -> str:
        return "  ".join(item.ljust(widths[idx]) for idx, item in enumerate(items)).rstrip()

    f.write(_render_row(columns) + "\n")
    f.write(_render_row(["-" * w for w in widths]) + "\n")
    for row in rows:
        f.write(_render_row(row) + "\n")


def _write_metric_tables(
    out_path: str,
    experiments: List[Dict],
) -> bool:
    specs = _metric_table_specs()
    tables_written = 0
    with open(out_path, "w") as f:
        f.write("# Metric Tables\n")
        f.write("# Rows: experiments\n")
        f.write("# Columns: epochs\n\n")

        for kind, key, title in specs:
            epoch_union = set()
            per_exp = []
            for exp in experiments:
                epochs, values = _series_for_table(exp, kind=kind, key=key)
                n = min(len(epochs), len(values))
                epochs = [int(round(float(e))) for e in epochs[:n]]
                values = values[:n]
                mapping: Dict[int, Optional[float]] = {}
                for e, v in zip(epochs, values):
                    try:
                        vf = float(v)
                    except (TypeError, ValueError):
                        vf = float("nan")
                    mapping[e] = vf
                    epoch_union.add(e)
                finite_values = [float(v) for v in mapping.values() if math.isfinite(float(v))]
                per_exp.append(
                    {
                        "label": exp["label"],
                        "job_id": str(exp.get("job_id", "-") or "-"),
                        "running_now": str(exp.get("running_now", "-") or "-"),
                        "mapping": mapping,
                        "has_data": len(finite_values) > 0,
                        "best_min": min(finite_values) if finite_values else float("inf"),
                        "best_max": max(finite_values) if finite_values else float("-inf"),
                    }
                )

            if not epoch_union:
                continue

            tables_written += 1
            sorted_epochs = sorted(epoch_union)
            sort_goal = _metric_sort_goal(key)

            def _sort_key(item: Dict) -> Tuple[int, float, str]:
                label_key = str(item["label"]).lower()
                if not item["has_data"]:
                    return (1, 0.0, label_key)
                if sort_goal == "max":
                    return (0, -float(item["best_max"]), label_key)
                return (0, float(item["best_min"]), label_key)

            per_exp = sorted(per_exp, key=_sort_key)

            f.write(f"=== {title} ({key}) | sorted by best-{sort_goal} over all epochs ===\n")
            columns = ["experiment", "slurm_job_id", "running_now"] + [f"ep{ep}" for ep in sorted_epochs]
            rows: List[List[str]] = []
            for item in per_exp:
                safe_label = str(item["label"]).replace("\t", " ").replace("\n", " ")
                safe_job_id = str(item.get("job_id", "-")).replace("\t", " ").replace("\n", " ")
                safe_running_now = str(item.get("running_now", "-")).replace("\t", " ").replace("\n", " ")
                row_vals: List[str] = []
                mapping = item["mapping"]
                for ep in sorted_epochs:
                    v = mapping.get(ep, float("nan"))
                    if isinstance(v, (int, float)) and math.isfinite(float(v)):
                        row_vals.append(_format_table_cell(float(v)))
                    else:
                        row_vals.append("-")
                rows.append([safe_label, safe_job_id, safe_running_now] + row_vals)
            _write_padded_table(f, columns=columns, rows=rows)
            f.write("\n")
    return tables_written > 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Overlay eval curves from multiple checkpoints or run directories."
    )
    ap.add_argument(
        "--run-dir",
        nargs="+",
        default=None,
        help="Run directories (2+) or glob patterns; checkpoints will be auto-resolved.",
    )
    ap.add_argument(
        "--ckpt",
        nargs="+",
        default=None,
        help="Checkpoint paths (2+) or glob patterns.",
    )
    ap.add_argument(
        "--label",
        nargs="*",
        default=None,
        help="Optional labels matching --ckpt order.",
    )
    ap.add_argument("--ckpt-a", default=None, help="Path to first checkpoint (legacy).")
    ap.add_argument("--ckpt-b", default=None, help="Path to second checkpoint (legacy).")
    ap.add_argument("--label-a", default=None, help="Label for first checkpoint (legacy).")
    ap.add_argument("--label-b", default=None, help="Label for second checkpoint (legacy).")
    ap.add_argument(
        "--eval-frequency",
        type=int,
        default=None,
        help="Eval frequency in epochs. For --run-dir, overrides config-derived frequency.",
    )
    ap.add_argument(
        "--prefer-best",
        action="store_true",
        help="When using --run-dir, prefer '*_best_model.pth' over '*_model.pth'.",
    )
    ap.add_argument("--out-dir", default=".", help="Output directory for plots.")
    ap.add_argument("--dpi", type=int, default=150, help="Output DPI.")
    ap.add_argument("--no-six-panel", action="store_true", help="Skip six-panel metrics plot.")
    ap.add_argument("--no-temporal", action="store_true", help="Skip temporal metrics plot.")
    ap.add_argument("--no-training", action="store_true", help="Skip training metrics plot.")
    ap.add_argument(
        "--show-grasp-baseline",
        action="store_true",
        help="Overlay dashed GRASP baselines from avg_grasp_* (if present).",
    )
    ap.add_argument(
        "--no-grasp-baseline",
        action="store_true",
        help="Disable GRASP baseline lines (enabled by default).",
    )
    ap.add_argument(
        "--tables-file",
        default="overlay_metric_tables.txt",
        help="Output text filename for per-metric tables (written under --out-dir).",
    )
    ap.add_argument(
        "--no-tables",
        action="store_true",
        help="Skip writing per-metric tables.",
    )
    ap.add_argument(
        "--filter-spf",
        nargs="+",
        type=int,
        default=None,
        help="Only include experiments whose inferred SPF is in this list (e.g. --filter-spf 2 36).",
    )
    args = ap.parse_args()

    def _build_experiment(
        ckpt: Dict,
        label: str,
        eval_frequency: int,
        spf: Optional[int],
        job_id: Optional[str],
        run_state_status: Optional[str],
    ) -> Dict:
        curves = {key: _as_list(ckpt.get(key, [])) for key, _, _, _, _ in EVAL_PANEL_LAYOUT}
        for key, _, _ in TEMPORAL_METRICS:
            curves[key] = _as_list(ckpt.get(key, []))
        for key, _, _ in TRAIN_METRICS:
            curves[key] = _as_list(ckpt.get(key, []))
        curves["eval_temporal_epochs"] = _as_list(ckpt.get("eval_temporal_epochs", []))
        curves["lr_epochs"] = _as_list(ckpt.get("lr_epochs", []))
        baselines = {
            key: _as_scalar(value)
            for key, value in ckpt.items()
            if key.startswith("avg_grasp_")
        }
        return {
            "label": _label_with_spf(label, spf),
            "curves": curves,
            "baselines": baselines,
            "eval_frequency": max(1, int(eval_frequency)),
            "spf": spf,
            "job_id": str(job_id) if job_id is not None else "-",
            "run_state_status": run_state_status if run_state_status is not None else "-",
        }

    if args.run_dir and args.ckpt:
        raise SystemExit("Use either --run-dir or --ckpt, not both.")

    experiments = []
    if args.run_dir:
        run_dirs = _expand_input_paths(args.run_dir, kind="run-dir")
        labels = list(args.label or [])
        if len(labels) < len(run_dirs):
            labels.extend([None] * (len(run_dirs) - len(labels)))
        labels = labels[: len(run_dirs)]
        for run_dir, label in zip(run_dirs, labels):
            try:
                ckpt_path = _resolve_checkpoint_from_run_dir(run_dir, prefer_best=args.prefer_best)
            except FileNotFoundError:
                print(f"Skipping {run_dir}: no checkpoint found.")
                continue
            eval_frequency = args.eval_frequency
            if eval_frequency is None:
                eval_frequency = _infer_eval_frequency_from_run_dir(run_dir)
            spf = _infer_spf_from_run_dir(run_dir)
            job_id = _infer_slurm_job_id_from_run_dir(run_dir)
            run_state_status = _infer_run_state_status_from_run_dir(run_dir)
            try:
                ckpt = _load_checkpoint(ckpt_path)
            except Exception as exc:
                print(f"Skipping {run_dir}: failed to load checkpoint '{ckpt_path}' ({exc}).")
                continue
            experiments.append(
                _build_experiment(
                    ckpt=ckpt,
                    label=label or os.path.basename(os.path.normpath(run_dir)),
                    eval_frequency=eval_frequency,
                    spf=spf,
                    job_id=job_id,
                    run_state_status=run_state_status,
                )
            )
    elif args.ckpt:
        ckpt_paths = _expand_input_paths(args.ckpt, kind="checkpoint")
        labels = list(args.label or [])
        if len(labels) < len(ckpt_paths):
            labels.extend([None] * (len(ckpt_paths) - len(labels)))
        labels = labels[: len(ckpt_paths)]
        eval_frequency = 1 if args.eval_frequency is None else args.eval_frequency
        for ckpt_path, label in zip(ckpt_paths, labels):
            ckpt = _load_checkpoint(ckpt_path)
            spf = _infer_spf_from_ckpt_path(ckpt_path)
            job_id = _infer_slurm_job_id_from_ckpt_path(ckpt_path)
            run_state_status = _infer_run_state_status_from_ckpt_path(ckpt_path)
            experiments.append(
                _build_experiment(
                    ckpt=ckpt,
                    label=label or _default_label(ckpt_path),
                    eval_frequency=eval_frequency,
                    spf=spf,
                    job_id=job_id,
                    run_state_status=run_state_status,
                )
            )
    else:
        if not args.ckpt_a or not args.ckpt_b:
            raise SystemExit("Provide --run-dir (2+) or --ckpt paths (2+) or legacy --ckpt-a/--ckpt-b.")
        ckpt_paths = [args.ckpt_a, args.ckpt_b]
        labels = [args.label_a, args.label_b]
        eval_frequency = 1 if args.eval_frequency is None else args.eval_frequency
        for ckpt_path, label in zip(ckpt_paths, labels):
            ckpt = _load_checkpoint(ckpt_path)
            spf = _infer_spf_from_ckpt_path(ckpt_path)
            job_id = _infer_slurm_job_id_from_ckpt_path(ckpt_path)
            run_state_status = _infer_run_state_status_from_ckpt_path(ckpt_path)
            experiments.append(
                _build_experiment(
                    ckpt=ckpt,
                    label=label or _default_label(ckpt_path),
                    eval_frequency=eval_frequency,
                    spf=spf,
                    job_id=job_id,
                    run_state_status=run_state_status,
                )
            )

    if args.filter_spf:
        allowed_spf = {int(v) for v in args.filter_spf}
        original_count = len(experiments)
        experiments = [exp for exp in experiments if exp.get("spf") in allowed_spf]
        removed = original_count - len(experiments)
        if removed > 0:
            print(
                f"Filtered out {removed} experiments not matching --filter-spf "
                f"{sorted(allowed_spf)}."
            )

    filtered = []
    for exp in experiments:
        if _experiment_has_any_data(exp):
            filtered.append(exp)
        else:
            print(f"Skipping {exp['label']}: no eval/temporal/training data points found.")
    experiments = filtered

    slurm_states = _query_slurm_states([str(exp.get("job_id", "-")) for exp in experiments])
    if slurm_states is None:
        print("Warning: unable to query squeue; using run_state.json for running_now.")
        slurm_states = {}
        slurm_available = False
    else:
        slurm_available = True

    for exp in experiments:
        job_id = str(exp.get("job_id", "-"))
        running_now = "-"
        if job_id and job_id != "-":
            if slurm_available:
                state = slurm_states.get(job_id, "")
                running_now = "yes" if state in {"RUNNING", "COMPLETING"} else "no"
            else:
                run_state_status = str(exp.get("run_state_status", "-")).lower()
                if run_state_status != "-" and run_state_status != "":
                    running_now = "yes" if run_state_status == "running" else "no"
        exp["running_now"] = running_now

    if len(experiments) < 2:
        raise SystemExit("Need at least two experiments with data to compare.")

    baseline_keys = [baseline_key for _, _, _, baseline_key, _ in EVAL_PANEL_LAYOUT]
    baseline_keys.extend(list(TEMPORAL_GRASP_BASELINE_KEYS.values()))
    grasp_baselines = _resolve_grasp_baselines(experiments, baseline_keys)

    os.makedirs(args.out_dir, exist_ok=True)

    show_grasp_baseline = args.show_grasp_baseline or not args.no_grasp_baseline

    if not args.no_six_panel:
        six_path = os.path.join(args.out_dir, "eval_metrics_overlay.png")
        wrote = _plot_six_panel(
            six_path,
            experiments,
            show_baseline=show_grasp_baseline,
            grasp_baselines=grasp_baselines,
            title="evaluation metrics",
            dpi=args.dpi,
        )
        if wrote:
            print(f"Wrote {six_path}")
        else:
            print("Skipped eval metrics overlay: no data points for any eval metric.")

    if not args.no_temporal:
        temporal_path = os.path.join(args.out_dir, "eval_temporal_metrics_overlay.png")
        wrote = _plot_temporal(
            temporal_path,
            experiments,
            show_baseline=show_grasp_baseline,
            grasp_baselines=grasp_baselines,
            title="temporal metrics",
            dpi=args.dpi,
        )
        if wrote:
            print(f"Wrote {temporal_path}")
        else:
            print("Skipped temporal metrics overlay: no data points for any temporal metric.")

    if not args.no_training:
        training_path = os.path.join(args.out_dir, "training_metrics_overlay.png")
        wrote = _plot_training(
            training_path,
            experiments,
            title="training metrics",
            dpi=args.dpi,
        )
        if wrote:
            print(f"Wrote {training_path}")
        else:
            print("Skipped training metrics overlay: no data points for any training metric.")

    if not args.no_tables:
        tables_path = os.path.join(args.out_dir, args.tables_file)
        wrote = _write_metric_tables(tables_path, experiments)
        if wrote:
            print(f"Wrote {tables_path}")
        else:
            print("Skipped metric tables: no data points found.")


if __name__ == "__main__":
    main()
