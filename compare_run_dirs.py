#!/usr/bin/env python3
"""Overlay eval and loss curves from two experiment output directories."""

import argparse
import glob
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml


EVAL_METRICS: List[Tuple[str, str, str, Optional[str]]] = [
    ("eval_ssims", "DRO SSIM", "SSIM", "avg_grasp_ssim"),
    ("eval_psnrs", "DRO PSNR", "PSNR", "avg_grasp_psnr"),
    ("eval_mses", "DRO Image MSE", "MSE", "avg_grasp_mse"),
    ("eval_lpipses", "DRO LPIPS", "LPIPS", "avg_grasp_lpips"),
    ("eval_dc_maes", "DRO k-space MAE (sim)", "MAE", "avg_grasp_dc_mae"),
    ("eval_raw_dc_maes", "Non-DRO k-space MAE", "MAE", "avg_grasp_raw_dc_mae"),
    ("eval_curve_corrs", "DRO Curve Correlation", "Pearson Correlation", "avg_grasp_curve_corr"),
    ("eval_dl_dc_mae_bestfits", "DRO k-space MAE (best-fit gain)", "MAE", "avg_grasp_dc_mae_bestfit"),
    ("eval_raw_ssdu_nmses", "Non-DRO SSDU NMSE", "NMSE", "avg_grasp_raw_ssdu_nmse"),
]

UNWEIGHTED_LOSS_METRICS: List[Tuple[str, str, str]] = [
    ("train_mc_losses", "Train MC Loss", "Loss"),
    ("train_ei_losses", "Train EI Loss", "Loss"),
    ("train_rebin_losses", "Train Rebin Loss", "Loss"),
    ("train_adj_losses", "Train Adjoint Loss", "Loss"),
]

WEIGHTED_LOSS_METRICS: List[Tuple[str, str, str]] = [
    ("weighted_train_mc_losses", "Weighted Train MC Loss", "Loss"),
    ("weighted_train_ei_losses", "Weighted Train EI Loss", "Loss"),
    ("weighted_train_rebin_losses", "Weighted Train Rebin Loss", "Loss"),
    ("weighted_train_adj_losses", "Weighted Train Adjoint Loss", "Loss"),
]

GRASP_LINE_STYLE = {
    "color": "tab:red",
    "linestyle": "--",
    "alpha": 0.85,
    "linewidth": 1.8,
}


def _as_list(value) -> List[float]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().flatten().tolist()
    if isinstance(value, (list, tuple)):
        out: List[float] = []
        for item in value:
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                continue
        return out
    try:
        return [float(value)]
    except (TypeError, ValueError):
        return []


def _resolve_checkpoint(run_dir: str, explicit_ckpt: Optional[str], prefer_best: bool) -> str:
    if explicit_ckpt:
        ckpt_path = explicit_ckpt
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(run_dir, ckpt_path)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path

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

    # Prefer exact <run_dir_basename>_model.pth or <basename>_best_model.pth when present.
    base = os.path.basename(os.path.normpath(run_dir))
    exact_priority = [
        os.path.join(run_dir, f"{base}_model.pth"),
        os.path.join(run_dir, f"{base}_best_model.pth"),
    ]
    for exact in exact_priority:
        if exact in candidates:
            return exact

    return max(candidates, key=os.path.getmtime)


def _infer_eval_frequency(run_dir: str, override: Optional[int]) -> int:
    if override is not None:
        if override < 1:
            raise ValueError("--eval-frequency must be >= 1.")
        return int(override)

    config_path = os.path.join(run_dir, "config.yaml")
    if not os.path.isfile(config_path):
        return 1
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        return 1
    data_cfg = cfg.get("data", {})
    if not isinstance(data_cfg, dict):
        return 1
    freq = data_cfg.get("eval_frequency", 1)
    try:
        freq_int = int(freq)
    except (TypeError, ValueError):
        return 1
    return max(1, freq_int)


def _infer_spf(run_dir: str) -> Optional[int]:
    config_path = os.path.join(run_dir, "config.yaml")
    if not os.path.isfile(config_path):
        return None
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        return None
    data_cfg = cfg.get("data", {})
    if not isinstance(data_cfg, dict):
        return None
    candidates = (
        data_cfg.get("eval_spokes"),
        data_cfg.get("train_spokes_per_frame"),
        data_cfg.get("fpg"),
    )
    for value in candidates:
        try:
            iv = int(value)
        except (TypeError, ValueError):
            continue
        if iv > 0:
            return iv
    return None


def _label_with_spf(label: str, spf: Optional[int]) -> str:
    if spf is None:
        return label
    return f"{label} (SPF={spf})"


def _resolve_shared_baseline(experiments: Sequence[Dict], baseline_key: str) -> Optional[float]:
    vals = []
    for exp in experiments:
        val = exp.get("baselines", {}).get(baseline_key)
        if val is None:
            continue
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fval):
            vals.append(fval)
    if not vals:
        return None
    return vals[0]


def _collect_unique_legend(axes_list):
    handles = []
    labels = []
    seen = set()
    for ax in axes_list:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label in seen:
                continue
            seen.add(label)
            handles.append(handle)
            labels.append(label)
    return handles, labels


def _build_eval_axis(ckpt: Dict, n: int, eval_frequency: int) -> List[int]:
    temporal_epochs = _as_list(ckpt.get("eval_temporal_epochs", []))
    if len(temporal_epochs) >= n:
        return [int(v) for v in temporal_epochs[:n]]

    eval_epochs = _as_list(ckpt.get("eval_epochs", []))
    if len(eval_epochs) >= n:
        return [int(v) for v in eval_epochs[:n]]

    return [i * eval_frequency for i in range(n)]


def _build_epoch_axis(n: int) -> List[int]:
    return list(range(1, n + 1))


def _plot_metric_grid(
    out_path: str,
    experiments: Sequence[Dict],
    metric_specs: Sequence[Tuple],
    title: str,
    metric_kind: str,
    dpi: int,
) -> None:
    cols = 3
    rows = int(math.ceil(len(metric_specs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 4.2))
    if not isinstance(axes, (list, tuple)):
        axes = axes.flatten()
    else:
        axes = list(axes)
    if hasattr(axes, "flatten"):
        axes = axes.flatten().tolist()

    fig.suptitle(title, fontsize=16)

    for idx, spec in enumerate(metric_specs):
        if metric_kind == "eval":
            key, panel_title, ylabel, baseline_key = spec
        else:
            key, panel_title, ylabel = spec
            baseline_key = None
        ax = axes[idx]
        plotted = False
        for exp in experiments:
            series = exp["series"].get(key, [])
            if not series:
                continue
            if not any(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in series):
                continue
            if metric_kind == "eval":
                x = _build_eval_axis(exp["ckpt"], len(series), exp["eval_frequency"])
            else:
                x = _build_epoch_axis(len(series))
            ax.plot(x, series, label=exp["label"], linewidth=1.8)
            plotted = True

        if metric_kind == "eval" and baseline_key:
            baseline = _resolve_shared_baseline(experiments, baseline_key)
            if baseline is not None:
                ax.axhline(y=baseline, label="GRASP", **GRASP_LINE_STYLE)

        ax.set_title(panel_title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if not plotted:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    for j in range(len(metric_specs), len(axes)):
        axes[j].axis("off")

    handles, labels = _collect_unique_legend(axes[: len(metric_specs)])
    if handles:
        legend_ax = fig.add_axes([0.82, 0.08, 0.17, 0.84])
        legend_ax.axis("off")
        legend_ax.legend(handles, labels, loc="center left", frameon=False, fontsize=9)

    fig.subplots_adjust(left=0.06, right=0.80, bottom=0.08, top=0.90, wspace=0.28, hspace=0.34)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_lr_overlay(out_path: str, experiments: Sequence[Dict], dpi: int, title: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    plotted = False
    for exp in experiments:
        lr_hist = exp["series"].get("lr_history", [])
        if not lr_hist:
            continue
        lr_epochs = exp["series"].get("lr_epochs", [])
        if len(lr_epochs) != len(lr_hist):
            lr_epochs = _build_epoch_axis(len(lr_hist))
        ax.plot(lr_epochs, lr_hist, label=exp["label"], linewidth=1.8)
        plotted = True

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.grid(alpha=0.3)
    if not plotted:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    if plotted:
        handles, labels = _collect_unique_legend([ax])
        if handles:
            legend_ax = fig.add_axes([0.82, 0.08, 0.17, 0.84])
            legend_ax.axis("off")
            legend_ax.legend(handles, labels, loc="center left", frameon=False, fontsize=9)
        fig.subplots_adjust(left=0.10, right=0.80, bottom=0.14, top=0.88)
    else:
        plt.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _load_experiment(
    run_dir: str,
    label: Optional[str],
    explicit_ckpt: Optional[str],
    prefer_best: bool,
    eval_frequency_override: Optional[int],
) -> Dict:
    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    ckpt_path = _resolve_checkpoint(run_dir, explicit_ckpt=explicit_ckpt, prefer_best=prefer_best)
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    series_keys = [key for key, *_ in EVAL_METRICS]
    series_keys.extend([key for key, _, _ in UNWEIGHTED_LOSS_METRICS + WEIGHTED_LOSS_METRICS])
    series_keys.extend(["lr_history", "lr_epochs"])
    series = {k: _as_list(ckpt.get(k, [])) for k in series_keys}
    baselines = {k: ckpt.get(k) for k in ckpt.keys() if isinstance(k, str) and k.startswith("avg_grasp_")}
    spf = _infer_spf(run_dir)
    base_label = label or os.path.basename(os.path.normpath(run_dir))
    legend_label = _label_with_spf(base_label, spf)

    return {
        "run_dir": run_dir,
        "ckpt_path": ckpt_path,
        "label": legend_label,
        "ckpt": ckpt,
        "series": series,
        "baselines": baselines,
        "spf": spf,
        "eval_frequency": _infer_eval_frequency(run_dir, eval_frequency_override),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two experiment output directories by overlaying eval and loss curves."
        )
    )
    parser.add_argument("--run-a", required=True, help="Path to first run output directory.")
    parser.add_argument("--run-b", required=True, help="Path to second run output directory.")
    parser.add_argument("--label-a", default=None, help="Optional label for first run.")
    parser.add_argument("--label-b", default=None, help="Optional label for second run.")
    parser.add_argument(
        "--ckpt-a",
        default=None,
        help="Optional explicit checkpoint path for run A (absolute or relative to --run-a).",
    )
    parser.add_argument(
        "--ckpt-b",
        default=None,
        help="Optional explicit checkpoint path for run B (absolute or relative to --run-b).",
    )
    parser.add_argument(
        "--prefer-best",
        action="store_true",
        help="Prefer '*_best_model.pth' over '*_model.pth' when auto-resolving checkpoints.",
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=None,
        help="Override eval-frequency for x-axis. Default reads each run's config.yaml.",
    )
    parser.add_argument("--out-dir", default="comparison_plots", help="Output directory for plots.")
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI.")
    args = parser.parse_args()

    experiments = [
        _load_experiment(
            run_dir=args.run_a,
            label=args.label_a,
            explicit_ckpt=args.ckpt_a,
            prefer_best=args.prefer_best,
            eval_frequency_override=args.eval_frequency,
        ),
        _load_experiment(
            run_dir=args.run_b,
            label=args.label_b,
            explicit_ckpt=args.ckpt_b,
            prefer_best=args.prefer_best,
            eval_frequency_override=args.eval_frequency,
        ),
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    spf_a = experiments[0].get("spf")
    spf_b = experiments[1].get("spf")
    spf_part = (
        f"A SPF={spf_a if spf_a is not None else 'NA'} | B SPF={spf_b if spf_b is not None else 'NA'}"
    )

    eval_out = os.path.join(args.out_dir, "eval_metrics_overlay.png")
    _plot_metric_grid(
        out_path=eval_out,
        experiments=experiments,
        metric_specs=EVAL_METRICS,
        title=f"Evaluation Metrics (Overlay) [{spf_part}]",
        metric_kind="eval",
        dpi=args.dpi,
    )

    loss_out = os.path.join(args.out_dir, "loss_metrics_overlay.png")
    _plot_metric_grid(
        out_path=loss_out,
        experiments=experiments,
        metric_specs=UNWEIGHTED_LOSS_METRICS + WEIGHTED_LOSS_METRICS,
        title=f"Train Loss Metrics (Overlay) [{spf_part}]",
        metric_kind="loss",
        dpi=args.dpi,
    )

    lr_out = os.path.join(args.out_dir, "learning_rate_overlay.png")
    _plot_lr_overlay(
        lr_out,
        experiments=experiments,
        dpi=args.dpi,
        title=f"Learning Rate [{spf_part}]",
    )

    for exp in experiments:
        print(f"Loaded: {exp['label']} -> {exp['ckpt_path']}")
    print(f"Wrote: {eval_out}")
    print(f"Wrote: {loss_out}")
    print(f"Wrote: {lr_out}")


if __name__ == "__main__":
    main()
