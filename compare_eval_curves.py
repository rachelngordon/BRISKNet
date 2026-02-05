#!/usr/bin/env python3
import argparse
import os
from typing import Dict, List, Optional, Tuple

import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


SIX_PANEL_METRICS = [
    ("eval_ssims", "DRO Evaluation SSIM", "SSIM", "avg_grasp_ssim"),
    ("eval_psnrs", "DRO Evaluation PSNR", "PSNR", "avg_grasp_psnr"),
    ("eval_mses", "DRO Evaluation Image MSE", "MSE", "avg_grasp_mse"),
    ("eval_lpipses", "Evaluation LPIPS", "LPIPS", "avg_grasp_lpips"),
    ("eval_raw_dc_maes", "Non-DRO Evaluation Raw k-space MAE", "MAE", "avg_grasp_raw_dc_mae"),
    ("eval_curve_corrs", "DRO Tumor Enhancement Curve Correlation", "Pearson Correlation Coefficient", "avg_grasp_curve_corr"),
]

TEMPORAL_METRICS = [
    ("eval_curve_maes", "Curve MAE", "MAE"),
    ("eval_ttae_secs", "Time to Arrival Error", "Seconds"),
    ("eval_iauc10_errs", "IAUC10 Error", "Error"),
    ("eval_peak_errs", "Peak Enhancement Error", "Error"),
]


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


def _load_checkpoint(path: str) -> Dict:
    return torch.load(path, map_location="cpu")


def _default_label(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _build_epoch_axis(n: int, eval_frequency: int) -> List[int]:
    return [i * eval_frequency for i in range(n)]


def _resolve_temporal_axes(
    curves: Dict[str, List[float]],
    eval_frequency: int,
) -> Optional[Tuple[List[int], Dict[str, List[float]]]]:
    temporal_epochs = _as_list(curves.get("eval_temporal_epochs", []))
    metric_lists = [curves.get(k, []) for k, _, _ in TEMPORAL_METRICS]

    if temporal_epochs and len(temporal_epochs) == len(metric_lists[0]):
        epochs = temporal_epochs
    else:
        epochs = _build_epoch_axis(len(metric_lists[0]), eval_frequency)

    min_len = min([len(epochs)] + [len(v) for v in metric_lists])
    if min_len == 0:
        return None

    epochs = epochs[:min_len]
    trimmed = {
        key: values[:min_len]
        for (key, _, _), values in zip(TEMPORAL_METRICS, metric_lists)
    }
    return epochs, trimmed


def _plot_six_panel(
    out_path: str,
    exp_a: Dict,
    exp_b: Dict,
    eval_frequency: int,
    show_baseline: bool,
    title: str,
    dpi: int,
) -> None:
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=20)

    colors = {"a": "tab:blue", "b": "tab:orange"}

    for idx, (key, panel_title, ylabel, baseline_key) in enumerate(SIX_PANEL_METRICS):
        ax = axes[idx // 3][idx % 3]
        for tag, exp in (("a", exp_a), ("b", exp_b)):
            values = exp["curves"].get(key, [])
            if not values:
                continue
            x = _build_epoch_axis(len(values), eval_frequency)
            ax.plot(x, values, label=exp["label"], color=colors[tag])
            if show_baseline:
                baseline = exp["baselines"].get(baseline_key)
                if baseline is not None:
                    ax.axhline(y=baseline, color=colors[tag], linestyle="--", alpha=0.5, linewidth=1.5)

        ax.set_title(panel_title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        if not exp_a["curves"].get(key) and not exp_b["curves"].get(key):
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    for idx in range(len(SIX_PANEL_METRICS)):
        ax = axes[idx // 3][idx % 3]
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best")
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_temporal(
    out_path: str,
    exp_a: Dict,
    exp_b: Dict,
    eval_frequency: int,
    title: str,
    dpi: int,
) -> None:
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=20)

    colors = {"a": "tab:blue", "b": "tab:orange"}

    resolved_a = _resolve_temporal_axes(exp_a["curves"], eval_frequency)
    resolved_b = _resolve_temporal_axes(exp_b["curves"], eval_frequency)

    for idx, (key, panel_title, ylabel) in enumerate(TEMPORAL_METRICS):
        ax = axes[idx // 2][idx % 2]
        plotted = False
        if resolved_a is not None:
            epochs_a, curves_a = resolved_a
            values_a = curves_a.get(key, [])
            if values_a:
                ax.plot(epochs_a, values_a, label=exp_a["label"], color=colors["a"])
                plotted = True
        if resolved_b is not None:
            epochs_b, curves_b = resolved_b
            values_b = curves_b.get(key, [])
            if values_b:
                ax.plot(epochs_b, values_b, label=exp_b["label"], color=colors["b"])
                plotted = True

        ax.set_title(panel_title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        if not plotted:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    for idx in range(len(TEMPORAL_METRICS)):
        ax = axes[idx // 2][idx % 2]
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best")
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Overlay eval curves from two checkpoints for six-panel and temporal metrics plots."
    )
    ap.add_argument("--ckpt-a", required=True, help="Path to first checkpoint.")
    ap.add_argument("--ckpt-b", required=True, help="Path to second checkpoint.")
    ap.add_argument("--label-a", default=None, help="Label for first checkpoint.")
    ap.add_argument("--label-b", default=None, help="Label for second checkpoint.")
    ap.add_argument("--eval-frequency", type=int, default=1, help="Eval frequency in epochs.")
    ap.add_argument("--out-dir", default=".", help="Output directory for plots.")
    ap.add_argument("--dpi", type=int, default=150, help="Output DPI.")
    ap.add_argument("--no-six-panel", action="store_true", help="Skip six-panel metrics plot.")
    ap.add_argument("--no-temporal", action="store_true", help="Skip temporal metrics plot.")
    ap.add_argument(
        "--show-grasp-baseline",
        action="store_true",
        help="Overlay dashed lines for avg_grasp_* baselines (if present).",
    )
    args = ap.parse_args()

    ckpt_a = _load_checkpoint(args.ckpt_a)
    ckpt_b = _load_checkpoint(args.ckpt_b)

    exp_a = {
        "label": args.label_a or _default_label(args.ckpt_a),
        "curves": {key: _as_list(ckpt_a.get(key, [])) for key, _, _, _ in SIX_PANEL_METRICS},
        "baselines": {
            "avg_grasp_ssim": _as_scalar(ckpt_a.get("avg_grasp_ssim")),
            "avg_grasp_psnr": _as_scalar(ckpt_a.get("avg_grasp_psnr")),
            "avg_grasp_mse": _as_scalar(ckpt_a.get("avg_grasp_mse")),
            "avg_grasp_lpips": _as_scalar(ckpt_a.get("avg_grasp_lpips")),
            "avg_grasp_raw_dc_mae": _as_scalar(ckpt_a.get("avg_grasp_raw_dc_mae")),
            "avg_grasp_curve_corr": _as_scalar(ckpt_a.get("avg_grasp_curve_corr")),
        },
    }
    exp_b = {
        "label": args.label_b or _default_label(args.ckpt_b),
        "curves": {key: _as_list(ckpt_b.get(key, [])) for key, _, _, _ in SIX_PANEL_METRICS},
        "baselines": {
            "avg_grasp_ssim": _as_scalar(ckpt_b.get("avg_grasp_ssim")),
            "avg_grasp_psnr": _as_scalar(ckpt_b.get("avg_grasp_psnr")),
            "avg_grasp_mse": _as_scalar(ckpt_b.get("avg_grasp_mse")),
            "avg_grasp_lpips": _as_scalar(ckpt_b.get("avg_grasp_lpips")),
            "avg_grasp_raw_dc_mae": _as_scalar(ckpt_b.get("avg_grasp_raw_dc_mae")),
            "avg_grasp_curve_corr": _as_scalar(ckpt_b.get("avg_grasp_curve_corr")),
        },
    }

    # Add temporal curves to each experiment.
    for key, _, _ in TEMPORAL_METRICS:
        exp_a["curves"][key] = _as_list(ckpt_a.get(key, []))
        exp_b["curves"][key] = _as_list(ckpt_b.get(key, []))
    exp_a["curves"]["eval_temporal_epochs"] = _as_list(ckpt_a.get("eval_temporal_epochs", []))
    exp_b["curves"]["eval_temporal_epochs"] = _as_list(ckpt_b.get("eval_temporal_epochs", []))

    os.makedirs(args.out_dir, exist_ok=True)

    if not args.no_six_panel:
        six_path = os.path.join(args.out_dir, "eval_metrics_overlay.png")
        _plot_six_panel(
            six_path,
            exp_a,
            exp_b,
            eval_frequency=args.eval_frequency,
            show_baseline=args.show_grasp_baseline,
            title="Evaluation Metrics Over Epochs (Overlay)",
            dpi=args.dpi,
        )
        print(f"Wrote {six_path}")

    if not args.no_temporal:
        temporal_path = os.path.join(args.out_dir, "eval_temporal_metrics_overlay.png")
        _plot_temporal(
            temporal_path,
            exp_a,
            exp_b,
            eval_frequency=args.eval_frequency,
            title="Temporal Fidelity Metrics Over Epochs (Overlay)",
            dpi=args.dpi,
        )
        print(f"Wrote {temporal_path}")


if __name__ == "__main__":
    main()
