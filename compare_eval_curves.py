#!/usr/bin/env python3
import argparse
import math
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

TEMPORAL_GRASP_BASELINE_KEYS = {
    "eval_curve_maes": "avg_grasp_curve_mae",
    "eval_ttae_secs": "avg_grasp_ttae_sec",
    "eval_iauc10_errs": "avg_grasp_iauc10_err",
    "eval_peak_errs": "avg_grasp_peak_err",
}

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


def _resolve_grasp_baselines(
    experiments: List[Dict],
    baseline_keys: List[str],
) -> Dict[str, Optional[float]]:
    baselines: Dict[str, Optional[float]] = {}
    for key in baseline_keys:
        values = []
        labels = []
        for exp in experiments:
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


def _plot_six_panel(
    out_path: str,
    experiments: List[Dict],
    eval_frequency: int,
    show_baseline: bool,
    grasp_baselines: Dict[str, Optional[float]],
    title: str,
    dpi: int,
) -> None:
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=20)

    if len(experiments) <= 10:
        palette = sns.color_palette("tab10", n_colors=len(experiments))
    else:
        palette = sns.color_palette("husl", n_colors=len(experiments))

    for idx, (key, panel_title, ylabel, baseline_key) in enumerate(SIX_PANEL_METRICS):
        ax = axes[idx // 3][idx % 3]
        for color, exp in zip(palette, experiments):
            values = exp["curves"].get(key, [])
            if not values:
                continue
            x = _build_epoch_axis(len(values), eval_frequency)
            ax.plot(x, values, label=exp["label"], color=color)
        if show_baseline:
            baseline = grasp_baselines.get(baseline_key)
            if _is_finite(baseline):
                ax.axhline(y=baseline, label="GRASP", **GRASP_LINE_STYLE)

        ax.set_title(panel_title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        if not any(exp["curves"].get(key) for exp in experiments):
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
    experiments: List[Dict],
    eval_frequency: int,
    show_baseline: bool,
    grasp_baselines: Dict[str, Optional[float]],
    title: str,
    dpi: int,
) -> None:
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=20)

    if len(experiments) <= 10:
        palette = sns.color_palette("tab10", n_colors=len(experiments))
    else:
        palette = sns.color_palette("husl", n_colors=len(experiments))

    resolved = [
        _resolve_temporal_axes(exp["curves"], eval_frequency) for exp in experiments
    ]

    for idx, (key, panel_title, ylabel) in enumerate(TEMPORAL_METRICS):
        ax = axes[idx // 2][idx % 2]
        plotted = False
        for color, exp, resolved_exp in zip(palette, experiments, resolved):
            if resolved_exp is None:
                continue
            epochs, curves = resolved_exp
            values = curves.get(key, [])
            if values:
                ax.plot(epochs, values, label=exp["label"], color=color)
                plotted = True
        if show_baseline:
            baseline_key = TEMPORAL_GRASP_BASELINE_KEYS.get(key)
            if baseline_key:
                baseline = grasp_baselines.get(baseline_key)
                if _is_finite(baseline):
                    ax.axhline(y=baseline, label="GRASP", **GRASP_LINE_STYLE)

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
        description="Overlay eval curves from multiple checkpoints for six-panel and temporal metrics plots."
    )
    ap.add_argument(
        "--ckpt",
        nargs="+",
        default=None,
        help="Checkpoint paths (2+).",
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
    ap.add_argument("--eval-frequency", type=int, default=1, help="Eval frequency in epochs.")
    ap.add_argument("--out-dir", default=".", help="Output directory for plots.")
    ap.add_argument("--dpi", type=int, default=150, help="Output DPI.")
    ap.add_argument("--no-six-panel", action="store_true", help="Skip six-panel metrics plot.")
    ap.add_argument("--no-temporal", action="store_true", help="Skip temporal metrics plot.")
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
    args = ap.parse_args()

    def _build_experiment(ckpt: Dict, label: str) -> Dict:
        curves = {key: _as_list(ckpt.get(key, [])) for key, _, _, _ in SIX_PANEL_METRICS}
        for key, _, _ in TEMPORAL_METRICS:
            curves[key] = _as_list(ckpt.get(key, []))
        curves["eval_temporal_epochs"] = _as_list(ckpt.get("eval_temporal_epochs", []))
        baselines = {
            key: _as_scalar(value)
            for key, value in ckpt.items()
            if key.startswith("avg_grasp_")
        }
        return {"label": label, "curves": curves, "baselines": baselines}

    if args.ckpt:
        ckpt_paths = list(args.ckpt)
        labels = list(args.label or [])
    else:
        if not args.ckpt_a or not args.ckpt_b:
            raise SystemExit("Provide --ckpt paths (2+) or legacy --ckpt-a/--ckpt-b.")
        ckpt_paths = [args.ckpt_a, args.ckpt_b]
        labels = [args.label_a, args.label_b]

    if len(ckpt_paths) < 2:
        raise SystemExit("Need at least two checkpoints to compare.")

    if len(labels) < len(ckpt_paths):
        labels.extend([None] * (len(ckpt_paths) - len(labels)))
    labels = labels[: len(ckpt_paths)]

    experiments = []
    for ckpt_path, label in zip(ckpt_paths, labels):
        ckpt = _load_checkpoint(ckpt_path)
        experiments.append(_build_experiment(ckpt, label or _default_label(ckpt_path)))

    baseline_keys = [baseline_key for _, _, _, baseline_key in SIX_PANEL_METRICS]
    baseline_keys.extend(list(TEMPORAL_GRASP_BASELINE_KEYS.values()))
    grasp_baselines = _resolve_grasp_baselines(experiments, baseline_keys)

    os.makedirs(args.out_dir, exist_ok=True)

    show_grasp_baseline = args.show_grasp_baseline or not args.no_grasp_baseline

    if not args.no_six_panel:
        six_path = os.path.join(args.out_dir, "eval_metrics_overlay.png")
        _plot_six_panel(
            six_path,
            experiments,
            eval_frequency=args.eval_frequency,
            show_baseline=show_grasp_baseline,
            grasp_baselines=grasp_baselines,
            title="Evaluation Metrics Over Epochs (Overlay)",
            dpi=args.dpi,
        )
        print(f"Wrote {six_path}")

    if not args.no_temporal:
        temporal_path = os.path.join(args.out_dir, "eval_temporal_metrics_overlay.png")
        _plot_temporal(
            temporal_path,
            experiments,
            eval_frequency=args.eval_frequency,
            show_baseline=show_grasp_baseline,
            grasp_baselines=grasp_baselines,
            title="Temporal Fidelity Metrics Over Epochs (Overlay)",
            dpi=args.dpi,
        )
        print(f"Wrote {temporal_path}")


if __name__ == "__main__":
    main()
