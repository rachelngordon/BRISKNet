#!/usr/bin/env python3
import argparse
import glob
import math
import os
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
    return torch.load(path, map_location="cpu")


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


def _experiment_has_any_data(exp: Dict) -> bool:
    metric_keys = [key for key, *_ in EVAL_PANEL_LAYOUT]
    metric_keys.extend([key for key, _, _ in TEMPORAL_METRICS])
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
    for key, panel_title, ylabel, baseline_key, (row, col) in EVAL_PANEL_LAYOUT:
        ax = axes[row][col]
        plotted = False
        for color, exp in zip(palette, experiments):
            values = exp["curves"].get(key, [])
            if not _has_finite(values):
                continue
            x = _build_epoch_axis(len(values), exp["eval_frequency"])
            if len(values) == 1:
                ax.plot(
                    x,
                    values,
                    label=exp["label"],
                    color=color,
                    linestyle="None",
                    marker="o",
                    markersize=5,
                )
            else:
                ax.plot(x, values, label=exp["label"], color=color, linewidth=1.8)
            plotted = True
        if show_baseline:
            baseline = grasp_baselines.get(baseline_key)
            if _is_finite(baseline):
                ax.axhline(y=baseline, label="GRASP", **GRASP_LINE_STYLE)

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

    axes_flat = [axes[row][col] for _, _, _, _, (row, col) in EVAL_PANEL_LAYOUT]
    handles, labels = _collect_unique_legend(axes_flat)
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

    resolved = [
        _resolve_temporal_axes(exp["curves"], exp["eval_frequency"]) for exp in experiments
    ]

    plotted_panels = 0
    for idx, (key, panel_title, ylabel) in enumerate(TEMPORAL_METRICS):
        ax = axes[idx // 2][idx % 2]
        plotted = False
        for color, exp, resolved_exp in zip(palette, experiments, resolved):
            if resolved_exp is None:
                continue
            epochs, curves = resolved_exp
            values = curves.get(key, [])
            if values:
                if len(values) == 1:
                    ax.plot(
                        epochs,
                        values,
                        label=exp["label"],
                        color=color,
                        linestyle="None",
                        marker="o",
                        markersize=5,
                    )
                else:
                    ax.plot(epochs, values, label=exp["label"], color=color, linewidth=1.8)
                plotted = True
        if show_baseline:
            baseline_key = TEMPORAL_GRASP_BASELINE_KEYS.get(key)
            if baseline_key:
                baseline = grasp_baselines.get(baseline_key)
                if _is_finite(baseline):
                    ax.axhline(y=baseline, label="GRASP", **GRASP_LINE_STYLE)

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

    axes_flat = [axes[idx // 2][idx % 2] for idx in range(len(TEMPORAL_METRICS))]
    handles, labels = _collect_unique_legend(axes_flat)
    if handles:
        legend_ax = fig.add_axes([0.69, 0.08, 0.30, 0.84])
        legend_ax.axis("off")
        legend_ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=9)

    fig.subplots_adjust(left=0.07, right=0.67, bottom=0.10, top=0.88, wspace=0.28, hspace=0.32)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


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

    def _build_experiment(ckpt: Dict, label: str, eval_frequency: int, spf: Optional[int]) -> Dict:
        curves = {key: _as_list(ckpt.get(key, [])) for key, _, _, _, _ in EVAL_PANEL_LAYOUT}
        for key, _, _ in TEMPORAL_METRICS:
            curves[key] = _as_list(ckpt.get(key, []))
        curves["eval_temporal_epochs"] = _as_list(ckpt.get("eval_temporal_epochs", []))
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
            experiments.append(
                _build_experiment(
                    ckpt=ckpt,
                    label=label or _default_label(ckpt_path),
                    eval_frequency=eval_frequency,
                    spf=spf,
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
            experiments.append(
                _build_experiment(
                    ckpt=ckpt,
                    label=label or _default_label(ckpt_path),
                    eval_frequency=eval_frequency,
                    spf=spf,
                )
            )

    filtered = []
    for exp in experiments:
        if _experiment_has_any_data(exp):
            filtered.append(exp)
        else:
            print(f"Skipping {exp['label']}: no eval/temporal data points found.")
    experiments = filtered

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


if __name__ == "__main__":
    main()
