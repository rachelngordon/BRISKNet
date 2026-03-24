"""Plot inference metric summaries from JSON logs. Run: python3 -m inference.plot_inference_metrics --help"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "val_inference_logs.json"

SPATIAL_DEFAULTS = ["ssim", "psnr", "mse", "lpips"]
MC_DEFAULTS = ["dro_dc_mae", "dro_dc_mse", "raw_dc_mae", "raw_dc_mse", "raw_dc_psnr", "raw_ssdu_nmse"]
TEMPORAL_DEFAULTS = [
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


def _parse_list(value: str) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _to_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_rows(log_path: str) -> List[Dict]:
    with open(log_path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError("Log file must contain a list of records.")
    return payload


def _temporal_block_name(temporal_subset: str, temporal_region: str) -> str:
    if temporal_subset == "all":
        prefix = "all_pixels"
    elif temporal_subset == "top10":
        prefix = "top10"
    elif temporal_subset == "top20":
        prefix = "top20"
    else:
        prefix = temporal_subset
    return f"{prefix}_{temporal_region}"


def _extract_structured_metric(node: Dict, metric: str) -> Tuple[float | None, float | None]:
    if not isinstance(node, dict):
        return None, None
    mean = _to_float(node.get(f"{metric}_mean"))
    std = _to_float(node.get(f"{metric}_stddev"))
    if std is None:
        std = _to_float(node.get(f"{metric}_std"))
    return mean, std


def _extract_flat_metric(row: Dict, metric: str) -> Tuple[float | None, float | None]:
    mean = _to_float(row.get(f"{metric}_mean"))
    std = _to_float(row.get(f"{metric}_std"))
    if std is None:
        std = _to_float(row.get(f"{metric}_stddev"))
    return mean, std


def _extract_timing_stats(row: Dict) -> Tuple[float | None, float | None]:
    mean = _to_float(row.get("avg_inference_time"))
    if mean is None:
        mean = _to_float(row.get("avg_grasp_recon_time"))
    if mean is None:
        mean = _to_float(row.get("avg_recon_time"))

    std = _to_float(row.get("std_inference_time"))
    if std is None:
        std = _to_float(row.get("std_grasp_recon_time"))
    if std is None:
        std = _to_float(row.get("std_recon_time"))
    return mean, std


def _find_exp_row(rows: List[Dict], exp_name: str) -> Dict | None:
    for row in rows:
        row_type = row.get("type") or row.get("row_type")
        if row_type and row_type != "BRISKNet":
            continue
        if row.get("exp_name") == exp_name:
            return row
    return None


def _select_grasp_row(grasp_rows: List[Dict], metric_type: str) -> Dict | None:
    if not grasp_rows:
        return None
    if metric_type == "timing":
        for row in grasp_rows:
            mean, _ = _extract_timing_stats(row)
            if mean is not None:
                return row
        return grasp_rows[0]
    for row in grasp_rows:
        if row.get("spatial_metrics") or row.get("dc_metrics") or row.get("temporal_metrics"):
            return row
    return grasp_rows[0]


def _match_grasp_row(exp_row: Dict, grasp_rows: List[Dict], metric_type: str) -> Dict | None:
    if exp_row is None:
        return None
    spf = int(float(exp_row.get("spokes_per_frame") or 0))
    frames = int(float(exp_row.get("num_frames") or 0))
    noise = exp_row.get("DRO_noise_level") or exp_row.get("dro_noise_level") or ""
    accel = _to_float(exp_row.get("acceleration") or exp_row.get("acceleration_factor"))

    candidates = []
    for row in grasp_rows:
        if int(float(row.get("spokes_per_frame") or 0)) != spf:
            continue
        if int(float(row.get("num_frames") or 0)) != frames:
            continue
        row_noise = row.get("DRO_noise_level") or row.get("dro_noise_level") or ""
        if str(row_noise) != str(noise):
            continue
        row_accel = _to_float(row.get("acceleration") or row.get("acceleration_factor"))
        if accel is not None and row_accel is not None and abs(row_accel - accel) > 1e-3:
            continue
        candidates.append(row)
    return _select_grasp_row(candidates, metric_type)


def _exp_group_key(exp_row: Dict) -> Tuple[int, int, str, float | None] | None:
    if exp_row is None:
        return None
    spf = int(float(exp_row.get("spokes_per_frame") or 0))
    frames = int(float(exp_row.get("num_frames") or 0))
    noise = exp_row.get("DRO_noise_level") or exp_row.get("dro_noise_level") or ""
    accel = _to_float(exp_row.get("acceleration") or exp_row.get("acceleration_factor"))
    accel_key = None if accel is None else round(accel, 3)
    return (spf, frames, str(noise), accel_key)


def _extract_metric(row: Dict, metric_type: str, metric: str, temporal_subset: str, temporal_region: str) -> Tuple[float | None, float | None]:
    if row is None:
        return None, None

    if metric_type == "timing":
        return _extract_timing_stats(row)

    if metric_type == "spatial":
        spatial = row.get("spatial_metrics")
        if spatial is not None:
            return _extract_structured_metric(spatial, metric)
        return _extract_flat_metric(row, f"dl_{metric}")

    if metric_type == "mc":
        dc = row.get("dc_metrics")
        if dc is not None:
            return _extract_structured_metric(dc, metric)
        return _extract_flat_metric(row, f"dl_{metric}")

    if metric_type == "temporal":
        temporal = row.get("temporal_metrics")
        block_name = _temporal_block_name(temporal_subset, temporal_region)
        if temporal is not None:
            return _extract_structured_metric(temporal.get(block_name, {}), metric)
        prefix = "" if temporal_region == "malignant" else "benign_"
        key = f"{prefix}dl_{temporal_subset}_{metric}"
        return _extract_flat_metric(row, key)

    raise ValueError(f"Unknown metric_type '{metric_type}'.")


def _plot_single_metric(exp_names: List[str], values: List[Tuple[float | None, float | None]],
                        grasp_groups: List[Dict] | None,
                        metric_type: str, temporal_subset: str, temporal_region: str,
                        metric: str, out_path: str, title: str | None, ylabel: str | None) -> None:
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(exp_names)), 4))
    x = list(range(len(exp_names)))
    means = [v[0] for v in values]
    stds = [v[1] for v in values]
    has_grasp = bool(grasp_groups) and any(
        _extract_metric(group.get("row"), metric_type, metric, temporal_subset, temporal_region)[0] is not None
        for group in grasp_groups
    )
    dx = 0.12 if has_grasp else 0.0
    grasp_color = "black"

    cmap = plt.get_cmap("tab10")
    for i, (mean, std) in enumerate(zip(means, stds)):
        if mean is None:
            continue
        color = cmap(i % 10)
        ax.errorbar(
            [x[i] - dx],
            [mean],
            yerr=[std] if std is not None else None,
            fmt="o",
            color=color,
            capsize=4,
            label=exp_names[i],
        )
    if has_grasp and grasp_groups is not None:
        x_right = (len(exp_names) - 1) + 0.6
        for group_idx, group in enumerate(grasp_groups):
            gmean, gstd = _extract_metric(
                group.get("row"),
                metric_type,
                metric,
                temporal_subset,
                temporal_region,
            )
            if gmean is None:
                continue
            group_x = x_right + group_idx * 0.2
            ax.errorbar(
                [group_x + dx],
                [gmean],
                yerr=[gstd] if gstd is not None else None,
                fmt="^",
                color=grasp_color,
                capsize=4,
                markerfacecolor="none",
            )

    ax.set_xticks(x)
    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax.set_ylabel(ylabel or metric)
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, borderaxespad=0.0)
    if has_grasp:
        fig.text(0.5, 0.02, "Markers: ● BRISKNet   △ GRASP", ha="center", fontsize=8)
        fig.tight_layout(rect=[0, 0.06, 0.8, 1])
    else:
        fig.tight_layout(rect=[0, 0, 0.8, 1])
    fig.savefig(out_path, dpi=200)


def _plot_multi_metric(exp_names: List[str], metrics: List[str],
                       values_by_exp: List[List[Tuple[float | None, float | None]]],
                       grasp_groups: List[Dict] | None,
                       metric_type: str, temporal_subset: str, temporal_region: str,
                       out_path: str, title: str | None, ylabel: str | None) -> None:
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(metrics)), 4))
    x = list(range(len(metrics)))
    width = 0.8 / max(1, len(exp_names))
    has_grasp = bool(grasp_groups) and any(
        _extract_metric(group.get("row"), metric_type, metric, temporal_subset, temporal_region)[0] is not None
        for group in grasp_groups
        for metric in metrics
    )
    method_delta = width * 0.25 if has_grasp else 0.0
    grasp_color = "black"

    cmap = plt.get_cmap("tab10")
    for exp_idx, exp_name in enumerate(exp_names):
        means = [v[0] for v in values_by_exp[exp_idx]]
        stds = [v[1] for v in values_by_exp[exp_idx]]
        offsets = [xi + (exp_idx - (len(exp_names) - 1) / 2) * width for xi in x]
        color = cmap(exp_idx % 10)
        ax.errorbar(
            [o - method_delta for o in offsets],
            means,
            yerr=stds,
            fmt="o",
            color=color,
            capsize=3,
            label=exp_name,
        )
    if has_grasp and grasp_groups is not None:
        for group_idx, group in enumerate(grasp_groups):
            indices = group.get("exp_indices", [])
            if not indices:
                continue
            for metric_idx, metric in enumerate(metrics):
                gmean, gstd = _extract_metric(
                    group.get("row"),
                    metric_type,
                    metric,
                    temporal_subset,
                    temporal_region,
                )
                if gmean is None:
                    continue
                cluster_right = x[metric_idx] + ((len(exp_names) - 1) / 2) * width
                group_x = cluster_right + (0.6 + group_idx * 0.25) * width
                ax.errorbar(
                    [group_x + method_delta],
                    [gmean],
                    yerr=[gstd] if gstd is not None else None,
                    fmt="^",
                    color=grasp_color,
                    capsize=3,
                    markerfacecolor="none",
                    label=None,
                )

    ax.set_xticks(x)
    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax.set_ylabel(ylabel or "Metric")
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, borderaxespad=0.0)
    if has_grasp:
        fig.text(0.5, 0.02, "Markers: ● BRISKNet   △ GRASP", ha="center", fontsize=8)
        fig.tight_layout(rect=[0, 0.06, 0.8, 1])
    else:
        fig.tight_layout(rect=[0, 0, 0.8, 1])
    fig.savefig(out_path, dpi=200)


def _plot_panel_metrics(exp_names: List[str], metrics: List[str],
                        values_by_exp: List[List[Tuple[float | None, float | None]]],
                        grasp_groups: List[Dict] | None,
                        metric_type: str, temporal_subset: str, temporal_region: str,
                        out_path: str, title: str | None, ylabel: str | None, panel_cols: int) -> None:
    n_metrics = len(metrics)
    ncols = max(1, min(panel_cols, n_metrics))
    nrows = (n_metrics + ncols - 1) // ncols
    fig_w = max(6, 3.6 * ncols)
    fig_h = 3.2 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(exp_names))]

    has_grasp = bool(grasp_groups) and any(
        _extract_metric(group.get("row"), metric_type, metric, temporal_subset, temporal_region)[0] is not None
        for group in grasp_groups
        for metric in metrics
    )
    dx = 0.12 if has_grasp else 0.0
    grasp_color = "black"

    x = list(range(len(exp_names)))
    for idx, metric in enumerate(metrics):
        ax = axes[idx // ncols][idx % ncols]
        for exp_idx, exp_name in enumerate(exp_names):
            mean, std = values_by_exp[exp_idx][idx]
            if mean is None:
                continue
            ax.errorbar(
                [x[exp_idx] - dx],
                [mean],
                yerr=[std] if std is not None else None,
                fmt="o",
                color=colors[exp_idx],
                capsize=3,
            )
        if has_grasp and grasp_groups is not None:
            x_right = (len(exp_names) - 1) + 0.6
            for group_idx, group in enumerate(grasp_groups):
                gmean, gstd = _extract_metric(
                    group.get("row"),
                    metric_type,
                    metric,
                    temporal_subset,
                    temporal_region,
                )
                if gmean is None:
                    continue
                group_x = x_right + group_idx * 0.2
                ax.errorbar(
                    [group_x + dx],
                    [gmean],
                    yerr=[gstd] if gstd is not None else None,
                    fmt="^",
                    color=grasp_color,
                    capsize=3,
                    markerfacecolor="none",
                )
        ax.set_xticks(x)
        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        ax.set_title(metric)
        ax.set_ylabel(ylabel or "Metric")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Hide unused axes
    for idx in range(n_metrics, nrows * ncols):
        ax = axes[idx // ncols][idx % ncols]
        ax.axis("off")

    handles = [
        Line2D([], [], marker="o", color=colors[i], linestyle="None", label=exp_name)
        for i, exp_name in enumerate(exp_names)
    ]
    legend_cols = min(len(exp_names), 4) if len(exp_names) > 0 else 1
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        fontsize=8,
        borderaxespad=0.0,
        ncol=legend_cols,
        frameon=False,
    )
    if has_grasp:
        fig.text(0.5, 0.08, "Markers: ● BRISKNet   △ GRASP", ha="center", fontsize=8)
        rect = [0, 0.14, 1, 1]
    else:
        rect = [0, 0.08, 1, 1]
    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=rect)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot inference metrics with error bars from val_inference_logs.")
    parser.add_argument(
        "--log_file",
        default=str(DEFAULT_LOG_PATH),
        help="Path to val_inference_logs.json (default: inference/val_inference_logs.json).",
    )
    parser.add_argument("--exp_names", required=True, help="Comma-separated experiment names to include.")
    parser.add_argument(
        "--metric_type",
        choices=("spatial", "mc", "temporal", "timing"),
        required=True,
        help="Metric type to plot.",
    )
    parser.add_argument(
        "--metric",
        default="",
        help="Metric name to plot (ignored for timing).",
    )
    parser.add_argument(
        "--metrics",
        default="",
        help="Comma-separated metrics to plot (overrides --metric when provided).",
    )
    parser.add_argument(
        "--panel",
        action="store_true",
        help="Render a panel of subplots (one per metric).",
    )
    parser.add_argument(
        "--panel_group",
        choices=("spatial", "mc", "temporal"),
        default="",
        help="Use the default metric list for a group when plotting a panel.",
    )
    parser.add_argument(
        "--panel_cols",
        type=int,
        default=3,
        help="Number of columns for panel plots.",
    )
    parser.add_argument(
        "--temporal_subset",
        choices=("all", "top10", "top20"),
        default="all",
        help="Temporal subset when metric_type=temporal.",
    )
    parser.add_argument(
        "--temporal_region",
        choices=("malignant", "benign"),
        default="malignant",
        help="Temporal region when metric_type=temporal.",
    )
    parser.add_argument(
        "--out",
        default="inference_metric_plot.png",
        help="Output image path.",
    )
    parser.add_argument("--title", default="", help="Plot title.")
    parser.add_argument("--ylabel", default="", help="Y-axis label.")
    args = parser.parse_args()

    exp_names = _parse_list(args.exp_names)
    if not exp_names:
        raise SystemExit("No experiment names provided.")

    metrics = _parse_list(args.metrics)
    if args.panel_group:
        if args.panel_group != args.metric_type:
            raise SystemExit("--panel_group must match --metric_type.")
        if args.panel_group == "spatial":
            metrics = SPATIAL_DEFAULTS
        elif args.panel_group == "mc":
            metrics = MC_DEFAULTS
        elif args.panel_group == "temporal":
            metrics = TEMPORAL_DEFAULTS
    if not metrics:
        if args.panel:
            if args.metric_type == "spatial":
                metrics = SPATIAL_DEFAULTS
            elif args.metric_type == "mc":
                metrics = MC_DEFAULTS
            elif args.metric_type == "temporal":
                metrics = TEMPORAL_DEFAULTS
            else:
                metrics = ["timing"]
        else:
            if args.metric_type != "timing" and not args.metric:
                raise SystemExit("Provide --metric or --metrics for non-timing plots.")
            metrics = [args.metric or "timing"]

    rows = _load_rows(args.log_file)
    exp_rows = [r for r in rows if (r.get("type") == "BRISKNet" or (r.get("row_type") or "exp") == "exp")]
    grasp_rows = [r for r in rows if (r.get("type") == "GRASP" or r.get("row_type") == "grasp_agg")]

    exp_to_row = {name: _find_exp_row(exp_rows, name) for name in exp_names}

    grasp_group_map: Dict[Tuple[int, int, str, float | None], Dict] = {}
    for idx, name in enumerate(exp_names):
        row = exp_to_row.get(name)
        key = _exp_group_key(row)
        if key is None:
            continue
        entry = grasp_group_map.get(key)
        if entry is None:
            entry = {"key": key, "exp_indices": [], "row": None}
            grasp_group_map[key] = entry
        entry["exp_indices"].append(idx)
        if entry["row"] is None:
            entry["row"] = _match_grasp_row(row, grasp_rows, args.metric_type)
    grasp_groups = list(grasp_group_map.values())

    missing = [name for name, row in exp_to_row.items() if row is None]
    if missing:
        missing_str = ", ".join(missing)
        print(f"Warning: no rows found for experiments: {missing_str}")

    if args.panel:
        values_by_exp = []
        for name in exp_names:
            row = exp_to_row.get(name)
            values_by_exp.append(
                [
                    _extract_metric(row, args.metric_type, metric, args.temporal_subset, args.temporal_region)
                    for metric in metrics
                ]
            )
        _plot_panel_metrics(
            exp_names,
            metrics,
            values_by_exp,
            grasp_groups,
            args.metric_type,
            args.temporal_subset,
            args.temporal_region,
            args.out,
            args.title or None,
            args.ylabel or None,
            args.panel_cols,
        )
    elif len(metrics) == 1:
        metric = metrics[0]
        values = [
            _extract_metric(exp_to_row.get(name), args.metric_type, metric, args.temporal_subset, args.temporal_region)
            for name in exp_names
        ]
        _plot_single_metric(
            exp_names,
            values,
            grasp_groups,
            args.metric_type,
            args.temporal_subset,
            args.temporal_region,
            metric,
            args.out,
            args.title or None,
            args.ylabel or None,
        )
    else:
        values_by_exp = []
        for name in exp_names:
            row = exp_to_row.get(name)
            values_by_exp.append(
                [
                    _extract_metric(row, args.metric_type, metric, args.temporal_subset, args.temporal_region)
                    for metric in metrics
                ]
            )
        _plot_multi_metric(
            exp_names,
            metrics,
            values_by_exp,
            grasp_groups,
            args.metric_type,
            args.temporal_subset,
            args.temporal_region,
            args.out,
            args.title or None,
            args.ylabel or None,
        )

    print(f"Saved plot to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
