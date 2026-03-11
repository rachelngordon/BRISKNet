import argparse
import json
from typing import Dict, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


METRIC_SPECS = [
    ("early_corr", r"$\rho_{\mathrm{early}}$"),
    ("early_mae", r"$MAE_{\mathrm{early}}$"),
    ("ttae_sec", r"$t_{\mathrm{arr}}\ \mathrm{Err}$"),
    ("wash_in_slope_err", r"$MAE_{\mathrm{wash\!-\!in}}$"),
    ("iauc10_err", r"$iAUC_{10}\ \mathrm{Err}$"),
]

TITLE_SIZE = 18
AXIS_LABEL_SIZE = 18
TICK_SIZE = 15
LEGEND_SIZE = 14


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
    with open(log_path, "r", encoding="utf-8") as f:
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


def _extract_temporal_metric(
    row: Dict,
    metric: str,
    temporal_subset: str,
    temporal_region: str,
) -> Tuple[float | None, float | None]:
    if row is None:
        return None, None
    temporal = row.get("temporal_metrics")
    block_name = _temporal_block_name(temporal_subset, temporal_region)
    if temporal is not None:
        return _extract_structured_metric(temporal.get(block_name, {}), metric)
    prefix = "" if temporal_region == "malignant" else "benign_"
    key = f"{prefix}dl_{temporal_subset}_{metric}"
    return _extract_flat_metric(row, key)


def _select_grasp_row(grasp_rows: List[Dict]) -> Dict | None:
    if not grasp_rows:
        return None
    for row in grasp_rows:
        if row.get("temporal_metrics"):
            return row
    return grasp_rows[0]


def _match_grasp_row(exp_row: Dict, grasp_rows: List[Dict]) -> Dict | None:
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
    return _select_grasp_row(candidates)


def _acceleration(row: Dict) -> float | None:
    return _to_float(row.get("acceleration") or row.get("acceleration_factor"))


def _exp_group_key(row: Dict) -> Tuple[int, int, str, float | None]:
    spf = int(float(row.get("spokes_per_frame") or 0))
    frames = int(float(row.get("num_frames") or 0))
    noise = row.get("DRO_noise_level") or row.get("dro_noise_level") or ""
    accel = _acceleration(row)
    accel_key = None if accel is None else round(accel, 3)
    return (spf, frames, str(noise), accel_key)


def _select_exp_row(
    exp_rows: List[Dict],
    exp_name: str,
    spf: int | None,
    frames: int | None,
    accel: float | None,
) -> Dict:
    if spf is None and frames is None and accel is None:
        if len(exp_rows) > 1:
            raise SystemExit(
                f"Multiple rows for '{exp_name}'. Provide --spokes_per_frame/--num_frames/--accel."
            )
        return exp_rows[0]
    candidates = []
    for row in exp_rows:
        if spf is not None and int(float(row.get("spokes_per_frame") or 0)) != spf:
            continue
        if frames is not None and int(float(row.get("num_frames") or 0)) != frames:
            continue
        row_accel = _acceleration(row)
        if accel is not None and row_accel is not None and abs(row_accel - accel) > 1e-3:
            continue
        candidates.append(row)
    if not candidates:
        raise SystemExit(f"No matching row for '{exp_name}' with the provided filters.")
    if len(candidates) > 1:
        raise SystemExit(
            f"Multiple rows for '{exp_name}' after filtering; tighten filters."
        )
    return candidates[0]


def _collect_metric_pair(
    exp_row: Dict,
    grasp_rows: List[Dict],
    metric: str,
    temporal_subset: str,
    temporal_region: str,
) -> Tuple[Tuple[float | None, float | None], Tuple[float | None, float | None]]:
    mean, std = _extract_temporal_metric(exp_row, metric, temporal_subset, temporal_region)
    grasp_row = _match_grasp_row(exp_row, grasp_rows)
    gmean, gstd = _extract_temporal_metric(grasp_row, metric, temporal_subset, temporal_region)
    return (mean, std), (gmean, gstd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot temporal metric panels for BRISKNet vs GRASP (single row by default)."
    )
    parser.add_argument(
        "--log_file",
        default="val_inference_logs.json",
        help="Path to inference log JSON (default: val_inference_logs.json).",
    )
    parser.add_argument(
        "--exp_names",
        required=True,
        help="Comma-separated experiment names (exp_name field in the log).",
    )
    parser.add_argument(
        "--exp_labels",
        default="",
        help="Comma-separated labels matching exp_names (for legend).",
    )
    parser.add_argument(
        "--temporal_subset",
        choices=("all", "top10", "top20"),
        default="all",
        help="Temporal subset to use.",
    )
    parser.add_argument(
        "--temporal_region",
        choices=("malignant", "benign"),
        default="malignant",
        help="Temporal region to use.",
    )
    parser.add_argument("--spokes_per_frame", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--accel", type=float, default=None, help="Acceleration filter (optional).")
    parser.add_argument(
        "--two_rows",
        action="store_true",
        help="Split metrics across two rows (default: single row).",
    )
    parser.add_argument(
        "--out",
        default="temporal_metrics_panel.png",
        help="Output plot path.",
    )
    parser.add_argument("--title", default="", help="Optional figure title.")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    exp_names = _parse_list(args.exp_names)
    if not exp_names:
        raise SystemExit("No experiment names provided.")

    exp_labels = _parse_list(args.exp_labels)
    if exp_labels and len(exp_labels) != len(exp_names):
        raise SystemExit("--exp_labels must match length of --exp_names.")
    if not exp_labels:
        exp_labels = exp_names

    rows = _load_rows(args.log_file)
    exp_rows_all = [
        r
        for r in rows
        if (r.get("type") == "BRISKNet" or (r.get("row_type") or "exp") == "exp")
    ]
    grasp_rows = [
        r for r in rows if (r.get("type") == "GRASP" or r.get("row_type") == "grasp_agg")
    ]

    exp_rows_map: Dict[str, List[Dict]] = {}
    for name in exp_names:
        exp_rows = [r for r in exp_rows_all if r.get("exp_name") == name]
        if not exp_rows:
            raise SystemExit(f"Experiment '{name}' not found in log.")
        exp_rows_map[name] = exp_rows

    exp_selected: List[Dict] = []
    for exp_idx, exp_name in enumerate(exp_names):
        exp_row = _select_exp_row(
            exp_rows_map[exp_name],
            exp_name,
            args.spokes_per_frame,
            args.num_frames,
            args.accel,
        )
        exp_selected.append(
            {
                "exp_idx": exp_idx,
                "exp_name": exp_name,
                "exp_label": exp_labels[exp_idx],
                "row": exp_row,
                "group_key": _exp_group_key(exp_row),
            }
        )

    group_map: Dict[Tuple[int, int, str, float | None], List[int]] = {}
    for item in exp_selected:
        group_map.setdefault(item["group_key"], []).append(item["exp_idx"])

    def _group_sort_key(key: Tuple[int, int, str, float | None]):
        accel_key = key[3]
        accel_sort = float("inf") if accel_key is None else accel_key
        return (accel_sort, key[0], key[1], key[2])

    group_order = sorted(group_map.keys(), key=_group_sort_key)

    x_positions: Dict[int, float] = {}
    group_grasp_x: Dict[Tuple[int, int, str, float | None], float] = {}
    group_grasp_row: Dict[Tuple[int, int, str, float | None], Dict | None] = {}
    x_cursor = 0.0
    group_gap = 0.8
    for group_key in group_order:
        exp_indices = group_map[group_key]
        for offset, exp_idx in enumerate(exp_indices):
            x_positions[exp_idx] = x_cursor + offset
        group_grasp_x[group_key] = x_cursor + len(exp_indices)
        first_exp_idx = exp_indices[0]
        exp_row = exp_selected[first_exp_idx]["row"]
        group_grasp_row[group_key] = _match_grasp_row(exp_row, grasp_rows)
        x_cursor += len(exp_indices) + 1 + group_gap

    all_x = list(x_positions.values()) + list(group_grasp_x.values())
    x_min = min(all_x) - 0.6
    x_max = max(all_x) + 0.6

    n_metrics = len(METRIC_SPECS)
    n_rows = 2 if args.two_rows else 1
    n_cols = int(np.ceil(n_metrics / n_rows))
    fig_w = max(16, 3.2 * n_cols)
    fig_h = 4.0 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(exp_names))]

    for metric_idx, (metric, label) in enumerate(METRIC_SPECS):
        row_idx, col_idx = divmod(metric_idx, n_cols)
        ax = axes[row_idx][col_idx]
        for exp_idx, exp_name in enumerate(exp_names):
            exp_row = exp_selected[exp_idx]["row"]
            (mean, std), (gmean, gstd) = _collect_metric_pair(
                exp_row,
                grasp_rows,
                metric,
                args.temporal_subset,
                args.temporal_region,
            )
            color = colors[exp_idx]
            if mean is not None:
                ax.errorbar(
                    [x_positions[exp_idx]],
                    [mean],
                    yerr=[std] if std is not None else None,
                    fmt="o",
                    color=color,
                    capsize=4,
                    markersize=6,
                )
        for group_key in group_order:
            grasp_row = group_grasp_row.get(group_key)
            gmean, gstd = _extract_temporal_metric(
                grasp_row, metric, args.temporal_subset, args.temporal_region
            )
            if gmean is None:
                continue
            ax.errorbar(
                [group_grasp_x[group_key]],
                [gmean],
                yerr=[gstd] if gstd is not None else None,
                fmt="^",
                color="black",
                capsize=4,
                markersize=6,
                markerfacecolor="black",
            )
        ax.set_title(label, fontsize=TITLE_SIZE)
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=TICK_SIZE)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_xlim(x_min, x_max)

    if args.two_rows:
        for row_idx in range(n_rows):
            axes[row_idx][0].set_ylabel("Metric value", fontsize=AXIS_LABEL_SIZE)
    else:
        axes[0][0].set_ylabel("Metric value", fontsize=AXIS_LABEL_SIZE)

    empty_axes = []
    for extra_idx in range(n_metrics, n_rows * n_cols):
        row_idx, col_idx = divmod(extra_idx, n_cols)
        axes[row_idx][col_idx].axis("off")
        empty_axes.append(axes[row_idx][col_idx])

    exp_handles = [
        Line2D([0], [0], color=colors[i], lw=2, marker="o", label=exp_labels[i])
        for i in range(len(exp_labels))
    ]
    method_handles = [
        Line2D([0], [0], color="black", lw=0, marker="o", label="BRISKNet"),
        Line2D([0], [0], color="black", lw=0, marker="^", markerfacecolor="black", label="GRASP"),
    ]

    legend_handles = exp_handles + method_handles
    legend_labels = [h.get_label() for h in legend_handles]
    legend_ncol = max(1, len(exp_handles))
    legend_fontsize = LEGEND_SIZE

    if not args.two_rows and len(exp_handles) == 5 and len(method_handles) == 2:
        exp_items = list(zip(exp_labels, exp_handles))

        def _pop_by_predicate(predicate):
            for idx, (label, handle) in enumerate(exp_items):
                if predicate(label.lower()):
                    return exp_items.pop(idx)
            return None

        baseline_item = _pop_by_predicate(lambda l: "baseline" in l)
        arrival_item = _pop_by_predicate(lambda l: "arrival" in l)
        rebin_item = _pop_by_predicate(lambda l: "rebin" in l)
        enh_item = _pop_by_predicate(lambda l: "enh" in l or "enhancement" in l)
        all_trans_item = _pop_by_predicate(lambda l: "transform" in l)

        if all(item is not None for item in (baseline_item, arrival_item, rebin_item, enh_item, all_trans_item)):
            top_handles = [
                baseline_item[1],
                arrival_item[1],
                enh_item[1],
                rebin_item[1],
            ]
            bottom_handles = [all_trans_item[1]] + method_handles
            legend_handles = [
                top_handles[0],
                bottom_handles[0],
                top_handles[1],
                bottom_handles[1],
                top_handles[2],
                bottom_handles[2],
                top_handles[3],
            ]
            legend_labels = [h.get_label() for h in legend_handles]
            legend_ncol = 4

    if args.two_rows and empty_axes:
        legend_ncol = 1

        def _fit_legend_in_axis(
            ax,
            handles,
            labels,
            ncol,
            max_fontsize,
            min_fontsize,
        ):
            legend = None
            for fontsize in range(max_fontsize, min_fontsize - 1, -1):
                if legend is not None:
                    legend.remove()
                legend = ax.legend(
                    handles,
                    labels,
                    loc="center",
                    ncol=ncol,
                    fontsize=fontsize,
                    handlelength=1.3,
                    handletextpad=0.5,
                    columnspacing=0.8,
                    labelspacing=0.6,
                    borderaxespad=0.0,
                    frameon=False,
                )
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                legend_bbox = legend.get_window_extent(renderer=renderer)
                ax_bbox = ax.get_window_extent(renderer=renderer)
                if (
                    legend_bbox.width <= ax_bbox.width * 0.98
                    and legend_bbox.height <= ax_bbox.height * 0.98
                ):
                    return legend
            return legend

        legend = _fit_legend_in_axis(
            empty_axes[0],
            legend_handles,
            legend_labels,
            legend_ncol,
            max(LEGEND_SIZE + 8, 18),
            8,
        )
    else:
        legend = fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=legend_ncol,
            fontsize=legend_fontsize,
            frameon=False,
        )

    if args.title:
        fig.suptitle(args.title, fontsize=TITLE_SIZE, y=1.02)
    bottom_pad = 0.06 if args.two_rows else 0.12
    fig.tight_layout(rect=[0.02, bottom_pad, 0.98, 0.98])
    fig.savefig(
        args.out,
        dpi=args.dpi,
        bbox_inches="tight",
        bbox_extra_artists=[legend],
        pad_inches=0.02,
    )


if __name__ == "__main__":
    main()
