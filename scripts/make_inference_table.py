"""Build a summary table from inference metrics JSON logs. Run: python3 -m inference.make_inference_table --help"""

import argparse
import json
import math
import os
from pathlib import Path
from collections import defaultdict
from typing import Iterable, List, Dict, Tuple

import yaml


DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "val_inference_logs.json"

SPATIAL_METRIC_SPECS: List[Tuple[str, str]] = [
    ("ssim", "SSIM"),
    ("psnr", "PSNR"),
    ("mse", "MSE"),
    ("lpips", "LPIPS"),
]

MC_METRIC_SPECS: List[Tuple[str, str]] = [
    ("dro_dc_mae", "DC MAE"),
    ("dro_dc_mse", "DC MSE"),
    ("raw_ssdu_nmse", "SSDU NMSE"),
]

METRIC_SPECS: Dict[str, List[Tuple[str, str]]] = {
    "spatial": [
        *SPATIAL_METRIC_SPECS,
    ],
    "mc": [
        *MC_METRIC_SPECS,
    ],
    "spatial_mc": [
        *SPATIAL_METRIC_SPECS,
        *MC_METRIC_SPECS,
    ],
    "temporal": [
        ("curve_corr", "$\\rho_{\\text{full}}$"),
        ("curve_mae", "\\emph{MAE}\\textsubscript{full}"),
        ("early_corr", "$\\rho_{\\text{early}}$"),
        ("early_mae", "\\emph{MAE}\\textsubscript{early}"),
        ("ttae_sec", "\\emph{t}\\textsubscript{arr} Err"),
        ("wash_in_slope_err", "\\emph{MAE}\\textsubscript{wash-in}"),
        ("iauc10_err", "\\emph{iAUC}\\textsubscript{10} Err"),
        ("peak_err", "\\emph{MAE}\\textsubscript{peak}"),
        ("ttpeak_err_sec", "\\emph{t}\\textsubscript{peak} Err"),
    ],
    "timing": [
        ("avg_inference_time", "Avg Inference Time (s)"),
    ],
}

METRIC_DIRECTIONS: Dict[str, str] = {
    "ssim": "up",
    "psnr": "up",
    "mse": "down",
    "lpips": "down",
    "dro_dc_mae": "down",
    "dro_dc_mse": "down",
    "raw_ssdu_nmse": "down",
    "curve_corr": "up",
    "curve_mae": "down",
    "early_corr": "up",
    "early_mae": "down",
    "ttae_sec": "down",
    "wash_in_slope_err": "down",
    "iauc10_err": "down",
    "peak_err": "down",
    "ttpeak_err_sec": "down",
    "avg_inference_time": "down",
}

METRIC_TABLE_FORMATS: Dict[str, str] = {
    "ssim": "1.3(1.3)",
    "psnr": "2.2(2.2)",
    "mse": "2.2(2.2)",
    "lpips": "1.3(1.3)",
    "dro_dc_mae": "1.3(1.3)",
    "dro_dc_mse": "1.3(1.3)",
    "raw_ssdu_nmse": "1.3(1.3)",
    "curve_corr": "1.2(1.2)",
    "curve_mae": "1.2(1.2)",
    "early_corr": "1.2(1.2)",
    "early_mae": "1.2(1.2)",
    "ttae_sec": "2.2(1.2)",
    "wash_in_slope_err": "1.2(1.2)",
    "iauc10_err": "2.1(2.1)",
    "peak_err": "2.2(2.2)",
    "ttpeak_err_sec": "2.2(2.2)",
    "avg_inference_time": "2.2(2.2)",
}

METRIC_PLOT_LABELS: Dict[str, str] = {
    "ssim": "SSIM",
    "psnr": "PSNR",
    "mse": "MSE",
    "lpips": "LPIPS",
    "dro_dc_mae": "DC MAE",
    "dro_dc_mse": "DC MSE",
    "raw_ssdu_nmse": "SSDU NMSE",
    "curve_corr": r"$\rho_{\mathrm{full}}$",
    "curve_mae": r"$\mathrm{MAE}_{\mathrm{full}}$",
    "early_corr": r"$\rho_{\mathrm{early}}$",
    "early_mae": r"$\mathrm{MAE}_{\mathrm{early}}$",
    "ttae_sec": r"$t_{\mathrm{arr}}\ \mathrm{Err}$",
    "wash_in_slope_err": r"$\mathrm{MAE}_{\mathrm{wash-in}}$",
    "iauc10_err": r"$\mathrm{iAUC}_{10}\ \mathrm{Err}$",
    "peak_err": r"$\mathrm{MAE}_{\mathrm{peak}}$",
    "ttpeak_err_sec": r"$t_{\mathrm{peak}}\ \mathrm{Err}$",
    "avg_inference_time": "Avg Inference Time (s)",
}

BAR_CHART_STYLE = {
    "target_square_axis_size": 3.8,
    "bar_width": 0.36,
    "title_fontsize": 30,
    "label_fontsize": 28,
    "tick_fontsize": 24,
    "legend_fontsize": 24,
    "bar_edgewidth": 1.8,
    "spine_linewidth": 2.0,
    "tick_width": 2.0,
    "std_linewidth": 3.0,
    "std_cap_width_fraction": 0.35,
    "subplot_left": 0.085,
    "subplot_right": 0.84,
    "subplot_right_no_side_legend": 0.97,
    "subplot_bottom": 0.16,
    "subplot_bottom_with_bottom_legend": 0.36,
    "subplot_top": 0.92,
    "subplot_wspace": 0.42,
    "subplot_hspace": 0.52,
    "legend_x_anchor": 0.86,
    "legend_y_anchor": 0.5,
    "legend_bottom_y_anchor": 0.035,
    "square_axis_box_aspect": 0.9,
}

SPATIAL_METRICS = [metric for metric, _ in METRIC_SPECS["spatial"]]
MC_METRICS = [metric for metric, _ in METRIC_SPECS["mc"]]
SPATIAL_MC_METRICS = [metric for metric, _ in METRIC_SPECS["spatial_mc"]]
TEMPORAL_METRICS = [metric for metric, _ in METRIC_SPECS["temporal"]]


def _normalize_metric_name(metric_name: str) -> str:
    return metric_name.strip().lower().replace("-", "_").replace(" ", "_")


def _build_metric_aliases() -> Dict[str, Dict[str, str]]:
    aliases: Dict[str, Dict[str, str]] = {}
    for metric_type, specs in METRIC_SPECS.items():
        alias_map: Dict[str, str] = {}
        for metric, _ in specs:
            alias_map[_normalize_metric_name(metric)] = metric
        aliases[metric_type] = alias_map

    aliases["temporal"].update({
        "full_curve_corr": "curve_corr",
        "full_curve_correlation": "curve_corr",
        "curve_correlation": "curve_corr",
        "full_curve_mae": "curve_mae",
        "early_curve_corr": "early_corr",
        "early_curve_correlation": "early_corr",
        "early_curve_mae": "early_mae",
        "arrival_time_error": "ttae_sec",
        "t_arr_error": "ttae_sec",
        "wash_in_mae": "wash_in_slope_err",
        "wash_in_error": "wash_in_slope_err",
        "washin_mae": "wash_in_slope_err",
        "iauc_error": "iauc10_err",
        "iauc10_error": "iauc10_err",
        "iauc_err": "iauc10_err",
        "peak_time_error": "ttpeak_err_sec",
    })
    for metric_type in ("mc", "spatial_mc"):
        aliases[metric_type].update({
            "dc_mae": "dro_dc_mae",
            "dc_mse": "dro_dc_mse",
            "dro_mae": "dro_dc_mae",
            "dro_mse": "dro_dc_mse",
            "data_consistency_mae": "dro_dc_mae",
            "data_consistency_mse": "dro_dc_mse",
            "raw_nmse": "raw_ssdu_nmse",
            "ssdu_nmse": "raw_ssdu_nmse",
            "raw_grasp_ssdu_nmse": "raw_ssdu_nmse",
        })
    return aliases


METRIC_ALIASES = _build_metric_aliases()


def _default_metrics(metric_type: str) -> List[str]:
    return [metric for metric, _ in METRIC_SPECS[metric_type]]


def _resolve_metrics(metric_type: str, metrics_value: str, temporal_metrics_value: str) -> List[str]:
    requested = _parse_list(metrics_value)
    if not requested and metric_type == "temporal":
        requested = _parse_list(temporal_metrics_value)
    if not requested:
        return _default_metrics(metric_type)

    selected = []
    alias_map = METRIC_ALIASES[metric_type]
    for metric in requested:
        canonical = alias_map.get(_normalize_metric_name(metric))
        if canonical is None:
            valid_metrics = ", ".join(_default_metrics(metric_type))
            raise ValueError(
                f"Unknown {metric_type} metric '{metric}'. Valid metrics are: {valid_metrics}."
            )
        if canonical not in selected:
            selected.append(canonical)
    return selected


def _parse_list(value: str) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_config_cols(value: str) -> List[Tuple[str, str]]:
    cols = []
    for item in _parse_list(value):
        if ":" in item:
            header, path = item.split(":", 1)
            header = header.strip()
            path = path.strip()
        else:
            path = item.strip()
            header = path.split(".")[-1] if path else ""
        if path:
            cols.append((header, path))
    return cols


def _to_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value):
    value = _to_float(value)
    if value is None:
        return None
    return int(value)


def _safe_filename_fragment(value: str) -> str:
    fragment = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))
    fragment = fragment.strip("_")
    return fragment or "metric"


def _metric_plot_label(metric_type: str, metric: str) -> str:
    if metric in METRIC_PLOT_LABELS:
        return METRIC_PLOT_LABELS[metric]
    return dict(METRIC_SPECS.get(metric_type, [])).get(metric, metric)


def _metric_title(metric_type: str, metric: str) -> str:
    metric_name = _metric_plot_label(metric_type, metric)
    direction = METRIC_DIRECTIONS.get(metric)
    if direction == "up":
        return f"{metric_name} ↑"
    if direction == "down":
        return f"{metric_name} ↓"
    return metric_name


def _metric_group_name(metric_type: str, metric: str) -> str:
    if metric in SPATIAL_METRICS:
        return "spatial"
    if metric in MC_METRICS:
        return "dc"
    if metric in TEMPORAL_METRICS:
        return "temporal"
    if metric_type == "mc":
        return "dc"
    return metric_type


def _group_metrics_for_panels(metric_type: str, metrics: List[str]) -> List[Tuple[str, List[str]]]:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for metric in metrics:
        grouped[_metric_group_name(metric_type, metric)].append(metric)

    panel_order = ["spatial", "dc", "temporal", "timing"]
    panels: List[Tuple[str, List[str]]] = []
    for panel_name in panel_order:
        metric_list = grouped.get(panel_name, [])
        if metric_list:
            panels.append((panel_name, metric_list))
    for panel_name, metric_list in grouped.items():
        if panel_name not in panel_order:
            panels.append((panel_name, metric_list))
    return panels


def _panel_figure_size(
    nrows: int,
    ncols: int,
    use_temporal_panel_legend: bool,
    use_bottom_horizontal_legend: bool,
) -> Tuple[float, float]:
    left = BAR_CHART_STYLE["subplot_left"]
    right = (
        BAR_CHART_STYLE["subplot_right_no_side_legend"]
        if (use_temporal_panel_legend or use_bottom_horizontal_legend)
        else BAR_CHART_STYLE["subplot_right"]
    )
    bottom = (
        BAR_CHART_STYLE["subplot_bottom_with_bottom_legend"]
        if use_bottom_horizontal_legend
        else BAR_CHART_STYLE["subplot_bottom"]
    )
    top = BAR_CHART_STYLE["subplot_top"]
    wspace = BAR_CHART_STYLE["subplot_wspace"]
    hspace = BAR_CHART_STYLE["subplot_hspace"]
    square_axis_size = BAR_CHART_STYLE["target_square_axis_size"]

    width_slots = ncols + max(0, ncols - 1) * wspace
    height_slots = nrows + max(0, nrows - 1) * hspace
    width_frac = max(1e-6, right - left)
    height_frac = max(1e-6, top - bottom)

    fig_width = square_axis_size * width_slots / width_frac
    fig_height = square_axis_size * height_slots / height_frac
    return fig_width, fig_height


def _format_mean_std(mean, std, decimals: int) -> str:
    if mean is None:
        return ""
    if std is None:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f}({std:.{decimals}f})"


def _format_mean_std_mri_journal(mean, std, decimals: int) -> str:
    if mean is None:
        return "NA"
    if std is None:
        return f"${mean:.{decimals}f}$"
    return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"


def _extract_mean_std_flat(row: Dict[str, str], prefix: str) -> Tuple[float | None, float | None]:
    return _to_float(row.get(f"{prefix}_mean")), _to_float(row.get(f"{prefix}_std"))


def _format_one_decimal(value) -> str:
    value = _to_float(value)
    if value is None:
        return ""
    return f"{value:.1f}"


def _format_int(value) -> str:
    value = _to_float(value)
    if value is None:
        return ""
    return str(int(value))


def _extract_timing_stats(row: Dict[str, str]) -> Tuple[float | None, float | None]:
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


def _extract_timing_stats_flat(row: Dict[str, str], prefix: str) -> Tuple[float | None, float | None]:
    mean = _to_float(row.get(f"{prefix}_avg_inference_time"))
    if mean is None:
        mean = _to_float(row.get(f"{prefix}_avg_grasp_recon_time"))
    if mean is None:
        mean = _to_float(row.get(f"{prefix}_avg_recon_time"))

    std = _to_float(row.get(f"{prefix}_std_inference_time"))
    if std is None:
        std = _to_float(row.get(f"{prefix}_std_grasp_recon_time"))
    if std is None:
        std = _to_float(row.get(f"{prefix}_std_recon_time"))
    return mean, std


def _load_config(exp_name: str, exp_base_dirs: List[str], cache: Dict[str, Dict]) -> Dict | None:
    if not exp_name:
        return None
    if exp_name in cache:
        return cache[exp_name]
    cfg = None
    for base_dir in exp_base_dirs:
        config_path = os.path.join(base_dir, exp_name, "config.yaml")
        if not os.path.exists(config_path):
            continue
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        break
    cache[exp_name] = cfg
    return cfg


def _get_config_value(cfg: Dict | None, path: str):
    if not cfg or not path:
        return None
    current = cfg
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit():
            idx = int(part)
            if idx < 0 or idx >= len(current):
                return None
            current = current[idx]
        else:
            return None
    return current


def _format_config_value(value, missing_default: str = "False") -> str:
    if value is None:
        return missing_default
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _extract_mean_std_structured(node: Dict[str, str]) -> Tuple[float | None, float | None]:
    if not isinstance(node, dict):
        return None, None
    return _to_float(node.get("mean")), _to_float(node.get("std"))


def _load_rows(log_path: str) -> List[Dict[str, str]]:
    with open(log_path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError("Log file must contain a list of records.")
    return payload


def _group_rows(rows: Iterable[Dict[str, str]]):
    exp_rows = [
        r for r in rows
        if r.get("type") == "BRISKNet" or (r.get("row_type") or "exp") == "exp"
    ]
    grasp_rows = [
        r for r in rows
        if r.get("type") == "GRASP" or r.get("row_type") == "grasp_agg"
    ]
    grasp_index = defaultdict(list)
    for row in grasp_rows:
        key = _match_key(row)
        grasp_index[key].append(row)
    return exp_rows, grasp_index


def _row_key(row: Dict[str, str]) -> Tuple[float, int, int, str]:
    accel = _to_float(row.get("acceleration_factor") or row.get("acceleration")) or 0.0
    spf = int(float(row.get("spokes_per_frame") or 0))
    frames = int(float(row.get("num_frames") or 0))
    noise = row.get("dro_noise_level") or row.get("DRO_noise_level") or ""
    return accel, spf, frames, noise


def _match_key(row: Dict[str, str]) -> Tuple[str, str, str, str]:
    accel = _to_float(row.get("acceleration_factor") or row.get("acceleration"))
    accel_key = "" if accel is None else f"{accel:.3f}"
    return (
        str(row.get("spokes_per_frame") or ""),
        str(row.get("num_frames") or ""),
        str(row.get("dro_noise_level") or row.get("DRO_noise_level") or ""),
        accel_key,
    )


def _format_row(values: List[str]) -> str:
    return " & ".join(values) + " \\\\"


def _method_name(row: Dict[str, str]) -> str:
    method_name = row.get("method_name") or row.get("method")
    if method_name:
        return str(method_name)
    exp_name = str(row.get("exp_name") or "").lower()
    if exp_name.startswith("ssdu"):
        return "SSDU"
    return str(row.get("type") or "BRISKNet")


def _method_sort_key(row: Dict[str, str]) -> Tuple[int, str]:
    method_name = _method_name(row)
    method_order = {"BRISKNet": 0, "SSDU": 1}
    return method_order.get(method_name, 2), method_name


def _metric_columns(metric_type: str, metrics: List[str], include_metric_arrows: bool) -> List[str]:
    metric_label_map = dict(METRIC_SPECS[metric_type])
    if not include_metric_arrows:
        return [metric_label_map[metric] for metric in metrics]

    out = []
    for metric in metrics:
        label = metric_label_map[metric]
        direction = METRIC_DIRECTIONS.get(metric)
        if direction == "up":
            out.append(f"{label} $\\uparrow$")
        elif direction == "down":
            out.append(f"{label} $\\downarrow$")
        else:
            out.append(label)
    return out


def _metric_column_spec(metrics: List[str]) -> List[str]:
    specs = []
    for metric in metrics:
        table_fmt = METRIC_TABLE_FORMATS.get(metric, "3.3(3.3)")
        specs.append(f"S[table-format={table_fmt}]")
    return specs


def _render_header_cells(left_headers: List[str], metric_headers: List[str]) -> List[str]:
    cells = list(left_headers)
    for metric_header in metric_headers:
        cells.append(f"\\multicolumn{{1}}{{c|}}{{{metric_header}}}")
    return cells


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


def _extract_metric_stats(row: Dict[str, str], model_prefix: str, metric_type: str, metric: str,
                          temporal_subset: str, temporal_region: str) -> Tuple[float | None, float | None]:
    if row.get("type") in ("BRISKNet", "GRASP"):
        if metric_type == "spatial":
            spatial = row.get("spatial_metrics", {})
            return _to_float(spatial.get(f"{metric}_mean")), _to_float(spatial.get(f"{metric}_stddev"))
        if metric_type == "mc":
            dc = row.get("dc_metrics", {})
            metric_key = metric
            if metric == "raw_ssdu_nmse" and row.get("type") == "GRASP":
                metric_key = "raw_grasp_ssdu_nmse"
            return _to_float(dc.get(f"{metric_key}_mean")), _to_float(dc.get(f"{metric_key}_stddev"))
        if metric_type == "spatial_mc":
            spatial = row.get("spatial_metrics", {})
            dc = row.get("dc_metrics", {})
            if metric in SPATIAL_METRICS:
                return _to_float(spatial.get(f"{metric}_mean")), _to_float(spatial.get(f"{metric}_stddev"))
            if metric in MC_METRICS:
                metric_key = metric
                if metric == "raw_ssdu_nmse" and row.get("type") == "GRASP":
                    metric_key = "raw_grasp_ssdu_nmse"
                return _to_float(dc.get(f"{metric_key}_mean")), _to_float(dc.get(f"{metric_key}_stddev"))
            return None, None
        if metric_type == "temporal":
            temporal = row.get("temporal_metrics", {})
            block = temporal.get(_temporal_block_name(temporal_subset, temporal_region), {})
            return _to_float(block.get(f"{metric}_mean")), _to_float(block.get(f"{metric}_stddev"))
        if metric_type == "timing":
            if metric != "avg_inference_time":
                return None, None
            return _extract_timing_stats(row)
        raise ValueError(f"Unknown metric_type {metric_type}")

    if metric_type == "spatial":
        return _extract_mean_std_flat(row, f"{model_prefix}_{metric}")
    if metric_type == "mc":
        metric_key = metric
        if metric == "raw_ssdu_nmse" and model_prefix == "grasp":
            metric_key = "raw_grasp_ssdu_nmse"
        return _extract_mean_std_flat(row, f"{model_prefix}_{metric_key}")
    if metric_type == "spatial_mc":
        if metric in SPATIAL_METRICS:
            return _extract_mean_std_flat(row, f"{model_prefix}_{metric}")
        if metric in MC_METRICS:
            metric_key = metric
            if metric == "raw_ssdu_nmse" and model_prefix == "grasp":
                metric_key = "raw_grasp_ssdu_nmse"
            return _extract_mean_std_flat(row, f"{model_prefix}_{metric_key}")
        return None, None
    if metric_type == "temporal":
        prefix = "" if temporal_region == "malignant" else "benign_"
        key = f"{prefix}{model_prefix}_{temporal_subset}_{metric}"
        return _extract_mean_std_flat(row, key)
    if metric_type == "timing":
        if metric != "avg_inference_time":
            return None, None
        return _extract_timing_stats_flat(row, model_prefix)
    raise ValueError(f"Unknown metric_type {metric_type}")


def _collect_metric_stats(row: Dict[str, str], model_prefix: str, metric_type: str, metrics: List[str],
                          temporal_subset: str, temporal_region: str) -> List[Tuple[float | None, float | None]]:
    return [
        _extract_metric_stats(row, model_prefix, metric_type, metric, temporal_subset, temporal_region)
        for metric in metrics
    ]


def _should_highlight_green(mean: float | None, peer_mean: float | None, peer_std: float | None,
                            std_threshold: float) -> bool:
    if mean is None or peer_mean is None or peer_std is None:
        return False
    return mean < (peer_mean - std_threshold * peer_std)


def _should_highlight_mri_journal(mean: float | None, grasp_mean: float | None,
                                  grasp_std: float | None, std_threshold: float) -> bool:
    if mean is None or grasp_mean is None or grasp_std is None:
        return False
    difference = abs(mean - grasp_mean)
    threshold = std_threshold * grasp_std
    return difference > threshold and not math.isclose(
        difference,
        threshold,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def _format_mri_journal_metric_cells(
    stats: List[Tuple[float | None, float | None]],
    grasp_stats: List[Tuple[float | None, float | None]] | None,
    decimals: int,
    std_threshold: float,
) -> List[str]:
    cells = []
    for idx, (mean, std) in enumerate(stats):
        cell = _format_mean_std_mri_journal(mean, std, decimals)
        if not cell:
            cells.append(cell)
            continue
        if grasp_stats is not None:
            grasp_mean, grasp_std = grasp_stats[idx]
            if _should_highlight_mri_journal(mean, grasp_mean, grasp_std, std_threshold):
                cell = f"\\cellcolor{{red}} {cell}"
        cells.append(cell)
    return cells


def _format_metric_cells(metrics: List[str], stats: List[Tuple[float | None, float | None]],
                         peer_stats: List[Tuple[float | None, float | None]] | None,
                         decimals: int, highlight_green_cells: bool,
                         green_cell_std_threshold: float) -> List[str]:
    cells = []
    for idx, metric in enumerate(metrics):
        mean, std = stats[idx]
        cell = _format_mean_std(mean, std, decimals)
        if not cell:
            cells.append(cell)
            continue
        if highlight_green_cells and peer_stats is not None:
            peer_mean, peer_std = peer_stats[idx]
            if _should_highlight_green(mean, peer_mean, peer_std, green_cell_std_threshold):
                table_fmt = METRIC_TABLE_FORMATS.get(metric, "3.3(3.3)")
                cell = (
                    f"\\multicolumn{{1}}{{|>{{\\cellcolor{{green}}}}S[table-format={table_fmt}]|}}"
                    f"{{{cell}}}"
                )
        cells.append(cell)
    return cells


def _select_grasp_row(grasp_rows: List[Dict[str, str]] | None, metric_type: str) -> Dict[str, str] | None:
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


def _emit_mri_journal_table(
    rows: List[Dict[str, str]],
    grasp_index: Dict[Tuple[str, str, str, str], List[Dict[str, str]]],
    metric_type: str,
    metrics: List[str],
    temporal_subset: str,
    temporal_region: str,
    decimals: int,
    caption: str,
    label: str,
    config_cols: List[Tuple[str, str]],
    exp_base_dirs: List[str],
    config_cache: Dict[str, Dict],
    include_af: bool,
    include_spf: bool,
    include_temporal_resolution: bool,
    include_metric_arrows: bool,
    red_cell_std_threshold: float,
) -> str:
    timing_cols = []
    if include_af:
        timing_cols.append("AF")
    if include_spf:
        timing_cols.append("SPF")
    if include_temporal_resolution:
        timing_cols.append("TR")
    left_headers = ["Method"] + [col for col, _ in config_cols] + timing_cols
    metric_headers = _metric_columns(metric_type, metrics, include_metric_arrows)
    header_cols = left_headers + metric_headers

    lines = [
        "\\begin{table}%[]",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular*}}{{\\tblwidth}}{{@{{}}{'L' * len(header_cols)}@{{}}}}",
        "\\toprule",
        _format_row(header_cols),
        "\\midrule",
    ]

    sorted_rows = sorted(rows, key=_row_key)
    grouped = []
    group_map: Dict[Tuple[str, str, str, str], List[Dict[str, str]]] = {}
    for row in sorted_rows:
        match_key = _match_key(row)
        if match_key not in group_map:
            group_map[match_key] = []
            grouped.append(match_key)
        group_map[match_key].append(row)

    for match_key in grouped:
        group_rows = sorted(group_map[match_key], key=_method_sort_key)
        grasp_row = _select_grasp_row(grasp_index.get(match_key), metric_type)
        grasp_stats = None
        if grasp_row is not None:
            grasp_stats = _collect_metric_stats(
                grasp_row,
                "grasp",
                metric_type,
                metrics,
                temporal_subset,
                temporal_region,
            )

        for row in group_rows:
            exp_name = row.get("exp_name") or ""
            if config_cols:
                cfg = _load_config(exp_name, exp_base_dirs, config_cache)
                config_vals = [
                    _format_config_value(
                        _get_config_value(cfg, path),
                        missing_default="False",
                    )
                    for _, path in config_cols
                ]
            else:
                config_vals = []

            timing_vals = []
            if include_af:
                timing_vals.append(
                    _format_one_decimal(row.get("acceleration_factor") or row.get("acceleration"))
                )
            if include_spf:
                timing_vals.append(_format_int(row.get("spokes_per_frame")))
            if include_temporal_resolution:
                timing_vals.append(_format_one_decimal(row.get("seconds_per_frame")))

            stats = _collect_metric_stats(
                row,
                "dl",
                metric_type,
                metrics,
                temporal_subset,
                temporal_region,
            )
            metric_vals = _format_mri_journal_metric_cells(
                stats,
                grasp_stats,
                decimals,
                red_cell_std_threshold,
            )
            lines.append(
                _format_row([_method_name(row)] + config_vals + timing_vals + metric_vals)
            )

        if grasp_row is not None:
            ref_row = group_rows[0]
            timing_vals = []
            if include_af:
                timing_vals.append(
                    _format_one_decimal(
                        ref_row.get("acceleration_factor")
                        or ref_row.get("acceleration")
                        or grasp_row.get("acceleration")
                    )
                )
            if include_spf:
                timing_vals.append(
                    _format_int(
                        ref_row.get("spokes_per_frame")
                        or grasp_row.get("spokes_per_frame")
                    )
                )
            if include_temporal_resolution:
                timing_vals.append(
                    _format_one_decimal(
                        ref_row.get("seconds_per_frame")
                        or grasp_row.get("seconds_per_frame")
                    )
                )
            grasp_metric_vals = [
                _format_mean_std_mri_journal(mean, std, decimals)
                for mean, std in (grasp_stats or [])
            ]
            lines.append(
                _format_row(
                    ["GRASP"] + (["NA"] * len(config_cols)) + timing_vals + grasp_metric_vals
                )
            )

    lines.extend([
        "\\bottomrule",
        "\\end{tabular*}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def _emit_table(rows: List[Dict[str, str]], grasp_index: Dict[Tuple[str, str, str, str], List[Dict[str, str]]],
                metric_type: str, metrics: List[str], temporal_subset: str, temporal_region: str,
                decimals: int, caption: str, label: str, config_cols: List[Tuple[str, str]],
                exp_base_dirs: List[str], config_cache: Dict[str, Dict],
                include_af: bool, include_spf: bool, include_temporal_resolution: bool,
                include_metric_arrows: bool, highlight_green_cells: bool,
                green_cell_std_threshold: float, format_mri_journal: bool = False) -> str:
    if format_mri_journal:
        return _emit_mri_journal_table(
            rows,
            grasp_index,
            metric_type,
            metrics,
            temporal_subset,
            temporal_region,
            decimals,
            caption,
            label,
            config_cols,
            exp_base_dirs,
            config_cache,
            include_af,
            include_spf,
            include_temporal_resolution,
            include_metric_arrows,
            green_cell_std_threshold,
        )

    timing_cols = []
    if include_af:
        timing_cols.append("AF")
    if include_spf:
        timing_cols.append("SPF")
    if include_temporal_resolution:
        timing_cols.append("TR")
    if config_cols:
        left_headers = ["Method"] + [col for col, _ in config_cols] + timing_cols
    else:
        left_headers = ["Method"] + timing_cols
    metric_headers = _metric_columns(metric_type, metrics, include_metric_arrows)
    header_cols = _render_header_cells(left_headers, metric_headers)

    left_col_spec = ["l"] + (["c"] * (len(left_headers) - 1))
    metric_col_spec = _metric_column_spec(metrics)
    col_spec = "|" + "|".join(left_col_spec + metric_col_spec) + "|"

    lines = []
    lines.append("\\begin{table}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\centering")
    lines.append("\\begin{adjustbox}{max width=\\textwidth}")
    lines.append("% \\fontsize{8}{9.6}\\selectfont")
    lines.append("")
    lines.append("\\sisetup{")
    lines.append("  separate-uncertainty = true,")
    lines.append("  uncertainty-separator = {\\,\\pm\\,},")
    lines.append("  retain-zero-uncertainty = true")
    lines.append("}")
    lines.append("")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")
    lines.append(_format_row(header_cols))
    lines.append("\\hline")

    sorted_rows = sorted(rows, key=_row_key)
    grouped = []
    group_map: Dict[Tuple[str, str, str, str], List[Dict[str, str]]] = {}
    for row in sorted_rows:
        match_key = _match_key(row)
        if match_key not in group_map:
            group_map[match_key] = []
            grouped.append(match_key)
        group_map[match_key].append(row)

    for group_idx, match_key in enumerate(grouped):
        group_rows = group_map[match_key]
        grasp_row = _select_grasp_row(grasp_index.get(match_key), metric_type)
        grasp_stats = None
        if grasp_row is not None:
            grasp_stats = _collect_metric_stats(
                grasp_row,
                "grasp",
                metric_type,
                metrics,
                temporal_subset,
                temporal_region,
            )
        shade_group = (group_idx % 2 == 0)

        for row in group_rows:
            accel = _format_one_decimal(row.get("acceleration_factor") or row.get("acceleration"))
            spf = _format_int(row.get("spokes_per_frame"))
            temporal_resolution = _format_one_decimal(row.get("seconds_per_frame"))
            exp_name = row.get("exp_name") or ""

            if config_cols:
                cfg = _load_config(exp_name, exp_base_dirs, config_cache)
                config_vals = [
                    _format_config_value(
                        _get_config_value(cfg, path),
                        missing_default="False",
                    )
                    for _, path in config_cols
                ]
            else:
                config_vals = []

            brisk_stats = _collect_metric_stats(
                row,
                "dl",
                metric_type,
                metrics,
                temporal_subset,
                temporal_region,
            )
            brisk_vals = _format_metric_cells(
                metrics,
                brisk_stats,
                grasp_stats,
                decimals,
                highlight_green_cells=highlight_green_cells,
                green_cell_std_threshold=green_cell_std_threshold,
            )
            timing_vals = []
            if include_af:
                timing_vals.append(str(accel))
            if include_spf:
                timing_vals.append(str(spf))
            if include_temporal_resolution:
                timing_vals.append(str(temporal_resolution))
            if shade_group:
                lines.append("\\rowcolor{groupbg}")
            lines.append(_format_row(["BRISKNet"] + config_vals + timing_vals + brisk_vals))

        if grasp_row is not None:
            empty_config_vals = ["NA"] * len(config_cols)
            ref_row = group_rows[0]
            grasp_accel = _format_one_decimal(
                ref_row.get("acceleration_factor")
                or ref_row.get("acceleration")
                or grasp_row.get("acceleration")
            )
            grasp_spf = _format_int(ref_row.get("spokes_per_frame") or grasp_row.get("spokes_per_frame"))
            grasp_temporal_resolution = _format_one_decimal(
                ref_row.get("seconds_per_frame") or grasp_row.get("seconds_per_frame")
            )
            if grasp_stats is None:
                grasp_stats = _collect_metric_stats(
                    grasp_row,
                    "grasp",
                    metric_type,
                    metrics,
                    temporal_subset,
                    temporal_region,
                )
            ref_brisk_stats = _collect_metric_stats(
                ref_row,
                "dl",
                metric_type,
                metrics,
                temporal_subset,
                temporal_region,
            )
            grasp_vals = _format_metric_cells(
                metrics,
                grasp_stats,
                ref_brisk_stats,
                decimals,
                highlight_green_cells=highlight_green_cells,
                green_cell_std_threshold=green_cell_std_threshold,
            )
            timing_vals = []
            if include_af:
                timing_vals.append(str(grasp_accel))
            if include_spf:
                timing_vals.append(str(grasp_spf))
            if include_temporal_resolution:
                timing_vals.append(str(grasp_temporal_resolution))
            if shade_group:
                lines.append("\\rowcolor{groupbg}")
            lines.append(_format_row(["GRASP"] + empty_config_vals + timing_vals + grasp_vals))
            lines.append("")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def _aggregate_stats(stats: List[Tuple[float | None, float | None]]) -> Tuple[float | None, float | None]:
    valid = [(mean, std) for mean, std in stats if mean is not None]
    if not valid:
        return None, None
    means = [mean for mean, _ in valid]
    mean_val = sum(means) / len(means)
    second_moment = 0.0
    for mean, std in valid:
        std_val = 0.0 if std is None else max(0.0, std)
        second_moment += (std_val * std_val) + (mean * mean)
    second_moment /= len(valid)
    var_val = max(0.0, second_moment - (mean_val * mean_val))
    return mean_val, math.sqrt(var_val)


def _collect_metric_bar_data(
    rows: List[Dict[str, str]],
    grasp_index: Dict[Tuple[str, str, str, str], List[Dict[str, str]]],
    metric_type: str,
    metrics: List[str],
    temporal_subset: str,
    temporal_region: str,
) -> Dict[str, Dict[str, Dict[int, List[Tuple[float | None, float | None]]]]]:
    data = {
        metric: {"BRISKNet": defaultdict(list), "GRASP": defaultdict(list)}
        for metric in metrics
    }
    seen_grasp_keys = set()

    for row in sorted(rows, key=_row_key):
        spf = _to_int(row.get("spokes_per_frame"))
        if spf is None:
            continue

        brisk_stats = _collect_metric_stats(
            row,
            "dl",
            metric_type,
            metrics,
            temporal_subset,
            temporal_region,
        )
        for idx, metric in enumerate(metrics):
            data[metric]["BRISKNet"][spf].append(brisk_stats[idx])

        match_key = _match_key(row)
        if match_key in seen_grasp_keys:
            continue
        grasp_row = _select_grasp_row(grasp_index.get(match_key), metric_type)
        if grasp_row is None:
            continue
        grasp_stats = _collect_metric_stats(
            grasp_row,
            "grasp",
            metric_type,
            metrics,
            temporal_subset,
            temporal_region,
        )
        for idx, metric in enumerate(metrics):
            data[metric]["GRASP"][spf].append(grasp_stats[idx])
        seen_grasp_keys.add(match_key)
    return data


def _save_metric_bar_charts(
    rows: List[Dict[str, str]],
    grasp_index: Dict[Tuple[str, str, str, str], List[Dict[str, str]]],
    metric_type: str,
    metrics: List[str],
    temporal_subset: str,
    temporal_region: str,
    output_dir: str,
    dpi: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except Exception as exc:
        raise RuntimeError(
            "Saving metric bar charts requires matplotlib. Install matplotlib or disable --save_metric_bar_charts."
        ) from exc

    metric_data = _collect_metric_bar_data(
        rows,
        grasp_index,
        metric_type,
        metrics,
        temporal_subset,
        temporal_region,
    )

    os.makedirs(output_dir, exist_ok=True)
    colors = {"BRISKNet": "#1f77b4", "GRASP": "#ff7f0e"}
    bar_width = BAR_CHART_STYLE["bar_width"]
    std_cap_half_width = bar_width * BAR_CHART_STYLE["std_cap_width_fraction"] * 0.5
    panel_groups = _group_metrics_for_panels(metric_type, metrics)
    legend_handles = [
        Patch(facecolor=colors["BRISKNet"], edgecolor="black", label="BRISKNet"),
        Patch(facecolor=colors["GRASP"], edgecolor="black", label="GRASP"),
    ]

    def _draw_std_range_lines(ax, x_positions: List[float], means: List[float], stds: List[float]) -> None:
        for x_pos, mean, std in zip(x_positions, means, stds):
            if math.isnan(mean) or math.isnan(std) or std < 0:
                continue
            lower = mean - std
            upper = mean + std
            ax.vlines(
                x_pos,
                lower,
                upper,
                colors="black",
                linewidth=BAR_CHART_STYLE["std_linewidth"],
                zorder=4,
            )
            ax.hlines(
                [lower, upper],
                x_pos - std_cap_half_width,
                x_pos + std_cap_half_width,
                colors="black",
                linewidth=BAR_CHART_STYLE["std_linewidth"],
                zorder=4,
            )

    for panel_name, panel_metrics in panel_groups:
        num_metrics = len(panel_metrics)
        if num_metrics == 0:
            continue

        use_temporal_panel_legend = panel_name == "temporal"
        use_bottom_horizontal_legend = panel_name in {"dc", "spatial"}
        legend_slot_idx = None
        if use_temporal_panel_legend:
            legend_slot_idx = 5 if num_metrics >= 6 else num_metrics

        ncols = min(3, max(1, num_metrics))
        required_slots = num_metrics + (1 if use_temporal_panel_legend else 0)
        nrows = math.ceil(required_slots / ncols)
        fig_width, fig_height = _panel_figure_size(
            nrows,
            ncols,
            use_temporal_panel_legend,
            use_bottom_horizontal_legend,
        )
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(fig_width, fig_height),
            squeeze=False,
        )
        flat_axes = list(axes.flat)
        plot_slot_indices = [
            slot_idx
            for slot_idx in range(len(flat_axes))
            if legend_slot_idx is None or slot_idx != legend_slot_idx
        ]
        assigned_plot_slots = set(plot_slot_indices[:num_metrics])
        any_visible = False

        for metric, slot_idx in zip(panel_metrics, plot_slot_indices):
            ax = flat_axes[slot_idx]
            model_data = metric_data.get(metric, {})
            spf_values = sorted(
                set(model_data.get("BRISKNet", {}).keys()) | set(model_data.get("GRASP", {}).keys())
            )
            if not spf_values:
                ax.set_visible(False)
                continue

            any_visible = True
            ax.set_box_aspect(BAR_CHART_STYLE["square_axis_box_aspect"])
            x_centers = list(range(len(spf_values)))
            x_brisk = [x - (bar_width / 2.0) for x in x_centers]
            x_grasp = [x + (bar_width / 2.0) for x in x_centers]

            brisk_agg = [_aggregate_stats(model_data["BRISKNet"].get(spf, [])) for spf in spf_values]
            grasp_agg = [_aggregate_stats(model_data["GRASP"].get(spf, [])) for spf in spf_values]

            brisk_means = [float("nan") if mean is None else mean for mean, _ in brisk_agg]
            brisk_stds = [0.0 if std is None else std for _, std in brisk_agg]
            grasp_means = [float("nan") if mean is None else mean for mean, _ in grasp_agg]
            grasp_stds = [0.0 if std is None else std for _, std in grasp_agg]

            ax.bar(
                x_brisk,
                brisk_means,
                width=bar_width,
                color=colors["BRISKNet"],
                edgecolor="black",
                linewidth=BAR_CHART_STYLE["bar_edgewidth"],
                zorder=2,
            )
            ax.bar(
                x_grasp,
                grasp_means,
                width=bar_width,
                color=colors["GRASP"],
                edgecolor="black",
                linewidth=BAR_CHART_STYLE["bar_edgewidth"],
                zorder=2,
            )
            _draw_std_range_lines(ax, x_brisk, brisk_means, brisk_stds)
            _draw_std_range_lines(ax, x_grasp, grasp_means, grasp_stds)

            metric_label = _metric_plot_label(metric_type, metric)
            metric_title = _metric_title(metric_type, metric)
            ax.set_xlabel("SPF", fontsize=BAR_CHART_STYLE["label_fontsize"], labelpad=8)
            ax.set_ylabel(metric_label, fontsize=BAR_CHART_STYLE["label_fontsize"])
            ax.set_title(metric_title, fontsize=BAR_CHART_STYLE["title_fontsize"])
            ax.set_xticks(x_centers)
            ax.set_xticklabels([str(spf) for spf in spf_values], fontsize=BAR_CHART_STYLE["tick_fontsize"])
            ax.tick_params(axis="x", labelsize=BAR_CHART_STYLE["tick_fontsize"], width=BAR_CHART_STYLE["tick_width"])
            ax.tick_params(axis="y", labelsize=BAR_CHART_STYLE["tick_fontsize"], width=BAR_CHART_STYLE["tick_width"])
            for spine in ax.spines.values():
                spine.set_linewidth(BAR_CHART_STYLE["spine_linewidth"])
            ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)

        for slot_idx, ax in enumerate(flat_axes):
            if legend_slot_idx is not None and slot_idx == legend_slot_idx:
                continue
            if slot_idx not in assigned_plot_slots:
                ax.set_visible(False)
        if not any_visible:
            plt.close(fig)
            continue

        if use_temporal_panel_legend and legend_slot_idx is not None and legend_slot_idx < len(flat_axes):
            legend_ax = flat_axes[legend_slot_idx]
            legend_ax.set_visible(True)
            legend_ax.set_xticks([])
            legend_ax.set_yticks([])
            legend_ax.set_frame_on(False)
            legend_ax.patch.set_alpha(0.0)
            for spine in legend_ax.spines.values():
                spine.set_visible(False)
            legend_ax.legend(
                handles=legend_handles,
                loc="center",
                ncol=1,
                fontsize=BAR_CHART_STYLE["legend_fontsize"],
                frameon=False,
            )
        elif use_bottom_horizontal_legend:
            fig.legend(
                handles=legend_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, BAR_CHART_STYLE["legend_bottom_y_anchor"]),
                ncol=2,
                fontsize=BAR_CHART_STYLE["legend_fontsize"],
                frameon=False,
            )
        else:
            fig.legend(
                handles=legend_handles,
                loc="center left",
                bbox_to_anchor=(
                    BAR_CHART_STYLE["legend_x_anchor"],
                    BAR_CHART_STYLE["legend_y_anchor"],
                ),
                ncol=1,
                fontsize=BAR_CHART_STYLE["legend_fontsize"],
                frameon=False,
            )

        subplot_right = (
            BAR_CHART_STYLE["subplot_right_no_side_legend"]
            if (use_temporal_panel_legend or use_bottom_horizontal_legend)
            else BAR_CHART_STYLE["subplot_right"]
        )
        subplot_bottom = (
            BAR_CHART_STYLE["subplot_bottom_with_bottom_legend"]
            if use_bottom_horizontal_legend
            else BAR_CHART_STYLE["subplot_bottom"]
        )
        fig.subplots_adjust(
            left=BAR_CHART_STYLE["subplot_left"],
            right=subplot_right,
            bottom=subplot_bottom,
            top=BAR_CHART_STYLE["subplot_top"],
            wspace=BAR_CHART_STYLE["subplot_wspace"],
            hspace=BAR_CHART_STYLE["subplot_hspace"],
        )

        name_parts = [metric_type, panel_name]
        if panel_name == "temporal":
            name_parts.extend([temporal_subset, temporal_region])
        file_name = f"{_safe_filename_fragment('_'.join(name_parts))}.png"
        fig.savefig(os.path.join(output_dir, file_name), dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from val_inference_logs.")
    parser.add_argument(
        "--log_file",
        default=str(DEFAULT_LOG_PATH),
        help="Path to val_inference_logs.json (default: inference/val_inference_logs.json).",
    )
    parser.add_argument("--exp_names", required=True, help="Comma-separated experiment names to include.")
    parser.add_argument(
        "--metric_type",
        choices=("spatial", "mc", "spatial_mc", "temporal", "timing"),
        required=True,
        help="Metric type to include in table.",
    )
    parser.add_argument(
        "--metrics",
        default="",
        help=(
            "Comma-separated metrics to include for the selected metric type. "
            "Examples: ssim,psnr (spatial), ssim,dro_dc_mae (spatial_mc), "
            "or early_corr,ttae_sec (temporal)."
        ),
    )
    parser.add_argument(
        "--temporal_subset",
        choices=("all", "top10", "top20"),
        default="all",
        help="Temporal subset to use when metric_type=temporal.",
    )
    parser.add_argument(
        "--temporal_region",
        choices=("malignant", "benign", "both"),
        default="malignant",
        help="Temporal region to use when metric_type=temporal.",
    )
    parser.add_argument(
        "--temporal_metrics",
        default="",
        help=(
            "Deprecated alias for temporal metric selection. "
            "Use --metrics instead."
        ),
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=None,
        help="Decimal places for mean/std (default: 3, or 2 with --format_mri_journal).",
    )
    parser.add_argument("--caption", default="", help="LaTeX table caption.")
    parser.add_argument("--label", default="", help="LaTeX table label.")
    parser.add_argument(
        "--format_mri_journal",
        action="store_true",
        help=(
            "Use MRI journal LaTeX formatting with booktabs/tabular*, explicit mean +/- SD, "
            "and red cells for experiment means more than one matched GRASP SD away."
        ),
    )
    af_group = parser.add_mutually_exclusive_group()
    af_group.add_argument(
        "--include_af",
        dest="include_af",
        action="store_true",
        default=None,
        help="Include AF column (default: disabled).",
    )
    af_group.add_argument(
        "--exclude_af",
        dest="include_af",
        action="store_false",
        help="Exclude AF column from the table.",
    )
    spf_group = parser.add_mutually_exclusive_group()
    spf_group.add_argument(
        "--include_spf",
        dest="include_spf",
        action="store_true",
        default=None,
        help="Include SPF column (default: enabled).",
    )
    spf_group.add_argument(
        "--exclude_spf",
        dest="include_spf",
        action="store_false",
        help="Exclude SPF column from the table.",
    )
    af_spf_group = parser.add_mutually_exclusive_group()
    af_spf_group.add_argument(
        "--include_af_spf",
        dest="include_af_spf_legacy",
        action="store_true",
        default=None,
        help="Deprecated: include both AF and SPF columns.",
    )
    af_spf_group.add_argument(
        "--exclude_af_spf",
        dest="include_af_spf_legacy",
        action="store_false",
        help="Deprecated: exclude both AF and SPF columns.",
    )
    parser.add_argument(
        "--exclude_timing_cols",
        dest="include_af_spf_legacy",
        action="store_false",
        help="Deprecated: use --exclude_af_spf.",
    )
    temporal_res_group = parser.add_mutually_exclusive_group()
    temporal_res_group.add_argument(
        "--include_temporal_resolution",
        dest="include_temporal_resolution",
        action="store_true",
        default=None,
        help="Include temporal resolution column (default: enabled).",
    )
    temporal_res_group.add_argument(
        "--exclude_temporal_resolution",
        dest="include_temporal_resolution",
        action="store_false",
        help="Exclude temporal resolution column.",
    )
    parser.add_argument(
        "--include_seconds_per_frame",
        dest="include_temporal_resolution",
        action="store_true",
        help="Deprecated alias for --include_temporal_resolution.",
    )
    parser.add_argument(
        "--exclude_seconds_per_frame",
        dest="include_temporal_resolution",
        action="store_false",
        help="Deprecated alias for --exclude_temporal_resolution.",
    )
    arrows_group = parser.add_mutually_exclusive_group()
    arrows_group.add_argument(
        "--include_metric_arrows",
        dest="include_metric_arrows",
        action="store_true",
        default=None,
        help="Append up/down arrows to metric headers (default: enabled).",
    )
    arrows_group.add_argument(
        "--exclude_metric_arrows",
        dest="include_metric_arrows",
        action="store_false",
        help="Do not append arrows to metric headers.",
    )
    highlight_group = parser.add_mutually_exclusive_group()
    highlight_group.add_argument(
        "--highlight_green_cells",
        dest="highlight_green_cells",
        action="store_true",
        default=None,
        help=(
            "Highlight metric cells in green when a value is more than one counterpart-row std "
            "below the counterpart-row mean within each BRISKNet/GRASP SPF pair."
        ),
    )
    highlight_group.add_argument(
        "--disable_green_cells",
        dest="highlight_green_cells",
        action="store_false",
        help="Disable green highlighting (default).",
    )
    parser.add_argument(
        "--green_cell_std_threshold",
        type=float,
        default=1.0,
        help=(
            "Standard-deviation multiplier for green highlighting relative to counterpart-row "
            "mean/std within an SPF pair (default: 1.0)."
        ),
    )
    parser.add_argument(
        "--exp_base_dir",
        default="output",
        help=(
            "Comma-separated base directories containing experiment configs "
            "(default: output)."
        ),
    )
    parser.add_argument(
        "--config_keys",
        default="",
        help=(
            "Comma-separated config paths to include as columns. "
            "Use Header:Path to override the column header."
        ),
    )
    parser.add_argument(
        "--save_metric_bar_charts",
        action="store_true",
        help=(
            "Also save grouped BRISKNet/GRASP metric bar-chart panels "
            "(one PNG per metric category: spatial, dc, or temporal)."
        ),
    )
    parser.add_argument(
        "--bar_chart_output_dir",
        default="figures/metric_bar_charts",
        help="Directory where metric bar chart PNGs are saved.",
    )
    parser.add_argument(
        "--bar_chart_dpi",
        type=int,
        default=300,
        help="DPI used for saved metric bar chart PNG files (default: 300).",
    )
    args = parser.parse_args()

    exp_names = set(_parse_list(args.exp_names))
    try:
        metrics = _resolve_metrics(args.metric_type, args.metrics, args.temporal_metrics)
    except ValueError as exc:
        parser.error(str(exc))
    config_cols = _parse_config_cols(args.config_keys)
    exp_base_dirs = _parse_list(args.exp_base_dir) or ["output"]
    fallback_dir = "/net/projects2/annawoodard/rachelgordon/experiments"
    if fallback_dir not in exp_base_dirs:
        exp_base_dirs.append(fallback_dir)
    config_cache = {}
    include_af = False if args.include_af is None else args.include_af
    include_spf = True if args.include_spf is None else args.include_spf
    if args.include_af_spf_legacy is not None:
        if args.include_af is None:
            include_af = args.include_af_spf_legacy
        if args.include_spf is None:
            include_spf = args.include_af_spf_legacy
    include_temporal_resolution = (
        True if args.include_temporal_resolution is None else args.include_temporal_resolution
    )
    include_metric_arrows = True if args.include_metric_arrows is None else args.include_metric_arrows
    highlight_green_cells = False if args.highlight_green_cells is None else args.highlight_green_cells
    green_cell_std_threshold = args.green_cell_std_threshold
    decimals = args.decimals
    if decimals is None:
        decimals = 2 if args.format_mri_journal else 3

    rows = _load_rows(args.log_file)
    exp_rows, grasp_index = _group_rows(rows)
    filtered = [r for r in exp_rows if r.get("exp_name") in exp_names]

    if args.metric_type != "temporal":
        print(
            _emit_table(
                filtered,
                grasp_index,
                args.metric_type,
                metrics,
                args.temporal_subset,
                "malignant",
                decimals,
                args.caption,
                args.label,
                config_cols,
                exp_base_dirs,
                config_cache,
                include_af=include_af,
                include_spf=include_spf,
                include_temporal_resolution=include_temporal_resolution,
                include_metric_arrows=include_metric_arrows,
                highlight_green_cells=highlight_green_cells,
                green_cell_std_threshold=green_cell_std_threshold,
                format_mri_journal=args.format_mri_journal,
            )
        )
        if args.save_metric_bar_charts:
            _save_metric_bar_charts(
                filtered,
                grasp_index,
                args.metric_type,
                metrics,
                args.temporal_subset,
                "malignant",
                args.bar_chart_output_dir,
                args.bar_chart_dpi,
            )
        return

    regions = ["malignant", "benign"] if args.temporal_region == "both" else [args.temporal_region]
    outputs = []
    for region in regions:
        outputs.append(
            _emit_table(
                filtered,
                grasp_index,
                args.metric_type,
                metrics,
                args.temporal_subset,
                region,
                decimals,
                args.caption,
                args.label,
                config_cols,
                exp_base_dirs,
                config_cache,
                include_af=include_af,
                include_spf=include_spf,
                include_temporal_resolution=include_temporal_resolution,
                include_metric_arrows=include_metric_arrows,
                highlight_green_cells=highlight_green_cells,
                green_cell_std_threshold=green_cell_std_threshold,
                format_mri_journal=args.format_mri_journal,
            )
        )
        if args.save_metric_bar_charts:
            _save_metric_bar_charts(
                filtered,
                grasp_index,
                args.metric_type,
                metrics,
                args.temporal_subset,
                region,
                args.bar_chart_output_dir,
                args.bar_chart_dpi,
            )
    print("\n\n".join(outputs))


if __name__ == "__main__":
    main()
