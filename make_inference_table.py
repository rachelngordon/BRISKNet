"""Build a summary table from inference metrics JSON logs. Run: python3 make_inference_table.py --help"""

import argparse
import json
import os
from collections import defaultdict
from typing import Iterable, List, Dict, Tuple

import yaml


SPATIAL_METRICS = ["ssim", "psnr", "mse", "lpips"]
MC_METRICS = ["dro_dc_mae", "dro_dc_mse", "raw_ssdu_nmse"]
TEMPORAL_METRICS = [
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


def _format_mean_std(mean, std, decimals: int) -> str:
    if mean is None:
        return ""
    if std is None:
        return f"{mean:.{decimals}f}"
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


def _format_config_value(value) -> str:
    if value is None:
        return ""
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


def _match_key(row: Dict[str, str]) -> Tuple[str, str, str]:
    return (
        row.get("spokes_per_frame"),
        row.get("num_frames"),
        row.get("dro_noise_level") or row.get("DRO_noise_level"),
    )


def _format_row(values: List[str]) -> str:
    return " & ".join(values) + " \\\\"


def _metric_columns(metric_type: str, temporal_metrics: List[str]) -> List[str]:
    if metric_type == "spatial":
        return ["SSIM", "PSNR", "MSE", "LPIPS"]
    if metric_type == "mc":
        return ["DRO DC MAE", "DRO DC MSE", "Raw SSDU NMSE"]
    if metric_type == "temporal":
        return [
            "$\\rho_\\text{full}$",
            "$MAE_\\text{full}$",
            "$\\rho_\\text{early}$",
            "$MAE_\\text{early}$",
            "$t_\\text{arr}$ Error",
            "$MAE_\\text{wash-in}$",
            "$iAUC_{10}$ Error",
            "$MAE_\\text{peak}$",
            "$t_\\text{peak}$ Error",
        ]
    if metric_type == "timing":
        return ["Avg Inference Time (s)"]
    raise ValueError(f"Unknown metric_type {metric_type}")


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


def _collect_metric_values(row: Dict[str, str], model_prefix: str, metric_type: str, temporal_metrics: List[str],
                           temporal_subset: str, temporal_region: str, decimals: int) -> List[str]:
    values = []
    if row.get("type") in ("BRISKNet", "GRASP"):
        if metric_type == "spatial":
            spatial = row.get("spatial_metrics", {})
            for metric in SPATIAL_METRICS:
                mean = _to_float(spatial.get(f"{metric}_mean"))
                std = _to_float(spatial.get(f"{metric}_stddev"))
                values.append(_format_mean_std(mean, std, decimals))
        elif metric_type == "mc":
            dc = row.get("dc_metrics", {})
            for metric in MC_METRICS:
                metric_key = metric
                if metric == "raw_ssdu_nmse" and row.get("type") == "GRASP":
                    metric_key = "raw_grasp_ssdu_nmse"
                mean = _to_float(dc.get(f"{metric_key}_mean"))
                std = _to_float(dc.get(f"{metric_key}_stddev"))
                values.append(_format_mean_std(mean, std, decimals))
        elif metric_type == "temporal":
            temporal = row.get("temporal_metrics", {})
            block = temporal.get(_temporal_block_name(temporal_subset, temporal_region), {})
            for metric in temporal_metrics:
                mean = _to_float(block.get(f"{metric}_mean"))
                std = _to_float(block.get(f"{metric}_stddev"))
                values.append(_format_mean_std(mean, std, decimals))
        elif metric_type == "timing":
            mean, std = _extract_timing_stats(row)
            values.append(_format_mean_std(mean, std, decimals))
        else:
            raise ValueError(f"Unknown metric_type {metric_type}")
        return values

    if metric_type == "spatial":
        for metric in SPATIAL_METRICS:
            mean, std = _extract_mean_std_flat(row, f"{model_prefix}_{metric}")
            values.append(_format_mean_std(mean, std, decimals))
    elif metric_type == "mc":
        for metric in MC_METRICS:
            metric_key = metric
            if metric == "raw_ssdu_nmse" and model_prefix == "grasp":
                metric_key = "raw_grasp_ssdu_nmse"
            mean, std = _extract_mean_std_flat(row, f"{model_prefix}_{metric_key}")
            values.append(_format_mean_std(mean, std, decimals))
        # mean, std = _extract_mean_std_flat(row, f"{model_prefix}_dc_mae")
        # values.append(_format_mean_std(mean, std, decimals))
        # if model_prefix == "dl":
        #     mean, std = _extract_mean_std_flat(row, "raw_ssdu_nmse")
        # else:
        #     mean, std = _extract_mean_std_flat(row, "raw_grasp_ssdu_nmse")
        # values.append(_format_mean_std(mean, std, decimals))
    elif metric_type == "temporal":
        prefix = "" if temporal_region == "malignant" else "benign_"
        for metric in temporal_metrics:
            key = f"{prefix}{model_prefix}_{temporal_subset}_{metric}"
            mean, std = _extract_mean_std_flat(row, key)
            values.append(_format_mean_std(mean, std, decimals))
    elif metric_type == "timing":
        mean, std = _extract_timing_stats_flat(row, model_prefix)
        values.append(_format_mean_std(mean, std, decimals))
    else:
        raise ValueError(f"Unknown metric_type {metric_type}")
    return values


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


def _emit_table(rows: List[Dict[str, str]], grasp_index: Dict[Tuple[str, str, str], List[Dict[str, str]]],
                metric_type: str, temporal_metrics: List[str], temporal_subset: str, temporal_region: str,
                decimals: int, caption: str, label: str, config_cols: List[Tuple[str, str]],
                exp_base_dirs: List[str], config_cache: Dict[str, Dict],
                include_af_spf: bool, include_seconds_per_frame: bool) -> str:
    timing_cols = []
    if include_af_spf:
        timing_cols.extend(["AF", "SPF"])
    if include_seconds_per_frame:
        timing_cols.append("Seconds/Frame")
    if config_cols:
        header_cols = ["Method"] + [col for col, _ in config_cols] + timing_cols
    else:
        header_cols = ["Method"] + timing_cols
    header_cols += _metric_columns(metric_type, temporal_metrics)
    col_spec = "|" + "|".join(["l"] * len(header_cols)) + "|"

    lines = []
    lines.append("\\begin{table}")
    lines.append(f"\\caption{{{caption}}}\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")
    lines.append(_format_row(header_cols))
    lines.append("\\hline")

    sorted_rows = sorted(rows, key=_row_key)
    grouped = []
    group_map: Dict[Tuple[str, str, str], List[Dict[str, str]]] = {}
    for row in sorted_rows:
        match_key = _match_key(row)
        if match_key not in group_map:
            group_map[match_key] = []
            grouped.append(match_key)
        group_map[match_key].append(row)

    for match_key in grouped:
        group_rows = group_map[match_key]
        grasp_row = _select_grasp_row(grasp_index.get(match_key), metric_type)

        for row in group_rows:
            accel = _format_one_decimal(row.get("acceleration_factor") or row.get("acceleration"))
            spf = _format_int(row.get("spokes_per_frame"))
            sec_per_frame = _format_one_decimal(row.get("seconds_per_frame"))
            exp_name = row.get("exp_name") or ""

            if config_cols:
                cfg = _load_config(exp_name, exp_base_dirs, config_cache)
                config_vals = [
                    _format_config_value(_get_config_value(cfg, path))
                    for _, path in config_cols
                ]
            else:
                config_vals = []

            brisk_vals = _collect_metric_values(
                row,
                "dl",
                metric_type,
                temporal_metrics,
                temporal_subset,
                temporal_region,
                decimals,
            )
            timing_vals = []
            if include_af_spf:
                timing_vals.extend([str(accel), str(spf)])
            if include_seconds_per_frame:
                timing_vals.append(str(sec_per_frame))
            lines.append(_format_row(["BRISKNet"] + config_vals + timing_vals + brisk_vals))

        if grasp_row is not None:
            empty_config_vals = [""] * len(config_cols)
            ref_row = group_rows[0]
            grasp_accel = _format_one_decimal(
                ref_row.get("acceleration_factor")
                or ref_row.get("acceleration")
                or grasp_row.get("acceleration")
            )
            grasp_spf = _format_int(ref_row.get("spokes_per_frame") or grasp_row.get("spokes_per_frame"))
            grasp_sec_per_frame = _format_one_decimal(
                ref_row.get("seconds_per_frame") or grasp_row.get("seconds_per_frame")
            )
            grasp_vals = _collect_metric_values(
                grasp_row,
                "grasp",
                metric_type,
                temporal_metrics,
                temporal_subset,
                temporal_region,
                decimals,
            )
            timing_vals = []
            if include_af_spf:
                timing_vals.extend([str(grasp_accel), str(grasp_spf)])
            if include_seconds_per_frame:
                timing_vals.append(str(grasp_sec_per_frame))
            lines.append(_format_row(["GRASP"] + empty_config_vals + timing_vals + grasp_vals))

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from val_inference_logs.")
    parser.add_argument("--log_file", default="val_inference_logs.json", help="Path to val_inference_logs.json.")
    parser.add_argument("--exp_names", required=True, help="Comma-separated experiment names to include.")
    parser.add_argument(
        "--metric_type",
        choices=("spatial", "mc", "temporal", "timing"),
        required=True,
        help="Metric type to include in table.",
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
        default=",".join(TEMPORAL_METRICS),
        help="Comma-separated temporal metrics to include (overrides defaults).",
    )
    parser.add_argument("--decimals", type=int, default=3, help="Decimal places for mean/std.")
    parser.add_argument("--caption", default="", help="LaTeX table caption.")
    parser.add_argument("--label", default="", help="LaTeX table label.")
    af_group = parser.add_mutually_exclusive_group()
    af_group.add_argument(
        "--include_af_spf",
        dest="include_af_spf",
        action="store_true",
        default=None,
        help="Include AF and SPF columns (default: enabled).",
    )
    af_group.add_argument(
        "--exclude_af_spf",
        dest="include_af_spf",
        action="store_false",
        help="Exclude AF and SPF columns from the table.",
    )
    parser.add_argument(
        "--exclude_timing_cols",
        dest="include_af_spf",
        action="store_false",
        help="Deprecated: use --exclude_af_spf.",
    )
    sec_group = parser.add_mutually_exclusive_group()
    sec_group.add_argument(
        "--include_seconds_per_frame",
        dest="include_seconds_per_frame",
        action="store_true",
        default=None,
        help="Include Seconds/Frame column.",
    )
    sec_group.add_argument(
        "--exclude_seconds_per_frame",
        dest="include_seconds_per_frame",
        action="store_false",
        help="Exclude Seconds/Frame column (default).",
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
    args = parser.parse_args()

    exp_names = set(_parse_list(args.exp_names))
    temporal_metrics = _parse_list(args.temporal_metrics) or TEMPORAL_METRICS
    config_cols = _parse_config_cols(args.config_keys)
    exp_base_dirs = _parse_list(args.exp_base_dir) or ["output"]
    fallback_dir = "/net/projects2/annawoodard/rachelgordon/experiments"
    if fallback_dir not in exp_base_dirs:
        exp_base_dirs.append(fallback_dir)
    config_cache = {}
    include_af_spf = True if args.include_af_spf is None else args.include_af_spf
    include_seconds_per_frame = False if args.include_seconds_per_frame is None else args.include_seconds_per_frame

    rows = _load_rows(args.log_file)
    exp_rows, grasp_index = _group_rows(rows)
    filtered = [r for r in exp_rows if r.get("exp_name") in exp_names]

    if args.metric_type != "temporal":
        print(
            _emit_table(
                filtered,
                grasp_index,
                args.metric_type,
                temporal_metrics,
                args.temporal_subset,
                "malignant",
                args.decimals,
                args.caption,
                args.label,
                config_cols,
                exp_base_dirs,
                config_cache,
                include_af_spf=include_af_spf,
                include_seconds_per_frame=include_seconds_per_frame,
            )
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
                temporal_metrics,
                args.temporal_subset,
                region,
                args.decimals,
                args.caption,
                args.label,
                config_cols,
                exp_base_dirs,
                config_cache,
                include_af_spf=include_af_spf,
                include_seconds_per_frame=include_seconds_per_frame,
            )
        )
    print("\n\n".join(outputs))


if __name__ == "__main__":
    main()
