import argparse
import json
from collections import defaultdict
from typing import Iterable, List, Dict, Tuple


SPATIAL_METRICS = ["ssim", "psnr", "mse", "lpips"]
MC_METRICS = ["dro_dc_mae", "raw_dc_mae"]
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
    return f"{mean:.{decimals}f} \\pm {std:.{decimals}f}"


def _extract_mean_std_flat(row: Dict[str, str], prefix: str) -> Tuple[float | None, float | None]:
    return _to_float(row.get(f"{prefix}_mean")), _to_float(row.get(f"{prefix}_std"))


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
    grasp_index = {}
    for row in grasp_rows:
        key = (
            row.get("acceleration_factor") if row.get("type") != "GRASP" else row.get("acceleration"),
            row.get("spokes_per_frame"),
            row.get("num_frames"),
            row.get("dro_noise_level") if row.get("type") != "GRASP" else row.get("DRO_noise_level"),
        )
        grasp_index[key] = row
    return exp_rows, grasp_index


def _row_key(row: Dict[str, str]) -> Tuple[float, int, int, str]:
    accel = _to_float(row.get("acceleration_factor") or row.get("acceleration")) or 0.0
    spf = int(float(row.get("spokes_per_frame") or 0))
    frames = int(float(row.get("num_frames") or 0))
    noise = row.get("dro_noise_level") or row.get("DRO_noise_level") or ""
    return accel, spf, frames, noise


def _format_row(values: List[str]) -> str:
    return " & ".join(values) + " \\\\"


def _metric_columns(metric_type: str, temporal_metrics: List[str]) -> List[str]:
    if metric_type == "spatial":
        return ["SSIM", "PSNR", "MSE", "LPIPS"]
    if metric_type == "mc":
        return ["DRO DC MAE", "Raw DC MAE"]
    if metric_type == "temporal":
        return temporal_metrics
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
            mean = _to_float(dc.get("dro_dc_mae_mean"))
            std = _to_float(dc.get("dro_dc_mae_stddev"))
            values.append(_format_mean_std(mean, std, decimals))
            mean = _to_float(dc.get("raw_dc_mae_mean"))
            std = _to_float(dc.get("raw_dc_mae_stddev"))
            values.append(_format_mean_std(mean, std, decimals))
        elif metric_type == "temporal":
            temporal = row.get("temporal_metrics", {})
            block = temporal.get(_temporal_block_name(temporal_subset, temporal_region), {})
            for metric in temporal_metrics:
                mean = _to_float(block.get(f"{metric}_mean"))
                std = _to_float(block.get(f"{metric}_stddev"))
                values.append(_format_mean_std(mean, std, decimals))
        else:
            raise ValueError(f"Unknown metric_type {metric_type}")
        return values

    if metric_type == "spatial":
        for metric in SPATIAL_METRICS:
            mean, std = _extract_mean_std_flat(row, f"{model_prefix}_{metric}")
            values.append(_format_mean_std(mean, std, decimals))
    elif metric_type == "mc":
        mean, std = _extract_mean_std_flat(row, f"{model_prefix}_dc_mae")
        values.append(_format_mean_std(mean, std, decimals))
        if model_prefix == "dl":
            mean, std = _extract_mean_std_flat(row, "raw_ssdu_nmse")
        else:
            mean, std = _extract_mean_std_flat(row, "raw_grasp_ssdu_nmse")
        values.append(_format_mean_std(mean, std, decimals))
    elif metric_type == "temporal":
        prefix = "" if temporal_region == "malignant" else "benign_"
        for metric in temporal_metrics:
            key = f"{prefix}{model_prefix}_{temporal_subset}_{metric}"
            mean, std = _extract_mean_std_flat(row, key)
            values.append(_format_mean_std(mean, std, decimals))
    else:
        raise ValueError(f"Unknown metric_type {metric_type}")
    return values


def _emit_table(rows: List[Dict[str, str]], grasp_index: Dict[Tuple[str, str, str, str], Dict[str, str]],
                metric_type: str, temporal_metrics: List[str], temporal_subset: str, temporal_region: str,
                decimals: int, caption: str, label: str) -> str:
    header_cols = ["Method", "AF", "Seconds/Frame"] + _metric_columns(metric_type, temporal_metrics)
    col_spec = "|" + "|".join(["l"] + ["c"] * (len(header_cols) - 1)) + "|"

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")
    lines.append(_format_row(header_cols))
    lines.append("\\hline")

    sorted_rows = sorted(rows, key=_row_key)
    for row in sorted_rows:
        accel = row.get("acceleration_factor") or row.get("acceleration") or ""
        sec_per_frame = row.get("seconds_per_frame") or ""
        key = (
            row.get("acceleration_factor") or row.get("acceleration"),
            row.get("spokes_per_frame"),
            row.get("num_frames"),
            row.get("dro_noise_level") or row.get("DRO_noise_level"),
        )
        grasp_row = grasp_index.get(key, row)

        brisk_vals = _collect_metric_values(
            row,
            "dl",
            metric_type,
            temporal_metrics,
            temporal_subset,
            temporal_region,
            decimals,
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

        lines.append(_format_row(["BRISKNet", str(accel), str(sec_per_frame)] + brisk_vals))
        lines.append("\\hline")
        lines.append(_format_row(["GRASP", str(accel), str(sec_per_frame)] + grasp_vals))
        lines.append("\\hline")

    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from val_inference_logs.")
    parser.add_argument("--log_file", default="val_inference_logs.json", help="Path to val_inference_logs.json.")
    parser.add_argument("--exp_names", required=True, help="Comma-separated experiment names to include.")
    parser.add_argument(
        "--metric_type",
        choices=("spatial", "mc", "temporal"),
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
    parser.add_argument("--decimals", type=int, default=4, help="Decimal places for mean/std.")
    parser.add_argument("--caption", default="", help="LaTeX table caption.")
    parser.add_argument("--label", default="", help="LaTeX table label.")
    args = parser.parse_args()

    exp_names = set(_parse_list(args.exp_names))
    temporal_metrics = _parse_list(args.temporal_metrics) or TEMPORAL_METRICS

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
            )
        )
    print("\n\n".join(outputs))


if __name__ == "__main__":
    main()
