"""Generate LaTeX significance tables from results/significance_all_comparisons.csv.

Transposed layout (default, --transposed):
  Rows = metrics grouped by family; columns = SPF values (single comparison)
  or (comparison, SPF) pairs (multi-comparison / temporal ablation).
  Each cell: mean diff on line 1, [CI] on line 2 via \\makecell.

Legacy layout (--legacy):
  Rows = comparison × SPF groups; columns = metrics.

Usage:
    # Transposed, single comparison (BRISKNet vs GRASP all SPFs):
    python scripts/make_significance_table.py --transposed \\
        --comparisons BRISKNet_vs_GRASP --spf 2,4,8,16,24,36 \\
        --families spatial,mc,temporal --label tab:sig_brisknet_vs_grasp

    # Transposed, multi-comparison (temporal ablation):
    python scripts/make_significance_table.py --transposed \\
        --comparisons full_vs_diffeo_only,no_arrival_shift_vs_diffeo_only,no_rebin_vs_diffeo_only \\
        --spf 8,36 --families spatial,mc,temporal --label tab:sig_temporal_ablation
"""

import argparse
import math
from pathlib import Path

import pandas as pd


DEFAULT_CSV = Path(__file__).resolve().parents[1] / "results" / "significance_all_comparisons.csv"

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

METRIC_DECIMALS = {
    "ssim": 3, "psnr": 2, "lpips": 3,
    "early_corr": 2, "early_mae": 2, "iauc10_err": 1,
    "ttae_sec": 2, "wash_in_slope_err": 2,
    "dc_mae": 3, "ssdu_nmse": 3,
}

# (metric_key, latex_header, higher_is_better)
METRIC_DISPLAY = [
    ("ssim",              "SSIM",                                        True),
    ("psnr",              "PSNR",                                        True),
    ("lpips",             "LPIPS",                                       False),
    ("early_corr",        r"$\rho_{\mathrm{early}}$",                   True),
    ("early_mae",         r"$\mathrm{MAE}_{\mathrm{early}}$",           False),
    ("iauc10_err",        r"$\mathrm{iAUC}_{10}\ \mathrm{Err}$",       False),
    ("ttae_sec",          r"$t_{\mathrm{arr}}\ \mathrm{Err}$",          False),
    ("wash_in_slope_err", r"$\mathrm{MAE}_{\mathrm{wash\text{-}in}}$",  False),
    ("dc_mae",            "DC MAE",                                      False),
    ("ssdu_nmse",         "SSDU NMSE",                                   False),
]
METRIC_INFO = {m: (hdr, hib) for m, hdr, hib in METRIC_DISPLAY}

METRIC_FAMILIES = {
    "spatial":  ["ssim", "psnr", "lpips"],
    "mc":       ["dc_mae", "ssdu_nmse"],
    "temporal": ["early_corr", "early_mae", "iauc10_err", "ttae_sec", "wash_in_slope_err"],
}
FAMILY_LABELS = {"spatial": "Spatial quality", "mc": "Measurement consistency", "temporal": "Temporal fidelity"}

SPF_ORDER = [2, 4, 8, 16, 24, 36]

COMPARISON_DISPLAY = {
    "BRISKNet_vs_GRASP":                  "BRISKNet vs GRASP",
    "SSDU_vs_GRASP":                      "SSDU vs GRASP",
    "BRISKNet_vs_SSDU":                   "BRISKNet vs SSDU",
    "EI_vs_MC":                           "EI vs MC-only",
    "arr_shift_vs_diffeo_only":           "Arr Shift vs Diffeo Only",
    "enh_scale_vs_diffeo_only":           "Enh Scale vs Diffeo Only",
    "arr_shift_enh_scale_vs_diffeo_only": "Arr Shift + Enh Scale vs Diffeo Only",
    "all_transforms_36_vs_diffeo_only":        "All Transforms vs Diffeo Only",
    "arr_shift_enh_scale_36_vs_diffeo_only":   "Arr Shift + Enh Scale vs Diffeo Only (36 SPF)",
    "arr_shift_rebin_36_vs_diffeo_only":       "Arr Shift + Rebin vs Diffeo Only",
    "enh_scale_rebin_36_vs_diffeo_only":       "Enh Scale + Rebin vs Diffeo Only",
}

# Short column headers for transposed multi-comparison tables
COMPARISON_COL_HEADER = {
    "arr_shift_vs_diffeo_only":           r"Arr Shift\\vs Diffeo.",
    "enh_scale_vs_diffeo_only":           r"Enh Scale\\vs Diffeo.",
    "arr_shift_enh_scale_vs_diffeo_only": r"Arr Shift\\+Enh Scale\\vs Diffeo.",
    "BRISKNet_vs_GRASP":                  r"BRISKNet\\vs GRASP",
    "SSDU_vs_GRASP":                      r"SSDU\\vs GRASP",
    "BRISKNet_vs_SSDU":                   r"BRISKNet\\vs SSDU",
    "EI_vs_MC":                           r"EI\\vs MC-only",
    "all_transforms_36_vs_diffeo_only":       r"All Transf.\\vs Diffeo.",
    "arr_shift_enh_scale_36_vs_diffeo_only":  r"Arr Shift\\+Enh Scale\\vs Diffeo.",
    "arr_shift_rebin_36_vs_diffeo_only":      r"Arr Shift\\+Rebin\\vs Diffeo.",
    "enh_scale_rebin_36_vs_diffeo_only":      r"Enh Scale\\+Rebin\\vs Diffeo.",
}

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _sig_stars(p: float, alpha: float) -> str:
    if math.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < alpha:
        return "*"
    return ""


def _fmt(v: float, decimals: int) -> str:
    if math.isnan(v):
        return r"\cdot"
    return f"{v:.{decimals}f}"


def _format_cell_makecell(
    mean_diff: float,
    ci_low: float,
    ci_high: float,
    p_adj: float,
    alpha: float,
    bold: bool,
    metric: str,
) -> str:
    """Two-line cell: Δ^{stars} on line 1, [CI] on line 2 via \\makecell."""
    decimals = METRIC_DECIMALS.get(metric, 3)
    if math.isnan(mean_diff):
        return "NA"
    stars = _sig_stars(p_adj, alpha)
    diff_str = _fmt(mean_diff, decimals)
    ci_str = f"[{_fmt(ci_low, decimals)},\\,{_fmt(ci_high, decimals)}]"
    line1 = f"${diff_str}^{{{stars}}}$" if stars else f"${diff_str}$"
    line2 = r"{\footnotesize " + f"${ci_str}$" + "}"
    cell = f"\\makecell[c]{{{line1}\\\\ {line2}}}"
    if bold and stars:
        cell = f"\\bfseries {cell}"
    return cell


def _format_cell_inline(
    mean_diff: float,
    ci_low: float,
    ci_high: float,
    p_adj: float,
    alpha: float,
    bold: bool,
    metric: str,
) -> str:
    """Legacy single-line cell: Δ\\;[CI]^{stars}."""
    decimals = METRIC_DECIMALS.get(metric, 3)
    if math.isnan(mean_diff):
        return "NA"
    stars = _sig_stars(p_adj, alpha)
    diff_str = _fmt(mean_diff, decimals)
    ci_str = f"[{_fmt(ci_low, decimals)},\\,{_fmt(ci_high, decimals)}]"
    cell = f"${diff_str}\\;{ci_str}^{{{stars}}}$" if stars else f"${diff_str}\\;{ci_str}$"
    if bold and stars:
        cell = f"\\textbf{{{cell}}}"
    return cell


def _format_row(values: list[str]) -> str:
    return " & ".join(values) + r" \\"


def _safe_float(val) -> float:
    if val is None:
        return float("nan")
    try:
        f = float(val)
        return float("nan") if math.isnan(f) else f
    except (TypeError, ValueError):
        return float("nan")


def _safe_str(val) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    return str(val)


def _lookup_cell(df: pd.DataFrame, comparison: str, spf: int, metric: str,
                 alpha: float, show_direction: bool, makecell: bool) -> str:
    row = df[(df["comparison"] == comparison) & (df["spf"] == spf) & (df["metric"] == metric)]
    if row.empty:
        return "NA"
    r = row.iloc[0]
    mean_diff = _safe_float(r.get("mean_diff"))
    ci_low    = _safe_float(r.get("ci_low"))
    ci_high   = _safe_float(r.get("ci_high"))
    p_adj     = _safe_float(r.get("p_adj_bh"))
    direction = _safe_str(r.get("direction"))
    bold = show_direction and direction == "a_better"
    fn = _format_cell_makecell if makecell else _format_cell_inline
    return fn(mean_diff, ci_low, ci_high, p_adj, alpha, bold, metric)


# ---------------------------------------------------------------------------
# Table builders — transposed (metrics as rows)
# ---------------------------------------------------------------------------

def _table_open(caption: str, label: str, n_cols: int) -> list[str]:
    col_spec = "L" * n_cols
    return [
        r"\begin{table}%[]",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\small",
        f"\\begin{{tabular*}}{{\\tblwidth}}{{@{{}}{col_spec}@{{}}}}",
        r"\toprule",
    ]


def _table_open_compact(caption: str, label: str) -> list[str]:
    """Compact table using plain tabular (not tabular*) so columns auto-size to content."""
    return [
        r"\begin{table}%[]",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\small",
        r"\begin{tabular}{@{}lc@{}}",
        r"\toprule",
    ]


def _table_close() -> list[str]:
    return [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]


def _table_close_compact() -> list[str]:
    return [r"\bottomrule", r"\end{tabular}", r"\end{table}"]


def _metric_label(m: str) -> str:
    hdr, hib = METRIC_INFO[m]
    arrow = r"$\uparrow$" if hib else r"$\downarrow$"
    return f"{hdr}~{arrow}"


def _emit_metric_rows(
    df: pd.DataFrame,
    metrics: list[str],
    families: list[str],
    col_keys: list[tuple],   # list of (comparison, spf) pairs
    n_cols: int,
    alpha: float,
    show_direction: bool,
    makecell: bool,
) -> list[str]:
    """Emit family-grouped metric rows for transposed tables."""
    lines = []
    for fam in families:
        fam_metrics = [m for m in metrics if m in METRIC_FAMILIES.get(fam, [])]
        for m in fam_metrics:
            cells = [_metric_label(m)]
            for cmp, spf in col_keys:
                cells.append(_lookup_cell(df, cmp, spf, m, alpha, show_direction, makecell))
            lines.append(_format_row(cells))
    return lines


def make_single_result_column(
    df: pd.DataFrame,
    comparison: str,
    spf: int,
    metrics: list[str],
    families: list[str],
    alpha: float,
    caption: str,
    label: str,
    show_direction: bool,
    col_header: str,
) -> str:
    """Compact table for a single comparison × single SPF.

    Rows = metrics; one value column with a descriptive header.
    Uses plain tabular (not tabular*) so the column auto-sizes to content
    instead of stretching across the full text width.
    """
    lines = _table_open_compact(caption, label)
    lines.append(_format_row(["Metric", col_header]))
    lines.append(r"\midrule")
    for fam in families:
        fam_metrics = [m for m in metrics if m in METRIC_FAMILIES.get(fam, [])]
        for m in fam_metrics:
            cell = _lookup_cell(df, comparison, spf, m, alpha, show_direction, makecell=True)
            lines.append(_format_row([_metric_label(m), cell]))
    lines.extend(_table_close_compact())
    return "\n".join(lines)


def make_transposed_single_comparison(
    df: pd.DataFrame,
    comparison: str,
    spf_list: list[int],
    metrics: list[str],
    families: list[str],
    alpha: float,
    caption: str,
    label: str,
    show_direction: bool,
) -> str:
    """Rows = metrics; cols = SPF values.  One comparison."""
    avail_spf = [
        s for s in SPF_ORDER
        if s in spf_list and not df[(df["comparison"] == comparison) & (df["spf"] == s)].empty
    ]
    col_keys = [(comparison, s) for s in avail_spf]
    n_cols = 1 + len(avail_spf)

    spf_headers = [f"SPF~{s}" for s in avail_spf]
    lines = _table_open(caption, label, n_cols)
    lines.append(_format_row(["Metric"] + spf_headers))
    lines.append(r"\midrule")
    lines.extend(_emit_metric_rows(df, metrics, families, col_keys, n_cols, alpha, show_direction, makecell=True))
    lines.extend(_table_close())
    return "\n".join(lines)


def make_transposed_multi_comparison(
    df: pd.DataFrame,
    comparisons: list[str],
    spf_list: list[int],
    metrics: list[str],
    families: list[str],
    alpha: float,
    caption: str,
    label: str,
    show_direction: bool,
) -> str:
    """Rows = metrics; cols = (comparison, SPF) pairs that have data."""
    col_keys = [
        (cmp, s)
        for cmp in comparisons
        for s in SPF_ORDER
        if s in spf_list and not df[(df["comparison"] == cmp) & (df["spf"] == s)].empty
    ]
    n_cols = 1 + len(col_keys)

    col_headers = []
    for cmp, spf in col_keys:
        short = COMPARISON_COL_HEADER.get(cmp, cmp.replace("_", " "))
        col_headers.append(f"\\makecell{{{short}\\\\ (SPF~{spf})}}")

    lines = _table_open(caption, label, n_cols)
    lines.append(_format_row(["Metric"] + col_headers))
    lines.append(r"\midrule")
    lines.extend(_emit_metric_rows(df, metrics, families, col_keys, n_cols, alpha, show_direction, makecell=True))
    lines.extend(_table_close())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SPF-rows table builders (rows = SPF, cols = metrics)
# ---------------------------------------------------------------------------

def make_spf_rows_single_comparison(
    df: pd.DataFrame,
    comparison: str,
    spf_list: list[int],
    metrics: list[str],
    alpha: float,
    caption: str,
    label: str,
    show_direction: bool,
) -> str:
    """Rows = SPF; cols = metrics.  Single comparison, one metric family."""
    n_cols = 1 + len(metrics)
    lines = _table_open(caption, label, n_cols)
    lines.append(_format_row(["SPF"] + [_metric_label(m) for m in metrics]))
    lines.append(r"\midrule")
    avail_spf = [s for s in SPF_ORDER if s in spf_list and
                 not df[(df["comparison"] == comparison) & (df["spf"] == s)].empty]
    for spf in avail_spf:
        cells = [str(spf)] + [
            _lookup_cell(df, comparison, spf, m, alpha, show_direction, makecell=True)
            for m in metrics
        ]
        lines.append(_format_row(cells))
    lines.extend(_table_close())
    return "\n".join(lines)


def make_spf_rows_multi_comparison(
    df: pd.DataFrame,
    comparisons: list[str],
    spf_list: list[int],
    metrics: list[str],
    alpha: float,
    caption: str,
    label: str,
    show_direction: bool,
) -> str:
    """Rows = SPF grouped by comparison; cols = metrics.  Multiple comparisons, one family."""
    n_cols = 1 + len(metrics)
    lines = _table_open(caption, label, n_cols)
    lines.append(_format_row(["SPF"] + [_metric_label(m) for m in metrics]))
    lines.append(r"\midrule")
    last_cmp_idx = len(comparisons) - 1
    for cmp_idx, cmp in enumerate(comparisons):
        disp = COMPARISON_DISPLAY.get(cmp, cmp.replace("_", " "))
        lines.append(f"\\multicolumn{{{n_cols}}}{{l}}{{\\textit{{{disp}}}}}\\\\ ")
        avail_spf = [s for s in SPF_ORDER if s in spf_list and
                     not df[(df["comparison"] == cmp) & (df["spf"] == s)].empty]
        for spf in avail_spf:
            cells = [str(spf)] + [
                _lookup_cell(df, cmp, spf, m, alpha, show_direction, makecell=True)
                for m in metrics
            ]
            lines.append(_format_row(cells))
        if cmp_idx < last_cmp_idx:
            lines.append(r"\addlinespace")
    lines.extend(_table_close())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Legacy table builders (rows = comparison×SPF, cols = metrics)
# ---------------------------------------------------------------------------

def _metric_header_cells(metrics: list[str], left_headers: list[str]) -> list[str]:
    cells = list(left_headers)
    for m in metrics:
        hdr, hib = METRIC_INFO[m]
        arrow = r" $\uparrow$" if hib else r" $\downarrow$"
        cells.append(f"{hdr}{arrow}")
    return cells


def _metric_list(families: list[str], metrics_override: list[str]) -> list[str]:
    if metrics_override:
        return [m for m in metrics_override if m in METRIC_INFO]
    out = []
    for fam in families:
        out.extend(METRIC_FAMILIES.get(fam, []))
    return out


def make_multi_comparison_table(
    df: pd.DataFrame,
    comparisons: list[str],
    spf_list: list[int],
    metrics: list[str],
    families: list[str],
    alpha: float,
    caption: str,
    label: str,
    show_direction: bool,
) -> str:
    """Legacy: rows = comparison sub-rows grouped by SPF; cols = metrics."""
    n_cols = 1 + len(metrics)
    col_spec = "L" * n_cols
    lines = [
        r"\begin{table}%[]",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\small",
        f"\\begin{{tabular*}}{{\\tblwidth}}{{@{{}}{col_spec}@{{}}}}",
        r"\toprule",
        _format_row(_metric_header_cells(metrics, ["Comparison"])),
        r"\midrule",
    ]
    avail_spf = [s for s in SPF_ORDER if s in spf_list]
    for i, spf in enumerate(avail_spf):
        lines.append(f"\\multicolumn{{{n_cols}}}{{l}}{{\\textit{{SPF\\,=\\,{spf}}}}}\\\\" )
        for cmp in comparisons:
            if df[(df["comparison"] == cmp) & (df["spf"] == spf)].empty:
                continue
            disp = COMPARISON_DISPLAY.get(cmp, cmp.replace("_", " "))
            cells = [disp] + [
                _lookup_cell(df, cmp, spf, m, alpha, show_direction, makecell=False)
                for m in metrics
            ]
            lines.append(_format_row(cells))
        if i < len(avail_spf) - 1:
            lines.append(r"\addlinespace")
    lines.extend(_table_close())
    return "\n".join(lines)


def make_single_comparison_table(
    df: pd.DataFrame,
    comparison: str,
    spf_list: list[int],
    metrics: list[str],
    families: list[str],
    alpha: float,
    caption: str,
    label: str,
    show_direction: bool,
) -> str:
    """Legacy: rows = SPF values; cols = metrics."""
    n_cols = 1 + len(metrics)
    col_spec = "L" * n_cols
    lines = [
        r"\begin{table}%[]",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\small",
        f"\\begin{{tabular*}}{{\\tblwidth}}{{@{{}}{col_spec}@{{}}}}",
        r"\toprule",
        _format_row(_metric_header_cells(metrics, ["SPF"])),
        r"\midrule",
    ]
    avail_spf = [s for s in SPF_ORDER if s in spf_list]
    for spf in avail_spf:
        cells = [str(spf)] + [
            _lookup_cell(df, comparison, spf, m, alpha, show_direction, makecell=False)
            for m in metrics
        ]
        lines.append(_format_row(cells))
    lines.extend(_table_close())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX significance table from significance CSV."
    )
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--comparisons", default="BRISKNet_vs_GRASP",
        help="Comma-separated comparison names.",
    )
    parser.add_argument(
        "--spf", default="",
        help="Comma-separated SPF values (default: all available).",
    )
    parser.add_argument(
        "--families", default="spatial,mc,temporal",
        help="Comma-separated metric families.",
    )
    parser.add_argument(
        "--metrics", default="",
        help="Comma-separated metrics, overrides --families.",
    )
    parser.add_argument("--label", default="tab:significance")
    parser.add_argument("--caption", default="")
    parser.add_argument(
        "--transposed", action="store_true", default=False,
        help="Transposed layout: rows = metrics, cols = SPF or (comparison, SPF).",
    )
    parser.add_argument(
        "--spf-rows", action="store_true", default=False,
        help="SPF-rows layout: rows = SPF grouped by comparison, cols = metrics (one family).",
    )
    parser.add_argument(
        "--col-header", default="",
        help="Override data column header (triggers compact single-result-column layout).",
    )
    parser.add_argument(
        "--show_direction", action="store_true", default=True,
    )
    parser.add_argument("--no_show_direction", dest="show_direction", action="store_false")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    comparisons = [c.strip() for c in args.comparisons.split(",") if c.strip()]
    families    = [f.strip() for f in args.families.split(",") if f.strip()]
    metrics_override = [m.strip() for m in args.metrics.split(",") if m.strip()]
    spf_list = [int(s) for s in args.spf.split(",") if s.strip()] if args.spf else SPF_ORDER

    metrics = _metric_list(families, metrics_override)
    df = df[df["comparison"].isin(comparisons) & df["spf"].isin(spf_list)].copy()

    if args.col_header and len(comparisons) == 1 and len(spf_list) == 1:
        table = make_single_result_column(
            df, comparisons[0], spf_list[0], metrics, families,
            args.alpha, args.caption, args.label, args.show_direction,
            col_header=args.col_header,
        )
    elif args.spf_rows:
        if len(comparisons) == 1:
            table = make_spf_rows_single_comparison(
                df, comparisons[0], spf_list, metrics,
                args.alpha, args.caption, args.label, args.show_direction,
            )
        else:
            table = make_spf_rows_multi_comparison(
                df, comparisons, spf_list, metrics,
                args.alpha, args.caption, args.label, args.show_direction,
            )
    elif args.transposed:
        if len(comparisons) == 1:
            table = make_transposed_single_comparison(
                df, comparisons[0], spf_list, metrics, families,
                args.alpha, args.caption, args.label, args.show_direction,
            )
        else:
            table = make_transposed_multi_comparison(
                df, comparisons, spf_list, metrics, families,
                args.alpha, args.caption, args.label, args.show_direction,
            )
    else:
        if len(comparisons) == 1:
            table = make_single_comparison_table(
                df, comparisons[0], spf_list, metrics, families,
                args.alpha, args.caption, args.label, args.show_direction,
            )
        else:
            table = make_multi_comparison_table(
                df, comparisons, spf_list, metrics, families,
                args.alpha, args.caption, args.label, args.show_direction,
            )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(table + "\n")
        print(f"Saved to: {args.out}")
    else:
        print(table)


if __name__ == "__main__":
    main()
