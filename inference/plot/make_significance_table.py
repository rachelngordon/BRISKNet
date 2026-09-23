"""Generate LaTeX significance tables from results/significance_all_comparisons.csv.

Supports single-comparison mode (rows = SPF) and multi-comparison mode
(rows = comparison × SPF, grouped by SPF).  Matches the mri-journal
booktabs/tabular* style used in make_inference_table.py.

Usage:
    # Multi-comparison (acceleration sweep spatial table):
    python inference/plot/make_significance_table.py \\
        --comparisons BRISKNet_vs_GRASP,SSDU_vs_GRASP,BRISKNet_vs_SSDU \\
        --spf 8,16,24,36 --families spatial --label tab:sig_acc_exp_spatial

    # Single-comparison (ultra-high, BRISKNet vs GRASP only):
    python inference/plot/make_significance_table.py \\
        --comparisons BRISKNet_vs_GRASP --spf 2,4 \\
        --families spatial --label tab:sig_ultra_acc_exp_spatial
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CSV = Path(__file__).resolve().parents[2] / "results" / "significance_all_comparisons.csv"

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
    ("ssim",             "SSIM",                                        True),
    ("psnr",             "PSNR",                                        True),
    ("lpips",            "LPIPS",                                       False),
    ("early_corr",       r"$\rho_{\mathrm{early}}$",                   True),
    ("early_mae",        r"$\mathrm{MAE}_{\mathrm{early}}$",           False),
    ("iauc10_err",       r"$\mathrm{iAUC}_{10}\ \mathrm{Err}$",       False),
    ("ttae_sec",         r"$t_{\mathrm{arr}}\ \mathrm{Err}$",          False),
    ("wash_in_slope_err",r"$\mathrm{MAE}_{\mathrm{wash\text{-}in}}$",  False),
    ("dc_mae",           "DC MAE",                                      False),
    ("ssdu_nmse",        "SSDU NMSE",                                   False),
]
METRIC_INFO = {m: (hdr, hib) for m, hdr, hib in METRIC_DISPLAY}

METRIC_FAMILIES = {
    "spatial":  ["ssim", "psnr", "lpips"],
    "temporal": ["early_corr", "early_mae", "iauc10_err", "ttae_sec", "wash_in_slope_err"],
    "mc":       ["dc_mae", "ssdu_nmse"],
}
FAMILY_LABELS = {"spatial": "Spatial", "temporal": "Temporal", "mc": "MC"}

SPF_ORDER = [2, 4, 8, 16, 24, 36]

COMPARISON_DISPLAY = {
    "BRISKNet_vs_GRASP":               "BRISKNet vs GRASP",
    "SSDU_vs_GRASP":                   "SSDU vs GRASP",
    "BRISKNet_vs_SSDU":                "BRISKNet vs SSDU",
    "EI_vs_MC":                        "EI vs MC-only",
    "full_vs_diffeo_only":             "Full vs Diffeo-only",
    "no_arrival_shift_vs_diffeo_only": "No arr-shift vs Diffeo-only",
    "no_rebin_vs_diffeo_only":         "No rebin vs Diffeo-only",
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


def _format_cell(
    mean_diff: float,
    ci_low: float,
    ci_high: float,
    p_adj: float,
    alpha: float,
    bold: bool,
    metric: str,
) -> str:
    decimals = METRIC_DECIMALS.get(metric, 3)
    if math.isnan(mean_diff):
        return "---"
    stars = _sig_stars(p_adj, alpha)
    diff_str = _fmt(mean_diff, decimals)
    ci_str = f"[{_fmt(ci_low, decimals)},\\,{_fmt(ci_high, decimals)}]"
    if stars:
        cell = f"${diff_str}$ ${ci_str}^{{{stars}}}$"
    else:
        cell = f"${diff_str}$ ${ci_str}$"
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


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def _metric_list(families: list[str], metrics_override: list[str]) -> list[str]:
    if metrics_override:
        return [m for m in metrics_override if m in METRIC_INFO]
    out = []
    for fam in families:
        out.extend(METRIC_FAMILIES.get(fam, []))
    return out


def _family_header_cells(metrics: list[str], families: list[str], n_left_cols: int) -> list[str]:
    cells = [""] * n_left_cols
    for fam in families:
        fam_metrics = [m for m in metrics if m in METRIC_FAMILIES.get(fam, [])]
        if fam_metrics:
            cells.append(f"\\multicolumn{{{len(fam_metrics)}}}{{c}}{{{FAMILY_LABELS[fam]}}}")
    return cells


def _metric_header_cells(metrics: list[str], left_headers: list[str]) -> list[str]:
    cells = list(left_headers)
    for m in metrics:
        hdr, hib = METRIC_INFO[m]
        arrow = r" $\uparrow$" if hib else r" $\downarrow$"
        cells.append(f"{hdr}{arrow}")
    return cells


def _get_cell(df: pd.DataFrame, comparison: str, spf: int, metric: str,
              alpha: float, show_direction: bool) -> str:
    row = df[(df["comparison"] == comparison) & (df["spf"] == spf) & (df["metric"] == metric)]
    if row.empty:
        return "---"
    r = row.iloc[0]
    mean_diff = _safe_float(r.get("mean_diff"))
    ci_low    = _safe_float(r.get("ci_low"))
    ci_high   = _safe_float(r.get("ci_high"))
    p_adj     = _safe_float(r.get("p_adj_bh"))
    direction = _safe_str(r.get("direction"))
    bold = show_direction and direction == "a_better"
    return _format_cell(mean_diff, ci_low, ci_high, p_adj, alpha, bold, metric)


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
    """Rows = comparison sub-rows grouped by SPF.  Used when len(comparisons) > 1."""
    n_left = 1   # one "Comparison" column
    n_cols = n_left + len(metrics)
    col_spec = "L" * n_cols

    lines = [
        r"\begin{table}%[]",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular*}}{{\\tblwidth}}{{@{{}}{col_spec}@{{}}}}",
        r"\toprule",
        _format_row(_family_header_cells(metrics, families, n_left)),
        _format_row(_metric_header_cells(metrics, ["Comparison"])),
        r"\midrule",
    ]

    avail_spf = [s for s in SPF_ORDER if s in spf_list]
    for i, spf in enumerate(avail_spf):
        # SPF group header
        lines.append(
            f"\\multicolumn{{{n_cols}}}{{l}}{{\\textit{{SPF\\,=\\,{spf}}}}}\\\\"
        )
        for cmp in comparisons:
            disp = COMPARISON_DISPLAY.get(cmp, cmp.replace("_", " "))
            cells = [disp]
            for m in metrics:
                cells.append(_get_cell(df, cmp, spf, m, alpha, show_direction))
            lines.append(_format_row(cells))
        if i < len(avail_spf) - 1:
            lines.append(r"\addlinespace")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table}",
    ])
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
    """Rows = SPF values.  Used for single-comparison tables."""
    n_left = 1   # SPF column
    n_cols = n_left + len(metrics)
    col_spec = "L" * n_cols

    lines = [
        r"\begin{table}%[]",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular*}}{{\\tblwidth}}{{@{{}}{col_spec}@{{}}}}",
        r"\toprule",
        _format_row(_family_header_cells(metrics, families, n_left)),
        _format_row(_metric_header_cells(metrics, ["SPF"])),
        r"\midrule",
    ]

    avail_spf = [s for s in SPF_ORDER if s in spf_list]
    for spf in avail_spf:
        cells = [str(spf)]
        for m in metrics:
            cells.append(_get_cell(df, comparison, spf, m, alpha, show_direction))
        lines.append(_format_row(cells))

    lines.extend([
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table}",
    ])
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
        help="Comma-separated comparison names (from the 'comparison' column in the CSV).",
    )
    parser.add_argument(
        "--spf", default="",
        help="Comma-separated SPF values to include (default: all available).",
    )
    parser.add_argument(
        "--families", default="spatial,temporal,mc",
        help="Comma-separated metric families (default: spatial,temporal,mc).",
    )
    parser.add_argument(
        "--metrics", default="",
        help="Comma-separated metrics to include, overriding --families.",
    )
    parser.add_argument("--label", default="tab:significance")
    parser.add_argument("--caption", default="")
    parser.add_argument(
        "--show_direction", action="store_true", default=True,
        help="Bold cells where method_a is significantly better (default: on).",
    )
    parser.add_argument("--no_show_direction", dest="show_direction", action="store_false")
    parser.add_argument("--out", default=None,
                        help="Optional output .tex file path (default: stdout).")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    comparisons = [c.strip() for c in args.comparisons.split(",") if c.strip()]
    families    = [f.strip() for f in args.families.split(",") if f.strip()]
    metrics_override = [m.strip() for m in args.metrics.split(",") if m.strip()]
    spf_list = [int(s) for s in args.spf.split(",") if s.strip()] if args.spf else SPF_ORDER

    metrics = _metric_list(families, metrics_override)

    # Filter df to requested comparisons and SPFs
    df = df[df["comparison"].isin(comparisons) & df["spf"].isin(spf_list)].copy()

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
