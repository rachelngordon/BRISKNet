"""Generate a LaTeX significance table from results/significance_brisknet_vs_grasp.csv.

Matches the mri-journal booktabs/tabular* style used in make_inference_table.py.

Usage:
    python -m inference.plot.make_significance_table [options]
    python inference/plot/make_significance_table.py --csv results/significance_brisknet_vs_grasp.csv
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CSV = Path(__file__).resolve().parents[2] / "results" / "significance_brisknet_vs_grasp.csv"

# Decimal places for mean_diff and CI per metric
METRIC_DECIMALS = {
    "ssim":             3,
    "psnr":             2,
    "lpips":            3,
    "early_corr":       2,
    "early_mae":        2,
    "iauc10_err":       1,
    "ttae_sec":         2,
    "wash_in_slope_err":2,
    "dc_mae":           3,
    "ssdu_nmse":        3,
}

# Ordered metric display: (metric_key, latex_header, higher_is_better)
METRIC_DISPLAY = [
    # Spatial
    ("ssim",             "SSIM",                                            True),
    ("psnr",             "PSNR",                                            True),
    ("lpips",            "LPIPS",                                           False),
    # Temporal
    ("early_corr",       r"$\rho_{\mathrm{early}}$",                       True),
    ("early_mae",        r"$\mathrm{MAE}_{\mathrm{early}}$",               False),
    ("iauc10_err",       r"$\mathrm{iAUC}_{10}\ \mathrm{Err}$",           False),
    ("ttae_sec",         r"$t_{\mathrm{arr}}\ \mathrm{Err}$",              False),
    ("wash_in_slope_err",r"$\mathrm{MAE}_{\mathrm{wash\text{-}in}}$",      False),
    # MC
    ("dc_mae",           "DC MAE",                                          False),
    ("ssdu_nmse",        "SSDU NMSE",                                       False),
]

METRIC_FAMILIES = {
    "spatial":  ["ssim", "psnr", "lpips"],
    "temporal": ["early_corr", "early_mae", "iauc10_err", "ttae_sec", "wash_in_slope_err"],
    "mc":       ["dc_mae", "ssdu_nmse"],
}

FAMILY_LABELS = {
    "spatial":  "Spatial",
    "temporal": "Temporal",
    "mc":       "MC",
}

SPF_ORDER = [2, 4, 8, 16, 24, 36]


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
    """Format a float for LaTeX, adding minus sign as needed."""
    if math.isnan(v):
        return r"\cdot"
    return f"{v:.{decimals}f}"


def _format_delta_ci(
    mean_diff: float,
    ci_low: float,
    ci_high: float,
    p_adj: float,
    alpha: float,
    show_direction: bool,
    direction: str,
    metric: str,
) -> str:
    """Format as: mean_diff [ci_low, ci_high]^{stars}  (bold if brisknet better & sig)."""
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

    if show_direction and direction == "brisknet_better" and stars:
        cell = f"\\textbf{{{cell}}}"

    return cell


def _format_row(values: list[str]) -> str:
    return " & ".join(values) + r" \\"


def _pivot_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame()
    return df.pivot_table(index="spf", columns="metric", values=col, aggfunc="first")


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


def make_table(
    df: pd.DataFrame,
    alpha: float,
    caption: str,
    label: str,
    show_direction: bool,
    families: list[str],
) -> str:
    metrics = [m for m, _, _ in METRIC_DISPLAY if any(m in METRIC_FAMILIES[f] for f in families)]
    metric_info = {m: (hdr, hib) for m, hdr, hib in METRIC_DISPLAY}

    available = df[df["metric"].isin(metrics)]
    pivot_p    = _pivot_col(available, "p_adj_bh")
    pivot_dir  = _pivot_col(available, "direction")
    pivot_diff = _pivot_col(available, "mean_diff")
    pivot_lo   = _pivot_col(available, "ci_low")
    pivot_hi   = _pivot_col(available, "ci_high")

    spf_values = [spf for spf in SPF_ORDER if not pivot_p.empty and spf in pivot_p.index]

    # ---- column spec ----
    n_cols = 1 + len(metrics)  # SPF + metrics
    col_spec = "L" * n_cols

    lines = [
        r"\begin{table}%[]",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular*}}{{\\tblwidth}}{{@{{}}{col_spec}@{{}}}}",
        r"\toprule",
    ]

    # ---- family header row ----
    family_cells = [""]  # blank for SPF column
    for fam in families:
        fam_metrics = [m for m in metrics if m in METRIC_FAMILIES[fam]]
        if fam_metrics:
            family_cells.append(
                f"\\multicolumn{{{len(fam_metrics)}}}{{c}}{{{FAMILY_LABELS[fam]}}}"
            )
    lines.append(_format_row(family_cells))

    # ---- metric header row ----
    header = ["SPF"]
    for m in metrics:
        hdr, hib = metric_info[m]
        arrow = r" $\uparrow$" if hib else r" $\downarrow$"
        header.append(f"{hdr}{arrow}")
    lines.append(_format_row(header))
    lines.append(r"\midrule")

    # ---- data rows ----
    for spf in spf_values:
        row = [str(spf)]
        for m in metrics:
            def _get(piv: pd.DataFrame) -> float | str:
                if piv.empty or spf not in piv.index or m not in piv.columns:
                    return float("nan")
                return piv.at[spf, m]

            p    = _safe_float(_get(pivot_p))
            diff = _safe_float(_get(pivot_diff))
            lo   = _safe_float(_get(pivot_lo))
            hi   = _safe_float(_get(pivot_hi))
            dirn = _safe_str(_get(pivot_dir))

            row.append(_format_delta_ci(diff, lo, hi, p, alpha, show_direction, dirn, m))
        lines.append(_format_row(row))

    lines.extend([
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX significance table from significance CSV."
    )
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_CSV),
        help="Path to significance_brisknet_vs_grasp.csv.",
    )
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance threshold (default: 0.05).")
    parser.add_argument(
        "--caption",
        default=(
            r"Paired Wilcoxon signed-rank test, BRISKNet vs.\ GRASP (two-sided, $n{=}25$; "
            r"$n{=}15$ for temporal). "
            r"Each cell: $\Delta$ = mean difference (BRISKNet $-$ GRASP) with 95\% CI; "
            r"$^{*}p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$ (BH-FDR corrected within "
            r"each metric family). \textbf{Bold}: BRISKNet significantly better."
        ),
    )
    parser.add_argument("--label", default="tab:significance_brisknet_vs_grasp")
    parser.add_argument(
        "--families",
        default="spatial,temporal,mc",
        help="Comma-separated metric families to include (default: spatial,temporal,mc).",
    )
    parser.add_argument(
        "--show_direction",
        action="store_true",
        default=True,
        help="Bold-highlight cells where BRISKNet is significantly better (default: on).",
    )
    parser.add_argument(
        "--no_show_direction",
        dest="show_direction",
        action="store_false",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output .tex file path (default: print to stdout).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    families = [f.strip() for f in args.families.split(",") if f.strip()]

    table = make_table(
        df,
        alpha=args.alpha,
        caption=args.caption,
        label=args.label,
        show_direction=args.show_direction,
        families=families,
    )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(table + "\n")
        print(f"Saved to: {args.out}")
    else:
        print(table)


if __name__ == "__main__":
    main()
