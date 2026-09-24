"""Paired Wilcoxon signed-rank significance tests for all paper comparisons.

Comparisons covered:
  - Acceleration sweep:  BRISKNet vs GRASP, SSDU vs GRASP, BRISKNet vs SSDU
  - EI ablation (8 SPF): EI+MC vs MC-only
  - Temporal ablation:   full / no-arrival-shift / no-rebin  vs  diffeo-only baseline

Usage:
    python inference/significance_test.py [--alpha 0.05]
    python inference/significance_test.py --comparisons accel_sweep
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, t as t_dist
from statsmodels.stats.multitest import multipletests

REPO_ROOT = Path(__file__).resolve().parents[1]
REVISED_LOG  = REPO_ROOT / "inference" / "test_inference_logs_mri_journal_revised.json"
ORIG_LOG     = REPO_ROOT / "inference" / "test_inference_logs_mri_journal.json"
PLOT_LOG     = REPO_ROOT / "inference" / "test_inference_logs.json"

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

# Within-exp comparisons (BRISKNet or SSDU vs GRASP):
#   metric -> (dl_col, grasp_col, higher_is_better)
WITHIN_SPATIAL = {
    "ssim":  ("dl_ssim",  "grasp_ssim",  True),
    "psnr":  ("dl_psnr",  "grasp_psnr",  True),
    "lpips": ("dl_lpips", "grasp_lpips", False),
}
WITHIN_TEMPORAL = {
    "early_corr":        ("dl_all_early_corr",       "grasp_all_early_corr",       True),
    "early_mae":         ("dl_all_early_mae",          "grasp_all_early_mae",        False),
    "iauc10_err":        ("dl_all_iauc10_err",         "grasp_all_iauc10_err",       False),
    "ttae_sec":          ("dl_all_ttae_sec",           "grasp_all_ttae_sec",         False),
    "wash_in_slope_err": ("dl_all_wash_in_slope_err",  "grasp_all_wash_in_slope_err",False),
}
WITHIN_MC = {
    "dc_mae":    ("dl_dc_mae",     "grasp_dc_mae",        False),
    "ssdu_nmse": ("raw_ssdu_nmse", "raw_grasp_ssdu_nmse", False),
}

# Cross-exp comparisons (both models evaluated on same test set):
#   metric -> (shared_col, higher_is_better)
#   Both experiments use the same column name; we load each from its own CSV.
CROSS_SPATIAL = {
    "ssim":  ("dl_ssim",  True),
    "psnr":  ("dl_psnr",  True),
    "lpips": ("dl_lpips", False),
}
CROSS_TEMPORAL = {
    "early_corr":        ("dl_all_early_corr",       True),
    "early_mae":         ("dl_all_early_mae",         False),
    "iauc10_err":        ("dl_all_iauc10_err",        False),
    "ttae_sec":          ("dl_all_ttae_sec",          False),
    "wash_in_slope_err": ("dl_all_wash_in_slope_err", False),
}
CROSS_MC = {
    "dc_mae":    ("dl_dc_mae",     False),
    "ssdu_nmse": ("raw_ssdu_nmse", False),
}

WITHIN_FAMILIES = {"spatial": WITHIN_SPATIAL, "temporal": WITHIN_TEMPORAL, "mc": WITHIN_MC}
CROSS_FAMILIES  = {"spatial": CROSS_SPATIAL,  "temporal": CROSS_TEMPORAL,  "mc": CROSS_MC}

# ---------------------------------------------------------------------------
# Comparison specifications
# ---------------------------------------------------------------------------
# Each "pair" entry: (exp_a, exp_b_or_None, spf)
#   exp_b is None  → within-exp vs GRASP (use grasp_* cols in exp_a's CSV)
#   exp_b is str   → cross-exp (load both CSVs, use dl_* col from each)

COMPARISONS = [

    # ---- Acceleration sweep: BRISKNet vs GRASP ----------------------------
    {
        "group":    "accel_sweep",
        "name":     "BRISKNet_vs_GRASP",
        "method_a": "BRISKNet",
        "method_b": "GRASP",
        "mode":     "within",
        "pairs": [
            ("ei_2spf_sampling_no_rebin_fop",   None, 2),
            ("ei_4spf_sampling_no_rebin_fop",   None, 4),
            ("ei_8spf_sampling_arrshift_fop",   None, 8),
            ("ei_16spf_sampling_arrshift_fop",  None, 16),
            ("ei_24spf_sampling_arrshift_fop",  None, 24),
            ("ei_36spf_sampling_arrshift_fop",  None, 36),
        ],
    },

    # ---- Acceleration sweep: SSDU vs GRASP --------------------------------
    {
        "group":    "accel_sweep",
        "name":     "SSDU_vs_GRASP",
        "method_a": "SSDU",
        "method_b": "GRASP",
        "mode":     "within",
        "pairs": [
            ("ssdu_8spf_sampling",  None, 8),
            ("ssdu_16spf_sampling", None, 16),
            ("ssdu_24spf_sampling", None, 24),
            ("ssdu_36spf_sampling", None, 36),
        ],
    },

    # ---- Acceleration sweep: BRISKNet vs SSDU -----------------------------
    {
        "group":    "accel_sweep",
        "name":     "BRISKNet_vs_SSDU",
        "method_a": "BRISKNet",
        "method_b": "SSDU",
        "mode":     "cross",
        "pairs": [
            ("ei_8spf_sampling_arrshift_fop",  "ssdu_8spf_sampling",  8),
            ("ei_16spf_sampling_arrshift_fop", "ssdu_16spf_sampling", 16),
            ("ei_24spf_sampling_arrshift_fop", "ssdu_24spf_sampling", 24),
            ("ei_36spf_sampling_arrshift_fop", "ssdu_36spf_sampling", 36),
        ],
    },

    # ---- EI ablation: EI+MC vs MC-only ------------------------------------
    {
        "group":    "ei_ablation",
        "name":     "EI_vs_MC",
        "method_a": "EI+MC",
        "method_b": "MC-only",
        "mode":     "cross",
        "pairs": [
            ("ei_8spf_sampling_arrshift_fop", "mc_8spf_slice_sampling", 8),
        ],
    },

    # ---- Temporal transform ablation vs diffeo-only baseline --------------
    # Experiments match those used in the temporal ablation figures exactly.
    # Baseline: ei_8spf_no_temporal_fop  (temporal_transform=none, diffeo spatial only)
    # Arr Shift: ei_8spf_fop_arrival_shift  (temporal_transform=arrival_shift)
    # Enh Scale: ei_8spf_fop_enh_scale     (temporal_transform=enh_scale)
    # Arr Shift + Enh Scale: ei_8spf_no_rebin_fop (temporal_transform=arrival_shift_enh_scale)
    {
        "group":    "temporal_ablation",
        "name":     "arr_shift_vs_diffeo_only",
        "method_a": "Arr Shift",
        "method_b": "Diffeo Only",
        "mode":     "cross",
        "pairs": [
            ("ei_8spf_fop_arrival_shift",  "ei_8spf_no_temporal_fop",  8),
            ("ei_36spf_sampling_arrshift_fop", "ei_36spf_no_temporal_fop", 36),
        ],
    },
    {
        "group":    "temporal_ablation",
        "name":     "enh_scale_vs_diffeo_only",
        "method_a": "Enh Scale",
        "method_b": "Diffeo Only",
        "mode":     "cross",
        "pairs": [
            ("ei_8spf_fop_enh_scale", "ei_8spf_no_temporal_fop", 8),
        ],
    },
    {
        "group":    "temporal_ablation",
        "name":     "arr_shift_enh_scale_vs_diffeo_only",
        "method_a": "Arr Shift + Enh Scale",
        "method_b": "Diffeo Only",
        "mode":     "cross",
        "pairs": [
            ("ei_8spf_no_rebin_fop", "ei_8spf_no_temporal_fop", 8),
        ],
    },
]

# ---------------------------------------------------------------------------
# Log loading / inference-dir lookup
# ---------------------------------------------------------------------------

def load_logs() -> dict[str, dict]:
    """Return exp_name -> entry dict, preferring revised > mri_journal > plot log."""
    index: dict[str, dict] = {}
    for log_path in (PLOT_LOG, ORIG_LOG, REVISED_LOG):   # later entries overwrite
        with open(log_path) as f:
            entries = json.load(f)
        for e in entries:
            name = e.get("exp_name")
            if name and e.get("inference_dir"):
                e["_log_source"] = log_path.name
                index[name] = e
    return index


def get_idir(exp_index: dict[str, dict], exp_name: str) -> tuple[str, str]:
    """Return (inference_dir, log_source) or raise."""
    entry = exp_index.get(exp_name)
    if entry is None:
        raise KeyError(f"Experiment '{exp_name}' not found in any log.")
    return entry["inference_dir"], entry["_log_source"]


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def check_nans(df: pd.DataFrame, col_a: str, col_b: str,
               label: str, metric: str) -> tuple[pd.DataFrame, int, int]:
    nan_mask = df[col_a].isna() | df[col_b].isna()
    n_nan = int(nan_mask.sum())
    if n_nan > 0:
        ids = df.loc[nan_mask].index.tolist()
        print(f"  WARNING {label}/{metric}: NaN rows {ids} — n_valid={len(df)-n_nan}")
    return df[~nan_mask].copy(), len(df) - n_nan, n_nan


def run_wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if len(a) < 2:
        return float("nan"), float("nan")
    try:
        res = wilcoxon(a, b, alternative="two-sided")
        return float(res.statistic), float(res.pvalue)
    except ValueError:
        return float("nan"), float("nan")


def mean_diff_ci(a: np.ndarray, b: np.ndarray, alpha: float) -> tuple[float, float]:
    diffs = a - b
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan")
    se = float(np.std(diffs, ddof=1)) / np.sqrt(n)
    t_crit = t_dist.ppf(1.0 - alpha / 2.0, df=n - 1)
    m = float(np.mean(diffs))
    return m - t_crit * se, m + t_crit * se


def rank_biserial(W: float, n: int) -> float:
    denom = n * (n + 1)
    return float("nan") if denom == 0 else float(1.0 - (2.0 * W) / denom)


def direction_label(mean_diff: float, higher_is_better: bool) -> str:
    if np.isnan(mean_diff):
        return "unknown"
    if higher_is_better:
        return "a_better" if mean_diff > 0 else "b_better"
    else:
        return "a_better" if mean_diff < 0 else "b_better"


def apply_fdr(rows: list[dict], group_key: tuple, alpha: float) -> None:
    """BH-FDR correction in-place for rows matching group_key=(group, name, family)."""
    subset = [r for r in rows
              if (r["comparison_group"], r["comparison"], r["metric_family"]) == group_key]
    p_raws = [r["p_raw"] for r in subset]
    valid = [not np.isnan(p) for p in p_raws]
    if not any(valid):
        for r in subset:
            r["p_adj_bh"] = float("nan")
        return
    p_valid = [p for p, v in zip(p_raws, valid) if v]
    _, corrected, _, _ = multipletests(p_valid, alpha=alpha, method="fdr_bh")
    it = iter(corrected)
    for r, v in zip(subset, valid):
        r["p_adj_bh"] = float(next(it)) if v else float("nan")


# ---------------------------------------------------------------------------
# Per-pair test runner
# ---------------------------------------------------------------------------

NAN_ROW_KEYS = ["n_valid", "n_nan", "mean_a", "mean_b", "mean_diff",
                "ci_low", "ci_high", "W_stat", "p_raw", "p_adj_bh", "effect_r"]

def _nan_row(base: dict) -> dict:
    return {**base, **{k: float("nan") for k in NAN_ROW_KEYS}, "direction": "unknown"}


def run_within_pair(
    exp_a: str,
    spf: int,
    idir_a: str,
    log_src_a: str,
    comp: dict,
    alpha: float,
) -> list[dict]:
    """Within-exp tests: load exp_a CSVs, compare dl_* vs grasp_* columns."""
    rows = []
    metrics_csv  = os.path.join(idir_a, "metrics.csv")
    temporal_csv = os.path.join(idir_a, "metrics_temporal_malignant_all.csv")

    if not os.path.exists(metrics_csv):
        print(f"  [SKIP] {exp_a}: metrics.csv not found")
        return rows

    df_m = pd.read_csv(metrics_csv)
    has_temp = os.path.exists(temporal_csv)
    if not has_temp:
        print(f"  [WARN] {exp_a}: temporal CSV not found — skipping temporal metrics")
    else:
        df_t = pd.read_csv(temporal_csv)

    label = f"{exp_a}(spf={spf})"

    for family, metric_dict in WITHIN_FAMILIES.items():
        if family == "temporal" and not has_temp:
            continue
        df = df_m if family in ("spatial", "mc") else df_t

        for metric, (col_a, col_b, hib) in metric_dict.items():
            base = {
                "comparison_group": comp["group"],
                "comparison":       comp["name"],
                "method_a":         comp["method_a"],
                "method_b":         comp["method_b"],
                "exp_a":            exp_a,
                "exp_b":            "GRASP",
                "spf":              spf,
                "log_source_a":     log_src_a,
                "log_source_b":     "—",
                "metric_family":    family,
                "metric":           metric,
                "p_adj_bh":         float("nan"),
            }
            if col_a not in df.columns or col_b not in df.columns:
                print(f"  SKIP {family}/{metric}: columns missing")
                rows.append(_nan_row(base))
                continue

            valid_df, n_valid, n_nan = check_nans(df, col_a, col_b, label, metric)
            if n_valid < 2:
                rows.append({**_nan_row(base), "n_valid": n_valid, "n_nan": n_nan})
                continue

            a_vals = valid_df[col_a].to_numpy(dtype=float)
            b_vals = valid_df[col_b].to_numpy(dtype=float)
            mean_a = float(np.mean(a_vals))
            mean_b = float(np.mean(b_vals))
            diff   = mean_a - mean_b
            ci_lo, ci_hi = mean_diff_ci(a_vals, b_vals, alpha)
            W, p_raw = run_wilcoxon(a_vals, b_vals)
            r = rank_biserial(W, n_valid)

            rows.append({**base,
                "n_valid":    n_valid,
                "n_nan":      n_nan,
                "mean_a":     mean_a,
                "mean_b":     mean_b,
                "mean_diff":  diff,
                "ci_low":     ci_lo,
                "ci_high":    ci_hi,
                "W_stat":     W,
                "p_raw":      p_raw,
                "effect_r":   r,
                "direction":  direction_label(diff, hib),
            })
    return rows


def run_cross_pair(
    exp_a: str,
    exp_b: str,
    spf: int,
    idir_a: str,
    idir_b: str,
    log_src_a: str,
    log_src_b: str,
    comp: dict,
    alpha: float,
) -> list[dict]:
    """Cross-exp tests: load both CSVs, compare dl_* col from each, merge by sample."""
    rows = []
    m_csv_a = os.path.join(idir_a, "metrics.csv")
    m_csv_b = os.path.join(idir_b, "metrics.csv")
    t_csv_a = os.path.join(idir_a, "metrics_temporal_malignant_all.csv")
    t_csv_b = os.path.join(idir_b, "metrics_temporal_malignant_all.csv")

    if not os.path.exists(m_csv_a):
        print(f"  [SKIP] {exp_a}: metrics.csv not found")
        return rows
    if not os.path.exists(m_csv_b):
        print(f"  [SKIP] {exp_b}: metrics.csv not found")
        return rows

    df_ma = pd.read_csv(m_csv_a)
    df_mb = pd.read_csv(m_csv_b)
    has_temp = os.path.exists(t_csv_a) and os.path.exists(t_csv_b)
    if not has_temp:
        print(f"  [WARN] temporal CSV missing for one of: {exp_a}, {exp_b}")
    else:
        df_ta = pd.read_csv(t_csv_a)
        df_tb = pd.read_csv(t_csv_b)

    label = f"{comp['name']}(spf={spf})"

    for family, metric_dict in CROSS_FAMILIES.items():
        if family == "temporal" and not has_temp:
            continue

        if family in ("spatial", "mc"):
            df_a, df_b = df_ma, df_mb
        else:
            df_a, df_b = df_ta, df_tb

        for metric, (col, hib) in metric_dict.items():
            base = {
                "comparison_group": comp["group"],
                "comparison":       comp["name"],
                "method_a":         comp["method_a"],
                "method_b":         comp["method_b"],
                "exp_a":            exp_a,
                "exp_b":            exp_b,
                "spf":              spf,
                "log_source_a":     log_src_a,
                "log_source_b":     log_src_b,
                "metric_family":    family,
                "metric":           metric,
                "p_adj_bh":         float("nan"),
            }
            if col not in df_a.columns or col not in df_b.columns:
                print(f"  SKIP {family}/{metric}: column '{col}' missing")
                rows.append(_nan_row(base))
                continue

            # Merge by sample index (same test set, same row order)
            merged = df_a[["sample", col]].rename(columns={col: "val_a"}).merge(
                df_b[["sample", col]].rename(columns={col: "val_b"}),
                on="sample", how="inner",
            )
            valid_df, n_valid, n_nan = check_nans(merged, "val_a", "val_b", label, metric)
            if n_valid < 2:
                rows.append({**_nan_row(base), "n_valid": n_valid, "n_nan": n_nan})
                continue

            a_vals = valid_df["val_a"].to_numpy(dtype=float)
            b_vals = valid_df["val_b"].to_numpy(dtype=float)
            mean_a = float(np.mean(a_vals))
            mean_b = float(np.mean(b_vals))
            diff   = mean_a - mean_b
            ci_lo, ci_hi = mean_diff_ci(a_vals, b_vals, alpha)
            W, p_raw = run_wilcoxon(a_vals, b_vals)
            r = rank_biserial(W, n_valid)

            rows.append({**base,
                "n_valid":    n_valid,
                "n_nan":      n_nan,
                "mean_a":     mean_a,
                "mean_b":     mean_b,
                "mean_diff":  diff,
                "ci_low":     ci_lo,
                "ci_high":    ci_hi,
                "W_stat":     W,
                "p_raw":      p_raw,
                "effect_r":   r,
                "direction":  direction_label(diff, hib),
            })
    return rows


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(rows: list[dict], alpha: float) -> None:
    groups = sorted({(r["comparison_group"], r["comparison"]) for r in rows})
    for grp, cmp in groups:
        families = sorted({r["metric_family"] for r in rows
                           if r["comparison_group"] == grp and r["comparison"] == cmp})
        print(f"\n{'='*90}")
        print(f"  {grp} / {cmp}  (BH-FDR α={alpha})")
        print(f"{'='*90}")
        for family in families:
            frows = [r for r in rows
                     if r["comparison_group"] == grp
                     and r["comparison"] == cmp
                     and r["metric_family"] == family]
            if not frows:
                continue
            print(f"\n  [{family.upper()}]")
            hdr = (f"  {'metric':<22} {'spf':>4}  {'mean_a':>9} {'mean_b':>9} "
                   f"{'diff':>9}  {'95% CI':<22}  {'p_raw':>8} {'p_adj':>8}  {'r':>6}  dir")
            print(hdr)
            print("  " + "-"*(len(hdr)-2))
            for r in sorted(frows, key=lambda x: (x["metric"], x["spf"])):
                sig = "*" if (not np.isnan(r["p_adj_bh"]) and r["p_adj_bh"] < alpha) else " "
                ci_str = (f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
                          if not np.isnan(r.get("ci_low", float("nan"))) else "")
                print(
                    f"  {r['metric']:<22} {r['spf']:>4}  "
                    f"{r['mean_a']:>9.4f} {r['mean_b']:>9.4f} {r['mean_diff']:>9.4f}  "
                    f"{ci_str:<22}  {r['p_raw']:>8.4f} {r['p_adj_bh']:>8.4f}  "
                    f"{r['effect_r']:>6.3f}  {r['direction']}{sig}"
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run all paper significance tests.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--comparisons", default="",
                        help="Comma-separated comparison groups to run (default: all). "
                             "Options: accel_sweep, ei_ablation, temporal_ablation")
    parser.add_argument("--out", default=str(
        REPO_ROOT / "results" / "significance_all_comparisons.csv"))
    args = parser.parse_args()

    requested_groups = {g.strip() for g in args.comparisons.split(",") if g.strip()}

    exp_index = load_logs()
    print(f"Loaded {len(exp_index)} experiments from logs.")

    all_rows: list[dict] = []

    for comp in COMPARISONS:
        if requested_groups and comp["group"] not in requested_groups:
            continue

        print(f"\n{'─'*70}")
        print(f"  {comp['group']} / {comp['name']}")
        print(f"{'─'*70}")

        for pair in comp["pairs"]:
            exp_a, exp_b_name, spf = pair
            print(f"\n  pair: {exp_a}  vs  {exp_b_name or 'GRASP'}  (spf={spf})")

            try:
                idir_a, log_src_a = get_idir(exp_index, exp_a)
            except KeyError as e:
                print(f"  [SKIP] {e}")
                continue

            if comp["mode"] == "within":
                pair_rows = run_within_pair(exp_a, spf, idir_a, log_src_a, comp, args.alpha)
            else:
                try:
                    idir_b, log_src_b = get_idir(exp_index, exp_b_name)
                except KeyError as e:
                    print(f"  [SKIP] {e}")
                    continue
                pair_rows = run_cross_pair(
                    exp_a, exp_b_name, spf,
                    idir_a, idir_b, log_src_a, log_src_b,
                    comp, args.alpha,
                )

            all_rows.extend(pair_rows)

    # ---- BH-FDR correction within (comparison_group, comparison, metric_family) ----
    fdr_keys = {(r["comparison_group"], r["comparison"], r["metric_family"])
                for r in all_rows}
    for key in fdr_keys:
        apply_fdr(all_rows, key, args.alpha)

    # ---- Print summary ----
    print_summary(all_rows, args.alpha)

    # ---- Save CSV ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"\nSaved {len(all_rows)} rows → {out_path}")


if __name__ == "__main__":
    main()
