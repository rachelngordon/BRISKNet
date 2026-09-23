"""Paired Wilcoxon signed-rank significance tests: BRISKNet vs GRASP (acceleration sweep).

Usage:
    python inference/significance_test.py [--alpha 0.05] [--out results/significance_brisknet_vs_grasp.csv]
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

REVISED_LOG = REPO_ROOT / "inference" / "test_inference_logs_mri_journal_revised.json"

TARGET_EXPS = [
    "ei_2spf_sampling_no_rebin_fop",
    "ei_4spf_sampling_no_rebin_fop",
    "ei_8spf_sampling_arrshift_fop",
    "ei_16spf_sampling_arrshift_fop",
    "ei_24spf_sampling_arrshift_fop",
    "ei_36spf_sampling_arrshift_fop",
]

# metric_name -> (dl_col, grasp_col, higher_is_better)
SPATIAL_METRICS = {
    "ssim":  ("dl_ssim",  "grasp_ssim",  True),
    "psnr":  ("dl_psnr",  "grasp_psnr",  True),
    "lpips": ("dl_lpips", "grasp_lpips", False),
}

TEMPORAL_METRICS = {
    "early_corr":        ("dl_all_early_corr",        "grasp_all_early_corr",        True),
    "early_mae":         ("dl_all_early_mae",          "grasp_all_early_mae",         False),
    "iauc10_err":        ("dl_all_iauc10_err",         "grasp_all_iauc10_err",        False),
    "ttae_sec":          ("dl_all_ttae_sec",           "grasp_all_ttae_sec",          False),
    "wash_in_slope_err": ("dl_all_wash_in_slope_err",  "grasp_all_wash_in_slope_err", False),
}

MC_METRICS = {
    "dc_mae":    ("dl_dc_mae",     "grasp_dc_mae",        False),
    "ssdu_nmse": ("raw_ssdu_nmse", "raw_grasp_ssdu_nmse", False),
}

METRIC_FAMILIES = {
    "spatial":  SPATIAL_METRICS,
    "temporal": TEMPORAL_METRICS,
    "mc":       MC_METRICS,
}


def load_log(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def resolve_experiments(log: list[dict]) -> tuple[dict, dict]:
    """Return (brisknet_by_name, grasp_by_spf) from a log."""
    brisknet = {}
    grasp_by_spf = {}
    for entry in log:
        if entry["type"] == "BRISKNet":
            name = entry.get("exp_name")
            if name:
                brisknet[name] = entry
        elif entry["type"] == "GRASP":
            spf = entry["spokes_per_frame"]
            grasp_by_spf[spf] = entry
    return brisknet, grasp_by_spf


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def check_nans(df: pd.DataFrame, sample_col: str, dl_col: str, grasp_col: str,
               exp_name: str, metric: str) -> tuple[pd.DataFrame, int, int]:
    """Report NaN samples and return the valid-row subset."""
    nan_mask = df[dl_col].isna() | df[grasp_col].isna()
    n_nan = int(nan_mask.sum())
    if n_nan > 0:
        nan_ids = df.loc[nan_mask, sample_col].tolist()
        print(f"  WARNING: {exp_name} / {metric}: NaN in samples {nan_ids} "
              f"— proceeding with n={len(df) - n_nan}")
    return df[~nan_mask].copy(), len(df) - n_nan, n_nan


def rank_biserial(W: float, n: int) -> float:
    """Rank-biserial correlation from Wilcoxon W statistic."""
    denom = n * (n + 1)
    if denom == 0:
        return float("nan")
    return float(1.0 - (2.0 * W) / denom)


def run_wilcoxon(dl_vals: np.ndarray, grasp_vals: np.ndarray
                 ) -> tuple[float, float]:
    """Return (W_stat, p_raw). Returns (nan, nan) if insufficient data."""
    if len(dl_vals) < 2:
        return float("nan"), float("nan")
    try:
        result = wilcoxon(dl_vals, grasp_vals, alternative="two-sided")
        return float(result.statistic), float(result.pvalue)
    except ValueError:
        # All differences are zero — test is undefined
        return float("nan"), float("nan")


def mean_diff_ci(dl_vals: np.ndarray, grasp_vals: np.ndarray,
                 alpha: float) -> tuple[float, float]:
    """95% CI for paired mean difference using t-distribution."""
    diffs = dl_vals - grasp_vals
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan")
    se = float(np.std(diffs, ddof=1)) / np.sqrt(n)
    t_crit = t_dist.ppf(1.0 - alpha / 2.0, df=n - 1)
    mean_d = float(np.mean(diffs))
    return mean_d - t_crit * se, mean_d + t_crit * se


def direction(mean_diff: float, higher_is_better: bool) -> str:
    if np.isnan(mean_diff):
        return "unknown"
    if higher_is_better:
        return "brisknet_better" if mean_diff > 0 else "grasp_better"
    else:
        return "brisknet_better" if mean_diff < 0 else "grasp_better"


def apply_fdr(rows: list[dict], family: str, alpha: float) -> list[dict]:
    """Apply BH-FDR within a metric family; update p_adj_bh in-place."""
    family_rows = [r for r in rows if r["metric_family"] == family]
    p_raws = [r["p_raw"] for r in family_rows]
    valid = [not np.isnan(p) for p in p_raws]

    if not any(valid):
        for r in family_rows:
            r["p_adj_bh"] = float("nan")
        return rows

    p_valid = [p for p, v in zip(p_raws, valid) if v]
    _, p_corrected, _, _ = multipletests(p_valid, alpha=alpha, method="fdr_bh")

    corrected_iter = iter(p_corrected)
    for r, v in zip(family_rows, valid):
        r["p_adj_bh"] = float(next(corrected_iter)) if v else float("nan")

    return rows


def print_summary(rows: list[dict], alpha: float) -> None:
    families = ["spatial", "temporal", "mc"]
    for family in families:
        frows = [r for r in rows if r["metric_family"] == family]
        if not frows:
            continue
        print(f"\n{'=' * 80}")
        print(f"  {family.upper()} METRICS  (BH-FDR α={alpha})")
        print(f"{'=' * 80}")
        metrics = sorted({r["metric"] for r in frows})
        exps = sorted({r["exp_name"] for r in frows}, key=lambda e: TARGET_EXPS.index(e) if e in TARGET_EXPS else 99)
        header = f"{'metric':<22} {'exp':<38} {'spf':>4}  {'mean_dl':>9} {'mean_gr':>9} {'mean_diff':>10}  {'p_raw':>8} {'p_adj':>8}  {'r':>6}  dir"
        print(header)
        print("-" * len(header))
        for metric in metrics:
            for r in [r for r in frows if r["metric"] == metric]:
                sig = "*" if (not np.isnan(r["p_adj_bh"]) and r["p_adj_bh"] < alpha) else " "
                print(
                    f"{metric:<22} {r['exp_name']:<38} {r['spf']:>4}  "
                    f"{r['mean_brisknet']:>9.4f} {r['mean_grasp']:>9.4f} {r['mean_diff']:>10.4f}  "
                    f"{r['p_raw']:>8.4f} {r['p_adj_bh']:>8.4f}  "
                    f"{r['effect_r']:>6.3f}  {r['direction']}{sig}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="BRISKNet vs GRASP significance tests.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", type=str,
                        default=str(REPO_ROOT / "results" / "significance_brisknet_vs_grasp.csv"))
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Step 1: Resolve experiments
    # ------------------------------------------------------------------ #
    log_name = REVISED_LOG.name
    log = load_log(REVISED_LOG)
    brisknet_by_name, grasp_by_spf = resolve_experiments(log)

    print(f"\nLog: {log_name}")
    print(f"Resolving {len(TARGET_EXPS)} target experiments:")
    resolved = []
    for name in TARGET_EXPS:
        if name in brisknet_by_name:
            entry = brisknet_by_name[name]
            spf = entry["spokes_per_frame"]
            idir = entry.get("inference_dir", "")
            grasp_entry = grasp_by_spf.get(spf, {})
            grasp_lamda = grasp_entry.get("grasp_lamda") or grasp_entry.get("grasp_lamdas")
            resolved.append({
                "exp_name": name,
                "spf": spf,
                "inference_dir": idir,
                "log_source": log_name,
                "grasp_lamda": grasp_lamda,
            })
            print(f"  [FOUND]   {name}  spf={spf}  grasp_lamda={grasp_lamda}")
        else:
            print(f"  [MISSING] {name} — not in {log_name}, skipping")

    if not resolved:
        print("No experiments resolved. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Steps 2–4: Load data, report NaNs, run Wilcoxon
    # ------------------------------------------------------------------ #
    rows: list[dict] = []

    for exp in resolved:
        name = exp["exp_name"]
        idir = exp["inference_dir"]
        spf = exp["spf"]

        metrics_csv = os.path.join(idir, "metrics.csv")
        temporal_csv = os.path.join(idir, "metrics_temporal_malignant_all.csv")

        if not os.path.exists(metrics_csv):
            print(f"\n[SKIP] {name}: metrics.csv not found at {metrics_csv}")
            continue

        df_metrics = load_csv(metrics_csv)

        has_temporal = os.path.exists(temporal_csv)
        if not has_temporal:
            print(f"\n[WARN] {name}: metrics_temporal_malignant_all.csv not found — skipping temporal metrics")
        else:
            df_temporal = load_csv(temporal_csv)

        print(f"\n--- {name}  (spf={spf}) ---")

        for family, metric_dict in METRIC_FAMILIES.items():
            if family == "temporal" and not has_temporal:
                continue

            df = df_metrics if family in ("spatial", "mc") else df_temporal

            for metric, (dl_col, grasp_col, hib) in metric_dict.items():
                if dl_col not in df.columns or grasp_col not in df.columns:
                    print(f"  SKIP {family}/{metric}: columns not found in CSV")
                    continue

                valid_df, n_valid, n_nan = check_nans(
                    df, "sample", dl_col, grasp_col, name, metric
                )

                if n_valid < 2:
                    print(f"  SKIP {family}/{metric}: only {n_valid} valid samples")
                    rows.append({
                        "log_source": exp["log_source"],
                        "exp_name": name,
                        "spf": spf,
                        "inference_dir": idir,
                        "grasp_lamda": exp["grasp_lamda"],
                        "metric_family": family,
                        "metric": metric,
                        "n_valid": n_valid,
                        "n_nan": n_nan,
                        "mean_brisknet": float("nan"),
                        "mean_grasp": float("nan"),
                        "mean_diff": float("nan"),
                        "ci_low": float("nan"),
                        "ci_high": float("nan"),
                        "W_stat": float("nan"),
                        "p_raw": float("nan"),
                        "p_adj_bh": float("nan"),
                        "effect_r": float("nan"),
                        "direction": "unknown",
                    })
                    continue

                dl_vals = valid_df[dl_col].to_numpy(dtype=float)
                grasp_vals = valid_df[grasp_col].to_numpy(dtype=float)

                mean_dl = float(np.mean(dl_vals))
                mean_gr = float(np.mean(grasp_vals))
                mean_diff = mean_dl - mean_gr
                ci_low, ci_high = mean_diff_ci(dl_vals, grasp_vals, args.alpha)

                W, p_raw = run_wilcoxon(dl_vals, grasp_vals)
                r = rank_biserial(W, n_valid)
                dirn = direction(mean_diff, hib)

                rows.append({
                    "log_source": exp["log_source"],
                    "exp_name": name,
                    "spf": spf,
                    "inference_dir": idir,
                    "grasp_lamda": exp["grasp_lamda"],
                    "metric_family": family,
                    "metric": metric,
                    "n_valid": n_valid,
                    "n_nan": n_nan,
                    "mean_brisknet": mean_dl,
                    "mean_grasp": mean_gr,
                    "mean_diff": mean_diff,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "W_stat": W,
                    "p_raw": p_raw,
                    "p_adj_bh": float("nan"),
                    "effect_r": r,
                    "direction": dirn,
                })

    # ------------------------------------------------------------------ #
    # Step 5: BH-FDR correction within each metric family
    # ------------------------------------------------------------------ #
    for family in METRIC_FAMILIES:
        rows = apply_fdr(rows, family, args.alpha)

    # ------------------------------------------------------------------ #
    # Step 6: Output
    # ------------------------------------------------------------------ #
    print_summary(rows, args.alpha)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}  ({len(df_out)} rows)")


if __name__ == "__main__":
    main()
