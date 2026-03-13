#!/usr/bin/env python3
"""Compare aggregate (mean) metrics between two CSVs. Run: python3 -m inference.compare_metrics --help"""

import argparse
import csv
import math
from typing import Dict, List, Tuple


def load_metrics(csv_path: str) -> Tuple[List[str], Dict[str, List[float]]]:
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"No header found in {csv_path}")
        metrics = {k: [] for k in reader.fieldnames if k != "sample"}

        for row in reader:
            for key in metrics.keys():
                val = row.get(key, "")
                if val is None or val == "":
                    metrics[key].append(math.nan)
                else:
                    try:
                        metrics[key].append(float(val))
                    except ValueError:
                        metrics[key].append(math.nan)

    return list(metrics.keys()), metrics


def aggregate_means(metrics: Dict[str, List[float]]) -> Dict[str, float]:
    means = {}
    for key, vals in metrics.items():
        clean = [v for v in vals if v is not None and not math.isnan(v)]
        means[key] = sum(clean) / len(clean) if clean else math.nan
    return means


def format_val(v: float, ndigits: int = 6) -> str:
    if v is None or math.isnan(v):
        return "NA"
    return f"{v:.{ndigits}f}"


def main():
    parser = argparse.ArgumentParser(
        description="Compare aggregate (mean) metrics between two CSVs."
    )
    parser.add_argument("--csv1", required=True, help="Path to first CSV.")
    parser.add_argument("--label1", required=True, help="Column label for first CSV.")
    parser.add_argument("--csv2", required=True, help="Path to second CSV.")
    parser.add_argument("--label2", required=True, help="Column label for second CSV.")
    parser.add_argument("--ndigits", type=int, default=6, help="Decimal places to show.")
    args = parser.parse_args()

    keys1, metrics1 = load_metrics(args.csv1)
    keys2, metrics2 = load_metrics(args.csv2)

    # Use intersection to avoid mismatches; preserve csv1 order.
    common_keys = [k for k in keys1 if k in set(keys2)]
    if not common_keys:
        raise ValueError("No matching metric columns between the two CSVs.")

    means1 = aggregate_means(metrics1)
    means2 = aggregate_means(metrics2)

    # Build a simple table.
    col0 = "metric"
    col1 = args.label1
    col2 = args.label2

    # Compute column widths
    w0 = max(len(col0), max(len(k) for k in common_keys))
    w1 = max(len(col1), max(len(format_val(means1[k], args.ndigits)) for k in common_keys))
    w2 = max(len(col2), max(len(format_val(means2[k], args.ndigits)) for k in common_keys))

    # Header
    print(f"{col0:<{w0}}  {col1:>{w1}}  {col2:>{w2}}")
    print(f"{'-'*w0}  {'-'*w1}  {'-'*w2}")

    # Rows
    for k in common_keys:
        v1 = format_val(means1[k], args.ndigits)
        v2 = format_val(means2[k], args.ndigits)
        print(f"{k:<{w0}}  {v1:>{w1}}  {v2:>{w2}}")


if __name__ == "__main__":
    main()
