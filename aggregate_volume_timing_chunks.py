#!/usr/bin/env python3
"""Aggregate chunked full-volume timing JSON outputs.

Combines chunk-level mean/std summaries into global per-SPF per-method summaries
using pooled-variance formulas.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ChunkStat:
    n: int
    mean: float
    std: float


def _pool_mean_std(chunks: List[ChunkStat]) -> Tuple[float | None, float | None, int]:
    if not chunks:
        return None, None, 0
    n_total = sum(c.n for c in chunks)
    if n_total <= 0:
        return None, None, 0

    mean_total = sum(c.n * c.mean for c in chunks) / float(n_total)

    if n_total == 1:
        return mean_total, 0.0, 1

    ss_total = 0.0
    for c in chunks:
        if c.n <= 0:
            continue
        within = (c.n - 1) * (c.std ** 2) if c.n > 1 else 0.0
        between = c.n * ((c.mean - mean_total) ** 2)
        ss_total += within + between

    var_total = ss_total / float(n_total - 1)
    std_total = math.sqrt(max(0.0, var_total))
    return mean_total, std_total, n_total


def _read_rows(paths: List[str]) -> List[dict]:
    rows: List[dict] = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        if isinstance(payload, list):
            for row in payload:
                if isinstance(row, dict):
                    row["_source_path"] = p
                    rows.append(row)
        elif isinstance(payload, dict):
            payload["_source_path"] = p
            rows.append(payload)
    return rows


def _add_stat(bag: Dict[Tuple[str, int, str], List[ChunkStat]], method: str, spf: int, metric: str, row: dict) -> None:
    mean_key = f"{method}_volume_{metric}_mean_s"
    std_key = f"{method}_volume_{metric}_std_s"
    if mean_key not in row:
        return
    n = int(row.get("num_volumes", 0) or 0)
    mean = row.get(mean_key)
    std = row.get(std_key)
    if n <= 0 or mean is None:
        return
    bag.setdefault((method, spf, metric), []).append(
        ChunkStat(n=n, mean=float(mean), std=float(std or 0.0))
    )


def _fmt_ms(mean_s: float | None, std_s: float | None) -> str:
    if mean_s is None:
        return "n/a"
    m = mean_s / 60.0
    if std_s is None:
        return f"{m:.2f}"
    return f"{m:.2f} +/- {std_s/60.0:.2f}"


def _fmt_x(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.2f}x"


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate chunked volume timing outputs")
    ap.add_argument("--brisk-glob", default="logs/volcmp40_b_spf*_full.json")
    ap.add_argument("--grasp-glob", default="logs/volcmp40_g_spf*_*_*.json")
    ap.add_argument("--expected-volumes", type=int, default=15)
    ap.add_argument("--out-json", default="logs/volcmp40_aggregate.json")
    ap.add_argument("--out-md", default="logs/volcmp40_aggregate.md")
    args = ap.parse_args()

    brisk_paths = sorted(glob.glob(args.brisk_glob))
    grasp_paths = sorted(glob.glob(args.grasp_glob))
    rows = _read_rows(brisk_paths + grasp_paths)

    stat_bag: Dict[Tuple[str, int, str], List[ChunkStat]] = {}
    ranges: Dict[Tuple[str, int], List[Tuple[int, int, str]]] = {}

    for row in rows:
        spf = int(row.get("spokes_per_frame", 0) or 0)
        if spf <= 0:
            continue

        if "brisknet_volume_solve_mean_s" in row:
            method = "brisknet"
            _add_stat(stat_bag, method, spf, "solve", row)
            _add_stat(stat_bag, method, spf, "e2e", row)
            a = int(row.get("volume_index_start", 0) or 0)
            b = int(row.get("volume_index_end", row.get("num_volumes", 0)) or 0)
            ranges.setdefault((method, spf), []).append((a, b, row.get("_source_path", "")))

        if "grasp_volume_solve_mean_s" in row:
            method = "grasp"
            _add_stat(stat_bag, method, spf, "solve", row)
            _add_stat(stat_bag, method, spf, "e2e", row)
            a = int(row.get("volume_index_start", 0) or 0)
            b = int(row.get("volume_index_end", row.get("num_volumes", 0)) or 0)
            ranges.setdefault((method, spf), []).append((a, b, row.get("_source_path", "")))

    out_rows: List[dict] = []
    for spf in sorted({k[1] for k in stat_bag.keys()}):
        b_solve = _pool_mean_std(stat_bag.get(("brisknet", spf, "solve"), []))
        b_e2e = _pool_mean_std(stat_bag.get(("brisknet", spf, "e2e"), []))
        g_solve = _pool_mean_std(stat_bag.get(("grasp", spf, "solve"), []))
        g_e2e = _pool_mean_std(stat_bag.get(("grasp", spf, "e2e"), []))

        b_mean, b_std, b_n = b_solve
        g_mean, g_std, g_n = g_solve
        speedup = (g_mean / b_mean) if (b_mean and g_mean) else None
        saved = ((g_mean - b_mean) / 60.0) if (b_mean is not None and g_mean is not None) else None

        def _coverage(method: str) -> dict:
            ivals = ranges.get((method, spf), [])
            covered = set()
            for a, b, _ in ivals:
                covered.update(range(max(0, a), max(0, b)))
            overlap = 0
            seen = set()
            for a, b, _ in ivals:
                for i in range(max(0, a), max(0, b)):
                    if i in seen:
                        overlap += 1
                    seen.add(i)
            return {
                "intervals": [{"start": a, "end": b, "source": src} for a, b, src in ivals],
                "covered_count": len(covered),
                "missing_indices": [i for i in range(args.expected_volumes) if i not in covered],
                "overlap_count": overlap,
            }

        out_rows.append(
            {
                "spf": spf,
                "brisknet": {
                    "solve_mean_s": b_mean,
                    "solve_std_s": b_std,
                    "solve_n": b_n,
                    "e2e_mean_s": b_e2e[0],
                    "e2e_std_s": b_e2e[1],
                    "e2e_n": b_e2e[2],
                    "coverage": _coverage("brisknet"),
                },
                "grasp": {
                    "solve_mean_s": g_mean,
                    "solve_std_s": g_std,
                    "solve_n": g_n,
                    "e2e_mean_s": g_e2e[0],
                    "e2e_std_s": g_e2e[1],
                    "e2e_n": g_e2e[2],
                    "coverage": _coverage("grasp"),
                },
                "speedup_x": speedup,
                "saved_min_per_volume": saved,
            }
        )

    out_payload = {
        "brisk_files": brisk_paths,
        "grasp_files": grasp_paths,
        "rows": out_rows,
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True) if os.path.dirname(args.out_json) else None
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    lines = []
    lines.append("# Volume Timing Aggregate")
    lines.append("")
    lines.append("| SPF | BRISKNet (min/vol) | GRASP (min/vol) | Speedup | Saved (min/vol) |")
    lines.append("|---:|---:|---:|---:|---:|")
    for r in sorted(out_rows, key=lambda x: x["spf"]):
        b = _fmt_ms(r["brisknet"]["solve_mean_s"], r["brisknet"]["solve_std_s"])
        g = _fmt_ms(r["grasp"]["solve_mean_s"], r["grasp"]["solve_std_s"])
        sp = _fmt_x(r["speedup_x"])
        sv = "n/a" if r["saved_min_per_volume"] is None else f"{r['saved_min_per_volume']:.2f}"
        lines.append(f"| {r['spf']} | {b} | {g} | {sp} | {sv} |")

    lines.append("")
    lines.append("## Coverage Checks")
    lines.append("")
    for r in sorted(out_rows, key=lambda x: x["spf"]):
        for method in ("brisknet", "grasp"):
            cov = r[method]["coverage"]
            miss = cov["missing_indices"]
            lines.append(
                f"- SPF {r['spf']} {method}: covered={cov['covered_count']} "
                f"missing={len(miss)} overlap={cov['overlap_count']}"
            )

    os.makedirs(os.path.dirname(args.out_md), exist_ok=True) if os.path.dirname(args.out_md) else None
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
