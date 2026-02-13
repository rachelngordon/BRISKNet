#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import subprocess
import sys
from typing import Any


VAL_LOG_PATH = os.path.join(os.path.dirname(__file__), "val_inference_logs.json")


def _run(cmd: list[str]) -> None:
    print("[probe] Running:", " ".join(shlex.quote(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def _load_val_log(path: str) -> list[dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing val inference log: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data).__name__}")
    return data


def _find_brisknet_entry(rows: list[dict[str, Any]], exp_name: str) -> dict[str, Any]:
    matches = [
        r for r in rows
        if r.get("type") == "BRISKNet" and str(r.get("exp_name", "")) == exp_name
    ]
    if not matches:
        raise RuntimeError(f"No BRISKNet entry found for exp_name='{exp_name}' in {VAL_LOG_PATH}")
    return matches[-1]


def _find_grasp_2spf_entry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [
        r for r in rows
        if r.get("type") == "GRASP" and int(r.get("spokes_per_frame", -1)) == 2
        and r.get("avg_grasp_recon_time") is not None
    ]
    if not matches:
        raise RuntimeError("No GRASP 2-SPF timing entry found in val_inference_logs.json")
    return matches[-1]


def _base_cmd(
    python_bin: str,
    exp_dir: str,
    device: str,
    num_samples: int,
    eval_spokes: int,
    disable_ssdu: bool,
) -> list[str]:
    cmd = [
        python_bin,
        "run_inference.py",
        "--exp_dir",
        exp_dir,
        "--device",
        device,
        "--num_samples",
        str(num_samples),
        "--eval_spokes",
        str(eval_spokes),
        "--overwrite_logs",
    ]
    if disable_ssdu:
        cmd.append("--disable_ssdu")
    return cmd


def _summarize_entry(label: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": label,
        "exp_name": row.get("exp_name"),
        "inference_dir": row.get("inference_dir"),
        "avg_inference_time": row.get("avg_inference_time"),
        "std_inference_time": row.get("std_inference_time"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe 2-SPF inference timing for Mamba + LSFP and print GRASP reference timing."
    )
    # submit.py always forwards these; keep them for compatibility.
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--mamba-exp-dir", required=True)
    parser.add_argument("--lsfp-exp-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=15)
    parser.add_argument("--eval-spokes", type=int, default=2)
    parser.add_argument("--disable-ssdu", action="store_true", default=True)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    mamba_exp_name = os.path.basename(os.path.abspath(args.mamba_exp_dir))
    lsfp_exp_name = os.path.basename(os.path.abspath(args.lsfp_exp_dir))

    for exp_dir in (args.mamba_exp_dir, args.lsfp_exp_dir):
        if not os.path.isdir(exp_dir):
            raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    _run(
        _base_cmd(
            python_bin=sys.executable,
            exp_dir=args.mamba_exp_dir,
            device=args.device,
            num_samples=args.num_samples,
            eval_spokes=args.eval_spokes,
            disable_ssdu=args.disable_ssdu,
        )
    )
    _run(
        _base_cmd(
            python_bin=sys.executable,
            exp_dir=args.lsfp_exp_dir,
            device=args.device,
            num_samples=args.num_samples,
            eval_spokes=args.eval_spokes,
            disable_ssdu=args.disable_ssdu,
        )
    )

    rows = _load_val_log(VAL_LOG_PATH)
    mamba_row = _find_brisknet_entry(rows, mamba_exp_name)
    lsfp_row = _find_brisknet_entry(rows, lsfp_exp_name)
    grasp_row = _find_grasp_2spf_entry(rows)

    summary = {
        "probe_exp_name": args.exp_name,
        "mamba": _summarize_entry("mamba", mamba_row),
        "lsfpnet": _summarize_entry("lsfpnet", lsfp_row),
        "grasp": {
            "spokes_per_frame": grasp_row.get("spokes_per_frame"),
            "avg_grasp_recon_time": grasp_row.get("avg_grasp_recon_time"),
            "std_grasp_recon_time": grasp_row.get("std_grasp_recon_time"),
            "num_samples": grasp_row.get("grasp_recon_time_num_samples", grasp_row.get("num_samples")),
        },
    }

    if args.output_json:
        out_path = os.path.abspath(args.output_json)
    else:
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(args.config)),
            f"{args.exp_name}_probe_summary.json",
        )
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("[probe] Summary written to:", out_path, flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
