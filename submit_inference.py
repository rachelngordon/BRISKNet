#!/usr/bin/env python3
"""Submit inference jobs via submitit on SLURM."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

import submitit


REPO_ROOT = Path(__file__).resolve().parents[1]

_DEFAULT_SLURM_PARAMS = {
    "nodes": 1,
    "gpus_per_node": 1,
    "cpus_per_task": 8,
    "partition": "general",
    "timeout_min": 700,
}

_SLURM_PRESETS = {
    "fast": {"timeout_min": 120},
    "standard": {"timeout_min": 700},
}


def _coerce_scalar(value: str):
    v = value.strip()
    low = v.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low == "none":
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _parse_key_value(items):
    parsed = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --slurm-param '{item}'. Expected key=value.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --slurm-param '{item}'. Key cannot be empty.")
        parsed[key] = _coerce_scalar(value)
    return parsed


def _resolve_abs_path(path: str) -> str:
    resolved = str(Path(path).expanduser())
    if not os.path.isabs(resolved):
        resolved = os.path.abspath(resolved)
    return resolved


def _resolve_slurm_params(args) -> dict[str, int | str]:
    resolved: dict[str, int | str] = dict(_DEFAULT_SLURM_PARAMS)
    if args.preset:
        resolved.update(_SLURM_PRESETS[args.preset])
    if args.nodes is not None:
        resolved["nodes"] = int(args.nodes)
    if args.gpus_per_node is not None:
        resolved["gpus_per_node"] = int(args.gpus_per_node)
    if args.cpus_per_task is not None:
        resolved["cpus_per_task"] = int(args.cpus_per_task)
    if args.partition is not None:
        resolved["partition"] = str(args.partition)
    if args.timeout_min is not None:
        resolved["timeout_min"] = int(args.timeout_min)
    return resolved


def _resolve_log_dir(log_root: str | None, exp_dir: str, job_name: str) -> str:
    if log_root:
        return os.path.join(_resolve_abs_path(log_root), job_name)
    return os.path.join(_resolve_abs_path(exp_dir), "submitit_logs")


def _validate_args(args, *, slurm_params: dict[str, int | str]):
    if int(slurm_params["nodes"]) < 1:
        raise ValueError("--nodes must be >= 1.")
    if int(slurm_params["gpus_per_node"]) < 1:
        raise ValueError("--gpus-per-node must be >= 1.")
    if int(slurm_params["cpus_per_task"]) < 1:
        raise ValueError("--cpus-per-task must be >= 1.")
    if int(slurm_params["timeout_min"]) < 1:
        raise ValueError("--timeout-min must be >= 1.")
    partition = str(slurm_params["partition"]).strip()
    if not partition:
        raise ValueError("--partition must be non-empty.")
    if not args.micromamba_path:
        raise ValueError("--micromamba-path is required.")
    if not args.env_name:
        raise ValueError("--env-name is required.")
    if not args.exp_dir:
        raise ValueError("--exp-dir is required.")


class InferenceRunner(submitit.helpers.Checkpointable):
    def __init__(
        self,
        exp_dir: str,
        log_file: str | None,
        skip_raw_grasp_metrics: bool,
        extra_args: list[str],
        micromamba_path: str,
        env_name: str,
        entry_script: str,
    ):
        self.exp_dir = exp_dir
        self.log_file = log_file
        self.skip_raw_grasp_metrics = skip_raw_grasp_metrics
        self.extra_args = extra_args
        self.micromamba_path = micromamba_path
        self.env_name = env_name
        self.entry_script = entry_script

    def _build_command(self) -> list[str]:
        cmd = [
            "python",
            self.entry_script,
            "--exp_dir",
            self.exp_dir,
        ]
        if self.log_file:
            cmd.extend(["--log_file", self.log_file])
        if self.skip_raw_grasp_metrics:
            cmd.append("--skip_raw_grasp_metrics")
        cmd.extend(self.extra_args)
        return cmd

    def __call__(self):
        cmd = self._build_command()
        cmd_str = " ".join(shlex.quote(arg) for arg in cmd)
        command_str = (
            "set -eo pipefail; "
            f"cd {shlex.quote(str(REPO_ROOT))}; "
            f"source {shlex.quote(self.micromamba_path)}; "
            f"micromamba activate {shlex.quote(self.env_name)}; "
            f"exec {cmd_str}"
        )
        subprocess.run(["/bin/bash", "-lc", command_str], check=True)

    def checkpoint(self, *args, **kwargs):
        new_runner = InferenceRunner(
            exp_dir=self.exp_dir,
            log_file=self.log_file,
            skip_raw_grasp_metrics=self.skip_raw_grasp_metrics,
            extra_args=self.extra_args,
            micromamba_path=self.micromamba_path,
            env_name=self.env_name,
            entry_script=self.entry_script,
        )
        return submitit.helpers.DelayedSubmission(new_runner)


def build_parser():
    p = argparse.ArgumentParser(description="Submit inference jobs via submitit on SLURM.")
    p.add_argument("--exp-dir", "--exp_dir", dest="exp_dir", required=True, help="Experiment directory.")
    p.add_argument("--log-file", "--log_file", dest="log_file", default=None, help="Path to inference log file.")
    p.add_argument(
        "--skip-raw-grasp-metrics",
        "--skip_raw_grasp_metrics",
        dest="skip_raw_grasp_metrics",
        action="store_true",
        help="Skip raw GRASP metrics during inference.",
    )
    p.add_argument(
        "--entry-script",
        default="inference/run_inference.py",
        help="Inference entry script.",
    )

    p.add_argument("--job-name", default=None, help="SLURM job name (defaults to infer_<exp-name>).")
    p.add_argument(
        "--preset",
        choices=sorted(_SLURM_PRESETS.keys()),
        default=None,
        help="Apply preset SLURM params (overridden by explicit flags).",
    )
    p.add_argument("--nodes", type=int, default=None, help="Number of SLURM nodes.")
    p.add_argument("--gpus-per-node", type=int, default=None, help="GPUs per node.")
    p.add_argument("--cpus-per-task", type=int, default=None, help="CPUs per task.")
    p.add_argument("--partition", default=None, help="SLURM partition.")
    p.add_argument("--timeout-min", type=int, default=None, help="SLURM timeout in minutes.")
    p.add_argument("--constraint", default=None, help="SLURM node constraint.")
    p.add_argument("--exclude", default=None, help="SLURM host exclude list.")
    p.add_argument("--qos", default=None, help="SLURM QoS.")
    p.add_argument("--account", default=None, help="SLURM account.")
    p.add_argument("--requeue", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--slurm-param",
        action="append",
        default=[],
        help="Additional slurm_additional_parameters entries as key=value. Repeatable.",
    )
    p.add_argument(
        "--log-root",
        default=None,
        help="Optional directory root for submitit logs (default: <exp_dir>/submitit_logs).",
    )
    p.add_argument(
        "--micromamba-path",
        default=os.environ.get("MICROMAMBA_SH"),
        help="Path to micromamba shell init script (or set MICROMAMBA_SH).",
    )
    p.add_argument(
        "--env-name",
        default=os.environ.get("MICROMAMBA_ENV"),
        help="Micromamba env name (or set MICROMAMBA_ENV).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate submit args, print resolved plan, and exit without submission.",
    )
    return p


def main():
    parser = build_parser()
    args, extra_args = parser.parse_known_args()
    extra_args = [arg for arg in extra_args if arg != "--"]

    slurm_params = _resolve_slurm_params(args)
    _validate_args(args, slurm_params=slurm_params)

    exp_dir = _resolve_abs_path(args.exp_dir)
    entry_script = _resolve_abs_path(args.entry_script)
    micromamba_path = _resolve_abs_path(args.micromamba_path)

    exp_name = Path(exp_dir).name
    job_name = str(args.job_name).strip() if args.job_name and str(args.job_name).strip() else f"infer_{exp_name}"

    log_dir = _resolve_log_dir(args.log_root, exp_dir, job_name)
    os.makedirs(log_dir, exist_ok=True)

    slurm_additional = {}
    if args.requeue:
        slurm_additional["requeue"] = True
    else:
        slurm_additional["no-requeue"] = True
    if args.constraint:
        slurm_additional["constraint"] = args.constraint
    if args.exclude:
        slurm_additional["exclude"] = args.exclude
    slurm_additional.update(_parse_key_value(args.slurm_param))

    update_kwargs = dict(
        slurm_partition=slurm_params["partition"],
        slurm_job_name=job_name,
        nodes=slurm_params["nodes"],
        tasks_per_node=1,
        cpus_per_task=slurm_params["cpus_per_task"],
        slurm_gres=f"gpu:{slurm_params['gpus_per_node']}",
        timeout_min=slurm_params["timeout_min"],
        slurm_additional_parameters=slurm_additional,
        srun_args=["--cpu-bind=none"],
    )
    if args.qos:
        update_kwargs["slurm_qos"] = args.qos
    if args.account:
        update_kwargs["slurm_account"] = args.account

    runner = InferenceRunner(
        exp_dir=exp_dir,
        log_file=args.log_file,
        skip_raw_grasp_metrics=args.skip_raw_grasp_metrics,
        extra_args=extra_args,
        micromamba_path=micromamba_path,
        env_name=args.env_name,
        entry_script=entry_script,
    )

    print(f"[submit] exp_dir: {exp_dir}")
    print(f"[submit] job_name: {job_name}")
    print(
        "[submit] nodes x gpus_per_node x cpus_per_task: "
        f"{slurm_params['nodes']} x {slurm_params['gpus_per_node']} x {slurm_params['cpus_per_task']}"
    )
    print(f"[submit] partition: {slurm_params['partition']}")
    print(f"[submit] timeout_min: {slurm_params['timeout_min']}")
    if args.preset:
        print(f"[submit] preset: {args.preset}")
    print(f"[submit] entry_script: {entry_script}")
    print(f"[submit] env_name: {args.env_name}")
    print(f"[submit] log_dir: {log_dir}")
    print(f"[submit] update_parameters: {update_kwargs}")
    if extra_args:
        print(f"[submit] extra inference args: {extra_args}")

    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(**update_kwargs)

    if args.dry_run:
        print("[submit] dry-run validator: PASS (submit args)")
        return

    job = executor.submit(runner)
    print(f"Submitted job with ID: {job.job_id}")


if __name__ == "__main__":
    main()
