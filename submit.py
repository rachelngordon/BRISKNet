import argparse
import os
import shlex
import subprocess
from pathlib import Path

import submitit
import yaml

from cluster_paths import apply_cluster_paths

_DEFAULT_SLURM_PARAMS = {
    "nodes": 1,
    "gpus_per_node": 4,
    "cpus_per_task": 8,
    "partition": "general",
    "timeout_min": 700,
}

_SLURM_PRESETS = {
    "overfit": {
        "nodes": 1,
        "gpus_per_node": 1,
        "cpus_per_task": 8,
        "partition": "general",
        "timeout_min": 240,
    },
    "memfit": {
        "nodes": 1,
        "gpus_per_node": 4,
        "cpus_per_task": 8,
        "partition": "general",
        "timeout_min": 240,
    },
    "selfsup": {
        "nodes": 1,
        "gpus_per_node": 4,
        "cpus_per_task": 8,
        "partition": "general",
        "timeout_min": 360,
    },
}


def _slurm_first_hostname() -> str:
    nodelist = os.environ.get("SLURM_NODELIST") or os.environ.get("SLURM_JOB_NODELIST")
    if not nodelist:
        raise RuntimeError(
            "SLURM_NODELIST/SLURM_JOB_NODELIST is not set. "
            "This launcher is intended to run inside a SLURM allocation."
        )

    output = subprocess.check_output(["scontrol", "show", "hostnames", nodelist], text=True)
    hostnames = [line.strip() for line in output.splitlines() if line.strip()]
    if not hostnames:
        raise RuntimeError(f"Could not resolve hostnames from SLURM nodelist '{nodelist}'.")
    return hostnames[0]


def _default_master_port() -> int:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or not job_id.isdigit():
        raise RuntimeError("SLURM_JOB_ID must be set to derive a rendezvous port.")
    return 29500 + (int(job_id) % 2000)


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


def _load_config(config_path: str) -> dict:
    resolved = _resolve_abs_path(config_path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Config file does not exist: {resolved}")
    with open(resolved, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must parse to a dict, got {type(config).__name__}.")
    return apply_cluster_paths(config)


def _resolve_experiment_output_dir(config: dict) -> str:
    exp_cfg = config.get("experiment")
    if not isinstance(exp_cfg, dict):
        raise KeyError("Config is missing required 'experiment' section.")
    output_dir = exp_cfg.get("output_dir")
    if not output_dir:
        raise KeyError("Config is missing required 'experiment.output_dir'.")
    return _resolve_abs_path(str(output_dir))


def _resolve_log_dir(log_root: str | None, config: dict, exp_name: str, job_name: str) -> str:
    if log_root:
        return os.path.join(_resolve_abs_path(log_root), job_name)
    output_root = _resolve_experiment_output_dir(config)
    return os.path.join(output_root, exp_name, "submitit_logs")


def _resolve_existing_file(path: str, label: str) -> str:
    resolved = _resolve_abs_path(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


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


class Trainer(submitit.helpers.Checkpointable):
    def __init__(
        self,
        exp_name,
        config_path,
        num_nodes,
        gpus_per_node,
        micromamba_path,
        env_name,
        entry_script,
        nccl_ib_disable,
        nccl_timeout,
        nccl_blocking_wait,
        nccl_async_error_handling,
    ):
        self.exp_name = exp_name
        self.config_path = config_path
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node
        self.micromamba_path = micromamba_path
        self.env_name = env_name
        self.entry_script = entry_script
        self.nccl_ib_disable = nccl_ib_disable
        self.nccl_timeout = nccl_timeout
        self.nccl_blocking_wait = nccl_blocking_wait
        self.nccl_async_error_handling = nccl_async_error_handling

    def _build_torchrun_command(self, master_addr: str, master_port: int, node_rank: int, rdzv_id: str):
        return [
            "torchrun",
            "--rdzv-backend=c10d",
            f"--rdzv-endpoint={master_addr}:{master_port}",
            f"--rdzv-id={rdzv_id}",
            f"--nnodes={self.num_nodes}",
            f"--node_rank={node_rank}",
            f"--nproc_per_node={self.gpus_per_node}",
            self.entry_script,
            "--config",
            self.config_path,
            "--exp_name",
            self.exp_name,
        ]

    def __call__(self):
        master_addr = _slurm_first_hostname()
        master_port = _default_master_port()
        node_rank = int(os.environ.get("SLURM_NODEID", "0"))
        rdzv_id = os.environ.get("SLURM_JOB_ID")
        if not rdzv_id:
            raise RuntimeError("SLURM_JOB_ID must be set inside the training job.")

        torchrun_command = self._build_torchrun_command(
            master_addr=master_addr,
            master_port=master_port,
            node_rank=node_rank,
            rdzv_id=rdzv_id,
        )
        torchrun_str = " ".join(shlex.quote(arg) for arg in torchrun_command)
        command_str = (
            "set -eo pipefail; "
            f"source {shlex.quote(self.micromamba_path)}; "
            f"micromamba activate {shlex.quote(self.env_name)}; "
            f"export NCCL_IB_DISABLE={int(self.nccl_ib_disable)}; "
            f"export NCCL_TIMEOUT={int(self.nccl_timeout)}; "
            f"export TORCH_NCCL_BLOCKING_WAIT={int(self.nccl_blocking_wait)}; "
            f"export TORCH_NCCL_ASYNC_ERROR_HANDLING={int(self.nccl_async_error_handling)}; "
            f"exec {torchrun_str}"
        )
        subprocess.run(["/bin/bash", "-lc", command_str], check=True)

    def checkpoint(self, *args, **kwargs):
        new_trainer = Trainer(
            exp_name=self.exp_name,
            config_path=self.config_path,
            num_nodes=self.num_nodes,
            gpus_per_node=self.gpus_per_node,
            micromamba_path=self.micromamba_path,
            env_name=self.env_name,
            entry_script=self.entry_script,
            nccl_ib_disable=self.nccl_ib_disable,
            nccl_timeout=self.nccl_timeout,
            nccl_blocking_wait=self.nccl_blocking_wait,
            nccl_async_error_handling=self.nccl_async_error_handling,
        )
        return submitit.helpers.DelayedSubmission(new_trainer)


def _validate_submission_plan(
    *,
    trainer: Trainer,
    log_dir: str,
    update_kwargs: dict[str, object],
) -> submitit.AutoExecutor:
    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(**update_kwargs)
    trainer._build_torchrun_command(
        master_addr="127.0.0.1",
        master_port=29500,
        node_rank=0,
        rdzv_id="dry-run",
    )
    return executor


def build_parser():
    p = argparse.ArgumentParser(description="Submit torchrun jobs via submitit on SLURM.")
    p.add_argument("--config", required=True, help="Path to YAML config.")
    p.add_argument("--exp-name", default=None, help="Experiment name for train script.")
    p.add_argument("--job-name", default=None, help="SLURM job name (defaults to exp-name).")
    p.add_argument(
        "--preset",
        choices=sorted(_SLURM_PRESETS.keys()),
        default=None,
        help=(
            "Apply preset SLURM params for common run types. "
            "Explicit --nodes/--gpus-per-node/--cpus-per-task/--partition/--timeout-min overrides preset values."
        ),
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
        help=(
            "Optional directory root for submitit logs. If omitted, logs are placed "
            "under <experiment.output_dir>/<exp-name>/submitit_logs."
        ),
    )
    p.add_argument(
        "--entry-script",
        default="multinode.py",
        help="Entry script used by torchrun.",
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

    p.add_argument("--nccl-ib-disable", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--nccl-timeout", type=int, default=3600)
    p.add_argument("--nccl-blocking-wait", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--nccl-async-error-handling", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config + submit args, print resolved plan, and exit without submission.",
    )
    return p


def main():
    args = build_parser().parse_args()
    slurm_params = _resolve_slurm_params(args)
    _validate_args(args, slurm_params=slurm_params)

    config_path = _resolve_existing_file(args.config, "Config file")
    entry_script = _resolve_existing_file(args.entry_script, "Entry script")
    micromamba_path = _resolve_existing_file(args.micromamba_path, "Micromamba init script")
    config = _load_config(config_path)

    exp_name = str(args.exp_name).strip() if args.exp_name and str(args.exp_name).strip() else Path(config_path).stem
    job_name = str(args.job_name).strip() if args.job_name and str(args.job_name).strip() else exp_name

    log_dir = _resolve_log_dir(args.log_root, config, exp_name, job_name)
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
    )
    if args.qos:
        update_kwargs["slurm_qos"] = args.qos
    if args.account:
        update_kwargs["slurm_account"] = args.account

    trainer = Trainer(
        exp_name=exp_name,
        config_path=config_path,
        num_nodes=int(slurm_params["nodes"]),
        gpus_per_node=int(slurm_params["gpus_per_node"]),
        micromamba_path=micromamba_path,
        env_name=args.env_name,
        entry_script=entry_script,
        nccl_ib_disable=args.nccl_ib_disable,
        nccl_timeout=args.nccl_timeout,
        nccl_blocking_wait=args.nccl_blocking_wait,
        nccl_async_error_handling=args.nccl_async_error_handling,
    )

    print(f"[submit] config: {config_path}")
    print(f"[submit] exp_name: {exp_name}")
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

    executor = _validate_submission_plan(
        trainer=trainer,
        log_dir=log_dir,
        update_kwargs=update_kwargs,
    )

    if args.dry_run:
        print("[submit] dry-run validator: PASS (config + submit args)")
        return

    job = executor.submit(trainer)
    print(f"Submitted job with ID: {job.job_id}")


if __name__ == "__main__":
    main()
