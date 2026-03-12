"""Bootstrap distributed env vars before launching training. Run: python3 train_zf_multinode.py"""

import os
import subprocess


def _slurm_first_hostname():
    nodelist = os.environ.get("SLURM_NODELIST") or os.environ.get("SLURM_JOB_NODELIST")
    if not nodelist:
        return None
    try:
        output = subprocess.check_output(
            ["scontrol", "show", "hostnames", nodelist], text=True
        )
        for line in output.splitlines():
            host = line.strip()
            if host:
                return host
    except Exception:
        pass

    head = nodelist.split(",")[0]
    if "[" in head and "]" in head:
        prefix = head.split("[", 1)[0]
        inside = head.split("[", 1)[1].split("]", 1)[0]
        first = inside.split(",", 1)[0]
        if "-" in first:
            first = first.split("-", 1)[0]
        return f"{prefix}{first}"
    return head


def _maybe_set_distributed_env():
    if os.environ.get("RANK") and os.environ.get("WORLD_SIZE"):
        return

    slurm_ntasks = os.environ.get("SLURM_NTASKS")
    slurm_procid = os.environ.get("SLURM_PROCID")
    slurm_localid = os.environ.get("SLURM_LOCALID")
    if slurm_ntasks and slurm_procid and slurm_localid:
        os.environ.setdefault("RANK", slurm_procid)
        os.environ.setdefault("WORLD_SIZE", slurm_ntasks)
        os.environ.setdefault("LOCAL_RANK", slurm_localid)

    if "MASTER_ADDR" not in os.environ:
        master_addr = _slurm_first_hostname()
        if master_addr:
            os.environ["MASTER_ADDR"] = master_addr

    if "MASTER_PORT" not in os.environ:
        job_id = os.environ.get("SLURM_JOB_ID")
        if job_id and job_id.isdigit():
            port = 29500 + (int(job_id) % 2000)
        else:
            port = 29500
        os.environ["MASTER_PORT"] = str(port)

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        ids = [entry for entry in visible.split(",") if entry.strip()]
        if len(ids) == 1:
            os.environ["LOCAL_RANK"] = "0"


def main():
    _maybe_set_distributed_env()
    from train_zf import main as train_main

    train_main()


if __name__ == "__main__":
    main()
