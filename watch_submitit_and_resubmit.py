#!/usr/bin/env python3
"""Watch submitit logs and resubmit failed jobs. Run: python3 watch_submitit_and_resubmit.py --help"""
import argparse
import glob
import json
import os
import subprocess
import time


DEFAULT_WATCH = {
    "submitit_logs/ei_2spf_slice_sampling_no_rebin": "python submit_2spf.py",
    "submitit_logs/ei_4spf_slice_sampling_no_rebin": "python submit_4spf.py",
    "submitit_logs/ei_8spf_slice_sampling_no_rebin": "python submit_8spf.py",
    "submitit_logs/ei_16spf_slice_sampling_no_rebin": "python submit_16spf_multinode.py",
    "submitit_logs/ei_36spf_slice_sampling_no_rebin": "python submit_36spf_multinode.py",
}

FAILURE_MARKERS = (
    "submitit ERROR",
    "ChildFailedError",
    "FAILED",
    "Traceback (most recent call last):",
    "ncclRemoteError",
    "Process group watchdog thread terminated with exception",
    "CUDA out of memory",
    "SIGABRT",
    "returned non-zero exit status",
)


def _latest_log(log_dir: str) -> str | None:
    patterns = [
        os.path.join(log_dir, "*_log.err"),
        os.path.join(log_dir, "*.err"),
        os.path.join(log_dir, "*.out"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _load_state(path: str) -> dict:
    if not os.path.isfile(path):
        return {"dirs": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"dirs": {}}


def _save_state(path: str, state: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _log_has_failure(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = f.read()
    except Exception:
        return False
    return any(marker in data for marker in FAILURE_MARKERS)


def _run_command(cmd: str, dry_run: bool) -> int:
    print(f"[resubmit] {cmd}")
    if dry_run:
        return 0
    try:
        completed = subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")
        return completed.returncode
    except subprocess.CalledProcessError as exc:
        print(f"[resubmit] command failed with exit {exc.returncode}")
        return exc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Monitor submitit logs and resubmit on failure."
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default="submitit_logs/.resubmit_state.json",
        help="Path to store resubmit state.",
    )
    parser.add_argument(
        "--max-resubmits",
        type=int,
        default=0,
        help="Max resubmits per log (0 = unlimited).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resubmits without executing.",
    )
    args = parser.parse_args()

    state = _load_state(args.state_file)
    dirs_state = state.setdefault("dirs", {})

    print("[watch] Watching submitit logs for failures. Press Ctrl-C to stop.")
    while True:
        for log_dir, command in DEFAULT_WATCH.items():
            abs_dir = os.path.abspath(log_dir)
            entry = dirs_state.setdefault(
                abs_dir,
                {
                    "latest_log": None,
                    "resubmitted_for": None,
                    "resubmit_count": 0,
                },
            )

            latest = _latest_log(log_dir)
            if latest is None:
                continue

            if entry.get("latest_log") != latest:
                entry["latest_log"] = latest
                entry["resubmitted_for"] = None

            if _log_has_failure(latest):
                if entry.get("resubmitted_for") == latest:
                    continue
                if args.max_resubmits > 0 and entry.get("resubmit_count", 0) >= args.max_resubmits:
                    print(f"[watch] Max resubmits reached for {latest}")
                    continue

                code = _run_command(command, args.dry_run)
                entry["resubmitted_for"] = latest
                entry["resubmit_count"] = entry.get("resubmit_count", 0) + 1
                entry["last_resubmit_exit"] = code
                entry["last_resubmit_time"] = time.time()

        _save_state(args.state_file, state)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
