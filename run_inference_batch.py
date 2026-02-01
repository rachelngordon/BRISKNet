import argparse
import os
import shlex
import subprocess
import sys


def _parse_list(value: str) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run inference for a list of experiment names.")
    parser.add_argument(
        "--exp_names",
        required=True,
        help="Comma-separated experiment names to run.",
    )
    parser.add_argument(
        "--exp_base_dir",
        default="output",
        help="Base directory containing experiment folders (default: output).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use (default: current interpreter).",
    )
    parser.add_argument(
        "--gpus",
        default="0,1,2,3",
        help="Comma-separated GPU indices to use (default: 0,1,2,3).",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Max concurrent runs (default: number of GPUs).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue running remaining experiments after a failure.",
    )
    args, passthrough = parser.parse_known_args()

    exp_names = _parse_list(args.exp_names)
    if not exp_names:
        raise SystemExit("No experiment names provided.")

    gpu_list = _parse_list(args.gpus)
    if not gpu_list:
        raise SystemExit("No GPUs provided.")
    max_workers = args.max_workers or len(gpu_list)
    max_workers = max(1, min(max_workers, len(gpu_list)))

    exit_code = 0
    running: list[tuple[subprocess.Popen, str]] = []

    def _launch(exp_name: str, gpu_id: str) -> subprocess.Popen:
        exp_dir = os.path.join(args.exp_base_dir, exp_name)
        cmd = [args.python, "run_inference.py", "--exp_dir", exp_dir, "--device", f"cuda:{gpu_id}"] + passthrough
        print("Running:", " ".join(shlex.quote(c) for c in cmd))
        if args.dry_run:
            return None
        return subprocess.Popen(cmd)

    exp_iter = iter(exp_names)
    gpu_iter = iter(gpu_list)

    while True:
        while len(running) < max_workers:
            try:
                exp_name = next(exp_iter)
            except StopIteration:
                break
            try:
                gpu_id = next(gpu_iter)
            except StopIteration:
                gpu_iter = iter(gpu_list)
                gpu_id = next(gpu_iter)
            proc = _launch(exp_name, gpu_id)
            if proc is None:
                continue
            running.append((proc, exp_name))

        if not running:
            break

        finished_idx = None
        for idx, (proc, exp_name) in enumerate(running):
            ret = proc.poll()
            if ret is not None:
                finished_idx = idx
                if ret != 0:
                    exit_code = ret
                    print(f"[Error] Inference failed for {exp_name} with code {ret}")
                    if not args.continue_on_error:
                        return exit_code
                break

        if finished_idx is None:
            # Avoid busy-wait
            for proc, _ in running:
                proc.wait(timeout=0.1)
        else:
            running.pop(finished_idx)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
