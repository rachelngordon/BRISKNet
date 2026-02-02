import argparse
import json
import math
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from utils import GRASPRecon_from_ktraj, prep_nufft


def _parse_list(value: str) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_noise_level(noise_level) -> Tuple[float, str | None]:
    if noise_level is None:
        return 0.0, None
    if isinstance(noise_level, str):
        label = noise_level.strip()
        if label == "":
            return 0.0, None
        try:
            value = float(label)
        except ValueError as exc:
            raise ValueError(f"noise_level must be numeric; got {noise_level!r}") from exc
        if value <= 0:
            return 0.0, None
        return value, label
    value = float(noise_level)
    if value <= 0:
        return 0.0, None
    return value, str(noise_level)


def _frames_from_spf(spokes_per_frame: int, total_spokes: int) -> int:
    if spokes_per_frame <= 0:
        raise ValueError("spokes_per_frame must be positive.")
    if total_spokes <= 0:
        raise ValueError("total_spokes must be positive.")
    if total_spokes % spokes_per_frame != 0:
        raise ValueError(
            f"total_spokes ({total_spokes}) must be divisible by spokes_per_frame ({spokes_per_frame})."
        )
    return total_spokes // spokes_per_frame


def _traj_suffix(traj_method: str, noise_value: float, noise_label: str | None, sim_source: str) -> str:
    if traj_method != "get_traj":
        suffix = ".npy"
    elif noise_value > 0:
        suffix = f"_correct_traj_n{noise_label}.npy"
    else:
        suffix = "_correct_traj.npy"
    if sim_source == "espirit" and suffix.endswith(".npy"):
        suffix = suffix[:-4] + "_espirit.npy"
    return suffix


def _load_split_ids(split_file: str, num_samples: int) -> List[str]:
    with open(split_file, "r") as f:
        splits = json.load(f)
    val_ids = splits.get("val_dro") or splits.get("val") or []
    if not val_ids:
        raise ValueError("No validation IDs found in split file.")
    return val_ids[:num_samples]


def _resolve_sample_dirs(dro_root: str, num_frames: int, sample_ids: List[str]) -> List[str]:
    dro_dir = os.path.join(dro_root, f"dro_{num_frames}frames")
    if not os.path.isdir(dro_dir):
        raise FileNotFoundError(f"DRO directory not found: {dro_dir}")
    sample_dirs = []
    missing = []
    for sample_id in sample_ids:
        sample_dir = os.path.join(dro_dir, sample_id)
        if not os.path.isdir(sample_dir):
            missing.append(sample_dir)
            continue
        sample_dirs.append(sample_dir)
    if missing:
        missing_str = "\n".join(missing[:5])
        raise FileNotFoundError(f"Missing sample directories (showing up to 5):\n{missing_str}")
    return sample_dirs


def _load_csmaps(sample_dir: str, source: str, espirit_dir: str | None) -> torch.Tensor:
    if source == "original":
        csmaps = np.load(os.path.join(sample_dir, "csmaps.npy"))
        csmaps_torch = torch.from_numpy(csmaps).permute(2, 0, 1)
    elif source == "espirit":
        dro_root = os.path.dirname(os.path.dirname(sample_dir))
        esp_root = espirit_dir or os.path.join(dro_root, "csmaps_espirit")
        sample_name = os.path.basename(sample_dir)
        csmap_path = os.path.join(esp_root, f"csmaps_{sample_name}.npy")
        csmaps = np.load(csmap_path)
        csmaps_torch = torch.from_numpy(csmaps)
        if not torch.is_complex(csmaps_torch):
            csmaps_torch = csmaps_torch.to(torch.complex64)
    else:
        raise ValueError(f"Unsupported dro_csmaps_source '{source}'.")
    return csmaps_torch


def _load_kspace(
    sample_dir: str,
    spokes_per_frame: int,
    num_frames: int,
    traj_method: str,
    noise_value: float,
    noise_label: str | None,
    sim_source: str,
) -> torch.Tensor:
    suffix = _traj_suffix(traj_method, noise_value, noise_label, sim_source)
    kspace_path = os.path.join(
        sample_dir,
        f"simulated_kspace_spf{spokes_per_frame}_frames{num_frames}{suffix}",
    )
    if not os.path.exists(kspace_path):
        raise FileNotFoundError(f"Missing simulated k-space file: {kspace_path}")
    kspace = np.load(kspace_path, allow_pickle=True)
    return torch.from_numpy(kspace)


def _noise_matches(row_noise, target_noise: float) -> bool:
    try:
        row_val = float(row_noise) if row_noise not in (None, "") else 0.0
    except (TypeError, ValueError):
        return False
    return abs(row_val - float(target_noise)) < 1e-8


def _update_grasp_log(
    rows: List[Dict],
    spokes_per_frame: int,
    num_frames: int,
    noise_level: float,
    mean_time: float,
    std_time: float,
    num_samples: int,
    total_scan_seconds: float,
) -> None:
    match = None
    for row in rows:
        if row.get("type") != "GRASP":
            continue
        if int(row.get("spokes_per_frame", -1)) != spokes_per_frame:
            continue
        if int(row.get("num_frames", -1)) != num_frames:
            continue
        if not _noise_matches(row.get("DRO_noise_level"), noise_level):
            continue
        match = row
        break

    if match is None:
        acceleration = (320.0 * math.pi / 2.0) / float(spokes_per_frame)
        seconds_per_frame = (
            float(total_scan_seconds) / float(num_frames - 1)
            if num_frames > 1 else float(total_scan_seconds)
        )
        match = {
            "type": "GRASP",
            "spokes_per_frame": int(spokes_per_frame),
            "num_frames": int(num_frames),
            "acceleration": float(acceleration),
            "seconds_per_frame": float(seconds_per_frame),
            "DRO_noise_level": float(noise_level),
            "num_samples": int(num_samples),
            "spatial_metrics": {},
            "dc_metrics": {},
            "temporal_metrics": {},
        }
        rows.append(match)

    match["avg_grasp_recon_time"] = float(mean_time)
    match["std_grasp_recon_time"] = float(std_time)
    match["grasp_recon_time_num_samples"] = int(num_samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Time GRASP reconstructions on DRO validation samples.")
    parser.add_argument("--log_file", default="val_inference_logs.json", help="Path to val_inference_logs.json.")
    parser.add_argument(
        "--split_file",
        default="data/data_split.json",
        help="Split file containing val_dro IDs.",
    )
    parser.add_argument(
        "--dro_root",
        default="/net/scratch2/rachelgordon/dro_dataset_frontpad",
        help="Root directory containing DRO datasets.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="Deprecated: num_frames is derived from spokes_per_frame and total_spokes.",
    )
    parser.add_argument(
        "--total_spokes",
        type=int,
        default=288,
        help="Total spokes per scan (used to derive num_frames).",
    )
    parser.add_argument(
        "--spokes_list",
        default="2,4,8,16,24,36",
        help="Comma-separated spokes per frame values.",
    )
    parser.add_argument("--num_samples", type=int, default=15, help="Number of DRO validation samples to time.")
    parser.add_argument(
        "--traj_method",
        default="get_traj",
        choices=("trajGR", "get_traj"),
        help="Trajectory source used to build ktraj.",
    )
    parser.add_argument(
        "--dro_noise_level",
        type=float,
        default=0.05,
        help="Noise level used in simulated k-space filenames.",
    )
    parser.add_argument(
        "--dro_sim_source",
        default="espirit",
        choices=("original", "espirit"),
        help="Simulated k-space filename source (original or espirit suffix).",
    )
    parser.add_argument(
        "--dro_csmaps_source",
        default="espirit",
        choices=("original", "espirit"),
        help="DRO csmaps source (original or espirit directory).",
    )
    parser.add_argument(
        "--dro_espirit_csmaps_dir",
        default=None,
        help="Override ESPIRiT csmaps dir (default: <dro_root>/csmaps_espirit).",
    )
    parser.add_argument(
        "--total_scan_seconds",
        type=float,
        default=150.0,
        help="Total scan duration in seconds for seconds_per_frame logging.",
    )
    parser.add_argument("--lamda", type=float, default=0.001, help="GRASP TV regularization weight.")
    parser.add_argument("--max_iter", type=int, default=10, help="GRASP max iterations.")
    parser.add_argument("--rho", type=float, default=0.1, help="GRASP ADMM rho.")
    args = parser.parse_args()

    spokes_list = [int(v) for v in _parse_list(args.spokes_list)]
    if not spokes_list:
        raise ValueError("No spokes per frame values provided.")

    noise_value, noise_label = _parse_noise_level(args.dro_noise_level)
    sample_ids = _load_split_ids(args.split_file, args.num_samples)
    sample_dirs_cache: Dict[int, List[str]] = {}

    if args.num_frames is not None:
        print(
            "Warning: --num_frames is deprecated and ignored; num_frames is derived from "
            "--total_spokes and spokes_per_frame."
        )

    if os.path.exists(args.log_file):
        with open(args.log_file, "r") as f:
            log_rows = json.load(f)
    else:
        log_rows = []

    for spf in spokes_list:
        num_frames = _frames_from_spf(spf, args.total_spokes)
        if num_frames not in sample_dirs_cache:
            sample_dirs_cache[num_frames] = _resolve_sample_dirs(args.dro_root, num_frames, sample_ids)
        sample_dirs = sample_dirs_cache[num_frames]

        print(f"Timing GRASP recon for {spf} spokes/frame ({num_frames} frames) ...")
        ktraj, _, _, _ = prep_nufft(640, spf, num_frames, traj_method=args.traj_method)

        times = []
        for sample_dir in sample_dirs:
            csmaps = _load_csmaps(sample_dir, args.dro_csmaps_source, args.dro_espirit_csmaps_dir)
            kspace = _load_kspace(
                sample_dir,
                spf,
                num_frames,
                args.traj_method,
                noise_value,
                noise_label,
                args.dro_sim_source,
            )
            start = time.perf_counter()
            GRASPRecon_from_ktraj(
                csmaps,
                kspace,
                ktraj,
                640,
                lamda=args.lamda,
                max_iter=args.max_iter,
                rho=args.rho,
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"  {os.path.basename(sample_dir)}: {elapsed:.3f}s")

        mean_time = float(np.mean(times)) if times else 0.0
        std_time = float(np.std(times, ddof=1)) if len(times) > 1 else 0.0
        print(f"  -> mean {mean_time:.3f}s, std {std_time:.3f}s over {len(times)} samples")

        _update_grasp_log(
            log_rows,
            spf,
            num_frames,
            noise_value,
            mean_time,
            std_time,
            len(times),
            args.total_scan_seconds,
        )

    with open(args.log_file, "w") as f:
        json.dump(log_rows, f, indent=2)

    print(f"Updated {args.log_file} with GRASP recon timing.")


if __name__ == "__main__":
    main()
