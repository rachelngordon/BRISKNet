"""Time BRISKNet vs GRASP reconstructions on full raw fastMRI volumes.

This script reconstructs all slices for each validation patient and reports
per-volume timing (mean +/- std over volumes) for each spokes-per-frame setting.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
from typing import Dict, List, Tuple

import h5py
import numpy as np
import sigpy as sp
import torch
import yaml
from einops import rearrange
from sigpy.mri import app

from model.model_factory import build_recon_model
from model.radial import MCNUFFT
from utils import (
    get_traj,
    prep_nufft,
    remove_module_prefix,
    sliding_window_inference,
    to_torch_complex,
)


def _parse_int_list(value: str) -> List[int]:
    if not value:
        return []
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _load_split_ids(split_file: str, split_key: str) -> List[str]:
    with open(split_file, "r", encoding="utf-8") as f:
        splits = json.load(f)
    if split_key not in splits:
        raise KeyError(f"Split key '{split_key}' not found in {split_file}.")
    ids = splits[split_key]
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"Split '{split_key}' is empty or invalid.")
    return ids


def _find_kspace_file(root_dir: str, patient_id: str) -> str:
    pattern = os.path.join(root_dir, f"*{patient_id}*.h5")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No k-space file matching {pattern}")
    if len(matches) > 1:
        sample = "\n".join(matches[:5])
        raise RuntimeError(f"Multiple files matched {patient_id}:\n{sample}")
    return matches[0]


def _torch_load_checkpoint(path: str, map_location: str = "cpu") -> Dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


def _load_config(config_path: str) -> Dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_model(config: Dict, device: torch.device) -> torch.nn.Module:
    model = build_recon_model(config, device=device, block_dir=None)
    model.eval()
    return model


def _load_weights(model: torch.nn.Module, checkpoint_path: str) -> None:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = _torch_load_checkpoint(checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(remove_module_prefix(state))


def _resolve_device(device_str: str | None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sync_torch(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _sync_sigpy(device: sp.Device) -> None:
    if hasattr(device, "synchronize"):
        device.synchronize()


def _sigpy_device_from_torch(device: torch.device) -> sp.Device:
    if device.type != "cuda":
        return sp.Device(-1)
    index = device.index if device.index is not None else 0
    return sp.Device(index)


def _load_csmap(csmap_root: str, patient_id: str, slice_idx: int, flip_kspace: bool) -> torch.Tensor:
    csmap_path = os.path.join(
        csmap_root, f"{patient_id}_2_cs_maps", f"cs_map_slice_{slice_idx:03d}.npy"
    )
    if not os.path.exists(csmap_path):
        raise FileNotFoundError(f"CSMAP file not found: {csmap_path}")
    csmap = np.load(csmap_path).squeeze()
    csmap_t = torch.from_numpy(csmap)
    if not torch.is_complex(csmap_t):
        csmap_t = csmap_t.to(torch.complex64)
    if csmap_t.ndim != 3:
        raise ValueError(f"Expected csmap shape (C,H,W), got {tuple(csmap_t.shape)}")
    csmap_t = csmap_t.unsqueeze(0)
    if flip_kspace:
        csmap_t = torch.rot90(csmap_t, k=2, dims=[-2, -1])
    return csmap_t


def _time_bin_kspace_train(
    kspace_slice: np.ndarray, spokes_per_frame: int, flip_kspace: bool
) -> Tuple[torch.Tensor, int, int]:
    if kspace_slice.ndim != 3:
        raise ValueError(
            f"Expected kspace slice shape (C,Spokes,Samples), got {kspace_slice.shape}"
        )
    n_coils, n_spokes, n_samples = kspace_slice.shape
    if n_spokes % spokes_per_frame != 0:
        raise ValueError(
            f"Total spokes ({n_spokes}) not divisible by spokes_per_frame ({spokes_per_frame})."
        )
    n_time = n_spokes // spokes_per_frame

    ksp_prep = np.swapaxes(kspace_slice, 0, 1)
    ksp_prep = ksp_prep.reshape(n_time, spokes_per_frame, n_coils, n_samples)
    ksp_prep = np.transpose(ksp_prep, (0, 2, 1, 3))

    real_part = torch.from_numpy(ksp_prep.real)
    imag_part = torch.from_numpy(ksp_prep.imag)
    kspace_final = torch.stack([real_part, imag_part], dim=0).float()
    if flip_kspace:
        kspace_final = torch.flip(kspace_final, dims=[-1])
    return kspace_final, n_time, n_samples


def _std_or_zero(values: List[float]) -> float:
    if len(values) > 1:
        return float(np.std(values, ddof=1))
    return 0.0


def _print_device_info(device: torch.device) -> None:
    if device.type != "cuda":
        print(f"[Device] Using {device.type.upper()}")
        return
    index = device.index if device.index is not None else 0
    name = torch.cuda.get_device_name(index)
    props = torch.cuda.get_device_properties(index)
    total_gb = props.total_memory / (1024**3)
    cap = f"{props.major}.{props.minor}"
    print(f"[Device] CUDA:{index} {name} | CC {cap} | {total_gb:.1f} GB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Time BRISKNet vs GRASP reconstructions on full validation volumes."
    )
    parser.add_argument("--split-file", default="data/split/data_split.json")
    parser.add_argument("--split-key", default="val")
    parser.add_argument("--kspace-root", required=True)
    parser.add_argument("--csmap-root", required=True)
    parser.add_argument("--dataset-key", default="kspace")
    parser.add_argument("--spokes-per-frame-list", default="8,16,24,36")
    parser.add_argument(
        "--model-template",
        required=True,
        help="Checkpoint path template. Can include {spf}.",
    )
    parser.add_argument(
        "--config-template",
        required=True,
        help="Config path template. Can include {spf}.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--lamda", type=float, default=0.001)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--total-scan-seconds", type=float, default=150.0)
    parser.add_argument("--num-samples", type=int, default=15)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index (inclusive) into selected split list.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="End index (exclusive) into selected split list.",
    )
    parser.add_argument("--slice-stride", type=int, default=1)
    parser.add_argument("--max-slices", type=int, default=None)
    parser.add_argument("--disable-grasp", action="store_true")
    parser.add_argument("--disable-brisknet", action="store_true")
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    if args.disable_grasp and args.disable_brisknet:
        raise ValueError("Cannot disable both GRASP and BRISKNet.")
    if args.slice_stride < 1:
        raise ValueError("--slice-stride must be >= 1.")

    spokes_list = _parse_int_list(args.spokes_per_frame_list)
    if not spokes_list:
        raise ValueError("No spokes_per_frame values provided.")

    val_ids = _load_split_ids(args.split_file, args.split_key)
    if args.num_samples:
        val_ids = val_ids[: args.num_samples]

    total_selected = len(val_ids)
    start_index = int(args.start_index or 0)
    end_index = total_selected if args.end_index is None else int(args.end_index)
    if start_index < 0 or end_index < 0:
        raise ValueError("--start-index and --end-index must be non-negative.")
    if start_index > total_selected:
        raise ValueError(f"--start-index {start_index} exceeds selected sample count {total_selected}.")
    end_index = min(end_index, total_selected)
    if end_index <= start_index:
        raise ValueError(
            f"Invalid index range: start={start_index}, end={end_index}, total={total_selected}."
        )
    val_ids = val_ids[start_index:end_index]
    if not val_ids:
        raise ValueError("No validation IDs selected after index filtering.")
    print(
        f"[Volume selection] split={args.split_key} total={total_selected} "
        f"range=[{start_index},{end_index}) count={len(val_ids)}",
        flush=True,
    )

    device = _resolve_device(args.device)
    _print_device_info(device)
    sigpy_device = _sigpy_device_from_torch(device)

    all_rows = []

    for spf in spokes_list:
        spf_start_time = time.perf_counter()
        config_path = args.config_template.format(spf=spf)
        model_path = args.model_template.format(spf=spf)
        config = _load_config(config_path)

        model = None
        if not args.disable_brisknet:
            model = _build_model(config, device)
            _load_weights(model, model_path)

        flip_kspace = bool(config.get("data", {}).get("flip_kspace", False))
        norm_mode = config.get("model", {}).get("norm", "both")
        encode_acc = bool(config.get("model", {}).get("encode_acceleration", False))
        encode_time = bool(config.get("model", {}).get("encode_time_index", False))
        eval_chunk_size = int(config.get("evaluation", {}).get("chunk_size", 0) or 0)
        eval_chunk_overlap = int(config.get("evaluation", {}).get("chunk_overlap", 0) or 0)
        traj_method = config.get("data", {}).get("traj_method", "get_traj")

        per_spf_setup_start = time.perf_counter()
        ktraj = dcomp = nufft_ob = adjnufft_ob = None
        coord = None
        n_time_ref = None
        acceleration_encoding = None
        start_timepoint_index = None
        per_spf_setup_time = None

        volume_brisk_solve: List[float] = []
        volume_brisk_e2e: List[float] = []
        volume_grasp_solve: List[float] = []
        volume_grasp_e2e: List[float] = []
        slices_per_volume: List[int] = []

        for vol_i, patient_id in enumerate(val_ids, start=1):
            kspace_path = _find_kspace_file(args.kspace_root, patient_id)
            with h5py.File(kspace_path, "r") as f:
                if args.dataset_key not in f:
                    raise KeyError(f"Dataset key '{args.dataset_key}' not found in {kspace_path}")
                n_slices_total = int(f[args.dataset_key].shape[0])
                max_count = min(args.max_slices, n_slices_total) if args.max_slices is not None else n_slices_total
                slice_indices = list(range(0, max_count, args.slice_stride))

                vol_brisk_solve = 0.0
                vol_brisk_e2e = 0.0
                vol_grasp_solve = 0.0
                vol_grasp_e2e = 0.0

                for slice_idx in slice_indices:
                    preprocess_start = time.perf_counter()
                    kspace_slice = np.asarray(f[args.dataset_key][slice_idx])
                    kspace_binned, n_time, n_samples = _time_bin_kspace_train(kspace_slice, spf, flip_kspace)
                    csmap = _load_csmap(args.csmap_root, patient_id, slice_idx, flip_kspace)

                    if n_time_ref is None:
                        n_time_ref = n_time
                        ktraj, dcomp, nufft_ob, adjnufft_ob = prep_nufft(
                            n_samples, spf, n_time, traj_method=traj_method
                        )
                        ktraj = ktraj.to(device)
                        dcomp = dcomp.to(device)
                        if device.type == "cuda":
                            nufft_ob = nufft_ob.to(device)
                            adjnufft_ob = adjnufft_ob.to(device)

                        h = csmap.shape[-2]
                        n_full = h * math.pi / 2.0
                        acceleration = torch.tensor([n_full / float(spf)], dtype=torch.float, device=device)
                        acceleration_encoding = acceleration if encode_acc else None
                        start_timepoint_index = (
                            torch.tensor([0], dtype=torch.float, device=device) if encode_time else None
                        )
                        coord = get_traj(
                            N_spokes=spf,
                            N_time=n_time,
                            base_res=int(n_samples // 2),
                            gind=1,
                        )
                        coord = np.asarray(coord, dtype=np.float32)
                        if coord.ndim == 3:
                            coord = coord[None, ...]
                        per_spf_setup_time = time.perf_counter() - per_spf_setup_start

                    kspace_cplx = to_torch_complex(kspace_binned.unsqueeze(0)).squeeze(0)
                    kspace_cplx = rearrange(kspace_cplx, "t co sp sam -> co (sp sam) t")
                    preprocess_time = time.perf_counter() - preprocess_start

                    if not args.disable_brisknet:
                        transfer_start = time.perf_counter()
                        kspace_dev = kspace_cplx.to(device)
                        csmap_dev = csmap.to(device).to(kspace_dev.dtype)
                        _sync_torch(device)
                        transfer_time = time.perf_counter() - transfer_start

                        use_sliding = eval_chunk_size > 0 and eval_chunk_size < n_time
                        physics_time = 0.0
                        with torch.no_grad():
                            if use_sliding:
                                _sync_torch(device)
                                t0 = time.perf_counter()
                                _ = sliding_window_inference(
                                    csmap_dev.shape[-2],
                                    csmap_dev.shape[-1],
                                    n_time,
                                    ktraj,
                                    dcomp,
                                    nufft_ob,
                                    adjnufft_ob,
                                    eval_chunk_size,
                                    eval_chunk_overlap,
                                    kspace_dev,
                                    csmap_dev,
                                    acceleration_encoding,
                                    start_timepoint_index,
                                    model,
                                    epoch="inference",
                                    device=device,
                                    norm=norm_mode,
                                    collect_adj_loss=False,
                                )
                                _sync_torch(device)
                                brisk_solve = time.perf_counter() - t0
                            else:
                                physics_start = time.perf_counter()
                                physics = MCNUFFT(nufft_ob, adjnufft_ob, ktraj, dcomp)
                                physics_time = time.perf_counter() - physics_start
                                _sync_torch(device)
                                t0 = time.perf_counter()
                                _ = model(
                                    kspace_dev,
                                    physics,
                                    csmap_dev,
                                    acceleration_encoding,
                                    start_timepoint_index,
                                    epoch="inference",
                                    norm=norm_mode,
                                )
                                _sync_torch(device)
                                brisk_solve = time.perf_counter() - t0

                        brisk_e2e = preprocess_time + transfer_time + physics_time + brisk_solve
                        vol_brisk_solve += brisk_solve
                        vol_brisk_e2e += brisk_e2e

                    if not args.disable_grasp:
                        grasp_total_start = time.perf_counter()
                        kspace_np = rearrange(
                            kspace_cplx.cpu(), "c (sp sam) t -> t c sp sam", sam=n_samples
                        ).unsqueeze(1).unsqueeze(3).numpy()
                        csmaps_np = rearrange(csmap, "b c h w -> c b h w").cpu().numpy()

                        kspace_sp = sp.to_device(kspace_np, sigpy_device)
                        csmaps_sp = sp.to_device(csmaps_np, sigpy_device)
                        coord_sp = sp.to_device(coord, sigpy_device)

                        recon_op = app.HighDimensionalRecon(
                            kspace_sp,
                            csmaps_sp,
                            combine_echo=False,
                            lamda=args.lamda,
                            coord=coord_sp,
                            regu="TV",
                            regu_axes=[0],
                            max_iter=args.max_iter,
                            solver="ADMM",
                            rho=args.rho,
                            device=sigpy_device,
                            show_pbar=False,
                            verbose=False,
                        )
                        _sync_sigpy(sigpy_device)
                        t0 = time.perf_counter()
                        recon_out = recon_op.run()
                        _sync_sigpy(sigpy_device)
                        grasp_solve = time.perf_counter() - t0
                        grasp_e2e = preprocess_time + (time.perf_counter() - grasp_total_start)

                        vol_grasp_solve += grasp_solve
                        vol_grasp_e2e += grasp_e2e

                        del recon_out
                        del recon_op

            slices_per_volume.append(len(slice_indices))
            if not args.disable_brisknet:
                volume_brisk_solve.append(vol_brisk_solve)
                volume_brisk_e2e.append(vol_brisk_e2e)
            if not args.disable_grasp:
                volume_grasp_solve.append(vol_grasp_solve)
                volume_grasp_e2e.append(vol_grasp_e2e)

            elapsed = time.perf_counter() - spf_start_time
            avg_per_vol = elapsed / float(vol_i)
            remaining = max(0, len(val_ids) - vol_i)
            eta_min = (avg_per_vol * remaining) / 60.0
            msg = (
                f"[SPF {spf}] volume {vol_i}/{len(val_ids)} {patient_id}: "
                f"slices={len(slice_indices)} elapsed={elapsed/60.0:.1f}m eta={eta_min:.1f}m"
            )
            if not args.disable_brisknet:
                msg += f" brisk_solve={vol_brisk_solve:.1f}s"
            if not args.disable_grasp:
                msg += f" grasp_solve={vol_grasp_solve:.1f}s"
            print(msg, flush=True)

        if n_time_ref is None:
            raise RuntimeError(f"No slices processed for SPF={spf}")
        if per_spf_setup_time is None:
            per_spf_setup_time = 0.0

        num_vols = max(1, len(val_ids))
        setup_share = per_spf_setup_time / float(num_vols)
        if not args.disable_brisknet:
            volume_brisk_e2e = [v + setup_share for v in volume_brisk_e2e]
        if not args.disable_grasp:
            volume_grasp_e2e = [v + setup_share for v in volume_grasp_e2e]

        seconds_per_frame = (
            float(args.total_scan_seconds) / float(n_time_ref - 1)
            if n_time_ref > 1
            else float(args.total_scan_seconds)
        )
        acceleration = (320.0 * math.pi / 2.0) / float(spf)

        row = {
            "spokes_per_frame": int(spf),
            "num_frames": int(n_time_ref),
            "acceleration": float(acceleration),
            "seconds_per_frame": float(seconds_per_frame),
            "num_volumes": int(len(val_ids)),
            "volume_index_start": int(start_index),
            "volume_index_end": int(end_index),
            "slices_per_volume_mean": float(np.mean(slices_per_volume)),
            "slices_per_volume_std": _std_or_zero([float(x) for x in slices_per_volume]),
            "spf_setup_time_s": float(per_spf_setup_time),
        }
        if not args.disable_brisknet:
            row.update(
                {
                    "brisknet_volume_solve_mean_s": float(np.mean(volume_brisk_solve)),
                    "brisknet_volume_solve_std_s": _std_or_zero(volume_brisk_solve),
                    "brisknet_volume_e2e_mean_s": float(np.mean(volume_brisk_e2e)),
                    "brisknet_volume_e2e_std_s": _std_or_zero(volume_brisk_e2e),
                }
            )
        if not args.disable_grasp:
            row.update(
                {
                    "grasp_volume_solve_mean_s": float(np.mean(volume_grasp_solve)),
                    "grasp_volume_solve_std_s": _std_or_zero(volume_grasp_solve),
                    "grasp_volume_e2e_mean_s": float(np.mean(volume_grasp_e2e)),
                    "grasp_volume_e2e_std_s": _std_or_zero(volume_grasp_e2e),
                }
            )

        all_rows.append(row)
        print(f"[SPF {spf}] summary: {row}", flush=True)

    print("\n=== Per-volume timing summary ===", flush=True)
    for row in all_rows:
        spf = row["spokes_per_frame"]
        b = (
            f"{row['brisknet_volume_solve_mean_s']:.3f} +/- {row['brisknet_volume_solve_std_s']:.3f}"
            if not args.disable_brisknet
            else "n/a"
        )
        g = (
            f"{row['grasp_volume_solve_mean_s']:.3f} +/- {row['grasp_volume_solve_std_s']:.3f}"
            if not args.disable_grasp
            else "n/a"
        )
        print(f"SPF {spf}: BRISKNet solve {b} s/volume | GRASP solve {g} s/volume", flush=True)

    if args.out_json:
        out_dir = os.path.dirname(args.out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=2)
        print(f"Wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
