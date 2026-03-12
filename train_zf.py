"""Train the DDEI reconstruction model. Run: python3 train_zf.py --config <config> --exp_name <name>"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import torch
import warnings
import yaml
from dataloader import ZFSliceDataset, SimulatedDataset, log_slice_sampling_startup_report
from einops import rearrange
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
from transform import (
    VideoRotate,
    VideoDiffeo,
    SubsampleTime,
    MonophasicTimeWarp,
    TemporalShiftJitterAfterBaseline,
    TemporalNoise,
    TimeReverse,
    BolusArrivalTimeShift,
    BaselineEnhancementScale,
)
from ei import EILoss
from mc import MCLoss
from model_factory import build_recon_model, is_lsfp_model
from radial_lsfp import MCNUFFT
from utils import prep_nufft, log_gradient_stats, log_lsfpnet_component_grads, plot_enhancement_curve, plot_rebin_consistency_diagnostic, get_cosine_ei_weight, plot_reconstruction_sample, get_git_commit, save_checkpoint, load_checkpoint, load_pretrained_weights, to_torch_complex, sliding_window_inference, set_seed, save_csmap_png
from eval import eval_grasp, eval_sample, compute_ssdu_kspace_nmse
import csv
import math
import random
import time
import threading
import atexit
import seaborn as sns
from rebin_loss import RebinConsistencyLoss
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import signal
from torch.utils.tensorboard import SummaryWriter
from cluster_paths import apply_cluster_paths
from contextlib import nullcontext, contextmanager
from datetime import timedelta


def setup():
    """Initializes the distributed process group."""
    # dist.init_process_group("nccl")
    dist.init_process_group("nccl", timeout=timedelta(seconds=3600))

def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

@contextmanager
def _temporary_rng(seed):
    if seed is None:
        yield
        return
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

def _load_run_state(run_state_path):
    if not os.path.exists(run_state_path):
        return {
            "attempt_count": 0,
            "cumulative_wall_time_sec": 0.0,
            "max_peak_mem_gb": 0.0,
            "attempt_history": [],
        }
    with open(run_state_path, "r") as file:
        return json.load(file)

def _save_run_state(run_state_path, run_state):
    with open(run_state_path, "w") as file:
        json.dump(run_state, file, indent=2)

def _sync_run_state_totals(run_state):
    attempt_history = run_state.get("attempt_history", [])
    cumulative_wall_time_sec = 0.0
    max_peak_mem_gb = 0.0
    for item in attempt_history:
        wall_time = float(item.get("wall_time_sec", 0.0) or 0.0)
        peak_mem = float(item.get("peak_mem_gb", 0.0) or 0.0)
        cumulative_wall_time_sec += wall_time
        if peak_mem > max_peak_mem_gb:
            max_peak_mem_gb = peak_mem
    run_state["cumulative_wall_time_sec"] = cumulative_wall_time_sec
    run_state["max_peak_mem_gb"] = max_peak_mem_gb
    run_state.setdefault("attempt_count", 0)
    run_state.setdefault("attempt_history", [])
    return run_state

def _get_param_counts(model):
    if isinstance(model, DDP):
        model = model.module
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def sample_start_time_index(max_idx, use_edge_mixture):
    if not use_edge_mixture:
        return random.randint(0, max_idx - 1)

    r = random.random()
    if r < 1.0 / 3.0:
        return 0
    if r < 2.0 / 3.0:
        return max_idx
    return random.randint(0, max_idx)


class _DatasetWithIndex(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        if isinstance(item, tuple):
            return (idx, *item)
        return (idx, item)


def _format_sample_id(dataset, idx):
    if hasattr(dataset, "slice_index_map"):
        try:
            file_path, slice_idx = dataset.slice_index_map[idx]
            patient_id = os.path.splitext(os.path.basename(file_path))[0]
            return f"{patient_id}:{slice_idx}"
        except Exception:
            return str(idx)
    if hasattr(dataset, "sample_paths"):
        try:
            sample_path = dataset.sample_paths[idx]
            return os.path.basename(sample_path)
        except Exception:
            return str(idx)
    return str(idx)


def _print_sample_check_summary(epoch, all_rank_samples, sample_check_batches):
    if not all_rank_samples:
        print(f"[DDP] Sample check epoch {epoch}: no samples collected.")
        return
    rank_sets = [set(samples) for samples in all_rank_samples]
    total_samples = sum(len(samples) for samples in all_rank_samples)
    total_unique = len(set().union(*rank_sets)) if rank_sets else 0
    sum_rank_unique = sum(len(s) for s in rank_sets)
    overlap = max(0, sum_rank_unique - total_unique)

    print(
        f"[DDP] Sample check epoch {epoch} (first {sample_check_batches} batches): "
        f"total={total_samples}, global_unique={total_unique}, cross_rank_overlap={overlap}"
    )
    for rank, samples in enumerate(all_rank_samples):
        unique_count = len(set(samples))
        dup_within = len(samples) - unique_count
        print(f"  - rank {rank}: samples={len(samples)}, unique={unique_count}, dup_within_rank={dup_within}")

    if overlap > 0:
        from collections import Counter

        counter = Counter()
        for samples in all_rank_samples:
            counter.update(samples)
        overlap_ids = [sid for sid, count in counter.items() if count > 1]
        preview = ", ".join(overlap_ids[:10])
        print(
            "[DDP] Overlap detected across ranks. "
            "If dataset size isn't divisible by world_size and drop_last=False, "
            "some overlap can be expected. Sample overlaps: "
            f"{preview}"
        )


def _build_strided_shard_indices(num_items, rank, world_size):
    if num_items < 0:
        raise ValueError(f"num_items must be non-negative, got {num_items}.")
    if world_size <= 1:
        return list(range(num_items))
    if rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid rank/world_size pair: rank={rank}, world_size={world_size}.")
    return list(range(rank, num_items, world_size))


def _flatten_gathered_records(gathered_records):
    flat = []
    for record_list in gathered_records:
        if not record_list:
            continue
        if isinstance(record_list, list):
            flat.extend(record_list)
        else:
            flat.append(record_list)
    return flat


def _as_int_scalar(val, default: int = 0) -> int:
    try:
        if torch.is_tensor(val):
            if val.numel() == 0:
                return int(default)
            return int(val.reshape(-1)[0].item())
        return int(val)
    except Exception:
        return int(default)


def _build_nufft_getter(device, traj_method: str):
    """Cache NUFFT trajectory/operators by (samples, spokes, time, traj_method)."""
    cache = {}

    def _get_nufft(N_samples, N_spokes, N_time):
        n_samples_i = _as_int_scalar(N_samples)
        n_spokes_i = _as_int_scalar(N_spokes)
        n_time_i = _as_int_scalar(N_time)
        key = (n_samples_i, n_spokes_i, n_time_i, str(traj_method))
        if key not in cache:
            ktraj, dcomp, nufft_ob, adjnufft_ob = prep_nufft(
                n_samples_i, n_spokes_i, n_time_i, traj_method=traj_method
            )
            cache[key] = (
                ktraj.to(device, non_blocking=True),
                dcomp.to(device, non_blocking=True),
                nufft_ob.to(device),
                adjnufft_ob.to(device),
            )
        return cache[key]

    return _get_nufft


def _normalize_optional_checkpoint_path(value, field_name: str):
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string path or null, got {type(value).__name__}.")
    path = value.strip()
    if path.lower() in ("", "none", "null"):
        return None
    return path


def main():

    set_seed(12)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train ReconResNet model.")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="config.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--exp_name", type=str, required=True, help="Name of the experiment"
    )
    parser.add_argument(
        "--from_checkpoint",
        type=bool,
        required=False,
        default=False,
        help="Deprecated (ignored). Checkpoints auto-resume based on exp_name.",
    )
    args = parser.parse_args()

    # print experiment name and git commit
    exp_name = args.exp_name

    # Load config and auto-resume if a last checkpoint exists for this exp_name
    with open(args.config, "r") as file:
        new_config = yaml.safe_load(file)
    new_config = apply_cluster_paths(new_config)

    experiment_cfg = new_config.get("experiment", {})
    resume_checkpoint_cfg = _normalize_optional_checkpoint_path(
        experiment_cfg.get("resume_checkpoint", None),
        "experiment.resume_checkpoint",
    )
    init_checkpoint_cfg = _normalize_optional_checkpoint_path(
        experiment_cfg.get("init_checkpoint", None),
        "experiment.init_checkpoint",
    )
    legacy_pretrained_cfg = _normalize_optional_checkpoint_path(
        experiment_cfg.get("pretrained_checkpoint", None),
        "experiment.pretrained_checkpoint",
    )
    if resume_checkpoint_cfg and init_checkpoint_cfg:
        raise ValueError(
            "experiment.resume_checkpoint and experiment.init_checkpoint are mutually exclusive."
        )
    if init_checkpoint_cfg and legacy_pretrained_cfg:
        raise ValueError(
            "Both experiment.init_checkpoint and deprecated experiment.pretrained_checkpoint are set. "
            "Use only experiment.init_checkpoint."
        )
    init_checkpoint = init_checkpoint_cfg or legacy_pretrained_cfg
    init_checkpoint_source = (
        "experiment.init_checkpoint"
        if init_checkpoint_cfg
        else ("experiment.pretrained_checkpoint" if legacy_pretrained_cfg else None)
    )

    output_dir = os.path.join(new_config["experiment"]["output_dir"], exp_name)
    default_checkpoint_file = os.path.join(output_dir, f"{exp_name}_model.pth")
    checkpoint_file = resume_checkpoint_cfg or default_checkpoint_file
    if resume_checkpoint_cfg:
        if not os.path.isfile(checkpoint_file):
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_file}")
        resume_from_checkpoint = True
    else:
        resume_from_checkpoint = os.path.isfile(checkpoint_file)

    if resume_from_checkpoint:
        config_path = os.path.join(os.path.dirname(checkpoint_file), "config.yaml")
        if os.path.isfile(config_path):
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
        else:
            config = new_config
        config = apply_cluster_paths(config)
        epochs = new_config['training']["epochs"]
    else:
        config = new_config
        epochs = config['training']["epochs"]

    # Keep output_dir aligned with the resolved checkpoint location.
    config.setdefault("experiment", {})["output_dir"] = os.path.dirname(output_dir)

    use_edge_time_index_sampling = config.get('training', {}).get('edge_time_index_sampling', False)
    traj_method = config.get("data", {}).get("traj_method", "get_traj")

    # create output directories
    output_dir = os.path.join(config["experiment"]["output_dir"], exp_name)
    eval_dir = os.path.join(output_dir, "eval_results")
    save_block_outputs = config.get("debugging", {}).get("save_block_outputs", False)
    save_enhancement_curves = config.get("debugging", {}).get("save_enhancement_curve_pngs", False)
    block_dir = os.path.join(output_dir, "block_outputs") if save_block_outputs else None
    ec_dir = os.path.join(output_dir, "enhancement_curves") if save_enhancement_curves else None

    attempt_start_time_epoch = time.time()
    attempt_start_time_monotonic = time.monotonic()
    attempt_peak_mem_gb = 0.0
    run_state_path = os.path.join(output_dir, "run_state.json")
    run_state = None
    state_written = False
    attempt_idx = None
    state_lock = threading.Lock()
    stop_run_state = threading.Event()
    run_state_thread = None
    run_state_heartbeat_sec = float(os.environ.get("RUN_STATE_HEARTBEAT_SEC", "60"))

        

    if config['training']['multigpu']:
        setup()

        # Get rank and world_size from the distributed package AFTER setup
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Get the local rank from the environment variable
        local_rank = int(os.environ["LOCAL_RANK"])

        # Set the device for this process
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        # Sanity check: verify collectives across all ranks.
        try:
            check_tensor = torch.ones(1, device=device)
            dist.all_reduce(check_tensor, op=dist.ReduceOp.SUM)
            expected = float(world_size)
            got = float(check_tensor.item())
            if global_rank == 0:
                if abs(got - expected) < 1e-6:
                    print(f"[DDP] Sanity check passed: all-reduce sum = {got} across {world_size} ranks.")
                else:
                    print(f"[DDP] Sanity check FAILED: all-reduce sum = {got}, expected {expected}.")
        except Exception as exc:
            if global_rank == 0:
                print(f"[DDP] Sanity check FAILED with exception: {exc}")

        if global_rank == 0:
            print(f"Starting distributed training with {world_size} GPUs.")
        print(f"  - [Rank {global_rank}] -> Using device {device}")

    else:
        global_rank = 0
        device = torch.device(config["training"]["device"])

    # AMP/mixed precision config (optional)
    amp_cfg = config.get("training", {}).get("amp", {})
    amp_enabled = (
        bool(amp_cfg.get("enabled", False))
        and torch.cuda.is_available()
        and torch.device(device).type == "cuda"
    )
    amp_dtype_str = str(amp_cfg.get("dtype", "bf16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_str in ("bf16", "bfloat16") else torch.float16
    use_scaler = bool(amp_cfg.get("use_grad_scaler", amp_dtype == torch.float16))
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    def amp_autocast():
        return torch.amp.autocast(device_type="cuda", dtype=amp_dtype) if amp_enabled else nullcontext()

    get_nufft = _build_nufft_getter(device=device, traj_method=traj_method)

    # Silence torchmetrics LPIPS FutureWarning about torch.load(weights_only=...)
    warnings.filterwarnings(
        "ignore",
        message="You are using `torch.load` with `weights_only=False`.*",
        category=FutureWarning,
    )

    
    if global_rank == 0 or not config['training']['multigpu']:
        commit_hash = get_git_commit()
        print(f"Running experiment on Git commit: {commit_hash}")

        print(f"Experiment: {exp_name}")
        print(f"[AMP] enabled={amp_enabled}, dtype={amp_dtype_str}, grad_scaler={use_scaler}")
        if resume_from_checkpoint:
            print(f"[Checkpoint] Resuming from {checkpoint_file}")
            if init_checkpoint:
                print(
                    f"[Checkpoint] Found {init_checkpoint_source}={init_checkpoint}, "
                    "but resume checkpoint takes precedence."
                )
        else:
            if init_checkpoint:
                if init_checkpoint_source == "experiment.pretrained_checkpoint":
                    print(
                        "[Checkpoint] experiment.pretrained_checkpoint is deprecated; "
                        "use experiment.init_checkpoint instead."
                    )
                print(
                    f"[Checkpoint] No run checkpoint found; warm-starting from "
                    f"{init_checkpoint_source}: {init_checkpoint}"
                )
            else:
                print("[Checkpoint] No existing checkpoint found; starting new run.")



    if global_rank == 0 or not config['training']['multigpu']:
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        if block_dir is not None:
            os.makedirs(block_dir, exist_ok=True)
        if ec_dir is not None:
            os.makedirs(ec_dir, exist_ok=True)

        run_state = _load_run_state(run_state_path)
        run_state.setdefault("attempt_history", [])
        run_state.setdefault("attempt_count", 0)

        # If a prior attempt was interrupted before finalization, mark it as such.
        last_attempt = run_state["attempt_history"][-1] if run_state["attempt_history"] else None
        if last_attempt is not None and last_attempt.get("status") == "running":
            last_attempt["status"] = "interrupted"
            last_attempt.setdefault("reason", "interrupted")
            last_attempt.setdefault("end_time_epoch", time.time())

        attempt_idx = run_state.get("attempt_count", 0) + 1
        run_state["attempt_count"] = attempt_idx
        run_state["attempt_history"].append(
            {
                "attempt": attempt_idx,
                "wall_time_sec": 0.0,
                "peak_mem_gb": 0.0,
                "status": "running",
                "reason": "running",
                "start_time_epoch": attempt_start_time_epoch,
            }
        )
        _sync_run_state_totals(run_state)
        _save_run_state(run_state_path, run_state)

        # Initialize TensorBoard SummaryWriter
        log_dir = os.path.join(output_dir, 'logs')
        writer = SummaryWriter(log_dir)



        # Save the configuration file
        if not resume_from_checkpoint:
            with open(os.path.join(output_dir, 'config.yaml'), 'w') as file:
                yaml.dump(config, file)

    def _update_run_state_progress(reason=None, final=False):
        nonlocal attempt_peak_mem_gb, run_state
        if run_state is None or attempt_idx is None:
            return
        if global_rank != 0 and config['training']['multigpu']:
            return
        with state_lock:
            wall_time_sec = time.monotonic() - attempt_start_time_monotonic
            if torch.cuda.is_available():
                try:
                    current_peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                except Exception:
                    current_peak_gb = 0.0
                attempt_peak_mem_gb = max(attempt_peak_mem_gb, current_peak_gb)

            attempt_entry = None
            for item in run_state.get("attempt_history", []):
                if item.get("attempt") == attempt_idx:
                    attempt_entry = item
                    break
            if attempt_entry is None:
                attempt_entry = {"attempt": attempt_idx}
                run_state.setdefault("attempt_history", []).append(attempt_entry)

            attempt_entry.setdefault("start_time_epoch", attempt_start_time_epoch)
            attempt_entry["wall_time_sec"] = wall_time_sec
            attempt_entry["peak_mem_gb"] = attempt_peak_mem_gb
            attempt_entry["status"] = "completed" if final else "running"
            if reason is not None:
                attempt_entry["reason"] = reason
            if final:
                attempt_entry["end_time_epoch"] = time.time()

            run_state["attempt_count"] = max(run_state.get("attempt_count", 0), attempt_idx)
            _sync_run_state_totals(run_state)
            _save_run_state(run_state_path, run_state)

    def _run_state_heartbeat():
        while not stop_run_state.wait(run_state_heartbeat_sec):
            _update_run_state_progress()

    if run_state is not None:
        run_state_thread = threading.Thread(target=_run_state_heartbeat, daemon=True)
        run_state_thread.start()

    def _finalize_run_state(reason):
        nonlocal attempt_peak_mem_gb, state_written, run_state
        if state_written:
            return
        if global_rank != 0 and config['training']['multigpu']:
            state_written = True
            return
        if run_state is None or attempt_idx is None:
            return

        stop_run_state.set()
        _update_run_state_progress(reason=reason, final=True)
        state_written = True

        wall_time_sec = run_state.get("attempt_history", [{}])[-1].get("wall_time_sec", 0.0)
        cumulative_wall_time_sec = run_state.get("cumulative_wall_time_sec", 0.0)
        max_peak_mem_gb = run_state.get("max_peak_mem_gb", 0.0)
        print(f"[RunState] Attempt {attempt_idx} wall time (sec): {wall_time_sec:.2f}")
        print(f"[RunState] Cumulative wall time (hours): {cumulative_wall_time_sec / 3600.0:.2f}")
        print(f"[RunState] Attempt peak GPU memory (GB): {attempt_peak_mem_gb:.2f}")
        print(f"[RunState] Global max peak GPU memory (GB): {max_peak_mem_gb:.2f}")

    def _handle_signal(sig, _frame):
        nonlocal attempt_peak_mem_gb
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            current_peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            attempt_peak_mem_gb = max(attempt_peak_mem_gb, current_peak_gb)
        _finalize_run_state(f"signal_{sig}")
        signal.signal(sig, signal.SIG_DFL)
        os.kill(os.getpid(), sig)

    for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGUSR1", None), getattr(signal, "SIGUSR2", None)):
        if sig is not None:
            signal.signal(sig, _handle_signal)

    if run_state is not None:
        atexit.register(_finalize_run_state, "exit")


    # load params
    split_file = config["data"]["split_file"]

    data_dir = config["data"]["root_dir"]

    batch_size = config["dataloader"]["batch_size"]
    max_subjects = config["dataloader"]["max_subjects"]
    slice_sampling_mode = config["dataloader"].get("slice_sampling_mode", "uniform")
    slice_sampling_uniform_fraction = config["dataloader"].get("slice_sampling_uniform_fraction", 1.0)
    slice_sampling_filter_quantile = config["dataloader"].get("slice_sampling_filter_quantile", 0.2)
    slice_sampling_no_replacement = config["dataloader"].get("slice_sampling_no_replacement", False)
    slice_sampling_cache_dir = config["dataloader"].get("slice_sampling_cache_dir", None)
    slice_sampling_cache_workers = config["dataloader"].get("slice_sampling_cache_workers", 0)
    slice_sampling_cache_rank = global_rank if config['training']['multigpu'] else None
    slice_sampling_cache_rank_only = 0 if config['training']['multigpu'] else None

    losses_cfg = config["model"]["losses"]
    mc_loss_weight = float(losses_cfg["mc_loss"]["weight"])
    adj_loss_cfg = losses_cfg.get("adj_loss", {})
    if not isinstance(adj_loss_cfg, dict):
        raise TypeError(
            f"model.losses.adj_loss must be a mapping, got {type(adj_loss_cfg).__name__}."
        )

    use_ei_loss = losses_cfg["use_ei_loss"]
    ei_cfg = losses_cfg["ei_loss"]
    target_w_ei = float(ei_cfg["weight"])
    warmup = int(ei_cfg["warmup"])
    duration = int(ei_cfg["duration"])
    if warmup < 0:
        raise ValueError("model.losses.ei_loss.warmup must be >= 0.")
    if duration < 0:
        raise ValueError("model.losses.ei_loss.duration must be >= 0.")
    transition_duration = max(duration, 1)
    ei_transition_start_epoch = warmup + 1
    ei_transition_end_epoch = warmup + transition_duration

    gradnorm_cfg = ei_cfg.get("gradnorm_transition", {})
    if isinstance(gradnorm_cfg, dict):
        ei_gradnorm_transition_enable = bool(gradnorm_cfg.get("enable", False))
        ei_gradnorm_ema_beta = float(gradnorm_cfg.get("ema_beta", 0.9))
        ei_gradnorm_eps = float(gradnorm_cfg.get("eps", 1e-8))
        ei_gradnorm_ratio_min = float(gradnorm_cfg.get("ratio_min", 0.1))
        ei_gradnorm_ratio_max = float(gradnorm_cfg.get("ratio_max", 10.0))
        ei_gradnorm_measure_every = int(gradnorm_cfg.get("measure_every_n_steps", 8))
        ei_gradnorm_target_scale_min = float(gradnorm_cfg.get("target_scale_min", 0.25))
        ei_gradnorm_target_scale_max = float(gradnorm_cfg.get("target_scale_max", 4.0))
    else:
        ei_gradnorm_transition_enable = bool(gradnorm_cfg)
        ei_gradnorm_ema_beta = 0.9
        ei_gradnorm_eps = 1e-8
        ei_gradnorm_ratio_min = 0.1
        ei_gradnorm_ratio_max = 10.0
        ei_gradnorm_measure_every = 8
        ei_gradnorm_target_scale_min = 0.25
        ei_gradnorm_target_scale_max = 4.0

    if not (0.0 <= ei_gradnorm_ema_beta < 1.0):
        raise ValueError("model.losses.ei_loss.gradnorm_transition.ema_beta must be in [0, 1).")
    if ei_gradnorm_eps <= 0:
        raise ValueError("model.losses.ei_loss.gradnorm_transition.eps must be > 0.")
    if ei_gradnorm_ratio_min <= 0 or ei_gradnorm_ratio_max <= 0:
        raise ValueError("model.losses.ei_loss.gradnorm_transition ratio bounds must be > 0.")
    if ei_gradnorm_ratio_max < ei_gradnorm_ratio_min:
        raise ValueError(
            "model.losses.ei_loss.gradnorm_transition.ratio_max must be >= ratio_min."
        )
    if ei_gradnorm_measure_every < 1:
        raise ValueError(
            "model.losses.ei_loss.gradnorm_transition.measure_every_n_steps must be >= 1."
        )
    if ei_gradnorm_target_scale_min <= 0 or ei_gradnorm_target_scale_max <= 0:
        raise ValueError(
            "model.losses.ei_loss.gradnorm_transition target scale bounds must be > 0."
        )
    if ei_gradnorm_target_scale_max < ei_gradnorm_target_scale_min:
        raise ValueError(
            "model.losses.ei_loss.gradnorm_transition.target_scale_max must be >= target_scale_min."
        )
    checkpoint_before_loss_transitions = bool(
        config.get("training", {}).get("checkpoint_before_loss_transitions", True)
    )

    rebin_cfg = losses_cfg.get("rebin_loss", {})
    use_rebin_loss = bool(rebin_cfg.get("enable", False))
    rebin_target_w = float(rebin_cfg.get("weight", 0.0))
    rebin_warmup = int(rebin_cfg.get("warmup", 0))
    rebin_duration = int(rebin_cfg.get("duration", 0))
    if rebin_warmup < 0:
        raise ValueError("model.losses.rebin_loss.warmup must be >= 0.")
    if rebin_duration < 0:
        raise ValueError("model.losses.rebin_loss.duration must be >= 0.")
    rebin_factor = int(rebin_cfg.get("factor", 2))
    rebin_metric_name = str(rebin_cfg.get("metric", "MSE"))
    rebin_time_index_mode = str(rebin_cfg.get("time_index_mode", "none"))
    rebin_teacher_branch = str(rebin_cfg.get("teacher_branch", "none"))
    rebin_teacher_stopgrad = bool(rebin_cfg.get("teacher_stopgrad", False))
    rebin_offset_mode = str(rebin_cfg.get("offset_mode", "none"))
    rebin_temporal_mode = str(rebin_cfg.get("temporal_mode", "absolute"))
    rebin_baseline_frames = int(rebin_cfg.get("baseline_frames", 4))
    rebin_percent_enhancement_eps = float(rebin_cfg.get("percent_enhancement_eps", 1e-4))

    rebin_dynamic_mask_cfg = rebin_cfg.get("dynamic_mask", {})
    if isinstance(rebin_dynamic_mask_cfg, dict):
        rebin_dynamic_mask_enable = bool(rebin_dynamic_mask_cfg.get("enable", False))
        rebin_dynamic_mask_fraction = float(rebin_dynamic_mask_cfg.get("fraction", 0.01))
        rebin_dynamic_mask_min_pixels = int(rebin_dynamic_mask_cfg.get("min_pixels", 256))
        rebin_dynamic_mask_warmup_epochs = int(rebin_dynamic_mask_cfg.get("warmup_epochs", 0))
        rebin_dynamic_mask_smooth_kernel = int(rebin_dynamic_mask_cfg.get("smooth_kernel", 0))
        rebin_dynamic_mask_clip_min = float(rebin_dynamic_mask_cfg.get("clip_min", 0.0))
        rebin_dynamic_mask_clip_max = float(rebin_dynamic_mask_cfg.get("clip_max", 1.0))
        rebin_dynamic_mask_stop_grad = bool(rebin_dynamic_mask_cfg.get("stop_grad", True))
    else:
        rebin_dynamic_mask_enable = bool(rebin_dynamic_mask_cfg)
        rebin_dynamic_mask_fraction = float(rebin_cfg.get("dynamic_mask_fraction", 0.01))
        rebin_dynamic_mask_min_pixels = int(rebin_cfg.get("dynamic_mask_min_pixels", 256))
        rebin_dynamic_mask_warmup_epochs = int(rebin_cfg.get("dynamic_mask_warmup_epochs", 0))
        rebin_dynamic_mask_smooth_kernel = int(rebin_cfg.get("dynamic_mask_smooth_kernel", 0))
        rebin_dynamic_mask_clip_min = float(rebin_cfg.get("dynamic_mask_clip_min", 0.0))
        rebin_dynamic_mask_clip_max = float(rebin_cfg.get("dynamic_mask_clip_max", 1.0))
        rebin_dynamic_mask_stop_grad = bool(rebin_cfg.get("dynamic_mask_stop_grad", True))

    save_interval = config["training"]["save_interval"]
    plot_interval = config["training"]["plot_interval"]

    model_type = config["model"]["name"]
    model_type_is_lsfp = is_lsfp_model(model_type)
    model_type_norm = str(model_type).strip().lower()
    mamba_variant = str(config.get("model", {}).get("mamba", {}).get("variant", "")).strip().lower()
    model_type_is_temporal_mamba = model_type_norm in {
        "mambatemporal",
        "mamba_temporal",
        "temporalmamba",
    } or (
        model_type_norm in {"mambarecon", "mamba_recon", "mamba"}
        and mamba_variant in {"temporal", "temporal_1d", "radial_temporal"}
    )
    use_adj_loss = model_type_is_lsfp
    if use_adj_loss:
        if "weight" not in adj_loss_cfg:
            raise KeyError("model.losses.adj_loss.weight is required for LSFPNet.")
        adj_loss_weight = float(adj_loss_cfg["weight"])
    else:
        # For non-LSFP architectures (e.g., Mamba), adjoint loss is disabled by design.
        # This is architecture-driven and not configurable.
        adj_loss_weight = 0.0

    H, W = config["data"]["height"], config["data"]["width"]
    N_time, N_samples, N_coils = (
        config["data"]["timeframes"],
        config["data"]["samples"],
        config["data"]["coils"]
    )
    Ng = config["data"]["fpg"] 

    total_spokes = config["data"]["total_spokes"]

    N_spokes = int(total_spokes / N_time)
    N_full = config['data']['height'] * math.pi / 2

    N_slices = config['data']['slices']
    num_slices_to_eval = config['data']['slices_to_eval']
    eval_frequency = config['data']['eval_frequency']

    eval_chunk_size = config["evaluation"]["chunk_size"]
    eval_chunk_overlap = config["evaluation"]["chunk_overlap"]

    def _use_eval_sliding_window(num_frames: int) -> bool:
        if int(num_frames) <= int(eval_chunk_size):
            return False
        if model_type_is_temporal_mamba:
            return False
        return True

    raw_grasp_slice_idx = config.get("evaluation", {}).get("raw_grasp_slice_idx", 95)
    eval_cfg = config.get("evaluation", {})
    val_noise_level = eval_cfg.get("val_noise_level", 0.05)
    dro_sim_source = eval_cfg.get("dro_sim_source", "espirit")
    dro_csmaps_source = eval_cfg.get("dro_csmaps_source", "espirit")
    dro_espirit_csmaps_dir = eval_cfg.get("dro_espirit_csmaps_dir")
    skip_raw_eval_if_invalid_slice = bool(eval_cfg.get("skip_raw_eval_if_invalid_slice", False))
    compute_ssdu_eval = bool(eval_cfg.get("compute_ssdu", False))
    try:
        ssdu_k_folds = int(eval_cfg.get("ssdu_k_folds", 4))
    except (TypeError, ValueError):
        ssdu_k_folds = 4
    ssdu_weighting = str(eval_cfg.get("ssdu_weighting", "sqrt_dcomp"))
    deterministic_val_ei = bool(eval_cfg.get("deterministic_val_ei", False))
    deterministic_val_ei_seed = eval_cfg.get("deterministic_val_ei_seed", 0)
    try:
        deterministic_val_ei_seed = int(deterministic_val_ei_seed)
    except (TypeError, ValueError):
        deterministic_val_ei_seed = 0

    cluster = config["experiment"].get("cluster", "Randi")

    flip_kspace = config["data"].get("flip_kspace", True)


    if config["data"]["train_spokes_per_frame"] != "None":
        train_spokes_per_frame = config["data"]["train_spokes_per_frame"]
    else:
        train_spokes_per_frame = None


    # Curriculum Learning Configuration
    curriculum_enabled = config['training']['curriculum_learning']['enabled']
    curriculum_phases = config['training']['curriculum_learning']['phases']
    low_spf_eval_targets = []
    if curriculum_enabled:
        low_spf_eval_targets = sorted({
            int(phase["eval_spokes_per_frame"])
            for phase in curriculum_phases
            if phase.get("eval_spokes_per_frame") is not None
            and int(phase["eval_spokes_per_frame"]) <= 8
        })

    # Initial setup for train_dataset based on the first phase if curriculum is enabled
    initial_train_spokes_range = [8, 16, 24, 36]
    if curriculum_enabled:
        if not curriculum_phases:
            raise ValueError("Curriculum learning enabled but no phases defined in config.yaml")
        initial_train_spokes_range = curriculum_phases[0]['train_spokes_range']
        print(f"Curriculum Learning Enabled. Initial training with spokes range: {initial_train_spokes_range}")


    # load data
    with open(split_file, "r") as fp:
        splits = json.load(fp)

    if max_subjects < 300:
        max_train = int(max_subjects * (1 - config["data"]["val_split_ratio"]))
        train_patient_ids = splits["train"][:max_train]
        
    else:
        train_patient_ids = splits["train"]

    val_patient_ids = splits["val"]
    val_dro_patient_ids = splits["val_dro"]


    # check for data leakage
    for val_id in val_patient_ids:
        if val_id in train_patient_ids:
            raise ValueError(f"Data Leakage encountered! Duplicate sample in train and val patient IDs: {val_id}")



    if config['dataloader']['slice_range_start'] == "None" or config['dataloader']['slice_range_end'] == "None":
        train_dataset = ZFSliceDataset(
            root_dir=data_dir,
            patient_ids=train_patient_ids,
            dataset_key=config["data"]["dataset_key"],
            file_pattern="*.h5",
            slice_idx=config["dataloader"]["slice_idx"],
            num_random_slices=config["dataloader"].get("num_random_slices", None),
            slice_sampling_mode=slice_sampling_mode,
            slice_sampling_uniform_fraction=slice_sampling_uniform_fraction,
            slice_sampling_filter_quantile=slice_sampling_filter_quantile,
            slice_sampling_no_replacement=slice_sampling_no_replacement,
            slice_sampling_cache_dir=slice_sampling_cache_dir,
            slice_sampling_cache_workers=slice_sampling_cache_workers,
            slice_sampling_cache_rank=slice_sampling_cache_rank,
            slice_sampling_cache_rank_only=slice_sampling_cache_rank_only,
            N_time=N_time,
            N_coils=N_coils,
            spf_aug=config['data']['spf_aug'],
            spokes_per_frame=train_spokes_per_frame,
            weight_accelerations=config['data']['weight_accelerations'],
            initial_spokes_range=initial_train_spokes_range,
            cluster=cluster,
            flip_kspace=flip_kspace,
        )
    else:
        train_dataset = ZFSliceDataset(
            root_dir=data_dir,
            patient_ids=train_patient_ids,
            dataset_key=config["data"]["dataset_key"],
            file_pattern="*.h5",
            slice_idx=range(config['dataloader']['slice_range_start'], config['dataloader']['slice_range_end']),
            num_random_slices=config["dataloader"].get("num_random_slices", None),
            slice_sampling_mode=slice_sampling_mode,
            slice_sampling_uniform_fraction=slice_sampling_uniform_fraction,
            slice_sampling_filter_quantile=slice_sampling_filter_quantile,
            slice_sampling_no_replacement=slice_sampling_no_replacement,
            slice_sampling_cache_dir=slice_sampling_cache_dir,
            slice_sampling_cache_workers=slice_sampling_cache_workers,
            slice_sampling_cache_rank=slice_sampling_cache_rank,
            slice_sampling_cache_rank_only=slice_sampling_cache_rank_only,
            N_time=N_time,
            N_coils=N_coils,
            spf_aug=config['data']['spf_aug'],
            spokes_per_frame=train_spokes_per_frame,
            weight_accelerations=config['data']['weight_accelerations'],
            initial_spokes_range=initial_train_spokes_range,
            cluster=cluster,
            flip_kspace=flip_kspace
        )


    if global_rank == 0:
        log_slice_sampling_startup_report(train_dataset, label="train", output_dir=output_dir)

    debug_cfg = config.get("debugging", {})
    sample_check_enabled = bool(debug_cfg.get("distributed_sample_check", False)) and config['training']['multigpu']
    sample_check_batches = int(debug_cfg.get("distributed_sample_check_batches", 5))
    if sample_check_batches < 0:
        sample_check_batches = 0

    train_dataset_for_loader = train_dataset
    include_sample_indices = False
    if sample_check_enabled:
        train_dataset_for_loader = _DatasetWithIndex(train_dataset)
        include_sample_indices = True
        if global_rank == 0:
            print(f"[DDP] Sample check enabled: collecting first {sample_check_batches} batches per epoch.")

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(12)

    if config['training']['multigpu']:
        train_sampler = DistributedSampler(train_dataset_for_loader, num_replicas=world_size, rank=global_rank)
        train_loader = DataLoader(
            train_dataset_for_loader,
            batch_size=config["dataloader"]["batch_size"],
            sampler=train_sampler,
            num_workers=config["dataloader"]["num_workers"],
            pin_memory=True,
            persistent_workers=bool(config["dataloader"]["num_workers"] > 0),
            worker_init_fn=seed_worker,
            generator=g,
        )
    else:
        train_loader = DataLoader(
            train_dataset_for_loader,
            batch_size=config["dataloader"]["batch_size"],
            shuffle=config["dataloader"]["shuffle"],
            num_workers=config["dataloader"]["num_workers"],
            pin_memory=True,
            persistent_workers=bool(config["dataloader"]["num_workers"] > 0),
            worker_init_fn=seed_worker,
            generator=g,
        )



    # Define eval physics and dataset. For curriculum learning, fix eval to the highest acceleration
    # (smallest spokes/frame) so plots remain consistent even before that phase is reached.
    fixed_eval_metrics = False
    if curriculum_enabled:
        eval_phases = [
            phase for phase in curriculum_phases
            if phase.get("eval_spokes_per_frame") is not None and phase.get("eval_num_frames") is not None
        ]
        if not eval_phases:
            raise ValueError("Curriculum learning is enabled but no eval settings were found in the phases.")
        eval_phase = min(
            eval_phases,
            key=lambda phase: phase["eval_spokes_per_frame"],
        )
        N_spokes_eval = eval_phase["eval_spokes_per_frame"]
        N_time_eval = eval_phase["eval_num_frames"]
        fixed_eval_metrics = True
    else:
        N_time_eval, N_spokes_eval = config["data"]["eval_timeframes"], config["data"]["eval_spokes"]

    if (
        model_type_is_temporal_mamba
        and int(N_time_eval) > int(eval_chunk_size)
        and (global_rank == 0 or not config["training"]["multigpu"])
    ):
        print(
            "[Eval] TemporalMamba detected: using direct full-sequence inference "
            f"(chunk_size={int(eval_chunk_size)} is ignored for model forward)."
        )

    # define physics object for evaluation
    eval_ktraj, eval_dcomp, eval_nufft_ob, eval_adjnufft_ob = get_nufft(
        N_samples, N_spokes_eval, N_time_eval
    )

    eval_physics = MCNUFFT(eval_nufft_ob, eval_adjnufft_ob, eval_ktraj, eval_dcomp)


    val_dro_dataset = SimulatedDataset(
        root_dir=config["evaluation"]["simulated_dataset_path"], 
        raw_kspace_path=data_dir,
        model_type=model_type, 
        patient_ids=val_dro_patient_ids,
        dataset_key=config["data"]["dataset_key"],
        spokes_per_frame=N_spokes_eval,
        num_frames=N_time_eval,
        traj_method=traj_method,
        grasp_slice_idx=raw_grasp_slice_idx,
        noise_level=val_noise_level,
        dro_csmaps_source=dro_csmaps_source,
        espirit_csmaps_dir=dro_espirit_csmaps_dir,
        dro_sim_source=dro_sim_source,
        skip_raw_eval_if_invalid_slice=skip_raw_eval_if_invalid_slice,
    )


    val_dro_loader = DataLoader(
        val_dro_dataset,
        batch_size=config["dataloader"]["batch_size"],
        shuffle=False,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=True,
        persistent_workers=bool(config["dataloader"]["num_workers"] > 0),
    )

    distributed_eval = bool(config.get("evaluation", {}).get("distributed_eval", True)) and config['training']['multigpu']

    def _build_val_dro_eval_loader(dataset):
        if not distributed_eval:
            return DataLoader(
                dataset,
                batch_size=config["dataloader"]["batch_size"],
                shuffle=False,
                num_workers=config["dataloader"]["num_workers"],
                pin_memory=True,
                persistent_workers=bool(config["dataloader"]["num_workers"] > 0),
            )
        shard_indices = _build_strided_shard_indices(len(dataset), global_rank, world_size)
        shard_dataset = Subset(dataset, shard_indices)
        return DataLoader(
            shard_dataset,
            batch_size=config["dataloader"]["batch_size"],
            shuffle=False,
            num_workers=config["dataloader"]["num_workers"],
            pin_memory=True,
            persistent_workers=bool(config["dataloader"]["num_workers"] > 0),
        )

    val_dro_eval_loader = _build_val_dro_eval_loader(val_dro_dataset)

    if distributed_eval:
        local_eval_count = len(getattr(val_dro_eval_loader, "dataset", []))
        local_eval_count_tensor = torch.tensor([local_eval_count], dtype=torch.int64, device=device)
        gathered_counts = [torch.zeros_like(local_eval_count_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_counts, local_eval_count_tensor)
        if global_rank == 0:
            per_rank_counts = [int(x.item()) for x in gathered_counts]
            print(
                "[Eval] Distributed validation enabled: "
                f"total_samples={len(val_dro_dataset)}, "
                f"per-rank min/median/max={min(per_rank_counts)}/{int(np.median(per_rank_counts))}/{max(per_rank_counts)}"
            )

    save_csmaps = config.get("debugging", {}).get("save_csmap_pngs", False)
    if save_csmaps and (global_rank == 0 or not config['training']['multigpu']):
        csmap_dir = os.path.join(output_dir, "csmap_checks")
        max_coils = config.get("debugging", {}).get("csmap_plot_max_coils", 16)
        try:
            train_batch = next(iter(train_loader))
            train_csmap = train_batch[1]
            save_csmap_png(train_csmap, csmap_dir, "train_raw", max_coils=max_coils)

            train_csmap_rot = torch.rot90(train_csmap, k=2, dims=[-2, -1])
            save_csmap_png(train_csmap_rot, csmap_dir, "train_rot", max_coils=max_coils)

        except Exception as exc:
            print(f"CSMap plot skipped for train_raw: {exc}")

        # try:
        val_batch = next(iter(val_dro_loader))
        _, val_csmap, _, _, _, _, _, _, val_raw_csmaps = val_batch

        # print("val_csmap norm check: ", np.sum(np.abs(val_csmap[:, :, :, 100, 100])**2))
        # print("val_raw_csmaps norm check: ", np.sum(np.abs(val_raw_csmaps[:, :, :, 100, 100])**2))

        save_csmap_png(val_csmap.squeeze(0), csmap_dir, "val_dro", max_coils=max_coils)
        raw_eval_available = _raw_eval_available(val_raw_csmaps)
        if raw_eval_available:
            save_csmap_png(val_raw_csmaps.squeeze(0), csmap_dir, "val_raw", max_coils=max_coils)


        # --- ensure S has shape (C, H, W) ---
        S_dro = val_csmap.squeeze()         # (C,H,W) if val_csmap was (1,C,H,W)
        # --- 1) Sum of magnitudes across coils: abs(S).sum(axis=0) ---
        dro_sumabs = np.abs(S_dro).sum(axis=0)   # (H,W)
        if raw_eval_available:
            S_raw = val_raw_csmaps.squeeze()    # (C,H,W) if val_raw_csmaps was (1,C,H,W)
            raw_sumabs = np.abs(S_raw).sum(axis=0)   # (H,W)

        # --- 2) (Optional but very useful) RSS magnitude across coils: sqrt(sum |S|^2) ---
        dro_rss = np.sqrt((np.abs(S_dro) ** 2).sum(axis=0))
        if raw_eval_available:
            raw_rss = np.sqrt((np.abs(S_raw) ** 2).sum(axis=0))

        # --- Plot & save ---
        os.makedirs(csmap_dir, exist_ok=True)

        plt.figure(figsize=(6, 6))
        plt.title("DRO: sum |S| across coils")
        plt.imshow(dro_sumabs, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(csmap_dir, "val_dro_sumabs.png"), dpi=200)
        plt.close()

        if raw_eval_available:
            plt.figure(figsize=(6, 6))
            plt.title("Raw: sum |S| across coils")
            plt.imshow(raw_sumabs, cmap="gray")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(csmap_dir, "val_raw_sumabs.png"), dpi=200)
            plt.close()

        # Optional RSS saves (recommended)
        plt.figure(figsize=(6, 6))
        plt.title("DRO: RSS = sqrt(sum |S|^2)")
        plt.imshow(dro_rss, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(csmap_dir, "val_dro_rss.png"), dpi=200)
        plt.close()

        if raw_eval_available:
            plt.figure(figsize=(6, 6))
            plt.title("Raw: RSS = sqrt(sum |S|^2)")
            plt.imshow(raw_rss, cmap="gray")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(csmap_dir, "val_raw_rss.png"), dpi=200)
            plt.close()
            
        # except Exception as exc:
        #     print(f"CSMap plot skipped for val data: {exc}")




    # define model
    model = build_recon_model(config, device=device, block_dir=block_dir)

    if config['training']['multigpu']:
        find_unused = config["training"].get("ddp_find_unused_parameters", True)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=find_unused)
        if config["training"].get("ddp_set_static_graph", False):
            model._set_static_graph()

    # Temporal Mamba EI calls a second model forward per iteration. Under DDP,
    # route EI through the underlying module to avoid reducer state conflicts.
    model_for_ei = model
    if config['training']['multigpu'] and model_type_is_temporal_mamba:
        model_for_ei = model.module
        if global_rank == 0:
            print("[EI] Using model.module for temporal Mamba EI forward under DDP.")
    ei_gradnorm_params = []
    if use_ei_loss and ei_gradnorm_transition_enable:
        ei_gradnorm_params = [p for p in model_for_ei.parameters() if p.requires_grad]
        if (global_rank == 0 or not config['training']['multigpu']) and not ei_gradnorm_params:
            print("[EI] GradNorm transition calibration disabled: no trainable parameters found.")
        if not ei_gradnorm_params:
            ei_gradnorm_transition_enable = False

    if global_rank == 0 or not config['training']['multigpu']:
        if run_state is not None and ("total_params" not in run_state or "trainable_params" not in run_state):
            total_params, trainable_params = _get_param_counts(model)
            with state_lock:
                run_state["total_params"] = total_params
                run_state["trainable_params"] = trainable_params
                _save_run_state(run_state_path, run_state)
            print(f"[RunState] Total parameters: {total_params:,}")
            print(f"[RunState] Trainable parameters: {trainable_params:,}")


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["model"]["optimizer"]["lr"],
        betas=(config["model"]["optimizer"]["b1"], config["model"]["optimizer"]["b2"]),
        eps=config["model"]["optimizer"]["eps"],
        weight_decay=config["model"]["optimizer"]["weight_decay"],
    )


    # Load the checkpoint to resume training
    if resume_from_checkpoint:
        model, optimizer, start_epoch, target_w_ei, step0_train_ei_loss, epoch_train_mc_loss, train_curves, val_curves, eval_curves, avg_grasp_ssim, avg_grasp_psnr, avg_grasp_mse, avg_grasp_lpips, avg_grasp_dc_mse, avg_grasp_dc_mae, avg_grasp_curve_corr, avg_grasp_raw_dc_mae, avg_grasp_raw_dc_mse = load_checkpoint(model, optimizer, checkpoint_file)
        if init_checkpoint and (global_rank == 0 or not config['training']['multigpu']):
            print(
                f"[Checkpoint] Found {init_checkpoint_source}={init_checkpoint}, "
                "but an existing resume checkpoint was detected; resuming instead."
            )

    else:
        start_epoch = 1
        if init_checkpoint:
            if not os.path.isfile(init_checkpoint):
                raise FileNotFoundError(
                    f"Init checkpoint not found: {init_checkpoint}"
                )

            skip_prefixes = []
            enc_skip_reason = None
            ckpt_cfg_path = os.path.join(os.path.dirname(init_checkpoint), "config.yaml")
            if os.path.isfile(ckpt_cfg_path):
                try:
                    with open(ckpt_cfg_path, "r") as file:
                        ckpt_cfg = yaml.safe_load(file) or {}
                    ckpt_model_cfg = ckpt_cfg.get("model", {})
                    ckpt_encode_acc = bool(ckpt_model_cfg.get("encode_acceleration", False))
                    ckpt_encode_time = bool(ckpt_model_cfg.get("encode_time_index", False))
                    cur_encode_acc = bool(config["model"].get("encode_acceleration", False))
                    cur_encode_time = bool(config["model"].get("encode_time_index", False))
                    if (cur_encode_acc and not ckpt_encode_acc) or (cur_encode_time and not ckpt_encode_time):
                        skip_prefixes.append("mapping_network.")
                        enc_skip_reason = (
                            f"checkpoint encodings (acc={ckpt_encode_acc}, time={ckpt_encode_time}) "
                            f"do not include current encodings (acc={cur_encode_acc}, time={cur_encode_time})"
                        )
                except Exception as exc:
                    if global_rank == 0 or not config['training']['multigpu']:
                        print(f"[Checkpoint] Warning: failed to read {ckpt_cfg_path}: {exc}")

            model, preload_info = load_pretrained_weights(
                model,
                init_checkpoint,
                skip_prefixes=skip_prefixes,
            )
            if global_rank == 0 or not config['training']['multigpu']:
                if init_checkpoint_source == "experiment.pretrained_checkpoint":
                    print(
                        "[Checkpoint] experiment.pretrained_checkpoint is deprecated; "
                        "use experiment.init_checkpoint instead."
                    )
                print(f"[Checkpoint] Loaded init weights from {init_checkpoint}")
                if enc_skip_reason:
                    print(f"[Checkpoint] Skipping encoding weights: {enc_skip_reason}")
                if preload_info.get("mismatched_keys"):
                    mismatched_preview = ", ".join(
                        k for k, _, _ in preload_info["mismatched_keys"][:5]
                    )
                    print(f"[Checkpoint] Skipped {len(preload_info['mismatched_keys'])} mismatched keys (e.g., {mismatched_preview})")
                print(
                    "[Checkpoint] Preload summary: "
                    f"loaded={preload_info.get('loaded_keys')}, "
                    f"skipped={preload_info.get('skipped_keys')}, "
                    f"missing={preload_info.get('missing_keys')}, "
                    f"unexpected={preload_info.get('unexpected_keys')}"
                )
                if run_state is not None:
                    with state_lock:
                        run_state["init_checkpoint"] = {
                            "loaded": True,
                            "path": init_checkpoint,
                            "source": init_checkpoint_source,
                            "epoch": preload_info.get("checkpoint_epoch"),
                        }
                        _save_run_state(run_state_path, run_state)
        # target_w_ei = 0.0


    # select metric for loss functions
    if config['model']['losses']['mc_loss']['metric'] == "MSE":
        mc_loss_fn = MCLoss(model_type=model_type)
    elif config['model']['losses']['mc_loss']['metric'] == "MAE":
        mc_loss_fn = MCLoss(model_type=model_type, metric=torch.nn.L1Loss())
    else:
        raise(ValueError, "Unsupported MC Loss Metric.")


    ei_metric_name = str(ei_cfg.get("metric", "MSE")).strip().upper()
    if ei_metric_name == "MAE":
        ei_loss_metric = torch.nn.L1Loss()
    elif ei_metric_name == "MSE":
        ei_loss_metric = torch.nn.MSELoss()
    else:
        raise ValueError(
            f"Unsupported EI Loss Metric '{ei_metric_name}'. Expected one of: "
            "MSE, MAE."
        )

    ei_no_grad = bool(ei_cfg.get("no_grad", False))
    # Temporal-Mamba EI can hit autograd versioning failures with full target-path
    # gradients; default to stop-grad target unless explicitly disabled.
    if model_type_is_temporal_mamba and bool(ei_cfg.get("force_no_grad_for_mamba", True)):
        ei_no_grad = True
    ei_checkpoint_model = ei_cfg.get("checkpoint_model", False)
    ei_checkpoint_mode = ei_cfg.get("checkpoint_mode", "none")
    ei_checkpoint_use_reentrant = ei_cfg.get("checkpoint_use_reentrant", False)

    if use_rebin_loss:
        if rebin_metric_name.upper() == "MAE":
            rebin_metric = torch.nn.L1Loss()
        else:
            rebin_metric = torch.nn.MSELoss()
        rebin_loss_fn = RebinConsistencyLoss(
            factor=rebin_factor,
            metric=rebin_metric,
            time_index_mode=rebin_time_index_mode,
            teacher_branch=rebin_teacher_branch,
            teacher_stopgrad=rebin_teacher_stopgrad,
            offset_mode=rebin_offset_mode,
            temporal_mode=rebin_temporal_mode,
            baseline_frames=rebin_baseline_frames,
            percent_enhancement_eps=rebin_percent_enhancement_eps,
            dynamic_mask_enable=rebin_dynamic_mask_enable,
            dynamic_mask_fraction=rebin_dynamic_mask_fraction,
            dynamic_mask_min_pixels=rebin_dynamic_mask_min_pixels,
            dynamic_mask_warmup_epochs=rebin_dynamic_mask_warmup_epochs,
            dynamic_mask_smooth_kernel=rebin_dynamic_mask_smooth_kernel,
            dynamic_mask_clip_min=rebin_dynamic_mask_clip_min,
            dynamic_mask_clip_max=rebin_dynamic_mask_clip_max,
            dynamic_mask_stop_grad=rebin_dynamic_mask_stop_grad,
        )
        if global_rank == 0 or not config['training']['multigpu']:
            print(
                "[Rebin] "
                f"factor={rebin_factor}, teacher={rebin_teacher_branch}, stopgrad={rebin_teacher_stopgrad}, "
                f"offset_mode={rebin_offset_mode}, temporal_mode={rebin_temporal_mode}, "
                f"dynamic_mask={rebin_dynamic_mask_enable}, "
                f"target_weight={rebin_target_w}, warmup={rebin_warmup}, duration={rebin_duration}"
            )
    else:
        rebin_loss_fn = None

    if not use_ei_loss:
        ei_gradnorm_transition_enable = False
    if ei_gradnorm_transition_enable and (global_rank == 0 or not config['training']['multigpu']):
        print(
            "[EI] GradNorm transition calibration enabled: "
            f"window=[{ei_transition_start_epoch}, {ei_transition_end_epoch}], "
            f"ema_beta={ei_gradnorm_ema_beta}, ratio_clip=[{ei_gradnorm_ratio_min}, {ei_gradnorm_ratio_max}], "
            f"target_scale_clip=[{ei_gradnorm_target_scale_min}, {ei_gradnorm_target_scale_max}], "
            f"measure_every_n_steps={ei_gradnorm_measure_every}"
        )

    # Optional: one-shot diagnostic plot for rebin consistency (runs before training).
    rebin_diag_cfg = config.get("debugging", {}).get("rebin_diagnostic", {})
    rebin_diag_enabled = bool(rebin_diag_cfg.get("enable", False))
    if rebin_diag_enabled and (global_rank == 0 or not config['training']['multigpu']):
        def _scalar_int(val, default: int = 0) -> int:
            try:
                if torch.is_tensor(val):
                    if val.numel() == 0:
                        return int(default)
                    if val.numel() == 1:
                        return int(val.item())
                    return int(val.reshape(-1)[0].item())
                return int(val)
            except Exception:
                return int(default)

        diag_factor = int(rebin_diag_cfg.get("factor", rebin_factor))
        diag_start_mode = str(rebin_diag_cfg.get("start_time_index", "middle")).lower()
        diag_search_batches = _scalar_int(rebin_diag_cfg.get("search_batches", 8), default=8)
        diag_mask_fraction = float(rebin_diag_cfg.get("mask_fraction", 0.01))
        diag_min_pixels = _scalar_int(rebin_diag_cfg.get("min_pixels", 256), default=256)
        diag_baseline_frames = _scalar_int(rebin_diag_cfg.get("baseline_frames", 4), default=4)
        diag_time_index_mode = str(rebin_diag_cfg.get("time_index_mode", rebin_time_index_mode)).lower()
        diag_out_subdir = str(rebin_diag_cfg.get("out_subdir", "rebin_diagnostics"))

        if diag_factor > 1:
            try:
                os.makedirs(os.path.join(output_dir, diag_out_subdir), exist_ok=True)

                # Pick a batch, preferably at the highest acceleration (smallest spokes/frame).
                best_batch = None
                best_spf = None
                min_spf_target = min(initial_train_spokes_range) if initial_train_spokes_range else None
                for i, batch in enumerate(iter(train_loader)):
                    if include_sample_indices:
                        _, measured_kspace, csmap, N_samples, N_spokes, N_time = batch
                    else:
                        measured_kspace, csmap, N_samples, N_spokes, N_time = batch

                    spf = _scalar_int(N_spokes, default=10**9)
                    if best_batch is None or spf < best_spf:
                        best_batch = (measured_kspace, csmap, N_samples, N_spokes, N_time)
                        best_spf = spf
                        if min_spf_target is not None and best_spf <= int(min_spf_target):
                            break

                    if i + 1 >= max(1, diag_search_batches):
                        break

                if best_batch is None:
                    raise RuntimeError("Rebin diagnostic: could not fetch a batch from train_loader.")

                measured_kspace, csmap, N_samples, N_spokes, N_time = best_batch
                if measured_kspace.shape[0] != 1:
                    measured_kspace = measured_kspace[:1]
                    csmap = csmap[:1]

                N_samples_i = _scalar_int(N_samples, default=0)
                N_spokes_i = _scalar_int(N_spokes, default=0)
                N_time_i = _scalar_int(N_time, default=0)

                # prepare inputs
                measured_kspace_cplx = to_torch_complex(measured_kspace)
                measured_kspace_cplx = measured_kspace_cplx[0]  # (T,co,sp,sam)
                measured_kspace_cplx = rearrange(measured_kspace_cplx, 't co sp sam -> co (sp sam) t')

                # prep physics operators
                ktraj, dcomp, nufft_ob, adjnufft_ob = get_nufft(
                    N_samples_i, N_spokes_i, N_time_i
                )

                # Crop to training window length (Ng) for a faithful diagnostic.
                if N_time_i > Ng:
                    max_idx = N_time_i - Ng
                    if diag_start_mode == "random":
                        start_idx = sample_start_time_index(max_idx, use_edge_time_index_sampling)
                    elif diag_start_mode == "zero":
                        start_idx = 0
                    elif diag_start_mode == "middle":
                        start_idx = max_idx // 2
                    else:
                        start_idx = _scalar_int(diag_start_mode, default=0)
                        start_idx = max(0, min(int(start_idx), int(max_idx)))

                    measured_kspace_cplx = measured_kspace_cplx[..., start_idx:start_idx + Ng]
                    ktraj_chunk = ktraj[..., start_idx:start_idx + Ng]
                    dcomp_chunk = dcomp[..., start_idx:start_idx + Ng]
                    physics = MCNUFFT(nufft_ob, adjnufft_ob, ktraj_chunk, dcomp_chunk)
                    start_timepoint_index = torch.tensor([start_idx], dtype=torch.float, device=device)
                else:
                    physics = MCNUFFT(nufft_ob, adjnufft_ob, ktraj, dcomp)
                    start_timepoint_index = torch.tensor([0], dtype=torch.float, device=device)

                y_hi = measured_kspace_cplx.to(device)
                csmap = csmap.to(device).to(y_hi.dtype)

                acceleration = torch.tensor([N_full / int(N_spokes_i)], dtype=torch.float, device=device)
                acceleration_encoding = acceleration if config['model']['encode_acceleration'] else None
                if config['model']['encode_time_index'] == False:
                    start_timepoint_index = None

                # High-temporal recon.
                model_was_training = model.training
                model.eval()
                with torch.no_grad():
                    with amp_autocast():
                        x_hi, *_ = model(
                            y_hi,
                            physics,
                            csmap,
                            acceleration_encoding,
                            start_timepoint_index,
                            epoch="rebin_diag_hi",
                            norm=config['model']['norm'],
                        )

                # Low-temporal recon from rebinned measurement/operator.
                x_hi_down = RebinConsistencyLoss._downsample_time_mean(x_hi, diag_factor)
                y_lo = RebinConsistencyLoss._rebin_kspace(y_hi, N_spokes_i, N_samples_i, diag_factor)
                ktraj_lo, dcomp_lo = RebinConsistencyLoss._rebin_ktraj_dcomp(
                    physics, N_spokes_i, N_samples_i, diag_factor
                )
                physics_lo = MCNUFFT(physics.nufft_ob, physics.adjnufft_ob, ktraj_lo, dcomp_lo).to(csmap.device)

                if acceleration_encoding is None:
                    acceleration_lo = None
                else:
                    acceleration_lo = acceleration_encoding / float(diag_factor)

                if diag_time_index_mode == "none":
                    start_idx_lo = None
                elif diag_time_index_mode == "scaled":
                    start_idx_lo = None if start_timepoint_index is None else (start_timepoint_index / float(diag_factor))
                else:  # inherit
                    start_idx_lo = start_timepoint_index

                with torch.no_grad():
                    with amp_autocast():
                        x_lo, *_ = model(
                            y_lo,
                            physics_lo,
                            csmap,
                            acceleration_lo,
                            start_idx_lo,
                            epoch="rebin_diag_lo",
                            norm=config['model']['norm'],
                        )

                out_path = os.path.join(
                    output_dir,
                    diag_out_subdir,
                    f"rebin_diag_spf{int(N_spokes_i)}_factor{int(diag_factor)}.png",
                )
                diag_title = (
                    f"Rebin diagnostic (spf_hi={int(N_spokes_i)}, AF={float(acceleration.item()):.2f}, "
                    f"factor={int(diag_factor)}, start={diag_start_mode})"
                )
                diag = plot_rebin_consistency_diagnostic(
                    x_hi,
                    x_lo,
                    diag_factor,
                    out_path,
                    x_hi_down=x_hi_down,
                    mask_fraction=diag_mask_fraction,
                    min_pixels=diag_min_pixels,
                    baseline_frames=diag_baseline_frames,
                    title=diag_title,
                )
                try:
                    curve_corr = float(np.corrcoef(diag["curve_hi_down"], diag["curve_lo"])[0, 1])
                except Exception:
                    curve_corr = float("nan")
                print(
                    "[Debug] Saved rebin diagnostic plot to "
                    f"{out_path} (corr(Avg(x_hi), x_lo)={curve_corr:.3f})."
                )

                if model_was_training:
                    model.train()
            except Exception as exc:
                print(f"[Debug] Rebin diagnostic plot skipped: {exc}")


    # define EI loss transformations
    if use_ei_loss:
        rotate = VideoRotate(n_trans=1, interpolation_mode="bilinear", degrees=config['model']['losses']['ei_loss'].get("rotate_range", 180))
        diffeo = VideoDiffeo(n_trans=1, device=device)

        subsample = SubsampleTime(
            n_trans=1,
            subsample_ratio_range=(
                config['model']['losses']['ei_loss'].get("subsample_ratio_min", 0.7),
                config['model']['losses']['ei_loss'].get("subsample_ratio_max", 0.95),
            ),
        )
        monophasic_warp = MonophasicTimeWarp(
            n_trans=1,
            warp_ratio_range=(
                config['model']['losses']['ei_loss'].get("warp_ratio_min", 0.7),
                config['model']['losses']['ei_loss'].get("warp_ratio_max", 1.3),
            ),
            pre_contrast_baseline=config['model']['losses']['ei_loss'].get("pre_contrast_baseline", "first_frame"),
            baseline_seconds=config['model']['losses']['ei_loss'].get("baseline_seconds", 20),
            buffer_frames=config['model']['losses']['ei_loss'].get("buffer_frames", 0),
        )
        shift_jitter = TemporalShiftJitterAfterBaseline(
            n_trans=1,
            max_shift=config['model']['losses']['ei_loss'].get("shift_jitter_max_shift", 2),
            pre_contrast_baseline=config['model']['losses']['ei_loss'].get("pre_contrast_baseline", "first_frame"),
            baseline_seconds=config['model']['losses']['ei_loss'].get("baseline_seconds", 20),
            buffer_frames=config['model']['losses']['ei_loss'].get("buffer_frames", 0),
        )
        arrival_shift = BolusArrivalTimeShift(
            n_trans=1,
            max_shift=config['model']['losses']['ei_loss'].get("arrival_shift_max_shift", 2),
            percentile=config['model']['losses']['ei_loss'].get("arrival_shift_percentile", 0.95),
            baseline_k=config['model']['losses']['ei_loss'].get("arrival_shift_baseline_k", 2.0),
            arrival_method=config['model']['losses']['ei_loss'].get("arrival_method", "threshold"),
            arrival_fraction=config['model']['losses']['ei_loss'].get("arrival_fraction", 0.1),
            pre_contrast_baseline=config['model']['losses']['ei_loss'].get("pre_contrast_baseline", "n_frames"),
            baseline_seconds=config['model']['losses']['ei_loss'].get("baseline_seconds", 20),
            total_seconds=config['model']['losses']['ei_loss'].get("total_seconds", 150.0),
        )
        enh_scale = BaselineEnhancementScale(
            n_trans=1,
            scale_range=(
                config['model']['losses']['ei_loss'].get("enh_scale_min", 0.8),
                config['model']['losses']['ei_loss'].get("enh_scale_max", 1.2),
            ),
            pre_contrast_baseline=config['model']['losses']['ei_loss'].get("pre_contrast_baseline", "first_frame"),
            baseline_seconds=config['model']['losses']['ei_loss'].get("baseline_seconds", 20),
            total_seconds=config['model']['losses']['ei_loss'].get("total_seconds", 150.0),
            buffer_frames=config['model']['losses']['ei_loss'].get("buffer_frames", 0),
            start_mode=config['model']['losses']['ei_loss'].get("enh_scale_start", "baseline"),
            arrival_percentile=config['model']['losses']['ei_loss'].get("arrival_shift_percentile", 0.95),
            arrival_baseline_k=config['model']['losses']['ei_loss'].get("arrival_shift_baseline_k", 2.0),
            arrival_method=config['model']['losses']['ei_loss'].get("arrival_method", "threshold"),
            arrival_fraction=config['model']['losses']['ei_loss'].get("arrival_fraction", 0.1),
        )
        temp_noise = TemporalNoise(n_trans=1, noise_strength=config['model']['losses']['ei_loss'].get("noise_strength", 0.5))
        time_reverse = TimeReverse(n_trans=1)

        if config['model']['losses']['ei_loss']['temporal_transform'] == "subsample":
            if config['model']['losses']['ei_loss']['spatial_transform'] == "none":
                ei_loss_fn = EILoss(subsample, metric=ei_loss_metric, model_type=model_type, no_grad=ei_no_grad, checkpoint_model=ei_checkpoint_model, checkpoint_mode=ei_checkpoint_mode, checkpoint_use_reentrant=ei_checkpoint_use_reentrant)
            else:
                ei_loss_fn = EILoss(subsample | (diffeo | rotate), metric=ei_loss_metric, model_type=model_type, no_grad=ei_no_grad, checkpoint_model=ei_checkpoint_model, checkpoint_mode=ei_checkpoint_mode, checkpoint_use_reentrant=ei_checkpoint_use_reentrant)
        elif config['model']['losses']['ei_loss']['temporal_transform'] == "warp":
            if config['model']['losses']['ei_loss']['spatial_transform'] == "none":
                ei_loss_fn = EILoss(monophasic_warp, metric=ei_loss_metric, model_type=model_type, no_grad=ei_no_grad, checkpoint_model=ei_checkpoint_model, checkpoint_mode=ei_checkpoint_mode, checkpoint_use_reentrant=ei_checkpoint_use_reentrant)
            else:
                ei_loss_fn = EILoss(monophasic_warp | (diffeo | rotate), metric=ei_loss_metric, model_type=model_type, no_grad=ei_no_grad, checkpoint_model=ei_checkpoint_model, checkpoint_mode=ei_checkpoint_mode, checkpoint_use_reentrant=ei_checkpoint_use_reentrant)
        elif config['model']['losses']['ei_loss']['temporal_transform'] == "shift_jitter":
            if config['model']['losses']['ei_loss']['spatial_transform'] == "none":
                ei_loss_fn = EILoss(shift_jitter, metric=ei_loss_metric, model_type=model_type, no_grad=ei_no_grad, checkpoint_model=ei_checkpoint_model, checkpoint_mode=ei_checkpoint_mode, checkpoint_use_reentrant=ei_checkpoint_use_reentrant)
            else:
                ei_loss_fn = EILoss(shift_jitter | (diffeo | rotate), metric=ei_loss_metric, model_type=model_type, no_grad=ei_no_grad, checkpoint_model=ei_checkpoint_model, checkpoint_mode=ei_checkpoint_mode, checkpoint_use_reentrant=ei_checkpoint_use_reentrant)
        elif config['model']['losses']['ei_loss']['temporal_transform'] == "arrival_shift":
            if config['model']['losses']['ei_loss']['spatial_transform'] == "none":
                ei_loss_fn = EILoss(arrival_shift, metric=ei_loss_metric, model_type=model_type, no_grad=ei_no_grad, checkpoint_model=ei_checkpoint_model, checkpoint_mode=ei_checkpoint_mode, checkpoint_use_reentrant=ei_checkpoint_use_reentrant)
            else:
                ei_loss_fn = EILoss(arrival_shift | (diffeo | rotate), metric=ei_loss_metric, model_type=model_type, no_grad=ei_no_grad, checkpoint_model=ei_checkpoint_model, checkpoint_mode=ei_checkpoint_mode, checkpoint_use_reentrant=ei_checkpoint_use_reentrant)
        elif config['model']['losses']['ei_loss']['temporal_transform'] == "enh_scale":
            if config['model']['losses']['ei_loss']['spatial_transform'] == "none":
                ei_loss_fn = EILoss(enh_scale, metric=ei_loss_metric, model_type=model_type, no_grad=ei_no_grad, checkpoint_model=ei_checkpoint_model, checkpoint_mode=ei_checkpoint_mode, checkpoint_use_reentrant=ei_checkpoint_use_reentrant)
            else:
                ei_loss_fn = EILoss(enh_scale | (diffeo | rotate), metric=ei_loss_metric, model_type=model_type, no_grad=ei_no_grad, checkpoint_model=ei_checkpoint_model, checkpoint_mode=ei_checkpoint_mode, checkpoint_use_reentrant=ei_checkpoint_use_reentrant)
        elif config['model']['losses']['ei_loss']['temporal_transform'] == "arrival_shift_enh_scale":
            ei_loss_fn = EILoss((arrival_shift | enh_scale) | (diffeo | rotate), metric=ei_loss_metric, model_type=model_type, no_grad=ei_no_grad, checkpoint_model=ei_checkpoint_model, checkpoint_mode=ei_checkpoint_mode, checkpoint_use_reentrant=ei_checkpoint_use_reentrant)
        elif config['model']['losses']['ei_loss']['temporal_transform'] == "noise":
            ei_loss_fn = EILoss(temp_noise, metric=ei_loss_metric, model_type=model_type, no_grad=ei_no_grad, checkpoint_model=ei_checkpoint_model, checkpoint_mode=ei_checkpoint_mode, checkpoint_use_reentrant=ei_checkpoint_use_reentrant)
        elif config['model']['losses']['ei_loss']['temporal_transform'] == "warp_subsample":
            ei_loss_fn = EILoss((subsample | monophasic_warp) | (diffeo | rotate), metric=ei_loss_metric, model_type=model_type, no_grad=ei_no_grad, checkpoint_model=ei_checkpoint_model, checkpoint_mode=ei_checkpoint_mode, checkpoint_use_reentrant=ei_checkpoint_use_reentrant)
        elif config['model']['losses']['ei_loss']['temporal_transform'] == "none":
            if config['model']['losses']['ei_loss']['spatial_transform'] == "rotate":
                ei_loss_fn = EILoss(rotate, metric=ei_loss_metric, model_type=model_type, no_grad=ei_no_grad, checkpoint_model=ei_checkpoint_model, checkpoint_mode=ei_checkpoint_mode, checkpoint_use_reentrant=ei_checkpoint_use_reentrant)
            elif config['model']['losses']['ei_loss']['spatial_transform'] == "diffeo":
                ei_loss_fn = EILoss(diffeo, metric=ei_loss_metric, model_type=model_type, no_grad=ei_no_grad, checkpoint_model=ei_checkpoint_model, checkpoint_mode=ei_checkpoint_mode, checkpoint_use_reentrant=ei_checkpoint_use_reentrant)
            else:
                ei_loss_fn = EILoss(rotate | diffeo, metric=ei_loss_metric, model_type=model_type, no_grad=ei_no_grad, checkpoint_model=ei_checkpoint_model, checkpoint_mode=ei_checkpoint_mode, checkpoint_use_reentrant=ei_checkpoint_use_reentrant)
        elif config['model']['losses']['ei_loss']['spatial_transform'] == "all":
            if config['model']['losses']['ei_loss']['temporal_transform'] == "all":
                ei_loss_fn = EILoss((subsample | monophasic_warp | temp_noise) | (diffeo | rotate), metric=ei_loss_metric, model_type=model_type, no_grad=ei_no_grad, checkpoint_model=ei_checkpoint_model, checkpoint_mode=ei_checkpoint_mode, checkpoint_use_reentrant=ei_checkpoint_use_reentrant)
        else:
            raise(ValueError, "Unsupported Temporal Transform.")



    if resume_from_checkpoint:
        train_mc_losses = train_curves["train_mc_losses"]
        val_mc_losses = val_curves["val_mc_losses"]
        train_ei_losses = train_curves["train_ei_losses"]
        val_ei_losses = val_curves["val_ei_losses"]
        train_adj_losses = train_curves["train_adj_losses"]
        val_adj_losses = val_curves["val_adj_losses"]
        weighted_train_mc_losses = train_curves["weighted_train_mc_losses"]
        weighted_train_ei_losses = train_curves["weighted_train_ei_losses"]
        weighted_train_adj_losses = train_curves["weighted_train_adj_losses"]
        train_rebin_losses = train_curves.get("train_rebin_losses", [])
        weighted_train_rebin_losses = train_curves.get("weighted_train_rebin_losses", [])
        lr_history = train_curves.get("lr_history", [])
        lr_epochs = train_curves.get("lr_epochs", [])
        ei_weight_history = train_curves.get("ei_weight_history", [])
        ei_weight_epochs = train_curves.get("ei_weight_epochs", [])
        ei_gradnorm_ratio_history = train_curves.get("ei_gradnorm_ratio_history", [])
        ei_gradnorm_ratio_epochs = train_curves.get("ei_gradnorm_ratio_epochs", [])
        rebin_weight_history = train_curves.get("rebin_weight_history", [])
        rebin_weight_epochs = train_curves.get("rebin_weight_epochs", [])
        ei_gradnorm_ratio_ema = train_curves.get("ei_gradnorm_ratio_ema", None)
        if ei_gradnorm_ratio_ema is not None:
            ei_gradnorm_ratio_ema = float(ei_gradnorm_ratio_ema)
        ei_gradnorm_samples = int(train_curves.get("ei_gradnorm_samples", 0))
        ei_gradnorm_locked = bool(train_curves.get("ei_gradnorm_locked", False))
        ei_target_weight_base = float(train_curves.get("ei_target_weight_base", target_w_ei))
        ei_target_weight_effective = float(train_curves.get("ei_target_weight_effective", target_w_ei))
        eval_ssims = eval_curves["eval_ssims"]
        eval_psnrs = eval_curves["eval_psnrs"]
        eval_mses = eval_curves["eval_mses"]
        eval_lpipses = eval_curves["eval_lpipses"]
        eval_dc_mses = eval_curves["eval_dc_mses"]
        eval_dc_maes = eval_curves["eval_dc_maes"]
        eval_raw_dc_mses = eval_curves.get("eval_raw_dc_mses", [])
        eval_raw_dc_maes = eval_curves.get("eval_raw_dc_maes", [])
        eval_curve_corrs = eval_curves["eval_curve_corrs"]
        eval_temporal_epochs = eval_curves.get("eval_temporal_epochs", [])
        eval_curve_maes = eval_curves.get("eval_curve_maes", [])
        eval_ttae_secs = eval_curves.get("eval_ttae_secs", [])
        eval_iauc10_errs = eval_curves.get("eval_iauc10_errs", [])
        eval_peak_errs = eval_curves.get("eval_peak_errs", [])
        eval_dl_dc_mae_bestfits = eval_curves.get("eval_dl_dc_mae_bestfits", [])
        eval_raw_ssdu_nmses = eval_curves.get("eval_raw_ssdu_nmses", [])
        avg_grasp_curve_mae = eval_curves.get("avg_grasp_curve_mae", float("nan"))
        avg_grasp_ttae_sec = eval_curves.get("avg_grasp_ttae_sec", float("nan"))
        avg_grasp_iauc10_err = eval_curves.get("avg_grasp_iauc10_err", float("nan"))
        avg_grasp_peak_err = eval_curves.get("avg_grasp_peak_err", float("nan"))
        avg_grasp_dc_mae_bestfit = eval_curves.get("avg_grasp_dc_mae_bestfit", float("nan"))
        avg_grasp_raw_ssdu_nmse = eval_curves.get("avg_grasp_raw_ssdu_nmse", float("nan"))
    else:
        train_mc_losses = []
        val_mc_losses = []
        train_ei_losses = []
        val_ei_losses = []
        train_adj_losses = []
        val_adj_losses = []
        weighted_train_mc_losses = []
        weighted_train_ei_losses = []
        weighted_train_adj_losses = []
        train_rebin_losses = []
        weighted_train_rebin_losses = []
        lr_history = []
        lr_epochs = []
        ei_weight_history = []
        ei_weight_epochs = []
        ei_gradnorm_ratio_history = []
        ei_gradnorm_ratio_epochs = []
        rebin_weight_history = []
        rebin_weight_epochs = []
        ei_gradnorm_ratio_ema = None
        ei_gradnorm_samples = 0
        ei_gradnorm_locked = False
        ei_target_weight_base = float(target_w_ei)
        ei_target_weight_effective = float(target_w_ei)
        eval_ssims = []
        eval_lpipses = []
        eval_psnrs = []
        eval_mses = []
        eval_dc_mses = []
        eval_dc_maes = []
        eval_raw_dc_mses = []
        eval_raw_dc_maes = []
        eval_curve_corrs = []
        eval_temporal_epochs = []
        eval_curve_maes = []
        eval_ttae_secs = []
        eval_iauc10_errs = []
        eval_peak_errs = []
        eval_dl_dc_mae_bestfits = []
        eval_raw_ssdu_nmses = []
        avg_grasp_curve_mae = float("nan")
        avg_grasp_ttae_sec = float("nan")
        avg_grasp_iauc10_err = float("nan")
        avg_grasp_peak_err = float("nan")
        avg_grasp_dc_mae_bestfit = float("nan")
        avg_grasp_raw_ssdu_nmse = float("nan")


    eval_spf_curves = {}
    if resume_from_checkpoint:
        eval_spf_curves = eval_curves.get("eval_spf_curves", {})
    for spf in low_spf_eval_targets:
        if spf not in eval_spf_curves:
            eval_spf_curves[spf] = dict(
                epochs=[],
                eval_ssims=[],
                eval_psnrs=[],
                eval_mses=[],
                eval_lpipses=[],
                eval_raw_dc_maes=[],
                eval_curve_corrs=[],
            )

    best_checkpoint_path = os.path.join(output_dir, f'{exp_name}_best_model.pth')
    best_psnr = -np.inf
    best_epoch = None
    if resume_from_checkpoint:
        psnr_array = np.array(eval_psnrs, dtype=float)
        if psnr_array.size and np.isfinite(psnr_array).any():
            best_idx = int(np.nanargmax(psnr_array))
            best_psnr = float(psnr_array[best_idx])
            if eval_temporal_epochs and len(eval_temporal_epochs) == len(eval_psnrs):
                best_epoch = eval_temporal_epochs[best_idx]
            else:
                best_epoch = best_idx * eval_frequency
        if global_rank == 0 or not config['training']['multigpu']:
            if best_epoch is not None:
                print(f"[Checkpoint] Loaded best PSNR {best_psnr:.4f} from epoch {best_epoch}")

    def _build_checkpoint_curves():
        train_curves = dict(
            train_mc_losses=train_mc_losses,
            train_ei_losses=train_ei_losses,
            train_adj_losses=train_adj_losses,
            weighted_train_mc_losses=weighted_train_mc_losses,
            weighted_train_ei_losses=weighted_train_ei_losses,
            weighted_train_adj_losses=weighted_train_adj_losses,
            train_rebin_losses=train_rebin_losses,
            weighted_train_rebin_losses=weighted_train_rebin_losses,
            lr_history=lr_history,
            lr_epochs=lr_epochs,
            ei_weight_history=ei_weight_history,
            ei_weight_epochs=ei_weight_epochs,
            ei_gradnorm_ratio_history=ei_gradnorm_ratio_history,
            ei_gradnorm_ratio_epochs=ei_gradnorm_ratio_epochs,
            ei_gradnorm_ratio_ema=ei_gradnorm_ratio_ema,
            ei_gradnorm_samples=ei_gradnorm_samples,
            ei_gradnorm_locked=ei_gradnorm_locked,
            ei_target_weight_base=ei_target_weight_base,
            ei_target_weight_effective=ei_target_weight_effective,
            rebin_weight_history=rebin_weight_history,
            rebin_weight_epochs=rebin_weight_epochs,
        )
        val_curves = dict(
            val_mc_losses=val_mc_losses,
            val_ei_losses=val_ei_losses,
            val_adj_losses=val_adj_losses,
        )
        eval_curves = dict(
            eval_ssims=eval_ssims,
            eval_psnrs=eval_psnrs,
            eval_mses=eval_mses,
            eval_lpipses=eval_lpipses,
            eval_dc_mses=eval_dc_mses,
            eval_dc_maes=eval_dc_maes,
            eval_raw_dc_mses=eval_raw_dc_mses,
            eval_raw_dc_maes=eval_raw_dc_maes,
            eval_curve_corrs=eval_curve_corrs,
            eval_temporal_epochs=eval_temporal_epochs,
            eval_curve_maes=eval_curve_maes,
            eval_ttae_secs=eval_ttae_secs,
            eval_iauc10_errs=eval_iauc10_errs,
            eval_peak_errs=eval_peak_errs,
            eval_dl_dc_mae_bestfits=eval_dl_dc_mae_bestfits,
            eval_raw_ssdu_nmses=eval_raw_ssdu_nmses,
            avg_grasp_curve_mae=avg_grasp_curve_mae,
            avg_grasp_ttae_sec=avg_grasp_ttae_sec,
            avg_grasp_iauc10_err=avg_grasp_iauc10_err,
            avg_grasp_peak_err=avg_grasp_peak_err,
            avg_grasp_dc_mae_bestfit=avg_grasp_dc_mae_bestfit,
            avg_grasp_raw_ssdu_nmse=avg_grasp_raw_ssdu_nmse,
            eval_spf_curves=eval_spf_curves,
            best_psnr=best_psnr,
            best_epoch=best_epoch,
        )
        return train_curves, val_curves, eval_curves

    def _grad_l2_norm(grad_tensors):
        total_sq = None
        for grad in grad_tensors:
            if grad is None:
                continue
            grad_sq = (grad.detach().float() ** 2).sum()
            total_sq = grad_sq if total_sq is None else (total_sq + grad_sq)
        if total_sq is None:
            return None
        return torch.sqrt(total_sq)

    loss_transition_checkpoints = {}

    def _register_transition_checkpoint(epoch_idx: int, tag: str):
        epoch_i = int(epoch_idx)
        if epoch_i <= 0:
            return
        if start_epoch > epoch_i:
            return
        loss_transition_checkpoints.setdefault(epoch_i, [])
        if tag not in loss_transition_checkpoints[epoch_i]:
            loss_transition_checkpoints[epoch_i].append(tag)

    if checkpoint_before_loss_transitions:
        if use_ei_loss and warmup > 0:
            _register_transition_checkpoint(warmup, "pre_ei_loss")
        if use_rebin_loss and rebin_warmup > 0:
            _register_transition_checkpoint(rebin_warmup, "pre_rebin_loss")

        if global_rank == 0 or not config['training']['multigpu']:
            if loss_transition_checkpoints:
                transitions = ", ".join(
                    [
                        f"epoch {ep}: {', '.join(tags)}"
                        for ep, tags in sorted(loss_transition_checkpoints.items())
                    ]
                )
                print(f"[Checkpoint] Transition checkpoints enabled: {transitions}")
            else:
                print("[Checkpoint] Transition checkpoints enabled, but no future transition epochs found.")
    else:
        if global_rank == 0 or not config['training']['multigpu']:
            print("[Checkpoint] Transition checkpoints disabled by config.")


    grasp_ssims = []
    grasp_psnrs = []
    grasp_mses = []
    grasp_lpipses = []
    grasp_dc_mses = []
    grasp_dc_maes = []
    grasp_dc_mae_bestfits = []
    grasp_curve_corrs = []
    raw_grasp_dc_mses = []
    raw_grasp_dc_maes = []
    grasp_curve_maes = []
    grasp_ttae_secs = []
    grasp_iauc10_errs = []
    grasp_peak_errs = []

    # Defaults so checkpointing works even if Step-0 validation is skipped
    if not resume_from_checkpoint:
        avg_grasp_ssim = 0.0
        avg_grasp_psnr = 0.0
        avg_grasp_mse = 0.0
        avg_grasp_lpips = 0.0
        avg_grasp_dc_mse = 0.0
        avg_grasp_dc_mae = 0.0
        avg_grasp_curve_corr = 0.0
        avg_grasp_raw_dc_mae = 0.0
        avg_grasp_raw_dc_mse = 0.0
        avg_grasp_curve_mae = float("nan")
        avg_grasp_ttae_sec = float("nan")
        avg_grasp_iauc10_err = float("nan")
        avg_grasp_peak_err = float("nan")
        avg_grasp_dc_mae_bestfit = float("nan")
        avg_grasp_raw_ssdu_nmse = float("nan")

    def _mean_or_nan(values):
        return float(np.mean(values)) if values else float("nan")

    def _std_or_zero(values):
        return float(np.std(values)) if values else 0.0

    def _raw_eval_available(raw_csmaps) -> bool:
        if raw_csmaps is None:
            return False
        try:
            if hasattr(raw_csmaps, "numel") and raw_csmaps.numel() == 0:
                return False
            finite = torch.isfinite(raw_csmaps)
            return bool(finite.any().item())
        except Exception:
            try:
                return bool(np.isfinite(raw_csmaps).any())
            except Exception:
                return True

    def _grasp_baseline_ready():
        required = [
            avg_grasp_ssim,
            avg_grasp_psnr,
            avg_grasp_mse,
            avg_grasp_lpips,
            avg_grasp_dc_mse,
            avg_grasp_dc_mae,
            avg_grasp_curve_mae,
            avg_grasp_ttae_sec,
            avg_grasp_iauc10_err,
            avg_grasp_peak_err,
        ]
        if (not skip_raw_eval_if_invalid_slice) or raw_grasp_dc_maes or raw_grasp_dc_mses:
            required.extend([avg_grasp_raw_dc_mae, avg_grasp_raw_dc_mse])
        return all(val is not None and np.isfinite(val) for val in required)

    lambda_Ls = []
    lambda_Ss = []
    lambda_spatial_Ls = []
    lambda_spatial_Ss = []
    gammas = []
    lambda_steps = []

    iteration_count = 0

    # Step 0: Evaluate the untrained model
    step0_do_val = config.get("debugging", {}).get("calc_step_0_val", True)
    # Optional debug knobs to cap expensive evaluation loops.
    max_step0_train_batches = config.get("debugging", {}).get("max_step0_train_batches", None)
    max_step0_val_batches = config.get("debugging", {}).get("max_step0_val_batches", None)
    max_val_batches = config.get("debugging", {}).get("max_val_batches", None)
    detect_anomaly = bool(config.get("debugging", {}).get("detect_anomaly", False))
    if detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
    if (not resume_from_checkpoint) and config['debugging']['calc_step_0'] == True:
        model.eval()
        initial_train_mc_loss = 0.0
        initial_val_mc_loss = 0.0
        initial_train_ei_loss = 0.0
        initial_val_ei_loss = 0.0
        initial_train_adj_loss = 0.0
        initial_val_adj_loss = 0.0
        initial_eval_ssims = []
        initial_eval_psnrs = []
        initial_eval_mses = []
        initial_eval_lpipses = []
        initial_eval_dc_mses = []
        initial_eval_dc_maes = []
        initial_eval_curve_corrs = []
        initial_eval_raw_dc_mses = []
        initial_eval_raw_dc_maes = []
        initial_eval_curve_maes = []
        initial_eval_ttae_secs = []
        initial_eval_iauc10_errs = []
        initial_eval_peak_errs = []
        initial_eval_dl_dc_mae_bestfits = []
        initial_eval_raw_ssdu_nmses = []


        with torch.no_grad():

            # Evaluate on training data
            step0_train_batches = 0
            for measured_kspace, csmap, N_samples, N_spokes, N_time in tqdm(train_loader, desc="Step 0 Training Evaluation"):

                # prepare inputs
                measured_kspace = to_torch_complex(measured_kspace).squeeze()
                measured_kspace = rearrange(measured_kspace, 't co sp sam -> co (sp sam) t')
                n_spokes_i = _as_int_scalar(N_spokes)
                n_time_i = _as_int_scalar(N_time)

                # prep physics operators
                ktraj, dcomp, nufft_ob, adjnufft_ob = get_nufft(
                    N_samples, n_spokes_i, n_time_i
                )

                
                if n_time_i > Ng:
                    max_idx = n_time_i - Ng
                    random_index = sample_start_time_index(max_idx, use_edge_time_index_sampling)

                    measured_kspace = measured_kspace[..., random_index:random_index + Ng]

                    ktraj_chunk = ktraj[..., random_index:random_index + Ng]
                    dcomp_chunk = dcomp[..., random_index:random_index + Ng]


                    physics = MCNUFFT(nufft_ob, adjnufft_ob, ktraj_chunk, dcomp_chunk)

                    start_timepoint_index = torch.tensor([random_index], dtype=torch.float, device=device)

                else:
                    physics = MCNUFFT(nufft_ob, adjnufft_ob, ktraj, dcomp)

                    start_timepoint_index = torch.tensor([0], dtype=torch.float, device=device)


                measured_kspace = measured_kspace.to(device, non_blocking=True)
                csmap = csmap.to(device, non_blocking=True).to(measured_kspace.dtype)

                acceleration = torch.tensor([N_full / int(n_spokes_i)], dtype=torch.float, device=device)

                if config['model']['encode_acceleration']:
                    acceleration_encoding = acceleration
                else: 
                    acceleration_encoding = None

                if config['model']['encode_time_index'] == False:
                    start_timepoint_index = None

                with amp_autocast():
                    x_recon, adj_loss, lambda_L, lambda_S, lambda_spatial_L, lambda_spatial_S, gamma, lambda_step = model(
                        measured_kspace, physics, csmap, acceleration_encoding, start_timepoint_index, epoch="train0", norm=config['model']['norm']
                    )

                    # calculate losses
                    if use_adj_loss:
                        initial_train_adj_loss += adj_loss.item()

                    mc_loss = mc_loss_fn(measured_kspace, x_recon, physics, csmap)
                    initial_train_mc_loss += mc_loss.item()

                    if use_ei_loss:
                        ei_loss, t_img = ei_loss_fn(
                            x_recon, physics, model_for_ei, csmap, acceleration_encoding, start_timepoint_index
                        )

                        initial_train_ei_loss += ei_loss.item()

                step0_train_batches += 1
                if max_step0_train_batches is not None and step0_train_batches >= int(max_step0_train_batches):
                    break
                    
            # record losses
            denom = step0_train_batches if step0_train_batches > 0 else len(train_loader)
            step0_train_mc_loss = initial_train_mc_loss / denom
            train_mc_losses.append(step0_train_mc_loss)

            step0_train_ei_loss = initial_train_ei_loss / denom
            train_ei_losses.append(step0_train_ei_loss)

            if use_adj_loss:
                step0_train_adj_loss = initial_train_adj_loss / denom
            else:
                step0_train_adj_loss = 0.0
            train_adj_losses.append(step0_train_adj_loss)


            if global_rank == 0 or not config['training']['multigpu']:
                writer.add_scalar('Loss/Train_MC', step0_train_mc_loss, 0)
                writer.add_scalar('Loss/Train_EI', step0_train_ei_loss, 0)
                writer.add_scalar('Loss/Train_Weighted_EI', 0, 0)
                writer.add_scalar('Loss/Train_Adj', step0_train_adj_loss, 0)


            lambda_Ls.append(lambda_L.item())
            lambda_Ss.append(lambda_S.item())
            lambda_spatial_Ls.append(lambda_spatial_L.item())
            lambda_spatial_Ss.append(lambda_spatial_S.item())
            gammas.append(gamma.item())
            lambda_steps.append(lambda_step.item())


            # Evaluate on validation data
            if step0_do_val:
                step0_val_batches = 0
                step0_val_infer_times = []
                step0_use_sliding_window = _use_eval_sliding_window(N_time_eval)
                for dro_kspace, csmap, ground_truth, dro_grasp_img, mask, grasp_path, raw_kspace, raw_grasp_img, raw_csmaps in tqdm(val_dro_loader, desc="Step 0 Validation Evaluation"):
    
                    csmap = csmap.squeeze(0).to(device)   # Remove batch dim
                    ground_truth = ground_truth.to(device) # Shape: (1, 2, T, H, W)
    
                    dro_kspace = dro_kspace.squeeze(0).to(device) # Remove batch dim
                    dro_grasp_img = dro_grasp_img.to(device) # Shape: (1, 2, H, T, W)
    
                    raw_eval_available = _raw_eval_available(raw_csmaps)
                    if raw_eval_available:
                        raw_kspace = raw_kspace.squeeze(0).to(device) # Remove batch dim
                        raw_grasp_img = raw_grasp_img.to(device) # Shape: (1, 2, H, T, W)
                        raw_csmaps = raw_csmaps.squeeze(0).to(device)   # Remove batch dim
                    else:
                        raw_kspace = None
                        raw_grasp_img = None
                        raw_csmaps = None
    
    
                    N_spokes = eval_ktraj.shape[1] / config['data']['samples']
                    acceleration = torch.tensor([N_full / int(N_spokes)], dtype=torch.float, device=device)
    
                    if config['model']['encode_acceleration']:
                        acceleration_encoding = acceleration
                    else: 
                        acceleration_encoding = None
    
                    if config['model']['encode_time_index'] == False:
                        start_timepoint_index = None
                    else:
                        start_timepoint_index = torch.tensor([0], dtype=torch.float, device=device)
    
                    # inference + losses
                    with amp_autocast():
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        infer_start = time.perf_counter()
                        if step0_use_sliding_window:
                            x_recon, adj_loss = sliding_window_inference(H, W, N_time_eval, eval_ktraj, eval_dcomp, eval_nufft_ob, eval_adjnufft_ob, eval_chunk_size, eval_chunk_overlap, dro_kspace, csmap, acceleration_encoding, start_timepoint_index, model, epoch="val0", device=device, norm=config["model"]["norm"], collect_adj_loss=use_adj_loss)  
                        else:
                            x_recon, adj_loss, *_ = model(
                            dro_kspace.to(device), eval_physics, csmap, acceleration_encoding, start_timepoint_index, epoch="val0", norm=config['model']['norm']
                            )
                            adj_loss = adj_loss.item()
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        step0_val_infer_times.append(float(time.perf_counter() - infer_start))
                        raw_x_recon = None
                        if raw_eval_available:
                            if step0_use_sliding_window:
                                raw_x_recon, _ = sliding_window_inference(
                                    H, W, N_time_eval,
                                    eval_ktraj, eval_dcomp, eval_nufft_ob, eval_adjnufft_ob,
                                    eval_chunk_size, eval_chunk_overlap,
                                    raw_kspace, raw_csmaps,
                                    acceleration_encoding, start_timepoint_index,
                                    model, epoch="val0", device=device,
                                    norm=config["model"]["norm"], collect_adj_loss=use_adj_loss
                                )
                            else:
                                raw_x_recon, *_ = model(
                                    raw_kspace.to(device), eval_physics, raw_csmaps,
                                    acceleration_encoding, start_timepoint_index,
                                    epoch="val0", norm=config['model']['norm']
                                )
    
                        # fix orientation of raw k-space recon
                        # raw_x_recon = torch.rot90(raw_x_recon, k=2, dims=[-3,-2])
    
                        # compute losses
                        if use_adj_loss:
                            initial_val_adj_loss += adj_loss
                        
                        mc_loss = mc_loss_fn(dro_kspace.to(device), x_recon, eval_physics, csmap)
                        initial_val_mc_loss += mc_loss.item()
    
                        if use_ei_loss:
                            if deterministic_val_ei:
                                val_seed = deterministic_val_ei_seed + int(step0_val_batches)
                                val_ei_ctx = _temporary_rng(val_seed)
                            else:
                                val_ei_ctx = nullcontext()
                            with val_ei_ctx:
                                ei_loss, t_img = ei_loss_fn(
                                    x_recon, eval_physics, model_for_ei, csmap, acceleration_encoding, start_timepoint_index
                                )

                            initial_val_ei_loss += ei_loss.item()
    
    
                    # calculate grasp metrics
                    if global_rank == 0 or not config['training']['multigpu']:
                        (
                            ssim_grasp,
                            psnr_grasp,
                            mse_grasp,
                            lpips_grasp,
                            dc_mse_grasp,
                            dc_mae_grasp,
                            grasp_aux,
                        ) = eval_grasp(
                            dro_kspace,
                            csmap,
                            ground_truth,
                            dro_grasp_img,
                            eval_physics,
                            device,
                            eval_dir,
                            rescale=config['evaluation']['rescale'],
                            dro_eval=True,
                            return_aux=True,
                        )
                        grasp_ssims.append(ssim_grasp)
                        grasp_psnrs.append(psnr_grasp)
                        grasp_mses.append(mse_grasp)
                        grasp_lpipses.append(lpips_grasp)
                        grasp_dc_mses.append(dc_mse_grasp)
                        grasp_dc_maes.append(dc_mae_grasp)
                        grasp_dc_mae_bestfit = None if grasp_aux is None else grasp_aux.get("grasp_dc_mae_bestfit")
                        if grasp_dc_mae_bestfit is not None and np.isfinite(grasp_dc_mae_bestfit):
                            grasp_dc_mae_bestfits.append(float(grasp_dc_mae_bestfit))
    
                        ssim, psnr, mse, lpips, dc_mse, dc_mae, recon_corr, grasp_corr, temporal_metrics = eval_sample(
                            dro_kspace,
                            csmap,
                            ground_truth,
                            x_recon,
                            eval_physics,
                            mask,
                            dro_grasp_img,
                            acceleration,
                            int(N_spokes),
                            eval_dir,
                            label='val0',
                            device=device,
                            cluster=cluster,
                            dro_eval=True,
                            grasp_path=grasp_path,
                            rescale=config['evaluation']['rescale'],
                            plot_arrival=True,
                            arrival_k=config['model']['losses']['ei_loss'].get("arrival_shift_baseline_k", 2.0),
                            arrival_percentile=config['model']['losses']['ei_loss'].get("arrival_shift_percentile", 0.95),
                            arrival_baseline_k=config['model']['losses']['ei_loss'].get("arrival_shift_baseline_k", 2.0),
                            arrival_method=config['model']['losses']['ei_loss'].get("arrival_method", "threshold"),
                            arrival_fraction=config['model']['losses']['ei_loss'].get("arrival_fraction", 0.1),
                            arrival_pre_contrast_baseline=config['model']['losses']['ei_loss'].get("pre_contrast_baseline", "n_frames"),
                            arrival_baseline_seconds=config['model']['losses']['ei_loss'].get("baseline_seconds", 20),
                            arrival_total_seconds=config['model']['losses']['ei_loss'].get("total_seconds", 150.0),
                        )
                        initial_eval_ssims.append(ssim)
                        initial_eval_psnrs.append(psnr)
                        initial_eval_mses.append(mse)
                        initial_eval_lpipses.append(lpips)
                        initial_eval_dc_mses.append(dc_mse)
                        initial_eval_dc_maes.append(dc_mae)
    
                        if recon_corr is not None:
                            initial_eval_curve_corrs.append(recon_corr)
                            grasp_curve_corrs.append(grasp_corr)
                        if temporal_metrics:
                            curve_mae = temporal_metrics.get("dl_all_curve_mae")
                            ttae_sec = temporal_metrics.get("dl_all_ttae_sec")
                            iauc10_err = temporal_metrics.get("dl_all_iauc10_err")
                            peak_err = temporal_metrics.get("dl_all_peak_err")
                            grasp_curve_mae = temporal_metrics.get("grasp_all_curve_mae")
                            grasp_ttae_sec = temporal_metrics.get("grasp_all_ttae_sec")
                            grasp_iauc10_err = temporal_metrics.get("grasp_all_iauc10_err")
                            grasp_peak_err = temporal_metrics.get("grasp_all_peak_err")
                            dl_dc_mae_bestfit = temporal_metrics.get("dl_dc_mae_bestfit")
                            if curve_mae is not None and np.isfinite(curve_mae):
                                initial_eval_curve_maes.append(curve_mae)
                            if ttae_sec is not None and np.isfinite(ttae_sec):
                                initial_eval_ttae_secs.append(ttae_sec)
                            if iauc10_err is not None and np.isfinite(iauc10_err):
                                initial_eval_iauc10_errs.append(iauc10_err)
                            if peak_err is not None and np.isfinite(peak_err):
                                initial_eval_peak_errs.append(peak_err)
                            if grasp_curve_mae is not None and np.isfinite(grasp_curve_mae):
                                grasp_curve_maes.append(float(grasp_curve_mae))
                            if grasp_ttae_sec is not None and np.isfinite(grasp_ttae_sec):
                                grasp_ttae_secs.append(float(grasp_ttae_sec))
                            if grasp_iauc10_err is not None and np.isfinite(grasp_iauc10_err):
                                grasp_iauc10_errs.append(float(grasp_iauc10_err))
                            if grasp_peak_err is not None and np.isfinite(grasp_peak_err):
                                grasp_peak_errs.append(float(grasp_peak_err))
                            if dl_dc_mae_bestfit is not None and np.isfinite(dl_dc_mae_bestfit):
                                initial_eval_dl_dc_mae_bestfits.append(float(dl_dc_mae_bestfit))
    
                        # raw k-space eval
                        if raw_eval_available:
                            print("performing non-DRO eval...")
                            dc_mse_raw_grasp, dc_mae_raw_grasp, _ = eval_grasp(
                                raw_kspace, raw_csmaps, ground_truth, raw_grasp_img,
                                eval_physics, device, eval_dir,
                                rescale=config['evaluation']['rescale'], dro_eval=False
                            )
                            dc_mse_raw, dc_mae_raw, _, _ = eval_sample(
                                raw_kspace,
                                raw_csmaps,
                                ground_truth,
                                raw_x_recon,
                                eval_physics,
                                mask,
                                raw_grasp_img,
                                acceleration,
                                int(N_spokes),
                                eval_dir,
                                label='val0',
                                device=device,
                                cluster=cluster,
                                dro_eval=False,
                                grasp_path=grasp_path,
                                raw_slice_idx=raw_grasp_slice_idx,
                                rescale=config['evaluation']['rescale'],
                                plot_arrival=True,
                                arrival_k=config['model']['losses']['ei_loss'].get("arrival_shift_baseline_k", 2.0),
                                arrival_percentile=config['model']['losses']['ei_loss'].get("arrival_shift_percentile", 0.95),
                                arrival_baseline_k=config['model']['losses']['ei_loss'].get("arrival_shift_baseline_k", 2.0),
                                arrival_method=config['model']['losses']['ei_loss'].get("arrival_method", "threshold"),
                                arrival_fraction=config['model']['losses']['ei_loss'].get("arrival_fraction", 0.1),
                                arrival_pre_contrast_baseline=config['model']['losses']['ei_loss'].get("pre_contrast_baseline", "n_frames"),
                                arrival_baseline_seconds=config['model']['losses']['ei_loss'].get("baseline_seconds", 20),
                                arrival_total_seconds=config['model']['losses']['ei_loss'].get("total_seconds", 150.0),
                            )

                            raw_grasp_dc_mses.append(dc_mse_raw_grasp)
                            raw_grasp_dc_maes.append(dc_mae_raw_grasp)
                            initial_eval_raw_dc_mses.append(dc_mse_raw)
                            initial_eval_raw_dc_maes.append(dc_mae_raw)
                            if compute_ssdu_eval:
                                ssdu_chunk_size = eval_chunk_size if step0_use_sliding_window else None
                                ssdu_result = compute_ssdu_kspace_nmse(
                                    model,
                                    raw_kspace,
                                    raw_csmaps,
                                    eval_ktraj,
                                    eval_dcomp,
                                    eval_nufft_ob,
                                    eval_adjnufft_ob,
                                    spokes_per_frame=int(N_spokes),
                                    K_folds=ssdu_k_folds,
                                    baseline_weighting=ssdu_weighting,
                                    device=device,
                                    acceleration_encoding=acceleration_encoding,
                                    start_timepoint_index=start_timepoint_index,
                                    norm=config["model"]["norm"],
                                    epoch="val0",
                                    chunk_size=ssdu_chunk_size,
                                    chunk_overlap=eval_chunk_overlap,
                                )
                                raw_ssdu_nmse = ssdu_result.get("ssdu_nmse_mean")
                                if raw_ssdu_nmse is not None and np.isfinite(raw_ssdu_nmse):
                                    initial_eval_raw_ssdu_nmses.append(float(raw_ssdu_nmse))

                    step0_val_batches += 1
                    if max_step0_val_batches is not None and step0_val_batches >= int(max_step0_val_batches):
                        break
    
    
            if step0_do_val:
                denom = step0_val_batches if step0_val_batches > 0 else len(val_dro_loader)
                step0_val_mc_loss = initial_val_mc_loss / denom
                val_mc_losses.append(step0_val_mc_loss)

                step0_val_ei_loss = initial_val_ei_loss / denom
                val_ei_losses.append(step0_val_ei_loss)

                if use_adj_loss:
                    step0_val_adj_loss = initial_val_adj_loss / denom
                else:
                    step0_val_adj_loss = 0.0
                val_adj_losses.append(step0_val_adj_loss)


                if global_rank == 0 or not config['training']['multigpu']:
                    writer.add_scalar('Loss/Val_MC', step0_val_mc_loss, 0)
                    writer.add_scalar('Loss/Val_EI', step0_val_ei_loss, 0)
                    writer.add_scalar('Loss/Val_Adj', step0_val_adj_loss, 0)


                # Calculate and store average validation evaluation metrics
                initial_eval_ssim = np.mean(initial_eval_ssims)
                initial_eval_psnr = np.mean(initial_eval_psnrs)
                initial_eval_mse = np.mean(initial_eval_mses)
                initial_eval_lpips = np.mean(initial_eval_lpipses)
                initial_eval_dc_mse = np.mean(initial_eval_dc_mses)
                initial_eval_dc_mae = np.mean(initial_eval_dc_maes)
                initial_eval_curve_corr = np.mean(initial_eval_curve_corrs)
                initial_eval_raw_dc_mse = np.mean(initial_eval_raw_dc_mses)
                initial_eval_raw_dc_mae = np.mean(initial_eval_raw_dc_maes)
                initial_eval_curve_mae = np.mean(initial_eval_curve_maes) if initial_eval_curve_maes else np.nan
                initial_eval_ttae_sec = np.mean(initial_eval_ttae_secs) if initial_eval_ttae_secs else np.nan
                initial_eval_iauc10_err = np.mean(initial_eval_iauc10_errs) if initial_eval_iauc10_errs else np.nan
                initial_eval_peak_err = np.mean(initial_eval_peak_errs) if initial_eval_peak_errs else np.nan
                initial_eval_dl_dc_mae_bestfit = (
                    np.mean(initial_eval_dl_dc_mae_bestfits) if initial_eval_dl_dc_mae_bestfits else np.nan
                )
                initial_eval_raw_ssdu_nmse = (
                    np.mean(initial_eval_raw_ssdu_nmses) if initial_eval_raw_ssdu_nmses else np.nan
                )

                eval_ssims.append(initial_eval_ssim)
                eval_psnrs.append(initial_eval_psnr)
                eval_mses.append(initial_eval_mse)
                eval_lpipses.append(initial_eval_lpips)
                eval_dc_mses.append(initial_eval_dc_mse) 
                eval_dc_maes.append(initial_eval_dc_mae) 
                eval_raw_dc_mses.append(initial_eval_raw_dc_mse) 
                eval_raw_dc_maes.append(initial_eval_raw_dc_mae) 
                eval_curve_corrs.append(initial_eval_curve_corr)
                eval_temporal_epochs.append(0)
                eval_curve_maes.append(initial_eval_curve_mae)
                eval_ttae_secs.append(initial_eval_ttae_sec)
                eval_iauc10_errs.append(initial_eval_iauc10_err)
                eval_peak_errs.append(initial_eval_peak_err)
                eval_dl_dc_mae_bestfits.append(initial_eval_dl_dc_mae_bestfit)
                eval_raw_ssdu_nmses.append(initial_eval_raw_ssdu_nmse)

                spf_key = int(N_spokes_eval)
                if spf_key in eval_spf_curves:
                    eval_spf_curves[spf_key]["epochs"].append(0)
                    eval_spf_curves[spf_key]["eval_ssims"].append(initial_eval_ssim)
                    eval_spf_curves[spf_key]["eval_psnrs"].append(initial_eval_psnr)
                    eval_spf_curves[spf_key]["eval_mses"].append(initial_eval_mse)
                    eval_spf_curves[spf_key]["eval_lpipses"].append(initial_eval_lpips)
                    eval_spf_curves[spf_key]["eval_raw_dc_maes"].append(initial_eval_raw_dc_mae)
                    eval_spf_curves[spf_key]["eval_curve_corrs"].append(initial_eval_curve_corr)

                if global_rank == 0 or not config['training']['multigpu']:
                    writer.add_scalar('Metric/SSIM', initial_eval_ssim, 0)
                    writer.add_scalar('Metric/PSNR', initial_eval_psnr, 0)
                    writer.add_scalar('Metric/MSE', initial_eval_mse, 0)
                    writer.add_scalar('Metric/LPIPS', initial_eval_lpips, 0)
                    writer.add_scalar('Metric/DC_MSE', initial_eval_dc_mse, 0)
                    writer.add_scalar('Metric/DC_MAE', initial_eval_dc_mae, 0)
                    writer.add_scalar('Metric/RAW_DC_MSE', initial_eval_raw_dc_mse, 0)
                    writer.add_scalar('Metric/RAW_DC_MAE', initial_eval_raw_dc_mae, 0)
                    writer.add_scalar('Metric/EC_Corr', initial_eval_curve_corr, 0)
                    if np.isfinite(initial_eval_curve_mae):
                        writer.add_scalar('Metric/Temporal_Curve_MAE', initial_eval_curve_mae, 0)
                    if np.isfinite(initial_eval_ttae_sec):
                        writer.add_scalar('Metric/Temporal_TTAE_sec', initial_eval_ttae_sec, 0)
                    if np.isfinite(initial_eval_iauc10_err):
                        writer.add_scalar('Metric/Temporal_IAUC10_err', initial_eval_iauc10_err, 0)
                    if np.isfinite(initial_eval_peak_err):
                        writer.add_scalar('Metric/Temporal_Peak_err', initial_eval_peak_err, 0)
                    if np.isfinite(initial_eval_dl_dc_mae_bestfit):
                        writer.add_scalar('Metric/DL_DC_MAE_BESTFIT', initial_eval_dl_dc_mae_bestfit, 0)
                    if np.isfinite(initial_eval_raw_ssdu_nmse):
                        writer.add_scalar('Metric/RAW_SSDU_NMSE', initial_eval_raw_ssdu_nmse, 0)

        print(f"Step 0 Train Losses: MC: {step0_train_mc_loss}, EI: {step0_train_ei_loss}, Adj: {step0_train_adj_loss}")
        if step0_do_val:
            print(f"Step 0 Val Losses: MC: {step0_val_mc_loss}, EI: {step0_val_ei_loss}, Adj: {step0_val_adj_loss}")
            if (global_rank == 0 or not config['training']['multigpu']) and step0_val_infer_times:
                step0_mean_infer = float(np.mean(step0_val_infer_times))
                step0_std_infer = float(np.std(step0_val_infer_times, ddof=1)) if len(step0_val_infer_times) > 1 else 0.0
                infer_mode = "sliding-window" if step0_use_sliding_window else "direct-full"
                print(
                    f"[Eval] Step 0: inference time/sample ({infer_mode}) = "
                    f"{step0_mean_infer:.3f}s ± {step0_std_infer:.3f}s (n={len(step0_val_infer_times)})"
                )
                writer.add_scalar('Timing/Val_Inference_Seconds_Mean', step0_mean_infer, 0)
                writer.add_scalar('Timing/Val_Inference_Seconds_Std', step0_std_infer, 0)
        else:
            print("Step 0 Val Losses: skipped (calc_step_0_val: false)")

        if step0_do_val:
            # calculate average GRASP metrics
            avg_grasp_ssim = np.mean(grasp_ssims)
            avg_grasp_psnr = np.mean(grasp_psnrs)
            avg_grasp_mse = np.mean(grasp_mses)
            avg_grasp_lpips = np.mean(grasp_lpipses)
            avg_grasp_dc_mse = np.mean(grasp_dc_mses)
            avg_grasp_dc_mae = np.mean(grasp_dc_maes)
            avg_grasp_curve_corr = np.mean(grasp_curve_corrs)
            avg_grasp_raw_dc_mae = np.mean(raw_grasp_dc_maes)
            avg_grasp_raw_dc_mse = np.mean(raw_grasp_dc_mses)
            avg_grasp_curve_mae = _mean_or_nan(grasp_curve_maes)
            avg_grasp_ttae_sec = _mean_or_nan(grasp_ttae_secs)
            avg_grasp_iauc10_err = _mean_or_nan(grasp_iauc10_errs)
            avg_grasp_peak_err = _mean_or_nan(grasp_peak_errs)
            avg_grasp_dc_mae_bestfit = (
                np.mean(grasp_dc_mae_bestfits) if grasp_dc_mae_bestfits else float("nan")
            )
            avg_grasp_raw_ssdu_nmse = float("nan")

            if global_rank == 0 or not config['training']['multigpu']:
                if np.isfinite(initial_eval_psnr) and initial_eval_psnr > best_psnr:
                    best_psnr = float(initial_eval_psnr)
                    best_epoch = 0
                    train_curves, val_curves, eval_curves = _build_checkpoint_curves()
                    save_checkpoint(
                        model,
                        optimizer,
                        1,
                        train_curves,
                        val_curves,
                        eval_curves,
                        ei_target_weight_effective,
                        step0_train_ei_loss,
                        step0_train_mc_loss,
                        avg_grasp_ssim,
                        avg_grasp_psnr,
                        avg_grasp_mse,
                        avg_grasp_lpips,
                        avg_grasp_dc_mse,
                        avg_grasp_dc_mae,
                        avg_grasp_curve_corr,
                        avg_grasp_raw_dc_mae,
                        avg_grasp_raw_dc_mse,
                        best_checkpoint_path,
                    )
                    print(
                        f"[Checkpoint] New best PSNR {best_psnr:.4f} at epoch 0. Saved to {best_checkpoint_path}"
                    )
                    if run_state is not None:
                        with state_lock:
                            run_state["best_checkpoint_epoch"] = int(best_epoch)
                            run_state["best_checkpoint_psnr"] = float(best_psnr)
                            _save_run_state(run_state_path, run_state)



    # Step-0 metrics are only populated when calc_step_0 is enabled (or restored from checkpoint).
    # Keep explicit defaults so checkpoint saving cannot fail when step-0 is skipped.
    if "step0_train_mc_loss" not in locals():
        step0_train_mc_loss = 0.0
    if "step0_train_ei_loss" not in locals():
        step0_train_ei_loss = 0.0
    if "step0_train_adj_loss" not in locals():
        step0_train_adj_loss = 0.0

    # Training Loop
    svd_fail_count = 0
    if (epochs + 1) == start_epoch:
        raise(ValueError("Full training epochs already complete."))

    else: 

        current_curriculum_phase_idx = -1

        for epoch in range(start_epoch, epochs + 1):
            model.train()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
            running_mc_loss = 0.0
            running_ei_loss = 0.0
            running_adj_loss = 0.0
            running_rebin_loss = 0.0
            epoch_eval_ssims = []
            epoch_eval_psnrs = []
            epoch_eval_mses = []
            epoch_eval_lpipses = []
            epoch_eval_dc_mses = []
            epoch_eval_dc_maes = []
            epoch_eval_curve_corrs = []
            epoch_eval_raw_dc_mses = []
            epoch_eval_raw_dc_maes = []
            epoch_eval_curve_maes = []
            epoch_eval_ttae_secs = []
            epoch_eval_iauc10_errs = []
            epoch_eval_peak_errs = []
            epoch_eval_dl_dc_mae_bestfits = []
            epoch_eval_raw_ssdu_nmses = []


            # LR schedule with optional cosine decay and warmup (set before first step of the epoch)
            total = epochs
            lr_sched_cfg = config.get("training", {}).get("lr_schedule", {})
            warm = int(lr_sched_cfg.get("warmup_epochs", 5))
            if warm < 0:
                warm = 0
            warmup_mode = str(lr_sched_cfg.get("warmup_mode", "linear")).lower()
            use_cosine_decay = bool(lr_sched_cfg.get("use_cosine_decay", True))
            lr_floor = lr_sched_cfg.get("min_lr_factor", 0.2)
            if warm > 0 and epoch <= warm:
                if warmup_mode == "cosine":
                    lr_scale = 0.5 * (1.0 - math.cos(math.pi * (epoch / warm)))
                else:
                    lr_scale = epoch / warm
            else:
                if use_cosine_decay:
                    p = (epoch - warm) / max(1, total - warm)
                    lr_scale = lr_floor + (1.0 - lr_floor) * 0.5 * (1 + math.cos(math.pi * p))
                else:
                    lr_scale = 1.0
            for pg in optimizer.param_groups:
                pg['lr'] = config["model"]["optimizer"]["lr"] * lr_scale

            current_lr = optimizer.param_groups[0]["lr"]
            lr_history.append(current_lr)
            lr_epochs.append(epoch)
            if global_rank == 0 or not config['training']['multigpu']:
                writer.add_scalar('LR', current_lr, epoch)

            train_loader_tqdm = tqdm(
                train_loader, desc=f"Epoch {epoch}/{epochs}  Training", unit="batch"
            )

            if hasattr(train_dataset, 'resample_slices'):
                print(f"Epoch {epoch}: Resampling training slices...")
                train_dataset.resample_slices()

            if curriculum_enabled:
                for i, phase in enumerate(curriculum_phases):
                    if epoch >= phase['start_epoch']:
                        if i > current_curriculum_phase_idx: # Transition to a new phase
                            print(f"\n--- Entering Curriculum Phase: {phase['name']} at Epoch {epoch} ---")
                            # Update training spokes range
                            train_dataset.spokes_range = phase['train_spokes_range']
                            train_dataset.update_spokes_weights()

                            current_curriculum_phase_idx = i

                            if not fixed_eval_metrics:
                                # define eval physics model and dataset based on evaluation parameters for curriculum phase
                                N_spokes_eval = phase['eval_spokes_per_frame']
                                N_time_eval = phase['eval_num_frames']

                                # define physics object for evaluation
                                eval_ktraj, eval_dcomp, eval_nufft_ob, eval_adjnufft_ob = get_nufft(
                                    N_samples, N_spokes_eval, N_time_eval
                                )

                                eval_physics = MCNUFFT(eval_nufft_ob, eval_adjnufft_ob, eval_ktraj, eval_dcomp)

                                val_dro_dataset.spokes_per_frame = N_spokes_eval
                                val_dro_dataset.num_frames = N_time_eval
                                val_dro_dataset._update_sample_paths()

                                val_dro_loader = DataLoader(
                                    val_dro_dataset,
                                    batch_size=config["dataloader"]["batch_size"],
                                    shuffle=False,
                                    num_workers=config["dataloader"]["num_workers"],
                                    pin_memory=True,
                                    persistent_workers=bool(config["dataloader"]["num_workers"] > 0),
                                )
                                val_dro_eval_loader = _build_val_dro_eval_loader(val_dro_dataset)

            # if use_ei_loss:

                # # --- Check if it's the transition epoch ---
                # if epoch < warmup + 1:
                #     target_w_ei = 0.0
                # elif epoch == warmup + 1:
                #     # Get the last known MC loss value from the previous epoch
                #     mc_loss_at_transition = epoch_train_mc_loss
                    
                #     print(f"Transitioning at Epoch {epoch}. MC Loss: {mc_loss_at_transition:.4e}")
                    
                #     # Calculate the final target weight ONCE
                #     if step0_train_ei_loss > 0:
                #         target_w_ei = mc_loss_at_transition / step0_train_ei_loss
                #     else:
                #         target_w_ei = 0.0 # Prevent division by zero
                        
                #     print(f"Dynamically calculated target EI weight: {target_w_ei:.4f}")


            if config['training']['multigpu']:
                train_loader.sampler.set_epoch(epoch)

            if (
                use_ei_loss
                and ei_gradnorm_transition_enable
                and (not ei_gradnorm_locked)
                and epoch > ei_transition_end_epoch
            ):
                if ei_gradnorm_ratio_ema is not None:
                    scale = float(np.clip(
                        ei_gradnorm_ratio_ema,
                        ei_gradnorm_target_scale_min,
                        ei_gradnorm_target_scale_max,
                    ))
                    ei_target_weight_effective = float(ei_target_weight_base * scale)
                    if global_rank == 0 or not config['training']['multigpu']:
                        print(
                            "[EI] Locked post-transition EI target weight from GradNorm EMA: "
                            f"base={ei_target_weight_base:.6g}, ratio_ema={ei_gradnorm_ratio_ema:.6g}, "
                            f"scale={scale:.6g}, effective_target={ei_target_weight_effective:.6g}."
                        )
                else:
                    ei_target_weight_effective = float(ei_target_weight_base)
                    if global_rank == 0 or not config['training']['multigpu']:
                        print(
                            "[EI] GradNorm transition ended without valid samples; "
                            f"using base EI target={ei_target_weight_effective:.6g}."
                        )
                ei_gradnorm_locked = True

            # EI schedule is epoch-wise; compute once per epoch to avoid per-iteration overhead.
            ei_loss_weight = 0.0
            compute_ei_this_epoch = False
            if use_ei_loss:
                ei_loss_weight = get_cosine_ei_weight(
                    current_epoch=epoch,
                    warmup_epochs=warmup,
                    schedule_duration=duration,
                    target_weight=ei_target_weight_effective,
                )
                compute_ei_this_epoch = ei_loss_weight > 0.0
            ei_transition_active_epoch = bool(
                use_ei_loss
                and ei_gradnorm_transition_enable
                and (not ei_gradnorm_locked)
                and (ei_transition_start_epoch <= epoch <= ei_transition_end_epoch)
            )
            ei_weight_history.append(ei_loss_weight)
            ei_weight_epochs.append(epoch)
            if (global_rank == 0 or not config['training']['multigpu']) and use_ei_loss:
                writer.add_scalar("Loss/EI_Target_Weight_Effective", ei_target_weight_effective, epoch)
                if ei_gradnorm_ratio_ema is not None:
                    writer.add_scalar("Loss/EI_GradNorm_Ratio_EMA_Epoch", ei_gradnorm_ratio_ema, epoch)

            # Rebin schedule is epoch-wise; skip rebin compute whenever its weight is zero.
            rebin_loss_weight = 0.0
            compute_rebin_this_epoch = False
            if use_rebin_loss:
                scheduled_rebin_w = get_cosine_ei_weight(
                    current_epoch=epoch,
                    warmup_epochs=rebin_warmup,
                    schedule_duration=rebin_duration,
                    target_weight=rebin_target_w,
                )
                rebin_loss_weight = scheduled_rebin_w
                compute_rebin_this_epoch = rebin_loss_weight > 0.0
            rebin_weight_history.append(rebin_loss_weight)
            rebin_weight_epochs.append(epoch)

            # Only set when EI is computed; keep defined to avoid UnboundLocalError in plotting.
            t_img = None
            val_t_img = None

            sample_check_ids = [] if sample_check_enabled else None

            for batch_idx, batch in enumerate(train_loader_tqdm):  # measured_kspace shape: (B, C, I, S, T)
                if include_sample_indices:
                    sample_indices, measured_kspace, csmap, N_samples, N_spokes, N_time = batch
                    if sample_check_enabled and batch_idx < sample_check_batches:
                        if torch.is_tensor(sample_indices):
                            sample_indices = sample_indices.tolist()
                        for idx in sample_indices:
                            sample_check_ids.append(_format_sample_id(train_dataset, idx))
                else:
                    measured_kspace, csmap, N_samples, N_spokes, N_time = batch
                
                start = time.time()

                # prepare inputs
                measured_kspace = to_torch_complex(measured_kspace).squeeze()
                measured_kspace = rearrange(measured_kspace, 't co sp sam -> co (sp sam) t')
                n_samples_i = _as_int_scalar(N_samples)
                n_spokes_i = _as_int_scalar(N_spokes)
                n_time_i = _as_int_scalar(N_time)

                # prep physics operators
                ktraj, dcomp, nufft_ob, adjnufft_ob = get_nufft(
                    n_samples_i, n_spokes_i, n_time_i
                )

                if n_time_i > Ng:

                    max_idx = n_time_i - Ng
                    random_index = sample_start_time_index(max_idx, use_edge_time_index_sampling)

                    measured_kspace = measured_kspace[..., random_index:random_index + Ng]
                    ktraj_chunk = ktraj[..., random_index:random_index + Ng]
                    dcomp_chunk = dcomp[..., random_index:random_index + Ng]

                    physics = MCNUFFT(nufft_ob, adjnufft_ob, ktraj_chunk, dcomp_chunk)

                    start_timepoint_index = torch.tensor([random_index], dtype=torch.float, device=device)
                    

                else:
                    physics = MCNUFFT(nufft_ob, adjnufft_ob, ktraj, dcomp)

                    start_timepoint_index = torch.tensor([0], dtype=torch.float, device=device)


                iteration_count += 1
                optimizer.zero_grad(set_to_none=True)

                measured_kspace = measured_kspace.to(device, non_blocking=True)

                csmap = csmap.to(device, non_blocking=True).to(measured_kspace.dtype)

                # calculate acceleration factor
                acceleration = torch.tensor([N_full / int(n_spokes_i)], dtype=torch.float, device=device)

                if config['model']['encode_acceleration']:
                    acceleration_encoding = acceleration
                else: 
                    acceleration_encoding = None

                if config['model']['encode_time_index'] == False:
                    start_timepoint_index = None
                # else:
                #     start_timepoint_index = torch.tensor([0], dtype=torch.float, device=device)

                # print("Time encoding: ", start_timepoint_index.item())

                try:
                    with amp_autocast():
                        x_recon, adj_loss, lambda_L, lambda_S, lambda_spatial_L, lambda_spatial_S, gamma, lambda_step = model(
                            measured_kspace, physics, csmap, acceleration_encoding, start_timepoint_index, epoch=f"train{epoch}", norm=config['model']['norm']
                        )

                        # compute losses
                        if use_adj_loss:
                            running_adj_loss += adj_loss.item()

                        mc_loss = mc_loss_fn(measured_kspace, x_recon, physics, csmap)
                        running_mc_loss += mc_loss.item()

                        if use_rebin_loss and compute_rebin_this_epoch:
                            rebin_loss = rebin_loss_fn(
                                measured_kspace,
                                x_recon,
                                physics,
                                model,
                                csmap,
                                acceleration_encoding,
                                start_timepoint_index,
                                spokes_per_frame=n_spokes_i,
                                samples_per_spoke=n_samples_i,
                                norm=config['model']['norm'],
                                epoch=epoch,
                            )
                            running_rebin_loss += rebin_loss.item()
                        else:
                            rebin_loss = None

                        ei_loss = None
                        if use_ei_loss and compute_ei_this_epoch:
                            ei_loss, t_img = ei_loss_fn(
                                x_recon, physics, model_for_ei, csmap, acceleration_encoding, start_timepoint_index
                            )

                            running_ei_loss += ei_loss.item()
                            if (
                                ei_transition_active_epoch
                                and (iteration_count % ei_gradnorm_measure_every == 0)
                                and ei_gradnorm_params
                            ):
                                mc_param_grads = torch.autograd.grad(
                                    mc_loss,
                                    ei_gradnorm_params,
                                    retain_graph=True,
                                    allow_unused=True,
                                )
                                ei_param_grads = torch.autograd.grad(
                                    ei_loss,
                                    ei_gradnorm_params,
                                    retain_graph=True,
                                    allow_unused=True,
                                )
                                mc_grad_norm = _grad_l2_norm(mc_param_grads)
                                ei_grad_norm = _grad_l2_norm(ei_param_grads)
                                if mc_grad_norm is not None and ei_grad_norm is not None:
                                    if config['training']['multigpu'] and dist.is_available() and dist.is_initialized():
                                        grad_pair = torch.stack((mc_grad_norm, ei_grad_norm))
                                        dist.all_reduce(grad_pair, op=dist.ReduceOp.SUM)
                                        grad_pair = grad_pair / float(world_size)
                                        mc_grad_norm = grad_pair[0]
                                        ei_grad_norm = grad_pair[1]
                                    ratio_value = float(
                                        mc_grad_norm / (ei_grad_norm + ei_gradnorm_eps)
                                    )
                                    ratio_clipped = float(
                                        np.clip(
                                            ratio_value,
                                            ei_gradnorm_ratio_min,
                                            ei_gradnorm_ratio_max,
                                        )
                                    )
                                    if ei_gradnorm_ratio_ema is None:
                                        ei_gradnorm_ratio_ema = ratio_clipped
                                    else:
                                        ei_gradnorm_ratio_ema = (
                                            ei_gradnorm_ema_beta * ei_gradnorm_ratio_ema
                                            + (1.0 - ei_gradnorm_ema_beta) * ratio_clipped
                                        )
                                    ei_gradnorm_samples += 1
                                    ei_gradnorm_ratio_history.append(ratio_clipped)
                                    ei_gradnorm_ratio_epochs.append(
                                        epoch + (batch_idx / max(1, len(train_loader)))
                                    )
                                    if global_rank == 0 or not config['training']['multigpu']:
                                        writer.add_scalar(
                                            "Loss/EI_GradNorm_Ratio",
                                            ratio_clipped,
                                            iteration_count,
                                        )
                                        writer.add_scalar(
                                            "Loss/EI_GradNorm_Ratio_EMA",
                                            ei_gradnorm_ratio_ema,
                                            iteration_count,
                                        )
                            train_loader_tqdm.set_postfix(
                                mc_loss=mc_loss.item(),
                                ei_loss=ei_loss.item(),
                                rebin_loss=(rebin_loss.item() if rebin_loss is not None else None),
                            )

                        else:
                            train_loader_tqdm.set_postfix(
                                mc_loss=mc_loss.item(),
                                rebin_loss=(rebin_loss.item() if rebin_loss is not None else None),
                            )

                        total_loss = mc_loss * mc_loss_weight
                        if ei_loss is not None:
                            total_loss = total_loss + ei_loss * ei_loss_weight
                        if use_adj_loss:
                            total_loss = total_loss + torch.mul(adj_loss_weight, adj_loss)
                        if rebin_loss is not None and rebin_loss_weight > 0:
                            total_loss = total_loss + rebin_loss * rebin_loss_weight

                    if torch.isnan(total_loss):
                        print(
                            "!!! ERROR: total_loss is NaN before backward pass. Aborting. !!!"
                        )
                        raise RuntimeError("total_loss is NaN")

                    if use_scaler:
                        if detect_anomaly:
                            with torch.autograd.detect_anomaly():
                                scaler.scale(total_loss).backward()
                        else:
                            scaler.scale(total_loss).backward()
                        scaler.unscale_(optimizer)
                    else:
                        if detect_anomaly:
                            with torch.autograd.detect_anomaly():
                                total_loss.backward()
                        else:
                            total_loss.backward()


                    debug_cfg = config.get("debugging", {})
                    if debug_cfg.get("enable_gradient_monitoring", False) and iteration_count % int(debug_cfg.get("monitoring_interval", 100)) == 0:
                    
                        log_gradient_stats(
                            model=model,
                            epoch=epoch,
                            iteration=iteration_count,
                            output_dir=output_dir,
                            log_filename="gradient_stats.csv"
                        )
                        log_lsfpnet_component_grads(
                            model=model,
                            epoch=epoch,
                            iteration=iteration_count,
                            output_dir=output_dir,
                            log_filename="lsfpnet_component_grads.csv"
                        )


                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if use_scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    end = time.time()

                    if global_rank == 0 or not config['training']['multigpu']:
                        print("time for one iteration: ", end-start)

                except RuntimeError as e:
                    # catch only SVD-related failures
                    if "svd" in str(e).lower():
                        svd_fail_count += 1
                        optimizer.zero_grad()
                        print(f"[Warning] Skipping batch {iteration_count} in epoch {epoch} due to SVD failure. "
                            f"Total failures so far: {svd_fail_count}")
                        continue  # skip this batch, go to next one
                    else:
                        raise  # re-raise other errors

            if sample_check_enabled and dist.is_available() and dist.is_initialized():
                try:
                    gathered = [None for _ in range(world_size)]
                    dist.all_gather_object(gathered, sample_check_ids or [])
                    if global_rank == 0:
                        _print_sample_check_summary(epoch, gathered, sample_check_batches)
                except Exception as exc:
                    if global_rank == 0:
                        print(f"[DDP] Sample check failed: {exc}")

            # plot training samples
            if epoch % save_interval == 0:

                if global_rank == 0 or not config['training']['multigpu']:

                    plot_reconstruction_sample(
                        x_recon,
                        f"Training Sample - Epoch {epoch} (AF = {round(acceleration.item(), 1)}, SPF = {int(N_spokes)}, FPG = {Ng})",
                        f"train_sample_epoch_{epoch}",
                        output_dir,
                    )

                    x_recon_reshaped = rearrange(x_recon, 'b c h w t -> b c t h w')

                    if ec_dir is not None:
                        plot_enhancement_curve(
                            x_recon_reshaped,
                            output_filename=os.path.join(
                                ec_dir, f"train_sample_enhancement_curve_epoch_{epoch}.png"
                            ),
                        )
                    
                    if use_ei_loss and t_img is not None:

                        plot_reconstruction_sample(
                            t_img,
                            f"Transformed Train Sample - Epoch {epoch} (AF = {round(acceleration.item(), 1)}, SPF = {int(N_spokes)})",
                            f"transforms/transform_train_sample_epoch_{epoch}",
                            output_dir,
                            x_recon,
                            transform=True
                        )

            # Calculate and store average epoch losses
            epoch_train_mc_loss = running_mc_loss / len(train_loader)
            train_mc_losses.append(epoch_train_mc_loss)
            weighted_train_mc_losses.append(epoch_train_mc_loss*mc_loss_weight)
            if use_ei_loss:
                epoch_train_ei_loss = running_ei_loss / len(train_loader)
            else:
                # Append 0 if EI loss is not used to keep lists aligned
                epoch_train_ei_loss = 0.0
                ei_loss_weight = 0

            train_ei_losses.append(epoch_train_ei_loss)
            weighted_train_ei_losses.append(epoch_train_ei_loss*ei_loss_weight)

            if use_adj_loss:
                epoch_train_adj_loss = running_adj_loss / len(train_loader)
            else:
                epoch_train_adj_loss = 0.0
            train_adj_losses.append(epoch_train_adj_loss)
            weighted_train_adj_losses.append(epoch_train_adj_loss*adj_loss_weight)

            if use_rebin_loss and compute_rebin_this_epoch:
                epoch_train_rebin_loss = running_rebin_loss / len(train_loader)
            else:
                epoch_train_rebin_loss = 0.0
            if use_rebin_loss:
                train_rebin_losses.append(epoch_train_rebin_loss)
                weighted_train_rebin_losses.append(epoch_train_rebin_loss * rebin_loss_weight)


            if global_rank == 0 or not config['training']['multigpu']:
                writer.add_scalar('Loss/Train_MC', epoch_train_mc_loss, epoch)
                writer.add_scalar('Loss/Train_EI', epoch_train_ei_loss, epoch)
                writer.add_scalar('Loss/Train_Adj', epoch_train_adj_loss, epoch)
                if use_rebin_loss:
                    writer.add_scalar('Loss/Rebin_Weight', rebin_loss_weight, epoch)
                    writer.add_scalar('Loss/Train_Rebin', epoch_train_rebin_loss, epoch)
                    writer.add_scalar('Loss/Train_Weighted_Rebin', epoch_train_rebin_loss * rebin_loss_weight, epoch)


            lambda_Ls.append(lambda_L.item())
            lambda_Ss.append(lambda_S.item())
            lambda_spatial_Ls.append(lambda_spatial_L.item())
            lambda_spatial_Ss.append(lambda_spatial_S.item())
            gammas.append(gamma.item())
            lambda_steps.append(lambda_step.item())


            # --- Validation Loop ---
            if epoch % eval_frequency == 0:
                model.eval()
                val_running_mc_loss = 0.0
                val_running_ei_loss = 0.0
                val_running_adj_loss = 0.0
                distributed_eval_this_epoch = distributed_eval and config['training']['multigpu']
                epoch_use_sliding_window = _use_eval_sliding_window(N_time_eval)
                run_eval_metrics_this_rank = (
                    (not config['training']['multigpu'])
                    or distributed_eval_this_epoch
                    or (global_rank == 0)
                )
                if distributed_eval_this_epoch:
                    if global_rank == 0:
                        collect_grasp_baseline_root = int((not _grasp_baseline_ready()) or len(grasp_ssims) == 0)
                    else:
                        collect_grasp_baseline_root = 0
                    collect_grasp_baseline_tensor = torch.tensor(
                        [collect_grasp_baseline_root], dtype=torch.int32, device=device
                    )
                    dist.broadcast(collect_grasp_baseline_tensor, src=0)
                    collect_grasp_baseline = bool(int(collect_grasp_baseline_tensor.item()))
                else:
                    collect_grasp_baseline = (
                        (global_rank == 0 or not config['training']['multigpu'])
                        and (not _grasp_baseline_ready() or len(grasp_ssims) == 0)
                    )
                if (global_rank == 0 or not config['training']['multigpu']) and collect_grasp_baseline:
                    grasp_ssims.clear()
                    grasp_psnrs.clear()
                    grasp_mses.clear()
                    grasp_lpipses.clear()
                    grasp_dc_mses.clear()
                    grasp_dc_maes.clear()
                    grasp_dc_mae_bestfits.clear()
                    grasp_curve_corrs.clear()
                    raw_grasp_dc_mses.clear()
                    raw_grasp_dc_maes.clear()
                    grasp_curve_maes.clear()
                    grasp_ttae_secs.clear()
                    grasp_iauc10_errs.clear()
                    grasp_peak_errs.clear()
                    print(f"[Eval] Epoch {epoch}: collecting GRASP baseline metrics from validation set.")
                local_eval_records = []
                val_infer_times_local = []
                epoch_eval_infer_times = []
                val_loader_tqdm = tqdm(
                    val_dro_eval_loader if distributed_eval_this_epoch else val_dro_loader,
                    desc=f"Epoch {epoch}/{epochs}  Validation",
                    unit="batch",
                    leave=False,
                    disable=config['training']['multigpu'] and global_rank != 0,
                )
                with torch.no_grad():
                    val_batches = 0
                    for (
                        val_dro_kspace_batch,
                        val_csmap,
                        val_ground_truth,
                        val_dro_grasp_img,
                        val_mask,
                        grasp_path,
                        val_raw_kspace,
                        val_raw_grasp_img,
                        val_raw_csmaps,
                    ) in val_loader_tqdm:

                        val_csmap = val_csmap.squeeze(0).to(device)   # Remove batch dim
                        val_ground_truth = val_ground_truth.to(device) # Shape: (1, 2, T, H, W)

                        # prepare inputs
                        val_dro_kspace_batch = val_dro_kspace_batch.squeeze(0).to(device) # Remove batch dim
                        val_dro_grasp_img = val_dro_grasp_img.to(device)

                        raw_eval_available = _raw_eval_available(val_raw_csmaps)
                        if raw_eval_available:
                            val_raw_kspace = val_raw_kspace.squeeze(0).to(device) # Remove batch dim
                            val_raw_grasp_img = val_raw_grasp_img.to(device)
                            val_raw_csmaps = val_raw_csmaps.squeeze(0).to(device)
                        else:
                            val_raw_kspace = None
                            val_raw_grasp_img = None
                            val_raw_csmaps = None

                        # calculate acceleration factor
                        N_spokes = eval_ktraj.shape[1] / config['data']['samples']
                        acceleration = torch.tensor([N_full / int(N_spokes)], dtype=torch.float, device=device)

                        if config['model']['encode_acceleration']:
                            acceleration_encoding = acceleration
                        else: 
                            acceleration_encoding = None

                        if config['model']['encode_time_index'] == False:
                            start_timepoint_index = None
                        else:
                            start_timepoint_index = torch.tensor([0], dtype=torch.float, device=device)
                            
                        try:
                            with amp_autocast():
                                if device.type == "cuda":
                                    torch.cuda.synchronize(device)
                                infer_start = time.perf_counter()
                                if epoch_use_sliding_window:
                                    val_x_recon, val_adj_loss = sliding_window_inference(H, W, N_time_eval, eval_ktraj, eval_dcomp, eval_nufft_ob, eval_adjnufft_ob, eval_chunk_size, eval_chunk_overlap, val_dro_kspace_batch, val_csmap, acceleration_encoding, start_timepoint_index, model, epoch=f"val{epoch}", device=device, norm=config["model"]["norm"], collect_adj_loss=use_adj_loss)  
                                else:
                                    val_x_recon, val_adj_loss, *_ = model(
                                    val_dro_kspace_batch.to(device), eval_physics, val_csmap, acceleration_encoding, start_timepoint_index, epoch=f"val{epoch}", norm=config['model']['norm']
                                    )
                                    val_adj_loss = val_adj_loss.item()
                                if device.type == "cuda":
                                    torch.cuda.synchronize(device)
                                val_infer_times_local.append(float(time.perf_counter() - infer_start))
                                val_raw_x_recon = None
                                if raw_eval_available:
                                    if epoch_use_sliding_window:
                                        val_raw_x_recon, _ = sliding_window_inference(
                                            H, W, N_time_eval,
                                            eval_ktraj, eval_dcomp, eval_nufft_ob, eval_adjnufft_ob,
                                            eval_chunk_size, eval_chunk_overlap,
                                            val_raw_kspace, val_raw_csmaps,
                                            acceleration_encoding, start_timepoint_index,
                                            model, epoch="val0", device=device,
                                            norm=config["model"]["norm"], collect_adj_loss=use_adj_loss
                                        )
                                    else:
                                        val_raw_x_recon, *_ = model(
                                            val_raw_kspace.to(device), eval_physics, val_raw_csmaps,
                                            acceleration_encoding, start_timepoint_index,
                                            epoch="val0", norm=config['model']['norm']
                                        )

                                # fix orientation of raw k-space recon
                                # val_raw_x_recon = torch.rot90(val_raw_x_recon, k=2, dims=[-3,-2])

                                # compute losses
                                if use_adj_loss:
                                    val_running_adj_loss += val_adj_loss

                                val_mc_loss = mc_loss_fn(val_dro_kspace_batch.to(device), val_x_recon, eval_physics, val_csmap)
                                val_running_mc_loss += val_mc_loss.item()

                                if use_ei_loss and compute_ei_this_epoch:
                                    if deterministic_val_ei:
                                        val_seed = deterministic_val_ei_seed + int(epoch) * 100000 + int(val_batches)
                                        val_ei_ctx = _temporary_rng(val_seed)
                                    else:
                                        val_ei_ctx = nullcontext()
                                    with val_ei_ctx:
                                        val_ei_loss, val_t_img = ei_loss_fn(
                                            val_x_recon, eval_physics, model_for_ei, val_csmap, acceleration_encoding, start_timepoint_index
                                        )

                                    val_running_ei_loss += val_ei_loss.item()
                                    val_loader_tqdm.set_postfix(
                                        val_mc_loss=val_mc_loss.item(), val_ei_loss=val_ei_loss.item(), ei_w=ei_loss_weight
                                    )
                                else:
                                    if use_ei_loss:
                                        val_loader_tqdm.set_postfix(val_mc_loss=val_mc_loss.item(), ei_w=ei_loss_weight)
                                    else:
                                        val_loader_tqdm.set_postfix(val_mc_loss=val_mc_loss.item())


                            ## Evaluation
                            if run_eval_metrics_this_rank:
                                ssim, psnr, mse, lpips, dc_mse, dc_mae, recon_corr, grasp_corr, temporal_metrics = eval_sample(
                                    val_dro_kspace_batch,
                                    val_csmap,
                                    val_ground_truth,
                                    val_x_recon,
                                    eval_physics,
                                    val_mask,
                                    val_dro_grasp_img,
                                    acceleration,
                                    int(N_spokes),
                                    eval_dir,
                                    f'epoch{epoch}',
                                    device,
                                    cluster=cluster,
                                    dro_eval=True,
                                    grasp_path=grasp_path,
                                    rescale=config['evaluation']['rescale'],
                                    plot_arrival=True,
                                    arrival_k=config['model']['losses']['ei_loss'].get("arrival_shift_baseline_k", 2.0),
                                    arrival_percentile=config['model']['losses']['ei_loss'].get("arrival_shift_percentile", 0.95),
                                    arrival_baseline_k=config['model']['losses']['ei_loss'].get("arrival_shift_baseline_k", 2.0),
                                    arrival_method=config['model']['losses']['ei_loss'].get("arrival_method", "threshold"),
                                    arrival_fraction=config['model']['losses']['ei_loss'].get("arrival_fraction", 0.1),
                                    arrival_pre_contrast_baseline=config['model']['losses']['ei_loss'].get("pre_contrast_baseline", "n_frames"),
                                    arrival_baseline_seconds=config['model']['losses']['ei_loss'].get("baseline_seconds", 20),
                                    arrival_total_seconds=config['model']['losses']['ei_loss'].get("total_seconds", 150.0),
                                )

                                ssim_grasp = None
                                psnr_grasp = None
                                mse_grasp = None
                                lpips_grasp = None
                                dc_mse_grasp = None
                                dc_mae_grasp = None
                                grasp_dc_mae_bestfit = None
                                if collect_grasp_baseline:
                                    (
                                        ssim_grasp,
                                        psnr_grasp,
                                        mse_grasp,
                                        lpips_grasp,
                                        dc_mse_grasp,
                                        dc_mae_grasp,
                                        grasp_aux,
                                    ) = eval_grasp(
                                        val_dro_kspace_batch,
                                        val_csmap,
                                        val_ground_truth,
                                        val_dro_grasp_img,
                                        eval_physics,
                                        device,
                                        eval_dir,
                                        rescale=config['evaluation']['rescale'],
                                        dro_eval=True,
                                        return_aux=True,
                                    )
                                    if grasp_aux is not None:
                                        grasp_dc_mae_bestfit = grasp_aux.get("grasp_dc_mae_bestfit")
                                curve_mae = None
                                ttae_sec = None
                                iauc10_err = None
                                peak_err = None
                                grasp_curve_mae = None
                                grasp_ttae_sec = None
                                grasp_iauc10_err = None
                                grasp_peak_err = None
                                dl_dc_mae_bestfit = None
                                if temporal_metrics:
                                    curve_mae = temporal_metrics.get("dl_all_curve_mae")
                                    ttae_sec = temporal_metrics.get("dl_all_ttae_sec")
                                    iauc10_err = temporal_metrics.get("dl_all_iauc10_err")
                                    peak_err = temporal_metrics.get("dl_all_peak_err")
                                    grasp_curve_mae = temporal_metrics.get("grasp_all_curve_mae")
                                    grasp_ttae_sec = temporal_metrics.get("grasp_all_ttae_sec")
                                    grasp_iauc10_err = temporal_metrics.get("grasp_all_iauc10_err")
                                    grasp_peak_err = temporal_metrics.get("grasp_all_peak_err")
                                    dl_dc_mae_bestfit = temporal_metrics.get("dl_dc_mae_bestfit")

                                # raw k-space eval
                                dc_mse_raw = None
                                dc_mae_raw = None
                                dc_mse_raw_grasp = None
                                dc_mae_raw_grasp = None
                                raw_ssdu_nmse = None
                                if raw_eval_available:
                                    dc_mse_raw, dc_mae_raw, _, _ = eval_sample(
                                        val_raw_kspace,
                                        val_raw_csmaps,
                                        val_ground_truth,
                                        val_raw_x_recon,
                                        eval_physics,
                                        val_mask,
                                        val_raw_grasp_img,
                                        acceleration,
                                        int(N_spokes),
                                        eval_dir,
                                        label=f'epoch{epoch}',
                                        device=device,
                                        cluster=cluster,
                                        dro_eval=False,
                                        grasp_path=grasp_path,
                                        raw_slice_idx=raw_grasp_slice_idx,
                                        rescale=config['evaluation']['rescale'],
                                        plot_arrival=True,
                                        arrival_k=config['model']['losses']['ei_loss'].get("arrival_shift_baseline_k", 2.0),
                                        arrival_percentile=config['model']['losses']['ei_loss'].get("arrival_shift_percentile", 0.95),
                                        arrival_baseline_k=config['model']['losses']['ei_loss'].get("arrival_shift_baseline_k", 2.0),
                                        arrival_method=config['model']['losses']['ei_loss'].get("arrival_method", "threshold"),
                                        arrival_fraction=config['model']['losses']['ei_loss'].get("arrival_fraction", 0.1),
                                        arrival_pre_contrast_baseline=config['model']['losses']['ei_loss'].get("pre_contrast_baseline", "n_frames"),
                                        arrival_baseline_seconds=config['model']['losses']['ei_loss'].get("baseline_seconds", 20),
                                        arrival_total_seconds=config['model']['losses']['ei_loss'].get("total_seconds", 150.0),
                                    )

                                    if collect_grasp_baseline:
                                        dc_mse_raw_grasp, dc_mae_raw_grasp, _ = eval_grasp(
                                            val_raw_kspace,
                                            val_raw_csmaps,
                                            val_ground_truth,
                                            val_raw_grasp_img,
                                            eval_physics,
                                            device,
                                            eval_dir,
                                            rescale=config['evaluation']['rescale'],
                                            dro_eval=False,
                                        )
                                    if compute_ssdu_eval:
                                        ssdu_chunk_size = eval_chunk_size if epoch_use_sliding_window else None
                                        ssdu_result = compute_ssdu_kspace_nmse(
                                            model,
                                            val_raw_kspace,
                                            val_raw_csmaps,
                                            eval_ktraj,
                                            eval_dcomp,
                                            eval_nufft_ob,
                                            eval_adjnufft_ob,
                                            spokes_per_frame=int(N_spokes),
                                            K_folds=ssdu_k_folds,
                                            baseline_weighting=ssdu_weighting,
                                            device=device,
                                            acceleration_encoding=acceleration_encoding,
                                            start_timepoint_index=start_timepoint_index,
                                            norm=config["model"]["norm"],
                                            epoch=f"val{epoch}",
                                            chunk_size=ssdu_chunk_size,
                                            chunk_overlap=eval_chunk_overlap,
                                        )
                                        raw_ssdu_nmse = ssdu_result.get("ssdu_nmse_mean")
                                if distributed_eval_this_epoch:
                                    local_eval_records.append(
                                        {
                                            "ssim": float(ssim),
                                            "psnr": float(psnr),
                                            "mse": float(mse),
                                            "lpips": float(lpips),
                                            "dc_mse": float(dc_mse),
                                            "dc_mae": float(dc_mae),
                                            "recon_corr": (float(recon_corr) if recon_corr is not None else None),
                                            "curve_mae": (float(curve_mae) if curve_mae is not None and np.isfinite(curve_mae) else None),
                                            "ttae_sec": (float(ttae_sec) if ttae_sec is not None and np.isfinite(ttae_sec) else None),
                                            "iauc10_err": (float(iauc10_err) if iauc10_err is not None and np.isfinite(iauc10_err) else None),
                                            "peak_err": (float(peak_err) if peak_err is not None and np.isfinite(peak_err) else None),
                                            "dl_dc_mae_bestfit": (
                                                float(dl_dc_mae_bestfit)
                                                if dl_dc_mae_bestfit is not None and np.isfinite(dl_dc_mae_bestfit)
                                                else None
                                            ),
                                            "raw_dc_mse": (float(dc_mse_raw) if dc_mse_raw is not None else None),
                                            "raw_dc_mae": (float(dc_mae_raw) if dc_mae_raw is not None else None),
                                            "raw_ssdu_nmse": (
                                                float(raw_ssdu_nmse)
                                                if raw_ssdu_nmse is not None and np.isfinite(raw_ssdu_nmse)
                                                else None
                                            ),
                                            "grasp_corr": (float(grasp_corr) if grasp_corr is not None else None),
                                            "grasp_ssim": (float(ssim_grasp) if ssim_grasp is not None else None),
                                            "grasp_psnr": (float(psnr_grasp) if psnr_grasp is not None else None),
                                            "grasp_mse": (float(mse_grasp) if mse_grasp is not None else None),
                                            "grasp_lpips": (float(lpips_grasp) if lpips_grasp is not None else None),
                                            "grasp_dc_mse": (float(dc_mse_grasp) if dc_mse_grasp is not None else None),
                                            "grasp_dc_mae": (float(dc_mae_grasp) if dc_mae_grasp is not None else None),
                                            "grasp_dc_mae_bestfit": (
                                                float(grasp_dc_mae_bestfit)
                                                if grasp_dc_mae_bestfit is not None and np.isfinite(grasp_dc_mae_bestfit)
                                                else None
                                            ),
                                            "grasp_curve_mae": (
                                                float(grasp_curve_mae)
                                                if grasp_curve_mae is not None and np.isfinite(grasp_curve_mae)
                                                else None
                                            ),
                                            "grasp_ttae_sec": (
                                                float(grasp_ttae_sec)
                                                if grasp_ttae_sec is not None and np.isfinite(grasp_ttae_sec)
                                                else None
                                            ),
                                            "grasp_iauc10_err": (
                                                float(grasp_iauc10_err)
                                                if grasp_iauc10_err is not None and np.isfinite(grasp_iauc10_err)
                                                else None
                                            ),
                                            "grasp_peak_err": (
                                                float(grasp_peak_err)
                                                if grasp_peak_err is not None and np.isfinite(grasp_peak_err)
                                                else None
                                            ),
                                            "raw_grasp_dc_mse": (float(dc_mse_raw_grasp) if dc_mse_raw_grasp is not None else None),
                                            "raw_grasp_dc_mae": (float(dc_mae_raw_grasp) if dc_mae_raw_grasp is not None else None),
                                        }
                                    )
                                else:
                                    if collect_grasp_baseline:
                                        grasp_ssims.append(ssim_grasp)
                                        grasp_psnrs.append(psnr_grasp)
                                        grasp_mses.append(mse_grasp)
                                        grasp_lpipses.append(lpips_grasp)
                                        grasp_dc_mses.append(dc_mse_grasp)
                                        grasp_dc_maes.append(dc_mae_grasp)
                                        if grasp_corr is not None:
                                            grasp_curve_corrs.append(grasp_corr)
                                        if dc_mse_raw_grasp is not None:
                                            raw_grasp_dc_mses.append(dc_mse_raw_grasp)
                                        if dc_mae_raw_grasp is not None:
                                            raw_grasp_dc_maes.append(dc_mae_raw_grasp)
                                        if grasp_curve_mae is not None and np.isfinite(grasp_curve_mae):
                                            grasp_curve_maes.append(float(grasp_curve_mae))
                                        if grasp_ttae_sec is not None and np.isfinite(grasp_ttae_sec):
                                            grasp_ttae_secs.append(float(grasp_ttae_sec))
                                        if grasp_iauc10_err is not None and np.isfinite(grasp_iauc10_err):
                                            grasp_iauc10_errs.append(float(grasp_iauc10_err))
                                        if grasp_peak_err is not None and np.isfinite(grasp_peak_err):
                                            grasp_peak_errs.append(float(grasp_peak_err))
                                    epoch_eval_ssims.append(ssim)
                                    epoch_eval_psnrs.append(psnr)
                                    epoch_eval_mses.append(mse)
                                    epoch_eval_lpipses.append(lpips)
                                    epoch_eval_dc_mses.append(dc_mse)
                                    epoch_eval_dc_maes.append(dc_mae)

                                    if recon_corr is not None:
                                        epoch_eval_curve_corrs.append(recon_corr)
                                    if curve_mae is not None and np.isfinite(curve_mae):
                                        epoch_eval_curve_maes.append(curve_mae)
                                    if ttae_sec is not None and np.isfinite(ttae_sec):
                                        epoch_eval_ttae_secs.append(ttae_sec)
                                    if iauc10_err is not None and np.isfinite(iauc10_err):
                                        epoch_eval_iauc10_errs.append(iauc10_err)
                                    if peak_err is not None and np.isfinite(peak_err):
                                        epoch_eval_peak_errs.append(peak_err)
                                    if dl_dc_mae_bestfit is not None and np.isfinite(dl_dc_mae_bestfit):
                                        epoch_eval_dl_dc_mae_bestfits.append(float(dl_dc_mae_bestfit))

                                    if dc_mse_raw is not None:
                                        epoch_eval_raw_dc_mses.append(dc_mse_raw)
                                    if dc_mae_raw is not None:
                                        epoch_eval_raw_dc_maes.append(dc_mae_raw)
                                    if raw_ssdu_nmse is not None and np.isfinite(raw_ssdu_nmse):
                                        epoch_eval_raw_ssdu_nmses.append(float(raw_ssdu_nmse))
                                    if (
                                        collect_grasp_baseline
                                        and grasp_dc_mae_bestfit is not None
                                        and np.isfinite(grasp_dc_mae_bestfit)
                                    ):
                                        grasp_dc_mae_bestfits.append(float(grasp_dc_mae_bestfit))

                            val_batches += 1
                            if max_val_batches is not None and val_batches >= int(max_val_batches):
                                break
                            
                            
                        except RuntimeError as e:
                            # catch only SVD-related failures
                            if "svd" in str(e).lower():
                                svd_fail_count += 1
                                optimizer.zero_grad()
                                print(f"[Warning] Skipping batch validation sample in epoch {epoch} due to SVD failure. "
                                    f"Total failures so far: {svd_fail_count}")
                                continue  # skip this batch, go to next one
                            else:
                                raise  # re-raise other errors


                if distributed_eval_this_epoch:
                    loss_reduce = torch.tensor(
                        [val_running_mc_loss, val_running_ei_loss, val_running_adj_loss, float(val_batches)],
                        dtype=torch.float64,
                        device=device,
                    )
                    dist.all_reduce(loss_reduce, op=dist.ReduceOp.SUM)
                    val_running_mc_loss = float(loss_reduce[0].item())
                    val_running_ei_loss = float(loss_reduce[1].item())
                    val_running_adj_loss = float(loss_reduce[2].item())
                    val_batches = int(loss_reduce[3].item())
                    gathered_eval_records = [None for _ in range(world_size)]
                    dist.all_gather_object(gathered_eval_records, local_eval_records)
                    gathered_infer_times = [None for _ in range(world_size)]
                    dist.all_gather_object(
                        gathered_infer_times,
                        [float(t) for t in val_infer_times_local],
                    )

                    if global_rank == 0:
                        all_eval_records = _flatten_gathered_records(gathered_eval_records)
                        if not all_eval_records:
                            raise RuntimeError(f"Epoch {epoch}: distributed eval collected zero records.")
                        epoch_eval_infer_times = _flatten_gathered_records(gathered_infer_times)

                        epoch_eval_ssims = [r["ssim"] for r in all_eval_records]
                        epoch_eval_psnrs = [r["psnr"] for r in all_eval_records]
                        epoch_eval_mses = [r["mse"] for r in all_eval_records]
                        epoch_eval_lpipses = [r["lpips"] for r in all_eval_records]
                        epoch_eval_dc_mses = [r["dc_mse"] for r in all_eval_records]
                        epoch_eval_dc_maes = [r["dc_mae"] for r in all_eval_records]
                        epoch_eval_raw_dc_mses = [
                            r["raw_dc_mse"] for r in all_eval_records if r.get("raw_dc_mse") is not None
                        ]
                        epoch_eval_raw_dc_maes = [
                            r["raw_dc_mae"] for r in all_eval_records if r.get("raw_dc_mae") is not None
                        ]
                        epoch_eval_curve_corrs = [r["recon_corr"] for r in all_eval_records if r.get("recon_corr") is not None]
                        epoch_eval_curve_maes = [r["curve_mae"] for r in all_eval_records if r.get("curve_mae") is not None]
                        epoch_eval_ttae_secs = [r["ttae_sec"] for r in all_eval_records if r.get("ttae_sec") is not None]
                        epoch_eval_iauc10_errs = [r["iauc10_err"] for r in all_eval_records if r.get("iauc10_err") is not None]
                        epoch_eval_peak_errs = [r["peak_err"] for r in all_eval_records if r.get("peak_err") is not None]
                        epoch_eval_dl_dc_mae_bestfits = [
                            r["dl_dc_mae_bestfit"] for r in all_eval_records if r.get("dl_dc_mae_bestfit") is not None
                        ]
                        epoch_eval_raw_ssdu_nmses = [
                            r["raw_ssdu_nmse"] for r in all_eval_records if r.get("raw_ssdu_nmse") is not None
                        ]

                        if collect_grasp_baseline:
                            grasp_ssims = [r["grasp_ssim"] for r in all_eval_records if r.get("grasp_ssim") is not None]
                            grasp_psnrs = [r["grasp_psnr"] for r in all_eval_records if r.get("grasp_psnr") is not None]
                            grasp_mses = [r["grasp_mse"] for r in all_eval_records if r.get("grasp_mse") is not None]
                            grasp_lpipses = [r["grasp_lpips"] for r in all_eval_records if r.get("grasp_lpips") is not None]
                            grasp_dc_mses = [r["grasp_dc_mse"] for r in all_eval_records if r.get("grasp_dc_mse") is not None]
                            grasp_dc_maes = [r["grasp_dc_mae"] for r in all_eval_records if r.get("grasp_dc_mae") is not None]
                            grasp_dc_mae_bestfits = [
                                r["grasp_dc_mae_bestfit"]
                                for r in all_eval_records
                                if r.get("grasp_dc_mae_bestfit") is not None
                            ]
                            grasp_curve_corrs = [r["grasp_corr"] for r in all_eval_records if r.get("grasp_corr") is not None]
                            raw_grasp_dc_mses = [r["raw_grasp_dc_mse"] for r in all_eval_records if r.get("raw_grasp_dc_mse") is not None]
                            raw_grasp_dc_maes = [r["raw_grasp_dc_mae"] for r in all_eval_records if r.get("raw_grasp_dc_mae") is not None]
                            grasp_curve_maes = [
                                r["grasp_curve_mae"] for r in all_eval_records if r.get("grasp_curve_mae") is not None
                            ]
                            grasp_ttae_secs = [
                                r["grasp_ttae_sec"] for r in all_eval_records if r.get("grasp_ttae_sec") is not None
                            ]
                            grasp_iauc10_errs = [
                                r["grasp_iauc10_err"] for r in all_eval_records if r.get("grasp_iauc10_err") is not None
                            ]
                            grasp_peak_errs = [
                                r["grasp_peak_err"] for r in all_eval_records if r.get("grasp_peak_err") is not None
                            ]
                elif global_rank == 0 or not config['training']['multigpu']:
                    epoch_eval_infer_times = [float(t) for t in val_infer_times_local]

                # Calculate and store average validation evaluation metrics
                if global_rank == 0 or not config['training']['multigpu']:
                    if collect_grasp_baseline:
                        if len(grasp_ssims) == 0:
                            raise RuntimeError(
                                f"Epoch {epoch}: GRASP baseline collection returned zero samples."
                            )
                        avg_grasp_ssim = _mean_or_nan(grasp_ssims)
                        avg_grasp_psnr = _mean_or_nan(grasp_psnrs)
                        avg_grasp_mse = _mean_or_nan(grasp_mses)
                        avg_grasp_lpips = _mean_or_nan(grasp_lpipses)
                        avg_grasp_dc_mse = _mean_or_nan(grasp_dc_mses)
                        avg_grasp_dc_mae = _mean_or_nan(grasp_dc_maes)
                        avg_grasp_dc_mae_bestfit = _mean_or_nan(grasp_dc_mae_bestfits)
                        avg_grasp_curve_corr = _mean_or_nan(grasp_curve_corrs)
                        avg_grasp_raw_dc_mae = _mean_or_nan(raw_grasp_dc_maes)
                        avg_grasp_raw_dc_mse = _mean_or_nan(raw_grasp_dc_mses)
                        avg_grasp_curve_mae = _mean_or_nan(grasp_curve_maes)
                        avg_grasp_ttae_sec = _mean_or_nan(grasp_ttae_secs)
                        avg_grasp_iauc10_err = _mean_or_nan(grasp_iauc10_errs)
                        avg_grasp_peak_err = _mean_or_nan(grasp_peak_errs)
                        avg_grasp_raw_ssdu_nmse = float("nan")
                        if not _grasp_baseline_ready():
                            raise RuntimeError(
                                f"Epoch {epoch}: GRASP baseline metrics are non-finite after collection."
                            )
                        print(
                            f"[Eval] Epoch {epoch}: GRASP baseline set "
                            f"(SSIM={avg_grasp_ssim:.4f}, PSNR={avg_grasp_psnr:.4f}, "
                            f"LPIPS={avg_grasp_lpips:.4f}, CurveCorr={avg_grasp_curve_corr:.4f})."
                        )

                    epoch_eval_ssim = np.mean(epoch_eval_ssims)
                    epoch_eval_psnr = np.mean(epoch_eval_psnrs)
                    epoch_eval_mse = np.mean(epoch_eval_mses)
                    epoch_eval_lpips = np.mean(epoch_eval_lpipses)
                    epoch_eval_dc_mse = np.mean(epoch_eval_dc_mses)
                    epoch_eval_dc_mae = np.mean(epoch_eval_dc_maes)
                    epoch_eval_curve_corr = np.mean(epoch_eval_curve_corrs)
                    epoch_eval_raw_dc_mse = np.mean(epoch_eval_raw_dc_mses)
                    epoch_eval_raw_dc_mae = np.mean(epoch_eval_raw_dc_maes)
                    epoch_eval_curve_mae = np.mean(epoch_eval_curve_maes) if epoch_eval_curve_maes else np.nan
                    epoch_eval_ttae_sec = np.mean(epoch_eval_ttae_secs) if epoch_eval_ttae_secs else np.nan
                    epoch_eval_iauc10_err = np.mean(epoch_eval_iauc10_errs) if epoch_eval_iauc10_errs else np.nan
                    epoch_eval_peak_err = np.mean(epoch_eval_peak_errs) if epoch_eval_peak_errs else np.nan
                    epoch_eval_dl_dc_mae_bestfit = (
                        np.mean(epoch_eval_dl_dc_mae_bestfits) if epoch_eval_dl_dc_mae_bestfits else np.nan
                    )
                    epoch_eval_raw_ssdu_nmse = (
                        np.mean(epoch_eval_raw_ssdu_nmses) if epoch_eval_raw_ssdu_nmses else np.nan
                    )
                    if epoch_eval_infer_times:
                        epoch_infer_mean = float(np.mean(epoch_eval_infer_times))
                        epoch_infer_std = float(np.std(epoch_eval_infer_times, ddof=1)) if len(epoch_eval_infer_times) > 1 else 0.0
                        infer_mode = "sliding-window" if epoch_use_sliding_window else "direct-full"
                        print(
                            f"[Eval] Epoch {epoch}: inference time/sample ({infer_mode}) = "
                            f"{epoch_infer_mean:.3f}s ± {epoch_infer_std:.3f}s "
                            f"(n={len(epoch_eval_infer_times)})"
                        )
                        writer.add_scalar('Timing/Val_Inference_Seconds_Mean', epoch_infer_mean, epoch)
                        writer.add_scalar('Timing/Val_Inference_Seconds_Std', epoch_infer_std, epoch)

                    eval_ssims.append(epoch_eval_ssim)
                    eval_psnrs.append(epoch_eval_psnr)
                    eval_mses.append(epoch_eval_mse)
                    eval_lpipses.append(epoch_eval_lpips)
                    eval_dc_mses.append(epoch_eval_dc_mse) 
                    eval_dc_maes.append(epoch_eval_dc_mae) 
                    eval_raw_dc_mses.append(epoch_eval_raw_dc_mse) 
                    eval_raw_dc_maes.append(epoch_eval_raw_dc_mae)    
                    eval_curve_corrs.append(epoch_eval_curve_corr)  
                    eval_temporal_epochs.append(epoch)
                    eval_curve_maes.append(epoch_eval_curve_mae)
                    eval_ttae_secs.append(epoch_eval_ttae_sec)
                    eval_iauc10_errs.append(epoch_eval_iauc10_err)
                    eval_peak_errs.append(epoch_eval_peak_err)
                    eval_dl_dc_mae_bestfits.append(epoch_eval_dl_dc_mae_bestfit)
                    eval_raw_ssdu_nmses.append(epoch_eval_raw_ssdu_nmse)

                    spf_key = int(N_spokes_eval)
                    if spf_key in eval_spf_curves:
                        eval_spf_curves[spf_key]["epochs"].append(epoch)
                        eval_spf_curves[spf_key]["eval_ssims"].append(epoch_eval_ssim)
                        eval_spf_curves[spf_key]["eval_psnrs"].append(epoch_eval_psnr)
                        eval_spf_curves[spf_key]["eval_mses"].append(epoch_eval_mse)
                        eval_spf_curves[spf_key]["eval_lpipses"].append(epoch_eval_lpips)
                        eval_spf_curves[spf_key]["eval_raw_dc_maes"].append(epoch_eval_raw_dc_mae)
                        eval_spf_curves[spf_key]["eval_curve_corrs"].append(epoch_eval_curve_corr)

    
                    writer.add_scalar('Metric/SSIM', epoch_eval_ssim, epoch)
                    writer.add_scalar('Metric/PSNR', epoch_eval_psnr, epoch)
                    writer.add_scalar('Metric/MSE', epoch_eval_mse, epoch)
                    writer.add_scalar('Metric/LPIPS', epoch_eval_lpips, epoch)
                    writer.add_scalar('Metric/DC_MSE', epoch_eval_dc_mse, epoch)
                    writer.add_scalar('Metric/DC_MAE', epoch_eval_dc_mae, epoch)
                    writer.add_scalar('Metric/RAW_DC_MSE', epoch_eval_raw_dc_mse, epoch)
                    writer.add_scalar('Metric/RAW_DC_MAE', epoch_eval_raw_dc_mae, epoch)
                    writer.add_scalar('Metric/EC_Corr', epoch_eval_curve_corr, epoch)
                    if np.isfinite(epoch_eval_curve_mae):
                        writer.add_scalar('Metric/Temporal_Curve_MAE', epoch_eval_curve_mae, epoch)
                    if np.isfinite(epoch_eval_ttae_sec):
                        writer.add_scalar('Metric/Temporal_TTAE_sec', epoch_eval_ttae_sec, epoch)
                    if np.isfinite(epoch_eval_iauc10_err):
                        writer.add_scalar('Metric/Temporal_IAUC10_err', epoch_eval_iauc10_err, epoch)
                    if np.isfinite(epoch_eval_peak_err):
                        writer.add_scalar('Metric/Temporal_Peak_err', epoch_eval_peak_err, epoch)
                    if np.isfinite(epoch_eval_dl_dc_mae_bestfit):
                        writer.add_scalar('Metric/DL_DC_MAE_BESTFIT', epoch_eval_dl_dc_mae_bestfit, epoch)
                    if np.isfinite(epoch_eval_raw_ssdu_nmse):
                        writer.add_scalar('Metric/RAW_SSDU_NMSE', epoch_eval_raw_ssdu_nmse, epoch)


                    
                    
                    # save a sample from the last validation batch of the epoch
                    if epoch % save_interval == 0:
                        
                        plot_reconstruction_sample(
                            val_x_recon,
                            f"Validation Sample - Epoch {epoch} (AF = {round(acceleration.item(), 1)}, SPF = {int(N_spokes)})",
                            f"val_sample_epoch_{epoch}",
                            output_dir,
                            val_dro_grasp_img
                        )

                        val_x_recon_reshaped = rearrange(val_x_recon, 'b c h w t -> b c t h w')

                        if ec_dir is not None:
                            plot_enhancement_curve(
                                val_x_recon_reshaped,
                                output_filename=os.path.join(
                                    ec_dir, f"val_dro_sample_enhancement_curve_epoch_{epoch}.png"
                                ),
                                show_arrival=True,
                                arrival_percentile=config['model']['losses']['ei_loss'].get("arrival_shift_percentile", 0.95),
                                arrival_baseline_k=config['model']['losses']['ei_loss'].get("arrival_shift_baseline_k", 2.0),
                                arrival_method=config['model']['losses']['ei_loss'].get("arrival_method", "threshold"),
                                arrival_fraction=config['model']['losses']['ei_loss'].get("arrival_fraction", 0.1),
                                arrival_pre_contrast_baseline=config['model']['losses']['ei_loss'].get("pre_contrast_baseline", "n_frames"),
                                arrival_baseline_seconds=config['model']['losses']['ei_loss'].get("baseline_seconds", 20),
                                arrival_total_seconds=config['model']['losses']['ei_loss'].get("total_seconds", 150.0),
                            )
                        
                        if ec_dir is not None:
                            plot_enhancement_curve(
                                val_dro_grasp_img,
                                output_filename=os.path.join(
                                    ec_dir, f"val_dro_grasp_sample_enhancement_curve_epoch_{epoch}.png"
                                ),
                                show_arrival=True,
                                arrival_percentile=config['model']['losses']['ei_loss'].get("arrival_shift_percentile", 0.95),
                                arrival_baseline_k=config['model']['losses']['ei_loss'].get("arrival_shift_baseline_k", 2.0),
                                arrival_method=config['model']['losses']['ei_loss'].get("arrival_method", "threshold"),
                                arrival_fraction=config['model']['losses']['ei_loss'].get("arrival_fraction", 0.1),
                                arrival_pre_contrast_baseline=config['model']['losses']['ei_loss'].get("pre_contrast_baseline", "n_frames"),
                                arrival_baseline_seconds=config['model']['losses']['ei_loss'].get("baseline_seconds", 20),
                                arrival_total_seconds=config['model']['losses']['ei_loss'].get("total_seconds", 150.0),
                            )


                        if use_ei_loss and val_t_img is not None:
                            plot_reconstruction_sample(
                                val_t_img,
                                f"Transformed Validation Sample - Epoch {epoch} (AF = {round(acceleration.item(), 1)}, SPF = {int(N_spokes)})",
                                f"transforms/transform_val_sample_epoch_{epoch}",
                                output_dir,
                                val_x_recon,
                                transform=True
                            )


                # Calculate and store average validation losses
                if distributed_eval_this_epoch:
                    denom = val_batches if val_batches > 0 else 1
                else:
                    denom = val_batches if val_batches > 0 else len(val_dro_loader)
                epoch_val_mc_loss = val_running_mc_loss / denom
                val_mc_losses.append(epoch_val_mc_loss)

                if use_ei_loss:
                    epoch_val_ei_loss = val_running_ei_loss / denom
                else:
                    epoch_val_ei_loss = 0.0

                val_ei_losses.append(epoch_val_ei_loss)

                if use_adj_loss:
                    epoch_val_adj_loss = val_running_adj_loss / denom
                else:
                    epoch_val_adj_loss = 0.0
                
                val_adj_losses.append(epoch_val_adj_loss)

                if global_rank == 0 or not config['training']['multigpu']:
                    writer.add_scalar('Loss/Val_MC', epoch_val_mc_loss, epoch)
                    writer.add_scalar('Loss/Val_EI', epoch_val_ei_loss, epoch)
                    writer.add_scalar('Loss/Val_Adj', epoch_val_adj_loss, epoch)

                    if np.isfinite(epoch_eval_psnr) and epoch_eval_psnr > best_psnr:
                        best_psnr = float(epoch_eval_psnr)
                        best_epoch = epoch
                        train_curves, val_curves, eval_curves = _build_checkpoint_curves()
                        save_checkpoint(
                            model,
                            optimizer,
                            epoch + 1,
                            train_curves,
                            val_curves,
                            eval_curves,
                            ei_target_weight_effective,
                            step0_train_ei_loss,
                            epoch_train_mc_loss,
                            avg_grasp_ssim,
                            avg_grasp_psnr,
                            avg_grasp_mse,
                            avg_grasp_lpips,
                            avg_grasp_dc_mse,
                            avg_grasp_dc_mae,
                            avg_grasp_curve_corr,
                            avg_grasp_raw_dc_mae,
                            avg_grasp_raw_dc_mse,
                            best_checkpoint_path,
                        )
                        print(
                            f"[Checkpoint] New best PSNR {best_psnr:.4f} at epoch {epoch}. Saved to {best_checkpoint_path}"
                        )
                        if run_state is not None:
                            with state_lock:
                                run_state["best_checkpoint_epoch"] = int(best_epoch)
                                run_state["best_checkpoint_psnr"] = float(best_psnr)
                                _save_run_state(run_state_path, run_state)




                # --- Plotting and Logging ---
                if epoch % save_interval == 0:

                    if global_rank == 0 or not config['training']['multigpu']:

                        # plot losses in one figure
                        # Set the seaborn style
                        sns.set_style("whitegrid")

                        # Create a figure and a set of subplots
                        fig, axes = plt.subplots(2, 4, figsize=(22, 10))

                        # Plot Training Adjoint Loss
                        sns.lineplot(x=range(len(train_adj_losses)), y=train_adj_losses, ax=axes[0, 0])
                        axes[0, 0].set_title("Training Adjoint Loss")
                        axes[0, 0].set_xlabel("Epoch")
                        axes[0, 0].set_ylabel("Adjoint Loss")

                        # Plot Training MC Loss
                        sns.lineplot(x=range(len(train_mc_losses)), y=train_mc_losses, ax=axes[0, 1])
                        axes[0, 1].set_title("Training MC Loss")
                        axes[0, 1].set_xlabel("Epoch")
                        axes[0, 1].set_ylabel("MC Loss")

                        # Plot Training EI Loss
                        sns.lineplot(x=range(len(train_ei_losses)), y=train_ei_losses, ax=axes[0, 2])
                        axes[0, 2].set_title("Training EI Loss")
                        axes[0, 2].set_xlabel("Epoch")
                        axes[0, 2].set_ylabel("EI Loss")

                        # Plot Learning Rate Schedule
                        if lr_history:
                            sns.lineplot(x=lr_epochs, y=lr_history, ax=axes[0, 3])
                        axes[0, 3].set_title("Learning Rate Schedule")
                        axes[0, 3].set_xlabel("Epoch")
                        axes[0, 3].set_ylabel("Learning Rate")

                        # Plot Validation Adjoint Loss
                        sns.lineplot(x=range(0, len(val_adj_losses)*eval_frequency, eval_frequency), y=val_adj_losses, ax=axes[1, 0], color='orange')
                        axes[1, 0].set_title(f"Validation Adjoint Loss ({N_spokes_eval} spokes/frame)")
                        axes[1, 0].set_xlabel("Epoch")
                        axes[1, 0].set_ylabel("Adjoint Loss")

                        # Plot Validation MC Loss
                        sns.lineplot(x=range(0, len(val_mc_losses)*eval_frequency, eval_frequency), y=val_mc_losses, ax=axes[1, 1], color='orange')
                        axes[1, 1].set_title(f"Validation MC Loss ({N_spokes_eval} spokes/frame)")
                        axes[1, 1].set_xlabel("Epoch")
                        axes[1, 1].set_ylabel("MC Loss")

                        # Plot Validation EI Loss
                        sns.lineplot(x=range(0, len(val_ei_losses)*eval_frequency, eval_frequency), y=val_ei_losses, ax=axes[1, 2], color='orange')
                        axes[1, 2].set_title(f"Validation EI Loss ({N_spokes_eval} spokes/frame)")
                        axes[1, 2].set_xlabel("Epoch")
                        axes[1, 2].set_ylabel("EI Loss")

                        # Plot EI Loss Weight Schedule
                        if ei_weight_history:
                            sns.lineplot(x=ei_weight_epochs, y=ei_weight_history, ax=axes[1, 3], color='orange')
                        axes[1, 3].set_title("EI Loss Weight Schedule")
                        axes[1, 3].set_xlabel("Epoch")
                        axes[1, 3].set_ylabel("EI Weight")

                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, "losses.png"))
                        plt.close()


                        # plot learnable parameters in one figure
                        # Set the seaborn style
                        sns.set_style("whitegrid")

                        # Create a figure and a set of subplots
                        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

                        sns.lineplot(x=range(len(lambda_Ls)), y=lambda_Ls, ax=axes[0, 0])
                        axes[0, 0].set_title("Lambda_L Parameter Value")
                        axes[0, 0].set_xlabel("Epoch")
                        axes[0, 0].set_ylabel("Lambda_L")

                        sns.lineplot(x=range(len(lambda_Ss)), y=lambda_Ss, ax=axes[0, 1])
                        axes[0, 1].set_title("Lambda_S Parameter Value")
                        axes[0, 1].set_xlabel("Epoch")
                        axes[0, 1].set_ylabel("Lambda_S")

                        sns.lineplot(x=range(len(lambda_spatial_Ls)), y=lambda_spatial_Ls, ax=axes[0, 2])
                        axes[0, 2].set_title("Spatial Lambda_L Parameter Value")
                        axes[0, 2].set_xlabel("Epoch")
                        axes[0, 2].set_ylabel("Spatial Lambda_L")

                        sns.lineplot(x=range(len(lambda_spatial_Ss)), y=lambda_spatial_Ss, ax=axes[1, 0])
                        axes[1, 0].set_title("Spatial Lambda_S Parameter Value")
                        axes[1, 0].set_xlabel("Epoch")
                        axes[1, 0].set_ylabel("Spatial Lambda_S")

                        sns.lineplot(x=range(len(gammas)), y=gammas, ax=axes[1, 1])
                        axes[1, 1].set_title("Gamma Parameter Value")
                        axes[1, 1].set_xlabel("Epoch")
                        axes[1, 1].set_ylabel("Gamma")

                        sns.lineplot(x=range(len(lambda_steps)), y=lambda_steps, ax=axes[1, 2])
                        axes[1, 2].set_title("Lambda Step Parameter Value")
                        axes[1, 2].set_xlabel("Epoch")
                        axes[1, 2].set_ylabel("Lambda Step")

                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, "parameters.png"))
                        plt.close()


                        # Plot Weighted Losses
                        plt.figure()
                        plt.plot(weighted_train_mc_losses, label="MC Loss")
                        plt.plot(weighted_train_ei_losses, label="EI Loss")
                        plt.plot(weighted_train_adj_losses, label="Adjoint Loss")
                        if weighted_train_rebin_losses:
                            plt.plot(weighted_train_rebin_losses, label="Rebin Loss")
                        plt.xlabel("Epoch")
                        plt.ylabel("Loss")
                        plt.title("Weighted Training Losses")
                        plt.legend()
                        plt.grid(True)
                        plt.savefig(os.path.join(output_dir, "weighted_losses.png"))
                        plt.close()

                        # Plot Learning Rate Schedule
                        if lr_history:
                            plt.figure()
                            plt.plot(lr_epochs, lr_history)
                            plt.xlabel("Epoch")
                            plt.ylabel("Learning Rate")
                            plt.title("Learning Rate Schedule")
                            plt.grid(True)
                            plt.savefig(os.path.join(output_dir, "learning_rate.png"))
                            plt.close()

                        # Plot Rebin Loss (if enabled)
                        if use_rebin_loss and train_rebin_losses:
                            plt.figure()
                            plt.plot(train_rebin_losses, label="Rebin Loss")
                            if weighted_train_rebin_losses:
                                plt.plot(weighted_train_rebin_losses, label="Weighted Rebin Loss")
                            plt.xlabel("Epoch")
                            plt.ylabel("Loss")
                            plt.title("Training Rebin Loss")
                            plt.legend()
                            plt.grid(True)
                            plt.savefig(os.path.join(output_dir, "rebin_loss.png"))
                            plt.close()

                        # plot evaluation metrics in one figure

                        # Set the seaborn style
                        sns.set_style("whitegrid")

                        eval_plot_specs = [
                            ("DRO SSIM", "SSIM", eval_ssims, avg_grasp_ssim),
                            ("DRO PSNR", "PSNR", eval_psnrs, avg_grasp_psnr),
                            ("DRO Image MSE", "MSE", eval_mses, avg_grasp_mse),
                            ("DRO LPIPS", "LPIPS", eval_lpipses, avg_grasp_lpips),
                            ("DRO k-space MAE (sim)", "MAE", eval_dc_maes, avg_grasp_dc_mae),
                            ("Non-DRO k-space MAE", "MAE", eval_raw_dc_maes, avg_grasp_raw_dc_mae),
                            (
                                "DRO Curve Correlation",
                                "Pearson Correlation Coefficient",
                                eval_curve_corrs,
                                avg_grasp_curve_corr,
                            ),
                            (
                                "DRO k-space MAE (best-fit gain)",
                                "MAE",
                                eval_dl_dc_mae_bestfits,
                                avg_grasp_dc_mae_bestfit,
                            ),
                            ("Non-DRO SSDU NMSE", "NMSE", eval_raw_ssdu_nmses, avg_grasp_raw_ssdu_nmse),
                        ]

                        ncols = 4
                        nrows = int(math.ceil(len(eval_plot_specs) / ncols))
                        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4.5))
                        axes = np.array(axes, ndmin=1).reshape(nrows, ncols)
                        fig.suptitle(f'Evaluation Metrics Over Epochs ({N_spokes_eval} spokes/frame)', fontsize=20)

                        for idx, (title, ylabel, series, grasp_baseline) in enumerate(eval_plot_specs):
                            ax = axes[idx // ncols, idx % ncols]
                            if eval_temporal_epochs and len(eval_temporal_epochs) == len(series):
                                x = eval_temporal_epochs
                            else:
                                x = list(range(0, len(series) * eval_frequency, eval_frequency))
                            finite_values = [v for v in series if v is not None and np.isfinite(v)]
                            if finite_values:
                                sns.lineplot(x=x, y=series, ax=ax)
                            else:
                                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                            if grasp_baseline is not None and np.isfinite(grasp_baseline):
                                ax.axhline(y=grasp_baseline, color='red', linestyle='--', linewidth=2)
                            ax.set_title(title)
                            ax.set_xlabel("Epoch")
                            ax.set_ylabel(ylabel)
                        for idx in range(len(eval_plot_specs), nrows * ncols):
                            axes[idx // ncols, idx % ncols].axis("off")

                        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                        plt.savefig(os.path.join(output_dir, "eval_metrics.png"))
                        plt.close()

                        if eval_curve_maes or eval_ttae_secs or eval_iauc10_errs or eval_peak_errs:
                            if eval_temporal_epochs and len(eval_temporal_epochs) == len(eval_curve_maes):
                                temporal_epochs = eval_temporal_epochs
                            else:
                                temporal_epochs = list(range(0, len(eval_curve_maes) * eval_frequency, eval_frequency))

                            min_len = min(
                                len(temporal_epochs),
                                len(eval_curve_maes),
                                len(eval_ttae_secs),
                                len(eval_iauc10_errs),
                                len(eval_peak_errs),
                            )
                            if min_len > 0:
                                temporal_epochs = temporal_epochs[:min_len]
                                curve_maes_plot = eval_curve_maes[:min_len]
                                ttae_plot = eval_ttae_secs[:min_len]
                                iauc10_plot = eval_iauc10_errs[:min_len]
                                peak_plot = eval_peak_errs[:min_len]

                                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                                fig.suptitle(
                                    f"Temporal Fidelity Metrics Over Epochs ({N_spokes_eval} spokes/frame)",
                                    fontsize=20,
                                )

                                sns.lineplot(x=temporal_epochs, y=curve_maes_plot, ax=axes[0, 0])
                                axes[0, 0].set_title("Curve MAE")
                                axes[0, 0].set_xlabel("Epoch")
                                axes[0, 0].set_ylabel("MAE")

                                sns.lineplot(x=temporal_epochs, y=ttae_plot, ax=axes[0, 1])
                                axes[0, 1].set_title("Time to Arrival Error")
                                axes[0, 1].set_xlabel("Epoch")
                                axes[0, 1].set_ylabel("Seconds")

                                sns.lineplot(x=temporal_epochs, y=iauc10_plot, ax=axes[1, 0])
                                axes[1, 0].set_title("IAUC10 Error")
                                axes[1, 0].set_xlabel("Epoch")
                                axes[1, 0].set_ylabel("Error")

                                sns.lineplot(x=temporal_epochs, y=peak_plot, ax=axes[1, 1])
                                axes[1, 1].set_title("Peak Enhancement Error")
                                axes[1, 1].set_xlabel("Epoch")
                                axes[1, 1].set_ylabel("Error")

                                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                                plt.savefig(os.path.join(output_dir, "eval_temporal_metrics.png"))
                                plt.close()

                        if curriculum_enabled and eval_spf_curves:
                            for spf in sorted(eval_spf_curves):
                                spf_curves = eval_spf_curves[spf]
                                if not spf_curves["epochs"]:
                                    continue

                                fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                                fig.suptitle(
                                    f"Evaluation Metrics Over Epochs ({spf} spokes/frame)",
                                    fontsize=20,
                                )

                                sns.lineplot(
                                    x=spf_curves["epochs"],
                                    y=spf_curves["eval_ssims"],
                                    ax=axes[0, 0],
                                )
                                axes[0, 0].set_title("DRO SSIM")
                                axes[0, 0].set_xlabel("Epoch")
                                axes[0, 0].set_ylabel("SSIM")

                                sns.lineplot(
                                    x=spf_curves["epochs"],
                                    y=spf_curves["eval_psnrs"],
                                    ax=axes[0, 1],
                                )
                                axes[0, 1].set_title("DRO PSNR")
                                axes[0, 1].set_xlabel("Epoch")
                                axes[0, 1].set_ylabel("PSNR")

                                sns.lineplot(
                                    x=spf_curves["epochs"],
                                    y=spf_curves["eval_mses"],
                                    ax=axes[0, 2],
                                )
                                axes[0, 2].set_title("DRO Image MSE")
                                axes[0, 2].set_xlabel("Epoch")
                                axes[0, 2].set_ylabel("MSE")

                                sns.lineplot(
                                    x=spf_curves["epochs"],
                                    y=spf_curves["eval_lpipses"],
                                    ax=axes[1, 0],
                                )
                                axes[1, 0].set_title("DRO LPIPS")
                                axes[1, 0].set_xlabel("Epoch")
                                axes[1, 0].set_ylabel("LPIPS")

                                sns.lineplot(
                                    x=spf_curves["epochs"],
                                    y=spf_curves["eval_raw_dc_maes"],
                                    ax=axes[1, 1],
                                )
                                axes[1, 1].set_title("Non-DRO k-space MAE")
                                axes[1, 1].set_xlabel("Epoch")
                                axes[1, 1].set_ylabel("MAE")

                                sns.lineplot(
                                    x=spf_curves["epochs"],
                                    y=spf_curves["eval_curve_corrs"],
                                    ax=axes[1, 2],
                                )
                                axes[1, 2].set_title("DRO Curve Correlation")
                                axes[1, 2].set_xlabel("Epoch")
                                axes[1, 2].set_ylabel("Pearson Correlation Coefficient")

                                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                                plt.savefig(
                                    os.path.join(output_dir, f"eval_metrics_{spf}spf.png")
                                )
                                plt.close()

                        epoch_labels = range(0, len(eval_dc_mses)*eval_frequency, eval_frequency)

                        

                        plt.figure()
                        plt.plot(epoch_labels, eval_dc_mses)
                        plt.axhline(
                            y=avg_grasp_dc_mse,
                            color="red",
                            linestyle="--",
                            linewidth=2,
                            label="GRASP Avg"
                        )
                        plt.xlabel("Epoch")
                        plt.ylabel("DRO k-space MSE (sim)")
                        plt.title("k-space MSE")
                        plt.grid(True)
                        plt.savefig(os.path.join(eval_dir, "eval_dc_mses.png"))
                        plt.close()


                        plt.figure()
                        plt.plot(epoch_labels, eval_dc_maes)
                        plt.axhline(
                            y=avg_grasp_dc_mae,
                            color="red",
                            linestyle="--",
                            linewidth=2,
                            label="GRASP Avg"
                        )
                        plt.xlabel("Epoch")
                        plt.ylabel("DRO k-space MAE (sim)")
                        plt.title("k-space MAE")
                        plt.grid(True)
                        plt.savefig(os.path.join(eval_dir, "eval_dc_maes.png"))
                        plt.close()

                        plt.figure()
                        plt.plot(epoch_labels, eval_raw_dc_mses)
                        plt.axhline(
                            y=avg_grasp_raw_dc_mse,
                            color="red",
                            linestyle="--",
                            linewidth=2,
                            label="GRASP Avg"
                        )
                        plt.xlabel("Epoch")
                        plt.ylabel("Non-DRO k-space MSE")
                        plt.title("k-space MSE")
                        plt.grid(True)
                        plt.savefig(os.path.join(eval_dir, "eval_raw_dc_mses.png"))
                        plt.close()


                if global_rank == 0 or not config['training']['multigpu']:
                    # Print epoch summary
                    print(
                        f"Epoch {epoch}: Training MC Loss: {epoch_train_mc_loss:.6f}, Validation MC Loss: {epoch_val_mc_loss:.6f}"
                    )
                    if use_ei_loss:
                        print(
                            f"Epoch {epoch}: Training EI Loss: {epoch_train_ei_loss:.6f}, Validation EI Loss: {epoch_val_ei_loss:.6f}"
                        )

                    if use_adj_loss:
                        print(
                            f"Epoch {epoch}: Training Adj Loss: {epoch_train_adj_loss:.6f}, Validation Adj Loss: {epoch_val_adj_loss:.6f}"
                        )
                    print(f"--- Evaluation Metrics: Epoch {epoch} ---")
                    print(f"Recon SSIM: {epoch_eval_ssim:.4f} ± {np.std(epoch_eval_ssims):.4f}")
                    print(f"Recon PSNR: {epoch_eval_psnr:.4f} ± {np.std(epoch_eval_psnrs):.4f}")
                    print(f"Recon MSE: {epoch_eval_mse:.4f} ± {np.std(epoch_eval_mses):.4f}")
                    print(f"Recon LPIPS: {epoch_eval_lpips:.4f} ± {np.std(epoch_eval_lpipses):.4f}")
                    print(f"Recon DC MSE: {epoch_eval_dc_mse:.4f} ± {np.std(epoch_eval_dc_mses):.4f}")
                    print(f"Recon DC MAE: {epoch_eval_dc_mae:.4f} ± {np.std(epoch_eval_dc_maes):.4f}")
                    print(f"Recon Raw DC MSE: {epoch_eval_raw_dc_mse:.4f} ± {np.std(epoch_eval_raw_dc_mses):.4f}")
                    print(f"Recon Raw DC MAE: {epoch_eval_raw_dc_mae:.4f} ± {np.std(epoch_eval_raw_dc_maes):.4f}")
                    print(f"Recon Enhancement Curve Correlation: {epoch_eval_curve_corr:.4f} ± {np.std(epoch_eval_curve_corrs):.4f}")
                    if not _grasp_baseline_ready():
                        raise RuntimeError(
                            f"Epoch {epoch}: GRASP baseline metrics unavailable; cannot log baseline."
                        )
                    print(f"GRASP SSIM: {avg_grasp_ssim:.4f} ± {_std_or_zero(grasp_ssims):.4f}")
                    print(f"GRASP PSNR: {avg_grasp_psnr:.4f} ± {_std_or_zero(grasp_psnrs):.4f}")
                    print(f"GRASP MSE: {avg_grasp_mse:.4f} ± {_std_or_zero(grasp_mses):.4f}")
                    print(f"GRASP LPIPS: {avg_grasp_lpips:.4f} ± {_std_or_zero(grasp_lpipses):.4f}")
                    print(f"GRASP DC MSE: {avg_grasp_dc_mse:.6f} ± {_std_or_zero(grasp_dc_mses):.4f}")
                    print(f"GRASP DC MAE: {avg_grasp_dc_mae:.6f} ± {_std_or_zero(grasp_dc_maes):.4f}")
                    print(f"GRASP Raw DC MSE: {avg_grasp_raw_dc_mse:.6f} ± {_std_or_zero(raw_grasp_dc_mses):.4f}")
                    print(f"GRASP Raw DC MAE: {avg_grasp_raw_dc_mae:.6f} ± {_std_or_zero(raw_grasp_dc_maes):.4f}")
                    print(f"GRASP Enhancement Curve Correlation: {avg_grasp_curve_corr:.6f} ± {_std_or_zero(grasp_curve_corrs):.4f}")

            # Always save the latest checkpoint after each epoch.
            if global_rank == 0 or not config['training']['multigpu']:
                train_curves, val_curves, eval_curves = _build_checkpoint_curves()
                model_save_path = os.path.join(output_dir, f'{exp_name}_model.pth')
                save_checkpoint(
                    model,
                    optimizer,
                    epoch + 1,
                    train_curves,
                    val_curves,
                    eval_curves,
                    ei_target_weight_effective,
                    step0_train_ei_loss,
                    epoch_train_mc_loss,
                    avg_grasp_ssim,
                    avg_grasp_psnr,
                    avg_grasp_mse,
                    avg_grasp_lpips,
                    avg_grasp_dc_mse,
                    avg_grasp_dc_mae,
                    avg_grasp_curve_corr,
                    avg_grasp_raw_dc_mae,
                    avg_grasp_raw_dc_mse,
                    model_save_path,
                )
                print(f"[Checkpoint] Latest model saved to {model_save_path}")

            transition_tags = loss_transition_checkpoints.get(epoch, [])
            if transition_tags:
                if global_rank == 0 or not config['training']['multigpu']:
                    train_curves, val_curves, eval_curves = _build_checkpoint_curves()
                    for tag in transition_tags:
                        transition_checkpoint_path = os.path.join(
                            output_dir, f"{exp_name}_{tag}_epoch{epoch}.pth"
                        )
                        save_checkpoint(
                            model,
                            optimizer,
                            epoch + 1,
                            train_curves,
                            val_curves,
                            eval_curves,
                            ei_target_weight_effective,
                            step0_train_ei_loss,
                            epoch_train_mc_loss,
                            avg_grasp_ssim,
                            avg_grasp_psnr,
                            avg_grasp_mse,
                            avg_grasp_lpips,
                            avg_grasp_dc_mse,
                            avg_grasp_dc_mae,
                            avg_grasp_curve_corr,
                            avg_grasp_raw_dc_mae,
                            avg_grasp_raw_dc_mse,
                            transition_checkpoint_path,
                        )
                        print(
                            f"[Checkpoint] Transition checkpoint ({tag}) saved to {transition_checkpoint_path}"
                        )

            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
                epoch_peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                attempt_peak_mem_gb = max(attempt_peak_mem_gb, epoch_peak_gb)





    # Save the model at the end of training
    if global_rank == 0 or not config['training']['multigpu']:
        train_curves, val_curves, eval_curves = _build_checkpoint_curves()
        model_save_path = os.path.join(output_dir, f'{exp_name}_model.pth')
        save_checkpoint(model, optimizer, epochs + 1, train_curves, val_curves, eval_curves, ei_target_weight_effective, step0_train_ei_loss, epoch_train_mc_loss, avg_grasp_ssim, avg_grasp_psnr, avg_grasp_mse, avg_grasp_lpips, avg_grasp_dc_mse, avg_grasp_dc_mae, avg_grasp_curve_corr, avg_grasp_raw_dc_mae, avg_grasp_raw_dc_mse, model_save_path)
        print(f'Model saved to {model_save_path}')


        # save final evaluation metrics
        if epoch > eval_frequency:
            
            metrics_path = os.path.join(eval_dir, "eval_metrics.csv")

            with open(metrics_path, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['Recon', 'SSIM', 'PSNR', 'MSE', 'LPIPS', 'DC MSE', 'DC MAE', 'EC Correlation'])
                csvwriter.writerow(['DL', 
                                f'{epoch_eval_ssim:.4f} +- {np.std(epoch_eval_ssims):.4f}', 
                                f'{epoch_eval_psnr:.4f} +- {np.std(epoch_eval_psnrs):.4f}', 
                                f'{epoch_eval_mse:.4f} +- {np.std(epoch_eval_mses):.4f}',
                                f'{epoch_eval_lpips:.4f} +- {np.std(epoch_eval_lpipses):.4f}',  
                                f'{epoch_eval_dc_mse:.4f} +- {np.std(epoch_eval_dc_mses):.4f}', 
                                f'{epoch_eval_dc_mae:.4f} +- {np.std(epoch_eval_dc_maes):.4f}', 
                                f"{epoch_eval_raw_dc_mse:.4f} +- {np.std(epoch_eval_raw_dc_mses):.4f}", 
                                f"{epoch_eval_raw_dc_mae:.4f} +- {np.std(epoch_eval_raw_dc_maes):.4f}", 
                                f'{epoch_eval_curve_corr:.4f} +- {np.std(epoch_eval_curve_corrs):.4f}'])
                csvwriter.writerow(['GRASP', 
                                f'{avg_grasp_ssim:.4f} +- {np.std(grasp_ssims):.4f}', 
                                f'{avg_grasp_psnr:.4f} +- {np.std(grasp_psnrs):.4f}', 
                                f'{avg_grasp_mse:.4f} +- {np.std(grasp_mses):.4f}', 
                                f'{avg_grasp_lpips:.4f} +- {np.std(grasp_lpipses):.4f}', 
                                f'{avg_grasp_dc_mse:.4f} +- {np.std(grasp_dc_mses):.4f}', 
                                f'{avg_grasp_dc_mae:.4f} +- {np.std(grasp_dc_maes):.4f}', 
                                f"{avg_grasp_raw_dc_mse:.6f} +- {np.std(raw_grasp_dc_mses):.4f}",
                                f"{avg_grasp_raw_dc_mae:.6f} +- {np.std(raw_grasp_dc_maes):.4f}", 
                                f'{avg_grasp_curve_corr:.4f} +- {np.std(grasp_curve_corrs):.4f}'])



        # # EVALUATE WITH VARIABLE SPOKES PER FRAME


        # # --- Stress Test Plan ---
        # # Designed to push the limits with very few spokes per frame.
        # # This has a different (lower) total spoke budget.

        # STRESS_TEST_PLAN = [
        #     {
        #         "spokes_per_frame": 2,
        #         "num_frames": 144, # 2 * 144 = 288 total spokes
        #         "description": "Stress test: max temporal points, 2 spokes"
        #     },
        #     {
        #         "spokes_per_frame": 4,
        #         "num_frames": 72, # 4 * 72 = 288 total spokes
        #         "description": "Stress test: max temporal points, 4 spokes"
        #     },
        #     {
        #         "spokes_per_frame": 8,
        #         "num_frames": 36, # 8 * 36 = 288 total spokes
        #         "description": "High temporal resolution"
        #     },
        #     {
        #         "spokes_per_frame": 16,
        #         "num_frames": 18, # 16 * 18 = 288 total spokes
        #         "description": "High temporal resolution"
        #     },
        #     {
        #         "spokes_per_frame": 24,
        #         "num_frames": 12, # 24 * 12 = 288 total spokes
        #         "description": "Good temporal resolution"
        #     },
        #     {
        #         "spokes_per_frame": 36,
        #         "num_frames": 8, # 36 * 8 = 288 total spokes
        #         "description": "Standard temporal resolution"
        #     }
        # ]


        # eval_spf_dataset = SimulatedSPFDataset(
        #     root_dir=config["evaluation"]["simulated_dataset_path"], 
        #     raw_kspace_path=data_dir,
        #     model_type=model_type, 
        #     patient_ids=val_dro_patient_ids,
        #     dataset_key=config["data"]["dataset_key"],
        #     grasp_slice_idx=raw_grasp_slice_idx
        #     )


        # eval_spf_loader = DataLoader(
        #     eval_spf_dataset,
        #     batch_size=config["dataloader"]["batch_size"],
        #     shuffle=False,
        #     num_workers=config["dataloader"]["num_workers"],
        # )




        # with torch.no_grad():

        #     spf_recon_ssim = {}
        #     spf_recon_psnr = {}
        #     spf_recon_mse = {}
        #     spf_recon_lpips = {}
        #     spf_recon_dc_mse = {}
        #     spf_recon_dc_mae = {}
        #     spf_recon_corr = {}
        #     spf_grasp_ssim = {}
        #     spf_grasp_psnr = {}
        #     spf_grasp_mse = {}
        #     spf_grasp_lpips = {}
        #     spf_grasp_dc_mse = {}
        #     spf_grasp_dc_mae = {}
        #     spf_grasp_corr = {}
        #     spf_raw_dc_mse = {}
        #     spf_raw_dc_mae = {}
        #     spf_raw_grasp_dc_mse = {}
        #     spf_raw_grasp_dc_mae = {}

        #     print("--- Running Stress Test Evaluation (Budget: 176 spokes) ---")
        #     for eval_config in STRESS_TEST_PLAN:

        #         stress_test_ssims = []
        #         stress_test_psnrs = []
        #         stress_test_mses = []
        #         stress_test_lpipses = []
        #         stress_test_dc_mses = []
        #         stress_test_dc_maes = []
        #         stress_test_corrs = []
        #         stress_test_grasp_ssims = []
        #         stress_test_grasp_psnrs = []
        #         stress_test_grasp_mses = []
        #         stress_test_grasp_lpipses = []
        #         stress_test_grasp_dc_mses = []
        #         stress_test_grasp_dc_maes = []
        #         stress_test_grasp_corrs = []
        #         stress_test_raw_dc_mses = []
        #         stress_test_raw_dc_maes = []
        #         stress_test_raw_grasp_dc_mses = []
        #         stress_test_raw_grasp_dc_maes = []


        #         spokes = eval_config["spokes_per_frame"]
        #         num_frames = eval_config["num_frames"]

        #         eval_spf_dataset.spokes_per_frame = spokes
        #         eval_spf_dataset.num_frames = num_frames
        #         eval_spf_dataset._update_sample_paths()


        #         for csmap, ground_truth, dro_grasp_img, mask, grasp_path, raw_kspace, raw_csmaps, raw_grasp_img in tqdm(eval_spf_loader, desc="Variable Spokes Per Frame Evaluation"):


        #             csmap = csmap.squeeze(0).to(device)   # Remove batch dim
        #             ground_truth = ground_truth.to(device) # Shape: (1, 2, T, H, W)

        #             dro_grasp_img = dro_grasp_img.to(device) # Shape: (1, 2, H, T, W)

        #             raw_kspace = raw_kspace.squeeze(0).to(device) # Remove batch dim
        #             raw_grasp_img = raw_grasp_img.to(device) # Shape: (1, 2, H, T, W)
        #             raw_csmaps = raw_csmaps.squeeze(0).to(device)   # Remove batch dim

        #             # SIMULATE KSPACE
        #             ktraj, dcomp, nufft_ob, adjnufft_ob = prep_nufft(N_samples, spokes, num_frames)
        #             physics = MCNUFFT(nufft_ob.to(device), adjnufft_ob.to(device), ktraj.to(device), dcomp.to(device))

        #             sim_kspace = physics(False, ground_truth, csmap)

        #             kspace = sim_kspace.squeeze(0).to(device) # Remove batch dim

        #             # calculate acceleration factor
        #             acceleration = torch.tensor([N_full / int(spokes)], dtype=torch.float, device=device)

        #             if config['model']['encode_acceleration']:
        #                 acceleration_encoding = acceleration
        #             else: 
        #                 acceleration_encoding = None

        #             if config['model']['encode_time_index'] == False:
        #                 start_timepoint_index = None
        #             else:
        #                 start_timepoint_index = torch.tensor([0], dtype=torch.float, device=device)


        #             # check if GRASP image exists or if we need to perform GRASP recon
        #             if type(dro_grasp_img) is int or len(dro_grasp_img.shape) == 1:
        #                 print(f"No GRASP file found, performing reconstruction with {spokes} spokes/frame and {num_frames} frames.")

        #                 dro_grasp_img = GRASPRecon(csmap, sim_kspace, spokes, num_frames, grasp_path[0])

        #                 grasp_recon_torch = torch.from_numpy(dro_grasp_img).permute(2, 0, 1) # T, H, W
        #                 grasp_recon_torch = torch.stack([grasp_recon_torch.real, grasp_recon_torch.imag], dim=0)

        #                 dro_grasp_img = torch.flip(grasp_recon_torch, dims=[-3])
        #                 dro_grasp_img = torch.rot90(dro_grasp_img, k=3, dims=[-3,-1]).unsqueeze(0)

        #             dro_grasp_img = dro_grasp_img.to(device)

        #             if num_frames > eval_chunk_size:
        #                 print("Performing sliding window eval...")
        #                 x_recon, _ = sliding_window_inference(H, W, num_frames, ktraj, dcomp, nufft_ob, adjnufft_ob, eval_chunk_size, eval_chunk_overlap, kspace, csmap, acceleration_encoding, start_timepoint_index, model, epoch=None, device=device)  
        #                 raw_x_recon, _ = sliding_window_inference(H, W, num_frames, ktraj, dcomp, nufft_ob, adjnufft_ob, eval_chunk_size, eval_chunk_overlap, raw_kspace, raw_csmaps, acceleration_encoding, start_timepoint_index, model, epoch="val0", device=device)  
        #             else:
        #                 x_recon, *_ = model(
        #                     kspace.to(device), physics, csmap, acceleration_encoding, start_timepoint_index, epoch=None, norm=config['model']['norm']
        #                 )
        #                 raw_x_recon, adj_loss, *_ = model(
        #                 raw_kspace.to(device), physics, raw_csmaps, acceleration_encoding, start_timepoint_index, epoch=None, norm=config['model']['norm']
        #                 )

        #             ground_truth = torch.stack([ground_truth.real, ground_truth.imag], dim=1)
        #             ground_truth = rearrange(ground_truth, 'b i h w t -> b i t h w')

        #             # fix orientation of raw k-space recon
        #             # raw_x_recon = torch.rot90(raw_x_recon, k=2, dims=[-3,-2])


        #             ## Evaluation
        #             ssim, psnr, mse, lpips, dc_mse, dc_mae, recon_corr, grasp_corr, _ = eval_sample(
        #                 kspace,
        #                 csmap,
        #                 ground_truth,
        #                 x_recon,
        #                 physics,
        #                 mask,
        #                 dro_grasp_img,
        #                 acceleration,
        #                 int(spokes),
        #                 eval_dir,
        #                 f"{spokes}spf",
        #                 device,
        #                 cluster=cluster,
        #                 dro_eval=True,
        #                 grasp_path=grasp_path,
        #                 rescale=config['evaluation']['rescale'],
        #                 filename_suffix=f"{spokes}spf",
        #             )
        #             stress_test_ssims.append(ssim)
        #             stress_test_psnrs.append(psnr)
        #             stress_test_mses.append(mse)
        #             stress_test_lpipses.append(lpips)
        #             stress_test_dc_mses.append(dc_mse)
        #             stress_test_dc_maes.append(dc_mae)

        #             if recon_corr is not None:
        #                 stress_test_corrs.append(recon_corr)
        #                 stress_test_grasp_corrs.append(grasp_corr)


        #             ssim_grasp, psnr_grasp, mse_grasp, lpips_grasp, dc_mse_grasp, dc_mae_grasp = eval_grasp(kspace, csmap, ground_truth, dro_grasp_img, physics, device, eval_dir, dro_eval=True)
        #             stress_test_grasp_ssims.append(ssim_grasp)
        #             stress_test_grasp_psnrs.append(psnr_grasp)
        #             stress_test_grasp_mses.append(mse_grasp)
        #             stress_test_grasp_lpipses.append(lpips_grasp)
        #             stress_test_grasp_dc_mses.append(dc_mse_grasp)
        #             stress_test_grasp_dc_maes.append(dc_mae_grasp)


        #             # raw k-space
        #             dc_mse_raw_grasp, dc_mae_raw_grasp = eval_grasp(raw_kspace, raw_csmaps, ground_truth, raw_grasp_img, physics, device, eval_dir, dro_eval=False)
        #             dc_mse_raw, dc_mae_raw, _ = eval_sample(
        #                 raw_kspace,
        #                 raw_csmaps,
        #                 ground_truth,
        #                 raw_x_recon,
        #                 physics,
        #                 mask,
        #                 raw_grasp_img,
        #                 acceleration,
        #                 int(N_spokes),
        #                 eval_dir,
        #                 label=f"{spokes}spf",
        #                 device=device,
        #                 cluster=cluster,
        #                 dro_eval=False,
        #                 grasp_path=grasp_path,
        #                 raw_slice_idx=raw_grasp_slice_idx,
        #                 rescale=config['evaluation']['rescale'],
        #                 filename_suffix=f"{spokes}spf",
        #             )

        #             stress_test_raw_grasp_dc_mses.append(dc_mse_raw_grasp)
        #             stress_test_raw_grasp_dc_maes.append(dc_mae_raw_grasp)
        #             stress_test_raw_dc_mses.append(dc_mse_raw)
        #             stress_test_raw_dc_maes.append(dc_mae_raw)




        #         spf_recon_ssim[spokes] = np.mean(stress_test_ssims)
        #         spf_recon_psnr[spokes] = np.mean(stress_test_psnrs)
        #         spf_recon_mse[spokes] = np.mean(stress_test_mses)
        #         spf_recon_lpips[spokes] = np.mean(stress_test_lpipses)
        #         spf_recon_dc_mse[spokes] = np.mean(stress_test_dc_mses)
        #         spf_recon_dc_mae[spokes] = np.mean(stress_test_dc_maes)
        #         spf_recon_corr[spokes] = np.mean(stress_test_corrs)

        #         spf_grasp_ssim[spokes] = np.mean(stress_test_grasp_ssims)
        #         spf_grasp_psnr[spokes] = np.mean(stress_test_grasp_psnrs)
        #         spf_grasp_mse[spokes] = np.mean(stress_test_grasp_mses)
        #         spf_grasp_lpips[spokes] = np.mean(stress_test_grasp_lpipses)
        #         spf_grasp_dc_mse[spokes] = np.mean(stress_test_grasp_dc_mses)
        #         spf_grasp_dc_mae[spokes] = np.mean(stress_test_grasp_dc_maes)
        #         spf_grasp_corr[spokes] = np.mean(stress_test_grasp_corrs)


        #         spf_raw_dc_mse[spokes] = np.mean(stress_test_raw_dc_mses)
        #         spf_raw_dc_mae[spokes] = np.mean(stress_test_raw_dc_maes)
        #         spf_raw_grasp_dc_mse[spokes] = np.mean(stress_test_raw_grasp_dc_mses)
        #         spf_raw_grasp_dc_mae[spokes] = np.mean(stress_test_raw_grasp_dc_maes)



        #         # Save Results
        #         spf_metrics_path = os.path.join(eval_dir, "eval_metrics.csv")
        #         with open(spf_metrics_path, 'a', newline='') as csvfile:
        #             csvwriter = csv.writer(csvfile)
        #             csvwriter.writerow(['Recon', 'Spokes Per Frame', 'SSIM', 'PSNR', 'MSE', "LPIPS", 'DC MSE', 'DC MAE', 'EC Correlation'])

        #             csvwriter.writerow(['DL', spokes, 
        #             f'{np.mean(stress_test_ssims):.4f} +- {np.std(stress_test_ssims):.4f}', 
        #             f'{np.mean(stress_test_psnrs):.4f} +- {np.std(stress_test_psnrs):.4f}', 
        #             f'{np.mean(stress_test_mses):.4f} +- {np.std(stress_test_mses):.4f}', 
        #             f'{np.mean(stress_test_lpipses):.4f} +- {np.std(stress_test_lpipses):.4f}', 
        #             f'{np.mean(stress_test_dc_mses):.4f} +- {np.std(stress_test_dc_mses):.4f}',
        #             f'{np.mean(stress_test_dc_maes):.4f} +- {np.std(stress_test_dc_maes):.4f}',
        #             f'{np.mean(stress_test_raw_dc_mses):.4f} +- {np.std(stress_test_raw_dc_mses):.4f}',
        #             f'{np.mean(stress_test_raw_dc_maes):.4f} +- {np.std(stress_test_raw_dc_maes):.4f}',
        #             f'{np.mean(stress_test_corrs):.4f} +- {np.std(stress_test_corrs):.4f}'
        #             ])

        #             csvwriter.writerow(['GRASP', spokes, 
        #             f'{np.mean(stress_test_grasp_ssims):.4f} +- {np.std(stress_test_grasp_ssims):.4f}', 
        #             f'{np.mean(stress_test_grasp_psnrs):.4f} +- {np.std(stress_test_grasp_psnrs):.4f}', 
        #             f'{np.mean(stress_test_grasp_mses):.4f} +- {np.std(stress_test_grasp_mses):.4f}', 
        #             f'{np.mean(stress_test_grasp_lpipses):.4f} +- {np.std(stress_test_grasp_lpipses):.4f}', 
        #             f'{np.mean(stress_test_grasp_dc_mses):.4f} +- {np.std(stress_test_grasp_dc_mses):.4f}',
        #             f'{np.mean(stress_test_grasp_dc_maes):.4f} +- {np.std(stress_test_grasp_dc_maes):.4f}',
        #             f'{np.mean(stress_test_raw_grasp_dc_mses):.4f} +- {np.std(stress_test_raw_grasp_dc_mses):.4f}',
        #             f'{np.mean(stress_test_raw_grasp_dc_maes):.4f} +- {np.std(stress_test_raw_grasp_dc_maes):.4f}',
        #             f'{np.mean(stress_test_grasp_corrs):.4f} +- {np.std(stress_test_grasp_corrs):.4f}',
        #             ])


            

        # # plot variable spokes/frame evaluation metrics in one figure
        # sns.set_style("whitegrid")
        # fig, axes = plt.subplots(2, 3, figsize=(18, 10))


        # sns.lineplot(x=list(spf_recon_ssim.keys()), 
        #             y=list(spf_recon_ssim.values()), 
        #             label="DL Recon", 
        #             marker='o',
        #             ax=axes[0, 0])

        # sns.lineplot(x=list(spf_grasp_ssim.keys()), 
        #             y=list(spf_grasp_ssim.values()), 
        #             label="Standard Recon", 
        #             marker='o',
        #             ax=axes[0, 0])

        # axes[0, 0].set_title("DRO Evaluation SSIM vs Spokes/Frame")
        # axes[0, 0].set_xlabel("Spokes per Frame")
        # axes[0, 0].set_ylabel("SSIM")


        # sns.lineplot(x=list(spf_recon_psnr.keys()), 
        #             y=list(spf_recon_psnr.values()), 
        #             label="DL Recon", 
        #             marker='o',
        #             ax=axes[0, 1])

        # sns.lineplot(x=list(spf_grasp_psnr.keys()), 
        #             y=list(spf_grasp_psnr.values()), 
        #             label="Standard Recon", 
        #             marker='o',
        #             ax=axes[0, 1])
        # axes[0, 1].set_title("DRO Evaluation PSNR vs Spokes/Frame")
        # axes[0, 1].set_xlabel("Spokes per Frame")
        # axes[0, 1].set_ylabel("PSNR")


        # sns.lineplot(x=list(spf_recon_mse.keys()), 
        #             y=list(spf_recon_mse.values()), 
        #             label="DL Recon", 
        #             marker='o',
        #             ax=axes[0, 2])

        # sns.lineplot(x=list(spf_grasp_mse.keys()), 
        #             y=list(spf_grasp_mse.values()), 
        #             label="Standard Recon", 
        #             marker='o',
        #             ax=axes[0, 2])
        # axes[0, 2].set_title("DRO Evaluation Image MSE vs Spokes/Frame")
        # axes[0, 2].set_xlabel("Spokes per Frame")
        # axes[0, 2].set_ylabel("MSE")


        # sns.lineplot(x=list(spf_recon_lpips.keys()), 
        #             y=list(spf_recon_lpips.values()), 
        #             label="DL Recon", 
        #             marker='o',
        #             ax=axes[1, 0])

        # sns.lineplot(x=list(spf_grasp_lpips.keys()), 
        #             y=list(spf_grasp_lpips.values()), 
        #             label="Standard Recon", 
        #             marker='o',
        #             ax=axes[1, 0])
        # axes[1, 0].set_title("Evaluation LPIPS vs Spokes/Frame")
        # axes[1, 0].set_xlabel("Spokes per Frame")
        # axes[1, 0].set_ylabel("LPIPS")





        # sns.lineplot(x=list(spf_raw_dc_mse.keys()), 
        #     y=list(spf_raw_dc_mse.values()), 
        #     label="DL Recon", 
        #     marker='o',
        #     ax=axes[1, 1])

        # sns.lineplot(x=list(spf_raw_grasp_dc_mse.keys()), 
        #             y=list(spf_raw_grasp_dc_mse.values()), 
        #             label="Standard Recon", 
        #             marker='o',
        #             ax=axes[1, 1])
        # axes[1, 1].set_title("Non-DRO Evaluation Raw k-space MAE vs Spokes/Frame")
        # axes[1, 1].set_xlabel("Spokes per Frame")
        # axes[1, 1].set_ylabel("MAE")

        # # sns.lineplot(x=list(spf_recon_dc_mae.keys()), 
        # #             y=list(spf_recon_dc_mae.values()), 
        # #             label="DL Recon", 
        # #             marker='o',
        # #             ax=axes[1, 1])

        # # sns.lineplot(x=list(spf_grasp_dc_mae.keys()), 
        # #             y=list(spf_grasp_dc_mae.values()), 
        # #             label="Standard Recon", 
        # #             marker='o',
        # #             ax=axes[1, 1])
        # # axes[1, 1].set_title("DRO Evaluation Simulated k-space MAE vs Spokes/Frame")
        # # axes[1, 1].set_xlabel("Spokes per Frame")
        # # axes[1, 1].set_ylabel("MAE")

        # sns.lineplot(x=list(spf_recon_corr.keys()), 
        #             y=list(spf_recon_corr.values()), 
        #             label="DL Recon", 
        #             marker='o',
        #             ax=axes[1, 2])

        # sns.lineplot(x=list(spf_grasp_corr.keys()), 
        #             y=list(spf_grasp_corr.values()), 
        #             label="Standard Recon", 
        #             marker='o',
        #             ax=axes[1, 2])
        # axes[1, 2].set_title("DRO Tumor Enhancement Curve Correlation vs Spokes/Frame")
        # axes[1, 2].set_xlabel("Spokes per Frame")
        # axes[1, 2].set_ylabel("Pearson Correlation Coefficient")

        # plt.tight_layout()
        # plt.savefig(os.path.join(output_dir, "spf_eval_metrics.png"))
        # plt.close()

    if global_rank == 0 or not config['training']['multigpu']:
        _finalize_run_state("completed")
        writer.close()

    cleanup()


if __name__ == '__main__':
    main()
