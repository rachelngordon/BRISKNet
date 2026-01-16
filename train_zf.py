import argparse
import json
import os
import matplotlib.pyplot as plt
import torch
import warnings
import yaml
from dataloader import ZFSliceDataset, SimulatedDataset, SimulatedSPFDataset
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from transform import VideoRotate, VideoDiffeo, SubsampleTime, MonophasicTimeWarp, TemporalNoise, TimeReverse
from ei import EILoss
from mc import MCLoss
from lsfpnet_encoding import LSFPNet, ArtifactRemovalLSFPNet
from radial_lsfp import MCNUFFT
from utils import prep_nufft, log_gradient_stats, log_lsfpnet_component_grads, plot_enhancement_curve, get_cosine_ei_weight, plot_reconstruction_sample, get_git_commit, save_checkpoint, load_checkpoint, to_torch_complex, GRASPRecon, sliding_window_inference, set_seed, save_csmap_png
from eval import eval_grasp, eval_sample
import csv
import math
import random
import time 
import seaborn as sns
from loss_metrics import LPIPSVideoMetric, SSIMVideoMetric
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import h5py
import signal
from torch.utils.tensorboard import SummaryWriter
from cluster_paths import apply_cluster_paths
from contextlib import nullcontext


def setup():
    """Initializes the distributed process group."""
    dist.init_process_group("nccl")

def cleanup():
    dist.destroy_process_group()

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
        help="Whether to load from a checkpoint",
    )
    args = parser.parse_args()

    # print experiment name and git commit
    exp_name = args.exp_name

    # NOTE: need to change for running on Randi
    config_dir = '/net/projects2/annawoodard/rachelgordon/experiments'
    # config_dir = 'output'


    # Load the configuration file
    if args.from_checkpoint == True:
        with open(os.path.join(config_dir, f"{exp_name}/config.yaml"), "r") as file:
            config = yaml.safe_load(file)

        with open(args.config, "r") as file:
            new_config = yaml.safe_load(file)
        
        epochs = new_config['training']["epochs"]
    else:
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)

        epochs = config['training']["epochs"]

    config = apply_cluster_paths(config)
    if args.from_checkpoint:
        new_config = apply_cluster_paths(new_config)

    use_edge_time_index_sampling = config.get('training', {}).get('edge_time_index_sampling', False)

    # create output directories
    output_dir = os.path.join(config["experiment"]["output_dir"], exp_name)
    eval_dir = os.path.join(output_dir, "eval_results")
    block_dir = os.path.join(output_dir, "block_outputs")
    ec_dir = os.path.join(output_dir, 'enhancement_curves')

    attempt_start_time = time.time()
    attempt_peak_mem_gb = 0.0
    run_state_path = os.path.join(output_dir, "run_state.json")
    run_state = None
    state_written = False
    attempt_idx = None

        

    if config['training']['multigpu']:
        setup()

        # Get rank and world_size from the distributed package AFTER setup
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Get the local rank from the environment variable
        local_rank = int(os.environ["LOCAL_RANK"])

        # Set the device for this process
        torch.cuda.set_device(local_rank)
        device = local_rank

        if global_rank == 0:
            print(f"Starting distributed training with {world_size} GPUs.")
        print(f"  - [Rank {global_rank}] -> Using device cuda:{device}")

    else:
        global_rank = 0
        device = torch.device(config["training"]["device"])

    # AMP/mixed precision config (optional)
    amp_cfg = config.get("training", {}).get("amp", {})
    amp_enabled = bool(amp_cfg.get("enabled", False)) and torch.cuda.is_available() and str(device).startswith("cuda")
    amp_dtype_str = str(amp_cfg.get("dtype", "bf16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_str in ("bf16", "bfloat16") else torch.float16
    use_scaler = bool(amp_cfg.get("use_grad_scaler", amp_dtype == torch.float16))
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    def amp_autocast():
        return torch.amp.autocast(device_type="cuda", dtype=amp_dtype) if amp_enabled else nullcontext()

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



    if global_rank == 0 or not config['training']['multigpu']:
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        os.makedirs(block_dir, exist_ok=True)
        os.makedirs(ec_dir, exist_ok=True)

        run_state = _load_run_state(run_state_path)
        attempt_idx = run_state.get("attempt_count", 0) + 1

        # Initialize TensorBoard SummaryWriter
        log_dir = os.path.join(output_dir, 'logs')
        writer = SummaryWriter(log_dir)



        # Save the configuration file
        if args.from_checkpoint == False:
            with open(os.path.join(output_dir, 'config.yaml'), 'w') as file:
                yaml.dump(config, file)

    def _finalize_run_state(reason):
        nonlocal attempt_peak_mem_gb, state_written, run_state
        if state_written:
            return
        if global_rank != 0 and config['training']['multigpu']:
            state_written = True
            return
        if run_state is None or attempt_idx is None:
            return

        wall_time_sec = time.time() - attempt_start_time
        cumulative_wall_time_sec = run_state.get("cumulative_wall_time_sec", 0.0) + wall_time_sec
        max_peak_mem_gb = max(run_state.get("max_peak_mem_gb", 0.0), attempt_peak_mem_gb)

        run_state["attempt_count"] = attempt_idx
        run_state["cumulative_wall_time_sec"] = cumulative_wall_time_sec
        run_state["max_peak_mem_gb"] = max_peak_mem_gb
        run_state.setdefault("attempt_history", []).append(
            {
                "attempt": attempt_idx,
                "wall_time_sec": wall_time_sec,
                "peak_mem_gb": attempt_peak_mem_gb,
                "reason": reason,
            }
        )

        _save_run_state(run_state_path, run_state)
        state_written = True

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


    # load params
    split_file = config["data"]["split_file"]

    data_dir = config["data"]["root_dir"]

    batch_size = config["dataloader"]["batch_size"]
    max_subjects = config["dataloader"]["max_subjects"]

    initial_lambdas = {'lambda_L': config['model']['lambda_L'], 
                    'lambda_S': config['model']['lambda_S'], 
                    'lambda_spatial_L': config['model']['lambda_spatial_L'],
                    'lambda_spatial_S': config['model']['lambda_spatial_S'],
                    'gamma': config['model']['gamma'],
                    'lambda_step': config['model']['lambda_step']}

    mc_loss_weight = config["model"]["losses"]["mc_loss"]["weight"]
    adj_loss_weight = config["model"]["losses"]["adj_loss"]["weight"]

    use_ei_loss = config["model"]["losses"]["use_ei_loss"]
    target_w_ei = config["model"]["losses"]["ei_loss"]["weight"]
    warmup = config["model"]["losses"]["ei_loss"]["warmup"]
    duration = config["model"]["losses"]["ei_loss"]["duration"]

    save_interval = config["training"]["save_interval"]
    plot_interval = config["training"]["plot_interval"]

    model_type = config["model"]["name"]

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

    raw_grasp_slice_idx = config.get("evaluation", {}).get("raw_grasp_slice_idx", 95)

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
            N_time=N_time,
            N_coils=N_coils,
            spf_aug=config['data']['spf_aug'],
            spokes_per_frame=train_spokes_per_frame,
            weight_accelerations=config['data']['weight_accelerations'],
            initial_spokes_range=initial_train_spokes_range,
            cluster=cluster,
            flip_kspace=flip_kspace
        )


    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(12)

    if config['training']['multigpu']:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["dataloader"]["batch_size"],
            sampler=train_sampler,
            num_workers=config["dataloader"]["num_workers"],
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["dataloader"]["batch_size"],
            shuffle=config["dataloader"]["shuffle"],
            num_workers=config["dataloader"]["num_workers"],
            pin_memory=True,
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

    # define physics object for evaluation
    eval_ktraj, eval_dcomp, eval_nufft_ob, eval_adjnufft_ob = prep_nufft(N_samples, N_spokes_eval, N_time_eval)
    eval_ktraj = eval_ktraj.to(device)
    eval_dcomp = eval_dcomp.to(device)
    eval_nufft_ob = eval_nufft_ob.to(device)
    eval_adjnufft_ob = eval_adjnufft_ob.to(device)

    eval_physics = MCNUFFT(eval_nufft_ob, eval_adjnufft_ob, eval_ktraj, eval_dcomp)


    val_dro_dataset = SimulatedDataset(
        root_dir=config["evaluation"]["simulated_dataset_path"], 
        raw_kspace_path=data_dir,
        model_type=model_type, 
        patient_ids=val_dro_patient_ids,
        dataset_key=config["data"]["dataset_key"],
        spokes_per_frame=N_spokes_eval,
        num_frames=N_time_eval,
        grasp_slice_idx=raw_grasp_slice_idx)


    val_dro_loader = DataLoader(
        val_dro_dataset,
        batch_size=config["dataloader"]["batch_size"],
        shuffle=False,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=True,
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

        print("val_csmap shape: ", val_csmap.shape)
        print("val_raw_csmaps shape: ", val_raw_csmaps.shape)
        print("val_csmap dtype: ", val_csmap.dtype)
        print("val_raw_csmaps dtype: ", val_raw_csmaps.dtype)
        # print("val_csmap norm check: ", np.sum(np.abs(val_csmap[:, :, :, 100, 100])**2))
        # print("val_raw_csmaps norm check: ", np.sum(np.abs(val_raw_csmaps[:, :, :, 100, 100])**2))

        save_csmap_png(val_csmap.squeeze(0), csmap_dir, "val_dro", max_coils=max_coils)
        save_csmap_png(val_raw_csmaps.squeeze(0), csmap_dir, "val_raw", max_coils=max_coils)


        # --- ensure S has shape (C, H, W) ---
        S_dro = val_csmap.squeeze()         # (C,H,W) if val_csmap was (1,C,H,W)
        S_raw = val_raw_csmaps.squeeze()    # (C,H,W) if val_raw_csmaps was (1,C,H,W)

        # --- 1) Sum of magnitudes across coils: abs(S).sum(axis=0) ---
        dro_sumabs = np.abs(S_dro).sum(axis=0)   # (H,W)
        raw_sumabs = np.abs(S_raw).sum(axis=0)   # (H,W)

        # --- 2) (Optional but very useful) RSS magnitude across coils: sqrt(sum |S|^2) ---
        dro_rss = np.sqrt((np.abs(S_dro) ** 2).sum(axis=0))
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
    kernel_size_L = config["model"].get("kernel_size_L", 3)
    kernel_size_S = config["model"].get("kernel_size_S", 3)

    lsfp_backbone = LSFPNet(LayerNo=config["model"]["num_layers"], 
                            lambdas=initial_lambdas, 
                            channels=config['model']['channels'],
                            style_dim=config['model']['style_dim'],
                            svd_mode=config['model']['svd_mode'],
                            use_lowk_dc=config['model']['use_lowk_dc'],
                            lowk_frac=config['model']['lowk_frac'],
                            lowk_alpha=config['model']['lowk_alpha'],
                            film_bounded=config['model']['film_bounded'],
                            film_gain=config['model']['film_gain'],
                            film_identity_init=config['model']['film_identity_init'],
                            svd_noise_std=config['model']['svd_noise_std'],
                            film_L=config['model']['film_L'],
                            kernel_size_L=kernel_size_L,
                            kernel_size_S=kernel_size_S,
                            activation_checkpointing=config["model"].get("activation_checkpointing", False),
                            checkpoint_use_reentrant=config["model"].get("checkpoint_use_reentrant", False),
                            )

    if config['model']['encode_acceleration'] and config['model']['encode_time_index']:
        model = ArtifactRemovalLSFPNet(lsfp_backbone, block_dir, channels=2).to(device)
    else:
        model = ArtifactRemovalLSFPNet(lsfp_backbone, block_dir, channels=1).to(device)

    if config['training']['multigpu']:
        find_unused = config["training"].get("ddp_find_unused_parameters", True)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=find_unused)
        if config["training"].get("ddp_set_static_graph", False):
            model._set_static_graph()

    if global_rank == 0 or not config['training']['multigpu']:
        if run_state is not None and ("total_params" not in run_state or "trainable_params" not in run_state):
            total_params, trainable_params = _get_param_counts(model)
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
    if args.from_checkpoint == True:
        # if global_rank == 0 or config['training']['multigpu'] == False:
        checkpoint_file = os.path.join(output_dir, f'{exp_name}_model.pth')
        model, optimizer, start_epoch, target_w_ei, step0_train_ei_loss, epoch_train_mc_loss, train_curves, val_curves, eval_curves, avg_grasp_ssim, avg_grasp_psnr, avg_grasp_mse, avg_grasp_lpips, avg_grasp_dc_mse, avg_grasp_dc_mae, avg_grasp_curve_corr, avg_grasp_raw_dc_mae, avg_grasp_raw_dc_mse = load_checkpoint(model, optimizer, checkpoint_file)

    else:
        start_epoch = 1
        # target_w_ei = 0.0


    # select metric for loss functions
    if config['model']['losses']['mc_loss']['metric'] == "MSE":
        mc_loss_fn = MCLoss(model_type=model_type)
    elif config['model']['losses']['mc_loss']['metric'] == "MAE":
        mc_loss_fn = MCLoss(model_type=model_type, metric=torch.nn.L1Loss())
    else:
        raise(ValueError, "Unsupported MC Loss Metric.")


    if config['model']['losses']['ei_loss']['metric'] == "LPIPS":
        ei_loss_metric = LPIPSVideoMetric(net_type='alex') 
    elif config['model']['losses']['ei_loss']['metric'] == "SSIM":
        ei_loss_metric = SSIMVideoMetric()
    else:
        ei_loss_metric = torch.nn.MSELoss()

    ei_no_grad = config['model']['losses']['ei_loss'].get("no_grad", False)
    ei_checkpoint_model = config['model']['losses']['ei_loss'].get("checkpoint_model", False)
    ei_checkpoint_mode = config['model']['losses']['ei_loss'].get("checkpoint_mode", "none")
    ei_checkpoint_use_reentrant = config['model']['losses']['ei_loss'].get("checkpoint_use_reentrant", False)


    # define EI loss transformations
    if use_ei_loss:
        rotate = VideoRotate(n_trans=1, interpolation_mode="bilinear", degrees=config['model']['losses']['ei_loss'].get("rotate_range", 180))
        diffeo = VideoDiffeo(n_trans=1, device=device)

        subsample = SubsampleTime(n_trans=1, subsample_ratio_range=(config['model']['losses']['ei_loss']['subsample_ratio_min'], config['model']['losses']['ei_loss']['subsample_ratio_max']))
        monophasic_warp = MonophasicTimeWarp(n_trans=1, warp_ratio_range=(config['model']['losses']['ei_loss']['warp_ratio_min'], config['model']['losses']['ei_loss']['warp_ratio_max']))
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



    if args.from_checkpoint:
        train_mc_losses = train_curves["train_mc_losses"]
        val_mc_losses = val_curves["val_mc_losses"]
        train_ei_losses = train_curves["train_ei_losses"]
        val_ei_losses = val_curves["val_ei_losses"]
        train_adj_losses = train_curves["train_adj_losses"]
        val_adj_losses = val_curves["val_adj_losses"]
        weighted_train_mc_losses = train_curves["weighted_train_mc_losses"]
        weighted_train_ei_losses = train_curves["weighted_train_ei_losses"]
        weighted_train_adj_losses = train_curves["weighted_train_adj_losses"]
        eval_ssims = eval_curves["eval_ssims"]
        eval_psnrs = eval_curves["eval_psnrs"]
        eval_mses = eval_curves["eval_mses"]
        eval_lpipses = eval_curves["eval_lpipses"]
        eval_dc_mses = eval_curves["eval_dc_mses"]
        eval_dc_maes = eval_curves["eval_dc_maes"]
        eval_raw_dc_mses = eval_curves.get("eval_raw_dc_mses", [])
        eval_raw_dc_maes = eval_curves.get("eval_raw_dc_maes", [])
        eval_curve_corrs = eval_curves["eval_curve_corrs"]
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
        eval_ssims = []
        eval_lpipses = []
        eval_psnrs = []
        eval_mses = []
        eval_dc_mses = []
        eval_dc_maes = []
        eval_raw_dc_mses = []
        eval_raw_dc_maes = []
        eval_curve_corrs = []


    eval_spf_curves = {}
    if args.from_checkpoint:
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


    grasp_ssims = []
    grasp_psnrs = []
    grasp_mses = []
    grasp_lpipses = []
    grasp_dc_mses = []
    grasp_dc_maes = []
    grasp_curve_corrs = []
    raw_grasp_dc_mses = []
    raw_grasp_dc_maes = []

    # Defaults so checkpointing works even if Step-0 validation is skipped
    if not args.from_checkpoint:
        avg_grasp_ssim = 0.0
        avg_grasp_psnr = 0.0
        avg_grasp_mse = 0.0
        avg_grasp_lpips = 0.0
        avg_grasp_dc_mse = 0.0
        avg_grasp_dc_mae = 0.0
        avg_grasp_curve_corr = 0.0
        avg_grasp_raw_dc_mae = 0.0
        avg_grasp_raw_dc_mse = 0.0

    lambda_Ls = []
    lambda_Ss = []
    lambda_spatial_Ls = []
    lambda_spatial_Ss = []
    gammas = []
    lambda_steps = []

    iteration_count = 0

    # Step 0: Evaluate the untrained model
    step0_do_val = config.get("debugging", {}).get("calc_step_0_val", True)
    if args.from_checkpoint == False and config['debugging']['calc_step_0'] == True:
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


        with torch.no_grad():

            # Evaluate on training data
            for measured_kspace, csmap, N_samples, N_spokes, N_time in tqdm(train_loader, desc="Step 0 Training Evaluation"):

                # prepare inputs
                measured_kspace = to_torch_complex(measured_kspace).squeeze()
                measured_kspace = rearrange(measured_kspace, 't co sp sam -> co (sp sam) t')

                # prep physics operators
                ktraj, dcomp, nufft_ob, adjnufft_ob = prep_nufft(N_samples, N_spokes, N_time)
                ktraj = ktraj.to(device)
                dcomp = dcomp.to(device)
                nufft_ob = nufft_ob.to(device)
                adjnufft_ob = adjnufft_ob.to(device)

                
                if N_time > Ng:
                    max_idx = N_time - Ng
                    random_index = sample_start_time_index(max_idx, use_edge_time_index_sampling)

                    measured_kspace = measured_kspace[..., random_index:random_index + Ng]

                    ktraj_chunk = ktraj[..., random_index:random_index + Ng]
                    dcomp_chunk = dcomp[..., random_index:random_index + Ng]


                    physics = MCNUFFT(nufft_ob, adjnufft_ob, ktraj_chunk, dcomp_chunk)

                    start_timepoint_index = torch.tensor([random_index], dtype=torch.float, device=device)

                else:
                    physics = MCNUFFT(nufft_ob, adjnufft_ob, ktraj, dcomp)

                    start_timepoint_index = torch.tensor([0], dtype=torch.float, device=device)


                csmap = csmap.to(device).to(measured_kspace.dtype)

                acceleration = torch.tensor([N_full / int(N_spokes)], dtype=torch.float, device=device)

                if config['model']['encode_acceleration']:
                    acceleration_encoding = acceleration
                else: 
                    acceleration_encoding = None

                if config['model']['encode_time_index'] == False:
                    start_timepoint_index = None

                with amp_autocast():
                    x_recon, adj_loss, lambda_L, lambda_S, lambda_spatial_L, lambda_spatial_S, gamma, lambda_step = model(
                        measured_kspace.to(device), physics, csmap, acceleration_encoding, start_timepoint_index, epoch="train0", norm=config['model']['norm']
                    )

                    # calculate losses
                    initial_train_adj_loss += adj_loss.item()

                    mc_loss = mc_loss_fn(measured_kspace.to(device), x_recon, physics, csmap)
                    initial_train_mc_loss += mc_loss.item()

                    if use_ei_loss:
                        ei_loss, t_img = ei_loss_fn(
                            x_recon, physics, model, csmap, acceleration_encoding, start_timepoint_index
                        )

                        initial_train_ei_loss += ei_loss.item()

                    
            # record losses
            step0_train_mc_loss = initial_train_mc_loss / len(train_loader)
            train_mc_losses.append(step0_train_mc_loss)

            step0_train_ei_loss = initial_train_ei_loss / len(train_loader)
            train_ei_losses.append(step0_train_ei_loss)

            step0_train_adj_loss = initial_train_adj_loss / len(train_loader)
            train_adj_losses.append(step0_train_adj_loss)


            if global_rank == 0 or not config['training']['multigpu']:
                writer.add_scalar('Loss/Train_MC', step0_train_mc_loss, 0)
                writer.add_scalar('Loss/Train_EI', step0_train_ei_loss, 0)
                writer.add_scalar('Loss/Train_Weighted_EI', 0, 0)
                writer.add_scalar('Loss/Train_Adj', step0_train_ei_loss, 0)


            lambda_Ls.append(lambda_L.item())
            lambda_Ss.append(lambda_S.item())
            lambda_spatial_Ls.append(lambda_spatial_L.item())
            lambda_spatial_Ss.append(lambda_spatial_S.item())
            gammas.append(gamma.item())
            lambda_steps.append(lambda_step.item())


            # Evaluate on validation data
            if step0_do_val:
                for dro_kspace, csmap, ground_truth, dro_grasp_img, mask, grasp_path, raw_kspace, raw_grasp_img, raw_csmaps in tqdm(val_dro_loader, desc="Step 0 Validation Evaluation"):
    
                    csmap = csmap.squeeze(0).to(device)   # Remove batch dim
                    ground_truth = ground_truth.to(device) # Shape: (1, 2, T, H, W)
    
                    dro_kspace = dro_kspace.squeeze(0).to(device) # Remove batch dim
                    dro_grasp_img = dro_grasp_img.to(device) # Shape: (1, 2, H, T, W)
    
                    raw_kspace = raw_kspace.squeeze(0).to(device) # Remove batch dim
                    raw_grasp_img = raw_grasp_img.to(device) # Shape: (1, 2, H, T, W)
                    raw_csmaps = raw_csmaps.squeeze(0).to(device)   # Remove batch dim
    
    
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
                        if N_time_eval > eval_chunk_size:
                            x_recon, adj_loss = sliding_window_inference(H, W, N_time_eval, eval_ktraj, eval_dcomp, eval_nufft_ob, eval_adjnufft_ob, eval_chunk_size, eval_chunk_overlap, dro_kspace, csmap, acceleration_encoding, start_timepoint_index, model, epoch="val0", device=device)  
                            raw_x_recon, _ = sliding_window_inference(H, W, N_time_eval, eval_ktraj, eval_dcomp, eval_nufft_ob, eval_adjnufft_ob, eval_chunk_size, eval_chunk_overlap, raw_kspace, raw_csmaps, acceleration_encoding, start_timepoint_index, model, epoch="val0", device=device)  
                        else:
                            x_recon, adj_loss, *_ = model(
                            dro_kspace.to(device), eval_physics, csmap, acceleration_encoding, start_timepoint_index, epoch="val0", norm=config['model']['norm']
                            )
                            raw_x_recon, *_ = model(
                            raw_kspace.to(device), eval_physics, raw_csmaps, acceleration_encoding, start_timepoint_index, epoch="val0", norm=config['model']['norm']
                            )
                            adj_loss = adj_loss.item()
    
                        # fix orientation of raw k-space recon
                        # raw_x_recon = torch.rot90(raw_x_recon, k=2, dims=[-3,-2])
    
                        # compute losses
                        initial_val_adj_loss += adj_loss
                        
                        mc_loss = mc_loss_fn(dro_kspace.to(device), x_recon, eval_physics, csmap)
                        initial_val_mc_loss += mc_loss.item()
    
                        if use_ei_loss:
                            ei_loss, t_img = ei_loss_fn(
                                x_recon, eval_physics, model, csmap, acceleration_encoding, start_timepoint_index
                            )
    
                            initial_val_ei_loss += ei_loss.item()
    
    
                    # calculate grasp metrics
                    if global_rank == 0 or not config['training']['multigpu']:
                        ssim_grasp, psnr_grasp, mse_grasp, lpips_grasp, dc_mse_grasp, dc_mae_grasp = eval_grasp(dro_kspace, csmap, ground_truth, dro_grasp_img, eval_physics, device, eval_dir, dro_eval=True)
                        grasp_ssims.append(ssim_grasp)
                        grasp_psnrs.append(psnr_grasp)
                        grasp_mses.append(mse_grasp)
                        grasp_lpipses.append(lpips_grasp)
                        grasp_dc_mses.append(dc_mse_grasp)
                        grasp_dc_maes.append(dc_mae_grasp)
    
                        ssim, psnr, mse, lpips, dc_mse, dc_mae, recon_corr, grasp_corr, _ = eval_sample(dro_kspace, csmap, ground_truth, x_recon, eval_physics, mask, dro_grasp_img, acceleration, int(N_spokes), eval_dir, label='val0', device=device, cluster=cluster, dro_eval=True, grasp_path=grasp_path, rescale=config['evaluation']['rescale'])
                        initial_eval_ssims.append(ssim)
                        initial_eval_psnrs.append(psnr)
                        initial_eval_mses.append(mse)
                        initial_eval_lpipses.append(lpips)
                        initial_eval_dc_mses.append(dc_mse)
                        initial_eval_dc_maes.append(dc_mae)
    
                        if recon_corr is not None:
                            initial_eval_curve_corrs.append(recon_corr)
                            grasp_curve_corrs.append(grasp_corr)
    
                        # raw k-space eval
                        print("performing non-DRO eval...")
                        dc_mse_raw_grasp, dc_mae_raw_grasp = eval_grasp(raw_kspace, raw_csmaps, ground_truth, raw_grasp_img, eval_physics, device, eval_dir, dro_eval=False)
                        dc_mse_raw, dc_mae_raw, _ = eval_sample(raw_kspace, raw_csmaps, ground_truth, raw_x_recon, eval_physics, mask, raw_grasp_img, acceleration, int(N_spokes), eval_dir, label='val0', device=device, cluster=cluster, dro_eval=False, grasp_path=grasp_path, raw_slice_idx=raw_grasp_slice_idx, rescale=config['evaluation']['rescale'])
    
                        raw_grasp_dc_mses.append(dc_mse_raw_grasp)
                        raw_grasp_dc_maes.append(dc_mae_raw_grasp)
                        initial_eval_raw_dc_mses.append(dc_mse_raw)
                        initial_eval_raw_dc_maes.append(dc_mae_raw)
    
    
            if step0_do_val:
                step0_val_mc_loss = initial_val_mc_loss / len(val_dro_loader)
                val_mc_losses.append(step0_val_mc_loss)

                step0_val_ei_loss = initial_val_ei_loss / len(val_dro_loader)
                val_ei_losses.append(step0_val_ei_loss)

                step0_val_adj_loss = initial_val_adj_loss / len(val_dro_loader)
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

                eval_ssims.append(initial_eval_ssim)
                eval_psnrs.append(initial_eval_psnr)
                eval_mses.append(initial_eval_mse)
                eval_lpipses.append(initial_eval_lpips)
                eval_dc_mses.append(initial_eval_dc_mse) 
                eval_dc_maes.append(initial_eval_dc_mae) 
                eval_raw_dc_mses.append(initial_eval_raw_dc_mse) 
                eval_raw_dc_maes.append(initial_eval_raw_dc_mae) 
                eval_curve_corrs.append(initial_eval_curve_corr)

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

        print(f"Step 0 Train Losses: MC: {step0_train_mc_loss}, EI: {step0_train_ei_loss}, Adj: {step0_train_adj_loss}")
        if step0_do_val:
            print(f"Step 0 Val Losses: MC: {step0_val_mc_loss}, EI: {step0_val_ei_loss}, Adj: {step0_val_adj_loss}")
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
            epoch_eval_ssims = []
            epoch_eval_psnrs = []
            epoch_eval_mses = []
            epoch_eval_lpipses = []
            epoch_eval_dc_mses = []
            epoch_eval_dc_maes = []
            epoch_eval_curve_corrs = []
            epoch_eval_raw_dc_mses = []
            epoch_eval_raw_dc_maes = []


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
                                eval_ktraj, eval_dcomp, eval_nufft_ob, eval_adjnufft_ob = prep_nufft(N_samples, N_spokes_eval, N_time_eval)
                                eval_ktraj = eval_ktraj.to(device)
                                eval_dcomp = eval_dcomp.to(device)
                                eval_nufft_ob = eval_nufft_ob.to(device)
                                eval_adjnufft_ob = eval_adjnufft_ob.to(device)

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
                                )

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

            for measured_kspace, csmap, N_samples, N_spokes, N_time in train_loader_tqdm:  # measured_kspace shape: (B, C, I, S, T)
                
                start = time.time()

                # prepare inputs
                measured_kspace = to_torch_complex(measured_kspace).squeeze()
                measured_kspace = rearrange(measured_kspace, 't co sp sam -> co (sp sam) t')

                # prep physics operators
                ktraj, dcomp, nufft_ob, adjnufft_ob = prep_nufft(N_samples, N_spokes, N_time)
                ktraj = ktraj.to(device)
                dcomp = dcomp.to(device)
                nufft_ob = nufft_ob.to(device)
                adjnufft_ob = adjnufft_ob.to(device)

                if N_time > Ng:

                    max_idx = N_time - Ng
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
                optimizer.zero_grad()

                csmap = csmap.to(device).to(measured_kspace.dtype)

                # calculate acceleration factor
                acceleration = torch.tensor([N_full / int(N_spokes)], dtype=torch.float, device=device)

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
                            measured_kspace.to(device), physics, csmap, acceleration_encoding, start_timepoint_index, epoch=f"train{epoch}", norm=config['model']['norm']
                        )

                        # compute losses
                        running_adj_loss += adj_loss.item()

                        mc_loss = mc_loss_fn(measured_kspace.to(device), x_recon, physics, csmap)
                        running_mc_loss += mc_loss.item()

                        if use_ei_loss:
                            ei_loss, t_img = ei_loss_fn(
                                x_recon, physics, model, csmap, acceleration_encoding, start_timepoint_index
                            )


                            ei_loss_weight = get_cosine_ei_weight(
                                current_epoch=epoch,
                                warmup_epochs=warmup,
                                schedule_duration=duration,
                                target_weight=target_w_ei
                            )


                            running_ei_loss += ei_loss.item()
                            total_loss = mc_loss * mc_loss_weight + ei_loss * ei_loss_weight + torch.mul(adj_loss_weight, adj_loss)
                            train_loader_tqdm.set_postfix(
                                mc_loss=mc_loss.item(), ei_loss=ei_loss.item()
                            )

                        else:
                            total_loss = mc_loss * mc_loss_weight + torch.mul(adj_loss_weight, adj_loss)
                            train_loader_tqdm.set_postfix(mc_loss=mc_loss.item())

                    if torch.isnan(total_loss):
                        print(
                            "!!! ERROR: total_loss is NaN before backward pass. Aborting. !!!"
                        )
                        raise RuntimeError("total_loss is NaN")

                    if use_scaler:
                        scaler.scale(total_loss).backward()
                        scaler.unscale_(optimizer)
                    else:
                        total_loss.backward()


                    if config["debugging"]["enable_gradient_monitoring"] == True and iteration_count % config["debugging"]["monitoring_interval"] == 0:
                    
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

                    # cosine LR with 5-epoch warmup
                    total = epochs; warm = 5
                    if epoch <= warm:
                        lr_scale = epoch / warm
                    else:
                        p = (epoch - warm) / max(1, total - warm)
                        lr_scale = 0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * p))
                    for pg in optimizer.param_groups:
                        pg['lr'] = config["model"]["optimizer"]["lr"] * lr_scale

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

                    plot_enhancement_curve(
                        x_recon_reshaped,
                        output_filename = os.path.join(output_dir, 'enhancement_curves', f'train_sample_enhancement_curve_epoch_{epoch}.png'))
                    
                    if use_ei_loss:

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

            epoch_train_adj_loss = running_adj_loss / len(train_loader)
            train_adj_losses.append(epoch_train_adj_loss)
            weighted_train_adj_losses.append(epoch_train_adj_loss*adj_loss_weight)


            if global_rank == 0 or not config['training']['multigpu']:
                writer.add_scalar('Loss/Train_MC', epoch_train_mc_loss, epoch)
                writer.add_scalar('Loss/Train_EI', epoch_train_ei_loss, epoch)
                writer.add_scalar('Loss/Train_Adj', epoch_train_adj_loss, epoch)


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
                val_loader_tqdm = tqdm(
                    val_dro_loader,
                    desc=f"Epoch {epoch}/{epochs}  Validation",
                    unit="batch",
                    leave=False,
                )
                with torch.no_grad():
                    for val_dro_kspace_batch, val_csmap, val_ground_truth, val_dro_grasp_img, val_mask, grasp_path, val_raw_kspace, val_raw_grasp_img, val_raw_csmaps in tqdm(val_dro_loader):

                        val_csmap = val_csmap.squeeze(0).to(device)   # Remove batch dim
                        val_ground_truth = val_ground_truth.to(device) # Shape: (1, 2, T, H, W)

                        # prepare inputs
                        val_dro_kspace_batch = val_dro_kspace_batch.squeeze(0).to(device) # Remove batch dim
                        val_dro_grasp_img = val_dro_grasp_img.to(device)

                        val_raw_kspace = val_raw_kspace.squeeze(0).to(device) # Remove batch dim
                        val_raw_grasp_img = val_raw_grasp_img.to(device)
                        val_raw_csmaps = val_raw_csmaps.squeeze(0).to(device)

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
                                if N_time_eval > eval_chunk_size:
                                    val_x_recon, val_adj_loss = sliding_window_inference(H, W, N_time_eval, eval_ktraj, eval_dcomp, eval_nufft_ob, eval_adjnufft_ob, eval_chunk_size, eval_chunk_overlap, val_dro_kspace_batch, val_csmap, acceleration_encoding, start_timepoint_index, model, epoch=f"val{epoch}", device=device)  
                                    val_raw_x_recon, _ = sliding_window_inference(H, W, N_time_eval, eval_ktraj, eval_dcomp, eval_nufft_ob, eval_adjnufft_ob, eval_chunk_size, eval_chunk_overlap, val_raw_kspace, val_raw_csmaps, acceleration_encoding, start_timepoint_index, model, epoch="val0", device=device)  
                                else:
                                    val_x_recon, val_adj_loss, *_ = model(
                                    val_dro_kspace_batch.to(device), eval_physics, val_csmap, acceleration_encoding, start_timepoint_index, epoch=f"val{epoch}", norm=config['model']['norm']
                                    )
                                    val_raw_x_recon, *_ = model(
                                    val_raw_kspace.to(device), eval_physics, val_raw_csmaps, acceleration_encoding, start_timepoint_index, epoch="val0", norm=config['model']['norm']
                                    )
                                    val_adj_loss = val_adj_loss.item()

                                # fix orientation of raw k-space recon
                                # val_raw_x_recon = torch.rot90(val_raw_x_recon, k=2, dims=[-3,-2])

                                # compute losses
                                val_running_adj_loss += val_adj_loss

                                val_mc_loss = mc_loss_fn(val_dro_kspace_batch.to(device), val_x_recon, eval_physics, val_csmap)
                                val_running_mc_loss += val_mc_loss.item()

                                if use_ei_loss:
                                    val_ei_loss, val_t_img = ei_loss_fn(
                                        val_x_recon, eval_physics, model, val_csmap, acceleration_encoding, start_timepoint_index
                                    )

                                    val_running_ei_loss += val_ei_loss.item()
                                    val_loader_tqdm.set_postfix(
                                        val_mc_loss=val_mc_loss.item(), val_ei_loss=val_ei_loss.item()
                                    )
                                else:
                                    val_loader_tqdm.set_postfix(val_mc_loss=val_mc_loss.item())


                            ## Evaluation
                            if global_rank == 0 or not config['training']['multigpu']:
                                ssim, psnr, mse, lpips, dc_mse, dc_mae, recon_corr, _, _ = eval_sample(val_dro_kspace_batch, val_csmap, val_ground_truth, val_x_recon, eval_physics, val_mask, val_dro_grasp_img, acceleration, int(N_spokes), eval_dir, f'epoch{epoch}', device, cluster=cluster, dro_eval=True, grasp_path=grasp_path, rescale=config['evaluation']['rescale'])
                                epoch_eval_ssims.append(ssim)
                                epoch_eval_psnrs.append(psnr)
                                epoch_eval_mses.append(mse)
                                epoch_eval_lpipses.append(lpips)
                                epoch_eval_dc_mses.append(dc_mse)
                                epoch_eval_dc_maes.append(dc_mae)

                                if recon_corr is not None:
                                    epoch_eval_curve_corrs.append(recon_corr)

                                # raw k-space eval
                                dc_mse_raw, dc_mae_raw, _ = eval_sample(val_raw_kspace, val_raw_csmaps, val_ground_truth, val_raw_x_recon, eval_physics, val_mask, val_raw_grasp_img, acceleration, int(N_spokes), eval_dir, label=f'epoch{epoch}', device=device, cluster=cluster, dro_eval=False, grasp_path=grasp_path, raw_slice_idx=raw_grasp_slice_idx, rescale=config['evaluation']['rescale'])

                                epoch_eval_raw_dc_mses.append(dc_mse_raw)
                                epoch_eval_raw_dc_maes.append(dc_mae_raw)

                            
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



                # Calculate and store average validation evaluation metrics
                if global_rank == 0 or not config['training']['multigpu']:
                    epoch_eval_ssim = np.mean(epoch_eval_ssims)
                    epoch_eval_psnr = np.mean(epoch_eval_psnrs)
                    epoch_eval_mse = np.mean(epoch_eval_mses)
                    epoch_eval_lpips = np.mean(epoch_eval_lpipses)
                    epoch_eval_dc_mse = np.mean(epoch_eval_dc_mses)
                    epoch_eval_dc_mae = np.mean(epoch_eval_dc_maes)
                    epoch_eval_curve_corr = np.mean(epoch_eval_curve_corrs)
                    epoch_eval_raw_dc_mse = np.mean(epoch_eval_raw_dc_mses)
                    epoch_eval_raw_dc_mae = np.mean(epoch_eval_raw_dc_maes)

                    eval_ssims.append(epoch_eval_ssim)
                    eval_psnrs.append(epoch_eval_psnr)
                    eval_mses.append(epoch_eval_mse)
                    eval_lpipses.append(epoch_eval_lpips)
                    eval_dc_mses.append(epoch_eval_dc_mse) 
                    eval_dc_maes.append(epoch_eval_dc_mae) 
                    eval_raw_dc_mses.append(epoch_eval_raw_dc_mse) 
                    eval_raw_dc_maes.append(epoch_eval_raw_dc_mae)    
                    eval_curve_corrs.append(epoch_eval_curve_corr)  

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

                        plot_enhancement_curve(
                            val_x_recon_reshaped,
                            output_filename = os.path.join(output_dir, 'enhancement_curves', f'val_dro_sample_enhancement_curve_epoch_{epoch}.png'))
                        
                        plot_enhancement_curve(
                            val_dro_grasp_img,
                            output_filename = os.path.join(output_dir, 'enhancement_curves', f'val_dro_grasp_sample_enhancement_curve_epoch_{epoch}.png'))


                        if use_ei_loss:
                            plot_reconstruction_sample(
                                val_t_img,
                                f"Transformed Validation Sample - Epoch {epoch} (AF = {round(acceleration.item(), 1)}, SPF = {int(N_spokes)})",
                                f"transforms/transform_val_sample_epoch_{epoch}",
                                output_dir,
                                val_x_recon,
                                transform=True
                            )


                # Calculate and store average validation losses
                epoch_val_mc_loss = val_running_mc_loss / len(val_dro_loader)
                val_mc_losses.append(epoch_val_mc_loss)

                if use_ei_loss:
                    epoch_val_ei_loss = val_running_ei_loss / len(val_dro_loader)
                else:
                    epoch_val_ei_loss = 0.0

                val_ei_losses.append(epoch_val_ei_loss)

                if model_type == "LSFPNet":
                    epoch_val_adj_loss = val_running_adj_loss / len(val_dro_loader)
                else:
                    epoch_val_adj_loss = 0.0
                
                val_adj_losses.append(epoch_val_adj_loss)

                if global_rank == 0 or not config['training']['multigpu']:
                    writer.add_scalar('Loss/Val_MC', epoch_val_mc_loss, epoch)
                    writer.add_scalar('Loss/Val_EI', epoch_val_ei_loss, epoch)
                    writer.add_scalar('Loss/Val_Adj', epoch_val_adj_loss, epoch)




                # --- Plotting and Logging ---
                if epoch % save_interval == 0:

                    if global_rank == 0 or not config['training']['multigpu']:

                        # Save the model checkpoint
                        train_curves = dict(
                            train_mc_losses=train_mc_losses,
                            train_ei_losses=train_ei_losses,
                            train_adj_losses=train_adj_losses,
                            weighted_train_mc_losses=weighted_train_mc_losses,
                            weighted_train_ei_losses=weighted_train_ei_losses,
                            weighted_train_adj_losses=weighted_train_adj_losses,
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
                            eval_spf_curves=eval_spf_curves,
                        )
                        model_save_path = os.path.join(output_dir, f'{exp_name}_model.pth')
                        save_checkpoint(model, optimizer, epoch + 1, train_curves, val_curves, eval_curves, target_w_ei, step0_train_ei_loss, epoch_train_mc_loss, avg_grasp_ssim, avg_grasp_psnr, avg_grasp_mse, avg_grasp_lpips, avg_grasp_dc_mse, avg_grasp_dc_mae, avg_grasp_curve_corr, avg_grasp_raw_dc_mae, avg_grasp_raw_dc_mse, model_save_path)
                        print(f'Model saved to {model_save_path}')


                        # plot losses in one figure
                        # Set the seaborn style
                        sns.set_style("whitegrid")

                        # Create a figure and a set of subplots
                        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

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
                        plt.xlabel("Epoch")
                        plt.ylabel("Loss")
                        plt.title("Weighted Training Losses")
                        plt.legend()
                        plt.grid(True)
                        plt.savefig(os.path.join(output_dir, "weighted_losses.png"))
                        plt.close()


                        # plot evaluation metrics in one figure

                        # Set the seaborn style
                        sns.set_style("whitegrid")

                        # Create a figure and a set of subplots
                        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                        fig.suptitle(f'Evaluation Metrics Over Epochs ({N_spokes_eval} spokes/frame)', fontsize=20)

                        sns.lineplot(x=range(0, len(eval_ssims)*eval_frequency, eval_frequency), y=eval_ssims, ax=axes[0, 0])
                        axes[0, 0].set_title("DRO Evaluation SSIM")
                        axes[0, 0].set_xlabel("Epoch")
                        axes[0, 0].set_ylabel("SSIM")
                        axes[0, 0].axhline(y=avg_grasp_ssim, color='red', linestyle='--', linewidth=2)

                        sns.lineplot(x=range(0, len(eval_psnrs)*eval_frequency, eval_frequency), y=eval_psnrs, ax=axes[0, 1])
                        axes[0, 1].set_title("DRO Evaluation PSNR")
                        axes[0, 1].set_xlabel("Epoch")
                        axes[0, 1].set_ylabel("PSNR")
                        axes[0, 1].axhline(y=avg_grasp_psnr, color='red', linestyle='--', linewidth=2)

                        sns.lineplot(x=range(0, len(eval_mses)*eval_frequency, eval_frequency), y=eval_mses, ax=axes[0, 2])
                        axes[0, 2].set_title("DRO Evaluation Image MSE")
                        axes[0, 2].set_xlabel("Epoch")
                        axes[0, 2].set_ylabel("MSE")
                        axes[0, 2].axhline(y=avg_grasp_mse, color='red', linestyle='--', linewidth=2)

                        sns.lineplot(x=range(0, len(eval_lpipses)*eval_frequency, eval_frequency), y=eval_lpipses, ax=axes[1, 0])
                        axes[1, 0].set_title("Evaluation LPIPS")
                        axes[1, 0].set_xlabel("Epoch")
                        axes[1, 0].set_ylabel("LPIPS")
                        axes[1, 0].axhline(y=avg_grasp_lpips, color='red', linestyle='--', linewidth=2)

                        # sns.lineplot(x=range(0, len(eval_dc_maes)*eval_frequency, eval_frequency), y=eval_dc_maes, ax=axes[1, 0])
                        # axes[1, 0].set_title("DRO Evaluation Simulated k-space MAE")
                        # axes[1, 0].set_xlabel("Epoch")
                        # axes[1, 0].set_ylabel("MAE")
                        # axes[1, 0].axhline(y=avg_grasp_dc_mae, color='red', linestyle='--', linewidth=2)

                        sns.lineplot(x=range(0, len(eval_raw_dc_maes)*eval_frequency, eval_frequency), y=eval_raw_dc_maes, ax=axes[1, 1])
                        axes[1, 1].set_title("Non-DRO Evaluation Raw k-space MAE")
                        axes[1, 1].set_xlabel("Epoch")
                        axes[1, 1].set_ylabel("MAE")
                        axes[1, 1].axhline(y=avg_grasp_raw_dc_mae, color='red', linestyle='--', linewidth=2)

                        sns.lineplot(x=range(0, len(eval_curve_corrs)*eval_frequency, eval_frequency), y=eval_curve_corrs, ax=axes[1, 2])
                        axes[1, 2].set_title("DRO Tumor Enhancement Curve Correlation")
                        axes[1, 2].set_xlabel("Epoch")
                        axes[1, 2].set_ylabel("Pearson Correlation Coefficient")
                        axes[1, 2].axhline(y=avg_grasp_curve_corr, color='red', linestyle='--', linewidth=2)

                        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                        plt.savefig(os.path.join(output_dir, "eval_metrics.png"))
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
                                axes[0, 0].set_title("DRO Evaluation SSIM")
                                axes[0, 0].set_xlabel("Epoch")
                                axes[0, 0].set_ylabel("SSIM")

                                sns.lineplot(
                                    x=spf_curves["epochs"],
                                    y=spf_curves["eval_psnrs"],
                                    ax=axes[0, 1],
                                )
                                axes[0, 1].set_title("DRO Evaluation PSNR")
                                axes[0, 1].set_xlabel("Epoch")
                                axes[0, 1].set_ylabel("PSNR")

                                sns.lineplot(
                                    x=spf_curves["epochs"],
                                    y=spf_curves["eval_mses"],
                                    ax=axes[0, 2],
                                )
                                axes[0, 2].set_title("DRO Evaluation Image MSE")
                                axes[0, 2].set_xlabel("Epoch")
                                axes[0, 2].set_ylabel("MSE")

                                sns.lineplot(
                                    x=spf_curves["epochs"],
                                    y=spf_curves["eval_lpipses"],
                                    ax=axes[1, 0],
                                )
                                axes[1, 0].set_title("Evaluation LPIPS")
                                axes[1, 0].set_xlabel("Epoch")
                                axes[1, 0].set_ylabel("LPIPS")

                                sns.lineplot(
                                    x=spf_curves["epochs"],
                                    y=spf_curves["eval_raw_dc_maes"],
                                    ax=axes[1, 1],
                                )
                                axes[1, 1].set_title("Non-DRO Evaluation Raw k-space MAE")
                                axes[1, 1].set_xlabel("Epoch")
                                axes[1, 1].set_ylabel("MAE")

                                sns.lineplot(
                                    x=spf_curves["epochs"],
                                    y=spf_curves["eval_curve_corrs"],
                                    ax=axes[1, 2],
                                )
                                axes[1, 2].set_title("DRO Tumor Enhancement Curve Correlation")
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
                        plt.ylabel("DRO Simulated k-space MSE")
                        plt.title("Evaluation Data Consistency (MSE)")
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
                        plt.ylabel("DRO Simulated k-space MAE")
                        plt.title("Evaluation Data Consistency (MAE)")
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
                        plt.ylabel("Non-DRO Raw k-space MSE")
                        plt.title("Evaluation Data Consistency (MSE)")
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

                    if model_type == "LSFPNet":
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
                    print(f"GRASP SSIM: {avg_grasp_ssim:.4f} ± {np.std(grasp_ssims):.4f}")
                    print(f"GRASP PSNR: {avg_grasp_psnr:.4f} ± {np.std(grasp_psnrs):.4f}")
                    print(f"GRASP MSE: {avg_grasp_mse:.4f} ± {np.std(grasp_mses):.4f}")
                    print(f"GRASP LPIPS: {avg_grasp_lpips:.4f} ± {np.std(grasp_lpipses):.4f}")
                    print(f"GRASP DC MSE: {avg_grasp_dc_mse:.6f} ± {np.std(grasp_dc_mses):.4f}")
                    print(f"GRASP DC MAE: {avg_grasp_dc_mae:.6f} ± {np.std(grasp_dc_maes):.4f}")
                    print(f"GRASP DC MSE: {avg_grasp_raw_dc_mse:.6f} ± {np.std(raw_grasp_dc_maes):.4f}")
                    print(f"GRASP DC MAE: {avg_grasp_raw_dc_mae:.6f} ± {np.std(raw_grasp_dc_mses):.4f}")
                    print(f"GRASP Enhancement Curve Correlation: {avg_grasp_curve_corr:.6f} ± {np.std(grasp_curve_corrs):.4f}")

            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
                epoch_peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                attempt_peak_mem_gb = max(attempt_peak_mem_gb, epoch_peak_gb)





    # Save the model at the end of training
    if global_rank == 0 or not config['training']['multigpu']:
        train_curves = dict(
            train_mc_losses=train_mc_losses,
            train_ei_losses=train_ei_losses,
            train_adj_losses=train_adj_losses,
            weighted_train_mc_losses=weighted_train_mc_losses,
            weighted_train_ei_losses=weighted_train_ei_losses,
            weighted_train_adj_losses=weighted_train_adj_losses,
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
            eval_spf_curves=eval_spf_curves,
        )
        model_save_path = os.path.join(output_dir, f'{exp_name}_model.pth')
        save_checkpoint(model, optimizer, epochs + 1, train_curves, val_curves, eval_curves, target_w_ei, step0_train_ei_loss, epoch_train_mc_loss, avg_grasp_ssim, avg_grasp_psnr, avg_grasp_mse, avg_grasp_lpips, avg_grasp_dc_mse, avg_grasp_dc_mae, avg_grasp_curve_corr, avg_grasp_raw_dc_mae, avg_grasp_raw_dc_mse, model_save_path)
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



        # EVALUATE WITH VARIABLE SPOKES PER FRAME


        # --- Stress Test Plan ---
        # Designed to push the limits with very few spokes per frame.
        # This has a different (lower) total spoke budget.

        STRESS_TEST_PLAN = [
            {
                "spokes_per_frame": 2,
                "num_frames": 144, # 2 * 144 = 288 total spokes
                "description": "Stress test: max temporal points, 2 spokes"
            },
            {
                "spokes_per_frame": 4,
                "num_frames": 72, # 4 * 72 = 288 total spokes
                "description": "Stress test: max temporal points, 4 spokes"
            },
            {
                "spokes_per_frame": 8,
                "num_frames": 36, # 8 * 36 = 288 total spokes
                "description": "High temporal resolution"
            },
            {
                "spokes_per_frame": 16,
                "num_frames": 18, # 16 * 18 = 288 total spokes
                "description": "High temporal resolution"
            },
            {
                "spokes_per_frame": 24,
                "num_frames": 12, # 24 * 12 = 288 total spokes
                "description": "Good temporal resolution"
            },
            {
                "spokes_per_frame": 36,
                "num_frames": 8, # 36 * 8 = 288 total spokes
                "description": "Standard temporal resolution"
            }
        ]


        eval_spf_dataset = SimulatedSPFDataset(
            root_dir=config["evaluation"]["simulated_dataset_path"], 
            raw_kspace_path=data_dir,
            model_type=model_type, 
            patient_ids=val_dro_patient_ids,
            dataset_key=config["data"]["dataset_key"],
            grasp_slice_idx=raw_grasp_slice_idx
            )


        eval_spf_loader = DataLoader(
            eval_spf_dataset,
            batch_size=config["dataloader"]["batch_size"],
            shuffle=False,
            num_workers=config["dataloader"]["num_workers"],
        )




        with torch.no_grad():

            spf_recon_ssim = {}
            spf_recon_psnr = {}
            spf_recon_mse = {}
            spf_recon_lpips = {}
            spf_recon_dc_mse = {}
            spf_recon_dc_mae = {}
            spf_recon_corr = {}
            spf_grasp_ssim = {}
            spf_grasp_psnr = {}
            spf_grasp_mse = {}
            spf_grasp_lpips = {}
            spf_grasp_dc_mse = {}
            spf_grasp_dc_mae = {}
            spf_grasp_corr = {}
            spf_raw_dc_mse = {}
            spf_raw_dc_mae = {}
            spf_raw_grasp_dc_mse = {}
            spf_raw_grasp_dc_mae = {}

            print("--- Running Stress Test Evaluation (Budget: 176 spokes) ---")
            for eval_config in STRESS_TEST_PLAN:

                stress_test_ssims = []
                stress_test_psnrs = []
                stress_test_mses = []
                stress_test_lpipses = []
                stress_test_dc_mses = []
                stress_test_dc_maes = []
                stress_test_corrs = []
                stress_test_grasp_ssims = []
                stress_test_grasp_psnrs = []
                stress_test_grasp_mses = []
                stress_test_grasp_lpipses = []
                stress_test_grasp_dc_mses = []
                stress_test_grasp_dc_maes = []
                stress_test_grasp_corrs = []
                stress_test_raw_dc_mses = []
                stress_test_raw_dc_maes = []
                stress_test_raw_grasp_dc_mses = []
                stress_test_raw_grasp_dc_maes = []


                spokes = eval_config["spokes_per_frame"]
                num_frames = eval_config["num_frames"]

                eval_spf_dataset.spokes_per_frame = spokes
                eval_spf_dataset.num_frames = num_frames
                eval_spf_dataset._update_sample_paths()


                for csmap, ground_truth, dro_grasp_img, mask, grasp_path, raw_kspace, raw_csmaps, raw_grasp_img in tqdm(eval_spf_loader, desc="Variable Spokes Per Frame Evaluation"):


                    csmap = csmap.squeeze(0).to(device)   # Remove batch dim
                    ground_truth = ground_truth.to(device) # Shape: (1, 2, T, H, W)

                    dro_grasp_img = dro_grasp_img.to(device) # Shape: (1, 2, H, T, W)

                    raw_kspace = raw_kspace.squeeze(0).to(device) # Remove batch dim
                    raw_grasp_img = raw_grasp_img.to(device) # Shape: (1, 2, H, T, W)
                    raw_csmaps = raw_csmaps.squeeze(0).to(device)   # Remove batch dim

                    # SIMULATE KSPACE
                    ktraj, dcomp, nufft_ob, adjnufft_ob = prep_nufft(N_samples, spokes, num_frames)
                    physics = MCNUFFT(nufft_ob.to(device), adjnufft_ob.to(device), ktraj.to(device), dcomp.to(device))

                    sim_kspace = physics(False, ground_truth, csmap)

                    kspace = sim_kspace.squeeze(0).to(device) # Remove batch dim

                    # calculate acceleration factor
                    acceleration = torch.tensor([N_full / int(spokes)], dtype=torch.float, device=device)

                    if config['model']['encode_acceleration']:
                        acceleration_encoding = acceleration
                    else: 
                        acceleration_encoding = None

                    if config['model']['encode_time_index'] == False:
                        start_timepoint_index = None
                    else:
                        start_timepoint_index = torch.tensor([0], dtype=torch.float, device=device)


                    # check if GRASP image exists or if we need to perform GRASP recon
                    if type(dro_grasp_img) is int or len(dro_grasp_img.shape) == 1:
                        print(f"No GRASP file found, performing reconstruction with {spokes} spokes/frame and {num_frames} frames.")

                        dro_grasp_img = GRASPRecon(csmap, sim_kspace, spokes, num_frames, grasp_path[0])

                        grasp_recon_torch = torch.from_numpy(dro_grasp_img).permute(2, 0, 1) # T, H, W
                        grasp_recon_torch = torch.stack([grasp_recon_torch.real, grasp_recon_torch.imag], dim=0)

                        dro_grasp_img = torch.flip(grasp_recon_torch, dims=[-3])
                        dro_grasp_img = torch.rot90(dro_grasp_img, k=3, dims=[-3,-1]).unsqueeze(0)

                    dro_grasp_img = dro_grasp_img.to(device)

                    if num_frames > eval_chunk_size:
                        print("Performing sliding window eval...")
                        x_recon, _ = sliding_window_inference(H, W, num_frames, ktraj, dcomp, nufft_ob, adjnufft_ob, eval_chunk_size, eval_chunk_overlap, kspace, csmap, acceleration_encoding, start_timepoint_index, model, epoch=None, device=device)  
                        raw_x_recon, _ = sliding_window_inference(H, W, num_frames, ktraj, dcomp, nufft_ob, adjnufft_ob, eval_chunk_size, eval_chunk_overlap, raw_kspace, raw_csmaps, acceleration_encoding, start_timepoint_index, model, epoch="val0", device=device)  
                    else:
                        x_recon, *_ = model(
                            kspace.to(device), physics, csmap, acceleration_encoding, start_timepoint_index, epoch=None, norm=config['model']['norm']
                        )
                        raw_x_recon, adj_loss, *_ = model(
                        raw_kspace.to(device), physics, raw_csmaps, acceleration_encoding, start_timepoint_index, epoch=None, norm=config['model']['norm']
                        )

                    ground_truth = torch.stack([ground_truth.real, ground_truth.imag], dim=1)
                    ground_truth = rearrange(ground_truth, 'b i h w t -> b i t h w')

                    # fix orientation of raw k-space recon
                    # raw_x_recon = torch.rot90(raw_x_recon, k=2, dims=[-3,-2])


                    ## Evaluation
                    ssim, psnr, mse, lpips, dc_mse, dc_mae, recon_corr, grasp_corr, _ = eval_sample(
                        kspace,
                        csmap,
                        ground_truth,
                        x_recon,
                        physics,
                        mask,
                        dro_grasp_img,
                        acceleration,
                        int(spokes),
                        eval_dir,
                        f"{spokes}spf",
                        device,
                        cluster=cluster,
                        dro_eval=True,
                        grasp_path=grasp_path,
                        rescale=config['evaluation']['rescale'],
                        filename_suffix=f"{spokes}spf",
                    )
                    stress_test_ssims.append(ssim)
                    stress_test_psnrs.append(psnr)
                    stress_test_mses.append(mse)
                    stress_test_lpipses.append(lpips)
                    stress_test_dc_mses.append(dc_mse)
                    stress_test_dc_maes.append(dc_mae)

                    if recon_corr is not None:
                        stress_test_corrs.append(recon_corr)
                        stress_test_grasp_corrs.append(grasp_corr)


                    ssim_grasp, psnr_grasp, mse_grasp, lpips_grasp, dc_mse_grasp, dc_mae_grasp = eval_grasp(kspace, csmap, ground_truth, dro_grasp_img, physics, device, eval_dir, dro_eval=True)
                    stress_test_grasp_ssims.append(ssim_grasp)
                    stress_test_grasp_psnrs.append(psnr_grasp)
                    stress_test_grasp_mses.append(mse_grasp)
                    stress_test_grasp_lpipses.append(lpips_grasp)
                    stress_test_grasp_dc_mses.append(dc_mse_grasp)
                    stress_test_grasp_dc_maes.append(dc_mae_grasp)


                    # raw k-space
                    dc_mse_raw_grasp, dc_mae_raw_grasp = eval_grasp(raw_kspace, raw_csmaps, ground_truth, raw_grasp_img, physics, device, eval_dir, dro_eval=False)
                    dc_mse_raw, dc_mae_raw, _ = eval_sample(
                        raw_kspace,
                        raw_csmaps,
                        ground_truth,
                        raw_x_recon,
                        physics,
                        mask,
                        raw_grasp_img,
                        acceleration,
                        int(N_spokes),
                        eval_dir,
                        label=f"{spokes}spf",
                        device=device,
                        cluster=cluster,
                        dro_eval=False,
                        grasp_path=grasp_path,
                        raw_slice_idx=raw_grasp_slice_idx,
                        rescale=config['evaluation']['rescale'],
                        filename_suffix=f"{spokes}spf",
                    )

                    stress_test_raw_grasp_dc_mses.append(dc_mse_raw_grasp)
                    stress_test_raw_grasp_dc_maes.append(dc_mae_raw_grasp)
                    stress_test_raw_dc_mses.append(dc_mse_raw)
                    stress_test_raw_dc_maes.append(dc_mae_raw)




                spf_recon_ssim[spokes] = np.mean(stress_test_ssims)
                spf_recon_psnr[spokes] = np.mean(stress_test_psnrs)
                spf_recon_mse[spokes] = np.mean(stress_test_mses)
                spf_recon_lpips[spokes] = np.mean(stress_test_lpipses)
                spf_recon_dc_mse[spokes] = np.mean(stress_test_dc_mses)
                spf_recon_dc_mae[spokes] = np.mean(stress_test_dc_maes)
                spf_recon_corr[spokes] = np.mean(stress_test_corrs)

                spf_grasp_ssim[spokes] = np.mean(stress_test_grasp_ssims)
                spf_grasp_psnr[spokes] = np.mean(stress_test_grasp_psnrs)
                spf_grasp_mse[spokes] = np.mean(stress_test_grasp_mses)
                spf_grasp_lpips[spokes] = np.mean(stress_test_grasp_lpipses)
                spf_grasp_dc_mse[spokes] = np.mean(stress_test_grasp_dc_mses)
                spf_grasp_dc_mae[spokes] = np.mean(stress_test_grasp_dc_maes)
                spf_grasp_corr[spokes] = np.mean(stress_test_grasp_corrs)


                spf_raw_dc_mse[spokes] = np.mean(stress_test_raw_dc_mses)
                spf_raw_dc_mae[spokes] = np.mean(stress_test_raw_dc_maes)
                spf_raw_grasp_dc_mse[spokes] = np.mean(stress_test_raw_grasp_dc_mses)
                spf_raw_grasp_dc_mae[spokes] = np.mean(stress_test_raw_grasp_dc_maes)



                # Save Results
                spf_metrics_path = os.path.join(eval_dir, "eval_metrics.csv")
                with open(spf_metrics_path, 'a', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(['Recon', 'Spokes Per Frame', 'SSIM', 'PSNR', 'MSE', "LPIPS", 'DC MSE', 'DC MAE', 'EC Correlation'])

                    csvwriter.writerow(['DL', spokes, 
                    f'{np.mean(stress_test_ssims):.4f} +- {np.std(stress_test_ssims):.4f}', 
                    f'{np.mean(stress_test_psnrs):.4f} +- {np.std(stress_test_psnrs):.4f}', 
                    f'{np.mean(stress_test_mses):.4f} +- {np.std(stress_test_mses):.4f}', 
                    f'{np.mean(stress_test_lpipses):.4f} +- {np.std(stress_test_lpipses):.4f}', 
                    f'{np.mean(stress_test_dc_mses):.4f} +- {np.std(stress_test_dc_mses):.4f}',
                    f'{np.mean(stress_test_dc_maes):.4f} +- {np.std(stress_test_dc_maes):.4f}',
                    f'{np.mean(stress_test_raw_dc_mses):.4f} +- {np.std(stress_test_raw_dc_mses):.4f}',
                    f'{np.mean(stress_test_raw_dc_maes):.4f} +- {np.std(stress_test_raw_dc_maes):.4f}',
                    f'{np.mean(stress_test_corrs):.4f} +- {np.std(stress_test_corrs):.4f}'
                    ])

                    csvwriter.writerow(['GRASP', spokes, 
                    f'{np.mean(stress_test_grasp_ssims):.4f} +- {np.std(stress_test_grasp_ssims):.4f}', 
                    f'{np.mean(stress_test_grasp_psnrs):.4f} +- {np.std(stress_test_grasp_psnrs):.4f}', 
                    f'{np.mean(stress_test_grasp_mses):.4f} +- {np.std(stress_test_grasp_mses):.4f}', 
                    f'{np.mean(stress_test_grasp_lpipses):.4f} +- {np.std(stress_test_grasp_lpipses):.4f}', 
                    f'{np.mean(stress_test_grasp_dc_mses):.4f} +- {np.std(stress_test_grasp_dc_mses):.4f}',
                    f'{np.mean(stress_test_grasp_dc_maes):.4f} +- {np.std(stress_test_grasp_dc_maes):.4f}',
                    f'{np.mean(stress_test_raw_grasp_dc_mses):.4f} +- {np.std(stress_test_raw_grasp_dc_mses):.4f}',
                    f'{np.mean(stress_test_raw_grasp_dc_maes):.4f} +- {np.std(stress_test_raw_grasp_dc_maes):.4f}',
                    f'{np.mean(stress_test_grasp_corrs):.4f} +- {np.std(stress_test_grasp_corrs):.4f}',
                    ])


            

        # plot variable spokes/frame evaluation metrics in one figure
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))


        sns.lineplot(x=list(spf_recon_ssim.keys()), 
                    y=list(spf_recon_ssim.values()), 
                    label="DL Recon", 
                    marker='o',
                    ax=axes[0, 0])

        sns.lineplot(x=list(spf_grasp_ssim.keys()), 
                    y=list(spf_grasp_ssim.values()), 
                    label="Standard Recon", 
                    marker='o',
                    ax=axes[0, 0])

        axes[0, 0].set_title("DRO Evaluation SSIM vs Spokes/Frame")
        axes[0, 0].set_xlabel("Spokes per Frame")
        axes[0, 0].set_ylabel("SSIM")


        sns.lineplot(x=list(spf_recon_psnr.keys()), 
                    y=list(spf_recon_psnr.values()), 
                    label="DL Recon", 
                    marker='o',
                    ax=axes[0, 1])

        sns.lineplot(x=list(spf_grasp_psnr.keys()), 
                    y=list(spf_grasp_psnr.values()), 
                    label="Standard Recon", 
                    marker='o',
                    ax=axes[0, 1])
        axes[0, 1].set_title("DRO Evaluation PSNR vs Spokes/Frame")
        axes[0, 1].set_xlabel("Spokes per Frame")
        axes[0, 1].set_ylabel("PSNR")


        sns.lineplot(x=list(spf_recon_mse.keys()), 
                    y=list(spf_recon_mse.values()), 
                    label="DL Recon", 
                    marker='o',
                    ax=axes[0, 2])

        sns.lineplot(x=list(spf_grasp_mse.keys()), 
                    y=list(spf_grasp_mse.values()), 
                    label="Standard Recon", 
                    marker='o',
                    ax=axes[0, 2])
        axes[0, 2].set_title("DRO Evaluation Image MSE vs Spokes/Frame")
        axes[0, 2].set_xlabel("Spokes per Frame")
        axes[0, 2].set_ylabel("MSE")


        sns.lineplot(x=list(spf_recon_lpips.keys()), 
                    y=list(spf_recon_lpips.values()), 
                    label="DL Recon", 
                    marker='o',
                    ax=axes[1, 0])

        sns.lineplot(x=list(spf_grasp_lpips.keys()), 
                    y=list(spf_grasp_lpips.values()), 
                    label="Standard Recon", 
                    marker='o',
                    ax=axes[1, 0])
        axes[1, 0].set_title("Evaluation LPIPS vs Spokes/Frame")
        axes[1, 0].set_xlabel("Spokes per Frame")
        axes[1, 0].set_ylabel("LPIPS")





        sns.lineplot(x=list(spf_raw_dc_mse.keys()), 
            y=list(spf_raw_dc_mse.values()), 
            label="DL Recon", 
            marker='o',
            ax=axes[1, 1])

        sns.lineplot(x=list(spf_raw_grasp_dc_mse.keys()), 
                    y=list(spf_raw_grasp_dc_mse.values()), 
                    label="Standard Recon", 
                    marker='o',
                    ax=axes[1, 1])
        axes[1, 1].set_title("Non-DRO Evaluation Raw k-space MAE vs Spokes/Frame")
        axes[1, 1].set_xlabel("Spokes per Frame")
        axes[1, 1].set_ylabel("MAE")

        # sns.lineplot(x=list(spf_recon_dc_mae.keys()), 
        #             y=list(spf_recon_dc_mae.values()), 
        #             label="DL Recon", 
        #             marker='o',
        #             ax=axes[1, 1])

        # sns.lineplot(x=list(spf_grasp_dc_mae.keys()), 
        #             y=list(spf_grasp_dc_mae.values()), 
        #             label="Standard Recon", 
        #             marker='o',
        #             ax=axes[1, 1])
        # axes[1, 1].set_title("DRO Evaluation Simulated k-space MAE vs Spokes/Frame")
        # axes[1, 1].set_xlabel("Spokes per Frame")
        # axes[1, 1].set_ylabel("MAE")

        sns.lineplot(x=list(spf_recon_corr.keys()), 
                    y=list(spf_recon_corr.values()), 
                    label="DL Recon", 
                    marker='o',
                    ax=axes[1, 2])

        sns.lineplot(x=list(spf_grasp_corr.keys()), 
                    y=list(spf_grasp_corr.values()), 
                    label="Standard Recon", 
                    marker='o',
                    ax=axes[1, 2])
        axes[1, 2].set_title("DRO Tumor Enhancement Curve Correlation vs Spokes/Frame")
        axes[1, 2].set_xlabel("Spokes per Frame")
        axes[1, 2].set_ylabel("Pearson Correlation Coefficient")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "spf_eval_metrics.png"))
        plt.close()

    if global_rank == 0 or not config['training']['multigpu']:
        _finalize_run_state("completed")
        writer.close()

    cleanup()


if __name__ == '__main__':
    main()
