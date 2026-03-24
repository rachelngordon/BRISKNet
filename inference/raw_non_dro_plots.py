"""Plot non-DRO raw reconstructions and tumor overlays. Run: python3 -m inference.raw_non_dro_plots (edit user inputs)"""

import json
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from einops import rearrange
from skimage.measure import find_contours

REPO_ROOT = Path(__file__).resolve().parents[1]
JOB_SCRIPTS_DIR = REPO_ROOT / "job-scripts"
if str(JOB_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(JOB_SCRIPTS_DIR))

from cluster_paths import apply_cluster_paths
from dataloader import SimulatedDataset
from inference.eval import _get_patient_id_from_grasp_path, _load_slice_map, _load_tumor_mask
from radial_lsfp import MCNUFFT
from run_inference import _build_model, _load_weights, _resolve_eval_params
from utils import prep_nufft, set_seed, sliding_window_inference, to_torch_complex


# ---- User inputs ----
exp_dir = '/net/projects2/annawoodard/rachelgordon/experiments/ei_warp_8spf_final'
device_override = None  # e.g., 'cuda:0' or 'cpu'
eval_spokes = None  # override spokes per frame if desired
eval_frames = None  # override num frames if desired
phase_index = None  # override curriculum phase index
seed = 12
max_malignant_samples = 5
output_dir = 'raw_non_dro_plots'

# For plotting
frames_to_show = None  # e.g., [0, 6, 13, 20]


set_seed(seed)
exp_name = os.path.basename(exp_dir.rstrip('/'))

config_path = os.path.join(exp_dir, 'config.yaml')
ckpt_path = os.path.join(exp_dir, f'{exp_name}_model.pth')

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

config = apply_cluster_paths(config)
device = torch.device(device_override or config['training']['device'])

# Resolve eval parameters (mirrors run_inference.py)
N_spokes_eval, N_time_eval = _resolve_eval_params(
    config,
    spokes=eval_spokes,
    frames=eval_frames,
    phase_idx=phase_index,
)

rescale = config.get('evaluation', {}).get('rescale', True)
raw_grasp_slice_idx = config.get('evaluation', {}).get('raw_grasp_slice_idx', 95)
cluster = config.get('experiment', {}).get('cluster', 'Randi')

# Load the validation split to match run_inference.py
with open(config['data']['split_file'], 'r') as fp:
    splits = json.load(fp)
val_ids = splits.get('val_dro') or splits.get('val') or []

dataset = SimulatedDataset(
    root_dir='/net/scratch2/rachelgordon/dro_dataset_frontpad',
    raw_kspace_path=config['data']['root_dir'],
    model_type=config['model']['name'],
    patient_ids=val_ids,
    dataset_key=config['data']['dataset_key'],
    spokes_per_frame=N_spokes_eval,
    num_frames=N_time_eval,
    grasp_slice_idx=raw_grasp_slice_idx,
)

print(f'Dataset size: {len(dataset)}')
os.makedirs(output_dir, exist_ok=True)

# Build model and load weights
block_dir = os.path.join(config['experiment']['output_dir'], exp_name, 'block_outputs')
os.makedirs(block_dir, exist_ok=True)
model = _build_model(config, device, block_dir)
model = _load_weights(model, ckpt_path)
model.eval()


# Prepare physics for inference
N_samples = config['data']['samples']
H, W = config['data']['height'], config['data']['width']
N_full = H * np.pi / 2

eval_ktraj, eval_dcomp, eval_nufft_ob, eval_adjnufft_ob = prep_nufft(
    N_samples, N_spokes_eval, N_time_eval
)
eval_ktraj = eval_ktraj.to(device)
eval_dcomp = eval_dcomp.to(device)
eval_nufft_ob = eval_nufft_ob.to(device)
eval_adjnufft_ob = eval_adjnufft_ob.to(device)
eval_physics = MCNUFFT(eval_nufft_ob, eval_adjnufft_ob, eval_ktraj, eval_dcomp)

eval_chunk_size = config.get('evaluation', {}).get('chunk_size', N_time_eval)
eval_chunk_overlap = config.get('evaluation', {}).get('chunk_overlap', 0)

acceleration_val = torch.tensor(
    [N_full / int(eval_ktraj.shape[1] / config['data']['samples'])],
    dtype=torch.float,
    device=device,
)
acceleration_encoding = acceleration_val if config['model']['encode_acceleration'] else None
start_timepoint_index = (
    torch.tensor([0], dtype=torch.float, device=device)
    if config['model']['encode_time_index']
    else None
)

def resolve_patient_and_mask(grasp_path, cluster_name, fallback_slice_idx):
    # DataLoader may wrap string in a list when batching; handle both.
    if isinstance(grasp_path, (list, tuple)):
        grasp_path = grasp_path[0] if grasp_path else None

    patient_id = _get_patient_id_from_grasp_path(grasp_path)
    if patient_id is None:
        return None, None

    slice_map = _load_slice_map()
    slice_idx = slice_map.get(patient_id, fallback_slice_idx)
    if slice_idx is None or slice_idx < 0:
        slice_idx = fallback_slice_idx

    tumor_mask = _load_tumor_mask(cluster_name, patient_id, slice_idx=slice_idx)
    return patient_id, tumor_mask


def plot_curve_with_roi(img_stack, tumor_mask, title, filename, frames_to_show=None):
    if tumor_mask is None or not tumor_mask.any():
        print('Empty or missing tumor mask; skipping plot.')
        return

    num_frames = img_stack.shape[2]
    time_points = np.linspace(0, 150, num_frames)

    if frames_to_show is None:
        interval = max(1, num_frames // 4)
        frames_to_show = [0, interval, 2 * interval, num_frames - 1]

    mean_curve = [img_stack[:, :, t][tumor_mask].mean() for t in range(num_frames)]

    fig = plt.figure(figsize=(18, 7.5))
    gs = fig.add_gridspec(2, 4, wspace=0.12, hspace=0.18)
    ax_curve = fig.add_subplot(gs[:, :2])
    ax_imgs = [
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[0, 3]),
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[1, 3]),
    ]

    ax_curve.plot(time_points, mean_curve, 'o-', linewidth=2, label='Mean tumor signal')
    ax_curve.set_title(title)
    ax_curve.set_xlabel('Time (s)')
    ax_curve.set_ylabel('Mean signal')
    ax_curve.grid(True, linestyle='--', alpha=0.5)

    highlight_times = [time_points[i] for i in frames_to_show]
    highlight_vals = [mean_curve[i] for i in frames_to_show]
    ax_curve.plot(highlight_times, highlight_vals, 'r*', markersize=12)

    contours = find_contours(tumor_mask, 0.5)
    vmin = np.percentile(img_stack, 1)
    vmax = np.percentile(img_stack, 99.5)

    for ax, frame_idx in zip(ax_imgs, frames_to_show):
        img = img_stack[:, :, frame_idx]
        ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='red')
        ax.set_title(f'Frame {frame_idx}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(filename)

malignant_count = 0
for sample_idx in range(len(dataset)):
    if malignant_count >= max_malignant_samples:
        break

    (
        _dro_kspace,
        _csmap,
        ground_truth,
        _dro_grasp_img,
        _mask,
        grasp_path,
        raw_kspace,
        raw_grasp_img,
        raw_csmaps,
    ) = dataset[sample_idx]

    patient_id, tumor_mask = resolve_patient_and_mask(
        grasp_path, cluster, raw_grasp_slice_idx
    )
    if tumor_mask is None or not tumor_mask.any():
        continue

    sample_label = patient_id or f'sample_{sample_idx:02d}'
    print(f'Processing {sample_label}')

    ground_truth = torch.as_tensor(ground_truth).to(device)
    raw_kspace = torch.as_tensor(raw_kspace).to(device)
    raw_grasp_img = torch.as_tensor(raw_grasp_img).to(device)
    raw_csmaps = torch.as_tensor(raw_csmaps).squeeze(0).to(device)
    raw_csmaps = raw_csmaps.unsqueeze(0)

    # Run raw (non-DRO) inference
    with torch.no_grad():
        if N_time_eval > eval_chunk_size:
            raw_x_recon, _ = sliding_window_inference(
                H,
                W,
                N_time_eval,
                eval_ktraj,
                eval_dcomp,
                eval_nufft_ob,
                eval_adjnufft_ob,
                eval_chunk_size,
                eval_chunk_overlap,
                raw_kspace,
                raw_csmaps,
                acceleration_encoding,
                start_timepoint_index,
                model,
                epoch='inference',
                device=device,
                norm=config['model']['norm'],
            )
        else:
            raw_x_recon, *_ = model(
                raw_kspace,
                eval_physics,
                raw_csmaps,
                acceleration_encoding,
                start_timepoint_index,
                epoch='inference',
                norm=config['model']['norm'],
            )

    raw_x_recon = raw_x_recon.cpu()
    raw_grasp_img = raw_grasp_img.cpu()
    ground_truth = ground_truth.cpu()

    # Optional rescaling to match run_inference.py behavior
    raw_x_recon_np = raw_x_recon.numpy()
    raw_grasp_np = raw_grasp_img.numpy()
    ground_truth_np = ground_truth.numpy()

    if rescale:
        c = np.dot(raw_x_recon_np.flatten(), ground_truth_np.flatten()) / np.dot(
            raw_x_recon_np.flatten(), raw_x_recon_np.flatten()
        )
        raw_x_recon_scaled = torch.tensor(c * raw_x_recon_np)
        c_grasp = np.dot(raw_grasp_np.flatten(), ground_truth_np.flatten()) / np.dot(
            raw_grasp_np.flatten(), raw_grasp_np.flatten()
        )
        raw_grasp_scaled = torch.tensor(c_grasp * raw_grasp_np)
    else:
        raw_x_recon_scaled = torch.tensor(raw_x_recon_np)
        raw_grasp_scaled = torch.tensor(raw_grasp_np)

    # Convert to magnitude stacks (H, W, T)
    raw_recon_complex = to_torch_complex(raw_x_recon_scaled).squeeze().cpu().numpy()
    raw_recon_mag = np.abs(raw_recon_complex)

    raw_grasp_scaled = raw_grasp_scaled.unsqueeze(0)
    raw_grasp_complex = rearrange(
        to_torch_complex(raw_grasp_scaled).squeeze(),
        'h t w -> h w t',
    )
    raw_grasp_complex = raw_grasp_complex.cpu().numpy()
    raw_grasp_mag = np.abs(raw_grasp_complex)

    plot_curve_with_roi(
        raw_recon_mag,
        tumor_mask,
        title=f'Raw DL reconstruction: malignant ROI ({sample_label})',
        filename=os.path.join(output_dir, f'{sample_label}_raw_dl_recon_curve.png'),
        frames_to_show=frames_to_show,
    )

    plot_curve_with_roi(
        raw_grasp_mag,
        tumor_mask,
        title=f'Raw GRASP reconstruction: malignant ROI ({sample_label})',
        filename=os.path.join(output_dir, f'{sample_label}_raw_grasp_recon_curve.png'),
        frames_to_show=frames_to_show,
    )

    malignant_count += 1

print(f'Generated plots for {malignant_count} malignant validation samples.')
