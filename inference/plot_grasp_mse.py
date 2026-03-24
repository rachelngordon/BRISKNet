"""Compute and plot GRASP MSE versus temporal resolution. Run: python3 -m inference.plot_grasp_mse"""

import json
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import numpy as np
from einops import rearrange
from radial import MCNUFFT
from dataloader import SimulatedSPFDataset
from inference.eval import eval_grasp
from utils import prep_nufft
import seaborn as sns


# load data
split_file = "/gpfs/data/karczmar-lab/workspaces/rachelgordon/breastMRI-recon/ddei/data/data_split.json"
with open(split_file, "r") as fp:
    splits = json.load(fp)

val_patient_ids = splits["val"]
val_dro_patient_ids = splits["val_dro"]


root_dir = "/ess/scratch/scratch1/rachelgordon/dro_dataset"
model_type = "LSFPNet"

eval_spf_dataset = SimulatedSPFDataset(
    root_dir=root_dir, 
    model_type=model_type, 
    patient_ids=val_dro_patient_ids,
    )


eval_spf_loader = DataLoader(
    eval_spf_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
)

device = torch.device("cuda")
N_samples = 640
exp_name = "plot_grasp_metrics"
output_dir = os.path.join("output", exp_name)
eval_dir = os.path.join(output_dir, "eval_results")


MAIN_EVALUATION_PLAN = [
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
            # {
            #     "spokes_per_frame": 24,
            #     "num_frames": 12, # 24 * 12 = 288 total spokes
            #     "description": "Good temporal resolution"
            # },
            # {
            #     "spokes_per_frame": 32,
            #     "num_frames": 8, # 36 * 8 = 288 total spokes
            #     "description": "Standard temporal resolution"
            # },
        ]



spf_grasp_mse = {}
 

for eval_config in MAIN_EVALUATION_PLAN:
    stress_test_grasp_mses = []

    spokes = eval_config["spokes_per_frame"]
    num_frames = eval_config["num_frames"]

    eval_spf_dataset.spokes_per_frame = spokes
    eval_spf_dataset.num_frames = num_frames
    eval_spf_dataset._update_sample_paths()

    for csmap, ground_truth, grasp_img, _, grasp_path in eval_spf_loader:
        print("grasp_path: ", grasp_path)

        csmap = csmap.squeeze(0).to(device)   # Remove batch dim
        ground_truth = ground_truth.to(device)  # Shape: (1, 2, T, H, W)

        # Simulate k-space for GRASP evaluation.
        ktraj, dcomp, nufft_ob, adjnufft_ob = prep_nufft(N_samples, spokes, num_frames)
        physics = MCNUFFT(
            nufft_ob.to(device),
            adjnufft_ob.to(device),
            ktraj.to(device),
            dcomp.to(device),
        )

        sim_kspace = physics(False, ground_truth, csmap)
        kspace = sim_kspace.squeeze(0).to(device)  # Remove batch dim

        grasp_img = grasp_img.to(device)
        ground_truth = torch.stack([ground_truth.real, ground_truth.imag], dim=1)
        ground_truth = rearrange(ground_truth, "b i h w t -> b i t h w")

        _, _, mse_grasp, _, _, _ = eval_grasp(
            kspace, csmap, ground_truth, grasp_img, physics, device, eval_dir
        )
        stress_test_grasp_mses.append(mse_grasp)

    spf_grasp_mse[spokes] = np.mean(stress_test_grasp_mses)


# temporal resolution = frames/second,  150 seconds / timeframes

# accelerations = {}
# for spf in spf_grasp_mse.keys():

#     acceleration = N_full / int(spf)
#     accelerations[acceleration] = spf_grasp_mse[spf]

temp_resolutions = {}
for spf in spf_grasp_mse.keys():
    num_timeframes = round(288 / int(spf), 0)
    temp_res = round(150 / num_timeframes, 0)
    temp_resolutions[temp_res] = spf_grasp_mse[spf]


# # Create the line plot
# sns.lineplot(x=list(temp_resolutions.keys()), y=list(temp_resolutions.values()), marker='o')

# plt.title("MSE of GRASP Reconstruction vs. Temporal Resolution", fontsize=16)
# plt.xlabel("Temporal Resolution (seconds/frame)", fontsize=14)
# plt.ylabel("MSE", fontsize=14)

# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)

# plt.grid(True) # Add a grid for better readability

# plt.savefig('grasp_mse_spf.png')


plt.figure(figsize=(8, 4))  # width, height in inches — reduce height here

# Create the line plot
sns.lineplot(x=list(temp_resolutions.keys()), y=list(temp_resolutions.values()), marker='o')

plt.title("MSE of GRASP Reconstruction vs. Temporal Resolution", fontsize=18)
plt.xlabel("Temporal Resolution (seconds/frame)", fontsize=16)
plt.ylabel("MSE", fontsize=16)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.grid(True)

plt.tight_layout()  # Ensures labels fit inside the frame
plt.savefig('grasp_mse_spf.png', dpi=300)
