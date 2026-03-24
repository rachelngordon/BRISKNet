<!-- # BRISKNet: Breast Rapid Imaging via Self-Supervised Kinetics
This repository contains relevant code for BRISKNet, an unsupervised, physics-informed framework for multi-coil, radial breast DCE-MRI reconstruction. 

## Environment Set Up
Install necessary dependencies:
```bash
bash env_setup.sh
```

Activate the environment:
```bash
micromamba actviate recon_mri
```

## Dataset 
We use the fastMRI breast dataset from Solomon et al. (2025), which can be accessed here: https://fastmri.med.nyu.edu/
The dataset was partitioned into training, validation, and test sets, and the IDs for the samples included in each dataset can be found at data/data_split.json
For evaluation, we use the Digital Reference Object Toolkit from Bae et al. (2024), and our fork of the repo can be found in external.

## Data Processing
```bash
bash process_all_data.sh {BASE_PATH} {OUT_PATH} {NUM_SLICES}
```
Code was adapted from the fastMRI preprocessing code to save the complex-valued zero-filled k-space. 
BASE_PATH is the directory where the raw data is stored, such as .../fastMRI_breast_data/fastMRI_breast_IDS_
OUT_PATH is the directory where the zero-filled k-space will be saved
NUM_SLICES was set to 192 to match the fastMRI default


## Training
The model architecture was adapted from Zhao et al. (2023) and the original code can be found here: https://github.com/aaronfeng369/LSFP-Net/tree/LSFP-Net
The loss constraints were adapted from Wang et al. (2025) and the original code can be found here: https://github.com/Andrewwango/ddei
Transformations were built on the deep inverse library, which can be found here: https://deepinv.github.io/deepinv/
```bash
python train_zf.py --exp_name {EXP_NAME} --config configs/example_config.yaml
```

## Inference
```bash
python inference/run_inference_new_dro.py --exp_name {EXP_NAME} --log_file inference/inference_logs.json
```
Plots will be saved to the experiment directory and aggregated metrics will be saved to the log file. 


## References
Preprocessing code is adapted from code provided with the fastMRI breast dataset: https://github.com/eddysolo/demo_dce_recon
LSFPNet architecture is adapted from https://github.com/aaronfeng369/LSFP-Net/tree/LSFP-Net
DDEI framework is adapted from: https://github.com/Andrewwango/ddei
Spatial transformations are adapted from: https://deepinv.github.io/deepinv/ -->


# BRISKNet: Breast Rapid Imaging via Self-Supervised Kinetics

This repository contains the official code for **BRISKNet**, an unsupervised, physics-informed framework for multi-coil radial breast DCE-MRI reconstruction.

The method is described in the paper:

> BRISKNet: Breast Rapid Imaging via Self-Supervised Kinetics

---

## Environment Setup

Install dependencies:

```bash
bash env_setup.sh
```

Activate the environment:
```bash
micromamba activate recon_mri
```

## Dataset

We use the fastMRI breast dataset from Solomon et al. (2024), available at:
https://fastmri.med.nyu.edu/

The dataset is partitioned into training, validation, and test sets.
The split file is located at:

```
data/data_split.json
```

For evaluation, we use the Digital Reference Object (DRO) Toolkit from Bae et al. (2024).
Our fork of the repository is included in:

```
external/
```

## Data Processing
```bash
bash process_all_data.sh {BASE_PATH} {OUT_PATH} {NUM_SLICES}
```

BASE_PATH — directory containing raw fastMRI breast data

Example:

```
.../fastMRI_breast_data/fastMRI_breast_IDS_
```

OUT_PATH — directory where zero-filled k-space will be saved
NUM_SLICES — number of slices to process 

Code is adapted from the fastMRI preprocessing pipeline
Saves complex-valued zero-filled k-space

## Training

The model architecture is adapted from [LSFP-Net](https://github.com/aaronfeng369/LSFP-Net/tree/LSFP-Net) (Zhao et al., 2023).
Loss constraints are adapted from [DDEI](https://github.com/Andrewwango/ddei) (Wang et al., 2025).
Transformations use the [DeepInverse](https://deepinv.github.io/deepinv/) library

Run training:

```bash
python train_zf.py \
    --exp_name {EXP_NAME} \
    --config configs/example_config.yaml
```

## Inference
```bash
python inference/run_inference_new_dro.py \
    --exp_name {EXP_NAME} \
    --log_file inference/inference_logs.json
```

Outputs:

Plots saved to experiment directory
Aggregated metrics saved to the log file

## References

<!-- If you use this code, please cite:

@article{brisknet2025,
  title={BRISKNet: Breast Rapid Imaging via Self-Supervised Kinetics},
  author={Anonymous},
  year={2025}
} -->

Solomon et al., fastMRI Breast dataset, 2025
Bae et al., Digital Reference Object Toolkit, 2024
He et al., LSFPNet, Nature Communications, 2023
Wang & Davies, Fully Unsupervised Dynamic MRI Reconstruction via Diffeo-Temporal Equivariance, ISBI 2025


<!-- ReconResNet code is adapted from: https://github.com/soumickmj/NCC1701/tree/main
Data Consistency code is adapted from: https://github.com/koflera/DynamicRadCineMRI/tree/main -->

<!-- ## Architecture Switch
- `model.name: LSFPNet` uses the original LSFP model.
- `model.name: MambaRecon` uses the radial-adapted MambaRecon model.
- Example Mamba config: `configs/config_mc_36spf_mamba.yaml`.
- 2-SPF Mamba + EI/Rebin schedule: `configs/config_sampling_2spf_rebin_v3_mamba.yaml`.
- Fast debug smoke config (train + EI + rebin + eval): `configs/config_sampling_2spf_rebin_v3_mamba_debug.yaml`. -->

<!-- # # Create the env (CUDA 12.4 runtime; works with newer drivers)
# micromamba create -n brisknet -f env_min.yaml -y
# micromamba activate brisknet

# # Install pip-only deps (includes pinned git repos).
# # Use constraints to prevent pip from upgrading conda-pinned numpy/h5py.
# python -m pip install --no-build-isolation -r requirements.txt -c constraints.txt

# # Optional: needed only for model.name=MambaRecon
# python -m pip install --no-build-isolation causal-conv1d mamba-ssm

# # Optional one-liner:
# # bash env_setup.sh
# # INSTALL_MAMBA=1 bash env_setup.sh -->
