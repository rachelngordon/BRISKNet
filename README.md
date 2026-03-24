# BRISKNet: Breast Rapid Imaging via Self-Supervised Kinetics

This repository contains the official code for **BRISKNet**, an unsupervised, physics-informed framework for multi-coil radial breast DCE-MRI reconstruction.

<!-- The method is described in the paper:

> BRISKNet: Breast Rapid Imaging via Self-Supervised Kinetics

--- -->

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

We use the [fastMRI breast dataset](https://fastmri.med.nyu.edu/) from Solomon et al. (2024). The dataset is partitioned into training, validation, and test sets.

The split file is located at:

```
data/data_split.json
```

For evaluation, we use the [Digital Reference Object (DRO) Toolkit](https://pubmed.ncbi.nlm.nih.gov/38775077/) from Bae et al. (2024).

Our fork of the repository is included in:

```
external/
```

## Data Processing
```bash
bash process_all_data.sh {BASE_PATH} {OUT_PATH} {NUM_SLICES}
```

- BASE_PATH is the directory containing raw fastMRI breast data. 

Example:

```
.../fastMRI_breast_data/fastMRI_breast_IDS_
```

- OUT_PATH is the directory where the complex-valued zero-filled k-space will be saved
- NUM_SLICES is the number of slices to process (we use 192)

Code is adapted from the [fastMRI preprocessing pipeline](https://github.com/eddysolo/demo_dce_recon).

## Training

The model architecture is adapted from [LSFP-Net](https://github.com/aaronfeng369/LSFP-Net/tree/LSFP-Net) (Zhao et al., 2023).
Loss constraints are adapted from [DDEI](https://github.com/Andrewwango/ddei) (Wang et al., 2025).
Transformations use the [DeepInverse](https://deepinv.github.io/deepinv/) library.

Run training:

```bash
python train.py \
    --exp_name {EXP_NAME} \
    --config example.yaml
```

## Inference
```bash
python inference/run_inference_new_dro.py \
    --exp_name {EXP_NAME} \
    --log_file inference/inference_logs.json
```

Outputs:

Plots saved to experiment directory.
Aggregated metrics saved to the log file.

## References

<!-- If you use this code, please cite:

@article{brisknet2025,
  title={BRISKNet: Breast Rapid Imaging via Self-Supervised Kinetics},
  author={Anonymous},
  year={2025}
} -->

- Solomon et al., fastMRI Breast dataset, 2025
- Bae et al., Digital Reference Object Toolkit, 2024
- He et al., LSFPNet, Nature Communications, 2023
- Wang & Davies, Fully Unsupervised Dynamic MRI Reconstruction via Diffeo-Temporal Equivariance, ISBI 2025


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
