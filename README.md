# breastMRI-recon
Breast MRI reconstruction with DCE-MRI dataset from NYU

## Environment Set Up
```bash
# Create the env (CUDA 12.4 runtime; works with newer drivers)
micromamba create -n brisknet -f env_min.yaml -y
micromamba activate brisknet

# Install pip-only deps (includes pinned git repos).
# Use constraints to prevent pip from upgrading conda-pinned numpy/h5py.
python -m pip install --no-build-isolation -r requirements.txt -c constraints.txt

# Optional one-liner:
# bash env_setup.sh
```

## References
Preprocessing code is adapted from code provided with the fastMRI breast dataset: https://github.com/eddysolo/demo_dce_recon
ReconResNet code is adapted from: https://github.com/soumickmj/NCC1701/tree/main
Data Consistency code is adapted from: https://github.com/koflera/DynamicRadCineMRI/tree/main
