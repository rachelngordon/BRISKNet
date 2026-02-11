#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="recon_mri"

# 0) Create env from YAML
micromamba create -n "${ENV_NAME}" -f env_min.yaml -y

# 1) Install pip requirements into that env (includes pinned git deps)
#    Use constraints to avoid pip overriding conda-pinned numpy/h5py.
micromamba run -n "${ENV_NAME}" python -m pip install --no-build-isolation -r requirements.txt -c constraints.txt

# 2) Optional: install Mamba kernels for `model.name: MambaRecon`
#    Set INSTALL_MAMBA=1 to enable.
if [[ "${INSTALL_MAMBA:-0}" == "1" ]]; then
  micromamba run -n "${ENV_NAME}" python -m pip install --no-build-isolation causal-conv1d mamba-ssm
fi

echo
echo "Env ${ENV_NAME} setup complete."
echo "To use it interactively:"
echo "  micromamba activate ${ENV_NAME}"
