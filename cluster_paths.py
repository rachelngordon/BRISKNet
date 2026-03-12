"""Cluster path normalization helpers. Run: imported by training/inference scripts (not intended to run directly)."""

import os

# Base directories for each cluster. The "code" base is used for config/output
# files and the "data" base for datasets and simulation assets.
CLUSTER_BASES = {
    "Randi": {
        "data": "/ess/scratch/scratch1/rachelgordon",
        "code": "/gpfs/data/karczmar-lab/workspaces/rachelgordon/breastMRI-recon/ddei",
    },
    "DSI": {
        "data": "/net/scratch2/rachelgordon",
        "code": "/home/rachelgordon/mri_recon/breastMRI-recon/ddei",
    },
}


def _swap_base(path: str, cluster: str, path_type: str) -> str:
    """
    Swap a path prefix to match the requested cluster.

    If the incoming path already has a known cluster prefix, it is replaced with
    the prefix for the requested cluster. Relative paths are anchored to the
    requested cluster base.
    """
    if path is None:
        return path

    if cluster not in CLUSTER_BASES:
        raise ValueError(f"Unknown cluster '{cluster}'. Supported clusters: {list(CLUSTER_BASES)}")

    base_for_cluster = CLUSTER_BASES[cluster][path_type]

    for bases in CLUSTER_BASES.values():
        candidate_base = bases[path_type]
        if path.startswith(candidate_base):
            suffix = path[len(candidate_base):].lstrip(os.sep)
            return os.path.join(base_for_cluster, suffix) if suffix else base_for_cluster

    if not os.path.isabs(path):
        return os.path.join(base_for_cluster, path)

    return path


def apply_cluster_paths(config: dict) -> dict:
    """Normalize config paths based on the chosen cluster."""
    cluster = config.get("experiment", {}).get("cluster", "Randi")

    data_cfg = config.get("data", {})
    eval_cfg = config.get("evaluation", {})
    exp_cfg = config.get("experiment", {})

    if "root_dir" in data_cfg:
        data_cfg["root_dir"] = _swap_base(data_cfg["root_dir"], cluster, "data")
    if "split_file" in data_cfg:
        data_cfg["split_file"] = _swap_base(data_cfg["split_file"], cluster, "code")

    if "simulated_dataset_path" in eval_cfg:
        eval_cfg["simulated_dataset_path"] = _swap_base(
            eval_cfg["simulated_dataset_path"], cluster, "data"
        )

    if "output_dir" in exp_cfg:
        exp_cfg["output_dir"] = _swap_base(exp_cfg["output_dir"], cluster, "code")
    if "pretrained_checkpoint" in exp_cfg:
        exp_cfg["pretrained_checkpoint"] = _swap_base(
            exp_cfg["pretrained_checkpoint"], cluster, "code"
        )

    return config
