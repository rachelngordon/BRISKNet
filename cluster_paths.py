import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SPLIT_FILE = REPO_ROOT / "data" / "split" / "data_split.json"

# Base directories for each cluster. The "code" base is used for config/output
# files and the "data" base for datasets and simulation assets.
CLUSTER_BASES = {
    "Randi": {
        "data": "/ess/scratch/scratch1/rachelgordon",
        "code": "/gpfs/data/karczmar-lab/workspaces/rachelgordon/breastMRI-recon/ddei",
    },
    "DSI": {
        "data": "/net/scratch2/rachelgordon",
        "code": str(REPO_ROOT),
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


def _normalize_split_file(path: str | None, cluster: str) -> str:
    """Resolve split-file path with a stable local fallback for moved repos."""
    candidate = _swap_base(path, cluster, "code")
    if candidate is None:
        candidate = str(DEFAULT_SPLIT_FILE)

    split_path = Path(candidate)
    if split_path.as_posix().endswith("/data/data_split.json") and DEFAULT_SPLIT_FILE.is_file():
        return str(DEFAULT_SPLIT_FILE)
    if split_path.as_posix().endswith("/data/split/data_split.json") and DEFAULT_SPLIT_FILE.is_file():
        return str(DEFAULT_SPLIT_FILE)
    if split_path.is_file():
        return str(split_path)

    raise FileNotFoundError(
        "split_file does not exist after cluster-path normalization: "
        f"{split_path}"
    )


def apply_cluster_paths(config: dict) -> dict:
    """Normalize config paths based on the chosen cluster."""
    cluster = config["experiment"]["cluster"]

    data_cfg = config["data"]
    eval_cfg = config["evaluation"]
    exp_cfg = config["experiment"]

    if "root_dir" in data_cfg:
        data_cfg["root_dir"] = _swap_base(data_cfg["root_dir"], cluster, "data")
    if "split_file" in data_cfg:
        data_cfg["split_file"] = _normalize_split_file(
            data_cfg["split_file"],
            cluster,
        )

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
