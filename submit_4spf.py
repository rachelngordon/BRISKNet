import os
import subprocess
import submitit


def _slurm_first_hostname():
    nodelist = os.environ.get("SLURM_NODELIST") or os.environ.get("SLURM_JOB_NODELIST")
    if not nodelist:
        return "127.0.0.1"
    try:
        output = subprocess.check_output(
            ["scontrol", "show", "hostnames", nodelist], text=True
        )
        for line in output.splitlines():
            host = line.strip()
            if host:
                return host
    except Exception:
        pass

    head = nodelist.split(",")[0]
    if "[" in head and "]" in head:
        prefix = head.split("[", 1)[0]
        inside = head.split("[", 1)[1].split("]", 1)[0]
        first = inside.split(",", 1)[0]
        if "-" in first:
            first = first.split("-", 1)[0]
        return f"{prefix}{first}"
    return head


def _default_master_port():
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id and job_id.isdigit():
        return 29500 + (int(job_id) % 2000)
    return 29500


class Trainer(submitit.helpers.Checkpointable):
    """
    A Checkpointable class to handle training and resubmission.
    """

    def __init__(self, exp_name, config_path, num_nodes, gpus_per_node):
        self.exp_name = exp_name
        self.config_path = config_path
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node

    def __call__(self):
        """
        Execute the training script.
        """
        micromamba_path = "/home/rachelgordon/micromamba/etc/profile.d/mamba.sh"
        env_name = "recon_mri"

        master_addr = _slurm_first_hostname()
        master_port = _default_master_port()
        node_rank = int(os.environ.get("SLURM_NODEID", "0"))
        rdzv_id = os.environ.get("SLURM_JOB_ID", "0")

        command_str = (
            f"source {micromamba_path} && "
            f"micromamba activate {env_name} && "
            f"export NCCL_IB_DISABLE=1 && "
            f"export NCCL_TIMEOUT=3600 && "
            f"export NCCL_BLOCKING_WAIT=1 && "
            f"export NCCL_ASYNC_ERROR_HANDLING=1 && "
            f"torchrun --rdzv-backend=c10d "
            f"--rdzv-endpoint={master_addr}:{master_port} "
            f"--rdzv-id={rdzv_id} "
            f"--nnodes={self.num_nodes} "
            f"--node_rank={node_rank} "
            f"--nproc_per_node={self.gpus_per_node} "
            f"train_zf_multinode.py "
            f"--config {self.config_path} "
            f"--exp_name {self.exp_name} "
        )

        subprocess.run(command_str, shell=True, check=True, executable="/bin/bash")

    def checkpoint(self, *args, **kwargs):
        """
        Called by submitit when the job is about to time out.
        """
        new_trainer = Trainer(
            exp_name=self.exp_name,
            config_path=self.config_path,
            num_nodes=self.num_nodes,
            gpus_per_node=self.gpus_per_node,
        )
        return submitit.helpers.DelayedSubmission(new_trainer)


def main():
    # --- Executor Configuration ---
    job_name = "ei_4spf_slice_sampling_all_transforms"
    config_path = "/home/rachelgordon/mri_recon/radial-breast-ddei/configs/config_sampling_4spf.yaml"
    num_nodes = 4
    gpus_per_node = 4

    log_dir = f"submitit_logs/{job_name}"
    os.makedirs(log_dir, exist_ok=True)

    executor = submitit.AutoExecutor(folder=log_dir)

    # --- SLURM Parameter Configuration ---
    executor.update_parameters(
        slurm_partition="general",
        slurm_job_name=job_name,
        nodes=num_nodes,
        tasks_per_node=1,
        cpus_per_task=8,
        slurm_gres=f"gpu:{gpus_per_node}",
        timeout_min=700,
        slurm_additional_parameters={"requeue": True, "exclude": "k002", "constraint": "a100|h100|h200"},
        srun_args=["--cpu-bind=none"],
    )

    # --- Job Submission ---
    initial_trainer = Trainer(
        exp_name=job_name,
        config_path=config_path,
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
    )
    job = executor.submit(initial_trainer)

    print(f"Submitted job with ID: {job.job_id}")


if __name__ == "__main__":
    main()
