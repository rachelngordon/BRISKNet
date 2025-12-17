
import submitit
import subprocess
import os

class Trainer(submitit.helpers.Checkpointable):
    """
    A Checkpointable class to handle training and resubmission.
    """
    def __init__(self, exp_name, config_path, num_gpus, from_checkpoint=False):
        self.exp_name = exp_name
        self.config_path = config_path
        self.num_gpus = num_gpus
        self.from_checkpoint = from_checkpoint

    def __call__(self):
        """
        Execute the training script.
        """
        # Activate your Micromamba environment
        # CORRECTED THE TYPO IN THIS PATH
        micromamba_path = "/home/rachelgordon/micromamba/etc/profile.d/mamba.sh"
        env_name = "recon_mri"
        
        command_str = (
            f"source {micromamba_path} && "
            f"micromamba activate {env_name} && "
            f"torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 "
            f"--nproc_per_node={self.num_gpus} "
            f"train_zf.py "
            f"--config {self.config_path} "
            f"--exp_name {self.exp_name} "
        )

        if self.from_checkpoint:
            command_str += " --from_checkpoint True"

        # Using shell=True to handle the source and && operators
        subprocess.run(command_str, shell=True, check=True, executable='/bin/bash')

    def checkpoint(self, *args, **kwargs):
        """
        This method is called by submitit when the job is about to time out.
        It returns a DelayedSubmission object for proper requeueing.
        """
        new_trainer_instance = Trainer(exp_name=self.exp_name, config_path=self.config_path, num_gpus=self.num_gpus, from_checkpoint=True)
        return submitit.helpers.DelayedSubmission(new_trainer_instance)

def main():
    # --- Executor Configuration ---
    job_name = "ei_warp"
    config_path = 'configs/config_ei_warp.yaml'
    num_gpus = 4

    log_dir = f"submitit_logs/{job_name}"
    os.makedirs(log_dir, exist_ok=True)

    executor = submitit.AutoExecutor(folder=log_dir)

    # --- SLURM Parameter Configuration --
    executor.update_parameters(
        slurm_partition="general",
        slurm_job_name=job_name,
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=8,                       # 8 CPUs for 4 GPUs is reasonable
        slurm_gres=f"gpu:h200:{num_gpus}",     # 4× H200 on a single node
        timeout_min=700,

        # IMPORTANT: no cpu_bind here anymore, this only affects sbatch
        # and your sbatch doesn't support --cpu-bind
        # slurm_additional_parameters={"export": "ALL,SLURM_CPU_BIND=off"},

        # Instead: tell the *srun* that submitit uses not to do CPU binding
        srun_args=["--cpu-bind=none"],
    )

    # --- Job Submission ---
    initial_trainer = Trainer(exp_name=job_name, config_path=config_path, num_gpus=num_gpus, from_checkpoint=False)
    job = executor.submit(initial_trainer)

    print(f"Submitted job with ID: {job.job_id}")

if __name__ == "__main__":
    main()