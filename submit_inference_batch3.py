import submitit
import subprocess
import os


class InferenceBatch(submitit.helpers.Checkpointable):
    def __init__(self, exp_names, exp_base_dir, num_gpus, extra_args=None):
        self.exp_names = exp_names
        self.exp_base_dir = exp_base_dir
        self.num_gpus = num_gpus
        self.extra_args = extra_args or []

    def __call__(self):
        micromamba_path = "/home/rachelgordon/micromamba/etc/profile.d/mamba.sh"
        env_name = "recon_mri"

        gpus = ",".join(str(i) for i in range(self.num_gpus))
        extra = " ".join(self.extra_args)

        command_str = (
            f"source {micromamba_path} && "
            f"micromamba activate {env_name} && "
            f"python run_inference_batch.py "
            f"--exp_names {self.exp_names} "
            f"--exp_base_dir {self.exp_base_dir} "
            f"--gpus {gpus} "
            f"{extra}"
        )

        subprocess.run(command_str, shell=True, check=True, executable="/bin/bash")

    def checkpoint(self, *args, **kwargs):
        return submitit.helpers.DelayedSubmission(
            InferenceBatch(
                exp_names=self.exp_names,
                exp_base_dir=self.exp_base_dir,
                num_gpus=self.num_gpus,
                extra_args=self.extra_args,
            )
        )


def main():
    job_name = "inference_batch3"
    exp_names = "mc_36spf_baseline"
    exp_base_dir = "/home/rachelgordon/mri_recon/radial-breast-ddei/output"
    num_gpus = 1
    extra_args = [
        # "--disable_ssdu",
    ]

    log_dir = f"submitit_logs/{job_name}"
    os.makedirs(log_dir, exist_ok=True)

    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(
        slurm_partition="general",
        slurm_job_name=job_name,
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=8,
        slurm_gres=f"gpu:{num_gpus}",
        timeout_min=200,
        slurm_additional_parameters={"requeue": True},
        qos="burst",
        srun_args=["--cpu-bind=none"],
    )

    job = executor.submit(
        InferenceBatch(
            exp_names=exp_names,
            exp_base_dir=exp_base_dir,
            num_gpus=num_gpus,
            extra_args=extra_args,
        )
    )
    print(f"Submitted job with ID: {job.job_id}")


if __name__ == "__main__":
    main()
