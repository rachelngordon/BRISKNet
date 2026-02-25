import submitit
import subprocess
import os


class InferenceBatch(submitit.helpers.Checkpointable):
    def __init__(self, exp_names, extra_args=None):
        self.exp_names = exp_names
        self.extra_args = extra_args or []

    def __call__(self):
        micromamba_path = "/home/rachelgordon/micromamba/etc/profile.d/mamba.sh"
        env_name = "recon_mri"
        extra = " ".join(self.extra_args)

        command_str = (
            f"source {micromamba_path} && "
            f"micromamba activate {env_name} && "
            f"python run_inference_new_dro.py "
            f"--exp_dir {self.exp_names} "
            f"{extra}"
        )

        subprocess.run(command_str, shell=True, check=True, executable="/bin/bash")

    def checkpoint(self, *args, **kwargs):
        return submitit.helpers.DelayedSubmission(
            InferenceBatch(
                exp_names=self.exp_names,
                extra_args=self.extra_args,
            )
        )


def main():
    job_name = "test_set_inference_36spf"
    exp_names = "/net/projects2/annawoodard/rachelgordon/experiments/ei_diffeo_36spf_slice_sampling"
    num_gpus = 1
    extra_args = [
        "--overwrite_logs",
        "--new_dro_root /net/scratch2/rachelgordon/dro_test_set",
        "--split_key test_dro",
        "--num_samples 25",
        "--skip_raw_grasp_metrics",
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
        timeout_min=700,
        slurm_additional_parameters={"requeue": True, "exclude": "k002"},
        srun_args=["--cpu-bind=none"],
    )

    job = executor.submit(
        InferenceBatch(
            exp_names=exp_names,
            extra_args=extra_args,
        )
    )
    print(f"Submitted job with ID: {job.job_id}")


if __name__ == "__main__":
    main()
