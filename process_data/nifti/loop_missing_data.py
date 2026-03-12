"""Run a batch NIfTI conversion loop for a hardcoded patient list. Run: python3 process_data/nifti/loop_missing_data.py"""

import re
import subprocess

# List of patient directories
patients = [
    "fastMRI_breast_014_2", "fastMRI_breast_041_2", "fastMRI_breast_057_2", "fastMRI_breast_055_2",
    "fastMRI_breast_025_2", "fastMRI_breast_023_2", "fastMRI_breast_018_2", "fastMRI_breast_056_2",
    "fastMRI_breast_021_2", "fastMRI_breast_020_2", "fastMRI_breast_049_2", "fastMRI_breast_051_2",
    "fastMRI_breast_024_2", "fastMRI_breast_039_2", "fastMRI_breast_011_2", "fastMRI_breast_036_2",
    "fastMRI_breast_035_2", "fastMRI_breast_026_2", "fastMRI_breast_017_2", "fastMRI_breast_050_2",
    "fastMRI_breast_027_2", "fastMRI_breast_015_2", "fastMRI_breast_032_2", "fastMRI_breast_054_2",
    "fastMRI_breast_043_2", "fastMRI_breast_033_2", "fastMRI_breast_038_2", "fastMRI_breast_022_2",
    "fastMRI_breast_045_2", "fastMRI_breast_058_2", "fastMRI_breast_030_2", "fastMRI_breast_040_2",
    "fastMRI_breast_034_2", "fastMRI_breast_053_2", "fastMRI_breast_037_2", "fastMRI_breast_031_2",
    "fastMRI_breast_044_2", "fastMRI_breast_013_2", "fastMRI_breast_028_2", "fastMRI_breast_012_2",
    "fastMRI_breast_IDS_001_010", "fastMRI_breast_016_2", "fastMRI_breast_048_2", "fastMRI_breast_042_2",
    "fastMRI_breast_046_2", "fastMRI_breast_019_2", "fastMRI_breast_052_2", "fastMRI_breast_047_2",
    "fastMRI_breast_059_2", "fastMRI_breast_029_2", "fastMRI_breast_060_2"
]

# Base directory
base_dir = "/ess/scratch/scratch1/rachelgordon/fastMRI_breast_data"

# Function to determine the parent directory based on the patient ID
def get_parent_dir(patient):
    # Extract the numeric ID from the patient name
    match = re.search(r"\d+", patient)
    if not match:
        raise ValueError(f"Invalid patient name format: {patient}")

    patient_id = int(match.group().lstrip("0"))  # Remove leading zeros and convert to integer
    start_id = (patient_id - 1) // 10 * 10 + 1
    end_id = start_id + 9
    parent_dir = f"fastMRI_breast_IDS_{start_id:03d}_{end_id:03d}"
    return parent_dir

# Main loop to process each patient directory
for patient in patients:
    try:
        # Get the parent directory
        parent_dir = get_parent_dir(patient)

        # Construct the input file path
        input_file = f"{base_dir}/{parent_dir}/{patient[:-2]}_2.h5"

        # Construct the shell command
        command = [
            "bash", "loop_single_data_nifti.sh",
            input_file,
            "/ess/scratch/scratch1/rachelgordon/reconresnet/fully_sampled_gt",
            "True", "2", "72", "0", "192", "100"
        ]

        # Print the command being executed
        print(f"Running: {' '.join(command)}")

        # Execute the command
        subprocess.run(command, check=True)

    except Exception as e:
        print(f"Error processing {patient}: {e}")
