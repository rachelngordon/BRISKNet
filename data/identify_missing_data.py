"""List missing fastMRI IDs (1-300) based on files present in a directory."""

import os

# Directory containing files like fastMRI_breast_XXX_2.h5
DIRECTORY = "/ess/scratch/scratch1/rachelgordon/dce-8tf/binned_kspace"
ID_RANGE = range(1, 301)
SUFFIX = "_2.h5"
PREFIX = "fastMRI_breast_"

# List all files in the directory
files = os.listdir(DIRECTORY)

# Extract patient IDs from filenames like: fastMRI_breast_XXX_2.h5
present_ids = set()
for filename in files:
    if filename.startswith(PREFIX) and filename.endswith(SUFFIX):
        try:
            id_str = filename.split('_')[2]
            patient_id = int(id_str)
            present_ids.add(patient_id)
        except (IndexError, ValueError):
            pass  # Skip files not matching expected pattern

# Compare with expected IDs 1 to 300
expected_ids = set(ID_RANGE)
missing_ids = sorted(expected_ids - present_ids)

# Print the missing IDs
print(f"Missing {len(missing_ids)} IDs:")
print(missing_ids)
