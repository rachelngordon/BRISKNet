"""Summarize lesion and laterality counts per split using DRO-to-fastMRI mappings."""

import json
import re

import pandas as pd

# --- File paths ---
SPLIT_PATH = "data_split.json"
MAPPING_PATH = "DROSubID_vs_fastMRIbreastID.csv"
DEMOGRAPHICS_PATH = "breast_fastMRI_final.xlsx"
PATIENT_COL = "Patient Coded Name"
LESION_COL = "Lesion status (0 = negative, 1= malignancy, 2= benign)"
LATERALITY_COL = "Laterality (1=right, 2=left)"

# Load data split
with open(SPLIT_PATH, "r") as f:
    splits = json.load(f)

# Load mapping file (fastMRIbreast <-> DRO)
mapping_df = pd.read_csv(MAPPING_PATH)
dro_to_fmri = dict(zip(mapping_df["DRO"], mapping_df["fastMRIbreast"]))

# Convert DRO-style IDs to fastMRI IDs
def extract_dro_id(s):
    match = re.search(r"sub(\d+)", s)
    return int(match.group(1)) if match else None

def dro_list_to_fmri_ids(dro_ids):
    return [
        f"fastMRI_breast_{dro_to_fmri[extract_dro_id(dro_id)]:03d}"
        for dro_id in dro_ids
    ]


test_fmri_ids = dro_list_to_fmri_ids(splits["test_dro"])
val_fmri_ids = dro_list_to_fmri_ids(splits["val_dro"])
train_fmri_ids = splits["train"]

# --- Load demographics ---
demo_df = pd.read_excel(DEMOGRAPHICS_PATH)

# Normalize column strings for safe matching
demo_df.columns = demo_df.columns.str.strip()
demo_df[PATIENT_COL] = demo_df[PATIENT_COL].str.strip()

# Helper to compute stats
def compute_counts(df_subset):
    total = len(df_subset)
    no_lesion = (df_subset[LESION_COL] == 0).sum()
    benign = (df_subset[LESION_COL] == 2).sum()
    malignant = (df_subset[LESION_COL] == 1).sum()

    malignant_right = ((df_subset[LESION_COL] == 1) & (df_subset[LATERALITY_COL] == 1)).sum()
    malignant_left = ((df_subset[LESION_COL] == 1) & (df_subset[LATERALITY_COL] == 2)).sum()

    return {
        "Total Patients": total,
        "No Lesion": no_lesion,
        "Benign": benign,
        "Malignant": malignant,
        "Malignant Right Breast": malignant_right,
        "Malignant Left Breast": malignant_left,
    }

# Filter by split
train_df = demo_df[demo_df[PATIENT_COL].isin(train_fmri_ids)]
val_df = demo_df[demo_df[PATIENT_COL].isin(val_fmri_ids)]
test_df = demo_df[demo_df[PATIENT_COL].isin(test_fmri_ids)]

# Compute stats
train_stats = compute_counts(train_df)
val_stats = compute_counts(val_df)
test_stats = compute_counts(test_df)

# --- Print summary ---
summary = {
    "Train": train_stats,
    "Validation": val_stats,
    "Test": test_stats
}

for split, stats in summary.items():
    print(f"\n--- {split} ---")
    for k, v in stats.items():
        print(f"{k}: {v}")
