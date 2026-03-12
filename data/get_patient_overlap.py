"""Map fastMRI patient splits to DRO sample IDs and report overlap."""

import json

import pandas as pd

# --- Configuration ---
SPLITS_FILE_PATH = (
    "/gpfs/data/karczmar-lab/workspaces/rachelgordon/breastMRI-recon/ddei/data/patient_splits.json"
)

OVERLAP_CSV_PATH = "DROSubID_vs_fastMRIbreastID.csv"
FASTMRI_COL = "fastMRIbreast"
DRO_COL = "DRO"

# --- Main Logic ---

def find_dro_samples_in_splits(splits_path, csv_path):
    """Return DRO IDs per split and any unused DRO IDs, or None on error."""
    # Load patient splits and counts
    try:
        with open(splits_path, 'r') as f:
            patient_splits = json.load(f)
        
        # Load train, validation, and test sets
        train_patients = patient_splits.get('train', [])
        val_patients = patient_splits.get('val', patient_splits.get('validation', []))
        test_patients = patient_splits.get('test', [])
        
        # Capture the total number of patients in each split
        total_train_count = len(train_patients)
        total_val_count = len(val_patients)
        total_test_count = len(test_patients)
        
        print(
            f"Successfully loaded {total_train_count} train, {total_val_count} val, "
            f"and {total_test_count} test patients from splits file."
        )
    except FileNotFoundError:
        print(f"Error: The JSON splits file '{splits_path}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{splits_path}'. Check the file format.")
        return None

    # Load overlap CSV
    try:
        overlap_df = pd.read_csv(csv_path)
        print(f"Successfully loaded the overlap data from '{csv_path}'.")
    except FileNotFoundError:
        print(f"Error: The CSV overlap file '{csv_path}' was not found.")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

    # Build lookup map and collect all DRO IDs
    fastmri_to_dro_map = {}
    for _, row in overlap_df.iterrows():
        fastmri_id_num = row[FASTMRI_COL]
        dro_id = row[DRO_COL]
        full_fastmri_id = f"fastMRI_breast_{int(fastmri_id_num):03d}"
        fastmri_to_dro_map[full_fastmri_id] = dro_id

    all_dro_ids = set(overlap_df[DRO_COL].unique())
    print(f"Created a lookup map with {len(fastmri_to_dro_map)} entries.")
    print(f"Found {len(all_dro_ids)} unique DRO samples in the overlap CSV.")

    # Map each split to DRO IDs
    dro_train = sorted([fastmri_to_dro_map[p] for p in train_patients if p in fastmri_to_dro_map])
    dro_val = sorted([fastmri_to_dro_map[p] for p in val_patients if p in fastmri_to_dro_map])
    dro_test = sorted([fastmri_to_dro_map[p] for p in test_patients if p in fastmri_to_dro_map])
    
    # DRO samples not used in any split
    found_dro_ids = set(dro_train + dro_val + dro_test)
    unused_dro_ids = all_dro_ids - found_dro_ids

    # Return results
    results = {
        "total_train_patients": total_train_count,
        "total_val_patients": total_val_count,
        "total_test_patients": total_test_count,
        "dro_train": dro_train,
        "dro_val": dro_val,
        "dro_test": dro_test,
        "dro_unused": sorted(list(unused_dro_ids)),
    }
    return results


# --- Execute the script and print results ---
if __name__ == "__main__":
    results = find_dro_samples_in_splits(SPLITS_FILE_PATH, OVERLAP_CSV_PATH)

    if results:
        print("\n" + "="*60)
        print("                     RESULTS SUMMARY")
        print("="*60)
        
        # --- Training Set Information ---
        print("\n--- Training Set ---")
        print(f"Total patients in original training split: {results['total_train_patients']}")
        print(f"Mapped DRO Samples in Training Set ({len(results['dro_train'])} found):")
        print(results['dro_train'])

        # --- Validation Set Information ---
        print("\n--- Validation Set ---")
        print(f"Total patients in original validation split: {results['total_val_patients']}")
        print(f"Mapped DRO Samples in Validation Set ({len(results['dro_val'])} found):")
        print(results['dro_val'])
        
        # --- Test Set Information ---
        print("\n--- Test Set ---")
        print(f"Total patients in original test split: {results['total_test_patients']}")
        print(f"Mapped DRO Samples in Test Set ({len(results['dro_test'])} found):")
        print(results['dro_test'])

        # --- Unused DRO Samples Information ---
        print("\n--- Unused DRO Samples ---")
        print(f"DRO Samples from CSV not found in any split ({len(results['dro_unused'])}):")
        print(results['dro_unused'])

        print("\n" + "="*60)
