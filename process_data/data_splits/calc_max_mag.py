"""Compute the max magnitude across NIfTI files for SSIM scaling. Run: python3 process_data/data_splits/calc_max_mag.py --help"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False
    print("Optional dependency 'tqdm' not found. Progress bar disabled.")
    # Define a dummy tqdm if not found
    def tqdm(iterable, *args, **kwargs):
        return iterable

def find_max_magnitude_in_nifti_files(directory: Path) -> float:
    """
    Calculates the maximum absolute magnitude across all NIfTI files
    in a directory, assuming the last dimension is (real, imag).

    Args:
        directory: Path object pointing to the directory containing .nii or .nii.gz files.

    Returns:
        The maximum absolute magnitude found across all files.
        Returns -1.0 if no valid files are found or an error occurs early.
    """
    nifti_files = list(directory.rglob('*.nii')) + list(directory.rglob('*.nii.gz'))

    if not nifti_files:
        print(f"Error: No '.nii' or '.nii.gz' files found in {directory}", file=sys.stderr)
        return -1.0

    print(f"Found {len(nifti_files)} NIfTI files to process...")

    global_max_magnitude = 0.0
    processed_files = 0

    for file_path in tqdm(nifti_files, desc="Processing NIfTI files"):
        try:
            # Load the NIfTI file
            img = nib.load(file_path)
            # Get the data as a NumPy array, converting to float32 for safety
            data = img.get_fdata(dtype=np.float32)

            # --- Critical Assumption Check ---
            if data.shape[0] != 2:
                print(f"\nWarning: Skipping file {file_path.name}. "
                      f"Expected last dimension to be 2 (real, imag), but got shape {data.shape}",
                      file=sys.stderr)
                continue
            # --------------------------------

            # Extract real and imaginary parts
            real_part = data[0]
            imag_part = data[1]

            # Calculate absolute magnitude (element-wise)
            # Use np.abs for potentially complex results if input wasn't float32,
            # or calculate directly for performance.
            # magnitude = np.abs(real_part + 1j * imag_part) # Cleaner but potentially slower
            magnitude = np.sqrt(real_part**2 + imag_part**2) # Direct calculation

            # Find the maximum magnitude in this file
            file_max_magnitude = np.max(magnitude)

            # Update the global maximum
            if file_max_magnitude > global_max_magnitude:
               # print(f"\nNew max found: {file_max_magnitude:.4f} in file {file_path.name}") # Optional: uncomment for verbose output
               global_max_magnitude = file_max_magnitude

            processed_files += 1

        except nib.filebasedimages.ImageFileError as e:
            print(f"\nError loading NIfTI file {file_path.name}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"\nAn unexpected error occurred processing file {file_path.name}: {e}", file=sys.stderr)
            # Decide if you want to stop or continue
            # return -1.0 # Stop on any error
            continue # Skip this file and continue

    if processed_files == 0:
        print("Error: No valid NIfTI files with the expected format were processed.", file=sys.stderr)
        return -1.0

    print(f"\nProcessed {processed_files} valid files.")
    return global_max_magnitude

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the maximum absolute magnitude across a directory of NIfTI files.")
    parser.add_argument("nifti_dir", type=str, help="Directory containing the NIfTI ground truth files (.nii or .nii.gz).")

    args = parser.parse_args()

    input_directory = Path(args.nifti_dir)

    if not input_directory.is_dir():
        print(f"Error: Directory not found: {input_directory}", file=sys.stderr)
        sys.exit(1)

    max_mag = find_max_magnitude_in_nifti_files(input_directory)

    if max_mag >= 0:
        print(f"\n========================================================")
        print(f"Maximum Absolute Magnitude across all files: {max_mag}")
        print(f"Recommended SSIM data_range: {max_mag:.4f} (or slightly higher, e.g., {np.ceil(max_mag)})")
        print(f"========================================================")
    else:
        print("\nCalculation failed or no valid files were found.")
        sys.exit(1)
