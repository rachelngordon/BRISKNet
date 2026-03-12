"""Resample a stack of NIfTI slices to a target slice count. Run: python3 process_data/dro_matching/resample_slices.py (edit paths at bottom)"""

import glob
import os

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

def resample_slices_from_directory(input_dir, output_dir, new_slice_count):
    """
    Loads individual slice files from a directory, resamples the volume to a
    new number of slices, and saves them as individual files in a new directory.

    Args:
        input_dir (str): The directory containing the original slice files
                         (e.g., 'slice_001_frame_000.nii').
        output_dir (str): The directory where the resampled slices will be saved.
        new_slice_count (int): The desired number of slices in the output.
    """

    sample_id = os.path.basename(input_dir)
    output_path = os.path.join(output_dir, sample_id)
    os.makedirs(output_path, exist_ok=True)

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found at '{input_dir}'")
        return

    os.makedirs(output_dir, exist_ok=True)

    search_pattern = os.path.join(input_dir, 'slice_*_frame_*.nii')
    slice_files = sorted(glob.glob(search_pattern))

    if not slice_files:
        print(f"Error: No slice files found in '{input_dir}' matching the pattern.")
        return

    print(f"Found {len(slice_files)} original slices.")

    all_slices_data = []
    first_slice_img = nib.load(slice_files[0])
    affine = first_slice_img.affine
    header = first_slice_img.header

    for slice_file in slice_files:
        img = nib.load(slice_file)
        # --- THIS IS THE FIX ---
        # Squeeze the data to ensure it's 2D, removing dimensions of size 1
        slice_data = np.squeeze(img.get_fdata())
        all_slices_data.append(slice_data)

    original_volume = np.stack(all_slices_data, axis=-1)

    original_volume = np.abs(original_volume[0] + 1j * original_volume[1])
    
    # Let's check the shape to be sure it's 3D now
    if original_volume.ndim != 3:
        print(f"Error: Stacked volume is not 3D! Its shape is {original_volume.shape}")
        return
        
    original_slice_count = original_volume.shape[2]

    print(f"Resampling from {original_slice_count} slices to {new_slice_count} slices...")

    slice_zoom_factor = new_slice_count / original_slice_count
    zoom_factors = [1, 1, slice_zoom_factor]

    # This call should now work correctly
    resampled_volume = zoom(original_volume, zoom_factors, order=3)

    print(f"New volume shape: {resampled_volume.shape}")

    new_affine = np.copy(affine)
    new_affine[2, 2] = affine[2, 2] * (original_slice_count / new_slice_count)

    for i in range(resampled_volume.shape[2]):
        new_slice_data = resampled_volume[:, :, i]
        new_slice_img = nib.Nifti1Image(new_slice_data, new_affine, header)
        output_filename = f"slice_{i:03d}_frame_000.nii"
        file_save_path = os.path.join(output_path, output_filename)
        nib.save(new_slice_img, file_save_path)

    print(f"Successfully saved {resampled_volume.shape[2]} resampled slices to '{file_save_path}'.")



input_directory = '/ess/scratch/scratch1/rachelgordon/192_slices_1_timeframe/fastMRI_breast_001_2'
output_directory = '/ess/scratch/scratch1/rachelgordon/83_slices_1_timeframe'
target_slice_count = 83

resample_slices_from_directory(input_directory, output_directory, target_slice_count)
