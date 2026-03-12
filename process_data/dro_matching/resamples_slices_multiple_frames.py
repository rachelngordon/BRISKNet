"""Resample multi-frame NIfTI slices to a target slice count. Run: python3 process_data/dro_matching/resamples_slices_multiple_frames.py (edit paths at bottom)"""

import glob
import os
import re
from collections import defaultdict

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
def resample_slices_from_directory_4d(input_dir, output_dir, new_slice_count):
    """
    Loads individual slice files with multiple timeframes from a directory, 
    resamples the volume for each timeframe to a new number of slices, 
    and saves them as individual files in a new directory.

    Args:
        input_dir (str): The directory containing the original slice files
                         (e.g., 'slice_001_frame_000.nii', 'slice_001_frame_001.nii').
        output_dir (str): The directory where the resampled slices will be saved.
        new_slice_count (int): The desired number of slices in the output for each timeframe.
    """

    sample_id = os.path.basename(input_dir)
    output_path = os.path.join(output_dir, sample_id)
    os.makedirs(output_path, exist_ok=True)

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found at '{input_dir}'")
        return

    search_pattern = os.path.join(input_dir, 'slice_*_frame_*.nii')
    slice_files = sorted(glob.glob(search_pattern))

    if not slice_files:
        print(f"Error: No slice files found in '{input_dir}' matching the pattern.")
        return

    # --- MODIFICATION START: Group files by timeframe ---
    timeframes = defaultdict(list)
    for f in slice_files:
        # Extract frame number from filename
        match = re.search(r'frame_(\d+)', f)
        if match:
            frame_num = int(match.group(1))
            timeframes[frame_num].append(f)

    print(f"Found {len(slice_files)} total files across {len(timeframes)} timeframes.")
    # --- MODIFICATION END ---


    # --- MODIFICATION START: Loop over each timeframe ---
    for frame_num, frame_files in timeframes.items():
        print(f"\nProcessing timeframe {frame_num}...")

        all_slices_data = []
        first_slice_img = nib.load(frame_files[0])
        affine = first_slice_img.affine
        header = first_slice_img.header

        for slice_file in frame_files:
            img = nib.load(slice_file)
            slice_data = np.squeeze(img.get_fdata())
            all_slices_data.append(slice_data)

        original_volume = np.stack(all_slices_data, axis=-1)

        # Assuming the complex data handling is still needed
        if original_volume.shape[0] == 2: # Check if the first dimension is for real and imaginary parts
             original_volume = original_volume[0] + 1j * original_volume[1]


        if original_volume.ndim != 3:
            print(f"Error: Stacked volume for timeframe {frame_num} is not 3D! Its shape is {original_volume.shape}")
            continue
            
        original_slice_count = original_volume.shape[2]

        print(f"Resampling timeframe {frame_num} from {original_slice_count} slices to {new_slice_count} slices...")

        slice_zoom_factor = new_slice_count / original_slice_count
        zoom_factors = [1, 1, slice_zoom_factor]

        resampled_volume = zoom(original_volume.astype(float), zoom_factors, order=3) # Convert to float for zoom

        print(f"New volume shape for timeframe {frame_num}: {resampled_volume.shape}")

        new_affine = np.copy(affine)
        new_affine[2, 2] = affine[2, 2] * (original_slice_count / new_slice_count)

        for i in range(resampled_volume.shape[2]):
            new_slice_data = resampled_volume[:, :, i]
            new_slice_img = nib.Nifti1Image(new_slice_data, new_affine, header)
            
            # --- MODIFICATION: Use the correct frame number in the output filename ---
            output_filename = f"slice_{i:03d}_frame_{frame_num:03d}.nii"
            file_save_path = os.path.join(output_path, output_filename)
            nib.save(new_slice_img, file_save_path)

        print(f"Successfully saved {resampled_volume.shape[2]} resampled slices for timeframe {frame_num} to '{output_path}'.")
    # --- MODIFICATION END ---


input_directory = '/ess/scratch/scratch1/rachelgordon/192_slices_22_timeframes/fastMRI_breast_001_2'
output_directory = '/ess/scratch/scratch1/rachelgordon/83_slices_22_timeframes'
target_slice_count = 83

resample_slices_from_directory_4d(input_directory, output_directory, target_slice_count)
