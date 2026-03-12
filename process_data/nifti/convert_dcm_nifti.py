"""Convert DICOM folders to NIfTI files. Run: python3 process_data/nifti/convert_dcm_nifti.py (edit paths at bottom)"""

import os

import nibabel as nib
import numpy as np
import pydicom

def convert_dicom_to_nifti(input_dir, output_dir):
    print(f"Processing input directory: {input_dir}")

    # Iterate through the subdirectories in the input directory
    for root, dirs, files in os.walk(input_dir):
        # Only process directories that contain DICOM files
        dicom_files = [f for f in files if f.endswith('.dcm')]
        if dicom_files:
            # Ensure the output directory exists for this specific scan
            base_name = os.path.basename(root).replace('_DCM', '')
            
            # Check if the corresponding fastMRI_breast_{ID}_2 directory already exists
            expected_output_dir = os.path.join(output_dir, base_name)
            print(expected_output_dir)

            if os.path.exists(expected_output_dir):
                print(f"Skipping {base_name} as {expected_output_dir} already exists.")
                continue

            output_scan_dir = os.path.join(output_dir, base_name)
            os.makedirs(output_scan_dir, exist_ok=True)

            # Sort and process each DICOM file
            for file_name in sorted(dicom_files):
                dcm_path = os.path.join(root, file_name)
                ds = pydicom.dcmread(dcm_path)

                # Extract image data
                img_array = ds.pixel_array.astype(np.float32)

                # Apply rescale if necessary
                if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                    img_array = img_array * ds.RescaleSlope + ds.RescaleIntercept

                # Convert to NIfTI image
                nifti_img = nib.Nifti1Image(img_array, np.eye(4))

                # Create output filename by removing "_DCM" and ".dcm"
                output_filename = file_name.replace('_DCM', '').replace('.dcm', '.nii')

                # Save the NIfTI file
                output_path = os.path.join(output_scan_dir, output_filename)
                nib.save(nifti_img, output_path)
                print(f"Saved NIfTI file: {output_path}")


# Example usage:
input_dir = "/ess/scratch/scratch1/rachelgordon/fastMRI_breast_IDS_150_300_DCM"
output_dir = "/ess/scratch/scratch1/rachelgordon/reconresnet/fully_sampled_gt"
convert_dicom_to_nifti(input_dir, output_dir)
