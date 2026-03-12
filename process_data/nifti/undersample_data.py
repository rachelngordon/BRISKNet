"""Undersample k-space HDF5 files by random spoke masking. Run: python3 process_data/nifti/undersample_data.py"""

import os

import h5py
import numpy as np

def process_file(file_path, output_dir, percentage_to_keep=0.1):
    """
    Process the HDF5 file to undersample k-space data and save to the output directory.
    
    Args:
        file_path (str): Path to the input HDF5 file.
        output_dir (str): Directory to save the processed output.
        percentage_to_keep (float): Percentage of spokes to keep.
    """


    # Open the HDF5 file
    with h5py.File(file_path, 'r') as f:
        # Load the datasets
        kspace = f['kspace'][:]
        #temptv = f['temptv'][:]

    # Undersample by masking a random percentage of spokes
    num_spokes = kspace.shape[1]  # Number of spokes
    mask = np.zeros((kspace.shape[0], num_spokes, kspace.shape[2], kspace.shape[3], kspace.shape[4]), dtype=np.float32)
    
    # Randomly choose which spokes to keep
    num_spokes_to_keep = int(num_spokes * percentage_to_keep)
    indices_to_keep = np.random.choice(num_spokes, num_spokes_to_keep, replace=False)
    
    # Set the selected spokes to 1 in the mask
    mask[:, indices_to_keep, :, :, :] = 1
    
    # Apply the mask to the k-space data
    undersampled_kspace_data = kspace * mask
    
    # Check the shape remains the same
    print("Original shape:", kspace.shape)
    print("Undersampled shape:", undersampled_kspace_data.shape)
    
    # Define output file path
    output_file_name = os.path.basename(file_path).replace('.h5', '_undersampled.h5')
    output_file_path = os.path.join(output_dir, output_file_name)
    
    # Create or open the HDF5 file
    with h5py.File(output_file_path, 'w') as h5f:
        # Create a dataset for the undersampled k-space data
        h5f.create_dataset('kspace', data=undersampled_kspace_data)
        
        # Optionally, save the mask or any other relevant information
        h5f.create_dataset('mask', data=mask)
        h5f.attrs['description'] = 'Undersampled k-space data'
        h5f.attrs['percentage_to_keep'] = percentage_to_keep
    
    print(f"Undersampled k-space data saved to {output_file_path}")

def main():
    base_dir = '/gpfs/data/karczmar-lab/workspaces/rachelgordon/demo_dce_recon'  # Base directory containing sample folders
    
    # List of sample directories to process
    #sample_dirs = [f'fastMRI_breast_{str(i).zfill(3)}_1' for i in range(1, 11)] + [f'fastMRI_breast_{str(i).zfill(3)}_2' for i in range(1, 11)]
    sample_dirs = [f'fastMRI_breast_{str(i).zfill(3)}_2' for i in range(1, 11)]
    
    for sample_dir in sample_dirs:
        input_dir = os.path.join(base_dir, sample_dir)
        
        # Process each .h5 file in the sample directory
        for file_name in os.listdir(input_dir):
            if file_name.endswith('.h5') and not file_name.endswith('_processed.h5') and not file_name.endswith('_undersampled.h5'):
                file_path = os.path.join(input_dir, file_name)
                process_file(file_path, input_dir)

if __name__ == '__main__':
    main()
