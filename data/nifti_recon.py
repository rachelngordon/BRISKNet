"""Convert processed HDF5 volumes into per-slice NIfTI files. Run: python3 process_data/nifti_recon.py --help"""

import argparse
import os

import h5py
import nibabel as nib
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Convert a h5py file to NIfTI')
    
    # parser.add_argument('--dcm',
    #                     default='GRASP_anno00001_anon.dcm',
    #                     help='a dicom file with tags (not used in NIfTI conversion)')
    
    parser.add_argument('--dir',
                        default='/ess/scratch/sctrach1/rachelgordon',
                        help='directory where the processed h5py file is located')

    parser.add_argument('--h5py',
                        default='/fastMRI_breast_001_1.h5',
                        help='radial k-space data')

    parser.add_argument('--spokes_per_frame', type=int, default=13,
                        help='number of spokes per frame')
    
    parser.add_argument('--out_dir', type=str, default='/ess/scratch/scratch1/rachelgordon/reconresnet/fully_sampled_gt/',
                        help='number of spokes per frame')

    parser.add_argument('--partitions', type=int, default=83,
                        help='total number of partitions')

    parser.add_argument('--TE', type=float, default=1.8,
                        help='echo time (ms)')

    parser.add_argument('--TR', type=float, default=4.87,
                        help='repetition time (ms)')
    parser.add_argument('--keep_complex', type=bool, default=False,
                        help='whether or not to keep the values as complex in two separate channels')

    args = parser.parse_args()

    # Output directory for NIfTI files
    # OUT_DIR = args.h5py + '_NIFTI_processed'
    # OUT_DIR = OUT_DIR.split('.h5')[0]
    # pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    basename = os.path.basename(args.h5py)
    OUT_DIR = os.path.join(args.out_dir, basename)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("NIFTI out_dir: ", OUT_DIR)

    # Load the HDF5 file
    f = h5py.File(args.h5py + '_processed.h5', 'r')
    R = f['temptv'][:]
    f.close()

    print("R dtype: ", R.dtype)


    #R = np.squeeze(abs(R))
    if not args.keep_complex:
        R = abs(R)

    R = np.flip(R, axis=(-2, -1))  # upward orientation
    print("R shape: ", R.shape)

    N_z, N_t, N_y, N_x = R.shape
    #print(R.shape)

    # Normalize images
    R = R * 533 / np.amax(R)
    print(np.amin(R))
    print(np.amax(R))
    print("R dtype: ", R.dtype)

    # Create and save NIfTI files
    for t in range(N_t):
        print("t: ", t)
        for z in range(N_z):

            # Create NIfTI image

            if args.keep_complex:
                # Split real and imaginary parts
                real_part = np.real(R[z, t])
                imag_part = np.imag(R[z, t])

                # Stack along the channel dimension
                img_data = np.stack((real_part, imag_part), axis=0)  # Shape: (2, N_y, N_x)
                print(f"img_data shape: {img_data.shape}, dtype: {img_data.dtype}")
            
            else:
                img_data = R[z, t]
                print(f"img_data shape: {img_data.shape}, dtype: {img_data.dtype}")
            
           

            nifti_img = nib.Nifti1Image(img_data, affine=np.eye(4))

            # Save NIfTI file
            nifti_filename = os.path.join(OUT_DIR, f'slice_{z:03d}_frame_{t:03d}.nii')
            nib.save(nifti_img, nifti_filename)

            print(f'> slice {z:03d} frame {t:03d} saved as {nifti_filename}')

    print('> done')
