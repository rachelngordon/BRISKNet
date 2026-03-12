"""Convert reconstructed HDF5 volumes into DICOM series. Run: python3 process_data/dicom/dcm_recon.py --help"""

import argparse
import os
from datetime import datetime, timedelta

import h5py
import numpy as np
import pydicom
# from pydicom import Dataset
from pydicom.datadict import add_dict_entry
# from pydicom.dataset import FileDataset, FileMetaDataset
# from pydicom.uid import UID


#DIR = os.path.dirname(os.path.realpath(__file__))
#print("dcm_recon dir: ", DIR)

# %%
if __name__ == "__main__":

    # %%
    parser = argparse.ArgumentParser('convert a h5py file to dicom')
    
    parser.add_argument('--dcm',
                        default='GRASP_anno00001_anon.dcm',
                        help='a dicom file with tags')
    
    parser.add_argument('--dir',
                        default='/ess/scratch/sctrach1/rachelgordon',
                        help='a dicom file with tags')

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

    args = parser.parse_args()

    # %%
    # read in the original dicom file
    ds = pydicom.dcmread(args.dcm)


    # OUT_DIR = args.h5py + '_DCM_processed'
    # OUT_DIR = OUT_DIR.split('.h5')[0]
    # pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    #print('> reconstructed files are stored in: ', OUT_DIR)
    basename = os.path.basename(args.h5py)
    OUT_DIR = os.path.join(args.out_dir, basename)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(args.h5py)

    f = h5py.File(args.h5py + '_processed.h5', 'r')
    R = f['temptv'][:]
    f.close()
    print(R.shape)
    print("R dtype: ", R.dtype)

    #R = np.squeeze(abs(R))
    #R = np.squeeze(R)

    print(R.shape)
    R = np.flip(R, axis=(-2, -1)) # upward orientation

    N_z, N_t, N_y, N_x = R.shape
    print(R.shape)
    print("R dtype: ", R.dtype)

    # normalize images
    R = R * 533 / np.amax(R)

    print(np.amin(R))
    print(np.amax(R))


    slice_thickness = ds[0x00180050].value
    print('> slice_thcikness: ', slice_thickness)

    # Set creation date/time
    dt = datetime.now()

    for t in range(N_t):

        # 288 is the total number of spokes
        dt_delay = dt + timedelta(minutes=t * 2.5 * args.spokes_per_frame / 288)

        for z in range(N_z):

            ds.ContentDate = dt.strftime('%Y%m%d')

            ds.ContentTime = dt_delay.strftime('%H%M%S.%f')
            ds.AcquisitionTime  = dt_delay.strftime('%H%M%S.%f') 

            # Combine real and imaginary parts for complex data
            real_part = np.real(R[z, t]).astype(np.float32)
            imag_part = np.imag(R[z, t]).astype(np.float32)
            complex_data = np.stack((real_part, imag_part), axis=-1)  # Shape (H, W, 2)
            ds.PixelData = complex_data.tobytes()

            print("complex data shape: ", complex_data.shape)

            # Add custom metadata for complex data shape and type
            ds.Rows = complex_data.shape[0]  # Height of the image
            ds.Columns = complex_data.shape[1]  # Width of the image
            ds.NumberOfFrames = 1  # Single frame per file
            ds.SamplesPerPixel = 2  # Real and imaginary parts
            ds.BitsAllocated = 32  # Each sample is 32 bits (float32)
            ds.BitsStored = 32
            ds.HighBit = 31
            ds.PixelRepresentation = 0  # Unsigned integer representation
            ds.PhotometricInterpretation = "MONOCHROME2"


            #ds.PixelData = R[z, t].astype(np.uint16).tobytes()
            ds['PixelData'].VR = 'OW'
            ds.is_little_endian = True
            ds.is_implicit_VR = False

            slice_loc = (- N_z // 2 + z) * slice_thickness

            ds[0x00100020].value = 'trial'  # PatientID
            ds[0x00201041].value = slice_loc  # SliceLocation
            ds.ImagePositionPatient = [-160.0, -160.0, slice_loc]  # ImagePosition
            ds[0x00200013].value = t * N_z + (z + 1)  # InstanceNumber
            ds[0x00200010].value = '1'  # StudyID
            ds[0x00080018].value = str(t * N_z + (z + 1))  # Unique SOP Instance UID !
            ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

            add_dict_entry(0x00186666, "DS", "TimeStamp", "Time Stamp", VM='1')
            ds.TimeStamp = str(t * N_z * args.TR * args.spokes_per_frame)

            print('> slice ' + str(z).zfill(3) + ' frame ' + str(t).zfill(3) + ' InstanceNumber ' + str(t * N_z + (z + 1)).zfill(4))
            print('  ds.ImagePositionPatient: ', ds.ImagePositionPatient)

            ds.save_as(OUT_DIR + '/slice_' + str(z).zfill(3) + '_frame_' + str(t).zfill(3) + '.dcm')
            print(f"file saved to {OUT_DIR + '/slice_' + str(z).zfill(3) + '_frame_' + str(t).zfill(3) + '.dcm'}")

    print('> done')
