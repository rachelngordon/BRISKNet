"""Reconstruct DCE volumes from radial k-space in NIfTI workflows. Run: python3 process_data/nifti/dce_recon.py --help"""

import argparse
import os

import cupy as cp
import h5py
import numpy as np
import sigpy as sp
import torch

from sigpy.mri import app


#DIR = os.path.dirname(os.path.realpath(__file__))

# %%
def get_traj(args, csmaps=False, N_spokes=13, N_time=1, base_res=320, gind=1):

    N_tot_spokes = N_spokes * N_time

    N_samples = base_res * 2

    base_lin = np.arange(N_samples).reshape(1, -1) - base_res

    tau = 0.5 * (1 + 5**0.5)
    base_rad = np.pi / (gind + tau - 1)

    base_rot = np.arange(N_tot_spokes).reshape(-1, 1) * base_rad

    traj = np.zeros((N_tot_spokes, N_samples, 2))
    traj[..., 0] = np.cos(base_rot) @ base_lin
    traj[..., 1] = np.sin(base_rot) @ base_lin

    traj = traj / 2

    traj = traj.reshape(N_time, N_spokes, N_samples, 2)

    if args.spokes_per_frame == 288 and csmaps == False:
        return traj
    else:
        return np.squeeze(traj)

# %% compute coil sensitivity maps
def get_coil(ksp, args, device=sp.Device(-1)):

    N_coils, N_spokes, N_samples = ksp.shape

    base_res = N_samples // 2

    ishape = [N_coils] + [base_res] * 2

    traj = get_traj(args, csmaps=True, N_spokes=N_spokes, N_time=1,
                    base_res=base_res, gind=1)

    dcf = (traj[..., 0]**2 + traj[..., 1]**2)**0.5

    F = sp.linop.NUFFT(ishape, traj)

    cim = F.H(ksp * dcf)
    cim = sp.fft(cim, axes=(-2, -1))

    mps = app.EspiritCalib(cim, device=device).run()
    # print(type(mps))

    mps = sp.to_device(mps)
    # print("After to device: ", type(mps))

    return mps


def undersample_data(kspace, out_dir, data_path, percentage_to_keep=10):
    """
    Undersample k-space data according to a given percentage of spokes to keep.
    
    Args:
        kspace (arr): Fully sampled k-space data.
        percentage_to_keep (float): Percentage of spokes to keep.
    """
    # convert percentage to a decimal
    percentage_to_keep = percentage_to_keep / 100

    # Undersample by masking a random percentage of spokes
    num_spokes = kspace.shape[1]  # Number of spokes
    mask = np.zeros((kspace.shape[0], num_spokes, kspace.shape[2], kspace.shape[3], kspace.shape[4]), dtype=np.float32)
    
    # Randomly choose which spokes to keep
    num_spokes_to_keep = round(num_spokes * percentage_to_keep)
    indices_to_keep = np.random.choice(num_spokes, num_spokes_to_keep, replace=False)
    
    # Set the selected spokes to 1 in the mask
    mask[:, indices_to_keep, :, :, :] = 1
    
    # Apply the mask to the k-space data
    undersampled_kspace_data = kspace * mask
    
    # Check the shape remains the same
    print("Original shape:", kspace.shape)
    print("Undersampled shape:", undersampled_kspace_data.shape)

    # Define output file path

    output_file_name = os.path.basename(data_path).replace('.h5', '_undersampled.h5')
    output_file_path = os.path.join(out_dir, output_file_name)

    
    # Create or open the HDF5 file
    with h5py.File(output_file_path, 'w') as h5f:
        # Create a dataset for the undersampled k-space data
        h5f.create_dataset('kspace', data=undersampled_kspace_data)
        
        # save the mask and other relevant information
        h5f.create_dataset('mask', data=mask)
        h5f.attrs['description'] = 'Undersampled k-space data'
        h5f.attrs['percentage_to_keep'] = percentage_to_keep
    
    print(f"Undersampled k-space data saved to {output_file_path}")

    return undersampled_kspace_data


# %%
if __name__ == "__main__":

    # %% parse
    parser = argparse.ArgumentParser(description='run dce reconstruction.')

    parser.add_argument('--dir', type=str,
                        default='fastMRI_breast_001_1',
                        help='the directory where raw .h5 data is stored')
    
    parser.add_argument('--out_dir', type=str,
                        default='',
                        help='the directory where kspace data and mask are stored')
    
    parser.add_argument('--save_cs_maps', type=bool,
                        default='',
                        help='whether or not to save coil sensitivity maps')

    parser.add_argument('--data',
                        default='fastMRI_breast_001_1.h5',
                        help='radial k-space data')

    parser.add_argument('--spokes_per_frame', type=int, default=12,
                        help='number of spokes per frame')

    parser.add_argument('--slice_idx', type=int, default=0,
                        help='which slice index for the reconstruction to begin with')

    parser.add_argument('--slice_inc', type=int, default=1,
                        help='number of slices to be reconstructed')
    
    parser.add_argument('--per_spokes', type=int, default=100,
                        help='percentage of spokes to keep when undersampling')

    parser.add_argument('--center_partition', type=int, default=31,
                        help='the center partition index [default: 31]')

    parser.add_argument('--images_per_slab', type=int, default=192,
                        help='total number of images per slab [default: 192]')
    parser.add_argument('--keep_complex', type=bool, default=False,
                        help='whether or not to keep the values as complex in two separate channels')

    args = parser.parse_args()

    device = sp.Device(0 if torch.cuda.is_available() else -1)
    # print('> device ', device)

    #OUT_DIR = args.dir
    # print("spokes per frame: ", args.spokes_per_frame)


    # %% read in k-space data
    IN_DIR = args.dir + '/' + args.data
    #print('> read in data ', IN_DIR)
    f = h5py.File(IN_DIR, 'r')
    ksp_f = f['kspace'][:].T
    ksp_f = np.transpose(ksp_f, (4, 3, 2, 1, 0))
    print('> kspace shape ', ksp_f.shape)
    f.close()

    # undersample k-space data 
    if args.per_spokes != 100:
        ksp_f = undersample_data(ksp_f, args.out_dir, args.data, percentage_to_keep=args.per_spokes)

    print("original k-space type: ", ksp_f.dtype)
    ksp = ksp_f[0] + 1j * ksp_f[1]
    print("k-space type after real/imag combined: ", ksp_f.dtype)
    ksp = np.transpose(ksp, (3, 2, 0, 1))

    # select only one of the middle partitions
    #ksp = ksp[40:41, :, :, :]

    


    # zero-fill the slice dimension
    partitions = ksp.shape[0]

    # adjust shift for zero filling depending on # of slices compared to partitions
    if args.images_per_slab > partitions + 1:
        shift = int(args.images_per_slab / 2 - args.center_partition)
    else:
        shift = 0
        print("slices less than or equal to partitions + 1.")



    ksp_zf = np.zeros_like(ksp, shape=[args.images_per_slab] + list(ksp.shape[1:]))
    ksp_zf[shift : shift + partitions, ...] = ksp

    ksp_zf = sp.fft(ksp_zf, axes=(0,))
    

    # save zero-filled k-space
    base_filename = os.path.splitext(args.data)[0]
    filename = os.path.join(args.out_dir, 'zf_kspace')

    if not os.path.isdir(filename):
        os.makedirs(filename)

    filename = os.path.join(filename, f'{base_filename}.h5')

    f = h5py.File(filename, "w")
    dset = f.create_dataset('kspace', data=ksp_zf)
    f.close()


    N_slices, N_coils, N_spokes, N_samples = ksp_zf.shape

    base_res = N_samples // 2

    N_time = N_spokes // args.spokes_per_frame

    N_spokes_prep = N_time * args.spokes_per_frame

    ksp_redu = ksp_zf[:, :, :N_spokes_prep, :]
    print('  ksp_redu shape: ', ksp_redu.shape)

    # %% retrospecitvely split spokes
    ksp_prep = np.swapaxes(ksp_redu, 0, 2)
    ksp_prep_shape = ksp_prep.shape
    ksp_prep = np.reshape(ksp_prep, [N_time, args.spokes_per_frame] + list(ksp_prep_shape[1:]))
    ksp_prep = np.transpose(ksp_prep, (3, 0, 2, 1, 4))



    # save binned k-space
    # base_filename = os.path.splitext(args.data)[0]
    # filename = os.path.join(args.out_dir, 'binned_kspace')

    # if not os.path.isdir(filename):
    #     os.makedirs(filename)

    # filename = os.path.join(filename, f'{base_filename}.h5')

    # f = h5py.File(filename, "w")
    # dset = f.create_dataset('ktspace', data=ksp_prep)
    # f.close()

    ksp_prep = ksp_prep[:, :, None, :, None, :, :]
    print('  ksp_prep shape: ', ksp_prep.shape)
    print('  ksp_prep dtype: ', ksp_prep.dtype)


    # %% trajectories
    traj = get_traj(args, N_spokes=args.spokes_per_frame,
                    N_time=N_time, base_res=base_res,
                    gind=1)
    print('  traj shape: ', traj.shape)

    # save traj 
    # traj_path = os.path.join(args.out_dir, 'trajectories')

    # if not os.path.isdir(traj_path):
    #     os.makedirs(traj_path)

    # traj_path = os.path.join(traj_path, f'{base_filename}_traj.npy')
    # np.save(traj_path, traj)


    # %% slice-by-slice recon

    if args.slice_idx >= 0:
        slice_loop = range(args.slice_idx, args.slice_idx + args.slice_inc, 1)
    else:
        slice_loop = range(N_slices)

    acq_slices = []

    print("out dir: ", args.out_dir)

    if args.save_cs_maps and args.per_spokes == 100:
        # base_out = os.path.basename(args.out_dir)
        # coil_maps_dir = os.path.join('/ess/scratch/scratch1/rachelgordon', base_out, 'cs_maps', base_filename + f'_cs_maps' )
        coil_maps_dir = os.path.join(args.out_dir, 'cs_maps', base_filename + f'_cs_maps' )

        print("coil_maps_dir: ", coil_maps_dir)
        os.makedirs(coil_maps_dir, exist_ok=True)

    for s in slice_loop:
        print('>>> slice ', str(s).zfill(3))

        # coil sensitivity maps
        print('> compute coil sensitivity maps')
        C = get_coil(ksp_zf[s], args, device=device)
        C = C[:, None, :, :]
        print('  coil shape: ', C.shape)


        if args.save_cs_maps and args.per_spokes == 100:
            save_path = os.path.join(coil_maps_dir, f'cs_map_slice_{s:03d}.npy')
            np.save(save_path, C)
            print(f"Map saved to {save_path}")

        # recon
        k1 = ksp_prep[s]
        print("k1 shape: ", k1.shape)
        print("k1 dtype: ", k1.dtype)
        
        print("---- k-space input shape: ", k1.shape) # ---- k-space input shape:  (8, 1, 16, 1, 36, 640)
        print("---- csmaps input shape: ", C.shape) # ---- csmaps input shape:  (16, 1, 320, 320)
        print("---- traj input shape: ", traj.shape) # ---- traj input shape:  (8, 36, 640, 2)
        R1 = app.HighDimensionalRecon(k1, C,
                        combine_echo=False,
                        lamda=0.001,
                        coord=traj,
                        regu='TV', regu_axes=[0],
                        max_iter=10,
                        solver='ADMM', rho=0.1,
                        device=device,
                        show_pbar=False,
                        verbose=False).run()
        print("R1 dtype: ", R1.dtype)
        acq_slices.append(R1)

    acq_slices = sp.to_device(acq_slices)

    acq_slices = cp.array(acq_slices)
    acq_slices = cp.asnumpy(acq_slices)
    print("acq_slices dtype: ", acq_slices.dtype)
    # acq_slices = np.squeeze(abs(acq_slices))

    if args.keep_complex == True:
        acq_slices = acq_slices.squeeze(axis=(2, 3, 4))
    else: 
        acq_slices = abs(acq_slices).squeeze(axis=(2, 3, 4))

    # acq_slices = acq_slices.squeeze(axis=(2, 3, 4))
    print("acq_slices dtype: ", acq_slices.dtype)

    print("acq_slices shape: ", acq_slices.shape)

    # save recon files
    filename = args.dir + '/' + base_filename + '_processed.h5'
    #print("save path: ",filename)
    f = h5py.File(filename, 'w')


    dset = f.create_dataset('temptv', data=acq_slices)
    dset.attrs['spokes_per_frame'] = args.spokes_per_frame
    dset.attrs['number_of_frames'] = N_time
    dset.attrs['number_of_slices'] = args.slice_inc
    f.close()
