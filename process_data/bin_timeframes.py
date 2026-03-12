"""Bin k-space spokes into time frames and save HDF5 outputs. Run: python3 process_data/bin_timeframes.py --help"""

import argparse

import h5py
import numpy as np
import sigpy as sp

parser = argparse.ArgumentParser(description='Bin k-space time frames.')
parser.add_argument('--spokes_per_frame', type=int, required=False, default=24, help='Number of spokes per timeframe')
parser.add_argument('--out_dir', type=str, required=False, default='/ess/scratch/scratch1/rachelgordon/binned_kspace_12_timeframes', help='Directory to save binned k-space to')
args = parser.parse_args()


def get_traj(N_spokes=13, N_time=1, base_res=320, gind=1):

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

    return np.squeeze(traj)



for val in range (0, 30):

    # read in k-space data
    start_range = val * 10 + 1
    end_range = start_range + 9

    ids_path = f'/ess/scratch/scratch1/rachelgordon/fastMRI_breast_data/fastMRI_breast_IDS_{start_range:03d}_{end_range:03d}/'

    for id in range(start_range, end_range):

        kspace_path = ids_path + f'fastMRI_breast_{id:03d}_2.h5'

        f = h5py.File(kspace_path, 'r')
        ksp_f = f['kspace'][:].T
        ksp_f = np.transpose(ksp_f, (4, 3, 2, 1, 0))
        print('> kspace shape ', ksp_f.shape)
        f.close()


        ksp = ksp_f[0] + 1j * ksp_f[1]
        ksp = np.transpose(ksp, (3, 2, 0, 1))
        ksp_zf = sp.fft(ksp, axes=(0,))


        N_slices, N_coils, N_spokes, N_samples = ksp_zf.shape

        base_res = N_samples // 2

        N_time = N_spokes // args.spokes_per_frame

        N_spokes_prep = N_time * args.spokes_per_frame

        ksp_redu = ksp_zf[:, :, :N_spokes_prep, :]
        print('  ksp_redu shape: ', ksp_redu.shape)


        # retrospecitvely split spokes
        ksp_prep = np.swapaxes(ksp_redu, 0, 2)
        ksp_prep_shape = ksp_prep.shape
        ksp_prep = np.reshape(ksp_prep, [N_time, args.spokes_per_frame] + list(ksp_prep_shape[1:]))
        ksp_prep = np.transpose(ksp_prep, (3, 0, 2, 1, 4))


        # save binned k-space to a file
        output_file_path = args.out_dir + f'/fastMRI_breast_{id:03d}_2.h5'

        with h5py.File(output_file_path, 'w') as h5f:
            h5f.create_dataset('ktspace', data=ksp_prep)
            


    # save trajectory on first iteration
    if val == 0:
        traj = get_traj(N_spokes=args.spokes_per_frame,
                        N_time=N_time, base_res=base_res,
                        gind=1)
        print('  traj shape: ', traj.shape)

        traj_path = args.out_dir + f'traj_{args.spokes_per_frame}_spf.npy'
        np.save(traj, traj_path)
