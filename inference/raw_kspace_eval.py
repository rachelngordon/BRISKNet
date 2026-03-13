"""Evaluate raw k-space recon and metrics for DL/GRASP baselines. Run: python3 -m inference.raw_kspace_eval"""

import glob
import os
import random

import h5py
import numpy as np
import sigpy as sp
import torch
from einops import rearrange
from sigpy.mri import app

from inference.eval import calc_dc
from utils import sliding_window_inference, to_torch_complex


def get_traj(spokes_per_frame, csmaps=False, N_spokes=13, N_time=1, base_res=320, gind=1):

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

    if spokes_per_frame == 288 and csmaps == False:
        return traj
    else:
        return np.squeeze(traj)

# %% compute coil sensitivity maps
def get_coil(ksp, spokes_per_frame, device=sp.Device(-1)):

    N_coils, N_spokes, N_samples = ksp.shape

    base_res = N_samples // 2

    ishape = [N_coils] + [base_res] * 2

    traj = get_traj(spokes_per_frame, csmaps=True, N_spokes=N_spokes, N_time=1,
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


def process_kspace(kspace_path, device, spokes_per_frame, images_per_slab, center_partition=31):

    f = h5py.File(kspace_path, 'r')
    ksp_zf = f['kspace'][:]#.T
    # ksp_zf = np.transpose(ksp_f, (4, 3, 2, 1, 0))
    # print('> kspace shape ', ksp_zf.shape)
    f.close()


    # print("original k-space type: ", ksp_f.dtype)
    # ksp = ksp_f[0] + 1j * ksp_f[1]
    # print("k-space type after real/imag combined: ", ksp_f.dtype)
    # ksp = np.transpose(ksp, (3, 2, 0, 1))


    # # zero-fill the slice dimension if necessaary
    # partitions = ksp.shape[0]

    # start = time.time()

    # if images_per_slab > partitions + 1:
    #     shift = int(images_per_slab / 2 - center_partition)
    # else:
    #     shift = 0
    #     print("slices less than or equal to partitions + 1.")

    # ksp_zf = np.zeros_like(ksp, shape=[images_per_slab] + list(ksp.shape[1:]))
    # ksp_zf[shift : shift + partitions, ...] = ksp


    # ksp_zf = sp.fft(ksp_zf, axes=(0,))

    # end = time.time()

    # print("zero-padded k-space shape: ", ksp_zf.shape) # zero-padded k-space shape:  (192, 16, 288, 640)
    # print("time for zero padding: ", end-start)


    N_slices, N_coils, N_spokes, N_samples = ksp_zf.shape

    base_res = N_samples // 2

    N_time = N_spokes // spokes_per_frame

    N_spokes_prep = N_time * spokes_per_frame

    ksp_redu = ksp_zf[:, :, :N_spokes_prep, :]
    # print('  ksp_redu shape: ', ksp_redu.shape) # (192, 16, 288, 640)

    # %% retrospecitvely split spokes
    ksp_prep = np.swapaxes(ksp_redu, 0, 2)

    # print("ksp_prep: ", ksp_prep.shape) # (288, 16, 192, 640)
    ksp_prep_shape = ksp_prep.shape
    ksp_prep = np.reshape(ksp_prep, [N_time, spokes_per_frame] + list(ksp_prep_shape[1:])) # (36, 8, 16, 192, 640)

    # print("ksp_prep after reshape: ", ksp_prep.shape) # (36, 8, 16, 192, 640)

    ksp_prep = np.transpose(ksp_prep, (3, 0, 2, 1, 4))


    ksp_prep = ksp_prep[:, :, None, :, None, :, :]
    # print('  ksp_prep shape: ', ksp_prep.shape)
    # print('  ksp_prep dtype: ', ksp_prep.dtype)

    traj = get_traj(spokes_per_frame, N_spokes=spokes_per_frame,
                N_time=N_time, base_res=base_res,
                gind=1)


    return ksp_zf, ksp_prep, traj


def raw_grasp_recon(ksp_zf, ksp_prep, traj, N_slices, spokes_per_frame, device):
    # %% slice-by-slice recon
    slice_loop = range(N_slices)

    # --- FIX 1: Pre-allocate a NumPy array on the CPU (RAM) to store the results ---
    # We need to determine the shape of a single reconstructed slice first.
    # Let's run one reconstruction to find out.
    
    print('>>> Determining output shape for pre-allocation...')
    s_test = 0
    C_test = get_coil(ksp_zf[s_test], spokes_per_frame, device=device)
    C_test = C_test[:, None, :, :]
    k1_test = ksp_prep[s_test]
    R1_test = app.HighDimensionalRecon(k1_test, C_test,
                    combine_echo=False, lamda=0.001, coord=traj,
                    regu='TV', regu_axes=[0], max_iter=10,
                    solver='ADMM', rho=0.1, device=device,
                    show_pbar=False, verbose=False).run()
    
    # Get the shape of a single slice reconstruction
    single_slice_shape = R1_test.shape
    print(f"  Detected single slice shape: {single_slice_shape}")
    
    # Create the placeholder array on the CPU
    final_recon_shape = (N_slices,) + single_slice_shape
    all_reconstructed_slices = np.zeros(final_recon_shape, dtype=R1_test.dtype)
    print(f"  Pre-allocating CPU array with shape: {all_reconstructed_slices.shape}")
    
    # Clean up memory from the test run
    del C_test, k1_test, R1_test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- END FIX 1 ---

    for s in slice_loop:
        print('>>> slice ', str(s).zfill(3))

        # coil sensitivity maps
        print('> compute coil sensitivity maps')
        C = get_coil(ksp_zf[s], spokes_per_frame, device=device)
        C = C[:, None, :, :]
        # print('  coil shape: ', C.shape) # Optional: uncomment for debugging

        # recon
        k1 = ksp_prep[s]
        
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
        
        # --- FIX 2: Move the result to CPU and place it in the pre-allocated array ---
        # Instead of: acq_slices.append(R1)
        
        # sp.to_device moves the SigPy array from GPU to CPU
        reconstructed_slice_cpu = sp.to_device(R1, sp.cpu_device)
        
        # Place the CPU-based result into our final numpy array
        all_reconstructed_slices[s] = reconstructed_slice_cpu

        # --- END FIX 2 ---

    # The loop is now finished, and `all_reconstructed_slices` is a complete NumPy array on the CPU.
    # The old, problematic post-processing is no longer needed.
    print("Final reconstructed shape: ", all_reconstructed_slices.shape)
    
    # Squeeze out unnecessary dimensions if they exist
    # Based on your original code, it seems you expect dimensions 2, 3, 4 to be size 1
    squeezed_slices = all_reconstructed_slices.squeeze(axis=(2, 3, 4))
    print("Final shape after squeeze: ", squeezed_slices.shape)

    return squeezed_slices


# def raw_grasp_recon(ksp_zf, ksp_prep, traj, N_slices, spokes_per_frame, device):
#     # %% slice-by-slice recon
#     slice_loop = range(N_slices)

#     acq_slices = []

#     slices_to_recon = 0

#     for s in slice_loop:
#         print('>>> slice ', str(s).zfill(3))

#         # coil sensitivity maps
#         print('> compute coil sensitivity maps')
#         C = get_coil(ksp_zf[s], spokes_per_frame, device=device)
#         C = C[:, None, :, :]
#         print('  coil shape: ', C.shape)


#         # recon
#         k1 = ksp_prep[s]
#         print("k1 shape: ", k1.shape)
#         print("k1 dtype: ", k1.dtype)
        
#         print("---- k-space input shape: ", k1.shape) # ---- k-space input shape:  (8, 1, 16, 1, 36, 640)
#         print("---- csmaps input shape: ", C.shape) # ---- csmaps input shape:  (16, 1, 320, 320)
#         print("---- traj input shape: ", traj.shape) # ---- traj input shape:  (8, 36, 640, 2)
#         R1 = app.HighDimensionalRecon(k1, C,
#                         combine_echo=False,
#                         lamda=0.001,
#                         coord=traj,
#                         regu='TV', regu_axes=[0],
#                         max_iter=10,
#                         solver='ADMM', rho=0.1,
#                         device=device,
#                         show_pbar=False,
#                         verbose=False).run()
#         print("R1 dtype: ", R1.dtype)
#         acq_slices.append(R1)

#         # NOTE: remove after testing
#         # slices_to_recon += 1

#         # if slices_to_recon > 2:
#         #     break



#     acq_slices = sp.to_device(acq_slices)

#     acq_slices = cp.array(acq_slices)
#     acq_slices = cp.asnumpy(acq_slices)
#     print("acq_slices dtype: ", acq_slices.dtype)
#     # acq_slices = np.squeeze(abs(acq_slices))

#     acq_slices = acq_slices.squeeze(axis=(2, 3, 4))
#     # acq_slices = acq_slices.squeeze()
    
#     # acq_slices = acq_slices.squeeze(axis=(2, 3, 4))
#     print("acq_slices dtype: ", acq_slices.dtype)

#     print("acq_slices shape: ", acq_slices.shape)

#     return acq_slices

    
def load_all_csmaps(dir, patient_id):
    """
    Loads all csmap slices for a given patient and stacks them into a single array.

    Args:
        patient_id (str): The ID of the patient.

    Returns:
        numpy.ndarray: A NumPy array containing all the stacked csmap slices
                    with the shape (1, C, H, W, Z_in).
    """
    ground_truth_dir = os.path.join(dir, 'cs_maps')
    patient_csmap_dir = os.path.join(ground_truth_dir, patient_id + '_cs_maps')

    # Find all slice files and sort them to ensure correct order
    slice_paths = sorted(glob.glob(os.path.join(patient_csmap_dir, 'cs_map_slice_*.npy')))

    if not slice_paths:
        raise FileNotFoundError(f"No csmap slices found for patient {patient_id} in {patient_csmap_dir}")

    # Load each slice and store it in a list
    all_slices = [np.load(path) for path in slice_paths]

    # Stack the slices along a new axis (the last axis)
    # Assuming each slice has a shape of (H, W, C)
    stacked_csmaps = np.stack(all_slices, axis=-1)

    print("stacked_csmaps: ", stacked_csmaps.shape)

    final_csmaps = rearrange(stacked_csmaps, 'c b h w z -> b c h w z')

    print("final_csmaps: ", final_csmaps.shape)

    # At this point, the shape is likely (H, W, C, Z_in)
    # We need to rearrange the axes to (C, H, W, Z_in)
    # The axes are indexed as H=0, W=1, C=2, Z_in=3
    # transposed_csmaps = np.transpose(stacked_csmaps, (2, 0, 1, 3))

    # # Add a new dimension at the beginning to get the final shape (1, C, H, W, Z_in)
    # final_csmaps = np.expand_dims(transposed_csmaps, axis=0)

    return final_csmaps


def eval_raw_kspace(num_slices_to_eval, val_patient_ids, data_dir, model, spokes_per_frame, N_slices, num_frames, eval_chunk_size, eval_chunk_overlap, H, W, ktraj, dcomp, nufft_ob, adjnufft_ob, physics, acceleration_encoding, start_timepoint_index, device, out_dir, label):
    
    sp_device = sp.Device(0 if torch.cuda.is_available() else -1)
    dtype = torch.complex64

    # select random slices to evaluate on
    # NOTE: fix after testing
    random_slice_indices = random.sample(range(N_slices), num_slices_to_eval)
    # random_slice_indices = [1]

    # NOTE: temporarily set val_patient_ids for testing
    # val_patient_ids = ['fastMRI_breast_001']
    
    avg_dc_mses = []
    avg_dc_maes = []
    # avg_grasp_dc_mses = []
    # avg_grasp_dc_maes = []
    with torch.no_grad():
        for patient_id in val_patient_ids:

            raw_kspace_path = os.path.join(data_dir, f'{patient_id}_2.h5')

            dir = os.path.dirname(data_dir)
            zf_kspace, binned_kspace, traj = process_kspace(raw_kspace_path, device=sp_device, spokes_per_frame=spokes_per_frame, images_per_slab=N_slices, center_partition=31)

            # grasp_img_path = os.path.join(dir, f'{patient_id}_2', f'grasp_recon_{spokes_per_frame}spf.npy')

            # if not os.path.exists(grasp_img_path):
            #     grasp_img_slices = raw_grasp_recon(zf_kspace, binned_kspace, traj, N_slices=N_slices, spokes_per_frame=spokes_per_frame, device=sp_device)
            #     np.save(grasp_img_path, grasp_img_slices)
            # else:
            #     grasp_img_slices = np.load(grasp_img_path)
            

            csmap = load_all_csmaps(dir, f'{patient_id}_2')


            slice_dc_mses = []
            slice_dc_maes = []
            # grasp_slice_dc_mses = []
            # grasp_slice_dc_maes = []

            for slice_idx in random_slice_indices:

                kspace_slice = torch.tensor(binned_kspace[slice_idx].squeeze())
                # grasp_img_slice = torch.tensor(grasp_img_slices[slice_idx])
                csmap_slice = torch.tensor(csmap[..., slice_idx])

                kspace_slice_flat = rearrange(kspace_slice, 't c sp sam -> c (sp sam) t').to(dtype)
                csmap_slice = csmap_slice.to(dtype)

                # model inference 
                if num_frames > eval_chunk_size:
                    x_recon, _ = sliding_window_inference(H, W, num_frames, ktraj, dcomp, nufft_ob, adjnufft_ob, eval_chunk_size, eval_chunk_overlap, kspace_slice_flat, csmap_slice, acceleration_encoding, start_timepoint_index, model, epoch=None, device=device)  
                else:
                    x_recon, *_ = model(
                        kspace_slice_flat.to(device), physics, csmap_slice, acceleration_encoding, start_timepoint_index, epoch=None, norm="both"
                    )

                # calculate data consistency of output with original k-space input
                # simulate k-space for DL and GRASP recons

                x_recon = to_torch_complex(x_recon)
                sim_kspace = physics(False, x_recon.to(device), csmap_slice.to(device))

                # print("grasp_img: ", grasp_img_slice.shape)
                
                # if grasp_img_slice.shape[1] == 2:
                #     grasp_img_slice = to_torch_complex(grasp_img_slice)

                # if grasp_img_slice.shape[-2] == num_frames: 
                #     grasp_img_slice = rearrange(grasp_img_slice, 'b h t w -> b h w t')


                # grasp_img_slice = rearrange(grasp_img_slice, 't h w -> h w t').unsqueeze(0)

                # sim_kspace_grasp = physics(False, grasp_img_slice.to(x_recon.dtype).to(device), csmap_slice.to(device))

                raw_dc_mse, raw_dc_mae = calc_dc(sim_kspace, kspace_slice_flat, device)
                # raw_grasp_dc_mse, raw_grasp_dc_mae = calc_dc(sim_kspace_grasp, kspace_slice_flat, device)

                slice_dc_mses.append(raw_dc_mse)
                slice_dc_maes.append(raw_dc_mae)
                # grasp_slice_dc_mses.append(raw_grasp_dc_mse)
                # grasp_slice_dc_maes.append(raw_grasp_dc_mae)


            avg_dc_mses.append(np.mean(slice_dc_mses))
            avg_dc_maes.append(np.mean(slice_dc_maes))
            # avg_grasp_dc_mses.append(np.mean(grasp_slice_dc_mses))
            # avg_grasp_dc_maes.append(np.mean(grasp_slice_dc_maes))

    
    avg_mse = np.mean(avg_dc_mses)
    avg_mae = np.mean(avg_dc_maes)
    # avg_grasp_mse = np.mean(avg_grasp_dc_mses)
    # avg_grasp_mae = np.mean(avg_grasp_dc_maes)

    std_mse = np.std(avg_dc_mses)
    std_mae = np.std(avg_dc_maes)
    # std_grasp_mse = np.std(avg_grasp_dc_mses)
    # std_grasp_mae = np.std(avg_grasp_dc_maes)


    # plot example image comparison
    # plot_path = os.path.join(out_dir, f'raw_kspace_recon_comparison_{label}.png')
    # timeframe = num_frames // 2 

    # # Extract the specific timeframe for both images.
    # # Since the first dimension is 1, we can squeeze it out.
    # x_recon_timeframe = x_recon[0, :, :, timeframe]
    # # grasp_img_timeframe = grasp_img_slice[..., timeframe]

    # # Create a figure with two subplots, arranged horizontally.
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # # Display the first image slice.
    # ax1.imshow(np.abs(x_recon_timeframe.squeeze().cpu().numpy()), cmap='gray')
    # ax1.set_title(f'DL Recon - Timeframe: {timeframe}, Slice: {slice_idx}')
    # ax1.axis('off')

    # # Display the second image slice.
    # ax2.imshow(np.abs(grasp_img_timeframe.squeeze().cpu().numpy()), cmap='gray')
    # ax2.set_title(f'GRASP Recon - Timeframe: {timeframe}, Slice: {slice_idx}')
    # ax2.axis('off')

    # # Adjust layout and display the plot.
    # plt.tight_layout()
    # plt.savefig(plot_path)
    # print(f"---- DL GRASP Orientation Comparsion Saved to: {plot_path} ----")


    return avg_mse, avg_mae, std_mse, std_mae




def eval_raw_kspace_grasp(slice_idx, val_patient_ids, data_dir, spokes_per_frame, N_slices, num_frames, physics, device):
    
    sp_device = sp.Device(0 if torch.cuda.is_available() else -1)
    dtype = torch.complex64

    # select random slices to evaluate on
    # NOTE: fix after testing
    # random_slice_indices = random.sample(range(N_slices), num_slices_to_eval)
    # random_slice_indices = [1]

    # NOTE: temporarily set val_patient_ids for testing
    # val_patient_ids = ['fastMRI_breast_001']
    
    # avg_dc_mses = []
    # avg_dc_maes = []
    avg_grasp_dc_mses = []
    avg_grasp_dc_maes = []
    with torch.no_grad():
        for patient_id in val_patient_ids:

            raw_kspace_path = os.path.join(data_dir, f'{patient_id}_2.h5')

            dir = os.path.dirname(data_dir)
            zf_kspace, binned_kspace, traj = process_kspace(raw_kspace_path, device=sp_device, spokes_per_frame=spokes_per_frame, images_per_slab=N_slices, center_partition=31)

            grasp_img_path = os.path.join(dir, f'{patient_id}_2', f'grasp_recon_{spokes_per_frame}spf_{num_frames}frames_slice{slice_idx}.npy')

            # if not os.path.exists(grasp_img_path):
            #     grasp_img_slices = raw_grasp_recon(zf_kspace, binned_kspace, traj, N_slices=N_slices, spokes_per_frame=spokes_per_frame, device=sp_device)
            #     np.save(grasp_img_path, grasp_img_slices)
            # else:
            grasp_img_slice = np.load(grasp_img_path)
            

            csmap = load_all_csmaps(dir, f'{patient_id}_2')


            # for slice_idx in random_slice_indices:

            kspace_slice = torch.tensor(binned_kspace[slice_idx].squeeze())
            # grasp_img_slice = torch.tensor(grasp_img_slices[slice_idx])
            csmap_slice = torch.tensor(csmap[..., slice_idx])

            kspace_slice_flat = rearrange(kspace_slice, 't c sp sam -> c (sp sam) t').to(dtype)
            csmap_slice = csmap_slice.to(dtype)

            # # model inference 
            # if num_frames > eval_chunk_size:
            #     x_recon, _ = sliding_window_inference(H, W, num_frames, ktraj, dcomp, nufft_ob, adjnufft_ob, eval_chunk_size, eval_chunk_overlap, kspace_slice_flat, csmap_slice, acceleration_encoding, start_timepoint_index, model, epoch=None, device=device)  
            # else:
            #     x_recon, *_ = model(
            #         kspace_slice_flat.to(device), physics, csmap_slice, acceleration_encoding, start_timepoint_index, epoch=None, norm="both"
            #     )

            # calculate data consistency of output with original k-space input
            # simulate k-space for DL and GRASP recons

            # x_recon = to_torch_complex(x_recon)
            # sim_kspace = physics(False, x_recon.to(device), csmap_slice.to(device))

            # print("grasp_img: ", grasp_img_slice.shape)
            
            if grasp_img_slice.shape[1] == 2:
                grasp_img_slice = to_torch_complex(grasp_img_slice)
                grasp_img_slice = rearrange(grasp_img_slice, 't h w -> h w t').unsqueeze(0)

            if grasp_img_slice.shape[-2] == num_frames: 
                grasp_img_slice = rearrange(grasp_img_slice, 'b h t w -> b h w t')


            grasp_img_slice = torch.tensor(grasp_img_slice, dtype=dtype, device=device)
            sim_kspace_grasp = physics(False, grasp_img_slice, csmap_slice.to(device))

            # raw_dc_mse, raw_dc_mae = calc_dc(sim_kspace, kspace_slice_flat, device)
            raw_grasp_dc_mse, raw_grasp_dc_mae = calc_dc(sim_kspace_grasp, kspace_slice_flat, device)

            # slice_dc_mses.append(raw_dc_mse)
            # slice_dc_maes.append(raw_dc_mae)
            # grasp_slice_dc_mses.append(raw_grasp_dc_mse)
            # grasp_slice_dc_maes.append(raw_grasp_dc_mae)


            # avg_dc_mses.append(np.mean(slice_dc_mses))
            # avg_dc_maes.append(np.mean(slice_dc_maes))
            avg_grasp_dc_mses.append(raw_grasp_dc_mse)
            avg_grasp_dc_maes.append(raw_grasp_dc_mae)

    
    # avg_mse = np.mean(avg_dc_mses)
    # avg_mae = np.mean(avg_dc_maes)
    avg_grasp_mse = np.mean(avg_grasp_dc_mses)
    avg_grasp_mae = np.mean(avg_grasp_dc_maes)

    # std_mse = np.std(avg_dc_mses)
    # std_mae = np.std(avg_dc_maes)
    std_grasp_mse = np.std(avg_grasp_dc_mses)
    std_grasp_mae = np.std(avg_grasp_dc_maes)


    # plot example image comparison
    # plot_path = os.path.join(out_dir, f'raw_kspace_recon_comparison_{label}.png')
    # timeframe = num_frames // 2 

    # # Extract the specific timeframe for both images.
    # # Since the first dimension is 1, we can squeeze it out.
    # x_recon_timeframe = x_recon[0, :, :, timeframe]
    # # grasp_img_timeframe = grasp_img_slice[..., timeframe]

    # # Create a figure with two subplots, arranged horizontally.
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # # Display the first image slice.
    # ax1.imshow(np.abs(x_recon_timeframe.squeeze().cpu().numpy()), cmap='gray')
    # ax1.set_title(f'DL Recon - Timeframe: {timeframe}, Slice: {slice_idx}')
    # ax1.axis('off')

    # # Display the second image slice.
    # ax2.imshow(np.abs(grasp_img_timeframe.squeeze().cpu().numpy()), cmap='gray')
    # ax2.set_title(f'GRASP Recon - Timeframe: {timeframe}, Slice: {slice_idx}')
    # ax2.axis('off')

    # # Adjust layout and display the plot.
    # plt.tight_layout()
    # plt.savefig(plot_path)
    # print(f"---- DL GRASP Orientation Comparsion Saved to: {plot_path} ----")


    return avg_grasp_mse, avg_grasp_mae, std_grasp_mse, std_grasp_mae
