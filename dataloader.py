import glob
import os
import csv
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
from einops import rearrange
import random
import sigpy as sp
from utils import prep_nufft
from radial_lsfp import MCNUFFT
import time
from typing import Union, List, Optional
import re
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
SLICE_MAP_PATH = REPO_ROOT / "data" / "largest_tumor_slices.csv"


def load_slice_map(csv_path: Path) -> dict:
    """Load patient -> slice index map; return empty dict if missing."""
    if not csv_path.exists():
        print(f"Slice map not found at {csv_path}; falling back to configured slice indices.")
        return {}

    mapping = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("fastMRI_breast_id")
            idx = row.get("largest_slice_idx")
            if pid is None or idx is None:
                continue
            pid = pid.replace(".nii", "")
            try:
                mapping[pid] = int(idx)
            except ValueError:
                continue
    return mapping


class ZFSliceDataset(Dataset):
    """
    A Dataset that:
      - Looks for all .h5/.hdf5 files under `root_dir`.
      - Each file is assumed to contain a dataset at `dataset_key`, with shape (... Z),
        where Z is the number of slices/partitions.
      - Can either use a fixed set of slices or randomly sample N slices per volume
        at the start of each epoch.
      - Returns each slice as a torch.Tensor.
    """

    def __init__(
        self,
        root_dir,
        patient_ids,
        dataset_key="kspace",
        file_pattern="*.h5",
        slice_idx: Optional[Union[int, range]] = 41,
        num_random_slices: Optional[int] = None,  # New parameter for random sampling
        N_time=8,
        N_coils=16,
        spf_aug=False,
        spokes_per_frame=None,
        weight_accelerations=False, 
        initial_spokes_range=[8, 16, 24, 36],
        cluster="Randi",
        flip_kspace=True,
    ):
        """
        Args:
            root_dir (str): Path to the folder containing all HDF5 k-space files.
            patient_ids (list): List of patient IDs to filter the files.
            dataset_key (str): The key/path inside each .h5 file to the k-space dataset.
            file_pattern (str): Glob pattern to match your HDF5 files.
            slice_idx (int, range, optional): A fixed slice index or range of indices to use.
                                              This is ignored if num_random_slices is set.
            num_random_slices (int, optional): If provided, the dataset will randomly sample
                                               this many slices from each volume at the beginning
                                               of each epoch.
        """
        super().__init__()

        self.root_dir = root_dir
        self.dataset_key = dataset_key
        self.slice_idx = slice_idx
        self.num_random_slices = num_random_slices
        self.N_time = N_time
        self.N_coils = N_coils
        self.spf_aug = spf_aug
        self.weight_acc = weight_accelerations
        self.cluster=cluster
        self.flip_kspace=flip_kspace

        # Find all matching HDF5 files under root_dir
        all_files = sorted(glob.glob(os.path.join(root_dir, file_pattern)))
        print("Number of files in root directory: ", len(all_files))

        if len(all_files) == 0:
            raise RuntimeError(
                f"No files found in {root_dir} matching pattern {file_pattern}"
            )

        # filter file list by patient ID substring
        filtered = []
        for fp in all_files:
            fname = os.path.basename(fp)
            if any(pid in fname for pid in patient_ids):
                filtered.append(fp)

        self.file_list = filtered

        if len(self.file_list) == 0:
            raise RuntimeError("No files matched the provided patient_ids filter.")

        # Logic for random slice sampling
        if self.num_random_slices is not None:
            print(f"Initializing in random slice sampling mode with N={self.num_random_slices} slices per volume.")
            self.volume_map = []
            for fp in self.file_list:
                with h5py.File(fp, "r") as f:
                    if self.dataset_key not in f:
                        raise KeyError(f"Dataset key '{self.dataset_key}' not found in file {fp}")
                    num_slices = f[self.dataset_key].shape[0]
                    self.volume_map.append((fp, num_slices))
            
            # Perform the initial random sampling for the first epoch
            self.resample_slices()
        
        # Original logic for fixed slices, executed only if not in random mode
        else:
            print(f"Initializing in fixed slice mode with slice_idx={self.slice_idx}.")
            self.slice_index_map = []
            for fp in self.file_list:
                with h5py.File(fp, "r") as f:
                    if self.dataset_key not in f:
                        raise KeyError(f"Dataset key '{self.dataset_key}' not found in file {fp}")
                    ds = f[self.dataset_key]
                    num_slices = ds.shape[0]

                slices_to_add = []
                if isinstance(self.slice_idx, int):
                    if self.slice_idx < num_slices:
                        slices_to_add = [self.slice_idx]
                    else:
                        print(f"Warning: slice_idx {self.slice_idx} is out of bounds for {fp} "
                              f"(size {num_slices}). Skipping this file for this slice.")
                elif isinstance(self.slice_idx, range):
                    slices_to_add = [s for s in self.slice_idx if s < num_slices]
                    if len(slices_to_add) < len(self.slice_idx):
                        print(f"Warning: Some requested slices were out of bounds for {fp}. "
                              f"Using only the valid slice indices from the provided range.")
                else:
                    raise TypeError(f"slice_idx must be an int, range, or None, but got {type(self.slice_idx)}")

                for z in slices_to_add:
                    self.slice_index_map.append((fp, z))

        print(f"Dataset initialized with {len(self.slice_index_map)} total slice examples.")

        self.spokes_per_frame = spokes_per_frame

        # NOTE: removed ultra-high accelerations until curriculum learning is implemented
        self.spokes_range = initial_spokes_range
        self.update_spokes_weights()
    
    def update_spokes_weights(self):

        if self.weight_acc:
            self.spf_weights = [1.0 / spf for spf in self.spokes_range]
        else:
            self.spf_weights = [1.0 for _ in self.spokes_range]


    def resample_slices(self):
        """
        Resamples N unique slices from each volume. This should be called at the
        beginning of each training epoch to ensure the model sees different data.
        """
        if self.num_random_slices is None:
            # If not in random sampling mode, do nothing.
            return

        self.slice_index_map = []
        for file_path, num_slices in self.volume_map:
            if num_slices >= self.num_random_slices:
                # Randomly sample N unique slices without replacement
                selected_slices = random.sample(range(num_slices), self.num_random_slices)
            else:
                # If the volume has fewer than N slices, take all of them.
                print(f"Warning: Volume {os.path.basename(file_path)} has only {num_slices} slices, "
                      f"which is less than the requested {self.num_random_slices}. Using all available slices.")
                selected_slices = list(range(num_slices))

            for z in selected_slices:
                self.slice_index_map.append((file_path, z))

    def load_dynamic_img(self, patient_id, slice):
        # This method remains unchanged
        H = W = 320
        data = np.empty((2, self.N_time, H, W), dtype=np.float32)

        dirname = os.path.dirname(self.root_dir)
        
        for t in range(self.N_time):
            if self.cluster == "Randi":
                img_path = os.path.join(dirname, f'{patient_id}/slice_{slice:03d}_frame_{t:03d}.nii')
            elif self.cluster == "DSI":
                img_path = os.path.join(dirname, f'{patient_id}/slice_{slice:03d}_frame_{t:03d}.nii')
            else:
                raise ValueError("Undefined cluster name.")
            img = nib.load(img_path)
            img_data = img.get_fdata()

            if img_data.shape != (2, H, W):
                raise ValueError(f"{img_path} has shape {img_data.shape}; expected (2, {H}, {W})")

            data[:, t] = img_data.astype(np.float32)
            
        return torch.from_numpy(data)

    def load_csmaps(self, patient_id, slice):
        # This method remains unchanged
        ground_truth_dir = os.path.join(os.path.dirname(self.root_dir), 'cs_maps')
        csmap_path = os.path.join(ground_truth_dir, patient_id + '_cs_maps', f'cs_map_slice_{slice:03d}.npy')
        csmap = np.load(csmap_path)
        return csmap.squeeze()
    

    def __len__(self):
        return len(self.slice_index_map)

    def __getitem__(self, idx):
        # This method remains unchanged as it relies on self.slice_index_map
        file_path, current_slice_idx = self.slice_index_map[idx]

        current_slice_idx = int(current_slice_idx)
        patient_id = file_path.split('/')[-1].strip('.h5')

        csmap = self.load_csmaps(patient_id, current_slice_idx)

        with h5py.File(file_path, "r") as f:
            kspace_slice = torch.tensor(f[self.dataset_key][current_slice_idx])


        # if aug, select random spokes per frame
        if self.spf_aug:
            print("setting random spokes per frame...")
            spokes_per_frame = random.choices(self.spokes_range, self.spf_weights, k=1)[0]
        else:
            spokes_per_frame = self.spokes_per_frame
            print(f"training with fixed spokes per frame ({spokes_per_frame})")

    
        # bin k-space according to desired spokes per frame (desired final shape (T, C, Sp, Sam))
        N_coils, N_spokes, N_samples = kspace_slice.shape
        N_time = N_spokes // spokes_per_frame
        N_spokes_prep = N_time * spokes_per_frame

        ksp_redu = kspace_slice[:, :N_spokes_prep, :] # (16, 288, 640)
        ksp_prep = np.swapaxes(ksp_redu, 0, 1) # (288, 16, 640)
        ksp_prep_shape = ksp_prep.shape
        ksp_prep = np.reshape(ksp_prep, [N_time, spokes_per_frame] + list(ksp_prep_shape[1:]))
        ksp_prep = rearrange(ksp_prep, 't sp c sam -> t c sp sam')

        real_part = ksp_prep.real
        imag_part = ksp_prep.imag
        kspace_final = torch.stack([real_part, imag_part], dim=0).float()

        if self.flip_kspace:
            kspace_final = torch.flip(kspace_final, dims=[-1])

            csmap_tensor = torch.from_numpy(csmap)
            csmap_tensor = torch.rot90(csmap_tensor, k=2, dims=[-2, -1])
            csmap = csmap_tensor.numpy()

        return kspace_final, csmap, N_samples, spokes_per_frame, N_time



class SliceDataset(Dataset):
    """
    A Dataset that:
      - Looks for all .h5/.hdf5 files under `root_dir`.
      - Each file is assumed to contain a dataset at `dataset_key`, with shape (... Z),
        where Z is the number of slices/partitions.
      - Can either use a fixed set of slices or randomly sample N slices per volume
        at the start of each epoch.
      - Returns each slice as a torch.Tensor.
    """

    def __init__(
        self,
        root_dir,
        patient_ids,
        dataset_key="kspace",
        file_pattern="*.h5",
        slice_idx: Optional[Union[int, range]] = 41,
        num_random_slices: Optional[int] = None,  # New parameter for random sampling
        N_time=8,
        N_coils=16,
        spf_aug=False,
        spokes_per_frame=None,
        weight_accelerations=False, 
        initial_spokes_range=[8, 16, 24, 36],
        interpolate_kspace=False,
        slices_to_interpolate=192,
        cluster="Randi"
    ):
        """
        Args:
            root_dir (str): Path to the folder containing all HDF5 k-space files.
            patient_ids (list): List of patient IDs to filter the files.
            dataset_key (str): The key/path inside each .h5 file to the k-space dataset.
            file_pattern (str): Glob pattern to match your HDF5 files.
            slice_idx (int, range, optional): A fixed slice index or range of indices to use.
                                              This is ignored if num_random_slices is set.
            num_random_slices (int, optional): If provided, the dataset will randomly sample
                                               this many slices from each volume at the beginning
                                               of each epoch.
        """
        super().__init__()

        self.root_dir = root_dir
        self.dataset_key = dataset_key
        self.slice_idx = slice_idx
        self.num_random_slices = num_random_slices
        self.N_time = N_time
        self.N_coils = N_coils
        self.spf_aug = spf_aug
        self.weight_acc = weight_accelerations
        self.cluster=cluster

        # Find all matching HDF5 files under root_dir
        all_files = sorted(glob.glob(os.path.join(root_dir, file_pattern)))
        print("Number of files in root directory: ", len(all_files))

        if len(all_files) == 0:
            raise RuntimeError(
                f"No files found in {root_dir} matching pattern {file_pattern}"
            )

        # filter file list by patient ID substring
        filtered = []
        for fp in all_files:
            fname = os.path.basename(fp)
            if any(pid in fname for pid in patient_ids):
                filtered.append(fp)

        self.file_list = filtered

        if len(self.file_list) == 0:
            raise RuntimeError("No files matched the provided patient_ids filter.")

        # Logic for random slice sampling
        if self.num_random_slices is not None:
            print(f"Initializing in random slice sampling mode with N={self.num_random_slices} slices per volume.")
            self.volume_map = []
            for fp in self.file_list:
                with h5py.File(fp, "r") as f:
                    if self.dataset_key not in f:
                        raise KeyError(f"Dataset key '{self.dataset_key}' not found in file {fp}")
                    num_slices = f[self.dataset_key].shape[0]
                    self.volume_map.append((fp, num_slices))
            
            # Perform the initial random sampling for the first epoch
            self.resample_slices()
        
        # Original logic for fixed slices, executed only if not in random mode
        else:
            print(f"Initializing in fixed slice mode with slice_idx={self.slice_idx}.")
            self.slice_index_map = []
            for fp in self.file_list:
                with h5py.File(fp, "r") as f:
                    if self.dataset_key not in f:
                        raise KeyError(f"Dataset key '{self.dataset_key}' not found in file {fp}")
                    ds = f[self.dataset_key]
                    num_slices = ds.shape[0]

                slices_to_add = []
                if isinstance(self.slice_idx, int):
                    if self.slice_idx < num_slices:
                        slices_to_add = [self.slice_idx]
                    else:
                        print(f"Warning: slice_idx {self.slice_idx} is out of bounds for {fp} "
                              f"(size {num_slices}). Skipping this file for this slice.")
                elif isinstance(self.slice_idx, range):
                    slices_to_add = [s for s in self.slice_idx if s < num_slices]
                    if len(slices_to_add) < len(self.slice_idx):
                        print(f"Warning: Some requested slices were out of bounds for {fp}. "
                              f"Using only the valid slice indices from the provided range.")
                else:
                    raise TypeError(f"slice_idx must be an int, range, or None, but got {type(self.slice_idx)}")

                for z in slices_to_add:
                    self.slice_index_map.append((fp, z))

        print(f"Dataset initialized with {len(self.slice_index_map)} total slice examples.")

        self.spokes_per_frame = spokes_per_frame

        # NOTE: removed ultra-high accelerations until curriculum learning is implemented
        # self.spokes_range = [2, 4, 8, 16, 24, 36]
        # self.spokes_range = [8, 16, 24, 36]
        self.spokes_range = initial_spokes_range
        self.update_spokes_weights()
    
    def update_spokes_weights(self):

        if self.weight_acc:
            self.spf_weights = [1.0 / spf for spf in self.spokes_range]
        else:
            self.spf_weights = [1.0 for spf in self.spokes_range]


    def resample_slices(self):
        """
        Resamples N unique slices from each volume. This should be called at the
        beginning of each training epoch to ensure the model sees different data.
        """
        if self.num_random_slices is None:
            # If not in random sampling mode, do nothing.
            return

        self.slice_index_map = []
        for file_path, num_slices in self.volume_map:
            if num_slices >= self.num_random_slices:
                # Randomly sample N unique slices without replacement
                selected_slices = random.sample(range(num_slices), self.num_random_slices)
            else:
                # If the volume has fewer than N slices, take all of them.
                print(f"Warning: Volume {os.path.basename(file_path)} has only {num_slices} slices, "
                      f"which is less than the requested {self.num_random_slices}. Using all available slices.")
                selected_slices = list(range(num_slices))

            for z in selected_slices:
                self.slice_index_map.append((file_path, z))

    def load_dynamic_img(self, patient_id, slice):
        # This method remains unchanged
        H = W = 320
        data = np.empty((2, self.N_time, H, W), dtype=np.float32)
        
        for t in range(self.N_time):
            if self.cluster == "Randi":
                img_path = f'/ess/scratch/scratch1/rachelgordon/dce-{self.N_time}tf/{patient_id}/slice_{slice:03d}_frame_{t:03d}.nii'
            elif self.cluster == "DSI":
                img_path = f'/net/scratch2/rachelgordon/dce-{self.N_time}tf/{patient_id}/slice_{slice:03d}_frame_{t:03d}.nii'
            else:
                raise ValueError("Undefined cluster name.")
            img = nib.load(img_path)
            img_data = img.get_fdata()

            if img_data.shape != (2, H, W):
                raise ValueError(f"{img_path} has shape {img_data.shape}; expected (2, {H}, {W})")

            data[:, t] = img_data.astype(np.float32)
            
        return torch.from_numpy(data)

    def load_csmaps(self, patient_id, slice):
        # This method remains unchanged
        ground_truth_dir = os.path.join(os.path.dirname(self.root_dir), 'cs_maps')
        csmap_path = os.path.join(ground_truth_dir, patient_id + '_cs_maps', f'cs_map_slice_{slice:03d}.npy')
        csmap = np.load(csmap_path)
        return csmap.squeeze()
    


    def __len__(self):
        return len(self.slice_index_map)

    def __getitem__(self, idx):
        # This method remains unchanged as it relies on self.slice_index_map
        file_path, current_slice_idx = self.slice_index_map[idx]
        current_slice_idx = int(current_slice_idx)
        patient_id = file_path.split('/')[-1].strip('.h5')

        # grasp_img = self.load_dynamic_img(patient_id, current_slice_idx)


        csmap = self.load_csmaps(patient_id, current_slice_idx)



        with h5py.File(file_path, "r") as f:

            kspace_slice = torch.tensor(f[self.dataset_key][current_slice_idx])

        print("loaded kspace shape: ", kspace_slice.shape) # torch.Size([8, 16, 36, 640])




        if self.spf_aug or self.spokes_per_frame:
            total_spokes = kspace_slice.shape[0] * kspace_slice.shape[2]
            N_samples = kspace_slice.shape[-1]
            kspace = rearrange(kspace_slice, 't c sp sam -> t sp c sam')
            kspace_flat = kspace.contiguous().view(total_spokes, self.N_coils, N_samples)
            # kspace_flat = kspace.contiguous().reshape(total_spokes, self.N_coils, N_samples)

            if self.spf_aug:
                print("setting random spokes per frame...")
                spokes_per_frame = random.choices(self.spokes_range, self.spf_weights, k=1)[0]
            else:
                spokes_per_frame = self.spokes_per_frame
                print(f"training with fixed spokes per frame ({spokes_per_frame})")

            N_time = total_spokes // spokes_per_frame
            kspace_binned = kspace_flat.view(N_time, spokes_per_frame, self.N_coils, N_samples)
            kspace_slice = rearrange(kspace_binned, 't sp c sam -> t c sp sam')
        else:
            N_time = self.N_time
            N_samples = kspace_slice.shape[-1]
            spokes_per_frame = kspace_slice.shape[-2]

        real_part = kspace_slice.real
        imag_part = kspace_slice.imag
        kspace_final = torch.stack([real_part, imag_part], dim=0).float()

        kspace_final = torch.flip(kspace_final, dims=[-1])

        csmap = torch.from_numpy(csmap)
        csmap_tensor = torch.rot90(csmap, k=2, dims=[-2, -1])
        csmap = csmap_tensor.numpy()

        return kspace_final, csmap, N_samples, spokes_per_frame, N_time
    

class SimulatedDataset(Dataset):
    """
    Dataset for loading the simulated data generated by your script.
    It loads the simulated k-space, coil sensitivity maps, and the
    ground truth dynamic image (DRO).
    """
    def __init__(self, root_dir, raw_kspace_path, model_type, patient_ids, dataset_key, grasp_slice_idx=95, spokes_per_frame=36, num_frames=8):

        self.root_dir = root_dir
        self.raw_kspace_path = raw_kspace_path
        self.patient_ids = patient_ids
        self.model_type = model_type
        self.spokes_per_frame = spokes_per_frame
        self.num_frames = num_frames
        self.grasp_slice_idx = grasp_slice_idx
        self.dataset_key = dataset_key
        self.slice_map = load_slice_map(SLICE_MAP_PATH)
        self._update_sample_paths()


        self.TISSUE_NAMES = [
            'glandular', 'benign', 'malignant', 'muscle',
            'skin', 'liver', 'heart', 'vascular'
        ]

    def _update_sample_paths(self):
        self.dro_dir = os.path.join(self.root_dir, f'dro_{self.num_frames}frames')

        # Find all sample directories, e.g., 'sample_001_sub1', 'sample_002_sub2', etc.
        self.sample_paths = sorted(glob.glob(os.path.join(self.dro_dir, 'sample_*')))
        if not self.sample_paths:
            raise FileNotFoundError(f"No sample directories found in {self.dro_dir}. "
                                    "Please check the path to your simulated dataset.")
        
        # filter file list by patient ID substring
        filtered = []
        for fp in self.sample_paths:
            fname = os.path.basename(fp)
            # Check if any patient_id appears in the filename
            if any(pid in fname for pid in self.patient_ids):
                filtered.append(fp)

        self.sample_paths = filtered

        print(f"Found {len(self.sample_paths)} simulated samples in {self.dro_dir} for {self.num_frames} frames.")


    def get_fastMRI_id(self, sample_dir):

        sample_file = os.path.basename(sample_dir)
        id_map = pd.read_csv('data/DROSubID_vs_fastMRIbreastID.csv')

        dro_id = int(sample_file.split("_")[1])
        dro_row = id_map[id_map["DRO"] == dro_id]
        fastmri_id = int(dro_row["fastMRIbreast"].iloc[0])

        return fastmri_id
  
            
    def __len__(self):
        return len(self.sample_paths)
    


    def __getitem__(self, idx):
        sample_dir = self.sample_paths[idx]


        # Load the data from .npy files
        csmaps = np.load(os.path.join(sample_dir, 'csmaps.npy'))
        dro = np.load(os.path.join(sample_dir, 'dro_ground_truth.npz'))
        grasp_path = os.path.join(sample_dir, f'grasp_spf{self.spokes_per_frame}_frames{self.num_frames}.npy')

        grasp_recon = np.load(grasp_path)

        # GRASP Recon: (H, W, T) -> (2, T, H, W) [real/imag, time, h, w]
        grasp_recon_torch = torch.from_numpy(grasp_recon).permute(2, 0, 1) # T, H, W
        grasp_recon_torch = torch.stack([grasp_recon_torch.real, grasp_recon_torch.imag], dim=0)

        grasp_recon_torch = torch.flip(grasp_recon_torch, dims=[-3])
        grasp_recon_torch = torch.rot90(grasp_recon_torch, k=3, dims=[-3,-1])

        kspace_path = os.path.join(sample_dir, f'simulated_kspace_spf{self.spokes_per_frame}_frames{self.num_frames}.npy')

        if os.path.exists(kspace_path):
            kspace_complex = np.load(kspace_path, allow_pickle=True)
            kspace_torch = torch.from_numpy(kspace_complex)
        else:
            kspace_torch = kspace_path


        # load raw k-space and GRASP recon
        fastmri_id = self.get_fastMRI_id(sample_dir)
        patient_id = f"fastMRI_breast_{fastmri_id:03d}_2"
        slice_idx = self.slice_map.get(patient_id, self.grasp_slice_idx)
        if slice_idx is None or slice_idx < 0:
            slice_idx = self.grasp_slice_idx

        raw_grasp_path = os.path.join(os.path.dirname(self.raw_kspace_path), f'{patient_id}/grasp_recon_{self.spokes_per_frame}spf_{self.num_frames}frames_slice{slice_idx}.npy')
        raw_kspace_path = os.path.join(self.raw_kspace_path, f'{patient_id}.h5')
        raw_csmap_path = os.path.join(os.path.dirname(self.raw_kspace_path), f'cs_maps/{patient_id}_cs_maps/cs_map_slice_{slice_idx:03d}.npy')
        
        raw_csmaps = np.load(raw_csmap_path)
        # raw_csmaps = rearrange(raw_csmaps, 'c b h w -> b c h w')

        raw_grasp_recon = np.load(raw_grasp_path).squeeze()


        # GRASP Recon: (H, W, T) -> (2, T, H, W) [real/imag, time, h, w]
        raw_grasp_recon = torch.from_numpy(raw_grasp_recon).permute(2, 0, 1) # T, H, W
        raw_grasp_recon = torch.stack([raw_grasp_recon.real, raw_grasp_recon.imag], dim=0)

        raw_grasp_recon = torch.flip(raw_grasp_recon, dims=[-3])
        raw_grasp_recon = torch.rot90(raw_grasp_recon, k=1, dims=[-3,-1])


        with h5py.File(raw_kspace_path, "r") as f:
            raw_kspace_slice = torch.tensor(f[self.dataset_key][slice_idx])

        # time-bin k-space
        N_spokes_prep = self.num_frames * self.spokes_per_frame

        ksp_redu = raw_kspace_slice[:, :N_spokes_prep, :] # (16, 288, 640)
        ksp_prep = np.swapaxes(ksp_redu, 0, 1) # (288, 16, 640)
        ksp_prep_shape = ksp_prep.shape
        ksp_prep = np.reshape(ksp_prep, [self.num_frames, self.spokes_per_frame] + list(ksp_prep_shape[1:]))

        ksp_prep = torch.flip(ksp_prep, dims=[-1])

        raw_kspace_slice = rearrange(ksp_prep, 't sp c sam -> c (sp sam) t').to(kspace_torch.dtype)


        ground_truth_complex = dro['ground_truth_images']

        parMap = dro['parMap']
        aif = dro['aif']
        S0 = dro['S0']
        T10 = dro['T10']
        # mask = dro['mask']

        # ==========================================================
        # --- RECONSTRUCT THE MASK DICTIONARY ---
        # ==========================================================
        mask_dictionary_rebuilt = {}
        for tissue_name in self.TISSUE_NAMES:
            # Check if the key for this tissue (e.g., 'malignant') exists in the file
            if tissue_name in dro:
                # Load the boolean array and add it to the dictionary
                mask_dictionary_rebuilt[tissue_name] = dro[tissue_name]
        
        # 'mask' is now the dictionary of boolean arrays, just like your functions expect
        mask = mask_dictionary_rebuilt


        # --- Convert to PyTorch Tensors ---
        # Ground truth: (H, W, T) -> (2, T, H, W) [real/imag, time, h, w]
        ground_truth_torch = torch.from_numpy(ground_truth_complex).permute(2, 0, 1) # T, H, W
        ground_truth_torch = torch.stack([ground_truth_torch.real, ground_truth_torch.imag], dim=0)

        # CSMaps: (H, W, C) -> (1, C, H, W) [batch, coils, h, w]
        csmaps_torch = torch.from_numpy(csmaps).permute(2, 0, 1).unsqueeze(0)
        raw_csmaps_torch = torch.from_numpy(raw_csmaps)#.permute(2, 0, 1).unsqueeze(0)
        raw_csmaps_torch = rearrange(raw_csmaps_torch, 'c b h w -> b c h w').to(csmaps_torch.dtype)

        raw_csmaps_torch = torch.rot90(raw_csmaps_torch, k=2, dims=[-2, -1])
        raw_csmaps_torch = raw_csmaps_torch.numpy()

        return kspace_torch, csmaps_torch, ground_truth_torch, grasp_recon_torch, mask, grasp_path, raw_kspace_slice, raw_grasp_recon, raw_csmaps_torch #, parMap, aif, S0, T10, mask
    
    


class SimulatedSPFDataset(Dataset):
    """
    Dataset for loading the simulated data generated by your script.
    It loads the simulated k-space, coil sensitivity maps, and the
    ground truth dynamic image (DRO).
    """
    def __init__(self, root_dir, raw_kspace_path, model_type, patient_ids, dataset_key, grasp_slice_idx=95):
        self.model_type = model_type
        self.root_dir = root_dir
        self.patient_ids = patient_ids

        self.raw_kspace_path = raw_kspace_path
        self.grasp_slice_idx = grasp_slice_idx
        self.dataset_key = dataset_key
        self.slice_map = load_slice_map(SLICE_MAP_PATH)

        # set default parameters to be changed before each call
        self.spokes_per_frame = 16
        self.num_frames = 18

        # Initialize sample paths based on default parameters
        self._update_sample_paths()
        

        self.TISSUE_NAMES = [
            'glandular', 'benign', 'malignant', 'muscle',
            'skin', 'liver', 'heart', 'vascular'
        ]

    def _update_sample_paths(self):
        self.dro_dir = os.path.join(self.root_dir, f'dro_{self.num_frames}frames')

        # Find all sample directories, e.g., 'sample_001_sub1', 'sample_002_sub2', etc.
        self.sample_paths = sorted(glob.glob(os.path.join(self.dro_dir, 'sample_*')))
        if not self.sample_paths:
            raise FileNotFoundError(f"No sample directories found in {self.dro_dir}. "
                                    "Please check the path to your simulated dataset.")
        
        # filter file list by patient ID substring
        filtered = []
        for fp in self.sample_paths:
            fname = os.path.basename(fp)
            # Check if any patient_id appears in the filename
            if any(pid in fname for pid in self.patient_ids):
                filtered.append(fp)

        self.sample_paths = filtered

        print(f"Found {len(self.sample_paths)} simulated samples in {self.dro_dir} for {self.num_frames} frames.")


    def get_fastMRI_id(self, sample_dir):

        sample_file = os.path.basename(sample_dir)
        id_map = pd.read_csv('data/DROSubID_vs_fastMRIbreastID.csv')

        dro_id = int(sample_file.split("_")[1])
        dro_row = id_map[id_map["DRO"] == dro_id]
        fastmri_id = int(dro_row["fastMRIbreast"].iloc[0])

        return fastmri_id
    
    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        sample_dir = self.sample_paths[idx]

        print(f"  Testing {self.spokes_per_frame} spokes/frame with {self.num_frames} frames.")


        # Load the data from .npy files
        csmaps = np.load(os.path.join(sample_dir, 'csmaps.npy'))
        dro = np.load(os.path.join(sample_dir, 'dro_ground_truth.npz'))

        grasp_path = os.path.join(sample_dir, f'grasp_spf{self.spokes_per_frame}_frames{self.num_frames}.npy')
        
        if os.path.exists(grasp_path):
            grasp_recon = np.load(grasp_path)

            # GRASP Recon: (H, W, T) -> (2, T, H, W) [real/imag, time, h, w]
            grasp_recon_torch = torch.from_numpy(grasp_recon).permute(2, 0, 1) # T, H, W
            grasp_recon_torch = torch.stack([grasp_recon_torch.real, grasp_recon_torch.imag], dim=0)

            grasp_recon_torch = torch.flip(grasp_recon_torch, dims=[-3])
            grasp_recon_torch = torch.rot90(grasp_recon_torch, k=3, dims=[-3,-1])

        else:
            grasp_recon_torch = 0


        ground_truth_complex = dro['ground_truth_images']

        # SELECT TIME WINDOW
        # ground_truth_complex = ground_truth_complex[..., self.window]

        smap_torch = rearrange(torch.tensor(csmaps), 'h w c -> c h w').unsqueeze(0)
        simImg_torch = torch.tensor(ground_truth_complex).to(torch.cfloat)



        parMap = dro['parMap']
        aif = dro['aif']
        S0 = dro['S0']
        T10 = dro['T10']
        # mask = dro['mask']

        # ==========================================================
        # --- RECONSTRUCT THE MASK DICTIONARY ---
        # ==========================================================
        mask_dictionary_rebuilt = {}
        for tissue_name in self.TISSUE_NAMES:
            # Check if the key for this tissue (e.g., 'malignant') exists in the file
            if tissue_name in dro:
                # Load the boolean array and add it to the dictionary
                mask_dictionary_rebuilt[tissue_name] = dro[tissue_name]
        
        # 'mask' is now the dictionary of boolean arrays, just like your functions expect
        mask = mask_dictionary_rebuilt


        # --- Convert to PyTorch Tensors ---
        # Ground truth: (H, W, T) -> (2, T, H, W) [real/imag, time, h, w]
        ground_truth_torch = torch.from_numpy(ground_truth_complex).permute(2, 0, 1) # T, H, W
        ground_truth_torch = torch.stack([ground_truth_torch.real, ground_truth_torch.imag], dim=0)


        # load raw k-space and GRASP recon
        fastmri_id = self.get_fastMRI_id(sample_dir)
        patient_id = f"fastMRI_breast_{fastmri_id:03d}_2"
        slice_idx = self.slice_map.get(patient_id, self.grasp_slice_idx)
        if slice_idx is None or slice_idx < 0:
            slice_idx = self.grasp_slice_idx

        raw_grasp_path = os.path.join(os.path.dirname(self.raw_kspace_path), f'{patient_id}/grasp_recon_{self.spokes_per_frame}spf_{self.num_frames}frames_slice{slice_idx}.npy')
        raw_kspace_path = os.path.join(self.raw_kspace_path, f'{patient_id}.h5')
        raw_csmap_path = os.path.join(os.path.dirname(self.raw_kspace_path), f'cs_maps/{patient_id}_cs_maps/cs_map_slice_{slice_idx:03d}.npy')
        
        raw_csmaps = np.load(raw_csmap_path)
        # raw_csmaps = rearrange(raw_csmaps, 'c b h w -> b c h w')

        raw_grasp_recon = np.load(raw_grasp_path).squeeze()


        # GRASP Recon: (H, W, T) -> (2, T, H, W) [real/imag, time, h, w]
        raw_grasp_recon = torch.from_numpy(raw_grasp_recon).permute(2, 0, 1) # T, H, W
        raw_grasp_recon = torch.stack([raw_grasp_recon.real, raw_grasp_recon.imag], dim=0)

        raw_grasp_recon = torch.flip(raw_grasp_recon, dims=[-3])
        raw_grasp_recon = torch.rot90(raw_grasp_recon, k=1, dims=[-3,-1])


        with h5py.File(raw_kspace_path, "r") as f:
            raw_kspace_slice = torch.tensor(f[self.dataset_key][slice_idx])

        # time-bin k-space
        N_spokes_prep = self.num_frames * self.spokes_per_frame

        ksp_redu = raw_kspace_slice[:, :N_spokes_prep, :] # (16, 288, 640)
        ksp_prep = np.swapaxes(ksp_redu, 0, 1) # (288, 16, 640)
        ksp_prep_shape = ksp_prep.shape
        ksp_prep = np.reshape(ksp_prep, [self.num_frames, self.spokes_per_frame] + list(ksp_prep_shape[1:]))
        
        ksp_prep = torch.flip(ksp_prep, dims=[-1])

        raw_kspace_slice = rearrange(ksp_prep, 't sp c sam -> c (sp sam) t').to(smap_torch.dtype)

        raw_csmaps_torch = torch.from_numpy(raw_csmaps)#.permute(2, 0, 1).unsqueeze(0)
        raw_csmaps_torch = rearrange(raw_csmaps_torch, 'c b h w -> b c h w').to(smap_torch.dtype)

        raw_csmaps_torch = torch.rot90(raw_csmaps_torch, k=2, dims=[-2, -1])
        raw_csmaps_torch = raw_csmaps_torch.numpy()


        return smap_torch, simImg_torch, grasp_recon_torch, mask, grasp_path, raw_kspace_slice, raw_csmaps_torch, raw_grasp_recon #, parMap, aif, S0, T10, mask
