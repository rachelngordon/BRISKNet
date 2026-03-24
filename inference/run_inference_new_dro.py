"""Run DRO and non-DRO inference for one or more experiment checkpoints. Run: python3 -m inference.run_inference_new_dro --help"""

import argparse
import csv
import glob
import json
import math
import os
import shlex
import shutil
import statistics
import sys
import time
import warnings
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import h5py.h5t as h5t
import numpy as np
import torch
import torchkbnufft as tkbn
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from einops import rearrange
from radial import to_torch_complex

REPO_ROOT = Path(__file__).resolve().parents[1]
JOB_SCRIPTS_DIR = REPO_ROOT / "job-scripts"
if str(JOB_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(JOB_SCRIPTS_DIR))

from cluster_paths import apply_cluster_paths
from dataloader import SLICE_MAP_PATH, load_slice_map
from inference.eval import (
    eval_grasp,
    eval_sample,
    eval_zf,
    compute_ssdu_kspace_nmse,
    compute_ssdu_kspace_nmse_grasp,
    calc_dc,
    calc_dc_psnr,
    _resolve_baseline_frames,
    _load_tumor_mask,
    _load_slice_map,
    _resolve_plot_label,
)
from model_factory import build_recon_model
from radial import MCNUFFT
from utils import (
    GRASPRecon_from_ktraj,
    prep_nufft,
    remove_module_prefix,
    set_seed,
    sliding_window_inference,
)

DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "val_inference_logs.json"

# Silence torchmetrics/torch FutureWarning about torch.load(weights_only=...) defaults.
warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch\.load` with `weights_only=False`.*",
    category=FutureWarning,
)


def _h5py_float_dtype(type_id: h5t.TypeID) -> np.dtype:
    size = type_id.get_size()
    if size == 4:
        return np.float32
    if size == 8:
        return np.float64
    raise TypeError(f"Unsupported float size for HDF5 dtype: {size}")


def _read_h5_float(dset: h5py.Dataset) -> np.ndarray:
    type_id = dset.id.get_type()
    if type_id.get_class() != h5t.FLOAT:
        raise TypeError(f"Expected float dataset; got class={type_id.get_class()}")
    np_dtype = _h5py_float_dtype(type_id)
    arr = np.empty(dset.shape, dtype=np_dtype)
    dset.read_direct(arr)
    return arr


def _read_h5_numeric(dset: h5py.Dataset) -> np.ndarray:
    type_id = dset.id.get_type()
    type_class = type_id.get_class()
    if type_class == h5t.FLOAT:
        return _read_h5_float(dset)
    if type_class != h5t.INTEGER:
        raise TypeError(f"Expected numeric dataset; got class={type_class}")
    size = type_id.get_size()
    signed = type_id.get_sign() == h5t.SGN_2
    if size == 1:
        np_dtype = np.int8 if signed else np.uint8
    elif size == 2:
        np_dtype = np.int16 if signed else np.uint16
    elif size == 4:
        np_dtype = np.int32 if signed else np.uint32
    elif size == 8:
        np_dtype = np.int64 if signed else np.uint64
    else:
        raise TypeError(f"Unsupported integer size for HDF5 dtype: {size}")
    arr = np.empty(dset.shape, dtype=np_dtype)
    dset.read_direct(arr)
    return arr


def _read_h5_complex(dset: h5py.Dataset) -> np.ndarray:
    type_id = dset.id.get_type()
    type_class = type_id.get_class()
    if type_class == h5t.COMPOUND:
        # MATLAB v7.3 complex arrays are stored as compound datasets with real/imag fields.
        memtype = h5t.create(h5t.COMPOUND, type_id.get_size())
        names = []
        formats = []
        offsets = []
        for idx in range(type_id.get_nmembers()):
            name = type_id.get_member_name(idx)
            name_str = name.decode() if isinstance(name, (bytes, bytearray)) else str(name)
            member = type_id.get_member_type(idx)
            if member.get_class() != h5t.FLOAT:
                raise TypeError("Unsupported compound member type in complex dataset.")
            np_fmt = _h5py_float_dtype(member)
            memtype.insert(name, type_id.get_member_offset(idx), h5t.NATIVE_FLOAT if np_fmt == np.float32 else h5t.NATIVE_DOUBLE)
            names.append(name_str)
            formats.append(np_fmt)
            offsets.append(type_id.get_member_offset(idx))
        np_dtype = np.dtype(
            {
                "names": names,
                "formats": formats,
                "offsets": offsets,
                "itemsize": type_id.get_size(),
            }
        )
        arr = np.empty(dset.shape, dtype=np_dtype)
        dset.id.read(h5py.h5s.ALL, h5py.h5s.ALL, arr, memtype)
        if "real" in arr.dtype.names and "imag" in arr.dtype.names:
            return arr["real"] + 1j * arr["imag"]
        if len(arr.dtype.names) >= 2:
            return arr[arr.dtype.names[0]] + 1j * arr[arr.dtype.names[1]]
        raise ValueError("Compound dataset did not contain real/imag fields.")
    if type_class == h5t.FLOAT:
        real = _read_h5_float(dset)
        return real.astype(np.complex64)
    raise TypeError(f"Unsupported HDF5 dtype class for complex read: {type_class}")


def _extract_int_suffix(name: str, token: str, suffix: str) -> int | None:
    if token not in name or not name.endswith(suffix):
        return None
    tail = name.split(token, 1)[1]
    if not tail.endswith(suffix):
        return None
    num_str = tail[: -len(suffix)]
    if num_str.isdigit():
        return int(num_str)
    return None


def _available_frames(paths: list[str], token: str, suffix: str) -> list[int]:
    frames: list[int] = []
    for path in paths:
        base = os.path.basename(path)
        num = _extract_int_suffix(base, token, suffix)
        if num is not None:
            frames.append(num)
    return sorted(set(frames))


def _normalize_root_dir(root_dir: str) -> str:
    return os.path.realpath(root_dir).rstrip(os.sep)


def _spf_suffix_policy(root_dir: str) -> str:
    normalized = _normalize_root_dir(root_dir)
    if normalized == "/net/scratch2/rachelgordon/dro_var_frames":
        return "require_frames"
    if normalized == "/net/scratch2/rachelgordon/dro":
        return "require_no_frames"
    return "flex"


def _kspace_ext_policy(root_dir: str) -> str:
    normalized = _normalize_root_dir(root_dir)
    if normalized == "/net/scratch2/rachelgordon/dro_var_frames":
        return "npy"
    if normalized == "/net/scratch2/rachelgordon/dro":
        return "mat"
    return "flex"


def _prep_nufft_from_dro_traj(
    kspace_path: str,
    spokes_per_frame: int,
    num_frames: int,
    expected_samples: int | None = None,
    traj_method: str = "get_traj",
):
    if kspace_path.endswith(".npy"):
        kspace = np.load(kspace_path, mmap_mode="r")
        if kspace.ndim == 4 and kspace.shape[-1] == 2 and not np.iscomplexobj(kspace):
            kspace = kspace[..., 0] + 1j * kspace[..., 1]
        if kspace.ndim != 3:
            raise ValueError(f"Unexpected kspace shape {kspace.shape} in {kspace_path}.")
        if kspace.shape[-1] != num_frames:
            if kspace.shape[1] == num_frames:
                kspace = np.transpose(kspace, (0, 2, 1))
            elif kspace.shape[0] == num_frames:
                kspace = np.transpose(kspace, (1, 2, 0))
            else:
                raise ValueError(
                    f"kspace frames ({kspace.shape[-1]}) != expected num_frames ({num_frames}) "
                    f"for {kspace_path}. Use --eval_frames to match the DRO files."
                )
        total_samples = kspace.shape[1]
        if total_samples % spokes_per_frame != 0:
            raise ValueError(
                f"kspace samples ({total_samples}) not divisible by spf {spokes_per_frame} "
                f"for {kspace_path}."
            )
        samples = total_samples // spokes_per_frame
        if expected_samples is not None and int(samples) != int(expected_samples):
            raise ValueError(
                f"kspace samples/spoke ({samples}) != expected_samples ({expected_samples})."
            )
        ktraj, dcomp, nufft_ob, adjnufft_ob = prep_nufft(
            int(samples), int(spokes_per_frame), int(num_frames), traj_method=traj_method
        )
        return ktraj, dcomp, nufft_ob, adjnufft_ob, int(samples)

    with h5py.File(kspace_path, "r") as f:
        if "traj" not in f:
            raise KeyError(f"{kspace_path} missing required key 'traj'.")
        traj = _read_h5_complex(f["traj"]).astype(np.complex64)

    if traj.ndim != 2:
        raise ValueError(f"Unexpected traj shape {traj.shape} in {kspace_path}.")
    total_spokes, samples = traj.shape
    expected_spokes = int(spokes_per_frame) * int(num_frames)
    if samples == expected_spokes and total_spokes != expected_spokes:
        traj = traj.T
        total_spokes, samples = traj.shape
    if total_spokes != expected_spokes:
        raise ValueError(
            f"traj spokes ({total_spokes}) != expected ({expected_spokes}) "
            f"for spf={spokes_per_frame}, frames={num_frames}."
        )
    if expected_samples is not None and int(samples) != int(expected_samples):
        raise ValueError(
            f"traj samples ({samples}) != expected_samples ({expected_samples})."
        )

    max_abs = float(np.max(np.abs(traj)))
    if max_abs <= 1.0:
        traj = traj * (2 * np.pi)
        print(f"[DRO traj] Scaling normalized traj by 2π (max_abs={max_abs:.3g}).")

    traj = traj.reshape(num_frames, spokes_per_frame, samples)
    kx = traj.real
    ky = traj.imag
    # ktraj = np.stack([kx, ky], axis=0)  # (2, T, spf, samples)
    ktraj = np.stack([ky, kx], axis=0)  # (2, T, spf, samples)
    ktraj = ktraj.reshape(2, num_frames, spokes_per_frame * samples)
    ktraj = np.transpose(ktraj, (0, 2, 1))  # (2, spf*samples, T)
    ktraj_torch = torch.tensor(ktraj, dtype=torch.float32)

    im_size = (int(samples // 2), int(samples // 2))
    grid_size = (int(samples), int(samples))
    dcomp = np.zeros((spokes_per_frame * samples, num_frames), dtype=np.complex64)
    for t in range(num_frames):
        dcomp_t = tkbn.calc_density_compensation_function(ktraj=ktraj_torch[:, :, t], im_size=im_size)
        dcomp_t = dcomp_t.squeeze()
        dcomp[:, t] = dcomp_t.detach().cpu().numpy().astype(np.complex64)
    dcomp_torch = torch.tensor(dcomp, dtype=torch.complex64)

    nufft_ob = tkbn.KbNufft(im_size=im_size, grid_size=grid_size)
    adjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size, grid_size=grid_size)

    return ktraj_torch, dcomp_torch, nufft_ob, adjnufft_ob, int(samples)


def _load_dro_fastmri_map(mapping_csv: str = "data/DROSubID_vs_fastMRIbreastID.csv") -> dict[int, int]:
    mapping = {}
    with open(mapping_csv, newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            try:
                dro_id = int(row["DRO"])
                fastmri_id = int(row["fastMRIbreast"])
            except (KeyError, ValueError):
                continue
            mapping[dro_id] = fastmri_id
    if not mapping:
        raise ValueError(f"No DRO-to-fastMRI mappings found in {mapping_csv}.")
    return mapping


class NewDROMatDataset(Dataset):
    """
    Dataset loader for the new MATLAB v7.3 DRO files stored as flat .mat files.
    Expects files:
      - <sample>_dro_<frames>frames.mat (contains simImg, smap, mask)
        - legacy: <sample>_dro.mat
      - <sample>_kspace_<spf>spf_<frames>frames.npy (contains kspace; dro_var_frames)
        - legacy: <sample>_kspace_<spf>spf.mat (dro)
        - legacy: <sample>_kspace_<spf>spf.npy
      - <sample>_recon_<spf>spf_<frames>frames.mat (contains grasp_bart) when dro_csmaps_source=original
        - legacy: <sample>_recon_<spf>spf.mat
      - espirit_grasp_recons/grasp_<sample>_<spf>spf_<frames>frames.npy when dro_csmaps_source=espirit
        - legacy: espirit_grasp_recons/grasp_<sample>_<spf>spf.npy
    """

    def __init__(
        self,
        root_dir: str,
        raw_kspace_path: str,
        model_type: str,
        patient_ids: list[str],
        dataset_key: str,
        grasp_slice_idx: int = 95,
        spokes_per_frame: int = 36,
        num_frames: int = 8,
        dro_csmaps_source: str = "espirit",
        espirit_csmaps_dir: str | None = None,
        espirit_grasp_recons_dir: str | None = None,
        skip_raw_eval_if_invalid_slice: bool = False,
        skip_raw_grasp_metrics: bool = False,
    ):
        self.root_dir = root_dir
        self.raw_kspace_path = raw_kspace_path
        self.model_type = model_type
        self.patient_ids = patient_ids or []
        self.dataset_key = dataset_key
        self.grasp_slice_idx = grasp_slice_idx
        self.spokes_per_frame = int(spokes_per_frame)
        self.num_frames = int(num_frames)
        self.raw_total_spokes = 288
        if self.raw_total_spokes % self.spokes_per_frame != 0:
            raise ValueError(
                f"raw_total_spokes ({self.raw_total_spokes}) is not divisible by "
                f"spokes_per_frame ({self.spokes_per_frame})."
            )
        self.raw_num_frames = self.raw_total_spokes // self.spokes_per_frame
        self.dro_csmaps_source = dro_csmaps_source
        self.espirit_csmaps_dir = espirit_csmaps_dir
        self.espirit_grasp_recons_dir = espirit_grasp_recons_dir
        self.spf_suffix_policy = _spf_suffix_policy(self.root_dir)
        self.kspace_ext_policy = _kspace_ext_policy(self.root_dir)
        if self.dro_csmaps_source not in ("original", "espirit"):
            raise ValueError(
                f"Unsupported dro_csmaps_source '{self.dro_csmaps_source}'. "
                "Expected 'original' or 'espirit'."
            )
        self.skip_raw_eval_if_invalid_slice = bool(skip_raw_eval_if_invalid_slice)
        self.skip_raw_grasp_metrics = bool(skip_raw_grasp_metrics)
        self.slice_map = load_slice_map(SLICE_MAP_PATH)
        self.dro_to_fastmri = _load_dro_fastmri_map()
        self.sample_ids = self._collect_sample_ids()

        self.TISSUE_NAMES = [
            "glandular",
            "benign",
            "malignant",
            "muscle",
            "skin",
            "liver",
            "heart",
            "vascular",
        ]

    def _collect_sample_ids(self) -> list[str]:
        frame_pattern = os.path.join(
            self.root_dir, f"sample_*_dro_{self.num_frames}frames.mat"
        )
        dro_files = glob.glob(frame_pattern)
        if not dro_files:
            legacy_pattern = os.path.join(self.root_dir, "sample_*_dro.mat")
            dro_files = glob.glob(legacy_pattern)
        if not dro_files:
            alt_pattern = os.path.join(self.root_dir, "sample_*_dro_*frames.mat")
            alt_files = glob.glob(alt_pattern)
            if alt_files:
                frames = _available_frames(alt_files, token="_dro_", suffix="frames.mat")
                frames_str = ", ".join(str(f) for f in frames) if frames else "unknown"
                raise FileNotFoundError(
                    f"No DRO .mat files found for {self.num_frames} frames in {self.root_dir}. "
                    f"Available frames: {frames_str}."
                )
            raise FileNotFoundError(f"No DRO .mat files found in {self.root_dir}.")

        available = []
        for fp in dro_files:
            base = os.path.basename(fp)
            if base.endswith("_dro.mat"):
                available.append(base[: -len("_dro.mat")])
            elif "_dro_" in base and base.endswith("frames.mat"):
                available.append(base.split("_dro_", 1)[0])
        available = sorted(set(available))
        if not self.patient_ids:
            return available

        missing = [pid for pid in self.patient_ids if pid not in available]
        if missing:
            missing_str = ", ".join(missing)
            raise FileNotFoundError(
                f"Missing DRO samples in {self.root_dir}: {missing_str}"
            )

        filtered = [sid for sid in available if any(pid in sid for pid in self.patient_ids)]
        if not filtered:
            raise FileNotFoundError(
                f"No DRO samples matched patient IDs in {self.root_dir}."
            )
        return filtered

    def __len__(self):
        return len(self.sample_ids)

    def _load_dro_mat(self, sample_id: str):
        dro_path = self._resolve_dro_mat_path(sample_id)
        with h5py.File(dro_path, "r") as f:
            if "simImg" not in f:
                raise KeyError(f"{dro_path} missing required key 'simImg'.")
            if "smap" not in f:
                raise KeyError(f"{dro_path} missing required key 'smap'.")
            sim_img = _read_h5_float(f["simImg"]).astype(np.float32)
            smap = _read_h5_complex(f["smap"]).astype(np.complex64)

            mask_dict = {}
            if "mask" in f and isinstance(f["mask"], h5py.Group):
                mask_group = f["mask"]
                for tissue in self.TISSUE_NAMES:
                    if tissue in mask_group:
                        mask_arr = _read_h5_numeric(mask_group[tissue])
                        mask_dict[tissue] = mask_arr > 0

        if smap.ndim != 3:
            raise ValueError(f"Unexpected smap shape {smap.shape} in {dro_path}.")
        if smap.shape[0] != 16 and smap.shape[-1] == 16:
            smap = np.transpose(smap, (2, 0, 1))

        if sim_img.ndim != 3:
            raise ValueError(f"Unexpected simImg shape {sim_img.shape} in {dro_path}.")
        if sim_img.shape[0] != self.num_frames:
            raise ValueError(
                f"simImg frames ({sim_img.shape[0]}) != expected num_frames ({self.num_frames}) "
                f"for {sample_id}. Use --eval_frames to match the DRO files."
            )
        return sim_img, smap, mask_dict

    def _load_kspace_mat(self, sample_id: str) -> np.ndarray:
        kspace_path = self._resolve_kspace_mat_path(sample_id)
        if kspace_path.endswith(".npy"):
            kspace = np.load(kspace_path)
            if kspace.ndim == 4 and kspace.shape[-1] == 2 and not np.iscomplexobj(kspace):
                kspace = kspace[..., 0] + 1j * kspace[..., 1]
        else:
            with h5py.File(kspace_path, "r") as f:
                if "kspace" not in f:
                    raise KeyError(f"{kspace_path} missing required key 'kspace'.")
                kspace = _read_h5_complex(f["kspace"])

        if kspace.ndim != 3:
            raise ValueError(f"Unexpected kspace shape {kspace.shape} in {kspace_path}.")
        if kspace_path.endswith(".mat"):
            kspace = kspace.astype(np.complex64, copy=False)
            total_spokes = kspace.shape[1]
            if total_spokes % self.spokes_per_frame != 0:
                raise ValueError(
                    f"kspace spokes dimension {total_spokes} not divisible by spf {self.spokes_per_frame} "
                    f"for {sample_id}."
                )
            inferred_frames = total_spokes // self.spokes_per_frame
            if inferred_frames != self.num_frames:
                raise ValueError(
                    f"kspace frames ({inferred_frames}) != expected num_frames ({self.num_frames}) "
                    f"for {sample_id}. Use --eval_frames to match the DRO files."
                )
            kspace = kspace.reshape(
                kspace.shape[0], self.num_frames, self.spokes_per_frame, kspace.shape[2]
            )
            kspace = rearrange(kspace, "c t sp sam -> c (sp sam) t")
            return kspace.astype(np.complex64, copy=False)
        if kspace.shape[-1] != self.num_frames:
            if kspace.shape[1] == self.num_frames:
                kspace = np.transpose(kspace, (0, 2, 1))
            elif kspace.shape[0] == self.num_frames:
                kspace = np.transpose(kspace, (1, 2, 0))
            else:
                raise ValueError(
                    f"kspace frames ({kspace.shape[-1]}) != expected num_frames ({self.num_frames}) "
                    f"for {sample_id}. Use --eval_frames to match the DRO files."
                )

        total_samples = kspace.shape[1]
        if total_samples % self.spokes_per_frame != 0:
            raise ValueError(
                f"kspace samples ({total_samples}) not divisible by spf {self.spokes_per_frame} "
                f"for {sample_id}."
            )
        kspace = kspace.astype(np.complex64, copy=False)
        return kspace

    def _resolve_dro_mat_path(self, sample_id: str) -> str:
        candidates = [
            os.path.join(self.root_dir, f"{sample_id}_dro_{self.num_frames}frames.mat"),
            os.path.join(self.root_dir, f"{sample_id}_dro.mat"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        alt_pattern = os.path.join(self.root_dir, f"{sample_id}_dro_*frames.mat")
        alt_files = glob.glob(alt_pattern)
        if alt_files:
            frames = _available_frames(alt_files, token="_dro_", suffix="frames.mat")
            frames_str = ", ".join(str(f) for f in frames) if frames else "unknown"
            raise FileNotFoundError(
                f"Missing DRO file for {sample_id} with {self.num_frames} frames. "
                f"Available frames: {frames_str}."
            )
        raise FileNotFoundError(f"Missing DRO file: {candidates[0]}")

    def _resolve_kspace_mat_path(self, sample_id: str) -> str:
        if self.kspace_ext_policy == "npy":
            extensions = [".npy"]
        elif self.kspace_ext_policy == "mat":
            extensions = [".mat"]
        else:
            extensions = [".npy", ".mat"]

        with_frames = [
            os.path.join(
                self.root_dir,
                f"{sample_id}_kspace_{self.spokes_per_frame}spf_{self.num_frames}frames{ext}",
            )
            for ext in extensions
        ]
        no_frames = [
            os.path.join(
                self.root_dir, f"{sample_id}_kspace_{self.spokes_per_frame}spf{ext}"
            )
            for ext in extensions
        ]
        if self.spf_suffix_policy == "require_frames":
            candidates = with_frames
        elif self.spf_suffix_policy == "require_no_frames":
            candidates = no_frames
        else:
            candidates = with_frames + no_frames
        for path in candidates:
            if os.path.exists(path):
                return path
        if self.spf_suffix_policy == "require_no_frames":
            raise FileNotFoundError(f"Missing kspace file: {candidates[0]}")
        token = f"_kspace_{self.spokes_per_frame}spf_"
        frames_found: set[int] = set()
        for ext in extensions:
            alt_pattern = os.path.join(
                self.root_dir,
                f"{sample_id}_kspace_{self.spokes_per_frame}spf_*frames{ext}",
            )
            alt_files = glob.glob(alt_pattern)
            if alt_files:
                frames_found.update(
                    _available_frames(alt_files, token=token, suffix=f"frames{ext}")
                )
        if frames_found:
            frames_str = ", ".join(str(f) for f in sorted(frames_found))
            raise FileNotFoundError(
                f"Missing kspace file for {sample_id} with spf={self.spokes_per_frame} "
                f"and {self.num_frames} frames. Available frames: {frames_str}."
            )
        raise FileNotFoundError(f"Missing kspace file: {candidates[0]}")

    def _resolve_grasp_recon_path(self, sample_id: str) -> str:
        if self.dro_csmaps_source == "espirit":
            esp_root = self.espirit_grasp_recons_dir or os.path.join(
                self.root_dir, "espirit_grasp_recons"
            )
            with_frames = os.path.join(
                esp_root,
                f"grasp_{sample_id}_{self.spokes_per_frame}spf_{self.num_frames}frames.npy",
            )
            no_frames = os.path.join(
                esp_root, f"grasp_{sample_id}_{self.spokes_per_frame}spf.npy"
            )
        else:
            with_frames = os.path.join(
                self.root_dir,
                f"{sample_id}_recon_{self.spokes_per_frame}spf_{self.num_frames}frames.mat",
            )
            no_frames = os.path.join(
                self.root_dir, f"{sample_id}_recon_{self.spokes_per_frame}spf.mat"
            )
        if self.spf_suffix_policy == "require_frames":
            candidates = [with_frames]
        elif self.spf_suffix_policy == "require_no_frames":
            candidates = [no_frames]
        else:
            candidates = [with_frames, no_frames]
        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[0]

    def _coerce_grasp_thw(self, grasp: np.ndarray, sample_id: str, grasp_path: str) -> np.ndarray:
        if grasp.ndim != 3:
            raise ValueError(f"Unexpected GRASP shape {grasp.shape} in {grasp_path}.")
        if grasp.shape[0] == self.num_frames:
            return grasp
        if grasp.shape[1] == self.num_frames:
            return np.transpose(grasp, (1, 0, 2))
        if grasp.shape[2] == self.num_frames:
            return np.transpose(grasp, (2, 0, 1))
        raise ValueError(
            f"GRASP frames ({grasp.shape}) do not match expected num_frames ({self.num_frames}) "
            f"for {sample_id}. Use --eval_frames to match the DRO files."
        )

    def _load_grasp_recon(self, sample_id: str) -> tuple[np.ndarray, str]:
        recon_path = self._resolve_grasp_recon_path(sample_id)
        if not os.path.exists(recon_path):
            if self.dro_csmaps_source == "espirit":
                raise FileNotFoundError(
                    f"Missing ESPIRiT GRASP recon file: {recon_path}. "
                    "Pass --dro_espirit_grasp_dir to override."
                )
            raise FileNotFoundError(f"Missing recon file: {recon_path}")

        if self.dro_csmaps_source == "espirit":
            grasp = np.load(recon_path)
            grasp = np.squeeze(grasp)
            if np.iscomplexobj(grasp):
                grasp = grasp.astype(np.complex64, copy=False)
            else:
                grasp = grasp.astype(np.float32, copy=False)
        else:
            with h5py.File(recon_path, "r") as f:
                if "grasp_bart" not in f:
                    raise KeyError(f"{recon_path} missing required key 'grasp_bart'.")
                grasp = _read_h5_float(f["grasp_bart"]).astype(np.float32)

        grasp = self._coerce_grasp_thw(grasp, sample_id, recon_path)
        return grasp, recon_path

    def _load_espirit_csmaps(self, sample_id: str, expected_coils: int) -> np.ndarray:
        esp_root = self.espirit_csmaps_dir or os.path.join(self.root_dir, "csmaps_espirit")
        esp_path = os.path.join(esp_root, f"csmaps_{sample_id}.npy")
        if not os.path.exists(esp_path):
            raise FileNotFoundError(
                f"Missing ESPIRiT csmaps file: {esp_path}. "
                "Pass --dro_espirit_csmaps_dir to override."
            )
        esp_maps = np.load(esp_path)
        if esp_maps.ndim != 3:
            raise ValueError(f"Unexpected ESPIRiT csmaps shape {esp_maps.shape} in {esp_path}.")
        if esp_maps.shape[0] != expected_coils and esp_maps.shape[-1] == expected_coils:
            esp_maps = np.transpose(esp_maps, (2, 0, 1))
        if esp_maps.shape[0] != expected_coils:
            raise ValueError(
                f"ESPIRiT csmaps coil dimension {esp_maps.shape[0]} != expected {expected_coils} "
                f"for {sample_id}."
            )
        return esp_maps.astype(np.complex64, copy=False)

    def get_fastmri_id(self, sample_id: str) -> int:
        dro_id = int(sample_id.split("_")[1])
        if dro_id not in self.dro_to_fastmri:
            raise KeyError(f"DRO id {dro_id} not found in DRO-to-fastMRI mapping.")
        return self.dro_to_fastmri[dro_id]

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]

        sim_img, smap, mask = self._load_dro_mat(sample_id)
        if self.dro_csmaps_source == "espirit":
            smap = self._load_espirit_csmaps(sample_id, expected_coils=smap.shape[0])
        kspace = self._load_kspace_mat(sample_id)
        grasp, grasp_path = self._load_grasp_recon(sample_id)

        # Ground truth: (T, H, W) -> (2, T, H, W)
        gt_torch = torch.from_numpy(sim_img)
        gt_torch = torch.stack([gt_torch, torch.zeros_like(gt_torch)], dim=0)

        # GRASP recon (T, H, W) -> (2, H, T, W) with real/imag channels.
        grasp_torch = torch.from_numpy(grasp)
        grasp_torch = grasp_torch.permute(1, 0, 2)  # (H, T, W)
        if torch.is_complex(grasp_torch):
            grasp_torch = torch.stack([grasp_torch.real, grasp_torch.imag], dim=0)
        else:
            grasp_torch = torch.stack([grasp_torch, torch.zeros_like(grasp_torch)], dim=0)

        # CSMaps: (C, H, W)
        csmaps_torch = torch.from_numpy(smap).to(torch.complex64)

        # k-space: (C, spf*sam, T)
        kspace_torch = torch.from_numpy(kspace).to(torch.complex64)

        # load raw k-space and GRASP recon
        fastmri_id = self.get_fastmri_id(sample_id)
        patient_id = f"fastMRI_breast_{fastmri_id:03d}_2"
        slice_idx = self.slice_map.get(patient_id, None)
        raw_slice_valid = slice_idx is not None and slice_idx >= 0
        if (not raw_slice_valid) and self.skip_raw_eval_if_invalid_slice:
            raw_grasp_recon = torch.full_like(grasp_torch, float("nan"))
            raw_kspace_slice = torch.full_like(kspace_torch, float("nan"))
            raw_csmaps_torch = torch.full_like(csmaps_torch, float("nan")).numpy()
        else:
            if slice_idx is None or slice_idx < 0:
                slice_idx = self.grasp_slice_idx

            raw_grasp_path = os.path.join(
                os.path.dirname(self.raw_kspace_path),
                f"{patient_id}/grasp_recon_{self.spokes_per_frame}spf_{self.raw_num_frames}frames_slice{slice_idx}.npy",
            )
            raw_kspace_path = os.path.join(self.raw_kspace_path, f"{patient_id}.h5")
            raw_csmap_path = os.path.join(
                os.path.dirname(self.raw_kspace_path),
                f"cs_maps/{patient_id}_cs_maps/cs_map_slice_{slice_idx:03d}.npy",
            )

            try:
                raw_csmaps = np.load(raw_csmap_path)
                if self.skip_raw_grasp_metrics:
                    raw_grasp_recon = torch.full(
                        (2, self.raw_num_frames, raw_csmaps.shape[-2], raw_csmaps.shape[-1]),
                        float("nan"),
                        dtype=torch.float32,
                    )
                else:
                    raw_grasp_recon = np.load(raw_grasp_path).squeeze()

                    # GRASP Recon: (H, W, T) -> (2, T, H, W)
                    raw_grasp_recon = torch.from_numpy(raw_grasp_recon).permute(2, 0, 1)
                    raw_grasp_recon = torch.stack([raw_grasp_recon.real, raw_grasp_recon.imag], dim=0)

                    raw_grasp_recon = torch.flip(raw_grasp_recon, dims=[-3])
                    raw_grasp_recon = torch.rot90(raw_grasp_recon, k=1, dims=[-3, -1])

                with h5py.File(raw_kspace_path, "r") as f:
                    raw_kspace_slice = torch.tensor(f[self.dataset_key][slice_idx])
            except OSError as exc:
                print(
                    f"[Raw] Skipping corrupted raw file for {patient_id}: {raw_kspace_path} "
                    f"(error: {exc})"
                )
                raw_grasp_recon = torch.full_like(grasp_torch, float("nan"))
                raw_kspace_slice = torch.full_like(kspace_torch, float("nan"))
                raw_csmaps_torch = torch.full_like(csmaps_torch, float("nan")).numpy()
                return (
                    kspace_torch,
                    csmaps_torch,
                    gt_torch,
                    grasp_torch,
                    mask,
                    grasp_path,
                    raw_kspace_slice,
                    raw_grasp_recon,
                    raw_csmaps_torch,
                )

            # time-bin k-space
            N_spokes_prep = self.raw_num_frames * self.spokes_per_frame

            ksp_redu = raw_kspace_slice[:, :N_spokes_prep, :]
            ksp_prep = np.swapaxes(ksp_redu, 0, 1)
            ksp_prep_shape = ksp_prep.shape
            ksp_prep = np.reshape(
                ksp_prep,
                [self.raw_num_frames, self.spokes_per_frame] + list(ksp_prep_shape[1:]),
            )

            ksp_prep = torch.flip(ksp_prep, dims=[-1])

            raw_kspace_slice = (
                rearrange(ksp_prep, "t sp c sam -> c (sp sam) t")
                .to(kspace_torch.dtype)
            )

            raw_csmaps_torch = torch.from_numpy(raw_csmaps)
            raw_csmaps_torch = rearrange(raw_csmaps_torch, "c b h w -> b c h w").to(csmaps_torch.dtype)
            raw_csmaps_torch = torch.rot90(raw_csmaps_torch, k=2, dims=[-2, -1])
            raw_csmaps_torch = raw_csmaps_torch.numpy()

        return (
            kspace_torch,
            csmaps_torch,
            gt_torch,
            grasp_torch,
            mask,
            grasp_path,
            raw_kspace_slice,
            raw_grasp_recon,
            raw_csmaps_torch,
        )


def _torch_load_checkpoint(path: str, map_location="cpu"):
    """Load a checkpoint in the safest available way across torch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # Older torch versions don't support weights_only.
        return torch.load(path, map_location=map_location)
    except Exception:
        # Some checkpoints include objects disallowed in weights_only mode
        # on torch>=2.6. Fall back to full trusted checkpoint load.
        return torch.load(path, map_location=map_location, weights_only=False)


def _resolve_eval_params(
    config: dict,
    spokes: int | None,
    frames: int | None,
    phase_idx: int | None,
) -> Tuple[int, int]:
    """Pick evaluation spokes/frame and num_frames using overrides or curriculum."""
    curriculum_cfg = config.get("training", {}).get("curriculum_learning", {})
    phases = curriculum_cfg.get("phases", [])
    if curriculum_cfg.get("enabled") and phases:
        # Default to the last phase unless the user specifies otherwise.
        phase_idx = len(phases) - 1 if phase_idx is None else phase_idx
        phase_idx = max(0, min(phase_idx, len(phases) - 1))
        phase = phases[phase_idx]
        base_spokes, base_frames = phase["eval_spokes_per_frame"], phase["eval_num_frames"]
    else:
        data_cfg = config["data"]
        base_spokes, base_frames = data_cfg["eval_spokes"], data_cfg["eval_timeframes"]

    if spokes is not None:
        base_spokes = spokes
    if frames is not None:
        base_frames = frames
    return int(base_spokes), int(base_frames)


def _build_model(config: dict, device, block_dir: str):
    """Create model from config."""
    model = build_recon_model(config, device=device, block_dir=block_dir)
    model.eval()
    return model


def _mean_std(values, key):
    vals = [v.get(key) for v in values if v.get(key) is not None and np.isfinite(v.get(key))]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def _compute_summaries(results, grasp_results, raw_results, zf_results):
    dl_summary = {
        "ssim": _mean_std(results, "ssim"),
        "psnr": _mean_std(results, "psnr"),
        "psnr_fg": _mean_std(results, "dl_psnr_fg"),
        "mse": _mean_std(results, "mse"),
        "mse_fg": _mean_std(results, "dl_mse_fg"),
        "lpips": _mean_std(results, "lpips"),
        "dc_mse": _mean_std(results, "dc_mse"),
        "dc_mae": _mean_std(results, "dc_mae"),
        "dc_mse_bestfit": _mean_std(results, "dl_dc_mse_bestfit"),
        "dc_mae_bestfit": _mean_std(results, "dl_dc_mae_bestfit"),
        "dc_scale_abs": _mean_std(results, "dl_dc_scale_abs"),
        "img_scale": _mean_std(results, "dl_img_scale"),
        "recon_corr": _mean_std(results, "recon_corr"),
        "grasp_corr": _mean_std(results, "grasp_corr"),
        "fg_fraction": _mean_std(results, "fg_fraction"),
    }

    grasp_summary = {
        "ssim": _mean_std(grasp_results, "ssim"),
        "psnr": _mean_std(grasp_results, "psnr"),
        "psnr_fg": _mean_std(results, "grasp_psnr_fg"),
        "mse": _mean_std(grasp_results, "mse"),
        "mse_fg": _mean_std(results, "grasp_mse_fg"),
        "lpips": _mean_std(grasp_results, "lpips"),
        "dc_mse": _mean_std(grasp_results, "dc_mse"),
        "dc_mae": _mean_std(grasp_results, "dc_mae"),
        "dc_mse_bestfit": _mean_std(grasp_results, "grasp_dc_mse_bestfit"),
        "dc_mae_bestfit": _mean_std(grasp_results, "grasp_dc_mae_bestfit"),
        "dc_scale_abs": _mean_std(grasp_results, "grasp_dc_scale_abs"),
        "img_scale": _mean_std(results, "grasp_img_scale"),
    }

    raw_summary = {
        "raw_dc_mse": _mean_std(raw_results, "raw_dc_mse"),
        "raw_dc_mae": _mean_std(raw_results, "raw_dc_mae"),
        "raw_dc_psnr": _mean_std(raw_results, "raw_dc_psnr"),
        "raw_grasp_dc_mse": _mean_std(raw_results, "raw_grasp_dc_mse"),
        "raw_grasp_dc_mae": _mean_std(raw_results, "raw_grasp_dc_mae"),
        "raw_grasp_dc_psnr": _mean_std(raw_results, "raw_grasp_dc_psnr"),
        "raw_ssdu_nmse": _mean_std(raw_results, "raw_ssdu_nmse"),
        "raw_grasp_ssdu_nmse": _mean_std(raw_results, "raw_grasp_ssdu_nmse"),
    }

    zf_summary = None
    if zf_results:
        zf_summary = {
            "ssim": _mean_std(zf_results, "ssim"),
            "psnr": _mean_std(zf_results, "psnr"),
            "psnr_fg": _mean_std(zf_results, "zf_psnr_fg"),
            "mse": _mean_std(zf_results, "mse"),
            "mse_fg": _mean_std(zf_results, "zf_mse_fg"),
            "lpips": _mean_std(zf_results, "lpips"),
            "dc_mse": _mean_std(zf_results, "dc_mse"),
            "dc_mae": _mean_std(zf_results, "dc_mae"),
            "dc_mse_bestfit": _mean_std(zf_results, "zf_dc_mse_bestfit"),
            "dc_mae_bestfit": _mean_std(zf_results, "zf_dc_mae_bestfit"),
            "dc_scale_abs": _mean_std(zf_results, "zf_dc_scale_abs"),
            "img_scale": _mean_std(zf_results, "zf_img_scale"),
        }

    return dl_summary, grasp_summary, raw_summary, zf_summary


def _write_metrics_csv(
    metrics_path: str,
    results,
    grasp_results,
    raw_results,
    zf_results,
):
    with open(metrics_path, "w") as f:
        headers = [
            "sample",
            "dro_csmap_scale",
            "dl_ssim",
            "dl_psnr",
            "dl_psnr_fg",
            "dl_mse",
            "dl_mse_fg",
            "dl_lpips",
            "dl_dc_mse",
            "dl_dc_mae",
            "dl_dc_mse_bestfit",
            "dl_dc_mae_bestfit",
            "dl_dc_scale_abs",
            "dl_dc_scale_phase",
            "dl_img_scale",
            "dl_recon_corr",
            "grasp_corr",
            "grasp_ssim",
            "grasp_psnr",
            "grasp_psnr_fg",
            "grasp_mse",
            "grasp_mse_fg",
            "grasp_lpips",
            "grasp_dc_mse",
            "grasp_dc_mae",
            "grasp_dc_mse_bestfit",
            "grasp_dc_mae_bestfit",
            "grasp_dc_scale_abs",
            "grasp_dc_scale_phase",
            "grasp_img_scale",
            "zf_ssim",
            "zf_psnr",
            "zf_psnr_fg",
            "zf_mse",
            "zf_mse_fg",
            "zf_lpips",
            "zf_dc_mse",
            "zf_dc_mae",
            "zf_dc_mse_bestfit",
            "zf_dc_mae_bestfit",
            "zf_dc_scale_abs",
            "zf_dc_scale_phase",
            "zf_img_scale",
            "fg_fraction",
            "raw_dc_mse",
            "raw_dc_mae",
            "raw_dc_psnr",
            "raw_grasp_dc_mse",
            "raw_grasp_dc_mae",
            "raw_grasp_dc_psnr",
            "raw_ssdu_nmse",
            "raw_grasp_ssdu_nmse",
        ]
        f.write(",".join(headers) + "\n")
        zf_lookup = {row["sample"]: row for row in zf_results}
        for dro_row, grasp_row, raw_row in zip(results, grasp_results, raw_results):
            zf_row = zf_lookup.get(dro_row["sample"], {})
            dl_psnr_fg = dro_row.get("dl_psnr_fg")
            dl_mse_fg = dro_row.get("dl_mse_fg")
            grasp_psnr_fg = dro_row.get("grasp_psnr_fg")
            grasp_mse_fg = dro_row.get("grasp_mse_fg")
            fg_fraction = dro_row.get("fg_fraction")
            zf_psnr_fg = zf_row.get("zf_psnr_fg")
            zf_mse_fg = zf_row.get("zf_mse_fg")
            row = [
                dro_row["sample"],
                "" if dro_row.get("dro_csmap_scale") is None else f"{dro_row['dro_csmap_scale']:.6f}",
                f"{dro_row['ssim']:.6f}",
                f"{dro_row['psnr']:.6f}",
                "" if dl_psnr_fg is None else f"{dl_psnr_fg:.6f}",
                f"{dro_row['mse']:.6f}",
                "" if dl_mse_fg is None else f"{dl_mse_fg:.6f}",
                f"{dro_row['lpips']:.6f}",
                f"{dro_row['dc_mse']:.6f}",
                f"{dro_row['dc_mae']:.6f}",
                "" if dro_row.get("dl_dc_mse_bestfit") is None else f"{dro_row['dl_dc_mse_bestfit']:.6f}",
                "" if dro_row.get("dl_dc_mae_bestfit") is None else f"{dro_row['dl_dc_mae_bestfit']:.6f}",
                "" if dro_row.get("dl_dc_scale_abs") is None else f"{dro_row['dl_dc_scale_abs']:.6f}",
                "" if dro_row.get("dl_dc_scale_phase") is None else f"{dro_row['dl_dc_scale_phase']:.6f}",
                "" if dro_row.get("dl_img_scale") is None else f"{dro_row['dl_img_scale']:.6f}",
                "" if dro_row["recon_corr"] is None else f"{dro_row['recon_corr']:.6f}",
                "" if dro_row["grasp_corr"] is None else f"{dro_row['grasp_corr']:.6f}",
                f"{grasp_row['ssim']:.6f}",
                f"{grasp_row['psnr']:.6f}",
                "" if grasp_psnr_fg is None else f"{grasp_psnr_fg:.6f}",
                f"{grasp_row['mse']:.6f}",
                "" if grasp_mse_fg is None else f"{grasp_mse_fg:.6f}",
                f"{grasp_row['lpips']:.6f}",
                f"{grasp_row['dc_mse']:.6f}",
                f"{grasp_row['dc_mae']:.6f}",
                "" if grasp_row.get("grasp_dc_mse_bestfit") is None else f"{grasp_row['grasp_dc_mse_bestfit']:.6f}",
                "" if grasp_row.get("grasp_dc_mae_bestfit") is None else f"{grasp_row['grasp_dc_mae_bestfit']:.6f}",
                "" if grasp_row.get("grasp_dc_scale_abs") is None else f"{grasp_row['grasp_dc_scale_abs']:.6f}",
                "" if grasp_row.get("grasp_dc_scale_phase") is None else f"{grasp_row['grasp_dc_scale_phase']:.6f}",
                "" if dro_row.get("grasp_img_scale") is None else f"{dro_row['grasp_img_scale']:.6f}",
                "" if not zf_row else f"{zf_row.get('ssim', float('nan')):.6f}",
                "" if not zf_row else f"{zf_row.get('psnr', float('nan')):.6f}",
                "" if zf_psnr_fg is None else f"{zf_psnr_fg:.6f}",
                "" if not zf_row else f"{zf_row.get('mse', float('nan')):.6f}",
                "" if zf_mse_fg is None else f"{zf_mse_fg:.6f}",
                "" if not zf_row else f"{zf_row.get('lpips', float('nan')):.6f}",
                "" if not zf_row else f"{zf_row.get('dc_mse', float('nan')):.6f}",
                "" if not zf_row else f"{zf_row.get('dc_mae', float('nan')):.6f}",
                "" if zf_row.get("zf_dc_mse_bestfit") is None else f"{zf_row['zf_dc_mse_bestfit']:.6f}",
                "" if zf_row.get("zf_dc_mae_bestfit") is None else f"{zf_row['zf_dc_mae_bestfit']:.6f}",
                "" if zf_row.get("zf_dc_scale_abs") is None else f"{zf_row['zf_dc_scale_abs']:.6f}",
                "" if zf_row.get("zf_dc_scale_phase") is None else f"{zf_row['zf_dc_scale_phase']:.6f}",
                "" if zf_row.get("zf_img_scale") is None else f"{zf_row['zf_img_scale']:.6f}",
                "" if fg_fraction is None else f"{fg_fraction:.6f}",
                f"{raw_row['raw_dc_mse']:.6f}",
                f"{raw_row['raw_dc_mae']:.6f}",
                "" if raw_row.get("raw_dc_psnr") is None else f"{raw_row['raw_dc_psnr']:.6f}",
                "" if raw_row.get("raw_grasp_dc_mse") is None else f"{raw_row['raw_grasp_dc_mse']:.6f}",
                "" if raw_row.get("raw_grasp_dc_mae") is None else f"{raw_row['raw_grasp_dc_mae']:.6f}",
                "" if raw_row.get("raw_grasp_dc_psnr") is None else f"{raw_row['raw_grasp_dc_psnr']:.6f}",
                "" if raw_row.get("raw_ssdu_nmse") is None else f"{raw_row['raw_ssdu_nmse']:.6f}",
                "" if raw_row.get("raw_grasp_ssdu_nmse") is None else f"{raw_row['raw_grasp_ssdu_nmse']:.6f}",
            ]
            f.write(",".join(row) + "\n")


def _write_temporal_metrics_csv(
    inference_dir: str,
    results,
    metric_names: list[str],
    suffix: str,
):
    for label, prefix in (("malignant", ""), ("benign", "benign_")):
        for subset in ("all", "top10", "top20"):
            keys = [
                f"{prefix}{model}_{subset}_{metric}"
                for model in ("dl", "grasp")
                for metric in metric_names
            ]
            temporal_metrics_path = os.path.join(
                inference_dir, f"metrics_temporal_{label}_{subset}{suffix}.csv"
            )
            with open(temporal_metrics_path, "w") as f:
                f.write(",".join(["sample"] + keys) + "\n")
                for dro_row in results:
                    row = [dro_row["sample"]]
                    for key in keys:
                        value = dro_row.get(key, np.nan)
                        if value is None or (isinstance(value, float) and np.isnan(value)):
                            row.append("")
                        else:
                            row.append(f"{value:.6f}")
                    f.write(",".join(row) + "\n")


def _load_weights(model, ckpt_path: str):
    ckpt = _torch_load_checkpoint(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(remove_module_prefix(state_dict))
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on DRO samples.")
    parser.add_argument("--exp_dir", required=True, help="Experiment directory location.")
    parser.add_argument(
        "--output_root",
        help=(
            "Override config experiment.output_dir when saving inference outputs "
            "(useful when exp_dir is read-only)."
        ),
    )
    parser.add_argument("--config", help="Path to config.yaml (defaults to output/<exp>/config.yaml).")
    parser.add_argument("--checkpoint", help="Path to model checkpoint (defaults to output/<exp>/<exp>_model.pth).")
    ckpt_group = parser.add_mutually_exclusive_group()
    ckpt_group.add_argument(
        "--use_best_checkpoint",
        dest="use_best_checkpoint",
        action="store_true",
        default=False,
        help="Use <exp>_best_model.pth if available.",
    )
    ckpt_group.add_argument(
        "--use_last_checkpoint",
        dest="use_best_checkpoint",
        action="store_false",
        help="Use <exp>_model.pth (last checkpoint, default).",
    )
    parser.add_argument(
        "--store_logs",
        dest="store_logs",
        action="store_true",
        default=True,
        help="Write/overwrite val_inference_logs for this experiment (default: true).",
    )
    parser.add_argument(
        "--no_store_logs",
        dest="store_logs",
        action="store_false",
        help="Disable writing val_inference_logs for this run.",
    )
    parser.add_argument(
        "--overwrite_logs",
        action="store_true",
        help=(
            "Overwrite matching entries in val_inference_logs.json "
            "(by exp_name for BRISKNet and by eval params for GRASP)."
        ),
    )
    parser.add_argument(
        "--log_file",
        default=str(DEFAULT_LOG_PATH),
        help="Path to val_inference_logs.json (default: inference/val_inference_logs.json).",
    )
    parser.add_argument("--num_samples", type=int, help="Number of validation samples to evaluate (default: config value).")
    parser.add_argument(
        "--split_key",
        default=None,
        help=(
            "Split key to use from data_split.json (e.g., val_dro, test_dro). "
            "If not set, uses val_dro then falls back to val."
        ),
    )
    parser.add_argument("--device", default=None, help="Torch device to use (default: config training.device).")
    parser.add_argument("--eval_spokes", type=int, help="Override spokes per frame for inference.")
    parser.add_argument("--eval_frames", type=int, help="Override number of frames for inference.")
    parser.add_argument("--phase_index", type=int, help="Curriculum phase index to use for eval params (default: last).")
    parser.add_argument("--disable_ssdu", action="store_true", help="Skip SSDU NMSE computation to speed up inference.")
    parser.add_argument(
        "--skip_raw_grasp_metrics",
        action="store_true",
        help="Skip raw GRASP image metrics/diagnostics (avoids loading raw GRASP recon files). SSDU still runs.",
    )
    parser.add_argument(
        "--dro_csmaps_source",
        default="espirit",
        choices=("original", "espirit"),
        help="DRO csmaps source ('original' or 'espirit'). Default: espirit.",
    )
    parser.add_argument(
        "--dro_sim_source",
        default="espirit",
        choices=("original", "espirit"),
        help=(
            "Use DRO simulated k-space/GRASP files from the sample directory (original) or "
            "ESPIRiT variants (espirit, suffix _espirit)."
        ),
    )
    parser.add_argument(
        "--traj_method",
        default=None,
        choices=("trajGR", "get_traj"),
        help="Override trajectory source ('trajGR' or 'get_traj'). Defaults to config data.traj_method.",
    )
    parser.add_argument(
        "--dro_espirit_csmaps_dir",
        default=None,
        help="Override ESPIRiT csmaps dir (default: <dro_root>/csmaps_espirit).",
    )
    parser.add_argument(
        "--dro_espirit_grasp_dir",
        default=None,
        help=(
            "Override ESPIRiT GRASP recon dir "
            "(default: <dro_root>/espirit_grasp_recons_lam{grasp_lamda})."
        ),
    )
    parser.add_argument(
        "--grasp_lamda",
        type=float,
        default=0.001,
        help=(
            "GRASP TV weight used to select the ESPIRiT GRASP recon directory "
            "(espirit_grasp_recons_lam{lamda}) when --dro_espirit_grasp_dir is not set."
        ),
    )
    parser.add_argument(
        "--grasp_lamdas",
        default=None,
        help="Comma-separated list of GRASP lambda values to evaluate (overrides --grasp_lamda).",
    )
    parser.add_argument(
        "--dro_noise_level",
        type=float,
        default=0.05,
        help="DRO noise level used for simulated k-space/GRASP filenames (e.g., 0 or 0.05). Default: 0.05.",
    )
    parser.add_argument(
        "--new_dro_root",
        default="/net/scratch2/rachelgordon/dro_var_frames",
        help="Root directory containing the new DRO .mat files.",
    )
    parser.add_argument(
        "--normalize_dro_csmaps",
        action="store_true",
        help=(
            "Normalize DRO coil sensitivity maps by a per-sample scalar (median RSS in tissue mask) "
            "and apply the same scaling to DRO k-space. Useful for debugging csmap OOD scaling."
        ),
    )
    parser.add_argument(
        "--plot_ei_label",
        action="store_true",
        help=(
            "Update BRISKNet spatial quality plot labels to reflect MC vs MC+EI "
            "based on model.losses.use_ei_loss in the config."
        ),
    )
    parser.add_argument("--diagnostics", action="store_true", help="Enable diagnostic plots per sample.")
    parser.add_argument(
        "--compute_zf_baseline",
        action="store_true",
        help="Compute and report adjoint (ZF) baseline image/k-space metrics.",
    )
    parser.add_argument(
        "--save_debug_arrays",
        action="store_true",
        help="Save per-sample magnitude arrays (gt/dl/grasp/zf) to NPZ for debugging.",
    )
    parser.add_argument("--diag_topk", type=int, default=16, help="Top-K pixels to plot in diagnostic curves.")
    parser.add_argument("--diag_num_frames", type=int, default=6, help="Number of frames for heatmap diagnostics.")
    parser.add_argument(
        "--diag_ref",
        default="gt",
        choices=("gt", "grasp", "raw_grasp", "raw_recon"),
        help="Reference image source for diagnostics.",
    )
    parser.add_argument(
        "--diag_normalize",
        action="store_true",
        help="Normalize pixel curves by baseline (t=0) reference intensity.",
    )
    parser.add_argument(
        "--baseline_mode",
        default="seconds",
        choices=("seconds", "fraction"),
        help="Baseline window selection mode for temporal metrics/plots.",
    )
    parser.add_argument(
        "--baseline_seconds",
        type=float,
        default=20.0,
        help="Baseline duration in seconds when baseline_mode=seconds.",
    )
    parser.add_argument(
        "--baseline_fraction",
        type=float,
        default=0.1,
        help="Baseline fraction of frames when baseline_mode=fraction.",
    )
    parser.add_argument(
        "--baseline_min_frames",
        type=int,
        default=1,
        help="Minimum baseline frames to use.",
    )
    parser.add_argument(
        "--baseline_max_frames",
        type=int,
        default=10,
        help="Maximum baseline frames when baseline_mode=fraction.",
    )
    parser.add_argument(
        "--total_scan_seconds",
        type=float,
        default=150.0,
        help="Total scan duration in seconds for temporal plots/metrics.",
    )
    parser.add_argument(
        "--arrival_k",
        type=float,
        default=None,
        help="Arrival threshold factor k for mu + k*sigma.",
    )
    parser.add_argument(
        "--arrival_method",
        default=None,
        help="Arrival method: threshold or fraction_of_peak (uses arrival_fraction).",
    )
    parser.add_argument(
        "--arrival_fraction",
        type=float,
        default=None,
        help="Fraction-of-peak arrival threshold (0..1).",
    )
    parser.add_argument(
        "--early_seconds",
        type=float,
        default=35.0,
        help="Early enhancement window length in seconds after arrival.",
    )
    parser.add_argument(
        "--early_min_frames",
        type=int,
        default=4,
        help="Minimum early enhancement window frames.",
    )
    parser.add_argument(
        "--early_max_frames",
        type=int,
        default=8,
        help="Maximum early enhancement window frames.",
    )
    parser.add_argument("--seed", type=int, default=12, help="Random seed.")
    return parser.parse_args()


def _write_inference_metadata(
    inference_dir: str,
    args,
    config: dict,
    config_path: str,
    ckpt_path: str,
    device,
    eval_spokes: int,
    eval_frames: int,
    inference_settings: dict | None = None,
    resolved_args: dict | None = None,
):
    metadata = {
        "argv": sys.argv,
        "command": " ".join([shlex.quote(sys.executable)] + [shlex.quote(arg) for arg in sys.argv]),
        "args": resolved_args or vars(args),
        "config_path": config_path,
        "checkpoint_path": ckpt_path,
        "resolved_device": str(device),
        "resolved_eval_spokes": int(eval_spokes),
        "resolved_eval_frames": int(eval_frames),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if inference_settings:
        metadata["inference_settings"] = inference_settings
    with open(os.path.join(inference_dir, "run_args.json"), "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    with open(os.path.join(inference_dir, "config_resolved.yaml"), "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    shutil.copy2(config_path, os.path.join(inference_dir, "config_source.yaml"))


def _to_numpy_img(x: torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if torch.is_complex(x):
            x = torch.abs(x)
        x = x.float().numpy()
    else:
        x = np.asarray(x)
        if np.iscomplexobj(x):
            x = np.abs(x)
        x = x.astype(np.float32, copy=False)
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == 2:
        x = x[None, ...]
    return x.astype(np.float32, copy=False)


def _normalize_mask(mask, T: int, H: int, W: int):
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = np.asarray(mask)
    if mask_np.ndim == 3 and mask_np.shape[0] == 1:
        mask_np = mask_np[0]
    if mask_np.ndim == 2:
        mask_stack = np.broadcast_to(mask_np, (T, H, W))
    elif mask_np.ndim == 3 and mask_np.shape[0] == T:
        mask_stack = mask_np
    else:
        return None
    return mask_stack > 0


def _parse_float_list(value: str) -> list[float]:
    if not value:
        return []
    parts = [v.strip() for v in value.split(",") if v.strip()]
    return [float(v) for v in parts]


def _coerce_grasp_thw(grasp: np.ndarray, num_frames: int, sample_id: str, grasp_path: str) -> np.ndarray:
    if grasp.ndim != 3:
        raise ValueError(f"Unexpected GRASP shape {grasp.shape} in {grasp_path}.")
    if grasp.shape[0] == num_frames:
        return grasp
    if grasp.shape[1] == num_frames:
        return np.transpose(grasp, (1, 0, 2))
    if grasp.shape[2] == num_frames:
        return np.transpose(grasp, (2, 0, 1))
    raise ValueError(
        f"GRASP frames ({grasp.shape}) do not match expected num_frames ({num_frames}) "
        f"for {sample_id}. Use --eval_frames to match the DRO files."
    )


def _load_grasp_from_dir(
    sample_id: str,
    spf: int,
    num_frames: int,
    grasp_dir: str,
    spf_suffix_policy: str = "flex",
) -> tuple[np.ndarray, str]:
    with_frames = os.path.join(grasp_dir, f"grasp_{sample_id}_{spf}spf_{num_frames}frames.npy")
    no_frames = os.path.join(grasp_dir, f"grasp_{sample_id}_{spf}spf.npy")
    if spf_suffix_policy == "require_frames":
        candidates = [with_frames]
    elif spf_suffix_policy == "require_no_frames":
        candidates = [no_frames]
    else:
        candidates = [with_frames, no_frames]
    recon_path = None
    for path in candidates:
        if os.path.exists(path):
            recon_path = path
            break
    if recon_path is None:
        raise FileNotFoundError(
            f"Missing ESPIRiT GRASP recon file in {grasp_dir} for {sample_id} "
            f"(spf={spf}, frames={num_frames})."
        )
    grasp = np.load(recon_path).squeeze()
    if np.iscomplexobj(grasp):
        grasp = grasp.astype(np.complex64, copy=False)
    else:
        grasp = grasp.astype(np.float32, copy=False)
    grasp = _coerce_grasp_thw(grasp, num_frames, sample_id, recon_path)
    return grasp, recon_path


def _grasp_np_to_torch(grasp: np.ndarray) -> torch.Tensor:
    grasp_torch = torch.from_numpy(grasp)
    grasp_torch = grasp_torch.permute(1, 0, 2)  # (H, T, W)
    if torch.is_complex(grasp_torch):
        grasp_torch = torch.stack([grasp_torch.real, grasp_torch.imag], dim=0)
    else:
        grasp_torch = torch.stack([grasp_torch, torch.zeros_like(grasp_torch)], dim=0)
    return grasp_torch


def _compute_dro_csmap_scale(csmap: torch.Tensor, mask) -> float | None:
    """Compute a per-sample scalar to normalize DRO sensitivity-map scale.

    Uses the median RSS of the coil maps inside the union of DRO tissue masks when available;
    falls back to nonzero RSS pixels otherwise.
    """
    smaps = csmap
    if smaps.ndim == 5 and smaps.shape[0] == 1:
        smaps = smaps.squeeze(0)
    if smaps.ndim == 4 and smaps.shape[0] == 1:
        smaps = smaps.squeeze(0)
    if smaps.ndim != 3:
        return None

    rss = torch.sqrt(torch.sum(torch.abs(smaps) ** 2, dim=0))  # (H,W)

    union_np = None
    if isinstance(mask, dict) and mask:
        for v in mask.values():
            if v is None:
                continue
            if not isinstance(v, torch.Tensor):
                continue
            v_np = v.detach().cpu().numpy().squeeze().astype(bool)
            if v_np.ndim != 2:
                continue
            union_np = v_np if union_np is None else (union_np | v_np)

    if union_np is not None and union_np.any() and union_np.mean() < 0.999:
        union_t = torch.from_numpy(union_np).to(rss.device)
        vals = rss[union_t]
    else:
        vals = rss[rss > 0]

    if vals.numel() == 0:
        return None
    scale = float(torch.median(vals).item())
    if not np.isfinite(scale) or scale <= 0:
        return None
    return scale


def _resample_temporal(tensor: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Nearest-neighbor resample along time dimension to match target frame count."""
    if tensor is None or target_frames is None:
        return tensor
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Expected tensor for temporal resampling.")
    if target_frames <= 0:
        raise ValueError(f"target_frames must be positive; got {target_frames}.")
    if tensor.ndim == 4:
        # (2, T, H, W)
        _, frames, _, _ = tensor.shape
        if frames == target_frames:
            return tensor
        idx = torch.linspace(0, frames - 1, steps=target_frames, device=tensor.device)
        idx = torch.round(idx).long()
        return tensor.index_select(1, idx)
    if tensor.ndim == 5:
        # (B, 2, T, H, W)
        _, _, frames, _, _ = tensor.shape
        if frames == target_frames:
            return tensor
        idx = torch.linspace(0, frames - 1, steps=target_frames, device=tensor.device)
        idx = torch.round(idx).long()
        return tensor.index_select(2, idx)
    return tensor


def _robust_limits(values: np.ndarray, low_q: float, high_q: float):
    finite_vals = values[np.isfinite(values)]
    if finite_vals.size == 0:
        return 0.0, 1.0
    return (float(np.percentile(finite_vals, low_q)), float(np.percentile(finite_vals, high_q)))


def _save_zf_plot(sample_dir: str, zf_complex: torch.Tensor, frame_idx: int | None = None):
    zf_mag = torch.abs(zf_complex).detach().cpu().numpy()
    if zf_mag.ndim == 2:
        frame = zf_mag
        frame_idx = 0
    elif zf_mag.ndim == 3:
        T = zf_mag.shape[2]
        if frame_idx is None:
            frame_idx = int(T // 2)
        frame_idx = max(0, min(int(frame_idx), int(T - 1)))
        frame = zf_mag[:, :, frame_idx]
    else:
        raise ValueError(f"Unexpected zf magnitude shape {zf_mag.shape}.")

    vmin, vmax = _robust_limits(frame, 1, 99.5)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(f"ZF frame {frame_idx}", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    out_path = os.path.join(sample_dir, f"zf_frame_{frame_idx:03d}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _select_timepoints(ref_roi_mean: np.ndarray, num_frames: int):
    T = ref_roi_mean.shape[0]
    if T == 0:
        return []
    num_frames = max(1, min(num_frames, T))
    evenly = np.linspace(0, T - 1, num_frames).round().astype(int).tolist()
    peak = int(np.nanargmax(ref_roi_mean)) if np.isfinite(ref_roi_mean).any() else 0
    selected = sorted(set(evenly + [peak]))
    return selected


def _overlay_plot(background, overlay, mask, vmin, vmax, cmap, alpha, outpath, title, cbar_label=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(background, cmap="gray")
    overlay_masked = np.ma.array(overlay, mask=~mask)
    im = ax.imshow(overlay_masked, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _contour_mask(ax, mask):
    ax.contour(mask.astype(float), levels=[0.5], colors="yellow", linewidths=0.8)


def _mask_bbox(mask: np.ndarray, pad: int, H: int, W: int):
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return (0, H, 0, W)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, H)
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, W)
    return (y0, y1, x0, x1)


def save_diagnostics(
    sample_dir: str,
    brisk,
    reference,
    mask,
    args,
    diag_subdir: str = "diagnostics",
    ref_label: str | None = None,
    brisk_label: str | None = None,
):
    diag_dir = os.path.join(sample_dir, diag_subdir)
    os.makedirs(diag_dir, exist_ok=True)

    brisk_np = _to_numpy_img(brisk).squeeze()
    brisk_np = np.abs(brisk_np[0] + 1j * brisk_np[1])
    brisk_np = rearrange(brisk_np, 'h w t -> t h w')

    ref_np = _to_numpy_img(reference).squeeze()
    ref_np = np.abs(ref_np[0] + 1j * ref_np[1])

    if ref_np.shape[0] == 320:
        ref_np = rearrange(ref_np, 'h t w -> t h w')

    if brisk_np.shape != ref_np.shape:
        raise ValueError(f"Diagnostics shape mismatch: brisk {brisk_np.shape} vs ref {ref_np.shape}")

    T, H, W = ref_np.shape
    mask_stack = _normalize_mask(mask, T, H, W)
    if mask_stack is None:
        raise ValueError(f"Diagnostics unsupported mask shape: {np.asarray(mask).shape}")

    mask_union = np.any(mask_stack, axis=0)
    if not np.any(mask_union):
        raise ValueError("Diagnostics tumor mask is empty.")

    ref_roi_mean = np.array(
        [
            np.nanmean(ref_np[t][mask_stack[t]])
            if np.any(mask_stack[t]) else np.nan
            for t in range(T)
        ],
        dtype=np.float32,
    )
    time_points = np.linspace(0, args.total_scan_seconds, T)
    n_baseline = _resolve_baseline_frames(
        num_frames=T,
        time_points=time_points,
        baseline_mode=args.baseline_mode,
        baseline_seconds=args.baseline_seconds,
        baseline_fraction=args.baseline_fraction,
        baseline_min_frames=args.baseline_min_frames,
        baseline_max_frames=args.baseline_max_frames,
    )
    baseline_idx = list(range(min(max(n_baseline, 1), T)))
    background = np.nanmean(ref_np[baseline_idx], axis=0)
    if not np.isfinite(background).any():
        fallback_idx = int(np.nanargmax(ref_roi_mean)) if np.isfinite(ref_roi_mean).any() else 0
        background = ref_np[fallback_idx]
    pad = max(4, int(0.05 * max(H, W)))
    y0, y1, x0, x1 = _mask_bbox(mask_union, pad, H, W)

    ref_vals = ref_np[mask_stack]
    vmin_ref, vmax_ref = _robust_limits(ref_vals, 1, 99)

    diff = brisk_np - ref_np
    diff_vals = diff[mask_stack]
    diff_abs = np.abs(diff_vals[np.isfinite(diff_vals)])
    diff_lim = float(np.percentile(diff_abs, 99)) if diff_abs.size else 1.0
    diff_lim = max(diff_lim, 1e-6)

    timepoints = _select_timepoints(ref_roi_mean, args.diag_num_frames)
    if ref_label is None:
        ref_label = {
            "gt": "Ground Truth",
            "grasp": "GRASP",
            "raw_grasp": "Raw GRASP",
            "raw_recon": "Raw Recon",
        }.get(args.diag_ref, "Reference")
    if brisk_label is None:
        brisk_label = "BRISKNet"
    if not timepoints:
        return
    nrows = len(timepoints)
    fig, axes = plt.subplots(nrows, 3, figsize=(12, 4 * nrows), squeeze=False)
    for row_idx, t in enumerate(timepoints):
        mask_t = mask_stack[t][y0:y1, x0:x1]
        bg_crop = background[y0:y1, x0:x1]
        ref_crop = ref_np[t][y0:y1, x0:x1]
        brisk_crop = brisk_np[t][y0:y1, x0:x1]
        diff_crop = diff[t][y0:y1, x0:x1]
        overlays = (ref_crop, brisk_crop, diff_crop)
        cmaps = ("magma", "magma", "coolwarm")
        vmins = (vmin_ref, vmin_ref, -diff_lim)
        vmaxs = (vmax_ref, vmax_ref, diff_lim)
        titles = (
            f"{ref_label} (t={t})",
            f"{brisk_label} (t={t})",
            f"Difference (t={t})",
        )
        cbar_labels = (
            f"{ref_label} intensity (a.u.)",
            f"{brisk_label} intensity (a.u.)",
            "Difference (a.u.)",
        )
        for col_idx in range(3):
            ax = axes[row_idx][col_idx]
            ax.imshow(bg_crop, cmap="gray")
            overlay_masked = np.ma.array(overlays[col_idx], mask=~mask_t)
            im = ax.imshow(overlay_masked, cmap=cmaps[col_idx], vmin=vmins[col_idx], vmax=vmaxs[col_idx], alpha=0.6)
            ax.set_title(titles[col_idx], fontsize=9)
            ax.axis("off")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(cbar_labels[col_idx], fontsize=8)
            cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(diag_dir, "heatmaps_triptych_grid.png"), dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)

    grasp_lamdas = (
        _parse_float_list(args.grasp_lamdas) if args.grasp_lamdas else [args.grasp_lamda]
    )
    if not grasp_lamdas:
        raise ValueError("No GRASP lambda values provided.")
    primary_grasp_lamda = grasp_lamdas[0]

    exp_name = args.exp_dir.split('/')[-1]

    # Resolve config/checkpoint paths and load config.
    config_path = args.config or os.path.join(args.exp_dir, "config.yaml")
    if args.checkpoint:
        ckpt_path = args.checkpoint
    elif args.use_best_checkpoint:
        best_ckpt_path = os.path.join(args.exp_dir, f"{exp_name}_best_model.pth")
        if os.path.exists(best_ckpt_path):
            ckpt_path = best_ckpt_path
        else:
            ckpt_path = os.path.join(args.exp_dir, f"{exp_name}_model.pth")
    else:
        ckpt_path = os.path.join(args.exp_dir, f"{exp_name}_model.pth")

    ckpt_meta = None
    try:
        ckpt_meta = _torch_load_checkpoint(ckpt_path, map_location="cpu")
    except Exception as exc:
        print(f"Warning: unable to load checkpoint metadata from {ckpt_path}: {exc}")
    trained_epochs = ckpt_meta.get("epoch") - 1 if isinstance(ckpt_meta, dict) else None

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = apply_cluster_paths(config)

    device = torch.device(args.device or config["training"]["device"])
    rescale = config.get("evaluation", {}).get("rescale", True)
    raw_grasp_slice_idx = config.get("evaluation", {}).get("raw_grasp_slice_idx", 95)
    cluster = config.get("experiment", {}).get("cluster", "Randi")

    # Where to save inference outputs.
    output_root = args.output_root or config["experiment"]["output_dir"]
    output_dir = os.path.join(output_root, exp_name)
    inference_dir = os.path.join(output_dir, f"inference_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(inference_dir, exist_ok=True)

    # Dataset setup.
    with open(config["data"]["split_file"], "r") as fp:
        splits = json.load(fp)

    if args.split_key:
        val_ids = splits.get(args.split_key) or []
        if not val_ids:
            raise KeyError(
                f"Split key '{args.split_key}' not found or empty in {config['data']['split_file']}."
            )
        print(f"[Split] Using '{args.split_key}' from data_split.json")
    else:
        val_ids = splits.get("val_dro") or splits.get("val") or []
        if not val_ids:
            raise KeyError(
                f"No 'val_dro' or 'val' split found in {config['data']['split_file']}."
            )

    _, N_time_eval_raw = _resolve_eval_params(
        config, spokes=None, frames=None, phase_idx=args.phase_index
    )
    N_spokes_eval, N_time_eval = _resolve_eval_params(
        config, spokes=args.eval_spokes, frames=args.eval_frames, phase_idx=args.phase_index
    )

    eval_chunk_size = config.get("evaluation", {}).get("chunk_size", N_time_eval)
    eval_chunk_overlap = config.get("evaluation", {}).get("chunk_overlap", 0)
    compute_ssdu = config.get("evaluation", {}).get("compute_ssdu", True)
    if args.disable_ssdu:
        compute_ssdu = False

    ssdu_k_folds = config.get("evaluation", {}).get("ssdu_k_folds", 4)
    ssdu_grasp_k_folds = config.get("evaluation", {}).get("ssdu_grasp_k_folds", ssdu_k_folds)
    ssdu_weighting = config.get("evaluation", {}).get("ssdu_weighting", "sqrt_dcomp")
    config_eval = config.get("evaluation", {})
    val_noise_level = (
        args.dro_noise_level
        if args.dro_noise_level is not None
        else config_eval.get("val_noise_level", 0.05)
    )
    dro_csmaps_source = (
        args.dro_csmaps_source
        if args.dro_csmaps_source is not None
        else config_eval.get("dro_csmaps_source", "espirit")
    )
    dro_sim_source = (
        args.dro_sim_source
        if args.dro_sim_source is not None
        else config_eval.get("dro_sim_source", "espirit")
    )
    if args.dro_noise_level is None and dro_sim_source == "espirit" and abs(float(val_noise_level) - 0.05) > 1e-8:
        # The current ESPIRiT DRO artifacts/GRASP recons in dro_dataset_frontpad are stored with the
        # `_correct_traj_n0.05_espirit` suffix. If an experiment config sets val_noise_level=0 (common
        # in older configs), we override to 0.05 to avoid FileNotFound errors and keep inference sane.
        print(
            f"Note: overriding DRO noise level from {val_noise_level} to 0.05 for dro_sim_source=espirit. "
            "Pass --dro_noise_level to override explicitly."
        )
        val_noise_level = 0.05

    losses_cfg = config.get("model", {}).get("losses", {})
    ei_cfg = losses_cfg.get("ei_loss", {})
    use_ei_loss = losses_cfg.get("use_ei_loss", False)
    if isinstance(use_ei_loss, str):
        use_ei_loss = use_ei_loss.strip().lower() in ("1", "true", "yes", "y")
    else:
        use_ei_loss = bool(use_ei_loss)
    plot_recon_label = None
    if args.plot_ei_label:
        label_suffix = "MC + EI" if use_ei_loss else "MC"
        plot_recon_label = rf"$|\mathrm{{BRISKNet}}_{{\mathrm{{pred}}}}|$ ({label_suffix})"
    arrival_method = (args.arrival_method or ei_cfg.get("arrival_method", "threshold")).lower()
    if args.arrival_fraction is None:
        arrival_fraction = float(ei_cfg.get("arrival_fraction", 0.1))
    else:
        arrival_fraction = float(args.arrival_fraction)
    if args.arrival_k is None:
        arrival_k = float(ei_cfg.get("arrival_shift_baseline_k", 2.0))
    else:
        arrival_k = float(args.arrival_k)

    inference_settings = {
        "csmaps_style": dro_csmaps_source,
        "dro_sim_source": dro_sim_source,
        "dro_espirit_csmaps_dir": args.dro_espirit_csmaps_dir,
        "dro_espirit_grasp_dir": args.dro_espirit_grasp_dir,
        "grasp_lamda": primary_grasp_lamda,
        "grasp_lamdas": grasp_lamdas,
        "traj_method": args.traj_method,
        "output_root": output_root,
        "baseline": {
            "mode": args.baseline_mode,
            "seconds": args.baseline_seconds,
            "fraction": args.baseline_fraction,
            "min_frames": args.baseline_min_frames,
            "max_frames": args.baseline_max_frames,
            "total_scan_seconds": args.total_scan_seconds,
        },
        "arrival_method": arrival_method,
        "arrival_fraction": arrival_fraction,
        "arrival_k": arrival_k,
        "early_window": {
            "mode": "seconds_after_arrival",
            "seconds": args.early_seconds,
            "min_frames": args.early_min_frames,
            "max_frames": args.early_max_frames,
        },
        "windowing": {
            "chunk_size": int(eval_chunk_size),
            "chunk_overlap": int(eval_chunk_overlap),
            "compute_ssdu": bool(compute_ssdu),
            "ssdu_k_folds": int(ssdu_k_folds),
            "ssdu_grasp_k_folds": int(ssdu_grasp_k_folds),
            "ssdu_weighting": ssdu_weighting,
        },
        "normalization": {
            "model_norm": config["model"]["norm"],
            "rescale": bool(rescale),
            "diag_normalize": bool(args.diag_normalize),
            "normalize_dro_csmaps": bool(args.normalize_dro_csmaps),
        },
        "dro_noise_level": val_noise_level,
        "eval_params": {
            "spokes_per_frame": int(N_spokes_eval),
            "num_frames": int(N_time_eval),
            "raw_num_frames": int(N_time_eval_raw),
            "phase_index": args.phase_index,
        },
        "data": {
            "dro_dataset_root": args.new_dro_root,
            "raw_kspace_root": config["data"]["root_dir"],
            "dataset_key": config["data"]["dataset_key"],
            "raw_grasp_slice_idx": raw_grasp_slice_idx,
        },
        "model": {
            "name": config["model"]["name"],
            "encode_acceleration": bool(config["model"]["encode_acceleration"]),
            "encode_time_index": bool(config["model"]["encode_time_index"]),
        },
    }

    data_dir = config["data"]["root_dir"]
    model_type = config["model"]["name"]
    model_type_norm = str(model_type).strip().lower()
    mamba_variant = str(config.get("model", {}).get("mamba", {}).get("variant", "")).strip().lower()
    model_type_is_temporal_mamba = model_type_norm in {
        "mambatemporal",
        "mamba_temporal",
        "temporalmamba",
    } or (
        model_type_norm in {"mambarecon", "mamba_recon", "mamba"}
        and mamba_variant in {"temporal", "temporal_1d", "radial_temporal"}
    )
    eval_uses_sliding_window = bool(N_time_eval > eval_chunk_size and not model_type_is_temporal_mamba)
    traj_method = args.traj_method or config.get("data", {}).get("traj_method", "get_traj")
    dro_dataset_root = args.new_dro_root
    spf_suffix_policy = _spf_suffix_policy(dro_dataset_root)

    resolved_espirit_dir = args.dro_espirit_csmaps_dir or os.path.join(dro_dataset_root, "csmaps_espirit")
    if len(grasp_lamdas) > 1 and args.dro_espirit_grasp_dir:
        raise ValueError(
            "Multiple GRASP lambdas requested, but --dro_espirit_grasp_dir was set. "
            "Use lambda-specific ESPIRiT dirs under the DRO root instead."
        )
    if len(grasp_lamdas) > 1 and dro_csmaps_source != "espirit":
        raise ValueError(
            "Multiple GRASP lambdas are only supported with dro_csmaps_source=espirit."
        )
    grasp_dirs = {
        lam: os.path.join(dro_dataset_root, f"espirit_grasp_recons_lam{lam:g}")
        for lam in grasp_lamdas
    }
    resolved_espirit_grasp_dir = args.dro_espirit_grasp_dir or grasp_dirs[primary_grasp_lamda]

    print("=== Inference Configuration (resolved) ===")
    print(f"DRO dataset root: {dro_dataset_root}")
    print(f"DRO noise level: {val_noise_level}")
    print(f"DRO sim source: {dro_sim_source}")
    print(f"DRO csmaps source: {dro_csmaps_source}")
    if dro_csmaps_source == "espirit":
        print(f"DRO ESPIRiT csmaps dir: {resolved_espirit_dir}")
        print(f"DRO ESPIRiT GRASP dir: {resolved_espirit_grasp_dir}")
        if len(grasp_lamdas) > 1:
            print(f"GRASP lambdas: {', '.join(f'{lam:g}' for lam in grasp_lamdas)}")
    else:
        print("DRO csmaps source path: smap inside each DRO .mat")
    print(f"Trajectory method: {traj_method}")
    print(f"Eval spokes/frame (BRISKNet): {int(N_spokes_eval)}")
    print(f"Eval frames (BRISKNet): {int(N_time_eval)}")
    if int(N_time_eval_raw) != int(N_time_eval):
        print(f"Raw eval frames (config): {int(N_time_eval_raw)}")
    print(f"Eval total spokes: {int(N_spokes_eval) * int(N_time_eval)}")
    print(f"Rescale (best-fit scalar): {rescale}")
    print(f"Normalize DRO csmaps: {bool(args.normalize_dro_csmaps)}")
    if model_type_is_temporal_mamba and int(N_time_eval) > int(eval_chunk_size):
        print(
            "[Inference] TemporalMamba detected: using direct full-sequence inference "
            f"(chunk_size={int(eval_chunk_size)} ignored for model forward)."
        )

    val_dataset = NewDROMatDataset(
        root_dir=dro_dataset_root,
        raw_kspace_path=data_dir,
        model_type=model_type,
        patient_ids=val_ids,
        dataset_key=config["data"]["dataset_key"],
        spokes_per_frame=N_spokes_eval,
        num_frames=N_time_eval,
        grasp_slice_idx=raw_grasp_slice_idx,
        dro_csmaps_source=dro_csmaps_source,
        espirit_csmaps_dir=args.dro_espirit_csmaps_dir,
        espirit_grasp_recons_dir=resolved_espirit_grasp_dir,
        skip_raw_grasp_metrics=args.skip_raw_grasp_metrics,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["dataloader"]["batch_size"],
        shuffle=False,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=True,
    )

    num_samples = args.num_samples or config.get("evaluation", {}).get("num_samples", len(val_dataset))
    num_samples = min(num_samples, len(val_dataset))

    resolved_phase_index = args.phase_index
    curriculum_cfg = config.get("training", {}).get("curriculum_learning", {})
    phases = curriculum_cfg.get("phases", [])
    if resolved_phase_index is None and curriculum_cfg.get("enabled") and phases:
        resolved_phase_index = len(phases) - 1

    resolved_args = dict(vars(args))
    resolved_args.update(
        {
            "checkpoint": ckpt_path,
            "config": config_path,
            "device": str(device),
            "eval_spokes": int(N_spokes_eval),
            "eval_frames": int(N_time_eval),
            "num_samples": int(num_samples),
            "phase_index": resolved_phase_index if resolved_phase_index is not None else "none",
            "dro_espirit_csmaps_dir": resolved_espirit_dir,
            "dro_espirit_grasp_dir": resolved_espirit_grasp_dir,
            "grasp_lamda": primary_grasp_lamda,
            "grasp_lamdas": grasp_lamdas,
            "dro_csmaps_source": dro_csmaps_source,
            "dro_sim_source": dro_sim_source,
            "traj_method": traj_method,
            "dro_noise_level": val_noise_level,
            "arrival_method": arrival_method,
            "arrival_fraction": arrival_fraction,
            "arrival_k": arrival_k,
            "early_max_frames": (
                args.early_max_frames if args.early_max_frames is not None else "no_max"
            ),
        }
    )

    inference_settings["data"]["dro_dataset_root"] = dro_dataset_root
    inference_settings["traj_method"] = traj_method
    inference_settings["dro_espirit_csmaps_dir"] = resolved_espirit_dir
    inference_settings["dro_espirit_grasp_dir"] = resolved_espirit_grasp_dir
    inference_settings["grasp_lamda"] = primary_grasp_lamda
    inference_settings["grasp_lamdas"] = grasp_lamdas
    inference_settings["eval_params"]["phase_index"] = (
        resolved_phase_index if resolved_phase_index is not None else "none"
    )
    inference_settings["eval_params"]["num_samples"] = int(num_samples)
    inference_settings["early_window"]["max_frames"] = (
        args.early_max_frames if args.early_max_frames is not None else "no_max"
    )

    _write_inference_metadata(
        inference_dir=inference_dir,
        args=args,
        config=config,
        config_path=config_path,
        ckpt_path=ckpt_path,
        device=device,
        eval_spokes=N_spokes_eval,
        eval_frames=N_time_eval,
        inference_settings=inference_settings,
        resolved_args=resolved_args,
    )

    with open(os.path.join(inference_dir, "grasp_lambda.txt"), "w") as f:
        for lam in grasp_lamdas:
            f.write(f"{lam:g}\n")

    # Prep physics for inference.
    N_samples = config["data"]["samples"]
    H, W = config["data"]["height"], config["data"]["width"]
    N_full = H * math.pi / 2

    if len(val_dataset.sample_ids) == 0:
        raise ValueError("No DRO samples available to load trajectory.")
    traj_sample_id = val_dataset.sample_ids[0]
    traj_mat_path = val_dataset._resolve_kspace_mat_path(traj_sample_id)
    eval_ktraj, eval_dcomp, eval_nufft_ob, eval_adjnufft_ob, traj_samples = _prep_nufft_from_dro_traj(
        traj_mat_path,
        spokes_per_frame=int(N_spokes_eval),
        num_frames=int(N_time_eval),
        expected_samples=int(N_samples),
        traj_method=traj_method,
    )
    print(f"Using DRO traj from: {traj_mat_path}")
    if traj_samples != int(N_samples):
        raise ValueError(f"Traj samples ({traj_samples}) != config samples ({N_samples}).")
    
    eval_ktraj = eval_ktraj.to(device)
    eval_dcomp = eval_dcomp.to(device)
    eval_nufft_ob = eval_nufft_ob.to(device)
    eval_adjnufft_ob = eval_adjnufft_ob.to(device)
    eval_physics = MCNUFFT(eval_nufft_ob, eval_adjnufft_ob, eval_ktraj, eval_dcomp)

    raw_total_spokes = 288
    if raw_total_spokes % int(N_spokes_eval) != 0:
        raise ValueError(
            f"raw_total_spokes ({raw_total_spokes}) is not divisible by eval_spokes ({int(N_spokes_eval)})."
        )
    raw_frames = raw_total_spokes // int(N_spokes_eval)
    if int(raw_frames) != int(N_time_eval_raw):
        print(
            f"Note: raw_frames ({int(raw_frames)}) != config eval_frames ({int(N_time_eval_raw)}); "
            "raw eval will use raw_frames."
        )
    raw_ktraj, raw_dcomp, raw_nufft_ob, raw_adjnufft_ob = prep_nufft(
        N_samples, N_spokes_eval, raw_frames, traj_method=traj_method
    )
    raw_ktraj = raw_ktraj.to(device)
    raw_dcomp = raw_dcomp.to(device)
    raw_nufft_ob = raw_nufft_ob.to(device)
    raw_adjnufft_ob = raw_adjnufft_ob.to(device)
    raw_physics = MCNUFFT(raw_nufft_ob, raw_adjnufft_ob, raw_ktraj, raw_dcomp)

    raw_chunk_size = min(int(eval_chunk_size), int(raw_frames))
    raw_chunk_overlap = min(int(eval_chunk_overlap), max(int(raw_chunk_size) - 1, 0))
    raw_uses_sliding_window = bool(raw_frames > raw_chunk_size and not model_type_is_temporal_mamba)

    # Build and load model.
    block_dir = os.path.join(output_dir, "block_outputs")
    os.makedirs(block_dir, exist_ok=True)
    model = _build_model(config, device, block_dir)
    model = _load_weights(model, ckpt_path)

    acceleration_val = torch.tensor([N_full / int(eval_ktraj.shape[1] / config["data"]["samples"])], dtype=torch.float, device=device)

    results_by_lam = {lam: [] for lam in grasp_lamdas}
    grasp_results_by_lam = {lam: [] for lam in grasp_lamdas}
    raw_results = []
    zf_results = []
    inference_times = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader, total=num_samples, desc="Inference on validation")):
            if idx >= num_samples:
                break
            label = f"sample{idx:02d}"

            (
                dro_kspace,
                csmap,
                ground_truth,
                dro_grasp_img,
                mask,
                grasp_path,
                raw_kspace,
                raw_grasp_img,
                raw_csmaps,
            ) = batch

            sample_id = val_dataset.sample_ids[idx]
            if dro_csmaps_source == "espirit":
                primary_dir = grasp_dirs[primary_grasp_lamda]
                grasp_np, grasp_path = _load_grasp_from_dir(
                    sample_id,
                    int(N_spokes_eval),
                    int(N_time_eval),
                    primary_dir,
                    spf_suffix_policy=spf_suffix_policy,
                )
                dro_grasp_img = _grasp_np_to_torch(grasp_np)

            # csmap = csmap.squeeze(0).to(device)
            csmap = csmap.to(device)
            ground_truth = ground_truth.to(device)
            dro_grasp_img = dro_grasp_img.to(device)
            dro_kspace = dro_kspace.squeeze(0).to(device)
            raw_kspace = raw_kspace.squeeze(0).to(device)
            raw_grasp_img = raw_grasp_img.to(device)
            raw_csmaps = raw_csmaps.squeeze(0).to(device)

            raw_valid = bool(torch.isfinite(raw_kspace).all().item())
            if raw_valid:
                raw_valid = bool(torch.isfinite(raw_csmaps).all().item())
            if not raw_valid:
                tqdm.write(
                    f"[Raw] Skipping raw eval for {sample_id}: invalid/corrupted raw k-space."
                )

            dro_csmap_scale = None
            if args.normalize_dro_csmaps:
                dro_csmap_scale = _compute_dro_csmap_scale(csmap, mask)
                if dro_csmap_scale is not None:
                    csmap = csmap / dro_csmap_scale
                    dro_kspace = dro_kspace / dro_csmap_scale
                    if idx == 0:
                        tqdm.write(
                            f"[DRO csmap normalize] Applied scale={dro_csmap_scale:.6f} "
                            "(csmap and kspace divided by this factor)."
                        )

            acceleration_encoding = acceleration_val if config["model"]["encode_acceleration"] else None
            start_timepoint_index = torch.tensor([0], dtype=torch.float, device=device) if config["model"]["encode_time_index"] else None

            if eval_uses_sliding_window:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                infer_start = time.perf_counter()
                x_recon, _ = sliding_window_inference(
                    H,
                    W,
                    N_time_eval,
                    eval_ktraj,
                    eval_dcomp,
                    eval_nufft_ob,
                    eval_adjnufft_ob,
                    eval_chunk_size,
                    eval_chunk_overlap,
                    dro_kspace,
                    csmap,
                    acceleration_encoding,
                    start_timepoint_index,
                    model,
                    epoch="inference",
                    device=device,
                    norm=config["model"]["norm"],
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                infer_time = time.perf_counter() - infer_start
                inference_times.append(infer_time)
                tqdm.write(
                    f"[Timing] {label}: DRO recon-only inference time = {infer_time:.3f}s"
                )
            else:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)

                infer_start = time.perf_counter()
                x_recon, *_ = model(
                    dro_kspace, eval_physics, csmap, acceleration_encoding, start_timepoint_index, epoch="inference", norm=config["model"]["norm"]
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                infer_time = time.perf_counter() - infer_start
                inference_times.append(infer_time)
                tqdm.write(
                    f"[Timing] {label}: DRO recon-only inference time = {infer_time:.3f}s"
                )
            raw_x_recon = None
            if raw_valid:
                if raw_uses_sliding_window:
                    raw_x_recon, _ = sliding_window_inference(
                        H,
                        W,
                        raw_frames,
                        raw_ktraj,
                        raw_dcomp,
                        raw_nufft_ob,
                        raw_adjnufft_ob,
                        raw_chunk_size,
                        raw_chunk_overlap,
                        raw_kspace,
                        raw_csmaps,
                        acceleration_encoding,
                        start_timepoint_index,
                        model,
                        epoch="inference",
                        device=device,
                        norm=config["model"]["norm"],
                    )
                else:
                    raw_x_recon, *_ = model(
                        raw_kspace, raw_physics, raw_csmaps, acceleration_encoding, start_timepoint_index, epoch="inference", norm=config["model"]["norm"]
                    )

            sample_dir = os.path.join(inference_dir, f"sample_{idx:02d}")
            os.makedirs(sample_dir, exist_ok=True)

            zf_complex = None
            if args.compute_zf_baseline or args.save_debug_arrays:
                zf_complex = eval_physics(True, dro_kspace, csmap)
            if zf_complex is not None:
                _save_zf_plot(sample_dir, zf_complex, frame_idx=N_time_eval // 2)

            if idx == 0 and (args.save_debug_arrays or args.compute_zf_baseline or args.diagnostics):
                def _mag_minmax(x: torch.Tensor):
                    if not isinstance(x, torch.Tensor):
                        return None, None
                    if x.numel() == 0:
                        return None, None
                    if x.shape[-1] == 0:
                        return None, None
                    if x.ndim >= 2 and x.shape[1] == 2:
                        mag = torch.sqrt(x[:, 0, ...] ** 2 + x[:, 1, ...] ** 2)
                        return float(mag.min().item()), float(mag.max().item())
                    if torch.is_complex(x):
                        mag = torch.abs(x)
                        return float(mag.min().item()), float(mag.max().item())
                    return float(x.min().item()), float(x.max().item())

                gt_min, gt_max = _mag_minmax(ground_truth)
                grasp_min, grasp_max = _mag_minmax(dro_grasp_img)
                dl_min, dl_max = _mag_minmax(x_recon)
                k_min, k_max = _mag_minmax(dro_kspace)
                tqdm.write("=== Debug: first-sample tensor stats ===")
                tqdm.write(f"grasp_path: {grasp_path}")
                tqdm.write(
                    "shapes: "
                    f"dro_kspace={tuple(dro_kspace.shape)}, csmap={tuple(csmap.shape)}, "
                    f"gt={tuple(ground_truth.shape)}, grasp={tuple(dro_grasp_img.shape)}, "
                    f"brisknet={tuple(x_recon.shape)}"
                )
                tqdm.write(
                    "magnitude ranges: "
                    f"kspace[{k_min:.3g},{k_max:.3g}] "
                    f"gt[{gt_min:.3g},{gt_max:.3g}] "
                    f"grasp[{grasp_min:.3g},{grasp_max:.3g}] "
                    f"brisknet[{dl_min:.3g},{dl_max:.3g}]"
                )

            if args.diagnostics:
                ref_map = {
                    "gt": ground_truth,
                    "grasp": dro_grasp_img,
                    "raw_grasp": raw_grasp_img,
                    "raw_recon": raw_x_recon,
                }
                if args.skip_raw_grasp_metrics and args.diag_ref == "raw_grasp":
                    tqdm.write(
                        "[Diagnostics] skip_raw_grasp_metrics enabled; "
                        "using raw_recon as diag_ref instead of raw_grasp."
                    )
                    ref_choice = ref_map.get("raw_recon")
                elif (not raw_valid) and args.diag_ref in ("raw_grasp", "raw_recon"):
                    tqdm.write(
                        "[Diagnostics] raw data unavailable; using grasp as diag_ref instead of raw."
                    )
                    ref_choice = ref_map.get("grasp")
                else:
                    ref_choice = ref_map.get(args.diag_ref)
                if ref_choice is None:
                    raise ValueError(f"Unknown diag_ref '{args.diag_ref}'.")
                diag_mask = None
                if isinstance(mask, dict):
                    if "malignant" in mask:
                        diag_mask = mask["malignant"]
                        mask_np = diag_mask.detach().cpu().numpy() if isinstance(diag_mask, torch.Tensor) else np.asarray(diag_mask)
                        if not np.any(mask_np):
                            diag_mask = None
                else:
                    diag_mask = mask
                if diag_mask is not None:
                    save_diagnostics(
                        sample_dir=sample_dir,
                        brisk=x_recon,
                        reference=ref_choice,
                        mask=diag_mask,
                        args=args,
                    )

                if raw_valid and (not args.skip_raw_grasp_metrics):
                    _, patient_id = _resolve_plot_label(label, grasp_path)
                    slice_map = _load_slice_map()
                    resolved_slice_idx = slice_map.get(patient_id, raw_grasp_slice_idx)
                    raw_tumor_mask = None
                    if resolved_slice_idx is not None and resolved_slice_idx >= 0:
                        raw_tumor_mask = _load_tumor_mask(cluster, patient_id, slice_idx=resolved_slice_idx)
                    if raw_tumor_mask is not None and np.any(raw_tumor_mask):
                        save_diagnostics(
                            sample_dir=sample_dir,
                            brisk=raw_x_recon,
                            reference=raw_grasp_img,
                            mask=raw_tumor_mask,
                            args=args,
                            diag_subdir="diagnostics_raw",
                            ref_label="Raw GRASP",
                            brisk_label="BRISKNet (raw)",
                        )

            ssdu_result = {}
            ssdu_grasp_result = {}
            if compute_ssdu and raw_valid:
                ssdu_chunk_size = raw_chunk_size if raw_uses_sliding_window else None
                ssdu_result = compute_ssdu_kspace_nmse(
                    model,
                    raw_kspace,
                    raw_csmaps,
                    raw_ktraj,
                    raw_dcomp,
                    raw_nufft_ob,
                    raw_adjnufft_ob,
                    spokes_per_frame=int(N_spokes_eval),
                    K_folds=ssdu_k_folds,
                    baseline_weighting=ssdu_weighting,
                    device=device,
                    acceleration_encoding=acceleration_encoding,
                    start_timepoint_index=start_timepoint_index,
                    norm=config["model"]["norm"],
                    epoch="inference",
                    chunk_size=ssdu_chunk_size,
                    chunk_overlap=raw_chunk_overlap,
                )
                ssdu_grasp_result = compute_ssdu_kspace_nmse_grasp(
                    lambda y_used, ktraj_used, dcomp_used, csmap, samples_per_spoke: GRASPRecon_from_ktraj(
                        csmap,
                        y_used,
                        ktraj_used,
                        samples_per_spoke,
                        device=None,
                    ),
                    raw_kspace,
                    raw_csmaps,
                    raw_ktraj,
                    raw_dcomp,
                    raw_nufft_ob,
                    raw_adjnufft_ob,
                    spokes_per_frame=int(N_spokes_eval),
                    K_folds=ssdu_grasp_k_folds,
                    orientation_transform="raw_grasp",
                    baseline_weighting=ssdu_weighting,
                    device=device,
                )

            x_recon = torch.rot90(x_recon, k=3, dims=[2, 3])
            ground_truth = torch.rot90(ground_truth, k=3, dims=[-2, -1])
            dro_grasp_img = dro_grasp_img.unsqueeze(0)
            grasp_primary = torch.rot90(dro_grasp_img, k=3, dims=[2, 4])
            dro_grasp_img = grasp_primary

            for tissue in mask.keys():
                mask[tissue] = torch.rot90(mask[tissue], k=3, dims=[-2, -1])

            for grasp_lamda in grasp_lamdas:
                if grasp_lamda == primary_grasp_lamda:
                    dro_grasp_img_lam = grasp_primary
                    grasp_path_lam = grasp_path
                else:
                    grasp_np, grasp_path_lam = _load_grasp_from_dir(
                        sample_id,
                        int(N_spokes_eval),
                        int(N_time_eval),
                        grasp_dirs[grasp_lamda],
                        spf_suffix_policy=spf_suffix_policy,
                    )
                    dro_grasp_img_lam = _grasp_np_to_torch(grasp_np).to(device)
                    dro_grasp_img_lam = dro_grasp_img_lam.unsqueeze(0)
                    dro_grasp_img_lam = torch.rot90(dro_grasp_img_lam, k=3, dims=[2, 4])

                filename_suffix = f"lam{grasp_lamda:g}" if len(grasp_lamdas) > 1 else ""

                dro_metrics = eval_sample(
                    dro_kspace,
                    csmap,
                    ground_truth,
                    x_recon,
                    eval_physics,
                    mask,
                    dro_grasp_img_lam,
                    acceleration_val,
                    int(N_spokes_eval),
                    sample_dir,
                    label,
                    device,
                    cluster,
                    dro_eval=True,
                    grasp_path=grasp_path_lam,
                    rescale=rescale,
                    filename_suffix=filename_suffix,
                    baseline_mode=args.baseline_mode,
                    baseline_seconds=args.baseline_seconds,
                    baseline_fraction=args.baseline_fraction,
                    baseline_min_frames=args.baseline_min_frames,
                    baseline_max_frames=args.baseline_max_frames,
                    arrival_k=arrival_k,
                    arrival_method=arrival_method,
                    arrival_fraction=arrival_fraction,
                    early_seconds=args.early_seconds,
                    early_min_frames=args.early_min_frames,
                    early_max_frames=args.early_max_frames,
                    total_scan_seconds=args.total_scan_seconds,
                    recon_label=plot_recon_label,
                    plot_malignant_curve=True,
                )

                (
                    grasp_ssim,
                    grasp_psnr,
                    grasp_mse,
                    grasp_lpips,
                    grasp_dc_mse,
                    grasp_dc_mae,
                    grasp_aux,
                ) = eval_grasp(
                    dro_kspace,
                    csmap,
                    ground_truth,
                    dro_grasp_img_lam,
                    eval_physics,
                    device,
                    sample_dir,
                    rescale=rescale,
                    dro_eval=True,
                    return_aux=True,
                )

                ssim, psnr, mse, lpips, dc_mse, dc_mae, recon_corr, grasp_corr, temporal_metrics = dro_metrics

                results_by_lam[grasp_lamda].append(
                    dict(
                        sample=label,
                        ssim=ssim,
                        psnr=psnr,
                        mse=mse,
                        lpips=lpips,
                        dc_mse=dc_mse,
                        dc_mae=dc_mae,
                        recon_corr=recon_corr,
                        grasp_corr=grasp_corr,
                        dro_csmap_scale=dro_csmap_scale,
                        **(temporal_metrics or {}),
                    )
                )
                grasp_results_by_lam[grasp_lamda].append(
                    dict(
                        sample=label,
                        ssim=grasp_ssim,
                        psnr=grasp_psnr,
                        mse=grasp_mse,
                        lpips=grasp_lpips,
                        dc_mse=grasp_dc_mse,
                        dc_mae=grasp_dc_mae,
                        **(grasp_aux or {}),
                    )
                )

            if args.compute_zf_baseline:
                (
                    zf_ssim,
                    zf_psnr,
                    zf_mse,
                    zf_lpips,
                    zf_dc_mse,
                    zf_dc_mae,
                    zf_aux,
                ) = eval_zf(
                    dro_kspace,
                    csmap,
                    ground_truth,
                    eval_physics,
                    mask,
                    device,
                    rescale=rescale,
                    zf_complex_override=zf_complex,
                    return_aux=True,
                )
                zf_results.append(
                    dict(
                        sample=label,
                        ssim=zf_ssim,
                        psnr=zf_psnr,
                        mse=zf_mse,
                        lpips=zf_lpips,
                        dc_mse=zf_dc_mse,
                        dc_mae=zf_dc_mae,
                        **(zf_aux or {}),
                    )
                )

            raw_ground_truth = ground_truth
            if ground_truth.shape[1] != raw_frames:
                raw_ground_truth = _resample_temporal(ground_truth, raw_frames)
                if idx == 0:
                    tqdm.write(
                        f"[Raw eval] Resampled GT frames from {ground_truth.shape[1]} to {raw_frames} "
                        "to match raw k-space."
                    )

            if not raw_valid:
                raw_dc_mse = None
                raw_dc_mae = None
                raw_dc_psnr = None
                raw_grasp_dc_mse = None
                raw_grasp_dc_mae = None
                raw_grasp_dc_psnr = None
            elif args.skip_raw_grasp_metrics:
                raw_x_recon_complex = to_torch_complex(raw_x_recon).squeeze()
                raw_kspace_squeezed = raw_kspace.squeeze()
                recon_kspace = raw_physics(False, raw_x_recon_complex, raw_csmaps)
                raw_dc_mse, raw_dc_mae = calc_dc(recon_kspace, raw_kspace_squeezed, device)
                raw_dc_psnr = calc_dc_psnr(raw_kspace_squeezed, raw_dc_mse, device)
                raw_grasp_dc_mse = None
                raw_grasp_dc_mae = None
                raw_grasp_dc_psnr = None
            else:
                raw_dc_mse, raw_dc_mae, raw_dc_psnr, _ = eval_sample(
                    raw_kspace,
                    raw_csmaps,
                    raw_ground_truth,
                    raw_x_recon,
                    raw_physics,
                    mask,
                    raw_grasp_img,
                    acceleration_val,
                    int(N_spokes_eval),
                    sample_dir,
                    f"{label}_raw",
                    device,
                    cluster,
                    dro_eval=False,
                    grasp_path=grasp_path,
                    raw_slice_idx=raw_grasp_slice_idx,
                    rescale=rescale,
                    baseline_mode=args.baseline_mode,
                    baseline_seconds=args.baseline_seconds,
                    baseline_fraction=args.baseline_fraction,
                    baseline_min_frames=args.baseline_min_frames,
                    baseline_max_frames=args.baseline_max_frames,
                    arrival_k=arrival_k,
                    arrival_method=arrival_method,
                    arrival_fraction=arrival_fraction,
                    early_seconds=args.early_seconds,
                    early_min_frames=args.early_min_frames,
                    early_max_frames=args.early_max_frames,
                    total_scan_seconds=args.total_scan_seconds,
                    recon_label=plot_recon_label,
                    plot_malignant_curve=True,
                )
                raw_grasp_dc_mse, raw_grasp_dc_mae, raw_grasp_dc_psnr = eval_grasp(
                    raw_kspace,
                    raw_csmaps,
                    raw_ground_truth,
                    raw_grasp_img,
                    raw_physics,
                    device,
                    sample_dir,
                    rescale=rescale,
                    dro_eval=False,
                )
            raw_results.append(
                dict(
                    sample=label,
                    raw_dc_mse=raw_dc_mse,
                    raw_dc_mae=raw_dc_mae,
                    raw_dc_psnr=raw_dc_psnr,
                    raw_grasp_dc_mse=raw_grasp_dc_mse,
                    raw_grasp_dc_mae=raw_grasp_dc_mae,
                    raw_grasp_dc_psnr=raw_grasp_dc_psnr,
                    raw_ssdu_nmse=ssdu_result.get("ssdu_nmse_mean"),
                    raw_grasp_ssdu_nmse=ssdu_grasp_result.get("ssdu_nmse_mean"),
                )
            )

            if args.save_debug_arrays:
                debug_npz = os.path.join(sample_dir, "debug_arrays_mag.npz")
                gt_mag = torch.sqrt(ground_truth[:, 0, ...] ** 2 + ground_truth[:, 1, ...] ** 2).squeeze(0)
                dl_mag = torch.sqrt(x_recon[:, 0, ...] ** 2 + x_recon[:, 1, ...] ** 2).squeeze(0)
                grasp_mag = torch.sqrt(dro_grasp_img[:, 0, ...] ** 2 + dro_grasp_img[:, 1, ...] ** 2).squeeze(0)
                gt_mag_np = gt_mag.detach().cpu().numpy().astype(np.float32, copy=False)  # (T,H,W)
                dl_mag_np = dl_mag.detach().cpu().numpy().astype(np.float32, copy=False)  # (H,W,T)
                dl_mag_np = np.transpose(dl_mag_np, (2, 0, 1))  # (T,H,W)
                grasp_mag_np = grasp_mag.detach().cpu().numpy().astype(np.float32, copy=False)
                # Canonicalize to (T,H,W) for easy comparison/debugging.
                if (
                    grasp_mag_np.ndim == 3
                    and grasp_mag_np.shape[0] != gt_mag_np.shape[0]
                    and grasp_mag_np.shape[1] == gt_mag_np.shape[0]
                ):
                    # Common case: (H,T,W) -> (T,H,W)
                    grasp_mag_np = np.transpose(grasp_mag_np, (1, 0, 2))
                zf_mag_np = np.array([], dtype=np.float32)
                if zf_complex is not None:
                    zf_mag_np = torch.abs(zf_complex).detach().cpu().numpy().astype(np.float32, copy=False)  # (H,W,T)
                    zf_mag_np = np.transpose(zf_mag_np, (2, 0, 1))  # (T,H,W)

                mask_union = np.array([], dtype=np.uint8)
                if isinstance(mask, dict) and mask:
                    union = None
                    for v in mask.values():
                        if v is None:
                            continue
                        v_np = v.detach().cpu().numpy().squeeze().astype(bool)
                        if v_np.ndim != 2:
                            continue
                        union = v_np if union is None else (union | v_np)
                    if union is not None:
                        mask_union = union.astype(np.uint8)

                np.savez_compressed(
                    debug_npz,
                    gt_mag=gt_mag_np,
                    dl_mag=dl_mag_np,
                    grasp_mag=grasp_mag_np,
                    zf_mag=zf_mag_np,
                    mask_union=mask_union,
                    spokes_per_frame=int(N_spokes_eval),
                    num_frames=int(N_time_eval),
                )

    mean_infer = None
    std_infer = None
    if inference_times:
        mean_infer = sum(inference_times) / len(inference_times)
        std_infer = statistics.stdev(inference_times) if len(inference_times) > 1 else 0.0
        infer_mode = "sliding-window" if eval_uses_sliding_window else "direct-full"
        print(
            f"Inference timing (recon only, {infer_mode}): "
            f"{mean_infer:.3f}s ± {std_infer:.3f}s per sample"
        )
        results_dir = os.path.join(REPO_ROOT, "results")
        os.makedirs(results_dir, exist_ok=True)
        times_path = os.path.join(results_dir, "inference_times")
        write_header = not os.path.exists(times_path)
        acceleration_report = (320.0 * math.pi / 2.0) / float(N_spokes_eval)
        seconds_per_frame = 150.0 / float(N_time_eval)
        with open(times_path, "a") as f:
            if write_header:
                f.write(
                    "exp_name,spokes_per_frame,time_frames,acceleration,seconds_per_frame,"
                    "mean_infer_s,std_infer_s\n"
                )
            f.write(
                f"{exp_name},{int(N_spokes_eval)},{int(N_time_eval)},"
                f"{acceleration_report:.6f},{seconds_per_frame:.6f},"
                f"{mean_infer:.6f},{std_infer:.6f}\n"
            )

    results = results_by_lam[primary_grasp_lamda]
    grasp_results = grasp_results_by_lam[primary_grasp_lamda]

    # Save metrics.
    metrics_path = os.path.join(inference_dir, "metrics.csv")
    with open(metrics_path, "w") as f:
        headers = [
            "sample",
            "dro_csmap_scale",
            "dl_ssim",
            "dl_psnr",
            "dl_psnr_fg",
            "dl_mse",
            "dl_mse_fg",
            "dl_lpips",
            "dl_dc_mse",
            "dl_dc_mae",
            "dl_dc_mse_bestfit",
            "dl_dc_mae_bestfit",
            "dl_dc_scale_abs",
            "dl_dc_scale_phase",
            "dl_img_scale",
            "dl_recon_corr",
            "grasp_corr",
            "grasp_ssim",
            "grasp_psnr",
            "grasp_psnr_fg",
            "grasp_mse",
            "grasp_mse_fg",
            "grasp_lpips",
            "grasp_dc_mse",
            "grasp_dc_mae",
            "grasp_dc_mse_bestfit",
            "grasp_dc_mae_bestfit",
            "grasp_dc_scale_abs",
            "grasp_dc_scale_phase",
            "grasp_img_scale",
            "zf_ssim",
            "zf_psnr",
            "zf_psnr_fg",
            "zf_mse",
            "zf_mse_fg",
            "zf_lpips",
            "zf_dc_mse",
            "zf_dc_mae",
            "zf_dc_mse_bestfit",
            "zf_dc_mae_bestfit",
            "zf_dc_scale_abs",
            "zf_dc_scale_phase",
            "zf_img_scale",
            "fg_fraction",
            "raw_dc_mse",
            "raw_dc_mae",
            "raw_dc_psnr",
            "raw_grasp_dc_mse",
            "raw_grasp_dc_mae",
            "raw_grasp_dc_psnr",
            "raw_ssdu_nmse",
            "raw_grasp_ssdu_nmse",
        ]
        f.write(",".join(headers) + "\n")
        zf_lookup = {row["sample"]: row for row in zf_results}
        for dro_row, grasp_row, raw_row in zip(results, grasp_results, raw_results):
            zf_row = zf_lookup.get(dro_row["sample"], {})
            dl_psnr_fg = dro_row.get("dl_psnr_fg")
            dl_mse_fg = dro_row.get("dl_mse_fg")
            grasp_psnr_fg = dro_row.get("grasp_psnr_fg")
            grasp_mse_fg = dro_row.get("grasp_mse_fg")
            fg_fraction = dro_row.get("fg_fraction")
            zf_psnr_fg = zf_row.get("zf_psnr_fg")
            zf_mse_fg = zf_row.get("zf_mse_fg")
            row = [
                dro_row["sample"],
                "" if dro_row.get("dro_csmap_scale") is None else f"{dro_row['dro_csmap_scale']:.6f}",
                f"{dro_row['ssim']:.6f}",
                f"{dro_row['psnr']:.6f}",
                "" if dl_psnr_fg is None else f"{dl_psnr_fg:.6f}",
                f"{dro_row['mse']:.6f}",
                "" if dl_mse_fg is None else f"{dl_mse_fg:.6f}",
                f"{dro_row['lpips']:.6f}",
                f"{dro_row['dc_mse']:.6f}",
                f"{dro_row['dc_mae']:.6f}",
                "" if dro_row.get("dl_dc_mse_bestfit") is None else f"{dro_row['dl_dc_mse_bestfit']:.6f}",
                "" if dro_row.get("dl_dc_mae_bestfit") is None else f"{dro_row['dl_dc_mae_bestfit']:.6f}",
                "" if dro_row.get("dl_dc_scale_abs") is None else f"{dro_row['dl_dc_scale_abs']:.6f}",
                "" if dro_row.get("dl_dc_scale_phase") is None else f"{dro_row['dl_dc_scale_phase']:.6f}",
                "" if dro_row.get("dl_img_scale") is None else f"{dro_row['dl_img_scale']:.6f}",
                "" if dro_row["recon_corr"] is None else f"{dro_row['recon_corr']:.6f}",
                "" if dro_row["grasp_corr"] is None else f"{dro_row['grasp_corr']:.6f}",
                f"{grasp_row['ssim']:.6f}",
                f"{grasp_row['psnr']:.6f}",
                "" if grasp_psnr_fg is None else f"{grasp_psnr_fg:.6f}",
                f"{grasp_row['mse']:.6f}",
                "" if grasp_mse_fg is None else f"{grasp_mse_fg:.6f}",
                f"{grasp_row['lpips']:.6f}",
                f"{grasp_row['dc_mse']:.6f}",
                f"{grasp_row['dc_mae']:.6f}",
                "" if grasp_row.get("grasp_dc_mse_bestfit") is None else f"{grasp_row['grasp_dc_mse_bestfit']:.6f}",
                "" if grasp_row.get("grasp_dc_mae_bestfit") is None else f"{grasp_row['grasp_dc_mae_bestfit']:.6f}",
                "" if grasp_row.get("grasp_dc_scale_abs") is None else f"{grasp_row['grasp_dc_scale_abs']:.6f}",
                "" if grasp_row.get("grasp_dc_scale_phase") is None else f"{grasp_row['grasp_dc_scale_phase']:.6f}",
                "" if dro_row.get("grasp_img_scale") is None else f"{dro_row['grasp_img_scale']:.6f}",
                "" if not zf_row else f"{zf_row.get('ssim', float('nan')):.6f}",
                "" if not zf_row else f"{zf_row.get('psnr', float('nan')):.6f}",
                "" if zf_psnr_fg is None else f"{zf_psnr_fg:.6f}",
                "" if not zf_row else f"{zf_row.get('mse', float('nan')):.6f}",
                "" if zf_mse_fg is None else f"{zf_mse_fg:.6f}",
                "" if not zf_row else f"{zf_row.get('lpips', float('nan')):.6f}",
                "" if not zf_row else f"{zf_row.get('dc_mse', float('nan')):.6f}",
                "" if not zf_row else f"{zf_row.get('dc_mae', float('nan')):.6f}",
                "" if zf_row.get("zf_dc_mse_bestfit") is None else f"{zf_row['zf_dc_mse_bestfit']:.6f}",
                "" if zf_row.get("zf_dc_mae_bestfit") is None else f"{zf_row['zf_dc_mae_bestfit']:.6f}",
                "" if zf_row.get("zf_dc_scale_abs") is None else f"{zf_row['zf_dc_scale_abs']:.6f}",
                "" if zf_row.get("zf_dc_scale_phase") is None else f"{zf_row['zf_dc_scale_phase']:.6f}",
                "" if zf_row.get("zf_img_scale") is None else f"{zf_row['zf_img_scale']:.6f}",
                "" if fg_fraction is None else f"{fg_fraction:.6f}",
                # f"{raw_row['raw_dc_mse']:.6f}",
                # f"{raw_row['raw_dc_mae']:.6f}",
                "" if raw_row.get("raw_dc_mse") is None else f"{raw_row['raw_dc_mse']:.6f}",
                "" if raw_row.get("raw_dc_mae") is None else f"{raw_row['raw_dc_mae']:.6f}",
                "" if raw_row.get("raw_dc_psnr") is None else f"{raw_row['raw_dc_psnr']:.6f}",
                # f"{raw_row['raw_grasp_dc_mse']:.6f}",
                # f"{raw_row['raw_grasp_dc_mae']:.6f}",
                "" if raw_row.get("raw_grasp_dc_mse") is None else f"{raw_row['raw_grasp_dc_mse']:.6f}",
                "" if raw_row.get("raw_grasp_dc_mae") is None else f"{raw_row['raw_grasp_dc_mae']:.6f}",

                "" if raw_row.get("raw_grasp_dc_psnr") is None else f"{raw_row['raw_grasp_dc_psnr']:.6f}",
                "" if raw_row.get("raw_ssdu_nmse") is None else f"{raw_row['raw_ssdu_nmse']:.6f}",
                "" if raw_row.get("raw_grasp_ssdu_nmse") is None else f"{raw_row['raw_grasp_ssdu_nmse']:.6f}",
            ]
            f.write(",".join(row) + "\n")

    metric_names = [
        "curve_corr",
        "curve_mae",
        "early_corr",
        "early_mae",
        "ttae_sec",
        "wash_in_slope_err",
        "iauc10_err",
        "peak_err",
        "ttpeak_err_sec",
    ]
    for label, prefix in (("malignant", ""), ("benign", "benign_")):
        for subset in ("all", "top10", "top20"):
            keys = [
                f"{prefix}{model}_{subset}_{metric}"
                for model in ("dl", "grasp")
                for metric in metric_names
            ]
            temporal_metrics_path = os.path.join(
                inference_dir, f"metrics_temporal_{label}_{subset}.csv"
            )
            with open(temporal_metrics_path, "w") as f:
                f.write(",".join(["sample"] + keys) + "\n")
                for dro_row in results:
                    row = [dro_row["sample"]]
                    for key in keys:
                        value = dro_row.get(key, np.nan)
                        if value is None or (isinstance(value, float) and np.isnan(value)):
                            row.append("")
                        else:
                            row.append(f"{value:.6f}")
                    f.write(",".join(row) + "\n")

    def _mean_std(values, key):
        vals = [v.get(key) for v in values if v.get(key) is not None and np.isfinite(v.get(key))]
        if not vals:
            return None, None
        mean = sum(vals) / len(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return mean, std

    dl_summary = {
        "ssim": _mean_std(results, "ssim"),
        "psnr": _mean_std(results, "psnr"),
        "psnr_fg": _mean_std(results, "dl_psnr_fg"),
        "mse": _mean_std(results, "mse"),
        "mse_fg": _mean_std(results, "dl_mse_fg"),
        "lpips": _mean_std(results, "lpips"),
        "dc_mse": _mean_std(results, "dc_mse"),
        "dc_mae": _mean_std(results, "dc_mae"),
        "dc_mse_bestfit": _mean_std(results, "dl_dc_mse_bestfit"),
        "dc_mae_bestfit": _mean_std(results, "dl_dc_mae_bestfit"),
        "dc_scale_abs": _mean_std(results, "dl_dc_scale_abs"),
        "img_scale": _mean_std(results, "dl_img_scale"),
        "recon_corr": _mean_std(results, "recon_corr"),
        "grasp_corr": _mean_std(results, "grasp_corr"),
        "fg_fraction": _mean_std(results, "fg_fraction"),
    }

    grasp_summary = {
        "ssim": _mean_std(grasp_results, "ssim"),
        "psnr": _mean_std(grasp_results, "psnr"),
        "psnr_fg": _mean_std(results, "grasp_psnr_fg"),
        "mse": _mean_std(grasp_results, "mse"),
        "mse_fg": _mean_std(results, "grasp_mse_fg"),
        "lpips": _mean_std(grasp_results, "lpips"),
        "dc_mse": _mean_std(grasp_results, "dc_mse"),
        "dc_mae": _mean_std(grasp_results, "dc_mae"),
        "dc_mse_bestfit": _mean_std(grasp_results, "grasp_dc_mse_bestfit"),
        "dc_mae_bestfit": _mean_std(grasp_results, "grasp_dc_mae_bestfit"),
        "dc_scale_abs": _mean_std(grasp_results, "grasp_dc_scale_abs"),
        "img_scale": _mean_std(results, "grasp_img_scale"),
    }

    def _format_mean_std(mean, std):
        if mean is None:
            return ""
        return f"{mean:.4f} ± {std:.4f}"

    def _format_mean_std_compact(mean, std, precision: int = 3):
        if mean is None:
            return ""
        return f"{mean:.{precision}f} ± {std:.{precision}f}"

    def _format_mean_std_precise(mean, std):
        if mean is None:
            return ""
        return f"{mean:.4e} ± {std:.4e}"

    def _render_table(headers, rows):
        widths = [len(str(h)) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))

        def fmt(row):
            return "| " + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) + " |"

        sep = "|-" + "-|-".join("-" * w for w in widths) + "-|"
        lines = [fmt(headers), sep]
        lines.extend(fmt(r) for r in rows)
        return "\n".join(lines)

    recon_corr_str = _format_mean_std(*dl_summary["recon_corr"])
    grasp_corr_str = _format_mean_std(*dl_summary["grasp_corr"])

    raw_summary = {
        "raw_dc_mse": _mean_std(raw_results, "raw_dc_mse"),
        "raw_dc_mae": _mean_std(raw_results, "raw_dc_mae"),
        "raw_dc_psnr": _mean_std(raw_results, "raw_dc_psnr"),
        "raw_grasp_dc_mse": _mean_std(raw_results, "raw_grasp_dc_mse"),
        "raw_grasp_dc_mae": _mean_std(raw_results, "raw_grasp_dc_mae"),
        "raw_grasp_dc_psnr": _mean_std(raw_results, "raw_grasp_dc_psnr"),
        "raw_ssdu_nmse": _mean_std(raw_results, "raw_ssdu_nmse"),
        "raw_grasp_ssdu_nmse": _mean_std(raw_results, "raw_grasp_ssdu_nmse"),
    }

    zf_summary = None
    if zf_results:
        zf_summary = {
            "ssim": _mean_std(zf_results, "ssim"),
            "psnr": _mean_std(zf_results, "psnr"),
            "psnr_fg": _mean_std(zf_results, "zf_psnr_fg"),
            "mse": _mean_std(zf_results, "mse"),
            "mse_fg": _mean_std(zf_results, "zf_mse_fg"),
            "lpips": _mean_std(zf_results, "lpips"),
            "dc_mse": _mean_std(zf_results, "dc_mse"),
            "dc_mae": _mean_std(zf_results, "dc_mae"),
            "dc_mse_bestfit": _mean_std(zf_results, "zf_dc_mse_bestfit"),
            "dc_mae_bestfit": _mean_std(zf_results, "zf_dc_mae_bestfit"),
            "dc_scale_abs": _mean_std(zf_results, "zf_dc_scale_abs"),
            "img_scale": _mean_std(zf_results, "zf_img_scale"),
        }

    print("=== Inference Summary (averaged over samples) ===")
    print("Image Metrics")
    image_headers = ["Method", "SSIM", "PSNR", "PSNR_FG", "MSE", "MSE_FG", "LPIPS"]
    image_rows = [
        [
            "BRISKNet",
            _format_mean_std(*dl_summary["ssim"]),
            _format_mean_std(*dl_summary["psnr"]),
            _format_mean_std(*dl_summary["psnr_fg"]),
            _format_mean_std(*dl_summary["mse"]),
            _format_mean_std(*dl_summary["mse_fg"]),
            _format_mean_std(*dl_summary["lpips"]),
        ],
        [
            "GRASP",
            _format_mean_std(*grasp_summary["ssim"]),
            _format_mean_std(*grasp_summary["psnr"]),
            _format_mean_std(*grasp_summary["psnr_fg"]),
            _format_mean_std(*grasp_summary["mse"]),
            _format_mean_std(*grasp_summary["mse_fg"]),
            _format_mean_std(*grasp_summary["lpips"]),
        ],
    ]
    if zf_summary is not None:
        image_rows.append(
            [
                "ZF",
                _format_mean_std(*zf_summary["ssim"]),
                _format_mean_std(*zf_summary["psnr"]),
                _format_mean_std(*zf_summary["psnr_fg"]),
                _format_mean_std(*zf_summary["mse"]),
                _format_mean_std(*zf_summary["mse_fg"]),
                _format_mean_std(*zf_summary["lpips"]),
            ]
        )
    print(_render_table(image_headers, image_rows))
    if recon_corr_str or grasp_corr_str:
        print(f"EC Corr (BRISKNet): {recon_corr_str}, EC Corr (GRASP): {grasp_corr_str}")
    fg_fraction_str = _format_mean_std(*dl_summary["fg_fraction"])
    if fg_fraction_str:
        print(f"Foreground mask fraction: {fg_fraction_str}")
    img_scale_parts = []
    dl_img_scale_str = _format_mean_std(*dl_summary["img_scale"])
    grasp_img_scale_str = _format_mean_std(*grasp_summary["img_scale"])
    zf_img_scale_str = _format_mean_std(*zf_summary["img_scale"]) if zf_summary is not None else ""
    if dl_img_scale_str:
        img_scale_parts.append(f"BRISKNet: {dl_img_scale_str}")
    if grasp_img_scale_str:
        img_scale_parts.append(f"GRASP: {grasp_img_scale_str}")
    if zf_img_scale_str:
        img_scale_parts.append(f"ZF: {zf_img_scale_str}")
    if img_scale_parts:
        print("Best-fit image gain (to GT): " + ", ".join(img_scale_parts))
    csmap_scale_str = _format_mean_std(*_mean_std(results, "dro_csmap_scale"))
    if csmap_scale_str:
        print(f"DRO csmap scale (median RSS): {csmap_scale_str}")
    print("K-space Metrics")
    k_headers = ["Method", "DC_MSE", "DC_MAE"]
    k_rows = [
        ["BRISKNet", _format_mean_std(*dl_summary["dc_mse"]), _format_mean_std(*dl_summary["dc_mae"])],
        ["GRASP", _format_mean_std(*grasp_summary["dc_mse"]), _format_mean_std(*grasp_summary["dc_mae"])],
    ]
    if zf_summary is not None:
        k_rows.append(["ZF", _format_mean_std(*zf_summary["dc_mse"]), _format_mean_std(*zf_summary["dc_mae"])])
    print("DRO")
    print(_render_table(k_headers, k_rows))
    if dl_summary["dc_mse_bestfit"][0] is not None or grasp_summary["dc_mse_bestfit"][0] is not None or (zf_summary is not None and zf_summary["dc_mse_bestfit"][0] is not None):
        zf_dc_bestfit = _format_mean_std(*zf_summary["dc_mse_bestfit"]) if zf_summary is not None else ""
        bestfit_line = (
            "DRO* -> "
            f"BRISKNet DC_MSE*: {_format_mean_std(*dl_summary['dc_mse_bestfit'])}, "
            f"GRASP DC_MSE*: {_format_mean_std(*grasp_summary['dc_mse_bestfit'])}"
        )
        if zf_dc_bestfit:
            bestfit_line += f", ZF DC_MSE*: {zf_dc_bestfit}"
        print(bestfit_line)

        zf_gain = _format_mean_std(*zf_summary["dc_scale_abs"]) if zf_summary is not None else ""
        gain_line = (
            "Gain -> "
            f"BRISKNet |c|: {_format_mean_std(*dl_summary['dc_scale_abs'])}, "
            f"GRASP |c|: {_format_mean_std(*grasp_summary['dc_scale_abs'])}"
        )
        if zf_gain:
            gain_line += f", ZF |c|: {zf_gain}"
        print(gain_line)
    raw_headers = ["Method", "DC_MSE", "DC_MAE", "DC_PSNR", "SSDU_NMSE"]
    raw_rows = [
        [
            "BRISKNet",
            _format_mean_std_precise(*raw_summary["raw_dc_mse"]),
            _format_mean_std_precise(*raw_summary["raw_dc_mae"]),
            _format_mean_std_precise(*raw_summary["raw_dc_psnr"]),
            _format_mean_std_precise(*raw_summary["raw_ssdu_nmse"]),
        ],
        [
            "GRASP",
            _format_mean_std_precise(*raw_summary["raw_grasp_dc_mse"]),
            _format_mean_std_precise(*raw_summary["raw_grasp_dc_mae"]),
            _format_mean_std_precise(*raw_summary["raw_grasp_dc_psnr"]),
            _format_mean_std_precise(*raw_summary["raw_grasp_ssdu_nmse"]),
        ],
    ]
    print("RAW")
    print(_render_table(raw_headers, raw_rows))
    def _has_any_metric(keys):
        return any(
            v.get(key) is not None and np.isfinite(v.get(key))
            for v in results
            for key in keys
        )

    def _print_temporal_table(label, prefix):
        if not _has_any_metric(
            [
                f"{prefix}{model}_{subset}_{metric}"
                for model in ("dl", "grasp")
                for subset in ("all", "top10", "top20")
                for metric in metric_names
            ]
        ):
            return
        print(label)
        temporal_headers = ["Subset", "Metric", "BRISKNet", "GRASP"]
        temporal_rows = []
        for subset in ("all", "top10", "top20"):
            for metric in metric_names:
                dl_key = f"{prefix}dl_{subset}_{metric}"
                grasp_key = f"{prefix}grasp_{subset}_{metric}"
                dl_val = _format_mean_std_compact(*_mean_std(results, dl_key))
                grasp_val = _format_mean_std_compact(*_mean_std(results, grasp_key))
                temporal_rows.append([subset, metric, dl_val, grasp_val])
        print(_render_table(temporal_headers, temporal_rows))

    print("----- Temporal Fidelity Metrics (mean ± std) -----")
    _print_temporal_table("Malignant", prefix="")
    _print_temporal_table("Benign", prefix="benign_")

    summaries_by_lam = {
        primary_grasp_lamda: {
            "dl_summary": dl_summary,
            "grasp_summary": grasp_summary,
            "raw_summary": raw_summary,
            "zf_summary": zf_summary,
        }
    }
    if len(grasp_lamdas) > 1:
        for grasp_lamda in grasp_lamdas:
            if grasp_lamda == primary_grasp_lamda:
                continue
            results_lam = results_by_lam[grasp_lamda]
            grasp_results_lam = grasp_results_by_lam[grasp_lamda]
            dl_sum, grasp_sum, raw_sum, zf_sum = _compute_summaries(
                results_lam, grasp_results_lam, raw_results, zf_results
            )
            summaries_by_lam[grasp_lamda] = {
                "dl_summary": dl_sum,
                "grasp_summary": grasp_sum,
                "raw_summary": raw_sum,
                "zf_summary": zf_sum,
            }
            suffix = f"_lam{grasp_lamda:g}"
            _write_metrics_csv(
                os.path.join(inference_dir, f"metrics{suffix}.csv"),
                results_lam,
                grasp_results_lam,
                raw_results,
                zf_results,
            )
            _write_temporal_metrics_csv(inference_dir, results_lam, metric_names, suffix)

    if args.store_logs:
        log_path = args.log_file
        if not os.path.isabs(log_path):
            log_path = os.path.join(REPO_ROOT, log_path)
        accel_factor = float(acceleration_val.item())
        seconds_per_frame = (
            float(args.total_scan_seconds) / float(N_time_eval - 1)
            if N_time_eval > 1 else float(args.total_scan_seconds)
        )
        sliding_window_used = bool(eval_uses_sliding_window)

        def _extract_mean_std(mean_std):
            if not mean_std:
                return {"mean": None, "std": None}
            mean, std = mean_std
            return {
                "mean": None if mean is None else float(mean),
                "std": None if std is None else float(std),
            }

        log_row = {
            "type": "BRISKNet",
            "exp_name": exp_name,
            "inference_dir": inference_dir,
            "spokes_per_frame": int(N_spokes_eval),
            "num_frames": int(N_time_eval),
            "acceleration": accel_factor,
            "seconds_per_frame": seconds_per_frame,
            "DRO_noise_level": val_noise_level,
            "grasp_lamdas": grasp_lamdas,
            "avg_inference_time": None if mean_infer is None else float(mean_infer),
            "std_inference_time": None if std_infer is None else float(std_infer),
            "num_samples": int(num_samples),
            "training_epochs": None if trained_epochs is None else int(trained_epochs),
            "spatial_metrics": {},
            "dc_metrics": {},
            "temporal_metrics": {},
        }

        grasp_agg_row = {
            "type": "GRASP",
            "spokes_per_frame": int(N_spokes_eval),
            "num_frames": int(N_time_eval),
            "acceleration": accel_factor,
            "seconds_per_frame": seconds_per_frame,
            "DRO_noise_level": val_noise_level,
            "grasp_lamda": primary_grasp_lamda,
            "num_samples": int(len(grasp_results)),
            "spatial_metrics": {},
            "dc_metrics": {},
            "temporal_metrics": {},
        }

        spatial_keys = ["ssim", "psnr", "mse", "lpips"]
        for metric in spatial_keys:
            dl_mean_std = _extract_mean_std(dl_summary.get(metric))
            grasp_mean_std = _extract_mean_std(grasp_summary.get(metric))
            log_row["spatial_metrics"][f"{metric}_mean"] = dl_mean_std["mean"]
            log_row["spatial_metrics"][f"{metric}_stddev"] = dl_mean_std["std"]
            grasp_agg_row["spatial_metrics"][f"{metric}_mean"] = grasp_mean_std["mean"]
            grasp_agg_row["spatial_metrics"][f"{metric}_stddev"] = grasp_mean_std["std"]

        dl_dc_mae = _extract_mean_std(dl_summary.get("dc_mae"))
        dl_dc_mse = _extract_mean_std(dl_summary.get("dc_mse"))
        grasp_dc_mae = _extract_mean_std(grasp_summary.get("dc_mae"))
        grasp_dc_mse = _extract_mean_std(grasp_summary.get("dc_mse"))
        raw_dc_mae = _extract_mean_std(raw_summary.get("raw_dc_mae"))
        raw_dc_mse = _extract_mean_std(raw_summary.get("raw_dc_mse"))
        raw_dc_psnr = _extract_mean_std(raw_summary.get("raw_dc_psnr"))
        raw_grasp_dc_mae = _extract_mean_std(raw_summary.get("raw_grasp_dc_mae"))
        raw_grasp_dc_mse = _extract_mean_std(raw_summary.get("raw_grasp_dc_mse"))
        raw_grasp_dc_psnr = _extract_mean_std(raw_summary.get("raw_grasp_dc_psnr"))
        raw_ssdu_nmse = _extract_mean_std(raw_summary.get("raw_ssdu_nmse"))
        raw_grasp_ssdu_nmse = _extract_mean_std(raw_summary.get("raw_grasp_ssdu_nmse"))

        log_row["dc_metrics"] = {
            "dro_dc_mae_mean": dl_dc_mae["mean"],
            "dro_dc_mae_stddev": dl_dc_mae["std"],
            "dro_dc_mse_mean": dl_dc_mse["mean"],
            "dro_dc_mse_stddev": dl_dc_mse["std"],
            "raw_dc_mae_mean": raw_dc_mae["mean"],
            "raw_dc_mae_stddev": raw_dc_mae["std"],
            "raw_dc_mse_mean": raw_dc_mse["mean"],
            "raw_dc_mse_stddev": raw_dc_mse["std"],
            "raw_dc_psnr_mean": raw_dc_psnr["mean"],
            "raw_dc_psnr_stddev": raw_dc_psnr["std"],
            "raw_ssdu_nmse_mean": raw_ssdu_nmse["mean"],
            "raw_ssdu_nmse_stddev": raw_ssdu_nmse["std"],
        }
        grasp_agg_row["dc_metrics"] = {
            "dro_dc_mae_mean": grasp_dc_mae["mean"],
            "dro_dc_mae_stddev": grasp_dc_mae["std"],
            "dro_dc_mse_mean": grasp_dc_mse["mean"],
            "dro_dc_mse_stddev": grasp_dc_mse["std"],
            "raw_dc_mae_mean": raw_grasp_dc_mae["mean"],
            "raw_dc_mae_stddev": raw_grasp_dc_mae["std"],
            "raw_dc_mse_mean": raw_grasp_dc_mse["mean"],
            "raw_dc_mse_stddev": raw_grasp_dc_mse["std"],
            "raw_dc_psnr_mean": raw_grasp_dc_psnr["mean"],
            "raw_dc_psnr_stddev": raw_grasp_dc_psnr["std"],
            "raw_grasp_ssdu_nmse_mean": raw_grasp_ssdu_nmse["mean"],
            "raw_grasp_ssdu_nmse_stddev": raw_grasp_ssdu_nmse["std"],
        }

        temporal_blocks = [
            ("all_pixels_malignant", "", "all"),
            ("all_pixels_benign", "benign_", "all"),
            ("top20_malignant", "", "top20"),
            ("top20_benign", "benign_", "top20"),
            ("top10_malignant", "", "top10"),
            ("top10_benign", "benign_", "top10"),
        ]
        for block_name, prefix, subset in temporal_blocks:
            log_row["temporal_metrics"][block_name] = {}
            grasp_agg_row["temporal_metrics"][block_name] = {}
            for metric in metric_names:
                dl_key = f"{prefix}dl_{subset}_{metric}"
                grasp_key = f"{prefix}grasp_{subset}_{metric}"
                dl_mean_std = _extract_mean_std(_mean_std(results, dl_key))
                grasp_mean_std = _extract_mean_std(_mean_std(results, grasp_key))
                log_row["temporal_metrics"][block_name][f"{metric}_mean"] = dl_mean_std["mean"]
                log_row["temporal_metrics"][block_name][f"{metric}_stddev"] = dl_mean_std["std"]
                grasp_agg_row["temporal_metrics"][block_name][f"{metric}_mean"] = grasp_mean_std["mean"]
                grasp_agg_row["temporal_metrics"][block_name][f"{metric}_stddev"] = grasp_mean_std["std"]

        existing_rows = []
        if os.path.exists(log_path):
            try:
                with open(log_path, "r") as f:
                    payload = json.load(f)
                if isinstance(payload, list):
                    existing_rows = payload
            except (json.JSONDecodeError, OSError) as exc:
                print(f"Warning: could not read {log_path}: {exc}")

        filtered_rows = []
        brisk_exists = False
        grasp_lamda_keys = {str(lam) for lam in grasp_lamdas}
        grasp_exists_by_lam = {key: False for key in grasp_lamda_keys}
        for row in existing_rows:
            row_type = row.get("type")
            if args.overwrite_logs:
                if row_type == "BRISKNet" and row.get("exp_name") == exp_name:
                    continue
                if row_type == "GRASP":
                    row_noise = row.get("DRO_noise_level") or row.get("dro_noise_level")
                    row_accel = row.get("acceleration") or row.get("acceleration_factor")
                    row_lam = row.get("grasp_lamda")
                    row_lam_key = str(row_lam) if row_lam is not None else None
                    if (
                        str(row.get("spokes_per_frame")) == str(int(N_spokes_eval))
                        and str(row.get("num_frames")) == str(int(N_time_eval))
                        and str(row_noise) == str(val_noise_level)
                        and str(row_accel) == str(accel_factor)
                        and row_lam_key in grasp_lamda_keys
                    ):
                        continue
                filtered_rows.append(row)
                continue

            if row_type == "BRISKNet" and row.get("exp_name") == exp_name:
                brisk_exists = True
            if row_type == "GRASP":
                row_lam_key = str(row.get("grasp_lamda")) if row.get("grasp_lamda") is not None else None
                if (
                    str(row.get("spokes_per_frame")) == str(int(N_spokes_eval))
                    and str(row.get("num_frames")) == str(int(N_time_eval))
                    and str(row.get("DRO_noise_level")) == str(val_noise_level)
                    and str(row.get("acceleration")) == str(accel_factor)
                    and row_lam_key in grasp_exists_by_lam
                ):
                    grasp_exists_by_lam[row_lam_key] = True
            filtered_rows.append(row)

        if args.overwrite_logs:
            filtered_rows.append(log_row)
        else:
            if not brisk_exists:
                filtered_rows.append(log_row)

        for grasp_lamda in grasp_lamdas:
            if grasp_lamda == primary_grasp_lamda:
                grasp_row = grasp_agg_row
                grasp_summary = summaries_by_lam[grasp_lamda]["grasp_summary"]
                dl_summary = summaries_by_lam[grasp_lamda]["dl_summary"]
                results_lam = results_by_lam[grasp_lamda]
            else:
                results_lam = results_by_lam[grasp_lamda]
                grasp_results_lam = grasp_results_by_lam[grasp_lamda]
                dl_summary, grasp_summary, _, _ = _compute_summaries(
                    results_lam, grasp_results_lam, raw_results, zf_results
                )
                grasp_row = {
                    "type": "GRASP",
                    "spokes_per_frame": int(N_spokes_eval),
                    "num_frames": int(N_time_eval),
                    "acceleration": accel_factor,
                    "seconds_per_frame": seconds_per_frame,
                    "DRO_noise_level": val_noise_level,
                    "grasp_lamda": grasp_lamda,
                    "num_samples": int(len(grasp_results_lam)),
                    "spatial_metrics": {},
                    "dc_metrics": {},
                    "temporal_metrics": {},
                }
                for metric in spatial_keys:
                    grasp_mean_std = _extract_mean_std(grasp_summary.get(metric))
                    grasp_row["spatial_metrics"][f"{metric}_mean"] = grasp_mean_std["mean"]
                    grasp_row["spatial_metrics"][f"{metric}_stddev"] = grasp_mean_std["std"]

                grasp_dc_mae = _extract_mean_std(grasp_summary.get("dc_mae"))
                grasp_dc_mse = _extract_mean_std(grasp_summary.get("dc_mse"))
                raw_grasp_dc_mae = _extract_mean_std(raw_summary.get("raw_grasp_dc_mae"))
                raw_grasp_dc_mse = _extract_mean_std(raw_summary.get("raw_grasp_dc_mse"))
                raw_grasp_dc_psnr = _extract_mean_std(raw_summary.get("raw_grasp_dc_psnr"))
                raw_grasp_ssdu_nmse = _extract_mean_std(raw_summary.get("raw_grasp_ssdu_nmse"))
                grasp_row["dc_metrics"] = {
                    "dro_dc_mae_mean": grasp_dc_mae["mean"],
                    "dro_dc_mae_stddev": grasp_dc_mae["std"],
                    "dro_dc_mse_mean": grasp_dc_mse["mean"],
                    "dro_dc_mse_stddev": grasp_dc_mse["std"],
                    "raw_dc_mae_mean": raw_grasp_dc_mae["mean"],
                    "raw_dc_mae_stddev": raw_grasp_dc_mae["std"],
                    "raw_dc_mse_mean": raw_grasp_dc_mse["mean"],
                    "raw_dc_mse_stddev": raw_grasp_dc_mse["std"],
                    "raw_dc_psnr_mean": raw_grasp_dc_psnr["mean"],
                    "raw_dc_psnr_stddev": raw_grasp_dc_psnr["std"],
                    "raw_grasp_ssdu_nmse_mean": raw_grasp_ssdu_nmse["mean"],
                    "raw_grasp_ssdu_nmse_stddev": raw_grasp_ssdu_nmse["std"],
                }

                temporal_blocks = [
                    ("all_pixels_malignant", "", "all"),
                    ("all_pixels_benign", "benign_", "all"),
                    ("top20_malignant", "", "top20"),
                    ("top20_benign", "benign_", "top20"),
                    ("top10_malignant", "", "top10"),
                    ("top10_benign", "benign_", "top10"),
                ]
                for block_name, prefix, subset in temporal_blocks:
                    grasp_row["temporal_metrics"][block_name] = {}
                    for metric in metric_names:
                        grasp_key = f"{prefix}grasp_{subset}_{metric}"
                        grasp_mean_std = _extract_mean_std(_mean_std(results_lam, grasp_key))
                        grasp_row["temporal_metrics"][block_name][f"{metric}_mean"] = grasp_mean_std["mean"]
                        grasp_row["temporal_metrics"][block_name][f"{metric}_stddev"] = grasp_mean_std["std"]

            lam_key = str(grasp_lamda)
            if args.overwrite_logs:
                filtered_rows.append(grasp_row)
            else:
                if not grasp_exists_by_lam.get(lam_key, False):
                    filtered_rows.append(grasp_row)

        with open(log_path, "w") as f:
            json.dump(filtered_rows, f, indent=2, sort_keys=False)

    print(f"Inference complete. Results saved to {inference_dir}")


if __name__ == "__main__":
    main()
