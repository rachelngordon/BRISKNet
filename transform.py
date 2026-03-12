"""Spatial/temporal transforms for EI and augmentation. Run: imported by training scripts (not intended to run directly)."""

from typing import Union

import deepinv as dinv
import torch
import torch.nn.functional as F
from deepinv.transform import Transform
from einops import rearrange



# class VideoRotate(dinv.transform.Rotate):
#     """A Rotate transform that correctly handles 5D video tensors by flattening time into the batch dimension."""

#     def _transform(self, x: torch.Tensor, **params) -> torch.Tensor:
#         # First, check if we even need to flatten. If it's already 4D, just rotate.
#         if not self._check_x_5D(x):
#             return super()._transform(x, **params)

#         # It's a 5D video tensor. Flatten time into the batch dimension.
#         B = x.shape[0]
#         x_flat = dinv.physics.TimeMixin.flatten(x)  # (B, C, T, H, W) -> (B*T, C, H, W)

#         # The parent's _transform method can now work correctly on the 4D tensor (batch of 2D images).
#         # We need to get the right parameters for this new batch size.
#         # The `get_params` is usually called before `_transform`, so we should be okay.
#         # However, to be safe, let's pass a modified params dictionary.
#         flat_params = self.get_params(x_flat)

#         transformed_flat = super()._transform(x_flat, **flat_params)

#         # Unflatten to restore the original 5D video shape.
#         return dinv.physics.TimeMixin.unflatten(transformed_flat, batch_size=B)

class VideoRotate(Transform):
    r"""
    CORRECTED 2D Rotation for Videos (Handles deepinv composition).
    
    This class correctly applies a single, consistent random rotation to all frames of a video.
    It samples angles uniformly from a continuous range and is robust to being called
    from a deepinv composition operator that pre-flattens the video tensor.

    :param tuple[float, float] or float degrees: Range of degrees to select from.
        If degrees is a number instead of sequence like (min, max), the range of degrees
        will be (-degrees, +degrees).
    :param str interpolation_mode: "bilinear" or "nearest".
    :param bool constant_shape: if True, output has the same shape as the input.
    """

    def __init__(
        self,
        *args,
        degrees: Union[float, tuple[float, float]] = 180.0,
        interpolation_mode: str = "bilinear",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if isinstance(degrees, (int, float)):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be non-negative.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of length 2.")
            self.degrees = degrees
            
        self.interpolation_mode = interpolation_mode
        # This flag tells the deepinv TimeMixin decorator not to flatten input for us.
        # We will handle the 5D logic ourselves.
        self.flatten_video_input = False

    def _get_params(self, x: torch.Tensor) -> dict:
        """
        Uniformly samples `n_trans` random angles from the specified continuous range.
        """
        # NOTE: self.n_trans comes from the parent Transform class
        angles = [
            torch.empty(1).uniform_(self.degrees[0], self.degrees[1]).item()
            for _ in range(self.n_trans)
        ]
        return {"theta": angles}

    def _transform(
        self,
        x: torch.Tensor,
        theta: Union[torch.Tensor, list] = [],
        **kwargs,
    ) -> torch.Tensor:
        """
        Applies the rotation transformations. This method now explicitly handles 5D video tensors.
        """
        if not self._check_x_5D(x):
             raise ValueError("VideoRotate is designed for 5D video tensors (B, C, T, H, W).")

        B, C, T, H, W = x.shape
        if not theta:
            # Important: Get params using the original 5D tensor shape
            params = self._get_params(x)
            theta = params["theta"]

        # For video transforms, we assume n_trans=1 and use the first generated angle
        # to ensure the same rotation is applied to all frames.
        if not theta:
            raise ValueError("Rotation angle 'theta' not provided.")
        angle_for_video = theta[0]
        
        # Create affine matrix for the rotation
        angle_rad = -torch.tensor(angle_for_video) * (torch.pi / 180.0)
        cos_a, sin_a = torch.cos(angle_rad), torch.sin(angle_rad)

        self.last_angle = angle_for_video
        
        # Matrix for a single rotation. Shape: (1, 2, 3)
        matrix = torch.tensor(
            [[cos_a, -sin_a, 0], [sin_a, cos_a, 0]], 
            dtype=torch.float32, device=x.device
        ).unsqueeze(0)
        
        # Expand matrix to apply to the whole batch
        matrix = matrix.repeat(B, 1, 1)

        # Generate the sampling grid once for a single 4D image shape
        grid_single = F.affine_grid(matrix, (B, C, H, W), align_corners=False)
        
        # Apply this same grid to all frames by expanding it and flattening the input
        grid_expanded = grid_single.repeat_interleave(T, dim=0)
        x_flat = dinv.physics.TimeMixin.flatten(x)
        
        transformed_flat = F.grid_sample(x_flat, grid_expanded, mode=self.interpolation_mode, padding_mode='zeros', align_corners=False)
        
        return dinv.physics.TimeMixin.unflatten(transformed_flat, batch_size=B)


class VideoDiffeo(dinv.transform.CPABDiffeomorphism):
    """A Diffeomorphism transform that correctly handles 5D video tensors."""

    def _transform(self, x: torch.Tensor, **params) -> torch.Tensor:
        if not self._check_x_5D(x):
            return super()._transform(x, **params)

        B = x.shape[0]
        x_flat = dinv.physics.TimeMixin.flatten(x)
        flat_params = self.get_params(x_flat)
        transformed_flat = super()._transform(x_flat, **flat_params)
        return dinv.physics.TimeMixin.unflatten(transformed_flat, batch_size=B)


# class SubsampleTime(Transform):
#     r"""
#     Augments a video by taking a random contiguous temporal sub-sequence.
#     This is suitable for non-cyclical data like contrast enhancement curves,
#     as it preserves the local arrow of time.

#     :param int n_trans: Number of transformed versions to generate per input image.
#     :param float subsample_ratio: The ratio of the total time frames to keep (e.g., 0.8 for 80%).
#     :param torch.Generator rng: Random number generator.
#     """

#     def __init__(self, *args, subsample_ratio: float = 0.8, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.flatten_video_input = False  # We operate directly on the 5D tensor
#         assert 0.0 < subsample_ratio <= 1.0, "subsample_ratio must be between 0 and 1."
#         self.subsample_ratio = subsample_ratio

#     def _get_params(self, x: torch.Tensor) -> dict:
#         """Generates a random start index for the temporal crop."""
#         total_time_frames = x.shape[2]  # Shape is (B, C, T, H, W)
#         subsample_length = int(total_time_frames * self.subsample_ratio)
#         if subsample_length >= total_time_frames:
#             # Handle edge case where ratio is 1.0 or rounds up
#             return {"start_indices": torch.zeros(self.n_trans, dtype=torch.long)}

#         max_start_index = total_time_frames - subsample_length
#         start_indices = torch.randint(
#             low=0, high=max_start_index + 1, size=(self.n_trans,), generator=self.rng
#         )
#         return {"start_indices": start_indices}

#     def _transform(
#         self, x: torch.Tensor, start_indices: torch.Tensor, **kwargs
#     ) -> torch.Tensor:
#         """Performs the temporal subsampling and resizes back to the original length."""
#         B, C, total_time_frames, H, W = x.shape
#         subsample_length = int(total_time_frames * self.subsample_ratio)

#         if subsample_length >= total_time_frames:
#             return x.repeat(self.n_trans, 1, 1, 1, 1)

#         output_list = []
#         for start_idx in start_indices:
#             # 1. Take the temporal subsequence
#             sub_sequence = x[:, :, start_idx : start_idx + subsample_length, :, :]

#             # 2. Flatten all non-time dimensions into one giant "channel" dimension for interpolation.
#             # Pattern: (Batch, Channels, Time, Height, Width) -> (Batch, (Channels*Height*Width), Time)
#             flat_for_interp = rearrange(sub_sequence, "b c t h w -> b (c h w) t")

#             # 3. Interpolate along the time dimension (the last dimension).
#             # This is a 1D interpolation.
#             resized_flat = torch.nn.functional.interpolate(
#                 flat_for_interp,
#                 size=total_time_frames,
#                 mode="linear",
#                 align_corners=False,
#             )

#             # 4. Un-flatten the dimensions back to the original video format.
#             # Einops can do this because it knows how (c h w) was constructed.
#             # Pattern: (Batch, (Channels*Height*Width), Time) -> (Batch, Channels, Time, Height, Width)
#             resized_sequence = rearrange(
#                 resized_flat, "b (c h w) t -> b c t h w", c=C, h=H, w=W
#             )

#             output_list.append(resized_sequence)

#         return torch.cat(output_list, dim=0)
    

class SubsampleTime(Transform):
    r"""
    Augments a video by taking a random contiguous temporal sub-sequence of a
    RANDOM length, and then interpolating it back to the original length.

    :param int n_trans: Number of transformed versions to generate per input image.
    :param tuple[float, float] subsample_ratio_range: The min and max ratio of the
                                                     total time frames to keep (e.g., (0.7, 0.95)).
    :param torch.Generator rng: Random number generator.
    """

    def __init__(self, *args, subsample_ratio_range: tuple[float, float] = (0.7, 0.95), **kwargs):
        super().__init__(*args, **kwargs)
        self.flatten_video_input = False  # Operate on 5D tensor
        
        min_r, max_r = subsample_ratio_range
        assert 0.0 < min_r <= max_r <= 1.0, "subsample_ratio_range must be a valid range (min, max) between 0 and 1."
        self.subsample_ratio_range = subsample_ratio_range

    def _get_params(self, x: torch.Tensor) -> dict:
        """
        Generates a random ratio and a random start index for each transform.
        """
        total_time_frames = x.shape[2]  # Shape is (B, C, T, H, W)
        min_r, max_r = self.subsample_ratio_range
        
        # 1. Sample a random ratio for each of the `n_trans` transforms
        #    torch.rand generates in [0, 1), so we scale and shift to get [min_r, max_r)
        ratios = min_r + (max_r - min_r) * torch.rand(self.n_trans, generator=self.rng)

        start_indices = []
        for ratio in ratios:
            # 2. Calculate length and max start index based on THIS transform's ratio
            subsample_length = int(total_time_frames * ratio.item())
            if subsample_length >= total_time_frames:
                start_indices.append(0)
                continue

            max_start_index = total_time_frames - subsample_length
            start_idx = torch.randint(
                low=0, high=max_start_index + 1, size=(1,), generator=self.rng
            ).item()
            start_indices.append(start_idx)
            
        return {"ratios": ratios, "start_indices": torch.tensor(start_indices, dtype=torch.long)}

    def _transform(
        self, x: torch.Tensor, ratios: torch.Tensor, start_indices: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Performs the temporal subsampling and resizing for each requested transform."""
        B, C, total_time_frames, H, W = x.shape
        assert B == 1, "This transform implementation assumes a batch size of 1 for simplicity."

        output_list = []
        # We generate `n_trans` augmented versions from the single input `x`
        for i in range(self.n_trans):
            ratio = ratios[i]
            start_idx = start_indices[i]
            subsample_length = int(total_time_frames * ratio.item())

            if subsample_length >= total_time_frames:
                output_list.append(x.clone()) # Use clone to avoid issues
                continue

            # 1. Take the subsequence
            sub_sequence = x[:, :, start_idx : start_idx + subsample_length, :, :]

            # 2. Flatten for interpolation
            flat_for_interp = rearrange(sub_sequence, "b c t h w -> b (c h w) t")

            # 3. Interpolate back to original time dimension length
            resized_flat = torch.nn.functional.interpolate(
                flat_for_interp,
                size=total_time_frames,
                mode="linear",
                align_corners=False,
            )

            # 4. Un-flatten back to video format
            resized_sequence = rearrange(
                resized_flat, "b (c h w) t -> b c t h w", c=C, h=H, w=W
            )
            output_list.append(resized_sequence)

        return torch.cat(output_list, dim=0)
    


class PeakAwareBiPhasicWarp(Transform):
    r"""
    An enhancement-peak-aware temporal augmentation that warps BOTH the wash-in
    and wash-out phases independently with different random ratios.

    It finds the time of peak enhancement, splits the video, and then
    time-warps (compresses/stretches) both phases before reassembling them.

    :param tuple[float, float] warp_ratio_range: The min/max ratio for
                                                 compressing a phase. e.g., (0.6, 0.95).
    """

    def __init__(self, *args, warp_ratio_range: tuple[float, float] = (0.6, 0.95), **kwargs):
        super().__init__(*args, **kwargs)
        self.flatten_video_input = False
        min_r, max_r = warp_ratio_range
        assert 0.0 < min_r <= max_r <= 1.0, "warp_ratio_range must be valid."
        self.warp_ratio_range = warp_ratio_range

    def _get_params(self, x: torch.Tensor) -> dict:
        """
        Generate two independent warp ratios for each requested transform:
        one for wash-in and one for wash-out.
        """
        params_list = []
        min_r, max_r = self.warp_ratio_range
        
        for _ in range(self.n_trans):
            # Generate TWO independent random ratios from the specified range
            washin_ratio = min_r + (max_r - min_r) * torch.rand(1, generator=self.rng).item()
            washout_ratio = min_r + (max_r - min_r) * torch.rand(1, generator=self.rng).item()
            
            params_list.append({
                "washin_ratio": washin_ratio,
                "washout_ratio": washout_ratio
            })
        
        return {"params_list": params_list}

    def _transform(self, x: torch.Tensor, params_list: list[dict], **kwargs) -> torch.Tensor:
        """Applies the independent, bi-phasic warp."""
        assert x.shape[0] == 1, "This transform assumes a batch size of 1 for the input."

        # Option 1: Find a single, global peak index for simplicity
        mean_signal_curve = torch.mean(x, dim=(0, 1, 3, 4)) # Avg over B, C, H, W
        peak_idx = torch.argmax(mean_signal_curve)

        # Edge case: If peak is at the start/end, there's no wash-in/wash-out to warp.
        if peak_idx <= 0 or peak_idx >= x.shape[2] - 1:
            return x.repeat(self.n_trans, 1, 1, 1, 1)

        output_list = []
        for params in params_list:
            # 1. Split the video into three parts
            wash_in_phase = x[:, :, :peak_idx, :, :]
            peak_frame = x[:, :, peak_idx:peak_idx+1, :, :]
            wash_out_phase = x[:, :, peak_idx+1:, :, :]

            # 2. Warp BOTH phases using their respective independent ratios
            warped_wash_in = self._warp_phase(wash_in_phase, params["washin_ratio"])
            warped_wash_out = self._warp_phase(wash_out_phase, params["washout_ratio"])

            # 3. Reassemble the video
            new_x = torch.cat([warped_wash_in, peak_frame, warped_wash_out], dim=2)
            output_list.append(new_x)

        return torch.cat(output_list, dim=0)

    def _warp_phase(self, phase_tensor: torch.Tensor, ratio: float) -> torch.Tensor:
        """Helper function to subsample and interpolate a video phase."""
        B, C, T_phase, H, W = phase_tensor.shape
        if T_phase <= 1: # Cannot warp a single frame or empty tensor
            return phase_tensor

        subsample_len = max(1, int(T_phase * ratio)) # Ensure at least 1 frame

        # Reshape for interpolation
        flat_for_interp = rearrange(phase_tensor, "b c t h w -> b (c h w) t")
        
        # Subsample by interpolating down, then interpolate back up.
        # This creates a smooth warp effect.
        subsampled_flat = F.interpolate(flat_for_interp, size=subsample_len, mode='linear', align_corners=False)
        resized_flat = F.interpolate(subsampled_flat, size=T_phase, mode='linear', align_corners=False)

        # Reshape back to video format
        warped_phase = rearrange(resized_flat, "b (c h w) t -> b c t h w", c=C, h=H, w=W)
        return warped_phase
    

class MonophasicTimeWarp(Transform):
    r"""
    A temporal augmentation specifically designed for monophasic enhancement curves
    (e.g., persistent or plateau types) where there is no wash-out phase.

    This transform keeps the first (pre-contrast) frame fixed and applies a
    single, smooth time-warp to the entire subsequent enhancement phase.

    :param tuple[float, float] warp_ratio_range: The min/max ratio for
        compressing/stretching the enhancement phase. e.g., (0.7, 1.3).
        Values < 1 compress time, values > 1 stretch time.
    :param str pre_contrast_baseline: Baseline strategy for the pre-contrast frame.
        "first_frame" keeps t=0; "n_frames" uses the mean of the first
        round(0.1*T) frames, clamped to [4, 10]; "m_seconds" uses the mean
        of the first round(m / (150 / T)) frames.
    :param float baseline_seconds: Baseline window in seconds used when
        pre_contrast_baseline="m_seconds".
    :param int buffer_frames: Number of frames to freeze immediately after the
        pre-contrast frame (k=0 disables the buffer).
    """
    def __init__(
        self,
        *args,
        warp_ratio_range: tuple[float, float] = (0.7, 1.3),
        pre_contrast_baseline: str = "first_frame",
        warp_baseline: str | None = None,
        baseline_seconds: float = 20.0,
        buffer_frames: int = 0,
        start_mode: str = "baseline",
        arrival_percentile: float = 0.95,
        arrival_baseline_k: float = 2.0,
        arrival_method: str = "threshold",
        arrival_fraction: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.flatten_video_input = False
        min_r, max_r = warp_ratio_range
        assert 0.0 < min_r <= max_r, "warp_ratio_range must be a valid positive range."
        self.warp_ratio_range = warp_ratio_range
        if warp_baseline is not None:
            pre_contrast_baseline = warp_baseline
        self.pre_contrast_baseline = pre_contrast_baseline
        self.baseline_seconds = float(baseline_seconds)
        self.buffer_frames = max(0, int(buffer_frames))
        self.start_mode = (start_mode or "baseline").lower()
        self.arrival_percentile = float(arrival_percentile)
        self.arrival_baseline_k = float(arrival_baseline_k)
        self.arrival_method = (arrival_method or "threshold").lower()
        self.arrival_fraction = float(arrival_fraction)

    def _get_params(self, x: torch.Tensor) -> dict:
        """
        Generates a single random warp ratio for the entire enhancement phase.
        """
        min_r, max_r = self.warp_ratio_range
        # Generate one random ratio from the specified range for each transform requested.
        ratios = [min_r + (max_r - min_r) * torch.rand(1, generator=self.rng) for _ in range(self.n_trans)]
        
        return {"ratios": ratios}

    def _transform(self, x: torch.Tensor, ratios: list[float], **kwargs) -> torch.Tensor:
        """Applies the monophasic time warp."""
        assert x.shape[0] == 1, "This transform assumes a batch size of 1 for the input."
        if x.shape[2] <= 1: # Cannot warp if there's only one frame
             return x.repeat(self.n_trans, 1, 1, 1, 1)

        output_list = []
        for ratio in ratios:
            # 1. Isolate the pre-contrast frame and the enhancement phase
            if self.pre_contrast_baseline == "n_frames":
                T = x.shape[2]
                n_base = int(round(0.1 * T))
                n_base = max(4, min(10, n_base))
                n_base = min(n_base, T)
                pre_contrast_frame = x[:, :, :n_base, :, :].mean(dim=2, keepdim=True)
            elif self.pre_contrast_baseline == "m_seconds":
                T = x.shape[2]
                seconds_per_frame = 150.0 / max(T, 1)
                n_base = int(round(self.baseline_seconds / seconds_per_frame))
                n_base = max(1, min(n_base, T))
                pre_contrast_frame = x[:, :, :n_base, :, :].mean(dim=2, keepdim=True)
            else:
                pre_contrast_frame = x[:, :, :1, :, :]
            buffer_len = min(self.buffer_frames, max(0, x.shape[2] - 1))
            buffer_frames = x[:, :, 1:1 + buffer_len, :, :]
            enhancement_phase = x[:, :, 1 + buffer_len:, :, :]

            # 2. Warp the enhancement phase
            warped_enhancement_phase = self._warp_phase(enhancement_phase, ratio)

            # 3. Reassemble the video
            new_x = torch.cat([pre_contrast_frame, buffer_frames, warped_enhancement_phase], dim=2)
            output_list.append(new_x)

        return torch.cat(output_list, dim=0)

    def _warp_phase(self, phase_tensor: torch.Tensor, ratio: float) -> torch.Tensor:
        """Helper function to interpolate a video phase to a new length."""
        B, C, T_phase, H, W = phase_tensor.shape
        if T_phase == 0:
            return phase_tensor
        
        ratio = ratio.item()

        # New length is the original length scaled by the ratio.
        # This allows for both compression (ratio < 1) and stretching (ratio > 1).
        new_length = int(round(T_phase * ratio))
        if new_length == 0: new_length = 1 # Ensure at least one frame

        # Reshape for interpolation
        flat_for_interp = rearrange(phase_tensor, "b c t h w -> b (c h w) t")
        
        # Interpolate to the new length
        resized_flat = F.interpolate(flat_for_interp, size=new_length, mode='linear', align_corners=False)

        # If the length changed, we need to interpolate back to the original length
        if new_length != T_phase:
            resized_flat = F.interpolate(resized_flat, size=T_phase, mode='linear', align_corners=False)

        # Reshape back to video format
        warped_phase = rearrange(resized_flat, "b (c h w) t -> b c t h w", c=C, h=H, w=W)
        return warped_phase


class TemporalShiftJitterAfterBaseline(Transform):
    r"""
    Applies a discrete temporal shift to the enhancement phase only, keeping the
    pre-contrast baseline fixed. The enhancement phase is edge-padded by repeating
    the first/last enhancement frame to preserve length and intensity scale.

    :param int max_shift: Maximum absolute frame shift (in frames). The shift is
        sampled uniformly from [-max_shift, max_shift].
    :param str pre_contrast_baseline: Baseline strategy for the pre-contrast frame.
        "first_frame" keeps t=0; "n_frames" uses the mean of the first
        round(0.1*T) frames, clamped to [4, 10]; "m_seconds" uses the mean
        of the first round(m / (150 / T)) frames.
    :param float baseline_seconds: Baseline window in seconds used when
        pre_contrast_baseline="m_seconds".
    :param int buffer_frames: Number of frames to freeze immediately after the
        pre-contrast frame (k=0 disables the buffer).
    """
    def __init__(
        self,
        *args,
        max_shift: int = 2,
        pre_contrast_baseline: str = "first_frame",
        baseline_seconds: float = 20.0,
        buffer_frames: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.flatten_video_input = False
        self.max_shift = max(0, int(max_shift))
        self.pre_contrast_baseline = pre_contrast_baseline
        self.baseline_seconds = float(baseline_seconds)
        self.buffer_frames = max(0, int(buffer_frames))

    def _get_params(self, x: torch.Tensor) -> dict:
        if self.max_shift == 0:
            shifts = [torch.tensor(0, device=x.device) for _ in range(self.n_trans)]
        else:
            shifts = [
                torch.randint(
                    low=-self.max_shift,
                    high=self.max_shift + 1,
                    size=(1,),
                    generator=self.rng,
                    device="cpu",
                )[0].to(x.device)
                for _ in range(self.n_trans)
            ]
        return {"shifts": shifts}

    def _transform(self, x: torch.Tensor, shifts: list[torch.Tensor], **kwargs) -> torch.Tensor:
        assert x.shape[0] == 1, "This transform assumes a batch size of 1 for the input."
        if x.shape[2] <= 1:
            return x.repeat(self.n_trans, 1, 1, 1, 1)

        T = x.shape[2]
        if self.pre_contrast_baseline == "n_frames":
            n_base = int(round(0.1 * T))
            n_base = max(4, min(10, n_base))
            n_base = min(n_base, T)
        elif self.pre_contrast_baseline == "m_seconds":
            seconds_per_frame = 150.0 / max(T, 1)
            n_base = int(round(self.baseline_seconds / seconds_per_frame))
            n_base = max(1, min(n_base, T))
        else:
            n_base = 1

        if T - n_base <= 1:
            return x.repeat(self.n_trans, 1, 1, 1, 1)

        buffer_len = min(self.buffer_frames, max(0, T - n_base))
        baseline = x[:, :, :n_base, :, :]
        buffer_frames = x[:, :, n_base:n_base + buffer_len, :, :]
        enh = x[:, :, n_base + buffer_len:, :, :]

        if enh.shape[2] <= 1:
            return x.repeat(self.n_trans, 1, 1, 1, 1)

        output_list = []
        for shift in shifts:
            shift_val = int(shift.item())
            if shift_val > 0:
                tail = enh[:, :, -1:, :, :].repeat(1, 1, shift_val, 1, 1)
                enh_shifted = torch.cat([enh[:, :, shift_val:, :, :], tail], dim=2)
            elif shift_val < 0:
                pad = enh[:, :, :1, :, :].repeat(1, 1, -shift_val, 1, 1)
                enh_shifted = torch.cat([pad, enh[:, :, :shift_val, :, :]], dim=2)
            else:
                enh_shifted = enh

            new_x = torch.cat([baseline, buffer_frames, enh_shifted], dim=2)
            output_list.append(new_x)

        return torch.cat(output_list, dim=0)


class TemporalNoise(Transform):
    """ 
    Adds low-frequency random noise to the temporal signal of a video.
    This simulates smooth, slowly varying noise sources over time.
    """
    def __init__(self, *args, noise_strength: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.flatten_video_input = False
        self.noise_strength = noise_strength

    def _get_params(self, x: torch.Tensor) -> dict:
        """
        Generates a single low-frequency noise vector for the transformation.
        """
        B, C, T, H, W = x.shape
        
        # Ensure there's at least one point in the low-res vector.
        low_res_T = max(1, T // 4) 
        
        # Create a 3D tensor in the format (Batch, Channels, Length)
        noise_low_res = torch.randn(B, 1, low_res_T, device=x.device)
        
        # --- CORRECTED LINE ---
        # Interpolate the 3D tensor directly. `size` refers to the target length.
        noise_high_res = F.interpolate(noise_low_res, size=T, mode='linear', align_corners=False)
        # The output shape is now (B, 1, T)
        
        # Normalize the noise to have zero mean and unit variance
        noise_norm = (noise_high_res - noise_high_res.mean(dim=-1, keepdim=True)) / (noise_high_res.std(dim=-1, keepdim=True) + 1e-8)
        
        # Scale by the desired strength
        final_noise = noise_norm * self.noise_strength
        
        # Return the noise in a dictionary, as required by deepinv
        return {'noise': final_noise}

    def _transform(self, x: torch.Tensor, noise: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Applies the pre-generated noise to the image tensor.
        """
        B, C, T, H, W = x.shape
        
        # Reshape noise to be broadcastable and add it to the image
        # noise shape: (B, 1, T) -> (B, 1, T, 1, 1)
        x_noisy = x + noise.view(B, 1, T, 1, 1)
        
        return x_noisy
    

class TimeReverse(Transform):
    r"""
    Reverses the temporal order of frames in a video tensor.

    This transform flips the video along the time axis, effectively playing it
    backwards. This is a deterministic transformation.

    :param int n_trans: Number of transformed versions to generate per input image.
                        Since this is deterministic, it will just repeat the same
                        output if n_trans > 1.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # We need the 5D video tensor (B, C, T, H, W) to access the time dimension.
        self.flatten_video_input = False

    def _get_params(self, x: torch.Tensor) -> dict:
        """
        No random parameters are needed for time reversal as it's a
        deterministic operation.
        """
        return {}

    def _transform(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Applies the time reversal transformation using torch.flip().
        """
        # Ensure we are working with a 5D tensor
        if len(x.shape) != 5:
            raise ValueError(f"TimeReverse expects a 5D tensor (B, C, T, H, W), but got shape {x.shape}.")

        # The core operation: flip along the time dimension (dim=2)
        # B, C, T, H, W
        # 0, 1, 2, 3, 4
        return torch.flip(x, dims=[2])


def estimate_bolus_arrival_index(
    x: torch.Tensor,
    percentile: float = 0.95,
    baseline_k: float = 2.0,
    arrival_method: str = "threshold",
    arrival_fraction: float = 0.1,
    pre_contrast_baseline: str = "n_frames",
    baseline_seconds: float = 20.0,
    total_seconds: float = 150.0,
) -> int:
    """
    Returns an integer arrival index in [0, T-1] estimated from a robust
    global curve (top-percentile intensity). This matches the logic used by
    BolusArrivalTimeShift.
    """
    if x.dim() != 5:
        raise ValueError(f"estimate_bolus_arrival_index expects 5D tensor, got {x.shape}.")
    B, C, T, H, W = x.shape
    if T <= 1:
        return 0

    def _baseline_len(num_frames: int) -> int:
        if pre_contrast_baseline == "n_frames":
            n_base = int(round(0.1 * num_frames))
            n_base = max(4, min(10, n_base))
        elif pre_contrast_baseline == "m_seconds":
            seconds_per_frame = total_seconds / max(num_frames, 1)
            n_base = int(round(baseline_seconds / max(seconds_per_frame, 1e-6)))
            n_base = max(1, n_base)
        else:
            n_base = 1
        return min(n_base, max(num_frames, 1))

    mag = torch.sqrt(x[:, 0, ...] ** 2 + x[:, 1, ...] ** 2 + 1e-8)  # (B, T, H, W)
    flat = mag.reshape(B, T, -1)
    q = float(percentile)
    q = min(max(q, 0.0), 1.0)

    thr = torch.quantile(flat, q, dim=-1, keepdim=True)  # (B, T, 1)
    mask = flat > thr
    bright_sum = (flat * mask).sum(dim=-1)
    bright_count = mask.sum(dim=-1)
    curve = torch.where(
        bright_count > 0,
        bright_sum / bright_count.clamp_min(1),
        thr.squeeze(-1),
    )  # (B, T)

    n_base = _baseline_len(T)
    baseline = curve[:, :n_base]
    mu = baseline.mean(dim=1)
    sigma = baseline.std(dim=1, unbiased=False)
    method = (arrival_method or "threshold").lower()
    if method in ("fraction", "fraction_of_peak", "fop"):
        peak = curve.max(dim=1).values
        frac = float(arrival_fraction)
        frac = max(0.0, min(1.0, frac))
        thr_curve = mu + frac * (peak - mu)
    else:
        thr_curve = mu + float(baseline_k) * sigma

    # Enforce arrival after baseline window.
    search_start = min(n_base, max(T - 1, 0))
    above = curve[0] > thr_curve[0]
    if search_start > 0:
        above = torch.cat(
            [
                torch.zeros(search_start, dtype=torch.bool, device=above.device),
                above[search_start:],
            ],
            dim=0,
        )
    if torch.any(above):
        return int(torch.argmax(above.int()).item())

    # Fallback: max derivative index (wash-in onset proxy)
    d = curve[0, 1:] - curve[0, :-1]
    if d.numel() == 0:
        return int(min(search_start, max(T - 1, 0)))
    if search_start > 0:
        d = d[search_start - 1 :]
        if d.numel() == 0:
            return int(min(search_start, max(T - 1, 0)))
        return int(torch.argmax(d).item() + search_start)
    return int(torch.argmax(d).item() + 1)


class BolusArrivalTimeShift(Transform):
    r"""
    Breast DCE temporal augmentation: jitter the sequence in time, but *anchored* to
    an estimated bolus-arrival (contrast arrival) index within the current window.

    Unlike MonophasicTimeWarp/SubsampleTime, this does **not** resample in time
    (no interpolation). It applies an integer-frame shift with edge padding.

    This is meant to model nuisance misalignment between acquisition window start
    and contrast arrival, without changing the enhancement curve shape.

    Notes:
      - Expects 5D video tensor (B, C, T, H, W) with C=2 (real/imag).
      - Assumes batch size B==1 (matches current training setup).

    Params:
      max_shift: maximum absolute jitter (frames) around the detected arrival.
      percentile: quantile (0..1) used to build a robust global curve from bright tissue.
      baseline_k: threshold is baseline_mean + baseline_k * baseline_std.
      pre_contrast_baseline/baseline_seconds: same conventions as other temporal transforms.
    """

    def __init__(
        self,
        *args,
        max_shift: int = 2,
        percentile: float = 0.95,
        baseline_k: float = 2.0,
        arrival_method: str = "threshold",
        arrival_fraction: float = 0.1,
        pre_contrast_baseline: str = "n_frames",
        baseline_seconds: float = 20.0,
        total_seconds: float = 150.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.flatten_video_input = False
        self.max_shift = max(0, int(max_shift))
        self.percentile = float(percentile)
        self.baseline_k = float(baseline_k)
        self.arrival_method = (arrival_method or "threshold").lower()
        self.arrival_fraction = float(arrival_fraction)
        self.pre_contrast_baseline = pre_contrast_baseline
        self.baseline_seconds = float(baseline_seconds)
        self.total_seconds = float(total_seconds)

    def _baseline_len(self, T: int) -> int:
        if self.pre_contrast_baseline == "n_frames":
            n_base = int(round(0.1 * T))
            n_base = max(4, min(10, n_base))
        elif self.pre_contrast_baseline == "m_seconds":
            seconds_per_frame = self.total_seconds / max(T, 1)
            n_base = int(round(self.baseline_seconds / max(seconds_per_frame, 1e-6)))
            n_base = max(1, n_base)
        else:
            n_base = 1
        return min(n_base, max(T, 1))

    @staticmethod
    def _magnitude(x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, T, H, W) -> mag: (B, T, H, W)
        return torch.sqrt(x[:, 0, ...] ** 2 + x[:, 1, ...] ** 2 + 1e-8)

    def _estimate_arrival_index(self, x: torch.Tensor) -> int:
        return estimate_bolus_arrival_index(
            x,
            percentile=self.percentile,
            baseline_k=self.baseline_k,
            arrival_method=self.arrival_method,
            arrival_fraction=self.arrival_fraction,
            pre_contrast_baseline=self.pre_contrast_baseline,
            baseline_seconds=self.baseline_seconds,
            total_seconds=self.total_seconds,
        )

    def _get_params(self, x: torch.Tensor) -> dict:
        if self.max_shift == 0:
            shifts = [0 for _ in range(self.n_trans)]
            return {"shifts": shifts}

        arrival_idx = self._estimate_arrival_index(x)
        shifts = []
        for _ in range(self.n_trans):
            delta = int(
                torch.randint(
                    low=-self.max_shift,
                    high=self.max_shift + 1,
                    size=(1,),
                    generator=self.rng,
                ).item()
            )
            # Clamp the target index to valid range, then convert to an actual shift.
            T = x.shape[2]
            target = min(max(arrival_idx + delta, 0), max(T - 1, 0))
            shifts.append(int(target - arrival_idx))
        return {"shifts": shifts}

    def _transform(self, x: torch.Tensor, shifts: list[int], **kwargs) -> torch.Tensor:
        if len(x.shape) != 5:
            raise ValueError(f"BolusArrivalTimeShift expects 5D tensor, got {x.shape}.")
        assert x.shape[0] == 1, "This transform assumes batch size 1."

        B, C, T, H, W = x.shape
        if T <= 1:
            return x.repeat(self.n_trans, 1, 1, 1, 1)

        out = []
        for shift in shifts:
            s = int(shift)
            if s == 0:
                out.append(x)
                continue
            if s > 0:
                pad = x[:, :, :1, :, :].repeat(1, 1, s, 1, 1)
                out.append(torch.cat([pad, x[:, :, : T - s, :, :]], dim=2))
            else:
                s_abs = -s
                pad = x[:, :, -1:, :, :].repeat(1, 1, s_abs, 1, 1)
                out.append(torch.cat([x[:, :, s_abs:, :, :], pad], dim=2))
        return torch.cat(out, dim=0)


class BaselineEnhancementScale(Transform):
    r"""
    Breast DCE augmentation: scale *enhancement above baseline* by a positive scalar.

    x'(t) = B + a * (x(t) - B)  for t >= baseline_end

    This preserves temporal kinetics shape (slopes/peak timing) and models nuisance
    gain/dose/B1 variation more plausibly than time resampling.

    Expects 5D tensor (B, C, T, H, W) with C=2 (real/imag) and batch size B==1.
    """

    def __init__(
        self,
        *args,
        scale_range: tuple[float, float] = (0.8, 1.2),
        pre_contrast_baseline: str = "first_frame",
        baseline_seconds: float = 20.0,
        total_seconds: float = 150.0,
        buffer_frames: int = 0,
        start_mode: str = "baseline",
        arrival_percentile: float = 0.95,
        arrival_baseline_k: float = 2.0,
        arrival_method: str = "threshold",
        arrival_fraction: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.flatten_video_input = False
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))
        self.pre_contrast_baseline = pre_contrast_baseline
        self.baseline_seconds = float(baseline_seconds)
        self.total_seconds = float(total_seconds)
        self.buffer_frames = max(0, int(buffer_frames))
        self.start_mode = (start_mode or "baseline").lower()
        self.arrival_percentile = float(arrival_percentile)
        self.arrival_baseline_k = float(arrival_baseline_k)
        self.arrival_method = (arrival_method or "threshold").lower()
        self.arrival_fraction = float(arrival_fraction)

    def _baseline_len(self, T: int) -> int:
        if self.pre_contrast_baseline == "n_frames":
            n_base = int(round(0.1 * T))
            n_base = max(4, min(10, n_base))
        elif self.pre_contrast_baseline == "m_seconds":
            seconds_per_frame = self.total_seconds / max(T, 1)
            n_base = int(round(self.baseline_seconds / max(seconds_per_frame, 1e-6)))
            n_base = max(1, n_base)
        else:
            n_base = 1
        return min(n_base, max(T, 1))

    def _get_params(self, x: torch.Tensor) -> dict:
        lo, hi = self.scale_range
        lo, hi = (min(lo, hi), max(lo, hi))
        scales = [float(lo + (hi - lo) * torch.rand(1, generator=self.rng).item()) for _ in range(self.n_trans)]
        return {"scales": scales}

    def _transform(self, x: torch.Tensor, scales: list[float], **kwargs) -> torch.Tensor:
        if len(x.shape) != 5:
            raise ValueError(f"BaselineEnhancementScale expects 5D tensor, got {x.shape}.")
        assert x.shape[0] == 1, "This transform assumes batch size 1."

        B, C, T, H, W = x.shape
        if T <= 1:
            return x.repeat(self.n_trans, 1, 1, 1, 1)

        n_base = self._baseline_len(T)
        buffer_len = min(self.buffer_frames, max(0, T - n_base))
        if self.start_mode == "arrival":
            arrival_idx = estimate_bolus_arrival_index(
                x,
                percentile=self.arrival_percentile,
                baseline_k=self.arrival_baseline_k,
                arrival_method=self.arrival_method,
                arrival_fraction=self.arrival_fraction,
                pre_contrast_baseline=self.pre_contrast_baseline,
                baseline_seconds=self.baseline_seconds,
                total_seconds=self.total_seconds,
            )
            enh_start = int(arrival_idx) + buffer_len
        else:
            enh_start = n_base + buffer_len
        enh_start = max(0, min(enh_start, T))

        # Baseline image (complex, per-pixel)
        baseline = x[:, :, :n_base, :, :].mean(dim=2, keepdim=True)  # (1, C, 1, H, W)

        out = []
        for a in scales:
            a_t = x.new_tensor(a)
            if enh_start < T:
                pre = x[:, :, :enh_start, :, :]
                enh = x[:, :, enh_start:, :, :]
                scaled_enh = baseline + a_t * (enh - baseline)
                x_new = torch.cat([pre, scaled_enh], dim=2)
            else:
                x_new = x.clone()
            out.append(x_new)
        return torch.cat(out, dim=0)
