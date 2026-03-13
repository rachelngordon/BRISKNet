"""Spatial/temporal transforms for EI and augmentation. Run: imported by training scripts (not intended to run directly)."""

from typing import Union

import deepinv as dinv
import torch
import torch.nn.functional as F
from deepinv.transform import Transform



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

    Unlike time-resampling transforms, this does **not** resample in time (no
    interpolation). It applies an integer-frame shift with edge padding.

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
