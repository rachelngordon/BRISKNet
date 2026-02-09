from __future__ import annotations

import re
from typing import Optional, Union

import torch
import torch.nn.functional as F
from deepinv.loss.loss import Loss
from deepinv.loss.metric.metric import Metric

from radial_lsfp import MCNUFFT


class RebinConsistencyLoss(Loss):
    r"""
    Operator-level (binning) consistency loss for radial dynamic MRI with optional
    teacher-student stop-grad, offset rebins, dynamic masking, and temporal targets.

    Given a high-temporal reconstruction x_hi from measurements y_hi with spf_hi spokes/frame,
    form a lower-temporal measurement y_lo by *rebinning* groups of consecutive frames
    (spf_lo = spf_hi * factor), reconstruct x_lo, then enforce:

        x_lo ~= Avg_factor(x_hi)

    This is "DDEI-ish" in the sense that it enforces consistency across a transformed
    measurement operator (different binning), rather than imposing arbitrary image-space
    time warps that change kinetics.

    Assumptions:
      - y_hi has shape (coils, spf_hi * N_samples, T_hi) complex
      - physics is an MCNUFFT with matching (ktraj, dcomp) for y_hi
      - x_net is the model output for y_hi with shape (B, 2, H, W, T_hi)
      - batch size B==1 (matches current training setup)
    """

    def __init__(
        self,
        factor: int = 2,
        metric: Union[Metric, torch.nn.Module] = torch.nn.MSELoss(),
        time_index_mode: str = "none",
        teacher_branch: str = "none",
        teacher_stopgrad: bool = False,
        offset_mode: str = "none",
        temporal_mode: str = "absolute",
        baseline_frames: int = 4,
        percent_enhancement_eps: float = 1e-4,
        dynamic_mask_enable: bool = False,
        dynamic_mask_fraction: float = 0.01,
        dynamic_mask_min_pixels: int = 256,
        dynamic_mask_warmup_epochs: int = 0,
        dynamic_mask_smooth_kernel: int = 0,
        dynamic_mask_clip_min: float = 0.0,
        dynamic_mask_clip_max: float = 1.0,
        dynamic_mask_stop_grad: bool = True,
    ):
        super().__init__()
        self.name = "rebin_consistency"
        self.factor = max(1, int(factor))
        self.metric = metric
        self.time_index_mode = str(time_index_mode).lower()
        self.teacher_branch = str(teacher_branch).lower()
        self.teacher_stopgrad = bool(teacher_stopgrad)
        self.offset_mode = str(offset_mode).lower()
        self.temporal_mode = str(temporal_mode).lower()
        self.baseline_frames = max(1, int(baseline_frames))
        self.percent_enhancement_eps = float(percent_enhancement_eps)

        self.dynamic_mask_enable = bool(dynamic_mask_enable)
        self.dynamic_mask_fraction = float(dynamic_mask_fraction)
        # Deprecated: kept only for backward-compatible config parsing.
        self.dynamic_mask_min_pixels = int(dynamic_mask_min_pixels)
        self.dynamic_mask_warmup_epochs = max(0, int(dynamic_mask_warmup_epochs))
        self.dynamic_mask_smooth_kernel = max(0, int(dynamic_mask_smooth_kernel))
        self.dynamic_mask_clip_min = float(dynamic_mask_clip_min)
        self.dynamic_mask_clip_max = float(dynamic_mask_clip_max)
        self.dynamic_mask_stop_grad = bool(dynamic_mask_stop_grad)

        if self.time_index_mode not in {"none", "inherit", "scaled"}:
            raise ValueError(
                f"Unsupported time_index_mode '{time_index_mode}'. Expected one of: none, inherit, scaled."
            )
        if self.teacher_branch not in {"none", "lo", "hi"}:
            raise ValueError(
                f"Unsupported teacher_branch '{teacher_branch}'. Expected one of: none, lo, hi."
            )
        if self.offset_mode not in {"none", "fixed", "random", "average"}:
            raise ValueError(
                f"Unsupported offset_mode '{offset_mode}'. Expected one of: none, fixed, random, average."
            )
        if self.temporal_mode not in {
            "absolute",
            "baseline_subtracted",
            "percent_enhancement",
            "temporal_difference",
        }:
            raise ValueError(
                "Unsupported temporal_mode "
                f"'{temporal_mode}'. Expected one of: absolute, baseline_subtracted, "
                "percent_enhancement, temporal_difference."
            )
        if not (0.0 < self.dynamic_mask_fraction <= 1.0):
            raise ValueError("dynamic_mask_fraction must be in (0, 1].")
        if self.dynamic_mask_clip_max < self.dynamic_mask_clip_min:
            raise ValueError("dynamic_mask_clip_max must be >= dynamic_mask_clip_min.")
        if self.dynamic_mask_smooth_kernel > 1 and self.dynamic_mask_smooth_kernel % 2 == 0:
            self.dynamic_mask_smooth_kernel += 1

    @staticmethod
    def _zero_like_loss_ref(ref: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), device=ref.device, dtype=ref.dtype)

    @staticmethod
    def _validate_offset(offset: int, factor: int) -> int:
        if factor <= 1:
            return 0
        offset = int(offset)
        if offset < 0 or offset >= factor:
            raise ValueError(f"Offset must satisfy 0 <= offset < factor. Got offset={offset}, factor={factor}.")
        return offset

    @staticmethod
    def _downsample_time_mean(x: torch.Tensor, factor: int, offset: int = 0) -> torch.Tensor:
        # x: (B, 2, H, W, T_hi) -> (B, 2, H, W, T_lo) via mean over groups.
        if factor <= 1:
            return x
        offset = RebinConsistencyLoss._validate_offset(offset, factor)
        if offset > 0:
            x = x[..., offset:]
        B, C, H, W, T = x.shape
        T_lo = T // factor
        if T_lo <= 0:
            return x[:, :, :, :, :0]
        x_cropped = x[:, :, :, :, : T_lo * factor]
        x_grouped = x_cropped.view(B, C, H, W, T_lo, factor)
        return x_grouped.mean(dim=-1)

    @staticmethod
    def _rebin_kspace(
        y: torch.Tensor, spf_hi: int, samples_per_spoke: int, factor: int, offset: int = 0
    ) -> torch.Tensor:
        # y: (coils, spf_hi * samples, T_hi) -> (coils, (spf_hi*factor)*samples, T_lo)
        if factor <= 1:
            return y
        offset = RebinConsistencyLoss._validate_offset(offset, factor)
        coils, sp_samp, T = y.shape
        expected = spf_hi * samples_per_spoke
        if sp_samp != expected:
            raise ValueError(f"Unexpected k-space shape: got {sp_samp}, expected {expected}.")

        if offset > 0:
            y = y[..., offset:]
            T = y.shape[-1]

        T_lo = T // factor
        if T_lo <= 0:
            return y[:, :, :0]
        y_cropped = y[:, :, : T_lo * factor]
        y_rs = y_cropped.reshape(coils, spf_hi, samples_per_spoke, T_lo, factor)
        # move factor into spokes axis
        y_rs = y_rs.permute(0, 4, 1, 2, 3).reshape(
            coils, spf_hi * factor, samples_per_spoke, T_lo
        )
        return y_rs.reshape(coils, spf_hi * factor * samples_per_spoke, T_lo)

    @staticmethod
    def _rebin_ktraj_dcomp(
        physics: MCNUFFT, spf_hi: int, samples_per_spoke: int, factor: int, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ktraj: (2, spf_hi * samples, T_hi) -> (2, (spf_hi*factor)*samples, T_lo)
        # dcomp: (spf_hi * samples, T_hi) -> ((spf_hi*factor)*samples, T_lo)
        if factor <= 1:
            return physics.ktraj, physics.dcomp
        offset = RebinConsistencyLoss._validate_offset(offset, factor)

        ktraj = physics.ktraj
        dcomp = physics.dcomp
        _, sp_samp, T = ktraj.shape
        expected = spf_hi * samples_per_spoke
        if sp_samp != expected:
            raise ValueError(f"Unexpected ktraj shape: got {sp_samp}, expected {expected}.")

        if offset > 0:
            ktraj = ktraj[..., offset:]
            dcomp = dcomp[..., offset:]
            T = ktraj.shape[-1]

        T_lo = T // factor
        if T_lo <= 0:
            return ktraj[:, :, :0], dcomp[:, :0]

        ktraj_c = ktraj[:, :, : T_lo * factor].reshape(2, spf_hi, samples_per_spoke, T_lo, factor)
        ktraj_c = ktraj_c.permute(0, 4, 1, 2, 3).reshape(
            2, spf_hi * factor, samples_per_spoke, T_lo
        )
        ktraj_lo = ktraj_c.reshape(2, spf_hi * factor * samples_per_spoke, T_lo)

        dcomp_c = dcomp[:, : T_lo * factor].reshape(spf_hi, samples_per_spoke, T_lo, factor)
        dcomp_c = dcomp_c.permute(3, 0, 1, 2).reshape(spf_hi * factor, samples_per_spoke, T_lo)
        dcomp_lo = dcomp_c.reshape(spf_hi * factor * samples_per_spoke, T_lo)

        return ktraj_lo, dcomp_lo

    @staticmethod
    def _to_magnitude(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        # x: (B,2,H,W,T) -> (B,1,H,W,T)
        return torch.sqrt(x[:, :1] ** 2 + x[:, 1:2] ** 2 + eps)

    @staticmethod
    def _parse_epoch(epoch: object) -> Optional[int]:
        if epoch is None:
            return None
        if isinstance(epoch, int):
            return epoch
        if isinstance(epoch, float):
            return int(epoch)
        if torch.is_tensor(epoch):
            if epoch.numel() == 0:
                return None
            return int(epoch.reshape(-1)[0].item())
        if isinstance(epoch, str):
            match = re.search(r"-?\d+", epoch)
            if match is not None:
                return int(match.group(0))
        return None

    def _select_offsets(self, ref: torch.Tensor) -> list[int]:
        if self.factor <= 1:
            return [0]
        if self.offset_mode in {"none", "fixed"}:
            return [0]
        if self.offset_mode == "random":
            offset = int(torch.randint(0, self.factor, (1,), device=ref.device).item())
            return [offset]
        return list(range(self.factor))

    def _adjust_start_index(
        self, start_timepoint_index: torch.Tensor | None, offset: int
    ) -> torch.Tensor | None:
        if self.time_index_mode == "none":
            return None
        if start_timepoint_index is None:
            return None

        shifted = start_timepoint_index + float(offset)
        if self.time_index_mode == "scaled":
            return shifted / float(self.factor)
        return shifted

    def _build_dynamic_mask(self, x_hi: torch.Tensor, epoch: object) -> torch.Tensor | None:
        if not self.dynamic_mask_enable:
            return None

        epoch_i = self._parse_epoch(epoch)
        if self.dynamic_mask_warmup_epochs > 0 and (epoch_i is None or epoch_i <= self.dynamic_mask_warmup_epochs):
            return None

        x_src = x_hi.detach() if self.dynamic_mask_stop_grad else x_hi
        if x_src.dim() != 5 or x_src.shape[1] != 2:
            raise ValueError(f"Expected x_hi shape (B,2,H,W,T), got {tuple(x_src.shape)}.")

        mag_hi = self._to_magnitude(x_src)  # (B,1,H,W,T)
        std_map = mag_hi.std(dim=-1, unbiased=False).squeeze(1)  # (B,H,W)
        B, H, W = std_map.shape
        numel = H * W
        k = int(round(self.dynamic_mask_fraction * numel))
        k = min(max(k, 1), numel)

        std_flat = std_map.reshape(B, numel)
        topk_idx = torch.topk(std_flat, k=k, dim=1, largest=True, sorted=False).indices
        mask_flat = torch.zeros_like(std_flat, dtype=x_hi.dtype)
        mask_flat.scatter_(1, topk_idx, 1.0)
        mask = mask_flat.view(B, 1, H, W)

        if self.dynamic_mask_smooth_kernel > 1:
            mask = F.avg_pool2d(
                mask,
                kernel_size=self.dynamic_mask_smooth_kernel,
                stride=1,
                padding=self.dynamic_mask_smooth_kernel // 2,
            )

        mask = mask.clamp(min=self.dynamic_mask_clip_min, max=self.dynamic_mask_clip_max)
        return mask.unsqueeze(-1)  # (B,1,H,W,1)

    def _temporal_representation(self, x: torch.Tensor) -> torch.Tensor:
        if self.temporal_mode == "absolute":
            return x

        x_mag = self._to_magnitude(x)  # (B,1,H,W,T)
        T = x_mag.shape[-1]
        if T == 0:
            return x_mag

        if self.temporal_mode == "temporal_difference":
            if T < 2:
                return x_mag[..., :0]
            return x_mag[..., 1:] - x_mag[..., :-1]

        baseline_frames = min(self.baseline_frames, T)
        baseline = x_mag[..., :baseline_frames].mean(dim=-1, keepdim=True)
        baseline_sub = x_mag - baseline

        if self.temporal_mode == "baseline_subtracted":
            return baseline_sub

        # percent_enhancement
        denom = baseline.abs().clamp_min(self.percent_enhancement_eps)
        return baseline_sub / denom

    def _metric_with_optional_mask(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        if mask is None:
            return self.metric(pred, target)

        if mask.dim() != 5 or mask.shape[0] != pred.shape[0]:
            raise ValueError(
                f"Mask shape {tuple(mask.shape)} is incompatible with prediction shape {tuple(pred.shape)}."
            )

        weights = mask.to(device=pred.device, dtype=pred.dtype).expand_as(pred)

        if isinstance(self.metric, torch.nn.L1Loss):
            err = (pred - target).abs()
            return (err * weights).sum() / weights.sum().clamp_min(1.0)

        if isinstance(self.metric, torch.nn.MSELoss):
            err = (pred - target).pow(2)
            return (err * weights).sum() / weights.sum().clamp_min(1.0)

        # Fallback for custom metrics: apply soft mask on both arguments.
        return self.metric(pred * weights, target * weights)

    def forward(
        self,
        y_hi: torch.Tensor,
        x_net: torch.Tensor,
        physics: MCNUFFT,
        model,
        csmap: torch.Tensor,
        acceleration: torch.Tensor | None,
        start_timepoint_index: torch.Tensor | None,
        spokes_per_frame: int,
        samples_per_spoke: int,
        norm: str = "both",
        **kwargs,
    ) -> torch.Tensor:
        if self.factor <= 1:
            return self._zero_like_loss_ref(x_net)

        epoch = kwargs.get("epoch", None)
        dynamic_mask = self._build_dynamic_mask(x_net, epoch)
        offsets = self._select_offsets(x_net)

        losses: list[torch.Tensor] = []

        for offset in offsets:
            # High-temporal recon (already computed outside) -> downsample to low temporal grid.
            x_hi_down = self._downsample_time_mean(x_net, self.factor, offset=offset)
            if x_hi_down.shape[-1] == 0:
                continue

            # Rebin measurements and corresponding operator.
            y_lo = self._rebin_kspace(
                y_hi, spokes_per_frame, samples_per_spoke, self.factor, offset=offset
            )
            ktraj_lo, dcomp_lo = self._rebin_ktraj_dcomp(
                physics, spokes_per_frame, samples_per_spoke, self.factor, offset=offset
            )
            physics_lo = MCNUFFT(physics.nufft_ob, physics.adjnufft_ob, ktraj_lo, dcomp_lo).to(csmap.device)

            # Adjust conditioning for the re-binned series.
            if acceleration is None:
                acceleration_lo = None
            else:
                acceleration_lo = acceleration / float(self.factor)

            start_idx_lo = self._adjust_start_index(start_timepoint_index, offset)

            run_teacher_no_grad = self.teacher_branch == "lo" and self.teacher_stopgrad
            if run_teacher_no_grad:
                with torch.no_grad():
                    x_lo, *_ = model(
                        y_lo.to(csmap.device),
                        physics_lo,
                        csmap,
                        acceleration_lo,
                        start_idx_lo,
                        epoch=None,
                        norm=norm,
                    )
            else:
                x_lo, *_ = model(
                    y_lo.to(csmap.device),
                    physics_lo,
                    csmap,
                    acceleration_lo,
                    start_idx_lo,
                    epoch=None,
                    norm=norm,
                )

            if self.teacher_branch == "lo":
                pred = x_hi_down
                target = x_lo.detach() if self.teacher_stopgrad else x_lo
            elif self.teacher_branch == "hi":
                pred = x_lo
                target = x_hi_down.detach() if self.teacher_stopgrad else x_hi_down
            else:
                pred = x_lo
                target = x_hi_down

            pred_cmp = self._temporal_representation(pred)
            target_cmp = self._temporal_representation(target)
            if pred_cmp.shape[-1] == 0:
                continue

            losses.append(self._metric_with_optional_mask(pred_cmp, target_cmp, dynamic_mask))

        if not losses:
            return self._zero_like_loss_ref(x_net)

        return torch.stack(losses).mean()
