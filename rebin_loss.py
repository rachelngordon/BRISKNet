from __future__ import annotations

from typing import Union

import torch
from deepinv.loss.loss import Loss
from deepinv.loss.metric.metric import Metric

from radial_lsfp import MCNUFFT
from utils import rebin_binned_sequence


class RebinConsistencyLoss(Loss):
    r"""
    Operator-level (binning) consistency loss for radial dynamic MRI.

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
        jitter_max_frames: int = 0,
    ):
        super().__init__()
        self.name = "rebin_consistency"
        self.factor = max(1, int(factor))
        self.metric = metric
        self.time_index_mode = str(time_index_mode).lower()
        self.jitter_max_frames = max(0, int(jitter_max_frames))

        if self.time_index_mode not in {"none", "inherit", "scaled"}:
            raise ValueError(
                f"Unsupported time_index_mode '{time_index_mode}'. Expected one of: none, inherit, scaled."
            )

    @staticmethod
    def _downsample_time_mean(x: torch.Tensor, factor: int) -> torch.Tensor:
        # x: (B, 2, H, W, T_hi) -> (B, 2, H, W, T_lo) via mean over groups.
        if factor <= 1:
            return x
        B, C, H, W, T = x.shape
        T_lo = T // factor
        if T_lo <= 0:
            return x[:, :, :, :, :0]
        x_cropped = x[:, :, :, :, : T_lo * factor]
        x_grouped = x_cropped.view(B, C, H, W, T_lo, factor)
        return x_grouped.mean(dim=-1)

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
        jitter_max_frames: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.factor <= 1:
            return torch.tensor(0.0, device=x_net.device, dtype=x_net.dtype)

        # Optional jitter: drop a small number of initial frames before rebinning.
        jitter_limit = self.jitter_max_frames if jitter_max_frames is None else int(jitter_max_frames)
        max_offset = min(jitter_limit, max(0, self.factor - 1))
        if max_offset > 0:
            offset = int(torch.randint(0, max_offset + 1, (1,), device=x_net.device).item())
            if offset > 0:
                x_net = x_net[..., offset:]
                y_hi = y_hi[..., offset:]
                ktraj_hi = physics.ktraj[..., offset:]
                dcomp_hi = physics.dcomp[..., offset:]
                if start_timepoint_index is not None:
                    start_timepoint_index = start_timepoint_index + float(offset)
            else:
                ktraj_hi = physics.ktraj
                dcomp_hi = physics.dcomp
        else:
            ktraj_hi = physics.ktraj
            dcomp_hi = physics.dcomp

        # High-temporal recon (already computed outside) -> downsample to low temporal grid.
        x_hi_down = self._downsample_time_mean(x_net, self.factor)
        if x_hi_down.shape[-1] == 0:
            return torch.tensor(0.0, device=x_net.device, dtype=x_net.dtype)

        # Rebin measurements and corresponding operator.
        y_lo = rebin_binned_sequence(y_hi, spokes_per_frame, samples_per_spoke, self.factor)
        ktraj_lo = rebin_binned_sequence(ktraj_hi, spokes_per_frame, samples_per_spoke, self.factor)
        dcomp_lo = rebin_binned_sequence(
            dcomp_hi.unsqueeze(0), spokes_per_frame, samples_per_spoke, self.factor
        ).squeeze(0)
        physics_lo = MCNUFFT(physics.nufft_ob, physics.adjnufft_ob, ktraj_lo, dcomp_lo).to(csmap.device)

        # Adjust conditioning for the re-binned series.
        if acceleration is None:
            acceleration_lo = None
        else:
            acceleration_lo = acceleration / float(self.factor)

        if self.time_index_mode == "none":
            start_idx_lo = None
        elif self.time_index_mode == "scaled":
            start_idx_lo = None if start_timepoint_index is None else (start_timepoint_index / float(self.factor))
        else:  # inherit
            start_idx_lo = start_timepoint_index

        # Reconstruct low-temporal series.
        x_lo, *_ = model(
            y_lo.to(csmap.device),
            physics_lo,
            csmap,
            acceleration_lo,
            start_idx_lo,
            epoch=None,
            norm=norm,
        )

        # Compare in complex (2-channel) image space.
        return self.metric(x_lo, x_hi_down)
