from __future__ import annotations

from typing import Union

import torch
from deepinv.loss.loss import Loss
from deepinv.loss.metric.metric import Metric

from radial_lsfp import MCNUFFT


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
    ):
        super().__init__()
        self.name = "rebin_consistency"
        self.factor = max(1, int(factor))
        self.metric = metric
        self.time_index_mode = str(time_index_mode).lower()

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

    @staticmethod
    def _rebin_kspace(y: torch.Tensor, spf_hi: int, samples_per_spoke: int, factor: int) -> torch.Tensor:
        # y: (coils, spf_hi * samples, T_hi) -> (coils, (spf_hi*factor)*samples, T_lo)
        if factor <= 1:
            return y
        coils, sp_samp, T = y.shape
        expected = spf_hi * samples_per_spoke
        if sp_samp != expected:
            raise ValueError(f"Unexpected k-space shape: got {sp_samp}, expected {expected}.")

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
        physics: MCNUFFT, spf_hi: int, samples_per_spoke: int, factor: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ktraj: (2, spf_hi * samples, T_hi) -> (2, (spf_hi*factor)*samples, T_lo)
        # dcomp: (spf_hi * samples, T_hi) -> ((spf_hi*factor)*samples, T_lo)
        if factor <= 1:
            return physics.ktraj, physics.dcomp

        ktraj = physics.ktraj
        dcomp = physics.dcomp
        _, sp_samp, T = ktraj.shape
        expected = spf_hi * samples_per_spoke
        if sp_samp != expected:
            raise ValueError(f"Unexpected ktraj shape: got {sp_samp}, expected {expected}.")

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
            return torch.tensor(0.0, device=x_net.device, dtype=x_net.dtype)

        # High-temporal recon (already computed outside) -> downsample to low temporal grid.
        x_hi_down = self._downsample_time_mean(x_net, self.factor)
        if x_hi_down.shape[-1] == 0:
            return torch.tensor(0.0, device=x_net.device, dtype=x_net.dtype)

        # Rebin measurements and corresponding operator.
        y_lo = self._rebin_kspace(y_hi, spokes_per_frame, samples_per_spoke, self.factor)
        ktraj_lo, dcomp_lo = self._rebin_ktraj_dcomp(physics, spokes_per_frame, samples_per_spoke, self.factor)
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
