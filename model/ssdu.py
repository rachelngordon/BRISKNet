"""SSDU spoke-wise split utilities and loss for self-supervised k-space training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from deepinv.loss.loss import Loss
from deepinv.loss.metric.metric import Metric

from .radial import MCNUFFT, to_torch_complex


def _build_ssdu_fold_indices(
    spokes_per_frame: int,
    samples_per_spoke: int,
    k_folds: int,
    device: torch.device,
    allow_single_spoke: bool,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    effective_k = min(int(k_folds), int(spokes_per_frame))
    if effective_k < 2:
        return []

    total_samples = int(spokes_per_frame) * int(samples_per_spoke)
    fold_indices: List[Tuple[torch.Tensor, torch.Tensor]] = []
    sample_offsets = torch.arange(samples_per_spoke, device=device)

    for fold_idx in range(effective_k):
        held_spokes = torch.arange(fold_idx, spokes_per_frame, effective_k, device=device)
        used_spokes = int(spokes_per_frame) - int(held_spokes.numel())
        if used_spokes < 2 and not allow_single_spoke:
            continue

        held_idx = (held_spokes[:, None] * samples_per_spoke + sample_offsets[None, :]).reshape(-1)
        held_mask = torch.zeros(total_samples, dtype=torch.bool, device=device)
        held_mask[held_idx] = True
        used_idx = (~held_mask).nonzero(as_tuple=False).squeeze(-1)
        fold_indices.append((held_idx, used_idx))

    return fold_indices


@dataclass
class SSDUSplit:
    y_theta: torch.Tensor
    y_lambda: torch.Tensor
    physics_theta: MCNUFFT
    physics_lambda: MCNUFFT
    held_idx: torch.Tensor
    used_idx: torch.Tensor


def _slice_dcomp(dcomp: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    if dcomp.ndim == 1:
        return dcomp[idx]
    if dcomp.ndim == 2:
        return dcomp[idx, :]
    raise ValueError(f"Unsupported dcomp shape for SSDU split: {tuple(dcomp.shape)}")


def build_spoke_wise_ssdu_split(
    y: torch.Tensor,
    physics: MCNUFFT,
    spokes_per_frame: int,
    samples_per_spoke: int,
    k_folds: int = 4,
    selection: str = "random",
    iteration_index: Optional[int] = None,
    allow_single_spoke: bool = False,
) -> Optional[SSDUSplit]:
    if y.ndim != 3:
        raise ValueError(f"SSDU expects y with shape (coils, samples, time), got {tuple(y.shape)}.")
    if spokes_per_frame <= 0 or samples_per_spoke <= 0:
        raise ValueError("spokes_per_frame and samples_per_spoke must be positive.")

    expected_samples = int(spokes_per_frame) * int(samples_per_spoke)
    if int(y.shape[1]) != expected_samples:
        raise ValueError(
            "SSDU split expects y.shape[1] == spokes_per_frame * samples_per_spoke, got "
            f"{int(y.shape[1])} vs {expected_samples}."
        )

    device = y.device
    fold_indices = _build_ssdu_fold_indices(
        spokes_per_frame=int(spokes_per_frame),
        samples_per_spoke=int(samples_per_spoke),
        k_folds=int(k_folds),
        device=device,
        allow_single_spoke=bool(allow_single_spoke),
    )
    if not fold_indices:
        return None

    selection_norm = str(selection).strip().lower()
    if selection_norm == "cyclic":
        if iteration_index is None:
            iteration_index = 0
        fold_pos = int(iteration_index) % len(fold_indices)
    elif selection_norm == "random":
        fold_pos = random.randrange(len(fold_indices))
    else:
        raise ValueError(
            f"Unsupported SSDU selection '{selection}'. Expected one of: random, cyclic."
        )

    held_idx, used_idx = fold_indices[fold_pos]

    if physics.ktraj.ndim != 3 or physics.ktraj.shape[0] != 2:
        raise ValueError(
            f"SSDU split expects physics.ktraj shape (2, samples, time), got {tuple(physics.ktraj.shape)}."
        )

    ktraj_theta = physics.ktraj[:, used_idx, :]
    ktraj_lambda = physics.ktraj[:, held_idx, :]
    dcomp_theta = _slice_dcomp(physics.dcomp, used_idx)
    dcomp_lambda = _slice_dcomp(physics.dcomp, held_idx)

    physics_theta = MCNUFFT(physics.nufft_ob, physics.adjnufft_ob, ktraj_theta, dcomp_theta)
    physics_lambda = MCNUFFT(physics.nufft_ob, physics.adjnufft_ob, ktraj_lambda, dcomp_lambda)

    y_theta = y[:, used_idx, :]
    y_lambda = y[:, held_idx, :]

    return SSDUSplit(
        y_theta=y_theta,
        y_lambda=y_lambda,
        physics_theta=physics_theta,
        physics_lambda=physics_lambda,
        held_idx=held_idx,
        used_idx=used_idx,
    )


class SSDULoss(Loss):
    """SSDU k-space loss on held-out spokes."""

    def __init__(
        self,
        model_type: str,
        metric: Union[Metric, torch.nn.Module] = torch.nn.MSELoss(),
        weighting: str = "sqrt_dcomp",
    ):
        super().__init__()
        self.name = "ssdu"
        self.metric = metric
        self.model_type = model_type
        self.weighting = str(weighting).strip().lower()

    def _weight_tensor(self, dcomp: torch.Tensor, target: torch.Tensor) -> Optional[torch.Tensor]:
        if self.weighting in ("none", "", "uniform"):
            return None
        if self.weighting != "sqrt_dcomp":
            raise ValueError(
                f"Unsupported SSDU weighting '{self.weighting}'. Expected one of: sqrt_dcomp, none."
            )

        if dcomp.ndim == 1:
            weight = torch.sqrt(torch.abs(dcomp)).unsqueeze(0).unsqueeze(-1)
        elif dcomp.ndim == 2:
            weight = torch.sqrt(torch.abs(dcomp)).unsqueeze(0).unsqueeze(-1)
        else:
            raise ValueError(f"Unsupported dcomp shape for SSDU weighting: {tuple(dcomp.shape)}")

        return weight.to(device=target.device, dtype=target.dtype)

    def forward(self, y_held, x_net, physics_held, csmap, **kwargs):
        if self.model_type not in {"LSFPNet", "MambaRecon", "MambaTemporal"}:
            raise ValueError(
                f"Unsupported model_type '{self.model_type}' for SSDULoss. "
                "Expected one of: LSFPNet, MambaRecon, MambaTemporal."
            )

        x_net_complex = to_torch_complex(x_net)
        y_hat = physics_held(inv=False, data=x_net_complex, smaps=csmap).to(y_held.device)

        y_hat = torch.stack([y_hat.real, y_hat.imag], dim=-1)
        y_ref = torch.stack([y_held.real, y_held.imag], dim=-1)

        weight = self._weight_tensor(physics_held.dcomp, y_hat)
        if weight is not None:
            y_hat = y_hat * weight
            y_ref = y_ref * weight

        return self.metric(y_hat, y_ref)
