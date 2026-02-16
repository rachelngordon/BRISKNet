from typing import Union

import torch
from deepinv.loss.loss import Loss
from deepinv.loss.metric.metric import Metric
from deepinv.transform.base import Transform
from radial_lsfp import to_torch_complex


class WeightedMAEMSELoss(torch.nn.Module):
    """Weighted combination of L1 and L2 losses."""

    def __init__(self, mae_weight: float = 1.0, mse_weight: float = 0.02):
        super().__init__()
        self.mae_weight = float(mae_weight)
        self.mse_weight = float(mse_weight)
        if self.mae_weight < 0.0 or self.mse_weight < 0.0:
            raise ValueError(
                f"mae_weight and mse_weight must be >= 0, got "
                f"{self.mae_weight} and {self.mse_weight}."
            )
        if self.mae_weight == 0.0 and self.mse_weight == 0.0:
            raise ValueError("At least one of mae_weight or mse_weight must be > 0.")
        self._l1 = torch.nn.L1Loss()
        self._l2 = torch.nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        out = 0.0
        if self.mae_weight > 0.0:
            out = out + self.mae_weight * self._l1(pred, target)
        if self.mse_weight > 0.0:
            out = out + self.mse_weight * self._l2(pred, target)
        return out


def _expand_sample_mask(sample_mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Broadcast a spoke/time mask to ref tensor shape for masked losses."""
    if sample_mask is None:
        return None
    if not torch.is_tensor(sample_mask):
        sample_mask = torch.as_tensor(sample_mask, device=ref.device)
    mask = sample_mask.to(device=ref.device, dtype=torch.bool)

    if mask.ndim == 2:
        # (M, T) -> (1, M, T)
        mask = mask.unsqueeze(0)

    if mask.ndim == ref.ndim - 1:
        # e.g. (C, M, T) -> (C, M, T, 1)
        mask = mask.unsqueeze(-1)

    if mask.ndim != ref.ndim:
        raise ValueError(
            f"Unsupported sample_mask ndim={mask.ndim} for ref ndim={ref.ndim}."
        )

    expand_shape = []
    for mask_dim, ref_dim in zip(mask.shape, ref.shape):
        if mask_dim == ref_dim:
            expand_shape.append(ref_dim)
        elif mask_dim == 1:
            expand_shape.append(ref_dim)
        else:
            raise ValueError(
                "sample_mask shape mismatch before broadcast: "
                f"mask={tuple(mask.shape)} vs ref={tuple(ref.shape)}."
            )
    mask = mask.expand(*expand_shape)

    if tuple(mask.shape) != tuple(ref.shape):
        raise ValueError(
            "sample_mask shape mismatch after broadcast: "
            f"mask={tuple(mask.shape)} vs ref={tuple(ref.shape)}."
        )
    return mask


def _masked_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    metric: Union[Metric, torch.nn.Module],
    sample_mask: torch.Tensor = None,
) -> torch.Tensor:
    if sample_mask is None:
        return metric(pred, target)

    mask = _expand_sample_mask(sample_mask, pred)
    selected = int(mask.sum().item())
    if selected <= 0:
        raise ValueError("sample_mask selected zero elements for metric computation.")
    return metric(pred[mask], target[mask])


def _to_ri_lastdim2(x: torch.Tensor) -> torch.Tensor:
    """Return tensor with real/imag stacked in the last dim."""
    if torch.is_complex(x):
        return torch.stack([x.real, x.imag], dim=-1)
    if x.shape[-1] != 2:
        raise ValueError(
            "Expected complex tensor or real/imag-stacked tensor with last dim=2, "
            f"got shape {tuple(x.shape)}."
        )
    return x


class MCLoss(Loss):
    r"""
    Measurement consistency loss

    This loss enforces that the reconstructions are measurement-consistent, i.e., :math:`y=\forw{\inverse{y}}`.

    The measurement consistency loss is defined as

    .. math::

        \|y-\forw{\inverse{y}}\|^2

    where :math:`\inverse{y}` is the reconstructed signal and :math:`A` is a forward operator.

    By default, the error is computed using the MSE metric, however any other metric (e.g., :math:`\ell_1`)
    can be used as well.

    :param Metric, torch.nn.Module metric: metric used for computing data consistency, which is set as the mean squared error by default.
    """

    def __init__(self, model_type, metric: Union[Metric, torch.nn.Module] = torch.nn.MSELoss()):
        super(MCLoss, self).__init__()
        self.name = "mc"
        self.metric = metric
        self.device = torch.device("cuda")
        self.model_type = model_type

    def forward(self, y, x_net, physics, csmap, sample_mask=None, **kwargs):
        r"""
        Computes the measurement splitting loss

        :param torch.Tensor y: measurements.
        :param torch.Tensor x_net: reconstructed image :math:`\inverse{y}`.
        :param deepinv.physics.Physics physics: forward operator associated with the measurements.
        :return: (:class:`torch.Tensor`) loss.
        """
        if self.model_type == "CRNN":
            y_hat = kwargs.get("y_hat_override", None)
            if y_hat is None:
                y_hat = physics.A(x_net, csmap)
            return _masked_metric(y_hat, y, self.metric, sample_mask=sample_mask)
        elif self.model_type in {"LSFPNet", "MambaRecon", "MambaTemporal"}:
            y_hat = kwargs.get("y_hat_override", None)
            if y_hat is None:
                x_net = to_torch_complex(x_net)
                y_hat = physics(inv=False, data=x_net, smaps=csmap).to(self.device)
            y_hat = _to_ri_lastdim2(y_hat)
            y = _to_ri_lastdim2(y)

            return _masked_metric(y_hat, y, self.metric, sample_mask=sample_mask)
        raise ValueError(
            f"Unsupported model_type '{self.model_type}' for MCLoss. "
            "Expected one of: CRNN, LSFPNet, MambaRecon, MambaTemporal."
        )


class SSDULoss(Loss):
    """Self-supervised data undersampling loss on held-out spokes."""

    def __init__(
        self,
        model_type,
        holdout_fraction: float = 0.2,
        metric: Union[Metric, torch.nn.Module] = torch.nn.MSELoss(),
        seed_mode: str = "step",
        seed_base: int = 0,
        min_heldout_spokes: int = 1,
    ):
        super().__init__()
        self.name = "ssdu"
        self.model_type = model_type
        self.metric = metric
        self.holdout_fraction = float(holdout_fraction)
        self.seed_mode = str(seed_mode).strip().lower()
        self.seed_base = int(seed_base)
        self.min_heldout_spokes = int(min_heldout_spokes)
        self.device = torch.device("cuda")

        if not (0.0 < self.holdout_fraction < 1.0):
            raise ValueError(
                f"holdout_fraction must be in (0, 1), got {self.holdout_fraction}."
            )
        if self.seed_mode not in {"step", "none"}:
            raise ValueError(
                f"Unsupported seed_mode '{seed_mode}'. Expected one of: step, none."
            )
        if self.min_heldout_spokes < 1:
            raise ValueError("min_heldout_spokes must be >= 1.")

    def split_measurements(self, y, samples_per_spoke: int, step: int = None):
        """
        Split measured k-space into used/held-out spokes for SSDU training.

        Args:
            y: complex k-space with shape (C, M, T), where M = spokes * samples_per_spoke.
            samples_per_spoke: number of samples per spoke.
            step: training step used for deterministic seed when seed_mode='step'.
        """
        if y.ndim != 3:
            raise ValueError(f"SSDU expects y with shape (C, M, T), got {tuple(y.shape)}.")
        samples_per_spoke = int(samples_per_spoke)
        if samples_per_spoke < 1:
            raise ValueError(f"samples_per_spoke must be >= 1, got {samples_per_spoke}.")
        _, M, T = y.shape
        if M % samples_per_spoke != 0:
            raise ValueError(
                f"k-space length M={M} is not divisible by samples_per_spoke={samples_per_spoke}."
            )
        spokes_per_frame = M // samples_per_spoke
        if spokes_per_frame < 2:
            raise ValueError(
                f"SSDU needs at least 2 spokes, got spokes_per_frame={spokes_per_frame}."
            )

        n_holdout = int(round(float(spokes_per_frame) * self.holdout_fraction))
        n_holdout = max(self.min_heldout_spokes, n_holdout)
        n_holdout = min(n_holdout, spokes_per_frame - 1)

        generator = None
        if self.seed_mode == "step":
            if step is None:
                raise ValueError("SSDU seed_mode='step' requires step to be provided.")
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(self.seed_base + int(step)))

        spoke_perm = torch.randperm(spokes_per_frame, generator=generator)
        spoke_perm = spoke_perm.to(y.device)
        held_spokes = spoke_perm[:n_holdout]
        held_mask_spoke = torch.zeros(spokes_per_frame, dtype=torch.bool, device=y.device)
        held_mask_spoke[held_spokes] = True

        held_mask_sample = held_mask_spoke[:, None].expand(spokes_per_frame, samples_per_spoke)
        held_mask_m = held_mask_sample.reshape(M)
        heldout_mask = held_mask_m[:, None].expand(M, T)
        used_mask = ~heldout_mask

        y_used = y.masked_fill(heldout_mask.unsqueeze(0), 0)
        return y_used, heldout_mask, used_mask

    def forward(self, y, x_net, physics, csmap, heldout_mask, **kwargs):
        """
        SSDU loss computed only on held-out measurements.

        Args:
            y: full measured k-space (complex), shape (C, M, T).
            x_net: reconstructed image.
            physics: forward model.
            csmap: coil sensitivity maps.
            heldout_mask: bool mask over (M, T), True where held-out.
        """
        if self.model_type == "CRNN":
            y_hat = kwargs.get("y_hat_override", None)
            if y_hat is None:
                y_hat = physics.A(x_net, csmap)
            return _masked_metric(y_hat, y, self.metric, sample_mask=heldout_mask)
        if self.model_type in {"LSFPNet", "MambaRecon", "MambaTemporal"}:
            y_hat = kwargs.get("y_hat_override", None)
            if y_hat is None:
                x_net = to_torch_complex(x_net)
                y_hat = physics(inv=False, data=x_net, smaps=csmap).to(self.device)

            y_hat = _to_ri_lastdim2(y_hat)
            y = _to_ri_lastdim2(y)
            return _masked_metric(y_hat, y, self.metric, sample_mask=heldout_mask)
        raise ValueError(
            f"Unsupported model_type '{self.model_type}' for SSDULoss. "
            "Expected one of: CRNN, LSFPNet, MambaRecon, MambaTemporal."
        )
