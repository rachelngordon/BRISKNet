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

    def forward(self, y, x_net, physics, csmap, **kwargs):
        r"""
        Computes the measurement splitting loss

        :param torch.Tensor y: measurements.
        :param torch.Tensor x_net: reconstructed image :math:`\inverse{y}`.
        :param deepinv.physics.Physics physics: forward operator associated with the measurements.
        :return: (:class:`torch.Tensor`) loss.
        """
        if self.model_type == "CRNN":
            return self.metric(physics.A(x_net, csmap), y)
        elif self.model_type in {"LSFPNet", "MambaRecon", "MambaTemporal"}:
            x_net = to_torch_complex(x_net)

            y_hat = physics(inv=False, data=x_net, smaps=csmap).to(self.device)

            y_hat = torch.stack([y_hat.real, y_hat.imag], dim=-1)
            y = torch.stack([y.real, y.imag], dim=-1)

            return self.metric(y_hat, y)
        raise ValueError(
            f"Unsupported model_type '{self.model_type}' for MCLoss. "
            "Expected one of: CRNN, LSFPNet, MambaRecon, MambaTemporal."
        )
