from typing import Union

import torch
from deepinv.loss.loss import Loss
from deepinv.loss.metric.metric import Metric
from deepinv.transform.base import Transform
from radial_lsfp import to_torch_complex


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
