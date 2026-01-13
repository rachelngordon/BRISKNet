from typing import Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from deepinv.loss.loss import Loss
from deepinv.loss.metric.metric import Metric
from deepinv.transform.base import Transform
from einops import rearrange
from radial_lsfp import to_torch_complex


class EILoss(Loss):
    r"""
    Equivariant imaging self-supervised loss.

    Assumes that the set of signals is invariant to a group of transformations (rotations, translations, etc.)
    in order to learn from incomplete measurement data alone https://https://arxiv.org/pdf/2103.14756.pdf.

    The EI loss is defined as

    .. math::

        \| T_g \hat{x} - \inverse{\forw{T_g \hat{x}}}\|^2


    where :math:`\hat{x}=\inverse{y}` is a reconstructed signal and
    :math:`T_g` is a transformation sampled at random from a group :math:`g\sim\group`.

    By default, the error is computed using the MSE metric, however any other metric (e.g., :math:`\ell_1`)
    can be used as well.

    :param deepinv.transform.Transform transform: Transform to generate the virtually augmented measurement.
        It can be any torch-differentiable function (e.g., a :class:`torch.nn.Module`)
        including `torchvision transforms <https://pytorch.org/vision/stable/transforms.html>`_.
    :param Metric, torch.nn.Module metric: Metric used to compute the error between the reconstructed augmented measurement and the reference
        image.
    :param bool apply_noise: if ``True``, the augmented measurement is computed with the full sensing model
        :math:`\sensor{\noise{\forw{\hat{x}}}}` (i.e., noise and sensor model),
        otherwise is generated as :math:`\forw{\hat{x}}`.
    :param float weight: Weight of the loss.
    :param bool no_grad: if ``True``, the gradient does not propagate through :math:`T_g`. Default: ``False``.
        This option is useful for super-resolution problems, see https://arxiv.org/abs/2312.11232.
    """

    def __init__(
        self,
        transform: Transform,
        model_type: str,
        metric: Union[Metric, nn.Module] = torch.nn.MSELoss(),
        apply_noise=True,
        weight=1.0,
        no_grad=False,
        checkpoint_model: bool = False,
        checkpoint_mode: str | None = None,
        checkpoint_use_reentrant: bool = False,
        *args,
        **kwargs,
    ):
        super(EILoss, self).__init__(*args, **kwargs)
        self.name = "ei"
        self.metric = metric
        self.weight = weight
        self.T = transform
        self.noise = apply_noise
        self.no_grad = no_grad
        self.checkpoint_use_reentrant = checkpoint_use_reentrant
        self.checkpoint_outer = checkpoint_model
        self.checkpoint_inner = True
        if checkpoint_mode is not None:
            mode = checkpoint_mode.lower()
            if mode == "none":
                self.checkpoint_outer = False
                self.checkpoint_inner = False
            elif mode == "outer":
                self.checkpoint_outer = True
                self.checkpoint_inner = False
            elif mode == "inner":
                self.checkpoint_outer = False
                self.checkpoint_inner = True
            elif mode == "both":
                self.checkpoint_outer = True
                self.checkpoint_inner = True
            else:
                raise ValueError(
                    f"Unsupported checkpoint_mode '{checkpoint_mode}'. Expected one of: none, outer, inner, both."
                )
        self.model_type = model_type

    def forward(self, x_net, physics, model, csmap, acceleration, start_timepoint_index, **kwargs):
        r"""
        Computes the EI loss

        :param torch.Tensor x_net: Reconstructed image :math:`\inverse{y}`.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
        :param torch.nn.Module model: Reconstruction function.
        :return: (:class:`torch.Tensor`) loss.
        """

        if self.no_grad:
            # NOTE: Calling both torch.no_grad() and detach() is not redundant.
            # One avoids unnecessary computations and makes the code more efficient
            # while the other ensures that x2 is marked as a leaf in the computational graph.
            with torch.no_grad():
                x_net_rearranged = rearrange(x_net, 'b c h w t -> b c t h w')
                x2_rearranged = self.T(x_net_rearranged)
                x2 = rearrange(x2_rearranged, 'b c t h w -> b c h w t')
                x2 = x2.detach()
        else:
            x_net_rearranged = rearrange(x_net, 'b c h w t -> b c t h w')
            x2_rearranged = self.T(x_net_rearranged)
            x2 = rearrange(x2_rearranged, 'b c t h w -> b c h w t')

        # if self.model_type == "CRNN":
        #     if self.noise:
        #         # NOTE: need to pass csmap for multi coil imp
        #         y = physics(x2, csmap)
        #     else:
        #         y = physics.A(x2, csmap)

        #     x3 = model(y, physics, csmap)

        # elif self.model_type == "LSFPNet":

        x2_complex = to_torch_complex(x2)
        y = physics(inv=False, data=x2_complex, smaps=csmap).to(csmap.device)
    
        disable_checkpointing = not self.checkpoint_inner
        if self.checkpoint_outer:
            def _run_model(y_in):
                x3_inner, *_ = model(
                    y_in,
                    physics,
                    csmap,
                    acceleration,
                    start_timepoint_index,
                    epoch=None,
                    disable_checkpointing=disable_checkpointing,
                )
                return x3_inner

            x3 = checkpoint(_run_model, y, use_reentrant=self.checkpoint_use_reentrant)
        else:
            x3, *_ = model(
                y,
                physics,
                csmap,
                acceleration,
                start_timepoint_index,
                epoch=None,
                disable_checkpointing=disable_checkpointing,
            )

        loss_ei = self.weight * self.metric(x3, x2)
        return loss_ei, x2
    
