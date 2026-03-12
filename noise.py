"""Noise model utilities. Run: imported by training scripts (not intended to run directly)."""

from typing import Callable
import warnings

import torch
import torch.nn as nn


class NoiseModel(nn.Module):
    r"""
    Base class for noise model.

    Noise models can be combined via :func:`deepinv.physics.NoiseModel.__mul__`.

    :param Callable noise_model: noise model function :math:`N(y)`.
    :param torch.Generator rng: (optional) a pseudorandom random number generator for the parameter generation.
        If provided, it should be on the same device as the input.
    """

    def __init__(self, noise_model: Callable = None, rng: torch.Generator = None):
        super().__init__()
        if noise_model is None:
            noise_model = lambda x: x
        self.noise_model = noise_model
        self.rng = rng
        if rng is not None:
            self.register_buffer("initial_random_state", rng.get_state())

    def forward(self, input: torch.Tensor, seed: int = None) -> torch.Tensor:
        r"""
        Add noise to the input

        :param torch.Tensor input: input tensor
        :param int seed: the seed for the random number generator.
        """
        self.rng_manual_seed(seed)
        return self.noise_model(input)

    def __mul__(self, other):
        r"""
        Concatenates two noise :math:`N = N_1 \circ N_2` via the mul operation

        The resulting operator will add the noise from both noise models and keep the `rng` of :math:`N_1`.

        :param deepinv.physics.NoiseModel other: Physics operator :math:`A_2`
        :return: (deepinv.physics.NoiseModel) concatenated operator

        """
        noise_model = lambda x: self.noise_model(other.noise_model(x))
        return NoiseModel(
            noise_model=noise_model,
            rng=self.rng,
        )

    def rng_manual_seed(self, seed: int = None):
        r"""
        Sets the seed for the random number generator.

        :param int seed: the seed to set for the random number generator.
            If not provided, the current state of the random number generator is used.
            .. note:: The seed will be ignored if the random number generator is not initialized.
        """
        if seed is not None:
            if self.rng is not None:
                self.rng = self.rng.manual_seed(seed)
            else:
                warnings.warn(
                    "Cannot set seed for random number generator because it is not initialized. The `seed` parameter is ignored."
                )

    def reset_rng(self):
        r"""
        Reset the random number generator to its initial state.
        """
        if self.rng is not None:
            self.rng.set_state(self.initial_random_state)
        else:
            warnings.warn(
                "Cannot reset state for random number generator because it was not initialized. This is ignored."
            )

    def rand_like(self, input: torch.Tensor, seed: int = None):
        r"""
        Equivalent to `torch.rand_like` but supports a pseudorandom number generator argument.

        :param int seed: the seed for the random number generator, if `rng` is provided.
        """
        self.rng_manual_seed(seed)
        return torch.empty_like(input).uniform_(generator=self.rng)

    def randn_like(self, input: torch.Tensor, seed: int = None):
        r"""
        Equivalent to `torch.randn_like` but supports a pseudorandom number generator argument.

        :param int seed: the seed for the random number generator, if `rng` is provided.
        """
        self.rng_manual_seed(seed)
        return torch.empty_like(input).normal_(generator=self.rng)

    def update_parameters(self, **kwargs):
        r"""
        Update the parameters of the noise model.

        :param dict kwargs: dictionary of parameters to update.
        """
        if kwargs:
            for key, value in kwargs.items():
                if (
                    value is not None
                    and hasattr(self, key)
                    and isinstance(value, (torch.Tensor, float))
                ):
                    self.register_buffer(key, self._float_to_tensor(value))

    def _float_to_tensor(self, value):
        r"""
        Convert a float or int to a torch.Tensor.

        :param value float or int or torch.Tensor: the input value

        :return: the same value as a torch.Tensor
        :rtype: torch.Tensor
        """
        if value is None:
            return value
        elif isinstance(value, (float, int)):
            return torch.tensor(value, dtype=torch.float32)
        elif isinstance(value, torch.Tensor):
            return value
        else:
            raise ValueError(
                f"Unsupported type for noise level. Expected float, int, or torch.Tensor, got {type(value)}."
            )

    # To handle the transfer between CPU/GPU properly
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = self._get_device_from_args(*args, **kwargs)
        if device is not None and self.rng is not None:
            state = self.rng.get_state()
            # Move the generator to the specified device
            self.rng = torch.Generator(device=device)
            try:
                self.rng.set_state(state)
            except RuntimeError:
                warnings.warn(
                    "Moving the random number generator between CPU/GPU is not possible. Reinitialize the generator on the correct device."
                )

        return self

    # Helper to extract device from .to() arguments
    def _get_device_from_args(self, *args, **kwargs):
        if args:
            if isinstance(args[0], torch.device):
                return args[0]
            elif isinstance(args[0], str):
                return torch.device(args[0])
        if "device" in kwargs:
            return (
                torch.device(kwargs["device"])
                if isinstance(kwargs["device"], str)
                else kwargs["device"]
            )
        return None


class ZeroNoise(NoiseModel):
    r"""
    Zero noise model :math:`y=x`, serve as a placeholder.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        r"""
        Return the same input.

        :param torch.Tensor x: measurements.
        :returns: x.
        """
        return x
