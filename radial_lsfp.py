import torch
import torch.nn as nn
import numpy as np
from time import time
from einops import rearrange

dtype = torch.complex64

def to_torch_complex(x: torch.Tensor):
    """(B, 2, ...) real -> (B, ...) complex"""
    assert x.shape[1] == 2, (
        f"Input tensor must have 2 channels (real, imag), but got shape {x.shape}"
    )
    xc = rearrange(x, "b c ... -> b ... c").contiguous()
    # view_as_complex supports only float/half/double real tensors.
    if xc.dtype not in (torch.float16, torch.float32, torch.float64):
        xc = xc.float()
    return torch.view_as_complex(xc)


def from_torch_complex(x: torch.Tensor):
    """(B, ...) complex -> (B, 2, ...) real"""
    return rearrange(torch.view_as_real(x), "b ... c -> b c ...").contiguous()


class MCNUFFT(nn.Module):
    def __init__(self, nufft_ob, adjnufft_ob, ktraj, dcomp):
        super(MCNUFFT, self).__init__()
        self.nufft_ob = nufft_ob
        self.adjnufft_ob = adjnufft_ob
        # Preserve dimensionality; squeezing singleton time can break temporal models.
        self.ktraj = ktraj
        self.dcomp = dcomp

    @staticmethod
    def _as_single_frame_ktraj(ktraj: torch.Tensor) -> torch.Tensor:
        if ktraj.ndim == 2 and ktraj.shape[0] == 2:
            return ktraj
        if ktraj.ndim == 3 and ktraj.shape[0] == 2 and ktraj.shape[-1] == 1:
            return ktraj[..., 0]
        raise ValueError(
            f"Expected single-frame ktraj with shape [2,samples] or [2,samples,1], got {tuple(ktraj.shape)}."
        )

    @staticmethod
    def _as_single_frame_dcomp(dcomp: torch.Tensor) -> torch.Tensor:
        if dcomp.ndim == 1:
            return dcomp
        if dcomp.ndim == 2 and dcomp.shape[-1] == 1:
            return dcomp[..., 0]
        raise ValueError(
            f"Expected single-frame dcomp with shape [samples] or [samples,1], got {tuple(dcomp.shape)}."
        )

    @staticmethod
    def _as_multi_frame_ktraj(ktraj: torch.Tensor, t: int) -> torch.Tensor:
        if ktraj.ndim != 3 or ktraj.shape[0] != 2:
            raise ValueError(
                f"Expected multi-frame ktraj with shape [2,samples,T], got {tuple(ktraj.shape)}."
            )
        if ktraj.shape[-1] != t:
            raise ValueError(
                f"ktraj time dim mismatch: expected T={t}, got {ktraj.shape[-1]}."
            )
        return ktraj

    @staticmethod
    def _as_multi_frame_dcomp(dcomp: torch.Tensor, t: int) -> torch.Tensor:
        if dcomp.ndim == 1:
            dcomp = dcomp.unsqueeze(-1)
        if dcomp.ndim != 2:
            raise ValueError(
                f"Expected multi-frame dcomp with shape [samples,T], got {tuple(dcomp.shape)}."
            )
        if dcomp.shape[-1] != t:
            raise ValueError(
                f"dcomp time dim mismatch: expected T={t}, got {dcomp.shape[-1]}."
            )
        return dcomp

    def forward(self, inv, data, smaps):
        # Training loss path can pass a singleton batch dim (e.g., [1,H,W,T]); drop it only.
        if data.ndim == 4 and data.shape[0] == 1:
            data = data[0]
        if data.ndim == 3 and data.shape[0] == 1 and data.shape[1] == 2:
            # Defensive: if an unexpected [1,2,*] slips in, this likely indicates
            # unconverted real/imag input; fail with context.
            raise ValueError(
                "MCNUFFT received real/imag channel-first input; expected complex tensor."
            )
        if data.ndim not in (2, 3):
            raise ValueError(
                f"Expected data ndim in {{2,3}} after optional singleton-batch strip, got {data.ndim}."
            )
        Nx, Ny = smaps.shape[2], smaps.shape[3]

        if data.ndim == 3:  # multi-frame
            t = data.shape[-1]
            ktraj = self._as_multi_frame_ktraj(self.ktraj, t=t)
            dcomp = self._as_multi_frame_dcomp(self.dcomp, t=t)

            # --- Vectorized approach ---
            if inv: # Adjoint NUFFT (k-space -> image)
                # Original shape: [coils, samples, time]
                # We need [batch, coils, samples] for nufft, so permute time to batch
                kd = data.permute(2, 0, 1) # -> [time, coils, samples]
                d = dcomp.permute(1, 0) # -> [time, samples]
                k = ktraj.permute(2, 0, 1) # -> [time, samples, 2]

                # Unsqueeze for coils/smaps dim
                d = d.unsqueeze(1) 
                
                # Perform one batched operation
                x_temp = self.adjnufft_ob(kd * d, k, smaps=smaps.to(dtype))
                # Output shape: [time, 1, Nx, Ny]
                
                # Reshape back to [Nx, Ny, time]
                x = x_temp.squeeze(1).permute(1, 2, 0) / np.sqrt(Nx * Ny)

            else: # Forward NUFFT (image -> k-space)
                # Original shape: [Nx, Ny, time]
                # We need [batch, 1, Nx, Ny] for nufft, so permute
                image = data.permute(2, 0, 1).unsqueeze(1) # -> [time, 1, Nx, Ny]
                k = ktraj.permute(2, 0, 1) # -> [time, samples, 2]
                
                # Perform one batched operation
                x_temp = self.nufft_ob(image, k, smaps=smaps)
                # Output shape: [time, coils, samples]
                
                # Reshape back to [coils, samples, time]
                x = x_temp.permute(1, 2, 0) / np.sqrt(Nx * Ny)
        else:  # single frame (original logic is fine)
            ktraj = self._as_single_frame_ktraj(self.ktraj)
            dcomp = self._as_single_frame_dcomp(self.dcomp)
            if inv:
                kd = data.unsqueeze(0)
                d = dcomp.unsqueeze(0).unsqueeze(0)
                x = self.adjnufft_ob(kd * d, ktraj, smaps=smaps.to(dtype))
                x = torch.squeeze(x) / np.sqrt(Nx * Ny)
            else:
                image = data.unsqueeze(0).unsqueeze(0)
                x = self.nufft_ob(image, ktraj, smaps=smaps)
                x = torch.squeeze(x) / np.sqrt(Nx * Ny)
        return x
