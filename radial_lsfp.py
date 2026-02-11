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
        self.ktraj = torch.squeeze(ktraj)
        self.dcomp = torch.squeeze(dcomp)

    def forward(self, inv, data, smaps):
        # Squeeze is fine, but let's get original shapes
        data = torch.squeeze(data)
        Nx, Ny = smaps.shape[2], smaps.shape[3]

        if len(data.shape) > 2:  # multi-frame
            is_complex_data = torch.is_complex(data)
            
            # --- Vectorized approach ---
            if inv: # Adjoint NUFFT (k-space -> image)
                # Original shape: [coils, samples, time]
                # We need [batch, coils, samples] for nufft, so permute time to batch
                kd = data.permute(2, 0, 1) # -> [time, coils, samples]
                d = self.dcomp.permute(1, 0) # -> [time, samples]
                k = self.ktraj.permute(2, 0, 1) # -> [time, samples, 2]

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
                k = self.ktraj.permute(2, 0, 1) # -> [time, samples, 2]
                
                # Perform one batched operation
                x_temp = self.nufft_ob(image, k, smaps=smaps)
                # Output shape: [time, coils, samples]
                
                # Reshape back to [coils, samples, time]
                x = x_temp.permute(1, 2, 0) / np.sqrt(Nx * Ny)
        else:  # single frame (original logic is fine)
            if inv:
                kd = data.unsqueeze(0)
                d = self.dcomp.unsqueeze(0).unsqueeze(0)
                x = self.adjnufft_ob(kd * d, self.ktraj, smaps=smaps.to(dtype))
                x = torch.squeeze(x) / np.sqrt(Nx * Ny)
            else:
                image = data.unsqueeze(0).unsqueeze(0)
                x = self.nufft_ob(image, self.ktraj, smaps=smaps)
                x = torch.squeeze(x) / np.sqrt(Nx * Ny)
        return x
