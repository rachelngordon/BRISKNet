import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from lsp import Project_inf, Wxs, Wtxs
from time import time
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np

dtype = torch.complex64

def to_torch_complex(x: torch.Tensor):
    """(B, 2, ...) real -> (B, ...) complex"""
    assert x.shape[1] == 2, (
        f"Input tensor must have 2 channels (real, imag), but got shape {x.shape}"
    )
    return torch.view_as_complex(rearrange(x, "b c ... -> b ... c").contiguous())


def from_torch_complex(x: torch.Tensor):
    """(B, ...) complex -> (B, 2, ...) real"""
    return rearrange(torch.view_as_real(x), "b ... c -> b c ...").contiguous()


def _realify(a: torch.Tensor) -> torch.Tensor:
    """2x2 real block representation of complex matrix
    returns R(a) = [[Re a, -Im a], [Im a, Re a]] with shape (2m, 2n)
    """
    ar = torch.view_as_real(a)                              # (m, n, 2)
    x = ar[..., 0]                                          # Re
    y = ar[..., 1]                                          # Im
    top = torch.cat([x, -y], dim=-1)                        # (m, 2n)
    bot = torch.cat([y,  x], dim=-1)                        # (m, 2n)
    return torch.cat([top, bot], dim=-2)                    # (2m, 2n)

def _de_realify(r: torch.Tensor) -> torch.Tensor:
    """inverse of _realify; expects r with shape (2m, 2n) laid out as [[X, -Y], [Y, X]]"""
    m2, n2 = r.shape[-2], r.shape[-1]
    m, n = m2 // 2, n2 // 2
    x = r[..., :m, :n]
    y = r[..., m:, :n]
    return x + 1j * y


class MappingNetwork(nn.Module):
    """Maps a scalar input to a style vector using a simple MLP."""
    def __init__(self, style_dim, channels, num_layers=4):
        super().__init__()
        # We start with a linear layer to project the scalar to the style dimension
        # layers = [nn.Linear(1, style_dim), nn.ReLU(True)]
        layers = [nn.Linear(channels, style_dim), nn.ReLU(True)]
        # Add subsequent layers
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(style_dim, style_dim), nn.ReLU(True)])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Ensure input is 2D for the linear layer: [batch_size, 1]
        if x.dim() == 0:
            x = x.unsqueeze(0)
        if x.dim() == 1:
            x = x.unsqueeze(1)

        return self.net(x)


# new BasicBlock with FiLM bounding and configurable SVD methods
class BasicBlock(nn.Module):
    """
    one unrolled LS+S block with:
      - nuclear-norm prox via SVD (two modes: 'detached_uv' or 'mag')
      - positivity via softplus on all hyper-parameters
      - optional hard low-k projection inside the DC gradient
      - FiLM conditioning with identity init and bounded modulation

    Args:
        lambdas: dict with keys {'lambda_L','lambda_S','lambda_spatial_L','lambda_spatial_S','gamma','lambda_step'}
        channels: conv width
        style_dim: FiLM MLP latent dim
        svd_mode: 'detached_uv' (recommended) or 'mag' (real-SVD on |Z|)
        use_lowk_dc: if True, replace low-k residual with measured data inside gradient
        lowk_frac: fraction of radii treated as “low-k” (e.g., 0.10–0.15)
        lowk_alpha: blend for low-k (1.0 hard replace, 0.9 soft blend)
        film_bounded: if True, use tanh-bounded modulation; else raw scale+1, bias
        film_gain: magnitude of modulation when film_bounded=True
        film_identity_init: if True, FiLM heads are zero-initialized (identity)
        svd_mag_noise_std: optional noise std added to |Z| in 'mag' mode (0 means none)
    """
    def __init__(
        self,
        lambdas,
        channels=32,
        style_dim=128,
        svd_mode: str = "detached_uv",
        use_lowk_dc: bool = True,
        lowk_frac: float = 0.10,
        lowk_alpha: float = 0.7,
        film_bounded: bool = True,
        film_gain: float = 0.10,
        film_identity_init: bool = True,
        svd_noise_std: float = 0.0,
        film_L: bool = True,
        kernel_size_L=3,
        kernel_size_S=3,
    ):
        super().__init__()
        self.channels = channels
        self.style_dim = style_dim
        self.film_L = film_L
        self.lowk_alpha = lowk_alpha
        self.lowk_radius_frac = lowk_frac
        self.kernel_size_L = self._to_3tuple(kernel_size_L)
        self.kernel_size_S = self._to_3tuple(kernel_size_S)
        self.padding_L = tuple(k // 2 for k in self.kernel_size_L)
        self.padding_S = tuple(k // 2 for k in self.kernel_size_S)

        # learnable raw params; we will softplus them in forward
        self.lambda_L        = nn.Parameter(torch.tensor([lambdas['lambda_L']]))
        self.lambda_S        = nn.Parameter(torch.tensor([lambdas['lambda_S']]))
        self.lambda_spatial_L= nn.Parameter(torch.tensor([lambdas['lambda_spatial_L']]))
        self.lambda_spatial_S= nn.Parameter(torch.tensor([lambdas['lambda_spatial_S']]))
        self.gamma           = nn.Parameter(torch.tensor([lambdas['gamma']]))
        self.lambda_step     = nn.Parameter(torch.tensor([lambdas['lambda_step']]))

        # FiLM heads
        if self.film_L:
            self.style_injector_L = nn.Linear(self.style_dim, self.channels * 2)
        self.style_injector_S = nn.Linear(self.style_dim, self.channels * 2)
        if film_identity_init:
            if self.film_L:
                init.zeros_(self.style_injector_L.weight); init.zeros_(self.style_injector_L.bias)
            init.zeros_(self.style_injector_S.weight); init.zeros_(self.style_injector_S.bias)

        # 3D conv (real/imag packed as channel=2 at input)
        self.conv1_forward_l = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, 1, *self.kernel_size_L)))
        self.conv2_forward_l = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, self.channels, *self.kernel_size_L)))
        self.conv3_forward_l = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, self.channels, *self.kernel_size_L)))

        self.conv1_backward_l = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, self.channels, *self.kernel_size_L)))
        self.conv2_backward_l = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, self.channels, *self.kernel_size_L)))
        self.conv3_backward_l = nn.Parameter(init.xavier_normal_(torch.Tensor(1, self.channels, *self.kernel_size_L)))

        self.conv1_forward_s = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, 1, *self.kernel_size_S)))
        self.conv2_forward_s = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, self.channels, *self.kernel_size_S)))
        self.conv3_forward_s = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, self.channels, *self.kernel_size_S)))

        self.conv1_backward_s = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, self.channels, *self.kernel_size_S)))
        self.conv2_backward_s = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, self.channels, *self.kernel_size_S)))
        self.conv3_backward_s = nn.Parameter(init.xavier_normal_(torch.Tensor(1, self.channels, *self.kernel_size_S)))

        # runtime knobs for A/B
        self.svd_mode          = svd_mode
        self.use_lowk_dc       = use_lowk_dc
        self.lowk_frac         = lowk_frac
        self.lowk_alpha        = lowk_alpha
        self.film_bounded      = film_bounded
        self.film_gain         = film_gain
        self.svd_noise_std = svd_noise_std

    @staticmethod
    def _film(x, style_head, style_embedding, bounded: bool, gain: float):
        """apply FiLM modulation; identity at init; bounded if requested"""
        params = style_head(style_embedding)
        scale_raw, bias_raw = params.chunk(2, dim=-1)
        if bounded:
            scale = gain * torch.tanh(scale_raw)
            bias  = gain * torch.tanh(bias_raw)
            scale = scale.view(1, -1, 1, 1, 1)
            bias  = bias.view(1, -1, 1, 1, 1)
            return F.relu(x * (1.0 + scale) + bias)
        else:
            scale = scale_raw.view(1, -1, 1, 1, 1)
            bias  = bias_raw.view(1, -1, 1, 1, 1)
            return F.relu(x * (scale + 1.0) + bias)

    @staticmethod
    def _lowk_project(k_pred: torch.Tensor, y: torch.Tensor, ktraj: torch.Tensor, frac: float, alpha: float):
        """replace (or blend) low-k samples with measurements (vectorized, complex-safe)"""
        # ktraj: (2, S, T) -> radii (S, T)
        r = (ktraj[0]**2 + ktraj[1]**2).sqrt()
        thr = torch.quantile(r.reshape(-1), frac)
        M = (r <= thr)
        M = rearrange(M, 's t -> 1 s t')  # broadcast over coils
        return torch.where(M, alpha * y + (1.0 - alpha) * k_pred, k_pred)

    @staticmethod
    def _to_3tuple(k):
        if isinstance(k, int):
            return (k, k, k)
        if isinstance(k, (list, tuple)) and len(k) == 3:
            return tuple(int(v) for v in k)
        raise ValueError("kernel_size must be an int or a tuple/list of three ints.")

    def _compute_gradient(self, M0, param_E, param_d, L, S, csmaps):
        nx, ny, nt = M0.size()
        x_sum = torch.reshape(L + S, [nx, ny, nt])
        k_pred = param_E(inv=False, data=x_sum, smaps=csmaps)

        residual = k_pred - param_d

        if self.use_lowk_dc:
            k = param_E.ktraj.to(param_d.device)
            if k.dim() == 3:
                kr = (k[0]**2 + k[1]**2).sqrt()
                r0 = self.lowk_radius_frac * kr.max()
                mask = (kr <= r0).to(residual.real.dtype)
                gate = 1.0 - self.lowk_alpha * rearrange(mask, 's t -> 1 s t')
            else:
                kr = (k[0]**2 + k[1]**2).sqrt()
                r0 = self.lowk_radius_frac * kr.max()
                mask = (kr <= r0).to(residual.real.dtype)
                gate = 1.0 - self.lowk_alpha * rearrange(mask, 's -> 1 s 1')
            residual = residual * gate

        gradient = param_E(inv=True, data=residual, smaps=csmaps)
        return torch.reshape(gradient, [nx * ny, nt])

    def _forward_with_gradient(self, M0, gradient, L, S, pt_L, pt_S, p_L, p_S, style_embedding=None):
        """
        runs one LS+S iteration with a precomputed image-space gradient
        inputs are complex in (nx, ny, nt) except p_*, pt_* which are packed as in your code
        returns same tuple as your original implementation
        """

        # positivity + stable scaling
        gamma          = F.softplus(self.gamma) + 1e-6
        lambda_step    = F.softplus(self.lambda_step) + 1e-6
        lambda_L_eff   = F.softplus(self.lambda_L) + 1e-8
        lambda_S_eff   = F.softplus(self.lambda_S) + 1e-8
        lam_sp_L_eff   = F.softplus(self.lambda_spatial_L) + 1e-8
        lam_sp_S_eff   = F.softplus(self.lambda_spatial_S) + 1e-8
        c = lambda_step / gamma

        nx, ny, nt = M0.size()

        # ===== L branch ======================================================
        # conv backprop on p_L
        pb_L = F.conv3d(p_L, self.conv1_backward_l, padding=self.padding_L); pb_L = F.relu(pb_L)
        pb_L = F.conv3d(pb_L, self.conv2_backward_l, padding=self.padding_L); pb_L = F.relu(pb_L)
        pb_L = F.conv3d(pb_L, self.conv3_backward_l, padding=self.padding_L)
        pb_L = rearrange(pb_L.squeeze(), 'two nx ny nt -> two (nx ny) nt', nx=nx, ny=ny)  # already flattened
        pb_L = pb_L[0, :, :] + 1j * pb_L[1, :, :]

        y_L = L - gamma * gradient - gamma * pt_L - gamma * pb_L

        # prox for nuclear norm on complex matrix
        Z = c * y_L + pt_L  # (nx*ny, nt) complex

        if self.svd_mode == "detached_uv":
            U, Svals, Vh = torch.linalg.svd(Z, full_matrices=False)
            U_d, Vh_d = U.detach(), Vh.detach()
            S_shrunk = Project_inf(Svals, lambda_L_eff).to(U_d.dtype)
            pt_L = U_d @ torch.diag_embed(S_shrunk) @ Vh_d

        elif self.svd_mode == "mag":
            mag   = Z.abs() + 1e-8
            phase = Z / mag
            if self.svd_noise_std > 0.0:
                mag = mag + torch.randn_like(mag) * self.svd_noise_std
            U, Svals, Vh = torch.linalg.svd(mag, full_matrices=False)
            S_shrunk = Project_inf(Svals, lambda_L_eff, to_complex=False)
            pt_L_mag = U @ torch.diag_embed(S_shrunk) @ Vh
            pt_L = pt_L_mag * phase

        elif self.svd_mode == "real":
            R = _realify(Z)                                     # (2M, 2N) real
            print("noise std: ", self.svd_noise_std)
            if self.svd_noise_std > 0.0:
                print("adding noise...")
                R = R + self.svd_noise_std * torch.randn_like(R)
            Ur, Sr, VrT = torch.linalg.svd(R, full_matrices=False)
            # Sr_shrunk_clamp = torch.clamp(Sr - self.lambda_L, min=0.0)        # same tau as complex case
            Sr_shrunk = Project_inf(Sr, self.lambda_L, to_complex=False)        # same tau as complex case
            R_prox = Ur @ torch.diag_embed(Sr_shrunk) @ VrT
            pt_L = _de_realify(R_prox) 

        else:
            raise ValueError(f"unsupported svd_mode: {self.svd_mode}")

        # conv forward L (+ FiLM)
        # tL_in  = torch.cat((torch.real(y_L), torch.imag(y_L)), 0).to(torch.float32)
        tL_in = from_torch_complex(y_L)
        tL_in  = rearrange(tL_in, '(nx ny) two nt -> two 1 nx ny nt', nx=nx, ny=ny)
        tL     = F.conv3d(tL_in, self.conv1_forward_l, padding=self.padding_L); tL = F.relu(tL)
        tL     = F.conv3d(tL,    self.conv2_forward_l, padding=self.padding_L)
        if style_embedding is not None and self.film_L:
            tL = self._film(tL, self.style_injector_L, style_embedding, self.film_bounded, self.film_gain)
        else:
            tL = F.relu(tL)
        tL_out = F.conv3d(tL, self.conv3_forward_l, padding=self.padding_L)

        tL_out_c = tL_out + p_L
        tL_out_c = tL_out_c[0, :, :, :, :] + 1j * tL_out_c[1, :, :, :, :]
        p_L = Project_inf(c * tL_out_c, lam_sp_L_eff)

        # new pb_L
        # p_L = torch.cat((torch.real(p_L), torch.imag(p_L)), 0).to(torch.float32)
        p_L = from_torch_complex(p_L)
        p_L = rearrange(p_L, 'ch two nx ny nt -> two ch nx ny nt')
        pb_L = F.conv3d(p_L, self.conv1_backward_l, padding=self.padding_L); pb_L = F.relu(pb_L)
        pb_L = F.conv3d(pb_L, self.conv2_backward_l, padding=self.padding_L); pb_L = F.relu(pb_L)
        pb_L_out = F.conv3d(pb_L, self.conv3_backward_l, padding=self.padding_L)
        pb_L = rearrange(pb_L_out.squeeze(), 'two nx ny nt -> two (nx ny) nt', nx=nx, ny=ny)
        # pb_L = rearrange(pb_L_out, 'two (nx ny) nt -> two (nx ny) nt', nx=nx, ny=ny)
        pb_L = pb_L[0, :, :] + 1j * pb_L[1, :, :]

        L = L - gamma * gradient - gamma * pt_L - gamma * pb_L
        adjloss_L = tL_out * p_L - pb_L_out * tL_in

        # ===== S branch ======================================================
        pb_S = F.conv3d(p_S, self.conv1_backward_s, padding=self.padding_S); pb_S = F.relu(pb_S)
        pb_S = F.conv3d(pb_S, self.conv2_backward_s, padding=self.padding_S); pb_S = F.relu(pb_S)
        pb_S = F.conv3d(pb_S, self.conv3_backward_s, padding=self.padding_S)
        pb_S = rearrange(pb_S.squeeze(), 'two nx ny nt -> two (nx ny) nt', nx=nx, ny=ny)
        # pb_S = rearrange(pb_S, 'two (nx ny) nt -> two (nx ny) nt', nx=nx, ny=ny)
        pb_S = pb_S[0, :, :] + 1j * pb_S[1, :, :]

        y_S  = S - gamma * gradient - gamma * Wtxs(pt_S) - gamma * pb_S
        pt_S = Project_inf(c * Wxs(y_S) + pt_S, lambda_S_eff)

        # tS_in  = torch.cat((torch.real(y_S), torch.imag(y_S)), 0).to(torch.float32)
        tS_in = from_torch_complex(y_S)
        tS_in  = rearrange(tS_in, '(nx ny) two nt -> two 1 nx ny nt', nx=nx, ny=ny)
        tS     = F.conv3d(tS_in, self.conv1_forward_s, padding=self.padding_S); tS = F.relu(tS)
        tS     = F.conv3d(tS,    self.conv2_forward_s, padding=self.padding_S)
        if style_embedding is not None:
            tS = self._film(tS, self.style_injector_S, style_embedding, self.film_bounded, self.film_gain)
        else:
            tS = F.relu(tS)
        tS_out = F.conv3d(tS, self.conv3_forward_s, padding=self.padding_S)

        tS_out_c = tS_out + p_S
        tS_out_c = tS_out_c[0, :, :, :, :] + 1j * tS_out_c[1, :, :, :, :]
        p_S = Project_inf(c * tS_out_c, lam_sp_S_eff)

        # p_S = torch.cat((torch.real(p_S), torch.imag(p_S)), 0).to(torch.float32)
        p_S = from_torch_complex(p_S)
        p_S = rearrange(p_S, 'ch two nx ny nt -> two ch nx ny nt')
        pb_S = F.conv3d(p_S, self.conv1_backward_s, padding=self.padding_S); pb_S = F.relu(pb_S)
        pb_S = F.conv3d(pb_S, self.conv2_backward_s, padding=self.padding_S); pb_S = F.relu(pb_S)
        pb_S_out = F.conv3d(pb_S, self.conv3_backward_s, padding=self.padding_S)
        pb_S = rearrange(pb_S_out.squeeze(), 'two nx ny nt -> two (nx ny) nt', nx=nx, ny=ny)
        # pb_S = rearrange(pb_S_out, 'two (nx ny) nt -> two (nx ny) nt', nx=nx, ny=ny)
        pb_S = pb_S[0, :, :] + 1j * pb_S[1, :, :]

        S = S - gamma * gradient - gamma * Wtxs(pt_S) - gamma * pb_S
        adjloss_S = tS_out * p_S - pb_S_out * tS_in

        # return the positive (reparam’d) scalars for logging
        return [
            L, S, adjloss_L, adjloss_S, pt_L, pt_S, p_L, p_S,
            lambda_L_eff, lambda_S_eff, lam_sp_L_eff, lam_sp_S_eff, gamma, lambda_step
        ]

    def forward(self, M0, param_E, param_d, L, S, pt_L, pt_S, p_L, p_S, csmaps, style_embedding=None):
        gradient = self._compute_gradient(M0, param_E, param_d, L, S, csmaps)
        return self._forward_with_gradient(M0, gradient, L, S, pt_L, pt_S, p_L, p_S, style_embedding)
    


# define LSFP-Net Block
# class BasicBlock(nn.Module):
#     def __init__(self, lambdas, channels=32, style_dim=128):
#         super(BasicBlock, self).__init__()

#         self.channels = channels
#         self.style_dim = style_dim

#         self.lambda_L = nn.Parameter(torch.tensor([lambdas['lambda_L']]))
#         self.lambda_S = nn.Parameter(torch.tensor([lambdas['lambda_S']]))
#         self.lambda_spatial_L = nn.Parameter(torch.tensor([lambdas['lambda_spatial_L']]))
#         self.lambda_spatial_S = nn.Parameter(torch.tensor([lambdas['lambda_spatial_S']]))

#         self.gamma = nn.Parameter(torch.tensor([lambdas['gamma']]))
#         self.lambda_step = nn.Parameter(torch.tensor([lambdas['lambda_step']]))


#         # Linear layers to project style vector to scale and bias
#         self.style_injector_L = nn.Linear(self.style_dim, self.channels * 2) # *2 for scale and bias
#         self.style_injector_S = nn.Linear(self.style_dim, self.channels * 2)

#         # identity-init the FiLM so training starts from no modulation
#         nn.init.zeros_(self.style_injector_L.weight)
#         nn.init.zeros_(self.style_injector_L.bias)
#         nn.init.zeros_(self.style_injector_S.weight)
#         nn.init.zeros_(self.style_injector_S.bias)


#         self.conv1_forward_l = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, 1, 3, 3, 3)))
#         self.conv2_forward_l = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, self.channels, 3, 3, 3)))
#         self.conv3_forward_l = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, self.channels, 3, 3, 3)))

#         self.conv1_backward_l = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, self.channels, 3, 3, 3)))
#         self.conv2_backward_l = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, self.channels, 3, 3, 3)))
#         self.conv3_backward_l = nn.Parameter(init.xavier_normal_(torch.Tensor(1, self.channels, 3, 3, 3)))

#         self.conv1_forward_s = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, 1, 3, 3, 3)))
#         self.conv2_forward_s = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, self.channels, 3, 3, 3)))
#         self.conv3_forward_s = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, self.channels, 3, 3, 3)))

#         self.conv1_backward_s = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, self.channels, 3, 3, 3)))
#         self.conv2_backward_s = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, self.channels, 3, 3, 3)))
#         self.conv3_backward_s = nn.Parameter(init.xavier_normal_(torch.Tensor(1, self.channels, 3, 3, 3)))

#     def forward(self, M0, param_E, param_d, L, S, pt_L, pt_S, p_L, p_S, csmaps, style_embedding=None):

#         # print(f"Checking M0 for NaNs: {torch.isnan(M0).any().item()}")
#         # print(f"Checking L for NaNs: {torch.isnan(L).any().item()}")
#         # print(f"Checking S for NaNs: {torch.isnan(S).any().item()}")

#         c = self.lambda_step / self.gamma
#         nx, ny, nt = M0.size()

#         # gradient
#         temp_data = torch.reshape(L + S, [nx, ny, nt])
#         temp_data = param_E(inv=False, data=temp_data, smaps=csmaps).to(param_d.device)
#         gradient = param_E(inv=True, data=temp_data - param_d, smaps=csmaps)
#         gradient = torch.reshape(gradient, [nx * ny, nt]).to(param_d.device)

#         # pb_L
#         pb_L = F.conv3d(p_L, self.conv1_backward_l, padding=1)
#         pb_L = F.relu(pb_L)
#         pb_L = F.conv3d(pb_L, self.conv2_backward_l, padding=1)
#         pb_L = F.relu(pb_L)
#         pb_L = F.conv3d(pb_L, self.conv3_backward_l, padding=1)

#         pb_L = torch.reshape(torch.squeeze(pb_L), [2, nx * ny, nt])
#         pb_L = pb_L[0, :, :] + 1j * pb_L[1, :, :]

#         # y_L
#         y_L = L - self.gamma * gradient - self.gamma * pt_L - self.gamma * pb_L

#         # pt_L

#         # --- START OF THE ROBUST COMPLEX SVD FIX ---

#         # """
#         # real SVD proximal on the 2x2 block realification (exact, phase-safe)
#         # """
#         # svd_input = c * y_L + pt_L                                 # (nx*ny, nt) complex
#         # R = _realify(svd_input)                                     # (2M, 2N) real
#         # Ur, Sr, VrT = torch.linalg.svd(R, full_matrices=False)
#         # Sr_shrunk = torch.clamp(Sr - self.lambda_L, min=0.0)        # same tau as complex case
#         # R_prox = Ur @ torch.diag_embed(Sr_shrunk) @ VrT
#         # pt_L = _de_realify(R_prox)                                  # back to complex

#         svd_input = c * y_L + pt_L                                # (nx*ny, nt) complex
#         U, Svals, Vh = torch.linalg.svd(svd_input, full_matrices=False)
#         U_d = U.detach()                                           # break phase-dependent path
#         Vh_d = Vh.detach()
#         St_shrunk = Project_inf(Svals, self.lambda_L)#, to_complex=False)
#         pt_L = U_d @ torch.diag_embed(St_shrunk) @ Vh_d    

#         # svd_input_complex = c * y_L + pt_L

#         # # 1. Store the original magnitude and phase of the complex matrix.
#         # #    Add a small epsilon to the magnitude to prevent division by zero when calculating phase.
#         # svd_input_mag = svd_input_complex.abs() + 1e-8
#         # original_phase = svd_input_complex / svd_input_mag

#         # noise_std = 1e-3 # A small standard deviation for the noise
#         # noise = torch.randn_like(svd_input_mag) * noise_std

#         # stable_svd_input = svd_input_mag + noise #epsilon


#         # # # Assume 'stable_svd_input' is your original non-square matrix A that causes the error
#         # # # It has shape (m, n)
#         # # A = stable_svd_input

#         # # # 1. Choose a small regularization parameter
#         # # alpha = 1e-6 # This is a hyperparameter you can tune

#         # # # 2. Get the dimensions
#         # # m, n = A.shape
#         # # device = A.device
#         # # dtype = A.dtype

#         # # # 3. Create the identity matrix for augmentation. It must be n x n.
#         # # #    Note: We take the square root of alpha for the augmentation.
#         # # identity_aug = torch.sqrt(torch.tensor(alpha)) * torch.eye(n, device=device, dtype=dtype)

#         # # # 4. Stack the original matrix A on top of the scaled identity
#         # # A_aug = torch.cat([A, identity_aug], dim=0)


#         # # print(f"Checking 'some_tensor' for NaNs before SVD: {torch.isnan(stable_svd_input).any().item()}")


#         # # Right before your SVD call
#         # if torch.isnan(stable_svd_input).any() or torch.isinf(stable_svd_input).any():
#         #     print("!!! SVD input contains NaN or Inf values. Halting. !!!")
#         #     # You might want to save the tensor here for debugging
#         #     # torch.save(stable_svd_input, 'svd_input_error_tensor.pt')
#         #     # Or enter a debugger
#         #     import pdb; pdb.set_trace()

        
#         # # # 5. Perform SVD on the well-conditioned augmented matrix
#         # # #    This should now converge without an error.
#         # # U_aug, S, Vh = torch.linalg.svd(A_aug, full_matrices=False)

#         # # # The resulting S and Vh are the regularized singular values and right singular vectors you need.
#         # # # Note: U_aug corresponds to the augmented (m+n) x n matrix. If you need U for the
#         # # # original m x n matrix, you would typically only use the first m rows of U_aug.
#         # # Ut = U_aug[:m, :]
#         # # Vt = Vh
#         # # # St is just S, but you can give it the same name for consistency
#         # # St = S #torch.diag(S) # or just use the vector S depending on your needs


#         # # 2. Perform SVD on the REAL-VALUED magnitude matrix. 
#         # #    This completely avoids the complex svd_backward error.
#         # #    Using linalg.svd is fine here since the input is real.
#         # Ut, St, Vt = torch.linalg.svd(stable_svd_input, full_matrices=False)

#         #  # 3. Apply the shrinkage/thresholding to the real singular values.
#         # #    (Project_inf operates on magnitudes, so this is correct).
#         # St_shrunk = Project_inf(St, self.lambda_L, to_complex=False)

#         # # 4. Reconstruct the new, thresholded MAGNITUDE matrix.
#         # pt_L_mag = Ut @ torch.diag_embed(St_shrunk) @ Vt

#         # # 5. Re-apply the original phase to our new magnitude matrix to get the
#         # #    final complex-valued update term.
#         # pt_L = pt_L_mag * original_phase

#         # Ut, St, Vt = torch.linalg.svd((c * y_L + pt_L), full_matrices=False)
#         # temp_St = torch.diag(Project_inf(St, self.lambda_L))
#         # pt_L = Ut.mm(temp_St).mm(Vt)


#         # update p_L
#         temp_y_L_input = torch.cat((torch.real(y_L), torch.imag(y_L)), 0).to(torch.float32)
#         temp_y_L_input = torch.reshape(temp_y_L_input, [2, nx, ny, nt]).unsqueeze(1)
#         temp_y_L = F.conv3d(temp_y_L_input, self.conv1_forward_l, padding=1)
#         temp_y_L = F.relu(temp_y_L)
#         temp_y_L = F.conv3d(temp_y_L, self.conv2_forward_l, padding=1)

#         # if style_embedding is not None:
#         #     # print("encoding acceleration...")
#         #     # Inject style here
#         #     style_params_L = self.style_injector_L(style_embedding)
#         #     # Assuming style_embedding is [1, style_dim], params will be [1, channels * 2]
#         #     scale_L, bias_L = style_params_L.chunk(2, dim=-1) # Split into [1, channels] each

#         #     # Reshape for broadcasting over the feature map: [2, channels, nx, ny, nt]
#         #     # We apply the same style to real and imaginary parts.
#         #     scale_L = scale_L.view(1, self.channels, 1, 1, 1)
#         #     bias_L = bias_L.view(1, self.channels, 1, 1, 1)

#         #     # Modulate and then apply ReLU. Add 1 to scale to initialize near identity.
#         #     temp_y_L = F.relu(temp_y_L * (scale_L + 1) + bias_L)
#         # else: 
#         #     temp_y_L = F.relu(temp_y_L)

#         # L branch FiLM (bounded modulation; identity at init)
#         style_params_L = self.style_injector_L(style_embedding) if style_embedding is not None else None
#         if style_params_L is not None:
#             scale_L_raw, bias_L_raw = style_params_L.chunk(2, dim=-1)
#             # keep modulation small and well-behaved
#             scale_L = 0.1 * torch.tanh(scale_L_raw)
#             bias_L  = 0.1 * torch.tanh(bias_L_raw)

#             # reshape for broadcasting over (H, W, T)
#             scale_L = scale_L.view(-1, self.channels, 1, 1, 1)
#             bias_L  = bias_L.view(-1, self.channels, 1, 1, 1)

#             temp_y_L = F.relu(temp_y_L * (1.0 + scale_L) + bias_L)
#         else:
#             temp_y_L = F.relu(temp_y_L)
        

#         temp_y_L_output = F.conv3d(temp_y_L, self.conv3_forward_l, padding=1)

#         temp_y_L = temp_y_L_output + p_L
#         temp_y_L = temp_y_L[0, :, :, :, :] + 1j * temp_y_L[1, :, :, :, :]
#         p_L = Project_inf(c * temp_y_L, self.lambda_spatial_L)

#         # new pb_L
#         p_L = torch.cat((torch.real(p_L), torch.imag(p_L)), 0).to(torch.float32)
#         p_L = torch.reshape(p_L, [2, self.channels, nx, ny, nt])
#         pb_L = F.conv3d(p_L, self.conv1_backward_l, padding=1)
#         pb_L = F.relu(pb_L)
#         pb_L = F.conv3d(pb_L, self.conv2_backward_l, padding=1)
#         pb_L = F.relu(pb_L)
#         pb_L_output = F.conv3d(pb_L, self.conv3_backward_l, padding=1)

#         pb_L = torch.reshape(pb_L_output, [2, nx * ny, nt])
#         pb_L = pb_L[0, :, :] + 1j * pb_L[1, :, :]

#         # L
#         L = L - self.gamma * gradient - self.gamma * pt_L - self.gamma * pb_L

#         # adjoint loss: adjloss_L = psi * x * y - psi_t * y * x
#         adjloss_L = temp_y_L_output * p_L - pb_L_output * temp_y_L_input

#         # pb_S
#         pb_S = F.conv3d(p_S, self.conv1_backward_s, padding=1)
#         pb_S = F.relu(pb_S)
#         pb_S = F.conv3d(pb_S, self.conv2_backward_s, padding=1)
#         pb_S = F.relu(pb_S)
#         pb_S = F.conv3d(pb_S, self.conv3_backward_s, padding=1)

#         pb_S = torch.reshape(pb_S, [2, nx * ny, nt])
#         pb_S = pb_S[0, :, :] + 1j * pb_S[1, :, :]

#         # y_S
#         y_S = S - self.gamma * gradient - self.gamma * Wtxs(pt_S) - self.gamma * pb_S

#         # pt_S
#         pt_S = Project_inf(c * Wxs(y_S) + pt_S, self.lambda_S)

#         # update p_S
#         temp_y_S_input = torch.cat((torch.real(y_S), torch.imag(y_S)), 0).to(torch.float32)
#         temp_y_S_input = torch.reshape(temp_y_S_input, [2, nx, ny, nt]).unsqueeze(1)
#         temp_y_S = F.conv3d(temp_y_S_input, self.conv1_forward_s, padding=1)
#         temp_y_S = F.relu(temp_y_S)
#         temp_y_S = F.conv3d(temp_y_S, self.conv2_forward_s, padding=1)

#         # if style_embedding is not None:
#         #     # print("encoding acceleration...")
#         #     # Inject style here
#         #     style_params_S = self.style_injector_S(style_embedding)
#         #     scale_S, bias_S = style_params_S.chunk(2, dim=-1)
#         #     scale_S = scale_S.view(1, self.channels, 1, 1, 1)
#         #     bias_S = bias_S.view(1, self.channels, 1, 1, 1)

#         #     temp_y_S = F.relu(temp_y_S * (scale_S + 1) + bias_S)
#         # else:
#         #     temp_y_S = F.relu(temp_y_S)

#         # S branch FiLM (same idea)
#         style_params_S = self.style_injector_S(style_embedding) if style_embedding is not None else None
#         if style_params_S is not None:
#             scale_S_raw, bias_S_raw = style_params_S.chunk(2, dim=-1)
#             scale_S = 0.1 * torch.tanh(scale_S_raw)
#             bias_S  = 0.1 * torch.tanh(bias_S_raw)

#             # reshape for broadcasting over (H, W, T)
#             scale_S = scale_S.view(-1, self.channels, 1, 1, 1)
#             bias_S  = bias_S.view(-1, self.channels, 1, 1, 1)

#             temp_y_S = F.relu(temp_y_S * (1.0 + scale_S) + bias_S)
#         else:
#             temp_y_S = F.relu(temp_y_S)


#         temp_y_S_output = F.conv3d(temp_y_S, self.conv3_forward_s, padding=1)

#         temp_y_Sp = temp_y_S_output + p_S
#         temp_y_Sp = temp_y_Sp[0, :, :, :, :] + 1j * temp_y_Sp[1, :, :, :, :]
#         p_S = Project_inf(c * temp_y_Sp, self.lambda_spatial_S)

#         # new pb_S
#         p_S = torch.cat((torch.real(p_S), torch.imag(p_S)), 0).to(torch.float32)
#         p_S = torch.reshape(p_S, [2, self.channels, nx, ny, nt])
#         pb_S = F.conv3d(p_S, self.conv1_backward_s, padding=1)
#         pb_S = F.relu(pb_S)
#         pb_S = F.conv3d(pb_S, self.conv2_backward_s, padding=1)
#         pb_S = F.relu(pb_S)
#         pb_S_output = F.conv3d(pb_S, self.conv3_backward_s, padding=1)

#         pb_S = torch.reshape(pb_S_output, [2, nx * ny, nt])
#         pb_S = pb_S[0, :, :] + 1j * pb_S[1, :, :]

#         # S
#         S = S - self.gamma * gradient - self.gamma * Wtxs(pt_S) - self.gamma * pb_S

#         # adjoint loss: adjloss_S = psi * x * y - psi_t * y * x
#         adjloss_S = temp_y_S_output * p_S - pb_S_output * temp_y_S_input

#         return [L, S, adjloss_L, adjloss_S, pt_L, pt_S, p_L, p_S, self.lambda_L, self.lambda_S, self.lambda_spatial_L, self.lambda_spatial_S, self.gamma, self.lambda_step]


# define LSFP-Net
class LSFPNet(nn.Module):
    def __init__(self, 
                 LayerNo: int, 
                 lambdas: dict, 
                 channels: int = 32, 
                 style_dim: int = 128,
                 svd_mode: str = "detached_uv",
                 use_lowk_dc: bool = True,
                 lowk_frac: float = 0.125,
                 lowk_alpha: float = 1.0,
                 film_bounded: bool = True,
                 film_gain: float = 0.10,
                 film_identity_init: bool = True,
                 svd_noise_std: float = 0.0,
                 film_L: bool = True,
                 kernel_size_L=3,
                 kernel_size_S=3,
                 activation_checkpointing: bool = False,
                 checkpoint_use_reentrant: bool = False,
        ):
        super(LSFPNet, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        self.channels = channels
        self.style_dim = style_dim
        self.activation_checkpointing = activation_checkpointing
        self.checkpoint_use_reentrant = checkpoint_use_reentrant

        for ii in range(LayerNo):
            # onelayer.append(BasicBlock(lambdas, channels=self.channels, style_dim=style_dim))
            onelayer.append(BasicBlock(lambdas=lambdas, 
                                       channels=self.channels, 
                                       style_dim=style_dim,
                                       svd_mode=svd_mode,
                                       use_lowk_dc=use_lowk_dc,
                                       lowk_frac=lowk_frac,
                                       lowk_alpha=lowk_alpha,
                                       film_bounded=film_bounded,
                                       film_gain=film_gain,
                                       film_identity_init=film_identity_init,
                                       svd_noise_std=svd_noise_std,
                                       film_L=film_L,
                                       kernel_size_L=kernel_size_L,
                                       kernel_size_S=kernel_size_S,
                                       ))

        self.fcs = nn.ModuleList(onelayer)

    
    def plot_block_output(self, M0, L, S, iter, epoch, output_dir):

        time_frame_index = 3

        nx, ny, nt = M0.size()
        L = torch.reshape(L, [nx, ny, nt])
        S = torch.reshape(S, [nx, ny, nt])

        output_image = L + S

        # Create a 1x3 plot grid
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        fig.suptitle(f"Basic Block Output at Time Frame {time_frame_index} and Iteration {iter}", fontsize=20)

        # --- Top Row: DL Reconstruction Comparison ---
        axes[0].imshow(np.abs(M0[..., time_frame_index].cpu().detach().numpy()), cmap='gray')
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        axes[1].imshow(np.abs(L[..., time_frame_index].cpu().detach().numpy()), cmap='gray')
        axes[1].set_title("Background Component (L)")
        axes[1].axis("off")

        axes[2].imshow(np.abs(S[..., time_frame_index].cpu().detach().numpy()), cmap='gray')
        axes[2].set_title("Dynamic Component (S)")
        axes[2].axis("off")

        axes[3].imshow(np.abs(output_image[..., time_frame_index].cpu().detach().numpy()), cmap='gray')
        axes[3].set_title("Combined Image (L + S)")
        axes[3].axis("off")

        filename = os.path.join(output_dir, f'basic_block_output_{epoch}_iter{iter}.png')
        plt.savefig(filename)
        plt.close()


    def forward(self, M0, param_E, param_d, csmap, epoch, output_dir, style_embedding=None, disable_checkpointing=False):

        # M0 = M0[..., 0] + 1j * M0[..., 1]
        # param_d = param_d[..., 0] + 1j * param_d[..., 1]

        nx, ny, nt = M0.size()
        L = torch.zeros([nx * ny, nt], dtype=dtype).to(param_d.device)
        S = torch.zeros([nx * ny, nt], dtype=dtype).to(param_d.device)
        pt_L = torch.zeros([nx * ny, nt], dtype=dtype).to(param_d.device)
        pt_S = torch.zeros([nx * ny, nt], dtype=dtype).to(param_d.device)
        p_L = torch.zeros([2, self.channels, nx, ny, nt], dtype=torch.float32).to(param_d.device)
        p_S = torch.zeros([2, self.channels, nx, ny, nt], dtype=torch.float32).to(param_d.device)

        layers_adj_L = []
        layers_adj_S = []

        use_checkpointing = self.activation_checkpointing and self.training and not disable_checkpointing
        checkpoint_dummy = None
        if use_checkpointing:
            checkpoint_dummy = torch.ones(1, device=param_d.device, requires_grad=True)

        for ii in range(self.LayerNo):
            if use_checkpointing:
                block = self.fcs[ii]
                gradient = block._compute_gradient(M0, param_E, param_d, L, S, csmap)

                def _run_block(L_t, S_t, pt_L_t, pt_S_t, p_L_t, p_S_t, grad_t, dummy_t, _block=block):
                    return tuple(_block._forward_with_gradient(M0, grad_t, L_t, S_t, pt_L_t, pt_S_t, p_L_t, p_S_t, style_embedding))

                (L, S, layer_adj_L, layer_adj_S, pt_L, pt_S, p_L, p_S, lambda_L, lambda_S,
                 lambda_spatial_L, lambda_spatial_S, gamma, lambda_step) = checkpoint(
                    _run_block, L, S, pt_L, pt_S, p_L, p_S, gradient, checkpoint_dummy,
                    use_reentrant=self.checkpoint_use_reentrant
                )
            else:
                (L, S, layer_adj_L, layer_adj_S, pt_L, pt_S, p_L, p_S, lambda_L, lambda_S,
                 lambda_spatial_L, lambda_spatial_S, gamma, lambda_step) = self.fcs[ii](
                    M0, param_E, param_d, L, S, pt_L, pt_S, p_L, p_S, csmap, style_embedding
                )
            layers_adj_L.append(layer_adj_L)
            layers_adj_S.append(layer_adj_S)

            # if epoch == "val10" or epoch == "train10":
            self.plot_block_output(M0, L, S, iter=ii, epoch=epoch, output_dir=output_dir)
                

        L = torch.reshape(L, [nx, ny, nt])
        S = torch.reshape(S, [nx, ny, nt])

        return [L, S, layers_adj_L, layers_adj_S, lambda_L, lambda_S, lambda_spatial_L, lambda_spatial_S, gamma, lambda_step]
    

class ArtifactRemovalLSFPNet(nn.Module):
    def __init__(self, backbone_net, output_dir, channels, **kwargs):
        super(ArtifactRemovalLSFPNet, self).__init__()
        self.backbone_net = backbone_net
        self.output_dir = output_dir

        # Define the style dimension and instantiate the mapping network
        self.style_dim = 128  # You can tune this hyperparameter
        self.mapping_network = MappingNetwork(style_dim=self.style_dim, channels=channels)

    @staticmethod
    def _normalise_both(zf: torch.Tensor, data: torch.Tensor):
        """
        Per-dynamic-series max-magnitude scaling (paper default).
        Both `zf` (image) and `data` (k-space) share the SAME scalar.
        """
        scale = zf.abs().max() + 1e-8                     # scalar, grads OK
        return zf / scale, data / scale, scale
    

    @staticmethod
    def _normalise_baseline(zf: torch.Tensor, data: torch.Tensor):
        """
        Per-dynamic-series max-magnitude scaling (paper default).
        Both `zf` (image) and `data` (k-space) share the SAME scalar.
        """
        scale = zf[..., 0].abs().mean() + 1e-8                     # scalar, grads OK
        return zf / scale, data / scale, scale
    
    
    @staticmethod
    def _normalise_indep(x: torch.Tensor):
        """
        Per-dynamic-series max-magnitude scaling (paper default).
        Both `zf` (image) and `data` (k-space) share the SAME scalar.
        """

        scale = torch.quantile(x.abs(), 0.99) + 1e-6
        if scale < 1e-6: # Handle case where input is all zeros
             scale = 1.0
        return x / scale, scale

    def forward(self, y, E, csmap, acceleration=None, start_timepoint_index=None, epoch=None, norm="both", disable_checkpointing=False, **kwargs):

        # 1. Get the initial ZF recon. This defines our target energy/scale.
        x_init = E(inv=True, data=y, smaps=csmap)

        if norm =="both":
            x_init_norm, y_norm, scale = self._normalise_both(x_init, y)
        elif norm == "independent":
            # 2. Normalize the image and k-space INDEPENDENTLY.
            x_init_norm, scale = self._normalise_indep(x_init)
            y_norm, scale_y = self._normalise_indep(y)
        elif norm == "baseline":
            x_init_norm, y_norm, scale = self._normalise_baseline(x_init, y)
        elif norm == "none":
            x_init_norm = x_init
            y_norm = y
            scale = 1.0

        # Generate style embedding from the acceleration factor and/or time start index
        if acceleration or start_timepoint_index:                                          # already small

            if start_timepoint_index is not None:
                # feature 2: start index as fraction of total frames
                T = x_init_norm.shape[-1]
                start_frac = (start_timepoint_index / max(T - 1, 1)).view(-1, 1)

                if acceleration is not None:
                    # feature 1: inv acceleration (≈ spf / (H*pi/2)), roughly in [~0.02, ~0.07]
                    H = x_init_norm.shape[-2]
                    N_full = H * np.pi / 2.0
                    inv_af = (1.0 / acceleration.clamp_min(1e-6)).view(-1, 1)          # smaller numbers are safer
                    spf_est = (N_full / acceleration.clamp_min(1e-6)).view(-1, 1)      # useful too
                    inv_af_feat = inv_af   

                    # concatenate normalized features
                    combined_input = torch.cat([inv_af_feat, start_frac], dim=1).to(x_init_norm.device)
                    # combined_input = torch.cat(
                    #     (acceleration.float().view(-1, 1), start_timepoint_index.float().view(-1, 1)),
                    #     dim=1
                    # )

                else:
                    # combined_input = start_timepoint_index
                    combined_input = start_frac
            else:
                # combined_input = acceleration
                # feature 1: inv acceleration (≈ spf / (H*pi/2)), roughly in [~0.02, ~0.07]
                H = x_init_norm.shape[-2]
                N_full = H * np.pi / 2.0
                inv_af = (1.0 / acceleration.clamp_min(1e-6)).view(-1, 1)          # smaller numbers are safer
                spf_est = (N_full / acceleration.clamp_min(1e-6)).view(-1, 1)      # useful too
                inv_af_feat = inv_af   
                combined_input = inv_af_feat

            style_embedding = self.mapping_network(combined_input)

            L, S, loss_layers_adj_L, loss_layers_adj_S, lambda_L, lambda_S, lambda_spatial_L, lambda_spatial_S, gamma, lambda_step  = self.backbone_net(
                x_init_norm,
                E,
                y_norm,
                csmap,
                epoch,
                self.output_dir,
                style_embedding,
                disable_checkpointing=disable_checkpointing,
            )

        else:
            L, S, loss_layers_adj_L, loss_layers_adj_S, lambda_L, lambda_S, lambda_spatial_L, lambda_spatial_S, gamma, lambda_step  = self.backbone_net(
                x_init_norm,
                E,
                y_norm,
                csmap,
                epoch,
                self.output_dir,
                None,
                disable_checkpointing=disable_checkpointing,
            )

        loss_constraint_L = torch.square(torch.mean(loss_layers_adj_L[0])) / self.backbone_net.LayerNo
        loss_constraint_S = torch.square(torch.mean(loss_layers_adj_S[0])) / self.backbone_net.LayerNo

        for k in range(self.backbone_net.LayerNo - 1):
            loss_constraint_S += torch.square(torch.mean(loss_layers_adj_S[k + 1])) / self.backbone_net.LayerNo
            loss_constraint_L += torch.square(torch.mean(loss_layers_adj_L[k + 1])) / self.backbone_net.LayerNo


        recon = (L + S) * scale                 # rescale to original units

        # 4) stack & convert back to (B,2,T,H,W) float32
        x_hat = torch.stack((recon.real, recon.imag), dim=0).unsqueeze(0)  # (B,2,H,W,T)

        return x_hat, loss_constraint_L + loss_constraint_S, lambda_L, lambda_S, lambda_spatial_L, lambda_spatial_S, gamma, lambda_step
