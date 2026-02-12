import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.utils.checkpoint import checkpoint

from radial_lsfp import from_torch_complex, to_torch_complex


def _require_selective_scan_fn():
    """Import mamba kernel lazily and fail with an actionable message."""
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    except Exception as exc:
        raise ImportError(
            "MambaRecon selected but mamba kernels are unavailable. "
            "Install `mamba-ssm` and `causal-conv1d` in this environment."
        ) from exc
    return selective_scan_fn


def _require_mamba_module():
    """Import Mamba block lazily and fail with an actionable message."""
    try:
        from mamba_ssm.modules.mamba_simple import Mamba
    except Exception as exc:
        raise ImportError(
            "Temporal Mamba selected but mamba modules are unavailable. "
            "Install `mamba-ssm` and `causal-conv1d` in this environment."
        ) from exc
    return Mamba


def _drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        return _drop_path(x, self.drop_prob, self.training)


class PatchEmbed2D(nn.Module):
    def __init__(self, patch_size=2, in_chans=2, embed_dim=128, norm_layer=None):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class Unpatchify(nn.Module):
    def __init__(self, dim, dim_scale=2):
        super().__init__()
        self.dim_scale = dim_scale
        self.layer = nn.Linear(dim, 2 * dim_scale**2, bias=False)

    def forward(self, x):
        x = self.layer(x)
        _, _, _, c = x.shape
        x = rearrange(
            x,
            "b h w (p1 p2 c) -> b (h p1) (w p2) c",
            p1=self.dim_scale,
            p2=self.dim_scale,
            c=c // (self.dim_scale**2),
        )
        return x.permute(0, 3, 1, 2)


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.selective_scan = _require_selective_scan_fn()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(
                self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs
            ),
            nn.Linear(
                self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs
            ),
            nn.Linear(
                self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs
            ),
            nn.Linear(
                self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs
            ),
        )
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in self.x_proj], dim=0)
        )
        del self.x_proj

        self.dt_projs = (
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
        )
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs], dim=0)
        )
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in self.dt_projs], dim=0)
        )
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    @staticmethod
    def dt_init(
        dt_rank,
        d_inner,
        dt_scale=1.0,
        dt_init="random",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        **factory_kwargs,
    ):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"Unsupported dt_init: {dt_init}")

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def _forward_core(self, x: torch.Tensor):
        b, _, h, w = x.shape
        length = h * w
        directions = 4

        x_hwwh = torch.stack(
            [
                x.view(b, -1, length),
                torch.transpose(x, dim0=2, dim1=3).contiguous().view(b, -1, length),
            ],
            dim=1,
        ).view(b, 2, -1, length)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l",
            xs.view(b, directions, -1, length),
            self.x_proj_weight,
        )
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum(
            "b k r l, k d r -> b k d l",
            dts.view(b, directions, -1, length),
            self.dt_projs_weight,
        )

        xs = xs.float().view(b, -1, length)
        dts = dts.contiguous().float().view(b, -1, length)
        Bs = Bs.float().view(b, directions, -1, length)
        Cs = Cs.float().view(b, directions, -1, length)

        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(b, directions, -1, length)

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(b, 2, -1, length)
        wh_y = torch.transpose(
            out_y[:, 1].view(b, -1, w, h), dim0=2, dim1=3
        ).contiguous().view(b, -1, length)
        invwh_y = torch.transpose(
            inv_y[:, 1].view(b, -1, w, h), dim0=2, dim1=3
        ).contiguous().view(b, -1, length)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(b, h, w, -1)
        y = self.out_norm(y).to(x.dtype)
        return y

    def forward(self, x: torch.Tensor):
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y = self._forward_core(x)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        drop_path: float = 0.0,
        norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
        attn_drop_rate: float = 0.0,
        d_state: int = 16,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(
            d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor):
        return x + self.drop_path(self.self_attention(self.ln_1(x)))


class VSSLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        attn_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
        use_checkpoint: bool = False,
        checkpoint_use_reentrant: bool = False,
        d_state: int = 16,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.checkpoint_use_reentrant = checkpoint_use_reentrant
        self.blocks = nn.ModuleList(
            [
                VSSBlock(
                    hidden_dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop,
                    d_state=d_state,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x, disable_checkpointing=False):
        use_ckpt = self.use_checkpoint and self.training and not disable_checkpointing
        for blk in self.blocks:
            if use_ckpt:
                x = checkpoint(blk, x, use_reentrant=self.checkpoint_use_reentrant)
            else:
                x = blk(x)
        return x


class TemporalMambaBlock(nn.Module):
    """Temporal mixer applied over frame axis per spatial token."""

    def __init__(
        self,
        hidden_dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_fast_path: bool = True,
        bidirectional: bool = True,
        use_relative_temporal_pos: bool = False,
        relative_temporal_pos_mode: str = "linear",
        relative_temporal_pos_scale: float = 1.0,
        use_spatial_mixer: bool = False,
        spatial_mixer_kernel_size: int = 3,
        drop_path: float = 0.0,
        norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
        use_checkpoint: bool = False,
        checkpoint_use_reentrant: bool = False,
        token_chunk_size: int = 0,
    ):
        super().__init__()
        Mamba = _require_mamba_module()
        self.norm = norm_layer(hidden_dim)
        self.bidirectional = bool(bidirectional)
        self.use_checkpoint = bool(use_checkpoint)
        self.checkpoint_use_reentrant = bool(checkpoint_use_reentrant)
        self.token_chunk_size = max(0, int(token_chunk_size))
        self.use_spatial_mixer = bool(use_spatial_mixer)
        self.use_relative_temporal_pos = bool(use_relative_temporal_pos)

        pos_mode = str(relative_temporal_pos_mode).strip().lower()
        if not self.use_relative_temporal_pos:
            pos_mode = "none"
        if pos_mode not in {"none", "linear", "linear_quadratic"}:
            raise ValueError(
                "relative_temporal_pos_mode must be one of: none, linear, linear_quadratic."
            )
        self.relative_temporal_pos_mode = pos_mode
        if self.relative_temporal_pos_mode == "none":
            self.temporal_pos_weight = None
            self.temporal_pos_scale = None
        else:
            feature_dim = 1 if self.relative_temporal_pos_mode == "linear" else 2
            self.temporal_pos_weight = nn.Parameter(
                torch.zeros(feature_dim, hidden_dim, dtype=torch.float32)
            )
            self.temporal_pos_scale = nn.Parameter(
                torch.tensor(float(relative_temporal_pos_scale), dtype=torch.float32)
            )

        self.mamba_fwd = Mamba(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=bool(use_fast_path),
        )
        if self.bidirectional:
            self.mamba_bwd = Mamba(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                use_fast_path=bool(use_fast_path),
            )
            self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.out_proj = nn.Identity()
        self.drop_path = DropPath(drop_path)
        if self.use_spatial_mixer:
            k = int(spatial_mixer_kernel_size)
            if k < 1 or k % 2 == 0:
                raise ValueError(
                    "spatial_mixer_kernel_size must be a positive odd integer."
                )
            self.spatial_norm = norm_layer(hidden_dim)
            self.spatial_dw = nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=k,
                padding=k // 2,
                groups=hidden_dim,
                bias=True,
            )
            self.spatial_pw = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=True)
            self.spatial_act = nn.SiLU()
            self.spatial_drop_path = DropPath(drop_path)
        else:
            self.spatial_norm = None
            self.spatial_dw = None
            self.spatial_pw = None
            self.spatial_act = None
            self.spatial_drop_path = None

    def _run_mamba(self, seq_chunk: torch.Tensor) -> torch.Tensor:
        # seq_chunk: (tokens, T, C)
        y_fwd = self.mamba_fwd(seq_chunk)
        if not self.bidirectional:
            return self.out_proj(y_fwd)

        seq_rev = torch.flip(seq_chunk, dims=[1])
        y_rev = torch.flip(self.mamba_bwd(seq_rev), dims=[1])
        return self.out_proj(torch.cat([y_fwd, y_rev], dim=-1))

    def _run_spatial_mixer(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, Hp, Wp, C)
        x = self.spatial_norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.spatial_dw(x)
        x = self.spatial_act(x)
        x = self.spatial_pw(x)
        return x.permute(0, 2, 3, 1).contiguous()

    def _apply_relative_temporal_pos(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, Hp, Wp, C)
        if self.relative_temporal_pos_mode == "none":
            return x

        t = int(x.shape[0])
        if t <= 1:
            rel = torch.zeros((t,), device=x.device, dtype=x.dtype)
        else:
            rel = torch.linspace(-1.0, 1.0, steps=t, device=x.device, dtype=x.dtype)

        if self.relative_temporal_pos_mode == "linear":
            feats = rel.unsqueeze(-1)
        else:
            feats = torch.stack((rel, rel * rel), dim=-1)

        weight = self.temporal_pos_weight.to(device=x.device, dtype=x.dtype)
        bias = torch.matmul(feats, weight)
        scale = self.temporal_pos_scale.to(device=x.device, dtype=x.dtype)
        return x + scale * bias[:, None, None, :]

    def forward(self, x: torch.Tensor, disable_checkpointing: bool = False) -> torch.Tensor:
        # x: (T, Hp, Wp, C)
        residual = x
        x = self.norm(x)
        x = self._apply_relative_temporal_pos(x)
        t, hp, wp, c = x.shape

        # Treat each spatial location as a separate sequence over time.
        seq = x.permute(1, 2, 0, 3).reshape(hp * wp, t, c).contiguous()
        seq_out = torch.empty_like(seq)

        chunk = self.token_chunk_size if self.token_chunk_size > 0 else seq.shape[0]
        use_ckpt = self.use_checkpoint and self.training and not disable_checkpointing

        for token_start in range(0, seq.shape[0], chunk):
            token_end = min(token_start + chunk, seq.shape[0])
            seq_chunk = seq[token_start:token_end]
            if use_ckpt:
                y_chunk = checkpoint(
                    self._run_mamba,
                    seq_chunk,
                    use_reentrant=self.checkpoint_use_reentrant,
                )
            else:
                y_chunk = self._run_mamba(seq_chunk)
            seq_out[token_start:token_end] = y_chunk

        x = seq_out.reshape(hp, wp, t, c).permute(2, 0, 1, 3).contiguous()
        x = residual + self.drop_path(x)

        if self.use_spatial_mixer:
            if use_ckpt:
                spatial_update = checkpoint(
                    self._run_spatial_mixer,
                    x,
                    use_reentrant=self.checkpoint_use_reentrant,
                )
            else:
                spatial_update = self._run_spatial_mixer(x)
            x = x + self.spatial_drop_path(spatial_update)

        return x


class TemporalMambaLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_fast_path: bool = True,
        bidirectional: bool = True,
        use_relative_temporal_pos: bool = False,
        relative_temporal_pos_mode: str = "linear",
        relative_temporal_pos_scale: float = 1.0,
        use_spatial_mixer: bool = False,
        spatial_mixer_kernel_size: int = 3,
        drop_path: float | list[float] = 0.0,
        norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
        use_checkpoint: bool = False,
        checkpoint_use_reentrant: bool = False,
        token_chunk_size: int = 0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TemporalMambaBlock(
                    hidden_dim=dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    use_fast_path=use_fast_path,
                    bidirectional=bidirectional,
                    use_relative_temporal_pos=use_relative_temporal_pos,
                    relative_temporal_pos_mode=relative_temporal_pos_mode,
                    relative_temporal_pos_scale=relative_temporal_pos_scale,
                    use_spatial_mixer=use_spatial_mixer,
                    spatial_mixer_kernel_size=spatial_mixer_kernel_size,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                    checkpoint_use_reentrant=checkpoint_use_reentrant,
                    token_chunk_size=token_chunk_size,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, disable_checkpointing: bool = False) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, disable_checkpointing=disable_checkpointing)
        return x


class RadialDataConsistency(nn.Module):
    """NUFFT-based data-consistency update in image space."""

    def __init__(
        self,
        num_of_feat_maps: int,
        patchify: bool,
        patch_size: int = 2,
        dc_step: float = 1.0,
        learnable_dc_step: bool = False,
        dc_step_min: float = 0.0,
        dc_step_max: float = 0.0,
    ):
        super().__init__()
        self.unpatchify = Unpatchify(num_of_feat_maps, dim_scale=patch_size)
        self.patchify = patchify
        self.learnable_dc_step = bool(learnable_dc_step)
        self.dc_step_min = float(dc_step_min)
        self.dc_step_max = float(dc_step_max)
        if self.dc_step_min < 0:
            raise ValueError("dc_step_min must be >= 0.")
        if self.dc_step_max > 0 and self.dc_step_max <= self.dc_step_min:
            raise ValueError("dc_step_max must be > dc_step_min when specified.")

        dc_step = float(dc_step)
        if self.learnable_dc_step:
            step_offset = dc_step - self.dc_step_min
            if step_offset <= 0:
                raise ValueError(
                    "Initial dc_step must be greater than dc_step_min when learnable_dc_step=true."
                )
            raw_init = math.log(math.expm1(step_offset))
            self.dc_step_raw = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))
            self.register_buffer(
                "dc_step_fixed", torch.tensor(dc_step, dtype=torch.float32), persistent=False
            )
        else:
            self.dc_step_raw = None
            self.register_buffer(
                "dc_step_fixed", torch.tensor(dc_step, dtype=torch.float32), persistent=False
            )

        if self.patchify:
            self.activation = nn.SiLU()
            self.patch_embed = PatchEmbed2D(
                patch_size=patch_size,
                in_chans=2,
                embed_dim=num_of_feat_maps,
                norm_layer=nn.LayerNorm,
            )
        else:
            self.activation = None
            self.patch_embed = None

    def _current_dc_step(self, ref_tensor: torch.Tensor) -> torch.Tensor:
        if self.learnable_dc_step:
            dc_step = F.softplus(self.dc_step_raw) + self.dc_step_min
            if self.dc_step_max > 0:
                dc_step = torch.clamp(dc_step, max=self.dc_step_max)
            return dc_step.to(device=ref_tensor.device, dtype=ref_tensor.real.dtype)
        return self.dc_step_fixed.to(device=ref_tensor.device, dtype=ref_tensor.real.dtype)

    def _data_cons_layer(self, im_btchw, measured_kspace, physics, coil_map):
        # (T,2,H,W) -> (H,W,T)
        im_complex = to_torch_complex(im_btchw)
        im_complex = rearrange(im_complex, "t h w -> h w t")

        pred_kspace = physics(inv=False, data=im_complex, smaps=coil_map)
        residual = measured_kspace - pred_kspace
        correction = physics(inv=True, data=residual, smaps=coil_map)
        dc_step = self._current_dc_step(im_complex)
        dc_im = im_complex + dc_step * correction

        # (H,W,T) -> (T,2,H,W)
        dc_im_btchw = from_torch_complex(rearrange(dc_im, "h w t -> t h w"))
        return dc_im_btchw

    def forward(self, x, measured_kspace, physics, coil_map):
        h = self.unpatchify(x)
        h = self._data_cons_layer(h, measured_kspace, physics, coil_map)
        if self.patchify:
            h = self.activation(h)
            h = self.patch_embed(h)
            return x + h
        return h


class MambaReconUnrolledRadial(nn.Module):
    def __init__(
        self,
        patch_size: int = 2,
        in_chans: int = 2,
        num_blocks: int = 6,
        block_depth: int = 2,
        embed_dim: int = 128,
        d_state: int = 16,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        checkpoint_use_reentrant: bool = False,
        dc_step: float = 1.0,
        dc_interval: int = 1,
        learnable_dc_step: bool = False,
        dc_step_min: float = 0.0,
        dc_step_max: float = 0.0,
    ):
        super().__init__()
        self.dc_interval = max(1, int(dc_interval))
        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm if patch_norm else None,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks * block_depth)]
        self.layers = nn.ModuleList()
        self.dc_layers = nn.ModuleList()
        self.dc_block_indices = []
        for i in range(num_blocks):
            dpr_slice = dpr[i * block_depth : (i + 1) * block_depth]
            self.layers.append(
                VSSLayer(
                    dim=embed_dim,
                    depth=block_depth,
                    d_state=d_state,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr_slice,
                    norm_layer=nn.LayerNorm,
                    use_checkpoint=use_checkpoint,
                    checkpoint_use_reentrant=checkpoint_use_reentrant,
                )
            )
            use_dc_here = ((i + 1) % self.dc_interval == 0) or (i == num_blocks - 1)
            if use_dc_here:
                self.dc_block_indices.append(i)
                self.dc_layers.append(
                    RadialDataConsistency(
                        embed_dim,
                        patchify=True,
                        patch_size=patch_size,
                        dc_step=dc_step,
                        learnable_dc_step=learnable_dc_step,
                        dc_step_min=dc_step_min,
                        dc_step_max=dc_step_max,
                    )
                )

        self.last_dc = RadialDataConsistency(
            embed_dim,
            patchify=False,
            patch_size=patch_size,
            dc_step=dc_step,
            learnable_dc_step=learnable_dc_step,
            dc_step_min=dc_step_min,
            dc_step_max=dc_step_max,
        )

    def forward(self, x_btchw, measured_kspace, physics, coil_map, disable_checkpointing=False):
        x = self.patch_embed(x_btchw)
        x = self.drop(x)
        dc_idx = 0
        for block_idx, layer in enumerate(self.layers):
            x = layer(x, disable_checkpointing=disable_checkpointing)
            if dc_idx < len(self.dc_block_indices) and block_idx == self.dc_block_indices[dc_idx]:
                x = self.dc_layers[dc_idx](x, measured_kspace, physics, coil_map)
                dc_idx += 1
        x = self.norm(x)
        return self.last_dc(x, measured_kspace, physics, coil_map)


class MambaTemporalUnrolledRadial(nn.Module):
    """Memory-aware temporal Mamba unrolled model for radial dynamic MRI."""

    def __init__(
        self,
        patch_size: int = 4,
        in_chans: int = 2,
        num_blocks: int = 6,
        block_depth: int = 2,
        embed_dim: int = 128,
        d_state: int = 16,
        temporal_d_conv: int = 4,
        temporal_expand: int = 2,
        temporal_bidirectional: bool = True,
        token_chunk_size: int = 1024,
        use_relative_temporal_pos: bool = False,
        relative_temporal_pos_mode: str = "linear",
        relative_temporal_pos_scale: float = 1.0,
        use_spatial_mixer: bool = False,
        spatial_mixer_kernel_size: int = 3,
        use_fast_path: bool = True,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        checkpoint_use_reentrant: bool = False,
        dc_step: float = 1.0,
        dc_interval: int = 1,
        learnable_dc_step: bool = False,
        dc_step_min: float = 0.0,
        dc_step_max: float = 0.0,
    ):
        super().__init__()
        self.dc_interval = max(1, int(dc_interval))
        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm if patch_norm else None,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks * block_depth)]
        self.layers = nn.ModuleList()
        self.dc_layers = nn.ModuleList()
        self.dc_block_indices = []
        for i in range(num_blocks):
            dpr_slice = dpr[i * block_depth : (i + 1) * block_depth]
            self.layers.append(
                TemporalMambaLayer(
                    dim=embed_dim,
                    depth=block_depth,
                    d_state=d_state,
                    d_conv=temporal_d_conv,
                    expand=temporal_expand,
                    use_fast_path=use_fast_path,
                    bidirectional=temporal_bidirectional,
                    use_relative_temporal_pos=use_relative_temporal_pos,
                    relative_temporal_pos_mode=relative_temporal_pos_mode,
                    relative_temporal_pos_scale=relative_temporal_pos_scale,
                    use_spatial_mixer=use_spatial_mixer,
                    spatial_mixer_kernel_size=spatial_mixer_kernel_size,
                    drop_path=dpr_slice,
                    norm_layer=nn.LayerNorm,
                    use_checkpoint=use_checkpoint,
                    checkpoint_use_reentrant=checkpoint_use_reentrant,
                    token_chunk_size=token_chunk_size,
                )
            )
            use_dc_here = ((i + 1) % self.dc_interval == 0) or (i == num_blocks - 1)
            if use_dc_here:
                self.dc_block_indices.append(i)
                self.dc_layers.append(
                    RadialDataConsistency(
                        embed_dim,
                        patchify=True,
                        patch_size=patch_size,
                        dc_step=dc_step,
                        learnable_dc_step=learnable_dc_step,
                        dc_step_min=dc_step_min,
                        dc_step_max=dc_step_max,
                    )
                )

        self.last_dc = RadialDataConsistency(
            embed_dim,
            patchify=False,
            patch_size=patch_size,
            dc_step=dc_step,
            learnable_dc_step=learnable_dc_step,
            dc_step_min=dc_step_min,
            dc_step_max=dc_step_max,
        )

    def forward(self, x_btchw, measured_kspace, physics, coil_map, disable_checkpointing=False):
        x = self.patch_embed(x_btchw)
        x = self.drop(x)
        dc_idx = 0
        for block_idx, layer in enumerate(self.layers):
            x = layer(x, disable_checkpointing=disable_checkpointing)
            if dc_idx < len(self.dc_block_indices) and block_idx == self.dc_block_indices[dc_idx]:
                x = self.dc_layers[dc_idx](x, measured_kspace, physics, coil_map)
                dc_idx += 1
        x = self.norm(x)
        return self.last_dc(x, measured_kspace, physics, coil_map)


class ArtifactRemovalMambaRecon(nn.Module):
    """Adapter to match LSFPNet forward contract used by train/inference code."""

    def __init__(
        self,
        backbone_net: nn.Module,
        predict_residual: bool = False,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        self.backbone_net = backbone_net
        self.predict_residual = bool(predict_residual)
        self.residual_scale = float(residual_scale)

    @staticmethod
    def _normalise_both(zf: torch.Tensor, data: torch.Tensor):
        scale = zf.abs().max() + 1e-8
        return zf / scale, data / scale, scale

    @staticmethod
    def _normalise_baseline(zf: torch.Tensor, data: torch.Tensor):
        scale = zf[..., 0].abs().mean() + 1e-8
        return zf / scale, data / scale, scale

    @staticmethod
    def _normalise_indep(x: torch.Tensor):
        scale = torch.quantile(x.abs(), 0.99) + 1e-6
        if scale < 1e-6:
            scale = torch.tensor(1.0, device=x.device, dtype=x.real.dtype)
        return x / scale, scale

    def forward(
        self,
        y,
        E,
        csmap,
        acceleration=None,
        start_timepoint_index=None,
        epoch=None,
        norm="both",
        disable_checkpointing=False,
        **kwargs,
    ):
        x_init = E(inv=True, data=y, smaps=csmap)

        if norm == "both":
            x_init_norm, y_norm, scale = self._normalise_both(x_init, y)
        elif norm == "independent":
            x_init_norm, scale = self._normalise_indep(x_init)
            y_norm, _ = self._normalise_indep(y)
        elif norm == "baseline":
            x_init_norm, y_norm, scale = self._normalise_baseline(x_init, y)
        elif norm == "none":
            x_init_norm = x_init
            y_norm = y
            scale = torch.tensor(1.0, device=x_init.device, dtype=x_init.real.dtype)
        else:
            raise ValueError(
                f"Unsupported normalization mode '{norm}'. Expected one of: both, independent, baseline, none."
            )

        x_init_btchw = from_torch_complex(rearrange(x_init_norm, "h w t -> t h w"))
        x_recon_btchw = self.backbone_net(
            x_init_btchw,
            y_norm,
            E,
            csmap,
            disable_checkpointing=disable_checkpointing,
        )
        if self.predict_residual:
            x_recon_btchw = x_init_btchw + self.residual_scale * x_recon_btchw

        recon = to_torch_complex(x_recon_btchw)
        recon = rearrange(recon, "t h w -> h w t") * scale
        x_hat = torch.stack((recon.real, recon.imag), dim=0).unsqueeze(0).to(torch.float32)

        # Keep output tuple compatible with LSFP callers.
        zero = torch.zeros((), device=x_hat.device, dtype=x_hat.dtype)
        return x_hat, zero, zero, zero, zero, zero, zero, zero
