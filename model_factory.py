"""Model factory helpers for constructing reconstruction networks. Run: imported by training/inference scripts (not intended to run directly)."""

from __future__ import annotations

from typing import Any

import torch


SUPPORTED_MODEL_NAMES = ("LSFPNet", "MambaRecon", "MambaTemporal")


def is_lsfp_model(model_name: str) -> bool:
    return str(model_name).strip().lower() == "lsfpnet"


def _parse_token_chunk_size(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"full", "all", "none", "auto"}:
            return 0
        return int(norm)
    return int(value)


def _build_lsfp_model(config: dict[str, Any], device: torch.device, block_dir: str):
    from lsfpnet_encoding import ArtifactRemovalLSFPNet, LSFPNet

    model_cfg = config["model"]
    lsfp_cfg = model_cfg.get("lsfpnet")
    if lsfp_cfg is None:
        lsfp_cfg = model_cfg
    elif not isinstance(lsfp_cfg, dict):
        raise TypeError(
            f"model.lsfpnet must be a mapping when provided, got {type(lsfp_cfg).__name__}."
        )

    initial_lambdas = {
        "lambda_L": lsfp_cfg["lambda_L"],
        "lambda_S": lsfp_cfg["lambda_S"],
        "lambda_spatial_L": lsfp_cfg["lambda_spatial_L"],
        "lambda_spatial_S": lsfp_cfg["lambda_spatial_S"],
        "gamma": lsfp_cfg["gamma"],
        "lambda_step": lsfp_cfg["lambda_step"],
    }

    lsfp_backbone = LSFPNet(
        LayerNo=lsfp_cfg["num_layers"],
        lambdas=initial_lambdas,
        channels=lsfp_cfg["channels"],
        style_dim=lsfp_cfg["style_dim"],
        svd_mode=lsfp_cfg["svd_mode"],
        use_lowk_dc=lsfp_cfg["use_lowk_dc"],
        lowk_frac=lsfp_cfg["lowk_frac"],
        lowk_alpha=lsfp_cfg["lowk_alpha"],
        film_bounded=lsfp_cfg["film_bounded"],
        film_gain=lsfp_cfg["film_gain"],
        film_identity_init=lsfp_cfg["film_identity_init"],
        svd_noise_std=lsfp_cfg["svd_noise_std"],
        film_L=lsfp_cfg["film_L"],
        kernel_size_L=lsfp_cfg.get("kernel_size_L", 3),
        kernel_size_S=lsfp_cfg.get("kernel_size_S", 3),
        activation_checkpointing=lsfp_cfg.get("activation_checkpointing", False),
        checkpoint_use_reentrant=lsfp_cfg.get("checkpoint_use_reentrant", False),
    )

    if model_cfg["encode_acceleration"] and model_cfg["encode_time_index"]:
        channels = 2
    else:
        channels = 1

    return ArtifactRemovalLSFPNet(lsfp_backbone, block_dir, channels=channels).to(device)


def _build_mamba_model(config: dict[str, Any], device: torch.device):
    from mamba_recon_arch import (
        ArtifactRemovalMambaRecon,
        MambaReconUnrolledRadial,
        MambaTemporalUnrolledRadial,
    )

    model_cfg = config["model"]
    if model_cfg.get("encode_acceleration", False) or model_cfg.get("encode_time_index", False):
        raise ValueError(
            "MambaRecon currently does not support acceleration/time-index encoding. "
            "Set `model.encode_acceleration: false` and `model.encode_time_index: false`."
        )

    mamba_cfg = model_cfg.get("mamba", {})
    variant = str(mamba_cfg.get("variant", "radial_2d")).strip().lower()
    if variant in {"temporal", "temporal_1d", "radial_temporal"}:
        backbone = MambaTemporalUnrolledRadial(
            patch_size=int(mamba_cfg.get("patch_size", 4)),
            in_chans=int(mamba_cfg.get("in_chans", 2)),
            num_blocks=int(mamba_cfg.get("num_blocks", 6)),
            block_depth=int(mamba_cfg.get("block_depth", 2)),
            embed_dim=int(mamba_cfg.get("hidden_dim", 128)),
            d_state=int(mamba_cfg.get("d_state", 16)),
            temporal_d_conv=int(mamba_cfg.get("temporal_d_conv", 4)),
            temporal_expand=int(mamba_cfg.get("temporal_expand", 2)),
            temporal_bidirectional=bool(mamba_cfg.get("temporal_bidirectional", True)),
            token_chunk_size=_parse_token_chunk_size(mamba_cfg.get("token_chunk_size", 1024)),
            use_relative_temporal_pos=bool(mamba_cfg.get("use_relative_temporal_pos", False)),
            relative_temporal_pos_mode=str(
                mamba_cfg.get("relative_temporal_pos_mode", "linear")
            ),
            relative_temporal_pos_scale=float(
                mamba_cfg.get("relative_temporal_pos_scale", 1.0)
            ),
            use_spatial_mixer=bool(mamba_cfg.get("use_spatial_mixer", False)),
            spatial_mixer_kernel_size=int(mamba_cfg.get("spatial_mixer_kernel_size", 3)),
            use_fast_path=bool(mamba_cfg.get("use_fast_path", True)),
            drop_rate=float(mamba_cfg.get("drop_rate", 0.0)),
            drop_path_rate=float(mamba_cfg.get("drop_path_rate", 0.0)),
            patch_norm=bool(mamba_cfg.get("patch_norm", True)),
            use_checkpoint=bool(mamba_cfg.get("use_checkpoint", False)),
            checkpoint_use_reentrant=bool(mamba_cfg.get("checkpoint_use_reentrant", False)),
            dc_step=float(mamba_cfg.get("dc_step", 1.0)),
            dc_interval=int(mamba_cfg.get("dc_interval", 1)),
            learnable_dc_step=bool(mamba_cfg.get("learnable_dc_step", False)),
            dc_step_min=float(mamba_cfg.get("dc_step_min", 0.0)),
            dc_step_max=float(mamba_cfg.get("dc_step_max", 0.0)),
        )
    else:
        backbone = MambaReconUnrolledRadial(
            patch_size=int(mamba_cfg.get("patch_size", 2)),
            in_chans=int(mamba_cfg.get("in_chans", 2)),
            num_blocks=int(mamba_cfg.get("num_blocks", 6)),
            block_depth=int(mamba_cfg.get("block_depth", 2)),
            embed_dim=int(mamba_cfg.get("hidden_dim", 128)),
            d_state=int(mamba_cfg.get("d_state", 16)),
            drop_rate=float(mamba_cfg.get("drop_rate", 0.0)),
            attn_drop_rate=float(mamba_cfg.get("attn_drop_rate", 0.0)),
            drop_path_rate=float(mamba_cfg.get("drop_path_rate", 0.0)),
            patch_norm=bool(mamba_cfg.get("patch_norm", True)),
            use_checkpoint=bool(mamba_cfg.get("use_checkpoint", False)),
            checkpoint_use_reentrant=bool(mamba_cfg.get("checkpoint_use_reentrant", False)),
            dc_step=float(mamba_cfg.get("dc_step", 1.0)),
            dc_interval=int(mamba_cfg.get("dc_interval", 1)),
            learnable_dc_step=bool(mamba_cfg.get("learnable_dc_step", False)),
            dc_step_min=float(mamba_cfg.get("dc_step_min", 0.0)),
            dc_step_max=float(mamba_cfg.get("dc_step_max", 0.0)),
        )
    return ArtifactRemovalMambaRecon(
        backbone,
        predict_residual=bool(mamba_cfg.get("predict_residual", False)),
        residual_scale=float(mamba_cfg.get("residual_scale", 1.0)),
    ).to(device)


def build_recon_model(config: dict[str, Any], device: torch.device, block_dir: str):
    model_name_raw = str(config["model"]["name"]).strip()
    model_name = model_name_raw.lower()
    if model_name == "lsfpnet":
        return _build_lsfp_model(config, device, block_dir)
    if model_name in {"mambatemporal", "mamba_temporal", "temporalmamba"}:
        cfg = dict(config)
        cfg_model = dict(cfg["model"])
        cfg_mamba = dict(cfg_model.get("mamba", {}))
        cfg_mamba.setdefault("variant", "temporal")
        cfg_model["mamba"] = cfg_mamba
        cfg["model"] = cfg_model
        return _build_mamba_model(cfg, device)
    if model_name in {"mambarecon", "mamba_recon", "mamba"}:
        return _build_mamba_model(config, device)
    raise ValueError(
        f"Unsupported model.name='{model_name_raw}'. Supported values: {', '.join(SUPPORTED_MODEL_NAMES)}."
    )
