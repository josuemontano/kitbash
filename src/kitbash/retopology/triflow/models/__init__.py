# Copyright (c) 2026 Haoxuan Li.
# Licensed under the Automotive Development Public Non-Commercial License v1.0.
# See LICENSE for details.
#
# Modified for kitbash: the hydra configs (configs/model/*.yaml, configs/trainer/trellis_latent_slatflow_trainer.yaml and
# configs/direct3ds2_sparse_nvv_vae_512.yaml) are hard-coded here as constants.
"""The three TriFlow networks (SDF VAE, NVV VAE, shape-conditioned latent flow model) with their inference hyper-parameters."""

from typing import Any

import torch.nn as nn

from .direct3ds2_sparse_vae import Direct3ds2SparseVAE
from .trellis_shape_conditioned_slat_flow import ShapeConditionedSlatFlowModel

__all__ = [
    "FLOW_MODEL_CONFIG",
    "NVV_VAE_CONFIG",
    "SDF_VAE_CONFIG",
    "Direct3ds2SparseVAE",
    "ShapeConditionedSlatFlowModel",
    "build_flow_model",
    "build_nvv_vae",
    "build_sdf_vae",
]

# configs/model/direct3ds2_sparse_vae.yaml (the SDF VAE uses it as is)
SDF_VAE_CONFIG: dict[str, Any] = {
    "use_checkpoint": False,  # upstream: true (activation checkpointing is a training memory trick)
    "embed_dim": 16,
    "num_head_channels_encoder": 64,
    "model_channels_encoder": 512,
    "num_heads_encoder": 8,
    "num_blocks_encoder": 4,
    "num_head_channels_decoder": 64,
    "model_channels_decoder": 512,
    "num_heads_decoder": 8,
    "num_blocks_decoder": 4,
    "resolution": 64,
    "in_channels": 1,
    "out_channels": 1,
    "out_active": "tanh",
    "use_fp16": False,
    "latents_scale": 1.0,
    "latents_shift": 0.0,
    "use_nvv_decoder": True,
    "decoder_channel_down_factors": [4, 8, 16],
    "use_nvv_encoder": True,
    "encoder_channel_down_factors": [4, 8, 16],
}

# ... overridden by configs/direct3ds2_sparse_nvv_vae_512.yaml for the NVV VAE
NVV_VAE_CONFIG: dict[str, Any] = {
    **SDF_VAE_CONFIG,
    "in_channels": 14,
    "out_channels": 4,
    "model_channels_decoder": 1024,
    "decoder_channel_down_factors": [2, 4, 4],
    "attn_mode": "swin",
}

# configs/model/trellis_slatflow_shapecond.yaml
FLOW_MODEL_CONFIG: dict[str, Any] = {
    "slat_flow_config": {
        "resolution": 64,
        "in_channels": 16,
        "out_channels": 16,
        "model_channels": 768,
        "cond_channels": 768,
        "num_blocks": 12,
        "num_heads": 12,
        "mlp_ratio": 4,
        "patch_size": 2,
        "num_io_res_blocks": 2,
        "io_block_channels": [128],
        "pe_mode": "ape",
        "qk_rms_norm": True,
        "use_fp16": False,
    },
    "cond_face_count": True,
    "cond_quad_ratio": True,
    "cond_sdf_latent": True,
    "sdf_feat_dim": 16,
}


def _eval(model: nn.Module) -> nn.Module:
    return model.eval().requires_grad_(False)


def build_sdf_vae() -> Direct3ds2SparseVAE:
    """SDF VAE (``sdf_vae.safetensors``): ``encode`` turns a narrow-band SDF into the 16-channel latent used as condition."""
    return _eval(Direct3ds2SparseVAE(**SDF_VAE_CONFIG))


def build_nvv_vae() -> Direct3ds2SparseVAE:
    """NVV VAE (``nvv_vae.safetensors``): ``decoder(latent, fine_coords=...)`` turns a flow sample into NVV features."""
    return _eval(Direct3ds2SparseVAE(**NVV_VAE_CONFIG))


def build_flow_model() -> ShapeConditionedSlatFlowModel:
    """Latent flow model (``flow_model.safetensors``), conditioned on the SDF latent, face count and quad ratio."""
    return _eval(ShapeConditionedSlatFlowModel(**FLOW_MODEL_CONFIG))
