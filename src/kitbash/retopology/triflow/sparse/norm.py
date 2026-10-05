# Vendored from TRELLIS (microsoft/TRELLIS, Copyright (c) Microsoft Corporation, MIT License) and Direct3D-S2
# (DreamTechAI/Direct3D-S2, Copyright (c) 2025 DreamTechAI, MIT License); the two `modules/sparse` packages are unified.
# Modified for kitbash: the DEBUG switch is gone and SparseLayerNorm normalises per voxel over the channels (upstream's
# per-batch reshape only worked when N == C).

import torch
import torch.nn as nn

from .basic import SparseTensor

__all__ = [
    "SparseGroupNorm",
    "SparseGroupNorm32",
    "SparseLayerNorm",
    "SparseLayerNorm32",
]


class SparseGroupNorm(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__(num_groups, num_channels, eps, affine)

    def forward(self, input: SparseTensor) -> SparseTensor:
        nfeats = torch.zeros_like(input.feats)
        for k in range(input.shape[0]):
            bfeats = input.feats[input.layout[k]]
            bfeats = bfeats.permute(1, 0).reshape(1, input.shape[1], -1)
            bfeats = super().forward(bfeats)
            bfeats = bfeats.reshape(input.shape[1], -1).permute(1, 0)
            nfeats[input.layout[k]] = bfeats
        return input.replace(nfeats)


class SparseLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, input: SparseTensor) -> SparseTensor:
        return input.replace(super().forward(input.feats))


class SparseGroupNorm32(SparseGroupNorm):
    """A GroupNorm layer that converts to float32 before the forward pass."""

    def forward(self, x: SparseTensor) -> SparseTensor:
        return super().forward(x.float()).type(x.dtype)


class SparseLayerNorm32(SparseLayerNorm):
    """A LayerNorm layer that converts to float32 before the forward pass."""

    def forward(self, x: SparseTensor) -> SparseTensor:
        return super().forward(x.float()).type(x.dtype)
