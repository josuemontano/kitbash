# Vendored from TRELLIS (microsoft/TRELLIS, Copyright (c) Microsoft Corporation, MIT License) and Direct3D-S2
# (DreamTechAI/Direct3D-S2, Copyright (c) 2025 DreamTechAI, MIT License): the unified ``modules/sparse`` package.
# Modified for kitbash: a single pure-PyTorch backend (CUDA, MPS and CPU) replaces the spconv / torchsparse sparse
# convolutions, the flash_attn / xformers attention and the BACKEND / ATTN environment switches. The Triton
# spatial-sparse-attention of Direct3D-S2 is not used by any TriFlow network and is not vendored.
"""Sparse tensors, convolutions, attention and transformer blocks in plain PyTorch."""

from . import transformer
from .attention import (
    SerializeMode,
    SerializeModes,
    SparseMultiHeadAttention,
    sparse_scaled_dot_product_attention,
    sparse_serialized_scaled_dot_product_self_attention,
    sparse_windowed_scaled_dot_product_self_attention,
)
from .basic import SparseTensor, sparse_batch_broadcast, sparse_batch_op, sparse_cat, sparse_unbind
from .conv import SparseConv3d, SparseInverseConv3d
from .linear import SparseLinear
from .nonlinearity import SparseActivation, SparseGELU, SparseReLU, SparseSigmoid, SparseSiLU, SparseTanh
from .norm import SparseGroupNorm, SparseGroupNorm32, SparseLayerNorm, SparseLayerNorm32
from .spatial import SparseDownsample, SparseSubdivide, SparseUpsample
from .utils import sparse2sparse_tensor

__all__ = [
    "SerializeMode",
    "SerializeModes",
    "SparseActivation",
    "SparseConv3d",
    "SparseDownsample",
    "SparseGELU",
    "SparseGroupNorm",
    "SparseGroupNorm32",
    "SparseInverseConv3d",
    "SparseLayerNorm",
    "SparseLayerNorm32",
    "SparseLinear",
    "SparseMultiHeadAttention",
    "SparseReLU",
    "SparseSiLU",
    "SparseSigmoid",
    "SparseSubdivide",
    "SparseTanh",
    "SparseTensor",
    "SparseUpsample",
    "sparse2sparse_tensor",
    "sparse_batch_broadcast",
    "sparse_batch_op",
    "sparse_cat",
    "sparse_scaled_dot_product_attention",
    "sparse_serialized_scaled_dot_product_self_attention",
    "sparse_unbind",
    "sparse_windowed_scaled_dot_product_self_attention",
    "transformer",
]
