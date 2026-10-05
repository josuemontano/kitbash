# Vendored from TRELLIS (microsoft/TRELLIS, Copyright (c) Microsoft Corporation, MIT License) and Direct3D-S2
# (DreamTechAI/Direct3D-S2, Copyright (c) 2025 DreamTechAI, MIT License); the two `modules/sparse` packages are unified.

import torch.nn as nn

from .basic import SparseTensor

__all__ = [
    "SparseActivation",
    "SparseGELU",
    "SparseReLU",
    "SparseSiLU",
    "SparseSigmoid",
    "SparseTanh",
]


class SparseSigmoid(nn.Sigmoid):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return input.replace(super().forward(input.feats))


class SparseReLU(nn.ReLU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return input.replace(super().forward(input.feats))


class SparseSiLU(nn.SiLU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return input.replace(super().forward(input.feats))


class SparseGELU(nn.GELU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return input.replace(super().forward(input.feats))


class SparseTanh(nn.Tanh):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return input.replace(super().forward(input.feats))


class SparseActivation(nn.Module):
    def __init__(self, activation: nn.Module):
        super().__init__()
        self.activation = activation

    def forward(self, input: SparseTensor) -> SparseTensor:
        return input.replace(self.activation(input.feats))
