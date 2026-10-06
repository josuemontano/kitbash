# Vendored from TRELLIS (microsoft/TRELLIS, Copyright (c) Microsoft Corporation, MIT License) and Direct3D-S2
# (DreamTechAI/Direct3D-S2, Copyright (c) 2025 DreamTechAI, MIT License): the dense helpers the sparse modules rely on
# (modules/norm.py, modules/utils.py, AbsolutePositionEmbedder from modules/transformer/blocks.py, MultiHeadRMSNorm and
# RotaryPositionEmbedder from modules/attention/modules.py). The unused dense attention / transformer blocks are not vendored.
# Modified for kitbash: fp16 module conversion helpers are gone (use autocast on CUDA instead).

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "AbsolutePositionEmbedder",
    "ChannelLayerNorm32",
    "GroupNorm32",
    "LayerNorm32",
    "MultiHeadRMSNorm",
    "RotaryPositionEmbedder",
    "zero_module",
]


class LayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class GroupNorm32(nn.GroupNorm):
    """A GroupNorm layer that converts to float32 before the forward pass."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class ChannelLayerNorm32(LayerNorm32):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        DIM = x.dim()
        x = x.permute(0, *range(2, DIM), 1).contiguous()
        x = super().forward(x)
        return x.permute(0, DIM - 1, *range(1, DIM - 1)).contiguous()


def zero_module(module: nn.Module) -> nn.Module:
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


class AbsolutePositionEmbedder(nn.Module):
    """Embeds spatial positions into vector representations."""

    def __init__(self, channels: int, in_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.freq_dim = channels // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000**self.freqs)

    def _sin_cos_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Sinusoidal embeddings of a 1-D tensor of N indices -> (N, D)."""
        self.freqs = self.freqs.to(x.device)
        out = torch.outer(x, self.freqs)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=-1)

    def forward(self, x: torch.Tensor, factor: float | None = None) -> torch.Tensor:
        """
        Args:
            x: (N, D) tensor of spatial positions
            factor: optional position scale (Direct3D-S2)
        """
        N, D = x.shape
        assert self.in_channels == D, "Input dimension must match number of input channels"
        if factor is not None:
            x = x * factor
        embed = self._sin_cos_embedding(x.reshape(-1))
        embed = embed.reshape(N, -1)
        if embed.shape[1] < self.channels:
            embed = torch.cat([embed, torch.zeros(N, self.channels - embed.shape[1], device=embed.device)], dim=-1)
        return embed


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.normalize(x.float(), dim=-1) * self.gamma * self.scale).to(x.dtype)


class RotaryPositionEmbedder(nn.Module):
    def __init__(self, hidden_size: int, in_channels: int = 3):
        super().__init__()
        assert hidden_size % 2 == 0, "Hidden size must be divisible by 2"
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.freq_dim = hidden_size // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000**self.freqs)

    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        self.freqs = self.freqs.to(indices.device)
        phases = torch.outer(indices, self.freqs)
        return torch.polar(torch.ones_like(phases), phases)

    def _rotary_embedding(self, x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * phases
        return torch.view_as_real(x_rotated).reshape(*x_rotated.shape[:-1], -1).to(x.dtype)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, indices: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: [..., N, D] tensor of queries
            k: [..., N, D] tensor of keys
            indices: [..., N, C] tensor of spatial positions
        """
        if indices is None:
            indices = torch.arange(q.shape[-2], device=q.device)
            if len(q.shape) > 2:
                indices = indices.unsqueeze(0).expand((*q.shape[:-2], -1))

        phases = self._get_phases(indices.reshape(-1)).reshape(*indices.shape[:-1], -1)
        if phases.shape[1] < self.hidden_size // 2:
            pad = self.hidden_size // 2 - phases.shape[1]
            phases = torch.cat(
                [
                    phases,
                    torch.polar(
                        torch.ones(*phases.shape[:-1], pad, device=phases.device),
                        torch.zeros(*phases.shape[:-1], pad, device=phases.device),
                    ),
                ],
                dim=-1,
            )
        return self._rotary_embedding(q, phases), self._rotary_embedding(k, phases)
