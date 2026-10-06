# Vendored from TRELLIS (microsoft/TRELLIS, Copyright (c) Microsoft Corporation, MIT License) and Direct3D-S2
# (DreamTechAI/Direct3D-S2, Copyright (c) 2025 DreamTechAI, MIT License); the two `modules/sparse` packages are unified.
# Modified for kitbash: flash_attn / xformers are replaced by padded scaled_dot_product_attention (``varlen.py``); the
# window partition is cached on the coordinate set instead of the pyramid level.

import math

import torch

from ..basic import SparseTensor
from .varlen import varlen_self_attention

__all__ = [
    "sparse_windowed_scaled_dot_product_self_attention",
]


def calc_window_partition(
    tensor: SparseTensor,
    window_size: int | tuple[int, ...],
    shift_window: int | tuple[int, ...] = 0,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    """Serialize and partition a set of coordinates into (shifted) windows.

    Args:
        tensor: The input tensor.
        window_size: The window size to use.
        shift_window: The shift of serialized coordinates.

    Returns:
        Forwards indices, backwards indices, sequence lengths and sequence batch indices.
    """
    DIM = tensor.coords.shape[1] - 1
    shift_window = (shift_window,) * DIM if isinstance(shift_window, int) else shift_window
    window_size = (window_size,) * DIM if isinstance(window_size, int) else window_size
    shifted_coords = tensor.coords.clone().detach().long()
    shifted_coords[:, 1:] += torch.tensor(shift_window, device=tensor.device, dtype=torch.long).unsqueeze(0)

    MAX_COORDS = shifted_coords[:, 1:].max(dim=0).values.tolist()
    NUM_WINDOWS = [math.ceil((mc + 1) / ws) for mc, ws in zip(MAX_COORDS, window_size, strict=True)]
    OFFSET = torch.cumprod(torch.tensor([1, *NUM_WINDOWS[::-1]]), dim=0).tolist()[::-1]

    shifted_coords[:, 1:] //= torch.tensor(window_size, device=tensor.device, dtype=torch.long).unsqueeze(0)
    shifted_indices = (shifted_coords * torch.tensor(OFFSET, device=tensor.device, dtype=torch.long).unsqueeze(0)).sum(dim=1)
    fwd_indices = torch.argsort(shifted_indices)
    bwd_indices = torch.empty_like(fwd_indices)
    bwd_indices[fwd_indices] = torch.arange(fwd_indices.shape[0], device=tensor.device)
    seq_lens = torch.bincount(shifted_indices)
    seq_batch_indices = torch.arange(seq_lens.shape[0], device=tensor.device, dtype=torch.long) // OFFSET[0]
    mask = seq_lens != 0
    seq_lens = seq_lens[mask].tolist()
    seq_batch_indices = seq_batch_indices[mask].tolist()

    return fwd_indices, bwd_indices, seq_lens, seq_batch_indices


def sparse_windowed_scaled_dot_product_self_attention(
    qkv: SparseTensor,
    window_size: int,
    shift_window: tuple[int, int, int] | int = (0, 0, 0),
) -> SparseTensor:
    """Windowed scaled dot product self attention.

    Args:
        qkv: [N, *, 3, H, C] sparse tensor containing Qs, Ks, and Vs.
        window_size: The window size to use.
        shift_window: The shift of serialized coordinates.
    """
    assert len(qkv.shape) == 4 and qkv.shape[1] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"

    cache_name = f"window_partition_{window_size}_{shift_window}"
    cached = qkv._coord_cache.get(cache_name)
    if cached is None:
        cached = calc_window_partition(qkv, window_size, shift_window)
        qkv._coord_cache[cache_name] = cached
    fwd_indices, bwd_indices, seq_lens, _ = cached

    q, k, v = qkv.feats[fwd_indices].unbind(dim=1)  # [M, H, C]
    out = varlen_self_attention(q, k, v, seq_lens)
    return qkv.replace(out[bwd_indices])  # [T, H, C]
