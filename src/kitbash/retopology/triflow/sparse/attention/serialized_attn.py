# Vendored from TRELLIS (microsoft/TRELLIS, Copyright (c) Microsoft Corporation, MIT License) and Direct3D-S2
# (DreamTechAI/Direct3D-S2, Copyright (c) 2025 DreamTechAI, MIT License); the two `modules/sparse` packages are unified.
# Modified for kitbash: flash_attn / xformers are replaced by padded scaled_dot_product_attention (``varlen.py``) and the
# vox2seq CUDA extension by a torch Z-order encoder. Hilbert orders are not ported. None of the TriFlow networks use
# serialized attention (they use "swin" windows and "full" attention).

import math
from enum import Enum

import torch

from ..basic import SparseTensor
from .varlen import varlen_self_attention

__all__ = [
    "sparse_serialized_scaled_dot_product_self_attention",
]


class SerializeMode(Enum):
    Z_ORDER = 0
    Z_ORDER_TRANSPOSED = 1
    HILBERT = 2
    HILBERT_TRANSPOSED = 3


SerializeModes = [
    SerializeMode.Z_ORDER,
    SerializeMode.Z_ORDER_TRANSPOSED,
    SerializeMode.HILBERT,
    SerializeMode.HILBERT_TRANSPOSED,
]


def _z_order_encode(coords: torch.Tensor, permute: list[int]) -> torch.Tensor:
    """30-bit Morton code of ``(N, 3)`` coordinates (10 bits per axis), axes taken in the order ``permute``."""
    xyz = coords[:, permute].long()
    code = torch.zeros(xyz.shape[0], dtype=torch.long, device=xyz.device)
    for bit in range(10):
        for axis in range(3):
            code |= ((xyz[:, axis] >> bit) & 1) << (3 * bit + 2 - axis)
    return code


def calc_serialization(
    tensor: SparseTensor,
    window_size: int,
    serialize_mode: SerializeMode = SerializeMode.Z_ORDER,
    shift_sequence: int = 0,
    shift_window: tuple[int, int, int] = (0, 0, 0),
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    """Serialize and partition a set of coordinates into windows of ``window_size`` along a space-filling curve.

    Returns:
        Forwards indices, backwards indices, sequence lengths and sequence batch indices.
    """
    fwd_indices = []
    bwd_indices = []
    seq_lens = []
    seq_batch_indices = []
    offsets = [0]

    serialize_coords = tensor.coords[:, 1:].clone()
    serialize_coords += torch.tensor(shift_window, dtype=serialize_coords.dtype, device=tensor.device).reshape(1, 3)
    if serialize_mode == SerializeMode.Z_ORDER:
        code = _z_order_encode(serialize_coords, [0, 1, 2])
    elif serialize_mode == SerializeMode.Z_ORDER_TRANSPOSED:
        code = _z_order_encode(serialize_coords, [1, 0, 2])
    elif serialize_mode in (SerializeMode.HILBERT, SerializeMode.HILBERT_TRANSPOSED):
        raise NotImplementedError("Hilbert serialization is not ported; use a Z-order mode")
    else:
        raise ValueError(f"Unknown serialize mode: {serialize_mode}")

    for bi, s in enumerate(tensor.layout):
        num_points = s.stop - s.start
        num_windows = (num_points + window_size - 1) // window_size
        valid_window_size = num_points / num_windows
        to_ordered = torch.argsort(code[s.start : s.stop])
        if num_windows == 1:
            fwd_indices.append(to_ordered)
            bwd_indices.append(torch.zeros_like(to_ordered).scatter_(0, to_ordered, torch.arange(num_points, device=tensor.device)))
            fwd_indices[-1] += s.start
            bwd_indices[-1] += offsets[-1]
            seq_lens.append(num_points)
            seq_batch_indices.append(bi)
            offsets.append(offsets[-1] + seq_lens[-1])
        else:
            offset = 0
            mids = [(i + 0.5) * valid_window_size + shift_sequence for i in range(num_windows)]
            split = [math.floor(i * valid_window_size + shift_sequence) for i in range(num_windows + 1)]
            bwd_index = torch.zeros((num_points,), dtype=torch.int64, device=tensor.device)
            for i in range(num_windows):
                mid = mids[i]
                valid_start = split[i]
                valid_end = split[i + 1]
                padded_start = math.floor(mid - 0.5 * window_size)
                padded_end = padded_start + window_size
                fwd_indices.append(to_ordered[torch.arange(padded_start, padded_end, device=tensor.device) % num_points])
                offset += valid_start - padded_start
                bwd_index.scatter_(
                    0,
                    fwd_indices[-1][valid_start - padded_start : valid_end - padded_start],
                    torch.arange(offset, offset + valid_end - valid_start, device=tensor.device),
                )
                offset += padded_end - valid_start
                fwd_indices[-1] += s.start
            seq_lens.extend([window_size] * num_windows)
            seq_batch_indices.extend([bi] * num_windows)
            bwd_indices.append(bwd_index + offsets[-1])
            offsets.append(offsets[-1] + num_windows * window_size)

    return torch.cat(fwd_indices), torch.cat(bwd_indices), seq_lens, seq_batch_indices


def sparse_serialized_scaled_dot_product_self_attention(
    qkv: SparseTensor,
    window_size: int,
    serialize_mode: SerializeMode = SerializeMode.Z_ORDER,
    shift_sequence: int = 0,
    shift_window: tuple[int, int, int] = (0, 0, 0),
) -> SparseTensor:
    """Serialized scaled dot product self attention.

    Args:
        qkv: [N, *, 3, H, C] sparse tensor containing Qs, Ks, and Vs.
        window_size: The window size to use.
        serialize_mode: The serialization mode to use.
        shift_sequence: The shift of serialized sequence.
        shift_window: The shift of serialized coordinates.
    """
    assert len(qkv.shape) == 4 and qkv.shape[1] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"

    cache_name = f"serialization_{serialize_mode}_{window_size}_{shift_sequence}_{shift_window}"
    cached = qkv._coord_cache.get(cache_name)
    if cached is None:
        cached = calc_serialization(qkv, window_size, serialize_mode, shift_sequence, shift_window)
        qkv._coord_cache[cache_name] = cached
    fwd_indices, bwd_indices, seq_lens, _ = cached

    q, k, v = qkv.feats[fwd_indices].unbind(dim=1)  # [M, H, C]
    out = varlen_self_attention(q, k, v, seq_lens)
    return qkv.replace(out[bwd_indices])  # [T, H, C]
