# Vendored from TRELLIS (microsoft/TRELLIS, Copyright (c) Microsoft Corporation, MIT License) and Direct3D-S2
# (DreamTechAI/Direct3D-S2, Copyright (c) 2025 DreamTechAI, MIT License); the two `modules/sparse` packages are unified.
# Modified for kitbash: the spconv / torchsparse kernels are replaced by a pure-PyTorch gather + matmul over a hashed
# neighbour map (``kmap.py``). The parameter lives in ``<module>.conv.weight`` with shape ``(out, kx, ky, kz, in)`` (the
# spconv layout used by the TRELLIS checkpoints). Direct3D-S2 checkpoints store a torchsparse ``conv.kernel`` of shape
# ``(kx*ky*kz, in, out)`` (``(in, out)`` for 1x1x1); it is converted while loading.

import math

import torch
import torch.nn as nn

from . import kmap as km
from .basic import SparseTensor

__all__ = ["SparseConv3d", "SparseInverseConv3d", "torchsparse_kernel_to_weight"]


def _triple(v) -> tuple[int, int, int]:
    return tuple(v) if isinstance(v, list | tuple) else (v, v, v)


def torchsparse_kernel_to_weight(kernel: torch.Tensor, kernel_size: tuple[int, int, int]) -> torch.Tensor:
    """``(kx*ky*kz, in, out)`` / ``(in, out)`` torchsparse kernel -> ``(out, kx, ky, kz, in)``.

    torchsparse numbers the offsets of an odd kernel with x fastest and z slowest.
    """
    kx, ky, kz = kernel_size
    if kernel.ndim == 2:
        kernel = kernel.unsqueeze(0)
    c_in, c_out = kernel.shape[1:]
    return kernel.reshape(kz, ky, kx, c_in, c_out).permute(4, 2, 1, 0, 3).contiguous()


class _ConvParams(nn.Module):
    """Holds the kernel (``weight``) and ``bias``; exists so that the state-dict keys read ``<name>.conv.weight``."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int, int], bias: bool):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, *kernel_size, in_channels))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        bound = 1.0 / math.sqrt(in_channels * math.prod(kernel_size))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def kernel(self) -> torch.Tensor:
        """``(K, in, out)`` view used by :func:`kmap.apply_kmap`."""
        return self.weight.permute(1, 2, 3, 4, 0).reshape(-1, self.in_channels, self.out_channels)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        legacy = prefix + "kernel"
        if legacy in state_dict:
            state_dict[prefix + "weight"] = torchsparse_kernel_to_weight(state_dict.pop(legacy), self.kernel_size)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


def _cache_key(kind: str, *parts) -> str:
    return f"{kind}_{'_'.join(str(p) for p in parts)}"


def _submanifold_kmap(x: SparseTensor, kernel_size, dilation) -> torch.Tensor:
    key = _cache_key("kmap", kernel_size, dilation)
    kmap = x._coord_cache.get(key)
    if kmap is None:
        kmap = km.submanifold_kmap(x.coords, kernel_size, dilation)
        x._coord_cache[key] = kmap
    return kmap


class SparseConv3d(nn.Module):
    """Sparse 3-D convolution (cross-correlation).

    ``stride == 1`` is a submanifold convolution (the output sites are the input sites, ``padding`` is irrelevant and the
    kernel must be odd). ``stride > 1`` produces every output site whose receptive field contains an active input.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=0, bias=True, indice_key=None):
        super().__init__()
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.dilation = dilation
        self.padding = _triple(padding if padding is not None else 0)
        if len(set(self.stride)) != 1:
            raise NotImplementedError("anisotropic strides are not supported")
        if self.stride[0] == 1 and any(k % 2 == 0 for k in self.kernel_size):
            raise NotImplementedError("submanifold convolutions need an odd kernel")
        self.conv = _ConvParams(in_channels, out_channels, self.kernel_size, bias)

    def forward(self, x: SparseTensor) -> SparseTensor:
        weight, bias = self.conv.kernel(), self.conv.bias
        c_out = self.conv.out_channels
        if self.stride[0] == 1:
            kmap = _submanifold_kmap(x, self.kernel_size, self.dilation)
            feats = km.apply_kmap(x.feats, kmap, weight, bias)
            return x.replace(feats)

        s = self.stride[0]
        out_coords = km.strided_out_coords(x.coords, self.kernel_size, s, self.padding, self.dilation)
        margin = self.dilation * (max(self.kernel_size) - 1) + max(self.padding) + s
        kmap = km.build_kmap(out_coords, km.CoordIndex(x.coords, margin), self.kernel_size, s, self.padding, self.dilation)
        feats = km.apply_kmap(x.feats, kmap, weight, bias)
        out = SparseTensor(feats, out_coords, torch.Size([x.shape[0], c_out]))
        out._scale = tuple(sc * st for sc, st in zip(x._scale, self.stride, strict=True))
        out._spatial_cache = x._spatial_cache
        out.register_spatial_cache(_cache_key("conv", self.stride, self.kernel_size, "inverse"), (x.coords, x.layout, kmap))
        return out


class SparseInverseConv3d(nn.Module):
    """Transposed counterpart of a strided :class:`SparseConv3d`: restores the sites that convolution consumed.

    It must be applied to (a descendant of) that convolution's output so the recorded kernel map can be reused.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None):
        super().__init__()
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.dilation = dilation
        self.conv = _ConvParams(in_channels, out_channels, self.kernel_size, bias)

    def forward(self, x: SparseTensor) -> SparseTensor:
        cached = x.get_spatial_cache(_cache_key("conv", self.stride, self.kernel_size, "inverse"))
        if cached is None:
            raise ValueError("Inverse convolution cache not found. SparseInverseConv3d must follow its strided SparseConv3d.")
        coords, layout, kmap = cached
        inv = km.invert_kmap(kmap, coords.shape[0])
        feats = km.apply_kmap(x.feats, inv, self.conv.kernel(), self.conv.bias)
        out = SparseTensor(feats, coords, torch.Size([x.shape[0], self.conv.out_channels]), layout)
        out._scale = tuple(sc // st for sc, st in zip(x._scale, self.stride, strict=True))
        out._spatial_cache = x._spatial_cache
        return out
