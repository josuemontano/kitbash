# Vendored from TRELLIS (microsoft/TRELLIS, Copyright (c) Microsoft Corporation, MIT License) and Direct3D-S2
# (DreamTechAI/Direct3D-S2, Copyright (c) 2025 DreamTechAI, MIT License); the two `modules/sparse` packages are unified.
# Modified for kitbash: pooling accumulates in float32 (upstream: float64, which MPS does not have; float32 is already safe
# against the fp16 overflow it guarded against). Pooling maps belong to their coordinate set; scale tracks voxel spacing.

from fractions import Fraction

import torch
import torch.nn as nn

from .basic import SparseTensor

__all__ = [
    "SparseDownsample",
    "SparseSubdivide",
    "SparseUpsample",
]


class SparseDownsample(nn.Module):
    """Downsample a sparse tensor by a factor of `factor`. Implemented as pooling (mean by default)."""

    def __init__(self, factor: int | tuple[int, ...] | list[int], mode="mean"):
        super().__init__()
        self.factor = tuple(factor) if isinstance(factor, list | tuple) else factor
        self.mode = mode

    def forward(self, input: SparseTensor) -> SparseTensor:
        DIM = input.coords.shape[-1] - 1
        factor = self.factor if isinstance(self.factor, tuple) else (self.factor,) * DIM
        assert len(factor) == DIM, "Input coordinates must have the same dimension as the downsample factor."

        # Reduction mode only affects features, so all modes share this coordinate-only map.
        key = ("downsample", factor)
        cached = input._coord_cache.get(key)
        if cached is None:
            coord = list(input.coords.long().unbind(dim=-1))
            for i, f in enumerate(factor):
                coord[i + 1] = coord[i + 1] // f

            MAX = [int(coord[i + 1].max().item()) + 1 for i in range(DIM)]
            OFFSET = [*torch.cumprod(torch.tensor(MAX[::-1]), 0).tolist()[::-1], 1]
            code = sum(c * o for c, o in zip(coord, OFFSET, strict=True))
            code, idx = code.unique(return_inverse=True)
            new_coords = torch.stack(
                [code // OFFSET[0], *[(code // OFFSET[i + 1]) % MAX[i] for i in range(DIM)]],
                dim=-1,
            ).to(input.coords.dtype)
            new_layout = SparseTensor._cal_layout(new_coords, input.shape[0])
            # The pyramid owns source caches; children store only an opaque key.
            # A direct back-reference would retain GPU maps until cyclic GC runs.
            source_key = ("pool_source", id(input.coords))
            input._spatial_cache[source_key] = input._coord_cache
            new_cache = {
                ("upsample", factor): (input.coords, input.layout, idx, source_key, input._scale),
            }
            cached = (new_coords, new_layout, idx, new_cache)
            input._coord_cache[key] = cached
        new_coords, new_layout, idx, new_cache = cached

        dtype = input.feats.dtype
        acc_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        # With include_self=True, the zero-initialised buffer makes "mean" sum / (count + 1).
        # That is what the pretrained networks were trained with, so it is kept as is.
        new_feats = torch.scatter_reduce(
            torch.zeros(new_coords.shape[0], input.feats.shape[1], device=input.feats.device, dtype=acc_dtype),
            dim=0,
            index=idx.unsqueeze(1).expand(-1, input.feats.shape[1]),
            src=input.feats.to(acc_dtype),
            reduce=self.mode,
            include_self=True,
        )
        new_feats = new_feats.to(dtype)

        return SparseTensor(
            new_feats,
            new_coords,
            input.shape,
            new_layout,
            scale=tuple(s * f for s, f in zip(input._scale, factor, strict=True)),
            spatial_cache=input._spatial_cache,
            coord_cache=new_cache,
        )


class SparseUpsample(nn.Module):
    """Upsample a sparse tensor by a factor of `factor`. Implemented as nearest neighbour interpolation."""

    def __init__(self, factor: int | tuple[int, int, int] | list[int]):
        super().__init__()
        self.factor = tuple(factor) if isinstance(factor, list | tuple) else factor

    def forward(self, input: SparseTensor) -> SparseTensor:
        DIM = input.coords.shape[-1] - 1
        factor = self.factor if isinstance(self.factor, tuple) else (self.factor,) * DIM
        assert len(factor) == DIM, "Input coordinates must have the same dimension as the upsample factor."

        cached = input._coord_cache.get(("upsample", factor))
        if cached is None:
            raise ValueError("Upsample cache not found. SparseUpsample must be paired with SparseDownsample.")
        new_coords, new_layout, idx, source_key, new_scale = cached
        new_cache = input._spatial_cache[source_key]
        return SparseTensor(
            input.feats[idx],
            new_coords,
            input.shape,
            new_layout,
            scale=new_scale,
            spatial_cache=input._spatial_cache,
            coord_cache=new_cache,
        )


class SparseSubdivide(nn.Module):
    """Upsample a sparse tensor by 2 along each axis, replicating each voxel's features onto its children."""

    def __init__(self):
        super().__init__()

    def forward(self, input: SparseTensor) -> SparseTensor:
        DIM = input.coords.shape[-1] - 1
        # upsample scale=2^DIM
        n_cube = torch.ones([2] * DIM, device=input.device, dtype=torch.int)
        n_coords = torch.nonzero(n_cube)
        n_coords = torch.cat([torch.zeros_like(n_coords[:, :1]), n_coords], dim=-1)
        factor = n_coords.shape[0]
        assert factor == 2**DIM
        new_coords = input.coords.clone()
        new_coords[:, 1:] *= 2
        new_coords = new_coords.unsqueeze(1) + n_coords.unsqueeze(0).to(new_coords.dtype)

        new_feats = input.feats.unsqueeze(1).expand(input.feats.shape[0], factor, *input.feats.shape[1:])
        out = SparseTensor(new_feats.flatten(0, 1), new_coords.flatten(0, 1), input.shape)
        out._scale = tuple(Fraction(s, 2) for s in input._scale)
        out._spatial_cache = input._spatial_cache
        return out
