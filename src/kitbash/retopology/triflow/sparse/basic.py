# Vendored from TRELLIS (microsoft/TRELLIS, Copyright (c) Microsoft Corporation, MIT License) and Direct3D-S2
# (DreamTechAI/Direct3D-S2, Copyright (c) 2025 DreamTechAI, MIT License); the two ``modules/sparse`` packages are unified.
# Modified for kitbash: the torchsparse / spconv tensor backends are gone. ``SparseTensor`` is a plain (feats, coords)
# pair; neighbour maps used by the convolutions live in a cache that is bound to the coordinate set.

from fractions import Fraction
from typing import Any

import torch

__all__ = [
    "SparseTensor",
    "sparse_batch_broadcast",
    "sparse_batch_op",
    "sparse_cat",
    "sparse_unbind",
]


def _cache_to_device(value: Any, device: torch.device, memo: dict[int, Any]) -> Any:
    """Move coordinate bookkeeping once, retaining shared pyramid caches."""
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, torch.Tensor):
        moved = value.to(device=device)
    elif isinstance(value, dict):
        moved = {}
        memo[id(value)] = moved
        moved.update((key, _cache_to_device(item, device, memo)) for key, item in value.items())
    elif isinstance(value, list):
        moved = []
        memo[id(value)] = moved
        moved.extend(_cache_to_device(item, device, memo) for item in value)
    elif isinstance(value, tuple):
        moved = tuple(_cache_to_device(item, device, memo) for item in value)
    else:
        return value
    memo[id(value)] = moved
    return moved


class SparseTensor:
    """Sparse tensor: ``feats`` (N, ...) at integer ``coords`` (N, 1 + D) with the batch index in column 0.

    Parameters:
    - feats: Features of the sparse tensor.
    - coords: Coordinates of the sparse tensor.
    - shape: ``(B, *feat_shape)``. Derived from the data when omitted.
    - layout: One ``slice`` per batch element. Derived from ``coords`` when omitted.
    - scale: Voxel spacing relative to the original grid; subdivision may use exact fractional spacing.

    NOTE: the data of one batch element must be contiguous (sorted by batch index).
    """

    def __init__(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        shape: torch.Size | None = None,
        layout: list[slice] | None = None,
        *,
        scale: tuple[int | Fraction, ...] = (1, 1, 1),
        spatial_cache: dict | None = None,
        coord_cache: dict | None = None,
    ):
        self.feats = feats
        self.coords = coords
        if shape is None:
            shape = self._cal_shape(feats, coords)
        if layout is None:
            layout = self._cal_layout(coords, shape[0])
        self._shape = shape
        self._layout = layout
        self._scale = scale
        # Bookkeeping shared by the convolution pyramid.
        self._spatial_cache = {} if spatial_cache is None else spatial_cache
        # Only valid for this coordinate set: neighbour maps, window partitions and pooling maps.
        self._coord_cache = {} if coord_cache is None else coord_cache

    @staticmethod
    def _cal_shape(feats: torch.Tensor, coords: torch.Tensor) -> torch.Size:
        return torch.Size([int(coords[:, 0].max().item()) + 1, *feats.shape[1:]])

    @staticmethod
    def _cal_layout(coords: torch.Tensor, batch_size: int) -> list[slice]:
        seq_len = torch.bincount(coords[:, 0].long(), minlength=batch_size).tolist()
        layout, start = [], 0
        for n in seq_len:
            layout.append(slice(start, start + n))
            start += n
        return layout

    @property
    def shape(self) -> torch.Size:
        return self._shape

    def dim(self) -> int:
        return len(self.shape)

    @property
    def layout(self) -> list[slice]:
        return self._layout

    @property
    def dtype(self):
        return self.feats.dtype

    @property
    def device(self):
        return self.feats.device

    def to(self, *args, **kwargs) -> SparseTensor:
        device = dtype = None
        if len(args) == 2:
            device, dtype = args
        elif len(args) == 1:
            if isinstance(args[0], torch.dtype):
                dtype = args[0]
            else:
                device = args[0]
        if "dtype" in kwargs:
            assert dtype is None, "to() received multiple values for argument 'dtype'"
            dtype = kwargs["dtype"]
        if "device" in kwargs:
            assert device is None, "to() received multiple values for argument 'device'"
            device = kwargs["device"]

        new_feats = self.feats.to(device=device, dtype=dtype)
        new_coords = self.coords.to(device=device)
        if new_coords is self.coords:
            return self.replace(new_feats)
        memo = {id(self.coords): new_coords}
        return SparseTensor(
            new_feats,
            new_coords,
            torch.Size([self.shape[0], *new_feats.shape[1:]]),
            self.layout,
            scale=self._scale,
            spatial_cache=_cache_to_device(self._spatial_cache, new_coords.device, memo),
            coord_cache=_cache_to_device(self._coord_cache, new_coords.device, memo),
        )

    def type(self, dtype) -> SparseTensor:
        return self.replace(self.feats.type(dtype))

    def cpu(self) -> SparseTensor:
        return self.to(device="cpu")

    def half(self) -> SparseTensor:
        return self.replace(self.feats.half())

    def float(self) -> SparseTensor:
        return self.replace(self.feats.float())

    def detach(self) -> SparseTensor:
        return self.replace(self.feats.detach())

    def dense(self) -> torch.Tensor:
        """Scatter into a dense ``(B, C, X, Y, Z)`` grid sized by the largest coordinate."""
        size = (self.coords[:, 1:].max(dim=0).values + 1).tolist()
        out = self.feats.new_zeros((self.shape[0], self.feats.shape[1], *size))
        c = self.coords.long()
        out[c[:, 0], :, c[:, 1], c[:, 2], c[:, 3]] = self.feats
        return out

    def reshape(self, *shape) -> SparseTensor:
        return self.replace(self.feats.reshape(self.feats.shape[0], *shape))

    def unbind(self, dim: int) -> list[SparseTensor]:
        return sparse_unbind(self, dim)

    def replace(self, feats: torch.Tensor, coords: torch.Tensor | None = None) -> SparseTensor:
        """Replace features, sharing bookkeeping only when the coordinate tensor is unchanged."""
        same_coords = coords is None or coords is self.coords
        new_shape = torch.Size([self.shape[0], *feats.shape[1:]])
        return SparseTensor(
            feats,
            self.coords if coords is None else coords,
            new_shape,
            self.layout if same_coords else None,
            scale=self._scale,
            spatial_cache=self._spatial_cache if same_coords else None,
            coord_cache=self._coord_cache if same_coords else None,
        )

    @staticmethod
    def full(aabb, dim, value, dtype=torch.float32, device=None) -> SparseTensor:
        N, C = dim
        x = torch.arange(aabb[0], aabb[3] + 1)
        y = torch.arange(aabb[1], aabb[4] + 1)
        z = torch.arange(aabb[2], aabb[5] + 1)
        coords = torch.stack(torch.meshgrid(x, y, z, indexing="ij"), dim=-1).reshape(-1, 3)
        coords = torch.cat(
            [torch.arange(N).view(-1, 1).repeat(1, coords.shape[0]).view(-1, 1), coords.repeat(N, 1)], dim=1
        ).to(dtype=torch.int32, device=device)
        feats = torch.full((coords.shape[0], C), value, dtype=dtype, device=device)
        return SparseTensor(feats, coords)

    def __neg__(self) -> SparseTensor:
        return self.replace(-self.feats)

    def __elemwise__(self, other: torch.Tensor | SparseTensor | float, op: callable) -> SparseTensor:
        if isinstance(other, torch.Tensor):
            try:
                other = torch.broadcast_to(other, self.shape)
                other = sparse_batch_broadcast(self, other)
            except Exception:
                pass
        if isinstance(other, SparseTensor):
            other = other.feats
        return self.replace(op(self.feats, other))

    def __add__(self, other):
        return self.__elemwise__(other, torch.add)

    def __radd__(self, other):
        return self.__elemwise__(other, torch.add)

    def __sub__(self, other):
        return self.__elemwise__(other, torch.sub)

    def __rsub__(self, other):
        return self.__elemwise__(other, lambda x, y: torch.sub(y, x))

    def __mul__(self, other):
        return self.__elemwise__(other, torch.mul)

    def __rmul__(self, other):
        return self.__elemwise__(other, torch.mul)

    def __truediv__(self, other):
        return self.__elemwise__(other, torch.div)

    def __rtruediv__(self, other):
        return self.__elemwise__(other, lambda x, y: torch.div(y, x))

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        elif isinstance(idx, slice):
            idx = range(*idx.indices(self.shape[0]))
        elif isinstance(idx, torch.Tensor):
            if idx.dtype == torch.bool:
                assert idx.shape == (self.shape[0],), f"Invalid index shape: {idx.shape}"
                idx = idx.nonzero().squeeze(1)
            elif idx.dtype in [torch.int32, torch.int64]:
                assert len(idx.shape) == 1, f"Invalid index shape: {idx.shape}"
            else:
                raise ValueError(f"Unknown index type: {idx.dtype}")
        else:
            raise ValueError(f"Unknown index type: {type(idx)}")

        coords, feats = [], []
        for new_idx, old_idx in enumerate(idx):
            coords.append(self.coords[self.layout[old_idx]].clone())
            coords[-1][:, 0] = new_idx
            feats.append(self.feats[self.layout[old_idx]])
        return SparseTensor(torch.cat(feats, dim=0).contiguous(), torch.cat(coords, dim=0).contiguous())

    def register_spatial_cache(self, key, value) -> None:
        """Register a cache entry for the current scale (kept across the tensors of one pyramid level)."""
        scale_key = self._scale
        if scale_key not in self._spatial_cache:
            self._spatial_cache[scale_key] = {}
        self._spatial_cache[scale_key][key] = value

    def get_spatial_cache(self, key=None):
        """Get a cache entry (or all entries) registered for the current scale."""
        cur_scale_cache = self._spatial_cache.get(self._scale, {})
        if key is None:
            return cur_scale_cache
        return cur_scale_cache.get(key, None)


def _batch_index(input: SparseTensor) -> torch.Tensor:
    """Batch index of every row, derived from the layout."""
    counts = torch.tensor([s.stop - s.start for s in input.layout], device=input.device)
    return torch.repeat_interleave(torch.arange(len(input.layout), device=input.device), counts)


def sparse_batch_broadcast(input: SparseTensor, other: torch.Tensor) -> torch.Tensor:
    """Broadcast a per-batch tensor ``other`` (B, ...) to one row per voxel of ``input``."""
    return other[_batch_index(input)].to(input.feats.dtype)


def sparse_batch_op(input: SparseTensor, other: torch.Tensor, op: callable = torch.add) -> SparseTensor:
    """Broadcast a per-batch tensor to a sparse tensor along the batch dimension, then apply ``op``."""
    return input.replace(op(input.feats, sparse_batch_broadcast(input, other)))


def sparse_cat(inputs: list[SparseTensor], dim: int = 0) -> SparseTensor:
    """Concatenate sparse tensors along the batch (``dim=0``) or the feature dimension."""
    if dim == 0:
        start = 0
        coords = []
        for input in inputs:
            coords.append(input.coords.clone())
            coords[-1][:, 0] += start
            start += input.shape[0]
        return SparseTensor(torch.cat([input.feats for input in inputs], dim=0), torch.cat(coords, dim=0))
    return inputs[0].replace(torch.cat([input.feats for input in inputs], dim=dim))


def sparse_unbind(input: SparseTensor, dim: int) -> list[SparseTensor]:
    """Unbind a sparse tensor along the batch (``dim=0``) or a feature dimension."""
    if dim == 0:
        return [input[i] for i in range(input.shape[0])]
    return [input.replace(f) for f in input.feats.unbind(dim)]
