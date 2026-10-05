# Written for kitbash (replaces the spconv / torchsparse kernel maps used by TRELLIS and Direct3D-S2).
# MIT-licensed vendored code in this package calls into it; the file itself carries no upstream code.
"""Neighbour maps for sparse 3-D convolutions in plain PyTorch.

A *kernel map* is an ``(M, K)`` int32 tensor: for output voxel ``m`` and kernel offset ``k`` it holds the row of the input
voxel that sits at ``out_coord * stride - padding + offset_k * dilation`` or ``-1`` if that voxel is not active. Voxels are
looked up with a sorted-key binary search (no python loops over voxels, no dense grid).
"""

import itertools

import torch


def kernel_offsets(kernel_size: tuple[int, int, int]) -> torch.Tensor:
    """``(K, 3)`` int offsets ``0..k-1`` per axis, x slowest / z fastest: the layout of a ``(.., kx, ky, kz, ..)`` weight."""
    return torch.tensor(list(itertools.product(*[range(k) for k in kernel_size])), dtype=torch.long)


class CoordIndex:
    """Sorted-key index over a set of ``(b, x, y, z)`` voxels supporting vectorised neighbour lookups."""

    def __init__(self, coords: torch.Tensor, margin: int):
        c = coords.long()
        self.lo = c[:, 1:].min(dim=0).values - margin
        hi = c[:, 1:].max(dim=0).values + margin
        self.dims = [int(v) for v in (hi - self.lo + 1).tolist()]
        self.volume = self.dims[0] * self.dims[1] * self.dims[2]
        keys = self.key(c)
        self.sorted_keys, self.order = keys.sort()
        self.n = c.shape[0]

    def key(self, c: torch.Tensor) -> torch.Tensor:
        """Linear key of ``(.., 4)`` long coords; exact as long as each axis lies inside the (margin-padded) box."""
        _, ny, nz = self.dims
        x = c[..., 1] - self.lo[0]
        y = c[..., 2] - self.lo[1]
        z = c[..., 3] - self.lo[2]
        return c[..., 0] * self.volume + (x * ny + y) * nz + z

    def shift_key(self, shift: tuple[int, int, int]) -> int:
        """Key delta for adding ``shift`` to every voxel (valid for shifts within the margin)."""
        return (shift[0] * self.dims[1] + shift[1]) * self.dims[2] + shift[2]

    def find(self, coords: torch.Tensor) -> torch.Tensor:
        """Row of each ``(b, x, y, z)`` coordinate in the indexed set, or -1 (also for coordinates outside the box)."""
        c = coords.long()
        lo = self.lo.to(c.device)
        hi = lo + torch.tensor(self.dims, device=c.device)
        inside = ((c[:, 1:] >= lo) & (c[:, 1:] < hi)).all(dim=1) & (c[:, 0] >= 0)
        return self.lookup(torch.where(inside, self.key(c), torch.full_like(c[:, 0], -1)))

    def lookup(self, query_keys: torch.Tensor) -> torch.Tensor:
        """Row of each query key in the indexed set, or -1."""
        pos = torch.searchsorted(self.sorted_keys, query_keys).clamp_(max=self.n - 1)
        found = self.sorted_keys[pos] == query_keys
        return torch.where(found, self.order[pos], torch.full_like(pos, -1))


def build_kmap(
    out_coords: torch.Tensor,
    index: CoordIndex,
    kernel_size: tuple[int, int, int],
    stride: int = 1,
    padding: tuple[int, int, int] = (0, 0, 0),
    dilation: int = 1,
) -> torch.Tensor:
    """Kernel map ``(M, K)`` (int32) from ``out_coords`` into the voxels held by ``index``."""
    base = out_coords.long().clone()
    base[:, 1:] *= stride
    base_key = index.key(base)
    offs = kernel_offsets(kernel_size).tolist()
    kmap = torch.empty((out_coords.shape[0], len(offs)), dtype=torch.int32, device=out_coords.device)
    for k, off in enumerate(offs):
        shift = tuple(o * dilation - p for o, p in zip(off, padding, strict=True))
        kmap[:, k] = index.lookup(base_key + index.shift_key(shift)).int()
    return kmap


def submanifold_kmap(coords: torch.Tensor, kernel_size: tuple[int, int, int], dilation: int = 1) -> torch.Tensor:
    """Kernel map of a stride-1 convolution whose output sites equal its input sites."""
    padding = tuple(dilation * (k // 2) for k in kernel_size)
    margin = max(dilation * (k - 1) for k in kernel_size)
    return build_kmap(coords, CoordIndex(coords, margin), kernel_size, 1, padding, dilation)


def strided_out_coords(
    coords: torch.Tensor,
    kernel_size: tuple[int, int, int],
    stride: int,
    padding: tuple[int, int, int],
    dilation: int = 1,
) -> torch.Tensor:
    """Output sites of a strided sparse convolution: every site whose receptive field holds an active input."""
    c = coords.long()
    in_size = c[:, 1:].max(dim=0).values + 1
    k_t = torch.tensor(kernel_size, device=c.device)
    pad_t = torch.tensor(padding, device=c.device)
    out_size = (in_size + 2 * pad_t - dilation * (k_t - 1) - 1) // stride + 1
    ny, nz = int(out_size[1]), int(out_size[2])
    volume = int(out_size[0]) * ny * nz
    keys = []
    for off in kernel_offsets(kernel_size).to(c.device):
        n = c[:, 1:] + pad_t - off * dilation
        ok = (n >= 0).all(dim=1) & (n % stride == 0).all(dim=1)
        o = n // stride
        ok &= (o < out_size).all(dim=1)
        o = o[ok]
        keys.append(c[ok, 0] * volume + (o[:, 0] * ny + o[:, 1]) * nz + o[:, 2])
    keys = torch.unique(torch.cat(keys))
    b = keys // volume
    rem = keys % volume
    out = torch.stack([b, rem // (ny * nz), (rem // nz) % ny, rem % nz], dim=1)
    return out.to(coords.dtype)


def invert_kmap(kmap: torch.Tensor, num_in: int) -> torch.Tensor:
    """Transpose a kernel map: ``(num_in, K)`` giving, for every input voxel and offset, the output row that reads it."""
    inv = torch.full((num_in, kmap.shape[1]), -1, dtype=torch.int32, device=kmap.device)
    m, k = (kmap >= 0).nonzero(as_tuple=True)
    inv[kmap[m, k].long(), k] = m.int()
    return inv


def apply_kmap(
    feats: torch.Tensor,
    kmap: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    row_budget_bytes: int = 192 * 2**20,
) -> torch.Tensor:
    """Sparse convolution as gather + matmul, accumulated one kernel offset at a time.

    Args:
        feats: ``(N, I)`` input features.
        kmap: ``(M, K)`` int32 kernel map into the rows of ``feats`` (-1 = no neighbour).
        weight: ``(K, I, O)`` kernel, offsets ordered like :func:`kernel_offsets`.
        bias: optional ``(O,)``.
        row_budget_bytes: upper bound for the temporary gathered ``(rows, I)`` block; rows are processed in chunks.
    """
    n_in, c_in = feats.shape
    m, k_vol = kmap.shape
    c_out = weight.shape[-1]
    weight = weight.to(feats.dtype)
    padded = torch.cat([feats, feats.new_zeros(1, c_in)], dim=0)
    idx = torch.where(kmap < 0, n_in, kmap)
    # offsets that are empty everywhere need no work; one host sync per kernel map
    live = (kmap >= 0).any(dim=0).tolist()
    rows = max(2048, row_budget_bytes // (4 * max(c_in, c_out)))
    out = torch.empty((m, c_out), dtype=feats.dtype, device=feats.device)
    for r0 in range(0, m, rows):
        r1 = min(r0 + rows, m)
        acc = feats.new_zeros(r1 - r0, c_out) if bias is None else bias.to(feats.dtype).expand(r1 - r0, c_out).clone()
        for k in range(k_vol):
            if not live[k]:
                continue
            g = padded.index_select(0, idx[r0:r1, k].long())
            acc.addmm_(g, weight[k])
        out[r0:r1] = acc
    return out
