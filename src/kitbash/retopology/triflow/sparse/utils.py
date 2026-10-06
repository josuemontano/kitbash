# Adapted from TriFlow (DerKleineLi/triflow, triflow/utils/direct3ds2_sparse.py and trellis_sparse.py).
# Copyright (c) 2026 Haoxuan Li.
# Licensed under the Automotive Development Public Non-Commercial License v1.0.
# See LICENSE for details.
#
# Modified for kitbash: the two upstream copies (one per SparseTensor class) are merged into one.

import torch

from .basic import SparseTensor

__all__ = ["sparse2sparse_tensor"]


def sparse2sparse_tensor(coords: torch.Tensor, feats: torch.Tensor) -> SparseTensor:
    """Wrap batched sparse ``(coords, feats)`` into a :class:`SparseTensor`.

    ``coords`` is ``(N, D+1)`` with the batch index in column 0. The per-batch voxel layout is computed and registered as a
    spatial cache, and the advertised shape is set to ``(B, C)`` so shape-dependent ops know the batch size without scanning
    ``coords``.
    """
    counts = torch.bincount(coords[:, 0].long()).tolist()
    layout, start = [], 0
    for n in counts:
        layout.append(slice(start, start + n))
        start += n

    sparse_tensor = SparseTensor(feats, coords.int(), torch.Size([len(counts), feats.shape[1]]), layout)
    sparse_tensor.register_spatial_cache("layout", layout)
    return sparse_tensor
