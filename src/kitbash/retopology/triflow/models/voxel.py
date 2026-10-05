# Adapted from TriFlow (DerKleineLi/triflow, triflow/utils/sparse_voxel.py).
# Copyright (c) 2026 Haoxuan Li.
# Licensed under the Automotive Development Public Non-Commercial License v1.0.
# See LICENSE for details.
#
# Modified for kitbash: ``find_coords_indices`` / ``fine_coords2coarse_coords`` re-implemented on the sorted-key index of
# ``sparse/kmap.py`` so the networks do not depend on the geometry package (and its scipy / scikit-image imports).

import torch

from ..sparse.kmap import CoordIndex


def find_coords_indices(query_coords: torch.Tensor, key_coords: torch.Tensor) -> torch.Tensor:
    """Index into ``key_coords`` of every ``query_coords`` row (both ``(N, 4)`` with the batch index first), -1 if absent."""
    return CoordIndex(key_coords, margin=0).find(query_coords)


def fine_coords2coarse_coords(coords_fine: torch.Tensor, ratio: int) -> torch.Tensor:
    """Unique parent coordinates (``coords // ratio``), sorted by batch then x, y, z."""
    c = coords_fine.long().clone()
    c[:, 1:] //= ratio
    size = int(c[:, 1:].max().item()) + 1
    keys = torch.unique(((c[:, 0] * size + c[:, 1]) * size + c[:, 2]) * size + c[:, 3])
    out = torch.stack([keys // size**3, (keys // size**2) % size, (keys // size) % size, keys % size], dim=1)
    return out.to(coords_fine.dtype)
