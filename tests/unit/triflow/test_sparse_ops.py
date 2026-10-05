"""Pooling / subdivision / coordinate helpers used by the networks."""

import torch

from kitbash.retopology.triflow import sparse as sp
from kitbash.retopology.triflow.models.direct3ds2_sparse_vae import SparseReferencedSubdivide
from kitbash.retopology.triflow.models.voxel import find_coords_indices, fine_coords2coarse_coords


def _coords():
    return torch.tensor(
        [[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 0, 0], [0, 2, 2, 2], [1, 0, 0, 1], [1, 5, 5, 5]], dtype=torch.int32
    )


def test_downsample_keeps_triflow_pooling_semantics_and_upsample_restores():
    # NOTE: upstream "mean" is scatter_reduce(include_self=True) over a zero buffer: sum / (count + 1). The weights rely on it.
    x = sp.SparseTensor(torch.arange(1.0, 7.0)[:, None], _coords())
    down = sp.SparseDownsample(2)(x)
    assert down.coords.tolist() == [[0, 0, 0, 0], [0, 1, 1, 1], [1, 0, 0, 0], [1, 2, 2, 2]]
    assert down.feats[:, 0].tolist() == [(1 + 2 + 3) / 4, 4 / 2, 5 / 2, 6 / 2]
    up = sp.SparseUpsample(2)(down)
    assert torch.equal(up.coords, x.coords)
    assert up.feats[:, 0].tolist() == [1.5, 1.5, 1.5, 2.0, 2.5, 3.0]


def test_subdivide_makes_eight_children_per_voxel():
    x = sp.SparseTensor(torch.randn(6, 3), _coords())
    out = sp.SparseSubdivide()(x)
    assert out.coords.shape[0] == 48
    assert torch.equal(out.coords[::8, 1:], x.coords[:, 1:] * 2)
    assert torch.equal(out.feats[3], x.feats[0])
    assert out._scale == (2, 2, 2)


def test_referenced_subdivide_equals_filtering_all_children():
    torch.manual_seed(0)
    x = sp.SparseTensor(torch.randn(6, 3), _coords())
    children = sp.SparseSubdivide()(x)
    keep = torch.tensor([5, 0, 17, 40, 9, 33])  # arbitrary subset, arbitrary order
    ref = children.coords[keep]
    out = SparseReferencedSubdivide()(x, reference_coords=ref)
    assert torch.equal(out.coords, ref)
    assert torch.equal(out.feats, children.feats[keep])


def test_find_coords_indices_and_coarse_coords():
    keys = _coords()
    queries = torch.tensor([[0, 2, 2, 2], [0, 9, 9, 9], [1, 5, 5, 5], [0, 0, 0, 1], [2, 0, 0, 0], [0, -1, 0, 0]], dtype=torch.int32)
    assert find_coords_indices(queries, keys).tolist() == [3, -1, 5, -1, -1, -1]
    assert fine_coords2coarse_coords(keys, 2).tolist() == [[0, 0, 0, 0], [0, 1, 1, 1], [1, 0, 0, 0], [1, 2, 2, 2]]


def test_sparse2sparse_tensor_sets_batch_shape_and_layout():
    t = sp.sparse2sparse_tensor(_coords(), torch.randn(6, 5))
    assert t.shape == (2, 5)
    assert t.layout == [slice(0, 4), slice(4, 6)]
    assert t.coords.dtype == torch.int32


def test_batch_broadcast_and_elementwise_ops():
    x = sp.SparseTensor(torch.ones(6, 2), _coords())
    y = x * torch.tensor([[2.0, 3.0], [5.0, 7.0]]) + 1
    assert y.feats[:4, 0].eq(3).all()
    assert y.feats[4:, 1].eq(8).all()
    assert sp.sparse_cat([x, x]).shape[0] == 4
    assert [t.feats.shape[0] for t in x.unbind(0)] == [4, 2]
