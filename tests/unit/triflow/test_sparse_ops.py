"""Pooling / subdivision / coordinate helpers used by the networks."""

import gc
import weakref
from fractions import Fraction

import pytest
import torch
import torch.nn.functional as F

from kitbash.retopology.triflow import sparse as sp
from kitbash.retopology.triflow.models.direct3ds2_sparse_vae import SparseReferencedSubdivide
from kitbash.retopology.triflow.models.voxel import find_coords_indices, fine_coords2coarse_coords
from kitbash.retopology.triflow.sparse import kmap

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + (["cuda"] if torch.cuda.is_available() else [])


def _coords():
    return torch.tensor(
        [[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 0, 0], [0, 2, 2, 2], [1, 0, 0, 1], [1, 5, 5, 5]], dtype=torch.int32
    )


def _line_coords():
    coords = torch.zeros(8, 4, dtype=torch.int32)
    coords[:, 1] = torch.arange(8)
    return coords


def test_downsample_keeps_triflow_pooling_semantics_and_upsample_restores():
    # NOTE: upstream "mean" is scatter_reduce(include_self=True) over a zero buffer: sum / (count + 1). The weights rely on it.
    x = sp.SparseTensor(torch.arange(1.0, 7.0)[:, None], _coords())
    down = sp.SparseDownsample(2)(x)
    assert down.coords.tolist() == [[0, 0, 0, 0], [0, 1, 1, 1], [1, 0, 0, 0], [1, 2, 2, 2]]
    assert down.feats[:, 0].tolist() == [(1 + 2 + 3) / 4, 4 / 2, 5 / 2, 6 / 2]
    up = sp.SparseUpsample(2)(down)
    assert torch.equal(up.coords, x.coords)
    assert up.feats[:, 0].tolist() == [1.5, 1.5, 1.5, 2.0, 2.5, 3.0]


@pytest.mark.parametrize("device", DEVICES)
def test_two_pooling_levels_restore_original_sites(device):
    x = sp.SparseTensor(torch.arange(1.0, 9.0, device=device)[:, None], _line_coords().to(device))
    down, up = sp.SparseDownsample(2), sp.SparseUpsample(2)
    coarse = down(down(x))
    restored = up(up(coarse))
    assert torch.equal(restored.coords, x.coords)
    assert restored.layout == x.layout
    assert restored._scale == x._scale
    torch.testing.assert_close(restored.feats[:, 0], torch.tensor([10 / 9] * 4 + [26 / 9] * 4, device=device))


@pytest.mark.parametrize("ownership", ["independent", "shared_pyramid", "replaced"])
def test_unrelated_coordinate_sets_keep_their_own_pooling_maps(ownership):
    x = sp.SparseTensor(torch.arange(1.0, 7.0)[:, None], _coords())
    down, up = sp.SparseDownsample(2), sp.SparseUpsample(2)
    pooled_x = down(x)
    other_coords = torch.tensor(
        [[0, 6, 0, 0], [0, 7, 0, 0], [1, 8, 0, 0], [1, 9, 0, 0], [1, 10, 0, 0], [1, 11, 0, 0]], dtype=torch.int32
    )
    feats = torch.arange(7.0, 13.0)[:, None]
    if ownership == "replaced":
        other = x.replace(feats, other_coords)
    else:
        other = sp.SparseTensor(feats, other_coords, spatial_cache=x._spatial_cache if ownership == "shared_pyramid" else None)
    pooled_other = down(other)
    restored_x, restored_other = up(pooled_x), up(pooled_other)
    assert torch.equal(restored_x.coords, x.coords)
    assert torch.equal(restored_other.coords, other_coords)
    assert restored_other.layout == [slice(0, 2), slice(2, 6)]
    torch.testing.assert_close(restored_x.feats[:, 0], torch.tensor([1.5, 1.5, 1.5, 2.0, 2.5, 3.0]))
    torch.testing.assert_close(restored_other.feats[:, 0], torch.tensor([5.0, 5.0, 19 / 3, 19 / 3, 23 / 3, 23 / 3]))
    # Replacing even a pooled tensor's coordinates cannot inherit its old inverse map.
    with pytest.raises(ValueError, match="cache"):
        up(pooled_x.replace(pooled_x.feats, pooled_x.coords.clone()))


def test_pooling_cache_separates_factors_and_recomputes_each_reduction():
    x = sp.SparseTensor(torch.arange(-4.0, 4.0)[:, None], _line_coords())
    for mode in ["mean", "sum", "amax", "amin", "prod"]:
        out = sp.SparseDownsample([2, 1, 1], mode=mode)(x)
        groups = torch.cat([torch.zeros(4, 1), x.feats.reshape(4, 2)], dim=1)
        expected = getattr(groups, mode)(dim=1)
        torch.testing.assert_close(out.feats[:, 0], expected)
        assert torch.equal(sp.SparseUpsample((2, 1, 1))(out).coords, x.coords)
    across_y = sp.SparseDownsample((1, 2, 1))(x)
    torch.testing.assert_close(across_y.feats, x.feats / 2)
    across_four = sp.SparseDownsample(4)(x)
    torch.testing.assert_close(across_four.feats[:, 0], x.feats.reshape(2, 4).sum(dim=1) / 5)
    assert torch.equal(sp.SparseUpsample(4)(across_four).coords, x.coords)


def test_cached_pooling_keeps_fresh_features_and_gradients():
    x = sp.SparseTensor(torch.zeros(8, 1), _line_coords())
    down, up = sp.SparseDownsample(2), sp.SparseUpsample(2)
    for channels in [1, 3]:
        feats = torch.arange(8 * channels, dtype=torch.float32).reshape(8, channels).requires_grad_()
        pooled = down(x.replace(feats))
        torch.testing.assert_close(pooled.feats, feats.reshape(4, 2, channels).sum(dim=1) / 3)
        restored = up(pooled)
        restored.feats.sum().backward()
        torch.testing.assert_close(feats.grad, torch.full_like(feats, 2 / 3))


def test_repeated_flow_reuses_pooling_layout_and_neighbor_maps(monkeypatch):
    x = sp.SparseTensor(torch.arange(1.0, 9.0)[:, None], _line_coords())
    down, up = sp.SparseDownsample(2), sp.SparseUpsample(2)
    conv = sp.SparseConv3d(1, 1, 3, bias=False)
    with torch.no_grad():
        conv.conv.weight.fill_(1)
    calls = {"neighbors": 0, "pooling": 0, "layout": 0}
    build_neighbors = kmap.submanifold_kmap
    unique = torch.Tensor.unique
    calculate_layout = sp.SparseTensor._cal_layout

    def count_neighbors(*args, **kwargs):
        calls["neighbors"] += 1
        return build_neighbors(*args, **kwargs)

    def count_pooling(*args, **kwargs):
        calls["pooling"] += 1
        return unique(*args, **kwargs)

    def count_layout(*args, **kwargs):
        calls["layout"] += 1
        return calculate_layout(*args, **kwargs)

    monkeypatch.setattr(kmap, "submanifold_kmap", count_neighbors)
    monkeypatch.setattr(torch.Tensor, "unique", count_pooling)
    monkeypatch.setattr(sp.SparseTensor, "_cal_layout", staticmethod(count_layout))
    with torch.no_grad():
        for step in range(3):
            current = x.replace(x.feats + step).to(torch.float32).detach()
            output = conv(up(conv(down(current))))
            expected = current.feats.reshape(1, 1, 4, 2).sum(dim=-1) / 3
            expected = F.conv1d(expected, torch.ones(1, 1, 3), padding=1).repeat_interleave(2, dim=-1)
            expected = F.conv1d(expected, torch.ones(1, 1, 3), padding=1)
            torch.testing.assert_close(output.feats[:, 0], expected.flatten())
            assert torch.equal(output.coords, x.coords)
            assert calls == {"neighbors": 2, "pooling": 1, "layout": 1}


@pytest.mark.parametrize("device", DEVICES)
def test_pyramid_cache_survives_device_and_dtype_transfers(device):
    x = sp.SparseTensor(torch.arange(1.0, 9.0)[:, None], _line_coords())
    down, up = sp.SparseDownsample(2), sp.SparseUpsample(2)
    coarse = down(down(x))
    moved = coarse.to(device=device, dtype=torch.float16)
    restored = up(up(moved))
    assert restored.feats.dtype == torch.float16
    assert restored.coords.device == moved.coords.device
    assert torch.equal(restored.coords.cpu(), x.coords)
    torch.testing.assert_close(restored.feats.cpu(), up(up(coarse)).feats.half())
    # Transferred bookkeeping remains usable for another full pyramid, without replacing the CPU cache.
    repeated = up(up(down(down(restored)))).cpu()
    expected = up(up(down(down(restored.cpu()))))
    torch.testing.assert_close(repeated.feats, expected.feats)
    assert torch.equal(repeated.coords, x.coords)
    assert torch.equal(up(up(coarse)).coords, x.coords)


def test_pooling_maps_release_without_waiting_for_cyclic_gc():
    enabled = gc.isenabled()
    gc.disable()
    try:
        x = sp.SparseTensor(torch.ones(8, 1), _line_coords())
        pooled = sp.SparseDownsample(2)(x)
        coordinates = [weakref.ref(x.coords), weakref.ref(pooled.coords)]
        del x, pooled
        assert all(reference() is None for reference in coordinates)
    finally:
        if enabled:
            gc.enable()


def test_subdivide_makes_eight_children_per_voxel():
    x = sp.SparseTensor(torch.randn(6, 3), _coords())
    out = sp.SparseSubdivide()(x)
    assert out.coords.shape[0] == 48
    assert torch.equal(out.coords[::8, 1:], x.coords[:, 1:] * 2)
    assert torch.equal(out.feats[3], x.feats[0])


def test_referenced_subdivide_equals_filtering_all_children():
    torch.manual_seed(0)
    x = sp.SparseTensor(torch.randn(6, 3), _coords())
    children = sp.SparseSubdivide()(x)
    keep = torch.tensor([5, 0, 17, 40, 9, 33])  # arbitrary subset, arbitrary order
    ref = children.coords[keep]
    out = SparseReferencedSubdivide()(x, reference_coords=ref)
    assert torch.equal(out.coords, ref)
    assert torch.equal(out.feats, children.feats[keep])


def test_fractional_spacing_survives_pooling_convolution_and_subdivision():
    x = sp.SparseTensor(torch.arange(1.0, 9.0)[:, None], _line_coords())
    refined = sp.SparseSubdivide()(x)
    referenced = SparseReferencedSubdivide()(x, reference_coords=refined.coords)
    down, up = sp.SparseDownsample(2), sp.SparseUpsample(2)
    for children in [refined, referenced]:
        assert children._scale == (Fraction(1, 2),) * 3
        pooled = down(children)
        assert pooled._scale == x._scale
        assert torch.equal(pooled.coords, x.coords)
        assert up(pooled)._scale == children._scale
        strided = sp.SparseConv3d(1, 1, 3, stride=2, padding=1)(children)
        assert strided._scale == pooled._scale
        inverse = sp.SparseInverseConv3d(1, 1, 3, stride=2)(strided)
        assert inverse._scale == children._scale
        assert torch.equal(inverse.coords, children.coords)
    conv_down = sp.SparseConv3d(1, 1, 3, stride=2, padding=1)
    conv_up = sp.SparseInverseConv3d(1, 1, 3, stride=2)
    restored = conv_up(conv_up(conv_down(conv_down(x))))
    assert restored._scale == x._scale
    assert torch.equal(restored.coords, x.coords)


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
