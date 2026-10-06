"""The pure-PyTorch sparse convolutions must equal a dense ``conv3d`` evaluated on the active sites."""

import pytest
import torch
import torch.nn.functional as F

from kitbash.retopology.triflow.sparse import SparseConv3d, SparseInverseConv3d, SparseTensor
from kitbash.retopology.triflow.sparse.conv import torchsparse_kernel_to_weight
from kitbash.retopology.triflow.sparse.kmap import invert_kmap, submanifold_kmap

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
GRID = 12


def _random_sparse(device, batch=2, c_in=5, density=0.15, seed=0):
    g = torch.Generator().manual_seed(seed)
    occ = torch.rand(batch, GRID, GRID, GRID, generator=g) < density
    occ[:, 0, 0, 0] = True  # keep every batch non-empty
    coords = occ.nonzero().int()  # sorted by (b, x, y, z)
    feats = torch.randn(coords.shape[0], c_in, generator=g)
    return SparseTensor(feats.to(device), coords.to(device)), occ


def _dense(x: SparseTensor) -> torch.Tensor:
    return x.dense()[:, :, :GRID, :GRID, :GRID] if x.coords[:, 1:].max() == GRID - 1 else _padded(x)


def _padded(x: SparseTensor) -> torch.Tensor:
    out = x.feats.new_zeros((x.shape[0], x.feats.shape[1], GRID, GRID, GRID))
    c = x.coords.long()
    out[c[:, 0], :, c[:, 1], c[:, 2], c[:, 3]] = x.feats
    return out


def _gather(dense: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    c = coords.long()
    return dense[c[:, 0], :, c[:, 1], c[:, 2], c[:, 3]]


def _dense_weight(conv) -> torch.Tensor:
    return conv.conv.weight.permute(0, 4, 1, 2, 3).detach().cpu()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("ks", "dilation"), [(3, 1), (1, 1), (3, 2), (5, 1)])
def test_submanifold_conv_matches_dense(device, ks, dilation):
    torch.manual_seed(1)
    x, _ = _random_sparse(device)
    conv = SparseConv3d(5, 7, ks, dilation=dilation).to(device)
    with torch.no_grad():
        y = conv(x)
        ref = F.conv3d(_padded(x.cpu()), _dense_weight(conv), conv.conv.bias.cpu(), padding=dilation * (ks // 2), dilation=dilation)
    assert torch.equal(y.coords, x.coords)
    torch.testing.assert_close(y.feats.cpu(), _gather(ref, x.coords.cpu()), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_strided_conv_matches_dense(device):
    torch.manual_seed(2)
    x, occ = _random_sparse(device)
    conv = SparseConv3d(5, 7, 3, stride=2, padding=1).to(device)
    with torch.no_grad():
        y = conv(x)
        xd = _padded(x.cpu())
        ref = F.conv3d(xd, _dense_weight(conv), conv.conv.bias.cpu(), stride=2, padding=1)
        ones = F.conv3d(occ[:, None].float(), torch.ones(1, 1, 3, 3, 3), stride=2, padding=1)
    expect = (ones[:, 0] > 0).nonzero().int()
    assert torch.equal(y.coords.cpu(), expect)
    torch.testing.assert_close(y.feats.cpu(), _gather(ref, expect), atol=1e-4, rtol=1e-4)
    assert y.shape[0] == x.shape[0]


@pytest.mark.parametrize("device", DEVICES)
def test_inverse_conv_matches_dense_transpose(device):
    torch.manual_seed(3)
    x, _ = _random_sparse(device)
    down = SparseConv3d(5, 7, 3, stride=2, padding=1).to(device)
    up = SparseInverseConv3d(7, 4, 3, stride=2).to(device)
    with torch.no_grad():
        y = down(x)
        z = up(y)
        yd = torch.zeros(y.shape[0], 7, 6, 6, 6)
        c = y.coords.long().cpu()
        yd[c[:, 0], :, c[:, 1], c[:, 2], c[:, 3]] = y.feats.cpu()
        ref = F.conv_transpose3d(
            yd, up.conv.weight.permute(4, 0, 1, 2, 3).cpu(), up.conv.bias.cpu(), stride=2, padding=1, output_padding=1
        )
    assert torch.equal(z.coords, x.coords)
    torch.testing.assert_close(z.feats.cpu(), _gather(ref, x.coords.cpu()), atol=1e-4, rtol=1e-4)


def test_inverse_conv_requires_its_forward_conv():
    x, _ = _random_sparse("cpu")
    with pytest.raises(ValueError, match="cache"):
        SparseInverseConv3d(5, 4, 3, stride=2)(x)


def test_inverted_kernel_map_is_the_transpose():
    x, _ = _random_sparse("cpu")
    kmap = submanifold_kmap(x.coords, (3, 3, 3))
    inv = invert_kmap(kmap, x.coords.shape[0])
    m, k = (kmap >= 0).nonzero(as_tuple=True)
    # submanifold maps are symmetric: neighbour at +offset is the reverse of neighbour at -offset
    assert torch.equal(inv[kmap[m, k].long(), k], m.int())
    assert int((kmap >= 0).sum()) == int((inv >= 0).sum())


def test_torchsparse_kernel_layout_is_converted_on_load():
    """Direct3D-S2 checkpoints hold ``conv.kernel`` (K^3, in, out) with x varying fastest; ``(in, out)`` for 1x1x1."""
    torch.manual_seed(4)
    conv = SparseConv3d(4, 6, 3)
    kernel = torch.randn(27, 4, 6)
    conv.load_state_dict({"conv.kernel": kernel, "conv.bias": torch.zeros(6)}, strict=True)
    for dx, dy, dz in [(0, 0, 0), (2, 0, 0), (0, 2, 1), (1, 2, 2)]:
        torch.testing.assert_close(conv.conv.weight[:, dx, dy, dz, :], kernel[dz * 9 + dy * 3 + dx].T)
    assert torchsparse_kernel_to_weight(torch.randn(4, 6), (1, 1, 1)).shape == (6, 1, 1, 1, 4)

    one = SparseConv3d(4, 6, 1)
    one.load_state_dict({"conv.kernel": torch.randn(4, 6), "conv.bias": torch.zeros(6)}, strict=True)

    # the spconv layout loads untouched
    other = SparseConv3d(4, 6, 3)
    other.load_state_dict(conv.state_dict(), strict=True)
    assert torch.equal(other.conv.weight, conv.conv.weight)


def test_even_kernel_needs_a_stride():
    with pytest.raises(NotImplementedError):
        SparseConv3d(2, 2, 2)
    SparseConv3d(2, 2, 2, stride=2)
