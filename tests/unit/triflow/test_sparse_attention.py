"""Windowed / full / cross / serialized sparse attention must equal a naive dense masked softmax(QK^T)V."""

import pytest
import torch

from kitbash.retopology.triflow.sparse import (
    SparseMultiHeadAttention,
    SparseTensor,
    sparse_scaled_dot_product_attention,
    sparse_serialized_scaled_dot_product_self_attention,
    sparse_windowed_scaled_dot_product_self_attention,
)
from kitbash.retopology.triflow.sparse.attention import varlen
from kitbash.retopology.triflow.sparse.attention.serialized_attn import calc_serialization

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
H, C = 3, 8


def _coords(batch_sizes, grid, seed=0):
    g = torch.Generator().manual_seed(seed)
    rows = []
    for b, n in enumerate(batch_sizes):
        flat = torch.randperm(grid**3, generator=g)[:n].sort().values
        xyz = torch.stack([flat // (grid * grid), (flat // grid) % grid, flat % grid], dim=1)
        rows.append(torch.cat([torch.full((n, 1), b), xyz], dim=1))
    return torch.cat(rows).int()


def _naive(q, k, v, allowed):
    """q/k/v (T, H, C); allowed (Tq, Tk) bool."""
    scores = torch.einsum("qhc,khc->hqk", q, k) / q.shape[-1] ** 0.5
    scores = scores.masked_fill(~allowed[None], float("-inf"))
    return torch.einsum("hqk,khc->qhc", scores.softmax(-1), v)


def _same_batch(coords):
    b = coords[:, 0]
    return b[:, None] == b[None, :]


@pytest.mark.parametrize("device", DEVICES)
def test_full_self_attention(device):
    torch.manual_seed(0)
    coords = _coords([37, 20], 8)
    qkv = SparseTensor(torch.randn(coords.shape[0], 3, H, C).to(device), coords.to(device))
    out = sparse_scaled_dot_product_attention(qkv)
    q, k, v = qkv.feats.cpu().unbind(1)
    ref = _naive(q, k, v, _same_batch(coords))
    torch.testing.assert_close(out.feats.cpu(), ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kv_kind", ["sparse", "dense", "separate"])
def test_cross_attention(device, kv_kind):
    torch.manual_seed(1)
    q_coords, kv_coords = _coords([25, 14], 8, seed=1), _coords([40, 33], 8, seed=2)
    q = SparseTensor(torch.randn(q_coords.shape[0], H, C).to(device), q_coords.to(device))
    kv_feats = torch.randn(kv_coords.shape[0], 2, H, C)
    allowed = q_coords[:, :1] == kv_coords[:, 0][None, :]
    if kv_kind == "sparse":
        out = sparse_scaled_dot_product_attention(q, SparseTensor(kv_feats.to(device), kv_coords.to(device)))
        k, v = kv_feats.unbind(1)
    elif kv_kind == "separate":
        k, v = kv_feats.unbind(1)
        out = sparse_scaled_dot_product_attention(
            q, SparseTensor(k.to(device), kv_coords.to(device)), SparseTensor(v.to(device), kv_coords.to(device))
        )
    else:  # dense context, equal length per batch element
        dense = torch.randn(2, 11, 2, H, C)
        out = sparse_scaled_dot_product_attention(q, dense.to(device))
        k = dense[:, :, 0].reshape(-1, H, C)
        v = dense[:, :, 1].reshape(-1, H, C)
        allowed = q_coords[:, :1] == torch.arange(2).repeat_interleave(11)[None, :]
    ref = _naive(q.feats.cpu(), k, v, allowed)
    torch.testing.assert_close(out.feats.cpu(), ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shift", [0, 4])
def test_windowed_self_attention(device, shift):
    torch.manual_seed(2)
    ws = 8
    coords = _coords([180, 95], 20, seed=3)
    qkv = SparseTensor(torch.randn(coords.shape[0], 3, H, C).to(device), coords.to(device))
    out = sparse_windowed_scaled_dot_product_self_attention(qkv, ws, shift)
    win = torch.cat([coords[:, :1], (coords[:, 1:] + shift) // ws], dim=1)
    allowed = (win[:, None, :] == win[None, :, :]).all(-1)
    q, k, v = qkv.feats.cpu().unbind(1)
    torch.testing.assert_close(out.feats.cpu(), _naive(q, k, v, allowed), atol=1e-4, rtol=1e-4)
    # the partition is cached on the coordinate set and reused by tensors derived from it
    assert len(qkv._coord_cache) == 1
    assert qkv.replace(qkv.feats * 2)._coord_cache is qkv._coord_cache


def test_windowed_attention_group_and_query_chunking(monkeypatch):
    """A tiny score budget forces several padded groups and chunked queries; the result must not change."""
    torch.manual_seed(3)
    coords = _coords([300], 16, seed=4)
    qkv = SparseTensor(torch.randn(coords.shape[0], 3, H, C), coords)
    full = sparse_windowed_scaled_dot_product_self_attention(qkv, 8, 0).feats
    monkeypatch.setattr(varlen, "SCORE_BUDGET_BYTES", 4 * 1024)
    qkv2 = SparseTensor(qkv.feats, coords)
    chunked = sparse_windowed_scaled_dot_product_self_attention(qkv2, 8, 0).feats
    torch.testing.assert_close(chunked, full, atol=1e-5, rtol=1e-5)
    cross = sparse_scaled_dot_product_attention(
        SparseTensor(qkv.feats[:, 0], coords), SparseTensor(qkv.feats[:, 1:], coords)
    ).feats
    monkeypatch.setattr(varlen, "SCORE_BUDGET_BYTES", 256 * 2**20)
    ref = sparse_scaled_dot_product_attention(SparseTensor(qkv.feats[:, 0], coords), SparseTensor(qkv.feats[:, 1:], coords)).feats
    torch.testing.assert_close(cross, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("window", [64, 16])
def test_serialized_self_attention(window):
    torch.manual_seed(4)
    coords = _coords([50, 41], 10, seed=5)
    qkv = SparseTensor(torch.randn(coords.shape[0], 3, H, C), coords)
    out = sparse_serialized_scaled_dot_product_self_attention(qkv, window)
    fwd, bwd, seq_lens, _ = calc_serialization(qkv, window)
    q, k, v = qkv.feats.unbind(1)
    pieces, start = [], 0
    for n in seq_lens:
        rows = fwd[start : start + n]
        pieces.append(_naive(q[rows], k[rows], v[rows], torch.ones(n, n, dtype=torch.bool)))
        start += n
    ref = torch.cat(pieces)[bwd]
    torch.testing.assert_close(out.feats, ref, atol=1e-4, rtol=1e-4)
    if window == 64:  # one window per batch element: plain full attention in a space-filling-curve order
        torch.testing.assert_close(out.feats, _naive(q, k, v, _same_batch(coords)), atol=1e-4, rtol=1e-4)


def test_multi_head_attention_module_modes():
    torch.manual_seed(5)
    coords = _coords([60, 30], 12, seed=6)
    x = SparseTensor(torch.randn(coords.shape[0], 24), coords)
    ctx = SparseTensor(torch.randn(50, 16), _coords([30, 20], 8, seed=7))
    for mode, kwargs in [("full", {}), ("windowed", {"window_size": 4, "shift_window": 2})]:
        attn = SparseMultiHeadAttention(24, num_heads=3, attn_mode=mode, qk_rms_norm=True, **kwargs)
        out = attn(x)
        assert out.feats.shape == (coords.shape[0], 24)
        assert torch.isfinite(out.feats).all()
    cross = SparseMultiHeadAttention(24, num_heads=3, ctx_channels=16, type="cross")
    out = cross(x, ctx)
    assert out.feats.shape == (coords.shape[0], 24)
    # permuting the context rows (within a batch element) cannot change cross attention
    perm = torch.cat([torch.randperm(30), 30 + torch.randperm(20)])
    shuffled = SparseTensor(ctx.feats[perm], ctx.coords[perm])
    torch.testing.assert_close(cross(x, shuffled).feats, out.feats, atol=1e-5, rtol=1e-5)
