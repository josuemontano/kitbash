# Vendored from TRELLIS (microsoft/TRELLIS, Copyright (c) Microsoft Corporation, MIT License) and Direct3D-S2
# (DreamTechAI/Direct3D-S2, Copyright (c) 2025 DreamTechAI, MIT License); the two `modules/sparse` packages are unified.
# Modified for kitbash: flash_attn / xformers are replaced by torch scaled_dot_product_attention (one call per batch element).

import torch

from ..basic import SparseTensor
from .varlen import sdpa

__all__ = [
    "sparse_scaled_dot_product_attention",
]


def sparse_scaled_dot_product_attention(*args, **kwargs):
    """Scaled dot product attention over (per batch element) variable-length sparse sequences.

    Accepted call forms (q/k/v/kv/qkv may be given positionally or by keyword):
      - ``(qkv)``: SparseTensor [N, *, 3, H, C].
      - ``(q, kv)``: q SparseTensor [N, *, H, C] with kv SparseTensor [N, *, 2, H, C] or dense [N, L, 2, H, C];
        or q dense [N, L, H, C] with kv SparseTensor.
      - ``(q, k, v)``: q SparseTensor with k, v SparseTensors [N, *, H, C] or dense [N, L, H, C]; or q dense with k, v sparse.
    Returns a SparseTensor (sparse q) or a dense [N, L, H, Co] tensor (dense q).
    """
    arg_names_dict = {1: ["qkv"], 2: ["q", "kv"], 3: ["q", "k", "v"]}
    num_all_args = len(args) + len(kwargs)
    assert num_all_args in arg_names_dict, f"Invalid number of arguments, got {num_all_args}, expected 1, 2, or 3"
    for key in arg_names_dict[num_all_args][len(args) :]:
        assert key in kwargs, f"Missing argument {key}"

    if num_all_args == 1:
        qkv = args[0] if len(args) > 0 else kwargs["qkv"]
        assert isinstance(qkv, SparseTensor), f"qkv must be a SparseTensor, got {type(qkv)}"
        assert len(qkv.shape) == 4 and qkv.shape[1] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"
        s = qkv
        q_seqlen = [qkv.layout[i].stop - qkv.layout[i].start for i in range(qkv.shape[0])]
        kv_seqlen = q_seqlen
        q, k, v = qkv.feats.unbind(dim=1)  # [T, H, C]

    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs["q"]
        kv = args[1] if len(args) > 1 else kwargs["kv"]
        assert (
            (isinstance(q, SparseTensor)
            and isinstance(kv, SparseTensor | torch.Tensor))
            or (isinstance(q, torch.Tensor)
            and isinstance(kv, SparseTensor))
        ), f"Invalid types, got {type(q)} and {type(kv)}"
        assert q.shape[0] == kv.shape[0], f"Batch size mismatch, got {q.shape[0]} and {kv.shape[0]}"

        if isinstance(q, SparseTensor):
            assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, C]"
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats  # [T_Q, H, C]
        else:
            assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, C]"
            s = None
            N, L, H, C = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, C)

        if isinstance(kv, SparseTensor):
            assert len(kv.shape) == 4 and kv.shape[1] == 2, f"Invalid shape for kv, got {kv.shape}, expected [N, *, 2, H, C]"
            kv_seqlen = [kv.layout[i].stop - kv.layout[i].start for i in range(kv.shape[0])]
            kv = kv.feats  # [T_KV, 2, H, C]
        else:
            assert len(kv.shape) == 5, f"Invalid shape for kv, got {kv.shape}, expected [N, L, 2, H, C]"
            N, L, _, H, C = kv.shape
            kv_seqlen = [L] * N
            kv = kv.reshape(N * L, 2, H, C)
        k, v = kv.unbind(dim=1)

    else:
        q = args[0] if len(args) > 0 else kwargs["q"]
        k = args[1] if len(args) > 1 else kwargs["k"]
        v = args[2] if len(args) > 2 else kwargs["v"]
        assert (
            (isinstance(q, SparseTensor)
            and isinstance(k, SparseTensor | torch.Tensor)
            and type(k) is type(v))
            or (isinstance(q, torch.Tensor)
            and isinstance(k, SparseTensor)
            and isinstance(v, SparseTensor))
        ), f"Invalid types, got {type(q)}, {type(k)}, and {type(v)}"
        assert q.shape[0] == k.shape[0] == v.shape[0], f"Batch size mismatch, got {q.shape[0]}, {k.shape[0]}, and {v.shape[0]}"

        if isinstance(q, SparseTensor):
            assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, Ci]"
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats
        else:
            assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, Ci]"
            s = None
            N, L, H, CI = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, CI)

        if isinstance(k, SparseTensor):
            assert len(k.shape) == 3, f"Invalid shape for k, got {k.shape}, expected [N, *, H, Ci]"
            assert len(v.shape) == 3, f"Invalid shape for v, got {v.shape}, expected [N, *, H, Co]"
            kv_seqlen = [k.layout[i].stop - k.layout[i].start for i in range(k.shape[0])]
            k = k.feats
            v = v.feats
        else:
            assert len(k.shape) == 4, f"Invalid shape for k, got {k.shape}, expected [N, L, H, Ci]"
            assert len(v.shape) == 4, f"Invalid shape for v, got {v.shape}, expected [N, L, H, Co]"
            N, L, H, CI, CO = *k.shape, v.shape[-1]
            kv_seqlen = [L] * N
            k = k.reshape(N * L, H, CI)
            v = v.reshape(N * L, H, CO)

    outs = []
    qs = ks = 0
    for lq, lk in zip(q_seqlen, kv_seqlen, strict=True):
        qb = q[qs : qs + lq].transpose(0, 1).unsqueeze(0)  # [1, H, Lq, C]
        kb = k[ks : ks + lk].transpose(0, 1).unsqueeze(0)
        vb = v[ks : ks + lk].transpose(0, 1).unsqueeze(0)
        outs.append(sdpa(qb, kb, vb)[0].transpose(0, 1))  # [Lq, H, Co]
        qs += lq
        ks += lk
    out = torch.cat(outs, dim=0)

    if s is not None:
        return s.replace(out)
    return out.reshape(len(q_seqlen), q_seqlen[0], out.shape[1], -1)
