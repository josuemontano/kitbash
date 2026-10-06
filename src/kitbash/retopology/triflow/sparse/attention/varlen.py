# Written for kitbash (replaces flash_attn / xformers block-diagonal attention). Carries no upstream code.
"""Variable-length scaled-dot-product attention on top of ``torch.nn.functional.scaled_dot_product_attention``."""

import torch
import torch.nn.functional as F

# Upper bound for the attention-score tensor (B * H * Lq * Lk * 4 bytes) of one SDPA call on devices that may materialise it.
SCORE_BUDGET_BYTES = 256 * 2**20


def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """``(B, H, Lq, C)``, ``(B, H, Lk, C)``, ``(B, H, Lk, Co)`` -> ``(B, H, Lq, Co)``; queries are chunked to bound memory.

    ``mask`` is an optional boolean key mask broadcastable to ``(B, H, Lq, Lk)`` (True = attend).
    """
    if q.device.type == "cuda":
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    B, H, Lq, _ = q.shape
    per_query = max(1, B * H * k.shape[2] * 4)
    chunk = max(1, SCORE_BUDGET_BYTES // per_query)
    if chunk >= Lq:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    outs = []
    for s in range(0, Lq, chunk):
        m = mask
        if m is not None and m.shape[2] != 1:
            m = m[:, :, s : s + chunk]
        outs.append(F.scaled_dot_product_attention(q[:, :, s : s + chunk], k, v, attn_mask=m))
    return torch.cat(outs, dim=2)


def varlen_self_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_lens: list[int]) -> torch.Tensor:
    """Attention inside each packed sequence: ``(M, H, C)`` tensors, sequences of ``seq_lens`` rows laid out back to back.

    Sequences are sorted by length and padded in groups (with a key mask), so thousands of small windows cost a few SDPA
    calls instead of thousands.
    """
    M, H, _ = q.shape
    device = q.device
    out = torch.empty((M, H, v.shape[-1]), dtype=q.dtype, device=device)
    starts, acc = [], 0
    for n in seq_lens:
        starts.append(acc)
        acc += n
    order = sorted(range(len(seq_lens)), key=seq_lens.__getitem__)

    qp, kp, vp = (torch.cat([t, t.new_zeros(1, *t.shape[1:])], dim=0) for t in (q, k, v))
    i = 0
    while i < len(order):
        j = i
        while j < len(order):
            width = seq_lens[order[j]]
            if j > i and (j - i + 1) * H * width * width * 4 > SCORE_BUDGET_BYTES:
                break
            j += 1
        group = order[i:j]
        i = j
        lmax = seq_lens[group[-1]]
        lens = torch.tensor([seq_lens[g] for g in group], device=device)
        st = torch.tensor([starts[g] for g in group], device=device)
        ar = torch.arange(lmax, device=device)
        valid = ar[None, :] < lens[:, None]
        idx = torch.where(valid, st[:, None] + ar[None, :], torch.full_like(valid, M, dtype=torch.long))
        qg, kg, vg = (t[idx].transpose(1, 2) for t in (qp, kp, vp))  # (G, H, L, C)
        mask = None if bool(valid.all()) else valid[:, None, None, :]
        og = sdpa(qg, kg, vg, mask).transpose(1, 2)  # (G, L, H, Co)
        out[idx[valid]] = og[valid]
    return out
