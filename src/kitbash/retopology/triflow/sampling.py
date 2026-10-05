# Copyright (c) 2026 Haoxuan Li.
# Licensed under the Automotive Development Public Non-Commercial License v1.0.
# See licenses/TriFlow-ADPNCL-1.0.txt for details.
#
# Adapted from TriFlow's triflow/utils/sampling.py and TRELLIS's trellis/pipelines/samplers/flow_euler.py
# (microsoft/TRELLIS, MIT).
# Modified for kitbash: standalone (no TRELLIS/easydict/tqdm imports); only the Euler loop inference needs, with a
# progress callback instead of a tqdm bar and a cancellation hook.

from collections.abc import Callable

import numpy as np
import torch


@torch.no_grad()
def euler_sample(
    model,
    noise,
    cond,
    *,
    steps: int = 50,
    on_step: Callable[[int, int], None] | None = None,
):
    """Integrate the flow from ``noise`` (t=1) to a sample (t=0) with ``steps`` Euler steps.

    ``model(x_t, t, cond)`` predicts the velocity; ``t`` is passed as a per-sample tensor in [0, 1000], the range the
    flow model was trained with. ``noise`` may be a SparseTensor: it only needs ``-`` and scalar ``*``.
    """
    sample = noise
    t_seq = np.linspace(1.0, 0.0, steps + 1)
    batch = noise.shape[0]
    device = noise.feats.device if hasattr(noise, "feats") else noise.device
    for i in range(steps):
        t, t_prev = float(t_seq[i]), float(t_seq[i + 1])
        t_tensor = torch.full((batch,), 1000.0 * t, device=device, dtype=torch.float32)
        velocity = model(sample, t_tensor, cond)
        sample = sample - (t - t_prev) * velocity
        if on_step is not None:
            on_step(i + 1, steps)
    return sample
