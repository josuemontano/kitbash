"""Device selection and numeric policy for the TriFlow networks.

Upstream hard-codes ``.cuda()`` and fp16 autocast; here the device is chosen at run time and half precision is only
used on CUDA, where it is known to be safe. MPS and CPU run in float32.
"""

import contextlib

import torch

DEVICES = ("auto", "cuda", "mps", "cpu")


def resolve_device(name: str = "auto") -> torch.device:
    """``auto`` prefers CUDA, then Apple MPS, then CPU. An explicit device that is unavailable raises ValueError."""
    if name not in DEVICES:
        raise ValueError(f"Unknown device {name!r}; use one of {', '.join(DEVICES)}")
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("device 'cuda' requested but CUDA is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("device 'mps' requested but MPS is not available")
    return torch.device(name)


def autocast(device: torch.device):
    """fp16 autocast on CUDA (upstream behaviour); a no-op everywhere else."""
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()
