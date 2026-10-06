import pytest
import torch

from kitbash.retopology.triflow.device import autocast, resolve_device


def test_cpu_is_rejected():
    with pytest.raises(ValueError, match="Unknown device"):
        resolve_device("cpu")


def test_auto_picks_an_available_device():
    if torch.cuda.is_available():
        assert resolve_device("auto").type == "cuda"
    elif torch.backends.mps.is_available():
        assert resolve_device("auto").type == "mps"
    else:
        with pytest.raises(ValueError, match="no GPU"):
            resolve_device("auto")


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError, match="Unknown device"):
        resolve_device("tpu")


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA present")
def test_explicit_cuda_without_cuda_fails_clearly():
    with pytest.raises(ValueError, match="CUDA is not available"):
        resolve_device("cuda")


def test_autocast_is_a_noop_off_cuda():
    with autocast(torch.device("cpu")):
        assert (torch.ones(2) @ torch.ones(2)).dtype == torch.float32
