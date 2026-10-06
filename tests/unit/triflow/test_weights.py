import contextlib
import hashlib

import pytest

from kitbash.errors import PreflightError, RetopologyError
from kitbash.retopology.triflow import weights

PAYLOAD = b"not really a checkpoint"


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def iter_bytes(self, size: int):
        for start in range(0, len(self._payload), size):
            yield self._payload[start : start + size]


@pytest.fixture
def fake_hub(monkeypatch):
    """Serve PAYLOAD for every file and pin a single checkpoint to PAYLOAD's hash."""
    calls: list[str] = []

    @contextlib.contextmanager
    def stream(method, url, **kwargs):
        calls.append(url)
        yield FakeResponse(PAYLOAD)

    monkeypatch.setattr(weights.httpx, "stream", stream)
    pinned = weights.Checkpoint("only", hashlib.sha256(PAYLOAD).hexdigest())
    monkeypatch.setattr(weights, "CHECKPOINTS", (pinned,))
    return calls


def test_check_reports_missing_weights_without_download(tmp_path):
    with pytest.raises(PreflightError, match="weights missing"):
        weights.check(tmp_path, allow_download=False)


def test_check_accepts_missing_weights_when_download_allowed(tmp_path):
    weights.check(tmp_path, allow_download=True)


def test_ensure_downloads_verifies_and_cleans_up(tmp_path, fake_hub):
    paths = weights.ensure(tmp_path)
    assert paths["only"].read_bytes() == PAYLOAD
    assert sorted(p.name for p in tmp_path.iterdir()) == ["only.safetensors"]  # no .part, no .lock
    assert fake_hub == [f"{weights.BASE_URL}/only.safetensors"]
    weights.ensure(tmp_path)
    assert len(fake_hub) == 1  # present files are never fetched again


def test_checksum_mismatch_leaves_nothing_behind(tmp_path, fake_hub, monkeypatch):
    monkeypatch.setattr(weights, "CHECKPOINTS", (weights.Checkpoint("only", "0" * 64),))
    with pytest.raises(RetopologyError, match="Checksum mismatch"):
        weights.ensure(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_concurrent_download_is_refused(tmp_path, fake_hub):
    (tmp_path / "only.safetensors.lock").write_text("")
    with pytest.raises(RetopologyError, match="Another process"):
        weights.ensure(tmp_path)
    assert (tmp_path / "only.safetensors.lock").exists()  # someone else's lock is not ours to remove
    assert fake_hub == []


def test_stale_partial_file_is_never_resumed(tmp_path, fake_hub):
    (tmp_path / "only.safetensors.part").write_bytes(b"garbage from an earlier crash")
    assert weights.ensure(tmp_path)["only"].read_bytes() == PAYLOAD


def test_corrupted_cached_checkpoint_is_rejected_without_download(tmp_path, fake_hub):
    path = tmp_path / "only.safetensors"
    path.write_bytes(b"wrong but already cached")
    with pytest.raises(RetopologyError, match="Checksum mismatch for cached"):
        weights.ensure(tmp_path)
    assert path.read_bytes() == b"wrong but already cached"
    assert fake_hub == []


def test_disallow_download_at_the_loading_boundary(tmp_path, fake_hub):
    with pytest.raises(RetopologyError, match="weights missing"):
        weights.ensure(tmp_path, allow_download=False)
    assert fake_hub == []
