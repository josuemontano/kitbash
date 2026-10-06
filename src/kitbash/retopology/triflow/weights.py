"""TriFlow checkpoints: pinned, downloaded once from Hugging Face, verified by SHA-256."""

import hashlib
import os
from pathlib import Path

import httpx
from attrs import frozen

from kitbash.errors import PreflightError, RetopologyError

REPO = "lihcxr/TriFlow"
REVISION = "e3403f35d0566a0487b9c8f3816bc6cdee814dca"
BASE_URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}"


@frozen
class Checkpoint:
    name: str
    sha256: str

    @property
    def filename(self) -> str:
        return f"{self.name}.safetensors"


CHECKPOINTS = (
    Checkpoint("sdf_vae", "a0ee5844b54d179b6724d332fa67d8fdc5c3a9904e103dc0b3a93b073311d94d"),
    Checkpoint("nvv_vae", "46c941c826c83f1e9f00d557dd655e41bc407693db33b94b422ae10313fe829d"),
    Checkpoint("flow_model", "f724b037546d1930e39c3f20edb16f5af3ff3f24f3969ade5daf74ffbdab624b"),
)


def checkpoint_path(weights_dir: Path, name: str) -> Path:
    return weights_dir / f"{name}.safetensors"


def missing(weights_dir: Path) -> list[Checkpoint]:
    return [c for c in CHECKPOINTS if not checkpoint_path(weights_dir, c.name).is_file()]


def _verify_cached(weights_dir: Path) -> None:
    for checkpoint in CHECKPOINTS:
        path = checkpoint_path(weights_dir, checkpoint.name)
        if not path.is_file():
            continue
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != checkpoint.sha256:
            raise RetopologyError(
                f"Checksum mismatch for cached {checkpoint.filename} in {weights_dir}",
                hint="Remove the corrupt checkpoint and download the pinned version again.",
            )


def check(weights_dir: Path, *, allow_download: bool) -> None:
    """Preflight: weights present, or downloadable."""
    _verify_cached(weights_dir)
    absent = missing(weights_dir)
    if absent and not allow_download:
        names = ", ".join(c.filename for c in absent)
        raise PreflightError(
            f"TriFlow weights missing in {weights_dir}: {names}",
            hint="Run `kitbash retopology download-weights`. (They are also downloaded automatically on first use.)",
        )


def ensure(weights_dir: Path, *, timeout_s: float = 60.0, allow_download: bool = True) -> dict[str, Path]:
    """Return SHA-256-verified checkpoint paths; download only when permitted."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    _verify_cached(weights_dir)
    absent = missing(weights_dir)
    if absent and not allow_download:
        raise RetopologyError(f"TriFlow weights missing in {weights_dir}: {', '.join(c.filename for c in absent)}")
    for checkpoint in absent:
        _download(checkpoint, weights_dir, timeout_s)
    return {c.name: checkpoint_path(weights_dir, c.name) for c in CHECKPOINTS}


def _download(checkpoint: Checkpoint, weights_dir: Path, timeout_s: float) -> None:
    target = checkpoint_path(weights_dir, checkpoint.name)
    part = target.with_suffix(".safetensors.part")
    lock = target.with_suffix(".safetensors.lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)  # one downloader per file, ever
    except FileExistsError:
        raise RetopologyError(
            f"Another process is downloading {checkpoint.filename} (lock file {lock})",
            hint="Wait for it to finish, or delete the lock file if no download is running.",
        ) from None
    try:
        os.close(descriptor)
        part.unlink(missing_ok=True)  # never resume: a partial file of unknown provenance is not trusted
        digest = hashlib.sha256()
        try:
            with httpx.stream("GET", f"{BASE_URL}/{checkpoint.filename}", follow_redirects=True, timeout=timeout_s) as response:
                response.raise_for_status()
                with part.open("wb") as handle:
                    for chunk in response.iter_bytes(1 << 20):
                        handle.write(chunk)
                        digest.update(chunk)
        except httpx.HTTPError as exc:
            raise RetopologyError(f"Could not download {checkpoint.filename}: {exc}", hint=f"Source: {BASE_URL}/{checkpoint.filename}") from exc
        if digest.hexdigest() != checkpoint.sha256:
            raise RetopologyError(
                f"Checksum mismatch for {checkpoint.filename} (got {digest.hexdigest()[:16]}…, expected {checkpoint.sha256[:16]}…)",
                hint="The download was corrupted or the pinned revision changed; try again.",
            )
        part.replace(target)
    finally:
        part.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)
