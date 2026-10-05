"""Text embedding backends. All vectors are L2-normalized so cosine similarity is a dot product."""

import hashlib
import itertools
import math
import os
import re
import threading
from collections.abc import Sequence
from typing import Protocol

import httpx

from kitbash.config import EmbeddingConfig
from kitbash.errors import ConfigError, KitbashError

type Vector = list[float]


class Embedder(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[Vector]: ...


def normalize(vector: Sequence[float]) -> Vector:
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [float(v) / norm for v in vector]


class SentenceTransformerEmbedder:
    """Local sentence-embedding model (loaded lazily on first use)."""

    def __init__(self, model: str, dimensions: int) -> None:
        self._model_name = model
        self._dimensions = dimensions
        self._model = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return f"sentence-transformers:{self._model_name}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        with self._lock:
            if self._model is None:
                self._model = self._load()
            vectors = self._model.encode(list(texts), normalize_embeddings=True)
        result = [normalize(v.tolist()) for v in vectors]
        if result and len(result[0]) != self._dimensions:
            raise ConfigError(
                f"Embedding model {self._model_name} returns {len(result[0])} dimensions, config says {self._dimensions}",
                hint="Set embedding.dimensions to match the model.",
            )
        return result

    def _load(self):
        # Keep Hugging Face progress bars and warnings out of the live terminal display.
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("TQDM_DISABLE", "1")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise KitbashError(
                "sentence-transformers is not installed", hint="Run `poetry install` or set embedding.backend."
            ) from exc
        return SentenceTransformer(self._model_name)


class HttpEmbedder:
    """OpenAI-compatible ``/embeddings`` endpoint, e.g. LM Studio or Ollama."""

    def __init__(self, base_url: str, model: str, dimensions: int, timeout_s: float) -> None:
        self._url = base_url.rstrip("/") + "/embeddings"
        self._model = model
        self._dimensions = dimensions
        self._timeout = timeout_s

    @property
    def name(self) -> str:
        return f"http:{self._model}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        try:
            response = httpx.post(self._url, json={"model": self._model, "input": list(texts)}, timeout=self._timeout)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise KitbashError(f"Embedding request to {self._url} failed: {exc}") from exc
        data = sorted(response.json()["data"], key=lambda item: item["index"])
        return [normalize(item["embedding"]) for item in data]


class HashingEmbedder:
    """Dependency-free bag-of-words hashing embedder (offline runs and tests)."""

    _TOKEN = re.compile(r"[a-z0-9]+")

    def __init__(self, dimensions: int = 256) -> None:
        self._dimensions = dimensions

    @property
    def name(self) -> str:
        return f"hashing:{self._dimensions}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> Vector:
        vector = [0.0] * self._dimensions
        tokens = self._TOKEN.findall(text.lower())
        features = tokens + [f"{a}_{b}" for a, b in itertools.pairwise(tokens)]
        for feature in features:
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self._dimensions
            vector[index] += 1.0 if digest[4] & 1 else -1.0
        return normalize(vector)


def make_embedder(config: EmbeddingConfig) -> Embedder:
    match config.backend:
        case "sentence-transformers":
            return SentenceTransformerEmbedder(config.model, config.dimensions)
        case "http":
            return HttpEmbedder(config.base_url, config.model, config.dimensions, config.timeout_s)
        case "hashing":
            return HashingEmbedder(config.dimensions)
    raise ConfigError(
        f"Unknown embedding backend {config.backend!r}", hint="Use sentence-transformers, http or hashing."
    )
