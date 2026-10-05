"""Poly Haven catalog lookups (HDRIs and textures) offered to the code writer as optional downloads."""

import json
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

API = "https://api.polyhaven.com/assets"
ENVIRONMENT_CATEGORIES = {"indoor": "indoor", "outdoor": "outdoor", "studio": "studio"}


class PolyHavenCatalog:
    def __init__(self, client: httpx.Client, cache_dir: Path, *, enabled: bool = True, ttl_s: float = 86_400) -> None:
        self._client = client
        self._cache_dir = cache_dir
        self._enabled = enabled
        self._ttl_s = ttl_s

    def hdris(self, environment: str, limit: int = 10) -> list[dict[str, Any]]:
        category = ENVIRONMENT_CATEGORIES.get(environment.lower(), "outdoor")
        assets = self._assets("hdris")
        chosen = [(k, v) for k, v in assets.items() if category in v.get("categories", [])]
        chosen.sort(key=lambda kv: -kv[1].get("download_count", 0))
        return [_summary(k, v) for k, v in chosen[:limit]]

    def textures(self, hints: Sequence[str], per_hint: int = 3) -> list[dict[str, Any]]:
        assets = self._assets("textures")
        results: dict[str, dict[str, Any]] = {}
        for hint in hints:
            words = set(re.findall(r"[a-z]+", hint.lower())) - {"and", "with", "the", "of"}
            scored = []
            for key, value in assets.items():
                vocabulary = set(value.get("tags", [])) | set(value.get("categories", [])) | set(value.get("name", "").lower().split())
                overlap = len(words & {w.lower() for w in vocabulary})
                if overlap:
                    scored.append((overlap, value.get("download_count", 0), key, value))
            scored.sort(key=lambda s: (-s[0], -s[1]))
            for _, _, key, value in scored[:per_hint]:
                results.setdefault(key, {**_summary(key, value), "for": hint})
        return list(results.values())

    def _assets(self, kind: str) -> dict[str, Any]:
        if not self._enabled:
            return {}
        cache = self._cache_dir / f"polyhaven_{kind}.json"
        if cache.is_file() and time.time() - cache.stat().st_mtime < self._ttl_s:
            return json.loads(cache.read_text(encoding="utf-8"))
        try:
            data = self._client.get(API, params={"t": kind}).raise_for_status().json()
        except httpx.HTTPError:
            return json.loads(cache.read_text(encoding="utf-8")) if cache.is_file() else {}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data), encoding="utf-8")
        return data


def _summary(key: str, value: dict[str, Any]) -> dict[str, Any]:
    return {"id": key, "name": value.get("name", key), "tags": value.get("tags", [])[:8]}
