"""Acquire references without model calls unless visual fallback is explicitly enabled."""

import hashlib
import json
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
from attrs import field, frozen

from kitbash.analytics import context
from kitbash.analytics.tracker import EventKind, Tracker
from kitbash.config import ReferenceConfig
from kitbash.domain.inventory import InventoryItem
from kitbash.domain.phases import PhaseName
from kitbash.domain.roles import Role
from kitbash.errors import KitbashError, LLMAccessError, LLMError
from kitbash.infra.image_search import Candidate, CandidateProvider, ImageDownloader
from kitbash.infra.imaging import contact_sheet
from kitbash.llm.service import LLMService
from kitbash.services.reference_quality import assess, rights_allowed, token_overlap


@frozen
class ReferenceChoice:
    path: Path
    source: str
    title: str
    reason: str
    license: str = ""
    provenance: dict[str, Any] = field(factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "source": self.source, "title": self.title, "reason": self.reason,
                "license": self.license, "provenance": self.provenance}


class ReferenceFinder:
    def __init__(
        self, providers: Sequence[CandidateProvider], downloader: ImageDownloader, llm: LLMService,
        config: ReferenceConfig, tracker: Tracker, *, cache_dir: Path | None = None,
    ) -> None:
        self._providers = tuple(providers)
        self._downloader = downloader
        self._llm = llm
        self._config = config
        self._tracker = tracker
        self._cache_dir = cache_dir

    def find(self, item: InventoryItem, directory: Path) -> ReferenceChoice | None:
        """Explicit input, adequate native crop, licensed APIs; uncertainty stops before Trellis.

        Searched candidates (crops, Commons, Openverse) are accepted only when the object is isolated: a transparent
        or flat neutral background. Anything else is dropped before ranking, so it can neither be auto-selected nor offered
        for review. A user-supplied image is the user's own choice and is used as given."""
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("review.json", "selection.json", "contact_sheet.png"):
            (directory / name).unlink(missing_ok=True)
        candidates_dir = directory / "candidates"
        candidates_dir.mkdir(exist_ok=True)
        if item.user_reference:
            source = Path(item.user_reference).expanduser()
            candidate = Candidate("user", source.name, path=source, license_id="user-provided", license="User supplied; rights asserted by user")
            fetched = self._downloader.fetch(candidate, candidates_dir / "user.png")
            return self._choose(fetched, item, directory, "provided by the user") if fetched else None

        crops: list[Candidate] = []
        for provider in self._providers:
            if provider.name != "input_crop":
                continue
            for candidate in self._search(provider, item.search_name or item.name, item, candidates_dir):
                fetched = self._downloader.fetch(candidate, candidates_dir / "crop.png")
                if fetched is not None and rights_allowed(fetched, self._config.allowed_licenses):
                    quality = assess(fetched, item)
                    if self._automatic(quality):
                        return self._choose(fetched, item, directory, "detailed source crop passed the configured heuristic threshold")
                    crops.append(fetched)
        candidates = self._collect(item, candidates_dir, crops)
        if not candidates:
            return None
        quality = [assess(c, item) for c in candidates]
        top = quality[0]
        margin = float(top["score"]) - float(quality[1]["score"]) if len(quality) > 1 else 1.0
        if self._automatic(top) and margin >= self._config.ambiguity_margin:
            return self._choose(candidates[0], item, directory, "passed explicit rights, quality and ranking-margin gates; heuristics are not visual proof")
        sheet = contact_sheet([c.path for c in candidates], [str(i + 1) for i in range(len(candidates))], directory / "contact_sheet.png")
        rows = [{**c.to_dict(), "quality": q, "rights_allowed": True} for c, q in zip(candidates, quality, strict=True)]
        (directory / "review.json").write_text(json.dumps({"contact_sheet": str(sheet), "candidates": rows}, indent=2), encoding="utf-8")
        if self._config.vision_fallback and any(self._automatic(q) for q in quality):
            try:
                picked = self._select(item, candidates, directory)
            except LLMAccessError:
                raise
            except LLMError as exc:
                self._tracker.event(EventKind.WARNING, "reference_selection_failed", error=str(exc)[:300])
                picked = None
            if picked is not None and self._automatic(assess(picked[0], item)):
                return self._choose(picked[0], item, directory, f"optional visual fallback: {picked[1]}")
        return None

    def _automatic(self, quality: Mapping[str, Any]) -> bool:
        return bool(quality["automatic_eligible"] and quality["score"] >= self._config.auto_select_threshold)

    def select_reviewed(self, item: InventoryItem, directory: Path, index: int) -> ReferenceChoice | None:
        """Explicit human visual choice; recheck current rights, file limits and pixel identity."""
        try:
            rows = json.loads((directory / "review.json").read_text(encoding="utf-8"))["candidates"]
            if type(index) is not int or not 0 <= index < len(rows):
                return None
            row = rows[index]
            candidate = Candidate.from_dict({k: v for k, v in row.items() if k not in {"quality", "rights_allowed"}})
            if not rights_allowed(candidate, self._config.allowed_licenses):
                return None
            if candidate.source == "input_crop" and min(candidate.original_width, candidate.original_height) < self._config.min_crop_side_px:
                return None
            fetched = self._downloader.fetch(candidate, directory / "reviewed.png")
            if fetched is None or fetched.content_hash != candidate.content_hash:
                return None
            if not assess(fetched, item)["isolated"]:
                return None
            return self._choose(fetched, item, directory, "explicit human visual selection from ranked contact sheet")
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _collect(self, item: InventoryItem, directory: Path, initial: Sequence[Candidate] = ()) -> list[Candidate]:
        found: list[Candidate] = []
        for level, query in enumerate(search_queries(item.search_name or item.name, self._config.query_suffix)):
            for provider in self._providers:
                if provider.name == "input_crop" or (provider.once_per_item and level > 0):
                    continue
                found.extend(self._search(provider, query, item, directory))
        # Rank metadata BEFORE downloads; a broad query cannot win just by arriving first.
        found = [c for c in found if rights_allowed(c, self._config.allowed_licenses)]
        found.sort(key=lambda c: (-token_overlap(c, item), c.provider_rank, c.source, c.provider_id, c.url or ""))
        local = list(initial)
        seen_urls = {c.url for c in initial if c.url}
        seen_hashes = {c.content_hash for c in initial if c.content_hash}
        # Bound work independently of usable count; inspect more than the display budget before ranking pixels.
        attempts = 0
        for index, candidate in enumerate(found):
            if candidate.url in seen_urls:
                continue
            if attempts >= self._config.max_candidates * 4:
                break
            attempts += 1
            if candidate.url:
                seen_urls.add(candidate.url)
            fetched = self._downloader.fetch(candidate, directory / f"{index:03d}.png")
            if fetched is not None and fetched.content_hash not in seen_hashes:
                seen_hashes.add(fetched.content_hash)
                local.append(fetched)
        ranked = [(c, quality) for c in local if (quality := assess(c, item))["isolated"]]
        ranked.sort(key=lambda pair: (-float(pair[1]["score"]), pair[0].provider_rank, pair[0].source, pair[0].provider_id, pair[0].content_hash))
        ranked = ranked[: self._config.max_candidates]
        (directory / "candidates.json").write_text(
            json.dumps([{**c.to_dict(), "quality": q, "rights_allowed": True} for c, q in ranked], indent=2), encoding="utf-8",
        )
        return [c for c, _ in ranked]

    def _search(self, provider: CandidateProvider, query: str, item: InventoryItem, directory: Path) -> list[Candidate]:
        cache = None
        if self._cache_dir is not None and provider.name != "input_crop":
            key = hashlib.sha256(json.dumps([1, provider.name, query, self._config.per_provider]).encode()).hexdigest()
            cache = self._cache_dir / "searches" / f"{key}.json"
            try:
                data = json.loads(cache.read_text(encoding="utf-8"))
                if 0 <= time.time() - data["created"] < self._config.search_cache_ttl_s:
                    return [Candidate.from_dict(row) for row in data["candidates"]]
            except (OSError, ValueError, KeyError, TypeError):
                pass
        try:
            candidates = provider.search(query, item, self._config.per_provider, directory)
        except (httpx.HTTPError, KitbashError, ValueError, KeyError, TypeError, OSError) as exc:
            self._tracker.event(EventKind.WARNING, "reference_search_failed", provider=provider.name, error=str(exc)[:300])
            return []
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", dir=cache.parent, delete=False, encoding="utf-8") as handle:
                temp = Path(handle.name)
                json.dump({"created": time.time(), "candidates": [c.to_dict() for c in candidates]}, handle)
            try:
                temp.replace(cache)
            finally:
                temp.unlink(missing_ok=True)
        return candidates

    def _select(self, item: InventoryItem, candidates: Sequence[Candidate], directory: Path) -> tuple[Candidate, str] | None:
        sheet = directory / "contact_sheet.png"
        crop = next((c.path for c in candidates if c.source == "input_crop"), None)
        described = "; ".join(f"{i + 1}: {c.source} ({c.title[:60]})" for i, c in enumerate(candidates))
        count = len(candidates)

        def validate(data: Any) -> tuple[int | None, str]:
            if not isinstance(data, Mapping) or "choice" not in data:
                raise LLMError("Expected {'choice': number or null, 'reason': ...}")
            choice = data["choice"]
            if choice is not None and (type(choice) is not int or not 1 <= choice <= count):
                raise LLMError(f"choice must be null or an integer between 1 and {count}")
            return choice, str(data.get("reason", ""))

        with context.bind(agent="reference_selector"):
            choice, reason = self._llm.ask_json(
                task="modelling.reference.select", role=Role.REFERENCE_SELECTION, phase=PhaseName.MODELLING,
                template="reference_select",
                variables={"name": item.name, "description": item.description, "category": item.category,
                           "materials": ", ".join(item.materials_hint) or "unknown", "count": count, "candidates": described,
                           "context": "The second attachment is the object's crop from the scene reference." if crop else ""},
                attachments=[sheet, *([crop] if crop else [])], validate=validate,
            )
        return None if choice is None else (candidates[choice - 1], reason)

    def _choose(self, candidate: Candidate, item: InventoryItem, directory: Path, reason: str) -> ReferenceChoice:
        path = directory / "reference.png"
        shutil.copy2(candidate.path, path)
        provenance = {**candidate.to_dict(), "quality": assess(candidate, item),
                      "rights_allowed": rights_allowed(candidate, self._config.allowed_licenses)}
        choice = ReferenceChoice(path, candidate.source, candidate.title, reason, candidate.license, provenance)
        (directory / "selection.json").write_text(json.dumps(choice.to_dict(), indent=2), encoding="utf-8")
        (directory / "review.json").unlink(missing_ok=True)
        return choice


def search_queries(name: str, suffix: str) -> list[str]:
    """Specific to broad noun phrases, with stable deduplication."""
    words = name.split()
    queries = [f"{name} {suffix}".strip(), name, *(" ".join(words[-n:]) for n in (4, 3, 2) if len(words) > n)]
    return list(dict.fromkeys(q for q in queries if q))
