"""Find the reference image Trellis will reconstruct an asset from."""

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
from attrs import frozen

from kitbash.analytics import context
from kitbash.analytics.tracker import EventKind, Tracker
from kitbash.config import ReferenceConfig
from kitbash.domain.inventory import InventoryItem
from kitbash.domain.phases import PhaseName
from kitbash.domain.roles import Role
from kitbash.errors import KitbashError, LLMAccessError, LLMError
from kitbash.infra.image_search import Candidate, CandidateProvider, ImageDownloader
from kitbash.infra.imaging import contact_sheet, normalize_to_png
from kitbash.llm.service import LLMService


@frozen
class ReferenceChoice:
    path: Path
    source: str
    title: str
    reason: str
    license: str = ""


class OmpWebImageProvider:
    """Asks the omp agent to find product photos with its web_search tool (slower, much better hits)."""

    name = "omp_web"
    once_per_item = True

    def __init__(self, llm: LLMService) -> None:
        self._llm = llm

    def search(self, query: str, item: InventoryItem, limit: int, workdir: Path) -> list[Candidate]:
        def validate(data: Any) -> list[Candidate]:
            images = data.get("images") if isinstance(data, Mapping) else None
            if not isinstance(images, list):
                raise LLMError("Expected {'images': [{'url': ..., 'title': ...}]}")
            return [
                Candidate(self.name, str(i.get("title", "")), url=str(i["url"]))
                for i in images
                if isinstance(i, Mapping) and str(i.get("url", "")).startswith("http")
            ][:limit]

        with context.bind(agent="reference_search"):
            return self._llm.ask_json(
                task="modelling.reference.search",
                role=Role.REFERENCE_SELECTION,
                phase=PhaseName.MODELLING,
                template="reference_search",
                variables={"name": item.search_name or item.name, "description": item.description, "count": limit, "query": query},
                tools=("web_search",),
                validate=validate,
            )


class ReferenceFinder:
    def __init__(
        self,
        providers: Sequence[CandidateProvider],
        downloader: ImageDownloader,
        llm: LLMService,
        config: ReferenceConfig,
        tracker: Tracker,
    ) -> None:
        self._providers = tuple(providers)
        self._downloader = downloader
        self._llm = llm
        self._config = config
        self._tracker = tracker

    def find(self, item: InventoryItem, directory: Path) -> ReferenceChoice | None:
        """Use the user's reference if given, else search, download and let the selection model pick."""
        directory.mkdir(parents=True, exist_ok=True)
        if item.user_reference:
            path = normalize_to_png(Path(item.user_reference).expanduser(), directory / "reference.png")
            return self._save(directory, ReferenceChoice(path, "user", Path(item.user_reference).name, "provided by the user"))
        candidates = self._collect(item, directory / "candidates")
        if not candidates:
            return None
        picked = self._select(item, candidates, directory)
        if picked is None:
            return None
        chosen, reason = picked
        path = directory / "reference.png"
        shutil.copy2(chosen.path, path)
        return self._save(directory, ReferenceChoice(path, chosen.source, chosen.title, reason, chosen.license))

    def _collect(self, item: InventoryItem, directory: Path) -> list[Candidate]:
        """Candidates from every query level and provider, interleaved so broad queries (which usually find
        the right kind of object) are represented even when the specific ones return noise."""
        directory.mkdir(parents=True, exist_ok=True)
        buckets: list[list[Candidate]] = []
        for level, query in enumerate(search_queries(item.search_name or item.name, self._config.query_suffix)):
            for provider in self._providers:
                if provider.once_per_item and level > 0:
                    continue
                try:
                    buckets.append(provider.search(query, item, self._config.per_provider, directory))
                except LLMAccessError:
                    raise
                except (httpx.HTTPError, KitbashError) as exc:
                    self._tracker.event(EventKind.WARNING, "reference_search_failed", provider=provider.name, error=str(exc)[:300])
        found = _interleave(buckets, limit=self._config.max_candidates * 2)
        local: list[Candidate] = []
        for candidate in found:
            if len(local) >= self._config.max_candidates:
                break
            fetched = self._downloader.fetch(candidate, directory / f"{len(local):02d}.png")
            if fetched is not None:
                local.append(fetched)
        (directory / "candidates.json").write_text(
            json.dumps([{"index": i, "source": c.source, "title": c.title, "url": c.url, "license": c.license} for i, c in enumerate(local)], indent=2),
            encoding="utf-8",
        )
        return local

    def _select(self, item: InventoryItem, candidates: Sequence[Candidate], directory: Path) -> tuple[Candidate, str] | None:
        sheet = contact_sheet([c.path for c in candidates], [str(i) for i in range(len(candidates))], directory / "contact_sheet.png")
        crop = next((c.path for c in candidates if c.source == "input_crop"), None)
        described = "; ".join(f"{i}: {c.source} ({c.title[:60]})" for i, c in enumerate(candidates))
        count = len(candidates)

        def validate(data: Any) -> tuple[int | None, str]:
            if not isinstance(data, Mapping) or "choice" not in data:
                raise LLMError("Expected {'choice': number or null, 'reason': ...}")
            choice = data["choice"]
            if choice is not None and (not isinstance(choice, int) or not 0 <= choice < count):
                raise LLMError(f"choice must be null or an integer between 0 and {count - 1}")
            return choice, str(data.get("reason", ""))

        with context.bind(agent="reference_selector"):
            choice, reason = self._llm.ask_json(
                task="modelling.reference.select",
                role=Role.REFERENCE_SELECTION,
                phase=PhaseName.MODELLING,
                template="reference_select",
                variables={
                    "name": item.name,
                    "description": item.description,
                    "category": item.category,
                    "materials": ", ".join(item.materials_hint) or "unknown",
                    "count": count,
                    "candidates": described,
                    "context": "The second attachment is the object's crop from the scene reference." if crop else "",
                },
                attachments=[sheet, *([crop] if crop else [])],
                validate=validate,
            )
        return None if choice is None else (candidates[choice], reason)

    @staticmethod
    def _save(directory: Path, choice: ReferenceChoice) -> ReferenceChoice:
        (directory / "selection.json").write_text(
            json.dumps({"path": str(choice.path), "source": choice.source, "title": choice.title, "reason": choice.reason, "license": choice.license}, indent=2),
            encoding="utf-8",
        )
        return choice


def search_queries(name: str, suffix: str) -> list[str]:
    """From most to least specific: the full name with the background hint, the name alone, then its
    trailing words (the head noun phrase: 'small round three-legged oak side table' -> 'side table')."""
    words = name.split()
    queries = [f"{name} {suffix}".strip(), name, *(" ".join(words[-n:]) for n in (4, 3, 2) if len(words) > n)]
    return list(dict.fromkeys(q for q in queries if q))


def _interleave(buckets: Sequence[Sequence[Candidate]], limit: int) -> list[Candidate]:
    """Round-robin over result lists, dropping duplicates (same URL or file)."""
    found: list[Candidate] = []
    seen: set[str] = set()
    for rank in range(max((len(b) for b in buckets), default=0)):
        for bucket in buckets:
            if rank < len(bucket):
                candidate = bucket[rank]
                key = candidate.url or str(candidate.path)
                if key not in seen:
                    seen.add(key)
                    found.append(candidate)
                if len(found) >= limit:
                    return found
    return found
