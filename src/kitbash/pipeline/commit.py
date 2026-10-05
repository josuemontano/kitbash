"""Saving approved assets to the backlot (never before approval)."""

import json
from pathlib import Path

from kitbash.agents.modelling import ModellingAgent
from kitbash.analytics.tracker import EventKind, Tracker
from kitbash.backlot.library import AssetBundle, Backlot, BacklotDraft, BacklotEntry
from kitbash.config import Config
from kitbash.critique.sessions import ResumableLoop
from kitbash.critique.store import CycleResult
from kitbash.domain.assets import AssetRecord
from kitbash.domain.inventory import InventoryItem
from kitbash.errors import BacklotError
from kitbash.paths import OutputLayout


class BacklotCommitter:
    def __init__(self, backlot: Backlot, loop: ResumableLoop, agent: ModellingAgent, config: Config, layout: OutputLayout, tracker: Tracker) -> None:
        self._backlot = backlot
        self._loop = loop
        self._agent = agent
        self._config = config
        self._layout = layout
        self._tracker = tracker

    def best(self, asset: AssetRecord, item: InventoryItem) -> CycleResult | None:
        return self._loop.best(self._agent.subject(item, asset)) if asset.has_build else None

    def committable(self, result: CycleResult | None) -> bool:
        artifacts = result.evaluation.artifacts if result else {}
        return all(artifacts.get(k) and Path(artifacts[k]).is_file() for k in ("blend", "usd", "preview"))

    def commit(self, asset: AssetRecord, item: InventoryItem) -> BacklotEntry:
        best = self.best(asset, item)
        if not self.committable(best):
            raise BacklotError(f"{asset.id} has no complete build (.blend, .usd and preview) to save")
        artifacts, facts = best.evaluation.artifacts, best.scorecard.facts
        entry = self._backlot.add(
            BacklotDraft(
                name=item.name,
                description=item.description,
                category=item.category,
                dimensions=tuple(facts.get("dimensions_m") or item.dimensions.as_tuple()),
                style=self._config.pipeline.style,
                usd_material_mode=str(facts.get("usd_material_mode", "preview_surface_baked")),
                usd_roundtrip_score=facts.get("usd_roundtrip_score"),
                source_reference=json.dumps(asset.extra.get("reference")) if asset.extra.get("reference") else asset.reference_path,
                tags=tuple(dict.fromkeys((item.category, *item.materials_hint))),
                metadata={
                    "slug": item.id,
                    "scene_output": str(self._layout.root),
                    "cycle": best.cycle,
                    "score": round(best.score, 4),
                    "scorecard": best.scorecard.to_dict(),
                    "reference": asset.extra.get("reference"),
                    "trellis": asset.extra.get("trellis"),
                    "usd": best.evaluation.report.get("usd", {}),
                },
            ),
            AssetBundle(
                root=Path(artifacts["build_dir"]),
                blend=Path(artifacts["blend"]),
                usd=Path(artifacts["usd"]),
                preview=Path(artifacts["preview"]),
            ),
        )
        self._tracker.event(EventKind.BACKLOT, "commit", asset=asset.id, backlot_id=entry.id, version=entry.version)
        return entry
