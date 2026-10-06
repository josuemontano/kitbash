"""Saving approved assets to the backlot (never before approval)."""

import json
from pathlib import Path

from kitbash.agents.modelling import BUILD_DIR, ModellingAgent
from kitbash.analytics.tracker import EventKind, Tracker
from kitbash.backlot.library import AssetBundle, Backlot, BacklotDraft, BacklotEntry
from kitbash.config import Config
from kitbash.critique.sessions import ResumableLoop
from kitbash.critique.store import CycleResult
from kitbash.domain.assets import AssetRecord, AssetState
from kitbash.domain.inventory import InventoryItem
from kitbash.domain.phases import PhaseName
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
        if not asset.has_build or asset.id != item.id:
            return None
        result = self._loop.best(self._agent.subject(item, asset))
        if result is None or result.cycle != asset.best_cycle or result.evidence is None:
            return None
        evidence = result.evidence
        if evidence.workspace != self._layout.root or evidence.subject_id != asset.id or evidence.phase is not PhaseName.MODELLING:
            return None
        return result

    def committable(self, result: CycleResult | None) -> bool:
        return result is not None and result.eligible and result.evaluation.artifacts.get("build_dir") == str(result.script_path.parent / BUILD_DIR)

    def commit(self, asset: AssetRecord, item: InventoryItem) -> BacklotEntry:
        if asset.state is not AssetState.AWAITING_REVIEW:
            raise BacklotError(f"{asset.id} is not awaiting approval")
        best = self.best(asset, item)
        if not self.committable(best):
            raise BacklotError(f"{asset.id} has no unchanged, owned, evaluated build (.blend, .usd and preview) to save")
        artifacts, facts = best.evaluation.artifacts, best.scorecard.facts
        evidence = best.evidence
        build = Path(artifacts["build_dir"])
        prefix = build.relative_to(evidence.root).as_posix() + "/"
        hashes = {key.removeprefix(prefix): value for key, value in evidence.hashes.items() if key.startswith(prefix)}
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
                    "evidence": evidence.to_dict(),
                    "reference": asset.extra.get("reference"),
                    "trellis": asset.extra.get("trellis"),
                    "retopology": asset.extra.get("retopology"),
                    "usd": best.evaluation.report.get("usd", {}),
                },
            ),
            AssetBundle(
                root=build,
                blend=Path(artifacts["blend"]),
                usd=Path(artifacts["usd"]),
                preview=Path(artifacts["preview"]),
                hashes=hashes,
                preview_hash=evidence.hashes[Path(artifacts["preview"]).relative_to(evidence.root).as_posix()],
            ),
        )
        self._tracker.event(EventKind.BACKLOT, "commit", asset=asset.id, backlot_id=entry.id, version=entry.version)
        return entry
