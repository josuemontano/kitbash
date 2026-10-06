"""Layout agent: places the approved assets, then sets camera, lighting and world for the chosen style."""

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from attrs import frozen

from kitbash.analytics import context
from kitbash.config import Config
from kitbash.critique.subject import CriticBrief, Evaluation
from kitbash.domain.inventory import Inventory, InventoryItem
from kitbash.domain.phases import PhaseName
from kitbash.domain.roles import Role
from kitbash.domain.run_input import RunInput
from kitbash.infra.imaging import side_by_side
from kitbash.infra.polyhaven import PolyHavenCatalog
from kitbash.llm.service import LLMService
from kitbash.paths import OutputLayout
from kitbash.services.api_reference import blender_api_reference
from kitbash.services.blender_toolkit import BlenderToolkit


@frozen
class PlacedAsset:
    """An approved asset the layout can place, keyed by the inventory item it was modelled for."""

    key: str
    slug: str  # object and collection names inside the asset .blend are derived from it
    name: str
    description: str
    blend: Path
    collection: str
    dimensions: tuple[float, float, float]
    backlot_id: str | None = None

    def spec(self, blend: Path | None = None) -> dict[str, Any]:
        return {
            "blend": str(blend or self.blend),
            "collection": self.collection,
            "name": self.slug,
            "dimensions": list(self.dimensions),
            "backlot_id": self.backlot_id,
        }


def asset_specs(assets: Sequence[PlacedAsset], blends: dict[str, Path] | None = None) -> dict[str, dict[str, Any]]:
    return {a.key: a.spec((blends or {}).get(a.key)) for a in assets}


class LayoutAgent:
    def __init__(
        self,
        llm: LLMService,
        toolkit: BlenderToolkit,
        catalog: PolyHavenCatalog,
        config: Config,
        layout: OutputLayout,
        run_input: RunInput,
    ) -> None:
        self._llm = llm
        self._toolkit = toolkit
        self._catalog = catalog
        self._config = config
        self._layout = layout
        self._input = run_input

    def write_script(self, inventory: Inventory, assets: Sequence[PlacedAsset], skipped: Sequence[InventoryItem]) -> str:
        style = self._config.style
        hdris = self._catalog.hdris(inventory.scene.environment)
        with context.bind(agent="layout_agent"):
            return self._llm.ask_python(
                task="layout.script",
                role=Role.CODE,
                phase=PhaseName.LAYOUT,
                template="layout_script",
                variables={
                    "scene": json.dumps(inventory.scene.to_dict(), indent=1),
                    "style": self._config.pipeline.style,
                    "style_guidance": style.guidance.strip(),
                    "engine": style.render_engine,
                    "assets": _table([
                        {"key": a.key, "name": a.name, "dimensions_m": a.dimensions, "description": a.description[:160]}
                        for a in assets
                    ]),
                    "placeholders": _table([
                        {"key": i.id, "name": i.name, "dimensions_m": i.dimensions.as_tuple()} for i in skipped
                    ]) or "(none)",
                    "inventory": _table(_inventory_for_layout(inventory)),
                    "hdris": _table(hdris) or "(none: use kb.color_world)",
                    "feedback": "(none)",
                    "api": blender_api_reference(),
                },
                attachments=self._input.references,
            )

    def subject(self, inventory: Inventory, assets: Sequence[PlacedAsset], skipped: Sequence[InventoryItem]) -> LayoutSubject:
        return LayoutSubject(inventory, tuple(assets), tuple(skipped), self._toolkit, self._config, self._layout, self._input)


class LayoutSubject:
    phase = PhaseName.LAYOUT
    subject_id = ""

    def __init__(
        self,
        inventory: Inventory,
        assets: tuple[PlacedAsset, ...],
        skipped: tuple[InventoryItem, ...],
        toolkit: BlenderToolkit,
        config: Config,
        layout: OutputLayout,
        run_input: RunInput,
    ) -> None:
        self._inventory = inventory
        self._assets = assets
        self._skipped = skipped
        self._toolkit = toolkit
        self._config = config
        self._layout = layout
        self._input = run_input

    def dependency_fingerprint(self) -> str:
        """Canonical inputs for layout generation, placement and style evaluation."""
        dependencies = {
            "scene": self._inventory.scene.to_dict(),
            "instances": sorted(_inventory_for_layout(self._inventory), key=lambda item: item["id"]),
            "assets": {
                asset.key: {**asset.spec(), "display_name": asset.name, "description": asset.description[:160]}
                for asset in self._assets
            },
            "placeholders": {
                item.id: {"name": item.name, "dimensions_m": item.dimensions.as_tuple()}
                for item in self._skipped
            },
            "style": {
                "name": self._config.pipeline.style,
                "guidance": self._config.style.guidance.strip(),
                "render_engine": self._config.style.render_engine,
            },
        }
        text = json.dumps(dependencies, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()

    def brief(self) -> CriticBrief:
        scene = self._inventory.scene
        return CriticBrief(
            phase=self.phase,
            subject="scene layout (placement, camera, lighting, world)",
            description=(
                f"{scene.description}\nEnvironment: {scene.environment}. Lighting: {scene.lighting}. "
                f"Placed assets: {', '.join(a.key for a in self._assets) or 'none'}; placeholders for skipped assets: "
                f"{', '.join(i.id for i in self._skipped) or 'none'}.\nStyle guidance: {self._config.style.guidance.strip()}"
            ),
            references=self._input.references,
            style=self._config.pipeline.style,
        )

    def api_reference(self) -> str:
        return blender_api_reference()

    def preview_resolution(self) -> tuple[int, int]:
        width = self._config.blender.preview_resolution[0]
        final_w, final_h = self._config.blender.final_resolution
        return width, max(1, round(width * final_h / final_w))

    def run_args(self, output_blend: Path, blends: dict[str, Path] | None = None, asset_mode: str = "append") -> dict[str, Any]:
        assets = asset_specs(self._assets, blends)
        counts = Counter(item.asset_key for item in self._inventory.items)
        for key, spec in assets.items():
            spec["instances"] = counts[key]
        airborne = Counter(item.asset_key for item in self._inventory.items if item.support == "airborne")
        return {"output_blend": str(output_blend), "assets": assets, "asset_mode": asset_mode, "airborne": dict(airborne)}

    def evaluate(self, script: Path, cycle_dir: Path, cycle: int) -> Evaluation:
        blend = cycle_dir / "layout.blend"
        args = self.run_args(blend)
        self._toolkit.run_script(script, args, cycle_dir / "layout_run.log")
        renders = self._layout.renders_dir(self.phase)
        render = self._toolkit.render_scene(
            blend, renders / f"cycle_{cycle:02d}.png", cycle_dir, resolution=self.preview_resolution(),
            samples=self._config.blender.preview_samples, engine=self._config.style.render_engine,
        )
        images = [render]
        if self._input.image is not None:
            images.append(side_by_side([self._input.image, render], ["reference", "layout"], renders / f"cycle_{cycle:02d}_compare.png"))
        report, facts = self._toolkit.inspect_scene(
            blend, args["assets"], [i.id for i in self._skipped], cycle_dir, "inspect", airborne=args["airborne"]
        )
        return Evaluation(ok=True, images=tuple(images), facts=facts, report=report, artifacts={"blend": str(blend), "render": str(render)})


def _table(rows: Sequence[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(row) for row in rows)


def _inventory_for_layout(inventory: Inventory) -> list[dict[str, Any]]:
    return [
        {
            "id": item.id,
            "asset_key": item.asset_key,
            "name": item.name,
            "dimensions_m": item.dimensions.as_tuple(),
            "location_m": item.position.location,
            "rotation_deg": item.position.rotation_deg,
            "support": item.support,
            "relationships": [
                {"type": r.kind, "target": r.target}
                for r in sorted(item.relationships, key=lambda r: (r.kind, r.target))
            ],
        }
        for item in inventory.items
    ]
