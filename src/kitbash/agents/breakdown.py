"""Breakdown agent: finds the objects in the reference (or designs them from the prompt) and checks the
inventory against the reference through a blockout render."""

import json
import pprint
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from string import Template
from typing import Any

from kitbash.analytics import context
from kitbash.backlot.library import Backlot, SearchHit
from kitbash.config import Config
from kitbash.critique.store import CycleResult
from kitbash.critique.subject import CriticBrief, Evaluation
from kitbash.domain.inventory import Inventory
from kitbash.domain.phases import PhaseName
from kitbash.domain.roles import Role
from kitbash.domain.run_input import InputMode, RunInput
from kitbash.infra.imaging import side_by_side
from kitbash.llm.prompts import PromptLibrary
from kitbash.llm.service import LLMService
from kitbash.paths import OutputLayout
from kitbash.services.blender_toolkit import BlenderToolkit

BLOCKOUT_TEMPLATE = "templates/breakdown_blockout.py.tmpl"


class BreakdownAgent:
    def __init__(
        self, llm: LLMService, prompts: PromptLibrary, toolkit: BlenderToolkit, config: Config, layout: OutputLayout, run_input: RunInput
    ) -> None:
        self._llm = llm
        self._prompts = prompts
        self._toolkit = toolkit
        self._config = config
        self._layout = layout
        self._input = run_input

    def analyze(self) -> Inventory:
        """Initial inventory from the image (image analysis role) or the prompt (prompt analysis role)."""
        image_mode = self._input.mode is InputMode.IMAGE
        max_items = self._config.breakdown.max_items
        with context.bind(agent="breakdown_agent"):
            return self._llm.ask_json(
                task=f"breakdown.analyze.{self._input.mode.value}",
                role=Role.IMAGE_ANALYSIS if image_mode else Role.PROMPT_ANALYSIS,
                phase=PhaseName.BREAKDOWN,
                template="breakdown_image" if image_mode else "breakdown_prompt",
                variables={"style": self._config.pipeline.style, "max_items": max_items, "prompt": self._input.prompt or ""},
                attachments=self._input.references,
                validate=lambda data: Inventory.from_dict(data, max_items=max_items),
            )

    def initial_script(self, inventory: Inventory) -> str:
        template = resources.files("kitbash.blender").joinpath(BLOCKOUT_TEMPLATE).read_text(encoding="utf-8")
        literal = pprint.pformat(inventory.to_dict(), width=110, sort_dicts=False)
        return Template(template).substitute(inventory=literal)

    def subject(self) -> BreakdownSubject:
        return BreakdownSubject(self._toolkit, self._prompts, self._config, self._layout, self._input)

    def inventory_of(self, result: CycleResult) -> Inventory:
        path = Path(result.evaluation.artifacts["inventory"])
        return Inventory.from_dict(json.loads(path.read_text(encoding="utf-8")), max_items=self._config.breakdown.max_items)

    def backlot_matches(self, inventory: Inventory, backlot: Backlot) -> dict[str, SearchHit]:
        """Best reusable backlot asset per modelled item, when it clears the similarity threshold."""
        settings = self._config.backlot
        style = self._config.pipeline.style if settings.match_style else None
        matches = {}
        for item in inventory.modelled_items():
            hits = backlot.search(item.embedding_text(), k=settings.top_k, style=style)
            if hits and hits[0].similarity >= settings.match_threshold:
                matches[item.id] = hits[0]
        return matches


class BreakdownSubject:
    phase = PhaseName.BREAKDOWN
    subject_id = ""

    def __init__(self, toolkit: BlenderToolkit, prompts: PromptLibrary, config: Config, layout: OutputLayout, run_input: RunInput) -> None:
        self._toolkit = toolkit
        self._prompts = prompts
        self._config = config
        self._layout = layout
        self._input = run_input

    def brief(self) -> CriticBrief:
        target = (
            "The inventory must list every salient object of the reference image with accurate real-world "
            "dimensions, positions, rotations and relationships; the blockout render (one labelled box per item, "
            "seen through the estimated camera) must line up with the reference."
            if self._input.mode is InputMode.IMAGE
            else f"The inventory must be a complete, specific and plausible set for this prompt: {self._input.prompt}"
        )
        return CriticBrief(
            phase=self.phase,
            subject="scene breakdown (inventory and blockout)",
            description=target,
            references=self._input.references,
            style=self._config.pipeline.style,
        )

    def api_reference(self) -> str:
        return (
            "Edit the INVENTORY dict literal at the top of the script; keep build() unchanged unless it fails.\n\n"
            + self._prompts.fragment("inventory_schema")
        )

    def evaluate(self, script: Path, cycle_dir: Path, cycle: int) -> Evaluation:
        blend = cycle_dir / "blockout.blend"
        result = self._toolkit.run_script(script, {"output_blend": str(blend)}, cycle_dir / "blockout.log")
        inventory = Inventory.from_dict(result["inventory"], max_items=self._config.breakdown.max_items)
        inventory_path = cycle_dir / "inventory.json"
        inventory_path.write_text(json.dumps(inventory.to_dict(), indent=2), encoding="utf-8")
        renders = self._layout.renders_dir(self.phase)
        render = self._toolkit.render_scene(
            blend, renders / f"cycle_{cycle:02d}_blockout.png", cycle_dir,
            resolution=self._config.blender.preview_resolution, samples=8, engine=self._config.blender.render_engine,
        )
        images = [render]
        if self._input.image is not None:
            images.insert(0, side_by_side([self._input.image, render], ["reference", "blockout"], renders / f"cycle_{cycle:02d}_compare.png"))
        return Evaluation(
            ok=True,
            images=tuple(images),
            facts=_inventory_facts(inventory, self._config.breakdown.confidence_threshold),
            report=inventory.to_dict(),
            artifacts={"inventory": str(inventory_path), "blend": str(blend), "render": str(render)},
        )


def _inventory_facts(inventory: Inventory, threshold: float) -> Mapping[str, Any]:
    items = inventory.items
    return {
        "items": len(items),
        "unrecognized_items": sum(1 for i in items if i.confidence < threshold),
        "mean_confidence": round(sum(i.confidence for i in items) / len(items), 3) if items else 0.0,
        "items_without_relationships": sum(1 for i in items if not i.relationships),
    }
