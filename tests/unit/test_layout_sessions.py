"""Layout dependency changes must not reuse completed or interrupted critic sessions."""

from types import SimpleNamespace

import pytest
from attrs import evolve
from rich.console import Console

from kitbash.agents.layout import LayoutAgent, LayoutSubject, PlacedAsset
from kitbash.analytics.tracker import Tracker
from kitbash.config import load_config
from kitbash.critique.sessions import LoopSessions, ResumableLoop
from kitbash.domain.inventory import Dimensions, Inventory, Relationship
from kitbash.domain.phases import PhaseName
from kitbash.domain.rubric import Rubric
from kitbash.domain.run_input import RunInput
from kitbash.interaction.autopilot import AutoPilot
from kitbash.paths import OutputLayout
from kitbash.phases.base import GateRounds
from kitbash.phases.layout import LayoutPhase
from kitbash.store.state import StateDB
from kitbash.ui.dashboard import Dashboard
from tests.unit.test_critic_loop import FakeSubject, ScriptedPatchWriter, make_loop, script

RUBRIC = Rubric.parse("| criterion | weight | pass condition | applies to |\n|-|-|-|-|\n| Quality | 1 | good enough | layout |")


@pytest.fixture
def scene(tmp_path, sample_inventory_dict, monkeypatch):
    state = StateDB(tmp_path / "state.db")
    layout = OutputLayout.at(tmp_path / "out")
    inventory = Inventory.from_dict(sample_inventory_dict)
    crate, mug = inventory.items
    scene = SimpleNamespace(
        state=state,
        layout=layout,
        config=load_config(),
        inventory=inventory,
        placed=[PlacedAsset(crate.id, crate.id, crate.name, crate.description, tmp_path / "crate.blend",
                            crate.id, crate.dimensions.as_tuple(), "backlot-crate")],
        skipped=[mug],
        writes=0,
        evaluations=[],
        interrupt=False,
        score=0.9,
    )
    writer = ScriptedPatchWriter([0.95])
    scene.writer = writer
    scene.loop = ResumableLoop(make_loop((state, layout, scene.config), writer, rubric=RUBRIC), LoopSessions(state.meta))

    def write_script(self, inventory, assets, skipped):
        scene.writes += 1
        return script(scene.score)

    def evaluate(self, path, cycle_dir, cycle):
        if scene.interrupt:
            raise KeyboardInterrupt
        scene.evaluations.append(cycle)
        return FakeSubject().evaluate(path, cycle_dir, cycle)

    monkeypatch.setattr(LayoutAgent, "write_script", write_script)
    monkeypatch.setattr(LayoutSubject, "evaluate", evaluate)

    def run():
        agent = LayoutAgent(None, None, None, scene.config, layout, RunInput.create(None, "a crate and mug"))
        cast = SimpleNamespace(load=lambda: (scene.inventory, scene.placed, scene.skipped))
        LayoutPhase(agent, scene.loop, cast, state.meta, AutoPilot(), Dashboard(Console(quiet=True), 4),
                    Tracker(state.spans), layout).run()

    scene.run = run
    yield scene
    state.close()


@pytest.mark.parametrize("change", [
    "asset_dimensions", "asset_identity", "asset_blend", "asset_collection", "asset_slug", "asset_description",
    "placement", "rotation", "inventory_dimensions", "relationships", "duplicate", "duplicate_binding",
    "placeholder_dimensions", "placeholder_added", "placeholder_removed", "scene", "environment", "lighting",
    "style_notes", "camera_location", "camera_rotation", "camera_lens", "style", "style_guidance", "style_engine",
])
def test_upstream_change_rebuilds_completed_layout(scene, change):
    scene.run()
    crate, mug = scene.inventory.items
    if change.startswith("asset_"):
        field, value = {
            "asset_dimensions": ("dimensions", (1.2, 0.8, 0.7)),
            "asset_identity": ("backlot_id", "replacement-crate"),
            "asset_blend": ("blend", scene.placed[0].blend.with_name("replacement.blend")),
            "asset_collection": ("collection", "new_collection"),
            "asset_slug": ("slug", "new_crate"),
            "asset_description": ("description", "a tall crate"),
        }[change]
        scene.placed = [evolve(scene.placed[0], **{field: value})]
    elif change in {"placement", "rotation"}:
        position = evolve(crate.position, **{"location" if change == "placement" else "rotation_deg": (2.0, 3.0, 4.0)})
        scene.inventory = scene.inventory.replace_item(evolve(crate, position=position))
    elif change == "inventory_dimensions":
        scene.inventory = scene.inventory.replace_item(evolve(crate, dimensions=Dimensions(1.0, 2.0, 3.0)))
    elif change == "relationships":
        scene.inventory = scene.inventory.replace_item(evolve(mug, relationships=(Relationship("next_to", crate.id),)))
    elif change == "duplicate":
        scene.inventory = evolve(scene.inventory, items=(*scene.inventory.items, evolve(crate, id="crate_copy", same_as=crate.id)))
    elif change == "duplicate_binding":
        scene.inventory = scene.inventory.replace_item(evolve(mug, same_as=crate.id))
        scene.skipped = []
    elif change == "placeholder_dimensions":
        mug = evolve(mug, dimensions=Dimensions(0.3, 0.4, 0.5))
        scene.inventory = scene.inventory.replace_item(mug)
        scene.skipped = [mug]
    elif change == "placeholder_added":
        scene.placed = []
        scene.skipped = list(scene.inventory.items)
    elif change == "placeholder_removed":
        scene.inventory = scene.inventory.without(mug.id)
        scene.skipped = []
    elif change in {"scene", "environment", "lighting", "style_notes"}:
        field = "description" if change == "scene" else change
        scene.inventory = evolve(scene.inventory, scene=evolve(scene.inventory.scene, **{field: "changed scene"}))
    elif change.startswith("camera_"):
        field, value = {
            "camera_location": ("location", (1.0, -8.0, 2.0)),
            "camera_rotation": ("rotation_deg", (70.0, 0.0, 20.0)),
            "camera_lens": ("focal_length_mm", 85.0),
        }[change]
        camera = evolve(scene.inventory.scene.camera, **{field: value})
        scene.inventory = evolve(scene.inventory, scene=evolve(scene.inventory.scene, camera=camera))
    elif change == "style":
        scene.config = evolve(scene.config, pipeline=evolve(scene.config.pipeline, style="2d"))
    else:
        field, value = ("guidance", "Use flat lighting") if change == "style_guidance" else ("render_engine", "BLENDER_EEVEE")
        style = evolve(scene.config.style, **{field: value})
        scene.config = evolve(scene.config, styles={**scene.config.styles, scene.config.pipeline.style: style})

    scene.score = 0.85  # An older, higher score must not win over the new dependencies.
    scene.run()
    assert scene.layout.layout_script.read_text() == script(0.85)
    assert scene.writes == 2 and scene.evaluations == [1, 2]
    assert scene.writer.requests == []  # Start from the new inputs, not a patch of stale output.
    scene.run()
    assert scene.writes == 2 and scene.evaluations == [1, 2]
    assert scene.layout.layout_script.read_text() == script(0.85)


def test_equivalent_normalized_inventory_resumes_completed_layout(scene):
    scene.run()
    data = scene.inventory.to_dict()
    data["items"].reverse()
    for item in data["items"]:
        item["dimensions"] = list(item.pop("dimensions_m").values())
        item["position"]["location"] = item["position"].pop("location_m")
    scene.inventory = Inventory.from_dict(data)
    scene.score = 0.95
    scene.run()
    assert scene.writes == 1 and scene.evaluations == [1]
    assert scene.layout.layout_script.read_text() == script(0.9)


def test_dependency_change_discards_old_gate_feedback(scene):
    scene.run()
    rounds = GateRounds(scene.state.meta, PhaseName.LAYOUT)
    rounds.add("move the crate left")
    scene.run()
    assert scene.writes == 1 and scene.evaluations == [1, 2]
    assert scene.writer.requests[0].edits[0].instruction == "move the crate left"
    scene.inventory = evolve(scene.inventory, scene=evolve(scene.inventory.scene, lighting="night"))
    scene.score = 0.85
    scene.run()
    assert rounds.rounds == []
    assert scene.writes == 2 and scene.evaluations == [1, 2, 3]
    assert scene.layout.layout_script.read_text() == script(0.85)


def test_changed_dependencies_abandon_interrupted_layout(scene):
    scene.interrupt = True
    with pytest.raises(KeyboardInterrupt):
        scene.run()
    scene.inventory = evolve(scene.inventory, scene=evolve(scene.inventory.scene, lighting="night"))
    scene.score = 0.85
    scene.interrupt = False
    scene.run()
    assert scene.writes == 2 and scene.evaluations == [2]
    assert scene.layout.layout_script.read_text() == script(0.85)
    assert [row.status for row in scene.state.cycles.cycles(PhaseName.LAYOUT, "")] == ["abandoned", "kept"]


def test_unchanged_dependencies_resume_interrupted_layout(scene):
    scene.interrupt = True
    with pytest.raises(KeyboardInterrupt):
        scene.run()
    scene.interrupt = False
    scene.score = 0.95
    scene.run()
    assert scene.writes == 1 and scene.evaluations == [1]
    assert scene.layout.layout_script.read_text() == script(0.9)


def test_interruption_after_dependency_checkpoint_still_starts_fresh(scene, monkeypatch):
    scene.run()
    scene.inventory = evolve(scene.inventory, scene=evolve(scene.inventory.scene, lighting="night"))
    run = scene.loop.run

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(scene.loop, "run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        scene.run()
    monkeypatch.setattr(scene.loop, "run", run)
    scene.score = 0.85
    scene.run()
    assert scene.writes == 2 and scene.evaluations == [1, 2]
    assert scene.layout.layout_script.read_text() == script(0.85)
    assert scene.writer.requests == []
