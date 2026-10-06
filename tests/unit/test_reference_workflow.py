"""Reference review must pause generation and preserve rights through approval."""

import json
import sys
import threading
from contextlib import nullcontext
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from attrs import evolve
from rich.console import Console

from kitbash.agents.modelling import ModellingAgent
from kitbash.analytics.tracker import Tracker
from kitbash.backlot.library import Backlot
from kitbash.config import load_config
from kitbash.critique.history import DiffStatus
from kitbash.critique.loop import LoopOutcome, LoopReason
from kitbash.critique.store import CycleResult, CycleStore
from kitbash.critique.subject import Evaluation
from kitbash.domain.assets import AssetState
from kitbash.domain.critique import CardEntry, ScoreCard
from kitbash.domain.inventory import Inventory
from kitbash.domain.phases import PhaseName
from kitbash.interaction.autopilot import AutoPilot
from kitbash.interaction.protocols import ReviewAction, ReviewDecision
from kitbash.interaction.terminal import TerminalUser
from kitbash.paths import OutputLayout
from kitbash.phases.modelling import ModellingPhase
from kitbash.pipeline.asset_pipeline import AssetPipeline
from kitbash.pipeline.commit import BacklotCommitter
from kitbash.pipeline.scheduler import Scheduler
from kitbash.services.references import ReferenceChoice
from kitbash.store.state import StateDB

S = AssetState


@pytest.fixture
def workflow(tmp_path, sample_inventory_dict):
    layout = OutputLayout.at(tmp_path / "out")
    layout.create()
    state = StateDB(layout.state_db)
    inventory = Inventory.from_dict(sample_inventory_dict)
    inventory = evolve(inventory, items=inventory.items[:1])
    state.inventory.save(inventory)
    agent = Mock(spec=ModellingAgent)
    agent.find_reference.return_value = None
    manifest = {
        "contact_sheet": str(tmp_path / "contact.jpg"),
        "candidates": [
            {"title": "Pine crate", "license_id": "cc-by-4.0", "creator": "Alice", "rights_allowed": True, "quality": {"score": 0.81}},
            {"title": "Wooden box", "license_id": "cc0-1.0", "creator": "Bob", "rights_allowed": True, "quality": {"score": 0.80}},
        ],
    }
    agent.reference_review.return_value = manifest
    config = load_config(None)
    tracker = Tracker(state.spans)
    loop = Mock()
    dashboard = SimpleNamespace(showing=lambda view: nullcontext(), paused=lambda: nullcontext())
    phase = ModellingPhase(agent, loop, Mock(), state, AutoPilot(), dashboard, tracker, config)
    board = phase._board(inventory)
    pipeline = AssetPipeline(board, inventory, agent, loop, tracker)
    yield SimpleNamespace(
        layout=layout, state=state, inventory=inventory, item=inventory.items[0], agent=agent,
        manifest=manifest, config=config, tracker=tracker, loop=loop, dashboard=dashboard,
        phase=phase, board=board, pipeline=pipeline, scheduler=Scheduler(config.pipeline.review_buffer, tracker),
    )
    state.close()


def test_ambiguity_stops_before_mesh_and_noninteractive_skips(workflow):
    env = workflow
    asset = env.pipeline.advance(env.item.id)
    assert asset.state is S.INPUT_NEEDED
    assert asset.extra["reference_review"] == env.manifest
    assert asset.reference_path is None
    env.agent.generate_mesh.assert_not_called()
    env.loop.run.assert_not_called()
    env.phase.run()
    assert env.state.assets.get(env.item.id).state is S.SKIPPED
    env.agent.generate_mesh.assert_not_called()


def test_terminal_selection_survives_retry_and_commit_with_remote_rights(workflow, tmp_path, monkeypatch, embedder):
    env = workflow
    asset = env.pipeline.advance(env.item.id)
    monkeypatch.setattr(sys, "stdin", StringIO("2\n"))
    shown = []
    user = TerminalUser(Console(file=StringIO(), color_system=None), env.dashboard, SimpleNamespace(show=shown.extend))
    decision = user.provide_input(asset, env.item, asset.input_request)
    assert decision.action is ReviewAction.PROVIDE_INPUT
    assert decision.reference_index == 1 and decision.reference_path is None
    assert shown == [tmp_path / "contact.jpg"]
    env.phase._apply(decision, asset, env.item, env.board, env.scheduler)
    assert env.board.get(asset.id).extra["reference_index"] == 1

    reference = tmp_path / "selected.png"
    reference.write_bytes(b"reference")
    provenance = {
        "source": "wikimedia", "title": "Wooden box", "creator": "Bob", "license_id": "cc-by-4.0",
        "url": "https://example.org/box.png", "page_url": "https://example.org/box",
        "license_url": "https://creativecommons.org/licenses/by/4.0/", "rights_allowed": True,
        "quality": {"score": 0.80},
    }
    choice = ReferenceChoice(reference, "wikimedia", "Wooden box", "selected by user", "CC BY 4.0", provenance=provenance)
    env.agent.select_reference.return_value = choice
    mesh = tmp_path / "mesh.obj"
    mesh.write_text("mesh")
    env.agent.generate_mesh.return_value = SimpleNamespace(mesh_path=mesh, duration_s=1.0, retries=0)
    cycle_dir = env.layout.cycle_dir(PhaseName.MODELLING, 1, asset.id)
    build = cycle_dir / "build"
    build.mkdir(parents=True)
    artifacts = {"build_dir": str(build)}
    for key, name in (("blend", "asset.blend"), ("usd", "asset.usd"), ("preview", "preview.png")):
        path = (cycle_dir if key == "preview" else build) / name
        path.write_bytes(b"artifact")
        artifacts[key] = str(path)
    script = cycle_dir / "script.py"
    script.write_text("# built")
    card = ScoreCard((CardEntry("geometry", "Geometry", 1.0, 1.0, True),), 1.0, True, 0.8)
    evaluation = Evaluation(ok=True, artifacts=artifacts)
    subject = SimpleNamespace(phase=PhaseName.MODELLING, subject_id=asset.id)
    evidence = CycleStore(env.state.cycles, env.layout).seal(subject, 1, script, evaluation)
    best = CycleResult(1, script, card, (), evaluation, DiffStatus.KEPT, PhaseName.MODELLING, evidence)

    def run_loop(subject, **kwargs):
        kwargs["observer"].critiquing(1)
        kwargs["observer"].evaluated(best)
        return LoopOutcome(LoopReason.PASSED, best, 1, "passed")

    env.loop.run.side_effect = run_loop
    env.loop.best.return_value = best
    reviewed = env.pipeline.advance(asset.id)
    assert reviewed.state is S.AWAITING_REVIEW
    assert reviewed.extra["reference_index"] is None and reviewed.extra["reference_review"] is None
    assert reviewed.extra.get("user_reference") is None
    assert reviewed.extra["reference"]["provenance"] == provenance
    assert reviewed.extra["cycle"] == 1

    library = Backlot(tmp_path / "backlot", embedder)
    try:
        committer = BacklotCommitter(library, env.loop, env.agent, env.config, env.layout, env.tracker)
        entry = committer.commit(reviewed, env.item)
        saved = library.get(entry.id)
        assert json.loads(saved.source_reference)["provenance"] == provenance
        assert saved.metadata["reference"]["provenance"] == provenance
    finally:
        library.close()


@pytest.mark.parametrize("decision", [
    ReviewDecision(ReviewAction.PROVIDE_INPUT, search_name="Oak storage box"),
    ReviewDecision(ReviewAction.PROVIDE_INPUT, reference_path="/tmp/user-reference.png"),
])
def test_new_input_clears_stale_candidate_selection(workflow, decision):
    env = workflow
    asset = env.pipeline.advance(env.item.id)
    asset = env.board.update(asset.id, extra={"reference_index": 1})
    env.phase._apply(decision, asset, env.item, env.board, env.scheduler)
    queued = env.board.get(asset.id)
    assert queued.extra["reference_index"] is None
    assert queued.extra["reference_review"] is None
    env.pipeline.advance(asset.id)
    env.agent.select_reference.assert_not_called()
    env.agent.generate_mesh.assert_not_called()


def test_rejected_explicit_selection_returns_to_input_without_mesh(workflow):
    env = workflow
    asset = env.pipeline.advance(env.item.id)
    env.phase._apply(ReviewDecision(ReviewAction.PROVIDE_INPUT, reference_index=0), asset, env.item, env.board, env.scheduler)
    env.agent.select_reference.return_value = None
    unresolved = env.pipeline.advance(asset.id)
    assert unresolved.state is S.INPUT_NEEDED
    assert unresolved.extra["reference_index"] is None
    env.agent.generate_mesh.assert_not_called()
    assert env.agent.find_reference.call_count == 1


def test_approved_reuse_never_acquires_a_reference_or_generates(workflow):
    env = workflow
    item = evolve(env.item, reuse_backlot_id="approved-crate")
    env.state.inventory.save(evolve(env.inventory, items=(item,)))
    env.phase.run()
    reused = env.state.assets.get(item.id)
    assert reused.state is S.APPROVED and reused.backlot_id == "approved-crate" and reused.reused
    env.agent.find_reference.assert_not_called()
    env.agent.select_reference.assert_not_called()
    env.agent.generate_mesh.assert_not_called()
    env.loop.run.assert_not_called()


@pytest.mark.parametrize("feedback", ["", "Use a taller crate"])
def test_rejected_reuse_survives_sqlite_reopen(workflow, sample_inventory_dict, feedback):
    env = workflow
    inventory = Inventory.from_dict(sample_inventory_dict)
    inventory = evolve(inventory, items=tuple(evolve(item, reuse_backlot_id=f"backlot-{item.id}") for item in inventory.items))
    env.state.inventory.save(inventory)
    board = env.phase._board(inventory)
    board.update(env.item.id, feedback=("Keep the pine material",))
    env.phase._reopen(board, env.item.id, feedback)
    expected_notes = ("Keep the pine material", feedback) if feedback else ("Keep the pine material",)
    queued = board.get(env.item.id)
    assert queued.state is S.QUEUED and not queued.reused and queued.backlot_id is None
    assert queued.feedback == expected_notes
    env.state.close()

    state = StateDB(env.layout.state_db)
    try:
        restored = state.inventory.load()
        assert restored == inventory.replace_item(evolve(inventory.item(env.item.id), reuse_backlot_id=None))
        tracker = Tracker(state.spans)
        phase = ModellingPhase(env.agent, env.loop, Mock(), state, AutoPilot(), env.dashboard, tracker, env.config)
        resumed = phase._board(restored)
        assert resumed.get(env.item.id) == queued
        other = resumed.get(inventory.items[1].id)
        assert other.state is S.APPROVED and other.reused and other.backlot_id == inventory.items[1].reuse_backlot_id

        pipeline = AssetPipeline(resumed, restored, env.agent, env.loop, tracker)
        pending = pipeline.advance(env.item.id)
        assert pending.state is S.INPUT_NEEDED and pending.feedback == expected_notes
        assert not pending.reused and pending.backlot_id is None

        # A later, explicit breakdown decision must still be able to select a new reuse.
        replacement = restored.replace_item(evolve(restored.item(env.item.id), reuse_backlot_id="replacement-crate"))
        state.inventory.save(replacement)
        accepted = phase._board(state.inventory.load()).get(env.item.id)
        assert accepted.state is S.APPROVED and accepted.reused and accepted.backlot_id == "replacement-crate"
    finally:
        state.close()


def test_interrupted_reuse_rejection_rolls_back_inventory_and_checkpoint(workflow, monkeypatch):
    env = workflow
    inventory = evolve(env.inventory, items=(evolve(env.item, reuse_backlot_id="approved-crate"),))
    env.state.inventory.save(inventory)
    board = env.phase._board(inventory)
    original = board.get(env.item.id)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(env.state.assets, "add_transition", interrupt)
    with pytest.raises(KeyboardInterrupt):
        env.phase._reopen(board, env.item.id, "Use a taller crate")
    env.state.close()

    state = StateDB(env.layout.state_db)
    try:
        assert state.inventory.load() == inventory
        assert state.assets.get(env.item.id) == original
    finally:
        state.close()


def test_input_prompts_run_on_main_thread(workflow):
    env = workflow
    threads = []

    class User(AutoPilot):
        def provide_input(self, asset, item, request):
            threads.append(threading.current_thread())
            return ReviewDecision(ReviewAction.SKIP)

    env.phase._user = User()
    env.phase.run()
    assert threads == [threading.main_thread()]
    assert env.state.assets.get(env.item.id).state is S.SKIPPED
