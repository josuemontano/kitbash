"""Procedural assets share build/review/resume rules without reconstructing a mesh."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from kitbash.agents.modelling import AssetSubject, ModellingAgent
from kitbash.analytics.tracker import Tracker
from kitbash.config import load_config
from kitbash.domain.assets import AssetRecord, AssetState, ReworkEntry
from kitbash.domain.inventory import Inventory
from kitbash.errors import KitbashError
from kitbash.paths import OutputLayout
from kitbash.phases.modelling import ModellingPhase
from kitbash.pipeline.asset_pipeline import AssetPipeline
from kitbash.pipeline.board import AssetBoard
from kitbash.store.state import StateDB

S = AssetState


@pytest.fixture
def procedural(tmp_path, sample_inventory_dict):
    config = load_config(None, {"modelling.method": "procedural"})
    layout = OutputLayout.at(tmp_path / "out")
    layout.create()
    state = StateDB(layout.state_db)
    inventory = Inventory.from_dict(sample_inventory_dict)
    tracker = Tracker(state.spans)
    agent = Mock(spec=ModellingAgent)
    loop = Mock()
    phase = ModellingPhase(agent, loop, Mock(), state, Mock(), Mock(), tracker, config)
    board = phase._board(inventory)
    pipeline = AssetPipeline(board, inventory, agent, loop, tracker)
    yield SimpleNamespace(
        config=config, layout=layout, state=state, inventory=inventory, tracker=tracker,
        agent=agent, loop=loop, phase=phase, board=board, pipeline=pipeline, item=inventory.items[0],
    )
    state.close()


@pytest.mark.parametrize("checkpoint", [S.QUEUED, S.BUILDING, S.CRITIQUING])
def test_procedural_resume_reaches_build_without_reconstruction(procedural, checkpoint):
    env = procedural
    asset_id = env.item.id
    if checkpoint is not S.QUEUED:
        env.board.transition(asset_id, S.BUILDING)
    if checkpoint is S.CRITIQUING:
        env.board.transition(asset_id, S.CRITIQUING)
    # Reload through the persistent record, not the current config or an in-memory default.
    board = AssetBoard(env.state.assets)
    pipeline = AssetPipeline(board, env.inventory, env.agent, env.loop, env.tracker)
    env.loop.run.side_effect = RuntimeError("interrupted build")
    with pytest.raises(RuntimeError, match="interrupted build"):
        pipeline.advance(asset_id)
    env.agent.find_reference.assert_not_called()
    env.agent.generate_mesh.assert_not_called()
    env.agent.retopologize.assert_not_called()
    env.loop.run.assert_called_once()
    _, record = env.agent.subject.call_args.args
    assert record.modelling_method == "procedural"
    assert record.mesh_path is None and record.reference_path is None
    assert record.state in (S.BUILDING, S.CRITIQUING)
    failed = pipeline.fail(asset_id, RuntimeError("interrupted build"))
    assert failed.state is S.AWAITING_REVIEW
    assert failed.error == "RuntimeError: interrupted build"
    assert not failed.has_build


@pytest.mark.parametrize("entry", list(ReworkEntry))
def test_procedural_rework_never_enters_reconstruction(procedural, entry):
    env = procedural
    original = AssetRecord(
        id=env.item.id, name=env.item.name, state=S.NEEDS_REWORK, modelling_method="procedural",
        rework_entry=entry, feedback=("Round the edges",), best_cycle=2, backlot_id="approved-v1",
    )
    env.board.reset(original)
    reworked = env.pipeline._rework(original, env.item)
    assert reworked.state is S.BUILDING and reworked.modelling_method == "procedural"
    assert reworked.best_cycle == 2 and reworked.backlot_id == "approved-v1"
    if entry is ReworkEntry.BUILD:
        assert reworked.extra["pending_feedback"] == "Round the edges"
        assert reworked.attempt == original.attempt
    else:
        assert reworked.extra["fresh_script"]
        assert reworked.attempt == original.attempt + 1
    env.agent.generate_mesh.assert_not_called()
    env.agent.find_reference.assert_not_called()


def test_procedural_approved_record_is_not_reset_on_resume(procedural):
    env = procedural
    approved = AssetRecord(
        id=env.item.id, name=env.item.name, state=S.APPROVED, modelling_method="procedural",
        best_cycle=3, backlot_id="owned-approved-build", score=0.95,
    )
    env.board.reset(approved)
    assert env.phase._board(env.inventory).get(approved.id) == approved
    assert env.pipeline.advance(approved.id) == approved
    env.loop.run.assert_not_called()
    env.agent.subject.assert_not_called()


def test_procedural_subject_evaluates_without_mesh_and_preserves_export_errors(procedural, tmp_path):
    env = procedural
    toolkit = Mock()
    preview = tmp_path / "preview.png"
    toolkit.render_views.return_value = (preview,)
    toolkit.inspect_asset.return_value = ({"faces": 100}, {"materials": 1, "missing_textures": 0})
    fidelity = Mock()
    fidelity.check.side_effect = KitbashError("export diagnostic")
    subject = AssetSubject(env.item, env.board.get(env.item.id), toolkit, fidelity, env.config, env.layout)
    script = tmp_path / "script.py"
    result = subject.evaluate(script, tmp_path / "cycle", 1)
    toolkit.run_script.assert_called_once()
    args = toolkit.run_script.call_args.args[1]
    assert args["mesh_path"] is None and args["retopology_method"] is None
    assert args["modelling_method"] == "procedural"
    toolkit.render_views.assert_called_once()
    toolkit.inspect_asset.assert_called_once()
    fidelity.check.assert_called_once()
    assert not result.ok and "export diagnostic" in result.error
    assert result.images == (preview,) and "blend" in result.artifacts
    assert subject.brief().references == ()


def test_trellis_subject_still_requires_input_mesh(procedural, tmp_path):
    env = procedural
    toolkit = Mock()
    record = AssetRecord(id=env.item.id, name=env.item.name)
    subject = AssetSubject(env.item, record, toolkit, Mock(), env.config, env.layout)
    with pytest.raises(KitbashError, match="has no Trellis mesh"):
        subject.evaluate(tmp_path / "script.py", tmp_path / "cycle", 1)
    toolkit.run_script.assert_not_called()


def test_procedural_script_uses_code_role_without_reference(procedural):
    env = procedural
    llm = Mock()
    llm.ask_python.return_value = "import bpy\n"
    catalog = Mock()
    catalog.textures.return_value = []
    agent = ModellingAgent(llm, Mock(), Mock(), None, None, None, catalog, env.config, env.layout, env.tracker)
    record = env.board.get(env.item.id)
    assert agent.write_script(env.item, record) == "import bpy\n"
    request = llm.ask_python.call_args.kwargs
    assert request["role"].value == "code" and request["attachments"] == ()
    assert request["variables"]["description"] == env.item.description
