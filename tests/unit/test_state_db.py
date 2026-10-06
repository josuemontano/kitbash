import time

import pytest

from kitbash.domain.assets import AssetRecord, AssetState, ReworkEntry
from kitbash.domain.inventory import Inventory
from kitbash.domain.phases import PhaseName, PhaseStatus
from kitbash.errors import StateError
from kitbash.infra.embeddings import HashingEmbedder
from kitbash.store.state import CycleRow, DiffRow, StateDB


@pytest.fixture
def state(tmp_path, embedder):
    db = StateDB(tmp_path / "state.db", embedder)
    yield db
    db.close()


def test_run_meta_round_trips_json(state):
    state.meta.set("input", {"mode": "image", "image": "/x.png"})
    state.meta.set("input", {"mode": "prompt", "prompt": "a room"})
    assert state.meta.get("input") == {"mode": "prompt", "prompt": "a room"}
    assert state.meta.get("missing", 42) == 42
    assert state.meta.all()["input"]["prompt"] == "a room"


def test_phase_status_lifecycle(state):
    assert state.phases.status(PhaseName.BREAKDOWN) is PhaseStatus.PENDING
    state.phases.set_status(PhaseName.BREAKDOWN, PhaseStatus.RUNNING)
    state.phases.set_status(PhaseName.BREAKDOWN, PhaseStatus.DONE)
    row = next(r for r in state.phases.rows() if r["name"] == "breakdown")
    assert row["status"] == "done" and row["finished_at"] >= row["started_at"]
    state.phases.reset(PhaseName.BREAKDOWN.and_later())
    assert state.phases.status(PhaseName.BREAKDOWN) is PhaseStatus.PENDING


def test_inventory_is_persisted_indexed_and_searchable(state, sample_inventory_dict):
    inventory = Inventory.from_dict(sample_inventory_dict)
    state.inventory.save(inventory)
    loaded = state.inventory.load()
    assert loaded == inventory
    rows = state.db.query("SELECT id, name, category FROM inventory_items ORDER BY ord")
    assert [tuple(r) for r in rows] == [("wooden_crate", "Wooden crate", "prop"), ("ceramic_mug", "Ceramic mug", "decor")]
    hits = state.inventory.search("white coffee mug", k=2)
    assert hits[0][0] == "ceramic_mug"


def test_assets_checkpoint_and_transition_log(state):
    record = AssetRecord(id="crate", name="Crate")
    state.assets.upsert(record, 0)
    state.assets.add_transition("crate", None, AssetState.QUEUED, "created")
    updated = AssetRecord(
        id="crate", name="Crate", state=AssetState.NEEDS_REWORK, rework_entry=ReworkEntry.BUILD,
        feedback=("taller",), extra={"cycle": 3},
    )
    state.assets.upsert(updated)
    state.assets.add_transition("crate", AssetState.QUEUED, AssetState.NEEDS_REWORK)
    assert state.assets.get("crate") == updated
    assert [t["to_state"] for t in state.assets.transitions()] == ["queued", "needs_rework"]
    state.assets.upsert(AssetRecord(id="mug", name="Mug"))
    state.assets.delete_missing(["mug"])
    assert [a.id for a in state.assets.all()] == ["mug"]


def test_cycles_and_diffs(state):
    state.cycles.save_cycle(CycleRow("modelling", "crate", 1, "/c/01/script.py", "/c/01/diff.patch", None, None, None, "pending", "", time.time()))
    state.cycles.save_diff(DiffRow("modelling", "crate", 1, "applied", "abc", "/c/01/diff.patch", "initial", None, None))
    state.cycles.save_cycle(CycleRow("modelling", "crate", 1, "/c/01/script.py", "/c/01/diff.patch", "/c/01/critique.json", 0.7, False, "kept", "ok", time.time()))
    state.cycles.update_diff(PhaseName.MODELLING, "crate", 1, status="kept", score_after=0.7)
    (cycle,) = state.cycles.cycles(PhaseName.MODELLING, "crate")
    assert cycle.score == 0.7 and cycle.passed is False and cycle.status == "kept"
    (diff,) = state.cycles.diffs(PhaseName.MODELLING, "crate")
    assert diff.status == "kept" and diff.score_after == 0.7
    assert state.cycles.cycles(PhaseName.MODELLING, "other") == []


def test_spans_and_events(state):
    state.spans.add_span("llm", "task", 1.0, 3.5, {"phase": "breakdown", "asset_id": None}, {"tokens_in": 10})
    state.spans.add_event("retry", "task", {"phase": "breakdown"}, {"reason": "timeout"})
    (span,) = state.spans.spans()
    assert span["meta"] == {"tokens_in": 10} and span["phase"] == "breakdown"
    assert state.spans.events()[0]["meta"]["reason"] == "timeout"


def test_vector_dimension_mismatch_is_reported(tmp_path):
    StateDB(tmp_path / "state.db", HashingEmbedder(32)).close()
    with pytest.raises(StateError, match="dimensions"):
        StateDB(tmp_path / "state.db", HashingEmbedder(48))


def test_inventory_rejects_different_model_at_same_dimensions(tmp_path, sample_inventory_dict):
    class OtherModel(HashingEmbedder):
        @property
        def name(self):
            return "other-model"

    path = tmp_path / "state.db"
    state = StateDB(path, HashingEmbedder(64))
    state.inventory.save(Inventory.from_dict(sample_inventory_dict))
    state.close()
    with pytest.raises(StateError):
        StateDB(path, OtherModel(64))
    reopened = StateDB(path, HashingEmbedder(64))
    try:
        assert reopened.inventory.load() == Inventory.from_dict(sample_inventory_dict)
        assert reopened.inventory.search("white coffee mug", k=1)[0][0] == "ceramic_mug"
    finally:
        reopened.close()
