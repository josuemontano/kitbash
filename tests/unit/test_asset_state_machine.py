import pytest

from kitbash.domain.assets import (
    TERMINAL_STATES,
    TRANSITIONS,
    AssetRecord,
    AssetState,
    asset_from_dict,
    asset_to_dict,
    check_transition,
)
from kitbash.errors import TransitionError
from kitbash.pipeline.board import AssetBoard
from kitbash.store.state import StateDB

S = AssetState
HAPPY_PATH = [S.REFERENCING, S.GENERATING, S.BUILDING, S.CRITIQUING, S.BUILDING, S.CRITIQUING, S.AWAITING_REVIEW, S.APPROVED]


@pytest.fixture
def board(tmp_path):
    state = StateDB(tmp_path / "state.db")
    board = AssetBoard(state.assets)
    board.ensure([AssetRecord(id="crate", name="Crate"), AssetRecord(id="mug", name="Mug")])
    yield board, state
    state.close()


def test_every_state_has_a_transition_table_entry():
    assert set(TRANSITIONS) == set(AssetState)
    assert {S.APPROVED, S.SKIPPED} == TERMINAL_STATES


def test_happy_path_is_allowed():
    current = S.QUEUED
    for target in HAPPY_PATH:
        check_transition(current, target)
        current = target


@pytest.mark.parametrize(
    "current, target",
    [
        (S.QUEUED, S.BUILDING),
        (S.REFERENCING, S.AWAITING_REVIEW),
        (S.GENERATING, S.APPROVED),
        (S.AWAITING_REVIEW, S.BUILDING),
        (S.INPUT_NEEDED, S.GENERATING),
        (S.SKIPPED, S.APPROVED),
    ],
)
def test_illegal_transitions_are_rejected(current, target):
    with pytest.raises(TransitionError, match=f"from {current.value} to {target.value}"):
        check_transition(current, target)


def test_rework_paths():
    for entry_state in (S.BUILDING, S.GENERATING, S.REFERENCING):
        check_transition(S.NEEDS_REWORK, entry_state)
    check_transition(S.AWAITING_REVIEW, S.NEEDS_REWORK)
    check_transition(S.INPUT_NEEDED, S.QUEUED)


def test_state_properties():
    assert S.AWAITING_REVIEW.is_with_user and S.INPUT_NEEDED.is_with_user
    assert S.BUILDING.is_worker_state and S.NEEDS_REWORK.is_worker_state
    assert not S.APPROVED.is_worker_state and S.APPROVED.is_terminal


def test_record_serialization_round_trip():
    record = AssetRecord(id="a", name="A", state=S.NEEDS_REWORK, feedback=("x",), extra={"cycle": 2})
    assert asset_from_dict(asset_to_dict(record)) == record


def test_board_persists_transitions_and_notifies(board):
    board, state = board
    seen = []
    board.subscribe(lambda record: seen.append(record.state))
    for target in HAPPY_PATH:
        board.transition("crate", target, note=target.value)
    assert state.assets.get("crate").state is S.APPROVED
    assert seen == HAPPY_PATH
    log = [t for t in state.assets.transitions() if t["asset_id"] == "crate"]
    assert [t["to_state"] for t in log] == ["queued", *[s.value for s in HAPPY_PATH]]
    assert not board.all_resolved()
    board.transition("mug", S.SKIPPED)
    assert board.all_resolved()


def test_board_rejects_illegal_moves_and_merges_extra(board):
    board, _ = board
    with pytest.raises(TransitionError):
        board.transition("crate", S.APPROVED)
    board.update("crate", extra={"cycle": 1})
    board.update("crate", extra={"score": 0.5})
    assert board.get("crate").extra == {"cycle": 1, "score": 0.5}


def test_board_reloads_checkpointed_states(tmp_path):
    state = StateDB(tmp_path / "state.db")
    board = AssetBoard(state.assets)
    board.ensure([AssetRecord(id="crate", name="Crate")])
    board.transition("crate", S.REFERENCING)
    board.transition("crate", S.GENERATING)
    state.close()
    reopened = StateDB(tmp_path / "state.db")
    again = AssetBoard(reopened.assets)
    again.ensure([AssetRecord(id="crate", name="Crate")])  # existing records keep their state
    assert again.get("crate").state is S.GENERATING
    again.reset(AssetRecord(id="crate", name="Crate", state=S.APPROVED, reused=True))
    assert again.get("crate").reused
    reopened.close()
