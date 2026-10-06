"""Per-asset state machine for the modelling phase.

queued -> referencing -> generating -> building -> critiquing -> awaiting_review -> approved | skipped | needs_rework
Procedural assets take queued -> building, then the same critique, review and approval path.

``input_needed`` extends the machine for assets that wait on the user (no reference image found,
unrecognized item, Trellis failing on every attempt). Workers never block on it: the asset leaves the
worker pool and comes back as ``queued`` once the user answers.
"""

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from attrs import field, frozen

from kitbash.errors import TransitionError


class AssetState(StrEnum):
    QUEUED = "queued"
    REFERENCING = "referencing"
    INPUT_NEEDED = "input_needed"
    GENERATING = "generating"
    BUILDING = "building"
    CRITIQUING = "critiquing"
    AWAITING_REVIEW = "awaiting_review"
    APPROVED = "approved"
    SKIPPED = "skipped"
    NEEDS_REWORK = "needs_rework"

    @property
    def is_terminal(self) -> bool:
        return self in TERMINAL_STATES

    @property
    def is_with_user(self) -> bool:
        """The asset waits on the user rather than on a worker."""
        return self in (AssetState.AWAITING_REVIEW, AssetState.INPUT_NEEDED)

    @property
    def is_worker_state(self) -> bool:
        return not (self.is_terminal or self.is_with_user)


S = AssetState

TRANSITIONS: Mapping[AssetState, frozenset[AssetState]] = {
    S.QUEUED: frozenset({S.REFERENCING, S.BUILDING, S.SKIPPED}),
    S.REFERENCING: frozenset({S.GENERATING, S.INPUT_NEEDED}),
    S.INPUT_NEEDED: frozenset({S.QUEUED, S.SKIPPED}),
    S.GENERATING: frozenset({S.BUILDING, S.INPUT_NEEDED}),
    S.BUILDING: frozenset({S.CRITIQUING, S.AWAITING_REVIEW}),
    S.CRITIQUING: frozenset({S.BUILDING, S.AWAITING_REVIEW}),
    S.AWAITING_REVIEW: frozenset({S.APPROVED, S.SKIPPED, S.NEEDS_REWORK}),
    S.NEEDS_REWORK: frozenset({S.BUILDING, S.GENERATING, S.REFERENCING}),
    S.APPROVED: frozenset({S.NEEDS_REWORK}),
    S.SKIPPED: frozenset({S.QUEUED}),
}

TERMINAL_STATES = frozenset({S.APPROVED, S.SKIPPED})


def check_transition(current: AssetState, target: AssetState) -> None:
    if target not in TRANSITIONS[current]:
        allowed = ", ".join(sorted(t.value for t in TRANSITIONS[current])) or "none"
        raise TransitionError(f"Asset cannot go from {current.value} to {target.value} (allowed: {allowed})")


class ReworkEntry(StrEnum):
    """Where a reworked asset re-enters the pipeline."""

    BUILD = "building"  # feedback: patch the build script
    REGENERATE = "generating"  # new Trellis mesh, new build script
    REFERENCE = "referencing"  # new reference image


@frozen
class AssetRecord:
    id: str
    name: str
    state: AssetState = AssetState.QUEUED
    attempt: int = 1
    seed: int = 42
    rework_entry: ReworkEntry | None = None
    feedback: tuple[str, ...] = ()
    reference_path: str | None = None
    mesh_path: str | None = None
    best_cycle: int | None = None
    score: float | None = None
    backlot_id: str | None = None
    reused: bool = False
    input_request: str | None = None
    error: str | None = None
    extra: Mapping[str, Any] = field(factory=dict)
    modelling_method: str = "trellis"

    @property
    def has_build(self) -> bool:
        return self.best_cycle is not None


def asset_to_dict(record: AssetRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "name": record.name,
        "state": record.state.value,
        "attempt": record.attempt,
        "seed": record.seed,
        "rework_entry": record.rework_entry.value if record.rework_entry else None,
        "feedback": list(record.feedback),
        "reference_path": record.reference_path,
        "mesh_path": record.mesh_path,
        "best_cycle": record.best_cycle,
        "score": record.score,
        "backlot_id": record.backlot_id,
        "reused": record.reused,
        "input_request": record.input_request,
        "error": record.error,
        "extra": dict(record.extra),
        "modelling_method": record.modelling_method,
    }


def asset_from_dict(data: Mapping[str, Any]) -> AssetRecord:
    return AssetRecord(
        id=data["id"],
        name=data["name"],
        state=AssetState(data["state"]),
        attempt=int(data.get("attempt", 1)),
        seed=int(data.get("seed", 42)),
        rework_entry=ReworkEntry(data["rework_entry"]) if data.get("rework_entry") else None,
        feedback=tuple(data.get("feedback", ())),
        reference_path=data.get("reference_path"),
        mesh_path=data.get("mesh_path"),
        best_cycle=data.get("best_cycle"),
        score=data.get("score"),
        backlot_id=data.get("backlot_id"),
        reused=bool(data.get("reused", False)),
        input_request=data.get("input_request"),
        error=data.get("error"),
        extra=dict(data.get("extra", {})),
        modelling_method=data.get("modelling_method", "trellis"),
    )
