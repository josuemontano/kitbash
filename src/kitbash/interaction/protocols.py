"""What the pipeline asks the user, as narrow interfaces (terminal or autopilot implement them)."""

from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from attrs import frozen

from kitbash.backlot.library import SearchHit
from kitbash.domain.assets import AssetRecord
from kitbash.domain.critique import ScoreCard
from kitbash.domain.inventory import InventoryItem
from kitbash.domain.phases import PhaseName


class ReviewAction(StrEnum):
    APPROVE = "approve"
    FEEDBACK = "feedback"
    REGENERATE = "regenerate"
    SKIP = "skip"
    PROVIDE_INPUT = "provide_input"


@frozen
class ReviewDecision:
    action: ReviewAction
    feedback: str = ""
    reference_path: str | None = None
    search_name: str | None = None
    reference_index: int | None = None
    procedural: bool = False  # the user explicitly chose programmatic modelling for this asset instead of Trellis


@frozen
class AssetReview:
    """Everything shown to the user for one finished asset."""

    asset: AssetRecord
    item: InventoryItem
    previews: tuple[Path, ...]
    reference: Path | None
    scorecard: ScoreCard | None
    loop_message: str
    usd_compare: Path | None
    queue_length: int
    committable: bool = False  # a complete build (.blend, .usd, preview) exists and can be approved


class AssetReviewer(Protocol):
    def review(self, review: AssetReview) -> ReviewDecision: ...

    def provide_input(self, asset: AssetRecord, item: InventoryItem, request: str) -> ReviewDecision: ...


class GateAction(StrEnum):
    APPROVE = "approve"
    PUBLISH_DEGRADED = "publish_degraded"
    FEEDBACK = "feedback"
    REWORK_ASSET = "rework_asset"
    ABORT = "abort"


@frozen
class GateDecision:
    action: GateAction
    feedback: str = ""
    asset_id: str | None = None


@frozen
class PhaseSummary:
    phase: PhaseName
    headline: str
    rows: tuple[tuple[str, ...], ...] = ()
    columns: tuple[str, ...] = ()
    images: tuple[Path, ...] = ()
    scorecard: ScoreCard | None = None
    message: str = ""
    asset_ids: tuple[str, ...] = ()  # assets the user may send back to rework (modelling gate)


class PhaseGate(Protocol):
    def confirm(self, summary: PhaseSummary) -> GateDecision: ...


class UnrecognizedAction(StrEnum):
    KEEP = "keep"
    SEARCH_NAME = "search_name"
    REFERENCE = "reference"
    DROP = "drop"


@frozen
class UnrecognizedAnswer:
    action: UnrecognizedAction
    value: str = ""


class InventoryQuestions(Protocol):
    def confirm_reuse(self, item: InventoryItem, hit: SearchHit) -> bool: ...

    def resolve_unrecognized(self, item: InventoryItem) -> UnrecognizedAnswer: ...


class UserChannel(AssetReviewer, PhaseGate, InventoryQuestions, Protocol):
    """A complete user (terminal or autopilot); consumers depend on the narrower protocols."""

    @property
    def interactive(self) -> bool: ...

    def notify(self, message: str) -> None: ...

    def show_images(self, images: Sequence[Path]) -> None: ...
