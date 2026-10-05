"""``--no-interactive``: every gate is approved, finished assets are approved in completion order,
input requests are skipped, reuse proposals above the threshold are accepted."""

import logging
from collections.abc import Sequence
from pathlib import Path

from kitbash.backlot.library import SearchHit
from kitbash.domain.assets import AssetRecord
from kitbash.domain.inventory import InventoryItem
from kitbash.interaction.protocols import (
    AssetReview,
    GateAction,
    GateDecision,
    PhaseSummary,
    ReviewAction,
    ReviewDecision,
    UnrecognizedAction,
    UnrecognizedAnswer,
)

log = logging.getLogger("kitbash.autopilot")


class AutoPilot:
    interactive = False

    def review(self, review: AssetReview) -> ReviewDecision:
        if not review.committable:
            log.warning("Skipping %s: it has no complete build to approve (%s)", review.asset.id, review.asset.error)
            return ReviewDecision(ReviewAction.SKIP)
        return ReviewDecision(ReviewAction.APPROVE)

    def provide_input(self, asset: AssetRecord, item: InventoryItem, request: str) -> ReviewDecision:
        log.warning("Input needed for %s, skipped in --no-interactive mode: %s", asset.id, request)
        return ReviewDecision(ReviewAction.SKIP)

    def confirm(self, summary: PhaseSummary) -> GateDecision:
        log.info("Auto-approved the %s gate: %s", summary.phase.value, summary.headline)
        return GateDecision(GateAction.APPROVE)

    def confirm_reuse(self, item: InventoryItem, hit: SearchHit) -> bool:
        log.info("Reusing backlot asset %s for %s (similarity %.2f)", hit.entry.id, item.id, hit.similarity)
        return True

    def resolve_unrecognized(self, item: InventoryItem) -> UnrecognizedAnswer:
        log.warning("Keeping unrecognized item %s (%s, confidence %.2f)", item.id, item.name, item.confidence)
        return UnrecognizedAnswer(UnrecognizedAction.KEEP)

    def notify(self, message: str) -> None:
        log.info(message)

    def show_images(self, images: Sequence[Path]) -> None:
        pass
