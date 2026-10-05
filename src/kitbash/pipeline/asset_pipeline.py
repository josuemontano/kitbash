"""Advances one asset through its states until it needs the user (review or input) or is done."""

from collections.abc import Callable, Mapping
from pathlib import Path

from attrs import evolve

from kitbash.agents.modelling import ModellingAgent
from kitbash.analytics.tracker import SpanKind, Tracker
from kitbash.critique.sessions import ResumableLoop
from kitbash.critique.store import CycleResult
from kitbash.domain.assets import TRANSITIONS, AssetRecord, AssetState, ReworkEntry
from kitbash.domain.inventory import Inventory, InventoryItem
from kitbash.errors import TrellisError
from kitbash.pipeline.board import AssetBoard

S = AssetState
type Step = Callable[[AssetRecord, InventoryItem], AssetRecord]


class AssetProgress:
    """Loop observer that mirrors critic-loop progress into the asset's state and board fields."""

    def __init__(self, board: AssetBoard, asset_id: str) -> None:
        self._board = board
        self._asset_id = asset_id

    def building(self, cycle: int) -> None:
        self._move(S.BUILDING, cycle)

    def critiquing(self, cycle: int) -> None:
        self._move(S.CRITIQUING, cycle)

    def evaluated(self, result: CycleResult) -> None:
        self._board.update(self._asset_id, extra={"cycle": result.cycle, "last_score": round(result.score, 3)})

    def _move(self, state: AssetState, cycle: int) -> None:
        current = self._board.get(self._asset_id)
        if current.state is not state:
            self._board.transition(self._asset_id, state, f"cycle {cycle:02d}")
        self._board.update(self._asset_id, extra={"cycle": cycle})


class AssetPipeline:
    def __init__(self, board: AssetBoard, inventory: Inventory, agent: ModellingAgent, loop: ResumableLoop, tracker: Tracker) -> None:
        self._board = board
        self._inventory = inventory
        self._agent = agent
        self._loop = loop
        self._tracker = tracker
        self._steps: Mapping[AssetState, Step] = {
            S.QUEUED: self._start,
            S.REFERENCING: self._reference,
            S.GENERATING: self._generate,
            S.BUILDING: self._build,
            S.CRITIQUING: self._build,
            S.NEEDS_REWORK: self._rework,
        }

    def item_for(self, asset: AssetRecord) -> InventoryItem:
        """The inventory item, with any name or reference the user supplied for this asset."""
        overrides = {key: asset.extra[key] for key in ("search_name", "user_reference") if key in asset.extra}
        return evolve(self._inventory.item(asset.id), **overrides)

    def advance(self, asset_id: str) -> AssetRecord:
        asset = self._board.get(asset_id)
        while asset.state.is_worker_state:
            with self._tracker.span(SpanKind.STEP, asset.state.value, attempt=asset.attempt):
                asset = self._steps[asset.state](asset, self.item_for(asset))
        return asset

    def fail(self, asset_id: str, error: BaseException) -> AssetRecord:
        """Hand an asset whose step crashed to the user, as a review or an input request."""
        asset = self._board.get(asset_id)
        message = f"{type(error).__name__}: {error}"
        if asset.state is S.QUEUED:
            asset = self._board.transition(asset_id, S.REFERENCING)
        if asset.state is S.NEEDS_REWORK:
            asset = self._board.transition(asset_id, S.BUILDING)
        if S.AWAITING_REVIEW in TRANSITIONS[asset.state]:
            return self._board.transition(asset_id, S.AWAITING_REVIEW, "failed", error=message)
        return self._board.transition(
            asset_id, S.INPUT_NEEDED, "failed", error=message,
            input_request=f"{message}\nGive another item name to search for, or a reference image path.",
        )

    # -- steps -------------------------------------------------------------------------------------

    def _start(self, asset: AssetRecord, item: InventoryItem) -> AssetRecord:
        return self._board.transition(asset.id, S.REFERENCING)

    def _reference(self, asset: AssetRecord, item: InventoryItem) -> AssetRecord:
        index = asset.extra.get("reference_index")
        if index is not None:
            asset = self._board.update(asset.id, extra={"reference_index": None})
            choice = self._agent.select_reference(item, index)
        else:
            choice = self._agent.find_reference(item)
        if choice is None:
            review = self._agent.reference_review(item)
            request = (
                f"Reference candidates for '{item.name}' need your review; deterministic quality and rights checks "
                "could not select one automatically. Choose a numbered candidate, give another item name, "
                "or provide a reference image path."
                if review else
                f"No usable reference image was found for '{item.name}'. "
                "Give an item name to search for, or the path to a reference image."
            )
            return self._board.transition(
                asset.id, S.INPUT_NEEDED, "reference needs input", input_request=request,
                extra={**asset.extra, "reference_review": review},
            )
        return self._board.transition(
            asset.id, S.GENERATING, choice.source, reference_path=str(choice.path), error=None,
            extra={**asset.extra, "reference": choice.to_dict(), "reference_review": None, "reference_index": None},
        )

    def _generate(self, asset: AssetRecord, item: InventoryItem) -> AssetRecord:
        try:
            result = self._agent.generate_mesh(asset, Path(asset.reference_path or ""))
        except TrellisError as exc:
            return self._board.transition(
                asset.id, S.INPUT_NEEDED, "trellis failed", error=str(exc),
                input_request=f"Trellis could not reconstruct '{item.name}' from its reference image. "
                "Give another item name to search for, or the path to a better reference image.",
            )
        trellis = asset.extra.get("trellis", {})
        return self._board.transition(
            asset.id, S.BUILDING, "mesh ready", mesh_path=str(result.mesh_path),
            extra={**asset.extra, "trellis": {
                "duration_s": round(trellis.get("duration_s", 0.0) + result.duration_s, 2),
                "retries": trellis.get("retries", 0) + result.retries,
                "runs": trellis.get("runs", 0) + 1,
            }},
        )

    def _build(self, asset: AssetRecord, item: InventoryItem) -> AssetRecord:
        subject = self._agent.subject(item, asset)
        outcome = self._loop.run(
            subject,
            request=f"attempt-{asset.attempt}-feedback-{len(asset.feedback)}",
            initial_script=lambda: self._agent.write_script(item, self._board.get(asset.id)),
            feedback=asset.extra.get("pending_feedback"),
            fresh=bool(asset.extra.get("fresh_script")),
            observer=AssetProgress(self._board, asset.id),
        )
        self._agent.keep_script(asset.id, outcome.best.script_path)
        return self._board.transition(
            asset.id, S.AWAITING_REVIEW, outcome.reason.value, best_cycle=outcome.best.cycle,
            score=round(outcome.best.score, 4), error=outcome.best.evaluation.error,
            extra={**self._board.get(asset.id).extra, "loop_reason": outcome.reason.value, "loop_message": outcome.message,
                   "pending_feedback": None, "fresh_script": False},
        )

    def _rework(self, asset: AssetRecord, item: InventoryItem) -> AssetRecord:
        match asset.rework_entry:
            case ReworkEntry.REGENERATE:
                return self._board.transition(
                    asset.id, S.GENERATING, "regenerate", attempt=asset.attempt + 1, seed=asset.seed + 1,
                    extra={**asset.extra, "fresh_script": True},
                )
            case ReworkEntry.REFERENCE:
                return self._board.transition(
                    asset.id, S.REFERENCING, "new reference", attempt=asset.attempt + 1, extra={**asset.extra, "fresh_script": True}
                )
            case _:
                feedback = asset.feedback[-1] if asset.feedback else None
                return self._board.transition(asset.id, S.BUILDING, "feedback", extra={**asset.extra, "pending_feedback": feedback})
