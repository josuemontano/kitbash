"""Phase 2: modelling, pipelined. Workers (producers) take assets through reference, Trellis, build and
critique; a single review loop (consumer) presents finished assets to the user in completion order while
generation continues. The phase ends at the barrier: every asset approved or skipped."""

import logging
import time
from pathlib import Path

from attrs import evolve

from kitbash.agents.modelling import ModellingAgent
from kitbash.analytics import context
from kitbash.analytics.tracker import EventKind, SpanKind, Tracker
from kitbash.config import Config
from kitbash.critique.sessions import ResumableLoop
from kitbash.domain.assets import AssetRecord, AssetState, ReworkEntry
from kitbash.domain.inventory import Inventory, InventoryItem
from kitbash.domain.phases import PhaseName
from kitbash.errors import KitbashError, StateError
from kitbash.infra.process import defer_interrupts
from kitbash.interaction.protocols import AssetReview, GateAction, PhaseSummary, ReviewAction, ReviewDecision, UserChannel
from kitbash.phases.base import ask_user, run_gate
from kitbash.pipeline.asset_pipeline import AssetPipeline
from kitbash.pipeline.board import AssetBoard
from kitbash.pipeline.commit import BacklotCommitter
from kitbash.pipeline.review_queue import EntryKind, ReviewEntry, ReviewQueue
from kitbash.pipeline.scheduler import Scheduler
from kitbash.pipeline.workers import WorkerPool
from kitbash.store.state import StateDB
from kitbash.ui.dashboard import Dashboard, ModellingView

S = AssetState
log = logging.getLogger("kitbash.modelling")
POLL_S = 0.25


class ModellingPhase:
    name = PhaseName.MODELLING

    def __init__(
        self,
        agent: ModellingAgent,
        loop: ResumableLoop,
        committer: BacklotCommitter,
        state: StateDB,
        user: UserChannel,
        dashboard: Dashboard,
        tracker: Tracker,
        config: Config,
    ) -> None:
        self._agent = agent
        self._loop = loop
        self._committer = committer
        self._state = state
        self._user = user
        self._dashboard = dashboard
        self._tracker = tracker
        self._config = config
        self._since: dict[str, float] = {}

    def run(self) -> None:
        inventory = self._state.inventory.load()
        if inventory is None:
            raise StateError("There is no inventory yet", hint="Run the breakdown phase first.")
        board = self._board(inventory)
        pipeline = AssetPipeline(board, inventory, self._agent, self._loop, self._tracker)
        while True:
            self._execute(board, pipeline)
            decision = run_gate(self._user, self._tracker, self._summary(board))
            if decision.action is GateAction.APPROVE:
                return
            if decision.action is GateAction.REWORK_ASSET and decision.asset_id:
                self._reopen(board, decision.asset_id, decision.feedback)

    # -- setup -------------------------------------------------------------------------------------

    def _board(self, inventory: Inventory) -> AssetBoard:
        board = AssetBoard(self._state.assets)
        modelled = inventory.modelled_items()
        board.retain(item.id for item in modelled)
        board.ensure(self._fresh_record(item.id, item.name, item.reuse_backlot_id) for item in modelled)
        for item in modelled:  # reuse decisions may change when the breakdown is re-run
            record = board.get(item.id)
            if record.reused and item.reuse_backlot_id != record.backlot_id:
                board.reset(self._fresh_record(item.id, item.name, item.reuse_backlot_id), "reuse decision changed")
            elif item.reuse_backlot_id and not record.reused and record.state is not S.APPROVED:
                board.reset(self._fresh_record(item.id, item.name, item.reuse_backlot_id), "reuse accepted")
        now = time.time()
        self._since = {record.id: now for record in board.all()}
        last_state = {record.id: record.state for record in board.all()}

        def track(record: AssetRecord) -> None:
            if last_state.get(record.id) is not record.state:
                last_state[record.id] = record.state
                self._since[record.id] = time.time()

        board.subscribe(track)
        return board

    def _fresh_record(self, asset_id: str, name: str, reuse_backlot_id: str | None) -> AssetRecord:
        """A new asset: approved straight away when it reuses a backlot asset, otherwise queued for modelling."""
        return AssetRecord(
            id=asset_id,
            name=name,
            state=S.APPROVED if reuse_backlot_id else S.QUEUED,
            seed=self._config.trellis.seed,
            reused=bool(reuse_backlot_id),
            backlot_id=reuse_backlot_id,
        )

    # -- producer / consumer -------------------------------------------------------------------------

    def _execute(self, board: AssetBoard, pipeline: AssetPipeline) -> None:
        scheduler = Scheduler(self._config.pipeline.review_buffer, self._tracker)
        queue = ReviewQueue()
        for record in board.all():
            self._route(record, scheduler, queue, initial=True)

        def failed(asset_id: str, error: BaseException) -> AssetRecord:
            log.exception("Asset %s failed", asset_id, exc_info=error)
            return pipeline.fail(asset_id, error)

        pool = WorkerPool(
            self._config.pipeline.threads, scheduler, pipeline.advance, failed,
            lambda record: self._route(record, scheduler, queue), self._tracker, self.name.value,
        )
        try:
            pool.start()
            with self._dashboard.showing(ModellingView(board, scheduler, queue, self._since)):
                while not board.all_resolved():
                    if pool.fatal is not None:
                        raise pool.fatal
                    entry = queue.get(timeout=POLL_S)
                    if entry is None:
                        self._recover_lost(board, scheduler, queue)
                        continue
                    try:
                        self._handle(entry, board, pipeline, scheduler, queue)
                    finally:
                        queue.done()
        finally:
            with defer_interrupts():
                pool.stop()
                pool.join()
        if pool.fatal is not None:
            raise pool.fatal

    def _route(self, record: AssetRecord, scheduler: Scheduler, queue: ReviewQueue, *, initial: bool = False) -> None:
        match record.state:
            case S.AWAITING_REVIEW:
                if initial:
                    scheduler.hold(record.id)
                queue.put(ReviewEntry(EntryKind.REVIEW, record.id))
            case S.INPUT_NEEDED:
                scheduler.release(record.id)
                queue.put(ReviewEntry(EntryKind.INPUT_NEEDED, record.id, record.input_request or ""))
            case state if state.is_worker_state:
                scheduler.submit(record.id)

    def _recover_lost(self, board: AssetBoard, scheduler: Scheduler, queue: ReviewQueue) -> None:
        """Safety net: if nothing is running or queued but assets are unresolved, requeue them."""
        if not scheduler.idle() or len(queue) or queue.current is not None:
            return
        for record in board.all():
            if not record.state.is_terminal:
                self._tracker.event(EventKind.WARNING, "requeued_lost_asset", asset=record.id, state=record.state.value)
                self._route(record, scheduler, queue, initial=True)

    def _handle(self, entry: ReviewEntry, board: AssetBoard, pipeline: AssetPipeline, scheduler: Scheduler, queue: ReviewQueue) -> None:
        asset = board.get(entry.asset_id)
        item = pipeline.item_for(asset)
        with context.bind(phase=self.name.value, asset_id=asset.id, agent="user"):
            self._tracker.record(SpanKind.REVIEW_WAIT, asset.id, entry.enqueued_at, time.time(), entry=entry.kind.value)
            if entry.kind is EntryKind.INPUT_NEEDED:
                decision = ask_user(
                    self._tracker, SpanKind.USER_INPUT, asset.id,
                    lambda: self._user.provide_input(asset, item, asset.input_request or entry.message), lambda d: d.action.value,
                )
            else:
                review = self._review(asset, item, len(queue))
                decision = ask_user(self._tracker, SpanKind.USER_REVIEW, asset.id, lambda: self._user.review(review), lambda d: d.action.value)
            if self._user.interactive and decision.action is not ReviewAction.APPROVE:
                self._tracker.event(EventKind.USER_INTERVENTION, decision.action.value, asset=asset.id, feedback=decision.feedback[:500])
            self._apply(decision, asset, item, board, scheduler)

    def _review(self, asset: AssetRecord, item: InventoryItem, queue_length: int) -> AssetReview:
        best = self._committer.best(asset, item)
        artifacts = best.evaluation.artifacts if best else {}
        compare = artifacts.get("usd_compare")
        return AssetReview(
            asset=asset,
            item=item,
            previews=best.evaluation.images if best else (),
            reference=Path(asset.reference_path) if asset.reference_path else None,
            scorecard=best.scorecard if best else None,
            loop_message=str(asset.extra.get("loop_message", "")),
            usd_compare=Path(compare) if compare else None,
            queue_length=queue_length,
            committable=self._committer.committable(best),
        )

    def _apply(self, decision: ReviewDecision, asset: AssetRecord, item: InventoryItem, board: AssetBoard, scheduler: Scheduler) -> None:
        match decision.action:
            case ReviewAction.APPROVE:
                try:
                    entry = self._committer.commit(asset, item)
                except KitbashError as exc:
                    self._user.notify(f"{asset.id} could not be saved to the backlot: {exc}")
                    board.transition(asset.id, S.NEEDS_REWORK, "commit failed", rework_entry=ReworkEntry.BUILD,
                                     feedback=(*asset.feedback, f"Fix this so the asset can be saved: {exc}"))
                    scheduler.submit_rework(asset.id)
                    return
                board.transition(asset.id, S.APPROVED, "approved", backlot_id=entry.id)
                scheduler.release(asset.id)
                self._user.notify(f"{asset.id} approved and saved to the backlot as {entry.id}")
            case ReviewAction.FEEDBACK | ReviewAction.REGENERATE:
                entry_point = ReworkEntry.BUILD if decision.action is ReviewAction.FEEDBACK else ReworkEntry.REGENERATE
                feedback = (*asset.feedback, decision.feedback) if decision.feedback else asset.feedback
                board.transition(asset.id, S.NEEDS_REWORK, decision.action.value, rework_entry=entry_point, feedback=feedback)
                scheduler.submit_rework(asset.id)
            case ReviewAction.SKIP:
                board.transition(asset.id, S.SKIPPED, "skipped")
                scheduler.release(asset.id)
            case ReviewAction.PROVIDE_INPUT:
                extra = {**asset.extra, "fresh_script": True, "reference_index": decision.reference_index}
                if decision.reference_index is None:
                    extra.update(search_name=decision.search_name, user_reference=decision.reference_path, reference_review=None)
                board.transition(
                    asset.id, S.QUEUED, "user input", input_request=None, error=None, attempt=asset.attempt + 1,
                    extra=extra,
                )
                scheduler.submit(asset.id)

    # -- gate ----------------------------------------------------------------------------------------

    def _reopen(self, board: AssetBoard, asset_id: str, feedback: str) -> None:
        asset = board.get(asset_id)
        notes = (*asset.feedback, feedback) if feedback else asset.feedback
        if asset.reused:  # model it from scratch instead of reusing the backlot asset
            # Resume must see the inventory decision and its checkpoint together.
            with self._state.db.transaction():
                self._state.inventory.clear_reuse(asset_id)
                board.reset(evolve(self._fresh_record(asset.id, asset.name, None), feedback=notes), "reuse dropped at the gate")
        elif asset.state is S.SKIPPED:
            board.transition(asset_id, S.QUEUED, "reopened", feedback=notes, extra={**asset.extra, "fresh_script": True})
        else:
            board.transition(asset_id, S.NEEDS_REWORK, "reopened", rework_entry=ReworkEntry.BUILD, feedback=notes)

    def _summary(self, board: AssetBoard) -> PhaseSummary:
        records = board.all()
        rows = tuple(
            (r.id, r.name, r.state.value, "-" if r.score is None else f"{r.score:.2f}", r.backlot_id or "-",
             "reused" if r.reused else str(r.extra.get("loop_reason", "")))
            for r in records
        )
        approved = sum(1 for r in records if r.state is S.APPROVED)
        return PhaseSummary(
            phase=self.name,
            headline=f"{approved} of {len(records)} assets approved, {len(records) - approved} skipped (placeholders in the layout).",
            columns=("asset", "name", "state", "score", "backlot id", "note"),
            rows=rows,
            asset_ids=tuple(r.id for r in records),
        )
