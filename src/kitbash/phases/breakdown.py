"""Phase 1: breakdown. Analyze the input, refine the inventory with the critic loop, settle reuse and
unrecognized items with the user, then pass the gate."""

import json
import shutil
from pathlib import Path
from typing import Any

from attrs import evolve

from kitbash.agents.breakdown import BreakdownAgent
from kitbash.analytics.tracker import EventKind, SpanKind, Tracker
from kitbash.backlot.library import Backlot
from kitbash.config import Config
from kitbash.critique.loop import LoopOutcome
from kitbash.critique.sessions import ResumableLoop
from kitbash.domain.inventory import Inventory, InventoryItem
from kitbash.domain.phases import PhaseName
from kitbash.interaction.protocols import GateAction, PhaseSummary, UnrecognizedAction, UnrecognizedAnswer, UserChannel
from kitbash.paths import OutputLayout
from kitbash.phases.base import GateRounds, ask_user, run_gate
from kitbash.store.state import StateDB
from kitbash.ui.dashboard import Dashboard, PhaseProgress

DECISIONS_KEY = "breakdown_decisions"


class BreakdownPhase:
    name = PhaseName.BREAKDOWN

    def __init__(
        self,
        agent: BreakdownAgent,
        loop: ResumableLoop,
        state: StateDB,
        backlot: Backlot,
        user: UserChannel,
        dashboard: Dashboard,
        tracker: Tracker,
        config: Config,
        layout: OutputLayout,
    ) -> None:
        self._agent = agent
        self._loop = loop
        self._state = state
        self._backlot = backlot
        self._user = user
        self._dashboard = dashboard
        self._tracker = tracker
        self._config = config
        self._layout = layout

    def run(self) -> None:
        analyzed = self._state.meta.get("breakdown_analysis")
        if analyzed is None:
            with self._tracker.span(SpanKind.STEP, "breakdown.analyze"):
                inventory = self._agent.analyze()
            analyzed = inventory.to_dict()
            self._state.meta.set("breakdown_analysis", analyzed)
            self._persist(inventory)
        initial = Inventory.from_dict(analyzed)
        subject = self._agent.subject()
        rounds = GateRounds(self._state.meta, self.name)
        while True:
            progress = PhaseProgress("Breakdown")
            with self._dashboard.showing(progress.view):
                outcome = self._loop.run(
                    subject,
                    request=rounds.request(),
                    initial_script=lambda: self._agent.initial_script(initial),
                    feedback=rounds.feedback,
                    observer=progress,
                )
            inventory = self._settle(self._agent.inventory_of(outcome.best))
            self._persist(inventory)
            decision = run_gate(self._user, self._tracker, self._summary(inventory, outcome))
            if decision.action is GateAction.APPROVE:
                return
            rounds.add(decision.feedback)

    # -- decisions ---------------------------------------------------------------------------------

    def _settle(self, inventory: Inventory) -> Inventory:
        """Apply (and ask for, once) the user's answers about unrecognized items and backlot reuse."""
        decisions: dict[str, dict[str, Any]] = self._state.meta.get(DECISIONS_KEY) or {"unrecognized": {}, "reuse": {}}
        threshold = self._config.breakdown.confidence_threshold
        for item in inventory.items:
            if item.confidence >= threshold:
                continue
            stored = decisions["unrecognized"].get(item.id)
            answer = UnrecognizedAnswer(UnrecognizedAction(stored["action"]), stored["value"]) if stored else self._ask_unrecognized(item)
            decisions["unrecognized"][item.id] = {"action": answer.action.value, "value": answer.value}
            inventory = self._apply_unrecognized(inventory, item.id, answer)
        inventory = inventory.with_duplicates_linked()
        for item_id, hit in self._agent.backlot_matches(inventory, self._backlot).items():
            stored = decisions["reuse"].get(item_id)
            if stored and stored["backlot_id"] == hit.entry.id:
                accepted = stored["accepted"]
            else:
                item = inventory.item(item_id)
                accepted = ask_user(
                    self._tracker, SpanKind.USER_INPUT, f"reuse:{item_id}",
                    lambda item=item, hit=hit: self._user.confirm_reuse(item, hit), str,
                )
                self._tracker.event(EventKind.BACKLOT, "reuse_decision", item=item_id, backlot_id=hit.entry.id, accepted=accepted, similarity=hit.similarity)
                if self._user.interactive:
                    self._tracker.event(EventKind.USER_INTERVENTION, "reuse_decision", item=item_id, accepted=accepted)
            decisions["reuse"][item_id] = {"backlot_id": hit.entry.id, "accepted": accepted}
            if accepted:
                inventory = inventory.replace_item(evolve(inventory.item(item_id), reuse_backlot_id=hit.entry.id))
        self._state.meta.set(DECISIONS_KEY, decisions)
        return inventory

    def _ask_unrecognized(self, item: InventoryItem) -> UnrecognizedAnswer:
        answer = ask_user(
            self._tracker, SpanKind.USER_INPUT, f"unrecognized:{item.id}",
            lambda: self._user.resolve_unrecognized(item), lambda a: a.action.value,
        )
        if self._user.interactive:
            self._tracker.event(EventKind.USER_INTERVENTION, "unrecognized_item", item=item.id, action=answer.action.value)
        return answer

    def _apply_unrecognized(self, inventory: Inventory, item_id: str, answer: UnrecognizedAnswer) -> Inventory:
        item = inventory.item(item_id)
        match answer.action:
            case UnrecognizedAction.SEARCH_NAME:
                return inventory.replace_item(evolve(item, search_name=answer.value))
            case UnrecognizedAction.REFERENCE:
                source = Path(answer.value)
                copy = self._layout.input_dir / f"reference_{item_id}{source.suffix}"
                if source.is_file() and not copy.exists():
                    shutil.copy2(source, copy)
                return inventory.replace_item(evolve(item, user_reference=str(copy if copy.exists() else source)))
            case UnrecognizedAction.DROP:
                return inventory.without(item_id)
        return inventory

    # -- output --------------------------------------------------------------------------------------

    def _persist(self, inventory: Inventory) -> None:
        self._state.inventory.save(inventory)
        self._layout.inventory_json.parent.mkdir(parents=True, exist_ok=True)
        self._layout.inventory_json.write_text(json.dumps(inventory.to_dict(), indent=2), encoding="utf-8")

    def _summary(self, inventory: Inventory, outcome: LoopOutcome) -> PhaseSummary:
        modelled = inventory.modelled_items()
        reused = sum(1 for i in modelled if i.reuse_backlot_id)
        rows = tuple(
            (i.id, i.name, i.category, "{:.2f} x {:.2f} x {:.2f}".format(*i.dimensions.as_tuple()), f"{i.confidence:.2f}",
             f"copy of {i.same_as}" if i.same_as else (f"reuse {i.reuse_backlot_id}" if i.reuse_backlot_id else "model"))
            for i in inventory.items
        )
        return PhaseSummary(
            phase=self.name,
            headline=(
                f"{len(inventory.items)} items: {len(modelled) - reused} to model, {reused} reused from the backlot, "
                f"{len(inventory.items) - len(modelled)} copies."
            ),
            columns=("id", "name", "category", "size (m)", "confidence", "plan"),
            rows=rows,
            images=outcome.best.evaluation.images[:1],
            scorecard=outcome.best.scorecard,
            message=outcome.message,
        )
