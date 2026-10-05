"""Phase 3: layout. Place the approved assets, set camera, lighting and world for the style, critique,
then pass the gate."""

import hashlib
import shutil
from collections.abc import Sequence

from kitbash.agents.layout import LayoutAgent, PlacedAsset
from kitbash.analytics.tracker import Tracker
from kitbash.critique.sessions import ResumableLoop
from kitbash.domain.inventory import InventoryItem
from kitbash.domain.phases import PhaseName
from kitbash.interaction.protocols import GateAction, PhaseSummary, UserChannel
from kitbash.paths import OutputLayout
from kitbash.phases.base import GateRounds, run_gate
from kitbash.phases.scene_assets import SceneCast
from kitbash.store.state import RunMetaRepository
from kitbash.ui.dashboard import Dashboard, PhaseProgress

CAST_KEY = "layout_cast"


class LayoutPhase:
    name = PhaseName.LAYOUT

    def __init__(
        self,
        agent: LayoutAgent,
        loop: ResumableLoop,
        cast: SceneCast,
        meta: RunMetaRepository,
        user: UserChannel,
        dashboard: Dashboard,
        tracker: Tracker,
        layout: OutputLayout,
    ) -> None:
        self._agent = agent
        self._loop = loop
        self._cast = cast
        self._state_meta = meta
        self._user = user
        self._dashboard = dashboard
        self._tracker = tracker
        self._layout = layout

    def run(self) -> None:
        inventory, placed, skipped = self._cast.load()
        subject = self._agent.subject(inventory, placed, skipped)
        rounds = GateRounds(self._state_meta, self.name)
        cast = cast_fingerprint(placed, skipped)
        fresh = self._state_meta.get(CAST_KEY) != cast  # the approved assets changed: write a new layout
        if fresh:
            rounds.reset()
            self._state_meta.set(CAST_KEY, cast)
        while True:
            progress = PhaseProgress("Layout")
            with self._dashboard.showing(progress.view):
                outcome = self._loop.run(
                    subject,
                    request=rounds.request(prefix=f"cast-{cast}-"),
                    initial_script=lambda: self._agent.write_script(inventory, placed, skipped),
                    feedback=rounds.feedback,
                    fresh=fresh and not rounds.rounds,
                    observer=progress,
                )
            shutil.copy2(outcome.best.script_path, self._layout.layout_script)
            summary = PhaseSummary(
                phase=self.name,
                headline=f"Layout with {len(placed)} assets and {len(skipped)} placeholders (best cycle {outcome.best.cycle:02d}).",
                images=outcome.best.evaluation.images[:2],
                scorecard=outcome.best.scorecard,
                message=outcome.message,
            )
            decision = run_gate(self._user, self._tracker, summary)
            if decision.action is GateAction.APPROVE:
                return
            rounds.add(decision.feedback)


def cast_fingerprint(placed: Sequence[PlacedAsset], skipped: Sequence[InventoryItem]) -> str:
    """Identifies which assets the layout places and which items are placeholders."""
    text = "|".join(sorted(f"{a.key}={a.backlot_id}" for a in placed)) + "#" + "|".join(sorted(i.id for i in skipped))
    return hashlib.sha1(text.encode()).hexdigest()[:10]
