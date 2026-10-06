"""Phase 3: layout. Place the approved assets, set camera, lighting and world for the style, critique,
then pass the gate."""

import shutil

from kitbash.agents.layout import LayoutAgent
from kitbash.analytics.tracker import Tracker
from kitbash.critique.sessions import ResumableLoop
from kitbash.domain.phases import PhaseName
from kitbash.interaction.protocols import GateAction, PhaseSummary, UserChannel
from kitbash.paths import OutputLayout
from kitbash.phases.base import GateRounds, run_gate
from kitbash.phases.scene_assets import SceneCast
from kitbash.store.state import RunMetaRepository
from kitbash.ui.dashboard import Dashboard, PhaseProgress

DEPENDENCIES_KEY = "layout_dependencies"


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
        dependencies = subject.dependency_fingerprint()
        if self._state_meta.get(DEPENDENCIES_KEY) != dependencies:
            rounds.reset()
            self._state_meta.set(DEPENDENCIES_KEY, dependencies)
        while True:
            progress = PhaseProgress("Layout")
            with self._dashboard.showing(progress.view):
                outcome = self._loop.run(
                    subject,
                    request=rounds.request(prefix=f"layout-{dependencies}-"),
                    initial_script=lambda: self._agent.write_script(inventory, placed, skipped),
                    feedback=rounds.feedback,
                    # Round zero is fresh even after interruption between the dependency
                    # checkpoint and session creation. Matching sessions still resume.
                    fresh=not rounds.rounds,
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
