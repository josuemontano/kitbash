"""Scene agent: builds the whole shot. Runs the phases in order, owns the global run state and hands
work to the breakdown, modelling and layout agents through their phases."""

import time
from collections.abc import Sequence

from kitbash.analytics import context
from kitbash.analytics.report import AnalyticsReport
from kitbash.analytics.tracker import SpanKind, Tracker
from kitbash.domain.phases import PhaseName, PhaseStatus
from kitbash.phases.base import Phase
from kitbash.store.state import StateDB


class SceneAgent:
    def __init__(self, phases: Sequence[Phase], state: StateDB, tracker: Tracker, report: AnalyticsReport) -> None:
        self._phases = tuple(phases)
        self._state = state
        self._tracker = tracker
        self._report = report

    def run(self, from_phase: PhaseName | None = None) -> dict:
        """Run every phase that is not done (or everything from ``from_phase``); analytics are always written."""
        if self._state.meta.get("run_started_at") is None:
            self._state.meta.set("run_started_at", time.time())
        if from_phase is not None:
            self._state.phases.reset(from_phase.and_later())
        try:
            for phase in self._phases:
                if self._state.phases.status(phase.name) is PhaseStatus.DONE:
                    continue
                with context.bind(phase=phase.name.value, agent="scene_agent"), self._tracker.span(SpanKind.PHASE, phase.name.value):
                    self._state.phases.set_status(phase.name, PhaseStatus.RUNNING)
                    phase.run()
                    self._state.phases.set_status(phase.name, PhaseStatus.DONE)
            self._state.meta.set("run_finished_at", time.time())
        finally:
            report = self._report.write()
        return report
