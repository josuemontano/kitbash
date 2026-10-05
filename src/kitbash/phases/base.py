"""Shared phase plumbing: the Phase interface and time-tracked user interactions."""

from collections.abc import Callable
from typing import Protocol

from kitbash.analytics.tracker import EventKind, SpanKind, Tracker
from kitbash.domain.phases import PhaseName
from kitbash.errors import UserAbort
from kitbash.interaction.protocols import GateAction, GateDecision, PhaseGate, PhaseSummary
from kitbash.store.state import RunMetaRepository


class Phase(Protocol):
    @property
    def name(self) -> PhaseName: ...

    def run(self) -> None: ...


def ask_user[T](tracker: Tracker, kind: SpanKind, name: str, question: Callable[[], T], describe: Callable[[T], str]) -> T:
    """Run a user interaction inside a span so user time is reported apart from compute time."""
    with tracker.span(kind, name) as span:
        answer = question()
        span.meta["answer"] = describe(answer)
    return answer


def run_gate(gate: PhaseGate, tracker: Tracker, summary: PhaseSummary) -> GateDecision:
    decision = ask_user(tracker, SpanKind.GATE, summary.phase.value, lambda: gate.confirm(summary), lambda d: d.action.value)
    if decision.action is not GateAction.APPROVE:
        tracker.event(EventKind.USER_INTERVENTION, f"gate_{decision.action.value}", phase=summary.phase.value, feedback=decision.feedback[:500])
    if decision.action is GateAction.ABORT:
        raise UserAbort(f"Stopped at the {summary.phase.value} gate", hint="Continue later with `kitbash resume`.")
    return decision


class GateRounds:
    """The feedback a phase gate received, persisted so each round is one resumable loop request."""

    def __init__(self, meta: RunMetaRepository, phase: PhaseName) -> None:
        self._meta = meta
        self._key = f"gate_feedback:{phase.value}"

    @property
    def rounds(self) -> list[str]:
        return list(self._meta.get(self._key) or [])

    @property
    def feedback(self) -> str | None:
        rounds = self.rounds
        return rounds[-1] if rounds else None

    def request(self, prefix: str = "") -> str:
        return f"{prefix}round-{len(self.rounds)}"

    def add(self, feedback: str) -> None:
        self._meta.set(self._key, [*self.rounds, feedback])

    def reset(self) -> None:
        self._meta.set(self._key, [])
