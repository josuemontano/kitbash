"""The automatic critic loop shared by breakdown, modelling (per asset) and layout.

Each cycle evaluates a script, asks every critic for rubric scores and edits, and has the code role turn
those edits into a unified diff against the latest kept script. Everything is checkpointed under
``cycles/NN/`` and in ``state.db`` so a loop resumes where it stopped.
"""

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from typing import Protocol

from attrs import define, frozen

from kitbash.analytics import context
from kitbash.analytics.tracker import EventKind, SpanKind, Tracker
from kitbash.config import CriticConfig
from kitbash.critique.critics import Critic, PatchRequest, PatchWriter, ReviewRequest
from kitbash.critique.history import DiffStatus, ScoreTrend
from kitbash.critique.store import CycleResult, CycleStore, LoopState
from kitbash.critique.subject import Evaluation, LoopSubject
from kitbash.domain.critique import Critique, Edit
from kitbash.domain.rubric import Rubric
from kitbash.errors import BlenderScriptError, KitbashError, LLMAccessError, LLMError, PatchError
from kitbash.infra.patching import apply_diff, make_diff
from kitbash.llm.parsing import check_python


class LoopObserver(Protocol):
    """Progress callbacks (UI, asset states). Every method is optional to implement meaningfully."""

    def building(self, cycle: int) -> None: ...

    def critiquing(self, cycle: int) -> None: ...

    def evaluated(self, result: CycleResult) -> None: ...


class _SilentObserver:
    def building(self, cycle: int) -> None:
        pass

    def critiquing(self, cycle: int) -> None:
        pass

    def evaluated(self, result: CycleResult) -> None:
        pass


class LoopReason(StrEnum):
    PASSED = "passed"
    MAX_CYCLES = "max_cycles"
    STALLED = "stalled"
    REPEATED_DIFF = "repeated_diff"
    PATCH_FAILED = "patch_failed"


@frozen
class LoopOutcome:
    reason: LoopReason
    best: CycleResult
    cycles_run: int
    message: str

    @property
    def passed(self) -> bool:
        return self.reason is LoopReason.PASSED


@define
class _Session:
    """One run of the loop: its first cycle, the result it builds on and whether it starts from scratch."""

    start: int
    base: CycleResult | None
    fresh: bool
    observer: LoopObserver

    def best(self, state: LoopState) -> CycleResult | None:
        """The session's highest-scoring result (what the loop hands back)."""
        return state.best(self.start) or self.base

    def head(self, state: LoopState) -> CycleResult | None:
        """The result the next patch builds on: the latest kept cycle of the session."""
        kept = [r for c, r in state.results.items() if c >= self.start and r.status is DiffStatus.KEPT]
        return max(kept, key=lambda r: r.cycle, default=None) or self.base


class CriticLoop:
    def __init__(
        self,
        *,
        critics: Sequence[Critic],
        patch_writer: PatchWriter,
        rubric: Rubric,
        config: CriticConfig,
        store: CycleStore,
        tracker: Tracker,
    ) -> None:
        self._critics = tuple(critics)
        self._patch_writer = patch_writer
        self._rubric = rubric
        self._config = config
        self._store = store
        self._tracker = tracker

    # -- public API --------------------------------------------------------------------------------

    def next_cycle(self, subject: LoopSubject) -> int:
        """Cycle number the next session starts at (persist it to resume a session's budget)."""
        state = self._store.load(subject)
        return state.pending[0] if state.pending else state.next_cycle

    def best(self, subject: LoopSubject, since: int = 1) -> CycleResult | None:
        return self._store.load(subject).best(since)

    def result(self, subject: LoopSubject, cycle: int) -> CycleResult | None:
        return self._store.load(subject).results.get(cycle)

    def abandon_pending(self, subject: LoopSubject) -> None:
        self._store.abandon_pending(subject)

    def run(
        self,
        subject: LoopSubject,
        *,
        initial_script: Callable[[], str],
        feedback: str | None = None,
        session_start: int | None = None,
        base_cycle: int | None = None,
        fresh: bool = False,
        max_cycles: int | None = None,
        observer: LoopObserver | None = None,
    ) -> LoopOutcome:
        """Run one session of at most ``max_cycles`` evaluated cycles.

        ``feedback`` becomes the first, highest priority edit of the session. ``fresh`` starts the session
        from a newly written script (e.g. after a new Trellis mesh). Within a session the best result is
        chosen among the session's own cycles, so user feedback is never reverted to an older result.
        """
        state = self._store.load(subject)
        start = session_start or (state.pending[0] if state.pending else state.next_cycle)
        base = state.results.get(base_cycle) if base_cycle else state.best()
        session = _Session(start=start, base=None if fresh else base, fresh=fresh or base is None, observer=observer or _SilentObserver())
        budget = max_cycles or self._config.max_cycles
        trend = ScoreTrend.of(
            [state.results[c].score for c in sorted(state.results) if c >= start],
            self._config.stall_cycles,
            self._config.stall_epsilon,
        )
        started = state.pending is not None or state.session_cycles(start) > 0
        if session.fresh and not started:
            session.observer.building(state.next_cycle)
            with context.bind(agent="script_writer"):
                script = initial_script()
            previous = base.script_path.read_text(encoding="utf-8") if base else ""
            self._store.write_cycle(subject, state, state.next_cycle, script, make_diff(previous, script), score_before=None, initial=True)
            started = True
        pending_feedback = None if started else feedback
        while state.session_cycles(start) < budget:
            if state.pending is None:
                escalation = self._propose(subject, state, session, pending_feedback)
                if escalation is not None:
                    return self._outcome(state, session, *escalation)
                pending_feedback = None
            result = self._evaluate(subject, state, session, feedback)
            trend.add(result.score)
            if result.scorecard.passed:
                return self._outcome(state, session, LoopReason.PASSED, f"Passed the rubric at cycle {result.cycle:02d}.")
            if trend.stalled():
                message = f"Scores stalled for {self._config.stall_cycles} cycles (best {trend.best:.2f})."
                return self._outcome(state, session, LoopReason.STALLED, message)
        return self._outcome(state, session, LoopReason.MAX_CYCLES, f"Used all {budget} critic cycles.")

    # -- steps -------------------------------------------------------------------------------------

    def _propose(
        self, subject: LoopSubject, state: LoopState, session: _Session, feedback: str | None
    ) -> tuple[LoopReason, str] | None:
        """Write the next cycle's script by patching the session head, or say why the loop must stop."""
        head = session.head(state)
        assert head is not None
        session.observer.building(state.next_cycle)
        script = head.script_path.read_text(encoding="utf-8")
        edits = head.edits()
        if feedback:
            edits.insert(0, Edit(instruction=feedback, source="user", priority="high"))
        cycle = state.next_cycle
        rejection: str | None = None
        for _ in range(self._config.patch_attempts):
            request = PatchRequest(
                brief=subject.brief(),
                script=script,
                edits=tuple(edits),
                history=state.history.render(),
                api_reference=subject.api_reference(),
                error=head.evaluation.error,
                rejection=rejection,
                cycle=cycle,
            )
            try:
                diff = self._patch_writer.propose(request)
            except LLMAccessError:
                raise
            except LLMError as exc:
                rejection = exc.message
                continue
            repeat = None if feedback else state.history.find_repeat(diff)
            if repeat is not None:
                reason = f"repeats the diff of cycle {repeat.cycle:02d} ({repeat.status.value})"
                self._store.reject(subject, state, cycle, diff, reason, head.score)
                return LoopReason.REPEATED_DIFF, f"Stopped: the new diff {reason}."
            try:
                new_script = apply_diff(script, diff)
                check_python(new_script)
                if new_script.strip() == script.strip():
                    raise PatchError("The diff does not change the script")
            except (PatchError, LLMError) as exc:
                self._store.reject(subject, state, cycle, diff, exc.message, head.score)
                rejection = exc.message
                continue
            self._store.write_cycle(subject, state, cycle, new_script, diff, score_before=head.score)
            return None
        return LoopReason.PATCH_FAILED, f"No usable patch after {self._config.patch_attempts} attempts: {rejection}"

    def _evaluate(self, subject: LoopSubject, state: LoopState, session: _Session, feedback: str | None) -> CycleResult:
        assert state.pending is not None
        cycle, script_path = state.pending
        cycle_dir = script_path.parent
        first_of_session = state.session_cycles(session.start) == 0
        parent_score = state.history.score_before(cycle)
        with self._tracker.span(SpanKind.STEP, f"{subject.phase.value}.cycle", cycle=cycle) as span:
            session.observer.building(cycle)
            try:
                evaluation = subject.evaluate(script_path, cycle_dir, cycle)
            except KitbashError as exc:
                detail = exc.traceback_text if isinstance(exc, BlenderScriptError) else ""
                evaluation = Evaluation(ok=False, error=f"{exc}\n{detail}".strip())
            review = ReviewRequest(
                brief=subject.brief(),
                evaluation=evaluation,
                script=script_path.read_text(encoding="utf-8"),
                history=state.history.render(),
                cycle=cycle,
                feedback=feedback or "",
            )
            session.observer.critiquing(cycle)
            critiques = self._review(review)
            card = self._rubric.score(
                subject.phase,
                critiques,
                evaluation.facts,
                threshold=self._config.pass_threshold,
                require_all_pass=self._config.require_all_pass,
            )
            span.meta.update(score=round(card.overall, 4), passed=card.passed, ok=evaluation.ok)
        # A patch that makes its parent script worse is reverted; a session's first cycle (initial script or
        # the user's feedback) is its baseline and is always kept.
        worse = not first_of_session and parent_score is not None and card.overall < parent_score - self._config.revert_epsilon
        status = DiffStatus.REVERTED if worse else DiffStatus.KEPT
        result = CycleResult(cycle, script_path, card, critiques, evaluation, status)
        self._store.save_evaluation(subject, result)
        state.history.settle(cycle, status, card.overall)
        state.results[cycle] = result
        state.pending = None
        session.observer.evaluated(result)
        self._tracker.event(
            EventKind.CRITIC_CYCLE, subject.phase.value, cycle=cycle, subject=subject.subject_id,
            score=card.overall, passed=card.passed, status=status.value,
        )
        return result

    def _review(self, request: ReviewRequest) -> tuple[Critique, ...]:
        def run(critic: Critic) -> Critique:
            try:
                return critic.review(request)
            except LLMAccessError:
                raise
            except KitbashError as exc:
                self._tracker.event(EventKind.WARNING, "critic_failed", critic=critic.kind.value, error=str(exc)[:500])
                return Critique(critic=critic.kind.value, summary=f"Critic failed: {exc}")

        with ThreadPoolExecutor(max_workers=max(1, len(self._critics))) as pool:
            return tuple(pool.map(context.propagate(run), self._critics))

    def _outcome(self, state: LoopState, session: _Session, reason: LoopReason, message: str) -> LoopOutcome:
        best = session.best(state)
        assert best is not None
        return LoopOutcome(reason=reason, best=best, cycles_run=state.session_cycles(session.start), message=message)
