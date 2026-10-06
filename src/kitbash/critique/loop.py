"""The automatic critic loop shared by breakdown, modelling (per asset) and layout.

Each cycle gathers evidence, evaluates bounded rubric decisions, and asks critics for edits only when
the result needs attention. The code role turns those edits into a unified diff against the latest kept
script. Everything is checkpointed under ``cycles/NN/`` and in ``state.db`` for resumption.
"""

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import StrEnum
from typing import Protocol

from attrs import define, evolve, frozen

from kitbash.analytics import context
from kitbash.analytics.tracker import EventKind, SpanKind, Tracker
from kitbash.config import CriticConfig
from kitbash.critique.critics import Critic, PatchRequest, PatchWriter, ReviewRequest
from kitbash.critique.evaluator import EvaluationCase, Evaluator
from kitbash.critique.history import DiffStatus, ScoreTrend
from kitbash.critique.store import CycleResult, CycleStore, LoopState
from kitbash.critique.subject import Evaluation, LoopSubject
from kitbash.domain.critique import Critique, Edit
from kitbash.domain.evaluation import CriterionAssessment, EvaluationResult
from kitbash.domain.rubric import Rubric
from kitbash.errors import BlenderScriptError, KitbashError, LLMAccessError, LLMError, PatchError, StateError
from kitbash.infra.patching import apply_diff, make_diff
from kitbash.infra.process import current_registry, defer_interrupts
from kitbash.llm.parsing import check_python
from kitbash.services.artifacts import file_hash


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

    def __attrs_post_init__(self) -> None:
        if not self.best.eligible or (self.reason is LoopReason.PASSED and not self.best.passed):
            raise StateError(f"Cycle {self.best.cycle:02d} is not eligible for a {self.reason.value} outcome")

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
        """The session's eligible result (what the loop hands back)."""
        return state.best(self.start, base_cycle=self.base.cycle if self.base else None)

    def head(self, state: LoopState) -> CycleResult | None:
        """The result the next patch builds on: the latest kept cycle of the session."""
        kept = [r for c, r in state.results.items() if c >= self.start and r.status is DiffStatus.KEPT]
        return max(kept, key=lambda r: r.cycle, default=None) or self.base


class CriticLoop:
    def __init__(
        self,
        *,
        critics: Sequence[Critic],
        evaluator: Evaluator,
        patch_writer: PatchWriter,
        rubric: Rubric,
        config: CriticConfig,
        store: CycleStore,
        tracker: Tracker,
    ) -> None:
        self._critics = tuple(critics)
        self._evaluator = evaluator
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

    def best(self, subject: LoopSubject, since: int = 1, *, base_cycle: int | None = None) -> CycleResult | None:
        return self._store.load(subject).best(since, base_cycle=base_cycle)

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
        # Evaluation is checkpointed before the session is marked done. A crash in between must
        # return the saved pass, even when it consumed the final cycle of the budget.
        saved = state.best(start)
        if saved is not None and saved.passed:
            return self._outcome(state, session, LoopReason.PASSED, f"Passed the rubric at cycle {saved.cycle:02d}.")
        trend = ScoreTrend.of([], self._config.stall_cycles, self._config.stall_epsilon)
        kept_score = session.base.score if session.base is not None else 0.0
        for cycle, prior in sorted(state.results.items()):
            if cycle >= start:
                if prior.status is DiffStatus.KEPT:
                    kept_score = prior.score
                trend.add(kept_score)
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
            head = session.head(state)
            trend.add(head.score if result.status is DiffStatus.REVERTED and head is not None else result.score)
            if result.passed:
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
                history=state.render_history(),
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
        parent = session.head(state)
        with self._tracker.span(SpanKind.STEP, f"{subject.phase.value}.cycle", cycle=cycle) as span:
            current_registry().check_cancelled()
            session.observer.building(cycle)
            script_digest = file_hash(script_path)
            try:
                evaluation = subject.evaluate(script_path, cycle_dir, cycle)
            except KitbashError as exc:
                detail = exc.traceback_text if isinstance(exc, BlenderScriptError) else ""
                evaluation = Evaluation(ok=False, error=f"{exc}\n{detail}".strip())
            evidence = self._store.seal(subject, cycle, script_path, evaluation)
            if evidence is not None and evidence.hashes.get(script_path.relative_to(evidence.root).as_posix()) != script_digest:
                evidence = None
            evaluation_id = f"{subject.phase.value}:{subject.subject_id}:{cycle}"
            case = EvaluationCase(
                evaluation_id=evaluation_id,
                iteration=cycle,
                brief=subject.brief(),
                script=script_path.read_text(encoding="utf-8"),
                history=state.render_history(),
                previous=state.previous(),
                feedback=feedback or "",
            )
            assessed = None
            diagnostic = None
            session.observer.critiquing(cycle)
            with context.bind(agent="clef_flash"):
                started_at = time.time()
                try:
                    assessed = self._evaluator.evaluate(case, self._rubric, evaluation)
                except KitbashError as exc:
                    # Provider failures must not become passes or leak provider payloads into analytics.
                    diagnostic = f"Evaluator unavailable ({type(exc).__name__})"
                    assessed = EvaluationResult(criteria=tuple(
                        CriterionAssessment(criterion.id, None, error=diagnostic)
                        for criterion in self._rubric.for_phase(subject.phase)
                    ))
                except BaseException as exc:
                    diagnostic = type(exc).__name__
                    raise
                finally:
                    self._tracker.record(
                        SpanKind.LLM if assessed is not None and assessed.model else SpanKind.STEP,
                        "clef_flash", started_at, time.time(), evaluation_id=evaluation_id, iteration=cycle,
                        model=assessed.model if assessed is not None else "",
                        resolved_model=assessed.model if assessed is not None else "",
                        tokens_in=assessed.input_tokens if assessed is not None else 0,
                        tokens_out=assessed.output_tokens if assessed is not None else 0,
                        cost_usd=None, error=diagnostic,
                    )
            # Explicit feedback/fresh inputs establish a baseline for changed goals, not a regression.
            # Compare every known dimension, even when another dimension of the parent is uncertain.
            baseline = first_of_session and (session.fresh or bool(feedback))
            previous = parent.scorecard if not baseline and parent is not None else None
            card = self._rubric.score(
                subject.phase,
                assessed.criteria,
                evaluation.facts,
                threshold=self._config.pass_threshold,
                require_all_pass=self._config.require_all_pass,
                confidence_threshold=self._config.confidence_threshold,
                previous=previous,
                regression_epsilon=self._config.revert_epsilon,
            )
            result = CycleResult(cycle, script_path, card, (), evaluation, DiffStatus.KEPT, subject.phase, evidence)
            escalation = []
            if not evaluation.ok or evaluation.error:
                escalation.append("evaluation_error")
            if evidence is None or not evidence.matches(result):
                escalation.append("invalid_evidence")
            if card.failing():
                escalation.append("failed_criteria")
            if card.unassessed():
                escalation.append("uncertain_criteria")
            if card.regressions():
                escalation.append("criterion_regression")
            if not card.passed and not escalation:
                escalation.append("below_threshold")
            if escalation:
                review = ReviewRequest(
                    brief=case.brief, evaluation=evaluation, script=case.script,
                    history=f"{case.history}\n\nCurrent escalation reasons: {', '.join(escalation)}",
                    cycle=cycle, scorecard=card, feedback=case.feedback,
                )
                result = evolve(result, critiques=self._review(review))
            current_registry().check_cancelled()
            # A single regressed dimension is sufficient: aggregate improvement cannot hide it.
            worse = previous is not None and parent is not None and parent.eligible and (
                not result.eligible or bool(card.regressions())
                or (not result.passed and result.score < parent.score - self._config.revert_epsilon)
            )
            if worse:
                result = evolve(result, status=DiffStatus.REVERTED)
            span.meta.update(score=card.overall, passed=result.passed, ok=evaluation.ok, escalation=escalation)
        self._store.save_evaluation(subject, result)
        state.history.settle(cycle, result.status, card.overall)
        state.results[cycle] = result
        state.pending = None
        session.observer.evaluated(result)
        self._tracker.event(
            EventKind.CRITIC_CYCLE, subject.phase.value, cycle=cycle, subject=subject.subject_id,
            evaluation_id=evaluation_id, iteration=cycle, model=assessed.model,
            score=card.overall, passed=result.passed, status=result.status.value, escalation=escalation,
            criteria={entry.criterion_id: {
                "score": entry.score, "raw_score": entry.raw_score, "confidence": entry.confidence,
                "probabilities": dict(entry.probabilities), "threshold": entry.threshold,
                "passed": entry.passed, "delta": entry.delta, "regressed": entry.regressed,
                "source": entry.decided_by,
            } for entry in card.entries},
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

        registry = current_registry()
        with registry.bind(), ThreadPoolExecutor(max_workers=max(1, len(self._critics))) as pool:
            try:
                registry.check_cancelled()
                futures = [pool.submit(context.propagate(run), critic) for critic in self._critics]
                # Observe a fatal result even when an earlier critic is still inside a subprocess.
                for future in as_completed(futures):
                    future.result()
                registry.check_cancelled()
                return tuple(future.result() for future in futures)
            except BaseException:
                with defer_interrupts():
                    registry.terminate_all()
                    pool.shutdown(wait=True, cancel_futures=True)
                raise

    def _outcome(self, state: LoopState, session: _Session, reason: LoopReason, message: str) -> LoopOutcome:
        best = session.best(state)
        if best is None:
            raise KitbashError(
                f"{message} No eligible critic result is available.",
                hint="A result needs successful evaluation, complete rubric scores, and all required artifacts. Check the cycle reports.",
            )
        return LoopOutcome(reason=reason, best=best, cycles_run=state.session_cycles(session.start), message=message)
