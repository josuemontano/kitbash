"""Resumable critic-loop sessions.

A session is one run of up to ``max_cycles`` cycles for one *request*: the first build, one round of user
feedback, or a fresh start after a new mesh. Callers name each request with a key. The same key resumes
(or returns) its session, so an interrupted run keeps its cycle budget and never applies the same feedback
twice; a new key always starts a new session, even when the previous one never finished.
"""

from collections.abc import Callable
from typing import Any

from attrs import asdict, evolve, frozen

from kitbash.critique.loop import CriticLoop, LoopObserver, LoopOutcome, LoopReason
from kitbash.critique.store import CycleResult
from kitbash.critique.subject import LoopSubject
from kitbash.errors import StateError
from kitbash.store.state import RunMetaRepository


@frozen
class LoopSession:
    start: int
    request: str = ""
    feedback: str | None = None
    base_cycle: int | None = None
    fresh: bool = False
    done: bool = False
    reason: str = ""
    message: str = ""


class LoopSessions:
    def __init__(self, meta: RunMetaRepository) -> None:
        self._meta = meta

    @staticmethod
    def _key(subject: LoopSubject) -> str:
        return f"loop_session:{subject.phase.value}:{subject.subject_id}"

    def get(self, subject: LoopSubject) -> LoopSession | None:
        data: dict[str, Any] | None = self._meta.get(self._key(subject))
        return LoopSession(**data) if data else None

    def save(self, subject: LoopSubject, session: LoopSession) -> LoopSession:
        self._meta.set(self._key(subject), asdict(session))
        return session


class ResumableLoop:
    def __init__(self, loop: CriticLoop, sessions: LoopSessions) -> None:
        self._loop = loop
        self._sessions = sessions

    def run(
        self,
        subject: LoopSubject,
        *,
        request: str,
        initial_script: Callable[[], str],
        feedback: str | None = None,
        fresh: bool = False,
        observer: LoopObserver | None = None,
    ) -> LoopOutcome:
        """Resume or return the session for ``request``, or start a new one for a new request."""
        session = self._sessions.get(subject)
        if session is not None and session.request == request:
            if session.done:
                best = self._best(subject, session)
                if best is None:
                    raise StateError("The completed critic session has no eligible result", hint="Check its required artifacts and cycle reports.")
                return LoopOutcome(reason=LoopReason(session.reason), best=best, cycles_run=0, message=session.message)
        else:
            if session is not None and not session.done:
                self._loop.abandon_pending(subject)
            previous = self._best(subject, session) if session else None
            session = self._sessions.save(
                subject,
                LoopSession(
                    start=self._loop.next_cycle(subject),
                    request=request,
                    feedback=feedback,
                    base_cycle=previous.cycle if previous else None,
                    fresh=fresh or previous is None,
                ),
            )
        outcome = self._loop.run(
            subject,
            initial_script=initial_script,
            feedback=session.feedback,
            session_start=session.start,
            base_cycle=session.base_cycle,
            fresh=session.fresh,
            observer=observer,
        )
        self._sessions.save(subject, evolve(session, done=True, reason=outcome.reason.value, message=outcome.message))
        return outcome

    def best(self, subject: LoopSubject) -> CycleResult | None:
        """The result of the latest session (what the user saw last)."""
        session = self._sessions.get(subject)
        return self._best(subject, session) if session else self._loop.best(subject)

    def _best(self, subject: LoopSubject, session: LoopSession) -> CycleResult | None:
        """A session's best cycle; a session without evaluated cycles still stands on its base."""
        return self._loop.best(subject, since=session.start, base_cycle=session.base_cycle)
