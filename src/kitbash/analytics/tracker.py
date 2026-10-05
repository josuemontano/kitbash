"""Records timed spans and discrete events into the state database."""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from typing import Any

from attrs import define, field

from kitbash.analytics import context
from kitbash.store.state import SpanRepository


class SpanKind(StrEnum):
    PHASE = "phase"
    STEP = "step"
    LLM = "llm"
    SUBPROCESS = "subprocess"
    ASSET_WORK = "asset_work"
    REVIEW_WAIT = "review_wait"
    USER_REVIEW = "user_review"
    GATE = "gate"
    USER_INPUT = "user_input"
    IDLE_BACKPRESSURE = "idle_backpressure"
    IDLE = "idle"


class EventKind(StrEnum):
    USER_INTERVENTION = "user_intervention"
    RETRY = "retry"
    CRITIC_CYCLE = "critic_cycle"
    BACKLOT = "backlot"
    WARNING = "warning"


@define
class SpanHandle:
    kind: str
    name: str
    started_at: float
    meta: dict[str, Any] = field(factory=dict)


class Tracker:
    def __init__(self, repository: SpanRepository) -> None:
        self._repository = repository

    @contextmanager
    def span(self, kind: str, name: str, **meta: Any) -> Iterator[SpanHandle]:
        handle = SpanHandle(kind=kind, name=name, started_at=time.time(), meta=dict(meta))
        try:
            yield handle
        except BaseException as exc:
            handle.meta.setdefault("error", f"{type(exc).__name__}: {exc}"[:500])
            raise
        finally:
            self.record(kind, name, handle.started_at, time.time(), **handle.meta)

    def record(self, kind: str, name: str, started_at: float, ended_at: float, **meta: Any) -> None:
        self._repository.add_span(kind, name, started_at, ended_at, context.current().as_dict(), meta)

    def event(self, kind: str, name: str, **meta: Any) -> None:
        self._repository.add_event(kind, name, context.current().as_dict(), meta)
