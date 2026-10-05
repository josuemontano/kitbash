"""Ambient trace context (phase, asset, agent, worker) carried through threads with contextvars."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from typing import Any

from attrs import asdict, evolve, frozen


@frozen
class TraceContext:
    phase: str | None = None
    asset_id: str | None = None
    agent: str | None = None
    worker: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_EMPTY = TraceContext()
_CURRENT: ContextVar[TraceContext] = ContextVar("kitbash_trace", default=_EMPTY)


def current() -> TraceContext:
    return _CURRENT.get()


@contextmanager
def bind(**values: str | None) -> Iterator[TraceContext]:
    token = _CURRENT.set(evolve(_CURRENT.get(), **values))
    try:
        yield _CURRENT.get()
    finally:
        _CURRENT.reset(token)


def propagate[T](fn: Callable[..., T]) -> Callable[..., T]:
    """Wrap ``fn`` so every call runs in its own copy of the caller's context (safe for parallel thread pools)."""
    context = copy_context()

    def run(*args: Any, **kwargs: Any) -> T:
        return context.copy().run(fn, *args, **kwargs)

    return run
