"""Producer threads: take assets from the scheduler and advance them until they need the user."""

import threading
import traceback
from collections.abc import Callable

from kitbash.analytics import context
from kitbash.analytics.tracker import EventKind, SpanKind, Tracker
from kitbash.errors import LLMAccessError
from kitbash.pipeline.scheduler import Scheduler

type WorkFn[T] = Callable[[str], T]
type FailureFn[T] = Callable[[str, BaseException], T]
type DeliverFn[T] = Callable[[T], None]


class WorkerPool[T]:
    """``work`` advances an asset and returns its outcome, ``on_failure`` turns a crash into one, and
    ``deliver`` hands the outcome on (e.g. to the review queue) before the worker reports it is done."""

    def __init__(
        self,
        size: int,
        scheduler: Scheduler,
        work: WorkFn[T],
        on_failure: FailureFn[T],
        deliver: DeliverFn[T],
        tracker: Tracker,
        phase: str,
    ) -> None:
        self._size = size
        self._scheduler = scheduler
        self._work = work
        self._on_failure = on_failure
        self._deliver = deliver
        self._tracker = tracker
        self._phase = phase
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.fatal: BaseException | None = None  # an error that must stop the whole run

    def start(self) -> None:
        for index in range(self._size):
            thread = threading.Thread(target=self._loop, name=f"worker-{index + 1}", args=(f"worker-{index + 1}",), daemon=True)
            self._threads.append(thread)
            thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._scheduler.close()

    def join(self, timeout: float | None = None) -> None:
        for thread in self._threads:
            thread.join(timeout)

    def _loop(self, worker: str) -> None:
        with context.bind(phase=self._phase, worker=worker):
            while (asset_id := self._scheduler.next(self._stop, worker)) is not None:
                with context.bind(asset_id=asset_id):
                    try:
                        outcome = self._run(asset_id)
                        if outcome is not None:
                            self._deliver(outcome)
                    finally:
                        self._scheduler.finished(asset_id)

    def _run(self, asset_id: str) -> T | None:
        try:
            with self._tracker.span(SpanKind.ASSET_WORK, asset_id):
                return self._work(asset_id)
        except LLMAccessError as exc:
            self.fatal = exc
            self.stop()
            return None
        except Exception as exc:  # one broken asset must not stop the pool
            self._tracker.event(EventKind.WARNING, "worker_failure", asset=asset_id, error=traceback.format_exc()[-2000:])
            try:
                return self._on_failure(asset_id, exc)
            except Exception:
                self._tracker.event(EventKind.WARNING, "failure_handler_failed", asset=asset_id, error=traceback.format_exc()[-2000:])
                return None
