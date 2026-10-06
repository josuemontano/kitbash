"""Producer threads: take assets from the scheduler and advance them until they need the user."""

import threading
import traceback
from collections.abc import Callable

from kitbash.analytics import context
from kitbash.analytics.tracker import EventKind, SpanKind, Tracker
from kitbash.errors import LLMAccessError
from kitbash.infra.process import ProcessCancelled, current_registry, defer_interrupts
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
        self._fatal_lock = threading.Lock()
        self._processes = current_registry()

    def start(self) -> None:
        with defer_interrupts():
            for index in range(self._size):
                if self._stop.is_set():
                    break
                thread = threading.Thread(
                    target=context.propagate(self._loop), name=f"worker-{index + 1}", args=(f"worker-{index + 1}",),
                )
                self._threads.append(thread)
                thread.start()

    def stop(self, *, cancel: bool = True) -> None:
        """Wake idle workers; cancel active work unless the phase has completed normally."""
        self._stop.set()
        self._scheduler.close()
        if cancel:
            self._processes.terminate_all()

    def join(self) -> None:
        """Do not release the services workers use until every worker has unwound."""
        for thread in self._threads:
            if thread.ident is not None:
                thread.join()

    def _loop(self, worker: str) -> None:
        try:
            with self._processes.bind(), context.bind(phase=self._phase, worker=worker):
                while (asset_id := self._scheduler.next(self._stop, worker)) is not None:
                    with context.bind(asset_id=asset_id):
                        try:
                            self._processes.check_cancelled()
                            outcome = self._run(asset_id)
                            self._processes.check_cancelled()
                            if outcome is not None and not self._stop.is_set():
                                self._deliver(outcome)
                        finally:
                            self._scheduler.finished(asset_id)
        except ProcessCancelled:
            pass
        except BaseException as exc:
            with self._fatal_lock:
                if self.fatal is None:
                    self.fatal = exc
            self.stop()

    def _run(self, asset_id: str) -> T | None:
        try:
            with self._tracker.span(SpanKind.ASSET_WORK, asset_id):
                return self._work(asset_id)
        except LLMAccessError:
            raise
        except Exception as exc:  # one broken asset must not stop the pool
            self._processes.check_cancelled()
            if self._stop.is_set():
                return None
            self._tracker.event(EventKind.WARNING, "worker_failure", asset=asset_id, error=traceback.format_exc()[-2000:])
            try:
                return self._on_failure(asset_id, exc)
            except LLMAccessError:
                raise
            except Exception:
                self._tracker.event(EventKind.WARNING, "failure_handler_failed", asset=asset_id, error=traceback.format_exc()[-2000:])
                return None
