"""Work scheduling for the modelling workers, with review-buffer backpressure.

Every asset that is in flight (being worked on) or waiting for the user's review holds one of
``capacity`` slots, so no more than ``capacity`` assets ever wait for review. When all slots are taken,
workers finish their current asset and then idle instead of starting a new one. Reworked assets keep
their slot and are served before new assets.
"""

import threading
import time
from collections import Counter, deque

from kitbash.analytics.tracker import SpanKind, Tracker


class Scheduler:
    def __init__(self, capacity: int, tracker: Tracker | None = None) -> None:
        self.capacity = capacity
        self._tracker = tracker
        self._cond = threading.Condition()
        self._new: deque[str] = deque()
        self._rework: deque[str] = deque()
        self._holding: set[str] = set()
        self._busy: Counter[str] = Counter()  # workers currently on each asset
        self._closed = False

    # -- producers of work -------------------------------------------------------------------------

    def submit(self, asset_id: str) -> None:
        """Queue an asset that needs a slot (new, resumed, or back from the user with a new input)."""
        with self._cond:
            if asset_id not in self._new:
                self._new.append(asset_id)
            self._cond.notify_all()

    def submit_rework(self, asset_id: str) -> None:
        """Queue an asset that already holds a slot (the user asked for changes)."""
        with self._cond:
            self._holding.add(asset_id)
            self._rework.append(asset_id)
            self._cond.notify_all()

    def hold(self, asset_id: str) -> None:
        """Count an asset that is waiting for review without being queued (resume)."""
        with self._cond:
            self._holding.add(asset_id)

    def release(self, asset_id: str) -> None:
        """The asset left the review buffer (approved, skipped, or waiting for user input)."""
        with self._cond:
            self._holding.discard(asset_id)
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    # -- workers -----------------------------------------------------------------------------------

    def next(self, stop: threading.Event, worker: str = "") -> str | None:
        """Block until there is work this worker may start; None when the scheduler is closed."""
        with self._cond:
            segment_start, segment_reason = time.time(), None
            try:
                while True:
                    if self._closed or stop.is_set():
                        return None
                    if self._rework:
                        return self._take(self._rework.popleft())
                    if self._new and len(self._holding) < self.capacity:
                        asset_id = self._new.popleft()
                        self._holding.add(asset_id)
                        return self._take(asset_id)
                    reason = SpanKind.IDLE_BACKPRESSURE if self._new else SpanKind.IDLE
                    if reason != segment_reason:
                        self._record(segment_reason, segment_start, worker)
                        segment_start, segment_reason = time.time(), reason
                    self._cond.wait(timeout=0.5)
            finally:
                self._record(segment_reason, segment_start, worker)

    def finished(self, asset_id: str) -> None:
        with self._cond:
            self._busy[asset_id] -= 1
            if self._busy[asset_id] <= 0:
                del self._busy[asset_id]
            self._cond.notify_all()

    # -- introspection -----------------------------------------------------------------------------

    def snapshot(self) -> dict[str, int]:
        with self._cond:
            return {
                "queued": len(self._new),
                "rework": len(self._rework),
                "holding": len(self._holding),
                "busy": sum(self._busy.values()),
                "capacity": self.capacity,
            }

    @property
    def backpressured(self) -> bool:
        with self._cond:
            return bool(self._new) and len(self._holding) >= self.capacity

    def idle(self) -> bool:
        """Nothing queued and nothing being worked on."""
        with self._cond:
            return not (self._new or self._rework or self._busy)

    def _take(self, asset_id: str) -> str:
        self._busy[asset_id] += 1
        return asset_id

    def _record(self, reason: SpanKind | None, started: float, worker: str) -> None:
        if reason is not None and self._tracker is not None:
            self._tracker.record(reason, "worker_idle", started, time.time(), worker=worker)
