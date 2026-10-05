"""FIFO of assets waiting for the user, in completion order."""

import threading
import time
from collections import deque
from enum import StrEnum

from attrs import field, frozen


class EntryKind(StrEnum):
    REVIEW = "review"
    INPUT_NEEDED = "input_needed"


@frozen
class ReviewEntry:
    kind: EntryKind
    asset_id: str
    message: str = ""
    enqueued_at: float = field(factory=time.time)


class ReviewQueue:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._entries: deque[ReviewEntry] = deque()
        self.current: ReviewEntry | None = None

    def put(self, entry: ReviewEntry) -> None:
        with self._cond:
            self._entries.append(entry)
            self._cond.notify_all()

    def get(self, timeout: float) -> ReviewEntry | None:
        with self._cond:
            if not self._entries:
                self._cond.wait(timeout=timeout)
            if not self._entries:
                return None
            self.current = self._entries.popleft()
            return self.current

    def done(self) -> None:
        with self._cond:
            self.current = None

    def pending(self) -> list[ReviewEntry]:
        with self._cond:
            return list(self._entries)

    def __len__(self) -> int:
        with self._cond:
            return len(self._entries)
