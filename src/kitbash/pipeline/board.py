"""Thread-safe, persisted view of every asset's state; the only place asset state changes."""

import threading
from collections import Counter
from collections.abc import Callable, Iterable
from typing import Any

from attrs import evolve

from kitbash.domain.assets import AssetRecord, AssetState, check_transition
from kitbash.errors import StateError
from kitbash.store.state import AssetRepository

type Listener = Callable[[AssetRecord], None]


class AssetBoard:
    def __init__(self, repository: AssetRepository) -> None:
        self._repository = repository
        self._lock = threading.RLock()
        self._records: dict[str, AssetRecord] = {r.id: r for r in repository.all()}
        self._listeners: list[Listener] = []

    def subscribe(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def ensure(self, records: Iterable[AssetRecord]) -> None:
        """Add assets that are not tracked yet (existing ones keep their checkpointed state)."""
        with self._lock:
            for order, record in enumerate(records):
                if record.id not in self._records:
                    self._records[record.id] = record
                    self._repository.upsert(record, order)
                    self._repository.add_transition(record.id, None, record.state, "created")

    def reset(self, record: AssetRecord, note: str = "reset") -> AssetRecord:
        """Replace an asset's record outright (e.g. the user changed a reuse decision)."""
        with self._lock:
            previous = self._records.get(record.id)
            self._store(record)
            self._repository.add_transition(record.id, previous.state if previous else None, record.state, note)
        self._notify(record)
        return record

    def retain(self, ids: Iterable[str]) -> None:
        """Forget assets whose inventory item no longer exists."""
        keep = set(ids)
        with self._lock:
            self._records = {k: v for k, v in self._records.items() if k in keep}
            self._repository.delete_missing(sorted(keep))

    def get(self, asset_id: str) -> AssetRecord:
        with self._lock:
            try:
                return self._records[asset_id]
            except KeyError:
                raise StateError(f"Unknown asset {asset_id!r}") from None

    def all(self) -> list[AssetRecord]:
        with self._lock:
            return list(self._records.values())

    def transition(self, asset_id: str, state: AssetState, note: str = "", **changes: Any) -> AssetRecord:
        with self._lock:
            current = self.get(asset_id)
            check_transition(current.state, state)
            updated = evolve(current, state=state, **changes)
            self._store(updated)
            self._repository.add_transition(asset_id, current.state, state, note)
        self._notify(updated)
        return updated

    def update(self, asset_id: str, **changes: Any) -> AssetRecord:
        """Change fields without a state transition; ``extra`` is merged, not replaced."""
        with self._lock:
            current = self.get(asset_id)
            if "extra" in changes:
                changes["extra"] = {**current.extra, **changes["extra"]}
            updated = evolve(current, **changes)
            self._store(updated)
        self._notify(updated)
        return updated

    def counts(self) -> Counter[AssetState]:
        with self._lock:
            return Counter(r.state for r in self._records.values())

    def all_resolved(self) -> bool:
        with self._lock:
            return all(r.state.is_terminal for r in self._records.values())

    def _store(self, record: AssetRecord) -> None:
        self._records[record.id] = record
        self._repository.upsert(record)

    def _notify(self, record: AssetRecord) -> None:
        for listener in self._listeners:
            listener(record)
