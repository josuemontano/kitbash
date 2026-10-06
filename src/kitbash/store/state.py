"""Scene state database (``<output>/state.db``): every checkpoint needed to resume a run."""

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from attrs import frozen

from kitbash.domain.assets import AssetRecord, AssetState, asset_from_dict, asset_to_dict
from kitbash.domain.inventory import Inventory, InventoryItem
from kitbash.domain.phases import PhaseName, PhaseStatus
from kitbash.infra.embeddings import Embedder
from kitbash.store.database import Database
from kitbash.store.vectors import VectorIndex

SCHEMA = """
CREATE TABLE IF NOT EXISTS run_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS phases (
    name TEXT PRIMARY KEY, status TEXT NOT NULL, started_at REAL, finished_at REAL
);
CREATE TABLE IF NOT EXISTS inventory_items (
    id TEXT PRIMARY KEY, ord INTEGER NOT NULL, name TEXT NOT NULL, description TEXT, category TEXT,
    confidence REAL, data TEXT NOT NULL, updated_at REAL
);
CREATE TABLE IF NOT EXISTS assets (
    id TEXT PRIMARY KEY, ord INTEGER NOT NULL, name TEXT NOT NULL, state TEXT NOT NULL, data TEXT NOT NULL,
    updated_at REAL
);
CREATE TABLE IF NOT EXISTS asset_transitions (
    id INTEGER PRIMARY KEY, asset_id TEXT NOT NULL, from_state TEXT, to_state TEXT NOT NULL, at REAL NOT NULL,
    note TEXT
);
CREATE TABLE IF NOT EXISTS cycles (
    phase TEXT NOT NULL, subject TEXT NOT NULL, cycle INTEGER NOT NULL, script_path TEXT NOT NULL,
    diff_path TEXT, critique_path TEXT, score REAL, passed INTEGER, status TEXT NOT NULL, summary TEXT,
    created_at REAL NOT NULL, PRIMARY KEY (phase, subject, cycle)
);
CREATE TABLE IF NOT EXISTS diffs (
    id INTEGER PRIMARY KEY, phase TEXT NOT NULL, subject TEXT NOT NULL, cycle INTEGER NOT NULL,
    status TEXT NOT NULL, fingerprint TEXT NOT NULL, diff_path TEXT NOT NULL, reason TEXT,
    score_before REAL, score_after REAL, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS spans (
    id INTEGER PRIMARY KEY, kind TEXT NOT NULL, name TEXT NOT NULL, phase TEXT, asset_id TEXT, agent TEXT,
    worker TEXT, started_at REAL NOT NULL, ended_at REAL NOT NULL, meta TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY, kind TEXT NOT NULL, name TEXT NOT NULL, phase TEXT, asset_id TEXT, at REAL NOT NULL,
    meta TEXT
);
CREATE INDEX IF NOT EXISTS spans_kind ON spans(kind);
CREATE INDEX IF NOT EXISTS transitions_asset ON asset_transitions(asset_id);
"""


def _dumps(value: Any) -> str:
    return json.dumps(value, default=str)


class RunMetaRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def get(self, key: str, default: Any = None) -> Any:
        row = self._db.one("SELECT value FROM run_meta WHERE key = ?", (key,))
        return json.loads(row["value"]) if row else default

    def set(self, key: str, value: Any) -> None:
        self._db.execute(
            "INSERT INTO run_meta VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, _dumps(value)),
        )

    def all(self) -> dict[str, Any]:
        return {row["key"]: json.loads(row["value"]) for row in self._db.query("SELECT key, value FROM run_meta")}


class PhaseRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def status(self, phase: PhaseName) -> PhaseStatus:
        row = self._db.one("SELECT status FROM phases WHERE name = ?", (phase.value,))
        return PhaseStatus(row["status"]) if row else PhaseStatus.PENDING

    def set_status(self, phase: PhaseName, status: PhaseStatus) -> None:
        now = time.time()
        self._db.execute(
            """INSERT INTO phases(name, status, started_at, finished_at) VALUES (?, ?, ?, NULL)
               ON CONFLICT(name) DO UPDATE SET status = excluded.status,
                 started_at = COALESCE(phases.started_at, excluded.started_at),
                 finished_at = CASE WHEN excluded.status = 'done' THEN ? ELSE NULL END""",
            (phase.value, status.value, now, now),
        )

    def reset(self, phases: Sequence[PhaseName]) -> None:
        for phase in phases:
            self._db.execute("UPDATE phases SET status = 'pending', finished_at = NULL WHERE name = ?", (phase.value,))

    def rows(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._db.query("SELECT * FROM phases")]


class InventoryRepository:
    """Inventory items, indexed by id/name/category and embedded for semantic search."""

    def __init__(self, db: Database, meta: RunMetaRepository, embedder: Embedder | None) -> None:
        self._db = db
        self._meta = meta
        self._embedder = embedder
        self._index = (
            VectorIndex(db, "inventory_vec", embedder.dimensions, embedder.name) if embedder is not None else None
        )

    def save(self, inventory: Inventory) -> None:
        now = time.time()
        with self._db.transaction() as conn:
            conn.execute("DELETE FROM inventory_items")
            for order, item in enumerate(inventory.items):
                conn.execute(
                    "INSERT INTO inventory_items VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (item.id, order, item.name, item.description, item.category, item.confidence,
                     _dumps(item.to_dict()), now),
                )
        self._meta.set("scene_info", inventory.scene.to_dict())
        self._reindex(inventory.items)

    def load(self) -> Inventory | None:
        rows = self._db.query("SELECT data FROM inventory_items ORDER BY ord")
        if not rows:
            return None
        return Inventory.from_dict({"scene": self._meta.get("scene_info", {}), "items": [json.loads(r["data"]) for r in rows]})

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        if self._index is None or self._embedder is None:
            return []
        neighbours = self._index.nearest(self._embedder.embed([query])[0], k)
        ids = {row["rowid"]: row["id"] for row in self._db.query("SELECT rowid, id FROM inventory_items")}
        return [(ids[n.rowid], n.similarity) for n in neighbours if n.rowid in ids]

    def _reindex(self, items: Sequence[InventoryItem]) -> None:
        if self._index is None or self._embedder is None:
            return
        self._index.clear()
        rowids = {row["id"]: row["rowid"] for row in self._db.query("SELECT rowid, id FROM inventory_items")}
        vectors = self._embedder.embed([item.embedding_text() for item in items])
        for item, vector in zip(items, vectors, strict=True):
            self._index.upsert(rowids[item.id], vector)


class AssetRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def upsert(self, record: AssetRecord, order: int | None = None) -> None:
        existing = self._db.one("SELECT ord FROM assets WHERE id = ?", (record.id,))
        ord_value = order if order is not None else (existing["ord"] if existing else self._next_order())
        self._db.execute(
            """INSERT INTO assets VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET name = excluded.name, state = excluded.state,
                 data = excluded.data, updated_at = excluded.updated_at, ord = excluded.ord""",
            (record.id, ord_value, record.name, record.state.value, _dumps(asset_to_dict(record)), time.time()),
        )

    def get(self, asset_id: str) -> AssetRecord | None:
        row = self._db.one("SELECT data FROM assets WHERE id = ?", (asset_id,))
        return asset_from_dict(json.loads(row["data"])) if row else None

    def all(self) -> list[AssetRecord]:
        return [asset_from_dict(json.loads(r["data"])) for r in self._db.query("SELECT data FROM assets ORDER BY ord")]

    def delete_missing(self, keep: Sequence[str]) -> None:
        placeholders = ",".join("?" for _ in keep) or "''"
        self._db.execute(f"DELETE FROM assets WHERE id NOT IN ({placeholders})", tuple(keep))

    def add_transition(self, asset_id: str, from_state: AssetState | None, to_state: AssetState, note: str = "") -> None:
        self._db.execute(
            "INSERT INTO asset_transitions(asset_id, from_state, to_state, at, note) VALUES (?, ?, ?, ?, ?)",
            (asset_id, from_state.value if from_state else None, to_state.value, time.time(), note),
        )

    def transitions(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._db.query("SELECT * FROM asset_transitions ORDER BY at, id")]

    def _next_order(self) -> int:
        row = self._db.one("SELECT COALESCE(MAX(ord), -1) + 1 AS n FROM assets")
        return int(row["n"])


@frozen
class CycleRow:
    phase: str
    subject: str
    cycle: int
    script_path: str
    diff_path: str | None
    critique_path: str | None
    score: float | None
    passed: bool | None
    status: str
    summary: str
    created_at: float


@frozen
class DiffRow:
    phase: str
    subject: str
    cycle: int
    status: str  # "applied", "kept", "reverted", "rejected"
    fingerprint: str
    diff_path: str
    reason: str
    score_before: float | None
    score_after: float | None


class CycleRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def save_cycle(self, row: CycleRow) -> None:
        self._db.execute(
            """INSERT INTO cycles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(phase, subject, cycle) DO UPDATE SET script_path = excluded.script_path,
                 diff_path = excluded.diff_path, critique_path = excluded.critique_path, score = excluded.score,
                 passed = excluded.passed, status = excluded.status, summary = excluded.summary""",
            (row.phase, row.subject, row.cycle, row.script_path, row.diff_path, row.critique_path, row.score,
             None if row.passed is None else int(row.passed), row.status, row.summary, row.created_at),
        )

    def cycles(self, phase: PhaseName, subject: str = "") -> list[CycleRow]:
        rows = self._db.query(
            "SELECT * FROM cycles WHERE phase = ? AND subject = ? ORDER BY cycle", (phase.value, subject)
        )
        return [
            CycleRow(**{**dict(r), "passed": None if r["passed"] is None else bool(r["passed"])}) for r in rows
        ]

    def all_cycles(self) -> list[CycleRow]:
        rows = self._db.query("SELECT * FROM cycles ORDER BY phase, subject, cycle")
        return [CycleRow(**{**dict(r), "passed": None if r["passed"] is None else bool(r["passed"])}) for r in rows]

    def save_diff(self, row: DiffRow) -> None:
        self._db.execute(
            """INSERT INTO diffs(phase, subject, cycle, status, fingerprint, diff_path, reason, score_before,
               score_after, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (row.phase, row.subject, row.cycle, row.status, row.fingerprint, row.diff_path, row.reason,
             row.score_before, row.score_after, time.time()),
        )

    def update_diff(self, phase: PhaseName, subject: str, cycle: int, *, status: str, score_after: float | None) -> None:
        self._db.execute(
            """UPDATE diffs SET status = ?, score_after = ? WHERE id = (
                 SELECT id FROM diffs WHERE phase = ? AND subject = ? AND cycle = ? AND status = 'applied'
                 ORDER BY id DESC LIMIT 1)""",
            (status, score_after, phase.value, subject, cycle),
        )

    def diffs(self, phase: PhaseName, subject: str = "") -> list[DiffRow]:
        rows = self._db.query(
            """SELECT phase, subject, cycle, status, fingerprint, diff_path, reason, score_before, score_after
               FROM diffs WHERE phase = ? AND subject = ? ORDER BY id""",
            (phase.value, subject),
        )
        return [DiffRow(**dict(r)) for r in rows]


class SpanRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def add_span(
        self, kind: str, name: str, started_at: float, ended_at: float, context: Mapping[str, Any], meta: Mapping[str, Any]
    ) -> None:
        self._db.execute(
            """INSERT INTO spans(kind, name, phase, asset_id, agent, worker, started_at, ended_at, meta)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (kind, name, context.get("phase"), context.get("asset_id"), context.get("agent"), context.get("worker"),
             started_at, ended_at, _dumps(dict(meta))),
        )

    def add_event(self, kind: str, name: str, context: Mapping[str, Any], meta: Mapping[str, Any]) -> None:
        self._db.execute(
            "INSERT INTO events(kind, name, phase, asset_id, at, meta) VALUES (?, ?, ?, ?, ?, ?)",
            (kind, name, context.get("phase"), context.get("asset_id"), time.time(), _dumps(dict(meta))),
        )

    def spans(self) -> list[dict[str, Any]]:
        return [_with_meta(r) for r in self._db.query("SELECT * FROM spans ORDER BY started_at, id")]

    def events(self) -> list[dict[str, Any]]:
        return [_with_meta(r) for r in self._db.query("SELECT * FROM events ORDER BY at, id")]


def _with_meta(row: Any) -> dict[str, Any]:
    data = dict(row)
    data["meta"] = json.loads(data.get("meta") or "{}")
    return data


class StateDB:
    """Facade over the scene state repositories."""

    def __init__(self, path: Path, embedder: Embedder | None = None) -> None:
        self.db = Database(path)
        try:
            self.db.executescript(SCHEMA)
            self.meta = RunMetaRepository(self.db)
            self.phases = PhaseRepository(self.db)
            self.inventory = InventoryRepository(self.db, self.meta, embedder)
            self.assets = AssetRepository(self.db)
            self.cycles = CycleRepository(self.db)
            self.spans = SpanRepository(self.db)
        except BaseException:
            self.db.close()
            raise

    def close(self) -> None:
        self.db.close()
