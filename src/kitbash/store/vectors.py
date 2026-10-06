"""k-nearest-neighbour index over a sqlite-vec ``vec0`` table (cosine distance)."""

import re
from collections.abc import Sequence

import sqlite_vec
from attrs import frozen

from kitbash.errors import StateError
from kitbash.store.database import Database

_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


@frozen
class Neighbour:
    rowid: int
    distance: float

    @property
    def similarity(self) -> float:
        """Cosine similarity in [0, 1] for normalized vectors (1 = identical)."""
        return max(0.0, 1.0 - self.distance)


class VectorIndex:
    def __init__(self, db: Database, table: str, dimensions: int, embedder_name: str, *, rebuild: bool = False) -> None:
        """``rebuild`` permits opening a mismatched index, without modifying it until ``replace``."""
        if not _IDENTIFIER.match(table):
            raise StateError(f"Invalid vector table name {table!r}")
        self._db = db
        self._table = table
        self.dimensions = dimensions
        self._embedder_name = embedder_name
        with self._db.transaction():
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS vector_meta (table_name TEXT PRIMARY KEY, dimensions INTEGER, embedder TEXT)"
            )
            row = self._db.one("SELECT dimensions, embedder FROM vector_meta WHERE table_name = ?", (self._table,))
            if row is None:
                self._create()
            elif not rebuild:
                self._check_identity()

    def upsert(self, rowid: int, vector: Sequence[float]) -> None:
        self._check(vector)
        with self._db.transaction() as conn:
            self._check_identity()
            conn.execute(f"DELETE FROM {self._table} WHERE rowid = ?", (rowid,))
            conn.execute(
                f"INSERT INTO {self._table}(rowid, embedding) VALUES (?, ?)", (rowid, sqlite_vec.serialize_float32(vector))
            )

    def delete(self, rowid: int) -> None:
        self._db.execute(f"DELETE FROM {self._table} WHERE rowid = ?", (rowid,))

    def clear(self) -> None:
        self._db.execute(f"DELETE FROM {self._table}")

    def nearest(self, vector: Sequence[float], k: int) -> list[Neighbour]:
        self._check(vector)
        with self._db.transaction():
            self._check_identity()
            rows = self._db.query(
                f"SELECT rowid, distance FROM {self._table} WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (sqlite_vec.serialize_float32(vector), k),
            )
        return [Neighbour(int(r["rowid"]), float(r["distance"])) for r in rows]

    def replace(self, entries: Sequence[tuple[int, Sequence[float]]]) -> None:
        """Atomically replace vectors and model identity, preserving the old index on failure."""
        with self._db.transaction() as conn:
            conn.execute(f"DROP TABLE IF EXISTS {self._table}")
            conn.execute("DELETE FROM vector_meta WHERE table_name = ?", (self._table,))
            self._create()
            for rowid, vector in entries:
                self._check(vector)
                conn.execute(
                    f"INSERT INTO {self._table}(rowid, embedding) VALUES (?, ?)",
                    (rowid, sqlite_vec.serialize_float32(vector)),
                )

    def _create(self) -> None:
        self._db.execute(
            f"CREATE VIRTUAL TABLE {self._table} "
            f"USING vec0(embedding float[{self.dimensions}] distance_metric=cosine)"
        )
        self._db.execute("INSERT INTO vector_meta VALUES (?, ?, ?)", (self._table, self.dimensions, self._embedder_name))

    def _check_identity(self) -> None:
        row = self._db.one("SELECT dimensions, embedder FROM vector_meta WHERE table_name = ?", (self._table,))
        if row is None or row["dimensions"] != self.dimensions or row["embedder"] != self._embedder_name:
            previous = f"{row['embedder']} ({row['dimensions']} dimensions)" if row else "an unknown model"
            hint = (
                "Switch back to the original embedding backend, or run `kitbash library reindex`."
                if self._table == "asset_vec" else
                "Switch back to the scene's original embedding backend, or start a new scene output directory."
            )
            raise StateError(
                f"{self._table} was indexed with {previous}, but the current embedder is "
                f"{self._embedder_name} ({self.dimensions} dimensions)",
                hint=hint,
            )

    def _check(self, vector: Sequence[float]) -> None:
        if len(vector) != self.dimensions:
            raise StateError(f"Vector has {len(vector)} dimensions, {self._table} expects {self.dimensions}")
