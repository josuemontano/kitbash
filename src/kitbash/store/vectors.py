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
        """``rebuild`` drops an index built with a different embedder instead of refusing to open it."""
        if not _IDENTIFIER.match(table):
            raise StateError(f"Invalid vector table name {table!r}")
        self._db = db
        self._table = table
        self.dimensions = dimensions
        if rebuild:
            self.reset(dimensions, embedder_name)
        else:
            self._ensure(dimensions, embedder_name)

    def upsert(self, rowid: int, vector: Sequence[float]) -> None:
        self._check(vector)
        with self._db.transaction() as conn:
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
        rows = self._db.query(
            f"SELECT rowid, distance FROM {self._table} WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (sqlite_vec.serialize_float32(vector), k),
        )
        return [Neighbour(int(r["rowid"]), float(r["distance"])) for r in rows]

    def reset(self, dimensions: int, embedder_name: str) -> None:
        """Drop and recreate the index for a different embedder."""
        self._db.execute(f"DROP TABLE IF EXISTS {self._table}")
        self._db.execute("DELETE FROM vector_meta WHERE table_name = ?", (self._table,))
        self._ensure(dimensions, embedder_name)

    def _ensure(self, dimensions: int, embedder_name: str) -> None:
        self.dimensions = dimensions
        self._db.executescript(
            "CREATE TABLE IF NOT EXISTS vector_meta (table_name TEXT PRIMARY KEY, dimensions INTEGER, embedder TEXT);"
        )
        row = self._db.one("SELECT dimensions, embedder FROM vector_meta WHERE table_name = ?", (self._table,))
        if row is None:
            self._db.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {self._table} "
                f"USING vec0(embedding float[{dimensions}] distance_metric=cosine)"
            )
            self._db.execute("INSERT INTO vector_meta VALUES (?, ?, ?)", (self._table, dimensions, embedder_name))
        elif row["dimensions"] != dimensions:
            raise StateError(
                f"{self._table} was indexed with {row['embedder']} ({row['dimensions']} dimensions) but the current "
                f"embedder produces {dimensions}",
                hint="Switch back to the original embedding backend, or run `kitbash library reindex`.",
            )

    def _check(self, vector: Sequence[float]) -> None:
        if len(vector) != self.dimensions:
            raise StateError(f"Vector has {len(vector)} dimensions, {self._table} expects {self.dimensions}")
