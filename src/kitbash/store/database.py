"""Thread-safe SQLite connection with sqlite-vec loaded."""

import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import sqlite_vec

from kitbash.errors import StateError


class Database:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=30)
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
        except (AttributeError, sqlite3.OperationalError) as exc:
            raise StateError(
                f"Could not load sqlite-vec into SQLite {sqlite3.sqlite_version}: {exc}",
                hint="Use a Python build whose sqlite3 module supports loadable extensions.",
            ) from exc
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Serialize writes; nested repositories participate through a savepoint."""
        with self._lock:
            nested = self._conn.in_transaction
            self._conn.execute("SAVEPOINT kitbash_nested" if nested else "BEGIN IMMEDIATE")
            try:
                yield self._conn
                self._conn.execute("RELEASE kitbash_nested" if nested else "COMMIT")
            except BaseException:
                if nested:
                    self._conn.execute("ROLLBACK TO kitbash_nested")
                    self._conn.execute("RELEASE kitbash_nested")
                elif self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def executescript(self, sql: str) -> None:
        with self._lock:
            self._conn.executescript(sql)

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()
