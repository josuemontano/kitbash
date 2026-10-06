"""The backlot: a global, reusable asset library with semantic search (SQLite + sqlite-vec)."""

import fcntl
import json
import shutil
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from attrs import field, frozen

from kitbash.domain.inventory import embedding_text
from kitbash.errors import BacklotError
from kitbash.infra.embeddings import Embedder
from kitbash.naming import slugify
from kitbash.store.database import Database
from kitbash.store.vectors import VectorIndex

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    rowid INTEGER PRIMARY KEY,
    id TEXT UNIQUE NOT NULL,
    slug TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    tags TEXT NOT NULL,
    category TEXT NOT NULL,
    dimensions TEXT NOT NULL,
    style TEXT NOT NULL,
    source_reference TEXT,
    blend_path TEXT NOT NULL,
    usd_path TEXT NOT NULL,
    preview_path TEXT NOT NULL,
    usd_material_mode TEXT NOT NULL,
    usd_roundtrip_score REAL,
    embedding BLOB,
    created_at REAL NOT NULL,
    version INTEGER NOT NULL,
    metadata TEXT NOT NULL,
    publication_status TEXT NOT NULL DEFAULT 'ready' CHECK(publication_status IN ('pending', 'ready'))
);
CREATE INDEX IF NOT EXISTS assets_slug ON assets(slug);
"""
VECTOR_TABLE = "asset_vec"


@frozen
class AssetBundle:
    """Files of a finished asset. ``root`` holds the .blend, its textures and the USD tree."""

    root: Path
    blend: Path
    usd: Path
    preview: Path


@frozen
class BacklotDraft:
    name: str
    description: str
    category: str
    dimensions: tuple[float, float, float]
    style: str
    usd_material_mode: str
    usd_roundtrip_score: float | None
    source_reference: str | None = None
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(factory=dict)


@frozen
class BacklotEntry:
    id: str
    name: str
    description: str
    tags: tuple[str, ...]
    category: str
    dimensions: tuple[float, float, float]
    style: str
    source_reference: str | None
    blend_path: Path
    usd_path: Path
    preview_path: Path
    usd_material_mode: str
    usd_roundtrip_score: float | None
    created_at: float
    version: int
    metadata: Mapping[str, Any]

    @property
    def directory(self) -> Path:
        return self.blend_path.parent

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "category": self.category,
            "dimensions": list(self.dimensions),
            "style": self.style,
            "source_reference": self.source_reference,
            "blend_path": str(self.blend_path),
            "usd_path": str(self.usd_path),
            "preview_path": str(self.preview_path),
            "usd_material_mode": self.usd_material_mode,
            "usd_roundtrip_score": self.usd_roundtrip_score,
            "created_at": self.created_at,
            "version": self.version,
            "metadata": dict(self.metadata),
        }


@frozen
class SearchHit:
    entry: BacklotEntry
    similarity: float


class Backlot:
    def __init__(self, root: Path, embedder: Embedder, *, rebuild_index: bool = False) -> None:
        self.root = root.expanduser()
        self.assets_dir = self.root / "assets"
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self._staging_dir = self.root / ".staging"
        self._staging_dir.mkdir(exist_ok=True)
        self._embedder = embedder
        self._db = Database(self.root / "backlot.db")
        try:
            self._db.executescript(SCHEMA)
            with self._db.transaction():
                columns = {row["name"] for row in self._db.query("PRAGMA table_info(assets)")}
                if "publication_status" not in columns:
                    self._db.execute("ALTER TABLE assets ADD COLUMN publication_status TEXT NOT NULL DEFAULT 'ready'")
            self._index = VectorIndex(self._db, VECTOR_TABLE, embedder.dimensions, embedder.name, rebuild=rebuild_index)
            with self._publication_lock(blocking=False) as acquired:
                if acquired:
                    self._recover()
        except BaseException:
            self._db.close()
            raise

    def close(self) -> None:
        self._db.close()

    def add(self, draft: BacklotDraft, bundle: AssetBundle) -> BacklotEntry:
        """Stage a complete bundle, commit its pending index entry, then publish by rename."""
        with self._publication_lock():
            self._recover()
            return self._add(draft, bundle)

    def _add(self, draft: BacklotDraft, bundle: AssetBundle) -> BacklotEntry:
        slug = slugify(draft.name)
        asset_id = f"{slug}-{uuid.uuid4().hex}"
        destination = self.assets_dir / asset_id
        staging = self._staging_dir / asset_id
        preview = destination / f"preview{bundle.preview.suffix}"
        row = {
            "id": asset_id,
            "slug": slug,
            "name": draft.name,
            "description": draft.description,
            "tags": json.dumps(list(draft.tags)),
            "category": draft.category,
            "dimensions": json.dumps(list(draft.dimensions)),
            "style": draft.style,
            "source_reference": draft.source_reference,
            "blend_path": str((destination / bundle.blend.relative_to(bundle.root)).relative_to(self.root)),
            "usd_path": str((destination / bundle.usd.relative_to(bundle.root)).relative_to(self.root)),
            "preview_path": str(preview.relative_to(self.root)),
            "usd_material_mode": draft.usd_material_mode,
            "usd_roundtrip_score": draft.usd_roundtrip_score,
            "created_at": time.time(),
            "metadata": json.dumps(dict(draft.metadata), default=str),
        }
        try:
            shutil.copytree(bundle.root, staging)
            shutil.copy2(bundle.preview, staging / preview.name)
            text = embedding_text(draft.name, draft.category, draft.description)
            vector = self._embedder.embed([f"{text} {' '.join(draft.tags)}".strip()])[0]
            with self._db.transaction():
                row["version"] = self._next_version(slug)
                row["embedding"] = json.dumps(vector)
                (staging / "metadata.json").write_text(json.dumps({**row, "embedding": None}, indent=2), encoding="utf-8")
                if not self._bundle_complete(staging, row):
                    raise BacklotError(f"Incomplete backlot bundle for {draft.name!r}")
                row["publication_status"] = "pending"
                cursor = self._db.execute(
                    f"INSERT INTO assets({', '.join(row)}) VALUES ({', '.join('?' for _ in row)})", tuple(row.values())
                )
                self._index.upsert(int(cursor.lastrowid), vector)
            staging.rename(destination)
            self._mark_ready(asset_id)
        except BaseException:
            # A committed pending entry owns its staging directory until reopen completes publication.
            if self._db.one("SELECT 1 FROM assets WHERE id = ?", (asset_id,)) is None:
                shutil.rmtree(staging, ignore_errors=True)
            raise
        return self.get(asset_id)

    def get(self, asset_id: str) -> BacklotEntry:
        row = self._db.one("SELECT * FROM assets WHERE id = ? AND publication_status = 'ready'", (asset_id,))
        if row is None:
            raise BacklotError(f"No backlot asset with id {asset_id!r}", hint="See `kitbash library list`.")
        return self._entry(row)

    def list(self) -> list[BacklotEntry]:
        return [self._entry(r) for r in self._db.query(
            "SELECT * FROM assets WHERE publication_status = 'ready' ORDER BY created_at DESC"
        )]

    def search(self, query: str, *, k: int = 5, style: str | None = None) -> list[SearchHit]:
        if not query.strip():
            return []
        vector = self._embedder.embed([query])[0]
        pending = self._db.one("SELECT COUNT(*) AS n FROM assets WHERE publication_status = 'pending'")["n"]
        neighbours = self._index.nearest(vector, (k * 4 if style else k) + pending)
        rows = {
            r["rowid"]: r
            for r in self._db.query(
                f"SELECT * FROM assets WHERE publication_status = 'ready' "
                f"AND rowid IN ({','.join('?' for _ in neighbours) or 'NULL'})",
                tuple(n.rowid for n in neighbours),
            )
        }
        hits = [SearchHit(self._entry(rows[n.rowid]), n.similarity) for n in neighbours if n.rowid in rows]
        if style:
            hits = [h for h in hits if h.entry.style == style]
        return hits[:k]

    def remove(self, asset_id: str) -> BacklotEntry:
        with self._publication_lock():
            entry = self.get(asset_id)
            with self._db.transaction():
                row = self._db.one("SELECT rowid FROM assets WHERE id = ?", (asset_id,))
                self._index.delete(int(row["rowid"]))
                self._db.execute("DELETE FROM assets WHERE id = ?", (asset_id,))
            shutil.rmtree(self.assets_dir / asset_id, ignore_errors=True)
            return entry

    def reindex(self) -> int:
        """Atomically re-embed every asset, retaining the previous model/index on failure."""
        with self._publication_lock():
            self._recover()
            rows = self._db.query("SELECT rowid, name, category, description, tags FROM assets")
            texts = [
                f"{embedding_text(r['name'], r['category'], r['description'])} {' '.join(json.loads(r['tags']))}"
                for r in rows
            ]
            vectors = self._embedder.embed(texts) if texts else []
            entries = [(int(row["rowid"]), vector) for row, vector in zip(rows, vectors, strict=True)]
            with self._db.transaction():
                for rowid, vector in entries:
                    self._db.execute("UPDATE assets SET embedding = ? WHERE rowid = ?", (json.dumps(vector), rowid))
                self._index.replace(entries)
            return len(rows)

    @contextmanager
    def _publication_lock(self, *, blocking: bool = True) -> Iterator[bool]:
        # OS-owned locks survive neither process death nor PID reuse. Never unlink this lock file.
        with (self.root / ".publication.lock").open("a+b") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            except BlockingIOError:
                yield False
                return
            try:
                yield True
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _mark_ready(self, asset_id: str) -> None:
        self._db.execute("UPDATE assets SET publication_status = 'ready' WHERE id = ?", (asset_id,))

    def _bundle_complete(self, directory: Path, row: Any) -> bool:
        try:
            metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
            if metadata["id"] != row["id"]:
                return False
            for key in ("blend_path", "usd_path", "preview_path"):
                relative = Path(row[key]).relative_to(Path("assets") / row["id"])
                if metadata[key] != row[key] or not (directory / relative).is_file():
                    return False
        except (OSError, ValueError, KeyError, TypeError):
            return False
        return True

    def _recover(self) -> None:
        """Reconcile only under the publication lock, so live writers' staging is untouched."""
        for row in self._db.query("SELECT * FROM assets WHERE publication_status = 'pending'"):
            destination = self.assets_dir / row["id"]
            staging = self._staging_dir / row["id"]
            if not destination.exists() and self._bundle_complete(staging, row):
                staging.rename(destination)
            if self._bundle_complete(destination, row):
                self._mark_ready(row["id"])
            else:
                with self._db.transaction():
                    self._index.delete(int(row["rowid"]))
                    self._db.execute("DELETE FROM assets WHERE id = ?", (row["id"],))
                shutil.rmtree(destination, ignore_errors=True)
        # Anything left here either predates the pending commit, or is redundant after publication.
        for staging in self._staging_dir.iterdir():
            if staging.is_dir():
                shutil.rmtree(staging)
            else:
                staging.unlink()

    def _next_version(self, slug: str) -> int:
        row = self._db.one("SELECT COALESCE(MAX(version), 0) + 1 AS v FROM assets WHERE slug = ?", (slug,))
        return int(row["v"])

    def _entry(self, row: Any) -> BacklotEntry:
        return BacklotEntry(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            tags=tuple(json.loads(row["tags"])),
            category=row["category"],
            dimensions=tuple(json.loads(row["dimensions"])),
            style=row["style"],
            source_reference=row["source_reference"],
            blend_path=self.root / row["blend_path"],
            usd_path=self.root / row["usd_path"],
            preview_path=self.root / row["preview_path"],
            usd_material_mode=row["usd_material_mode"],
            usd_roundtrip_score=row["usd_roundtrip_score"],
            created_at=row["created_at"],
            version=row["version"],
            metadata=json.loads(row["metadata"]),
        )


def best_match(hits: Sequence[SearchHit], threshold: float) -> SearchHit | None:
    return next((h for h in hits if h.similarity >= threshold), None)
