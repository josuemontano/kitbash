"""The backlot: a global, reusable asset library with semantic search (SQLite + sqlite-vec)."""

import fcntl
import hashlib
import json
import shutil
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Any

from attrs import field, frozen

from kitbash.domain.inventory import embedding_text
from kitbash.errors import BacklotError
from kitbash.infra.embeddings import Embedder
from kitbash.naming import slugify
from kitbash.services.artifacts import snapshot
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
    publication_status TEXT NOT NULL DEFAULT 'ready' CHECK(publication_status IN ('pending', 'ready')),
    artifact_manifest TEXT
);
CREATE INDEX IF NOT EXISTS assets_slug ON assets(slug);
"""
VECTOR_TABLE = "asset_vec"


@frozen
class AssetBundle:
    """Evaluated asset bytes: the complete root tree and a separately hashed preview."""

    root: Path
    blend: Path
    usd: Path
    preview: Path
    hashes: Mapping[str, str] = field(converter=lambda hashes: MappingProxyType(dict(hashes)))
    preview_hash: str


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
                    self._db.execute("ALTER TABLE assets ADD COLUMN publication_status TEXT NOT NULL DEFAULT 'pending'")
                if "artifact_manifest" not in columns:
                    self._db.execute("ALTER TABLE assets ADD COLUMN artifact_manifest TEXT")
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
        blend_relative, usd_relative = self._check_source(bundle)
        slug = slugify(draft.name)
        asset_id = f"{slug}-{uuid.uuid4().hex}"
        destination = self.assets_dir / asset_id
        staging = self._staging_dir / asset_id
        preview = destination / f"preview{bundle.preview.suffix}"
        hashes = {**bundle.hashes, preview.name: bundle.preview_hash}
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
            "blend_path": str((destination / blend_relative).relative_to(self.root)),
            "usd_path": str((destination / usd_relative).relative_to(self.root)),
            "preview_path": str(preview.relative_to(self.root)),
            "usd_material_mode": draft.usd_material_mode,
            "usd_roundtrip_score": draft.usd_roundtrip_score,
            "created_at": time.time(),
            "metadata": json.dumps(dict(draft.metadata), default=str),
        }
        try:
            shutil.copytree(bundle.root, staging, symlinks=True)
            for name in ("metadata.json", preview.name):
                if (staging / name).exists() or (staging / name).is_symlink():
                    raise BacklotError(f"Staged asset collides with publisher-created {name!r}")
            shutil.copy2(bundle.preview, staging / preview.name, follow_symlinks=False)
            self._check_source(bundle)
            try:
                if snapshot(staging, (staging,)) != hashes:
                    raise ValueError("copied files do not match evaluated content")
            except (OSError, ValueError) as exc:
                raise BacklotError(f"Invalid staged backlot bundle for {draft.name!r}: {exc}") from exc
            text = embedding_text(draft.name, draft.category, draft.description)
            vector = self._embedder.embed([f"{text} {' '.join(draft.tags)}".strip()])[0]
            with self._db.transaction():
                row["version"] = self._next_version(slug)
                row["embedding"] = json.dumps(vector)
                metadata = json.dumps(self._metadata(row), indent=2)
                if (staging / "metadata.json").exists() or (staging / "metadata.json").is_symlink():
                    raise BacklotError("Staged asset collides with publisher-created 'metadata.json'")
                (staging / "metadata.json").write_text(metadata, encoding="utf-8")
                hashes["metadata.json"] = hashlib.sha256(metadata.encode("utf-8")).hexdigest()
                row["artifact_manifest"] = json.dumps({
                    "id": asset_id, "root": destination.relative_to(self.root).as_posix(), "hashes": hashes,
                })
                if not self._bundle_complete(staging, row):
                    raise BacklotError(f"Invalid staged backlot bundle for {draft.name!r}")
                row["publication_status"] = "pending"
                cursor = self._db.execute(
                    f"INSERT INTO assets({', '.join(row)}) VALUES ({', '.join('?' for _ in row)})", tuple(row.values())
                )
                self._index.upsert(int(cursor.lastrowid), vector)
            if not self._bundle_complete(staging, row):
                raise BacklotError(f"Changed pending backlot bundle for {draft.name!r}")
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
        with self._db.transaction(immediate=False):
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

    @staticmethod
    def _check_source(bundle: AssetBundle) -> tuple[Path, Path]:
        try:
            if any(".." in path.parts for path in (bundle.root, bundle.blend, bundle.usd, bundle.preview)):
                raise ValueError("bundle paths must not contain '..'")
            blend = bundle.blend.absolute().relative_to(bundle.root.absolute())
            usd = bundle.usd.absolute().relative_to(bundle.root.absolute())
            if snapshot(bundle.root, (bundle.root,)) != bundle.hashes:
                raise ValueError("asset tree does not match evaluated content")
            for relative in (blend, usd):
                if relative.as_posix() not in bundle.hashes or (bundle.root / relative).stat().st_size == 0:
                    raise ValueError("blend and USD paths must select nonempty evaluated files")
            if (
                snapshot(bundle.preview.parent, (bundle.preview,)) != {bundle.preview.name: bundle.preview_hash}
                or bundle.preview.stat().st_size == 0
            ):
                raise ValueError("preview does not match evaluated content")
            for name in ("metadata.json", f"preview{bundle.preview.suffix}"):
                if (bundle.root / name).exists() or (bundle.root / name).is_symlink():
                    raise ValueError(f"asset tree collides with publisher-created {name!r}")
        except (OSError, ValueError) as exc:
            raise BacklotError(f"Invalid or changed backlot bundle: {exc}") from exc
        return blend, usd

    @staticmethod
    def _metadata(row: Any) -> dict[str, Any]:
        # Publication state and digest evidence live only in SQLite; metadata cannot hash itself.
        return {
            **{key: value for key, value in dict(row).items() if key not in {
                "rowid", "embedding", "publication_status", "artifact_manifest",
            }},
            "embedding": None,
        }

    def _mark_ready(self, asset_id: str) -> None:
        row = self._db.one("SELECT * FROM assets WHERE id = ? AND publication_status = 'pending'", (asset_id,))
        if row is None or not self._bundle_complete(self.assets_dir / asset_id, row):
            raise BacklotError(f"Invalid pending backlot bundle {asset_id!r}")
        self._db.execute("UPDATE assets SET publication_status = 'ready' WHERE id = ?", (asset_id,))

    def _bundle_complete(self, directory: Path, row: Any) -> bool:
        try:
            asset_id = row["id"]
            if not asset_id or asset_id in {".", ".."} or Path(asset_id).name != asset_id:
                return False
            owner = Path("assets") / asset_id
            if directory not in (self.assets_dir / asset_id, self._staging_dir / asset_id):
                return False
            manifest = json.loads(row["artifact_manifest"])
            if manifest["id"] != asset_id or manifest["root"] != owner.as_posix():
                return False
            if snapshot(directory, (directory,)) != manifest["hashes"]:
                return False
            metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
            if metadata != self._metadata(row):
                return False
            for key in ("blend_path", "usd_path", "preview_path"):
                relative = Path(row[key]).relative_to(owner)
                artifact = directory / relative
                if (
                    ".." in relative.parts
                    or relative.as_posix() not in manifest["hashes"]
                    or artifact.stat().st_size == 0
                ):
                    return False
        except (OSError, ValueError, KeyError, TypeError):
            return False
        return True

    def _recover(self) -> None:
        """Reconcile only under the publication lock, so live writers' staging is untouched."""
        for row in self._db.query("SELECT * FROM assets WHERE publication_status = 'pending'"):
            asset_id = row["id"]
            valid_id = bool(asset_id) and asset_id not in {".", ".."} and Path(asset_id).name == asset_id
            if valid_id:
                destination = self.assets_dir / asset_id
                staging = self._staging_dir / asset_id
                if not destination.exists() and self._bundle_complete(staging, row):
                    staging.rename(destination)
                try:
                    self._mark_ready(asset_id)
                except BacklotError:
                    pass
                else:
                    continue
            with self._db.transaction():
                self._index.delete(int(row["rowid"]))
                self._db.execute("DELETE FROM assets WHERE id = ?", (asset_id,))
            if valid_id:
                self._remove_directory(destination)
        # Anything left here either predates the pending commit, or is redundant after publication.
        for staging in self._staging_dir.iterdir():
            self._remove_directory(staging)
        # Deletion may have committed just before process death interrupted directory cleanup.
        owned = {row["id"] for row in self._db.query("SELECT id FROM assets")}
        for destination in self.assets_dir.iterdir():
            if destination.name not in owned:
                self._remove_directory(destination)

    @staticmethod
    def _remove_directory(directory: Path) -> None:
        if directory.is_dir() and not directory.is_symlink():
            shutil.rmtree(directory)
        else:
            directory.unlink(missing_ok=True)

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
