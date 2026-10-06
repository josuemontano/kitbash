import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from kitbash.backlot.library import AssetBundle, Backlot, BacklotDraft, best_match
from kitbash.errors import BacklotError, StateError
from kitbash.infra.embeddings import HashingEmbedder
from kitbash.store.database import Database


class OtherModel(HashingEmbedder):
    @property
    def name(self):
        return "other-model"


def interrupted_add(root: Path, boundary: str) -> None:
    """Exit without unwinding Python/SQLite at a real publication boundary."""
    import kitbash.backlot.library as module

    library = Backlot(root / "backlot", HashingEmbedder(64))
    source = bundle(root, "source")
    if boundary == "copy":
        def copy_partial(src, destination):
            destination.mkdir()
            (destination / "asset.blend").write_bytes(b"partial")
            os._exit(73)
        module.shutil.copytree = copy_partial
    elif boundary == "index":
        upsert = library._index.upsert

        def uncommitted_vector(rowid, vector):
            upsert(rowid, vector)
            os._exit(73)
        library._index.upsert = uncommitted_vector
    elif boundary == "rename":
        Path.rename = lambda *args: os._exit(73)
    else:
        library._mark_ready = lambda asset_id: os._exit(73)
    library.add(draft("Oak chair", "Oak dining chair"), source)


def bundle(root: Path, name: str) -> AssetBundle:
    build = root / name
    (build / "textures").mkdir(parents=True)
    (build / "usd" / "textures").mkdir(parents=True)
    (build / "asset.blend").write_bytes(b"blend")
    (build / "textures" / "wood.png").write_bytes(b"png")
    (build / "usd" / "asset.usd").write_text("#usda 1.0")
    preview = root / f"{name}_preview.png"
    preview.write_bytes(b"png")
    return AssetBundle(root=build, blend=build / "asset.blend", usd=build / "usd" / "asset.usd", preview=preview)


def draft(name: str, description: str, category: str = "furniture", style: str = "photorealistic") -> BacklotDraft:
    return BacklotDraft(
        name=name, description=description, category=category, dimensions=(0.5, 0.5, 0.9), style=style,
        usd_material_mode="materialx", usd_roundtrip_score=0.93, tags=("wood",), metadata={"slug": name.lower().replace(" ", "_")},
    )


@pytest.fixture
def backlot(tmp_path, embedder):
    library = Backlot(tmp_path / "backlot", embedder)
    yield library
    library.close()


def test_add_copies_files_and_keeps_relative_layout(backlot, tmp_path):
    entry = backlot.add(draft("Oak chair", "Oak dining chair with a slatted back"), bundle(tmp_path, "chair"))
    assert entry.blend_path.is_file() and entry.usd_path.is_file() and entry.preview_path.is_file()
    assert (entry.directory / "textures" / "wood.png").is_file()
    assert entry.usd_path.relative_to(entry.directory) == Path("usd/asset.usd")
    assert entry.usd_material_mode == "materialx" and entry.usd_roundtrip_score == pytest.approx(0.93)
    assert (entry.directory / "metadata.json").is_file()


def test_search_ranks_similar_assets_first(backlot, tmp_path):
    backlot.add(draft("Oak chair", "Oak dining chair with a slatted back"), bundle(tmp_path, "chair"))
    backlot.add(draft("Ceramic mug", "White glazed coffee mug", "decor"), bundle(tmp_path, "mug"))
    backlot.add(draft("Floor lamp", "Tall brass floor lamp with a linen shade", "lighting"), bundle(tmp_path, "lamp"))
    hits = backlot.search("white coffee mug. decor", k=3)
    assert hits[0].entry.name == "Ceramic mug"
    assert hits[0].similarity > hits[-1].similarity
    assert best_match(hits, threshold=0.0).entry.name == "Ceramic mug"
    assert best_match(hits, threshold=1.01) is None


def test_style_filter_and_versions(backlot, tmp_path):
    first = backlot.add(draft("Oak chair", "Oak dining chair"), bundle(tmp_path, "a"))
    second = backlot.add(draft("Oak chair", "Oak dining chair", style="2d"), bundle(tmp_path, "b"))
    assert (first.version, second.version) == (1, 2)
    assert [h.entry.id for h in backlot.search("oak dining chair", k=5, style="2d")] == [second.id]


def test_get_list_remove(backlot, tmp_path):
    entry = backlot.add(draft("Oak chair", "Oak dining chair"), bundle(tmp_path, "chair"))
    assert backlot.get(entry.id).name == "Oak chair"
    assert [e.id for e in backlot.list()] == [entry.id]
    backlot.remove(entry.id)
    assert backlot.list() == [] and not entry.directory.exists()
    assert backlot.search("oak chair") == []
    with pytest.raises(BacklotError):
        backlot.get(entry.id)


def test_changing_embedder_requires_reindex(tmp_path):
    root = tmp_path / "backlot"
    library = Backlot(root, HashingEmbedder(64))
    library.add(draft("Oak chair", "Oak dining chair"), bundle(tmp_path, "chair"))
    library.close()
    with pytest.raises(StateError, match="reindex"):
        Backlot(root, HashingEmbedder(32))


def test_reindex_after_embedder_change(tmp_path):
    root = tmp_path / "backlot"
    library = Backlot(root, HashingEmbedder(64))
    library.add(draft("Oak chair", "Oak dining chair"), bundle(tmp_path, "chair"))
    library.close()
    library = Backlot(root, HashingEmbedder(32), rebuild_index=True)
    assert library.reindex() == 1
    assert library.search("oak chair")[0].entry.name == "Oak chair"
    library.close()


@pytest.mark.parametrize("boundary,published", [("copy", False), ("index", False), ("rename", True), ("ready", True)])
def test_reopen_recovers_process_interruption(tmp_path, boundary, published):
    process = multiprocessing.get_context("spawn").Process(target=interrupted_add, args=(tmp_path, boundary))
    process.start()
    process.join(timeout=15)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail("publication subprocess did not exit")
    assert process.exitcode == 73

    root = tmp_path / "backlot"
    # Only the post-rename interruption may have made a public directory, already complete.
    public = list((root / "assets").iterdir())
    assert bool(public) == (boundary == "ready")
    if public:
        assert (public[0] / "metadata.json").is_file()
        assert (public[0] / "asset.blend").read_bytes() == b"blend"

    for _ in range(2):
        library = Backlot(root, HashingEmbedder(64))
        try:
            assert [entry.name for entry in library.list()] == (["Oak chair"] if published else [])
            assert [hit.entry.name for hit in library.search("oak chair")] == (["Oak chair"] if published else [])
            assert list((root / ".staging").iterdir()) == []
            assert library._db.one("SELECT COUNT(*) AS n FROM assets")["n"] == int(published)
            assert library._db.one("SELECT COUNT(*) AS n FROM asset_vec")["n"] == int(published)
        finally:
            library.close()


def test_pending_bundle_is_hidden_until_reopen(backlot, tmp_path, monkeypatch):
    ready = backlot.add(draft("Ceramic mug", "White coffee mug", "decor"), bundle(tmp_path, "mug"))

    def interrupt(asset_id):
        raise OSError("ready update interrupted")

    monkeypatch.setattr(backlot, "_mark_ready", interrupt)
    with pytest.raises(OSError, match="interrupted"):
        backlot.add(draft("Oak chair", "Oak dining chair"), bundle(tmp_path, "chair"))
    pending = backlot._db.one("SELECT id FROM assets WHERE publication_status = 'pending'")["id"]
    assert [entry.id for entry in backlot.list()] == [ready.id]
    assert [hit.entry.id for hit in backlot.search("oak chair", k=1)] == [ready.id]
    with pytest.raises(BacklotError):
        backlot.get(pending)
    reopened = Backlot(backlot.root, HashingEmbedder(64))
    try:
        assert reopened.get(pending).name == "Oak chair"
        assert reopened.search("oak chair", k=1)[0].entry.id == pending
    finally:
        reopened.close()


def test_incomplete_pending_bundle_is_removed(backlot, tmp_path, monkeypatch):
    def interrupt(asset_id):
        raise OSError("ready update interrupted")

    monkeypatch.setattr(backlot, "_mark_ready", interrupt)
    with pytest.raises(OSError):
        backlot.add(draft("Oak chair", "Oak dining chair"), bundle(tmp_path, "chair"))
    public = next(backlot.assets_dir.iterdir())
    (public / "asset.blend").unlink()
    reopened = Backlot(backlot.root, HashingEmbedder(64))
    try:
        assert reopened.list() == []
        assert reopened.search("oak chair") == []
        assert not public.exists()
        assert reopened._db.one("SELECT COUNT(*) AS n FROM asset_vec")["n"] == 0
    finally:
        reopened.close()


def test_reopen_preserves_active_writer_staging(tmp_path):
    entered = Event()
    release = Event()

    class BlockingEmbedder(HashingEmbedder):
        def embed(self, texts):
            entered.set()
            if not release.wait(timeout=15):
                raise RuntimeError("writer was not released")
            return super().embed(texts)

    root = tmp_path / "backlot"
    writer = Backlot(root, BlockingEmbedder(64))
    source = bundle(tmp_path, "chair")
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(writer.add, draft("Oak chair", "Oak dining chair"), source)
            try:
                assert entered.wait(timeout=10)
                observer = Backlot(root, HashingEmbedder(64))
                try:
                    staging = next((root / ".staging").iterdir())
                    assert (staging / "asset.blend").read_bytes() == b"blend"
                    assert observer.list() == []
                    assert list(observer.assets_dir.iterdir()) == []
                finally:
                    observer.close()
            finally:
                release.set()
            entry = future.result(timeout=10)
        assert writer.get(entry.id).blend_path.read_bytes() == b"blend"
    finally:
        writer.close()


def test_vector_write_failure_rolls_back_asset_and_staging(backlot, tmp_path, monkeypatch):
    upsert = backlot._index.upsert

    def fail_after_insert(rowid, vector):
        upsert(rowid, vector)
        raise RuntimeError("vector write failed")

    monkeypatch.setattr(backlot._index, "upsert", fail_after_insert)
    with pytest.raises(RuntimeError, match="vector write failed"):
        backlot.add(draft("Oak chair", "Oak dining chair"), bundle(tmp_path, "chair"))
    assert backlot.list() == []
    assert backlot.search("oak chair") == []
    assert list(backlot.assets_dir.iterdir()) == []
    assert list((backlot.root / ".staging").iterdir()) == []
    assert backlot._db.one("SELECT COUNT(*) AS n FROM asset_vec")["n"] == 0


def test_migration_preserves_existing_published_assets(tmp_path):
    root = tmp_path / "backlot"
    library = Backlot(root, HashingEmbedder(64))
    entry = library.add(draft("Oak chair", "Oak dining chair"), bundle(tmp_path, "chair"))
    library.close()
    db = Database(root / "backlot.db")
    db.execute("ALTER TABLE assets DROP COLUMN publication_status")
    db.close()
    reopened = Backlot(root, HashingEmbedder(64))
    try:
        assert reopened.get(entry.id) == entry
        assert reopened.search("oak chair")[0].entry.id == entry.id
    finally:
        reopened.close()


def test_equal_dimensions_still_require_model_reindex(tmp_path):
    root = tmp_path / "backlot"
    library = Backlot(root, HashingEmbedder(64))
    entry = library.add(draft("Oak chair", "Oak dining chair"), bundle(tmp_path, "chair"))
    try:
        with pytest.raises(StateError, match="reindex"):
            Backlot(root, OtherModel(64))
        replacement = Backlot(root, OtherModel(64), rebuild_index=True)
        try:
            with pytest.raises(StateError, match="reindex"):
                replacement.search("oak chair")
            assert replacement.reindex() == 1
            assert replacement.search("oak chair")[0].entry.id == entry.id
            with pytest.raises(StateError, match="reindex"):
                library.search("oak chair")
        finally:
            replacement.close()
    finally:
        library.close()


@pytest.mark.parametrize("failure", ["embedding", "vector"])
def test_failed_reindex_preserves_previous_model_and_complete_index(tmp_path, failure):
    class BrokenModel(OtherModel):
        def embed(self, texts):
            if failure == "embedding":
                raise RuntimeError("embedding unavailable")
            vectors = super().embed(texts)
            vectors[-1] = [1.0]
            return vectors

    root = tmp_path / "backlot"
    library = Backlot(root, HashingEmbedder(64))
    chair = library.add(draft("Oak chair", "Oak dining chair"), bundle(tmp_path, "chair"))
    mug = library.add(draft("Ceramic mug", "White coffee mug", "decor"), bundle(tmp_path, "mug"))
    embeddings = [row["embedding"] for row in library._db.query("SELECT embedding FROM assets ORDER BY rowid")]
    library.close()
    replacement = Backlot(root, BrokenModel(64), rebuild_index=True)
    try:
        with pytest.raises((StateError, RuntimeError)):
            replacement.reindex()
    finally:
        replacement.close()
    with pytest.raises(StateError, match="reindex"):
        Backlot(root, OtherModel(64))
    reopened = Backlot(root, HashingEmbedder(64))
    try:
        assert reopened.search("oak chair", k=1)[0].entry.id == chair.id
        assert reopened.search("white coffee mug", k=1)[0].entry.id == mug.id
        assert [row["embedding"] for row in reopened._db.query("SELECT embedding FROM assets ORDER BY rowid")] == embeddings
    finally:
        reopened.close()


def test_reindex_fresh_library(tmp_path):
    library = Backlot(tmp_path / "backlot", OtherModel(64), rebuild_index=True)
    try:
        assert library.reindex() == 0
        assert library.search("oak chair") == []
        entry = library.add(draft("Oak chair", "Oak dining chair"), bundle(tmp_path, "chair"))
        assert library.search("oak chair")[0].entry.id == entry.id
    finally:
        library.close()
