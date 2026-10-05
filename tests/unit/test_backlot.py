from pathlib import Path

import pytest

from kitbash.backlot.library import AssetBundle, Backlot, BacklotDraft, best_match
from kitbash.errors import BacklotError, StateError
from kitbash.infra.embeddings import HashingEmbedder


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
