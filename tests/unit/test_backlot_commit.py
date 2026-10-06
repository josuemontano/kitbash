from pathlib import Path
from types import SimpleNamespace

import pytest
from attrs import evolve

from kitbash.analytics.tracker import Tracker
from kitbash.backlot.library import Backlot
from kitbash.config import load_config
from kitbash.critique.history import DiffStatus
from kitbash.critique.store import CycleResult, CycleStore
from kitbash.critique.subject import Evaluation
from kitbash.domain.assets import AssetRecord, AssetState
from kitbash.domain.critique import CardEntry, ScoreCard
from kitbash.domain.inventory import Inventory
from kitbash.domain.phases import PhaseName
from kitbash.errors import BacklotError
from kitbash.paths import OutputLayout
from kitbash.pipeline.commit import BacklotCommitter
from kitbash.store.state import StateDB


@pytest.fixture
def publication(tmp_path, embedder, sample_inventory_dict):
    layout = OutputLayout.at(tmp_path / "out")
    layout.create()
    state = StateDB(layout.state_db)
    library = Backlot(tmp_path / "backlot", embedder)
    item = Inventory.from_dict(sample_inventory_dict).items[0]
    asset = AssetRecord(item.id, item.name, state=AssetState.AWAITING_REVIEW, best_cycle=1)
    subject = SimpleNamespace(phase=PhaseName.MODELLING, subject_id=asset.id)
    store = CycleStore(state.cycles, layout)
    directory = store.cycle_dir(subject, 1)
    build = directory / "build"
    build.mkdir(parents=True)
    artifacts = {"build_dir": str(build)}
    for key, path in {
        "blend": build / "asset.blend", "usd": build / "asset.usd", "preview": directory / "preview.png",
    }.items():
        path.write_bytes(key.encode())
        artifacts[key] = str(path)
    script = directory / "script.py"
    script.write_text("# verified build")
    evaluation = Evaluation(ok=True, artifacts=artifacts)
    card = ScoreCard((CardEntry("geometry", "Geometry", 1, 1, True),), 1, True, 0.8)
    result = CycleResult(1, script, card, (), evaluation, DiffStatus.KEPT, subject.phase,
                         store.seal(subject, 1, script, evaluation))
    store.save_evaluation(subject, result)
    loop = SimpleNamespace(best=lambda subject: store.load(subject).best())
    agent = SimpleNamespace(subject=lambda item, asset: SimpleNamespace(phase=subject.phase, subject_id=asset.id))
    committer = BacklotCommitter(library, loop, agent, load_config(None), layout, Tracker(state.spans))
    yield SimpleNamespace(committer=committer, library=library, asset=asset, item=item, result=result, layout=layout)
    library.close()
    state.close()


@pytest.mark.parametrize("mismatch", ["state", "item", "reviewed_cycle", "workspace"])
def test_commit_requires_current_review_and_matching_owner(publication, mismatch):
    env = publication
    asset, item = env.asset, env.item
    if mismatch == "state":
        asset = evolve(asset, state=AssetState.BUILDING)
    elif mismatch == "item":
        item = evolve(item, id="different-item")
    elif mismatch == "reviewed_cycle":
        asset = evolve(asset, best_cycle=2)
    else:
        env.committer._layout = OutputLayout.at(env.layout.root.parent / "another-workspace")
    with pytest.raises(BacklotError):
        env.committer.commit(asset, item)
    assert env.library.list() == []
    assert list(env.library.assets_dir.iterdir()) == []


def test_commit_does_not_publish_mutated_reviewed_bytes(publication):
    env = publication
    assert env.committer.committable(env.committer.best(env.asset, env.item))
    Path(env.result.evaluation.artifacts["blend"]).write_bytes(b"changed after approval display")
    with pytest.raises(BacklotError):
        env.committer.commit(env.asset, env.item)
    assert env.library.list() == []


def test_commit_preserves_the_reviewed_content_and_identity(publication):
    env = publication
    entry = env.committer.commit(env.asset, env.item)
    assert entry.blend_path.read_bytes() == b"blend"
    assert entry.usd_path.read_bytes() == b"usd"
    assert entry.preview_path.read_bytes() == b"preview"
    assert entry.metadata["evidence"]["subject"] == env.asset.id
    assert entry.metadata["evidence"]["cycle"] == env.asset.best_cycle
