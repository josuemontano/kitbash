"""Publication boundaries, including recovery between directory renames and SQLite writes."""

import fcntl
import json
import shutil
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from kitbash.config import default_rubric_path, load_config
from kitbash.domain.rubric import Rubric
from kitbash.errors import StateError
from kitbash.infra.imaging import ImageComparison
from kitbash.paths import OutputLayout
from kitbash.phases.assembly import AssemblyPhase
from kitbash.services.usd_fidelity import UsdCheck
from kitbash.store.state import StateDB


class SceneWriter:
    def __init__(self):
        self.interrupt = False
        self.missing_usd = False
        self.score = 1.0
        self.content = b"first scene"

    def run_script(self, script, args, log):
        Path(args["output_blend"]).write_bytes(self.content)
        if self.interrupt:
            raise KeyboardInterrupt

    def localize(self, blend, logs):
        return {"missing": [], "copied": []}

    def inspect_scene(self, blend, assets, placeholders, logs, name):
        return {"camera": {"name": "Camera"}}, {
            "missing_assets": 0, "unexpected_assets": 0,
            "missing_placeholders": 0, "unexpected_placeholders": 0,
            "floating_assets": 0, "has_camera": True, "missing_textures": 0, "naming_violations": 0,
        }

    def render_scene(self, blend, destination, logs, **kwargs):
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"render")
        (destination.parent / "render.json").write_text(json.dumps({"images": [str(destination)]}))
        return destination

    def check(self, blend, usd, **kwargs):
        if not self.missing_usd:
            usd.write_bytes(b"usd")
        roundtrip = {
            "scene_facts": {
                "missing_assets": 0, "unexpected_assets": 0,
                "missing_placeholders": 0, "unexpected_placeholders": 0,
                "floating_assets": 0, "has_camera": True, "missing_textures": 0, "naming_violations": 0,
            }
        }
        return UsdCheck(usd, {}, roundtrip, (ImageComparison(self.score, 0.0),), None)


@pytest.fixture
def assembly(tmp_path):
    layout = OutputLayout.at(tmp_path)
    layout.create()
    state = StateDB(layout.state_db)
    writer = SceneWriter()
    subject = SimpleNamespace(run_args=lambda output, blends, mode: {"output_blend": str(output), "assets": {}})
    phase = AssemblyPhase(
        SimpleNamespace(subject=lambda *args: subject),
        SimpleNamespace(best=lambda subject: SimpleNamespace(script_path=tmp_path / "layout.py")),
        SimpleNamespace(load=lambda: (None, [], [])),
        writer, writer, Rubric.load(default_rubric_path()), state,
        SimpleNamespace(notify=lambda message: None, show_images=lambda paths: None),
        SimpleNamespace(showing=lambda view: nullcontext()),
        SimpleNamespace(span=lambda *args: nullcontext()), load_config(), layout,
    )
    yield phase, writer, state, layout
    state.close()


def test_interrupted_first_build_never_publishes(assembly):
    phase, writer, state, layout = assembly
    writer.interrupt = True
    with pytest.raises(KeyboardInterrupt):
        phase.run()
    assert not layout.scene_dir.exists()
    assert not (layout.root / ".scene-staging").exists()
    acceptance = (state.meta.get("assembly") or {}).get("acceptance", {})
    assert not acceptance.get("published")


def test_interrupted_rebuild_preserves_published_scene(assembly):
    phase, writer, state, layout = assembly
    phase.run()
    original = {path.relative_to(layout.scene_dir): path.read_bytes() for path in layout.scene_dir.rglob("*") if path.is_file()}
    writer.content = b"replacement scene"
    writer.interrupt = True
    with pytest.raises(KeyboardInterrupt):
        phase.run()
    assert {path.relative_to(layout.scene_dir): path.read_bytes() for path in layout.scene_dir.rglob("*") if path.is_file()} == original
    assert not (state.meta.get("assembly") or {}).get("acceptance", {}).get("published")
    assert not (layout.root / ".scene-staging").exists()

def test_replacement_drops_stale_files_and_relocates_reports(assembly):
    phase, writer, state, layout = assembly
    phase.run()
    (layout.scene_dir / "obsolete-texture.png").write_bytes(b"obsolete")
    writer.content = b"replacement scene"
    phase.run()
    assert layout.scene_blend.read_bytes() == b"replacement scene"
    assert not (layout.scene_dir / "obsolete-texture.png").exists()
    report = json.loads((layout.scene_dir / "assembly.json").read_text())
    assert Path(report["usd"]["usd_path"]) == layout.scene_usd
    assert Path(state.meta.get("assembly")["final_render"]).is_file()
    render_report = json.loads((layout.scene_renders_dir / "render.json").read_text())
    assert render_report["images"] == [str(layout.scene_renders_dir / "final.png")]
    assert not (layout.root / ".scene-previous").exists()


@pytest.mark.parametrize("renamed_new", [False, True])
def test_reopen_reconciles_interrupted_rename_and_database_update(assembly, renamed_new):
    phase, writer, state, layout = assembly
    phase.run()
    previous = layout.root / ".scene-previous"
    layout.scene_dir.rename(previous)
    if renamed_new:
        # A validated replacement was renamed, but SQLite still describes the old generation.
        shutil.copytree(previous, layout.scene_dir)
        layout.scene_blend.write_bytes(b"new validated scene")
        report_path = layout.scene_dir / "assembly.json"
        report = json.loads(report_path.read_text())
        report["usd"]["roundtrip_score"] = 0.91
        report_path.write_text(json.dumps(report))
    stale = layout.root / ".scene-staging"
    stale.mkdir()
    (stale / "partial.blend").write_bytes(b"partial")
    phase._recover_previous(previous)
    assert layout.scene_blend.read_bytes() == (b"new validated scene" if renamed_new else b"first scene")
    assert state.meta.get("assembly")["usd_roundtrip_score"] == (0.91 if renamed_new else 1.0)
    writer.interrupt = True
    with pytest.raises(KeyboardInterrupt):
        phase.run()
    assert layout.scene_blend.read_bytes() == (b"new validated scene" if renamed_new else b"first scene")
    assert not (state.meta.get("assembly") or {}).get("acceptance", {}).get("published")
    assert not previous.exists()
    assert not stale.exists()


def test_concurrent_publisher_cannot_remove_active_staging(assembly):
    phase, _writer, state, layout = assembly
    staged = layout.root / ".scene-staging"
    staged.mkdir()
    active = staged / "active.blend"
    active.write_bytes(b"active writer")
    with (layout.root / ".scene.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(StateError):
            phase.run()
        assert active.read_bytes() == b"active writer"
    assert state.meta.get("assembly") is None
