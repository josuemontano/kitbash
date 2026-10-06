"""Publication boundaries, including recovery between directory renames and SQLite writes."""

import fcntl
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from kitbash.analytics.tracker import Tracker
from kitbash.config import default_rubric_path, load_config
from kitbash.domain.phases import PhaseName
from kitbash.domain.rubric import Rubric
from kitbash.errors import StateError
from kitbash.infra.imaging import ImageComparison
from kitbash.interaction.autopilot import AutoPilot
from kitbash.interaction.protocols import GateAction, GateDecision
from kitbash.paths import OutputLayout
from kitbash.phases.assembly import AssemblyPhase
from kitbash.services.usd_fidelity import UsdCheck
from kitbash.store.state import StateDB


class SceneWriter:
    def __init__(self):
        self.interrupt = False
        self.missing_usd = False
        self.empty_artifact = None
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
        if self.empty_artifact == "final.png":
            destination.write_bytes(b"")
            return destination
        destination.write_bytes(b"render")
        (destination.parent / "render.json").write_text(json.dumps({"images": [str(destination)]}))
        return destination

    def check(self, blend, usd, **kwargs):
        if not self.missing_usd:
            usd.write_bytes(b"" if self.empty_artifact == "scene.usd" else b"usd")
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
        AutoPilot(),
        SimpleNamespace(showing=lambda view: nullcontext()),
        Tracker(state.spans), load_config(), layout,
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
        staged = layout.root / ".scene-staging"
        staged.mkdir()
        writer.content = b"new validated scene"
        writer.score = 0.91
        report = phase._assemble(staged)
        phase._relocate_reports(staged, layout.scene_dir, report)
        staged.rename(layout.scene_dir)
    stale = layout.root / ".scene-staging"
    stale.mkdir()
    (stale / "partial.blend").write_bytes(b"partial")
    writer.interrupt = True
    with pytest.raises(KeyboardInterrupt):
        phase.run()
    assert layout.scene_blend.read_bytes() == (b"new validated scene" if renamed_new else b"first scene")
    assert state.meta.get("published_assembly")["usd_roundtrip_score"] == (0.91 if renamed_new else 1.0)
    assert state.meta.get("assembly")["acceptance"]["status"] == "pending"
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


def test_failed_validation_retains_rejected_outputs_without_replacing_scene(assembly):
    phase, writer, state, layout = assembly
    phase.run()
    original = {path.relative_to(layout.scene_dir): path.read_bytes() for path in layout.scene_dir.rglob("*") if path.is_file()}
    published = state.meta.get("published_assembly")
    writer.content = b"failed replacement"
    writer.score = 0.0
    with pytest.raises(StateError):
        phase.run()
    assert {path.relative_to(layout.scene_dir): path.read_bytes() for path in layout.scene_dir.rglob("*") if path.is_file()} == original
    assert state.meta.get("published_assembly") == published
    attempt = state.meta.get("assembly")
    assert attempt["acceptance"]["status"] == "failed" and not attempt["acceptance"]["published"]
    rejected = layout.phase_dir(PhaseName.ASSEMBLY) / "rejected"
    assert Path(attempt["scene_blend"]) == rejected / "scene.blend"
    assert Path(attempt["scene_blend"]).read_bytes() == b"failed replacement"
    assert json.loads((rejected / "renders" / "render.json").read_text())["images"] == [str(rejected / "renders" / "final.png")]
    assert not (layout.root / ".scene-staging").exists()


@pytest.mark.parametrize("artifact", ["scene.blend", "scene.usd", "final.png", "missing_usd"])
def test_incomplete_outputs_cannot_publish_even_by_human_override(assembly, artifact):
    class Override(AutoPilot):
        interactive = True

        def confirm(self, summary):
            pytest.fail("Missing deliverables cannot be overridden")

    phase, writer, state, layout = assembly
    phase.run()
    original = layout.scene_blend.read_bytes()
    writer.content = b"" if artifact == "scene.blend" else b"replacement scene"
    writer.empty_artifact = artifact
    writer.missing_usd = artifact == "missing_usd"
    phase._user = Override()
    with pytest.raises(StateError):
        phase.run()
    assert layout.scene_blend.read_bytes() == original
    assert state.meta.get("assembly")["acceptance"]["status"] == "failed"
    assert not state.meta.get("assembly")["acceptance"]["published"]


def test_rejected_first_build_does_not_create_public_scene(assembly):
    phase, writer, state, layout = assembly
    writer.score = 0.0
    with pytest.raises(StateError):
        phase.run()
    assert not layout.scene_dir.exists()
    assert state.meta.get("published_assembly") is None
    assert Path(state.meta.get("assembly")["scene_blend"]).is_file()


def test_explicit_degraded_rebuild_publishes_only_after_confirmation(assembly):
    phase, writer, state, layout = assembly
    phase.run()
    original = layout.scene_blend.read_bytes()

    class Override(AutoPilot):
        interactive = True

        def confirm(self, summary):
            assert layout.scene_blend.read_bytes() == original
            assert state.meta.get("assembly")["acceptance"]["status"] == "pending"
            return GateDecision(GateAction.PUBLISH_DEGRADED)

    writer.content = b"explicitly degraded replacement"
    writer.score = 0.0
    phase._user = Override()
    phase.run()
    assert layout.scene_blend.read_bytes() == writer.content
    assert state.meta.get("published_assembly")["acceptance"]["status"] == "overridden"
    assert not state.meta.get("published_assembly")["scorecard"]["passed"]


@pytest.mark.parametrize("prior_scene", [False, True])
def test_first_or_replacement_publish_recovers_after_sqlite_failure(assembly, monkeypatch, prior_scene):
    phase, writer, state, layout = assembly
    if prior_scene:
        phase.run()
    writer.content = b"validated replacement"
    record = phase._record

    def fail_after_rename(report):
        if layout.scene_blend.is_file() and layout.scene_blend.read_bytes() == writer.content:
            raise OSError("SQLite write unavailable")
        record(report)

    monkeypatch.setattr(phase, "_record", fail_after_rename)
    with pytest.raises(OSError):
        phase.run()
    assert layout.scene_blend.read_bytes() == b"validated replacement"
    monkeypatch.setattr(phase, "_record", record)
    writer.interrupt = True
    with pytest.raises(KeyboardInterrupt):
        phase.run()
    assert state.meta.get("published_assembly")["acceptance"]["published"]
    assert state.meta.get("published_assembly")["scene_blend"] == str(layout.scene_blend)
    assert state.meta.get("assembly")["acceptance"]["status"] == "pending"
    assert layout.scene_blend.read_bytes() == b"validated replacement"


def test_failure_between_directory_renames_restores_previous_scene(assembly, monkeypatch):
    phase, writer, state, layout = assembly
    phase.run()
    writer.content = b"replacement scene"
    rename = Path.rename

    def fail_publish(path, target):
        if path.name == ".scene-staging":
            raise OSError("rename interrupted")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_publish)
    with pytest.raises(OSError):
        phase.run()
    assert layout.scene_blend.read_bytes() == b"first scene"
    assert state.meta.get("assembly")["acceptance"]["status"] == "pending"
    assert not (layout.root / ".scene-previous").exists()


def damage_publication(scene, damage, foreign_owner):
    journal = scene / "assembly.json"
    if damage == "changed":
        (scene / "scene.blend").write_bytes(b"uninspected scene")
    elif damage == "missing":
        (scene / "scene.usd").unlink()
    elif damage == "added":
        (scene / "uninspected.txt").write_text("new output")
    elif damage == "symlink":
        (scene / "scene.usd").unlink()
        (scene / "scene.usd").symlink_to(scene / "scene.blend")
    elif damage == "missing_journal":
        journal.unlink()
    elif damage == "malformed_journal":
        journal.write_text("not a journal")
    else:
        report = json.loads(journal.read_text())
        if damage == "foreign_owner":
            report["integrity"]["owner"] = str(foreign_owner)
        elif damage == "legacy":
            report.pop("integrity")
        elif damage == "missing_manifest":
            report["integrity"].pop("manifest")
        elif damage == "rejected":
            report["acceptance"]["published"] = False
        elif damage == "invalid_override":
            report["acceptance"].update(status="overridden", automatic_pass=False, override="approve")
            report["scorecard"]["passed"] = False
        journal.write_text(json.dumps(report))


@pytest.mark.parametrize("damage", [
    "changed", "missing", "added", "symlink", "missing_journal", "malformed_journal",
    "foreign_owner", "legacy", "missing_manifest", "rejected", "invalid_override",
])
def test_invalid_interrupted_candidate_restores_intact_previous_scene(assembly, damage):
    phase, writer, state, layout = assembly
    phase.run()
    original = {path.relative_to(layout.scene_dir): path.read_bytes() for path in layout.scene_dir.rglob("*") if path.is_file()}
    published = state.meta.get("published_assembly")
    previous = layout.root / ".scene-previous"
    layout.scene_dir.rename(previous)
    staged = layout.root / ".scene-staging"
    staged.mkdir()
    writer.content = b"replacement scene"
    writer.score = 0.91
    report = phase._assemble(staged)
    phase._relocate_reports(staged, layout.scene_dir, report)
    staged.rename(layout.scene_dir)
    damage_publication(layout.scene_dir, damage, layout.root / "another-workspace")

    writer.interrupt = True
    with pytest.raises(KeyboardInterrupt):
        phase.run()

    assert {path.relative_to(layout.scene_dir): path.read_bytes() for path in layout.scene_dir.rglob("*") if path.is_file()} == original
    assert state.meta.get("published_assembly") == published
    assert not state.meta.get("assembly")["acceptance"]["published"]
    assert not previous.exists()
    assert not staged.exists()


@pytest.mark.parametrize("damage", ["changed", "missing", "foreign_owner", "legacy", "missing_manifest", "missing_journal"])
def test_unverified_existing_scene_is_preserved_but_not_reported_as_published(assembly, damage):
    phase, writer, state, layout = assembly
    phase.run()
    damage_publication(layout.scene_dir, damage, layout.root / "another-workspace")
    original = {path.relative_to(layout.scene_dir): path.read_bytes() for path in layout.scene_dir.rglob("*") if path.is_file()}
    writer.interrupt = True

    with pytest.raises(KeyboardInterrupt):
        phase.run()

    assert {path.relative_to(layout.scene_dir): path.read_bytes() for path in layout.scene_dir.rglob("*") if path.is_file()} == original
    assert state.meta.get("published_assembly") is None
    assert not state.meta.get("assembly")["acceptance"]["published"]
    assert not (layout.root / ".scene-previous").exists()
    assert not (layout.root / ".scene-staging").exists()


def test_legacy_previous_directory_survives_failed_replacement(assembly, monkeypatch):
    phase, writer, state, layout = assembly
    phase.run()
    damage_publication(layout.scene_dir, "legacy", layout.root)
    original = {path.relative_to(layout.scene_dir): path.read_bytes() for path in layout.scene_dir.rglob("*") if path.is_file()}
    rename = Path.rename

    def fail_publish(path, target):
        if path.name == ".scene-staging":
            raise OSError("rename interrupted")
        return rename(path, target)

    writer.content = b"replacement scene"
    monkeypatch.setattr(Path, "rename", fail_publish)
    with pytest.raises(OSError):
        phase.run()

    assert {path.relative_to(layout.scene_dir): path.read_bytes() for path in layout.scene_dir.rglob("*") if path.is_file()} == original
    assert state.meta.get("published_assembly") is None
    assert not state.meta.get("assembly")["acceptance"]["published"]
    assert not (layout.root / ".scene-previous").exists()


@pytest.mark.parametrize("damage", ["changed", "missing", "foreign_owner", "missing_manifest", "rejected"])
def test_change_after_report_relocation_cannot_replace_published_scene(assembly, monkeypatch, damage):
    phase, writer, state, layout = assembly
    phase.run()
    original = {path.relative_to(layout.scene_dir): path.read_bytes() for path in layout.scene_dir.rglob("*") if path.is_file()}
    published = state.meta.get("published_assembly")
    relocate = phase._relocate_reports

    def change_after_relocation(scene, destination, report):
        result = relocate(scene, destination, report)
        damage_publication(scene, damage, layout.root / "another-workspace")
        return result

    writer.content = b"replacement scene"
    monkeypatch.setattr(phase, "_relocate_reports", change_after_relocation)
    with pytest.raises(StateError):
        phase.run()

    assert {path.relative_to(layout.scene_dir): path.read_bytes() for path in layout.scene_dir.rglob("*") if path.is_file()} == original
    assert state.meta.get("published_assembly") == published
    assert not state.meta.get("assembly")["acceptance"]["published"]
    assert not (layout.root / ".scene-previous").exists()
    assert not (layout.root / ".scene-staging").exists()
