"""End-to-end runs with a fake omp and a fake Trellis, and real headless Blender."""

import json
import time

import pytest
from PIL import Image
from rich.console import Console
from typer.testing import CliRunner

from kitbash.app import Application, create_workspace
from kitbash.cli import app
from kitbash.domain.run_input import RunInput
from kitbash.infra.blender import BlenderRunner
from kitbash.interaction.autopilot import AutoPilot
from kitbash.interaction.protocols import AssetReview, ReviewDecision
from kitbash.store.state import StateDB
from tests.helpers import omp_calls, reference_image, requires_blender, write_test_config

pytestmark = [pytest.mark.integration, pytest.mark.blender, requires_blender]
runner = CliRunner()


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    log = tmp_path / "omp_calls.jsonl"
    monkeypatch.setenv("FAKE_OMP_LOG", str(log))
    # Reconstruction/reuse tests supply an explicit reference, independent of search heuristics.
    monkeypatch.setenv("FAKE_OMP_REFERENCE", str(reference_image(tmp_path / "asset.png")))
    return tmp_path, write_test_config(tmp_path), log


def build(config, image, output):
    result = runner.invoke(app, ["build", "--image", str(image), "--output", str(output), "--config", str(config), "--no-interactive"])
    assert result.exit_code == 0, result.output + (str(result.exception) if result.exception else "")
    return result


def test_full_pipeline_then_reuse_from_the_backlot(workspace):
    tmp, config, log = workspace
    output = tmp / "out"
    build(config, reference_image(tmp / "room.png"), output)

    # Deterministic output layout with every deliverable.
    for relative in (
        "config.snapshot.toml", "rubric.snapshot.md", "state.db", "input/reference.png",
        "phases/01_breakdown/inventory.json", "phases/01_breakdown/cycles/01/script.py",
        "phases/01_breakdown/cycles/01/diff.patch", "phases/01_breakdown/cycles/01/critique.json",
        "phases/02_modelling/wooden_crate/reference/reference.png", "phases/02_modelling/wooden_crate/cycles/01/script.py",
        "phases/02_modelling/wooden_crate/script.py",
        "phases/03_layout/script.py", "scene/scene.blend", "scene/scene.usd", "scene/renders/final.png",
        "analytics/analytics.json", "analytics/analytics.md", "logs/kitbash.log",
    ):
        assert (output / relative).exists(), relative
    assert list((output / "phases/02_modelling/wooden_crate/trellis").rglob("wooden_crate.obj"))
    assert list((output / "phases/02_modelling/wooden_crate/previews").glob("*.png"))
    assert not (output / "phases/02_modelling/wooden_crate/retopo").exists()  # retopology.method = "decimate": the old path
    build_args = json.loads((output / "phases/02_modelling/wooden_crate/cycles/01/build.args.json").read_text())["args"]
    assert build_args["retopology_method"] == "decimate" and "/trellis/" in build_args["mesh_path"]
    assert list((output / "phases/02_modelling/wooden_crate/usd_roundtrip").glob("*_report.json"))
    assert (output / "scene/assets/wooden_crate/asset.blend").is_file()

    analytics = json.loads((output / "analytics/analytics.json").read_text())
    assert analytics["run"]["versions"]["omp"] == "omp/0.0.0-fake" and analytics["run"]["versions"]["blender"].startswith("Blender")
    assert set(analytics["assets"]) == {"wooden_crate", "ceramic_mug"}
    for asset in analytics["assets"].values():
        assert asset["state"] == "approved" and asset["usd_roundtrip_score"] > 0.8
        assert asset["usd_material_mode"] in ("materialx", "preview_surface_baked") and asset["trellis_time_s"] > 0
    assert analytics["totals"]["llm_calls"] > 0 and analytics["totals"]["critic_cycles"] >= 4
    assert analytics["scene"]["usd_roundtrip_score"] > 0.8
    assembly = json.loads((output / "scene/assembly.json").read_text())
    assert assembly["scorecard"]["criteria"]["usd_material_fidelity"]["pass"] is True

    first_calls = {c["task"] for c in omp_calls(log)}
    assert {"breakdown.analyze.image", "modelling.script", "layout.script"} <= first_calls
    assert "modelling.reference.select" not in first_calls

    # A second run on a similar image reuses the approved assets instead of regenerating them.
    log.write_text("")
    second = tmp / "out2"
    build(config, reference_image(tmp / "room2.png", tint=12), second)
    tasks = [c["task"] for c in omp_calls(log)]
    assert "modelling.script" not in tasks and "modelling.reference.select" not in tasks
    state = StateDB(second / "state.db")
    assets = {a.id: a for a in state.assets.all()}
    assert all(a.reused and a.backlot_id for a in assets.values())
    state.close()
    assert not list((second / "phases/02_modelling").glob("*/trellis"))
    assert (second / "scene/scene.usd").is_file()


class SlowReviewer(AutoPilot):
    """Takes a while over each review, like a person would."""

    def review(self, review: AssetReview) -> ReviewDecision:
        time.sleep(3)
        return super().review(review)


def test_generation_continues_while_the_user_reviews(workspace, monkeypatch):
    tmp, config, _ = workspace
    monkeypatch.setenv("FAKE_TRELLIS_SLEEP_ceramic_mug", "6")  # the mug finishes well after the crate
    monkeypatch.setenv("FAKE_TRELLIS_FAIL_wooden_crate", "1")  # and the crate needs one Trellis retry
    settings, layout, run_input = create_workspace(tmp / "out", RunInput.create(reference_image(tmp / "room.png"), None), config, None, {})
    application = Application(settings, layout, run_input, interactive=False, console=Console(quiet=True))
    application.user = SlowReviewer()
    try:
        report = application.run()
    finally:
        application.close()
    reviews = {r["asset"]: r for r in report["concurrency"]["reviews"]}
    crate = reviews["wooden_crate"]
    assert crate["duration_s"] >= 3
    assert crate["other_assets_compute_s"] > 0 and crate["assets_in_progress"] == ["ceramic_mug"]
    assert report["assets"]["wooden_crate"]["trellis_retries"] == 1
    assert report["totals"]["user_time_s"] >= 6 and report["totals"]["review_queue_wait_s"] >= 0
    assert "U" in layout.analytics_md.read_text()


def test_resume_is_idempotent(workspace):
    tmp, config, log = workspace
    output = tmp / "out"
    build(config, reference_image(tmp / "room.png"), output)
    log.write_text("")
    result = runner.invoke(app, ["resume", "--output", str(output), "--no-interactive"])
    assert result.exit_code == 0, result.output
    work = [c for c in omp_calls(log) if c["task"] != "preflight.ping"]
    assert work == []  # every phase is done: nothing is recomputed (only the startup model pings run)
    result = runner.invoke(app, ["resume", "--output", str(output), "--no-interactive", "--from-phase", "layout"])
    assert result.exit_code == 0, result.output
    tasks = {c["task"] for c in omp_calls(log)}
    assert "modelling.script" not in tasks and "breakdown.analyze.image" not in tasks


def test_unresolved_references_leave_placeholders_without_mesh_generation(workspace, monkeypatch):
    tmp, config, log = workspace
    monkeypatch.delenv("FAKE_OMP_REFERENCE")
    output = tmp / "partial"
    build(config, reference_image(tmp / "small-room.png"), output)
    state = StateDB(output / "state.db")
    try:
        assets = state.assets.all()
        assert {asset.id: asset.state.value for asset in assets} == {"wooden_crate": "skipped", "ceramic_mug": "skipped"}
        assert all(asset.reference_path is None and asset.mesh_path is None for asset in assets)
    finally:
        state.close()
    assert not list((output / "phases/02_modelling").glob("*/trellis"))
    assert "modelling.script" not in {call["task"] for call in omp_calls(log)}
    assembly = json.loads((output / "scene/assembly.json").read_text())
    assert set(assembly["placeholders"]) == {"wooden_crate", "ceramic_mug"}
    assert assembly["assets"] == {}
    assert (output / "scene/scene.usd").is_file()


def test_procedural_prompt_build_critique_export_and_resume(workspace, monkeypatch):
    tmp, _, log = workspace
    monkeypatch.delenv("FAKE_OMP_REFERENCE")

    def unexpected(*args, **kwargs):
        pytest.fail("Procedural modelling must not construct reference or Trellis services")

    monkeypatch.setattr("kitbash.app.ReferenceFinder", unexpected)
    monkeypatch.setattr("kitbash.app.TrellisRunner", unexpected)
    config = write_test_config(
        tmp, modelling={"method": "procedural"},
        paths={"trellis": str(tmp / "no-trellis"), "triflow_weights": str(tmp / "no-weights")},
        tools={"trellis_python": str(tmp / "no-python")},
        retopology={"method": "triflow"},
        blender={"final_resolution": [160, 120], "final_samples": 3},
    )
    output = tmp / "procedural"
    result = runner.invoke(app, [
        "build", "--prompt", "A wooden crate with a ceramic mug on top in a studio.",
        "--output", str(output), "--config", str(config), "--no-interactive",
    ])
    assert result.exit_code == 0, result.output + str(result.exception or "")
    state = StateDB(output / "state.db")
    try:
        assets = state.assets.all()
        assert len(assets) == 2
        for asset in assets:
            assert asset.state.value == "approved" and asset.backlot_id
            assert asset.modelling_method == "procedural"
            assert asset.mesh_path is None and asset.reference_path is None
            assert not ({"reference", "trellis", "retopology"} & asset.extra.keys())
            transitions = [t["to_state"] for t in state.assets.transitions() if t["asset_id"] == asset.id]
            assert transitions[:2] == ["queued", "building"]
            assert "critiquing" in transitions and transitions[-1] == "approved"
            assert not ({"referencing", "generating", "skipped"} & set(transitions))
            cycle = output / "phases/02_modelling" / asset.id / "cycles/01"
            assert (cycle / "critique.json").is_file()
            assert (cycle / "build/asset.blend").is_file()
            assert (cycle / "build/usd/asset.usd").is_file()
    finally:
        state.close()
    calls = omp_calls(log)
    scripts = [call for call in calls if call["task"] == "modelling.script"]
    assert len(scripts) == 2 and all(not call["attachments"] for call in scripts)
    assert any(call["task"].startswith("modelling.critic.") for call in calls)
    assert not any(call["task"].startswith("modelling.reference") for call in calls)
    assembly = json.loads((output / "scene/assembly.json").read_text())
    assert not assembly["placeholders"] and len(assembly["assets"]) == 2
    for relative in ("scene/scene.blend", "scene/scene.usd", "scene/renders/final.png"):
        assert (output / relative).is_file()
    assert not list((output / "phases/02_modelling").glob("*/trellis"))
    assert not list((output / "phases/02_modelling").glob("*/retopo"))
    # Reopening the delivered .blend and pressing Render must use the same output contract,
    # not Blender's startup resolution or the generated script's preview sample count.
    render_script = tmp / "render_saved_scene.py"
    render_script.write_text(
        "import bpy\nimport kitbash_bpy as kb\n"
        "scene = bpy.context.scene\n"
        "kb.emit('settings', {'engine': scene.render.engine, 'samples': scene.cycles.samples})\n"
        "scene.render.filepath = kb.args()['render_path']\n"
        "bpy.ops.render.render(write_still=True)\n"
    )
    reopened_render = tmp / "reopened.png"
    saved = BlenderRunner("blender", timeout_s=120).run(
        render_script, args={"render_path": str(reopened_render)}, blend=output / "scene/scene.blend",
        log_path=tmp / "render_saved_scene.log",
    )
    assert saved["settings"] == {"engine": "CYCLES", "samples": 3}
    with Image.open(reopened_render) as image:
        assert image.size == (160, 120)
    preserved = {path: path.read_bytes() for path in (output / "scene/assets").rglob("asset.blend")}
    assert len(preserved) == 2
    log.write_text("")
    result = runner.invoke(app, ["resume", "--output", str(output), "--no-interactive"])
    assert result.exit_code == 0, result.output
    assert not [call for call in omp_calls(log) if call["task"] != "preflight.ping"]
    assert all(path.read_bytes() == content for path, content in preserved.items())
