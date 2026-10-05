"""End-to-end runs with a fake omp and a fake Trellis, and real headless Blender."""

import json
import time

import pytest
from rich.console import Console
from typer.testing import CliRunner

from kitbash.app import Application, create_workspace
from kitbash.cli import app
from kitbash.domain.run_input import RunInput
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
