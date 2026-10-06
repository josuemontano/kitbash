"""The Kitbash flow: 1 reference image -> Trellis -> retopology -> USD/Blender model -> backlot, with real headless Blender."""

import json
import sqlite3

import pytest
from typer.testing import CliRunner

from kitbash.app import ModelApplication, create_model_workspace
from kitbash.backlot.library import Backlot
from kitbash.cli import app
from kitbash.errors import PreflightError, RetopologyError, TrellisError
from kitbash.infra.embeddings import HashingEmbedder
from kitbash.pipeline.image_to_model import ModelRequest
from tests.fakes.retopology import FakeTriflowRetopologizer
from tests.helpers import omp_calls, reference_image, requires_blender, write_test_config

pytestmark = [pytest.mark.integration, pytest.mark.blender, requires_blender]

runner = CliRunner()


@pytest.fixture
def omp_log(tmp_path, monkeypatch):
    log = tmp_path / "omp_calls.jsonl"
    monkeypatch.setenv("FAKE_OMP_LOG", str(log))
    return log


def application(tmp_path, fake, **sections):
    config = write_test_config(tmp_path, retopology={"method": "triflow"}, **sections)
    settings, layout = create_model_workspace(tmp_path / "out", config, {})
    return ModelApplication(settings, layout, retopology_factory=lambda cfg, recorder: fake)


def request(tmp_path, **changes):
    return ModelRequest(image=reference_image(tmp_path / "mug.png"), name="Ceramic mug", category="drinkware", height_m=0.1, **changes)


def backlot_entries(tmp_path):
    backlot = Backlot(tmp_path / "backlot", HashingEmbedder(64))
    try:
        return backlot.list()
    finally:
        backlot.close()


def test_one_image_becomes_a_backlot_asset_without_agents_or_critics(tmp_path, omp_log):
    fake = FakeTriflowRetopologizer()
    app_ = application(tmp_path, fake)
    try:
        result = app_.run(request(tmp_path))
    finally:
        app_.close()

    # Trellis ran, then retopology ran on its mesh, and the build imported the retopologized mesh.
    assert len(fake.calls) == 1 and fake.calls[0][0].parent.name == "attempt_01" and fake.calls[0][0].suffix == ".obj"
    assert result.mesh == tmp_path / "out/phases/02_modelling/ceramic_mug/retopo/attempt_01/ceramic_mug.obj" and result.mesh.is_file()
    args = json.loads((tmp_path / "out/phases/02_modelling/ceramic_mug/build.args.json").read_text())["args"]
    assert args["mesh_path"] == str(result.mesh) and args["retopology_method"] == "triflow"

    # The USD/Blender model exists and went into the backlot.
    assert result.blend.is_file() and result.usd.is_file() and result.preview.is_file()
    (entry,) = backlot_entries(tmp_path)
    assert entry.id == result.entry.id and entry.name == "Ceramic mug" and entry.blend_path.is_file() and entry.usd_path.is_file()
    assert entry.metadata["modelling_method"] == "trellis" and entry.metadata["flow"] == "image_to_model"
    assert entry.metadata["retopology"]["method"] == "triflow" and entry.metadata["reference"]["path"].endswith("reference.png")
    assert entry.dimensions[2] == pytest.approx(0.1, rel=0.01)

    # No critic loop, no scene phases and not a single LLM call.
    assert omp_calls(omp_log) == []
    with sqlite3.connect(tmp_path / "out" / "state.db") as db:
        assert db.execute("SELECT COUNT(*) FROM cycles").fetchone() == (0,)
        assert db.execute("SELECT COUNT(*) FROM spans WHERE kind = 'llm'").fetchone() == (0,)
    assert not (tmp_path / "out/scene").exists()
    assert set(result.timings_s) == {"trellis", "retopology", "build", "preview", "usd", "backlot"}
    assert json.loads(result.report.read_text())["backlot_id"] == entry.id


def test_failed_retopology_stops_the_flow_instead_of_using_the_raw_trellis_mesh(tmp_path, omp_log):
    app_ = application(tmp_path, FakeTriflowRetopologizer(fail="out of memory"))
    try:
        with pytest.raises(RetopologyError, match="out of memory"):
            app_.run(request(tmp_path))
    finally:
        app_.close()
    assert backlot_entries(tmp_path) == []
    assert not (tmp_path / "out/phases/02_modelling/ceramic_mug/build").exists()


def test_failed_trellis_stops_the_flow_instead_of_modelling_programmatically(tmp_path, omp_log, monkeypatch):
    monkeypatch.setenv("FAKE_TRELLIS_FAIL_ceramic_mug", "9")
    fake = FakeTriflowRetopologizer()
    app_ = application(tmp_path, fake, trellis={"retries": 0, "timeout_s": 60})
    try:
        with pytest.raises(TrellisError):
            app_.run(request(tmp_path))
    finally:
        app_.close()
    assert fake.calls == [] and backlot_entries(tmp_path) == [] and omp_calls(omp_log) == []


def test_the_flow_uses_trellis_even_when_the_config_selects_procedural_modelling(tmp_path, omp_log):
    fake = FakeTriflowRetopologizer()
    app_ = application(tmp_path, fake, modelling={"method": "procedural"})
    try:
        result = app_.run(request(tmp_path))
    finally:
        app_.close()
    assert len(fake.calls) == 1 and result.entry.metadata["modelling_method"] == "trellis"


def test_preflight_checks_retopology_before_any_trellis_run(tmp_path, omp_log):
    fake = FakeTriflowRetopologizer(missing="TriFlow weights not found in /nowhere")
    app_ = application(tmp_path, fake)
    try:
        with pytest.raises(PreflightError, match="TriFlow weights not found"):
            app_.run(request(tmp_path))
    finally:
        app_.close()
    assert fake.calls == []


def test_cli_runs_the_flow_end_to_end(tmp_path, omp_log):
    config = write_test_config(tmp_path)  # decimate retopology: no GPU needed
    image = reference_image(tmp_path / "ceramic_mug.png")
    result = runner.invoke(app, ["model", "--image", str(image), "--output", str(tmp_path / "out"), "--config", str(config), "--height", "0.1"])
    assert result.exit_code == 0, result.output
    assert "Added to the backlot as ceramic_mug-" in result.output
    (entry,) = backlot_entries(tmp_path)
    assert entry.name == "ceramic mug" and entry.metadata["retopology"]["method"] == "decimate"
    assert omp_calls(omp_log) == []


def test_cli_refuses_a_missing_image_and_an_existing_run(tmp_path):
    missing = runner.invoke(app, ["model", "--image", str(tmp_path / "nope.png"), "--output", str(tmp_path / "out")])
    assert missing.exit_code == 1 and "Input image not found" in missing.output
    config = write_test_config(tmp_path)
    image = reference_image(tmp_path / "a.png")
    create_model_workspace(tmp_path / "out", config, {})
    again = runner.invoke(app, ["model", "--image", str(image), "--output", str(tmp_path / "out"), "--config", str(config)])
    assert again.exit_code == 1 and "already contains kitbash run" in " ".join(again.output.split())
