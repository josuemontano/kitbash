"""The retopology step in a full run: a fake TriFlow engine, a fake Trellis and a fake omp, with real headless Blender."""

import json
import subprocess
from pathlib import Path

import pytest
from rich.console import Console

from kitbash.app import Application, create_workspace
from kitbash.domain.assets import AssetState
from kitbash.domain.phases import PhaseName
from kitbash.domain.run_input import RunInput
from kitbash.errors import PreflightError
from kitbash.retopology.base import RetopologyMethod
from kitbash.store.state import StateDB
from tests.fakes.retopology import FakeTriflowRetopologizer
from tests.helpers import omp_calls, reference_image, requires_blender, write_test_config

pytestmark = [pytest.mark.integration, pytest.mark.blender, requires_blender]


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    log = tmp_path / "omp_calls.jsonl"
    monkeypatch.setenv("FAKE_OMP_LOG", str(log))
    # Exercise retopology with an explicit reference, not the crop quality gate.
    monkeypatch.setenv("FAKE_OMP_REFERENCE", str(reference_image(tmp_path / "asset.png")))
    return tmp_path, log


def application(tmp, fake, **retopology):
    config = write_test_config(tmp, retopology={"method": "triflow", **retopology})
    settings, layout, run_input = create_workspace(tmp / "out", RunInput.create(reference_image(tmp / "room.png"), None), config, None, {})
    return Application(settings, layout, run_input, interactive=False, console=Console(quiet=True), retopology_factory=lambda cfg, rec: fake)


def run(app):
    try:
        return app.run()
    finally:
        app.close()


def assets_of(layout):
    state = StateDB(layout.state_db)
    try:
        return {a.id: a for a in state.assets.all()}
    finally:
        state.close()


def test_triflow_mesh_feeds_the_build_script(workspace):
    tmp, log = workspace
    fake = FakeTriflowRetopologizer(sleep_s=0.2)
    app = application(tmp, fake)
    report = run(app)
    assets = assets_of(app.layout)
    assert set(assets) == {"wooden_crate", "ceramic_mug"}
    for asset_id, asset in assets.items():
        assert asset.state is AssetState.APPROVED
        retopo_mesh = app.layout.asset_retopo_dir(asset_id) / "attempt_01" / f"{asset_id}.obj"
        assert retopo_mesh.is_file() and asset.mesh_path == str(retopo_mesh)
        assert asset.extra["retopology"]["method"] == "triflow" and asset.extra["retopology"]["fallback_reason"] is None
        assert asset.extra["trellis"]["runs"] == 1
        assert report["assets"][asset_id]["retopology_time_s"] >= 0.2 and report["assets"][asset_id]["trellis_time_s"] > 0
        script_args = json.loads((app.layout.cycle_dir(PhaseName.MODELLING, 1, asset_id) / "build.args.json").read_text())["args"]
        assert script_args["mesh_path"] == str(retopo_mesh) and script_args["retopology_method"] == "triflow"
        assert script_args["mesh_up_axis"] == "Z"
    assert report["totals"]["retopology_time_s"] >= 0.4 and "retopology" in {k.split(":")[1] for k in report["steps"]}
    analytics = json.loads(app.layout.analytics_json.read_text())
    assert analytics["totals"]["retopology_time_s"] == report["totals"]["retopology_time_s"]
    assert "retopology time (s)" in app.layout.analytics_md.read_text()
    scripts = [c for c in omp_calls(log) if c["task"] == "modelling.script"]
    assert scripts and len(fake.calls) == 2


def test_failed_retopology_falls_back_to_the_trellis_mesh(workspace):
    tmp, _ = workspace
    app = application(tmp, FakeTriflowRetopologizer(fail="out of memory", fail_for=("wooden_crate",)))
    run(app)
    assets = assets_of(app.layout)
    crate, mug = assets["wooden_crate"], assets["ceramic_mug"]
    assert crate.state is AssetState.APPROVED and "/trellis/" in crate.mesh_path and "/retopo/" not in crate.mesh_path
    assert crate.extra["retopology"]["method"] == "decimate" and crate.extra["retopology"]["fallback_reason"] == "triflow: out of memory"
    assert mug.state is AssetState.APPROVED and "/retopo/" in mug.mesh_path and mug.extra["retopology"]["method"] == "triflow"


def test_failed_retopology_without_fallback_fails_the_asset(workspace):
    tmp, _ = workspace
    app = application(tmp, FakeTriflowRetopologizer(fail="out of memory", fail_for=("wooden_crate",)), fallback_on_error=False)
    run(app)
    assets = assets_of(app.layout)
    assert assets["wooden_crate"].state is AssetState.SKIPPED and "out of memory" in assets["wooden_crate"].error
    assert assets["ceramic_mug"].state is AssetState.APPROVED


def test_preflight_collects_the_retopology_problem(workspace):
    tmp, _ = workspace
    fake = FakeTriflowRetopologizer(missing="TriFlow weights not found in /nowhere")
    app = application(tmp, fake)
    try:
        with pytest.raises(PreflightError, match="Preflight failed") as raised:
            app.run()
    finally:
        app.close()
    assert "TriFlow weights not found in /nowhere" in raised.value.message and "Download the TriFlow weights" in raised.value.hint
    assert fake.method is RetopologyMethod.TRIFLOW and fake.calls == []


def test_blender_preserves_triflow_topology_but_reduces_fallback_meshes(tmp_path):
    helpers = Path(__file__).resolve().parents[2] / "src/kitbash/blender"
    result = tmp_path / "topology.json"
    script = tmp_path / "topology.py"
    script.write_text(f'''
import json
import sys
import bpy
sys.path.insert(0, {str(helpers)!r})
import kitbash_bpy as kb

kb.reset_scene()
bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4)
obj = bpy.context.object
bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=1e-5, location=(2, 0, 0))
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
bpy.ops.object.join()

def geometry():
    return ([tuple(v.co) for v in obj.data.vertices],
            sorted(tuple(sorted(f.vertices)) for f in obj.data.polygons))

before = geometry()
kb._configure({{"retopology_method": "triflow", "max_faces": 100}})
kb.clean_mesh(obj, min_island_ratio=.03)
kb.decimate(obj)
kb.decimate(obj, max_faces=50)
preserved = geometry() == before
protected_faces = len(obj.data.polygons)
kb._configure({{"retopology_method": "decimate", "max_faces": 100}})
fallback_faces = kb.decimate(obj)
with open({str(result)!r}, "w") as handle:
    json.dump({{"preserved": preserved, "protected_faces": protected_faces,
               "original_faces": len(before[1]), "fallback_faces": fallback_faces}}, handle)
''')
    subprocess.run(
        ["blender", "-b", "--factory-startup", "--python-exit-code", "1", "--python", str(script)],
        check=True, capture_output=True, text=True, timeout=120,
    )
    observed = json.loads(result.read_text())
    assert observed["preserved"]
    assert observed["protected_faces"] == observed["original_faces"] > 100
    assert 0 < observed["fallback_faces"] <= 100
