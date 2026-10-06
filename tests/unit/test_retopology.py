import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kitbash.agents.modelling import ModellingAgent, retopology_method
from kitbash.analytics.report import AnalyticsReport
from kitbash.analytics.tracker import EventKind, SpanKind, Tracker
from kitbash.app import create_workspace, open_workspace
from kitbash.cli import Retopology, _overrides, app
from kitbash.config import load_config
from kitbash.domain.assets import AssetRecord, AssetState
from kitbash.domain.inventory import Inventory
from kitbash.domain.rubric import Rubric
from kitbash.domain.run_input import RunInput
from kitbash.errors import ConfigError, PreflightError, RetopologyError
from kitbash.paths import OutputLayout
from kitbash.pipeline.asset_pipeline import AssetPipeline
from kitbash.pipeline.board import AssetBoard
from kitbash.retopology import NullRecorder, make_retopologizer
from kitbash.retopology.base import RetopologyMethod, RetopologyResult
from kitbash.retopology.decimate import DecimateRetopologizer
from kitbash.services.plan import Planner
from kitbash.store.state import StateDB
from tests.fakes.retopology import FakeTriflowRetopologizer

runner = CliRunner()
S = AssetState


# -- config and CLI ------------------------------------------------------------------------------


def test_defaults_select_triflow():
    config = load_config()
    r = config.retopology
    assert r.method == "triflow" and r.method_enum is RetopologyMethod.TRIFLOW
    assert (r.face_count, r.qem_threshold, r.quad_ratio, r.flow_steps, r.device) == (4000, 12.0, 0.95, 50, "auto")
    assert r.fallback_on_error is True
    assert config.paths.triflow_weights == Path("~/.cache/kitbash/triflow").expanduser()


def test_user_file_and_override_set_the_method(tmp_path):
    user = tmp_path / "mine.toml"
    user.write_text('[retopology]\nmethod = "decimate"\nface_count = 2000\nfallback_on_error = false\n')
    config = load_config(user)
    assert config.retopology.method == "decimate" and config.retopology.face_count == 2000 and not config.retopology.fallback_on_error
    assert load_config(user, {"retopology.method": "triflow"}).retopology.method == "triflow"


@pytest.mark.parametrize(
    "text, message",
    [
        ('[retopology]\nmethod = "quadriflow"\n', "Unknown retopology method"),
        ('[retopology]\ndevice = "tpu"\n', "Unknown retopology.device"),
        ("[retopology]\nface_count = 0\n", "at least 1"),
        ("[retopology]\nquad_ratio = 1.5\n", "quad_ratio"),
        ("[retopology]\nfaces = 10\n", "Unknown config key"),
        ("[retopology]\nflow_steps = 'many'\n", "must be int"),
    ],
)
def test_invalid_retopology_config(tmp_path, text, message):
    path = tmp_path / "bad.toml"
    path.write_text(text)
    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_cli_flag_maps_to_the_config_key():
    ctx = types.SimpleNamespace(args=[])
    assert _overrides(ctx, retopology=Retopology.decimate, style=None) == {"retopology.method": "decimate"}
    assert _overrides(ctx, retopology=None) == {}


def test_dry_run_reports_the_method(tmp_path):
    base = ["build", "--output", str(tmp_path / "out"), "--prompt", "a chair", "--dry-run"]
    decimate = runner.invoke(app, [*base, "--retopology", "decimate"])
    assert decimate.exit_code == 0, decimate.output
    assert "per asset: retopology" in decimate.output and "triflow weights" not in decimate.output
    triflow = runner.invoke(app, [*base, "--retopology", "triflow"])
    assert triflow.exit_code == 0, triflow.output
    assert "triflow weights" in triflow.output
    assert runner.invoke(app, [*base, "--retopology", "quadriflow"]).exit_code != 0


def test_plan_reports_missing_weights_through_check(tmp_path):
    config = load_config(None, {"retopology.method": "triflow"})
    rubric = Rubric.load(Path(__file__).parents[2] / "src/kitbash/defaults/rubric.md")
    planner = Planner(
        config, OutputLayout.at(tmp_path), RunInput.create(None, "a chair"), rubric,
        lambda cfg, recorder: FakeTriflowRetopologizer(missing="TriFlow weights not found"),
    )
    (row,) = [p for p in planner.paths() if p[0] == "triflow weights"]
    assert row[2] == "NOT READY: TriFlow weights not found"
    ready = Planner(config, OutputLayout.at(tmp_path), RunInput.create(None, "a chair"), rubric, lambda cfg, recorder: FakeTriflowRetopologizer())
    assert [p[2] for p in ready.paths() if p[0] == "triflow weights"] == ["ok"]


def test_resume_keeps_the_method_from_the_snapshot(tmp_path):
    for method in ("decimate", "triflow"):
        output = tmp_path / method
        config, layout, _ = create_workspace(output, RunInput.create(None, "a chair"), None, None, {"retopology.method": method})
        assert config.retopology.method == method
        reopened, _, _ = open_workspace(output, {})
        assert reopened.retopology == config.retopology
        assert "[retopology]" in layout.config_snapshot.read_text()


def test_old_snapshot_without_a_retopology_table_still_resumes(tmp_path):
    output = tmp_path / "old"
    _, layout, _ = create_workspace(output, RunInput.create(None, "a chair"), None, None, {})
    text = layout.config_snapshot.read_text()
    head, _, rest = text.partition("[retopology]")
    layout.config_snapshot.write_text(head + rest[rest.index("[blender]"):])
    assert "[retopology]" not in layout.config_snapshot.read_text()
    config, _, _ = open_workspace(output, {})
    assert config.retopology.method == "triflow"


# -- factory -------------------------------------------------------------------------------------


def test_factory_returns_the_decimate_pass_through(tmp_path):
    sys.modules.pop("kitbash.retopology.triflow.engine", None)
    retopologizer = make_retopologizer(load_config(None, {"retopology.method": "decimate"}), NullRecorder())
    assert isinstance(retopologizer, DecimateRetopologizer) and retopologizer.method is RetopologyMethod.DECIMATE
    retopologizer.check()
    mesh = tmp_path / "a.obj"
    result = retopologizer.retopologize(mesh, tmp_path / "retopo", "a")
    assert result.mesh_path == mesh and result.faces_in is None and result.faces_out is None and result.device is None
    assert result.fallback_reason is None and "kitbash.retopology.triflow.engine" not in sys.modules


def test_factory_builds_the_engine_lazily_with_the_configured_settings(monkeypatch):
    captured = {}

    class Engine:
        method = RetopologyMethod.TRIFLOW

        def __init__(self, **kwargs):
            captured.update(kwargs)

    package = types.ModuleType("kitbash.retopology.triflow")
    package.__path__ = []
    engine = types.ModuleType("kitbash.retopology.triflow.engine")
    engine.TriflowRetopologizer = Engine
    monkeypatch.setitem(sys.modules, "kitbash.retopology.triflow", package)
    monkeypatch.setitem(sys.modules, "kitbash.retopology.triflow.engine", engine)
    config = load_config(None, {"retopology.face_count": 1234, "retopology.device": "mps"})
    recorder = NullRecorder()
    assert isinstance(make_retopologizer(config, recorder), Engine)
    assert captured == {
        "face_count": 1234, "qem_threshold": 12.0, "quad_ratio": 0.95, "flow_steps": 50, "device": "mps",
        "weights_dir": config.paths.triflow_weights, "recorder": recorder, "seed": config.trellis.seed,
    }


def test_missing_engine_is_a_preflight_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "kitbash.retopology.triflow.engine", None)
    with pytest.raises(PreflightError, match="TriFlow retopology is not available") as raised:
        make_retopologizer(load_config(), NullRecorder())
    assert "--retopology decimate" in raised.value.hint


# -- agent: fallback ---------------------------------------------------------------------------------


@pytest.fixture
def state(tmp_path):
    state = StateDB(tmp_path / "state.db")
    yield state
    state.close()


def make_agent(tmp_path, state, retopologizer, **overrides):
    config = load_config(None, overrides)
    agent = ModellingAgent(None, None, None, None, None, retopologizer, None, config, OutputLayout.at(tmp_path / "out"), Tracker(state.spans))
    return agent


def trellis_mesh(tmp_path):
    mesh = tmp_path / "trellis" / "chair.obj"
    mesh.parent.mkdir(parents=True)
    mesh.write_text("v 0 0 0\nf 1 1 1\nf 1 1 1\n")
    return mesh


def test_agent_uses_the_retopology_result_and_records_a_span(tmp_path, state):
    fake = FakeTriflowRetopologizer()
    agent = make_agent(tmp_path, state, fake)
    mesh = trellis_mesh(tmp_path)
    result = agent.retopologize(AssetRecord(id="chair", name="Chair", attempt=2), mesh)
    assert result.mesh_path == tmp_path / "out/phases/02_modelling/chair/retopo/attempt_02/chair.obj" and result.mesh_path.is_file()
    assert fake.calls == [(mesh, result.mesh_path.parent, "chair")]
    assert [(s["kind"], s["name"]) for s in state.spans.spans()] == [(SpanKind.SUBPROCESS, "retopology")]


def test_agent_falls_back_to_the_trellis_mesh(tmp_path, state):
    agent = make_agent(tmp_path, state, FakeTriflowRetopologizer(fail="CUDA out of memory"))
    mesh = trellis_mesh(tmp_path)
    result = agent.retopologize(AssetRecord(id="chair", name="Chair"), mesh)
    assert result.mesh_path == mesh and result.method is RetopologyMethod.DECIMATE
    assert result.fallback_reason == "triflow: CUDA out of memory"
    assert result.to_extra()["fallback_reason"] == "triflow: CUDA out of memory"
    (event,) = state.spans.events()
    assert event["kind"] == EventKind.WARNING and event["name"] == "retopology_fallback"
    (span,) = state.spans.spans()
    assert span["name"] == "retopology" and "CUDA out of memory" in span["meta"]["error"]


def test_agent_raises_without_fallback(tmp_path, state):
    agent = make_agent(tmp_path, state, FakeTriflowRetopologizer(fail="boom"), **{"retopology.fallback_on_error": False})
    with pytest.raises(RetopologyError, match="boom"):
        agent.retopologize(AssetRecord(id="chair", name="Chair"), trellis_mesh(tmp_path))


def test_decimate_agent_is_a_silent_pass_through(tmp_path, state):
    agent = make_agent(tmp_path, state, DecimateRetopologizer(), **{"retopology.method": "decimate"})
    mesh = trellis_mesh(tmp_path)
    result = agent.retopologize(AssetRecord(id="chair", name="Chair"), mesh)
    assert result.mesh_path == mesh and state.spans.spans() == []
    assert not (tmp_path / "out").exists()


def test_build_script_knows_the_method_that_produced_the_mesh():
    assert retopology_method(AssetRecord(id="a", name="A")) == "decimate"
    assert retopology_method(AssetRecord(id="a", name="A", extra={"retopology": {"method": "triflow"}})) == "triflow"


# -- pipeline ----------------------------------------------------------------------------------------------


class StubAgent:
    def __init__(self, tmp_path, retopologizer):
        self._tmp = tmp_path
        self._retopologizer = retopologizer
        self.retopology_calls = 0

    def generate_mesh(self, asset, reference):
        mesh = self._tmp / "trellis" / f"{asset.id}.obj"
        mesh.parent.mkdir(exist_ok=True)
        mesh.write_text("v 0 0 0\nf 1 1 1\n")
        return types.SimpleNamespace(mesh_path=mesh, duration_s=3.0, retries=1)

    def retopologize(self, asset, mesh_path):
        self.retopology_calls += 1
        return self._retopologizer(asset, mesh_path)


@pytest.fixture
def generating(tmp_path, state, sample_inventory_dict):
    board = AssetBoard(state.assets)
    board.ensure([AssetRecord(id="wooden_crate", name="Wooden crate", state=S.GENERATING, extra={"reference": {"source": "x"}})])
    inventory = Inventory.from_dict(sample_inventory_dict)

    def build(retopologizer):
        agent = StubAgent(tmp_path, retopologizer)
        return AssetPipeline(board, inventory, agent, None, Tracker(state.spans)), agent

    return board, build


def test_pipeline_builds_from_the_retopology_result(tmp_path, generating):
    board, build = generating
    result_mesh = tmp_path / "retopo" / "wooden_crate.obj"

    def retopologize(asset, mesh_path):
        return RetopologyResult(RetopologyMethod.TRIFLOW, result_mesh, 900, 400, 1.5, device="cpu")

    pipeline, agent = build(retopologize)
    asset = board.get("wooden_crate")
    after = pipeline._generate(asset, pipeline.item_for(asset))
    assert after.state is S.BUILDING and after.mesh_path == str(result_mesh) and agent.retopology_calls == 1
    assert after.extra["trellis"] == {"duration_s": 3.0, "retries": 1, "runs": 1}
    assert after.extra["reference"] == asset.extra["reference"]
    assert after.extra["retopology"] == {
        "method": "triflow", "faces_in": 900, "faces_out": 400, "duration_s": 1.5, "device": "cpu", "fallback_reason": None,
    }
    assert retopology_method(after) == "triflow"


def test_pipeline_keeps_the_trellis_mesh_after_a_fallback(tmp_path, generating):
    board, build = generating

    def retopologize(asset, mesh_path):
        return RetopologyResult(RetopologyMethod.DECIMATE, mesh_path, None, None, 0.2, fallback_reason="triflow: no GPU")

    pipeline, _ = build(retopologize)
    asset = board.get("wooden_crate")
    after = pipeline._generate(asset, pipeline.item_for(asset))
    assert after.state is S.BUILDING and after.mesh_path == str(tmp_path / "trellis" / "wooden_crate.obj")
    assert after.extra["retopology"]["fallback_reason"] == "triflow: no GPU" and retopology_method(after) == "decimate"
    assert after.extra["reference"] == asset.extra["reference"]


def test_pipeline_fails_the_asset_like_a_trellis_failure(generating):
    board, build = generating

    def retopologize(asset, mesh_path):
        raise RetopologyError("degenerate mesh")

    pipeline, _ = build(retopologize)
    asset = board.get("wooden_crate")
    after = pipeline._generate(asset, pipeline.item_for(asset))
    assert after.state is S.INPUT_NEEDED and "degenerate mesh" in after.error and "Retopology of the mesh for 'Wooden crate'" in after.input_request
    assert after.mesh_path is None and after.extra["trellis"]["runs"] == 1 and after.extra["reference"] == {"source": "x"}


# -- analytics -----------------------------------------------------------------------------------------------


def test_analytics_reports_retopology_next_to_trellis(tmp_path, state):
    from kitbash.analytics import context

    tracker = Tracker(state.spans)
    with context.bind(phase="modelling", asset_id="a"):
        tracker.record(SpanKind.SUBPROCESS, "trellis", 1000.0, 1030.0, attempt=1)
        tracker.record(SpanKind.SUBPROCESS, "retopology", 1030.0, 1042.0, attempt=1, method="triflow")
    state.assets.upsert(AssetRecord(id="a", name="A", extra={"retopology": {"method": "triflow"}}))
    report = AnalyticsReport(state, OutputLayout.at(tmp_path / "out")).write()
    assert report["totals"]["trellis_time_s"] == 30.0 and report["totals"]["retopology_time_s"] == 12.0
    assert report["assets"]["a"]["retopology_time_s"] == 12.0 and report["assets"]["a"]["retopology"] == {"method": "triflow"}
    assert "retopology (s)" in (tmp_path / "out/analytics/analytics.md").read_text()
