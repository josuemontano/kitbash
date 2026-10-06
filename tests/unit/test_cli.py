import pytest
from typer.testing import CliRunner

from kitbash.app import create_workspace, open_workspace
from kitbash.cli import app, parse_extra_args
from kitbash.domain.phases import PhaseName, PhaseStatus
from kitbash.domain.run_input import RunInput
from kitbash.errors import ConfigError
from kitbash.store.state import StateDB
from tests.helpers import reference_image

runner = CliRunner()


def test_model_flags_are_parsed():
    assert parse_extra_args(["--model.code=claude-sonnet-5", "--model.layout.visual_critic", "gemini-x"]) == {
        "models.code": "claude-sonnet-5",
        "models.phases.layout.visual_critic": "gemini-x",
    }
    with pytest.raises(ConfigError, match="Unknown option"):
        parse_extra_args(["--colour=red"])
    with pytest.raises(ConfigError, match="needs a model"):
        parse_extra_args(["--model.code"])


def test_dry_run_prints_the_plan_and_creates_nothing(tmp_path):
    image = tmp_path / "room.png"
    from PIL import Image

    Image.new("RGB", (64, 64), "white").save(image)
    output = tmp_path / "out"
    result = runner.invoke(app, ["build", "--output", str(output), "--image", str(image), "--dry-run", "--model.code=my-coder", "--max-cycles", "2"])
    assert result.exit_code == 0, result.output
    assert "Plan (dry run" in result.output and "my-coder" in result.output and "<= 2 cycles" in result.output
    assert not output.exists()


def test_exactly_one_input_is_required(tmp_path):
    result = runner.invoke(app, ["build", "--output", str(tmp_path / "o"), "--dry-run"])
    assert result.exit_code == 1 and "exactly one of --image or --prompt" in result.output
    result = runner.invoke(app, ["build", "--output", str(tmp_path / "o"), "--prompt", "a room", "--image", "x.png", "--dry-run"])
    assert result.exit_code == 1


def test_resume_without_a_run_fails_clearly(tmp_path):
    result = runner.invoke(app, ["resume", "--output", str(tmp_path / "nothing")])
    assert result.exit_code == 1 and "No kitbash run" in result.output


@pytest.mark.parametrize("mode", ["prompt", "image"])
def test_rebuild_preserves_completed_run(tmp_path, monkeypatch, mode):
    if mode == "image":
        image = reference_image(tmp_path / "room.png")
        run_input = RunInput.create(image, None)
        input_args = ["--image", str(image)]
    else:
        run_input = RunInput.create(None, "a room")
        input_args = ["--prompt", "a room"]
    config, layout, stored_input = create_workspace(
        tmp_path / "out", run_input, None, None, {"pipeline.style": "photorealistic"}
    )
    state = StateDB(layout.state_db)
    try:
        for phase in PhaseName:
            state.phases.set_status(phase, PhaseStatus.DONE)
        assembly = {"acceptance": {"status": "accepted", "published": True}}
        state.meta.set("assembly", assembly)
        state.meta.set("published_assembly", assembly)
    finally:
        state.close()
    layout.scene_dir.mkdir()
    layout.scene_blend.write_bytes(b"published blend")
    layout.scene_usd.write_bytes(b"published usd")
    before = {path.relative_to(layout.root): path.read_bytes() for path in layout.root.rglob("*") if path.is_file()}
    rubric = tmp_path / "changed-rubric.md"
    rubric.write_text(layout.rubric_snapshot.read_text().replace("|", " |"))

    def unexpected_startup(*args, **kwargs):
        pytest.fail("Repeated build reached application startup")

    monkeypatch.setattr("kitbash.cli.Application", unexpected_startup)
    result = runner.invoke(
        app, ["build", "--output", str(layout.root), *input_args, "--style", "2d", "--rubric", str(rubric), "--no-interactive"]
    )
    assert result.exit_code == 1, result.output
    assert "kitbash resume" in result.output
    assert "--output" in result.output
    assert {path.relative_to(layout.root): path.read_bytes() for path in layout.root.rglob("*") if path.is_file()} == before
    reopened_config, _, reopened_input = open_workspace(layout.root, {})
    assert reopened_config.pipeline.style == config.pipeline.style
    assert reopened_input == stored_input
    state = StateDB(layout.state_db)
    try:
        assert all(state.phases.status(phase) is PhaseStatus.DONE for phase in PhaseName)
        assert state.meta.get("assembly") == assembly
        assert state.meta.get("published_assembly") == assembly
    finally:
        state.close()


def test_rebuild_rejects_identical_input_and_settings_before_any_phase(tmp_path):
    run_input = RunInput.create(None, "a room")
    _, layout, _ = create_workspace(tmp_path / "out", run_input, None, None, {})
    with pytest.raises(ConfigError):
        create_workspace(layout.root, run_input, None, None, {})


@pytest.mark.parametrize("existing_file", ["state.db", "config.snapshot.toml", "rubric.snapshot.md"])
def test_build_preserves_incomplete_run_files(tmp_path, existing_file):
    output = tmp_path / "out"
    output.mkdir()
    marker = output / existing_file
    marker.write_bytes(b"incomplete run")
    with pytest.raises(ConfigError):
        create_workspace(output, RunInput.create(None, "a room"), None, None, {})
    assert list(output.iterdir()) == [marker]
    assert marker.read_bytes() == b"incomplete run"


def test_library_commands_on_an_empty_backlot(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(f'[paths]\nbacklot = "{tmp_path / "backlot"}"\n[embedding]\nbackend = "hashing"\ndimensions = 32\n')
    assert runner.invoke(app, ["library", "list", "--config", str(config)]).exit_code == 0
    assert runner.invoke(app, ["library", "search", "chair", "--config", str(config)]).exit_code == 0
    shown = runner.invoke(app, ["library", "show", "missing-id", "--config", str(config)])
    assert shown.exit_code == 1 and "No backlot asset" in shown.output
