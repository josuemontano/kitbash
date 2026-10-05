import pytest
from typer.testing import CliRunner

from kitbash.cli import app, parse_extra_args
from kitbash.errors import ConfigError

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


def test_library_commands_on_an_empty_backlot(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(f'[paths]\nbacklot = "{tmp_path / "backlot"}"\n[embedding]\nbackend = "hashing"\ndimensions = 32\n')
    assert runner.invoke(app, ["library", "list", "--config", str(config)]).exit_code == 0
    assert runner.invoke(app, ["library", "search", "chair", "--config", str(config)]).exit_code == 0
    shown = runner.invoke(app, ["library", "show", "missing-id", "--config", str(config)])
    assert shown.exit_code == 1 and "No backlot asset" in shown.output
