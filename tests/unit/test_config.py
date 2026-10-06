from pathlib import Path

import pytest

from kitbash.config import load_config, parse_model_overrides, structure_config
from kitbash.domain.phases import PhaseName
from kitbash.domain.roles import Role
from kitbash.errors import ConfigError


def test_defaults_load_with_spec_values():
    config = load_config()
    assert config.pipeline.threads == 2 and config.pipeline.review_buffer == 6
    assert config.critic.max_cycles == 4 and config.pipeline.style == "photorealistic"
    assert config.models.model_for(Role.VISUAL_CRITIC) == "gemini-3.8-flash"
    assert config.models.model_for(Role.PROMPT_ANALYSIS) == "gpt-6-sol"
    assert config.trellis.steps == 64 and config.trellis.pipeline_type == "1024" and config.trellis.no_texture
    assert config.modelling.method == "trellis"
    assert set(config.styles) == {"photorealistic", "2d", "animated-3d"}
    assert config.paths.backlot == Path("~/.local/share/backlot").expanduser()


def test_user_file_and_overrides_merge(tmp_path):
    user = tmp_path / "mine.toml"
    user.write_text('[pipeline]\nthreads = 5\n[models]\ncode = "claude-sonnet-5"\n[models.phases.layout]\ncode = "gpt-6-sol"\n')
    config = load_config(user, {"pipeline.review_buffer": 3, **parse_model_overrides({"breakdown.visual_critic": "x-vision"})})
    assert config.pipeline.threads == 5 and config.pipeline.review_buffer == 3
    assert config.models.model_for(Role.CODE) == "claude-sonnet-5"
    assert config.models.model_for(Role.CODE, PhaseName.LAYOUT) == "gpt-6-sol"
    assert config.models.model_for(Role.VISUAL_CRITIC, PhaseName.BREAKDOWN) == "x-vision"
    assert config.models.model_for(Role.VISUAL_CRITIC, PhaseName.MODELLING) == "gemini-3.8-flash"
    phases = {(p.value if p else None, r.value) for p, r, _ in config.models.all_assignments()}
    assert ("layout", "code") in phases and ("breakdown", "visual_critic") in phases


@pytest.mark.parametrize("method", ["trellis", "procedural"])
def test_snapshot_round_trips(tmp_path, method):
    config = load_config(None, {"pipeline.threads": 7, "modelling.method": method})
    snapshot = tmp_path / "config.snapshot.toml"
    snapshot.write_text(config.snapshot_toml())
    assert load_config(snapshot) == config


@pytest.mark.parametrize(
    "text, message",
    [
        ("[pipeline]\nthreadz = 1\n", "Unknown config key"),
        ("[pipeline]\nthreads = 'two'\n", "must be int"),
        ('[pipeline]\nstyle = "noir"\n', "Unknown style"),
        ("[pipeline]\nthreads = 0\n", "at least 1"),
        ('[models]\nfoo = "x"\n', "Unknown config key"),
        ('[models.phases.nowhere]\ncode = "x"\n', "Unknown phase"),
        ('[usd]\nmaterialx = "sometimes"\n', "materialx"),
        ('[modelling]\nmethod = "fallback"\n', "modelling.method"),
        ('[evaluation]\nmodel = " "\n', "evaluation.model"),
        ('[evaluation]\nbase_url = "file:///tmp/clef"\n', "evaluation.base_url"),
        ('[evaluation]\nbase_url = "http://user:secret@localhost"\n', "evaluation.base_url"),
        ('[evaluation]\ntimeout_s = inf\n', "evaluation.timeout_s"),
        ('[evaluation]\ntimeout_s = 0\n', "evaluation.timeout_s"),
        ('[critic]\nconfidence_threshold = nan\n', "critic.confidence_threshold"),
        ('[critic]\nconfidence_threshold = 1.1\n', "critic.confidence_threshold"),
        ('[critic]\nrevert_epsilon = -1\n', "critic.revert_epsilon"),
        ("[pipeline\n", "Invalid TOML"),
    ],
)
def test_invalid_configs_fail_with_clear_messages(tmp_path, text, message):
    path = tmp_path / "bad.toml"
    path.write_text(text)
    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_model_override_syntax():
    assert parse_model_overrides({"code": "m", "layout.visual-critic": "v"}) == {
        "models.code": "m",
        "models.phases.layout.visual_critic": "v",
    }
    with pytest.raises(ConfigError, match="Unknown model role"):
        parse_model_overrides({"painter": "m"})
    with pytest.raises(ConfigError, match="Invalid model override"):
        parse_model_overrides({"a.b.c": "m"})


def test_missing_model_for_role_is_an_error():
    config = load_config()
    data = dict(config.raw)
    data["models"] = {k: v for k, v in data["models"].items() if k != "code"}
    with pytest.raises(ConfigError, match="code"):
        structure_config(data)


def test_legacy_snapshot_defaults_to_trellis():
    data = dict(load_config().raw)
    del data["modelling"]
    config = structure_config(data)
    assert config.modelling.method == "trellis"
    assert config.raw["modelling"] == {"method": "trellis"}
