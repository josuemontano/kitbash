"""The interactive terminal flow, answered through stdin: gate feedback, asset feedback and approvals."""

import json

import pytest
from typer.testing import CliRunner

from kitbash.cli import app
from kitbash.store.state import StateDB
from tests.helpers import omp_calls, reference_image, requires_blender, write_test_config

pytestmark = [pytest.mark.integration, pytest.mark.blender, requires_blender]


def test_interactive_feedback_reenters_the_critic_loops(tmp_path, monkeypatch):
    log = tmp_path / "omp_calls.jsonl"
    monkeypatch.setenv("FAKE_OMP_LOG", str(log))
    monkeypatch.setenv("FAKE_OMP_REFERENCE", str(reference_image(tmp_path / "asset.png")))
    config = write_test_config(tmp_path, pipeline={"threads": 1})
    answers = [
        "f", "Make the crate a little taller",  # breakdown gate: feedback -> another critic cycle
        "a",                                    # breakdown gate: approve
        "f", "Warmer wood colour",              # first asset review: feedback -> rework
        "a", "a",                               # approve the reworked asset and the other one (either order)
        "a", "a", "a", "a",                     # remaining reviews and the modelling and layout gates
    ]
    output = tmp_path / "out"
    result = CliRunner().invoke(
        app,
        ["build", "--image", str(reference_image(tmp_path / "room.png")), "--output", str(output), "--config", str(config)],
        input="\n".join(answers) + "\n",
    )
    assert result.exit_code == 0, result.output + (str(result.exception) if result.exception else "")

    breakdown_cycles = sorted(p.name for p in (output / "phases/01_breakdown/cycles").iterdir())
    assert breakdown_cycles == ["01", "02"]
    assert "patched by the fake" in (output / "phases/01_breakdown/cycles/02/script.py").read_text()
    patch_calls = [c for c in omp_calls(log) if c["task"] == "breakdown.patch"]
    assert len(patch_calls) == 1

    state = StateDB(output / "state.db")
    reworked = [a for a in state.assets.all() if a.feedback]
    assert len(reworked) == 1 and reworked[0].feedback == ("Warmer wood colour",)
    assert all(a.state.value == "approved" for a in state.assets.all())
    transitions = [t["to_state"] for t in state.assets.transitions() if t["asset_id"] == reworked[0].id]
    assert "needs_rework" in transitions and transitions[-1] == "approved"
    state.close()
    assert len(list((output / f"phases/02_modelling/{reworked[0].id}/cycles").iterdir())) == 2

    analytics = json.loads((output / "analytics/analytics.json").read_text())
    assert analytics["totals"]["user_interventions"] >= 2
    assert analytics["phases"]["breakdown"]["user_interventions"] >= 1
