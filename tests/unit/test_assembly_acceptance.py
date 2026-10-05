"""Final publication must reflect assembled scene validation, not just completed export."""

import json
import sys
from contextlib import nullcontext
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from attrs import evolve
from rich.console import Console

from kitbash.agents.layout import LayoutSubject
from kitbash.agents.scene import SceneAgent
from kitbash.analytics.report import AnalyticsReport
from kitbash.analytics.tracker import Tracker
from kitbash.config import default_rubric_path, load_config
from kitbash.domain.inventory import Inventory
from kitbash.domain.phases import PhaseName, PhaseStatus
from kitbash.domain.rubric import Rubric
from kitbash.domain.run_input import RunInput
from kitbash.errors import StateError, UserAbort
from kitbash.interaction.autopilot import AutoPilot
from kitbash.interaction.protocols import GateAction, GateDecision, PhaseSummary
from kitbash.interaction.terminal import TerminalUser
from kitbash.paths import OutputLayout
from kitbash.phases.assembly import AssemblyPhase
from kitbash.store.state import StateDB

SCENE_FACTS = {
    "missing_assets": 0, "unexpected_assets": 0,
    "missing_placeholders": 0, "unexpected_placeholders": 0,
    "floating_assets": 0, "has_camera": True, "missing_textures": 0, "naming_violations": 0,
}


class Toolkit:
    def __init__(self):
        self.facts = dict(SCENE_FACTS)

    def run_script(self, script, args, log):
        Path(args["output_blend"]).write_text("assembled blend")

    def localize(self, blend, logs):
        return {"missing": [], "copied": []}

    def inspect_scene(self, blend, assets, placeholders, logs, name):
        return {"camera": {"name": "Camera"} if self.facts["has_camera"] else None}, dict(self.facts)

    def render_scene(self, blend, output, logs, **settings):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("render")
        return output


class Fidelity:
    mode = "preview_surface_baked"

    def __init__(self):
        self.score = 0.95
        self.scene_facts = dict(SCENE_FACTS)

    def check(self, blend, output, **settings):
        output.write_text("exported USD")
        return self

    def facts(self):
        return {
            **{f"usd_{key}": value for key, value in self.scene_facts.items()},
            "usd_roundtrip_score": self.score, "usd_broken_materials": 0, "usd_absolute_texture_paths": 0,
        }

    def report(self):
        return {"usd_material_mode": self.mode, "roundtrip_score": self.score, "scene_facts": self.scene_facts}


@pytest.fixture
def assembly_env(tmp_path, sample_inventory_dict):
    layout = OutputLayout.at(tmp_path / "out")
    layout.create()
    state = StateDB(layout.state_db)
    tracker = Tracker(state.spans)
    config = load_config(None)
    inventory = Inventory.from_dict(sample_inventory_dict)
    toolkit, fidelity = Toolkit(), Fidelity()
    subject = LayoutSubject(inventory, (), inventory.items, toolkit, config, layout, RunInput.create(None, "studio"))
    script = tmp_path / "layout.py"
    script.write_text("# approved layout\n")

    def make_phase(*, user=None, rubric=None, require_all_pass=True):
        settings = evolve(config, critic=evolve(config.critic, require_all_pass=require_all_pass))
        return AssemblyPhase(
            SimpleNamespace(subject=lambda *args: subject), SimpleNamespace(best=lambda subject: SimpleNamespace(script_path=script)),
            SimpleNamespace(load=lambda: (inventory, [], list(inventory.items))), toolkit, fidelity,
            rubric or Rubric.load(default_rubric_path()), state, user or AutoPilot(),
            SimpleNamespace(showing=lambda view: nullcontext()), tracker, settings, layout,
        )

    def run(phase, from_phase=None):
        return SceneAgent([phase], state, tracker, AnalyticsReport(state, layout)).run(from_phase)

    yield SimpleNamespace(layout=layout, state=state, toolkit=toolkit, fidelity=fidelity, make_phase=make_phase, run=run)
    state.close()


def test_failed_final_scorecard_cannot_complete_noninteractive_run(assembly_env):
    env = assembly_env
    env.fidelity.score = 0.2
    with pytest.raises(StateError, match="Final assembly validation failed"):
        env.run(env.make_phase())
    assert env.state.phases.status(PhaseName.ASSEMBLY) is not PhaseStatus.DONE
    assert env.state.meta.get("run_finished_at") is None
    report = json.loads((env.layout.scene_dir / "assembly.json").read_text())
    assert not report["scorecard"]["passed"]
    assert report["acceptance"]["status"] == "failed"
    assert not report["acceptance"]["published"]


@pytest.mark.parametrize("source", ["blend", "usd"])
@pytest.mark.parametrize("key, value", [
    ("missing_assets", 1), ("unexpected_assets", 1),
    ("missing_placeholders", 1), ("unexpected_placeholders", 1),
    ("floating_assets", 1), ("has_camera", False), ("missing_textures", 1),
    ("missing_assets", None),
])
def test_scene_integrity_cannot_be_waived_by_a_custom_rubric(assembly_env, source, key, value):
    env = assembly_env
    measured = env.toolkit.facts if source == "blend" else env.fidelity.scene_facts
    measured[key] = value
    rubric = Rubric.parse(
        "| criterion | weight | pass condition | applies to |\n|-|-|-|-|\n"
        "| USD quality | 1 | `usd_roundtrip_score >= 0.8` | assembly |"
    )
    with pytest.raises(StateError, match="Final assembly validation failed"):
        env.run(env.make_phase(rubric=rubric))
    result = env.state.meta.get("assembly")
    assert not result["scorecard"]["passed"] and not result["acceptance"]["automatic_pass"]
    prefix = "usd_" if source == "usd" else ""
    assert any(prefix + key in issue for issue in result["acceptance"]["issues"])


def test_unscored_final_criteria_are_not_an_automatic_pass(assembly_env):
    env = assembly_env
    rubric = Rubric.parse(
        "| criterion | weight | pass condition | applies to |\n|-|-|-|-|\n"
        "| USD quality | 1 | `usd_roundtrip_score >= 0.8` | assembly |\n"
        "| Composition | 1 | Looks good | assembly |"
    )
    with pytest.raises(StateError, match="Final assembly validation failed"):
        env.run(env.make_phase(rubric=rubric))
    assert not env.state.meta.get("assembly")["scorecard"]["passed"]
    criteria = env.state.meta.get("assembly")["scorecard"]["criteria"]
    assert criteria["composition"]["status"] == "unassessed" and criteria["composition"]["pass"] is None
    assert criteria["usd_quality"]["status"] == "passed"


class Human(AutoPilot):
    interactive = True

    def __init__(self, action):
        self.action = action

    def confirm(self, summary):
        assert summary.phase is PhaseName.ASSEMBLY and not summary.scorecard.passed
        return GateDecision(self.action)


def test_explicit_human_override_publishes_degraded_not_passed(assembly_env):
    env = assembly_env
    env.fidelity.score = 0.2
    report = env.run(env.make_phase(user=Human(GateAction.PUBLISH_DEGRADED)))
    scene = report["scene"]
    assert scene["acceptance"]["status"] == "overridden"
    assert scene["acceptance"]["published"] and not scene["acceptance"]["automatic_pass"]
    assert not scene["scorecard"]["passed"]
    assert env.state.phases.status(PhaseName.ASSEMBLY) is PhaseStatus.DONE
    stored = json.loads((env.layout.scene_dir / "assembly.json").read_text())
    assert stored["acceptance"]["override"] == "publish_degraded"
    assert not stored["scorecard"]["passed"]
    assert report["totals"]["user_interventions"] == 1


@pytest.mark.parametrize("action, error", [(GateAction.APPROVE, StateError), (GateAction.ABORT, UserAbort)])
def test_ordinary_approval_or_abort_cannot_override_failed_acceptance(assembly_env, action, error):
    env = assembly_env
    env.fidelity.score = 0.2
    with pytest.raises(error):
        env.run(env.make_phase(user=Human(action)))
    assert not env.state.meta.get("assembly")["acceptance"]["published"]
    assert env.state.phases.status(PhaseName.ASSEMBLY) is not PhaseStatus.DONE


def test_noninteractive_mode_never_requests_an_override(assembly_env):
    class Automated(AutoPilot):
        def confirm(self, summary):
            pytest.fail("noninteractive code requested a human override")

    env = assembly_env
    env.fidelity.score = 0.2
    with pytest.raises(StateError, match="Final assembly validation failed"):
        env.run(env.make_phase(user=Automated()))


def test_passing_final_scene_is_published_automatically(assembly_env):
    env = assembly_env
    report = env.run(env.make_phase())
    assert report["scene"]["acceptance"] == {"status": "passed", "automatic_pass": True, "published": True, "issues": []}
    assert report["scene"]["scorecard"]["passed"]
    assert env.state.phases.status(PhaseName.ASSEMBLY) is PhaseStatus.DONE


def test_failed_rerun_clears_previous_publication_and_completion(assembly_env):
    env = assembly_env
    env.run(env.make_phase())
    assert env.state.meta.get("run_finished_at") is not None
    env.fidelity.score = 0.2
    with pytest.raises(StateError, match="Final assembly validation failed"):
        env.run(env.make_phase(), from_phase=PhaseName.ASSEMBLY)
    assert env.state.meta.get("run_finished_at") is None
    assert env.state.meta.get("assembly")["acceptance"]["status"] == "failed"
    assert not env.state.meta.get("assembly")["acceptance"]["published"]


def test_interrupted_rebuild_does_not_leave_a_previous_pass(assembly_env, monkeypatch):
    env = assembly_env
    env.run(env.make_phase())

    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(env.toolkit, "run_script", interrupt)
    with pytest.raises(KeyboardInterrupt):
        env.run(env.make_phase(), from_phase=PhaseName.ASSEMBLY)
    result = env.state.meta.get("assembly")["acceptance"]
    assert result["status"] == "pending" and not result["automatic_pass"] and not result["published"]


@pytest.mark.parametrize("answer, expected", [("\n", GateAction.ABORT), ("a\nq\n", GateAction.ABORT), ("p\n", GateAction.PUBLISH_DEGRADED)])
def test_terminal_requires_explicit_degraded_publication(monkeypatch, answer, expected):
    monkeypatch.setattr(sys, "stdin", StringIO(answer))
    console = Console(file=StringIO(), color_system=None)
    user = TerminalUser(console, SimpleNamespace(paused=lambda: nullcontext()), SimpleNamespace(show=lambda images: None))
    decision = user.confirm(PhaseSummary(phase=PhaseName.ASSEMBLY, headline="Final validation failed"))
    assert decision.action is expected


def test_resume_revalidates_an_old_completed_but_unaccepted_assembly(assembly_env):
    env = assembly_env
    env.state.phases.set_status(PhaseName.ASSEMBLY, PhaseStatus.DONE)
    env.state.meta.set("assembly", {"scorecard": {"passed": False}})
    env.state.meta.set("run_finished_at", 1.0)
    env.fidelity.score = 0.2
    with pytest.raises(StateError, match="Final assembly validation failed"):
        env.run(env.make_phase())
    assert env.state.phases.status(PhaseName.ASSEMBLY) is not PhaseStatus.DONE
    assert env.state.meta.get("run_finished_at") is None
    assert env.state.meta.get("assembly")["acceptance"]["status"] == "failed"


def test_lenient_critic_setting_cannot_automatically_waive_a_final_failure(assembly_env):
    env = assembly_env
    rubric = Rubric.parse(
        "| criterion | weight | pass condition | applies to |\n|-|-|-|-|\n"
        "| USD quality | 10 | `usd_roundtrip_score >= 0.8` | assembly |\n"
        "| Required detail | 0.1 | `false` | assembly |"
    )
    with pytest.raises(StateError, match="Final assembly validation failed"):
        env.run(env.make_phase(rubric=rubric, require_all_pass=False))
    assert not env.state.meta.get("assembly")["acceptance"]["automatic_pass"]
