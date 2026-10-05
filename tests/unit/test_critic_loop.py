"""Critic loop behaviour with a fake subject, fake critics and a scripted patch writer."""

import json
import re
from pathlib import Path

import pytest

from kitbash.analytics.tracker import Tracker
from kitbash.config import load_config
from kitbash.critique.critics import PatchRequest, ReviewRequest
from kitbash.critique.history import DiffStatus
from kitbash.critique.loop import CriticLoop, LoopReason
from kitbash.critique.sessions import LoopSessions, ResumableLoop
from kitbash.critique.store import CycleStore
from kitbash.critique.subject import CriticBrief, Evaluation
from kitbash.domain.critique import CriterionScore, CriticKind, Critique, Edit
from kitbash.domain.phases import PhaseName
from kitbash.domain.rubric import Rubric
from kitbash.errors import BlenderScriptError
from kitbash.infra.patching import make_diff
from kitbash.paths import OutputLayout
from kitbash.store.state import StateDB

RUBRIC = Rubric.parse("| criterion | weight | pass condition | applies to |\n|-|-|-|-|\n| Quality | 1 | good enough | modelling |")
SCORE = re.compile(r"^SCORE = ([0-9.]+)$", re.MULTILINE)


def script(score: float) -> str:
    return f"import kitbash_bpy as kb\n\nkb.reset_scene()\nSCORE = {score}\nkb.save_asset(None)\n"


class FakeSubject:
    phase = PhaseName.MODELLING
    subject_id = "crate"

    def __init__(self, fail_on: set[int] | None = None) -> None:
        self.evaluated: list[int] = []
        self.fail_on = fail_on or set()

    def brief(self) -> CriticBrief:
        return CriticBrief(phase=self.phase, subject="asset crate", description="a crate")

    def api_reference(self) -> str:
        return "kb.* helpers"

    def evaluate(self, path: Path, cycle_dir: Path, cycle: int) -> Evaluation:
        self.evaluated.append(cycle)
        if cycle in self.fail_on:
            raise BlenderScriptError("boom", traceback_text="Traceback: line 4")
        score = float(SCORE.search(path.read_text()).group(1))
        return Evaluation(ok=True, facts={"score": score}, report={"cycle": cycle})


class FakeCritic:
    def __init__(self, kind: CriticKind) -> None:
        self.kind = kind

    def review(self, request: ReviewRequest) -> Critique:
        score = request.evaluation.facts.get("score", 0.0)
        return Critique(
            critic=self.kind.value,
            summary=f"score {score}",
            scores=(CriterionScore("quality", score, score >= 0.8),),
            edits=(Edit(instruction="raise the score", source=self.kind.value),),
        )


class ScriptedPatchWriter:
    """Returns diffs that set SCORE to the next scripted value (or a fixed diff text)."""

    def __init__(self, scores: list[float | str]) -> None:
        self.scores = list(scores)
        self.requests: list[PatchRequest] = []

    def propose(self, request: PatchRequest) -> str:
        self.requests.append(request)
        value = self.scores.pop(0)
        if isinstance(value, str):
            return value
        current = float(SCORE.search(request.script).group(1))
        return make_diff(request.script, request.script.replace(f"SCORE = {current}", f"SCORE = {value}"))


@pytest.fixture
def env(tmp_path):
    state = StateDB(tmp_path / "state.db")
    layout = OutputLayout.at(tmp_path / "out")
    config = load_config(None, {"critic.patch_attempts": 2})
    yield state, layout, config
    state.close()


def make_loop(env, writer: ScriptedPatchWriter) -> CriticLoop:
    state, layout, config = env
    return CriticLoop(
        critics=(FakeCritic(CriticKind.VISUAL), FakeCritic(CriticKind.TECHNICAL)),
        patch_writer=writer,
        rubric=RUBRIC,
        config=config.critic,
        store=CycleStore(state.cycles, layout),
        tracker=Tracker(state.spans),
    )


def test_passes_on_the_first_cycle_and_checkpoints_files(env):
    loop = make_loop(env, ScriptedPatchWriter([]))
    outcome = loop.run(FakeSubject(), initial_script=lambda: script(0.9))
    assert outcome.reason is LoopReason.PASSED and outcome.cycles_run == 1 and outcome.best.cycle == 1
    cycle_dir = env[1].cycle_dir(PhaseName.MODELLING, 1, "crate")
    assert {p.name for p in cycle_dir.iterdir()} >= {"script.py", "diff.patch", "critique.json", "report.json"}
    critique = json.loads((cycle_dir / "critique.json").read_text())
    assert critique["scorecard"]["passed"] and critique["scorecard"]["criteria"]["quality"]["score"] == 0.9
    assert (cycle_dir / "diff.patch").read_text().startswith("--- a/script.py")


def test_improves_through_patches_until_it_passes(env):
    writer = ScriptedPatchWriter([0.7, 0.9])
    outcome = make_loop(env, writer).run(FakeSubject(), initial_script=lambda: script(0.5))
    assert outcome.reason is LoopReason.PASSED and outcome.best.cycle == 3 and outcome.cycles_run == 3
    assert [r.cycle for r in writer.requests] == [2, 3]
    instructions = [e.instruction for e in writer.requests[0].edits]
    assert "raise the score" in instructions and instructions[0] == "Make 'Quality' pass."  # failing criteria come first
    diffs = env[0].cycles.diffs(PhaseName.MODELLING, "crate")
    assert [d.status for d in diffs] == ["kept", "kept", "kept"]


def test_worse_patches_are_reverted_and_the_best_script_is_patched_next(env):
    writer = ScriptedPatchWriter([0.3, 0.9])
    outcome = make_loop(env, writer).run(FakeSubject(), initial_script=lambda: script(0.6))
    assert outcome.passed and outcome.best.cycle == 3
    assert "SCORE = 0.6" in writer.requests[1].script  # patched from cycle 1, not the reverted cycle 2
    assert "cycle 02: reverted" in writer.requests[1].history
    statuses = [d.status for d in env[0].cycles.diffs(PhaseName.MODELLING, "crate")]
    assert statuses == ["kept", "reverted", "kept"]


def test_small_dips_are_kept_and_built_upon(env):
    writer = ScriptedPatchWriter([0.59, 0.9])  # within revert_epsilon (0.02) of 0.6
    outcome = make_loop(env, writer).run(FakeSubject(), initial_script=lambda: script(0.6))
    assert outcome.passed and "SCORE = 0.59" in writer.requests[1].script
    assert [d.status for d in env[0].cycles.diffs(PhaseName.MODELLING, "crate")] == ["kept", "kept", "kept"]


def test_stops_early_when_scores_stall(env):
    outcome = make_loop(env, ScriptedPatchWriter([0.605, 0.6, 0.95])).run(FakeSubject(), initial_script=lambda: script(0.6))
    assert outcome.reason is LoopReason.STALLED and outcome.cycles_run == 3


def test_stops_when_a_diff_repeats(env):
    repeated = make_diff(script(0.6), script(0.61))
    writer = ScriptedPatchWriter([repeated, repeated])
    outcome = make_loop(env, writer).run(FakeSubject(), initial_script=lambda: script(0.6))
    assert outcome.reason is LoopReason.REPEATED_DIFF and "repeats the diff of cycle 02" in outcome.message


def test_patches_that_do_not_apply_are_rejected_then_escalated(env):
    bad = ["@@ -1 +1 @@\n-nothing like this\n+x = 1\n", "@@ -1 +1 @@\n-nor this\n+y = 2\n"]
    outcome = make_loop(env, ScriptedPatchWriter(bad)).run(FakeSubject(), initial_script=lambda: script(0.5))
    assert outcome.reason is LoopReason.PATCH_FAILED
    cycle_dir = env[1].cycle_dir(PhaseName.MODELLING, 2, "crate")
    assert sorted(p.name for p in cycle_dir.glob("rejected_*.patch")) == ["rejected_1.patch", "rejected_2.patch"]
    assert [d.status for d in env[0].cycles.diffs(PhaseName.MODELLING, "crate")] == ["kept", "rejected", "rejected"]


def test_repeating_a_rejected_diff_escalates(env):
    bad = "@@ -1 +1 @@\n-nothing like this\n+x = 1\n"
    outcome = make_loop(env, ScriptedPatchWriter([bad, bad])).run(FakeSubject(), initial_script=lambda: script(0.5))
    assert outcome.reason is LoopReason.REPEATED_DIFF and "(rejected)" in outcome.message


def test_rejection_reason_is_sent_back_to_the_patch_writer(env):
    bad = "@@ -1 +1 @@\n-nothing like this\n+x = 1\n"
    writer = ScriptedPatchWriter([bad, 0.9])
    outcome = make_loop(env, writer).run(FakeSubject(), initial_script=lambda: script(0.5))
    assert outcome.passed and "does not apply" in writer.requests[1].rejection


def test_uses_all_cycles_and_reports_max_cycles(env):
    outcome = make_loop(env, ScriptedPatchWriter([0.3, 0.5, 0.7])).run(FakeSubject(), initial_script=lambda: script(0.1))
    assert outcome.reason is LoopReason.MAX_CYCLES and outcome.cycles_run == 4 and outcome.best.cycle == 4


def test_script_errors_reach_the_patch_writer(env):
    writer = ScriptedPatchWriter([0.9])
    outcome = make_loop(env, writer).run(FakeSubject(fail_on={1}), initial_script=lambda: script(0.5))
    assert outcome.passed
    assert "boom" in writer.requests[0].error and "Traceback: line 4" in writer.requests[0].error


def test_resume_re_evaluates_a_pending_cycle_without_rewriting_the_script(env):
    calls = []

    class Crash(FakeSubject):
        def evaluate(self, path, cycle_dir, cycle):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        make_loop(env, ScriptedPatchWriter([])).run(Crash(), initial_script=lambda: calls.append(1) or script(0.9))
    subject = FakeSubject()
    outcome = make_loop(env, ScriptedPatchWriter([])).run(subject, initial_script=lambda: calls.append(1) or script(0.9))
    assert outcome.passed and calls == [1] and subject.evaluated == [1]


def test_feedback_session_builds_on_the_reviewed_result_and_is_never_reverted(env):
    state = env[0]
    writer = ScriptedPatchWriter([0.4, 0.5, 0.55, 0.6])
    loop = ResumableLoop(make_loop(env, writer), LoopSessions(state.meta))
    subject = FakeSubject()
    first = loop.run(subject, request="r0", initial_script=lambda: script(0.9))
    assert first.passed and first.best.cycle == 1
    again = loop.run(subject, request="r0", initial_script=lambda: script(0.9))
    assert again.cycles_run == 0 and again.best.cycle == 1  # finished session: nothing re-runs
    feedback = loop.run(subject, request="r1", initial_script=lambda: script(0.9), feedback="make it taller")
    request = writer.requests[0]
    assert request.edits[0].instruction == "make it taller" and request.edits[0].source == "user"
    assert "SCORE = 0.9" in request.script
    assert feedback.reason is LoopReason.MAX_CYCLES and feedback.cycles_run == 4
    assert feedback.best.cycle == 5 and feedback.best.score == 0.6  # session best, even though cycle 1 scored higher
    statuses = [d.status for d in state.cycles.diffs(PhaseName.MODELLING, "crate")]
    assert statuses == ["kept"] * 5  # the feedback cycle is the session baseline, never reverted


def test_fresh_session_rewrites_the_script(env):
    state = env[0]
    loop = ResumableLoop(make_loop(env, ScriptedPatchWriter([])), LoopSessions(state.meta))
    subject = FakeSubject()
    loop.run(subject, request="a1", initial_script=lambda: script(0.9))
    outcome = loop.run(subject, request="a2", initial_script=lambda: script(0.95), fresh=True)
    assert outcome.best.cycle == 2 and "SCORE = 0.95" in outcome.best.script_path.read_text()
    diffs = state.cycles.diffs(PhaseName.MODELLING, "crate")
    assert diffs[-1].reason == "initial" and diffs[-1].status == DiffStatus.KEPT.value


def test_a_new_request_replaces_a_crashed_session(env):
    """Feedback given after a crash must not be swallowed by the unfinished old session."""
    state = env[0]

    class CrashOnce(FakeSubject):
        crashed = False

        def evaluate(self, path, cycle_dir, cycle):
            if not CrashOnce.crashed:
                CrashOnce.crashed = True
                raise RuntimeError("unexpected")
            return super().evaluate(path, cycle_dir, cycle)

    writer = ScriptedPatchWriter([0.9])
    loop = ResumableLoop(make_loop(env, writer), LoopSessions(state.meta))
    subject = CrashOnce()
    with pytest.raises(RuntimeError):
        loop.run(subject, request="first", initial_script=lambda: script(0.5))
    outcome = loop.run(subject, request="second", initial_script=lambda: script(0.6), fresh=True)
    rows = {c.cycle: c.status for c in state.cycles.cycles(PhaseName.MODELLING, "crate")}
    assert rows[1] == "abandoned"  # the stale pending cycle is not re-evaluated
    assert outcome.passed and outcome.best.cycle == 3 and loop.best(subject).cycle == 3
    assert "SCORE = 0.6" in writer.requests[0].script  # patched from the new session's fresh script


def test_a_session_without_its_own_cycles_falls_back_to_its_base(env):
    state = env[0]
    bad = ["@@ -1 +1 @@\n-nothing\n+x\n", "@@ -1 +1 @@\n-nope\n+y\n"]
    writer = ScriptedPatchWriter([0.3, 0.301, 0.302, *bad])
    loop = ResumableLoop(make_loop(env, writer), LoopSessions(state.meta))
    subject = FakeSubject()
    loop.run(subject, request="r0", initial_script=lambda: script(0.9))  # cycle 1 passes at 0.9
    second = loop.run(subject, request="r1", initial_script=lambda: script(0.9), feedback="change it")
    assert second.reason.value == "stalled" and second.best.cycle == 4  # its own best, below cycle 1's score
    third = loop.run(subject, request="r2", initial_script=lambda: script(0.9), feedback="and again")
    assert third.reason.value == "patch_failed"
    assert third.best.cycle == second.best.cycle and loop.best(subject).cycle == second.best.cycle
