"""Critic loop behaviour with a fake subject, fake critics and a scripted patch writer."""

import json
import os
import re
import signal
import sys
from pathlib import Path

import pytest
from attrs import evolve

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
from kitbash.errors import BlenderScriptError, KitbashError, LLMAccessError, StateError
from kitbash.infra.patching import make_diff
from kitbash.infra.process import ProcessCancelled, current_registry, run_process
from kitbash.paths import OutputLayout
from kitbash.store.state import StateDB
from tests.helpers import process_gone, wait_for

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
        artifacts = {}
        for key in ("blend", "usd", "preview", "inventory", "render"):
            artifact = cycle_dir / key
            artifact.write_text(f"cycle {cycle}")
            artifacts[key] = str(artifact)
        return Evaluation(ok=True, facts={"score": score}, report={"cycle": cycle}, artifacts=artifacts)


class FakeCritic:
    def __init__(self, kind: CriticKind, *, fail_on: set[int] | None = None) -> None:
        self.kind = kind
        self.fail_on = fail_on or set()

    def review(self, request: ReviewRequest) -> Critique:
        score = request.evaluation.facts.get("score", 0.0)
        return Critique(
            critic=self.kind.value,
            summary=f"score {score}",
            scores=(CriterionScore("quality", score, score >= 0.8 and request.cycle not in self.fail_on),),
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


def make_loop(env, writer: ScriptedPatchWriter, *, rubric=RUBRIC, critics=None) -> CriticLoop:
    state, layout, config = env
    return CriticLoop(
        critics=critics if critics is not None else (FakeCritic(CriticKind.VISUAL), FakeCritic(CriticKind.TECHNICAL)),
        patch_writer=writer,
        rubric=rubric,
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


def test_interrupted_fresh_session_does_not_expose_the_previous_result(env):
    loop = ResumableLoop(make_loop(env, ScriptedPatchWriter([])), LoopSessions(env[0].meta))
    subject = FakeSubject()
    loop.run(subject, request="old", initial_script=lambda: script(0.9))

    def interrupt():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        loop.run(subject, request="changed-inputs", initial_script=interrupt, fresh=True)
    assert loop.best(subject) is None
    outcome = loop.run(subject, request="changed-inputs", initial_script=lambda: script(0.85))
    assert outcome.best.cycle == 2 and outcome.best.score == 0.85


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


@pytest.mark.parametrize("failure", ["criterion", "evaluation"])
def test_passing_lower_score_replaces_failed_high_score(env, failure):
    class Subject(FakeSubject):
        def evaluate(self, path, cycle_dir, cycle):
            evaluation = super().evaluate(path, cycle_dir, cycle)
            if failure == "evaluation" and cycle == 1:
                return evolve(evaluation, ok=False, error="USD export failed")
            return evaluation

    critics = (
        FakeCritic(CriticKind.VISUAL, fail_on={1} if failure == "criterion" else set()),
        FakeCritic(CriticKind.TECHNICAL),
    )
    writer = ScriptedPatchWriter([0.85])
    loop = ResumableLoop(make_loop(env, writer, critics=critics), LoopSessions(env[0].meta))
    subject = Subject()
    outcome = loop.run(subject, request="build", initial_script=lambda: script(0.95))
    assert outcome.passed and outcome.best.cycle == 2 and outcome.best.score == 0.85
    assert outcome.best.passed and outcome.best.status is DiffStatus.KEPT
    rows = env[0].cycles.cycles(subject.phase, subject.subject_id)
    assert [(row.cycle, row.passed) for row in rows] == [(1, False), (2, True)]
    cached = loop.run(subject, request="build", initial_script=lambda: script(0.95))
    assert cached.passed and cached.best.cycle == 2 and cached.cycles_run == 0
    assert subject.evaluated == [1, 2]


@pytest.mark.parametrize("budget", [2, 4])
@pytest.mark.parametrize("interruption", ["pending", "evaluated"])
def test_interrupted_session_resumes_the_exact_passing_cycle(env, tmp_path, budget, interruption):
    class Subject(FakeSubject):
        interrupted = False

        def evaluate(self, path, cycle_dir, cycle):
            if interruption == "pending" and cycle == 2 and not self.interrupted:
                self.interrupted = True
                raise KeyboardInterrupt
            return super().evaluate(path, cycle_dir, cycle)

    class Observer:
        def building(self, cycle):
            pass

        def critiquing(self, cycle):
            pass

        def evaluated(self, result):
            if interruption == "evaluated" and result.cycle == 2:
                raise KeyboardInterrupt

    config = evolve(env[2], critic=evolve(env[2].critic, max_cycles=budget))
    critics = (FakeCritic(CriticKind.VISUAL, fail_on={1}), FakeCritic(CriticKind.TECHNICAL))
    loop = ResumableLoop(make_loop((*env[:2], config), ScriptedPatchWriter([0.85]), critics=critics), LoopSessions(env[0].meta))
    subject = Subject()
    with pytest.raises(KeyboardInterrupt):
        loop.run(subject, request="build", initial_script=lambda: script(0.95), observer=Observer())
    assert not LoopSessions(env[0].meta).get(subject).done

    reopened = StateDB(tmp_path / "state.db")
    try:
        writer = ScriptedPatchWriter([])
        resumed = ResumableLoop(make_loop((reopened, env[1], config), writer, critics=critics), LoopSessions(reopened.meta))
        outcome = resumed.run(subject, request="build", initial_script=lambda: pytest.fail("rewrote initial script"))
        assert outcome.reason is LoopReason.PASSED and outcome.best.cycle == 2
        assert outcome.best.score == 0.85 and outcome.best.passed and outcome.cycles_run == 2
        assert subject.evaluated == [1, 2] and writer.requests == []
        cached = resumed.run(subject, request="build", initial_script=lambda: pytest.fail("rewrote completed session"))
        assert cached.passed and cached.best.cycle == 2 and cached.cycles_run == 0
    finally:
        reopened.close()


@pytest.mark.parametrize("require_all_pass", [True, False])
def test_unscored_required_criterion_is_not_an_eligible_result(env, require_all_pass):
    rubric = Rubric.parse(
        "| criterion | weight | pass condition | applies to |\n|-|-|-|-|\n"
        "| Quality | 1 | good enough | modelling |\n| Safety | 1 | safe | modelling |"
    )
    config = evolve(env[2], critic=evolve(env[2].critic, require_all_pass=require_all_pass))
    loop = make_loop((*env[:2], config), ScriptedPatchWriter([]), rubric=rubric)
    subject = FakeSubject()
    with pytest.raises(KitbashError, match="No eligible critic result"):
        loop.run(subject, initial_script=lambda: script(0.95), max_cycles=1)
    assert loop.best(subject) is None
    assert env[0].cycles.cycles(subject.phase, subject.subject_id)[0].passed is False


@pytest.mark.parametrize(
    "phase, missing",
    [
        (PhaseName.BREAKDOWN, "inventory"), (PhaseName.BREAKDOWN, "blend"), (PhaseName.BREAKDOWN, "render"),
        (PhaseName.MODELLING, "blend"), (PhaseName.MODELLING, "usd"), (PhaseName.MODELLING, "preview"),
        (PhaseName.LAYOUT, "blend"), (PhaseName.LAYOUT, "render"),
    ],
)
def test_required_artifact_cannot_be_omitted(env, phase, missing):
    class Subject(FakeSubject):
        def evaluate(self, path, cycle_dir, cycle):
            evaluation = super().evaluate(path, cycle_dir, cycle)
            return evolve(evaluation, artifacts={key: value for key, value in evaluation.artifacts.items() if key != missing})

    subject = Subject()
    subject.phase = phase
    subject.subject_id = "crate" if phase is PhaseName.MODELLING else ""
    rubric = Rubric.parse("| criterion | weight | pass condition | applies to |\n|-|-|-|-|\n| Quality | 1 | good enough | all |")
    loop = make_loop(env, ScriptedPatchWriter([]), rubric=rubric)
    with pytest.raises(KitbashError, match="No eligible critic result"):
        loop.run(subject, initial_script=lambda: script(0.95), max_cycles=1)
    assert loop.best(subject) is None
    assert env[0].cycles.cycles(subject.phase, subject.subject_id)[0].passed is False


@pytest.mark.parametrize("damage", ["deleted", "directory"])
def test_completed_session_rechecks_required_artifact_files(env, damage):
    subject = FakeSubject()
    loop = ResumableLoop(make_loop(env, ScriptedPatchWriter([])), LoopSessions(env[0].meta))
    outcome = loop.run(subject, request="build", initial_script=lambda: script(0.9))
    usd = Path(outcome.best.evaluation.artifacts["usd"])
    usd.unlink()
    if damage == "directory":
        usd.mkdir()
    with pytest.raises(StateError, match="no eligible result"):
        loop.run(subject, request="build", initial_script=lambda: pytest.fail("rewrote completed session"))
    assert loop.best(subject) is None


@pytest.mark.parametrize("failure", ["reverted", "evaluation"])
def test_ineligible_high_score_cannot_replace_a_passing_checkpoint(env, failure):
    subject = FakeSubject()
    loop = make_loop(env, ScriptedPatchWriter([]))
    first = loop.run(subject, initial_script=lambda: script(0.85)).best
    store = CycleStore(env[0].cycles, env[1])
    state = store.load(subject)
    store.write_cycle(subject, state, 2, script(0.99), make_diff(script(0.85), script(0.99)), score_before=0.85)
    path = store.cycle_dir(subject, 2) / "script.py"
    evaluation = subject.evaluate(path, path.parent, 2)
    result = evolve(
        first, cycle=2, script_path=path, scorecard=evolve(first.scorecard, overall=0.99),
        evaluation=evolve(evaluation, ok=False) if failure == "evaluation" else evaluation,
        status=DiffStatus.REVERTED if failure == "reverted" else DiffStatus.KEPT,
    )
    store.save_evaluation(subject, result)
    outcome = loop.run(subject, initial_script=lambda: pytest.fail("rewrote passing checkpoint"), session_start=1, max_cycles=2)
    assert outcome.passed and outcome.best.cycle == 1 and outcome.best.score == 0.85
    assert loop.best(subject).cycle == 1
    assert env[0].cycles.cycles(subject.phase, subject.subject_id)[1].passed is False


def test_failed_feedback_session_does_not_fall_back_to_an_old_pass(env):
    subject = FakeSubject(fail_on={2})
    config = evolve(env[2], critic=evolve(env[2].critic, max_cycles=1))
    loop = ResumableLoop(make_loop((*env[:2], config), ScriptedPatchWriter([0.95])), LoopSessions(env[0].meta))
    first = loop.run(subject, request="first", initial_script=lambda: script(0.9))
    assert first.passed
    with pytest.raises(KitbashError, match="No eligible critic result"):
        loop.run(subject, request="feedback", initial_script=lambda: script(0.9), feedback="make it taller")
    assert loop.best(subject) is None


def test_cached_pass_cannot_transfer_to_a_nonpassing_cycle(env):
    subject = FakeSubject()
    critics = (FakeCritic(CriticKind.VISUAL, fail_on={1}), FakeCritic(CriticKind.TECHNICAL))
    loop = ResumableLoop(make_loop(env, ScriptedPatchWriter([0.85]), critics=critics), LoopSessions(env[0].meta))
    outcome = loop.run(subject, request="build", initial_script=lambda: script(0.95))
    Path(outcome.best.evaluation.artifacts["usd"]).unlink()
    assert loop.best(subject).cycle == 1  # the only surviving candidate failed the rubric
    with pytest.raises(StateError, match="not eligible for a passed outcome"):
        loop.run(subject, request="build", initial_script=lambda: pytest.fail("rewrote completed session"))


def test_fatal_critic_cancels_earlier_running_critic_before_join(env, tmp_path):
    pid_path = tmp_path / "critic-pid"
    original = LLMAccessError("provider budget exhausted")

    class RunningCritic(FakeCritic):
        def review(self, request):
            run_process([
                sys.executable, "-c",
                f"import os,time; from pathlib import Path; Path({str(pid_path)!r}).write_text(str(os.getpid())); time.sleep(60)",
            ], timeout_s=90)
            raise AssertionError("cancelled critic continued")

    class FatalCritic(FakeCritic):
        def review(self, request):
            wait_for(lambda: pid_path.exists() and pid_path.stat().st_size)
            raise original

    loop = make_loop(env, ScriptedPatchWriter([]))
    loop._critics = (RunningCritic(CriticKind.VISUAL), FatalCritic(CriticKind.TECHNICAL))
    try:
        with pytest.raises(LLMAccessError) as caught:
            loop.run(FakeSubject(), initial_script=lambda: script(0.9))
        assert caught.value is original
        assert process_gone(int(pid_path.read_text()))
        with pytest.raises(ChildProcessError):
            os.waitpid(int(pid_path.read_text()), os.WNOHANG)
        pending = env[0].cycles.cycles(PhaseName.MODELLING, "crate")[0]
        assert pending.status == "pending" and pending.passed is None
    finally:
        if pid_path.exists() and not process_gone(int(pid_path.read_text())):
            os.killpg(int(pid_path.read_text()), signal.SIGKILL)


def test_last_critic_cancellation_leaves_cycle_pending_and_next_run_is_fresh(env):
    class CancellingCritic(FakeCritic):
        def review(self, request):
            current_registry().terminate_all()
            return super().review(request)

    loop = make_loop(env, ScriptedPatchWriter([]), critics=(CancellingCritic(CriticKind.VISUAL),))
    with pytest.raises(ProcessCancelled):
        loop.run(FakeSubject(), initial_script=lambda: script(0.9))
    pending = env[0].cycles.cycles(PhaseName.MODELLING, "crate")[0]
    assert pending.status == "pending" and pending.passed is None
    fresh = make_loop(env, ScriptedPatchWriter([]))
    assert fresh.run(FakeSubject(), initial_script=lambda: script(0.9)).reason is LoopReason.PASSED


@pytest.mark.parametrize("artifact", ["script", "blend", "usd", "preview"])
def test_changed_reviewed_bytes_cannot_resume_as_passed(env, artifact):
    subject = FakeSubject()
    loop = ResumableLoop(make_loop(env, ScriptedPatchWriter([])), LoopSessions(env[0].meta))
    result = loop.run(subject, request="build", initial_script=lambda: script(0.9)).best
    path = result.script_path if artifact == "script" else Path(result.evaluation.artifacts[artifact])
    original = path.read_bytes()
    path.write_bytes(b"x" * len(original))
    assert not result.eligible
    with pytest.raises(StateError, match="no eligible result"):
        loop.run(subject, request="build", initial_script=lambda: pytest.fail("rewrote a reviewed build"))


def test_mutation_during_independent_review_cannot_be_sealed_as_a_pass(env):
    class MutatingCritic(FakeCritic):
        def review(self, request):
            Path(request.evaluation.artifacts["blend"]).write_bytes(b"not the inspected build")
            return super().review(request)

    loop = make_loop(env, ScriptedPatchWriter([]), critics=(MutatingCritic(CriticKind.VISUAL),))
    with pytest.raises(KitbashError, match="No eligible critic result"):
        loop.run(FakeSubject(), initial_script=lambda: script(0.9), max_cycles=1)
    assert loop.best(FakeSubject()) is None


@pytest.mark.parametrize("damage", ["legacy", "workspace", "subject", "cycle"])
def test_saved_evidence_must_belong_to_the_requested_cycle(env, damage):
    subject = FakeSubject()
    loop = make_loop(env, ScriptedPatchWriter([]))
    result = loop.run(subject, initial_script=lambda: script(0.9)).best
    checkpoint = result.script_path.parent / "critique.json"
    data = json.loads(checkpoint.read_text())
    if damage == "legacy":
        del data["evidence"]
    else:
        data["evidence"][damage] = 2 if damage == "cycle" else "another-owner"
    checkpoint.write_text(json.dumps(data))
    assert loop.best(subject) is None


@pytest.mark.parametrize("damage", ["foreign", "symlink", "empty"])
def test_evaluation_rejects_unowned_or_empty_files(env, tmp_path, damage):
    foreign = tmp_path / "foreign.blend"
    foreign.write_bytes(b"foreign")

    class Subject(FakeSubject):
        def evaluate(self, path, cycle_dir, cycle):
            evaluation = super().evaluate(path, cycle_dir, cycle)
            blend = Path(evaluation.artifacts["blend"])
            if damage == "foreign":
                return evolve(evaluation, artifacts={**evaluation.artifacts, "blend": str(foreign)})
            blend.unlink()
            if damage == "symlink":
                blend.symlink_to(foreign)
            else:
                blend.touch()
            return evaluation

    loop = make_loop(env, ScriptedPatchWriter([]))
    with pytest.raises(KitbashError, match="No eligible critic result"):
        loop.run(Subject(), initial_script=lambda: script(0.9), max_cycles=1)


@pytest.mark.parametrize("change", ["added", "changed", "deleted"])
def test_entire_evaluated_bundle_is_bound_including_textures(env, change):
    class Subject(FakeSubject):
        def evaluate(self, path, cycle_dir, cycle):
            evaluation = super().evaluate(path, cycle_dir, cycle)
            build = cycle_dir / "build"
            build.mkdir()
            (build / "texture.png").write_bytes(b"texture")
            return evolve(evaluation, artifacts={**evaluation.artifacts, "build_dir": str(build)})

    subject = Subject()
    loop = make_loop(env, ScriptedPatchWriter([]))
    result = loop.run(subject, initial_script=lambda: script(0.9)).best
    build = Path(result.evaluation.artifacts["build_dir"])
    if change == "added":
        (build / "unexpected.png").write_bytes(b"new")
    elif change == "changed":
        (build / "texture.png").write_bytes(b"changed")
    else:
        (build / "texture.png").unlink()
    assert loop.best(subject) is None
