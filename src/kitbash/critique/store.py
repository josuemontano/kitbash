"""Checkpoints of the critic loop: ``cycles/NN/`` files plus the cycles and diffs tables of state.db."""

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from attrs import define, field, frozen

from kitbash.critique.history import DiffHistory, DiffStatus, HistoryEntry
from kitbash.critique.subject import Evaluation, LoopSubject
from kitbash.domain.critique import Critique, Edit, ScoreCard, scorecard_from_dict
from kitbash.domain.phases import PhaseName
from kitbash.paths import OutputLayout
from kitbash.services.artifacts import snapshot
from kitbash.store.state import CycleRepository, CycleRow, DiffRow

PENDING = "pending"
ABANDONED = "abandoned"  # written but never evaluated: superseded by a new request
PRIORITY = {"high": 0, "medium": 1, "low": 2}
REQUIRED_ARTIFACTS = {
    PhaseName.BREAKDOWN: ("inventory", "blend", "render"),
    PhaseName.MODELLING: ("blend", "usd", "preview"),
    PhaseName.LAYOUT: ("blend", "render"),
}


@frozen
class CycleEvidence:
    """The owner and bytes evaluated before critic review; never resealed on resume."""

    workspace: Path
    phase: PhaseName
    subject_id: str
    cycle: int
    hashes: Mapping[str, str]

    @property
    def root(self) -> Path:
        layout = OutputLayout.at(self.workspace)
        return layout.asset_dir(self.subject_id) if self.subject_id else layout.phase_dir(self.phase)

    def capture(self, script: Path, evaluation: Evaluation) -> dict[str, str]:
        layout = OutputLayout.at(self.workspace)
        expected = layout.cycle_dir(self.phase, self.cycle, self.subject_id) / "script.py"
        if (
            script.absolute() != expected or self.root.resolve() != self.root
            or not self.root.is_relative_to(layout.phase_dir(self.phase))
        ):
            raise ValueError("Cycle script does not belong to its workspace, subject and cycle")
        for key in REQUIRED_ARTIFACTS[self.phase]:
            path = Path(evaluation.artifacts[key])
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing or empty required artifact: {key}")
        return snapshot(self.root, [
            script, *(Path(path) for path in evaluation.artifacts.values() if path), *evaluation.images,
        ])

    def matches(self, result: CycleResult) -> bool:
        try:
            return (
                self.phase is result.phase and self.cycle == result.cycle and bool(self.hashes)
                and self.capture(result.script_path, result.evaluation) == self.hashes
            )
        except (OSError, ValueError, KeyError):
            return False

    def to_dict(self) -> dict:
        return {
            "workspace": str(self.workspace), "phase": self.phase.value,
            "subject": self.subject_id, "cycle": self.cycle, "hashes": dict(self.hashes),
        }


@frozen
class CycleResult:
    cycle: int
    script_path: Path
    scorecard: ScoreCard
    critiques: tuple[Critique, ...]
    evaluation: Evaluation
    status: DiffStatus
    phase: PhaseName
    evidence: CycleEvidence | None = None

    @property
    def score(self) -> float:
        return self.scorecard.overall

    @property
    def eligible(self) -> bool:
        """A kept, fully evaluated result still owned by the same task with unchanged bytes."""
        return (
            self.status is DiffStatus.KEPT
            and self.evaluation.ok
            and bool(self.scorecard.entries)
            and all(entry.score is not None and entry.passed is not None for entry in self.scorecard.entries)
            and not self.scorecard.regressions()
            and self.evidence is not None
            and self.evidence.matches(self)
        )

    @property
    def passed(self) -> bool:
        return self.eligible and self.scorecard.passed

    def edits(self) -> list[Edit]:
        """Critic edits, fixes for failed criteria, and evidence requests for unassessed criteria."""
        edits = [edit for critique in self.critiques for edit in critique.edits]
        edits += [
            Edit(instruction=f"Make '{entry.name}' pass. {' '.join(entry.notes)}".strip(), source="rubric", priority="high")
            for entry in self.scorecard.failing()
            if entry.decided_by == "check" or not entry.notes
        ]
        edits += [
            Edit(instruction=f"Provide evidence to assess '{entry.name}'. {' '.join(entry.notes)}".strip(), source="rubric", priority="high")
            for entry in self.scorecard.unassessed()
        ]
        return sorted(edits, key=lambda e: PRIORITY.get(e.priority, 1))


@define
class LoopState:
    results: dict[int, CycleResult] = field(factory=dict)
    pending: tuple[int, Path] | None = None
    history: DiffHistory = field(factory=DiffHistory)
    last_cycle: int = 0

    @property
    def next_cycle(self) -> int:
        return self.last_cycle + 1

    def best(self, since: int = 1, *, base_cycle: int | None = None) -> CycleResult | None:
        """Prefer eligible passes, then score; only an unevaluated session can fall back to its base."""
        candidates = [r for c, r in self.results.items() if c >= since]
        if not candidates and base_cycle is not None and (base := self.results.get(base_cycle)) is not None:
            candidates = [base]
        return max((r for r in candidates if r.eligible), key=lambda r: (r.scorecard.passed, r.score, r.cycle), default=None)

    def session_cycles(self, start: int) -> int:
        return sum(1 for c in self.results if c >= start)

    def previous(self) -> tuple[Mapping[str, Any], ...]:
        """All completed decisions, including reverted builds, in evaluation order."""
        return tuple(
            {"iteration": cycle, "status": result.status.value, "scorecard": result.scorecard.to_dict()}
            for cycle, result in sorted(self.results.items())
        )

    def render_history(self) -> str:
        """Keep decision evidence alongside diffs for evaluators, critics and the patch writer."""
        decisions = json.dumps(self.previous(), indent=2, default=str)
        return f"{self.history.render()}\n\n## Previous rubric results\n```json\n{decisions}\n```"



class CycleStore:
    """Reads and writes everything a loop needs to resume: scripts, diffs, critiques and statuses."""

    def __init__(self, cycles: CycleRepository, layout: OutputLayout) -> None:
        self._cycles = cycles
        self._layout = layout

    def cycle_dir(self, subject: LoopSubject, cycle: int) -> Path:
        return self._layout.cycle_dir(subject.phase, cycle, subject.subject_id)

    def seal(self, subject: LoopSubject, cycle: int, script: Path, evaluation: Evaluation) -> CycleEvidence | None:
        if not evaluation.ok:
            return None
        evidence = CycleEvidence(self._layout.root, subject.phase, subject.subject_id, cycle, {})
        try:
            hashes = evidence.capture(script, evaluation)
        except (OSError, ValueError, KeyError):
            return None
        return CycleEvidence(evidence.workspace, evidence.phase, evidence.subject_id, cycle, hashes)

    def write_cycle(
        self, subject: LoopSubject, state: LoopState, cycle: int, script: str, diff: str, *, score_before: float | None, initial: bool = False
    ) -> None:
        directory = self.cycle_dir(subject, cycle)
        directory.mkdir(parents=True, exist_ok=True)
        script_path = directory / "script.py"
        diff_path = directory / "diff.patch"
        script_path.write_text(script, encoding="utf-8")
        diff_path.write_text(diff, encoding="utf-8")
        entry = HistoryEntry(cycle=cycle, status=DiffStatus.APPLIED, diff=diff, score_before=score_before, initial=initial)
        self._cycles.save_cycle(
            CycleRow(
                phase=subject.phase.value, subject=subject.subject_id, cycle=cycle, script_path=str(script_path),
                diff_path=str(diff_path), critique_path=None, score=None, passed=None, status=PENDING,
                summary="initial script" if initial else "", created_at=time.time(),
            )
        )
        self._cycles.save_diff(
            DiffRow(
                phase=subject.phase.value, subject=subject.subject_id, cycle=cycle, status=DiffStatus.APPLIED.value,
                fingerprint=entry.fingerprint, diff_path=str(diff_path), reason="initial" if initial else "",
                score_before=score_before, score_after=None,
            )
        )
        state.history.add(entry)
        state.pending = (cycle, script_path)
        state.last_cycle = max(state.last_cycle, cycle)

    def reject(self, subject: LoopSubject, state: LoopState, cycle: int, diff: str, reason: str, score: float) -> None:
        directory = self.cycle_dir(subject, cycle)
        directory.mkdir(parents=True, exist_ok=True)
        attempt = len(list(directory.glob("rejected_*.patch"))) + 1
        path = directory / f"rejected_{attempt}.patch"
        path.write_text(f"# rejected: {reason}\n{diff}", encoding="utf-8")
        entry = HistoryEntry(cycle=cycle, status=DiffStatus.REJECTED, diff=diff, reason=reason, score_before=score)
        state.history.add(entry)
        self._cycles.save_diff(
            DiffRow(
                phase=subject.phase.value, subject=subject.subject_id, cycle=cycle, status=DiffStatus.REJECTED.value,
                fingerprint=entry.fingerprint, diff_path=str(path), reason=reason, score_before=score, score_after=None,
            )
        )

    def save_evaluation(self, subject: LoopSubject, result: CycleResult) -> None:
        directory = result.script_path.parent
        critique_path = directory / "critique.json"
        (directory / "report.json").write_text(json.dumps(result.evaluation.report, indent=2, default=str), encoding="utf-8")
        critique_path.write_text(
            json.dumps(
                {
                    "cycle": result.cycle,
                    "status": result.status.value,
                    "scorecard": result.scorecard.to_dict(),
                    "critiques": [c.to_dict() for c in result.critiques],
                    "evaluation": result.evaluation.to_dict(),
                    "evidence": result.evidence.to_dict() if result.evidence else None,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        summary = " | ".join(c.summary for c in result.critiques if c.summary)[:1000]
        self._cycles.save_cycle(
            CycleRow(
                phase=subject.phase.value, subject=subject.subject_id, cycle=result.cycle,
                script_path=str(result.script_path), diff_path=str(directory / "diff.patch"),
                critique_path=str(critique_path), score=result.score, passed=result.passed,
                status=result.status.value, summary=summary, created_at=time.time(),
            )
        )
        self._cycles.update_diff(subject.phase, subject.subject_id, result.cycle, status=result.status.value, score_after=result.score)

    def abandon_pending(self, subject: LoopSubject) -> None:
        """Retire a cycle that was written but never evaluated (a crashed session a new request replaces)."""
        state = self.load(subject)
        if state.pending is None:
            return
        cycle, script_path = state.pending
        self._cycles.save_cycle(
            CycleRow(
                phase=subject.phase.value, subject=subject.subject_id, cycle=cycle, script_path=str(script_path),
                diff_path=str(script_path.parent / "diff.patch"), critique_path=None, score=None, passed=None,
                status=ABANDONED, summary="superseded before evaluation", created_at=time.time(),
            )
        )
        self._cycles.update_diff(subject.phase, subject.subject_id, cycle, status=DiffStatus.REJECTED.value, score_after=None)

    def load(self, subject: LoopSubject) -> LoopState:
        state = LoopState()
        for row in self._cycles.cycles(subject.phase, subject.subject_id):
            state.last_cycle = max(state.last_cycle, row.cycle)
            if row.status == ABANDONED:
                continue
            if row.status == PENDING or not row.critique_path:
                state.pending = (row.cycle, Path(row.script_path))
                continue
            state.results[row.cycle] = _result_from_files(row, self._layout)
        for diff in self._cycles.diffs(subject.phase, subject.subject_id):
            path = Path(diff.diff_path)
            text = path.read_text(encoding="utf-8") if path.is_file() else ""
            if diff.status == DiffStatus.REJECTED.value and text.startswith("# rejected:"):
                text = text.split("\n", 1)[1] if "\n" in text else ""
            state.history.add(
                HistoryEntry(
                    cycle=diff.cycle, status=DiffStatus(diff.status), diff=text, reason=diff.reason or "",
                    score_before=diff.score_before, score_after=diff.score_after, initial=diff.reason == "initial",
                )
            )
        return state


def _result_from_files(row: CycleRow, layout: OutputLayout) -> CycleResult:
    data = json.loads(Path(row.critique_path).read_text(encoding="utf-8"))
    evaluation_data = data.get("evaluation", {})
    report_path = Path(row.script_path).parent / "report.json"
    evaluation = Evaluation(
        ok=bool(evaluation_data.get("ok")),
        images=tuple(Path(p) for p in evaluation_data.get("images", [])),
        facts=evaluation_data.get("facts", {}),
        report=json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {},
        error=evaluation_data.get("error"),
        artifacts=evaluation_data.get("artifacts", {}),
    )
    critiques = tuple(Critique.parse(c["critic"], c, model=c.get("model", "")) for c in data.get("critiques", []))
    evidence = None
    saved = data.get("evidence")
    if isinstance(saved, dict) and (
        saved.get("workspace") == str(layout.root) and saved.get("phase") == row.phase
        and saved.get("subject") == row.subject and saved.get("cycle") == row.cycle
        and isinstance(saved.get("hashes"), dict)
    ):
        evidence = CycleEvidence(layout.root, PhaseName(row.phase), row.subject, row.cycle, saved["hashes"])
    return CycleResult(
        cycle=row.cycle,
        script_path=Path(row.script_path),
        scorecard=scorecard_from_dict(data.get("scorecard", {})),
        critiques=critiques,
        evaluation=evaluation,
        status=DiffStatus(row.status),
        phase=PhaseName(row.phase),
        evidence=evidence,
    )
