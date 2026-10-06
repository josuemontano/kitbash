"""Critic feedback and application-owned rubric decisions."""

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

from attrs import field, frozen


class CriticKind(StrEnum):
    VISUAL = "visual"
    TECHNICAL = "technical"


@frozen
class Edit:
    """One requested change to the script, in words. The code writer turns edits into a diff."""

    instruction: str
    target: str = ""
    issue: str = ""
    priority: str = "medium"
    source: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "instruction": self.instruction,
            "target": self.target,
            "issue": self.issue,
            "priority": self.priority,
            "source": self.source,
        }


@frozen
class Critique:
    critic: str
    summary: str
    edits: tuple[Edit, ...] = ()
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "critic": self.critic,
            "model": self.model,
            "summary": self.summary,
            "edits": [e.to_dict() for e in self.edits],
        }

    @classmethod
    def parse(cls, critic: str, data: Mapping[str, Any], *, model: str = "") -> Critique:
        edits = tuple(
            Edit(
                instruction=str(e.get("instruction") or e.get("change") or e.get("fix") or ""),
                target=str(e.get("target", "")),
                issue=str(e.get("issue", "")),
                priority=str(e.get("priority", "medium")),
                source=critic,
            )
            for e in data.get("edits") or []
            if isinstance(e, Mapping) and (e.get("instruction") or e.get("change") or e.get("fix"))
        )
        return cls(critic=critic, summary=str(data.get("summary", "")), edits=edits, model=model)


@frozen
class CardEntry:
    criterion_id: str
    name: str
    weight: float
    score: float | None
    passed: bool | None
    notes: tuple[str, ...] = ()
    decided_by: str = ""
    raw_score: float | None = None
    confidence: float | None = None
    probabilities: Mapping[str, float] = field(factory=dict)
    threshold: float | None = None
    delta: float | None = None
    regressed: bool = False

    @property
    def status(self) -> str:
        if self.passed is None:
            return "unassessed"
        return "passed" if self.passed else "failed"


@frozen
class ScoreCard:
    entries: tuple[CardEntry, ...]
    overall: float
    passed: bool
    threshold: float
    facts: Mapping[str, Any] = field(factory=dict)

    @property
    def status(self) -> str:
        if self.regressions():
            return "fail"
        if self.unassessed():
            return "uncertain"
        return "pass" if self.passed else "fail"

    def regressions(self) -> Sequence[CardEntry]:
        return [e for e in self.entries if e.regressed]

    def failing(self) -> Sequence[CardEntry]:
        return [e for e in self.entries if e.passed is False]

    def unassessed(self) -> Sequence[CardEntry]:
        return [e for e in self.entries if e.passed is None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall,
            "passed": self.passed,
            "status": self.status,
            "threshold": self.threshold,
            "criteria": {
                e.criterion_id: {
                    "name": e.name,
                    "weight": e.weight,
                    "score": e.score,
                    "raw_score": e.raw_score,
                    "confidence": e.confidence,
                    "probabilities": dict(e.probabilities),
                    "threshold": e.threshold,
                    "delta": e.delta,
                    "regressed": e.regressed,
                    "pass": e.passed,
                    "status": e.status,
                    "decided_by": e.decided_by,
                    "notes": list(e.notes),
                }
                for e in self.entries
            },
            "facts": dict(self.facts),
        }


def scorecard_from_dict(data: Mapping[str, Any]) -> ScoreCard:
    entries = tuple(
        CardEntry(
            criterion_id=cid,
            name=str(item.get("name", cid)),
            weight=float(item.get("weight", 1.0)),
            score=item.get("score"),
            passed=item.get("pass"),
            notes=tuple(item.get("notes", ())),
            decided_by=str(item.get("decided_by", "")),
            raw_score=item.get("raw_score"),
            confidence=item.get("confidence"),
            probabilities=dict(item.get("probabilities", {})),
            threshold=item.get("threshold"),
            delta=item.get("delta"),
            regressed=bool(item.get("regressed", False)),
        )
        for cid, item in (data.get("criteria") or {}).items()
    )
    return ScoreCard(
        entries=entries,
        overall=float(data.get("overall", 0.0)),
        passed=bool(data.get("passed", False)),
        threshold=float(data.get("threshold", 0.0)),
        facts=dict(data.get("facts", {})),
    )
