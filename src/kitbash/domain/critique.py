"""Critic output and aggregated scores."""

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

from attrs import field, frozen


class CriticKind(StrEnum):
    VISUAL = "visual"
    TECHNICAL = "technical"


@frozen
class CriterionScore:
    criterion_id: str
    score: float | None
    passed: bool | None
    notes: str = ""


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
    scores: tuple[CriterionScore, ...] = ()
    edits: tuple[Edit, ...] = ()
    model: str = ""

    def score_for(self, criterion_id: str) -> CriterionScore | None:
        return next((s for s in self.scores if s.criterion_id == criterion_id), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "critic": self.critic,
            "model": self.model,
            "summary": self.summary,
            "scores": {s.criterion_id: {"score": s.score, "pass": s.passed, "notes": s.notes} for s in self.scores},
            "edits": [e.to_dict() for e in self.edits],
        }

    @classmethod
    def parse(cls, critic: str, data: Mapping[str, Any], *, model: str = "") -> Critique:
        raw_scores = data.get("scores") or {}
        if isinstance(raw_scores, list):
            raw_scores = {str(s.get("criterion") or s.get("id")): s for s in raw_scores if isinstance(s, Mapping)}
        scores = tuple(_parse_score(cid, value) for cid, value in raw_scores.items())
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
        return cls(critic=critic, summary=str(data.get("summary", "")), scores=scores, edits=edits, model=model)


def _parse_score(criterion_id: str, value: Any) -> CriterionScore:
    if isinstance(value, Mapping):
        score, passed, notes = value.get("score"), value.get("pass", value.get("passed")), str(value.get("notes", ""))
    else:
        score, passed, notes = value, None, ""
    return CriterionScore(
        criterion_id=criterion_id,
        score=_unit_score(score),
        passed=passed if isinstance(passed, bool) else None,
        notes=notes,
    )


def _unit_score(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number > 1.0:  # tolerate 0-10 or 0-100 scales
        number = number / 10.0 if number <= 10.0 else number / 100.0
    return min(max(number, 0.0), 1.0)


@frozen
class CardEntry:
    criterion_id: str
    name: str
    weight: float
    score: float | None
    passed: bool | None
    notes: tuple[str, ...] = ()
    decided_by: str = ""  # "check", "critics" or "" when unscored


@frozen
class ScoreCard:
    entries: tuple[CardEntry, ...]
    overall: float
    passed: bool
    threshold: float
    facts: Mapping[str, Any] = field(factory=dict)

    def failing(self) -> Sequence[CardEntry]:
        return [e for e in self.entries if e.passed is False]

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": round(self.overall, 4),
            "passed": self.passed,
            "threshold": self.threshold,
            "criteria": {
                e.criterion_id: {
                    "name": e.name,
                    "weight": e.weight,
                    "score": None if e.score is None else round(e.score, 4),
                    "pass": e.passed,
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
