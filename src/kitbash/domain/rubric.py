"""Editable criteria and application-owned decisions over bounded assessments.

Required columns: ``criterion | weight | pass condition | applies to``. Optional
``critic``, JSON ``levels`` and raw-scale ``threshold`` columns preserve explicit
rubric meanings. Backtick expressions are authoritative checks over measured facts.
"""

import ast
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from attrs import evolve, frozen

from kitbash.domain.critique import CardEntry, CriticKind, ScoreCard
from kitbash.domain.evaluation import CriterionAssessment
from kitbash.domain.phases import PhaseName
from kitbash.errors import KitbashError
from kitbash.naming import slugify

REQUIRED_COLUMNS = ("criterion", "weight", "pass condition", "applies to")
_BACKTICKS = re.compile(r"`([^`]+)`")
_CELL_SPLIT = re.compile(r"(?<!\\)\|")


class RubricError(KitbashError):
    """The rubric file cannot be parsed."""


@frozen
class Check:
    """A boolean expression over facts: comparisons, ``and``, ``or``, ``not`` and numbers."""

    source: str
    tree: ast.Expression

    @classmethod
    def compile(cls, source: str) -> Check:
        try:
            tree = ast.parse(source.strip(), mode="eval")
        except SyntaxError as exc:
            raise RubricError(f"Invalid check expression `{source}`: {exc.msg}") from exc
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_NODES):
                raise RubricError(f"Unsupported syntax {type(node).__name__} in check `{source}`")
        return cls(source=source.strip(), tree=tree)

    @property
    def names(self) -> frozenset[str]:
        return frozenset(n.id for n in ast.walk(self.tree) if isinstance(n, ast.Name) and n.id not in _CONSTANTS)

    def evaluate(self, facts: Mapping[str, Any]) -> bool | None:
        """True/False, or None when a fact it needs was not measured."""
        if any(facts.get(name) is None for name in self.names):
            return None
        try:
            return bool(_eval(self.tree.body, facts))
        except (TypeError, ValueError, ArithmeticError):
            return None


@frozen
class Criterion:
    id: str
    name: str
    weight: float
    pass_condition: str
    applies_to: frozenset[PhaseName]
    critics: frozenset[CriticKind]
    check: Check | None = None
    levels: tuple[tuple[float, str], ...] = ()
    threshold: float | None = None

    def applies(self, phase: PhaseName, critic: CriticKind | None = None) -> bool:
        return phase in self.applies_to and (critic is None or critic in self.critics)


@frozen
class Rubric:
    criteria: tuple[Criterion, ...]
    source: str = ""

    @classmethod
    def load(cls, path: Path) -> Rubric:
        if not path.is_file():
            raise RubricError(f"Rubric file not found: {path}")
        return cls.parse(path.read_text(encoding="utf-8"), source=str(path))

    @classmethod
    def parse(cls, text: str, *, source: str = "") -> Rubric:
        header, rows = _find_table(text)
        columns = {name: index for index, name in enumerate(header)}
        missing = [c for c in REQUIRED_COLUMNS if c not in columns]
        if missing:
            raise RubricError(f"Rubric table is missing columns: {', '.join(missing)}")
        criteria = [_parse_row(row, columns, line) for line, row in rows]
        ids = [c.id for c in criteria]
        if duplicates := sorted({i for i in ids if ids.count(i) > 1}):
            raise RubricError(f"Duplicate rubric criteria: {', '.join(duplicates)}")
        if not criteria:
            raise RubricError("Rubric has no criteria")
        for phase in PhaseName:
            applicable = [c for c in criteria if c.applies(phase)]
            if applicable and sum(c.weight for c in applicable) <= 0:
                raise RubricError(f"Rubric requires positive total applicable weight for {phase.value}")
        return cls(criteria=tuple(criteria), source=source)

    def for_phase(self, phase: PhaseName, critic: CriticKind | None = None) -> tuple[Criterion, ...]:
        return tuple(c for c in self.criteria if c.applies(phase, critic))

    def render(self, phase: PhaseName, critic: CriticKind | None = None) -> str:
        """Criteria and explicit scale meanings for feedback prompts."""
        lines = ["| id | criterion | weight | pass condition | levels | threshold |", "|---|---|---|---|---|---|"]
        lines += [
            f"| {c.id} | {c.name} | {c.weight:g} | {c.pass_condition} | "
            f"{json.dumps(dict(c.levels)) if c.levels else ''} | {c.threshold if c.threshold is not None else ''} |"
            for c in self.for_phase(phase, critic)
        ]
        return "\n".join(lines)

    def score(
        self,
        phase: PhaseName,
        assessments: Sequence[CriterionAssessment],
        facts: Mapping[str, Any],
        *,
        threshold: float,
        require_all_pass: bool,
        confidence_threshold: float = 0.7,
        previous: ScoreCard | None = None,
        regression_epsilon: float = 0.02,
    ) -> ScoreCard:
        if not _in_range(threshold, 0.0, 1.0) or not _in_range(confidence_threshold, 0.0, 1.0):
            raise RubricError("Score and confidence thresholds must be finite numbers in [0, 1]")
        if not _in_range(regression_epsilon, 0.0, 1.0):
            raise RubricError("Regression epsilon must be a finite number in [0, 1]")
        criteria = self.for_phase(phase)
        if any(not _in_range(c.weight, 0.0, math.inf) for c in criteria) or sum(c.weight for c in criteria) <= 0:
            raise RubricError(f"Rubric requires finite nonnegative weights and positive total applicable weight for {phase.value}")
        by_id: dict[str, CriterionAssessment] = {}
        for assessment in assessments:
            if assessment.criterion_id in by_id:
                by_id[assessment.criterion_id] = CriterionAssessment(assessment.criterion_id, None, error="Duplicate assessment")
            else:
                by_id[assessment.criterion_id] = assessment
        old = {e.criterion_id: e for e in previous.entries} if previous else {}
        entries = []
        for criterion in criteria:
            entry = _score_criterion(criterion, by_id.get(criterion.id), facts, threshold, confidence_threshold)
            prior = old.get(criterion.id)
            delta = entry.score - prior.score if prior and entry.score is not None and prior.score is not None else None
            regressed = bool(prior and (
                (prior.passed is not None and delta is not None and delta < -regression_epsilon)
                or (prior.passed is True and entry.passed is not True)
            ))
            entries.append(evolve(entry, delta=delta, regressed=regressed))
        scored = [e for e in entries if e.score is not None]
        weight = sum(e.weight for e in scored)
        overall = sum(e.weight * e.score for e in scored) / weight if weight else 0.0
        certain = all(e.passed is not None for e in entries)
        all_pass = all(e.passed is True for e in entries)
        passed = bool(scored) and certain and overall >= threshold and (all_pass or not require_all_pass)
        passed = passed and not any(e.regressed for e in entries)
        return ScoreCard(entries=tuple(entries), overall=overall, passed=passed, threshold=threshold, facts=dict(facts))


def _in_range(value: Any, low: float, high: float) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and low <= value <= high


def _score_criterion(
    criterion: Criterion,
    assessment: CriterionAssessment | None,
    facts: Mapping[str, Any],
    threshold: float,
    confidence_threshold: float,
) -> CardEntry:
    low, high = (criterion.levels[0][0], criterion.levels[-1][0]) if criterion.levels else (0.0, 1.0)
    raw_threshold = criterion.threshold if criterion.threshold is not None else low + threshold * (high - low)
    base = dict(criterion_id=criterion.id, name=criterion.name, weight=criterion.weight, threshold=raw_threshold)
    verdict = criterion.check.evaluate(facts) if criterion.check else None
    if verdict is not None:
        return CardEntry(
            **base, score=float(verdict), raw_score=high if verdict else low, passed=verdict,
            confidence=1.0, decided_by="check",
        )
    if assessment is None:
        return CardEntry(**base, score=None, passed=None, notes=("Criterion assessment unavailable",))
    details = dict(confidence=assessment.confidence, probabilities=dict(assessment.probabilities), decided_by=assessment.source)
    if assessment.error or not _in_range(assessment.value, low, high):
        return CardEntry(**base, **details, score=None, passed=None, notes=(assessment.error or "Invalid or missing raw score",))
    score = (assessment.value - low) / (high - low)
    confident = _in_range(assessment.confidence, confidence_threshold, 1.0)
    return CardEntry(
        **base, **details, score=score, raw_score=assessment.value,
        passed=assessment.value >= raw_threshold if confident else None,
        notes=() if confident else ("Confidence unavailable or below threshold",),
    )


def _find_table(text: str) -> tuple[list[str], list[tuple[int, list[str]]]]:
    lines = text.splitlines()
    for start, line in enumerate(lines):
        cells = _cells(line)
        if cells and {"criterion", "weight"} <= {c.lower() for c in cells}:
            header = [c.lower() for c in cells]
            rows: list[tuple[int, list[str]]] = []
            for number, row_line in enumerate(lines[start + 1 :], start=start + 2):
                if not row_line.strip().startswith("|"):
                    break
                row = _cells(row_line)
                if all(set(cell) <= set("-: ") for cell in row):
                    continue
                rows.append((number, row))
            return header, rows
    raise RubricError("No rubric table found (expected a Markdown table with 'criterion' and 'weight' columns)")


def _cells(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return []
    parts = _CELL_SPLIT.split(stripped.strip("|"))
    return [part.strip().replace("\\|", "|") for part in parts]


def _parse_row(row: list[str], columns: Mapping[str, int], line: int) -> Criterion:
    def cell(name: str, default: str = "") -> str:
        index = columns.get(name)
        return row[index] if index is not None and index < len(row) else default

    name = cell("criterion")
    if not name:
        raise RubricError(f"Rubric line {line}: empty criterion")
    try:
        weight = float(cell("weight"))
    except ValueError:
        raise RubricError(f"Rubric line {line}: weight must be a number") from None
    if not _in_range(weight, 0.0, math.inf):
        raise RubricError(f"Rubric line {line}: weight must be finite and nonnegative")
    levels = _levels(cell("levels"), line)
    threshold = None
    if cell("threshold"):
        try:
            threshold = float(cell("threshold"))
        except ValueError:
            raise RubricError(f"Rubric line {line}: threshold must be a number") from None
        low, high = (levels[0][0], levels[-1][0]) if levels else (0.0, 1.0)
        if not _in_range(threshold, low, high):
            raise RubricError(f"Rubric line {line}: threshold must be finite and within the raw scale")
    condition = cell("pass condition")
    checks = _BACKTICKS.findall(condition)
    return Criterion(
        id=slugify(name),
        name=name,
        weight=weight,
        pass_condition=condition,
        applies_to=_phases(cell("applies to"), line),
        critics=_critics(cell("critic", "both"), line),
        check=Check.compile(checks[0]) if checks else None,
        levels=levels,
        threshold=threshold,
    )


def _levels(text: str, line: int) -> tuple[tuple[float, str], ...]:
    if not text:
        return ()
    try:
        # Preserve pairs so duplicate JSON keys cannot silently replace meanings.
        pairs = json.loads(text, object_pairs_hook=lambda pairs: pairs)
        if not text.lstrip().startswith("{") or not isinstance(pairs, list) or not 2 <= len(pairs) <= 26:
            raise ValueError
        levels = []
        for key, description in pairs:
            value = float(key)
            if not math.isfinite(value) or not isinstance(description, str) or not description.strip():
                raise ValueError
            levels.append((value, description))
        if len({value for value, _ in levels}) != len(levels):
            raise ValueError
        return tuple(sorted(levels))
    except (ValueError, TypeError):
        raise RubricError(f"Rubric line {line}: levels must be a JSON object with 2..26 unique finite numeric keys and nonempty descriptions") from None


def _phases(text: str, line: int) -> frozenset[PhaseName]:
    names = [part.strip().lower() for part in re.split(r"[,/;]| and ", text) if part.strip()]
    if not names or "all" in names:
        return frozenset(PhaseName)
    try:
        return frozenset(PhaseName(n) for n in names)
    except ValueError:
        raise RubricError(f"Rubric line {line}: unknown phase in {text!r}") from None


def _critics(text: str, line: int) -> frozenset[CriticKind]:
    value = text.strip().lower() or "both"
    if value == "both":
        return frozenset(CriticKind)
    try:
        return frozenset(CriticKind(part.strip()) for part in value.split(","))
    except ValueError:
        raise RubricError(f"Rubric line {line}: critic must be visual, technical or both") from None


# -- safe evaluation ---------------------------------------------------------------------------------

_CONSTANTS = {"true": True, "false": False, "True": True, "False": False}
_ALLOWED_NODES = (
    ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.Compare,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.Name, ast.Load, ast.Constant,
)
_COMPARATORS = {
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
}


def _eval(node: ast.AST, facts: Mapping[str, Any]) -> Any:
    match node:
        case ast.Constant(value=value):
            return value
        case ast.Name(id=name):
            return _CONSTANTS[name] if name in _CONSTANTS else facts[name]
        case ast.UnaryOp(op=ast.Not(), operand=operand):
            return not _eval(operand, facts)
        case ast.UnaryOp(op=ast.USub(), operand=operand):
            return -_eval(operand, facts)
        case ast.BoolOp(op=ast.And(), values=values):
            return all(_eval(v, facts) for v in values)
        case ast.BoolOp(op=ast.Or(), values=values):
            return any(_eval(v, facts) for v in values)
        case ast.Compare(left=left, ops=ops, comparators=comparators):
            return _compare(_eval(left, facts), ops, [_eval(c, facts) for c in comparators])
    raise RubricError(f"Cannot evaluate {ast.dump(node)}")


def _compare(left: Any, ops: Iterable[ast.cmpop], rights: list[Any]) -> bool:
    for op, right in zip(ops, rights, strict=True):
        if not _COMPARATORS[type(op)](left, right):
            return False
        left = right
    return True
