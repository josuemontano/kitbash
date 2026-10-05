"""Rubric: parsed from an editable Markdown table, used to prompt critics and to score their answers.

Table columns (header names are case-insensitive): ``criterion | weight | pass condition | applies to``
and an optional ``critic`` column (``visual``, ``technical`` or ``both``). Text in backticks inside the
pass condition is a machine check over measured facts, e.g. ``usd_roundtrip_score >= 0.85``.
"""

import ast
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

from attrs import frozen

from kitbash.domain.critique import CardEntry, CriticKind, Critique, ScoreCard
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
        return bool(_eval(self.tree.body, facts))


@frozen
class Criterion:
    id: str
    name: str
    weight: float
    pass_condition: str
    applies_to: frozenset[PhaseName]
    critics: frozenset[CriticKind]
    check: Check | None = None

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
        return cls(criteria=tuple(criteria), source=source)

    def for_phase(self, phase: PhaseName, critic: CriticKind | None = None) -> tuple[Criterion, ...]:
        return tuple(c for c in self.criteria if c.applies(phase, critic))

    def render(self, phase: PhaseName, critic: CriticKind | None = None) -> str:
        """Markdown table of the criteria a critic must score, for prompts."""
        lines = ["| id | criterion | weight | pass condition |", "|---|---|---|---|"]
        lines += [f"| {c.id} | {c.name} | {c.weight:g} | {c.pass_condition} |" for c in self.for_phase(phase, critic)]
        return "\n".join(lines)

    def score(
        self,
        phase: PhaseName,
        critiques: Sequence[Critique],
        facts: Mapping[str, Any],
        *,
        threshold: float,
        require_all_pass: bool,
    ) -> ScoreCard:
        entries = tuple(_score_criterion(c, critiques, facts, threshold) for c in self.for_phase(phase))
        scored = [e for e in entries if e.score is not None]
        weight = sum(e.weight for e in scored)
        overall = sum(e.weight * e.score for e in scored) / weight if weight else 0.0
        all_pass = all(e.passed is True for e in entries)
        passed = bool(scored) and overall >= threshold and (all_pass or not require_all_pass)
        return ScoreCard(entries=entries, overall=overall, passed=passed, threshold=threshold, facts=dict(facts))


def _score_criterion(criterion: Criterion, critiques: Sequence[Critique], facts: Mapping[str, Any], threshold: float) -> CardEntry:
    opinions = [
        (critique.critic, score)
        for critique in critiques
        if (score := critique.score_for(criterion.id)) is not None and CriticKind(critique.critic) in criterion.critics
    ]
    notes = tuple(f"{critic}: {s.notes}" for critic, s in opinions if s.notes)
    base = dict(criterion_id=criterion.id, name=criterion.name, weight=criterion.weight, notes=notes)
    verdict = criterion.check.evaluate(facts) if criterion.check else None
    if verdict is not None:
        return CardEntry(**base, score=1.0 if verdict else 0.0, passed=verdict, decided_by="check")
    assessed = [(critic, s) for critic, s in opinions if s.score is not None]
    missing = criterion.critics - {CriticKind(critic) for critic, _ in assessed}
    reported = {CriticKind(critic): s for critic, s in opinions}
    base["notes"] = notes + tuple(
        f"{critic.value}: criterion unassessed (no score provided)"
        for critic in sorted(missing)
        if critic not in reported or not reported[critic].notes
    )
    if not assessed:
        return CardEntry(**base, score=None, passed=None)
    score = fmean(s.score for _, s in assessed)
    flags = [s.passed if s.passed is not None else s.score >= threshold for _, s in assessed]
    # A known failure remains a failure; missing evidence must never become either a pass or
    # an invented negative assessment. Machine checks above remain authoritative.
    passed = False if not all(flags) else (None if missing else True)
    return CardEntry(**base, score=score, passed=passed, decided_by="critics")


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
    )


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
