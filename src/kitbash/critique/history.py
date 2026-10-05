"""Diff history across critic cycles: statuses, repeat detection and score stalls."""

from collections.abc import Sequence
from enum import StrEnum

from attrs import Factory, define, evolve, field, frozen

from kitbash.infra.patching import fingerprint as diff_fingerprint
from kitbash.infra.patching import similarity

REPEAT_SIMILARITY = 0.9


class DiffStatus(StrEnum):
    APPLIED = "applied"  # applied, not yet evaluated
    KEPT = "kept"  # evaluated and kept
    REVERTED = "reverted"  # evaluated, made things worse, rolled back
    REJECTED = "rejected"  # never applied (did not apply, repeated, or broke syntax)


@frozen
class HistoryEntry:
    cycle: int
    status: DiffStatus
    diff: str
    reason: str = ""
    score_before: float | None = None
    score_after: float | None = None
    initial: bool = False  # the first script of the loop, not a patch
    fingerprint: str = field(default=Factory(lambda self: diff_fingerprint(self.diff), takes_self=True))

    @property
    def forbidden(self) -> bool:
        return self.status in (DiffStatus.REVERTED, DiffStatus.REJECTED)


@define
class DiffHistory:
    entries: list[HistoryEntry] = field(factory=list)

    def add(self, entry: HistoryEntry) -> None:
        self.entries.append(entry)

    def settle(self, cycle: int, status: DiffStatus, score_after: float | None) -> None:
        """Record the outcome of the diff applied for ``cycle``."""
        for index in range(len(self.entries) - 1, -1, -1):
            entry = self.entries[index]
            if entry.cycle == cycle and entry.status is DiffStatus.APPLIED:
                self.entries[index] = evolve(entry, status=status, score_after=score_after)
                return

    def score_before(self, cycle: int) -> float | None:
        """Score of the script the diff applied for ``cycle`` was patched from."""
        for entry in reversed(self.entries):
            if entry.cycle == cycle and entry.status is not DiffStatus.REJECTED:
                return entry.score_before
        return None

    def find_repeat(self, diff: str, *, include_kept: bool = True) -> HistoryEntry | None:
        """An earlier diff that makes the same change (same fingerprint or near-identical changed lines)."""
        new_fingerprint = diff_fingerprint(diff)
        for entry in self.entries:
            if entry.initial:
                continue
            if not include_kept and not entry.forbidden:
                continue
            if entry.fingerprint == new_fingerprint or similarity(entry.diff, diff) >= REPEAT_SIMILARITY:
                return entry
        return None

    def render(self, *, max_chars: int = 30_000) -> str:
        """Chronological history for prompts; the newest diffs are kept whole, older ones truncated."""
        if not self.entries:
            return "(no previous cycles)"
        blocks: list[str] = []
        budget = max_chars
        for entry in reversed(self.entries):
            header = (
                f"### cycle {entry.cycle:02d}: {entry.status.value}"
                f"{f' ({entry.reason})' if entry.reason else ''}"
                f", score {_fmt(entry.score_before)} -> {_fmt(entry.score_after)}"
            )
            body = entry.diff if len(entry.diff) <= budget else entry.diff[:budget] + "\n... (truncated)"
            budget = max(0, budget - len(body))
            blocks.append(f"{header}\n```diff\n{body.rstrip()}\n```" if body else header)
        return "\n\n".join(reversed(blocks))


def _fmt(score: float | None) -> str:
    return "n/a" if score is None else f"{score:.2f}"


@define
class ScoreTrend:
    """Tracks evaluated scores to detect stalls: no improvement above ``epsilon`` for ``window`` cycles."""

    window: int
    epsilon: float
    scores: list[float] = field(factory=list)

    def add(self, score: float) -> None:
        self.scores.append(score)

    @property
    def best(self) -> float:
        return max(self.scores, default=0.0)

    def stalled(self) -> bool:
        if len(self.scores) <= self.window:
            return False
        before = max(self.scores[: -self.window])
        return max(self.scores[-self.window :]) <= before + self.epsilon

    @classmethod
    def of(cls, scores: Sequence[float], window: int, epsilon: float) -> ScoreTrend:
        return cls(window=window, epsilon=epsilon, scores=list(scores))
