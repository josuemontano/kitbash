"""What the critic loop needs from a phase: a way to evaluate a script and a brief for the critics."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from attrs import field, frozen

from kitbash.domain.phases import PhaseName


@frozen
class Evaluation:
    ok: bool
    images: tuple[Path, ...] = ()
    facts: Mapping[str, Any] = field(factory=dict)
    report: Mapping[str, Any] = field(factory=dict)
    error: str | None = None
    artifacts: Mapping[str, str] = field(factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "error": self.error,
            "images": [str(p) for p in self.images],
            "facts": dict(self.facts),
            "artifacts": dict(self.artifacts),
        }


@frozen
class CriticBrief:
    """Static context shared by every critic and the patch writer."""

    phase: PhaseName
    subject: str  # e.g. "asset 'oak_chair' (Oak dining chair)" or "scene layout"
    description: str  # what the result must be
    references: tuple[Path, ...] = ()
    style: str = "photorealistic"


class LoopSubject(Protocol):
    @property
    def phase(self) -> PhaseName: ...

    @property
    def subject_id(self) -> str:
        """Empty for phase-level loops, the asset id in modelling."""
        ...

    def brief(self) -> CriticBrief: ...

    def api_reference(self) -> str:
        """Documentation of the helpers the script may use (given to the patch writer)."""
        ...

    def evaluate(self, script: Path, cycle_dir: Path, cycle: int) -> Evaluation: ...
