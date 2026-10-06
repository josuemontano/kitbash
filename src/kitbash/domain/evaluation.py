"""Bounded rubric decisions, independent of any provider transport or SDK."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from attrs import field, frozen

from kitbash.errors import KitbashError


class DecisionError(KitbashError):
    """Provider failure whose message is safe to persist (no payloads or secrets)."""


@frozen
class DecisionQuestion:
    criterion_id: str
    instructions: str
    values: tuple[float, ...]
    descriptions: tuple[str, ...]
    binary: bool = False


@frozen
class CriterionAssessment:
    """A raw rubric value, not a pass decision.

    Binary confidence is derived distribution concentration (1 - H / ln(2)),
    not a probability of correctness. Missing confidence cannot establish a pass.
    Probability keys are raw ordinal values or ``false``/``true`` for binary decisions.
    """

    criterion_id: str
    value: float | None
    confidence: float | None = None
    probabilities: Mapping[str, float] = field(factory=dict)
    source: str = "clef-flash"
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion_id": self.criterion_id,
            "value": self.value,
            "confidence": self.confidence,
            "probabilities": dict(self.probabilities),
            "source": self.source,
            "error": self.error,
        }


@frozen
class EvaluationResult:
    criteria: tuple[CriterionAssessment, ...]
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "criteria": [criterion.to_dict() for criterion in self.criteria],
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


class DecisionAdapter(Protocol):
    def evaluate(
        self,
        state: Mapping[str, Any],
        questions: tuple[DecisionQuestion, ...],
        images: tuple[Path, ...],
    ) -> EvaluationResult: ...
