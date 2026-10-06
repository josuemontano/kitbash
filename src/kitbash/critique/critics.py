"""Visual and technical critics: actionable feedback grounded in bounded rubric results."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from attrs import frozen

from kitbash.analytics import context
from kitbash.critique.subject import CriticBrief, Evaluation
from kitbash.domain.critique import CriticKind, Critique, Edit, ScoreCard
from kitbash.domain.roles import Role
from kitbash.domain.rubric import Rubric
from kitbash.errors import LLMError
from kitbash.llm.service import LLMService

MAX_IMAGES = 8
MAX_REPORT_CHARS = 40_000


@frozen
class ReviewRequest:
    brief: CriticBrief
    evaluation: Evaluation
    script: str
    history: str
    cycle: int
    scorecard: ScoreCard
    feedback: str = ""


class Critic(Protocol):
    @property
    def kind(self) -> CriticKind: ...

    def review(self, request: ReviewRequest) -> Critique: ...


class LLMCritic:
    """Shared flow; subclasses pick the role, the template and what the model gets to see."""

    kind: CriticKind
    role: Role
    template: str

    def __init__(self, llm: LLMService, rubric: Rubric) -> None:
        self._llm = llm
        self._rubric = rubric

    def review(self, request: ReviewRequest) -> Critique:
        phase = request.brief.phase
        criteria = self._rubric.for_phase(phase, self.kind)
        if not criteria:
            return Critique(critic=self.kind.value, summary="No rubric criteria for this critic.")
        if reason := self.cannot_review(request.evaluation):
            return Critique(critic=self.kind.value, summary=f"Skipped: {reason}")
        model = self._llm.model_for(self.role, phase)

        def validate(data: Any) -> Critique:
            if not isinstance(data, Mapping) or not isinstance(data.get("summary"), str) or not isinstance(data.get("edits"), list):
                raise LLMError("Expected a JSON object with 'summary' text and an 'edits' list")
            if "scores" in data:
                raise LLMError("Critics provide feedback only; do not return rubric scores")
            return Critique.parse(self.kind.value, data, model=model)

        with context.bind(agent=f"{self.kind.value}_critic"):
            return self._llm.ask_json(
                task=f"{phase.value}.critic.{self.kind.value}",
                role=self.role,
                phase=phase,
                template=self.template,
                variables={
                    "subject": request.brief.subject,
                    "description": request.brief.description,
                    "style": request.brief.style,
                    "phase": phase.value,
                    "cycle": request.cycle,
                    "criteria": self._rubric.render(phase, self.kind),
                    "scorecard": _json(request.scorecard.to_dict()),
                    "history": request.history,
                    "feedback": request.feedback or "(none)",
                    **self.variables(request),
                },
                attachments=self.attachments(request),
                validate=validate,
            )

    def cannot_review(self, evaluation: Evaluation) -> str | None:
        return None

    def variables(self, request: ReviewRequest) -> dict[str, str]:
        return {}

    def attachments(self, request: ReviewRequest) -> Sequence[Path]:
        return ()


class VisualCritic(LLMCritic):
    kind = CriticKind.VISUAL
    role = Role.VISUAL_CRITIC
    template = "critic_visual"

    def cannot_review(self, evaluation: Evaluation) -> str | None:
        return None if any(image.is_file() for image in evaluation.images) else "the script produced no available renders"

    def variables(self, request: ReviewRequest) -> dict[str, str]:
        references = [p.name for p in request.brief.references]
        renders = [p.name for p in request.evaluation.images]
        return {
            "references": ", ".join(references) or "(no reference image: judge against the description)",
            "renders": ", ".join(renders),
            "facts": _json(request.evaluation.facts),
        }

    def attachments(self, request: ReviewRequest) -> Sequence[Path]:
        images = [*request.brief.references, *request.evaluation.images]
        return tuple(p for p in images if p.is_file())[:MAX_IMAGES]


class TechnicalCritic(LLMCritic):
    kind = CriticKind.TECHNICAL
    role = Role.TECHNICAL_CRITIC
    template = "critic_technical"

    def variables(self, request: ReviewRequest) -> dict[str, str]:
        return {
            "script": request.script,
            "facts": _json(request.evaluation.facts),
            "report": _json(request.evaluation.report)[:MAX_REPORT_CHARS],
            "error": request.evaluation.error or "(none: the script ran successfully)",
        }


@frozen
class PatchRequest:
    brief: CriticBrief
    script: str
    edits: tuple[Edit, ...]
    history: str
    api_reference: str
    error: str | None = None
    rejection: str | None = None
    cycle: int = 0


class PatchWriter:
    """Turns edits into a unified diff against the script. All code is written by the code role."""

    def __init__(self, llm: LLMService) -> None:
        self._llm = llm

    def propose(self, request: PatchRequest) -> str:
        phase = request.brief.phase
        with context.bind(agent="patch_writer"):
            return self._llm.ask_diff(
                task=f"{phase.value}.patch",
                role=Role.CODE,
                phase=phase,
                template="patch",
                variables={
                    "subject": request.brief.subject,
                    "description": request.brief.description,
                    "style": request.brief.style,
                    "phase": phase.value,
                    "script": request.script,
                    "edits": json.dumps([e.to_dict() for e in request.edits], indent=2),
                    "error": request.error or "(none)",
                    "history": request.history,
                    "api": request.api_reference,
                    "rejection": request.rejection or "(none)",
                },
                attachments=tuple(p for p in request.brief.references if p.is_file())[:2],
            )


def _json(value: Any) -> str:
    return json.dumps(value, indent=1, default=str)
