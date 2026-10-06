"""Convert collected evidence into bounded decisions without collecting it again."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from attrs import frozen

from kitbash.critique.subject import CriticBrief, Evaluation
from kitbash.domain.critique import CriticKind
from kitbash.domain.evaluation import CriterionAssessment, DecisionAdapter, DecisionError, DecisionQuestion, EvaluationResult
from kitbash.domain.rubric import Rubric
from kitbash.errors import KitbashError


@frozen
class EvaluationCase:
    evaluation_id: str
    iteration: int
    brief: CriticBrief
    script: str
    history: str = ""
    previous: tuple[Mapping[str, Any], ...] = ()
    feedback: str = ""


class Evaluator(Protocol):
    def evaluate(self, case: EvaluationCase, rubric: Rubric, evidence: Evaluation) -> EvaluationResult: ...


@frozen
class ClefFlashEvaluator:
    adapter: DecisionAdapter

    def evaluate(self, case: EvaluationCase, rubric: Rubric, evidence: Evaluation) -> EvaluationResult:
        images: list[Path] = []
        indices: dict[Path, int] = {}
        manifest: list[dict[str, Any]] = []

        def add_image(path: Path, role: str, name: str = "") -> bool:
            available = path.is_file()
            index = None
            if available:
                identity = path.resolve()
                if identity not in indices:
                    indices[identity] = len(images)
                    images.append(path)
                index = indices[identity]
            manifest.append({"path": str(path), "role": role, "name": name, "available": available, "image_index": index})
            return available

        for path in case.brief.references:
            add_image(path, "reference")
        render_available = False
        for path in evidence.images:
            render_available = add_image(path, "render") or render_available
        for name, value in evidence.artifacts.items():
            path = Path(value)
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif"}:
                render_available = add_image(path, "artifact", name) or render_available

        state = {
            "evaluation_id": case.evaluation_id,
            "iteration": case.iteration,
            "brief": {
                "phase": case.brief.phase.value,
                "subject": case.brief.subject,
                "description": case.brief.description,
                "style": case.brief.style,
                "references": [str(path) for path in case.brief.references],
            },
            "script": case.script,
            "history": case.history,
            "previous": [dict(result) for result in case.previous],
            "feedback": case.feedback,
            "evidence": {
                "ok": evidence.ok,
                "error": evidence.error,
                "facts": dict(evidence.facts),
                "report": dict(evidence.report),
                "artifacts": dict(evidence.artifacts),
                "images": [str(path) for path in evidence.images],
            },
            "image_manifest": manifest,
        }
        criteria = rubric.for_phase(case.brief.phase)
        results: dict[str, CriterionAssessment] = {}
        questions = []
        for criterion in criteria:
            verdict = criterion.check.evaluate(evidence.facts) if criterion.check else None
            if verdict is not None:
                results[criterion.id] = CriterionAssessment(criterion.id, float(verdict), confidence=1.0, source="check")
                continue
            error = ""
            if not evidence.ok or evidence.error:
                error = "Evaluation evidence unavailable or failed"
            elif CriticKind.VISUAL in criterion.critics and not render_available:
                error = "Rendered visual evidence unavailable"
            if error:
                results[criterion.id] = CriterionAssessment(criterion.id, None, error=error)
                continue
            levels = criterion.levels
            questions.append(DecisionQuestion(
                criterion_id=criterion.id,
                instructions=f"{criterion.name}\nPass condition: {criterion.pass_condition}",
                values=tuple(value for value, _ in levels) if levels else (0.0, 1.0),
                descriptions=tuple(description for _, description in levels) if levels else (
                    f"Pass condition not met: {criterion.pass_condition}",
                    f"Pass condition met: {criterion.pass_condition}",
                ),
                binary=not levels,
            ))
        result = EvaluationResult(())
        if questions:
            try:
                result = self.adapter.evaluate(state, tuple(questions), tuple(images))
            except KitbashError as exc:
                diagnostic = str(exc) if isinstance(exc, DecisionError) else "Decision provider unavailable"
                for question in questions:
                    results[question.criterion_id] = CriterionAssessment(question.criterion_id, None, error=diagnostic)
            else:
                requested = {question.criterion_id for question in questions}
                for assessment in result.criteria:
                    if assessment.criterion_id not in requested:
                        continue
                    if assessment.criterion_id in results:
                        results[assessment.criterion_id] = CriterionAssessment(assessment.criterion_id, None, error="Duplicate decision answer")
                    else:
                        results[assessment.criterion_id] = assessment
        return EvaluationResult(
            criteria=tuple(
                results.get(criterion.id, CriterionAssessment(criterion.id, None, error="Decision answer unavailable"))
                for criterion in criteria
            ),
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
