"""Ollama's local System One decision API (Clef vision requires Ollama >=0.35.1).

Wire schema and body limits: https://docs.ollama.com/api/systemone
Examples and model requirements: https://docs.ollama.com/capabilities/decision
The caller owns the HTTP client, its timeout, and its lifecycle. No evidence is
truncated, no generation controls are sent, and requests are not retried here.
"""

import base64
import json
import math
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import httpx

from kitbash.domain.evaluation import CriterionAssessment, DecisionError, DecisionQuestion, EvaluationResult

_MAX_QUESTIONS = 64
_TEXT_LIMIT = 64 * 1024
_IMAGE_LIMIT = 32 * 1024 * 1024
# The official examples round probabilities and scores to four decimal places.
_ROUNDING = 0.00005


def _number(value: Any, minimum: float, maximum: float) -> float:
    if type(value) not in (int, float):
        raise ValueError("Invalid numeric answer")
    try:
        result = float(value)
    except OverflowError:
        raise ValueError("Invalid numeric answer") from None
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError("Answer outside its scale")
    return result


def _question_payload(question: DecisionQuestion) -> dict[str, Any]:
    if not isinstance(question.criterion_id, str) or not question.criterion_id.strip():
        raise ValueError("Question identifier must not be blank")
    if not isinstance(question.instructions, str) or not question.instructions.strip():
        raise ValueError("Question instructions must not be blank")
    if type(question.binary) is not bool:
        raise ValueError("Invalid question type")
    if not 2 <= len(question.values) <= 26 or len(question.values) != len(question.descriptions):
        raise ValueError("Questions require 2 to 26 matching levels")
    values = tuple(_number(value, -math.inf, math.inf) for value in question.values)
    if any(a >= b for a, b in pairwise(values)):
        raise ValueError("Question levels must be strictly increasing")
    if any(not isinstance(text, str) or not text.strip() for text in question.descriptions):
        raise ValueError("Question descriptions must not be blank")
    if question.binary and values != (0.0, 1.0):
        raise ValueError("Binary questions require the zero-to-one scale")
    return {
        "type": "noul" if question.binary else "score",
        "instructions": question.instructions,
        "criteria": (
            dict(zip(("false", "true"), question.descriptions, strict=True))
            if question.binary else list(question.descriptions)
        ),
    }


def _assessment(question: DecisionQuestion, answer: Any) -> CriterionAssessment:
    try:
        if answer is None:
            raise ValueError("Missing criterion answer")
        if not isinstance(answer, dict):
            raise ValueError("Invalid criterion answer")
        expected_type = "noul" if question.binary else "score"
        if answer.get("type") != expected_type:
            raise ValueError("Unexpected criterion answer type")
        if question.binary:
            probability = _number(answer.get("noul"), 0.0, 1.0)
            probabilities = {"false": 1.0 - probability, "true": probability}
            # Noul supplies no confidence. This is derived distribution
            # concentration, NOT calibrated probability of being correct.
            entropy = -math.fsum(p * math.log(p) for p in probabilities.values() if p > 0)
            confidence = min(1.0, max(0.0, 1.0 - entropy / math.log(2)))
            value = probability
        else:
            count = len(question.values)
            legend = {str(index): description for index, description in enumerate(question.descriptions)}
            if answer.get("legend") != legend:
                raise ValueError("Invalid score legend")
            distribution = answer.get("probabilities")
            if not isinstance(distribution, dict) or distribution.keys() != legend.keys():
                raise ValueError("Invalid score probability options")
            probabilities_by_index = [_number(distribution[str(index)], 0.0, 1.0) for index in range(count)]
            total = math.fsum(probabilities_by_index)
            if abs(total - 1.0) > count * _ROUNDING + 1e-9:
                raise ValueError("Score probabilities do not sum to one")
            score = _number(answer.get("score"), 0.0, count - 1)
            expectation = math.fsum(index * p for index, p in enumerate(probabilities_by_index))
            tolerance = _ROUNDING * (1 + count * (count - 1) / 2) + 1e-9
            if abs(score - expectation) > tolerance:
                raise ValueError("Score disagrees with its probabilities")
            confidence = _number(answer.get("confidence"), 0.0, 1.0)
            # Normalize harmless wire rounding before remapping nonuniform raw
            # levels. Linear interpolation of the index score would be wrong.
            probabilities = {
                str(float(raw)): p / total
                for raw, p in zip(question.values, probabilities_by_index, strict=True)
            }
            value = math.fsum(
                float(raw) * (p / total)
                for raw, p in zip(question.values, probabilities_by_index, strict=True)
            )
            value = min(float(question.values[-1]), max(float(question.values[0]), value))
        return CriterionAssessment(
            criterion_id=question.criterion_id, value=value, confidence=confidence, probabilities=probabilities,
        )
    except ValueError as exc:
        return CriterionAssessment(criterion_id=question.criterion_id, value=None, error=str(exc))


def _envelope(response: httpx.Response) -> tuple[str, dict[str, Any], int, int]:
    try:
        data = response.json()
    except (ValueError, UnicodeError):
        raise DecisionError("Clef-Flash returned invalid JSON") from None
    if not isinstance(data, dict) or "error" in data:
        raise DecisionError("Clef-Flash returned an invalid response envelope")
    model, answers, usage = data.get("model"), data.get("answers"), data.get("usage")
    if not isinstance(model, str) or not model.strip() or not isinstance(answers, dict) or not isinstance(usage, dict):
        raise DecisionError("Clef-Flash returned an invalid response envelope")
    input_tokens, output_tokens = usage.get("input_tokens"), usage.get("output_tokens")
    if any(type(count) is not int or count < 0 for count in (input_tokens, output_tokens)):
        raise DecisionError("Clef-Flash returned invalid token usage")
    return model, answers, input_tokens, output_tokens


class ClefFlashAdapter:
    """Bounded, typed rubric assessments through ``POST /v1/systemone``."""

    def __init__(self, client: httpx.Client, *, model: str = "clef-flash") -> None:
        if not isinstance(model, str) or not model.strip():
            raise DecisionError("Clef-Flash requires a model name")
        self._client = client
        self._model = model

    def evaluate(
        self, state: Mapping[str, Any], questions: tuple[DecisionQuestion, ...], images: tuple[Path, ...],
    ) -> EvaluationResult:
        if not questions:
            return EvaluationResult(criteria=(), model=self._model)
        try:
            question_payloads = {question.criterion_id: _question_payload(question) for question in questions}
        except ValueError as exc:
            raise DecisionError(f"Invalid Clef-Flash question: {exc}") from None
        if len(question_payloads) != len(questions):
            raise DecisionError("Clef-Flash question identifiers must be unique")
        limit = _IMAGE_LIMIT if images else _TEXT_LIMIT
        encoded_images = []
        try:
            if sum(4 * ((image.stat().st_size + 2) // 3) for image in images) > limit:
                self._payload_too_large(bool(images))
            encoded_images = [base64.b64encode(image.read_bytes()).decode("ascii") for image in images]
        except OSError:
            raise DecisionError("Clef-Flash could not read an evidence image") from None
        assessments = []
        input_tokens = output_tokens = 0
        result_model = ""
        for start in range(0, len(questions), _MAX_QUESTIONS):
            batch = questions[start:start + _MAX_QUESTIONS]
            payload = {
                "model": self._model,
                "state": dict(state),
                "questions": {question.criterion_id: question_payloads[question.criterion_id] for question in batch},
            }
            if images:
                payload["images"] = encoded_images
            try:
                body = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
            except (TypeError, ValueError, UnicodeError, OverflowError):
                raise DecisionError("Clef-Flash evidence is not valid JSON") from None
            if len(body) > limit:
                self._payload_too_large(bool(images))
            try:
                response = self._client.post(
                    "/v1/systemone", content=body, headers={"Content-Type": "application/json"}, follow_redirects=False,
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                hints = {
                    400: "check the local model, question schema, and loaded context window",
                    404: "download the local Clef-Flash model before assessing",
                    413: "request body exceeded the server's 64 KiB/32 MiB limit",
                    500: "the local model could not load or score this request",
                }
                hint = hints.get(exc.response.status_code, "check the local Ollama service")
                raise DecisionError(f"Clef-Flash request failed (HTTP {exc.response.status_code}): {hint}") from None
            except httpx.TimeoutException:
                raise DecisionError("Clef-Flash request timed out") from None
            except httpx.HTTPError:
                raise DecisionError("Clef-Flash request failed during transport") from None
            model, answers, used_input, used_output = _envelope(response)
            if result_model and result_model != model:
                raise DecisionError("Clef-Flash returned inconsistent models across batches")
            if answers.keys() - payload["questions"].keys():
                raise DecisionError("Clef-Flash returned answers for unknown criteria")
            result_model = model
            input_tokens += used_input
            output_tokens += used_output
            assessments.extend(_assessment(question, answers.get(question.criterion_id)) for question in batch)
        return EvaluationResult(
            criteria=tuple(assessments), model=result_model, input_tokens=input_tokens, output_tokens=output_tokens,
        )

    @staticmethod
    def _payload_too_large(has_images: bool) -> None:
        limit = "32 MiB with images" if has_images else "64 KiB without images"
        raise DecisionError(f"Clef-Flash request exceeds {limit}; evidence was not truncated")
