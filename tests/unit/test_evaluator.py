from unittest.mock import Mock

import pytest

from kitbash.critique.evaluator import ClefFlashEvaluator, EvaluationCase
from kitbash.critique.subject import CriticBrief, Evaluation
from kitbash.domain.evaluation import CriterionAssessment, DecisionAdapter, DecisionError, EvaluationResult
from kitbash.domain.phases import PhaseName
from kitbash.domain.rubric import Rubric
from kitbash.errors import KitbashError

TABLE = """
| criterion | weight | pass condition | applies to | critic | levels |
|---|---|---|---|---|---|
| Measured scale | 1 | `scale_error <= 0.25` | modelling | technical | |
| Shape | 2 | preserves the reference silhouette | modelling | visual | {"5":"exact silhouette", "1":"unrecognizable", "3":"recognizable"} |
| Materials | 1 | uses principled shading | modelling | technical | |
| Other phase | 1 | balanced | layout | visual | |
"""


def make_case(**kwargs):
    return EvaluationCase(
        evaluation_id="modelling:chair:3", iteration=3,
        brief=kwargs.pop("brief", CriticBrief(PhaseName.MODELLING, "chair", "wooden chair")),
        script=kwargs.pop("script", "create_chair()"), **kwargs,
    )


def adapter_returning(*criteria):
    adapter = Mock(spec=DecisionAdapter)
    adapter.evaluate.return_value = EvaluationResult(tuple(criteria), "clef-flash", 17, 4)
    return adapter


def test_case_conversion_preserves_all_evidence_and_previous_results(tmp_path):
    reference = tmp_path / "reference.png"
    render = tmp_path / "render.png"
    compare = tmp_path / "usd_compare.png"
    for path in (reference, render, compare):
        path.write_bytes(b"image")
    previous = ({"iteration": 1, "status": "fail", "criteria": {"shape": {"raw_score": 2.123456789}}},
                {"iteration": 2, "status": "rejected", "criteria": {"shape": {"raw_score": 1.123456789}}})
    case = make_case(
        brief=CriticBrief(PhaseName.MODELLING, "chair", "wooden chair", (reference,), "stylized"),
        script="script-content" * 10000, history="all historical feedback", previous=previous, feedback="preserve the legs",
    )
    evidence = Evaluation(
        True, (render, render), {"scale_error": 0.1, "nested": {"measurement": [1, 2, 3]}},
        {"geometry": {"all": "report data"}}, artifacts={"usd_compare": str(compare), "render": str(render), "usd": "mesh.usdc"},
    )
    adapter = adapter_returning(CriterionAssessment("shape", 4.9, 0.99), CriterionAssessment("materials", 0.99, 0.99))
    result = ClefFlashEvaluator(adapter).evaluate(case, Rubric.parse(TABLE), evidence)
    state, questions, images = adapter.evaluate.call_args.args
    assert state["evaluation_id"] == case.evaluation_id and state["iteration"] == 3
    assert state["script"] == case.script and state["history"] == case.history
    assert state["previous"] == list(previous) and state["feedback"] == case.feedback
    assert state["brief"] == {"phase": "modelling", "subject": "chair", "description": "wooden chair", "style": "stylized", "references": [str(reference)]}
    assert state["evidence"] == {
        "ok": True, "error": None, "facts": dict(evidence.facts), "report": dict(evidence.report),
        "artifacts": dict(evidence.artifacts), "images": [str(render), str(render)],
    }
    assert images == (reference, render, compare)
    manifest = state["image_manifest"]
    assert [item["role"] for item in manifest] == ["reference", "render", "render", "artifact", "artifact"]
    assert [item["image_index"] for item in manifest] == [0, 1, 1, 2, 1]
    assert {question.criterion_id for question in questions} == {"shape", "materials"}
    assert [assessment.criterion_id for assessment in result.criteria] == ["measured_scale", "shape", "materials"]
    assert result.criteria[0] == CriterionAssessment("measured_scale", 1.0, 1.0, source="check")
    assert result.model == "clef-flash" and result.input_tokens == 17 and result.output_tokens == 4


def test_questions_preserve_explicit_ordinal_meanings_and_binary_pass_conditions(tmp_path):
    render = tmp_path / "render.png"
    render.touch()
    adapter = adapter_returning()
    ClefFlashEvaluator(adapter).evaluate(make_case(), Rubric.parse(TABLE), Evaluation(True, (render,), {"scale_error": 0.1}))
    _, questions, _ = adapter.evaluate.call_args.args
    shape, materials = questions
    assert shape.criterion_id == "shape" and shape.binary is False
    assert shape.values == (1.0, 3.0, 5.0)
    assert shape.descriptions == ("unrecognizable", "recognizable", "exact silhouette")
    assert "preserves the reference silhouette" in shape.instructions
    assert materials.criterion_id == "materials" and materials.binary is True
    assert materials.values == (0.0, 1.0)
    assert materials.descriptions == ("Pass condition not met: uses principled shading", "Pass condition met: uses principled shading")


def test_missing_render_does_not_turn_references_into_output_evidence(tmp_path):
    reference = tmp_path / "reference.png"
    reference.touch()
    adapter = adapter_returning(CriterionAssessment("materials", 0.99, 0.99))
    case = make_case(brief=CriticBrief(PhaseName.MODELLING, "chair", "wooden chair", (reference,)))
    result = ClefFlashEvaluator(adapter).evaluate(case, Rubric.parse(TABLE), Evaluation(True, facts={"scale_error": 0.0}))
    assert result.criteria[1].value is None and result.criteria[1].error
    assert [q.criterion_id for q in adapter.evaluate.call_args.args[1]] == ["materials"]
    card = Rubric.parse(TABLE).score(PhaseName.MODELLING, result.criteria, {"scale_error": 0.0}, threshold=0.8, require_all_pass=False)
    assert not card.passed and card.status == "uncertain"


def test_already_collected_comparison_artifact_is_visual_evidence(tmp_path):
    compare = tmp_path / "usd_compare.png"
    compare.touch()
    adapter = adapter_returning(CriterionAssessment("shape", 4.9, 0.99))
    result = ClefFlashEvaluator(adapter).evaluate(make_case(), Rubric.parse(TABLE), Evaluation(True, artifacts={"usd_compare": str(compare)}))
    assert result.criteria[1].value == 4.9
    assert adapter.evaluate.call_args.args[2] == (compare,)


def test_missing_image_paths_are_recorded_but_not_sent_to_provider(tmp_path):
    missing = tmp_path / "missing.png"
    adapter = adapter_returning()
    ClefFlashEvaluator(adapter).evaluate(make_case(), Rubric.parse(TABLE), Evaluation(True, (missing,)))
    state, questions, images = adapter.evaluate.call_args.args
    assert images == ()
    assert state["image_manifest"][0]["available"] is False
    assert state["image_manifest"][0]["image_index"] is None
    assert "shape" not in {q.criterion_id for q in questions}


@pytest.mark.parametrize("ok, error", [(False, "render failed"), (False, None), (True, "render failed")])
def test_failed_evidence_never_uses_provider_to_invent_success(ok, error):
    adapter = adapter_returning()
    result = ClefFlashEvaluator(adapter).evaluate(make_case(), Rubric.parse(TABLE), Evaluation(ok, facts={"scale_error": 0.4}, error=error))
    adapter.evaluate.assert_not_called()
    assert result.criteria[0].value == 0.0 and result.criteria[0].source == "check"
    assert all(criterion.value is None for criterion in result.criteria[1:])


def test_provider_failure_is_safe_unavailable_and_preserves_measured_results(tmp_path):
    render = tmp_path / "render.png"
    render.touch()
    adapter = adapter_returning()
    adapter.evaluate.side_effect = KitbashError("secret token or image payload")
    result = ClefFlashEvaluator(adapter).evaluate(make_case(), Rubric.parse(TABLE), Evaluation(True, (render,), {"scale_error": 0.1}))
    assert result.criteria[0].value == 1.0
    assert all(criterion.value is None and criterion.error for criterion in result.criteria[1:])
    assert "secret" not in str(result.to_dict())


def test_missing_extra_and_duplicate_provider_answers_cannot_pass(tmp_path):
    render = tmp_path / "render.png"
    render.touch()
    adapter = adapter_returning(
        CriterionAssessment("shape", 5.0, 0.99), CriterionAssessment("shape", 4.0, 0.99),
        CriterionAssessment("unrequested", 1.0, 0.99), CriterionAssessment("measured_scale", 0.0, 0.99),
    )
    result = ClefFlashEvaluator(adapter).evaluate(make_case(), Rubric.parse(TABLE), Evaluation(True, (render,), {"scale_error": 0.1}))
    assert [entry.criterion_id for entry in result.criteria] == ["measured_scale", "shape", "materials"]
    assert result.criteria[0].value == 1.0
    assert result.criteria[1].value is None and result.criteria[2].value is None


def test_raw_values_confidence_and_probabilities_survive_consumer_boundary(tmp_path):
    render = tmp_path / "render.png"
    render.touch()
    ordinal = CriterionAssessment("shape", 4.6, 0.912345678, {"1.0": 0.025, "3.0": 0.15, "5.0": 0.825})
    binary = CriterionAssessment("materials", 0.91, 0.123456789, {"false": 0.09, "true": 0.91})
    adapter = adapter_returning(ordinal, binary)
    rubric = Rubric.parse(TABLE)
    result = ClefFlashEvaluator(adapter).evaluate(make_case(), rubric, Evaluation(True, (render,), {"scale_error": 0.1}))
    assert result.criteria[1:] == (ordinal, binary)
    assert result.to_dict()["criteria"][1] == ordinal.to_dict()
    card = rubric.score(PhaseName.MODELLING, result.criteria, {"scale_error": 0.1}, threshold=0.8, require_all_pass=False)
    assert card.entries[1].score == pytest.approx(0.9)
    assert card.entries[1].probabilities == ordinal.probabilities
    assert card.entries[2].confidence == binary.confidence
    assert card.entries[2].probabilities == binary.probabilities
    assert card.entries[2].passed is None and not card.passed


def test_safe_provider_failure_retains_actionable_diagnostic():
    adapter = adapter_returning()
    adapter.evaluate.side_effect = DecisionError("Clef-Flash HTTP 404: check model availability")
    result = ClefFlashEvaluator(adapter).evaluate(make_case(), Rubric.parse(TABLE), Evaluation(True, facts={"scale_error": 0.1}))
    assert result.criteria[2].error == "Clef-Flash HTTP 404: check model availability"
    assert result.criteria[2].value is None
