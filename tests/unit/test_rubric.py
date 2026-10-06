import pytest

from kitbash.config import default_rubric_path
from kitbash.domain.critique import CriticKind, Critique, scorecard_from_dict
from kitbash.domain.evaluation import CriterionAssessment
from kitbash.domain.phases import PhaseName
from kitbash.domain.rubric import Check, Rubric, RubricError

TABLE = """
# My rubric

| criterion | weight | pass condition | applies to | critic |
|---|---|---|---|---|
| Scale is plausible | 2 | within 25% `scale_error <= 0.25` | modelling | technical |
| Looks right | 3 | matches the reference | breakdown, modelling | visual |
| Lighting | 1 | balanced exposure | layout | both |
"""


def test_default_rubric_has_every_required_criterion():
    rubric = Rubric.load(default_rubric_path())
    ids = {c.id for c in rubric.criteria}
    assert {
        "real_world_scale_is_plausible",
        "origin_at_the_base_z_up",
        "asset_name_is_correct_and_follows_the_naming_convention",
        "matches_the_reference_shape_proportions_color",
        "uses_principled_bsdf_with_sensible_pbr_values",
        "usd_material_fidelity",
        "spatial_arrangement_matches_the_reference",
        "style_compliance",
        "lighting",
    } <= ids
    usd = next(c for c in rubric.criteria if c.id == "usd_material_fidelity")
    assert usd.applies_to == {PhaseName.MODELLING, PhaseName.ASSEMBLY}
    assert usd.check is not None and {"usd_roundtrip_score", "missing_textures"} <= usd.check.names


def test_parse_columns_phases_and_critics():
    rubric = Rubric.parse(TABLE)
    scale, looks, lighting = rubric.criteria
    assert scale.id == "scale_is_plausible" and scale.weight == 2.0
    assert scale.critics == {CriticKind.TECHNICAL}
    assert looks.applies_to == {PhaseName.BREAKDOWN, PhaseName.MODELLING}
    assert lighting.critics == {CriticKind.VISUAL, CriticKind.TECHNICAL}
    assert [c.id for c in rubric.for_phase(PhaseName.MODELLING, CriticKind.VISUAL)] == ["looks_right"]
    assert "scale_is_plausible" in rubric.render(PhaseName.MODELLING)


def test_critic_column_is_optional_and_all_phases_keyword():
    rubric = Rubric.parse("| criterion | weight | pass condition | applies to |\n|-|-|-|-|\n| Anything | 1 | ok | all |")
    (criterion,) = rubric.criteria
    assert criterion.applies_to == set(PhaseName)
    assert criterion.critics == set(CriticKind)


@pytest.mark.parametrize(
    "text, message",
    [
        ("no table here", "No rubric table"),
        ("| criterion | weight |\n|-|-|\n| A | 1 |", "missing columns"),
        ("| criterion | weight | pass condition | applies to |\n|-|-|-|-|\n| A | heavy | ok | modelling |", "weight must be a number"),
        ("| criterion | weight | pass condition | applies to |\n|-|-|-|-|\n| A | 1 | ok | nowhere |", "unknown phase"),
        ("| criterion | weight | pass condition | applies to |\n|-|-|-|-|\n| A | 1 | `import os` | modelling |", "Invalid check"),
        ("| criterion | weight | pass condition | applies to |\n|-|-|-|-|\n| A | 1 | `f(x) > 1` | modelling |", "Unsupported syntax"),
        ("| criterion | weight | pass condition | applies to |\n|-|-|-|-|\n| A | 1 | ok | modelling |\n| A | 2 | ok | layout |", "Duplicate"),
    ],
)
def test_invalid_rubrics_are_rejected(text, message):
    with pytest.raises(RubricError, match=message):
        Rubric.parse(text)


def test_check_expressions():
    check = Check.compile("usd_roundtrip_score >= 0.85 and missing_textures == 0 and not broken")
    assert check.evaluate({"usd_roundtrip_score": 0.9, "missing_textures": 0, "broken": False}) is True
    assert check.evaluate({"usd_roundtrip_score": 0.8, "missing_textures": 0, "broken": False}) is False
    assert check.evaluate({"usd_roundtrip_score": 0.9}) is None
    assert Check.compile("0 <= x < 1").evaluate({"x": 0.5}) is True
    assert Check.compile("up_axis_ok or -x > 1").evaluate({"up_axis_ok": True, "x": 0}) is True


def assess(criterion_id, value, confidence=0.95, **kwargs):
    return CriterionAssessment(criterion_id, value, confidence, **kwargs)


def score(rubric, assessments=(), facts=None, **kwargs):
    options = {"threshold": 0.8, "require_all_pass": True, **kwargs}
    return rubric.score(PhaseName.MODELLING, assessments, facts or {}, **options)


def ordinal_rubric(threshold=""):
    return Rubric.parse(
        "| criterion | weight | pass condition | applies to | levels | threshold |\n"
        "|---|---|---|---|---|---|\n"
        '| Shape | 1 | matches reference | modelling | {"5":"excellent", "1":"poor", "3":"adequate"} | '
        + threshold + " |"
    )


def test_machine_check_is_authoritative_over_provider_and_available_without_provider():
    rubric = Rubric.parse(TABLE)
    assessments = [assess("scale_is_plausible", 1.0), assess("looks_right", 0.9)]
    failed = score(rubric, assessments, {"scale_error": 0.4})
    assert failed.entries[0].decided_by == "check"
    assert failed.entries[0].score == 0.0 and failed.entries[0].passed is False
    assert not failed.passed
    passed = score(rubric, [assess("looks_right", 0.9)], {"scale_error": 0.1})
    assert passed.passed and passed.overall == pytest.approx((2 + 3 * 0.9) / 5)
    assert passed.entries[0].confidence == 1.0


def test_binary_rubric_does_not_invent_an_ordinal_scale():
    rubric = Rubric.parse(TABLE)
    assert all(not c.levels and c.threshold is None for c in rubric.criteria)
    card = score(rubric, [assess("looks_right", 0.83)], {"scale_error": 0.0})
    assert card.entries[1].raw_score == card.entries[1].score == 0.83
    assert card.entries[1].threshold == 0.8


def test_explicit_ordinal_meanings_are_sorted_and_raw_scale_normalized():
    rubric = ordinal_rubric()
    assert rubric.criteria[0].levels == ((1.0, "poor"), (3.0, "adequate"), (5.0, "excellent"))
    card = score(rubric, [assess("shape", 4.6)])
    entry = card.entries[0]
    assert entry.raw_score == 4.6
    assert entry.score == pytest.approx(0.9)
    assert entry.threshold == pytest.approx(4.2)
    assert card.passed


def test_explicit_raw_threshold_overrides_mapped_global_threshold():
    assessment = [assess("shape", 4.4)]
    assert score(ordinal_rubric(), assessment).passed
    card = score(ordinal_rubric("4.5"), assessment)
    assert card.overall > 0.8 and not card.passed
    assert card.entries[0].passed is False and card.entries[0].threshold == 4.5
    assert score(ordinal_rubric("4.5"), assessment, require_all_pass=False).passed


@pytest.mark.parametrize("confidence", [None, 0.699, -1.0, 1.1, float("nan"), True])
@pytest.mark.parametrize("require_all_pass", [False, True])
def test_uncertainty_blocks_pass_even_when_lenient(confidence, require_all_pass):
    card = score(Rubric.parse(TABLE), [assess("looks_right", 1.0, confidence)], {"scale_error": 0.0}, require_all_pass=require_all_pass)
    assert not card.passed and card.status == "uncertain"
    assert card.entries[1].score == 1.0 and card.entries[1].passed is None
    assert [entry.criterion_id for entry in card.unassessed()] == ["looks_right"]


def test_confidence_boundary_is_inclusive():
    assert score(ordinal_rubric(), [assess("shape", 5.0, 0.7)]).passed
    assert not score(ordinal_rubric(), [assess("shape", 5.0, 0.7)], confidence_threshold=0.71).passed


@pytest.mark.parametrize("require_all_pass", [False, True])
def test_missing_assessment_never_passes(require_all_pass):
    card = score(Rubric.parse(TABLE), facts={"scale_error": 0.0}, require_all_pass=require_all_pass)
    assert card.overall == 1.0 and not card.passed
    assert card.entries[1].score is None and card.entries[1].passed is None


@pytest.mark.parametrize("value", [None, -0.01, 1.01, float("nan"), float("inf"), True, "0.9"])
def test_invalid_binary_values_are_unavailable_not_clamped(value):
    card = score(Rubric.parse(TABLE), [assess("looks_right", value)], {"scale_error": 0.0})
    assert card.entries[1].score is None and not card.passed


def test_provider_errors_and_duplicate_answers_cannot_establish_pass():
    rubric = ordinal_rubric()
    assert score(rubric, [assess("shape", 5.0, error="unavailable")]).status == "uncertain"
    duplicate = score(rubric, [assess("shape", 5.0), assess("shape", 1.0)])
    assert duplicate.entries[0].score is None and not duplicate.passed


def test_per_dimension_regression_blocks_rising_aggregate():
    rubric = Rubric.parse(TABLE)
    previous = score(rubric, [assess("scale_is_plausible", 0.9), assess("looks_right", 0.5)])
    current = score(rubric, [assess("scale_is_plausible", 0.85), assess("looks_right", 1.0)], previous=previous)
    assert current.overall > previous.overall
    assert current.entries[0].passed is True and current.entries[0].delta == pytest.approx(-0.05)
    assert current.regressions() == [current.entries[0]]
    assert not current.passed and current.status == "fail"


@pytest.mark.parametrize("value, confidence", [(4.199, 0.99), (4.3, 0.1), (None, None)])
def test_pass_to_fail_or_unknown_is_regression_even_without_large_drop(value, confidence):
    rubric = ordinal_rubric()
    previous = score(rubric, [assess("shape", 4.201)])
    current = score(rubric, [assess("shape", value, confidence)], previous=previous)
    assert current.regressions() == list(current.entries)
    assert not current.passed


def test_regression_epsilon_and_recovery_from_unavailable():
    rubric = ordinal_rubric()
    previous = score(rubric, [assess("shape", 4.8)])
    current = score(rubric, [assess("shape", 4.76)], previous=previous)
    assert current.passed and not current.regressions()
    unavailable = score(rubric)
    recovered = score(rubric, [assess("shape", 5.0)], previous=unavailable)
    assert recovered.passed and recovered.entries[0].delta is None


def test_scorecard_roundtrip_retains_full_precision_and_decision_fields():
    rubric = ordinal_rubric("4.123456789")
    probabilities = {"1.0": 0.0123456789, "3.0": 0.123456789, "5.0": 0.8641975321}
    previous = score(rubric, [assess("shape", 4.987654321)])
    card = score(rubric, [assess("shape", 4.7037037064, 0.876543219, probabilities=probabilities)], previous=previous)
    data = card.to_dict()
    restored = scorecard_from_dict(data)
    assert restored == card
    assert data["overall"] == card.overall
    assert data["criteria"]["shape"]["score"] == card.entries[0].score
    assert data["criteria"]["shape"]["raw_score"] == 4.7037037064
    assert data["criteria"]["shape"]["confidence"] == 0.876543219
    assert data["criteria"]["shape"]["probabilities"] == probabilities
    assert restored.regressions() and restored.status == "fail"


def test_scorecard_roundtrip_preserves_unassessed_versus_failed():
    card = score(Rubric.parse(TABLE), facts={"scale_error": 0.4})
    restored = scorecard_from_dict(card.to_dict())
    assert restored == card
    assert [e.criterion_id for e in restored.failing()] == ["scale_is_plausible"]
    assert [e.criterion_id for e in restored.unassessed()] == ["looks_right"]


def test_critique_is_feedback_only_and_ignores_llm_score_claims():
    critique = Critique.parse("visual", {"summary": "fix shape", "scores": {"shape": {"pass": True, "score": 100}}, "edits": [{"instruction": "Widen base"}]})
    assert critique.summary == "fix shape" and critique.edits[0].instruction == "Widen base"
    assert "scores" not in critique.to_dict()
    assert not hasattr(critique, "scores")


@pytest.mark.parametrize("weight", ["-1", "nan", "inf", "-inf"])
def test_invalid_weights_are_rejected(weight):
    with pytest.raises(RubricError):
        Rubric.parse(TABLE.replace("| 2 |", f"| {weight} |"))


def test_zero_applicable_weight_is_rejected():
    with pytest.raises(RubricError):
        Rubric.parse(TABLE.replace("| 1 |", "| 0 |"))


@pytest.mark.parametrize("levels", [
    '{"1":"only"}', '{"1":"bad", "1":"good"}', '{"1":"bad", "1.0":"good"}',
    '{"NaN":"bad", "5":"good"}', '{"1":"", "5":"good"}', '{"1":true, "5":"good"}',
    '["bad", "good"]', '{"bad":"bad", "good":"good"}',
])
def test_invalid_ordinal_levels_are_rejected(levels):
    with pytest.raises(RubricError):
        Rubric.parse("| criterion | weight | pass condition | applies to | levels |\n|-|-|-|-|-|\n| Shape | 1 | matches | modelling | " + levels + " |")


@pytest.mark.parametrize("threshold", ["nan", "inf", "0", "6", "high"])
def test_invalid_raw_threshold_is_rejected(threshold):
    with pytest.raises(RubricError):
        ordinal_rubric(threshold)


def test_uncertain_baseline_can_recover_without_hiding_other_dimension_regressions():
    rubric = Rubric.parse(TABLE)
    previous = score(rubric, [assess("scale_is_plausible", 0.99), assess("looks_right", 0.99, 0.1)])
    current = score(rubric, [assess("scale_is_plausible", 0.85), assess("looks_right", 0.85)], previous=previous)
    assert current.entries[0].delta == pytest.approx(-0.14)
    assert current.entries[0].regressed
    assert current.entries[1].delta == pytest.approx(-0.14)
    assert current.entries[1].passed is True and not current.entries[1].regressed
    assert current.regressions() == [current.entries[0]]
    assert not current.passed
    recovered = score(rubric, [assess("scale_is_plausible", 0.99), assess("looks_right", 0.85)], previous=previous)
    assert recovered.passed and not recovered.regressions()
