import pytest

from kitbash.config import default_rubric_path
from kitbash.domain.critique import CriterionScore, CriticKind, Critique, scorecard_from_dict
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


def critique(critic: CriticKind, **scores: tuple[float | None, bool | None]) -> Critique:
    return Critique(
        critic=critic.value,
        summary="",
        scores=tuple(CriterionScore(cid, score, passed) for cid, (score, passed) in scores.items()),
    )


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
    assert check.evaluate({"usd_roundtrip_score": 0.9}) is None  # unmeasured facts leave it to critics
    assert Check.compile("0 <= x < 1").evaluate({"x": 0.5}) is True
    assert Check.compile("up_axis_ok or -x > 1").evaluate({"up_axis_ok": True, "x": 0}) is True


def test_machine_check_decides_over_critics():
    rubric = Rubric.parse(TABLE)
    critics = [critique(CriticKind.TECHNICAL, scale_is_plausible=(1.0, True)), critique(CriticKind.VISUAL, looks_right=(0.9, True))]
    card = rubric.score(PhaseName.MODELLING, critics, {"scale_error": 0.4}, threshold=0.8, require_all_pass=True)
    scale = next(e for e in card.entries if e.criterion_id == "scale_is_plausible")
    assert scale.decided_by == "check" and scale.passed is False and scale.score == 0.0
    assert not card.passed
    card = rubric.score(PhaseName.MODELLING, critics, {"scale_error": 0.1}, threshold=0.8, require_all_pass=True)
    assert card.passed and card.overall == pytest.approx((2 * 1.0 + 3 * 0.9) / 5)


def test_scores_from_critics_that_are_not_assigned_are_ignored():
    rubric = Rubric.parse(TABLE)
    card = rubric.score(
        PhaseName.MODELLING,
        [critique(CriticKind.TECHNICAL, looks_right=(0.1, False)), critique(CriticKind.VISUAL, looks_right=(0.9, True))],
        {"scale_error": 0.0},
        threshold=0.8,
        require_all_pass=True,
    )
    looks = next(e for e in card.entries if e.criterion_id == "looks_right")
    assert looks.score == 0.9 and looks.passed


def test_require_all_pass_and_unscored_criteria():
    rubric = Rubric.parse(TABLE)
    card = rubric.score(
        PhaseName.MODELLING,
        [critique(CriticKind.VISUAL, looks_right=(0.95, False))],
        {"scale_error": 0.0},
        threshold=0.5,
        require_all_pass=True,
    )
    assert not card.passed and [e.criterion_id for e in card.failing()] == ["looks_right"]
    lenient = rubric.score(PhaseName.MODELLING, [critique(CriticKind.VISUAL, looks_right=(0.95, False))], {"scale_error": 0.0}, threshold=0.5, require_all_pass=False)
    assert lenient.passed
    nothing = rubric.score(PhaseName.LAYOUT, [], {}, threshold=0.5, require_all_pass=True)
    assert not nothing.passed and nothing.entries[0].score is None


def test_editing_the_rubric_changes_the_verdict():
    critics = [critique(CriticKind.VISUAL, looks_right=(0.7, None))]
    strict = Rubric.parse(TABLE)
    relaxed = Rubric.parse(TABLE.replace("`scale_error <= 0.25`", "`scale_error <= 0.5`"))
    facts = {"scale_error": 0.4}
    assert not strict.score(PhaseName.MODELLING, critics, facts, threshold=0.6, require_all_pass=False).entries[0].passed
    assert relaxed.score(PhaseName.MODELLING, critics, facts, threshold=0.6, require_all_pass=False).entries[0].passed


def test_required_unknown_criterion_blocks_an_otherwise_perfect_score():
    card = Rubric.parse(TABLE).score(
        PhaseName.MODELLING, [], {"scale_error": 0.0}, threshold=0.8, require_all_pass=True,
    )
    assert card.overall == 1.0
    assert not card.passed
    assert card.entries[1].passed is None


def test_lenient_scoring_keeps_unknown_criteria_explicit_without_relabeling_them():
    card = Rubric.parse(TABLE).score(
        PhaseName.MODELLING, [], {"scale_error": 0.0}, threshold=0.8, require_all_pass=False,
    )
    assert card.passed
    assert [entry.criterion_id for entry in card.unassessed()] == ["looks_right"]
    assert card.entries[1].status == "unassessed" and card.entries[1].score is None
    assert not card.failing()


@pytest.mark.parametrize("unavailable", [False, True], ids=["missing-critic", "explicitly-unavailable"])
def test_one_assigned_critic_cannot_pass_for_an_unavailable_other_critic(unavailable):
    critics = [critique(CriticKind.TECHNICAL, lighting=(0.95, True))]
    if unavailable:
        critics.append(critique(CriticKind.VISUAL, lighting=(None, None)))
    card = Rubric.parse(TABLE).score(PhaseName.LAYOUT, critics, {}, threshold=0.8, require_all_pass=True)
    assert card.overall == 0.95 and not card.passed
    assert card.entries[0].passed is None and card.entries[0].status == "unassessed"
    assert not card.failing() and card.unassessed() == list(card.entries)


def test_known_failure_is_not_erased_by_an_unavailable_second_critic():
    critics = [
        critique(CriticKind.TECHNICAL, lighting=(0.95, False)),
        critique(CriticKind.VISUAL, lighting=(None, None)),
    ]
    card = Rubric.parse(TABLE).score(PhaseName.LAYOUT, critics, {}, threshold=0.8, require_all_pass=True)
    assert not card.passed and card.entries[0].status == "failed"
    assert card.failing() == list(card.entries) and not card.unassessed()


def test_measured_check_can_assess_a_criterion_when_critic_evidence_is_unavailable():
    critics = [
        critique(CriticKind.TECHNICAL, scale_is_plausible=(None, None)),
        critique(CriticKind.VISUAL, looks_right=(0.95, True)),
    ]
    card = Rubric.parse(TABLE).score(PhaseName.MODELLING, critics, {"scale_error": 0.0}, threshold=0.8, require_all_pass=True)
    assert card.passed and not card.unassessed()
    assert card.entries[0].decided_by == "check" and card.entries[0].status == "passed"


def test_pass_verdict_without_a_score_is_not_evidence():
    critics = [critique(CriticKind.VISUAL, looks_right=(None, True))]
    card = Rubric.parse(TABLE).score(PhaseName.MODELLING, critics, {"scale_error": 0.0}, threshold=0.8, require_all_pass=True)
    assert not card.passed and card.entries[1].status == "unassessed"


def test_scorecard_roundtrip_preserves_unassessed_versus_failed_criteria():
    card = Rubric.parse(TABLE).score(PhaseName.MODELLING, [], {"scale_error": 0.4}, threshold=0.8, require_all_pass=True)
    data = card.to_dict()
    assert data["criteria"]["scale_is_plausible"]["status"] == "failed"
    assert data["criteria"]["scale_is_plausible"]["pass"] is False
    assert data["criteria"]["looks_right"]["status"] == "unassessed"
    assert data["criteria"]["looks_right"]["pass"] is None
    restored = scorecard_from_dict(data)
    assert not restored.passed
    assert [entry.criterion_id for entry in restored.failing()] == ["scale_is_plausible"]
    assert [entry.criterion_id for entry in restored.unassessed()] == ["looks_right"]
