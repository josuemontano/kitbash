"""Critics must account for every assigned criterion without inventing missing evidence."""

import json

import pytest
from PIL import Image

from kitbash.config import load_config
from kitbash.critique.critics import ReviewRequest, TechnicalCritic, VisualCritic
from kitbash.critique.subject import CriticBrief, Evaluation
from kitbash.domain.phases import PhaseName
from kitbash.domain.rubric import Rubric
from kitbash.errors import LLMError
from kitbash.llm.client import LLMResponse, Usage
from kitbash.llm.prompts import PromptLibrary
from kitbash.llm.service import LLMService

RUBRIC = Rubric.parse(
    "| criterion | weight | pass condition | applies to | critic |\n|-|-|-|-|-|\n"
    "| Shape | 1 | matches the reference | modelling | both |\n"
    "| Geometry | 1 | closed mesh | modelling | technical |\n"
    "| Lighting | 1 | good exposure | layout | visual |"
)
ASSESSED = {"score": 0.95, "pass": True, "notes": "measured evidence"}


class ScriptedClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return LLMResponse(text=json.dumps(next(self.responses)), usage=Usage(), model=request.model)


def service(responses, *, parse_retries=0):
    client = ScriptedClient(responses)
    return LLMService(client, PromptLibrary(), load_config().models, parse_retries=parse_retries), client


def request(*, images=()):
    return ReviewRequest(
        brief=CriticBrief(PhaseName.MODELLING, "crate", "wooden crate"),
        evaluation=Evaluation(ok=True, images=images), script="pass", history="", cycle=1,
    )


def test_partial_critic_response_is_not_accepted():
    llm, _ = service([{"scores": {"shape": ASSESSED}}])
    with pytest.raises(LLMError, match="geometry"):
        TechnicalCritic(llm, RUBRIC).review(request())


def test_partial_response_uses_existing_repair_path_before_acceptance():
    llm, client = service([
        {"scores": {"shape": ASSESSED}},
        {"scores": {"shape": {"score": 0.2, "pass": False}, "geometry": ASSESSED}},
    ], parse_retries=1)
    result = TechnicalCritic(llm, RUBRIC).review(request())
    assert result.score_for("shape").passed is False
    assert result.score_for("shape").score == 0.2
    assert result.score_for("geometry").passed is True
    assert len(client.requests) == 2


def test_critic_can_explicitly_mark_all_applicable_evidence_unavailable():
    unavailable = {"score": None, "pass": None, "notes": "The inspection report is unavailable"}
    llm, _ = service([{"scores": {"shape": unavailable, "geometry": unavailable}}])
    result = TechnicalCritic(llm, RUBRIC).review(request())
    assert {score.criterion_id for score in result.scores} == {"shape", "geometry"}
    card = RUBRIC.score(PhaseName.MODELLING, [result], {}, threshold=0.8, require_all_pass=True)
    assert not card.passed and not card.failing()
    assert {entry.criterion_id for entry in card.unassessed()} == {"shape", "geometry"}
    assert all(result.score_for(key).notes == unavailable["notes"] for key in ("shape", "geometry"))


@pytest.mark.parametrize("score", [
    {"score": None, "pass": True, "notes": "no evidence"},
    {"score": None, "pass": False, "notes": "no evidence"},
    {"score": None, "pass": None, "notes": "  "},
    {"score": 0.95, "pass": None, "notes": "no verdict"},
    {},
])
def test_incomplete_or_contradictory_assessment_is_rejected(score):
    llm, _ = service([{"scores": {"shape": ASSESSED, "geometry": score}}])
    with pytest.raises(LLMError, match="geometry"):
        TechnicalCritic(llm, RUBRIC).review(request())


@pytest.mark.parametrize("missing_file", [False, True], ids=["no-renders", "deleted-render"])
def test_skipped_visual_review_marks_each_applicable_criterion_unassessed(tmp_path, missing_file):
    images = (tmp_path / "missing.png",) if missing_file else ()
    llm, client = service([])
    result = VisualCritic(llm, RUBRIC).review(request(images=images))
    assert client.requests == []
    assert {score.criterion_id for score in result.scores} == {"shape"}
    score = result.score_for("shape")
    assert score.score is None and score.passed is None and score.notes


def test_complete_role_specific_reviews_can_pass_together(tmp_path):
    render = tmp_path / "render.png"
    Image.new("RGB", (2, 2)).save(render)
    visual_llm, _ = service([{"scores": {"shape": ASSESSED}}])
    technical_llm, _ = service([{"scores": {"shape": ASSESSED, "geometry": ASSESSED}}])
    review = request(images=(render,))
    critics = [VisualCritic(visual_llm, RUBRIC).review(review), TechnicalCritic(technical_llm, RUBRIC).review(review)]
    card = RUBRIC.score(PhaseName.MODELLING, critics, {}, threshold=0.8, require_all_pass=True)
    assert card.passed and not card.unassessed() and not card.failing()
