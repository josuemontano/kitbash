"""Generative critics consume bounded decisions and return feedback, never scoring authority."""

import json

import pytest
from PIL import Image

from kitbash.config import load_config
from kitbash.critique.critics import ReviewRequest, TechnicalCritic, VisualCritic
from kitbash.critique.subject import CriticBrief, Evaluation
from kitbash.domain.evaluation import CriterionAssessment
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
FEEDBACK = {"summary": "Close the open mesh without changing its silhouette.", "edits": [
    {"target": "geometry", "instruction": "Cap the open base", "priority": "high"},
]}


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
    card = RUBRIC.score(PhaseName.MODELLING, (
        CriterionAssessment("shape", 0.95, confidence=0.9),
        CriterionAssessment("geometry", 0.2, confidence=0.95),
    ), {}, threshold=0.8, require_all_pass=True)
    return ReviewRequest(
        brief=CriticBrief(PhaseName.MODELLING, "crate", "wooden crate"), scorecard=card,
        evaluation=Evaluation(ok=True, images=images), script="pass", history="prior criteria and regressions", cycle=1,
    )


def test_technical_critic_returns_feedback_without_scores():
    llm, client = service([FEEDBACK])
    review = request()
    result = TechnicalCritic(llm, RUBRIC).review(review)
    assert result.summary == FEEDBACK["summary"]
    assert result.edits[0].instruction == "Cap the open base" and result.edits[0].source == "technical"
    assert "scores" not in result.to_dict()
    assert json.dumps(review.scorecard.to_dict(), indent=1) in client.requests[0].prompt
    assert review.history in client.requests[0].prompt
    assert "feedback only" in client.requests[0].prompt


@pytest.mark.parametrize("response", [[], {}, {"summary": "missing edits"}, {"summary": 3, "edits": []}])
def test_feedback_output_requires_summary_and_edits(response):
    llm, _ = service([response])
    with pytest.raises(LLMError, match="summary"):
        TechnicalCritic(llm, RUBRIC).review(request())


def test_generated_scores_are_rejected_through_existing_repair_path():
    llm, client = service([
        {**FEEDBACK, "scores": {"shape": {"score": 1.0, "pass": True}}}, FEEDBACK,
    ], parse_retries=1)
    result = TechnicalCritic(llm, RUBRIC).review(request())
    assert len(client.requests) == 2
    assert "scores" not in result.to_dict() and result.edits


@pytest.mark.parametrize("missing_file", [False, True], ids=["no-renders", "deleted-render"])
def test_visual_critic_does_not_invent_feedback_without_renders(tmp_path, missing_file):
    images = (tmp_path / "missing.png",) if missing_file else ()
    llm, client = service([])
    result = VisualCritic(llm, RUBRIC).review(request(images=images))
    assert client.requests == []
    assert result.summary.startswith("Skipped:") and not result.edits
    assert "scores" not in result.to_dict()


def test_visual_critic_receives_scorecard_and_existing_images(tmp_path):
    render = tmp_path / "render.png"
    Image.new("RGB", (2, 2)).save(render)
    llm, client = service([FEEDBACK])
    review = request(images=(render,))
    result = VisualCritic(llm, RUBRIC).review(review)
    sent = client.requests[0]
    assert sent.attachments == (render,)
    assert json.dumps(review.scorecard.to_dict(), indent=1) in sent.prompt
    assert "| geometry |" not in sent.prompt  # role-specific rubric guidance remains scoped
    assert "scores" not in result.to_dict()
