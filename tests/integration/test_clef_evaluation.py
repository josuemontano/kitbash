"""Replay the existing crate case through HTTP scoring, real critics, patching and checkpoints."""

import json
from pathlib import Path

import httpx

from kitbash.analytics.tracker import Tracker
from kitbash.config import default_rubric_path, load_config
from kitbash.critique.critics import PatchWriter, TechnicalCritic, VisualCritic
from kitbash.critique.evaluator import ClefFlashEvaluator
from kitbash.critique.loop import CriticLoop
from kitbash.critique.store import CycleStore
from kitbash.critique.subject import CriticBrief, Evaluation
from kitbash.domain.phases import PhaseName
from kitbash.domain.rubric import Rubric
from kitbash.infra.clef_flash import ClefFlashAdapter
from kitbash.infra.patching import make_diff
from kitbash.llm.client import LLMResponse, Usage
from kitbash.llm.prompts import PromptLibrary
from kitbash.llm.service import LLMService
from kitbash.paths import OutputLayout
from kitbash.store.state import StateDB
from tests.helpers import reference_image

FIXTURE = Path(__file__).parents[1] / "fixtures" / "clef_crate.json"
BEFORE = "PINE_COLOR = (0.8, 0.2, 0.1)\n"
AFTER = "PINE_COLOR = (0.55, 0.36, 0.2)\n"
VISUAL_ID = "matches_the_reference_shape_proportions_color"


def test_crate_failure_gets_feedback_then_passes_and_resumes_without_model_calls(tmp_path):
    fixture = json.loads(FIXTURE.read_text())
    config = load_config()
    rubric = Rubric.load(default_rubric_path())
    layout = OutputLayout.at(tmp_path / "out")
    layout.create()
    reference = reference_image(tmp_path / "reference.png")
    api_requests, llm_requests = [], []

    class CrateEvidence:
        """Replay collected Blender evidence; production rendering remains on LoopSubject."""

        phase = PhaseName.MODELLING
        subject_id = "wooden_crate"

        def brief(self):
            return CriticBrief(self.phase, fixture["subject"], fixture["description"], (reference,))

        def api_reference(self):
            return "PINE_COLOR is the pine material's base color."

        def evaluate(self, script, cycle_dir, cycle):
            artifacts = {}
            for key in ("blend", "usd"):
                path = cycle_dir / f"crate.{key}"
                path.write_text(f"crate fixture artifact: cycle {cycle}")
                artifacts[key] = str(path)
            image = reference_image(cycle_dir / "preview.png", tint=20 if cycle == 1 else 0)
            artifacts["preview"] = str(image)
            return Evaluation(True, (image,), fixture["facts"], fixture["report"], artifacts=artifacts)

    class FeedbackClient:
        def complete(self, request):
            llm_requests.append(request)
            text = make_diff(BEFORE, AFTER) if request.task.endswith(".patch") else json.dumps(fixture["feedback"])
            return LLMResponse(text, Usage(), request.model)

    def respond(request):
        payload = json.loads(request.content)
        api_requests.append(payload)
        assert request.url.path == "/v1/systemone"
        assert set(payload["questions"]) == {VISUAL_ID}  # All measured checks are authoritative, not re-judged.
        assert payload["images"]
        return httpx.Response(200, json=fixture["responses"][len(api_requests) - 1])

    state = StateDB(layout.state_db)
    try:
        with httpx.Client(base_url="http://fixture.local", transport=httpx.MockTransport(respond)) as client:
            llm = LLMService(FeedbackClient(), PromptLibrary(), config.models)
            store = CycleStore(state.cycles, layout)
            loop = CriticLoop(
                evaluator=ClefFlashEvaluator(ClefFlashAdapter(client)),
                critics=(VisualCritic(llm, rubric), TechnicalCritic(llm, rubric)),
                patch_writer=PatchWriter(llm), rubric=rubric, config=config.critic,
                store=store, tracker=Tracker(state.spans),
            )
            subject = CrateEvidence()
            outcome = loop.run(subject, initial_script=lambda: BEFORE)
            assert outcome.passed and outcome.best.cycle == 2
            assert outcome.best.script_path.read_text() == AFTER
            saved = store.load(subject)
            failed = saved.results[1].scorecard
            assert [entry.criterion_id for entry in failed.failing()] == [VISUAL_ID]
            assert saved.results[1].critiques and not saved.results[2].critiques
            current = next(entry for entry in outcome.best.scorecard.entries if entry.criterion_id == VISUAL_ID)
            assert current.raw_score == 0.995 and current.delta == 0.985
            assert current.confidence > config.critic.confidence_threshold
            assert current.probabilities["true"] == 0.995
            assert all(entry.decided_by == "check" for entry in failed.entries if entry.criterion_id != VISUAL_ID)
            critic_requests = [request for request in llm_requests if ".critic." in request.task]
            assert critic_requests and all(VISUAL_ID in request.prompt and '"pass": false' in request.prompt for request in critic_requests)
            patch = next(request for request in llm_requests if request.task.endswith(".patch"))
            assert "PINE_COLOR" in patch.prompt and VISUAL_ID in patch.prompt
            assert api_requests[1]["state"]["previous"]
            assert store.load(subject).results[2].scorecard.to_dict() == outcome.best.scorecard.to_dict()
            resumed = loop.run(subject, initial_script=lambda: BEFORE, session_start=1)
            assert resumed.passed and resumed.best.cycle == 2 and len(api_requests) == 2
            spans = [span for span in state.spans.spans() if span["name"] == "clef_flash"]
            assert sum(span["meta"]["tokens_in"] for span in spans) == 2000
            assert all(span["meta"]["cost_usd"] is None for span in spans)
    finally:
        state.close()
