"""Exercise the documented System One wire format without contacting Ollama."""

import base64
import json
import math

import httpx
import pytest

from kitbash.domain.evaluation import DecisionError, DecisionQuestion
from kitbash.infra.clef_flash import ClefFlashAdapter


@pytest.fixture
def binary():
    return DecisionQuestion("visible", "Is the reference visible?", (0.0, 1.0), ("not visible", "visible"), True)


@pytest.fixture
def ordinal():
    # Uneven levels make incorrect linear remapping of the index score observable.
    return DecisionQuestion("quality", "Judge fit to the reference.", (1.0, 2.0, 5.0), ("poor", "fair", "excellent"))


def score_answer(probabilities=None, *, score=1.5, confidence=0.42, legend=None):
    return {
        "type": "score", "score": score,
        "legend": legend if legend is not None else {"0": "poor", "1": "fair", "2": "excellent"},
        "probabilities": probabilities if probabilities is not None else {"0": 0.1, "1": 0.3, "2": 0.6},
        "confidence": confidence,
    }


def response(answers, *, model="clef-flash", input_tokens=175, output_tokens=2):
    return {"model": model, "answers": answers, "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}}


def client_for(handler):
    return httpx.Client(base_url="http://clef.test", timeout=2, transport=httpx.MockTransport(handler))


def test_systemone_request_and_typed_responses(binary, ordinal, tmp_path):
    picture = tmp_path / "sample.png"
    picture.write_bytes(b"\x89PNG\r\n\x1a\nraw-example")
    evidence = {"report": {"details": "No truncation: 晴"}, "renders": [picture.name]}
    calls = []

    def handle(request):
        calls.append(request)
        assert request.method == "POST"
        assert request.url.path == "/v1/systemone"
        assert request.headers["content-type"] == "application/json"
        wire = json.loads(request.content)
        assert wire == {
            "model": "clef-flash", "state": evidence,
            "images": [base64.b64encode(picture.read_bytes()).decode("ascii")],
            "questions": {
                "visible": {"type": "noul", "instructions": binary.instructions,
                            "criteria": {"false": "not visible", "true": "visible"}},
                "quality": {"type": "score", "instructions": ordinal.instructions,
                            "criteria": ["poor", "fair", "excellent"]},
            },
        }
        return httpx.Response(200, json=response({
            "visible": {"type": "noul", "noul": 0.9},
            "quality": score_answer(),
        }))

    with client_for(handle) as client:
        result = ClefFlashAdapter(client).evaluate(evidence, (binary, ordinal), (picture,))
    assert len(calls) == 1
    assert result.model == "clef-flash"
    assert (result.input_tokens, result.output_tokens) == (175, 2)
    yes, quality = result.criteria
    assert yes.criterion_id == "visible" and yes.source == "clef-flash" and yes.error == ""
    assert yes.value == pytest.approx(0.9)
    assert yes.probabilities == pytest.approx({"false": 0.1, "true": 0.9})
    expected_concentration = 1 + (0.1 * math.log(0.1) + 0.9 * math.log(0.9)) / math.log(2)
    assert yes.confidence == pytest.approx(expected_concentration)
    assert quality.value == pytest.approx(3.7)  # Not score 1.5 mapped linearly to 4.
    assert quality.probabilities == pytest.approx({"1.0": 0.1, "2.0": 0.3, "5.0": 0.6})
    assert quality.confidence == 0.42


def test_rounded_distribution_is_accepted_and_normalized(ordinal):
    def handle(request):
        return httpx.Response(200, json=response({
            "quality": score_answer({"0": 0.3333, "1": 0.3333, "2": 0.3333}, score=0.9999),
        }))

    with client_for(handle) as client:
        result = ClefFlashAdapter(client).evaluate({"text": "full"}, (ordinal,), ())
    assert result.criteria[0].value == pytest.approx(8 / 3)
    assert sum(result.criteria[0].probabilities.values()) == pytest.approx(1)


@pytest.mark.parametrize("answer", [
    {"type": "noul", "noul": True},
    {"type": "noul", "noul": "0.9"},
    {"type": "noul", "noul": float("nan")},
    {"type": "noul", "noul": -0.01},
    {"type": "noul", "noul": 1.01},
    {"type": "score", "score": 0.9},
    ["noul", 0.9],
])
def test_invalid_binary_answer_is_unavailable(binary, answer):
    def handle(request):
        return httpx.Response(200, content=json.dumps(response({"visible": answer})).encode("utf-8"))

    with client_for(handle) as client:
        criterion, = ClefFlashAdapter(client).evaluate({"text": "evidence"}, (binary,), ()).criteria
    assert criterion.criterion_id == binary.criterion_id
    assert criterion.value is None and criterion.confidence is None and criterion.error
    assert criterion.probabilities == {}


@pytest.mark.parametrize("answer", [
    score_answer(score=True),
    score_answer(score="1.5"),
    score_answer(score=float("inf")),
    score_answer(score=-0.5),
    score_answer(score=2.1),
    score_answer(probabilities={"0": 0.1, "1": 0.3}),
    score_answer(probabilities={"0": 0.1, "1": 0.3, "2": 0.6, "3": 0.0}),
    score_answer(probabilities={"0": 0.1, "1": True, "2": 0.6}),
    score_answer(probabilities={"0": -0.1, "1": 0.5, "2": 0.6}),
    score_answer(probabilities={"0": 0.1, "1": 0.2, "2": 0.3}),
    score_answer(score=0.5),
    score_answer(confidence="0.8"),
    score_answer(confidence=1.01),
    score_answer(legend={"0": "poor", "1": "fair", "2": "wrong"}),
    score_answer(legend=["poor", "fair", "excellent"]),
    {"type": "choice", "choice": "1"},
])
def test_invalid_ordinal_answer_is_unavailable(ordinal, answer):
    def handle(request):
        return httpx.Response(200, content=json.dumps(response({"quality": answer})).encode("utf-8"))

    with client_for(handle) as client:
        criterion, = ClefFlashAdapter(client).evaluate({"text": "evidence"}, (ordinal,), ()).criteria
    assert criterion.criterion_id == ordinal.criterion_id
    assert criterion.value is None and criterion.confidence is None and criterion.error


def test_missing_answer_does_not_discard_valid_peer(binary, ordinal):
    def handle(request):
        return httpx.Response(200, json=response({"quality": score_answer()}))

    with client_for(handle) as client:
        result = ClefFlashAdapter(client).evaluate({"brief": "render"}, (binary, ordinal), ())
    assert [item.criterion_id for item in result.criteria] == ["visible", "quality"]
    assert result.criteria[0].value is None and "Missing" in result.criteria[0].error
    assert result.criteria[1].value == pytest.approx(3.7)
    assert result.input_tokens == 175


@pytest.mark.parametrize("body", [
    [],
    {"model": "clef-flash", "answers": {}},
    response([]),
    response({}, input_tokens=True),
    response({}, output_tokens=-1),
    response({}, input_tokens=1.5),
    response({}, model=""),
    {"error": "sensitive data echoed by server"},
])
def test_invalid_response_envelope_raises_safe_error(binary, body):
    def handle(request):
        return httpx.Response(200, json=body)

    with client_for(handle) as client, pytest.raises(DecisionError) as error:
        ClefFlashAdapter(client).evaluate({"private": "secret"}, (binary,), ())
    assert "secret" not in str(error.value)
    assert "sensitive data" not in str(error.value)


def test_invalid_json_is_not_interpreted_as_answer(binary):
    with (
        client_for(lambda request: httpx.Response(200, content=b"broken-json")) as client,
        pytest.raises(DecisionError, match="invalid JSON"),
    ):
        ClefFlashAdapter(client).evaluate({"brief": "hi"}, (binary,), ())


@pytest.mark.parametrize("status", [400, 404, 413, 500, 302])
def test_http_errors_do_not_include_server_payload_or_credentials(binary, status):
    def handle(request):
        return httpx.Response(status, json={"error": "Authorization: Bearer secret; evidence: sensitive"},
                              headers={"location": "http://creds.example/private"} if status == 302 else None)

    with client_for(handle) as client, pytest.raises(DecisionError) as error:
        ClefFlashAdapter(client).evaluate({"secret": "sensitive"}, (binary,), ())
    assert str(status) in str(error.value)
    assert "secret" not in str(error.value) and "sensitive" not in str(error.value)


def test_transport_error_and_timeout_are_safe(binary):
    for failure, diagnostic in [(httpx.ConnectError("Bearer secret"), "transport"),
                                (httpx.ReadTimeout("Bearer secret"), "timed out")]:
        def handle(request, failure=failure):
            raise failure

        with client_for(handle) as client, pytest.raises(DecisionError) as error:
            ClefFlashAdapter(client).evaluate({"secret": "sensitive"}, (binary,), ())
        assert diagnostic in str(error.value)
        assert "secret" not in str(error.value)


def test_batches_at_64_without_dropping_answers_or_counting_usage_once(binary):
    questions = tuple(DecisionQuestion(f"criterion_{index}", binary.instructions,
                                       binary.values, binary.descriptions, binary=True) for index in range(130))
    sizes = []

    def handle(request):
        body = json.loads(request.content)
        assert body["state"] == {"brief": "shared across all batches"}
        keys = list(body["questions"])
        assert keys == [question.criterion_id for question in questions[len(sizes) * 64:len(sizes) * 64 + 64]]
        sizes.append(len(keys))
        return httpx.Response(200, json=response({key: {"type": "noul", "noul": 1.0} for key in keys},
                                                 input_tokens=100, output_tokens=len(keys)))

    with client_for(handle) as client:
        result = ClefFlashAdapter(client).evaluate({"brief": "shared across all batches"}, questions, ())
    assert sizes == [64, 64, 2]
    assert (result.input_tokens, result.output_tokens) == (300, 130)
    assert [item.criterion_id for item in result.criteria] == [question.criterion_id for question in questions]
    assert all(item.value == 1 and item.confidence == 1 for item in result.criteria)


def test_unexpected_answer_identifier_is_envelope_error(binary):
    with (
        client_for(lambda request: httpx.Response(200, json=response({"other": {"type": "noul", "noul": 0.3}}))) as client,
        pytest.raises(DecisionError, match="unknown criteria"),
    ):
        ClefFlashAdapter(client).evaluate({"brief": "hi"}, (binary,), ())


def test_inconsistent_model_across_batches(binary):
    questions = tuple(DecisionQuestion(str(i), binary.instructions, binary.values, binary.descriptions, True)
                      for i in range(65))
    seen = 0

    def handle(request):
        nonlocal seen
        seen += 1
        return httpx.Response(200, json=response({}, model="clef-flash" if seen == 1 else "nimble"))

    with client_for(handle) as client, pytest.raises(DecisionError, match="inconsistent models"):
        ClefFlashAdapter(client).evaluate({"brief": "hi"}, questions, ())


def test_text_payload_over_64_kib_is_refused_without_request(binary):
    with (
        client_for(lambda request: pytest.fail("oversized body sent")) as client,
        pytest.raises(DecisionError, match="64 KiB without images"),
    ):
        ClefFlashAdapter(client).evaluate({"unabridged": "x" * (64 * 1024)}, (binary,), ())


def test_image_payload_over_32_mib_is_refused_without_request(binary, tmp_path):
    picture = tmp_path / "large.png"
    with picture.open("wb") as image:
        image.truncate(25 * 1024 * 1024)  # Base64 alone exceeds 32 MiB.
    with (
        client_for(lambda request: pytest.fail("oversized body sent")) as client,
        pytest.raises(DecisionError, match="32 MiB with images"),
    ):
        ClefFlashAdapter(client).evaluate({"text": "all"}, (binary,), (picture,))


def test_images_use_32_mib_limit_including_json_overhead(binary, tmp_path):
    picture = tmp_path / "reference.png"
    picture.write_bytes(b"\x89PNG\r\n\x1a\n")
    state = {"full_report": "x" * (64 * 1024)}
    seen = []

    def handle(request):
        seen.append(len(request.content))
        assert json.loads(request.content)["state"] == state
        return httpx.Response(200, json=response({"visible": {"type": "noul", "noul": 0.5}}))

    with client_for(handle) as client:
        result = ClefFlashAdapter(client).evaluate(state, (binary,), (picture,))
    assert len(seen) == 1 and seen[0] > 64 * 1024
    assert result.criteria[0].value == 0.5
    assert result.criteria[0].confidence == pytest.approx(0)


def test_duplicate_question_identifiers_rejected_without_network(binary):
    with (
        client_for(lambda request: pytest.fail("duplicate criteria sent")) as client,
        pytest.raises(DecisionError, match="must be unique"),
    ):
        ClefFlashAdapter(client).evaluate({}, (binary, binary), ())


def test_empty_questions_does_not_make_request():
    with client_for(lambda request: pytest.fail("empty question set sent")) as client:
        result = ClefFlashAdapter(client).evaluate({"text": "all"}, (), ())
    assert result.criteria == () and result.model == "clef-flash"


def test_unserializable_state_and_missing_image_are_safe(binary, tmp_path):
    with client_for(lambda request: pytest.fail("invalid evidence sent")) as client:
        with pytest.raises(DecisionError, match="could not read an evidence image"):
            ClefFlashAdapter(client).evaluate({}, (binary,), (tmp_path / "missing.png",))
        with pytest.raises(DecisionError, match="not valid JSON"):
            ClefFlashAdapter(client).evaluate({"private": object()}, (binary,), ())
