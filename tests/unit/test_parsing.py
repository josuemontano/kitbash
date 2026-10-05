import json

import pytest

from kitbash.errors import LLMError
from kitbash.infra.omp import CatalogModel, OmpCatalog, parse_print_output
from kitbash.llm.parsing import extract_diff, extract_json, extract_python
from kitbash.llm.prompts import PromptLibrary


def test_extract_json_from_fences_and_prose():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('The answer is {"choice": 2, "reason": "x"} as requested.') == {"choice": 2, "reason": "x"}
    assert extract_json("[1, 2]") == [1, 2]
    truncated = '{"summary": "ok", "scores": {"a": {"score": 0.9}}, "edits": [{"instruction": "x"}]'
    assert extract_json(truncated)["scores"] == {"a": {"score": 0.9}}  # the missing final brace is repaired
    assert extract_json('{"text": "brace } inside", "list": [1, 2') == {"text": "brace } inside", "list": [1, 2]}
    with pytest.raises(LLMError):
        extract_json("no json at all {")


def test_extract_python_checks_syntax_for_blender_python():
    code = extract_python("Here:\n```python\nimport kitbash_bpy as kb\nkb.reset_scene()\n```\nDone.")
    assert code == "import kitbash_bpy as kb\nkb.reset_scene()\n"
    with pytest.raises(LLMError, match="syntax error"):
        extract_python("```python\ndef broken(:\n```")
    with pytest.raises(LLMError, match="syntax error"):
        extract_python("```python\ntry:\n    pass\nexcept A, B:\n    pass\n```")  # 3.14-only syntax, Blender runs 3.13


def test_extract_diff_from_fence_or_raw():
    diff = extract_diff("Patch:\n```diff\n--- a/script.py\n+++ b/script.py\n@@ -1 +1 @@\n-a\n+b\n```")
    assert diff.startswith("--- a/script.py") and diff.endswith("+b\n")
    assert extract_diff("@@ -1 +1 @@\n-a\n+b").startswith("@@")
    with pytest.raises(LLMError):
        extract_diff("I would change the roughness.")


def test_parse_omp_json_events():
    lines = [
        json.dumps({"type": "session"}),
        json.dumps({"type": "message_end", "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]}}),
        json.dumps({
            "type": "message_end",
            "message": {
                "role": "assistant", "provider": "aigw", "model": "gemini-3.8-flash", "stopReason": "stop",
                "content": [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "pong"}],
                "usage": {"input": 12, "output": 3, "cacheRead": 1, "reasoningTokens": 2, "cost": {"total": 0.002}},
            },
        }),
        "not json",
    ]
    response = parse_print_output(lines, duration_s=1.5)
    assert response.text == "pong" and response.model == "gemini-3.8-flash"
    assert (response.usage.input_tokens, response.usage.output_tokens, response.usage.cost_usd) == (12, 3, 0.002)


def test_parse_omp_errors():
    error = json.dumps({"type": "message_end", "message": {"role": "assistant", "stopReason": "error", "errorMessage": "quota", "content": []}})
    with pytest.raises(LLMError, match="quota"):
        parse_print_output([error])
    with pytest.raises(LLMError, match="no assistant message"):
        parse_print_output([])


def test_catalog_matching():
    catalog = OmpCatalog(
        models=(
            CatalogModel("aigw", "claude-opus-5", "aigw/claude-opus-5", "claude-opus-5", frozenset({"text", "image"})),
            CatalogModel("aigw", "gpt-6-sol", "aigw/gpt-6-sol", "gpt-6-sol", frozenset({"text"})),
        )
    )
    assert catalog.find("aigw/claude-opus-5").id == "claude-opus-5"
    assert catalog.find("CLAUDE-OPUS-5") is not None
    assert catalog.find("opus-5.5") is None
    assert catalog.suggestions("opus-5.5")[0] == "claude-opus-5"


def test_prompt_templates_render_with_task_marker_and_includes():
    prompts = PromptLibrary()
    text = prompts.render("breakdown_image", "breakdown.analyze.image", {"style": "2d", "max_items": 7, "prompt": ""})
    assert text.startswith("<!-- kitbash-task: breakdown.analyze.image -->")
    assert "at most 7" in text and "Inventory JSON format" in text  # the shared schema is included
    assert prompts.system("visual_critic")
    with pytest.raises(Exception, match="needs variable"):
        prompts.render("breakdown_image", "t", {})


def test_reference_search_queries_broaden_step_by_step():
    from kitbash.services.references import search_queries

    assert search_queries("Small round three-legged oak side table", "isolated white background") == [
        "Small round three-legged oak side table isolated white background",
        "Small round three-legged oak side table",
        "three-legged oak side table",
        "oak side table",
        "side table",
    ]
    assert search_queries("Mug", "") == ["Mug"]




def test_omp_command_enables_only_requested_tools(tmp_path):
    from kitbash.domain.roles import Role
    from kitbash.infra.omp import OmpClient
    from kitbash.llm.client import LLMRequest

    client = OmpClient("omp", timeout_s=1, retries=0, retry_backoff_s=0, extra_args=(), workdir=tmp_path, transcript_dir=tmp_path)
    plain = client._command(LLMRequest("t", Role.CODE, "m", "hello"))
    assert "--no-tools" in plain and "--tools" not in plain and plain[-1] == "hello"
    searching = client._command(LLMRequest("t", Role.REFERENCE_SELECTION, "m", "find", tools=("web_search",)))
    assert searching[searching.index("--tools") + 1] == "web_search" and "--no-tools" not in searching




def test_budget_exhaustion_is_fatal_and_not_retried(tmp_path):
    import json as _json

    from kitbash.domain.roles import Role
    from kitbash.errors import LLMAccessError
    from kitbash.infra.omp import OmpClient
    from kitbash.llm.client import LLMRequest

    fake = tmp_path / "omp"
    calls = tmp_path / "calls"
    event = {"type": "message_end", "message": {"role": "assistant", "content": [], "stopReason": "error",
             "errorMessage": "429 Budget has been exceeded! Current cost: 20.01, Max budget: 20.0"}}
    fake.write_text(f"#!/bin/sh\necho x >> {calls}\necho '{_json.dumps(event)}'\n")
    fake.chmod(0o755)
    client = OmpClient(str(fake), timeout_s=10, retries=3, retry_backoff_s=0, extra_args=(), workdir=tmp_path, transcript_dir=tmp_path / "llm")
    with pytest.raises(LLMAccessError, match="budget or quota exhausted"):
        client.complete(LLMRequest("t", Role.CODE, "m", "hello"))
    assert calls.read_text().count("x") == 1  # no retries


def test_rejected_credentials_are_fatal_with_a_clean_message(tmp_path):
    import json as _json

    from kitbash.domain.roles import Role
    from kitbash.errors import LLMAccessError
    from kitbash.infra.omp import OmpClient
    from kitbash.llm.client import LLMRequest

    fake = tmp_path / "omp"
    event = {"type": "agent_end", "messages": [], "errorMessage": '422 {"msg":"Not enough segments"}'}
    fake.write_text(f"#!/bin/sh\necho '{_json.dumps(event)}'\nexit 1\n")
    fake.chmod(0o755)
    client = OmpClient(str(fake), timeout_s=10, retries=3, retry_backoff_s=0, extra_args=(), workdir=tmp_path, transcript_dir=tmp_path / "llm")
    with pytest.raises(LLMAccessError) as caught:
        client.complete(LLMRequest("t", Role.CODE, "m", "hello"))
    assert "credentials rejected (422" in caught.value.message and "agent_end" not in caught.value.message
    assert "Refresh the provider credentials" in caught.value.hint
