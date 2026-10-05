"""``omp`` agent adapter: every LLM call in kitbash goes through here."""

import json
import re
import threading
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from attrs import frozen

from kitbash.errors import LLMAccessError, LLMError, PreflightError
from kitbash.infra.process import ProcessResult, run_process
from kitbash.llm.client import LLMRequest, LLMResponse, Usage

PRINT_FLAGS = ("-p", "--mode", "json", "--no-session", "--no-extensions", "--no-skills", "--no-rules", "--no-lsp", "--no-title")
NON_RETRYABLE_MARKERS = ("not found", "No API key", "Unknown model")
ACCESS_PROBLEMS = (
    (re.compile(r"budget (has been )?exceeded|budget_exceeded|insufficient_quota|quota exceeded", re.IGNORECASE),
     "budget or quota exhausted", "Raise the budget or switch models (--model.<role>=...), then `kitbash resume`."),
    (re.compile(r"\b40[13]\b|unauthori[sz]ed|forbidden|not enough segments|invalid (api )?(key|token)|token (has )?expired",
                re.IGNORECASE),
     "credentials rejected", "Refresh the provider credentials omp uses (for example the gateway token), then `kitbash resume`."),
)


class OmpClient:
    def __init__(
        self,
        executable: str,
        *,
        timeout_s: float,
        retries: int,
        retry_backoff_s: float,
        extra_args: Sequence[str],
        workdir: Path,
        transcript_dir: Path,
    ) -> None:
        self._executable = executable
        self._timeout_s = timeout_s
        self._retries = retries
        self._backoff = retry_backoff_s
        self._extra_args = tuple(extra_args)
        self._workdir = workdir
        self._transcripts = transcript_dir
        self._counter = 0
        self._lock = threading.Lock()

    def complete(self, request: LLMRequest) -> LLMResponse:
        for attempt in range(self._retries + 1):
            try:
                return self._complete_once(request)
            except LLMAccessError:
                raise
            except LLMError as exc:
                retryable = not any(marker in str(exc) for marker in NON_RETRYABLE_MARKERS)
                if not retryable or attempt == self._retries:
                    raise
                time.sleep(self._backoff * (attempt + 1))
        raise AssertionError("unreachable")

    def _complete_once(self, request: LLMRequest) -> LLMResponse:
        transcript = self._transcript_path(request)
        result = run_process(self._command(request), timeout_s=self._timeout_s, cwd=self._workdir)
        if result.timed_out:
            self._write_transcript(transcript, request, result, None)
            raise LLMError(f"omp timed out after {self._timeout_s:.0f}s on task {request.task!r}", log_path=transcript)
        try:
            response = parse_print_output(result.stdout.splitlines(), duration_s=result.duration_s)
        except LLMError as exc:
            self._write_transcript(transcript, request, result, None)
            detail = _error_message(result.stdout) or (result.stderr.strip().splitlines() or [str(exc)])[-1]
            message = f"omp failed on task {request.task!r} with model {request.model!r}: {detail}"
            evidence = f"{exc} {result.stdout[-4000:]} {result.stderr[-4000:]}"
            for pattern, problem, hint in ACCESS_PROBLEMS:
                if pattern.search(evidence):
                    raise LLMAccessError(
                        f"The model provider refused {request.model!r}: {problem} ({detail[:200]})", hint=hint, log_path=transcript
                    ) from exc
            raise LLMError(message, log_path=transcript) from exc
        self._write_transcript(transcript, request, result, response)
        return response

    def _command(self, request: LLMRequest) -> list[str]:
        args = [self._executable, *PRINT_FLAGS, "--model", request.model, "--cwd", str(self._workdir)]
        args += ["--tools", ",".join(request.tools)] if request.tools else ["--no-tools"]
        if request.thinking:
            args += ["--thinking", request.thinking]
        if request.system:
            args += ["--system-prompt", request.system]
        args += self._extra_args
        args += [f"@{path}" for path in request.attachments]
        args.append(request.prompt)
        return args

    def _transcript_path(self, request: LLMRequest) -> Path:
        with self._lock:
            self._counter += 1
            number = self._counter
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return self._transcripts / f"{stamp}_{number:04d}_{request.task.replace('.', '_')}.json"

    @staticmethod
    def _write_transcript(path: Path, request: LLMRequest, result: ProcessResult, response: LLMResponse | None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {
            "task": request.task,
            "role": request.role.value,
            "model": request.model,
            "thinking": request.thinking,
            "attachments": [str(p) for p in request.attachments],
            "system": request.system,
            "prompt": request.prompt,
            "exit_code": result.returncode,
            "duration_s": round(result.duration_s, 3),
        }
        if response is not None:
            record["response"] = response.text
            record["usage"] = {
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
                "cost_usd": response.usage.cost_usd,
            }
        else:
            record["stdout_tail"] = result.stdout[-4000:]
            record["stderr_tail"] = result.stderr[-4000:]
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def parse_print_output(lines: Iterable[str], *, duration_s: float = 0.0) -> LLMResponse:
    """Extract the final assistant message from ``omp -p --mode json`` event lines."""
    final: dict[str, Any] | None = None
    for line in lines:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = event.get("message") if event.get("type") == "message_end" else None
        if message and message.get("role") == "assistant":
            final = message
    if final is None:
        raise LLMError("omp returned no assistant message")
    if final.get("stopReason") == "error" or final.get("errorMessage"):
        raise LLMError(f"omp model error: {final.get('errorMessage') or 'unknown error'}")
    text = "".join(part.get("text", "") for part in final.get("content", []) if part.get("type") == "text")
    if not text.strip():
        raise LLMError("omp returned an empty answer")
    usage = final.get("usage") or {}
    cost = (usage.get("cost") or {}).get("total")
    return LLMResponse(
        text=text,
        usage=Usage(
            input_tokens=int(usage.get("input", 0)),
            output_tokens=int(usage.get("output", 0)),
            cache_read_tokens=int(usage.get("cacheRead", 0)),
            cache_write_tokens=int(usage.get("cacheWrite", 0)),
            reasoning_tokens=int(usage.get("reasoningTokens", 0)),
            cost_usd=float(cost) if cost is not None else None,
        ),
        model=str(final.get("model", "")),
        provider=str(final.get("provider", "")),
        duration_s=duration_s,
    )


def _error_message(stdout: str) -> str | None:
    """The last error message reported in omp's JSON events, if any."""
    message = None
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        found = event.get("errorMessage") or (event.get("message") or {}).get("errorMessage") if isinstance(event, dict) else None
        message = found or message
    return " ".join(message.split())[:300] if message else None


# -- catalog and version -----------------------------------------------------------------------------


@frozen
class CatalogModel:
    provider: str
    id: str
    selector: str
    name: str
    inputs: frozenset[str]

    def matches(self, wanted: str) -> bool:
        wanted = wanted.lower()
        return wanted in {self.id.lower(), self.selector.lower(), self.name.lower()}


@frozen
class OmpCatalog:
    models: tuple[CatalogModel, ...]

    def find(self, wanted: str) -> CatalogModel | None:
        return next((m for m in self.models if m.matches(wanted)), None)

    def suggestions(self, wanted: str, limit: int = 3) -> list[str]:
        tokens = {t for t in wanted.lower().replace("/", "-").split("-") if t}
        ranked = sorted(self.models, key=lambda m: -len(tokens & set(m.id.lower().split("-"))))
        return [m.id for m in ranked[:limit]]


def omp_version(executable: str, timeout_s: float = 30) -> str:
    result = run_process([executable, "--version"], timeout_s=timeout_s)
    if not result.ok:
        raise PreflightError(f"`{executable} --version` failed: {result.tail(5)}")
    return result.stdout.strip().splitlines()[0]


def load_catalog(executable: str, timeout_s: float = 120) -> OmpCatalog:
    result = run_process([executable, "models", "--json"], timeout_s=timeout_s)
    if not result.ok:
        raise PreflightError(f"`{executable} models --json` failed: {result.tail(5)}")
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise PreflightError(f"`{executable} models --json` returned invalid JSON: {exc}") from exc
    entries = data.get("models", data) if isinstance(data, dict) else data
    return OmpCatalog(
        models=tuple(
            CatalogModel(
                provider=str(m.get("provider", "")),
                id=str(m.get("id", "")),
                selector=str(m.get("selector", f"{m.get('provider', '')}/{m.get('id', '')}")),
                name=str(m.get("name", m.get("id", ""))),
                inputs=frozenset(m.get("input", ["text"])),
            )
            for m in entries
            if isinstance(m, dict) and m.get("kind", "chat") == "chat"
        )
    )
