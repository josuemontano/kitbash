"""Provider-neutral LLM request/response types. The only implementation talks to ``omp``."""

from pathlib import Path
from typing import Protocol

from attrs import frozen

from kitbash.domain.roles import Role


@frozen
class LLMRequest:
    task: str
    role: Role
    model: str
    prompt: str
    system: str | None = None
    attachments: tuple[Path, ...] = ()
    thinking: str | None = None
    tools: tuple[str, ...] = ()  # omp agent tools the model may use (e.g. "web_search"); none by default


@frozen
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float | None = None


@frozen
class LLMResponse:
    text: str
    usage: Usage
    model: str
    provider: str = ""
    duration_s: float = 0.0


class LLMClient(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse: ...
