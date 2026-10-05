"""Decorator that records every LLM call (tokens, cost, duration) as an analytics span."""

from kitbash.analytics.tracker import EventKind, SpanKind, Tracker
from kitbash.llm.client import LLMClient, LLMRequest, LLMResponse


class TrackedLLMClient:
    def __init__(self, inner: LLMClient, tracker: Tracker) -> None:
        self._inner = inner
        self._tracker = tracker

    def complete(self, request: LLMRequest) -> LLMResponse:
        with self._tracker.span(SpanKind.LLM, request.task, model=request.model, role=request.role.value) as span:
            response = self._inner.complete(request)
            usage = response.usage
            span.meta.update(
                tokens_in=usage.input_tokens,
                tokens_out=usage.output_tokens,
                cache_read_tokens=usage.cache_read_tokens,
                reasoning_tokens=usage.reasoning_tokens,
                cost_usd=usage.cost_usd,
                provider=response.provider,
                resolved_model=response.model,
            )
            return response

    def note_retry(self, request: LLMRequest, reason: str) -> None:
        self._tracker.event(EventKind.RETRY, request.task, reason=reason[:300], model=request.model)
