"""Typed LLM calls: render a prompt, pick the model for (role, phase), parse and validate the answer."""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from attrs import evolve

from kitbash.config import ModelsConfig
from kitbash.domain.phases import PhaseName
from kitbash.domain.roles import Role
from kitbash.errors import KitbashError, LLMError
from kitbash.llm.client import LLMClient, LLMRequest
from kitbash.llm.parsing import extract_diff, extract_json, extract_python
from kitbash.llm.prompts import PromptLibrary

REPAIR_NOTE = (
    "\n\n## Your previous answer could not be used\n{error}\n\n"
    "Answer again, following the output format exactly."
)


class LLMService:
    def __init__(self, client: LLMClient, prompts: PromptLibrary, models: ModelsConfig, *, parse_retries: int = 1) -> None:
        self._client = client
        self._prompts = prompts
        self._models = models
        self._parse_retries = parse_retries

    def model_for(self, role: Role, phase: PhaseName | None) -> str:
        return self._models.model_for(role, phase)

    def ask[T](
        self,
        *,
        task: str,
        role: Role,
        phase: PhaseName | None,
        template: str,
        variables: Mapping[str, Any],
        parse: Callable[[str], T],
        attachments: Sequence[Path] = (),
        tools: Sequence[str] = (),
    ) -> T:
        """Call the model and parse its answer; a parse failure is sent back once for a corrected answer."""
        prompt = self._prompts.render(template, task, variables)
        request = LLMRequest(
            task=task,
            role=role,
            model=self._models.model_for(role, phase),
            prompt=prompt,
            system=self._prompts.system(role.value),
            attachments=tuple(attachments),
            thinking=self._models.thinking_for(role),
            tools=tuple(tools),
        )
        for attempt in range(self._parse_retries + 1):
            response = self._client.complete(request)
            try:
                return parse(response.text)
            except KitbashError as exc:
                if attempt == self._parse_retries:
                    raise LLMError(f"{task}: unusable answer from {request.model}: {exc.message}") from exc
                request = evolve(request, prompt=prompt + REPAIR_NOTE.format(error=exc.message))
        raise AssertionError("unreachable")

    def ask_json[T](self, *, validate: Callable[[Any], T], **kwargs: Any) -> T:
        return self.ask(parse=lambda text: validate(extract_json(text)), **kwargs)

    def ask_python(self, **kwargs: Any) -> str:
        return self.ask(parse=extract_python, **kwargs)

    def ask_diff(self, **kwargs: Any) -> str:
        return self.ask(parse=extract_diff, **kwargs)
