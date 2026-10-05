"""Prompt templates stored as Markdown files in ``kitbash/prompts``.

Templates use ``$name`` placeholders (``$$`` for a literal dollar) and may include shared fragments
with a line ``<<include fragment_name>>``.
"""

import re
from collections.abc import Mapping
from importlib import resources
from string import Template
from typing import Any

from kitbash.errors import KitbashError

_INCLUDE = re.compile(r"^<<include\s+([\w-]+)>>\s*$", re.MULTILINE)
TASK_MARKER = "<!-- kitbash-task: {task} -->"


class PromptLibrary:
    def __init__(self, package: str = "kitbash.prompts") -> None:
        self._package = package
        self._cache: dict[str, str] = {}

    def render(self, name: str, task: str, variables: Mapping[str, Any]) -> str:
        template = Template(self._load(name))
        try:
            body = template.substitute({k: _text(v) for k, v in variables.items()})
        except KeyError as exc:
            raise KitbashError(f"Prompt template {name!r} needs variable {exc.args[0]!r}") from None
        return f"{TASK_MARKER.format(task=task)}\n{body.strip()}\n"

    def fragment(self, name: str) -> str:
        """A template's raw text (for sharing documentation such as the inventory schema)."""
        return self._load(name).strip()

    def system(self, role: str) -> str | None:
        name = f"system_{role}"
        return self._load(name).strip() if self._exists(name) else None

    def _load(self, name: str) -> str:
        if name not in self._cache:
            if not self._exists(name):
                raise KitbashError(f"Missing prompt template {name}.md")
            text = resources.files(self._package).joinpath(f"{name}.md").read_text(encoding="utf-8")
            self._cache[name] = _INCLUDE.sub(lambda m: self._load(m.group(1)).strip(), text)
        return self._cache[name]

    def _exists(self, name: str) -> bool:
        return resources.files(self._package).joinpath(f"{name}.md").is_file()


def _text(value: Any) -> str:
    return value if isinstance(value, str) else str(value)
