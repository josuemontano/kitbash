"""What the user asked for: a reference image or a text prompt."""

from enum import StrEnum
from pathlib import Path
from typing import Any

from attrs import frozen

from kitbash.errors import ConfigError


class InputMode(StrEnum):
    IMAGE = "image"
    PROMPT = "prompt"


@frozen
class RunInput:
    mode: InputMode
    image: Path | None = None
    prompt: str | None = None

    @classmethod
    def create(cls, image: Path | None, prompt: str | None) -> RunInput:
        if (image is None) == (not prompt):
            raise ConfigError("Pass exactly one of --image or --prompt")
        if image is not None:
            if not image.is_file():
                raise ConfigError(f"Input image not found: {image}")
            return cls(InputMode.IMAGE, image=image.expanduser().resolve())
        return cls(InputMode.PROMPT, prompt=prompt.strip())

    @property
    def references(self) -> tuple[Path, ...]:
        return (self.image,) if self.image else ()

    def describe(self) -> str:
        return f"reference image {self.image.name}" if self.image else f"prompt: {self.prompt}"

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode.value, "image": str(self.image) if self.image else None, "prompt": self.prompt}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunInput:
        return cls(InputMode(data["mode"]), image=Path(data["image"]) if data.get("image") else None, prompt=data.get("prompt"))
