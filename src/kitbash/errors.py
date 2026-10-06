"""Error hierarchy. Every error carries a message the user can act on."""

from pathlib import Path


class KitbashError(Exception):
    """Base class for expected, user-facing failures."""

    def __init__(self, message: str, *, hint: str | None = None, log_path: Path | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.log_path = log_path

    def __str__(self) -> str:
        parts = [self.message]
        if self.hint:
            parts.append(f"Hint: {self.hint}")
        if self.log_path:
            parts.append(f"Log: {self.log_path}")
        return "\n".join(parts)


class ConfigError(KitbashError):
    """Invalid or inconsistent configuration."""


class PreflightError(KitbashError):
    """A required tool, path or model is missing."""


class SubprocessError(KitbashError):
    """A subprocess failed, crashed or timed out."""


class BlenderScriptError(SubprocessError):
    """A script failed inside Blender."""

    def __init__(self, message: str, *, traceback_text: str = "", **kwargs) -> None:
        super().__init__(message, **kwargs)
        self.traceback_text = traceback_text


class TrellisError(SubprocessError):
    """Trellis failed to produce a mesh."""


class LLMError(KitbashError):
    """An omp call failed or returned an unusable answer."""


class LLMAccessError(LLMError):
    """The model provider refuses every call (budget, quota or credentials). Fatal: never retried or swallowed."""


class PatchError(KitbashError):
    """A unified diff does not apply to the script it targets."""


class StateError(KitbashError):
    """The state database is inconsistent with the requested operation."""


class TransitionError(StateError):
    """An asset state transition that the state machine does not allow."""


class BacklotError(KitbashError):
    """The asset library cannot satisfy a request."""


class ReferenceNotFoundError(KitbashError):
    """No usable reference image could be found for an asset."""


class UserAbort(KitbashError):
    """The user stopped the run at a gate."""


class RetopologyError(KitbashError):
    """Retopology of a generated mesh failed."""
