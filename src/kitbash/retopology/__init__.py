"""Retopology methods: turn the raw Trellis mesh into the mesh the build script imports."""

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from kitbash.errors import PreflightError
from kitbash.retopology.base import Retopologizer, RetopologyMethod
from kitbash.retopology.decimate import DecimateRetopologizer

if TYPE_CHECKING:
    from kitbash.config import Config
    from kitbash.infra.blender import SpanRecorder


class NullRecorder:
    """A SpanRecorder that records nothing, for checks that run outside a pipeline (dry run)."""

    @contextmanager
    def span(self, kind: str, name: str, **meta: Any) -> Iterator[SimpleNamespace]:
        yield SimpleNamespace(kind=kind, name=name, meta=dict(meta))


def make_retopologizer(config: Config, recorder: SpanRecorder) -> Retopologizer:
    """The configured method. TriFlow (and torch with it) is imported only when it is selected."""
    settings = config.retopology
    match settings.method_enum:
        case RetopologyMethod.DECIMATE:
            return DecimateRetopologizer()
        case RetopologyMethod.TRIFLOW:
            try:
                from kitbash.retopology.triflow.engine import TriflowRetopologizer
            except ImportError as exc:
                raise PreflightError(
                    f"TriFlow retopology is not available: {exc}",
                    hint="Install the TriFlow dependencies, or run with --retopology decimate.",
                ) from exc

            return TriflowRetopologizer(
                face_count=settings.face_count,
                qem_threshold=settings.qem_threshold,
                quad_ratio=settings.quad_ratio,
                flow_steps=settings.flow_steps,
                device=settings.device,
                weights_dir=config.paths.triflow_weights,
                recorder=recorder,
                seed=config.trellis.seed,
            )
