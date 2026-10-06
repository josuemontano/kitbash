"""The mandatory retopology step that follows every Trellis mesh, shared by the modelling pipeline and the Kitbash flow."""

from pathlib import Path

from kitbash.analytics.tracker import SpanKind, Tracker
from kitbash.paths import OutputLayout
from kitbash.retopology.base import Retopologizer, RetopologyMethod, RetopologyResult


def retopologize_mesh(
    retopologizer: Retopologizer, layout: OutputLayout, tracker: Tracker, asset_id: str, attempt: int, mesh_path: Path
) -> RetopologyResult:
    """Retopologize the Trellis mesh. A failure raises RetopologyError: the raw Trellis mesh is never used instead."""
    if retopologizer.method is RetopologyMethod.DECIMATE:  # reduction happens in Blender (kb.decimate), nothing to run here
        return retopologizer.retopologize(mesh_path, mesh_path.parent, asset_id)
    directory = layout.asset_retopo_dir(asset_id) / f"attempt_{attempt:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    with tracker.span(SpanKind.SUBPROCESS, "retopology", attempt=attempt, method=retopologizer.method.value) as span:
        result = retopologizer.retopologize(mesh_path, directory, asset_id)
        span.meta.update(faces_in=result.faces_in, faces_out=result.faces_out, device=result.device)
    return result
