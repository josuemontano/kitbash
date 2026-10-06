"""Retopology contract shared by every method: turn the raw Trellis mesh into the mesh the build script imports."""

from enum import StrEnum
from pathlib import Path
from typing import Protocol

from attrs import frozen


class RetopologyMethod(StrEnum):
    TRIFLOW = "triflow"  # learned artist-like topology (vendored TriFlow), the default
    DECIMATE = "decimate"  # the original behaviour: the build script collapses/voxel-remeshes in Blender (kb.decimate)


@frozen
class RetopologyResult:
    method: RetopologyMethod
    mesh_path: Path  # mesh the build script imports; same coordinate frame (and up axis) as the input
    faces_in: int | None
    faces_out: int | None
    duration_s: float
    device: str | None = None

    def to_extra(self) -> dict:
        return {
            "method": self.method.value, "faces_in": self.faces_in, "faces_out": self.faces_out,
            "duration_s": round(self.duration_s, 2), "device": self.device,
        }


class Retopologizer(Protocol):
    method: RetopologyMethod

    def check(self) -> None:
        """Preflight: raise PreflightError when the method cannot run (missing dependency, weights, device)."""

    def retopologize(self, mesh_path: Path, out_dir: Path, asset_name: str) -> RetopologyResult:
        """Return a result whose mesh_path lives in ``out_dir`` (or is ``mesh_path`` itself for pass-through methods).
        Raises RetopologyError on failure."""
