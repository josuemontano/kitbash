"""The original behaviour: no retopology step, the build script decimates the raw Trellis mesh (``kb.decimate``)."""

from pathlib import Path

from kitbash.retopology.base import RetopologyMethod, RetopologyResult


class DecimateRetopologizer:
    method = RetopologyMethod.DECIMATE

    def check(self) -> None:
        return None

    def retopologize(self, mesh_path: Path, out_dir: Path, asset_name: str) -> RetopologyResult:
        return RetopologyResult(method=self.method, mesh_path=mesh_path, faces_in=None, faces_out=None, duration_s=0.0)
