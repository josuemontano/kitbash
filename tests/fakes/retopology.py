"""Stand-in for the TriFlow engine: rewrites the Trellis OBJ into ``out_dir`` and reports face counts."""

import time
from pathlib import Path

from kitbash.errors import PreflightError, RetopologyError
from kitbash.retopology.base import RetopologyMethod, RetopologyResult


class FakeTriflowRetopologizer:
    method = RetopologyMethod.TRIFLOW

    def __init__(
        self, *, fail: str | None = None, fail_for: tuple[str, ...] | None = None, missing: str | None = None, sleep_s: float = 0.0
    ) -> None:
        self.fail = fail
        self.fail_for = fail_for
        self.missing = missing
        self.sleep_s = sleep_s
        self.calls: list[tuple[Path, Path, str]] = []

    def check(self) -> None:
        if self.missing:
            raise PreflightError(self.missing, hint="Download the TriFlow weights (fake).")

    def retopologize(self, mesh_path: Path, out_dir: Path, asset_name: str) -> RetopologyResult:
        self.calls.append((mesh_path, out_dir, asset_name))
        time.sleep(self.sleep_s)
        if self.fail and (self.fail_for is None or asset_name in self.fail_for):
            raise RetopologyError(self.fail)
        out_dir.mkdir(parents=True, exist_ok=True)
        text = mesh_path.read_text(encoding="utf-8")
        target = out_dir / f"{asset_name}{mesh_path.suffix}"
        target.write_text("# retopologized (fake)\n" + text, encoding="utf-8")
        faces = sum(1 for line in text.splitlines() if line.startswith("f "))
        return RetopologyResult(
            method=self.method, mesh_path=target, faces_in=faces, faces_out=faces // 2, duration_s=self.sleep_s or 0.01, device="cpu"
        )
