"""Trellis image-to-3D runner with retries, logs and timing."""

import threading
from pathlib import Path

from attrs import frozen

from kitbash.errors import PreflightError, TrellisError
from kitbash.infra.blender import SpanRecorder
from kitbash.infra.process import run_process

MESH_SUFFIXES = (".glb", ".obj")


@frozen
class TrellisAttempt:
    attempt: int
    returncode: int
    duration_s: float
    timed_out: bool
    log_path: Path


@frozen
class TrellisResult:
    mesh_path: Path
    attempts: tuple[TrellisAttempt, ...]

    @property
    def duration_s(self) -> float:
        return sum(a.duration_s for a in self.attempts)

    @property
    def retries(self) -> int:
        return len(self.attempts) - 1


class TrellisRunner:
    def __init__(
        self,
        trellis_dir: Path,
        *,
        python: str,
        steps: int,
        pipeline_type: str,
        no_texture: bool,
        timeout_s: float,
        retries: int,
        max_concurrent: int,
        recorder: SpanRecorder,
    ) -> None:
        self._dir = trellis_dir
        self._python = python
        self._steps = steps
        self._pipeline_type = pipeline_type
        self._no_texture = no_texture
        self._timeout_s = timeout_s
        self._retries = retries
        self._slots = threading.BoundedSemaphore(max(1, max_concurrent))
        self._recorder = recorder

    def check(self) -> None:
        if not (self._dir / "generate.py").is_file():
            raise PreflightError(
                f"Trellis not found: {self._dir / 'generate.py'} does not exist",
                hint="Set paths.trellis in your config to the trellis-mac checkout.",
            )

    def command(self, image: Path, output_stem: Path, seed: int) -> list[str]:
        args = [
            self._python, "generate.py", str(image),
            "--output", str(output_stem),
            "--steps", str(self._steps),
            "--pipeline-type", self._pipeline_type,
            "--seed", str(seed),
        ]
        if self._no_texture:
            args.append("--no-texture")
        return args

    def generate(self, image: Path, out_dir: Path, asset_name: str, *, seed: int) -> TrellisResult:
        """Generate a mesh from ``image``; retries failures, raises TrellisError when every attempt fails."""
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = out_dir / asset_name
        attempts: list[TrellisAttempt] = []
        with self._slots:
            for attempt in range(1, self._retries + 2):
                log_path = out_dir / f"trellis_attempt_{attempt}.log"
                with self._recorder.span("subprocess", "trellis", attempt=attempt, seed=seed + attempt - 1) as span:
                    process = run_process(
                        self.command(image, stem, seed + attempt - 1),
                        timeout_s=self._timeout_s,
                        cwd=self._dir,
                        log_path=log_path,
                    )
                    span.meta.update(exit_code=process.returncode, timed_out=process.timed_out, log=str(log_path))
                attempts.append(
                    TrellisAttempt(attempt, process.returncode, process.duration_s, process.timed_out, log_path)
                )
                mesh = _find_mesh(stem)
                if process.ok and mesh is not None:
                    return TrellisResult(mesh_path=mesh, attempts=tuple(attempts))
        last = attempts[-1]
        reason = "timed out" if last.timed_out else f"exited with code {last.returncode}"
        raise TrellisError(
            f"Trellis {reason} on all {len(attempts)} attempts for {asset_name}",
            hint="Try another reference image (clean background, whole object visible).",
            log_path=last.log_path,
        )


def _find_mesh(stem: Path) -> Path | None:
    for suffix in MESH_SUFFIXES:
        candidate = stem.with_suffix(suffix)
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None
