"""Headless Blender: run a script with JSON arguments and read back a JSON result."""

import json
import re
from collections.abc import Mapping
from contextlib import nullcontext
from importlib import resources
from pathlib import Path
from typing import Any, Protocol

from attrs import frozen

from kitbash.errors import BlenderScriptError, PreflightError
from kitbash.infra.process import run_process

RUNNER = "kb_runner.py"


def blender_scripts_dir() -> Path:
    return Path(str(resources.files("kitbash.blender")))


def blender_script(name: str) -> Path:
    return blender_scripts_dir() / name


class SpanRecorder(Protocol):
    def span(self, kind: str, name: str, **meta: Any): ...


@frozen
class BlenderCapabilities:
    version: str
    version_tuple: tuple[int, ...]
    python_version: str
    materialx_export: bool
    usd_export_options: tuple[str, ...]

    @classmethod
    def from_probe(cls, data: Mapping[str, Any]) -> BlenderCapabilities:
        options = tuple(data.get("usd_export_options", []))
        return cls(
            version=str(data.get("version", "")),
            version_tuple=tuple(data.get("version_tuple", [])),
            python_version=str(data.get("python_version", "")),
            materialx_export="generate_materialx_network" in options,
            usd_export_options=options,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "version_tuple": list(self.version_tuple),
            "python_version": self.python_version,
            "materialx_export": self.materialx_export,
            "usd_export_options": list(self.usd_export_options),
        }


class BlenderRunner:
    def __init__(self, executable: str, *, timeout_s: float, recorder: SpanRecorder | None = None) -> None:
        self._executable = executable
        self._timeout_s = timeout_s
        self._recorder = recorder

    def run(
        self,
        script: Path,
        *,
        args: Mapping[str, Any],
        log_path: Path,
        blend: Path | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Run ``script`` inside Blender; returns what the script emitted, raises on any failure."""
        args_path = log_path.with_name(log_path.stem + ".args.json")
        result_path = log_path.with_name(log_path.stem + ".result.json")
        args_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.unlink(missing_ok=True)
        payload = {
            "script": str(script),
            "helpers_dir": str(blender_scripts_dir()),
            "result_path": str(result_path),
            "args": dict(args),
        }
        args_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        command = [self._executable, "--factory-startup", "-b"]
        if blend is not None:
            command.append(str(blend))
        command += ["--python-exit-code", "3", "-P", str(blender_script(RUNNER)), "--", str(args_path)]
        timeout = timeout_s or self._timeout_s
        with self._span(script, log_path, blend) as span:
            process = run_process(command, timeout_s=timeout, cwd=script.parent, log_path=log_path)
            if span is not None:
                span.meta.update(exit_code=process.returncode, timed_out=process.timed_out, log=str(log_path))
        outcome = _read_result(result_path)
        if process.timed_out:
            raise BlenderScriptError(f"Blender timed out after {timeout:.0f}s running {script.name}", log_path=log_path)
        if outcome is None:
            raise BlenderScriptError(
                f"Blender exited with code {process.returncode} before {script.name} finished",
                traceback_text=process.tail(30),
                log_path=log_path,
            )
        if not outcome.get("ok"):
            raise BlenderScriptError(
                f"{script.name} failed in Blender: {outcome.get('error', 'unknown error')}",
                traceback_text=str(outcome.get("traceback", "")),
                log_path=log_path,
            )
        return dict(outcome.get("result", {}))

    def probe(self, log_path: Path) -> BlenderCapabilities:
        try:
            data = self.run(blender_script("probe.py"), args={}, log_path=log_path, timeout_s=180)
        except BlenderScriptError as exc:
            raise PreflightError(f"Blender capability probe failed: {exc.message}", log_path=exc.log_path) from exc
        return BlenderCapabilities.from_probe(data)

    def version(self) -> str:
        result = run_process([self._executable, "--version"], timeout_s=60)
        if not result.ok:
            raise PreflightError(f"`{self._executable} --version` failed: {result.tail(5)}")
        match = re.search(r"Blender\s+[\w.\- ]+", result.stdout)
        return match.group(0).strip() if match else result.stdout.strip().splitlines()[0]

    def _span(self, script: Path, log_path: Path, blend: Path | None):
        if self._recorder is None:
            return nullcontext()
        fixed = script.parent == blender_scripts_dir()
        name = f"blender:{script.stem if fixed else log_path.stem}"  # generated scripts are all called script.py
        return self._recorder.span("subprocess", name, blend=str(blend) if blend else None)


def _read_result(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
