"""Startup checks: omp, Blender, Trellis and every configured model, failing fast with clear messages."""

import shutil
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from attrs import frozen

from kitbash.analytics import context
from kitbash.config import Config
from kitbash.domain.roles import Role
from kitbash.errors import LLMError, PreflightError
from kitbash.infra.blender import BlenderCapabilities, BlenderRunner
from kitbash.infra.omp import OmpClient, load_catalog, omp_version
from kitbash.infra.process import current_registry, defer_interrupts
from kitbash.llm.client import LLMRequest
from kitbash.llm.prompts import TASK_MARKER
from kitbash.retopology.base import Retopologizer, RetopologyMethod


@frozen
class PreflightReport:
    omp_version: str
    blender_version: str
    blender: BlenderCapabilities
    trellis_python: str
    models: dict[str, str]

    def versions(self) -> dict[str, str]:
        return {"omp": self.omp_version, "blender": self.blender_version, "blender_python": self.blender.python_version}


def resolve_executable(name: str, what: str, hint: str) -> str:
    path = shutil.which(name) or (name if Path(name).is_file() else None)
    if path is None:
        raise PreflightError(f"{what} not found: {name!r} is not on PATH", hint=hint)
    return path


def ping_models(config: Config, omp: str, models: Mapping[str, Role], log_dir: Path) -> tuple[list[str], list[str]]:
    """One tiny call per model, in parallel: catches rejected credentials or budgets before any work."""
    client = OmpClient(
        omp, timeout_s=min(config.omp.timeout_s, 180), retries=0, retry_backoff_s=0, extra_args=config.omp.extra_args,
        workdir=log_dir, transcript_dir=log_dir / "llm",
    )

    def ping(model: str, role: Role) -> tuple[str, str | None] | None:
        prompt = f"{TASK_MARKER.format(task='preflight.ping')}\nReply with exactly: pong"
        try:
            client.complete(LLMRequest(task="preflight.ping", role=role, model=model, prompt=prompt))
        except LLMError as exc:
            return f"Model {model!r} does not answer through omp: {exc.message}", exc.hint
        return None

    registry = current_registry()
    with registry.bind(), ThreadPoolExecutor(max_workers=max(1, len(models))) as pool:
        try:
            registry.check_cancelled()
            futures = [pool.submit(context.propagate(ping), model, role) for model, role in models.items()]
            for future in as_completed(futures):
                future.result()
            answers = [answer for future in futures if (answer := future.result()) is not None]
            registry.check_cancelled()
        except BaseException:
            with defer_interrupts():
                registry.terminate_all()
                pool.shutdown(wait=True, cancel_futures=True)
            raise
    return [message for message, _ in answers], [hint for _, hint in answers if hint]


def run_preflight(config: Config, log_dir: Path, retopologizer: Retopologizer | None) -> PreflightReport:
    problems: list[str] = []
    retopology_hints: list[str] = []
    omp = resolve_executable(config.tools.omp, "omp", "Install omp or set tools.omp in your config.")
    blender = resolve_executable(config.tools.blender, "Blender", "Install Blender or set tools.blender in your config.")
    trellis_python = ""
    if config.modelling.method == "trellis":
        trellis_python = config.tools.resolved_trellis_python(config.paths.trellis)
        if not (config.paths.trellis / "generate.py").is_file():
            problems.append(f"Trellis not found: {config.paths.trellis / 'generate.py'} does not exist (set paths.trellis)")
        if shutil.which(trellis_python) is None and not Path(trellis_python).is_file():
            problems.append(f"Trellis Python not found: {trellis_python} (set tools.trellis_python)")
        if retopologizer is None:
            problems.append("Trellis modelling requires a retopologizer")
        elif retopologizer.method is RetopologyMethod.TRIFLOW:
            try:
                retopologizer.check()
            except PreflightError as exc:
                problems.append(exc.message)
                if exc.hint:
                    retopology_hints.append(exc.hint)

    version = omp_version(omp)
    catalog = load_catalog(omp)
    models: dict[str, str] = {}
    for phase, role, model in config.models.all_assignments():
        where = f"{role.value}" + (f" in {phase.value}" if phase else "")
        found = catalog.find(model)
        if found is None:
            suggestions = ", ".join(catalog.suggestions(model))
            problems.append(f"Model {model!r} ({where}) is not available in omp. Close matches: {suggestions}")
            continue
        if config.models.needs_images(role) and "image" not in found.inputs:
            problems.append(f"Model {model!r} ({where}) does not accept images, but the {role.value} role sends images")
        models[where] = found.selector
    hints = ["Fix the config (see `omp models` for model names) or override with --model.<role>=<name>."]
    if config.omp.preflight_ping and not problems:
        failures, ping_hints = ping_models(config, omp, {model: role for _, role, model in config.models.all_assignments()}, log_dir)
        problems += failures
        hints = list(dict.fromkeys(ping_hints)) or hints
    runner = BlenderRunner(blender, timeout_s=config.blender.timeout_s)
    blender_version = runner.version()
    capabilities = runner.probe(log_dir / "blender_probe.log")
    if problems:
        raise PreflightError("Preflight failed:\n  - " + "\n  - ".join(problems), hint=" ".join([*retopology_hints, *hints]))
    return PreflightReport(
        omp_version=version,
        blender_version=blender_version,
        blender=capabilities,
        trellis_python=trellis_python,
        models=models,
    )

