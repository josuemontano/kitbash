"""The plan printed by ``--dry-run``: phases, models, paths and estimated steps. Runs nothing."""

import shutil
from collections.abc import Callable
from pathlib import Path

from attrs import frozen

from kitbash.config import Config
from kitbash.domain.phases import PhaseName
from kitbash.domain.roles import Role
from kitbash.domain.rubric import Rubric
from kitbash.domain.run_input import InputMode, RunInput
from kitbash.errors import KitbashError
from kitbash.infra.blender import SpanRecorder
from kitbash.paths import OutputLayout
from kitbash.retopology import NullRecorder, make_retopologizer
from kitbash.retopology.base import Retopologizer, RetopologyMethod


@frozen
class PlanRow:
    phase: str
    step: str
    estimate: str


class Planner:
    def __init__(
        self,
        config: Config,
        layout: OutputLayout,
        run_input: RunInput,
        rubric: Rubric,
        retopology_factory: Callable[[Config, SpanRecorder], Retopologizer] = make_retopologizer,
    ) -> None:
        self._retopology_factory = retopology_factory
        self._config = config
        self._layout = layout
        self._input = run_input
        self._rubric = rubric

    def _model(self, role: Role, phase: PhaseName) -> str:
        return self._config.models.model_for(role, phase)

    def _critics(self, phase: PhaseName) -> str:
        return f"{self._model(Role.VISUAL_CRITIC, phase)} + {self._model(Role.TECHNICAL_CRITIC, phase)}"

    def steps(self) -> list[PlanRow]:
        c = self._config.critic.max_cycles
        cfg = self._config
        analysis_role = Role.IMAGE_ANALYSIS if self._input.mode is InputMode.IMAGE else Role.PROMPT_ANALYSIS
        B, M, L = PhaseName.BREAKDOWN, PhaseName.MODELLING, PhaseName.LAYOUT
        return [
            PlanRow("breakdown", f"analyze the {self._input.mode.value}", f"1 call to {self._model(analysis_role, B)}"),
            PlanRow("breakdown", "critic loop (blockout render vs reference)", f"<= {c} cycles x (2 Blender runs, 2 critics: {self._critics(B)})"),
            PlanRow("breakdown", "patches", f"<= {c - 1} calls to {self._model(Role.CODE, B)}"),
            PlanRow("breakdown", "backlot lookup, unrecognized items, user gate", "embedding search per item"),
            PlanRow("modelling", "per asset: reference image", f"{', '.join(cfg.reference.providers)} + 1 call to {self._model(Role.REFERENCE_SELECTION, M)}"),
            PlanRow("modelling", "per asset: Trellis", f"1 run (<= {cfg.trellis.retries} retries), {cfg.trellis.steps} steps, pipeline {cfg.trellis.pipeline_type}"),
            PlanRow("modelling", "per asset: retopology", self._retopology_estimate()),
            PlanRow("modelling", "per asset: build script", f"1 call to {self._model(Role.CODE, M)}"),
            PlanRow("modelling", "per asset: critic loop", f"<= {c} cycles x (7 Blender runs incl. USD export + round trip, 2 critics: {self._critics(M)})"),
            PlanRow("modelling", "per asset: patches", f"<= {c - 1} calls to {self._model(Role.CODE, M)}"),
            PlanRow("modelling", "pipeline", f"{cfg.pipeline.threads} workers, review buffer {cfg.pipeline.review_buffer}, max LLM calls per asset {3 * c + 1}"),
            PlanRow("layout", "layout script", f"1 call to {self._model(Role.CODE, L)}"),
            PlanRow("layout", "critic loop", f"<= {c} cycles x (3 Blender runs, 2 critics: {self._critics(L)})"),
            PlanRow("layout", "patches and user gate", f"<= {c - 1} calls to {self._model(Role.CODE, L)}"),
            PlanRow("assembly", "copy assets, rebuild scene, localize files", "2 Blender runs"),
            PlanRow("assembly", "final render", f"{cfg.blender.final_resolution[0]}x{cfg.blender.final_resolution[1]}, {cfg.blender.final_samples} samples"),
            PlanRow("assembly", "USD export + round trip", f"materialx={cfg.usd.materialx}, bake fallback {cfg.usd.bake_resolution}px"),
        ]

    def _retopology_estimate(self) -> str:
        r = self._config.retopology
        if r.method_enum is RetopologyMethod.DECIMATE:
            return "decimate: none (the build script collapses/voxel-remeshes the Trellis mesh in Blender)"
        fallback = "falls back to the Trellis mesh on error" if r.fallback_on_error else "fails the asset on error"
        return f"triflow: 1 run, {r.face_count} faces, {r.flow_steps} flow steps, device {r.device}; {fallback}"

    def _retopology_paths(self) -> list[tuple[str, str, str]]:
        cfg = self._config
        if cfg.retopology.method_enum is RetopologyMethod.DECIMATE:
            return []
        try:
            self._retopology_factory(cfg, NullRecorder()).check()
        except KitbashError as exc:
            status = f"NOT READY: {exc.message}"
        else:
            status = "ok"
        return [("triflow weights", str(cfg.paths.triflow_weights), status)]

    def models(self) -> list[tuple[str, str, str]]:
        return [(phase.value if phase else "all", role.value, model) for phase, role, model in self._config.models.all_assignments()]

    def paths(self) -> list[tuple[str, str, str]]:
        cfg = self._config
        return [
            ("output", str(self._layout.root), "exists" if self._layout.root.exists() else "will be created"),
            ("input", self._input.describe(), ""),
            ("backlot", str(cfg.paths.backlot), "exists" if cfg.paths.backlot.exists() else "will be created"),
            ("trellis", str(cfg.paths.trellis), "ok" if (cfg.paths.trellis / "generate.py").is_file() else "MISSING generate.py"),
            ("trellis python", cfg.tools.resolved_trellis_python(cfg.paths.trellis), ""),
            *self._retopology_paths(),
            ("blender", cfg.tools.blender, "found" if _found(cfg.tools.blender) else "NOT FOUND"),
            ("omp", cfg.tools.omp, "found" if _found(cfg.tools.omp) else "NOT FOUND"),
            ("rubric", self._rubric.source, f"{len(self._rubric.criteria)} criteria"),
            ("style", cfg.pipeline.style, cfg.style.render_engine),
        ]


def _found(executable: str) -> bool:
    return shutil.which(executable) is not None or Path(executable).is_file()
