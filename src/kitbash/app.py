"""Composition root: builds every service for one scene output directory and wires them together."""

import logging
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rich.console import Console

from kitbash import __version__
from kitbash.agents.breakdown import BreakdownAgent
from kitbash.agents.layout import LayoutAgent
from kitbash.agents.modelling import ModellingAgent
from kitbash.agents.scene import SceneAgent
from kitbash.analytics.report import AnalyticsReport
from kitbash.analytics.tracker import Tracker
from kitbash.backlot.library import Backlot
from kitbash.config import Config, default_rubric_path, load_config
from kitbash.critique.critics import PatchWriter, TechnicalCritic, VisualCritic
from kitbash.critique.loop import CriticLoop
from kitbash.critique.sessions import LoopSessions, ResumableLoop
from kitbash.critique.store import CycleStore
from kitbash.domain.phases import PhaseName
from kitbash.domain.rubric import Rubric
from kitbash.domain.run_input import RunInput
from kitbash.errors import ConfigError, StateError
from kitbash.infra.blender import BlenderRunner
from kitbash.infra.embeddings import Embedder, make_embedder
from kitbash.infra.image_search import (
    CandidateProvider,
    ImageDownloader,
    InputCropProvider,
    OpenverseProvider,
    WikimediaProvider,
    make_http_client,
)
from kitbash.infra.omp import OmpClient
from kitbash.infra.polyhaven import PolyHavenCatalog
from kitbash.infra.trellis import TrellisRunner
from kitbash.interaction.autopilot import AutoPilot
from kitbash.interaction.protocols import UserChannel
from kitbash.interaction.terminal import TerminalUser
from kitbash.llm.prompts import PromptLibrary
from kitbash.llm.service import LLMService
from kitbash.llm.tracked import TrackedLLMClient
from kitbash.paths import OutputLayout
from kitbash.phases.assembly import AssemblyPhase
from kitbash.phases.breakdown import BreakdownPhase
from kitbash.phases.layout import LayoutPhase
from kitbash.phases.modelling import ModellingPhase
from kitbash.phases.scene_assets import SceneCast
from kitbash.pipeline.commit import BacklotCommitter
from kitbash.services.blender_toolkit import BlenderToolkit
from kitbash.services.preflight import PreflightReport, run_preflight
from kitbash.services.references import ReferenceFinder
from kitbash.services.usd_fidelity import UsdFidelityChecker
from kitbash.store.state import StateDB
from kitbash.ui.dashboard import Dashboard
from kitbash.ui.images import ImagePresenter

log = logging.getLogger("kitbash")


# -- output directories --------------------------------------------------------------------------------


def create_workspace(
    output: Path, run_input: RunInput, config_path: Path | None, rubric_path: Path | None, overrides: Mapping[str, Any]
) -> tuple[Config, OutputLayout, RunInput]:
    """Validate everything, then snapshot config, rubric and input into a (new or matching) output dir."""
    config = load_config(config_path, overrides)
    rubric_source = rubric_path or default_rubric_path()
    Rubric.load(rubric_source)
    layout = OutputLayout.at(output)
    stored = _stored_input(layout)
    if stored and not _same_input(stored, run_input):
        raise ConfigError(
            f"{layout.root} already holds a run for {stored.describe()}",
            hint="Use `kitbash resume --output ...` to continue it, or choose another --output.",
        )
    layout.create()
    local_input = _copy_input(layout, run_input)
    layout.config_snapshot.write_text(config.snapshot_toml(), encoding="utf-8")
    shutil.copy2(rubric_source, layout.rubric_snapshot)
    state = StateDB(layout.state_db)
    state.meta.set("input", local_input.to_dict())
    state.close()
    return config, layout, local_input


def open_workspace(output: Path, overrides: Mapping[str, Any]) -> tuple[Config, OutputLayout, RunInput]:
    layout = OutputLayout.at(output)
    stored = _stored_input(layout)
    if stored is None or not layout.config_snapshot.exists():
        raise StateError(f"No kitbash run in {layout.root}", hint="Start one with `kitbash build`.")
    return load_config(layout.config_snapshot, overrides), layout, stored


def _stored_input(layout: OutputLayout) -> RunInput | None:
    if not layout.state_db.exists():
        return None
    state = StateDB(layout.state_db)
    try:
        data = state.meta.get("input")
    finally:
        state.close()
    return RunInput.from_dict(data) if data else None


def _copy_input(layout: OutputLayout, run_input: RunInput) -> RunInput:
    if run_input.image is None:
        (layout.input_dir / "prompt.txt").write_text(run_input.prompt or "", encoding="utf-8")
        return run_input
    target = layout.input_dir / f"reference{run_input.image.suffix.lower()}"
    if run_input.image.resolve() != target.resolve():
        shutil.copy2(run_input.image, target)
    return RunInput(run_input.mode, image=target, prompt=None)


def _same_input(stored: RunInput, new: RunInput) -> bool:
    if stored.mode is not new.mode:
        return False
    if stored.image and new.image:
        return stored.image.name.startswith("reference") and stored.image.read_bytes() == new.image.read_bytes()
    return stored.prompt == new.prompt


def configure_logging(layout: OutputLayout) -> None:
    handler = logging.FileHandler(layout.logs_dir / "kitbash.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s"))
    root = logging.getLogger("kitbash")
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    root.propagate = False


# -- the application ------------------------------------------------------------------------------------


class Application:
    """Everything one run needs, built once. Every dependency is injected here and nowhere else."""

    def __init__(self, config: Config, layout: OutputLayout, run_input: RunInput, *, interactive: bool, console: Console) -> None:
        configure_logging(layout)
        self.config, self.layout, self.run_input, self.console = config, layout, run_input, console
        self.embedder: Embedder = make_embedder(config.embedding)
        self.state = StateDB(layout.state_db, self.embedder)
        self.tracker = Tracker(self.state.spans)
        self.rubric = Rubric.load(layout.rubric_snapshot)
        self.dashboard = Dashboard(console, config.ui.refresh_per_second)
        images = ImagePresenter(console, enabled=config.ui.show_previews, open_files=interactive)
        self.user: UserChannel = TerminalUser(console, self.dashboard, images) if interactive else AutoPilot()
        self.backlot = Backlot(config.paths.backlot, self.embedder)
        self._http = make_http_client(config.reference.timeout_s)

    def preflight(self) -> PreflightReport:
        report = run_preflight(self.config, self.layout.logs_dir)
        meta = self.state.meta
        meta.set("style", self.config.pipeline.style)
        meta.set("versions", {**report.versions(), "kitbash": __version__})
        meta.set("blender_capabilities", report.blender.to_dict())
        meta.set("models", report.models)
        return report

    def scene_agent(self, report: PreflightReport) -> SceneAgent:
        config, layout, tracker, state = self.config, self.layout, self.tracker, self.state
        prompts = PromptLibrary()
        omp = OmpClient(
            config.tools.omp,
            timeout_s=config.omp.timeout_s,
            retries=config.omp.retries,
            retry_backoff_s=config.omp.retry_backoff_s,
            extra_args=config.omp.extra_args,
            workdir=layout.root,
            transcript_dir=layout.logs_dir / "llm",
        )
        llm = LLMService(TrackedLLMClient(omp, tracker), prompts, config.models)
        runner = BlenderRunner(config.tools.blender, timeout_s=config.blender.timeout_s, recorder=tracker)
        toolkit = BlenderToolkit(runner, config.blender, config.usd, config.naming, config.paths.downloads)
        fidelity = UsdFidelityChecker(toolkit)
        loop = ResumableLoop(
            CriticLoop(
                critics=(VisualCritic(llm, self.rubric), TechnicalCritic(llm, self.rubric)),
                patch_writer=PatchWriter(llm),
                rubric=self.rubric,
                config=config.critic,
                store=CycleStore(state.cycles, layout),
                tracker=tracker,
            ),
            LoopSessions(state.meta),
        )
        catalog = PolyHavenCatalog(
            self._http, config.paths.downloads, enabled=config.polyhaven.enabled, ttl_s=config.polyhaven.cache_ttl_s
        )
        reference_cache = config.paths.downloads / "references"
        finder = ReferenceFinder(
            self._providers(),
            ImageDownloader(
                self._http, max_bytes=int(config.reference.max_download_mb * 1024 * 1024),
                min_side=config.reference.min_side_px, cache_dir=reference_cache,
            ),
            llm,
            config.reference,
            tracker,
            cache_dir=reference_cache,
        )
        trellis = TrellisRunner(
            config.paths.trellis,
            python=report.trellis_python,
            steps=config.trellis.steps,
            pipeline_type=config.trellis.pipeline_type,
            no_texture=config.trellis.no_texture,
            timeout_s=config.trellis.timeout_s,
            retries=config.trellis.retries,
            max_concurrent=config.trellis_concurrency(),
            recorder=tracker,
        )
        breakdown = BreakdownAgent(llm, prompts, toolkit, config, layout, self.run_input)
        modelling = ModellingAgent(llm, toolkit, fidelity, finder, trellis, catalog, config, layout)
        layout_agent = LayoutAgent(llm, toolkit, catalog, config, layout, self.run_input)
        cast = SceneCast(state, self.backlot, config.naming)
        committer = BacklotCommitter(self.backlot, loop, modelling, config, layout, tracker)
        user, dashboard = self.user, self.dashboard
        phases = (
            BreakdownPhase(breakdown, loop, state, self.backlot, user, dashboard, tracker, config, layout),
            ModellingPhase(modelling, loop, committer, state, user, dashboard, tracker, config),
            LayoutPhase(layout_agent, loop, cast, state.meta, user, dashboard, tracker, layout),
            AssemblyPhase(layout_agent, loop, cast, toolkit, fidelity, self.rubric, state, user, dashboard, tracker, config, layout),
        )
        return SceneAgent(phases, state, tracker, AnalyticsReport(state, layout))

    def run(self, from_phase: PhaseName | None = None) -> dict:
        report = self.preflight()
        return self.scene_agent(report).run(from_phase)

    def analytics(self) -> dict:
        return AnalyticsReport(self.state, self.layout).write()

    def close(self) -> None:
        self._http.close()
        self.backlot.close()
        self.state.close()

    def _providers(self) -> list[CandidateProvider]:
        factories = {
            "input_crop": lambda: InputCropProvider(self.run_input.image, self.config.reference.min_crop_side_px),
            "wikimedia": lambda: WikimediaProvider(self._http),
            "openverse": lambda: OpenverseProvider(self._http),
        }
        unknown = [name for name in self.config.reference.providers if name not in factories]
        if unknown:
            raise ConfigError(f"Unknown reference providers: {', '.join(unknown)}", hint=f"Use {', '.join(factories)}.")
        return [factories[name]() for name in self.config.reference.providers]
