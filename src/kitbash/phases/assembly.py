"""Phase 4: assembly and export. Copy the backlot assets into the scene, rebuild the approved layout
against those copies, localize external files, render, export USD and validate the round trip."""

import json
import shutil
from collections.abc import Sequence
from pathlib import Path

from kitbash.agents.layout import LayoutAgent, PlacedAsset
from kitbash.analytics.tracker import SpanKind, Tracker
from kitbash.config import Config
from kitbash.critique.sessions import ResumableLoop
from kitbash.domain.phases import PhaseName
from kitbash.domain.rubric import Rubric
from kitbash.errors import StateError
from kitbash.interaction.protocols import UserChannel
from kitbash.paths import OutputLayout
from kitbash.phases.scene_assets import SceneCast
from kitbash.services.blender_toolkit import BlenderToolkit
from kitbash.services.usd_fidelity import UsdFidelityChecker
from kitbash.store.state import StateDB
from kitbash.ui.dashboard import Dashboard, PhaseProgress


class AssemblyPhase:
    name = PhaseName.ASSEMBLY

    def __init__(
        self,
        agent: LayoutAgent,
        loop: ResumableLoop,
        cast: SceneCast,
        toolkit: BlenderToolkit,
        fidelity: UsdFidelityChecker,
        rubric: Rubric,
        state: StateDB,
        user: UserChannel,
        dashboard: Dashboard,
        tracker: Tracker,
        config: Config,
        layout: OutputLayout,
    ) -> None:
        self._agent = agent
        self._loop = loop
        self._cast = cast
        self._toolkit = toolkit
        self._fidelity = fidelity
        self._rubric = rubric
        self._state = state
        self._user = user
        self._dashboard = dashboard
        self._tracker = tracker
        self._config = config
        self._layout = layout

    def run(self) -> None:
        inventory, placed, skipped = self._cast.load()
        best = self._loop.best(self._agent.subject(inventory, placed, skipped))
        if best is None:
            raise StateError("There is no layout script yet", hint="Run the layout phase first.")
        layout = self._layout
        logs = layout.logs_dir / "assembly"
        progress = PhaseProgress("Assembly")
        with self._dashboard.showing(progress.view):
            progress.status = "copying assets into the scene"
            blends = self._copy_assets(placed)
            progress.status = "rebuilding the layout against the copies"
            with self._tracker.span(SpanKind.STEP, "assembly.build"):
                subject = self._agent.subject(inventory, placed, skipped)
                args = subject.run_args(layout.scene_blend, blends, self._config.assembly.mode)
                self._toolkit.run_script(best.script_path, args, logs / "scene_build.log")
                localized = self._toolkit.localize(layout.scene_blend, logs)
            progress.status = "rendering the final frame"
            with self._tracker.span(SpanKind.STEP, "assembly.render"):
                final = self._toolkit.render_scene(
                    layout.scene_blend, layout.scene_renders_dir / "final.png", logs,
                    resolution=self._config.blender.final_resolution, samples=self._config.blender.final_samples,
                    engine=self._config.style.render_engine,
                )
            progress.status = "exporting USD and validating the round trip"
            with self._tracker.span(SpanKind.STEP, "assembly.usd"):
                usd = self._fidelity.check(
                    layout.scene_blend, layout.scene_usd, work_dir=layout.scene_dir / "_usd_work",
                    roundtrip_dir=layout.scene_renders_dir / "usd_roundtrip", prefix="scene", log_dir=logs,
                    scene=True, engine=self._config.style.render_engine,
                )
        facts = {
            **usd.facts(),
            "missing_textures": usd.facts()["usd_missing_textures"] + len(localized["missing"]),
            "localized_files": len(localized["copied"]),
        }
        card = self._rubric.score(
            self.name, [], facts, threshold=self._config.critic.pass_threshold, require_all_pass=self._config.critic.require_all_pass
        )
        report = {
            "scene_blend": str(layout.scene_blend),
            "scene_usd": str(layout.scene_usd),
            "final_render": str(final),
            "assets": {asset.key: str(blends[asset.key]) for asset in placed},
            "placeholders": [item.id for item in skipped],
            "localized": localized,
            "usd": usd.report(),
            "scorecard": card.to_dict(),
        }
        (layout.scene_dir / "assembly.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        self._state.meta.set("assembly", {k: report[k] for k in ("scene_blend", "scene_usd", "final_render", "scorecard")} | {
            "usd_material_mode": usd.mode, "usd_roundtrip_score": round(usd.score, 4),
        })
        self._user.notify(
            f"Scene assembled: {layout.relative(layout.scene_blend)}, {layout.relative(layout.scene_usd)} "
            f"(USD round trip {usd.score:.2f}, {usd.mode})"
        )
        self._user.show_images([final])

    def _copy_assets(self, placed: Sequence[PlacedAsset]) -> dict[str, Path]:
        """Copy each backlot asset folder (blend, textures, USD) so the scene export is self-contained."""
        blends = {}
        for asset in placed:
            source = asset.blend.parent
            target = self._layout.scene_assets_dir / asset.key
            shutil.copytree(source, target, dirs_exist_ok=True)
            blends[asset.key] = target / asset.blend.name
        return blends
