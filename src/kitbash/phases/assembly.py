"""Phase 4: assembly and export. Copy the backlot assets into the scene, rebuild the approved layout
against those copies, localize external files, render, export USD and validate the round trip."""

import json
import shutil
from collections.abc import Sequence
from pathlib import Path

from attrs import evolve

from kitbash.agents.layout import LayoutAgent, PlacedAsset
from kitbash.analytics.tracker import SpanKind, Tracker
from kitbash.config import Config
from kitbash.critique.sessions import ResumableLoop
from kitbash.domain.phases import PhaseName
from kitbash.domain.rubric import Rubric
from kitbash.errors import StateError
from kitbash.interaction.protocols import GateAction, PhaseSummary, UserChannel
from kitbash.paths import OutputLayout
from kitbash.phases.base import run_gate
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
        report = {
            "scene_blend": str(self._layout.scene_blend),
            "scene_usd": str(self._layout.scene_usd),
            "acceptance": {"status": "pending", "automatic_pass": False, "published": False, "issues": []},
        }
        self._save_report(report)
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
            progress.status = "inspecting the assembled scene"
            with self._tracker.span(SpanKind.STEP, "assembly.inspect"):
                inspection, blend_facts = self._toolkit.inspect_scene(
                    layout.scene_blend, args["assets"], [item.id for item in skipped], logs, "scene_inspect"
                )
            final = None
            if blend_facts["has_camera"]:
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
                    scene_expectations={
                        "assets": args["assets"],
                        "expected_assets": {key: spec["instances"] for key, spec in args["assets"].items()},
                        "expected_placeholders": [item.id for item in skipped],
                        "camera_name": inspection["camera"]["name"] if inspection["camera"] else None,
                    },
                )
        usd_facts = usd.facts()
        facts = {
            **blend_facts,
            **usd_facts,
            "blend_missing_textures": blend_facts["missing_textures"],
            "missing_textures": blend_facts["missing_textures"] + usd_facts["usd_missing_textures"] + len(localized["missing"]),
            "localized_files": len(localized["copied"]),
        }
        card = self._rubric.score(
            self.name, [], facts, threshold=self._config.critic.pass_threshold, require_all_pass=True
        )
        issues = []
        if card.failing():
            issues.append("Failed final criteria: " + ", ".join(entry.name for entry in card.failing()))
        elif not card.passed and not card.unassessed():
            issues.append("Final rubric score did not meet the pass threshold")
        for entry in card.unassessed():
            issues.append(f"Unassessed final criterion: {entry.name}")
        # These are publication invariants, not optional rubric weights. A custom or lenient rubric
        # cannot automatically publish missing instances, unsupported objects or broken textures.
        for prefix in ("", "usd_"):
            for key in ("missing_assets", "unexpected_assets", "missing_placeholders", "unexpected_placeholders", "floating_assets", "missing_textures"):
                if facts.get(prefix + key) != 0:
                    issues.append(f"{prefix}{key}: {facts.get(prefix + key, 'not measured')}")
            if facts.get(prefix + "has_camera") is not True:
                issues.append(f"{prefix}has_camera: missing or invalid active camera")
        outputs_exist = layout.scene_blend.is_file() and layout.scene_usd.is_file()
        if not outputs_exist:
            issues.append("Required scene.blend or scene.usd output is missing")
        card = evolve(card, passed=card.passed and not issues)
        acceptance = {"status": "passed" if card.passed else "failed", "automatic_pass": card.passed, "published": card.passed, "issues": issues}
        report.update({
            "final_render": str(final) if final else None,
            "assets": {asset.key: str(blends[asset.key]) for asset in placed},
            "placeholders": [item.id for item in skipped],
            "localized": localized,
            "blend_inspection": inspection,
            "usd": usd.report(),
            "scorecard": card.to_dict(),
            "acceptance": acceptance,
        })
        self._save_report(report)
        if not card.passed:
            if self._user.interactive and outputs_exist:
                decision = run_gate(self._user, self._tracker, PhaseSummary(
                    phase=self.name,
                    headline="Final validation failed. This scene is not an automatic pass.",
                    columns=("acceptance failure",), rows=tuple((issue,) for issue in issues),
                    images=(final,) if final else (), scorecard=card,
                    message="Explicitly publish a degraded result, or stop and fix the scene. An ordinary approval does not override validation.",
                ))
                if decision.action is GateAction.PUBLISH_DEGRADED:
                    acceptance.update(status="overridden", published=True, override="publish_degraded")
                    self._save_report(report)
            if not acceptance["published"]:
                raise StateError(
                    "Final assembly validation failed",
                    hint="; ".join(issues) + ". Outputs are retained for inspection, not accepted. Fix the layout and resume from layout, or explicitly publish degraded in interactive mode.",
                )
        label = "Scene passed final validation" if card.passed else "Degraded scene published by human override (validation failed)"
        self._user.notify(
            f"{label}: {layout.relative(layout.scene_blend)}, {layout.relative(layout.scene_usd)} "
            f"(USD round trip {usd.score:.2f}, {usd.mode})"
        )
        if final:
            self._user.show_images([final])

    def _save_report(self, report: dict) -> None:
        (self._layout.scene_dir / "assembly.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        usd = report.get("usd", {})
        self._state.meta.set("assembly", {
            **{key: report.get(key) for key in ("scene_blend", "scene_usd", "final_render", "scorecard", "acceptance")},
            "usd_material_mode": usd.get("usd_material_mode"), "usd_roundtrip_score": usd.get("roundtrip_score"),
        })

    def _copy_assets(self, placed: Sequence[PlacedAsset]) -> dict[str, Path]:
        """Copy each backlot asset folder (blend, textures, USD) so the scene export is self-contained."""
        blends = {}
        for asset in placed:
            source = asset.blend.parent
            target = self._layout.scene_assets_dir / asset.key
            shutil.copytree(source, target, dirs_exist_ok=True)
            blends[asset.key] = target / asset.blend.name
        return blends
