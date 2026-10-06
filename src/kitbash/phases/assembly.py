"""Phase 4: assembly and export. Copy the backlot assets into the scene, rebuild the approved layout
against those copies, localize external files, render, export USD and validate the round trip."""

import fcntl
import json
import shutil
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

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
        with self._staging() as scene:
            report = self._assemble(scene)
            previous = self._layout.root / ".scene-previous"
            if self._layout.scene_dir.exists():
                self._layout.scene_dir.rename(previous)
            scene.rename(self._layout.scene_dir)
            self._record(report)
            if previous.exists():
                shutil.rmtree(previous)
        acceptance = report.get("acceptance", {})
        if not acceptance.get("published"):
            issues = acceptance.get("issues", [])
            raise StateError(
                "Final assembly validation failed",
                hint="; ".join(issues) + ". Outputs are retained for inspection, not accepted. Fix the layout and resume from layout, or explicitly publish degraded in interactive mode.",
            )
        card_passed = report.get("scorecard", {}).get("passed")
        label = "Scene passed final validation" if card_passed else "Degraded scene published by human override (validation failed)"
        self._user.notify(
            f"{label}: {self._layout.relative(self._layout.scene_blend)}, {self._layout.relative(self._layout.scene_usd)} "
            f"(USD round trip {report['usd']['roundtrip_score']:.2f}, {report['usd']['usd_material_mode']})"
        )
        if report.get("final_render"):
            self._user.show_images([Path(report["final_render"])])

    def _assemble(self, scene: Path) -> dict[str, Any]:
        inventory, placed, skipped = self._cast.load()
        best = self._loop.best(self._agent.subject(inventory, placed, skipped))
        if best is None:
            raise StateError("There is no layout script yet", hint="Run the layout phase first.")
        layout = self._layout
        logs = layout.logs_dir / "assembly"
        progress = PhaseProgress("Assembly")
        with self._dashboard.showing(progress.view):
            progress.status = "copying assets into the scene"
            blends = self._copy_assets(placed, scene / "assets")
            progress.status = "rebuilding the layout against the copies"
            with self._tracker.span(SpanKind.STEP, "assembly.build"):
                subject = self._agent.subject(inventory, placed, skipped)
                args = subject.run_args(scene / "scene.blend", blends, self._config.assembly.mode)
                self._toolkit.run_script(best.script_path, args, logs / "scene_build.log")
                localized = self._toolkit.localize(scene / "scene.blend", logs)
            progress.status = "inspecting the assembled scene"
            with self._tracker.span(SpanKind.STEP, "assembly.inspect"):
                inspection, blend_facts = self._toolkit.inspect_scene(
                    scene / "scene.blend", args.get("assets", {}), [item.id for item in skipped], logs, "scene_inspect"
                )
            final = None
            if blend_facts["has_camera"]:
                progress.status = "rendering the final frame"
                with self._tracker.span(SpanKind.STEP, "assembly.render"):
                    final = self._toolkit.render_scene(
                        scene / "scene.blend", scene / "renders" / "final.png", logs,
                        resolution=self._config.blender.final_resolution, samples=self._config.blender.final_samples,
                        engine=self._config.style.render_engine,
                    )
            progress.status = "exporting USD and validating the round trip"
            with self._tracker.span(SpanKind.STEP, "assembly.usd"):
                usd = self._fidelity.check(
                    scene / "scene.blend", scene / "scene.usd", work_dir=scene / "_usd_work",
                    roundtrip_dir=scene / "renders" / "usd_roundtrip", prefix="scene", log_dir=logs,
                    scene=True, engine=self._config.style.render_engine,
                    scene_expectations={
                        "assets": args.get("assets", {}),
                        "expected_assets": {key: spec["instances"] for key, spec in args.get("assets", {}).items()},
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
        outputs_exist = (scene / "scene.blend").is_file() and (scene / "scene.usd").is_file()
        if not outputs_exist:
            issues.append("Required scene.blend or scene.usd output is missing")
        card = evolve(card, passed=card.passed and not issues)
        acceptance = {"status": "passed" if card.passed else "failed", "automatic_pass": card.passed, "published": card.passed, "issues": issues}
        report = {
            "scene_blend": str(scene / "scene.blend"),
            "scene_usd": str(scene / "scene.usd"),
            "final_render": str(final) if final else None,
            "assets": {asset.key: str(blends[asset.key]) for asset in placed},
            "placeholders": [item.id for item in skipped],
            "localized": localized,
            "blend_inspection": inspection,
            "usd": usd.report(),
            "scorecard": card.to_dict(),
            "acceptance": acceptance,
        }
        if not card.passed and getattr(self._user, "interactive", False) and outputs_exist:
            decision = run_gate(self._user, self._tracker, PhaseSummary(
                phase=self.name,
                headline="Final validation failed. This scene is not an automatic pass.",
                columns=("acceptance failure",), rows=tuple((issue,) for issue in issues),
                images=(final,) if final else (), scorecard=card,
                message="Explicitly publish a degraded result, or stop and fix the scene. An ordinary approval does not override validation.",
            ))
            if decision.action is GateAction.PUBLISH_DEGRADED:
                acceptance.update(status="overridden", published=True, override="publish_degraded")
                report["acceptance"] = acceptance
        report = self._published_paths(report, scene)
        # Reports written by Blender also contain absolute output paths. Keep them valid after rename.
        for path in scene.rglob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            path.write_text(json.dumps(self._published_paths(data, scene), indent=2), encoding="utf-8")
        (scene / "assembly.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    @contextmanager
    def _staging(self) -> Iterator[Path]:
        """Serialize publishers and recover either side of an interrupted directory replacement."""
        layout = self._layout
        scene = layout.root / ".scene-staging"
        previous = layout.root / ".scene-previous"
        with (layout.root / ".scene.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise StateError("Another process is publishing this scene.") from exc
            self._recover_previous(previous)
            if scene.exists():
                shutil.rmtree(scene)
            self._state.meta.set("assembly", {
                "scene_blend": str(layout.scene_blend),
                "scene_usd": str(layout.scene_usd),
                "acceptance": {"status": "pending", "automatic_pass": False, "published": False, "issues": []},
            })
            scene.mkdir()
            try:
                yield scene
            finally:
                self._recover_previous(previous)
                if scene.exists():
                    shutil.rmtree(scene)

    def _recover_previous(self, previous: Path) -> None:
        if previous.exists():
            if self._layout.scene_dir.exists():
                shutil.rmtree(previous)
            else:
                previous.rename(self._layout.scene_dir)
            published = self._layout.scene_dir / "assembly.json"
            if published.is_file():
                self._record(json.loads(published.read_text(encoding="utf-8")))

    def _record(self, report: dict[str, Any]) -> None:
        usd = report.get("usd", {})
        self._state.meta.set("assembly", {
            **{key: report.get(key) for key in ("scene_blend", "scene_usd", "final_render", "scorecard", "acceptance")},
            "usd_material_mode": usd.get("usd_material_mode"),
            "usd_roundtrip_score": usd.get("roundtrip_score"),
        })

    def _published_paths(self, value: Any, scene: Path) -> Any:
        if isinstance(value, dict):
            return {key: self._published_paths(item, scene) for key, item in value.items()}
        if isinstance(value, list):
            return [self._published_paths(item, scene) for item in value]
        if isinstance(value, str) and (value == str(scene) or value.startswith(f"{scene}/")):
            return str(self._layout.scene_dir / Path(value).relative_to(scene))
        return value

    def _copy_assets(self, placed: Sequence[PlacedAsset], destination: Path) -> dict[str, Path]:
        """Copy each backlot asset folder (blend, textures, USD) so the scene export is self-contained."""
        blends = {}
        for asset in placed:
            source = asset.blend.parent
            target = destination / asset.key
            shutil.copytree(source, target)
            blends[asset.key] = target / asset.blend.name
        return blends
