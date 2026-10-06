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
from kitbash.services.artifacts import file_hash, snapshot
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
            if not report["acceptance"]["published"]:
                rejected = self._retain_rejected(scene, report)
                raise StateError(
                    "Final assembly validation failed",
                    hint="; ".join(report["acceptance"]["issues"])
                    + f". Rejected outputs are in {rejected}; the published scene is unchanged. "
                    "Fix the layout and resume from layout, or explicitly publish degraded in interactive mode.",
                )
            self._verify_content(scene, report)
            report = self._relocate_reports(scene, self._layout.scene_dir, report)
            if self._published_report(scene) != report:
                raise StateError("Final assembly publication evidence is invalid or changed.")
            previous = self._layout.root / ".scene-previous"
            if self._layout.scene_dir.exists():
                self._layout.scene_dir.rename(previous)
            scene.rename(self._layout.scene_dir)
            self._record(report)
        card_passed = report["scorecard"]["passed"]
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
                    scene / "scene.blend", args["assets"], [item.id for item in skipped], logs, "scene_inspect"
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
        required = [scene / "scene.blend", scene / "scene.usd"]
        if final is not None:
            required.append(final)
        missing = [path for path in required if not path.is_file() or path.stat().st_size == 0]
        outputs_exist = not missing and (final is not None or blend_facts["has_camera"] is False)
        if missing:
            issues.append("Required outputs are missing or empty: " + ", ".join(path.name for path in missing))
        if final is None:
            issues.append("Final render is unavailable")
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
        report["integrity"] = {
            "owner": str(self._layout.root.resolve()),
            "manifest": self._content_manifest(scene),
        }
        if not card.passed and self._user.interactive and outputs_exist:
            try:
                decision = run_gate(self._user, self._tracker, PhaseSummary(
                    phase=self.name,
                    headline="Final validation failed. This scene is not an automatic pass.",
                    columns=("acceptance failure",), rows=tuple((issue,) for issue in issues),
                    images=(final,) if final else (), scorecard=card,
                    message="Explicitly publish a degraded result, or stop and fix the scene. An ordinary approval does not override validation.",
                ))
            except BaseException:
                self._retain_rejected(scene, report)
                raise
            if decision.action is GateAction.PUBLISH_DEGRADED:
                acceptance.update(status="overridden", published=True, override="publish_degraded")
        return report

    def _retain_rejected(self, scene: Path, report: dict[str, Any]) -> Path:
        destination = self._layout.phase_dir(self.name) / "rejected"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        report = self._relocate_reports(scene, destination, report)
        scene.rename(destination)
        self._record(report)
        return destination

    def _relocate_reports(self, scene: Path, destination: Path, report: dict[str, Any]) -> dict[str, Any]:
        report = self._relocated_paths(report, scene, destination)
        # Only generated reports belong to assembly; copied asset bundles may contain arbitrary JSON.
        for path in (scene / "renders").rglob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            path.write_text(json.dumps(self._relocated_paths(data, scene, destination), indent=2), encoding="utf-8")
            report["integrity"]["manifest"][path.relative_to(scene).as_posix()] = file_hash(path)
        # Assembly owns this journal, never generated scripts or copied asset bundles.
        with (scene / "assembly.json").open("x", encoding="utf-8") as journal:
            journal.write(json.dumps(report, indent=2))
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
            report = self._recover_previous(previous)
            if report is not None:
                self._record(report)
            else:
                self._state.meta.set("published_assembly", None)
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
                if previous.exists():
                    self._recover_previous(previous)
                if scene.exists():
                    shutil.rmtree(scene)

    def _recover_previous(self, previous: Path) -> dict[str, Any] | None:
        report = self._published_report(self._layout.scene_dir)
        if previous.exists():
            if report is not None:
                shutil.rmtree(previous)
            else:
                scene = self._layout.scene_dir
                if scene.is_symlink() or (scene.exists() and not scene.is_dir()):
                    scene.unlink()
                elif scene.exists():
                    shutil.rmtree(scene)
                previous.rename(scene)
                report = self._published_report(scene)
        return report

    def _content_manifest(self, scene: Path) -> dict[str, str]:
        return snapshot(scene, (path for path in scene.iterdir() if path.name != "assembly.json"))

    def _verify_content(self, scene: Path, report: dict[str, Any]) -> None:
        integrity = report.get("integrity")
        if not isinstance(integrity, dict) or integrity.get("owner") != str(self._layout.root.resolve()):
            raise StateError("Final assembly has no matching workspace owner.")
        try:
            manifest = self._content_manifest(scene)
        except (OSError, ValueError) as exc:
            raise StateError("Final assembly content cannot be verified.") from exc
        if integrity.get("manifest") != manifest:
            raise StateError("Final assembly content changed after inspection.")

    def _published_report(self, scene: Path) -> dict[str, Any] | None:
        """Recover only an accepted journal belonging to this workspace and these exact bytes."""
        journal = scene / "assembly.json"
        try:
            if journal.is_symlink() or not journal.is_file():
                return None
            report = json.loads(journal.read_text(encoding="utf-8"))
            if not isinstance(report, dict):
                return None
            self._verify_content(scene, report)
            acceptance = report.get("acceptance")
            card = report.get("scorecard")
            if not isinstance(acceptance, dict) or not isinstance(card, dict) or acceptance.get("published") is not True:
                return None
            passed = (
                acceptance.get("status") == "passed" and acceptance.get("automatic_pass") is True
                and card.get("passed") is True and acceptance.get("issues") == []
            )
            overridden = (
                acceptance.get("status") == "overridden" and acceptance.get("automatic_pass") is False
                and card.get("passed") is False and acceptance.get("override") == "publish_degraded"
            )
            if not (passed or overridden):
                return None
            required = ["scene.blend", "scene.usd"]
            for key, name in (("scene_blend", "scene.blend"), ("scene_usd", "scene.usd")):
                if report.get(key) != str(self._layout.scene_dir / name):
                    return None
            final = report.get("final_render")
            if final is not None:
                required.append(Path(final).relative_to(self._layout.scene_dir).as_posix())
            elif passed:
                return None
            manifest = report["integrity"]["manifest"]
            if any(name not in manifest or (scene / name).stat().st_size == 0 for name in required):
                return None
            return report
        except (OSError, ValueError, TypeError, StateError):
            return None

    def _record(self, report: dict[str, Any]) -> None:
        usd = report.get("usd", {})
        metadata = {
            **{key: report.get(key) for key in ("scene_blend", "scene_usd", "final_render", "scorecard", "acceptance")},
            "usd_material_mode": usd.get("usd_material_mode"),
            "usd_roundtrip_score": usd.get("roundtrip_score"),
        }
        with self._state.db.transaction():
            self._state.meta.set("assembly", metadata)
            if (report.get("acceptance") or {}).get("published"):
                self._state.meta.set("published_assembly", metadata)

    def _relocated_paths(self, value: Any, scene: Path, destination: Path) -> Any:
        if isinstance(value, dict):
            return {key: self._relocated_paths(item, scene, destination) for key, item in value.items()}
        if isinstance(value, list):
            return [self._relocated_paths(item, scene, destination) for item in value]
        if isinstance(value, str) and (value == str(scene) or value.startswith(f"{scene}/")):
            return str(destination / Path(value).relative_to(scene))
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
