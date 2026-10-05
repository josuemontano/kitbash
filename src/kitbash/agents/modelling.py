"""Modelling agent: one asset at a time (reference image, Trellis mesh, Blender build script, checks)."""

import json
import logging
import shutil
import time
from pathlib import Path

from kitbash.analytics import context
from kitbash.analytics.tracker import EventKind, SpanKind, Tracker
from kitbash.config import Config
from kitbash.critique.subject import CriticBrief, Evaluation
from kitbash.domain.assets import AssetRecord
from kitbash.domain.inventory import InventoryItem
from kitbash.domain.phases import PhaseName
from kitbash.domain.roles import Role
from kitbash.errors import KitbashError, RetopologyError
from kitbash.infra.polyhaven import PolyHavenCatalog
from kitbash.infra.trellis import TrellisResult, TrellisRunner
from kitbash.llm.service import LLMService
from kitbash.paths import OutputLayout
from kitbash.retopology.base import Retopologizer, RetopologyMethod, RetopologyResult
from kitbash.services.api_reference import blender_api_reference
from kitbash.services.blender_toolkit import BlenderToolkit
from kitbash.services.references import ReferenceChoice, ReferenceFinder
from kitbash.services.usd_fidelity import UsdFidelityChecker

BUILD_DIR = "build"
BLEND_NAME = "asset.blend"
USD_RELATIVE = Path("usd") / "asset.usd"

log = logging.getLogger("kitbash")


class ModellingAgent:
    def __init__(
        self,
        llm: LLMService,
        toolkit: BlenderToolkit,
        fidelity: UsdFidelityChecker,
        finder: ReferenceFinder,
        trellis: TrellisRunner,
        retopologizer: Retopologizer,
        catalog: PolyHavenCatalog,
        config: Config,
        layout: OutputLayout,
        tracker: Tracker,
    ) -> None:
        self._llm = llm
        self._toolkit = toolkit
        self._fidelity = fidelity
        self._finder = finder
        self._trellis = trellis
        self._retopologizer = retopologizer
        self._catalog = catalog
        self._config = config
        self._layout = layout
        self._tracker = tracker

    def find_reference(self, item: InventoryItem) -> ReferenceChoice | None:
        with context.bind(agent="modelling_agent"):
            return self._finder.find(item, self._layout.asset_reference_dir(item.id))

    def generate_mesh(self, asset: AssetRecord, reference: Path) -> TrellisResult:
        directory = self._layout.asset_trellis_dir(asset.id) / f"attempt_{asset.attempt:02d}"
        return self._trellis.generate(reference, directory, asset.id, seed=asset.seed)

    def retopologize(self, asset: AssetRecord, mesh_path: Path) -> RetopologyResult:
        """Retopologize the Trellis mesh. With ``fallback_on_error`` a failure keeps the Trellis mesh (the decimate path)."""
        retopologizer = self._retopologizer
        if retopologizer.method is RetopologyMethod.DECIMATE:
            return retopologizer.retopologize(mesh_path, mesh_path.parent, asset.id)
        directory = self._layout.asset_retopo_dir(asset.id) / f"attempt_{asset.attempt:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        try:
            with self._tracker.span(SpanKind.SUBPROCESS, "retopology", attempt=asset.attempt, method=retopologizer.method.value) as span:
                result = retopologizer.retopologize(mesh_path, directory, asset.id)
                span.meta.update(faces_in=result.faces_in, faces_out=result.faces_out, device=result.device)
        except RetopologyError as exc:
            if not self._config.retopology.fallback_on_error:
                raise
            reason = f"{retopologizer.method.value}: {exc.message}"
            log.warning("Retopology failed for %s, keeping the Trellis mesh: %s", asset.id, reason)
            self._tracker.event(EventKind.WARNING, "retopology_fallback", asset=asset.id, reason=reason)
            return RetopologyResult(
                method=RetopologyMethod.DECIMATE, mesh_path=mesh_path, faces_in=None, faces_out=None,
                duration_s=time.monotonic() - started, fallback_reason=reason,
            )
        return result

    def write_script(self, item: InventoryItem, asset: AssetRecord) -> str:
        naming = self._config.naming
        textures = self._catalog.textures(item.materials_hint)
        with context.bind(agent="modelling_agent"):
            return self._llm.ask_python(
                task="modelling.script",
                role=Role.CODE,
                phase=PhaseName.MODELLING,
                template="asset_script",
                variables={
                    "slug": item.id,
                    "name": item.name,
                    "description": item.description,
                    "category": item.category,
                    "dimensions": _dimensions(item),
                    "materials": ", ".join(item.materials_hint) or "not specified",
                    "style": self._config.pipeline.style,
                    "retopology": _retopology_note(retopology_method(asset)),
                    "feedback": "\n".join(f"- {f}" for f in asset.feedback) or "(none)",
                    "naming": naming.describe(item.id),
                    "textures": json.dumps(textures) if textures else "(none)",
                    "api": blender_api_reference(),
                },
                attachments=tuple(p for p in (Path(asset.reference_path),) if asset.reference_path and p.is_file()),
            )

    def subject(self, item: InventoryItem, asset: AssetRecord) -> AssetSubject:
        return AssetSubject(item, asset, self._toolkit, self._fidelity, self._config, self._layout)

    def keep_script(self, asset_id: str, script: Path) -> None:
        """The asset's current best build script, at ``02_modelling/<asset_id>/script.py``."""
        shutil.copy2(script, self._layout.asset_script(asset_id))


class AssetSubject:
    phase = PhaseName.MODELLING

    def __init__(
        self,
        item: InventoryItem,
        asset: AssetRecord,
        toolkit: BlenderToolkit,
        fidelity: UsdFidelityChecker,
        config: Config,
        layout: OutputLayout,
    ) -> None:
        self._item = item
        self._asset = asset
        self._toolkit = toolkit
        self._fidelity = fidelity
        self._config = config
        self._layout = layout

    @property
    def subject_id(self) -> str:
        return self._asset.id

    def brief(self) -> CriticBrief:
        item = self._item
        return CriticBrief(
            phase=self.phase,
            subject=f"asset '{item.id}' ({item.name})",
            description=(
                f"{item.description}\nCategory: {item.category}. Real-world size (width, depth, height): "
                f"{_dimensions(item)}. Materials: {', '.join(item.materials_hint) or 'unspecified'}.\n"
                f"Naming convention:\n{self._config.naming.describe(item.id)}"
            ),
            references=tuple(Path(p) for p in (self._asset.reference_path,) if p),
            style=self._config.pipeline.style,
        )

    def api_reference(self) -> str:
        return blender_api_reference()

    def evaluate(self, script: Path, cycle_dir: Path, cycle: int) -> Evaluation:
        """Build the asset, render previews, inspect it and validate the USD round trip."""
        if not self._asset.mesh_path:
            raise KitbashError(f"Asset {self._asset.id} has no Trellis mesh")
        build = cycle_dir / BUILD_DIR
        blend = build / BLEND_NAME
        prefix = f"cycle_{cycle:02d}"
        self._toolkit.run_script(
            script,
            {
                "mesh_path": self._asset.mesh_path,
                "mesh_up_axis": self._config.trellis.mesh_up_axis,
                "retopology_method": retopology_method(self._asset),
                "output_blend": str(blend),
                "textures_dir": str(build / "textures"),
                "slug": self._item.id,
                "name": self._item.name,
                "dimensions_m": list(self._item.dimensions.as_tuple()),
            },
            cycle_dir / "build.log",
        )
        previews = self._toolkit.render_views(blend, self._layout.asset_previews_dir(self._asset.id), prefix, cycle_dir)
        report, facts = self._toolkit.inspect_asset(blend, self._item.id, self._item.dimensions.as_tuple(), cycle_dir, "inspect")
        artifacts = {"blend": str(blend), "build_dir": str(build), "preview": str(previews[0])}
        error = None
        try:
            usd = self._fidelity.check(
                blend, build / USD_RELATIVE, work_dir=cycle_dir / "usd_work",
                roundtrip_dir=self._layout.asset_roundtrip_dir(self._asset.id), prefix=prefix, log_dir=cycle_dir,
            )
        except KitbashError as exc:
            error = f"USD export or round trip failed: {exc}"
            facts = {**facts, "usd_roundtrip_score": 0.0, "usd_broken_materials": facts.get("materials", 1)}
            report = {"asset": report, "usd": {"error": str(exc)}}
        else:
            usd_facts = usd.facts()
            facts = {**facts, **usd_facts, "missing_textures": facts["missing_textures"] + usd_facts["usd_missing_textures"]}
            report = {"asset": report, "usd": usd.report()}
            artifacts.update(usd=str(usd.usd_path), usd_compare=str(usd.compare_image or ""))
            (self._layout.asset_roundtrip_dir(self._asset.id) / f"{prefix}_report.json").write_text(
                json.dumps(usd.report(), indent=2), encoding="utf-8"
            )
        return Evaluation(ok=error is None, images=tuple(previews), facts=facts, report=report, error=error, artifacts=artifacts)


def retopology_method(asset: AssetRecord) -> str:
    """The method that produced the asset's mesh: ``decimate`` for a raw Trellis mesh (including after a fallback)."""
    return asset.extra.get("retopology", {}).get("method", RetopologyMethod.DECIMATE.value)


def _retopology_note(method: str) -> str:
    if method == RetopologyMethod.TRIFLOW.value:
        return (
            "The mesh was already retopologized (low-poly triangles), so `kb.decimate(obj)` preserves it and "
            "`kb.clean_mesh(obj)` only updates normals and shading; still call them, and do not reduce or remesh it yourself."
        )
    return "The mesh is the raw Trellis output (dense triangles), so `kb.decimate(obj)` does the real reduction."


def _dimensions(item: InventoryItem) -> str:
    return "({:.3f}, {:.3f}, {:.3f}) m".format(*item.dimensions.as_tuple())
