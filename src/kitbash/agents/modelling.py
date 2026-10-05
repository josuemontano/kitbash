"""Modelling agent: one asset at a time (reference image, Trellis mesh, Blender build script, checks)."""

import json
import shutil
from pathlib import Path

from kitbash.analytics import context
from kitbash.config import Config
from kitbash.critique.subject import CriticBrief, Evaluation
from kitbash.domain.assets import AssetRecord
from kitbash.domain.inventory import InventoryItem
from kitbash.domain.phases import PhaseName
from kitbash.domain.roles import Role
from kitbash.errors import KitbashError
from kitbash.infra.polyhaven import PolyHavenCatalog
from kitbash.infra.trellis import TrellisResult, TrellisRunner
from kitbash.llm.service import LLMService
from kitbash.paths import OutputLayout
from kitbash.services.api_reference import blender_api_reference
from kitbash.services.blender_toolkit import BlenderToolkit
from kitbash.services.references import ReferenceChoice, ReferenceFinder
from kitbash.services.usd_fidelity import UsdFidelityChecker

BUILD_DIR = "build"
BLEND_NAME = "asset.blend"
USD_RELATIVE = Path("usd") / "asset.usd"


class ModellingAgent:
    def __init__(
        self,
        llm: LLMService,
        toolkit: BlenderToolkit,
        fidelity: UsdFidelityChecker,
        finder: ReferenceFinder,
        trellis: TrellisRunner,
        catalog: PolyHavenCatalog,
        config: Config,
        layout: OutputLayout,
    ) -> None:
        self._llm = llm
        self._toolkit = toolkit
        self._fidelity = fidelity
        self._finder = finder
        self._trellis = trellis
        self._catalog = catalog
        self._config = config
        self._layout = layout

    def find_reference(self, item: InventoryItem) -> ReferenceChoice | None:
        with context.bind(agent="modelling_agent"):
            return self._finder.find(item, self._layout.asset_reference_dir(item.id))

    def reference_review(self, item: InventoryItem) -> dict | None:
        manifest = self._layout.asset_reference_dir(item.id) / "review.json"
        return json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else None

    def select_reference(self, item: InventoryItem, index: int) -> ReferenceChoice | None:
        with context.bind(agent="modelling_agent"):
            return self._finder.select_reviewed(item, self._layout.asset_reference_dir(item.id), index)

    def generate_mesh(self, asset: AssetRecord, reference: Path) -> TrellisResult:
        directory = self._layout.asset_trellis_dir(asset.id) / f"attempt_{asset.attempt:02d}"
        return self._trellis.generate(reference, directory, asset.id, seed=asset.seed)

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


def _dimensions(item: InventoryItem) -> str:
    return "({:.3f}, {:.3f}, {:.3f}) m".format(*item.dimensions.as_tuple())
