"""The Kitbash flow: 1 reference image -> Trellis -> retopology -> USD/Blender model -> backlot.

A minimal pipeline that validates image-to-model end to end. It starts from exactly one reference image, always
reconstructs it with Trellis, always retopologizes the mesh, builds the .blend with a fixed script, exports and
validates the USD, and adds the asset to the backlot. There are no critics, no evaluation loops, no LLM calls and no
agents, and no procedural geometry: a failed step fails the flow instead of producing something else."""

import json
import time
from pathlib import Path

from attrs import frozen

from kitbash.analytics.tracker import EventKind, SpanKind, Tracker
from kitbash.backlot.library import AssetBundle, Backlot, BacklotDraft, BacklotEntry
from kitbash.config import Config
from kitbash.errors import KitbashError
from kitbash.infra.imaging import dominant_color
from kitbash.infra.trellis import TrellisResult, TrellisRunner
from kitbash.naming import slugify
from kitbash.paths import OutputLayout
from kitbash.retopology.base import Retopologizer, RetopologyResult
from kitbash.retopology.step import retopologize_mesh
from kitbash.services.artifacts import file_hash, snapshot
from kitbash.services.blender_toolkit import BlenderToolkit
from kitbash.services.reference_quality import is_isolated
from kitbash.services.usd_fidelity import UsdFidelityChecker

BLEND_NAME = "asset.blend"
USD_RELATIVE = Path("usd") / "asset.usd"
MODEL_REPORT = "model.json"


@frozen
class ModelRequest:
    image: Path  # exactly one reference image
    name: str
    category: str = "object"
    description: str = ""
    height_m: float = 1.0  # real-world height; the mesh is scaled to it
    seed: int = 42

    @property
    def slug(self) -> str:
        return slugify(self.name)


@frozen
class ModelResult:
    entry: BacklotEntry
    blend: Path
    usd: Path
    preview: Path
    reference: Path
    mesh: Path
    report: Path
    timings_s: dict[str, float]


class ImageToModelFlow:
    def __init__(
        self,
        trellis: TrellisRunner,
        retopologizer: Retopologizer,
        toolkit: BlenderToolkit,
        fidelity: UsdFidelityChecker,
        backlot: Backlot,
        config: Config,
        layout: OutputLayout,
        tracker: Tracker,
    ) -> None:
        self._trellis = trellis
        self._retopologizer = retopologizer
        self._toolkit = toolkit
        self._fidelity = fidelity
        self._backlot = backlot
        self._config = config
        self._layout = layout
        self._tracker = tracker

    def run(self, request: ModelRequest) -> ModelResult:
        slug = request.slug
        layout = self._layout
        timings: dict[str, float] = {}

        def timed(step: str, action):
            started = time.monotonic()
            with self._tracker.span(SpanKind.STEP, step):
                result = action()
            timings[step] = round(time.monotonic() - started, 2)
            return result

        reference = self._reference(request)
        trellis: TrellisResult = timed("trellis", lambda: self._trellis.generate(
            reference, layout.asset_trellis_dir(slug) / "attempt_01", slug, seed=request.seed
        ))
        retopology: RetopologyResult = timed("retopology", lambda: retopologize_mesh(
            self._retopologizer, layout, self._tracker, slug, 1, trellis.mesh_path
        ))
        build = layout.asset_dir(slug) / "build"
        blend = build / BLEND_NAME
        built = timed("build", lambda: self._toolkit.build_trellis_model(
            {
                "mesh_path": str(retopology.mesh_path),
                "mesh_up_axis": self._config.trellis.mesh_up_axis,
                "retopology_method": retopology.method.value,
                "fit_mode": "height",
                "base_color": list(dominant_color(reference)),
                "output_blend": str(blend),
                "textures_dir": str(build / "textures"),
                "slug": slug,
                "name": request.name,
                "dimensions_m": [request.height_m] * 3,
            },
            layout.asset_dir(slug),
        ))
        previews = timed("preview", lambda: self._toolkit.render_views(
            blend, layout.asset_previews_dir(slug), "model", layout.asset_dir(slug)
        ))
        usd = timed("usd", lambda: self._fidelity.check(
            blend, build / USD_RELATIVE, work_dir=layout.asset_dir(slug) / "usd_work",
            roundtrip_dir=layout.asset_roundtrip_dir(slug), prefix="model", log_dir=layout.asset_dir(slug),
        ))
        dimensions = tuple(built["asset"]["dimensions_m"])
        preview = previews[0]
        metadata = {
            "slug": slug,
            "flow": "image_to_model",
            "modelling_method": "trellis",
            "scene_output": str(layout.root),
            "reference": {"path": str(reference), "sha256": file_hash(reference)},
            "trellis": {"duration_s": round(trellis.duration_s, 2), "retries": trellis.retries, "runs": 1, "seed": request.seed},
            "retopology": retopology.to_extra(),
            "usd": usd.report(),
        }
        entry = timed("backlot", lambda: self._backlot.add(
            BacklotDraft(
                name=request.name,
                description=request.description or request.name,
                category=request.category,
                dimensions=dimensions,
                style=self._config.pipeline.style,
                usd_material_mode=usd.mode,
                usd_roundtrip_score=round(usd.score, 4),
                source_reference=json.dumps(metadata["reference"]),
                tags=(request.category,),
                metadata=metadata,
            ),
            AssetBundle(
                root=build, blend=blend, usd=usd.usd_path, preview=preview,
                hashes=snapshot(build, (build,)), preview_hash=file_hash(preview),
            ),
        ))
        report = layout.root / MODEL_REPORT
        report.write_text(json.dumps({**metadata, "backlot_id": entry.id, "timings_s": timings}, indent=2), encoding="utf-8")
        return ModelResult(entry, blend, usd.usd_path, preview, reference, retopology.mesh_path, report, timings)

    def _reference(self, request: ModelRequest) -> Path:
        """The one reference image, copied into the run. Exactly what the user gave: never searched, replaced or generated."""
        source = request.image.expanduser()
        if not source.is_file():
            raise KitbashError(f"Reference image not found: {source}")
        try:
            isolated = is_isolated(source)
        except OSError as exc:
            raise KitbashError(f"Not a readable image: {source} ({exc})") from exc
        target = self._layout.input_dir / f"reference{source.suffix.lower()}"
        if source.resolve() != target.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        if not isolated:
            self._tracker.event(
                EventKind.WARNING, "reference_not_isolated", image=str(source),
                note="The object is not on a transparent or flat neutral background; Trellis may reconstruct it poorly.",
            )
        return target

