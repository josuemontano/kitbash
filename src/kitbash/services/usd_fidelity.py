"""USD export with the fidelity ladder plus the required round-trip validation."""

import shutil
from collections.abc import Mapping
from pathlib import Path
from statistics import fmean
from typing import Any

from attrs import frozen

from kitbash.infra.imaging import ImageComparison, compare_images, side_by_side
from kitbash.services.blender_toolkit import BlenderToolkit


@frozen
class UsdCheck:
    usd_path: Path
    export: Mapping[str, Any]
    roundtrip: Mapping[str, Any]
    comparisons: tuple[ImageComparison, ...]
    compare_image: Path | None

    @property
    def score(self) -> float:
        return fmean(c.score for c in self.comparisons) if self.comparisons else 0.0

    @property
    def mode(self) -> str:
        return str(self.export.get("usd_material_mode", "preview_surface_baked"))

    def facts(self) -> dict[str, Any]:
        broken = int(self.roundtrip.get("materials_total", 0)) - int(self.roundtrip.get("materials_ok", 0))
        scene_facts = {f"usd_{key}": value for key, value in self.roundtrip.get("scene_facts", {}).items()}
        scene_missing = self.roundtrip.get("scene_report", {}).get("missing_textures", [])
        imported_missing = set(self.roundtrip.get("missing_textures", [])) | set(scene_missing)
        return {
            **scene_facts,
            "usd_roundtrip_score": round(self.score, 4),
            "usd_missing_textures": len(self.export.get("missing_textures", [])) + len(imported_missing),
            "usd_broken_materials": broken,
            "usd_absolute_texture_paths": len(self.export.get("absolute_texture_paths", [])),
            "usd_material_mode": self.mode,
        }

    def report(self) -> dict[str, Any]:
        return {
            "usd_path": str(self.usd_path),
            "materialx_supported": self.export.get("materialx_supported"),
            "usd_material_mode": self.mode,
            "materials": {
                name: {
                    "rung": info.get("rung"),
                    "materialx_lossless": info.get("materialx_lossless"),
                    "baked": info.get("baked"),
                    "lost_in_preview": info.get("lost_in_preview"),
                    "roundtrip": self.roundtrip.get("materials", {}).get(info.get("usd_prim") or name),
                }
                for name, info in self.export.get("materials", {}).items()
            },
            "missing_textures": list(self.export.get("missing_textures", [])) + list(self.roundtrip.get("missing_textures", [])),
            "comparisons": [c.to_dict() for c in self.comparisons],
            "roundtrip_score": round(self.score, 4),
            **({"scene_report": self.roundtrip["scene_report"]} if "scene_report" in self.roundtrip else {}),
        }


class UsdFidelityChecker:
    def __init__(self, toolkit: BlenderToolkit) -> None:
        self._toolkit = toolkit

    def check(
        self,
        blend: Path,
        usd_path: Path,
        *,
        work_dir: Path,
        roundtrip_dir: Path,
        prefix: str,
        log_dir: Path,
        scene: bool = False,
        engine: str | None = None,
        scene_expectations: Mapping[str, Any] | None = None,
    ) -> UsdCheck:
        """Export ``blend`` to ``usd_path``, re-import it in a clean session, render both and compare."""
        toolkit = self._toolkit
        export = toolkit.export_usd(blend, usd_path, work_dir, log_dir, f"{prefix}_usd_export", scene=scene)
        shutil.rmtree(work_dir, ignore_errors=True)
        expected = {info.get("usd_prim") or name: info.get("expected_preview_channels", []) for name, info in export["materials"].items()}
        roundtrip = toolkit.usd_roundtrip(
            usd_path, expected, mode="scene" if scene else "asset", output_dir=roundtrip_dir,
            prefix=f"{prefix}_usd", log_dir=log_dir, engine=engine, scene_expectations=scene_expectations,
        )
        settings = {"resolution": toolkit.roundtrip_resolution, "samples": toolkit.roundtrip_samples}
        if scene:
            originals = (
                [toolkit.render_scene(blend, roundtrip_dir / f"{prefix}_blend_camera.png", log_dir, engine=engine, **settings)]
                if roundtrip["images"] else []
            )
        else:
            originals = toolkit.render_views(blend, roundtrip_dir, f"{prefix}_blend", log_dir, **settings)
        renders = [Path(p) for p in roundtrip["images"]]
        comparisons = tuple(compare_images(a, b) for a, b in zip(originals, renders, strict=False))
        compare_image = None
        if originals and renders:
            compare_image = side_by_side(
                [originals[0], renders[0]], [".blend", "USD round trip"], roundtrip_dir / f"{prefix}_compare.png", height=320
            )
        return UsdCheck(usd_path, export, roundtrip, comparisons, compare_image)
