"""The fixed (kitbash-authored) Blender scripts, with the render and naming settings of the run."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from kitbash.config import BlenderConfig, UsdConfig
from kitbash.infra.blender import BlenderRunner, blender_script
from kitbash.naming import NamingConvention


class BlenderToolkit:
    def __init__(
        self, runner: BlenderRunner, blender: BlenderConfig, usd: UsdConfig, naming: NamingConvention, downloads: Path
    ) -> None:
        self._runner = runner
        self._blender = blender
        self._usd = usd
        self._naming = naming
        self._downloads = downloads

    def base_args(self, **extra: Any) -> dict[str, Any]:
        """Arguments every script receives (naming, download cache, face budget)."""
        return {
            "naming": self._naming.as_dict(),
            "download_dir": str(self._downloads),
            "max_faces": self._blender.max_faces,
            **extra,
        }

    def run_script(
        self, script: Path, args: Mapping[str, Any], log_path: Path, *, blend: Path | None = None, timeout_s: float | None = None
    ) -> dict[str, Any]:
        return self._runner.run(script, args=self.base_args(**args), log_path=log_path, blend=blend, timeout_s=timeout_s)

    def _fixed(self, name: str, blend: Path | None, log_dir: Path, log_name: str, timeout_s: float | None = None, **args: Any) -> dict[str, Any]:
        return self.run_script(blender_script(name), args, log_dir / f"{log_name}.log", blend=blend, timeout_s=timeout_s)

    # -- assets ------------------------------------------------------------------------------------

    def render_views(
        self,
        blend: Path,
        output_dir: Path,
        prefix: str,
        log_dir: Path,
        *,
        views: Sequence[str] | None = None,
        resolution: Sequence[int] | None = None,
        samples: int | None = None,
    ) -> list[Path]:
        result = self._fixed(
            "render_views.py", blend, log_dir, f"{prefix}_render",
            engine=self._blender.render_engine, device=self._blender.cycles_device,
            views=list(views or self._blender.preview_views),
            resolution=list(resolution or self._blender.preview_resolution),
            samples=samples or self._blender.preview_samples,
            output_dir=str(output_dir), prefix=prefix,
        )
        return [Path(p) for p in result["images"]]

    def inspect_asset(self, blend: Path, slug: str, expected_dimensions: Sequence[float], log_dir: Path, log_name: str) -> tuple[dict, dict]:
        result = self._fixed("inspect_asset.py", blend, log_dir, log_name, slug=slug, expected_dimensions=list(expected_dimensions))
        return result["report"], result["facts"]

    # -- USD ---------------------------------------------------------------------------------------

    def export_usd(self, blend: Path, output_usd: Path, work_dir: Path, log_dir: Path, log_name: str, *, scene: bool = False) -> dict[str, Any]:
        result = self._fixed(
            "usd_export.py", blend, log_dir, log_name, timeout_s=self._blender.bake_timeout_s,
            output_usd=str(output_usd), work_dir=str(work_dir), scene=scene,
            materialx=self._usd.materialx, bake_preview_fallback=self._usd.bake_preview_fallback,
            bake_resolution=self._usd.bake_resolution, bake_samples=self._usd.bake_samples,
            device=self._blender.cycles_device,
        )
        return result["usd"]

    def usd_roundtrip(
        self, usd: Path, expected: Mapping[str, Sequence[str]], *, mode: str, output_dir: Path, prefix: str, log_dir: Path,
        engine: str | None = None, scene_expectations: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = self._fixed(
            "usd_roundtrip.py", None, log_dir, f"{prefix}_roundtrip",
            usd_path=str(usd), expected={k: list(v) for k, v in expected.items()}, mode=mode,
            views=list(self._blender.preview_views), resolution=list(self._usd.roundtrip_resolution),
            samples=self._usd.roundtrip_samples, engine=engine or self._blender.render_engine,
            device=self._blender.cycles_device, output_dir=str(output_dir), prefix=prefix,
            scene_expectations=dict(scene_expectations) if scene_expectations is not None else None,
        )
        return result["roundtrip"]

    # -- scenes ------------------------------------------------------------------------------------

    def render_scene(
        self, blend: Path, output_path: Path, log_dir: Path, *, resolution: Sequence[int], samples: int, engine: str | None = None
    ) -> Path:
        result = self._fixed(
            "render_scene.py", blend, log_dir, f"{output_path.stem}_render",
            resolution=list(resolution), samples=samples, engine=engine,
            device=self._blender.cycles_device, output_path=str(output_path),
        )
        return Path(result["images"][0])

    def inspect_scene(
        self, blend: Path, assets: Mapping[str, Any], expected_placeholders: Sequence[str], log_dir: Path, log_name: str
    ) -> tuple[dict, dict]:
        result = self._fixed(
            "inspect_scene.py", blend, log_dir, log_name,
            assets=dict(assets), expected_assets={key: int(spec.get("instances", 1)) for key, spec in assets.items()},
            expected_placeholders=list(expected_placeholders),
        )
        return result["report"], result["facts"]

    def localize(self, blend: Path, log_dir: Path) -> dict[str, Any]:
        return self._fixed("localize_files.py", blend, log_dir, "localize")["localized"]

    @property
    def roundtrip_resolution(self) -> tuple[int, int]:
        return self._usd.roundtrip_resolution

    @property
    def roundtrip_samples(self) -> int:
        return self._usd.roundtrip_samples

    @property
    def views(self) -> tuple[str, ...]:
        return self._blender.preview_views
