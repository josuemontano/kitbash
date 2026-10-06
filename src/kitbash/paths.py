"""Deterministic layout of a scene output directory."""

from pathlib import Path

from attrs import frozen

from kitbash.domain.phases import PhaseName


@frozen
class OutputLayout:
    root: Path

    @classmethod
    def at(cls, root: Path) -> OutputLayout:
        return cls(root.expanduser().resolve())

    # -- top level ---------------------------------------------------------------------------------
    @property
    def config_snapshot(self) -> Path:
        return self.root / "config.snapshot.toml"

    @property
    def rubric_snapshot(self) -> Path:
        return self.root / "rubric.snapshot.md"

    @property
    def state_db(self) -> Path:
        return self.root / "state.db"

    @property
    def input_dir(self) -> Path:
        return self.root / "input"

    @property
    def phases_dir(self) -> Path:
        return self.root / "phases"

    @property
    def scene_dir(self) -> Path:
        return self.root / "scene"

    @property
    def analytics_dir(self) -> Path:
        return self.root / "analytics"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    # -- phases ------------------------------------------------------------------------------------
    def phase_dir(self, phase: PhaseName) -> Path:
        return self.phases_dir / phase.dirname

    def cycles_dir(self, phase: PhaseName, subject: str = "") -> Path:
        base = self.asset_dir(subject) if subject else self.phase_dir(phase)
        return base / "cycles"

    def cycle_dir(self, phase: PhaseName, cycle: int, subject: str = "") -> Path:
        return self.cycles_dir(phase, subject) / f"{cycle:02d}"

    def renders_dir(self, phase: PhaseName) -> Path:
        return self.phase_dir(phase) / "renders"

    @property
    def inventory_json(self) -> Path:
        return self.phase_dir(PhaseName.BREAKDOWN) / "inventory.json"

    @property
    def layout_script(self) -> Path:
        return self.phase_dir(PhaseName.LAYOUT) / "script.py"

    # -- modelling ---------------------------------------------------------------------------------
    def asset_dir(self, asset_id: str) -> Path:
        return self.phase_dir(PhaseName.MODELLING) / asset_id

    def asset_reference_dir(self, asset_id: str) -> Path:
        return self.asset_dir(asset_id) / "reference"

    def asset_trellis_dir(self, asset_id: str) -> Path:
        return self.asset_dir(asset_id) / "trellis"

    def asset_retopo_dir(self, asset_id: str) -> Path:
        return self.asset_dir(asset_id) / "retopo"

    def asset_script(self, asset_id: str) -> Path:
        return self.asset_dir(asset_id) / "script.py"

    def asset_previews_dir(self, asset_id: str) -> Path:
        return self.asset_dir(asset_id) / "previews"

    def asset_roundtrip_dir(self, asset_id: str) -> Path:
        return self.asset_dir(asset_id) / "usd_roundtrip"

    # -- scene -------------------------------------------------------------------------------------
    @property
    def scene_blend(self) -> Path:
        return self.scene_dir / "scene.blend"

    @property
    def scene_usd(self) -> Path:
        return self.scene_dir / "scene.usd"

    @property
    def scene_renders_dir(self) -> Path:
        return self.scene_dir / "renders"

    @property
    def scene_assets_dir(self) -> Path:
        return self.scene_dir / "assets"

    @property
    def analytics_json(self) -> Path:
        return self.analytics_dir / "analytics.json"

    @property
    def analytics_md(self) -> Path:
        return self.analytics_dir / "analytics.md"

    def create(self) -> None:
        for directory in (
            self.input_dir,
            self.logs_dir,
            self.analytics_dir,
            *(self.phase_dir(phase) for phase in PhaseName if phase is not PhaseName.ASSEMBLY),
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def relative(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.root))
        except ValueError:
            return str(path)
