"""Configuration: packaged defaults <- user TOML <- CLI overrides, structured into attrs classes."""

import copy
import tomllib
import types
import typing
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Any

import attrs
import tomli_w
from attrs import frozen

from kitbash.domain.phases import PhaseName
from kitbash.domain.roles import Role
from kitbash.errors import ConfigError
from kitbash.naming import NamingConvention
from kitbash.retopology.base import RetopologyMethod

DEFAULTS_PACKAGE = "kitbash.defaults"
FREE_FORM_TABLES = frozenset({"models.phases", "styles"})
RETOPOLOGY_DEVICES = ("auto", "cuda", "mps")
# Keys removed on purpose; old config files and run snapshots that still set them keep loading.
REMOVED_KEYS = {"retopology": {"fallback_on_error"}}


@frozen
class PathsConfig:
    backlot: Path
    trellis: Path
    downloads: Path
    triflow_weights: Path


@frozen
class ToolsConfig:
    omp: str
    blender: str
    trellis_python: str

    def resolved_trellis_python(self, trellis_dir: Path) -> str:
        if self.trellis_python:
            return self.trellis_python
        venv_python = trellis_dir / ".venv" / "bin" / "python"
        return str(venv_python) if venv_python.exists() else "python3"


@frozen
class CapabilitiesConfig:
    image_roles: tuple[str, ...]


@frozen
class ModelsConfig:
    roles: Mapping[str, str]
    phases: Mapping[str, Mapping[str, str]]
    thinking: Mapping[str, str]
    capabilities: CapabilitiesConfig

    def model_for(self, role: Role, phase: PhaseName | None = None) -> str:
        if phase is not None and (override := self.phases.get(phase.value, {}).get(role.value)):
            return override
        return self.roles[role.value]

    def thinking_for(self, role: Role) -> str | None:
        return self.thinking.get(role.value) or None

    def needs_images(self, role: Role) -> bool:
        return role.value in self.capabilities.image_roles

    def all_assignments(self) -> list[tuple[PhaseName | None, Role, str]]:
        """Every (phase, role, model) the run can use; phase is None for the global default."""
        rows: list[tuple[PhaseName | None, Role, str]] = [(None, role, self.roles[role.value]) for role in Role]
        for phase_name, overrides in self.phases.items():
            rows.extend((PhaseName(phase_name), Role(role), model) for role, model in overrides.items())
        return rows


@frozen
class OmpConfig:
    timeout_s: float
    retries: int
    retry_backoff_s: float
    extra_args: tuple[str, ...]
    preflight_ping: bool


@frozen
class PipelineConfig:
    style: str
    threads: int
    review_buffer: int


@frozen
class CriticConfig:
    max_cycles: int
    pass_threshold: float
    require_all_pass: bool
    stall_cycles: int
    stall_epsilon: float
    revert_epsilon: float
    patch_attempts: int


@frozen
class BreakdownConfig:
    confidence_threshold: float
    max_items: int


@frozen
class BacklotConfig:
    match_threshold: float
    top_k: int
    match_style: bool


@frozen
class EmbeddingConfig:
    backend: str
    model: str
    base_url: str
    dimensions: int
    timeout_s: float


@frozen
class ReferenceConfig:
    providers: tuple[str, ...]
    max_candidates: int
    per_provider: int
    min_side_px: int
    max_download_mb: float
    timeout_s: float
    query_suffix: str
    allowed_licenses: tuple[str, ...]
    min_crop_side_px: int
    auto_select_threshold: float
    ambiguity_margin: float
    vision_fallback: bool
    search_cache_ttl_s: float


@frozen
class PolyHavenConfig:
    enabled: bool
    cache_ttl_s: float


@frozen
class ModellingConfig:
    method: str


@frozen
class TrellisConfig:
    steps: int
    pipeline_type: str
    no_texture: bool
    retries: int
    timeout_s: float
    max_concurrent: int
    seed: int
    mesh_up_axis: str


@frozen
class RetopologyConfig:
    method: str
    face_count: int
    qem_threshold: float
    quad_ratio: float
    flow_steps: int
    device: str

    @property
    def method_enum(self) -> RetopologyMethod:
        return RetopologyMethod(self.method)


@frozen
class BlenderConfig:
    timeout_s: float
    bake_timeout_s: float
    render_engine: str
    cycles_device: str
    preview_resolution: tuple[int, int]
    preview_samples: int
    preview_views: tuple[str, ...]
    final_resolution: tuple[int, int]
    final_samples: int
    max_faces: int


@frozen
class UsdConfig:
    materialx: str
    bake_preview_fallback: bool
    bake_resolution: int
    bake_samples: int
    roundtrip_resolution: tuple[int, int]
    roundtrip_samples: int
    roundtrip_threshold: float


@frozen
class AssemblyConfig:
    mode: str


@frozen
class UiConfig:
    refresh_per_second: float
    show_previews: bool


@frozen
class StyleConfig:
    render_engine: str
    guidance: str


@frozen
class Config:
    paths: PathsConfig
    tools: ToolsConfig
    models: ModelsConfig
    omp: OmpConfig
    pipeline: PipelineConfig
    critic: CriticConfig
    breakdown: BreakdownConfig
    backlot: BacklotConfig
    embedding: EmbeddingConfig
    reference: ReferenceConfig
    polyhaven: PolyHavenConfig
    modelling: ModellingConfig
    trellis: TrellisConfig
    retopology: RetopologyConfig
    blender: BlenderConfig
    usd: UsdConfig
    naming: NamingConvention
    assembly: AssemblyConfig
    ui: UiConfig
    styles: Mapping[str, StyleConfig]
    raw: Mapping[str, Any] = attrs.field(eq=False, repr=False)

    @property
    def style(self) -> StyleConfig:
        return self.styles[self.pipeline.style]

    def trellis_concurrency(self) -> int:
        return self.trellis.max_concurrent or self.pipeline.threads

    def snapshot_toml(self) -> str:
        return tomli_w.dumps(_jsonable(self.raw))


# -- loading -------------------------------------------------------------------------------------


def default_config_text() -> str:
    return resources.files(DEFAULTS_PACKAGE).joinpath("config.toml").read_text(encoding="utf-8")


def default_rubric_path() -> Path:
    return Path(str(resources.files(DEFAULTS_PACKAGE).joinpath("rubric.md")))


def load_config(user_path: Path | None = None, overrides: Mapping[str, Any] | None = None) -> Config:
    """Merge defaults, an optional user file and dotted-key overrides, then validate."""
    merged = _parse_toml(default_config_text(), "packaged defaults")
    if user_path is not None:
        if not user_path.is_file():
            raise ConfigError(f"Config file not found: {user_path}")
        user = _drop_removed_keys(_parse_toml(user_path.read_text(encoding="utf-8"), str(user_path)))
        _reject_unknown_keys(user, merged, "")
        merged = deep_merge(merged, user)
    for dotted, value in (overrides or {}).items():
        _set_dotted(merged, dotted, value)
    return structure_config(merged)


def structure_config(data: Mapping[str, Any]) -> Config:
    raw = copy.deepcopy(dict(data))
    raw.setdefault("modelling", {"method": "trellis"})
    body = dict(raw)
    models = dict(body.pop("models"))
    body["models"] = {
        "roles": {k: v for k, v in models.items() if isinstance(v, str)},
        "phases": models.get("phases", {}),
        "thinking": models.get("thinking", {}),
        "capabilities": models.get("capabilities", {}),
    }
    config = _structure(Config, {**body, "raw": raw}, "")
    _validate(config)
    return config


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def parse_model_overrides(pairs: Mapping[str, str]) -> dict[str, str]:
    """Turn ``{"code": m, "layout.code": m}`` into dotted config keys."""
    overrides: dict[str, str] = {}
    for key, model in pairs.items():
        parts = key.split(".")
        match parts:
            case [role]:
                overrides[f"models.{_role(role)}"] = model
            case [phase, role]:
                overrides[f"models.phases.{_phase(phase)}.{_role(role)}"] = model
            case _:
                raise ConfigError(f"Invalid model override --model.{key}", hint="Use --model.<role>=<name> or --model.<phase>.<role>=<name>.")
    return overrides


def _role(name: str) -> str:
    try:
        return Role(name.replace("-", "_")).value
    except ValueError:
        raise ConfigError(f"Unknown model role {name!r}", hint=f"Roles: {', '.join(r.value for r in Role)}.") from None


def _phase(name: str) -> str:
    try:
        return PhaseName(name).value
    except ValueError:
        raise ConfigError(f"Unknown phase {name!r}", hint=f"Phases: {', '.join(p.value for p in PhaseName)}.") from None


def _parse_toml(text: str, origin: str) -> dict[str, Any]:
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {origin}: {exc}") from exc


def _drop_removed_keys(user: dict[str, Any]) -> dict[str, Any]:
    """Retopology of a Trellis mesh is mandatory, so ``retopology.fallback_on_error`` no longer exists."""
    for table, keys in REMOVED_KEYS.items():
        if isinstance(user.get(table), dict):
            user[table] = {k: v for k, v in user[table].items() if k not in keys}
    return user


def _reject_unknown_keys(user: Mapping[str, Any], defaults: Mapping[str, Any], prefix: str) -> None:
    for key, value in user.items():
        dotted = f"{prefix}{key}"
        if prefix.rstrip(".") in FREE_FORM_TABLES:
            continue
        if key not in defaults:
            raise ConfigError(f"Unknown config key {dotted!r}")
        if isinstance(value, Mapping) and isinstance(defaults[key], Mapping):
            _reject_unknown_keys(value, defaults[key], f"{dotted}.")


def _set_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    *parents, leaf = dotted.split(".")
    node = data
    for part in parents:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ConfigError(f"Cannot override {dotted!r}: {part!r} is not a table")
    node[leaf] = value


def _validate(config: Config) -> None:
    missing = [role.value for role in Role if not config.models.roles.get(role.value)]
    if missing:
        raise ConfigError(f"No model configured for roles: {', '.join(missing)}")
    for phase, overrides in config.models.phases.items():
        _phase(phase)
        for role in overrides:
            _role(role)
    if config.pipeline.style not in config.styles:
        raise ConfigError(
            f"Unknown style {config.pipeline.style!r}", hint=f"Styles: {', '.join(sorted(config.styles))}."
        )
    if config.pipeline.threads < 1 or config.pipeline.review_buffer < 1:
        raise ConfigError("--threads and --review-buffer must be at least 1")
    if config.critic.max_cycles < 1:
        raise ConfigError("--max-cycles must be at least 1")
    if config.modelling.method not in {"trellis", "procedural"}:
        raise ConfigError("modelling.method must be 'trellis' or 'procedural'")
    methods = [m.value for m in RetopologyMethod]
    if config.retopology.method not in methods:
        raise ConfigError(f"Unknown retopology method {config.retopology.method!r}", hint=f"Methods: {', '.join(methods)}.")
    if config.retopology.device not in RETOPOLOGY_DEVICES:
        raise ConfigError(f"Unknown retopology.device {config.retopology.device!r}", hint=f"Devices: {', '.join(RETOPOLOGY_DEVICES)}.")
    if config.blender.cycles_device != "GPU":
        raise ConfigError(
            f"blender.cycles_device must be \"GPU\", got {config.blender.cycles_device!r}", hint="CPU rendering is not supported."
        )
    if config.retopology.face_count < 1 or config.retopology.flow_steps < 1:
        raise ConfigError("retopology.face_count and retopology.flow_steps must be at least 1")
    if config.retopology.qem_threshold < 0 or not 0.0 <= config.retopology.quad_ratio <= 1.0:
        raise ConfigError("retopology.qem_threshold must be >= 0 and retopology.quad_ratio must be in [0, 1]")
    if config.usd.materialx not in {"auto", "off"}:
        raise ConfigError("usd.materialx must be 'auto' or 'off'")
    if config.assembly.mode not in {"append", "link"}:
        raise ConfigError("assembly.mode must be 'append' or 'link'")
    reference = config.reference
    if set(reference.providers) - {"input_crop", "wikimedia", "openverse"}:
        raise ConfigError("reference.providers supports input_crop, wikimedia and openverse; use vision_fallback for optional visual selection")
    if min(reference.max_candidates, reference.per_provider, reference.min_side_px, reference.min_crop_side_px) < 1:
        raise ConfigError("reference candidate counts and minimum dimensions must be positive")
    if not (0 < reference.auto_select_threshold <= 1 and 0 <= reference.ambiguity_margin <= 1):
        raise ConfigError("reference.auto_select_threshold must be in (0, 1] and ambiguity_margin in [0, 1]")
    if not (reference.max_download_mb > 0 and reference.timeout_s > 0 and reference.search_cache_ttl_s >= 0):
        raise ConfigError("reference download/timeout limits must be positive and cache TTL nonnegative")


# -- generic structuring ---------------------------------------------------------------------------


def _structure(tp: Any, value: Any, where: str) -> Any:
    origin = typing.get_origin(tp)
    if attrs.has(tp):
        return _structure_attrs(tp, value, where)
    if tp is Any:
        return value
    if tp is Path:
        return Path(str(value)).expanduser()
    if origin is tuple:
        args = typing.get_args(tp)
        items = _expect(value, (list, tuple), where)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_structure(args[0], item, f"{where}[{i}]") for i, item in enumerate(items))
        if len(items) != len(args):
            raise ConfigError(f"{where} expects {len(args)} values, got {len(items)}")
        return tuple(_structure(arg, item, f"{where}[{i}]") for i, (arg, item) in enumerate(zip(args, items, strict=True)))
    if origin in (Mapping, dict):
        _, value_type = typing.get_args(tp)
        table = _expect(value, (dict,), where)
        return types.MappingProxyType({str(k): _structure(value_type, v, f"{where}.{k}") for k, v in table.items()})
    if tp is bool:
        return _expect(value, (bool,), where)
    if tp is int:
        if isinstance(value, bool):
            raise ConfigError(f"{where} must be an integer")
        return int(_expect(value, (int,), where))
    if tp is float:
        if isinstance(value, bool):
            raise ConfigError(f"{where} must be a number")
        return float(_expect(value, (int, float), where))
    if tp is str:
        return _expect(value, (str,), where)
    raise ConfigError(f"Unsupported config type {tp!r} at {where}")


def _structure_attrs(cls: type, value: Any, where: str) -> Any:
    table = _expect(value, (dict,), where)
    attrs.resolve_types(cls)
    kwargs = {}
    known = set()
    for field in attrs.fields(cls):
        known.add(field.name)
        key = f"{where}.{field.name}".lstrip(".")
        if field.name in table:
            kwargs[field.name] = _structure(field.type, table[field.name], key)
        elif field.default is attrs.NOTHING:
            raise ConfigError(f"Missing config key {key!r}")
    unknown = set(table) - known
    if unknown:
        raise ConfigError(f"Unknown config keys under {where or 'root'!r}: {', '.join(sorted(unknown))}")
    return cls(**kwargs)


def _expect(value: Any, kinds: tuple[type, ...], where: str) -> Any:
    if not isinstance(value, kinds):
        names = " or ".join(k.__name__ for k in kinds)
        raise ConfigError(f"{where} must be {names}, got {type(value).__name__}")
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value
