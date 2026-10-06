"""Scene inventory: the objects found in the reference, in world units (meters, Z up)."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from attrs import evolve, frozen

from kitbash.errors import KitbashError
from kitbash.naming import unique_slug

RELATIONSHIP_KINDS = ("on", "next_to", "inside", "under", "attached_to", "in_front_of", "behind")
Vec3 = tuple[float, float, float]


class InventoryError(KitbashError):
    """An inventory document is malformed."""


def embedding_text(name: str, category: str, description: str) -> str:
    """Text embedded for inventory items and backlot assets alike, so their vectors are comparable."""
    return f"{name}. {category}. {description}"


@frozen
class Dimensions:
    width: float  # X
    depth: float  # Y
    height: float  # Z

    def as_tuple(self) -> Vec3:
        return (self.width, self.depth, self.height)

    @classmethod
    def parse(cls, value: Any) -> Dimensions:
        if isinstance(value, Mapping):
            keys = [("width", "x"), ("depth", "y"), ("height", "z")]
            numbers = [_first(value, names, 0.3) for names in keys]
        else:
            numbers = _vector(value, 3, default=0.3)
        return cls(*(max(float(n), 0.001) for n in numbers))


@frozen
class Placement:
    location: Vec3 = (0.0, 0.0, 0.0)  # base center of the object
    rotation_deg: Vec3 = (0.0, 0.0, 0.0)
    image_bbox: tuple[float, float, float, float] | None = None  # normalized, top-left origin
    image_center: tuple[float, float] | None = None

    @classmethod
    def parse(cls, value: Mapping[str, Any]) -> Placement:
        bbox = value.get("image_bbox")
        center = value.get("image_center")
        parsed_bbox = tuple(_clamp01(v) for v in _vector(bbox, 4)) if bbox else None
        if parsed_bbox and center is None:
            center = ((parsed_bbox[0] + parsed_bbox[2]) / 2, (parsed_bbox[1] + parsed_bbox[3]) / 2)
        return cls(
            location=_vector(value.get("location_m", value.get("location")), 3, default=0.0),
            rotation_deg=_vector(value.get("rotation_deg", value.get("rotation")), 3, default=0.0),
            image_bbox=parsed_bbox,
            image_center=tuple(_clamp01(v) for v in _vector(center, 2)) if center else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_bbox": list(self.image_bbox) if self.image_bbox else None,
            "image_center": list(self.image_center) if self.image_center else None,
            "location_m": list(self.location),
            "rotation_deg": list(self.rotation_deg),
        }


@frozen
class Relationship:
    kind: str
    target: str

    @classmethod
    def parse(cls, value: Mapping[str, Any]) -> Relationship:
        kind = str(value.get("type", value.get("kind", "next_to"))).strip().lower().replace(" ", "_")
        return cls(kind=kind if kind in RELATIONSHIP_KINDS else "next_to", target=str(value.get("target", "")))


@frozen
class CameraEstimate:
    location: Vec3 = (0.0, -4.0, 1.6)
    rotation_deg: Vec3 = (80.0, 0.0, 0.0)
    focal_length_mm: float = 35.0

    @classmethod
    def parse(cls, value: Mapping[str, Any] | None) -> CameraEstimate:
        value = value or {}
        default = cls()
        return cls(
            location=_vector(value.get("location_m", value.get("location")), 3, default=None) or default.location,
            rotation_deg=_vector(value.get("rotation_deg"), 3, default=None) or default.rotation_deg,
            focal_length_mm=float(value.get("focal_length_mm", default.focal_length_mm)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "location_m": list(self.location),
            "rotation_deg": list(self.rotation_deg),
            "focal_length_mm": self.focal_length_mm,
        }


@frozen
class SceneInfo:
    description: str = ""
    environment: str = "indoor"
    lighting: str = ""
    style_notes: str = ""
    camera: CameraEstimate = CameraEstimate()

    @classmethod
    def parse(cls, value: Mapping[str, Any] | None) -> SceneInfo:
        value = value or {}
        return cls(
            description=str(value.get("description", "")),
            environment=str(value.get("environment", "indoor")),
            lighting=str(value.get("lighting", "")),
            style_notes=str(value.get("style_notes", "")),
            camera=CameraEstimate.parse(value.get("camera")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "environment": self.environment,
            "lighting": self.lighting,
            "style_notes": self.style_notes,
            "camera": self.camera.to_dict(),
        }


@frozen
class InventoryItem:
    id: str
    name: str
    description: str
    category: str
    dimensions: Dimensions
    position: Placement
    relationships: tuple[Relationship, ...] = ()
    materials_hint: tuple[str, ...] = ()
    confidence: float = 1.0
    # Decisions taken at the breakdown gate.
    reuse_backlot_id: str | None = None
    user_reference: str | None = None
    search_name: str | None = None
    same_as: str | None = None  # another item this one is an identical copy of (modelled once)
    support: str = "contact"  # "airborne" deliberately needs no physical support

    @property
    def asset_key(self) -> str:
        """The modelled asset this item is placed with."""
        return self.same_as or self.id

    def embedding_text(self) -> str:
        return embedding_text(self.name, self.category, self.description)

    def is_unrecognized(self, threshold: float) -> bool:
        return self.confidence < threshold and not (self.user_reference or self.search_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "dimensions_m": {"width": self.dimensions.width, "depth": self.dimensions.depth, "height": self.dimensions.height},
            "position": self.position.to_dict(),
            "relationships": [{"type": r.kind, "target": r.target} for r in self.relationships],
            "materials_hint": list(self.materials_hint),
            "confidence": self.confidence,
            "support": self.support,
            "reuse_backlot_id": self.reuse_backlot_id,
            "user_reference": self.user_reference,
            "search_name": self.search_name,
            "same_as": self.same_as,
        }


@frozen
class Inventory:
    scene: SceneInfo
    items: tuple[InventoryItem, ...]

    def item(self, item_id: str) -> InventoryItem:
        for item in self.items:
            if item.id == item_id:
                return item
        raise InventoryError(f"No inventory item with id {item_id!r}")

    def replace_item(self, item: InventoryItem) -> Inventory:
        return evolve(self, items=tuple(item if i.id == item.id else i for i in self.items))

    def without(self, item_id: str) -> Inventory:
        """Drop an item and every relationship that points at it."""
        items = []
        for item in self.items:
            if item.id == item_id:
                continue
            relationships = tuple(r for r in item.relationships if r.target != item_id)
            items.append(evolve(item, relationships=relationships, same_as=None if item.same_as == item_id else item.same_as))
        return evolve(self, items=tuple(items))

    def modelled_items(self) -> tuple[InventoryItem, ...]:
        """Items that need their own asset (not copies of another item)."""
        return tuple(item for item in self.items if item.same_as is None)

    def with_duplicates_linked(self) -> Inventory:
        """Link copies with matching text, category, materials and exact dimensions."""
        first: dict[tuple[str, str, str, frozenset[str], Dimensions], str] = {}
        items = []
        for item in self.items:
            key = (
                item.name.strip().lower(),
                item.description.strip().lower(),
                item.category.strip().lower(),
                frozenset(material.strip().lower() for material in item.materials_hint),
                item.dimensions,
            )
            original = first.setdefault(key, item.id)
            items.append(evolve(item, same_as=None if original == item.id else original))
        return evolve(self, items=tuple(items))

    def to_dict(self) -> dict[str, Any]:
        return {"scene": self.scene.to_dict(), "items": [item.to_dict() for item in self.items]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, max_items: int | None = None) -> Inventory:
        if not isinstance(data, Mapping) or not isinstance(data.get("items"), list):
            raise InventoryError("Inventory must be an object with an 'items' list")
        raw_items = data["items"][:max_items] if max_items else data["items"]
        taken: set[str] = set()
        items = [_parse_item(raw, taken) for raw in raw_items if isinstance(raw, Mapping)]
        if not items:
            raise InventoryError("Inventory has no items")
        return cls(scene=SceneInfo.parse(data.get("scene")), items=tuple(_resolve_targets(items)))


def _parse_item(raw: Mapping[str, Any], taken: set[str]) -> InventoryItem:
    name = str(raw.get("name") or raw.get("id") or "item").strip()
    support = raw.get("support", "contact")
    if support not in ("contact", "airborne"):
        raise InventoryError(f"Invalid support policy {support!r} for {name!r}; use contact or airborne")
    return InventoryItem(
        id=unique_slug(str(raw.get("id") or name), taken),
        name=name,
        description=str(raw.get("description", "")).strip(),
        category=str(raw.get("category", "prop")).strip().lower() or "prop",
        dimensions=Dimensions.parse(raw.get("dimensions_m", raw.get("dimensions", {}))),
        position=Placement.parse(raw.get("position") or {}),
        relationships=tuple(Relationship.parse(r) for r in raw.get("relationships", []) if isinstance(r, Mapping)),
        materials_hint=tuple(str(m) for m in _as_list(raw.get("materials_hint", raw.get("materials", [])))),
        confidence=_clamp01(raw.get("confidence", 1.0)),
        reuse_backlot_id=raw.get("reuse_backlot_id") or None,
        user_reference=raw.get("user_reference") or None,
        search_name=raw.get("search_name") or None,
        same_as=raw.get("same_as") or None,
        support=support,
    )


def _resolve_targets(items: Sequence[InventoryItem]) -> Iterable[InventoryItem]:
    """Prefer exact ids, then unambiguous names; drop unknown and ambiguous targets."""
    ids = {item.id for item in items}
    by_name: dict[str, str | None] = {}
    for item in items:
        name = item.name.lower()
        by_name[name] = None if name in by_name else item.id
    for item in items:
        relationships = []
        for rel in item.relationships:
            target = rel.target if rel.target in ids else by_name.get(rel.target.lower())
            if target and target != item.id:
                relationships.append(evolve(rel, target=target))
        yield evolve(item, relationships=tuple(relationships))


def _first(mapping: Mapping[str, Any], names: Sequence[str], default: float) -> float:
    for name in names:
        if name in mapping:
            return _number(mapping[name], default)
    return default


def _vector(value: Any, size: int, default: float | None = 0.0) -> Any:
    if not isinstance(value, (list, tuple)) or len(value) < size:
        return None if default is None else tuple(default for _ in range(size))
    return tuple(_number(v, default or 0.0) for v in value[:size])


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: Any) -> float:
    return min(max(_number(value, 0.0), 0.0), 1.0)


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return list(value) if isinstance(value, (list, tuple)) else []
