"""Which assets a scene places (approved, from the backlot) and which items become placeholders."""

from collections.abc import Sequence

from kitbash.agents.layout import PlacedAsset
from kitbash.backlot.library import Backlot
from kitbash.domain.assets import AssetRecord, AssetState
from kitbash.domain.inventory import Inventory, InventoryItem
from kitbash.errors import StateError
from kitbash.naming import NamingConvention, slugify
from kitbash.store.state import StateDB


def collect_scene_assets(
    inventory: Inventory, records: Sequence[AssetRecord], backlot: Backlot, naming: NamingConvention
) -> tuple[list[PlacedAsset], list[InventoryItem]]:
    by_id = {r.id: r for r in records}
    placed: list[PlacedAsset] = []
    for item in inventory.modelled_items():
        record = by_id.get(item.id)
        if record is None or record.state is not AssetState.APPROVED or not record.backlot_id:
            continue
        entry = backlot.get(record.backlot_id)
        slug = str(entry.metadata.get("slug") or slugify(entry.name))
        placed.append(
            PlacedAsset(
                key=item.id,
                slug=slug,
                name=item.name,
                description=item.description,
                blend=entry.blend_path,
                collection=naming.collection.format(slug=slug),
                dimensions=tuple(entry.dimensions),
                backlot_id=entry.id,
            )
        )
    keys = {asset.key for asset in placed}
    return placed, [item for item in inventory.items if item.asset_key not in keys]


class SceneCast:
    """Loads the inventory with the placed assets and placeholder items (layout and assembly share it)."""

    def __init__(self, state: StateDB, backlot: Backlot, naming: NamingConvention) -> None:
        self._state = state
        self._backlot = backlot
        self._naming = naming

    def load(self) -> tuple[Inventory, list[PlacedAsset], list[InventoryItem]]:
        inventory = self._state.inventory.load()
        if inventory is None:
            raise StateError("There is no inventory yet", hint="Run the breakdown phase first.")
        placed, skipped = collect_scene_assets(inventory, self._state.assets.all(), self._backlot, self._naming)
        return inventory, placed, skipped
