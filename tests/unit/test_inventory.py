import pytest
from attrs import evolve

from kitbash.domain.inventory import Inventory, InventoryError
from kitbash.naming import NamingConvention, slugify, unique_slug


def test_from_dict_normalizes_llm_variations(sample_inventory_dict):
    inventory = Inventory.from_dict(sample_inventory_dict)
    crate, mug = inventory.items
    assert crate.dimensions.as_tuple() == (0.6, 0.4, 0.35)
    assert mug.dimensions.as_tuple() == (0.12, 0.09, 0.1)  # list form
    assert mug.materials_hint == ("glazed ceramic", "white")  # comma string form
    assert mug.relationships[0].kind == "on" and mug.relationships[0].target == "wooden_crate"  # name -> id
    assert mug.position.image_center == pytest.approx((0.5, 0.36))
    assert inventory.scene.camera.focal_length_mm == 50
    assert Inventory.from_dict(inventory.to_dict()) == inventory


def test_ids_are_unique_slugs_and_bad_documents_fail():
    data = {"items": [{"name": "Chair"}, {"name": "Chair"}, {"name": "9 Lives Cat"}]}
    ids = [i.id for i in Inventory.from_dict(data).items]
    assert ids == ["chair", "chair_02", "item_9_lives_cat"]
    with pytest.raises(InventoryError):
        Inventory.from_dict({"items": []})
    with pytest.raises(InventoryError):
        Inventory.from_dict({"objects": []})


def test_duplicates_are_modelled_once(sample_inventory_dict):
    data = dict(sample_inventory_dict)
    chair = {"name": "Dining chair", "description": "Oak chair", "dimensions_m": [0.45, 0.5, 0.9]}
    data["items"] = [*data["items"], {**chair, "id": "chair_01"}, {**chair, "id": "chair_02"}]
    inventory = Inventory.from_dict(data).with_duplicates_linked()
    assert inventory.item("chair_02").same_as == "chair_01" and inventory.item("chair_02").asset_key == "chair_01"
    assert [i.id for i in inventory.modelled_items()] == ["wooden_crate", "ceramic_mug", "chair_01"]
    without = inventory.without("wooden_crate")
    assert without.item("ceramic_mug").relationships == ()


def test_unrecognized_items(sample_inventory_dict):
    data = dict(sample_inventory_dict)
    data["items"] = [{**data["items"][0], "confidence": 0.2}]
    item = Inventory.from_dict(data).items[0]
    assert item.is_unrecognized(0.45)
    assert not evolve(item, search_name="fruit crate").is_unrecognized(0.45)


def test_naming_helpers():
    assert slugify("Mid-century Walnut Armchair!") == "mid_century_walnut_armchair"
    assert slugify("Café Table") == "cafe_table" and slugify("***") == "item"
    taken = {"lamp"}
    assert unique_slug("Lamp", taken) == "lamp_02" and "lamp_02" in taken
    convention = NamingConvention()
    assert "mat_oak_chair_<part>" in convention.describe("oak_chair")
