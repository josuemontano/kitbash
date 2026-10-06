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


@pytest.mark.parametrize("reverse_order", [False, True])
def test_relationship_ids_take_priority_over_names(reverse_order):
    items = [
        {"id": "chair", "name": "Seat", "relationships": [{"type": "next_to", "target": "chair"}]},
        {"id": "chair_02", "name": "Chair"},
        {"id": "lamp", "relationships": [{"type": "next_to", "target": "chair"}]},
    ]
    inventory = Inventory.from_dict({"items": items[::-1] if reverse_order else items})

    assert inventory.item("lamp").relationships[0].target == "chair"
    assert inventory.item("chair").relationships == ()  # exact self-id must not fall back to another item's name
    assert Inventory.from_dict(inventory.to_dict()) == inventory


@pytest.mark.parametrize("reverse_order", [False, True])
def test_relationship_names_resolve_only_when_unambiguous(reverse_order):
    items = [
        {"id": "chair_01", "name": "Dining Chair"},
        {"id": "chair_02", "name": "DINING CHAIR"},
        {"id": "chair_03", "name": "dining chair"},
        {"id": "table", "name": "Side table"},
        {"id": "lamp", "name": "Lamp", "relationships": [
            {"type": "next_to", "target": "Dining Chair"},
            {"type": "on", "target": "SIDE TABLE"},
            {"type": "next_to", "target": "chair_02"},
            {"type": "next_to", "target": "missing"},
            {"type": "next_to", "target": "Lamp"},
        ]},
    ]
    inventory = Inventory.from_dict({"items": items[::-1] if reverse_order else items})

    assert [(rel.kind, rel.target) for rel in inventory.item("lamp").relationships] == [
        ("on", "table"), ("next_to", "chair_02"),
    ]


def test_duplicates_are_modelled_once(sample_inventory_dict):
    data = dict(sample_inventory_dict)
    chair = {"name": "Dining chair", "description": "Oak chair", "dimensions_m": [0.45, 0.5, 0.9]}
    data["items"] = [*data["items"], {**chair, "id": "chair_01"}, {**chair, "id": "chair_02"}]
    inventory = Inventory.from_dict(data).with_duplicates_linked()
    assert inventory.item("chair_02").same_as == "chair_01" and inventory.item("chair_02").asset_key == "chair_01"
    assert [i.id for i in inventory.modelled_items()] == ["wooden_crate", "ceramic_mug", "chair_01"]
    without = inventory.without("wooden_crate")
    assert without.item("ceramic_mug").relationships == ()


@pytest.mark.parametrize(
    "variant",
    [
        {"materials_hint": ["metal", "brown"]},
        {"materials_hint": []},
        {"category": "decor"},
        {"dimensions_m": [0.46, 0.5, 0.9]},
        {"dimensions_m": [0.45, 0.51, 0.9]},
        {"dimensions_m": [0.45, 0.5, 0.91]},
    ],
    ids=["material", "unknown-material", "category", "width", "depth", "height"],
)
def test_duplicate_linking_keeps_variants_separate(variant):
    chair = {
        "name": "Chair",
        "description": "Dining chair",
        "category": "furniture",
        "materials_hint": ["wood", "brown"],
        "dimensions_m": [0.45, 0.5, 0.9],
    }
    inventory = Inventory.from_dict({"items": [
        {**chair, "id": "chair"},
        {**chair, **variant, "id": "variant"},
        {**chair, **variant, "id": "variant_copy"},
    ]}).with_duplicates_linked()

    assert [item.asset_key for item in inventory.items] == ["chair", "variant", "variant"]
    assert [item.id for item in inventory.modelled_items()] == ["chair", "variant"]
    assert inventory.with_duplicates_linked() == inventory


def test_duplicate_linking_normalizes_identity_but_ignores_placement():
    chair = {
        "name": "Chair",
        "description": "Dining chair",
        "category": "furniture",
        "materials_hint": ["wood", "brown"],
        "dimensions_m": [0.45, 0.5, 0.9],
    }
    inventory = Inventory.from_dict({"items": [
        {**chair, "id": "chair"},
        {
            **chair,
            "id": "copy",
            "name": " CHAIR ",
            "description": " DINING CHAIR ",
            "category": " FURNITURE ",
            "materials_hint": [" BROWN ", "WOOD"],
            "position": {"location_m": [2, 0, 0], "rotation_deg": [0, 0, 90]},
        },
    ]}).with_duplicates_linked()

    assert inventory.item("copy").asset_key == "chair"
    assert [item.id for item in inventory.modelled_items()] == ["chair"]
    assert inventory.item("copy").position.location == (2, 0, 0)
    assert inventory.item("copy").position.rotation_deg == (0, 0, 90)


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


def test_airborne_support_survives_inventory_roundtrip_and_invalid_policy_fails():
    inventory = Inventory.from_dict({"items": [{"id": "cloud", "support": "airborne"}, {"id": "house"}]})
    restored = Inventory.from_dict(inventory.to_dict())
    assert restored.item("cloud").support == "airborne"
    assert restored.item("house").support == "contact"
    with pytest.raises(InventoryError, match="support policy"):
        Inventory.from_dict({"items": [{"id": "cloud", "support": "ignore"}]})
