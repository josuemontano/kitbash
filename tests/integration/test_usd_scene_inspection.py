"""Final scene acceptance inspects real imported USD geometry and the selected camera."""

import shutil

import pytest

from kitbash.config import load_config
from kitbash.infra.blender import BlenderRunner
from kitbash.services.blender_toolkit import BlenderToolkit
from kitbash.services.usd_fidelity import UsdFidelityChecker
from tests.helpers import requires_blender

pytestmark = [pytest.mark.integration, pytest.mark.blender, requires_blender]

ASSET = """import bpy
import kitbash_bpy as kb
kb.reset_scene()
bpy.ops.mesh.primitive_cube_add(size=0.5)
root = bpy.context.object
kb.asset_collection().objects.link(root)
kb.origin_to_base(root)
material = kb.principled("body", base_color=(0.15, 0.45, 0.8), roughness=0.65)
kb.assign(root, material)
bpy.ops.mesh.primitive_cube_add(size=0.15, location=(0.12, 0.0, 0.575))
kb.assign(bpy.context.object, material)
kb.save_asset(root)
"""

LAYOUT = """import bpy
import kitbash_bpy as kb
kb.reset_scene()
kb.ground_plane(size=10.0)
kb.place_asset("crate", (-1.0, 0.0, 0.0))
kb.place_asset("crate", (0.0, 0.0, 0.0))
kb.place_placeholder("skipped", (0.4, 0.4, 0.6), (1.0, 0.0, 0.0))
kb.place_placeholder("skipped", (0.4, 0.4, 0.6), (1.0, 1.0, 0.0))
kb.camera((0.0, -5.0, 3.0), look_at_point=(0.0, -10.0, 8.0), name="a_decoy", focal_length_mm=20)
kb.camera((4.0, -6.0, 4.0), look_at_point=(0.0, 0.3, 0.4), name="z_selected.camera", focal_length_mm=45)
kb.light("SUN", (0.0, 0.0, 6.0), energy=3.0, rotation_deg=(30.0, 0.0, 20.0))
kb.color_world((0.5, 0.5, 0.5), 0.5)
kb.save_scene()
"""

EDIT_USD = """from pxr import Sdf, Usd
import kitbash_bpy as kb
args = kb.args()
stage = Usd.Stage.Open(args["usd_path"])
for key, value in args["tags"].items():
    prims = [prim for prim in stage.Traverse() if (attribute := prim.GetAttribute(key)) and attribute.Get() == value]
    assert len(prims) == args["count"], (key, [str(prim.GetPath()) for prim in prims])
    path = prims[0].GetPath()
    if args["operation"] == "remove":
        stage.RemovePrim(path)
    else:
        destination = path.GetParentPath().AppendChild(prims[0].GetName() + "_extra")
        Sdf.CopySpec(stage.GetRootLayer(), path, stage.GetRootLayer(), destination)
stage.Save()
"""


@pytest.fixture(scope="module", params=["append", "link"])
def scene_export(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp(f"usd_scene_{request.param}")
    config = load_config(None, {"usd.materialx": "off", "usd.roundtrip_resolution": [96, 96], "usd.roundtrip_samples": 4})
    kit = BlenderToolkit(
        BlenderRunner("blender", timeout_s=300), config.blender, config.usd, config.naming, directory / "downloads"
    )
    asset_script = directory / "asset.py"
    asset_script.write_text(ASSET)
    asset_blend = directory / "asset" / "asset.blend"
    kit.run_script(asset_script, {"slug": "crate", "output_blend": str(asset_blend)}, directory / "asset.log")
    assets = {"crate": {"blend": str(asset_blend), "collection": "crate", "name": "crate", "instances": 2}}
    layout_script = directory / "layout.py"
    layout_script.write_text(LAYOUT)
    blend = directory / "scene.blend"
    kit.run_script(
        layout_script, {"assets": assets, "asset_mode": request.param, "output_blend": str(blend)}, directory / "layout.log"
    )
    expectations = {
        "assets": assets, "expected_assets": {"crate": 2}, "expected_placeholders": ["skipped", "skipped"],
        "camera_name": "z_selected.camera",
    }
    result = UsdFidelityChecker(kit).check(
        blend, directory / "scene.usda", work_dir=directory / "work", roundtrip_dir=directory / "roundtrip",
        prefix="scene", log_dir=directory / "logs", scene=True, scene_expectations=expectations,
    )
    return kit, blend, expectations, result


def edited_usd(scene_export, tmp_path, *, tags, count, operation="remove"):
    kit, _, _, result = scene_export
    # Keep the layer next to its textures, but never mutate the shared fixture's export.
    usd = result.usd_path.with_name(f"{tmp_path.name}.usda")
    shutil.copyfile(result.usd_path, usd)
    script = tmp_path / "edit_usd.py"
    script.write_text(EDIT_USD)
    kit.run_script(
        script, {"usd_path": str(usd), "tags": tags, "count": count, "operation": operation}, tmp_path / "edit.log"
    )
    return usd


def inspect_usd(scene_export, usd, tmp_path):
    kit, _, expectations, _ = scene_export
    return kit.usd_roundtrip(
        usd, {}, mode="scene", output_dir=tmp_path / "roundtrip", prefix="edited", log_dir=tmp_path / "logs",
        scene_expectations=expectations,
    )


def test_scene_roundtrip_keeps_instances_placeholders_and_active_camera(scene_export):
    _, _, _, result = scene_export
    facts = result.facts()
    for key in (
        "missing_assets", "unexpected_assets", "missing_placeholders", "unexpected_placeholders", "floating_assets",
        "missing_textures", "naming_violations",
    ):
        assert facts[f"usd_{key}"] == 0, result.report()
    assert facts["usd_has_camera"] is True
    scene = result.report()["scene_report"]
    assert len(scene["placed"]["crate"]) == 2
    assert scene["placeholders"] == ["skipped", "skipped"]
    assert scene["camera"]["location"] == [4.0, -6.0, 4.0]
    assert scene["camera"]["lens_mm"] == 45.0
    # The alphabetically first camera looks away: choosing it changes the actual comparison images.
    assert result.score > 0.85


@pytest.mark.parametrize("operation", ["remove", "duplicate"])
def test_imported_instance_counts_detect_usd_loss_and_excess(scene_export, tmp_path, operation):
    usd = edited_usd(
        scene_export, tmp_path, tags={"userProperties:kb_asset_key": "crate", "userProperties:kb_placeholder": "skipped"},
        count=2, operation=operation,
    )
    imported = inspect_usd(scene_export, usd, tmp_path)
    facts = imported["scene_facts"]
    missing, extra, actual = (1, 0, 1) if operation == "remove" else (0, 1, 3)
    assert facts["missing_assets"] == facts["missing_placeholders"] == missing
    assert facts["unexpected_assets"] == facts["unexpected_placeholders"] == extra
    assert len(imported["scene_report"]["placed"]["crate"]) == actual
    assert imported["scene_report"]["placeholders"] == ["skipped"] * actual
    assert imported["scene_report"]["expected_assets"] == {"crate": 2}
    assert imported["scene_report"]["actual_assets"] == {"crate": actual}
    assert imported["scene_report"]["expected_placeholders"] == {"skipped": 2}
    assert imported["scene_report"]["actual_placeholders"] == {"skipped": actual}


def test_usd_without_support_geometry_fails_grounding(scene_export, tmp_path):
    usd = edited_usd(scene_export, tmp_path, tags={"userProperties:blender:object_name": "ground"}, count=1)
    imported = inspect_usd(scene_export, usd, tmp_path)
    assert imported["scene_facts"]["missing_assets"] == 0
    # z=0 is not itself support, and each asset's own multi-mesh hierarchy must not support itself.
    assert imported["scene_facts"]["floating_assets"] == 4


def test_missing_expected_camera_does_not_render_another_camera(scene_export, tmp_path):
    usd = edited_usd(scene_export, tmp_path, tags={"userProperties:kb_camera_name": "z_selected.camera"}, count=1)
    imported = inspect_usd(scene_export, usd, tmp_path)
    assert imported["scene_facts"]["has_camera"] is False
    assert imported["scene_report"]["camera"] is None
    assert imported["images"] == []


def test_missing_source_active_camera_returns_failure_without_source_render(scene_export, tmp_path):
    kit, blend, expectations, _ = scene_export
    script = tmp_path / "unset_camera.py"
    script.write_text("import bpy\nimport kitbash_bpy as kb\nbpy.context.scene.camera = None\nkb.save_scene()\n")
    no_camera_blend = tmp_path / "no_active_camera.blend"
    kit.run_script(script, {"output_blend": str(no_camera_blend)}, tmp_path / "unset_camera.log", blend=blend)
    result = UsdFidelityChecker(kit).check(
        no_camera_blend, tmp_path / "scene.usda", work_dir=tmp_path / "work", roundtrip_dir=tmp_path / "roundtrip",
        prefix="missing_camera", log_dir=tmp_path / "logs", scene=True,
        scene_expectations={**expectations, "camera_name": None},
    )
    assert result.facts()["usd_has_camera"] is False
    assert result.roundtrip["images"] == []
    assert result.comparisons == ()
    assert result.compare_image is None
    assert result.score == 0.0


@pytest.mark.parametrize("textured_world", [False, True], ids=["color-world", "environment-map"])
def test_world_strength_and_color_grade_survive_rendered_roundtrip(tmp_path, textured_world):
    config = load_config(None, {"usd.materialx": "off", "usd.roundtrip_resolution": [128, 128], "usd.roundtrip_samples": 32})
    kit = BlenderToolkit(BlenderRunner("blender", timeout_s=300), config.blender, config.usd, config.naming, tmp_path)
    script = tmp_path / "world.py"
    script.write_text("""import os
import bpy
import kitbash_bpy as kb
kb.reset_scene()
kb.ground_plane(size=200)
bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, location=(0, 0, 1))
kb.assign(bpy.context.object, kb.principled("sphere", base_color=(0.25, 0.5, 0.15), roughness=0.8))
cloud = bpy.context.object
cloud["kb_asset_key"] = "sphere"
cloud.location.z = 1.5
kb.camera((4, -6, 3.5), look_at_point=(0, 0, 1), focal_length_mm=45)
kb.color_world((0.4, 0.6, 0.8), 0.15)
if kb.args()["textured_world"]:
    image = bpy.data.images.new("environment", width=16, height=8, float_buffer=True)
    image.pixels[:] = [0.4, 0.6, 0.8, 1.0] * (16 * 8)
    image.filepath_raw = os.path.join(os.path.dirname(kb.args()["output_blend"]), "environment.exr")
    image.file_format = "OPEN_EXR"
    image.save()
    tree = bpy.context.scene.world.node_tree
    texture = tree.nodes.new("ShaderNodeTexEnvironment")
    texture.image = image
    background = next(n for n in tree.nodes if n.type == "BACKGROUND")
    tree.links.new(texture.outputs["Color"], background.inputs["Color"])
bpy.context.scene.view_settings.view_transform = "AgX"
bpy.context.scene.view_settings.look = "AgX - Punchy"
bpy.context.scene.view_settings.exposure = 0.7
kb.save_scene()
""")
    blend = tmp_path / "scene.blend"
    kit.run_script(script, {"output_blend": str(blend), "textured_world": textured_world}, tmp_path / "build.log")
    result = UsdFidelityChecker(kit).check(
        blend, tmp_path / "scene.usda", work_dir=tmp_path / "work", roundtrip_dir=tmp_path / "roundtrip",
        prefix="scene", log_dir=tmp_path / "logs", scene=True,
        scene_expectations={"assets": {"sphere": {}}, "airborne": {"sphere": 1}},
    )
    assert result.score > 0.96, result.report()
    assert result.facts()["usd_floating_assets"] == 0
    assert result.report()["scene_report"]["intentional_airborne"] == ["Sphere"]
