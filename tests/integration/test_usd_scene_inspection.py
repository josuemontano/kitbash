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


ORTHOGRAPHIC_SCENE = """import bpy
import kitbash_bpy as kb
kb.reset_scene()
scene = bpy.context.scene
scene.unit_settings.scale_length = 0.25
scene.render.pixel_aspect_x, scene.render.pixel_aspect_y = kb.args()["pixel_aspect"]
kb.ground_plane(size=30)
for index, (position, color) in enumerate([
    ((-1.3, -0.5, 0.6), (0.8, 0.1, 0.05)),
    ((1.0, 0.4, 0.9), (0.1, 0.6, 0.2)),
    ((-0.3, 1.5, 0.4), (0.1, 0.2, 0.8)),
]):
    bpy.ops.mesh.primitive_cube_add(size=1, location=position)
    obj = bpy.context.object
    obj.scale = (0.7 + index * 0.2, 0.6, position[2] * 2)
    material = kb.principled(f"marker_{index}", base_color=color, roughness=0.6)
    kb.assign(obj, material)
    if index == 0:
        # Force the bake path that temporarily changes resolution to 64 square.
        noise = material.node_tree.nodes.new("ShaderNodeTexNoise")
        bsdf = next(n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        material.node_tree.links.new(noise.outputs["Fac"], bsdf.inputs["Roughness"])
kb.camera((0, -5, 3), look_at_point=(0, -10, 8), name="a_decoy")
camera = kb.camera((5, -7, 5), look_at_point=(0, 0, 0.8), name="z_selected.ortho")
camera.data.type = "ORTHO"
camera.data.ortho_scale = 7
camera.data.shift_x = 0.13
camera.data.shift_y = -0.09
camera.data.clip_start = 0.3
camera.data.clip_end = 85
parent = bpy.data.objects.new("camera_parent", None)
scene.collection.objects.link(parent)
parent.location = (2.3, -1.1, 0.7)
parent.rotation_euler = (0.1, -0.2, 0.35)
bpy.context.view_layer.update()
world = camera.matrix_world.copy()
camera.parent = parent
camera.matrix_world = world
kb.light("SUN", (0, 0, 6), energy=2, rotation_deg=(25, -15, 30))
kb.color_world((0.5, 0.5, 0.5), 0.4)
kb.save_scene()
"""

CHECK_ORTHOGRAPHIC_USD = """import bpy
import kb_usd_cameras
import kitbash_bpy as kb
from pxr import Usd, UsdGeom
scene = bpy.context.scene
source = scene.camera
name = source.name
frame = sorted(tuple(v) for v in source.data.view_frame(scene=scene))
matrix = source.matrix_world.copy()
render = {key: getattr(scene.render, key) for key in ("resolution_x", "resolution_y", "pixel_aspect_x", "pixel_aspect_y")}
stage = Usd.Stage.Open(kb.args()["usd_path"])
cameras = [UsdGeom.Camera(p) for p in stage.Traverse() if p.IsA(UsdGeom.Camera)]
assert len(cameras) == 2, [str(c.GetPath()) for c in cameras]
camera = next(c for c in cameras if c.GetProjectionAttr().Get() == "orthographic")
unit = UsdGeom.GetStageMetersPerUnit(stage)
xs, ys = [v[0] for v in frame], [v[1] for v in frame]
assert abs(camera.GetHorizontalApertureAttr().Get() * unit / 10 - (max(xs) - min(xs))) < 1e-5
assert abs(camera.GetVerticalApertureAttr().Get() * unit / 10 - (max(ys) - min(ys))) < 1e-5
assert abs(camera.GetHorizontalApertureOffsetAttr().Get() * unit / 10 - (max(xs) + min(xs)) / 2) < 1e-5
assert abs(camera.GetVerticalApertureOffsetAttr().Get() * unit / 10 - (max(ys) + min(ys)) / 2) < 1e-5
assert all(abs(a * unit - b) < 1e-5 for a, b in zip(camera.GetClippingRangeAttr().Get(), (0.3, 85)))
assert stage.GetRootLayer().customLayerData["kitbash"]["render_settings"] == render
# Calling authoring again simulates an exporter that already supports the camera.
kb_usd_cameras.author_orthographic_cameras(stage, scene)
assert len([p for p in stage.Traverse() if p.IsA(UsdGeom.Camera)]) == 2
kb.reset_scene()
bpy.ops.wm.usd_import(filepath=kb.args()["usd_path"], import_cameras=True, property_import_mode="USER", merge_parent_xform=True)
scene = bpy.context.scene
imported = [o for o in scene.objects if o.type == "CAMERA"]
assert len(imported) == 2
camera = next(o for o in imported if o.get("kb_camera_name") == name)
assert camera.data.type == "ORTHO" and camera.get("kb_active_camera")
assert not next(o for o in imported if o != camera).get("kb_active_camera")
# This only fixes Blender's known aperture-unit bug, never creates a camera.
kb_usd_cameras.normalize_imported_orthographic_scale(stage, scene)
for key, value in render.items():
    setattr(scene.render, key, value)
bpy.context.view_layer.update()
assert max(abs(camera.matrix_world[r][c] - matrix[r][c]) for r in range(4) for c in range(4)) < 1e-5
actual = sorted(tuple(v) for v in camera.data.view_frame(scene=scene))
# Frame Z is an arbitrary display depth for orthographic cameras; XY are projection bounds.
assert max(abs(a[i] - b[i]) for a, b in zip(actual, frame) for i in (0, 1)) < 1e-5
assert abs(camera.data.clip_start - 0.3) < 1e-5 and abs(camera.data.clip_end - 85) < 1e-5
"""


@pytest.mark.parametrize(
    ("resolution", "pixel_aspect"),
    [([240, 120], [1.25, 1.0]), ([120, 240], [1.0, 1.5])],
    ids=["landscape-shifted", "portrait-shifted"],
)
def test_orthographic_camera_standard_usd_and_rendered_framing(tmp_path, resolution, pixel_aspect):
    config = load_config(None, {
        "blender.final_resolution": resolution,
        "usd.materialx": "off", "usd.bake_resolution": 32,
        "usd.roundtrip_resolution": [128, 128], "usd.roundtrip_samples": 16,
    })
    kit = BlenderToolkit(BlenderRunner("blender", timeout_s=300), config.blender, config.usd, config.naming, tmp_path)
    script = tmp_path / "orthographic.py"
    script.write_text(ORTHOGRAPHIC_SCENE)
    blend = tmp_path / "scene.blend"
    kit.run_script(script, {"output_blend": str(blend), "pixel_aspect": pixel_aspect}, tmp_path / "build.log")
    result = UsdFidelityChecker(kit).check(
        blend, tmp_path / "scene.usda", work_dir=tmp_path / "work", roundtrip_dir=tmp_path / "roundtrip",
        prefix="scene", log_dir=tmp_path / "logs", scene=True,
        scene_expectations={"camera_name": "z_selected.ortho"},
    )
    assert result.facts()["usd_has_camera"] is True
    assert result.score > 0.85, result.report()
    assert result.roundtrip["render_resolution"] == ([128, 64] if resolution[0] > resolution[1] else [64, 128])
    assert any(info["baked"] for info in result.export["materials"].values())
    script = tmp_path / "inspect_camera.py"
    script.write_text(CHECK_ORTHOGRAPHIC_USD)
    kit.run_script(script, {"usd_path": str(result.usd_path)}, tmp_path / "inspect.log", blend=blend)
