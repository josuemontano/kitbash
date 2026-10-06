"""Scene acceptance inspects real geometry, placement multiplicities and external dependencies."""

from pathlib import Path

import pytest
from PIL import Image

from kitbash.infra.blender import BlenderRunner, blender_script
from tests.helpers import requires_blender

pytestmark = [pytest.mark.integration, pytest.mark.blender, requires_blender]

PRELUDE = """import bpy
import kitbash_bpy as kb
from inspect_scene import inspect_scene

kb.reset_scene()

def snapshot(name, **overrides):
    options = {**kb.args().get("expectations", {}), **overrides}
    report, facts = inspect_scene(options)
    kb.emit(name, {"report": report, "facts": facts})

def cube(name, location, parent=None):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.parent = parent
    return obj
"""


def run_scene(directory: Path, source: str, **args) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "scene.py"
    script.write_text(PRELUDE + source)
    return BlenderRunner("blender", timeout_s=120).run(script, args=args, log_path=directory / "scene.log")


def library(directory: Path, *, empty_root: bool = False, texture: Path | None = None) -> Path:
    blend = directory / "crate.blend"
    run_scene(directory, """
body = cube("crate", (0, 0, 0))
for vertex in body.data.vertices:
    vertex.co.z += 0.5
home = kb.collection("crate")
for collection in list(body.users_collection):
    collection.objects.unlink(body)
home.objects.link(body)
if kb.args()["empty_root"]:
    root = bpy.data.objects.new("crate", None)
    body.name = "crate_body"
    root.name = "crate"
    home.objects.link(root)
    body.parent = root
if kb.args().get("texture"):
    material = bpy.data.materials.new("crate_material")
    material.use_nodes = True
    group = bpy.data.node_groups.new("nested_texture", "ShaderNodeTree")
    node = group.nodes.new("ShaderNodeTexImage")
    node.image = bpy.data.images.load(kb.args()["texture"])
    material.node_tree.nodes.new("ShaderNodeGroup").node_tree = group
    body.data.materials.append(material)
bpy.ops.wm.save_as_mainfile(filepath=kb.args()["output_blend"])
bpy.ops.file.make_paths_relative()
bpy.ops.wm.save_mainfile()
""", output_blend=str(blend), empty_root=empty_root, texture=str(texture) if texture else None)
    return blend


@pytest.mark.parametrize("mode", ["append", "link"])
@pytest.mark.parametrize("empty_root", [False, True], ids=["mesh-root", "empty-root"])
def test_repeated_assets_count_instances_and_evaluate_transformed_geometry(tmp_path, mode, empty_root):
    blend = library(tmp_path / "library", empty_root=empty_root)
    assets = {"crate": {"blend": str(blend), "collection": "crate", "name": "crate", "instances": 2}}
    result = run_scene(tmp_path / "layout", """
ground = kb.ground_plane(size=20)
a = kb.place_asset("crate", (-1.5, 0, 0))
b = kb.place_asset("crate", (1.5, 0, 0), rotation_deg=(0, 0, 30))
kb.camera((0, -6, 3), look_at_point=(0, 0, 0))
snapshot("exact")
snapshot("missing", expected_assets={"crate": 4})
snapshot("extra", expected_assets={"crate": 1})
b.location.z = 0.4
snapshot("floating")
ground.hide_render = True
snapshot("no_ground")
""", assets=assets, asset_mode=mode, expectations={"assets": assets})

    exact = result["exact"]
    assert exact["report"]["actual_assets"] == {"crate": 2}
    assert exact["report"]["expected_assets"] == {"crate": 2}
    assert exact["facts"]["missing_assets"] == exact["facts"]["unexpected_assets"] == 0
    assert exact["facts"]["floating_assets"] == 0
    assert exact["facts"]["has_camera"] is True
    assert result["missing"]["facts"]["missing_assets"] == 2
    assert result["missing"]["report"]["missing_assets"] == ["crate", "crate"]
    assert result["extra"]["facts"]["unexpected_assets"] == 1
    assert result["extra"]["report"]["unexpected_assets"] == ["crate"]
    floating = result["floating"]
    assert floating["facts"]["floating_assets"] == 1
    assert floating["report"]["floating"][0]["gap_m"] == pytest.approx(0.4, abs=1e-4)
    assert floating["report"]["floating"][0]["reason"] == "support_too_far"
    assert result["no_ground"]["facts"]["floating_assets"] == 2
    assert {item["reason"] for item in result["no_ground"]["report"]["floating"]} == {"no_external_support"}


def test_placeholder_multiplicity_and_grounding_are_independent_of_asset_tags(tmp_path):
    result = run_scene(tmp_path, """
kb.ground_plane(size=20)
kb.place_placeholder("skipped", (1, 1, 1), (-2, 0, 0))
kb.place_placeholder("skipped", (1, 1, 1), (0, 0, 0))
kb.place_placeholder("extra", (1, 1, 1), (2, 0, 0.5))
snapshot("inspection")
""", expectations={"expected_placeholders": ["skipped", "missing", "missing"]})
    report, facts = result["inspection"]["report"], result["inspection"]["facts"]
    assert report["actual_placeholders"] == {"skipped": 2, "extra": 1}
    assert report["missing_placeholders"] == ["missing", "missing"]
    assert report["unexpected_placeholders"] == ["extra", "skipped"]
    assert facts["missing_placeholders"] == facts["unexpected_placeholders"] == 2
    assert facts["floating_assets"] == 1
    assert report["floating"][0]["object"] == "extra_placeholder"
    assert report["floating"][0]["gap_m"] == pytest.approx(0.5, abs=1e-4)


def test_empty_invalid_and_self_supporting_hierarchies_fail_integrity(tmp_path):
    result = run_scene(tmp_path, """
empty = bpy.data.objects.new("empty", None)
bpy.context.scene.collection.objects.link(empty)
empty["kb_asset_key"] = "empty"
invalid_mesh = bpy.data.meshes.new("invalid_mesh")
invalid = bpy.data.objects.new("invalid", invalid_mesh)
bpy.context.scene.collection.objects.link(invalid)
invalid["kb_asset_key"] = "invalid"
root = bpy.data.objects.new("self_support", None)
bpy.context.scene.collection.objects.link(root)
root["kb_asset_key"] = "self_support"
# The lower child is not an external support for the upper child or its EMPTY root.
cube("lower", (0, 0, 0.5), root)
cube("upper", (0, 0, 1.5), root)
snapshot("unsupported")
kb.ground_plane(size=10)
snapshot("grounded")
""", expectations={"expected_assets": {"empty": 1, "invalid": 1, "self_support": 1}})
    unsupported = result["unsupported"]
    assert unsupported["facts"]["missing_assets"] == 0
    assert unsupported["facts"]["floating_assets"] == 3
    assert {item["object"]: item["reason"] for item in unsupported["report"]["floating"]} == {
        "empty": "no_inspectable_geometry", "invalid": "uninspectable_geometry", "self_support": "no_external_support",
    }
    assert result["grounded"]["facts"]["floating_assets"] == 2
    assert {item["object"] for item in result["grounded"]["report"]["floating"]} == {"empty", "invalid"}


def test_one_linked_placement_can_support_another_but_not_itself(tmp_path):
    blend = library(tmp_path / "library", empty_root=True)
    assets = {"crate": {"blend": str(blend), "collection": "crate", "name": "crate", "instances": 2}}
    result = run_scene(tmp_path / "layout", """
kb.ground_plane(size=10)
kb.place_asset("crate", (0, 0, 0))
upper = kb.place_asset("crate", (0, 0, 1))
snapshot("stacked")
upper.location.x = 3
snapshot("separate")
""", assets=assets, asset_mode="link", expectations={"assets": assets})
    assert result["stacked"]["facts"]["floating_assets"] == 0
    assert result["separate"]["facts"]["floating_assets"] == 1
    assert result["separate"]["report"]["floating"][0]["gap_m"] == pytest.approx(1.0)


def test_support_under_an_off_center_base_is_not_missed(tmp_path):
    result = run_scene(tmp_path, """
obj = cube("crate", (0, 0, 0.7))
obj["kb_asset_key"] = "crate"
pedestal = cube("pedestal", (0.25, 0, 0.1))
pedestal.scale = (0.1, 0.1, 0.2)
snapshot("supported")
pedestal.location.x = 2
snapshot("unsupported")
""", expectations={"assets": {"crate": {}}})
    assert result["supported"]["facts"]["floating_assets"] == 0
    assert result["unsupported"]["facts"]["floating_assets"] == 1


def test_no_implicit_ground_plane_and_camera_must_be_active_in_the_scene(tmp_path):
    result = run_scene(tmp_path, """
obj = cube("crate", (0, 0, 0.5))
obj["kb_asset_key"] = "crate"
snapshot("empty_space")
cam = kb.camera((0, -5, 3), look_at_point=(0, 0, 0), name="camera.001")
snapshot("camera")
snapshot("wrong_camera", camera_name="other")
cam["kb_camera_name"] = "camera.original"
snapshot("renamed_camera", camera_name="camera.original")
for collection in list(cam.users_collection):
    collection.objects.unlink(cam)
bpy.context.scene.camera = cam
snapshot("unlinked_camera")
""", expectations={"assets": {"crate": {}}})
    assert result["empty_space"]["facts"]["floating_assets"] == 1
    assert result["empty_space"]["report"]["floating"][0]["gap_m"] is None
    assert result["empty_space"]["facts"]["has_camera"] is False
    assert result["camera"]["facts"]["has_camera"] is True
    assert result["wrong_camera"]["facts"]["has_camera"] is False
    assert result["renamed_camera"]["facts"]["has_camera"] is True
    assert result["unlinked_camera"]["facts"]["has_camera"] is False
    assert result["unlinked_camera"]["report"]["camera"] is None


def test_linked_nested_material_and_world_texture_paths_are_checked(tmp_path):
    source = tmp_path / "library"
    source.mkdir()
    texture = source / "present.png"
    Image.new("RGB", (2, 2), (120, 90, 70)).save(texture)
    blend = library(source, texture=texture)
    assets = {"crate": {"blend": str(blend), "collection": "crate", "name": "crate"}}
    result = run_scene(tmp_path / "layout", """
import os
kb.place_asset("crate")
kb.ground_plane(size=10)
snapshot("linked_present")
os.unlink(kb.args()["texture"])
world = bpy.data.worlds.new("test_world")
world.use_nodes = True
bpy.context.scene.world = world
node = world.node_tree.nodes.new("ShaderNodeTexEnvironment")
node.image = bpy.data.images.new("missing_world", width=2, height=2)
node.image.source = "FILE"
node.image.filepath = kb.args()["missing_world"]
snapshot("missing")
""", assets=assets, asset_mode="link", texture=str(texture), missing_world=str(tmp_path / "missing.hdr"),
        expectations={"assets": assets})
    assert result["linked_present"]["facts"]["missing_textures"] == 0
    assert result["missing"]["facts"]["missing_textures"] == 2
    assert set(result["missing"]["report"]["missing_textures"]) == {str(texture), str(tmp_path / "missing.hdr")}
    assert result["missing"]["report"]["environment_images"][0]["exists"] is False


def test_fixed_inspection_script_preserves_report_and_fact_emission(tmp_path):
    blend = tmp_path / "scene.blend"
    run_scene(tmp_path / "build", """
kb.ground_plane(size=10)
obj = cube("crate", (0, 0, 0.5))
obj["kb_asset_key"] = "crate"
kb.camera((0, -5, 3), look_at_point=(0, 0, 0))
bpy.ops.wm.save_as_mainfile(filepath=kb.args()["output_blend"])
""", output_blend=str(blend))
    result = BlenderRunner("blender", timeout_s=120).run(
        blender_script("inspect_scene.py"), args={"expected_assets": {"crate": 2}}, blend=blend,
        log_path=tmp_path / "inspect.log",
    )
    assert result["facts"]["missing_assets"] == 1
    assert result["facts"]["floating_assets"] == 0
    assert result["report"]["camera"]["name"] == "camera"
