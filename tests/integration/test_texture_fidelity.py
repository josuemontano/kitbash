"""Texture identity and preview-loss regressions exercised in real Blender."""

import hashlib
import shutil
from pathlib import Path

import pytest
from PIL import Image

from kitbash.errors import KitbashError
from kitbash.services.usd_fidelity import UsdFidelityChecker
from tests.helpers import requires_blender
from tests.integration.test_usd_export import build_asset, check
from tests.integration.test_usd_export import toolkit as toolkit

pytestmark = [pytest.mark.integration, pytest.mark.blender, requires_blender]

RAW_IMAGE = """
import bpy
kb.ensure_uv(obj)
mat = kb.principled("label", roughness=0.4)
image = kb.add_node(mat, "ShaderNodeTexImage")
image.image = bpy.data.images.load(kb.args()["texture"])
kb.link(mat, image.outputs["Color"], kb.principled_node(mat).inputs["Base Color"])
"""

IMAGE_REPORT = """import hashlib
import bpy
import kb_materials as km
import kitbash_bpy as kb

materials = {}
for material in bpy.data.materials:
    if not material.users:
        continue
    images = []
    for image in km.images_of(material):
        info = km.image_info(image)
        path = bpy.path.abspath(image.filepath, library=image.library)
        with open(path, "rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        images.append({**info, "digest": digest, "linked": image.library is not None})
    if images:
        materials[material.name] = images
kb.emit("materials", materials)
"""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_report(kit, blend, tmp):
    script = tmp / "image_report.py"
    script.write_text(IMAGE_REPORT)
    return kit.run_script(script, {}, tmp / "image_report.log", blend=blend)["materials"]


@pytest.mark.parametrize("boundary", ["asset", "scene"])
def test_same_basename_textures_keep_their_content_after_copy(toolkit, tmp_path, boundary):
    kit = toolkit(materialx="off")
    paths = []
    for name, color in [("red", (200, 30, 30)), ("blue", (30, 30, 200))]:
        path = tmp_path / "sources" / name / "albedo.png"
        path.parent.mkdir(parents=True)
        Image.new("RGB", (32, 32), color).save(path)
        paths.append(path)
    expected = {digest(path) for path in paths}
    materials = """
import bpy
kb.ensure_uv(obj)
for index, path in enumerate(kb.args()["textures"]):
    mat = kb.principled("red" if index == 0 else "blue", roughness=0.4)
    if kb.args()["boundary"] == "asset":
        image = kb.image_texture(mat, path, projection="UV")
    else:
        image = kb.add_node(mat, "ShaderNodeTexImage")
        image.image = bpy.data.images.load(path)
    kb.link(mat, image.outputs["Color"], kb.principled_node(mat).inputs["Base Color"])
    if index == 0:
        child = obj.copy()
        child.data = obj.data.copy()
        kb.asset_collection().objects.link(child)
        child.location.x = 0.4
        kb.assign(child, mat)
bpy.context.view_layer.update()
"""
    blend = build_asset(kit, tmp_path, "pair", materials, textures=[str(p) for p in paths], boundary=boundary)
    if boundary == "scene":
        assert kit.localize(blend, tmp_path / "logs")["missing"] == []
    shutil.rmtree(tmp_path / "sources")
    before = image_report(kit, blend, tmp_path)
    images = [image for material in before.values() for image in material]
    assert {image["digest"] for image in images} == expected
    assert all(image["exists"] and image["relative"] for image in images)
    assert len({image["filepath"] for image in images}) == 2
    assert kit.localize(blend, tmp_path / "logs")["missing"] == []
    assert image_report(kit, blend, tmp_path) == before
    result = check(kit, blend, tmp_path / "check")
    assert {digest(result.usd_path.parent / path) for path in result.export["textures"]} == expected
    assert result.facts()["usd_missing_textures"] == 0
    assert result.facts()["usd_broken_materials"] == 0
    assert result.score > 0.85


@pytest.mark.parametrize("mode", ["append", "link"])
def test_copied_libraries_with_same_basename_textures_survive_scene_relocation(toolkit, tmp_path, mode):
    kit = toolkit(materialx="off")
    scene_dir = tmp_path / "scene"
    assets, expected = {}, {}
    for name, color in [("red", (200, 30, 30)), ("blue", (30, 30, 200))]:
        # Existing libraries need not have been authored with content-addressed names.
        texture = tmp_path / name / "build" / "textures" / "albedo.png"
        texture.parent.mkdir(parents=True)
        Image.new("RGB", (32, 32), color).save(texture)
        expected[f"mat_{name}_label"] = digest(texture)
        blend = build_asset(kit, tmp_path, name, RAW_IMAGE, texture=str(texture))
        copied = scene_dir / "assets" / name
        shutil.copytree(blend.parent, copied)
        assets[name] = {"blend": str(copied / "asset.blend"), "collection": name, "name": name}
        shutil.rmtree(blend.parent)
    layout = tmp_path / "layout.py"
    layout.write_text("""import kitbash_bpy as kb
kb.reset_scene()
kb.ground_plane(size=4.0)
kb.place_asset("red", (-0.25, 0.0, 0.0))
kb.place_asset("blue", (0.25, 0.0, 0.0))
kb.camera((0.0, -2.0, 0.7), look_at_point=(0.0, 0.0, 0.2), focal_length_mm=40)
kb.light("SUN", (0.0, 0.0, 5.0), energy=3.0, rotation_deg=(40.0, 0.0, 30.0))
kb.color_world((0.5, 0.5, 0.5), 1.0)
kb.save_scene()
""")
    scene_blend = scene_dir / "scene.blend"
    kit.run_script(layout, {"output_blend": str(scene_blend), "assets": assets, "asset_mode": mode}, tmp_path / "layout.log")
    localized = kit.localize(scene_blend, tmp_path / "logs")
    assert localized["missing"] == []
    relocated = tmp_path / "relocated"
    shutil.move(scene_dir, relocated)
    scene_blend = relocated / "scene.blend"
    before = scene_blend.read_bytes()
    report = image_report(kit, scene_blend, tmp_path)
    for name, content in expected.items():
        image, = report[name]
        assert image["digest"] == content
        assert image["exists"] and image["relative"]
        assert image["linked"] == (mode == "link")
    result = UsdFidelityChecker(kit).check(
        scene_blend, relocated / "scene.usd", work_dir=tmp_path / "work", roundtrip_dir=tmp_path / "rt",
        prefix="scene", log_dir=tmp_path / "logs", scene=True,
    )
    assert {digest(result.usd_path.parent / path) for path in result.export["textures"]} == set(expected.values())
    assert result.facts()["usd_missing_textures"] == 0
    assert result.facts()["usd_absolute_texture_paths"] == 0
    assert result.facts()["usd_broken_materials"] == 0
    assert result.score > 0.85
    assert scene_blend.read_bytes() == before


@pytest.mark.parametrize("source", ["procedural", "direct_image"])
def test_known_preview_channel_loss_fails_fidelity_check(toolkit, tmp_path, source):
    kit = toolkit(materialx="off")
    texture = tmp_path / "transmission.png"
    Image.new("RGB", (32, 32), (200, 200, 200)).save(texture)
    materials = """
kb.ensure_uv(obj)
mat = kb.principled("glass")
if kb.args()["source"] == "direct_image":
    node = kb.image_texture(mat, kb.args()["texture"], projection="UV")
else:
    node = kb.add_node(mat, "ShaderNodeTexNoise")
kb.link(mat, node.outputs["Color"], kb.principled_node(mat).inputs["Transmission Weight"])
"""
    blend = build_asset(kit, tmp_path, "lossy", materials, texture=str(texture), source=source)
    with pytest.raises(KitbashError, match=r"mat_lossy_glass.*Transmission Weight"):
        check(kit, blend, tmp_path / "check")
    assert not (tmp_path / "check" / "roundtrip").exists()
