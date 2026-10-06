"""USD material fidelity ladder and self-contained exports, with real headless Blender."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from kitbash.backlot.library import AssetBundle, Backlot, BacklotDraft
from kitbash.config import load_config
from kitbash.infra.blender import BlenderRunner
from kitbash.infra.embeddings import HashingEmbedder
from kitbash.services.artifacts import file_hash, snapshot
from kitbash.services.blender_toolkit import BlenderToolkit
from kitbash.services.usd_fidelity import UsdFidelityChecker
from tests.fakes.trellis.generate import cylinder_obj
from tests.helpers import requires_blender

pytestmark = [pytest.mark.integration, pytest.mark.blender, requires_blender]

PROCEDURAL_NOISE = """
mat = kb.principled("body", base_color=(0.5, 0.3, 0.2), roughness=0.5)
bsdf = kb.principled_node(mat)
noise = kb.add_node(mat, "ShaderNodeTexNoise")
kb.link(mat, noise.outputs["Color"], bsdf.inputs["Base Color"])
"""
PROCEDURAL_VORONOI_MIX = """
mat = kb.principled("body", base_color=(0.5, 0.3, 0.2), roughness=0.5)
bsdf = kb.principled_node(mat)
mix = kb.add_node(mat, "ShaderNodeMix", data_type="RGBA")
mix.inputs["A"].default_value = (0.9, 0.1, 0.1, 1.0)
mix.inputs["B"].default_value = (0.1, 0.1, 0.9, 1.0)
voronoi = kb.add_node(mat, "ShaderNodeTexVoronoi")
kb.link(mat, voronoi.outputs["Distance"], mix.inputs["Factor"])
kb.link(mat, mix.outputs["Result"], bsdf.inputs["Base Color"])
kb.link(mat, voronoi.outputs["Distance"], bsdf.inputs["Roughness"])
"""
IMAGE_TEXTURE = """
kb.ensure_uv(obj)
mat = kb.principled("label", roughness=0.4)
image = kb.image_texture(mat, kb.args()["texture"], projection="UV", part="label")
kb.link(mat, image.outputs["Color"], kb.principled_node(mat).inputs["Base Color"])
"""


def asset_script(materials: str) -> str:
    return (
        "import kitbash_bpy as kb\n\nkb.reset_scene()\nobj = kb.import_mesh()\nkb.clean_mesh(obj)\n"
        "kb.fit_dimensions(obj, mode='height')\nkb.origin_to_base(obj)\n"
        + materials
        + "\nkb.assign(obj, mat)\nkb.save_asset(obj)\n"
    )


@pytest.fixture
def toolkit(tmp_path):
    def make(**usd) -> BlenderToolkit:
        config = load_config(
            None,
            {
                "blender.preview_views": ["front_3q"],
                "usd.roundtrip_resolution": [96, 96],
                "usd.roundtrip_samples": 4,
                "usd.bake_resolution": 128,
                "usd.bake_samples": 2,
                **{f"usd.{k}": v for k, v in usd.items()},
            },
        )
        return BlenderToolkit(BlenderRunner("blender", timeout_s=300), config.blender, config.usd, config.naming, tmp_path / "downloads")

    return make


def build_asset(toolkit: BlenderToolkit, tmp: Path, name: str, materials: str, **args) -> Path:
    mesh = tmp / f"{name}.obj"
    mesh.write_text(cylinder_obj())
    script = tmp / f"{name}_script.py"
    script.write_text(asset_script(materials))
    build = tmp / name / "build"
    toolkit.run_script(
        script,
        {"mesh_path": str(mesh), "output_blend": str(build / "asset.blend"), "textures_dir": str(build / "textures"),
         "slug": name, "dimensions_m": [0.3, 0.3, 0.4], **args},
        tmp / name / "build.log",
    )
    return build / "asset.blend"


def check(toolkit: BlenderToolkit, blend: Path, tmp: Path):
    usd = blend.parent / "usd" / "asset.usd"
    return UsdFidelityChecker(toolkit).check(
        blend, usd, work_dir=tmp / "work", roundtrip_dir=tmp / "roundtrip", prefix="t", log_dir=tmp / "logs"
    )


def usd_text(path: Path) -> str:
    """ASCII dump of a (possibly binary) USD layer, via Blender's bundled pxr."""
    script = path.with_suffix(".dump.py")
    script.write_text(
        "import sys\nfrom pxr import Sdf\nprint('<<USDA>>' + Sdf.Layer.FindOrOpen(sys.argv[-1]).ExportToString())\n"
    )
    result = subprocess.run(
        ["blender", "-b", "--factory-startup", "--python", str(script), "--", str(path)], capture_output=True, text=True, timeout=120
    )
    return result.stdout.split("<<USDA>>", 1)[1]


def test_materialx_network_is_used_when_it_keeps_every_link(toolkit, tmp_path):
    kit = toolkit()
    result = check(kit, build_asset(kit, tmp_path, "noisy", PROCEDURAL_NOISE), tmp_path)
    material = result.export["materials"]["mat_noisy_body"]
    assert result.export["materialx_supported"] is True
    assert material["rung"] == "materialx" and material["materialx_lossless"] is True
    assert result.mode == "materialx"
    text = usd_text(result.usd_path)
    assert "outputs:mtlx:surface" in text and "ND_open_pbr_surface_surfaceshader" in text
    # The preview fallback is baked too, so UsdPreviewSurface readers (Blender's importer) match the .blend.
    assert set(material["baked"]) == {"Base Color"}
    assert result.facts()["usd_broken_materials"] == 0 and result.facts()["usd_missing_textures"] == 0
    assert result.score > 0.85


def test_unsupported_nodes_fall_back_to_a_baked_preview_surface(toolkit, tmp_path):
    kit = toolkit()
    blend = build_asset(kit, tmp_path, "voronoi", PROCEDURAL_VORONOI_MIX)
    before = blend.read_bytes()
    result = check(kit, blend, tmp_path)
    material = result.export["materials"]["mat_voronoi_body"]
    assert material["rung"] == "preview_surface_baked" and material["materialx_lossless"] is False
    assert set(material["baked"]) == {"Base Color", "Roughness"}
    for relative in material["baked"].values():
        assert not Path(relative).is_absolute() and (result.usd_path.parent / relative).is_file()
    text = usd_text(result.usd_path)
    assert "UsdUVTexture" in text
    assert "outputs:mtlx:surface" not in text  # the lossy MaterialX network is not offered
    assert result.roundtrip["materials"]["mat_voronoi_body"]["missing_channels"] == []
    assert result.score > 0.8
    assert blend.read_bytes() == before  # baking happened in a temporary copy


def test_materialx_can_be_disabled(toolkit, tmp_path):
    kit = toolkit(materialx="off")
    result = check(kit, build_asset(kit, tmp_path, "noisy", PROCEDURAL_NOISE), tmp_path)
    assert result.export["materialx_supported"] is False
    assert result.export["materials"]["mat_noisy_body"]["rung"] == "preview_surface_baked"
    assert "outputs:mtlx:surface" not in usd_text(result.usd_path)


LAYOUT = """import kitbash_bpy as kb
kb.reset_scene()
kb.ground_plane(size=4.0)
kb.place_asset("labelled_can", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
kb.camera((0.0, -2.0, 0.6), look_at_point=(0.0, 0.0, 0.2), focal_length_mm=50)
kb.light("SUN", (0.0, 0.0, 5.0), energy=3.0, rotation_deg=(40.0, 0.0, 30.0))
kb.color_world((0.5, 0.5, 0.5), 1.0)
kb.save_scene()
"""

CHECK_IMAGES = """import bpy, json, os
print("<<IMAGES>>" + json.dumps([
    {"path": i.filepath, "exists": os.path.isfile(bpy.path.abspath(i.filepath, library=i.library))}
    for i in bpy.data.images if i.source == "FILE"
]))
"""


def test_texture_paths_stay_relative_after_copying_into_a_scene(toolkit, tmp_path):
    kit = toolkit()
    texture = tmp_path / "label.png"
    Image.new("RGB", (64, 64), (200, 30, 30)).save(texture)
    blend = build_asset(kit, tmp_path, "labelled_can", IMAGE_TEXTURE, texture=str(texture))
    asset_usd = check(kit, blend, tmp_path / "asset_check")
    assert asset_usd.export["materials"]["mat_labelled_can_label"]["baked"] == {}  # plain image textures need no baking

    backlot = Backlot(tmp_path / "backlot", HashingEmbedder(32))
    entry = backlot.add(
        BacklotDraft("Labelled can", "Tin can with a red label", "prop", (0.3, 0.3, 0.4), "photorealistic", asset_usd.mode, asset_usd.score),
        AssetBundle(
            root=blend.parent, blend=blend, usd=asset_usd.usd_path, preview=texture,
            hashes=snapshot(blend.parent, (blend.parent,)), preview_hash=file_hash(texture),
        ),
    )
    texture.unlink()  # the source is gone: only the copies may be referenced now

    scene_dir = tmp_path / "scene"
    copy = scene_dir / "assets" / "labelled_can"
    shutil.copytree(entry.directory, copy)
    layout = tmp_path / "layout.py"
    layout.write_text(LAYOUT)
    scene_blend = scene_dir / "scene.blend"
    assets = {"labelled_can": {"blend": str(copy / "asset.blend"), "collection": "labelled_can", "name": "labelled_can"}}
    kit.run_script(layout, {"output_blend": str(scene_blend), "assets": assets, "asset_mode": "append"}, tmp_path / "logs" / "scene.log")
    kit.localize(scene_blend, tmp_path / "logs")

    probe = tmp_path / "check_images.py"
    probe.write_text(CHECK_IMAGES)
    output = subprocess.run(["blender", "-b", str(scene_blend), "--python", str(probe)], capture_output=True, text=True, timeout=120).stdout
    images = json.loads(output.split("<<IMAGES>>", 1)[1].splitlines()[0])
    label, = images
    assert label["path"].startswith("//") and label["exists"]

    scene_usd = UsdFidelityChecker(kit).check(
        scene_blend, scene_dir / "scene.usd", work_dir=tmp_path / "scene_work", roundtrip_dir=tmp_path / "scene_rt",
        prefix="scene", log_dir=tmp_path / "logs", scene=True,
    )
    assert scene_usd.export["missing_textures"] == [] and scene_usd.export["absolute_texture_paths"] == []
    assert all(not Path(t).is_absolute() for t in scene_usd.export["textures"])
    assert scene_usd.roundtrip["missing_textures"] == []
    assert scene_usd.score > 0.8
    backlot.close()


LINKED_LAYOUT = """import kitbash_bpy as kb
kb.reset_scene()
kb.ground_plane(size=4.0)
kb.place_asset("voronoi", (0.0, 0.0, 0.0))
kb.place_asset("voronoi", (0.6, 0.0, 0.0), (0.0, 0.0, 45.0))
kb.camera((0.0, -2.0, 0.6), look_at_point=(0.3, 0.0, 0.2), focal_length_mm=40)
kb.light("SUN", (0.0, 0.0, 5.0), energy=3.0, rotation_deg=(40.0, 0.0, 30.0))
kb.color_world((0.5, 0.5, 0.5), 1.0)
kb.save_scene()
"""


def test_linked_assets_are_baked_in_the_scene_export(toolkit, tmp_path):
    """assembly.mode = "link": collection instances of a library asset still get their procedural
    materials baked for the scene USD."""
    kit = toolkit()
    blend = build_asset(kit, tmp_path, "voronoi", PROCEDURAL_VORONOI_MIX)
    scene_dir = tmp_path / "scene"
    copy = scene_dir / "assets" / "voronoi"
    shutil.copytree(blend.parent, copy)
    layout = tmp_path / "linked_layout.py"
    layout.write_text(LINKED_LAYOUT)
    scene_blend = scene_dir / "scene.blend"
    assets = {"voronoi": {"blend": str(copy / "asset.blend"), "collection": "voronoi", "name": "voronoi"}}
    kit.run_script(layout, {"output_blend": str(scene_blend), "assets": assets, "asset_mode": "link"}, tmp_path / "logs" / "linked.log")
    before = scene_blend.read_bytes()
    result = UsdFidelityChecker(kit).check(
        scene_blend, scene_dir / "scene.usd", work_dir=tmp_path / "work", roundtrip_dir=tmp_path / "rt",
        prefix="scene", log_dir=tmp_path / "logs", scene=True,
    )
    material = result.export["materials"]["mat_voronoi_body"]
    assert set(material["baked"]) == {"Base Color", "Roughness"}
    assert result.roundtrip["materials"][material["usd_prim"]]["missing_channels"] == []
    assert result.facts()["usd_broken_materials"] == 0 and result.score > 0.8
    assert scene_blend.read_bytes() == before  # the scene keeps its library links


SHARED_WITH_CHILD = PROCEDURAL_VORONOI_MIX + """
child = obj.copy()
child.data = obj.data.copy()
child.data.name = "Cylinder.001"
child.name = "Cylinder.001"
kb.asset_collection().objects.link(child)
child.parent = obj
child.location = (0.4, 0.0, 0.0)
kb.assign(child, mat)
"""


def test_materials_shared_by_several_meshes_round_trip(toolkit, tmp_path):
    kit = toolkit()
    result = check(kit, build_asset(kit, tmp_path, "pair", SHARED_WITH_CHILD), tmp_path)
    names = set(result.export["materials"])
    assert len(names) == 2 and all("." not in n for n in names)  # one baked copy per mesh, no dotted names
    assert result.roundtrip["materials_ok"] == result.roundtrip["materials_total"] == 2
    assert result.facts()["usd_broken_materials"] == 0
