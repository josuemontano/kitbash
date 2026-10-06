"""Malformed procedural faces fail before publication, not after lossy USD import."""

import pytest

from kitbash.config import load_config
from kitbash.errors import BlenderScriptError
from kitbash.infra.blender import BlenderRunner
from kitbash.services.blender_toolkit import BlenderToolkit
from tests.helpers import requires_blender

pytestmark = [pytest.mark.integration, pytest.mark.blender, requires_blender]

RING = """import math
import bpy
import kitbash_bpy as kb

kb.reset_scene()
options = kb.args()
sides, rows = 4, 3
vertices = [(math.cos(j * 2 * math.pi / sides), math.sin(j * 2 * math.pi / sides), row)
            for row in range(rows) for j in range(sides)]
# A second material component makes index drift consumer-visible after bad faces are removed.
vertices += [(x, y, z) for z in (2.5, 3.5) for y in (-0.5, 0.5) for x in (-0.5, 0.5)]
faces = [tuple(reversed(range(sides)))]
for row in range(rows - 1):
    for j in range(sides):
        nxt = (j + 1) % sides
        a, b = row * sides + j, (row + 1) * sides + j
        # Exact round_tree_01 failure: b already includes j. At the seam b+nxt repeats b.
        upper_next = b + nxt if options.get("malformed") else (row + 1) * sides + nxt
        faces.append((a, row * sides + nxt, upper_next, b))
faces.append(tuple((rows - 1) * sides + j for j in range(sides)))
faces += [(12, 14, 15, 13), (16, 17, 19, 18), (12, 13, 17, 16),
          (14, 18, 19, 15), (12, 16, 18, 14), (13, 15, 19, 17)]
mesh = bpy.data.meshes.new("ring_mesh")
mesh.from_pydata(vertices, [], faces)
obj = bpy.data.objects.new("ring", mesh)
bpy.context.scene.collection.objects.link(obj)
mesh.materials.append(kb.principled("wood"))
mesh.materials.append(kb.principled("foliage", base_color=(0.1, 0.5, 0.2)))
for polygon in mesh.polygons:
    polygon.material_index = int(polygon.index >= 10)
if options.get("out_of_range"):
    mesh.loops[mesh.polygons[0].loop_start].vertex_index = len(vertices)
if options.get("child"):
    bpy.ops.mesh.primitive_cube_add(size=0.1)
    root = bpy.context.object
    obj.parent = root
else:
    root = obj
if options.get("probe_only"):
    before = [(tuple(p.vertices), p.material_index) for p in mesh.polygons]
    try:
        kb.validate_meshes([obj])
    except ValueError as exc:
        kb.emit("error_type", type(exc).__name__)
    kb.emit("unchanged", before == [(tuple(p.vertices), p.material_index) for p in mesh.polygons])
    kb.emit("faces", len(mesh.polygons))
elif options.get("raw_save"):
    bpy.ops.wm.save_as_mainfile(filepath=options["output_blend"])
else:
    kb.save_asset(root)
"""

ROUNDTRIP = """from collections import Counter
import bpy
import kitbash_bpy as kb

def snapshot():
    faces = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        for polygon in obj.data.polygons:
            points = tuple(sorted(tuple(round(v, 5) for v in obj.matrix_world @ obj.data.vertices[i].co)
                                  for i in polygon.vertices))
            material = obj.material_slots[polygon.material_index].material.name
            faces.append((points, material))
    return sorted(faces)

before = snapshot()
kb.reset_scene()
bpy.ops.wm.usd_import(filepath=kb.args()["usd_path"])
kb.validate_meshes(bpy.context.scene.objects)
after = snapshot()
kb.emit("same_faces_and_materials", before == after)
kb.emit("faces", len(after))
kb.emit("materials", dict(Counter(material for _, material in after)))
"""


@pytest.fixture
def toolkit(tmp_path):
    config = load_config(None, {"usd.materialx": "off"})
    return BlenderToolkit(
        BlenderRunner("blender", timeout_s=120), config.blender, config.usd, config.naming, tmp_path / "downloads"
    )


def build(toolkit, tmp_path, **options):
    script = tmp_path / "ring.py"
    script.write_text(RING)
    blend = tmp_path / "asset.blend"
    toolkit.run_script(
        script, {"slug": "ring", "output_blend": str(blend), **options}, tmp_path / "build.log"
    )
    return blend


@pytest.mark.parametrize("child", [False, True], ids=["root", "child"])
def test_save_asset_rejects_repeated_face_vertices_before_writing(toolkit, tmp_path, child):
    with pytest.raises(BlenderScriptError):
        build(toolkit, tmp_path, malformed=True, child=child)
    assert not (tmp_path / "asset.blend").exists()


@pytest.mark.parametrize("options", [{"malformed": True}, {"out_of_range": True}], ids=["repeated-index", "out-of-range"])
def test_mesh_validation_does_not_repair_geometry_or_materials(toolkit, tmp_path, options):
    script = tmp_path / "probe.py"
    script.write_text(RING)
    result = toolkit.run_script(script, {"slug": "ring", "probe_only": True, **options}, tmp_path / "probe.log")
    assert result["error_type"] == "ValueError"
    assert result["unchanged"] is True
    assert result["faces"] == 16


@pytest.mark.parametrize("gate", ["inspect", "export"])
def test_cached_malformed_asset_cannot_bypass_save_validation(toolkit, tmp_path, gate):
    blend = build(toolkit, tmp_path, malformed=True, raw_save=True)
    usd = tmp_path / "asset.usda"
    with pytest.raises(BlenderScriptError):
        if gate == "inspect":
            toolkit.inspect_asset(blend, "ring", (2.0, 2.0, 3.5), tmp_path, "inspect")
        else:
            toolkit.export_usd(blend, usd, tmp_path / "work", tmp_path, "export")
    assert not usd.exists()


def test_correct_ring_indices_preserve_all_faces_and_materials_on_usd_import(toolkit, tmp_path):
    blend = build(toolkit, tmp_path)
    usd = tmp_path / "asset.usda"
    toolkit.export_usd(blend, usd, tmp_path / "work", tmp_path, "export")
    script = tmp_path / "roundtrip.py"
    script.write_text(ROUNDTRIP)
    log = tmp_path / "roundtrip.log"
    result = toolkit.run_script(script, {"usd_path": str(usd)}, log, blend=blend)
    assert result["same_faces_and_materials"] is True
    assert result["faces"] == 16
    assert result["materials"] == {"mat_ring_wood": 10, "mat_ring_foliage": 6}
