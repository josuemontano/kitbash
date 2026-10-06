"""Inspect an asset .blend: objects, scale, origin, naming and material graphs, plus facts for rubric checks."""

import math

import bpy
import kb_materials
import kitbash_bpy as kb
from mathutils import Vector

options = kb.args()
expected = [max(float(v), 1e-4) for v in options["expected_dimensions"]]
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if not meshes:
    raise RuntimeError("The asset has no mesh objects")
kb.validate_meshes(meshes)

object_rx = kb.naming_regex("object")
root = next((o for o in meshes if object_rx.match(o.name)), None) or max(meshes, key=lambda o: len(o.data.polygons))
lo, hi = kb.world_bounds(meshes)
size = [round(v, 5) for v in (hi - lo)]
base_center = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))
origin_offset = (root.matrix_world.translation - base_center).length
rotation_identity = all(abs(a) < 1e-4 for a in root.matrix_world.to_euler()) and all(
    abs(s - 1.0) < 1e-4 for s in root.matrix_world.to_scale()
)


def _orientation_ok(expected, actual, ratio=1.5, tolerance=1.15):
    """False when a clearly tall (or clearly flat) object does not come out tall (or flat) along Z."""
    footprint_max, footprint_min = max(expected[0], expected[1]), min(expected[0], expected[1])
    if expected[2] >= ratio * footprint_max:
        return actual[2] * tolerance >= max(actual[0], actual[1])
    if footprint_min >= ratio * expected[2]:
        return actual[2] <= min(actual[0], actual[1]) * tolerance
    return True


orientation_ok = _orientation_ok(expected, size)
volume = max(size[0] * size[1] * size[2], 1e-12)
scale_error = abs((volume / (expected[0] * expected[1] * expected[2])) ** (1 / 3) - 1.0)

violations = []
if not object_rx.match(root.name):
    violations.append(f"object {root.name!r} should be {kb.naming('object')!r}")
if not kb.naming_regex("mesh").match(root.data.name):
    violations.append(f"mesh {root.data.name!r} should be {kb.naming('mesh')!r}")
for obj in meshes:
    if obj is not root and not obj.name.startswith(kb.slug()):
        violations.append(f"object {obj.name!r} should start with {kb.slug()!r}")
materials = kb_materials.mesh_materials(meshes)
material_rx, image_rx = kb.naming_regex("material"), kb.naming_regex("image")
for material in materials:
    if not material_rx.match(material.name):
        violations.append(f"material {material.name!r} should match {kb.naming('material', '<part>')!r}")
    for image in kb_materials.images_of(material):
        if not image_rx.match(image.name):
            violations.append(f"image {image.name!r} should match {kb.naming('image', '<part>')!r}")

summaries = [kb_materials.material_summary(m) for m in materials]
missing = sorted({i["filepath"] for s in summaries for i in s["images"] if not i["exists"] and not i["packed"]})
absolute = sorted({i["filepath"] for s in summaries for i in s["images"] if not i["relative"] and not i["packed"]})
faces_without_material = sum(1 for o in meshes if not o.material_slots or all(s.material is None for s in o.material_slots))

kb.emit(
    "report",
    {
        "root": root.name,
        "objects": [
            {
                "name": o.name,
                "type": o.type,
                "data": o.data.name,
                "parent": o.parent.name if o.parent else None,
                "dimensions_m": kb.dimensions(o),
                "location": [round(v, 4) for v in o.location],
                "rotation_deg": [round(math.degrees(v), 2) for v in o.rotation_euler],
                "scale": [round(v, 4) for v in o.scale],
                "vertices": len(o.data.vertices),
                "faces": len(o.data.polygons),
                "uv_layers": [uv.name for uv in o.data.uv_layers],
                "materials": [s.material.name for s in o.material_slots if s.material],
            }
            for o in meshes
        ],
        "bounds_m": {"min": list(lo), "max": list(hi), "size": size},
        "expected_dimensions_m": expected,
        "origin_world": list(root.matrix_world.translation),
        "base_center_world": list(base_center),
        "naming_violations": violations,
        "materials": summaries,
        "missing_textures": missing,
        "absolute_texture_paths": absolute,
    },
)
kb.emit(
    "facts",
    {
        "dimensions_m": size,
        "scale_error": round(scale_error, 4),
        "origin_offset_m": round(origin_offset, 5),
        "up_axis_ok": bool(rotation_identity and orientation_ok),
        "naming_violations": len(violations),
        "non_principled_materials": sum(1 for s in summaries if not s["ends_in_principled"]),
        "materials": len(summaries),
        "objects_without_material": faces_without_material,
        "missing_textures": len(missing),
        "absolute_texture_paths": len(absolute),
        "faces": sum(len(o.data.polygons) for o in meshes),
    },
)
