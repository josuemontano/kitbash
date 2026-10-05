"""Inspect a laid-out scene: placements, placeholders, camera, lights, world, and grounding of objects."""

import math

import bpy
import kb_materials as km
import kitbash_bpy as kb
from mathutils import Vector

options = kb.args()
expected_assets = set(options.get("expected_assets", []))
expected_placeholders = set(options.get("expected_placeholders", []))
scene = bpy.context.scene
depsgraph = bpy.context.evaluated_depsgraph_get()


def support_gap(obj):
    """Distance from the bottom center of ``obj`` down to the first surface below it (None if nothing)."""
    lo, hi = kb.world_bounds([obj, *[c for c in obj.children_recursive if c.type == "MESH"]])
    origin = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z - 1e-4))
    hit, location, *_ = scene.ray_cast(depsgraph, origin, Vector((0, 0, -1)), distance=100.0)
    if hit:
        return round(origin.z - location.z, 4)
    return round(lo.z, 4) if lo.z > 0 else 0.0


placed, placeholders, violations = {}, [], []
for obj in scene.objects:
    key = obj.get("kb_asset_key")
    if key:
        placed.setdefault(key, []).append(obj)
        spec = kb.args().get("assets", {}).get(key, {})
        if not obj.name.split(".")[0].startswith(spec.get("name", key)):
            violations.append(f"{obj.name!r} should be named after its asset {spec.get('name', key)!r}")
    if obj.get("kb_placeholder"):
        placeholders.append(obj.get("kb_placeholder"))

floating = []
roots = [o for objects in placed.values() for o in objects if o.type == "MESH"]
for obj in roots:
    gap = support_gap(obj)
    if gap is not None and gap > 0.03:
        floating.append({"object": obj.name, "gap_m": gap})

camera = scene.camera
lights = [o for o in scene.objects if o.type == "LIGHT"]
world_nodes = sorted({n.bl_idname for n in scene.world.node_tree.nodes}) if scene.world and scene.world.node_tree else []
environment_images = [
    km.image_info(n.image) for n in (scene.world.node_tree.nodes if scene.world and scene.world.node_tree else []) if getattr(n, "image", None)
]
materials = [m for m in bpy.data.materials if m.users]
missing_textures = sorted(
    {i["filepath"] for m in materials for i in (km.image_info(img) for img in km.images_of(m)) if not i["exists"] and not i["packed"]}
    | {i["filepath"] for i in environment_images if not i["exists"] and not i["packed"]}
)

kb.emit(
    "report",
    {
        "placed": {k: [{"name": o.name, "location": [round(v, 3) for v in o.location],
                        "rotation_deg": [round(math.degrees(v), 1) for v in o.rotation_euler],
                        "dimensions_m": kb.dimensions(o) if o.type == "MESH" else None} for o in objects]
                   for k, objects in placed.items()},
        "placeholders": placeholders,
        "missing_assets": sorted(expected_assets - set(placed)),
        "missing_placeholders": sorted(expected_placeholders - set(placeholders)),
        "floating": floating,
        "naming_violations": violations,
        "camera": None if camera is None else {
            "name": camera.name, "location": [round(v, 3) for v in camera.location],
            "rotation_deg": [round(math.degrees(v), 1) for v in camera.rotation_euler],
            "type": camera.data.type, "lens_mm": round(camera.data.lens, 1),
        },
        "lights": [{"name": o.name, "type": o.data.type, "energy": round(o.data.energy, 2),
                    "location": [round(v, 2) for v in o.location]} for o in lights],
        "world_nodes": world_nodes,
        "environment_images": environment_images,
        "render_engine": scene.render.engine,
        "view_transform": scene.view_settings.view_transform,
        "freestyle": scene.render.use_freestyle,
        "missing_textures": missing_textures,
    },
)
kb.emit(
    "facts",
    {
        "missing_assets": len(expected_assets - set(placed)),
        "missing_placeholders": len(expected_placeholders - set(placeholders)),
        "floating_assets": len(floating),
        "naming_violations": len(violations),
        "has_camera": camera is not None,
        "lights": len(lights),
        "missing_textures": len(missing_textures),
    },
)
