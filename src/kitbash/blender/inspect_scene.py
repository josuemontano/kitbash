"""Inspect a laid-out scene without mutating it, including evaluated placement geometry."""

import math
import os
from collections import Counter
from itertools import chain

import bpy
import kb_materials as km
import kitbash_bpy as kb
from mathutils import Vector
from mathutils.bvhtree import BVHTree

GROUND_TOLERANCE = 0.03
RAY_EPSILON = 1e-4
GEOMETRY_TYPES = {"MESH", "CURVE", "SURFACE", "META", "FONT"}


def _placement_ancestors(obj, roots):
    owners = set()
    while obj is not None:
        original = obj.original
        if original in roots:
            owners.add(original)
        obj = original.parent
    return owners


def _geometry(depsgraph, roots):
    """World-space evaluated surfaces, with instance-specific ownership for self exclusion."""
    surfaces, errors, materials = [], {}, set()
    for instance in depsgraph.object_instances:
        obj = instance.object
        if obj.type not in GEOMETRY_TYPES or obj.hide_render or not instance.show_self:
            continue
        parent = instance.parent if instance.is_instance else obj
        owners = _placement_ancestors(parent, roots)
        if any(owner.hide_render for owner in owners):
            continue
        materials.update(slot.material for slot in obj.material_slots if slot.material is not None)
        converted = obj.type != "MESH"
        try:
            mesh = obj.to_mesh() if converted else obj.data
            if mesh is None or not mesh.polygons:
                for owner in owners:
                    errors.setdefault(owner, []).append(f"{obj.name}: no polygon geometry")
                continue
            matrix = instance.matrix_world
            vertices = [matrix @ vertex.co for vertex in mesh.vertices]
            if not all(math.isfinite(v) for point in vertices for v in point):
                raise ValueError("non-finite geometry coordinates")
            mesh.calc_loop_triangles()
            faces = [tuple(face.vertices) for face in mesh.loop_triangles if mesh.polygons[face.polygon_index].area > 0]
            if not faces or abs(matrix.determinant()) < 1e-12:
                raise ValueError("degenerate geometry")
            surfaces.append({
                "owners": owners,
                "vertices": vertices,
                "centers": [sum((vertices[index] for index in face), Vector()) / 3 for face in faces],
                "tree": BVHTree.FromPolygons(vertices, faces, all_triangles=True),
            })
        except (RuntimeError, ValueError) as exc:
            for owner in owners:
                errors.setdefault(owner, []).append(f"{obj.name}: {exc}")
        finally:
            if converted:
                obj.to_mesh_clear()
    return surfaces, errors, materials


def _support(root, surfaces, errors):
    own = [surface for surface in surfaces if root in surface["owners"]]
    if root in errors:
        return {"object": root.name, "gap_m": None, "reason": "uninspectable_geometry", "details": errors[root]}
    if not own:
        return {"object": root.name, "gap_m": None, "reason": "no_inspectable_geometry"}

    bottom = min(point.z for surface in own for point in surface["vertices"])
    samples = set()
    for surface in own:
        vertices = surface["vertices"]
        samples.update(tuple(point) for point in vertices if point.z <= bottom + GROUND_TOLERANCE)
        samples.update(tuple(point) for point in surface["centers"] if point.z <= bottom + GROUND_TOLERANCE)

    other = [surface for surface in surfaces if root not in surface["owners"]]
    nearest = None
    down = Vector((0, 0, -1))
    for point in samples:
        origin = Vector(point) + Vector((0, 0, RAY_EPSILON))
        for surface in other:
            hit, _, _, distance = surface["tree"].ray_cast(origin, down)
            if hit is None:
                continue
            gap = max(0.0, distance - RAY_EPSILON)
            if gap <= GROUND_TOLERANCE:
                return None
            nearest = gap if nearest is None else min(nearest, gap)
    # Also probe from the support side: a narrow pedestal may lie between all base samples.
    up = -down
    for surface in other:
        for point in chain(surface["vertices"], surface["centers"]):
            if not bottom - GROUND_TOLERANCE <= point.z <= bottom + RAY_EPSILON:
                continue
            origin = point - Vector((0, 0, RAY_EPSILON))
            for target in own:
                hit, _, _, distance = target["tree"].ray_cast(origin, up, GROUND_TOLERANCE + RAY_EPSILON)
                if hit is not None and distance - RAY_EPSILON <= GROUND_TOLERANCE:
                    return None
    # Wall-mounted pieces and terrain intersections have real contact without a support below
    # their lowest vertices. Check actual surfaces, never bounding-box overlap or script tags.
    for source in own:
        for target in other:
            if source["tree"].overlap(target["tree"]):
                return None
            for first, second in ((source, target), (target, source)):
                for point in chain(first["vertices"], first["centers"]):
                    hit, _, _, distance = second["tree"].find_nearest(point, GROUND_TOLERANCE)
                    if hit is not None and distance <= GROUND_TOLERANCE:
                        return None
    return {
        "object": root.name,
        "gap_m": None if nearest is None else round(nearest, 4),
        "reason": "no_external_support" if nearest is None else "support_too_far",
    }


def _images(tree, seen=None):
    """Image nodes in a material/world, including shared and nested node groups."""
    if tree is None:
        return set()
    seen = set() if seen is None else seen
    if tree in seen:
        return set()
    seen.add(tree)
    result = set()
    for node in tree.nodes:
        image = getattr(node, "image", None)
        if image is not None:
            result.add(image)
        result.update(_images(getattr(node, "node_tree", None), seen))
    return result


def _image_info(image):
    info = km.image_info(image)
    # A linked image's // path belongs to its library, not the assembled scene's directory.
    absolute = bpy.path.abspath(info["filepath"], library=image.library) if info["filepath"] else ""
    info.update(
        resolved_path=absolute,
        exists=bool(absolute) and os.path.isfile(absolute),
        source=image.source,
    )
    return info


def _counts(options):
    expected = options.get("expected_assets")
    if expected is None:
        expected = {key: spec.get("instances", 1) for key, spec in options.get("assets", {}).items()}
    return Counter(expected)


def _camera(scene, options):
    camera = scene.camera
    reason = None
    if camera is None:
        reason = "no_active_camera"
    elif camera.type != "CAMERA":
        reason = "active_object_is_not_camera"
    elif camera.name not in scene.objects:
        reason = "active_camera_not_in_scene"
    elif options.get("camera_name") is not None and camera.get("kb_camera_name", camera.name) != options["camera_name"]:
        reason = "active_camera_does_not_match_expected"
    if reason:
        return None, reason
    return {
        "name": camera.name,
        "location": [round(v, 3) for v in camera.location],
        "rotation_deg": [round(math.degrees(v), 1) for v in camera.rotation_euler],
        "type": camera.data.type,
        "lens_mm": round(camera.data.lens, 1),
    }, None


def inspect_scene(options: dict) -> tuple[dict, dict]:
    """Return (report, facts) for the active scene; importing this module emits nothing.

    ``expected_assets`` maps keys to instance counts, or defaults to each asset spec's
    ``instances`` (one when omitted). ``expected_placeholders`` counts repeated IDs too.
    Tags identify placements, but evaluated geometry must establish their integrity and support.
    """
    scene = bpy.context.scene
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    expected_assets = _counts(options)
    expected_placeholders = Counter(options.get("expected_placeholders", []))
    placed, placeholders, violations, roots = {}, [], [], set()
    for obj in scene.objects:
        key = obj.get("kb_asset_key")
        if key:
            placed.setdefault(key, []).append(obj)
            roots.add(obj)
            name = options.get("assets", {}).get(key, {}).get("name", key)
            if not obj.name.startswith(name):
                violations.append(f"{obj.name!r} should be named after its asset {name!r}")
        if obj.get("kb_placeholder"):
            placeholders.append(obj["kb_placeholder"])
            roots.add(obj)

    actual_assets = Counter({key: len(objects) for key, objects in placed.items()})
    actual_placeholders = Counter(placeholders)
    missing_assets = expected_assets - actual_assets
    unexpected_assets = actual_assets - expected_assets
    missing_placeholders = expected_placeholders - actual_placeholders
    unexpected_placeholders = actual_placeholders - expected_placeholders
    surfaces, errors, materials = _geometry(depsgraph, roots)
    floating = []
    airborne = Counter(options.get("airborne", {}))
    intentional_airborne = []
    for root in sorted(roots, key=lambda obj: obj.name):
        failure = _support(root, surfaces, errors)
        key = root.get("kb_asset_key") or root.get("kb_placeholder")
        if failure is not None and failure["reason"] in {"no_external_support", "support_too_far"} and airborne[key] > 0:
            airborne[key] -= 1
            intentional_airborne.append(root.name)
            continue
        if failure is not None:
            floating.append(failure)

    camera, camera_error = _camera(scene, options)
    lights = [obj for obj in scene.objects if obj.type == "LIGHT"]
    world_tree = scene.world.node_tree if scene.world else None
    world_nodes = sorted({node.bl_idname for node in world_tree.nodes}) if world_tree else []
    world_images = _images(world_tree)
    environment_images = [_image_info(image) for image in sorted(world_images, key=lambda image: image.name)]
    images = set(world_images)
    for material in materials:
        images.update(_images(material.node_tree))
    texture_images = [_image_info(image) for image in sorted(images, key=lambda image: image.name)]
    missing_textures = sorted({
        info["resolved_path"] or info["filepath"] or info["name"] for info in texture_images
        if info["source"] not in {"GENERATED", "VIEWER"} and not info["exists"] and not info["packed"]
    })
    report = {
        "placed": {
            key: [{
                "name": obj.name,
                "location": [round(v, 3) for v in obj.location],
                "rotation_deg": [round(math.degrees(v), 1) for v in obj.rotation_euler],
                "dimensions_m": kb.dimensions(obj) if obj.type == "MESH" else None,
            } for obj in objects]
            for key, objects in placed.items()
        },
        "placeholders": placeholders,
        "expected_assets": dict(expected_assets),
        "actual_assets": dict(actual_assets),
        "expected_placeholders": dict(expected_placeholders),
        "actual_placeholders": dict(actual_placeholders),
        "missing_assets": sorted(missing_assets.elements()),
        "unexpected_assets": sorted(unexpected_assets.elements()),
        "missing_placeholders": sorted(missing_placeholders.elements()),
        "unexpected_placeholders": sorted(unexpected_placeholders.elements()),
        "floating": floating,
        "intentional_airborne": intentional_airborne,
        "naming_violations": violations,
        "camera": camera,
        "camera_error": camera_error,
        "expected_camera": options.get("camera_name"),
        "lights": [{
            "name": obj.name, "type": obj.data.type, "energy": round(obj.data.energy, 2),
            "location": [round(v, 2) for v in obj.location],
        } for obj in lights],
        "world_nodes": world_nodes,
        "environment_images": environment_images,
        "texture_images": texture_images,
        "render_engine": scene.render.engine,
        "view_transform": scene.view_settings.view_transform,
        "freestyle": scene.render.use_freestyle,
        "missing_textures": missing_textures,
    }
    facts = {
        "missing_assets": sum(missing_assets.values()),
        "unexpected_assets": sum(unexpected_assets.values()),
        "missing_placeholders": sum(missing_placeholders.values()),
        "unexpected_placeholders": sum(unexpected_placeholders.values()),
        "floating_assets": len(floating),
        "naming_violations": len(violations),
        "has_camera": camera is not None,
        "lights": len(lights),
        "missing_textures": len(missing_textures),
    }
    return report, facts


if __name__ == "__main__":
    report, facts = inspect_scene(kb.args())
    kb.emit("report", report)
    kb.emit("facts", facts)
