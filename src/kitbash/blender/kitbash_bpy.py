"""kitbash helper library for Blender scripts. Import it as ``import kitbash_bpy as kb``.

Every function marked ``@api`` is documented for, and meant to be used by, generated scripts.
Units are meters, Z is up, angles are in degrees unless stated otherwise.
"""

import json
import math
import os
import re
import shutil
import urllib.request

import bmesh
import bpy
import numpy as np
from kb_files import copy_texture
from mathutils import Euler, Matrix, Vector

API = []
_ARGS = {}
_RESULT = {}
_PLACED = {}
DEFAULT_NAMING = {
    "object": "{slug}",
    "mesh": "{slug}_mesh",
    "material": "mat_{slug}_{part}",
    "image": "tex_{slug}_{part}",
    "collection": "{slug}",
}
POLYHAVEN_API = "https://api.polyhaven.com/files/"
POLYHAVEN_MAPS = ("Diffuse", "Rough", "nor_gl", "arm", "AO", "Displacement", "Metal")


def api(function):
    API.append(function.__name__)
    return function


def _configure(arguments):
    _ARGS.clear()
    _ARGS.update(arguments)
    _RESULT.clear()
    _PLACED.clear()


def _result():
    return dict(_RESULT)


# -- run arguments ---------------------------------------------------------------------------------


@api
def args():
    """Arguments kitbash passed to this run: paths (mesh_path, output_blend, textures_dir), slug, dimensions_m, naming, assets..."""
    return _ARGS


@api
def emit(key, value):
    """Send a JSON-serializable value back to kitbash under ``key``."""
    _RESULT[key] = value


def slug():
    return _ARGS.get("slug", "asset")


def naming(kind, part=None, asset_slug=None):
    pattern = {**DEFAULT_NAMING, **_ARGS.get("naming", {})}[kind]
    return pattern.format(slug=asset_slug or slug(), part=part_name(part or "main"))


def naming_regex(kind, asset_slug=None):
    """Regex that matches names following the convention for ``kind`` (``{part}`` = any snake_case part)."""
    pattern = {**DEFAULT_NAMING, **_ARGS.get("naming", {})}[kind]
    escaped = re.escape(pattern).replace(re.escape("{slug}"), re.escape(asset_slug or slug()))
    return re.compile("^" + escaped.replace(re.escape("{part}"), "[a-z0-9_]+") + r"(\.\d{3})?$")


def part_name(text):
    part = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return part or "main"


# -- scene -----------------------------------------------------------------------------------------


@api
def reset_scene():
    """Start from an empty scene with metric units (1 Blender unit = 1 meter)."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    units = bpy.context.scene.unit_settings
    units.system = "METRIC"
    units.scale_length = 1.0
    units.length_unit = "METERS"
    _PLACED.clear()


def collection(name, parent=None):
    """Get or create a collection linked under ``parent`` (default: the scene collection)."""
    existing = bpy.data.collections.get(name)
    if existing is None:
        existing = bpy.data.collections.new(name)
    parent = parent or bpy.context.scene.collection
    if existing.name not in parent.children and existing is not parent:
        parent.children.link(existing)
    return existing


def asset_collection():
    return collection(naming("collection"))


def activate(obj):
    view_layer = bpy.context.view_layer
    view_layer.update()
    for other in view_layer.objects:
        if other is not None:
            other.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _vertices(mesh):
    coords = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", coords)
    return coords.reshape(-1, 3)


def _local_bounds(obj):
    coords = _vertices(obj.data)
    if not len(coords):
        return Vector((0, 0, 0)), Vector((0, 0, 0))
    return Vector(coords.min(axis=0)), Vector(coords.max(axis=0))


def world_bounds(objects):
    """(min, max) world-space corners of the evaluated bounding boxes of ``objects``."""
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    if not points:
        return Vector((0, 0, 0)), Vector((0, 0, 0))
    lo = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    hi = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    return lo, hi


@api
def dimensions(obj):
    """World-space (width, depth, height) of ``obj`` in meters."""
    lo, hi = world_bounds([obj])
    return tuple(round(v, 5) for v in (hi - lo))


# -- meshes ----------------------------------------------------------------------------------------


@api
def import_mesh(path=None, name=None, up_axis=None):
    """Import the Trellis mesh (GLB, OBJ, PLY, STL or FBX; default ``args()['mesh_path']``) as ONE mesh object.

    ``up_axis`` is the up axis of the file's vertices ('Z' for Trellis output, the default from
    ``args()['mesh_up_axis']``; 'Y' for conventional glTF/OBJ). All parts are joined, transforms are applied
    and imported materials are dropped. The object is named per the naming convention and placed in the
    asset collection.
    """
    path = path or _ARGS["mesh_path"]
    up_axis = (up_axis or _ARGS.get("mesh_up_axis", "Y")).upper()
    before = {o.name for o in bpy.data.objects}
    extension = os.path.splitext(path)[1].lower()
    undo_gltf_axis = False
    if extension in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
        undo_gltf_axis = up_axis == "Z"  # the glTF importer always converts Y-up to Z-up
    elif extension == ".obj":
        forward, up = ("Y", "Z") if up_axis == "Z" else ("NEGATIVE_Z", "Y")
        bpy.ops.wm.obj_import(filepath=path, forward_axis=forward, up_axis=up)
    elif extension == ".ply":
        bpy.ops.wm.ply_import(filepath=path)
    elif extension == ".stl":
        bpy.ops.wm.stl_import(filepath=path)
    elif extension == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    else:
        raise ValueError(f"Unsupported mesh format: {path}")
    new_objects = [o for o in bpy.data.objects if o.name not in before]
    meshes = [o for o in new_objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh found in {path}")
    bpy.context.view_layer.update()
    joined = bmesh.new()
    for obj in meshes:
        data = obj.data.copy()
        data.transform(obj.matrix_world)
        joined.from_mesh(data)
        bpy.data.meshes.remove(data)
    bmesh.ops.remove_doubles(joined, verts=joined.verts, dist=1e-6)  # exporters often split vertices at every face
    mesh = bpy.data.meshes.new(naming("mesh", asset_slug=name))
    joined.to_mesh(mesh)
    joined.free()
    if undo_gltf_axis:
        mesh.transform(Matrix.Rotation(math.radians(-90.0), 4, "X"))
    for obj in new_objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for material in list(bpy.data.materials):
        if material.users == 0:
            bpy.data.materials.remove(material)
    mesh.materials.clear()
    obj = bpy.data.objects.new(naming("object", asset_slug=name), mesh)
    asset_collection().objects.link(obj)
    activate(obj)
    emit("import", {"path": path, "vertices": len(mesh.vertices), "faces": len(mesh.polygons)})
    return obj


def _replace_with_evaluated(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(evaluated, preserve_all_data_layers=True, depsgraph=depsgraph)
    old = obj.data
    obj.modifiers.clear()
    obj.data = mesh
    name = old.name
    bpy.data.meshes.remove(old)
    mesh.name = name


def _collapse(obj, limit):
    obj.data.calc_loop_triangles()  # the collapse ratio is relative to triangles, not faces
    modifier = obj.modifiers.new("kb_decimate", "DECIMATE")
    modifier.decimate_type = "COLLAPSE"
    modifier.ratio = min(1.0, limit / max(len(obj.data.loop_triangles), 1))
    _replace_with_evaluated(obj)


@api
def decimate(obj, max_faces=None, voxel_resolution=384):
    """Reduce ``obj`` to at most ``max_faces`` faces (default ``args()['max_faces']``) with collapse decimation.
    Non-manifold meshes that collapse cannot reduce enough are first voxel-remeshed (watertight) at
    ``voxel_resolution`` voxels across their largest side. TriFlow meshes are left unchanged, even
    with an explicit face limit. Returns the face count."""
    if _ARGS.get("retopology_method") == "triflow":
        return len(obj.data.polygons)
    limit = int(max_faces or _ARGS.get("max_faces", 150000))
    if len(obj.data.polygons) <= limit:
        return len(obj.data.polygons)
    _collapse(obj, limit)
    if len(obj.data.polygons) > limit * 1.25:
        lo, hi = _local_bounds(obj)
        remesh = obj.modifiers.new("kb_remesh", "REMESH")
        remesh.mode = "VOXEL"
        remesh.voxel_size = max(max(hi - lo) / voxel_resolution, 1e-5)
        remesh.adaptivity = 0.0
        _replace_with_evaluated(obj)
        if len(obj.data.polygons) > limit:
            _collapse(obj, limit)
    return len(obj.data.polygons)


@api
def clean_mesh(obj, merge_distance=0.0001, smooth_angle=40.0, min_island_ratio=0.002):
    """Merge duplicates, drop loose geometry/tiny islands and update normals and smooth shading.
    For TriFlow meshes, preserve connectivity and update only normals and shading."""
    mesh = obj.data
    work = bmesh.new()
    work.from_mesh(mesh)
    if _ARGS.get("retopology_method") != "triflow":
        bmesh.ops.remove_doubles(work, verts=work.verts, dist=merge_distance)
        loose = [v for v in work.verts if not v.link_faces]
        if loose:
            bmesh.ops.delete(work, geom=loose, context="VERTS")
        _remove_small_islands(work, min_island_ratio)
    bmesh.ops.recalc_face_normals(work, faces=work.faces)
    work.to_mesh(mesh)
    work.free()
    mesh.shade_smooth()
    if hasattr(mesh, "set_sharp_from_angle"):
        mesh.set_sharp_from_angle(angle=math.radians(smooth_angle))
    return len(mesh.polygons)


def _remove_small_islands(work, ratio):
    total = len(work.faces)
    if not total or ratio <= 0:
        return
    work.faces.ensure_lookup_table()
    seen = set()
    small = []
    for face in work.faces:
        if face.index in seen:
            continue
        island, stack = [], [face]
        seen.add(face.index)
        while stack:
            current = stack.pop()
            island.append(current)
            for edge in current.edges:
                for neighbour in edge.link_faces:
                    if neighbour.index not in seen:
                        seen.add(neighbour.index)
                        stack.append(neighbour)
        if len(island) < total * ratio:
            small.extend(island)
    if small and len(small) < total:
        bmesh.ops.delete(work, geom=small, context="FACES")


@api
def rotate(obj, degrees=(0.0, 0.0, 0.0)):
    """Rotate the mesh data of ``obj`` about its origin by XYZ Euler ``degrees`` (applied; object rotation stays zero)."""
    rotation = Euler([math.radians(d) for d in degrees], "XYZ").to_matrix().to_4x4()
    obj.data.transform(rotation)
    obj.data.update()


@api
def fit_dimensions(obj, target=None, mode="height"):
    """Scale the mesh data so its bounds match ``target`` (width, depth, height in meters; default
    ``args()['dimensions_m']``). ``mode``: 'height' (match Z), 'max' (largest side), 'footprint'
    (largest of X/Y), 'volume' (overall size) or 'exact' (non-uniform). Returns the new dimensions."""
    target = [max(float(v), 1e-4) for v in (target or _ARGS["dimensions_m"])]
    lo, hi = _local_bounds(obj)
    size = [max(v, 1e-6) for v in (hi - lo)]
    if mode == "exact":
        factors = [t / s for t, s in zip(target, size, strict=True)]
    else:
        uniform = {
            "height": target[2] / size[2],
            "max": max(target) / max(size),
            "footprint": max(target[0], target[1]) / max(size[0], size[1]),
            "volume": (target[0] * target[1] * target[2] / (size[0] * size[1] * size[2])) ** (1 / 3),
        }[mode]
        factors = [uniform] * 3
    obj.data.transform(Matrix.Diagonal((*factors, 1.0)))
    obj.data.update()
    return dimensions(obj)


@api
def origin_to_base(obj):
    """Move the geometry so the origin is at the bottom center of its bounds and put the object at the world origin."""
    lo, hi = _local_bounds(obj)
    base = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))
    obj.data.transform(Matrix.Translation(-base))
    obj.data.update()
    obj.location = (0.0, 0.0, 0.0)
    obj.rotation_euler = (0.0, 0.0, 0.0)
    obj.scale = (1.0, 1.0, 1.0)


@api
def ensure_uv(obj, angle_limit=66.0, island_margin=0.02):
    """Create a UV map with Smart UV Project if ``obj`` has none (needed for UV-mapped image textures)."""
    if obj.data.uv_layers:
        return obj.data.uv_layers.active.name
    activate(obj)
    with bpy.context.temp_override(active_object=obj, object=obj, selected_objects=[obj], selected_editable_objects=[obj]):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=math.radians(angle_limit), island_margin=island_margin)
        bpy.ops.object.mode_set(mode="OBJECT")
    return obj.data.uv_layers.active.name


# -- materials -------------------------------------------------------------------------------------


def node_tree(owner):
    if owner.node_tree is None:
        owner.use_nodes = True
    return owner.node_tree


def _socket_name(node, key):
    wanted = key.replace("_", " ").lower()
    for socket in node.inputs:
        if socket.name.lower() == wanted or socket.identifier.lower() == key.lower():
            return socket.name
    raise KeyError(f"{node.bl_idname} has no input {key!r}")


@api
def material_name(part):
    """Conventional material name for a part of the current asset, e.g. material_name('wood')."""
    return naming("material", part)


@api
def principled(part, base_color=(0.8, 0.8, 0.8), roughness=0.5, metallic=0.0, **inputs):
    """New material with a Principled BSDF, named per convention from ``part`` (e.g. 'wood', 'metal_legs').

    Extra Principled inputs by socket name, spaces as underscores: principled('glass', transmission_weight=1.0, ior=1.45).
    Returns the material; use principled_node(mat) for the BSDF node.
    """
    name = part if naming_regex("material").match(part) else material_name(part)
    material = bpy.data.materials.new(name)
    tree = node_tree(material)
    bsdf = principled_node(material)
    color = tuple(base_color)
    bsdf.inputs["Base Color"].default_value = color if len(color) == 4 else (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    for key, value in inputs.items():
        bsdf.inputs[_socket_name(bsdf, key)].default_value = value
    tree.nodes.active = bsdf
    return material


@api
def principled_node(material):
    """The Principled BSDF of ``material`` (created and wired to the output if missing)."""
    tree = node_tree(material)
    for node in tree.nodes:
        if node.bl_idname == "ShaderNodeBsdfPrincipled":
            return node
    bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
    output = next((n for n in tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial"), None)
    output = output or tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return bsdf


@api
def add_node(material, node_type, location=None, **properties):
    """Add a shader node by bl_idname (e.g. 'ShaderNodeTexNoise', 'ShaderNodeValToRGB') and set node properties."""
    node = node_tree(material).nodes.new(node_type)
    if location is not None:
        node.location = location
    for key, value in properties.items():
        setattr(node, key, value)
    return node


@api
def link(material, from_socket, to_socket):
    """Connect ``from_socket`` to ``to_socket`` in the material's node tree."""
    return node_tree(material).links.new(from_socket, to_socket)


def _texture_coordinates(material, projection, scale):
    tree = node_tree(material)
    coords = tree.nodes.new("ShaderNodeTexCoord")
    mapping = tree.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (scale, scale, scale)
    tree.links.new(coords.outputs["Object" if projection == "BOX" else "UV"], mapping.inputs["Vector"])
    return mapping


@api
def image_texture(material, path, non_color=False, projection="BOX", scale=1.0, part=None):
    """Image Texture node for ``path``. The file is content-hashed into the asset's textures folder
    (saved with a relative path). projection 'BOX' uses object coordinates; 'UV' needs ensure_uv(obj)."""
    textures_dir = _ARGS.get("textures_dir") or os.path.join(os.path.dirname(_ARGS["output_blend"]), "textures")
    destination = copy_texture(path, textures_dir)
    image = bpy.data.images.load(destination, check_existing=True)
    image.name = naming("image", part or os.path.splitext(os.path.basename(path))[0])
    if non_color:
        image.colorspace_settings.name = "Non-Color"
    node = node_tree(material).nodes.new("ShaderNodeTexImage")
    node.image = image
    mapping = _texture_coordinates(material, projection, scale)
    node_tree(material).links.new(mapping.outputs["Vector"], node.inputs["Vector"])
    if projection == "BOX":
        node.projection = "BOX"
        node.projection_blend = 0.2
    return node


@api
def pbr_textures(material, maps, scale=1.0, projection="BOX"):
    """Wire texture maps (e.g. from fetch_polyhaven: 'diffuse', 'rough', 'nor_gl', 'arm', 'metal') into the
    material's Principled BSDF. Returns the created image nodes by map name."""
    bsdf = principled_node(material)
    tree = node_tree(material)
    nodes = {}
    lowered = {k.lower(): v for k, v in maps.items()}
    if "diffuse" in lowered:
        nodes["diffuse"] = image_texture(material, lowered["diffuse"], projection=projection, scale=scale, part="diffuse")
        tree.links.new(nodes["diffuse"].outputs["Color"], bsdf.inputs["Base Color"])
    if "rough" in lowered:
        nodes["rough"] = image_texture(material, lowered["rough"], True, projection, scale, part="rough")
        tree.links.new(nodes["rough"].outputs["Color"], bsdf.inputs["Roughness"])
    elif "arm" in lowered:
        nodes["arm"] = image_texture(material, lowered["arm"], True, projection, scale, part="arm")
        split = tree.nodes.new("ShaderNodeSeparateColor")
        tree.links.new(nodes["arm"].outputs["Color"], split.inputs["Color"])
        tree.links.new(split.outputs["Green"], bsdf.inputs["Roughness"])
        tree.links.new(split.outputs["Blue"], bsdf.inputs["Metallic"])
    if "metal" in lowered:
        nodes["metal"] = image_texture(material, lowered["metal"], True, projection, scale, part="metal")
        tree.links.new(nodes["metal"].outputs["Color"], bsdf.inputs["Metallic"])
    if "nor_gl" in lowered:
        nodes["nor_gl"] = image_texture(material, lowered["nor_gl"], True, projection, scale, part="normal")
        normal_map = tree.nodes.new("ShaderNodeNormalMap")
        tree.links.new(nodes["nor_gl"].outputs["Color"], normal_map.inputs["Color"])
        tree.links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
    return nodes


@api
def assign(obj, material, where=None):
    """Assign ``material`` to the faces of ``obj`` where ``where(center, normal)`` is True (all faces if None).

    ``center`` is the face center in normalized bounds coordinates (0..1 on X, Y and Z; z=0 is the bottom),
    ``normal`` the world-space face normal (a mathutils Vector). Returns the number of faces assigned.
    """
    mesh = obj.data
    if material.name not in [m.name for m in mesh.materials if m]:
        mesh.materials.append(material)
    index = [m.name if m else "" for m in mesh.materials].index(material.name)
    if where is None:
        for polygon in mesh.polygons:
            polygon.material_index = index
        return len(mesh.polygons)
    lo, hi = _local_bounds(obj)
    size = Vector([max(v, 1e-9) for v in (hi - lo)])
    rotation = obj.matrix_world.to_3x3()
    count = 0
    for polygon in mesh.polygons:
        center = polygon.center - lo
        normalized = Vector((center.x / size.x, center.y / size.y, center.z / size.z))
        if where(normalized, (rotation @ polygon.normal).normalized()):
            polygon.material_index = index
            count += 1
    return count


# -- downloads -------------------------------------------------------------------------------------


def _download(url, destination):
    if os.path.isfile(destination) and os.path.getsize(destination) > 0:
        return destination
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "kitbash/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, open(destination + ".part", "wb") as handle:
        shutil.copyfileobj(response, handle)
    os.replace(destination + ".part", destination)
    return destination


@api
def fetch_polyhaven(kind, asset_id, resolution="1k"):
    """Download a Poly Haven asset (cached). kind 'hdri' returns {'hdri': path}; kind 'texture' returns
    {'diffuse', 'rough', 'nor_gl', 'arm', 'ao', 'displacement', ...: path} for the maps that exist."""
    cache = os.path.expanduser(_ARGS.get("download_dir") or "~/.cache/kitbash/downloads")
    request = urllib.request.Request(POLYHAVEN_API + asset_id, headers={"User-Agent": "kitbash/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        files = json.load(response)
    if kind == "hdri":
        entry = files["hdri"][resolution]["hdr"]
        return {"hdri": _download(entry["url"], os.path.join(cache, "hdri", os.path.basename(entry["url"])))}
    maps = {}
    for map_name in POLYHAVEN_MAPS:
        variants = files.get(map_name, {}).get(resolution)
        if not variants:
            continue
        entry = variants.get("jpg") or variants.get("png")
        if entry:
            maps[map_name.lower()] = _download(entry["url"], os.path.join(cache, "textures", asset_id, os.path.basename(entry["url"])))
    if not maps:
        raise RuntimeError(f"Poly Haven asset {asset_id!r} has no {resolution} texture maps")
    return maps


# -- finishing an asset ----------------------------------------------------------------------------


@api
def apply_naming(obj):
    """Rename ``obj``, its mesh, materials and images to the naming convention (called by save_asset)."""
    obj.name = naming("object")
    obj.data.name = naming("mesh")
    material_rx = naming_regex("material")
    image_rx = naming_regex("image")
    for material in [m for m in obj.data.materials if m]:
        if not material_rx.match(material.name):
            material.name = naming("material", material.name)
        for node in node_tree(material).nodes:
            image = getattr(node, "image", None)
            if image is not None and not image_rx.match(image.name):
                image.name = naming("image", os.path.splitext(image.name)[0])
    for child in obj.children_recursive:
        if child.type == "MESH":
            if not child.name.startswith(slug()):
                child.name = f"{slug()}_{part_name(child.name)}"
            child.data.name = f"{part_name(child.name)}_mesh"


@api
def save_asset(obj):
    """Finish the asset: link ``obj`` into the asset collection, apply naming and parent other meshes to it.
    Purge unused data and save ``args()['output_blend']`` with relative texture paths."""
    home = asset_collection()
    if obj.name not in home.objects:
        home.objects.link(obj)
    for other in list(bpy.data.objects):
        if other.type != "MESH" and other.name not in home.objects:
            bpy.data.objects.remove(other, do_unlink=True)
    for other in [o for o in bpy.data.objects if o.type == "MESH" and o is not obj]:
        if other.name not in home.objects:
            home.objects.link(other)
        if other.parent is None:
            world = other.matrix_world.copy()
            other.parent = obj
            other.matrix_world = world
    for scene_child in list(bpy.context.scene.collection.objects):
        bpy.context.scene.collection.objects.unlink(scene_child)
    apply_naming(obj)
    obj["kb_slug"] = slug()
    bpy.data.orphans_purge(do_recursive=True)
    _save(_ARGS["output_blend"])
    emit("asset", {"object": obj.name, "dimensions_m": dimensions(obj), "faces": len(obj.data.polygons)})


def _save(path):
    bpy.context.preferences.filepaths.save_version = 0  # no .blend1 backups
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=path, check_existing=False)
    bpy.ops.file.make_paths_relative()
    bpy.ops.wm.save_mainfile()


# -- layout ----------------------------------------------------------------------------------------


@api
def place_asset(key, location=(0.0, 0.0, 0.0), rotation_deg=(0.0, 0.0, 0.0), scale=1.0):
    """Place an approved asset (``args()['assets'][key]``) with its base center at ``location`` and an XYZ
    Euler rotation in degrees. Placing the same key again adds a linked duplicate. Returns the root object."""
    spec = _ARGS["assets"][key]
    root = _instance(key, spec)
    root.location = location
    root.rotation_euler = [math.radians(d) for d in rotation_deg]
    root.scale = (scale, scale, scale) if isinstance(scale, (int, float)) else scale
    root["kb_asset_key"] = key
    if spec.get("backlot_id"):
        root["kb_backlot_id"] = spec["backlot_id"]
    return root


def _instance(key, spec):
    layout_collection = collection("assets")
    if _ARGS.get("asset_mode", "append") == "link":
        source = _load_collection(key, spec, link=True)
        empty = bpy.data.objects.new(spec.get("name", key), None)
        empty.instance_type = "COLLECTION"
        empty.instance_collection = source
        layout_collection.objects.link(empty)
        return empty
    placed = _PLACED.get(key)
    if placed is None:
        source = _load_collection(key, spec, link=False)
        if source.name not in layout_collection.children:
            layout_collection.children.link(source)
        roots = [o for o in source.objects if o.parent is None]
        _PLACED[key] = {"collection": source, "root": roots[0]}
        return roots[0]
    return _duplicate_hierarchy(placed["root"], placed["collection"])


def _load_collection(key, spec, link):
    cached = _PLACED.get(f"{key}::source")
    if cached is not None:
        return cached
    wanted = spec.get("collection")
    with bpy.data.libraries.load(spec["blend"], link=link) as (source, target):
        if wanted in source.collections:
            target.collections = [wanted]
        else:
            target.objects = list(source.objects)
    if target.collections:
        result = target.collections[0]
    else:
        result = bpy.data.collections.new(wanted or key)
        for obj in target.objects:
            result.objects.link(obj)
    _PLACED[f"{key}::source"] = result
    return result


def _duplicate_hierarchy(root, destination):
    mapping = {}
    for original in [root, *root.children_recursive]:
        copy = original.copy()
        destination.objects.link(copy)
        mapping[original] = copy
    for original, copy in mapping.items():
        if original.parent in mapping:
            copy.parent = mapping[original.parent]
    return mapping[root]


@api
def place_placeholder(key, dimensions_m, location, rotation_deg=(0.0, 0.0, 0.0), label=None):
    """Labelled proxy box (origin at its base) standing in for a skipped asset."""
    width, depth, height = dimensions_m
    mesh = bpy.data.meshes.new(f"{key}_placeholder_mesh")
    work = bmesh.new()
    bmesh.ops.create_cube(work, size=1.0)
    bmesh.ops.scale(work, vec=(width, depth, height), verts=work.verts)
    bmesh.ops.translate(work, vec=(0, 0, height / 2), verts=work.verts)
    work.to_mesh(mesh)
    work.free()
    box = bpy.data.objects.new(f"{key}_placeholder", mesh)
    collection("placeholders").objects.link(box)
    material = bpy.data.materials.get("mat_placeholder") or principled("placeholder", (1.0, 0.35, 0.05), 0.6)
    material.name = "mat_placeholder"
    mesh.materials.append(material)
    text_curve = bpy.data.curves.new(f"{key}_label", type="FONT")
    text_curve.body = label or key
    # Glyphs are about 0.6 x size wide: keep the label within the box width.
    text_curve.size = max(0.02, min(min(width, height) * 0.25, width / (0.6 * max(len(text_curve.body), 1))))
    text_curve.align_x = "CENTER"
    text = bpy.data.objects.new(f"{key}_label", text_curve)
    text.location = (0.0, -depth / 2 - 0.01, height / 2)
    text.rotation_euler = (math.radians(90), 0.0, 0.0)
    text.parent = box
    collection("placeholders").objects.link(text)
    box.location = location
    box.rotation_euler = [math.radians(d) for d in rotation_deg]
    box["kb_placeholder"] = key
    return box


@api
def look_at(obj, target):
    """Rotate ``obj`` (camera or light) so it points at the world position ``target``."""
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


@api
def camera(location, look_at_point=None, rotation_deg=None, focal_length_mm=35.0, name="camera", ortho_scale=None, sensor_width_mm=36.0):
    """Create the scene camera and make it active. Aim with ``look_at_point`` or ``rotation_deg``; a value for
    ``ortho_scale`` makes it orthographic."""
    data = bpy.data.cameras.new(name)
    data.lens = focal_length_mm
    data.sensor_width = sensor_width_mm
    if ortho_scale:
        data.type = "ORTHO"
        data.ortho_scale = ortho_scale
    data.clip_end = 1000.0
    obj = bpy.data.objects.new(name, data)
    collection("cameras").objects.link(obj)
    obj.location = location
    if look_at_point is not None:
        look_at(obj, look_at_point)
    elif rotation_deg is not None:
        obj.rotation_euler = [math.radians(d) for d in rotation_deg]
    bpy.context.scene.camera = obj
    return obj


@api
def light(kind, location, energy, color=(1.0, 1.0, 1.0), look_at_point=None, rotation_deg=None, size=1.0, name=None):
    """Add a light. kind: 'SUN' (energy = strength in W/m²), 'AREA', 'POINT' or 'SPOT' (energy in W)."""
    data = bpy.data.lights.new(name or kind.lower(), type=kind)
    data.energy = energy
    data.color = color
    if kind == "AREA":
        data.size = size
    elif kind in ("POINT", "SPOT"):
        data.shadow_soft_size = size
    elif kind == "SUN":
        data.angle = math.radians(max(size, 0.5))
    obj = bpy.data.objects.new(name or kind.lower(), data)
    collection("lights").objects.link(obj)
    obj.location = location
    if look_at_point is not None:
        look_at(obj, look_at_point)
    elif rotation_deg is not None:
        obj.rotation_euler = [math.radians(d) for d in rotation_deg]
    return obj


def _world():
    world = bpy.context.scene.world or bpy.data.worlds.new("world")
    bpy.context.scene.world = world
    tree = node_tree(world)
    tree.nodes.clear()
    return world, tree


@api
def hdri_world(path, strength=1.0, rotation_deg=0.0):
    """Node-based world lit by an equirectangular HDRI (e.g. fetch_polyhaven('hdri', id)['hdri'])."""
    world, tree = _world()
    coords = tree.nodes.new("ShaderNodeTexCoord")
    mapping = tree.nodes.new("ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(rotation_deg))
    environment = tree.nodes.new("ShaderNodeTexEnvironment")
    environment.image = bpy.data.images.load(path, check_existing=True)
    background = tree.nodes.new("ShaderNodeBackground")
    background.inputs["Strength"].default_value = strength
    output = tree.nodes.new("ShaderNodeOutputWorld")
    tree.links.new(coords.outputs["Generated"], mapping.inputs["Vector"])
    tree.links.new(mapping.outputs["Vector"], environment.inputs["Vector"])
    tree.links.new(environment.outputs["Color"], background.inputs["Color"])
    tree.links.new(background.outputs["Background"], output.inputs["Surface"])
    return world


@api
def color_world(color=(0.05, 0.05, 0.05), strength=1.0):
    """Node-based world with a flat background color."""
    world, tree = _world()
    background = tree.nodes.new("ShaderNodeBackground")
    background.inputs["Color"].default_value = (*color[:3], 1.0)
    background.inputs["Strength"].default_value = strength
    output = tree.nodes.new("ShaderNodeOutputWorld")
    tree.links.new(background.outputs["Background"], output.inputs["Surface"])
    return world


@api
def render_settings(engine=None, samples=None, view_transform="AgX", look=None, exposure=0.0, gamma=1.0, film_transparent=False):
    """Render engine ('CYCLES' or 'BLENDER_EEVEE') and color management. kitbash sets resolution itself."""
    scene = bpy.context.scene
    if engine:
        scene.render.engine = engine
    if samples:
        if scene.render.engine == "CYCLES":
            scene.cycles.samples = samples
        else:
            scene.eevee.taa_render_samples = samples
    try:
        scene.view_settings.view_transform = view_transform
        if look:
            scene.view_settings.look = look
    except TypeError:
        scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = exposure
    scene.view_settings.gamma = gamma
    scene.render.film_transparent = film_transparent


@api
def freestyle_outlines(thickness=1.5, color=(0.0, 0.0, 0.0)):
    """Enable Freestyle outlines (for the '2d' style)."""
    scene = bpy.context.scene
    scene.render.use_freestyle = True
    scene.render.line_thickness_mode = "ABSOLUTE"
    scene.render.line_thickness = thickness
    view_layer = bpy.context.view_layer
    view_layer.use_freestyle = True
    settings = view_layer.freestyle_settings
    lineset = settings.linesets[0] if settings.linesets else settings.linesets.new("outlines")
    lineset.linestyle.color = color


@api
def ground_plane(size=20.0, material=None, name="ground"):
    """Square ground plane at z=0 (optionally with ``material``)."""
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    half = size / 2
    mesh.from_pydata([(-half, -half, 0), (half, -half, 0), (half, half, 0), (-half, half, 0)], [], [(0, 1, 2, 3)])
    obj = bpy.data.objects.new(name, mesh)
    collection("environment").objects.link(obj)
    if material is not None:
        mesh.materials.append(material)
    return obj


@api
def save_scene():
    """Save the scene to ``args()['output_blend']`` with relative paths. Call it last."""
    _save(_ARGS["output_blend"])
    emit("scene", {"objects": len(bpy.data.objects), "camera": bpy.context.scene.camera.name if bpy.context.scene.camera else None})
