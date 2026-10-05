"""Render configuration and the neutral studio rig used for asset previews and USD round-trip renders.

The rig is built in memory only (never saved into an asset), so the .blend render and the round-trip
render of the re-imported USD see exactly the same camera, lights and floor.
"""

import math
import os

import bpy
import kitbash_bpy as kb
from mathutils import Vector

VIEWS = {
    # name: (azimuth degrees from the front (-Y) towards +X, elevation degrees)
    "front": (0.0, 6.0),
    "front_3q": (-35.0, 18.0),
    "side": (90.0, 8.0),
    "back_3q": (145.0, 22.0),
    "top": (0.0, 75.0),
}
RIG_PREFIX = "kb_studio"


def configure_render(engine="CYCLES", samples=32, device="CPU", resolution=(768, 768)):
    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.resolution_x, scene.render.resolution_y = int(resolution[0]), int(resolution[1])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    if engine == "CYCLES":
        scene.cycles.samples = int(samples)
        scene.cycles.use_denoising = True
        scene.cycles.device = _cycles_device(device)
    else:
        scene.eevee.taa_render_samples = int(samples)


def _cycles_device(device):
    if device != "GPU":
        return "CPU"
    preferences = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("METAL", "OPTIX", "CUDA", "HIP", "ONEAPI"):
        try:
            preferences.compute_device_type = backend
        except TypeError:
            continue
        preferences.get_devices()
        if any(d.type == backend for d in preferences.devices):
            for d in preferences.devices:
                d.use = True
            return "GPU"
    return "CPU"


def render_to(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return path


def scene_meshes():
    return [o for o in bpy.context.scene.objects if o.type == "MESH" and not o.name.startswith(RIG_PREFIX)]


def studio_rig(objects, view, focal_length_mm=50.0):
    """Neutral studio: gray world, key/fill/rim area lights, floor, and a camera framing ``objects``."""
    lo, hi = kb.world_bounds(objects)
    center = (lo + hi) / 2
    radius = max((hi - lo).length / 2, 0.05)
    _studio_world()
    _studio_lights(center, radius, lo.z)
    _studio_floor(center, radius, lo.z)
    azimuth, elevation = VIEWS[view]
    fov = 2 * math.atan(18.0 / focal_length_mm)
    distance = radius / math.sin(fov / 2) * 1.15
    direction = Vector(
        (
            math.sin(math.radians(azimuth)) * math.cos(math.radians(elevation)),
            -math.cos(math.radians(azimuth)) * math.cos(math.radians(elevation)),
            math.sin(math.radians(elevation)),
        )
    )
    name = f"{RIG_PREFIX}_camera"
    camera = bpy.data.objects.get(name)
    if camera is None:
        camera = bpy.data.objects.new(name, bpy.data.cameras.new(name))
        bpy.context.scene.collection.objects.link(camera)
    camera.data.lens = focal_length_mm
    camera.data.clip_start = max(radius / 100, 0.001)
    camera.data.clip_end = distance + radius * 50
    camera.location = center + direction * distance
    kb.look_at(camera, center)
    bpy.context.scene.camera = camera
    return camera


def _studio_world():
    world = bpy.data.worlds.get(f"{RIG_PREFIX}_world") or bpy.data.worlds.new(f"{RIG_PREFIX}_world")
    tree = kb.node_tree(world)
    tree.nodes.clear()
    background = tree.nodes.new("ShaderNodeBackground")
    background.inputs["Color"].default_value = (0.3, 0.3, 0.3, 1.0)
    background.inputs["Strength"].default_value = 0.6
    output = tree.nodes.new("ShaderNodeOutputWorld")
    tree.links.new(background.outputs["Background"], output.inputs["Surface"])
    bpy.context.scene.world = world
    scene = bpy.context.scene
    try:
        scene.view_settings.view_transform = "AgX"
    except TypeError:
        scene.view_settings.view_transform = "Standard"


def _studio_lights(center, radius, floor_z):
    setups = {
        "key": ((-1.6, -2.0, 2.2), 600.0, 1.2),
        "fill": ((2.2, -1.2, 1.0), 180.0, 1.6),
        "rim": ((0.6, 2.4, 2.0), 300.0, 1.0),
    }
    for role, (offset, energy, size) in setups.items():
        name = f"{RIG_PREFIX}_{role}"
        light = bpy.data.objects.get(name)
        if light is None:
            light = bpy.data.objects.new(name, bpy.data.lights.new(name, type="AREA"))
            bpy.context.scene.collection.objects.link(light)
        light.data.size = size * radius * 2
        light.data.energy = energy * radius * radius
        light.location = center + Vector(offset) * radius * 2
        kb.look_at(light, center)


def _studio_floor(center, radius, floor_z):
    name = f"{RIG_PREFIX}_floor"
    floor = bpy.data.objects.get(name)
    if floor is None:
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], [], [(0, 1, 2, 3)])
        floor = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(floor)
        material = bpy.data.materials.new(f"{RIG_PREFIX}_floor_material")
        bsdf = kb.principled_node(material)
        bsdf.inputs["Base Color"].default_value = (0.42, 0.42, 0.42, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.8
        mesh.materials.append(material)
    floor.location = (center.x, center.y, floor_z - radius * 0.001)
    floor.scale = (radius * 25, radius * 25, 1.0)


def render_views(objects, views, output_dir, prefix):
    """Render each named view of ``objects``; returns the image paths."""
    paths = []
    for view in views:
        studio_rig(objects, view)
        paths.append(render_to(os.path.join(output_dir, f"{prefix}_{view}.png")))
    return paths
