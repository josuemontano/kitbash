"""Re-import an exported USD into a clean session, check the material graphs and render it.

In ``asset`` mode the neutral studio rig is used (the same one as the .blend previews); in ``scene`` mode
the camera, lights and world that came back from the USD are used.
"""

import os
import re

import bpy
import kb_materials as km
import kb_render
import kb_usd_cameras
import kitbash_bpy as kb
from kb_files import image_path
from pxr import Usd

options = kb.args()
kb.reset_scene()
if options["mode"] == "scene":
    # The importer converts the USD dome light into the scene's world only when a world exists.
    bpy.context.scene.world = bpy.data.worlds.new("world")
import_options = {p.identifier for p in bpy.ops.wm.usd_import.get_rna_type().properties}
wanted = {
    "filepath": options["usd_path"],
    "import_materials": True,
    "import_usd_preview": True,
    "import_cameras": options["mode"] == "scene",
    "import_lights": options["mode"] == "scene",
    "create_world_material": options["mode"] == "scene",
    "property_import_mode": "USER",
    # Merging a geometry prim into its Xform transfers the Xform's placement/camera properties
    # onto the imported object. Multi-object placement roots remain separate Empty parents.
    "merge_parent_xform": True,
}
bpy.ops.wm.usd_import(**{k: v for k, v in wanted.items() if k in import_options})
scene_expectations = options.get("scene_expectations")
scene_inspection = {}
if options["mode"] == "scene":
    render = bpy.context.scene.render
    stage = Usd.Stage.Open(options["usd_path"])
    metadata = stage.GetRootLayer().customLayerData.get("kitbash", {})
    view_settings = metadata.get("view_settings", {})
    render_settings = metadata.get("render_settings", {})
    if render_settings:
        # Preserve the source raster aspect within the requested comparison budget.
        # Pixel aspect is separate: changing it would also change camera framing.
        width, height = render_settings["resolution_x"], render_settings["resolution_y"]
        scale = min(options["resolution"][0] / width, options["resolution"][1] / height)
        options["resolution"] = [max(1, round(width * scale)), max(1, round(height * scale))]
        render.pixel_aspect_x = render_settings["pixel_aspect_x"]
        render.pixel_aspect_y = render_settings["pixel_aspect_y"]
    kb_usd_cameras.normalize_imported_orthographic_scale(stage, bpy.context.scene)
    for name in ("view_transform", "look", "exposure", "gamma"):
        if name in view_settings:
            setattr(bpy.context.scene.view_settings, name, view_settings[name])
    cameras = [obj for obj in bpy.context.scene.objects if obj.type == "CAMERA"]
    if scene_expectations is not None and "camera_name" in scene_expectations:
        expected_camera = scene_expectations["camera_name"]
        camera = next((obj for obj in cameras if obj.get("kb_camera_name", obj.name) == expected_camera), None)
    else:
        camera = next((obj for obj in cameras if obj.get("kb_active_camera")), None)
        if camera is None and scene_expectations is None:
            camera = next(iter(cameras), None)
    bpy.context.scene.camera = camera
    if scene_expectations is not None:
        from inspect_scene import inspect_scene

        report, facts = inspect_scene({**options, **scene_expectations})
        scene_inspection = {"scene_report": report, "scene_facts": facts}

materials_report, missing = {}, set()
for name, expected_channels in options.get("expected", {}).items():
    material = bpy.data.materials.get(name) or next((m for m in bpy.data.materials if re.sub(r"\.\d{3}$", "", m.name) == name), None)
    if material is None:
        materials_report[name] = {"found": False, "principled": False, "missing_channels": list(expected_channels)}
        continue
    bsdfs = km.principled_nodes(material)
    missing_channels = []
    for channel in expected_channels:
        socket = bsdfs[0].inputs.get(channel) if bsdfs else None
        if socket is None or not socket.is_linked or "ShaderNodeTexImage" not in km.upstream_types(socket):
            missing_channels.append(channel)
    materials_report[name] = {
        "found": True,
        "principled": bool(bsdfs),
        "missing_channels": missing_channels,
        "node_types": sorted({n.bl_idname for n in material.node_tree.nodes}),
    }

for image in bpy.data.images:
    if image.source != "FILE" or image.packed_file is not None:
        continue
    path = image_path(image)
    if not os.path.isfile(path):
        missing.add(image.filepath)

kb_render.configure_render(options["engine"], options["samples"], options.get("device", "CPU"), options["resolution"])
if options["mode"] == "scene":
    if bpy.context.scene.camera is None and scene_expectations is None:
        raise RuntimeError("The exported USD has no camera")
    images = (
        [kb_render.render_to(os.path.join(options["output_dir"], f"{options['prefix']}_camera.png"))]
        if bpy.context.scene.camera is not None else []
    )
else:
    images = kb_render.render_views(kb_render.scene_meshes(), options["views"], options["output_dir"], options["prefix"])

kb.emit(
    "roundtrip",
    {
        "materials": materials_report,
        "materials_total": len(materials_report),
        "materials_ok": sum(1 for m in materials_report.values() if m["found"] and m["principled"] and not m["missing_channels"]),
        "missing_textures": sorted(missing),
        "images": images,
        "render_resolution": list(options["resolution"]),
        **scene_inspection,
    },
)
