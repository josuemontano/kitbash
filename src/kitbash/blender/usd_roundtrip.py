"""Re-import an exported USD into a clean session, check the material graphs and render it.

In ``asset`` mode the neutral studio rig is used (the same one as the .blend previews); in ``scene`` mode
the camera, lights and world that came back from the USD are used.
"""

import os
import re

import bpy
import kb_materials as km
import kb_render
import kitbash_bpy as kb

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
}
bpy.ops.wm.usd_import(**{k: v for k, v in wanted.items() if k in import_options})

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
    path = bpy.path.abspath(image.filepath)
    if not os.path.isfile(path):
        missing.add(image.filepath)

kb_render.configure_render(options["engine"], options["samples"], options.get("device", "CPU"), options["resolution"])
if options["mode"] == "scene":
    cameras = [o for o in bpy.context.scene.objects if o.type == "CAMERA"]
    if not cameras:
        raise RuntimeError("The exported USD has no camera")
    bpy.context.scene.camera = cameras[0]
    images = [kb_render.render_to(os.path.join(options["output_dir"], f"{options['prefix']}_camera.png"))]
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
    },
)
