"""Render the open scene through its active camera."""

import bpy
import kb_render
import kitbash_bpy as kb

options = kb.args()
scene = bpy.context.scene
if scene.camera is None:
    cameras = [o for o in scene.objects if o.type == "CAMERA"]
    if not cameras:
        raise RuntimeError("The scene has no camera")
    scene.camera = cameras[0]
engine = options.get("engine") or scene.render.engine
kb_render.configure_render(engine, options["samples"], options.get("device", "GPU"), options["resolution"])
kb.emit("images", [kb_render.render_to(options["output_path"])])
