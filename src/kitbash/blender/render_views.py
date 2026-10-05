"""Render studio views of every mesh in the open .blend (asset previews and USD round-trip renders)."""

import kb_render
import kitbash_bpy as kb

options = kb.args()
kb_render.configure_render(options["engine"], options["samples"], options.get("device", "CPU"), options["resolution"])
meshes = kb_render.scene_meshes()
if not meshes:
    raise RuntimeError("The file has no mesh objects to render")
kb.emit("images", kb_render.render_views(meshes, options["views"], options["output_dir"], options["prefix"]))
