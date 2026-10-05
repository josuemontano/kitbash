"""Make the open .blend self-contained: copy external images (HDRIs, downloaded textures) next to it,
reference everything with relative paths and save."""

import os
import shutil

import bpy
import kitbash_bpy as kb

options = kb.args()
blend_dir = os.path.dirname(bpy.data.filepath)
target_dir = os.path.join(blend_dir, options.get("textures_subdir", "textures"))
copied, missing = [], []
for image in bpy.data.images:
    if image.source not in ("FILE", "SEQUENCE", "TILED") or image.packed_file is not None or not image.filepath:
        continue
    absolute = os.path.normpath(bpy.path.abspath(image.filepath))
    if not os.path.isfile(absolute):
        missing.append(image.filepath)
        continue
    if os.path.commonpath([absolute, blend_dir]) == blend_dir:
        continue
    os.makedirs(target_dir, exist_ok=True)
    destination = os.path.join(target_dir, os.path.basename(absolute))
    if not os.path.isfile(destination):
        shutil.copy2(absolute, destination)
    image.filepath = bpy.path.relpath(destination)
    copied.append(os.path.relpath(destination, blend_dir))
for library in bpy.data.libraries:
    absolute = os.path.normpath(bpy.path.abspath(library.filepath))
    if not os.path.isfile(absolute):
        missing.append(library.filepath)
bpy.ops.file.make_paths_relative()
bpy.context.preferences.filepaths.save_version = 0  # no .blend1 backups
bpy.ops.wm.save_mainfile()
kb.emit("localized", {"copied": copied, "missing": missing})
