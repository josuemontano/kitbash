"""Copy local file images beside the scene and save relative paths.

Assembly bundles linked libraries and their textures under scene/assets before this runs;
their image paths belong to those libraries and must not be rewritten relative to the scene.
"""

import os

import bpy
import kitbash_bpy as kb
from kb_files import copy_texture, image_path

options = kb.args()
blend_dir = os.path.abspath(os.path.dirname(bpy.data.filepath))
target_dir = os.path.join(blend_dir, options.get("textures_subdir", "textures"))
copied, missing = [], []
for image in bpy.data.images:
    if image.source not in ("FILE", "SEQUENCE", "TILED") or image.packed_file is not None or not image.filepath:
        continue
    absolute = image_path(image)
    if not os.path.isfile(absolute):
        missing.append(image.filepath)
        continue
    if image.library is not None:
        continue
    if os.path.commonpath([absolute, blend_dir]) == blend_dir:
        continue
    if image.source != "FILE":
        raise RuntimeError(f"Cannot localize multi-file {image.source} image {image.name!r}: {image.filepath}")
    destination = copy_texture(absolute, target_dir)
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
