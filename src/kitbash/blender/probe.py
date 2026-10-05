"""Report Blender capabilities: version, Python and the options of bpy.ops.wm.usd_export."""

import sys

import bpy
import kitbash_bpy as kb

export_options = [p.identifier for p in bpy.ops.wm.usd_export.get_rna_type().properties if p.identifier != "rna_type"]
import_options = [p.identifier for p in bpy.ops.wm.usd_import.get_rna_type().properties if p.identifier != "rna_type"]
kb.emit("version", bpy.app.version_string)
kb.emit("version_tuple", list(bpy.app.version))
kb.emit("python_version", sys.version.split()[0])
kb.emit("usd_export_options", export_options)
kb.emit("usd_import_options", import_options)
