"""Build the asset from a retopologized Trellis mesh: import, clean, scale, ground, one flat material, save. No LLM-written code."""

import kitbash_bpy as kb

options = kb.args()
kb.reset_scene()
obj = kb.import_mesh()
kb.decimate(obj)  # keeps a TriFlow mesh untouched; reduces a raw mesh only when --retopology decimate was chosen
kb.clean_mesh(obj)
kb.fit_dimensions(obj, mode=options.get("fit_mode", "height"))
kb.origin_to_base(obj)
material = kb.principled("body", base_color=tuple(options["base_color"]), roughness=0.55)
kb.assign(obj, material)
kb.save_asset(obj)
