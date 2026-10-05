# Task: write the Blender build script for one asset

Trellis generated an untextured mesh of the object from the attached reference image. Write the script
that turns it into a finished, reusable asset.

## Asset
- slug: `$slug`
- name: $name
- description: $description
- category: $category
- target dimensions (width, depth, height in meters): $dimensions
- materials hint: $materials
- target style of the shot: $style
- user feedback to honour: $feedback

## Naming convention
$naming

## Optional Poly Haven textures
Use `kb.fetch_polyhaven("texture", id)` with `kb.pbr_textures(mat, maps)` only when a photographic
texture clearly beats procedural nodes. Candidates matching the materials hint: $textures

## What the script must do
1. `kb.reset_scene()`, then `obj = kb.import_mesh()`.
2. `kb.decimate(obj)` and `kb.clean_mesh(obj)`.
3. Fix the orientation with `kb.rotate(obj, (x, y, z))` if the object is not upright with its front
   facing -Y (Trellis meshes usually are already upright).
4. Real-world scale with `kb.fit_dimensions(obj, mode=...)` ('height' for most objects), then `kb.origin_to_base(obj)`.
5. Build one Principled BSDF material per visible material part with `kb.principled(part, ...)`, adding
   procedural nodes (noise, voronoi, wave, color ramp, bump) for wood grain, fabric weave, wear and so on.
   Match the reference colors. Assign parts with `kb.assign(obj, mat, where=lambda c, n: ...)` using the
   normalized face center `c` (0..1 on X, Y, Z) and world normal `n`; assign a base material to all faces first.
6. End with `kb.save_asset(obj)`.

Keep it one straightforward top-level script (no argument parsing, no main guard), well commented,
under about 200 lines.

## Helper API (kitbash_bpy)
$api

## Output
One complete Python script in a single ```python block, nothing else.
