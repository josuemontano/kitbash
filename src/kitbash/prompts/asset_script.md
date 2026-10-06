# Task: write the Blender build script for one asset

Write a complete script that constructs a finished, reusable asset using the geometry method below.

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
$geometry

Build one Principled BSDF material per visible material part with `kb.principled(part, ...)`, adding
procedural nodes (noise, voronoi, wave, color ramp, bump) when appropriate for the requested style.
Assign materials to the appropriate components or with `kb.assign(obj, mat, where=lambda c, n: ...)`
using normalized face center `c` (0..1 on X, Y, Z) and world normal `n`. Every face needs a material.
End with `kb.save_asset(obj)`. This same built asset is rendered, inspected, critiqued and exported to USD.
Every polygon must reference at least three distinct, in-range vertex indices. Saving and export reject
malformed faces; fix the source indexing rather than deleting faces or relying on importer repair.

Keep it one straightforward top-level script (no argument parsing, no main guard), well commented,
under about 200 lines.

## Helper API (kitbash_bpy)
$api

## Output
One complete Python script in a single ```python block, nothing else.
