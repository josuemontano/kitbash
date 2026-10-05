# Task: write the Blender layout script for the whole shot

Place every approved asset, add placeholders for skipped ones, and set the camera, lights and world
for the requested style. The attached image is the reference (if any).

## Shot
$scene

## Style: $style
$style_guidance
Render engine for this style: $engine

## Approved assets (use these keys with kb.place_asset)
$assets

## Skipped assets (use kb.place_placeholder with these keys)
$placeholders

## Inventory positions and relationships (meters, Z up, base centers)
$inventory

## Poly Haven HDRI candidates for this environment
$hdris
Use `kb.hdri_world(kb.fetch_polyhaven("hdri", id, "2k")["hdri"], strength, rotation_deg)` when the style calls for an HDRI.

## User feedback to honour
$feedback

## What the script must do
1. `kb.reset_scene()`.
2. Build the environment the inventory leaves out (floor or ground, walls if the shot needs them) with
   simple node-based Principled materials; `kb.ground_plane()` is a good start.
3. Place every approved asset once per inventory item with `kb.place_asset(key, location, rotation_deg)`.
   Respect relationships: an object `on` another sits at the top of its support (use `kb.dimensions()`).
4. Skipped assets get `kb.place_placeholder(key, dimensions_m, location, rotation_deg, label=name)`.
5. Camera with `kb.camera(...)` matching the reference framing, lights with `kb.light(...)`, world with
   `kb.hdri_world(...)` or `kb.color_world(...)`, and `kb.render_settings(engine=...)` for the style.
6. End with `kb.save_scene()`.

Keep it one straightforward top-level script, well commented.

## Helper API (kitbash_bpy)
$api

## Output
One complete Python script in a single ```python block, nothing else.
