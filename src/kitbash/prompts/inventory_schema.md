## Inventory JSON format

World conventions: meters, Z up, the floor is z = 0, the origin is on the floor under the center of the
composition, the camera looks roughly along +Y and image right is +X. `location_m` is the BASE CENTER of
an object (where it touches its support). Rotations are Blender XYZ Euler angles in degrees; a camera with
rotation_deg [90, 0, 0] looks horizontally along +Y, [75, 0, 0] looks slightly down.

```json
{
  "scene": {
    "description": "One paragraph describing the whole shot",
    "environment": "indoor | outdoor | studio",
    "lighting": "Light sources, direction, color temperature, time of day",
    "style_notes": "Palette, mood, notable look",
    "camera": {"location_m": [0.0, -4.0, 1.6], "rotation_deg": [80.0, 0.0, 0.0], "focal_length_mm": 35}
  },
  "items": [
    {
      "id": "walnut_armchair",
      "name": "Mid-century walnut armchair",
      "description": "Shape, parts, proportions, colors and materials a modeller needs",
      "category": "furniture | lighting | decor | plant | appliance | electronics | vehicle | prop | fixture",
      "dimensions_m": {"width": 0.8, "depth": 0.75, "height": 0.9},
      "position": {
        "image_bbox": [0.12, 0.40, 0.38, 0.92],
        "location_m": [-1.2, 0.5, 0.0],
        "rotation_deg": [0.0, 0.0, 25.0]
      },
      "relationships": [{"type": "next_to", "target": "coffee_table"}],
      "materials_hint": ["walnut wood", "black leather"],
      "confidence": 0.9
    }
  ]
}
```

- `id`: unique snake_case. `image_bbox`: normalized [x0, y0, x1, y1], origin top-left (use null without an image).
- `relationships[].type`: on, next_to, inside, under, attached_to, in_front_of or behind; `target` is another item id.
- `confidence`: how sure you are what the object is (below 0.45 means "I cannot tell what this is").
- Do not list the floor, walls, ceiling, ground, sky or backdrop; the layout builds those.
- List each distinct physical object once. Repeated identical objects (e.g. 4 dining chairs) are separate
  items with ids like `dining_chair_01`, `dining_chair_02` and the same name and description.
