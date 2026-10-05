# Task: design an inventory from a text prompt

There is no reference image. Design the set for this shot so it can be rebuilt in Blender one asset at
a time. Target style: $style.

## Prompt
$prompt

List every object the shot needs (at most $max_items), from most to least important, with specific
names and descriptions detailed enough to search for reference photos and to model from. Choose a
camera that frames the composition well. Set `image_bbox` to null for every item.

<<include inventory_schema>>

Respond with the JSON object only.
