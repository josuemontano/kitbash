# Task: choose the reference image for an asset

The object to model:
- name: $name
- description: $description
- category: $category
- materials: $materials

The first attachment is a contact sheet of $count candidates numbered from 1: $candidates
$context

The chosen image is fed to an image-to-3D model (Trellis) that only reconstructs the object's SHAPE:
it removes the background itself, and colors, graphics and materials are rebuilt afterwards from the
description. So judge candidates on geometry:

- Required: the right kind of object, shown whole (not cropped), not hidden behind other things, and
  clearly the main subject of the photo.
- Prefer: a silhouette and proportions close to the description, a three-quarter or front view, a
  plain background, a sharp image.
- Ignore: color, printed graphics, lighting, and busy backgrounds when the object itself is fully visible.

Pick the closest usable candidate even if it differs in details. Answer null only if no candidate
shows this kind of object whole.

Respond with JSON only:
{"choice": <candidate number or null>, "reason": "one sentence", "background": "white | black | transparent | other"}
