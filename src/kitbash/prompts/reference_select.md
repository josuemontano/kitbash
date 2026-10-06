# Task: choose the reference image for an asset

The object to model:
- name: $name
- description: $description
- category: $category
- materials: $materials

The first attachment is a contact sheet of $count candidates numbered from 1: $candidates
$context

The chosen image is fed to an image-to-3D model (Trellis) that only reconstructs the object's SHAPE;
colors, graphics and materials are rebuilt afterwards from the description. So judge candidates on geometry:

- Required: the right kind of object, shown whole (not cropped), not hidden behind other things, clearly
  the main subject, and ISOLATED: no background, or a plain pure/neutral background (white, grey, black).
  Reject any candidate with a scene, surface, people or clutter behind the object.
- Prefer: a silhouette and proportions close to the description, a three-quarter or front view, a sharp image.
- Ignore: color, printed graphics and lighting.

Pick the closest usable candidate even if it differs in details. Answer null if no candidate shows this kind
of object whole and isolated.

Respond with JSON only:
{"choice": <candidate number or null>, "reason": "one sentence", "background": "white | black | transparent | other"}
