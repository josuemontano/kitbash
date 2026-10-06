"""Stand-in for the ``omp`` CLI used by the integration tests.

It answers ``--version``, ``models --json`` and ``-p --mode json`` calls. Print-mode answers are chosen
from the ``<!-- kitbash-task: ... -->`` marker kitbash puts at the top of every prompt, and every call is
appended to ``$FAKE_OMP_LOG`` (JSON lines) so tests can assert which agents ran.
"""

import json
import os
import re
import sys

MODELS = ["claude-opus-5", "claude-sonnet-5", "gemini-3.8-flash", "gpt-6-sol"]
TASK = re.compile(r"<!-- kitbash-task: ([\w.]+) -->")

INVENTORY = {
    "scene": {
        "description": "A wooden crate with a ceramic mug on top, in a bright studio.",
        "environment": "studio",
        "lighting": "soft key light from the upper left",
        "style_notes": "clean product shot",
        "camera": {"location_m": [0.0, -2.5, 0.9], "rotation_deg": [80.0, 0.0, 0.0], "focal_length_mm": 50},
    },
    "items": [
        {
            "id": "wooden_crate", "name": "Wooden crate", "description": "Slatted pine storage crate with dark nail heads",
            "category": "prop", "dimensions_m": {"width": 0.6, "depth": 0.4, "height": 0.35},
            "position": {"image_bbox": [0.25, 0.45, 0.75, 0.95], "location_m": [0.0, 0.0, 0.0], "rotation_deg": [0.0, 0.0, 10.0]},
            "relationships": [], "materials_hint": ["pine wood"], "confidence": 0.92,
        },
        {
            "id": "ceramic_mug", "name": "Ceramic mug", "description": "White glazed coffee mug with a round handle",
            "category": "decor", "dimensions_m": {"width": 0.12, "depth": 0.09, "height": 0.1},
            "position": {"image_bbox": [0.42, 0.25, 0.58, 0.45], "location_m": [0.05, 0.0, 0.35], "rotation_deg": [0.0, 0.0, 0.0]},
            "relationships": [{"type": "on", "target": "wooden_crate"}], "materials_hint": ["glazed ceramic"], "confidence": 0.88,
        },
    ],
}

BUILD_SCRIPT = '''import kitbash_bpy as kb

kb.reset_scene()
obj = kb.import_mesh()
kb.decimate(obj)
kb.clean_mesh(obj)
kb.fit_dimensions(obj, mode="height")
kb.origin_to_base(obj)

body = kb.principled("body", base_color=(0.55, 0.36, 0.2), roughness=0.6)
bsdf = kb.principled_node(body)
noise = kb.add_node(body, "ShaderNodeTexNoise")
noise.inputs["Scale"].default_value = 12.0
kb.link(body, noise.outputs["Color"], bsdf.inputs["Base Color"])
kb.assign(obj, body)
kb.save_asset(obj)
'''

PROCEDURAL_SCRIPT = '''import kitbash_bpy as kb

import bpy
import math

kb.reset_scene()
parts = []
if kb.args()["slug"] == "wooden_crate":
    def plank(location, size):
        bpy.ops.mesh.primitive_cube_add(size=1, location=location)
        obj = bpy.context.object
        obj.scale = size
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        parts.append(obj)

    plank((0, 0, 0.015), (0.6, 0.4, 0.03))
    for z in (0.08, 0.19, 0.30):
        for y in (-0.185, 0.185):
            plank((0, y, z), (0.6, 0.03, 0.10))
        for x in (-0.285, 0.285):
            plank((x, 0, z), (0.03, 0.34, 0.10))
    # The scene places its mug on top, so this storage crate needs a slatted lid.
    for y in (-0.16, -0.08, 0, 0.08, 0.16):
        plank((0, y, 0.335), (0.6, 0.075, 0.03))
    body = kb.principled("pine", base_color=(0.55, 0.36, 0.2), roughness=0.6)
else:
    # Closed, hollow cup with an annular rim and a curved handle.
    vertices = []
    for radius, z in ((0.045, 0), (0.045, 0.1), (0.039, 0.1), (0.039, 0.008)):
        vertices.extend((radius * math.cos(i * math.tau / 32), radius * math.sin(i * math.tau / 32), z) for i in range(32))
    faces = []
    for ring in range(3):
        for i in range(32):
            j = (i + 1) % 32
            faces.append((ring * 32 + i, ring * 32 + j, (ring + 1) * 32 + j, (ring + 1) * 32 + i))
    faces.extend((tuple(reversed(range(32))), tuple(range(96, 128))))
    mesh = bpy.data.meshes.new("cup")
    mesh.from_pydata(vertices, [], faces)
    cup = bpy.data.objects.new("cup", mesh)
    bpy.context.collection.objects.link(cup)
    parts.append(cup)
    bpy.ops.mesh.primitive_torus_add(major_radius=0.025, minor_radius=0.007, location=(0.055, 0, 0.053), rotation=(math.pi / 2, 0, 0))
    parts.append(bpy.context.object)
    body = kb.principled("ceramic", base_color=(0.9, 0.9, 0.9), roughness=0.25)
bpy.ops.object.select_all(action="DESELECT")
for obj in parts:
    obj.select_set(True)
bpy.context.view_layer.objects.active = parts[0]
bpy.ops.object.join()
obj = bpy.context.object
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
kb.fit_dimensions(obj, mode="exact")
kb.origin_to_base(obj)
kb.assign(obj, body)
kb.save_asset(obj)
'''

LAYOUT_SCRIPT = '''import kitbash_bpy as kb

PLACEMENTS = {placements}
PLACEHOLDERS = {placeholders}

kb.reset_scene()
kb.ground_plane(size=10.0, material=kb.principled("floor", base_color=(0.5, 0.5, 0.52), roughness=0.8))
for key, location, rotation in PLACEMENTS:
    kb.place_asset(key, location, rotation)
for key, size, location, rotation, label in PLACEHOLDERS:
    kb.place_placeholder(key, size, location, rotation, label=label)
kb.camera((0.0, -2.5, 0.9), rotation_deg=(80.0, 0.0, 0.0), focal_length_mm=50)
kb.light("SUN", (0.0, 0.0, 5.0), energy=3.0, rotation_deg=(40.0, 0.0, 30.0))
kb.color_world((0.6, 0.6, 0.65), 0.8)
kb.render_settings(engine="CYCLES", samples=8)
kb.save_scene()
'''


def section_rows(prompt: str, heading: str) -> list[dict]:
    """JSON-lines rows under a '## heading' section of the prompt."""
    match = re.search(rf"^## {re.escape(heading)}.*?\n(.*?)(?=^## |\Z)", prompt, re.MULTILINE | re.DOTALL)
    rows = []
    for line in (match.group(1) if match else "").splitlines():
        line = line.strip()
        if line.startswith("{"):
            rows.append(json.loads(line))
    return rows


def layout_script(prompt: str) -> str:
    approved = {row["key"] for row in section_rows(prompt, "Approved assets")}
    skipped = {row["key"]: row for row in section_rows(prompt, "Skipped assets")}
    placements, placeholders = [], []
    for row in section_rows(prompt, "Inventory positions"):
        if row["asset_key"] in approved:
            placements.append((row["asset_key"], row["location_m"], row["rotation_deg"]))
        elif row["id"] in skipped:
            placeholders.append((row["id"], row["dimensions_m"], row["location_m"], row["rotation_deg"], row["name"]))
    return LAYOUT_SCRIPT.format(placements=repr(placements), placeholders=repr(placeholders))


def respond(task: str, prompt: str) -> str:
    if task == "preflight.ping":
        return "pong"
    if task.startswith("breakdown.analyze"):
        inventory = INVENTORY
        if reference := os.environ.get("FAKE_OMP_REFERENCE"):
            inventory = {**INVENTORY, "items": [{**item, "user_reference": reference} for item in INVENTORY["items"]]}
        return "```json\n" + json.dumps(inventory) + "\n```"
    if ".critic." in task:
        return json.dumps({"summary": "Inspect the supplied rubric results and evidence.", "edits": []})
    if task == "modelling.reference.select":
        return json.dumps({"choice": 1, "reason": "the crop shows the whole object", "background": "other"})
    if task == "modelling.script":
        script = PROCEDURAL_SCRIPT if "PROCEDURAL CONSTRUCTION:" in prompt else BUILD_SCRIPT
        return "```python\n" + script + "```"
    if task == "layout.script":
        return "```python\n" + layout_script(prompt) + "```"
    if task.endswith(".patch"):
        return "```diff\n--- a/script.py\n+++ b/script.py\n@@ -1,2 +1,3 @@\n import kitbash_bpy as kb\n+# patched by the fake\n \n```"
    raise SystemExit(f"fake omp: no answer for task {task!r}")


def emit(text: str, model: str) -> None:
    usage = {"input": len(text) // 4 + 100, "output": len(text) // 4, "cacheRead": 0, "cacheWrite": 0, "cost": {"total": 0.0001}}
    message = {"role": "assistant", "content": [{"type": "text", "text": text}], "provider": "fake", "model": model, "usage": usage, "stopReason": "stop"}
    print(json.dumps({"type": "session", "id": "fake"}))
    print(json.dumps({"type": "message_end", "message": message}))


def main(argv: list[str]) -> int:
    if argv == ["--version"]:
        print("omp/0.0.0-fake")
        return 0
    if argv[:2] == ["models", "--json"]:
        models = [{"provider": "fake", "kind": "chat", "id": m, "selector": f"fake/{m}", "name": m, "input": ["text", "image"]} for m in MODELS]
        print(json.dumps({"models": models}))
        return 0
    if "-p" not in argv:
        print("fake omp: unsupported call", argv, file=sys.stderr)
        return 2
    prompt = argv[-1]
    model = argv[argv.index("--model") + 1]
    marker = TASK.search(prompt)
    task = marker.group(1) if marker else "unknown"
    attachments = [a[1:] for a in argv if a.startswith("@")]
    if log := os.environ.get("FAKE_OMP_LOG"):
        with open(log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"task": task, "model": model, "attachments": attachments}) + "\n")
    emit(respond(task, prompt), model)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
