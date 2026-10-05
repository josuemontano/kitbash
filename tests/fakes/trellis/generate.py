"""Stand-in for trellis-mac's generate.py: writes a small Y-up cylinder OBJ at ``<output>.obj``.

Environment knobs for tests:
- ``FAKE_TRELLIS_SLEEP_<name>``: seconds to wait before producing ``<name>`` (to stagger assets).
- ``FAKE_TRELLIS_FAIL_<name>``: number of attempts for ``<name>`` that fail before one succeeds.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path


def cylinder_obj(segments: int = 24, radius: float = 0.5, height: float = 1.0) -> str:
    lines = []
    for ring_y in (-height / 2, height / 2):
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            lines.append(f"v {radius * math.cos(angle):.5f} {ring_y:.5f} {radius * math.sin(angle):.5f}")
    lines.append(f"v 0 {-height / 2} 0")
    lines.append(f"v 0 {height / 2} 0")
    bottom_center, top_center = 2 * segments + 1, 2 * segments + 2
    for i in range(segments):
        a, b = i + 1, (i + 1) % segments + 1
        c, d = a + segments, b + segments
        lines += [f"f {a} {b} {d}", f"f {a} {d} {c}", f"f {bottom_center} {b} {a}", f"f {top_center} {c} {d}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--pipeline-type")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-texture", action="store_true")
    args = parser.parse_args()
    if not os.path.exists(args.image):
        print(f"Error: {args.image} not found")
        return 1
    output = Path(args.output)
    name = output.name
    time.sleep(float(os.environ.get(f"FAKE_TRELLIS_SLEEP_{name}", "0")))
    counter = output.with_suffix(".attempts")
    attempts = int(counter.read_text()) + 1 if counter.exists() else 1
    counter.write_text(str(attempts))
    if attempts <= int(os.environ.get(f"FAKE_TRELLIS_FAIL_{name}", "0")):
        print("GPU watchdog killed the decoder (fake)", file=sys.stderr)
        return 2
    output.with_suffix(".obj").write_text(cylinder_obj(), encoding="utf-8")
    print(f"Saved: {output}.obj (steps={args.steps}, pipeline={args.pipeline_type}, seed={args.seed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
