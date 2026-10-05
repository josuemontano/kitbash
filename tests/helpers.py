"""Test helpers: fake tool wiring and synthetic reference images."""

import json
import shutil
import stat
import sys
from pathlib import Path

import pytest
import tomli_w
from PIL import Image, ImageDraw

FAKES = Path(__file__).parent / "fakes"

requires_blender = pytest.mark.skipif(shutil.which("blender") is None, reason="Blender is not installed")


def fake_omp(directory: Path) -> Path:
    """Executable wrapper that runs the fake omp with the test interpreter."""
    wrapper = directory / "omp"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKES / "fake_omp.py"}" "$@"\n', encoding="utf-8")
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    return wrapper


def reference_image(path: Path, *, tint: int = 0) -> Path:
    """A crate with a mug on it, roughly where the fake inventory's bounding boxes say."""
    image = Image.new("RGB", (512, 512), (235, 235, 235))
    draw = ImageDraw.Draw(image)
    draw.rectangle((128, 230, 384, 486), fill=(150 + tint, 100, 55))
    for y in range(250, 480, 40):
        draw.line((128, y, 384, y), fill=(110, 70, 40), width=4)
    draw.rectangle((215, 128, 297, 230), fill=(250, 250, 250), outline=(180, 180, 180))
    image.save(path)
    return path


def write_test_config(tmp: Path, **sections: dict) -> Path:
    """A config TOML that points kitbash at the fakes, tiny renders and a private backlot."""
    base = {
        "paths": {"backlot": str(tmp / "backlot"), "trellis": str(FAKES / "trellis"), "downloads": str(tmp / "downloads")},
        "tools": {"omp": str(fake_omp(tmp)), "blender": "blender", "trellis_python": sys.executable},
        "omp": {"timeout_s": 60, "retries": 0},
        "embedding": {"backend": "hashing", "dimensions": 64},
        "reference": {"providers": ["input_crop"]},
        "polyhaven": {"enabled": False},
        "trellis": {"timeout_s": 60, "retries": 2},
        "retopology": {"method": "decimate"},
        "critic": {"max_cycles": 2},
        "blender": {
            "timeout_s": 300, "preview_resolution": [128, 128], "preview_samples": 4, "preview_views": ["front_3q"],
            "final_resolution": [192, 108], "final_samples": 4, "max_faces": 20000,
        },
        "usd": {"roundtrip_resolution": [96, 96], "roundtrip_samples": 4, "bake_resolution": 128, "bake_samples": 2},
        "ui": {"show_previews": False},
    }
    for name, values in sections.items():
        base.setdefault(name, {}).update(values)
    path = tmp / "config.toml"
    path.write_text(tomli_w.dumps(base), encoding="utf-8")
    return path


def omp_calls(log: Path) -> list[dict]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
