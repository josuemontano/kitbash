import hashlib
import os
from pathlib import Path

from kitbash.blender.kb_files import copy_texture


def test_same_basename_textures_keep_distinct_contents(tmp_path):
    first = tmp_path / "red" / "albedo.png"
    second = tmp_path / "blue" / "albedo.png"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"red texture")
    second.write_bytes(b"blue texture")
    target = tmp_path / "textures"

    copied_first = Path(copy_texture(first, target))
    copied_second = Path(copy_texture(second, target))

    assert copied_first != copied_second
    assert copied_first.read_bytes() == first.read_bytes()
    assert copied_second.read_bytes() == second.read_bytes()
    assert copied_first.name == hashlib.sha256(first.read_bytes()).hexdigest() + ".png"
    assert copied_second.name == hashlib.sha256(second.read_bytes()).hexdigest() + ".png"


def test_identical_textures_reuse_copy_without_rewriting(tmp_path):
    first = tmp_path / "first.exr"
    second = tmp_path / "second.exr"
    first.write_bytes(b"same texture")
    second.write_bytes(first.read_bytes())
    target = tmp_path / "textures"
    destination = Path(copy_texture(first, target))
    os.utime(destination, ns=(1_000_000_000, 1_000_000_000))
    modified = destination.stat().st_mtime_ns

    assert Path(copy_texture(second, target)) == destination
    assert Path(copy_texture(destination, target)) == destination
    assert destination.read_bytes() == b"same texture"
    assert destination.stat().st_mtime_ns == modified


def test_changed_texture_creates_new_copy_without_corrupting_old(tmp_path):
    source = tmp_path / "albedo.png"
    target = tmp_path / "textures"
    source.write_bytes(b"original texture")
    original = Path(copy_texture(source, target))

    source.write_bytes(b"changed texture")
    changed = Path(copy_texture(source, target))

    assert original != changed
    assert original.read_bytes() == b"original texture"
    assert changed.read_bytes() == b"changed texture"


def test_existing_digest_filename_does_not_hide_corrupt_contents(tmp_path):
    source = tmp_path / "albedo.png"
    target = tmp_path / "textures"
    source.write_bytes(b"correct texture")
    destination = Path(copy_texture(source, target))
    destination.write_bytes(b"corrupted texture")

    assert Path(copy_texture(source, target)) == destination
    assert destination.read_bytes() == b"correct texture"
