"""Library-aware image paths and content-addressed texture copies."""

import hashlib
import os
import shutil


def image_path(image, base_dir=None):
    """Resolve a file image against its owning library, or the current blend."""
    import bpy

    path = image.filepath or image.filepath_raw
    if not path:
        return ""
    return os.path.abspath(bpy.path.abspath(path, start=base_dir, library=image.library))


def _digest(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def copy_texture(path, target_dir):
    """Copy one file by its full SHA-256 and extension; never reuse a basename."""
    source = os.path.abspath(path)
    digest = _digest(source)
    destination = os.path.join(os.path.abspath(target_dir), digest + os.path.splitext(source)[1])
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if source != destination and (not os.path.isfile(destination) or _digest(destination) != digest):
        shutil.copy2(source, destination)
    return destination
