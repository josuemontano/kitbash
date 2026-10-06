"""Content manifests for trusted publishers; not a sandbox for generated Python."""

import hashlib
import stat
from collections.abc import Iterable
from pathlib import Path


def file_hash(path: Path) -> str:
    """Hash a regular file without accepting symbolic links or special files."""
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"Artifact is not a regular file: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot(root: Path, paths: Iterable[Path]) -> dict[str, str]:
    """Hash selected files/trees, rejecting escapes and links, including directory links.

    Keys are relative to ``root`` so a verified tree can be copied and checked again.
    Repeating this operation with the same selections also detects added/deleted files.
    """
    source_root = root.absolute()
    if source_root.is_symlink():
        raise ValueError(f"Artifact root must not be a symbolic link: {root}")
    root = source_root.resolve()
    hashes: dict[str, str] = {}

    def visit(path: Path) -> None:
        relative = path.absolute().relative_to(source_root)
        path = root / relative
        if ".." in relative.parts or path.resolve() != path:
            raise ValueError(f"Artifact escapes its owner or contains symbolic links: {path}")
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            for child in sorted(path.iterdir()):
                visit(source_root / child.relative_to(root))
        else:
            hashes[relative.as_posix()] = file_hash(path)

    for path in paths:
        visit(path)
    return hashes
