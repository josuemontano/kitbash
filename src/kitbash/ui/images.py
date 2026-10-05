"""Show images in the terminal (kitty or iTerm2 inline image protocols) or fall back to the file path."""

import base64
import contextlib
import io
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from PIL import Image
from rich.console import Console

CHUNK = 4096


def detect_protocol(env: dict[str, str] | None = None) -> str | None:
    env = dict(os.environ if env is None else env)
    program = env.get("TERM_PROGRAM", "")
    if env.get("TERM") == "xterm-kitty" or "KITTY_WINDOW_ID" in env or program.lower() == "ghostty":
        return "kitty"
    if program in ("iTerm.app", "WezTerm") or env.get("LC_TERMINAL") == "iTerm2":
        return "iterm2"
    return None


def kitty_sequence(png: bytes, columns: int) -> str:
    payload = base64.standard_b64encode(png).decode()
    chunks = [payload[i : i + CHUNK] for i in range(0, len(payload), CHUNK)] or [""]
    parts = []
    for index, chunk in enumerate(chunks):
        more = 1 if index < len(chunks) - 1 else 0
        header = f"f=100,a=T,c={columns},m={more}" if index == 0 else f"m={more}"
        parts.append(f"\x1b_G{header};{chunk}\x1b\\")
    return "".join(parts)


def iterm2_sequence(png: bytes, columns: int) -> str:
    payload = base64.standard_b64encode(png).decode()
    return f"\x1b]1337;File=inline=1;width={columns};preserveAspectRatio=1;size={len(png)}:{payload}\x07"


class ImagePresenter:
    def __init__(self, console: Console, *, enabled: bool = True, open_files: bool = True, protocol: str | None = "auto") -> None:
        self._console = console
        self._enabled = enabled
        self._open_files = open_files
        self._protocol = detect_protocol() if protocol == "auto" else protocol

    def show(self, paths: Sequence[Path], columns: int = 60) -> None:
        existing = [p for p in paths if p and Path(p).is_file()]
        if not self._enabled or not existing:
            return
        if self._protocol and self._console.is_terminal:
            for path in existing:
                self._console.print(f"[dim]{path}[/dim]")
                writer = kitty_sequence if self._protocol == "kitty" else iterm2_sequence
                self._console.file.write(writer(_png_bytes(path), columns) + "\n")
                self._console.file.flush()
            return
        for path in existing:
            self._console.print(f"Preview: [link=file://{path}]{path}[/link]")
        if self._open_files:
            _open(existing[0])


def _png_bytes(path: Path, max_side: int = 1024) -> bytes:
    with Image.open(path) as image:
        image = image.convert("RGBA") if image.mode in ("RGBA", "LA", "P") else image.convert("RGB")
        image.thumbnail((max_side, max_side))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def _open(path: Path) -> None:
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    with contextlib.suppress(OSError):
        subprocess.Popen([opener, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
