"""Unified diffs: create, apply (tolerant to wrong line numbers and whitespace) and fingerprint."""

import difflib
import hashlib
import re
from collections.abc import Callable

from attrs import frozen

from kitbash.errors import PatchError

_HUNK_HEADER = re.compile(r"^@@\s*-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s*@@")
_FILE_HEADERS = ("--- ", "+++ ", "diff --git", "index ", "new file mode", "deleted file mode")


@frozen
class Hunk:
    old_start: int | None  # 1-based, None when the header had no numbers
    before: tuple[str, ...]  # context + removed lines
    after: tuple[str, ...]  # context + added lines


def make_diff(old: str, new: str, path: str = "script.py") -> str:
    lines = difflib.unified_diff(
        _with_newline(old).splitlines(keepends=True),
        _with_newline(new).splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    return "".join(lines)


def parse_diff(diff: str) -> list[Hunk]:
    hunks: list[Hunk] = []
    before: list[str] = []
    after: list[str] = []
    start: int | None = None
    in_hunk = False

    def flush() -> None:
        if in_hunk and (before or after):
            hunks.append(Hunk(start, tuple(before), tuple(after)))

    for line in diff.splitlines():
        if line.startswith("@@"):
            flush()
            match = _HUNK_HEADER.match(line)
            start = int(match.group(1)) if match else None
            before, after, in_hunk = [], [], True
        elif not in_hunk or line.startswith(_FILE_HEADERS) or line.startswith("\\"):
            continue
        elif line.startswith("-"):
            before.append(line[1:])
        elif line.startswith("+"):
            after.append(line[1:])
        else:
            context = line[1:] if line.startswith(" ") else line
            before.append(context)
            after.append(context)
    flush()
    if not hunks:
        raise PatchError("The diff contains no hunks")
    return hunks


def apply_diff(text: str, diff: str) -> str:
    """Apply every hunk of ``diff`` to ``text``. Raises PatchError naming the hunk that does not fit."""
    lines = text.splitlines()
    cursor = 0
    offset = 0
    for number, hunk in enumerate(parse_diff(diff), start=1):
        expected = (hunk.old_start - 1 + offset) if hunk.old_start else cursor
        index = _locate(lines, hunk.before, expected, cursor)
        if index is None:
            preview = "\n".join(hunk.before[:4])
            raise PatchError(f"Hunk {number} does not apply: its context was not found in the script:\n{preview}")
        lines[index : index + len(hunk.before)] = list(hunk.after)
        cursor = index + len(hunk.after)
        offset += len(hunk.after) - len(hunk.before)
    return "\n".join(lines) + "\n"


def changed_lines(diff: str) -> frozenset[str]:
    """Normalized '+'/'-' lines of a diff, ignoring whitespace and blank lines."""
    result = set()
    for line in diff.splitlines():
        if line.startswith(("+++", "---")) or not line.startswith(("+", "-")):
            continue
        body = " ".join(line[1:].split())
        if body:
            result.add(line[0] + body)
    return frozenset(result)


def fingerprint(diff: str) -> str:
    return hashlib.sha1("\n".join(sorted(changed_lines(diff))).encode()).hexdigest()


def similarity(first: str, second: str) -> float:
    a, b = changed_lines(first), changed_lines(second)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _locate(lines: list[str], block: tuple[str, ...], expected: int, cursor: int) -> int | None:
    if not block:
        return min(max(expected, 0), len(lines))
    for normalize in _NORMALIZERS:
        wanted = [normalize(line) for line in block]
        normalized = [normalize(line) for line in lines]
        matches = [i for i in range(len(lines) - len(block) + 1) if normalized[i : i + len(block)] == wanted]
        if matches:
            after_cursor = [i for i in matches if i >= cursor] or matches
            return min(after_cursor, key=lambda i: abs(i - expected))
    return None


_NORMALIZERS: tuple[Callable[[str], str], ...] = (
    lambda line: line,
    lambda line: line.rstrip(),
    lambda line: " ".join(line.split()),
)


def _with_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"
