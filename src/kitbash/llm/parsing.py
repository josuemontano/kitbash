"""Pull JSON, code and diffs out of model answers."""

import ast
import json
import re
from typing import Any

from kitbash.errors import LLMError

_FENCE = re.compile(r"```([\w+-]*)[^\n]*\n(.*?)```", re.DOTALL)
BLENDER_PYTHON = (3, 13)


def fenced_blocks(text: str) -> list[tuple[str, str]]:
    """(language, body) of every fenced block; an unterminated final fence is included."""
    blocks = [(lang.lower(), body) for lang, body in _FENCE.findall(text)]
    tail = text.rsplit("```", 1)
    if text.count("```") % 2 == 1 and len(tail) == 2:
        lang, _, body = tail[1].partition("\n")
        blocks.append((lang.strip().lower(), body))
    return blocks


def extract_json(text: str) -> Any:
    """The answer's JSON value: fenced blocks first, then the whole text. A top-level object that is
    only missing its closing brackets is repaired before falling back to inner objects."""
    candidates = [body for lang, body in fenced_blocks(text) if lang in ("json", "")] + [text]
    decoder = json.JSONDecoder()
    starts = [(candidate, [m.start() for m in re.finditer(r"[{\[]", candidate)]) for candidate in candidates]
    for candidate, positions in starts:
        if not positions:
            continue
        chunk = candidate[positions[0] :]
        try:
            return decoder.raw_decode(chunk)[0]
        except json.JSONDecodeError:
            repaired = _close_open_brackets(chunk)
            if repaired is not None:
                try:
                    value = json.loads(repaired)
                except json.JSONDecodeError:
                    value = None
                if value:  # an empty object from a stray brace in prose is not an answer
                    return value
    for candidate, positions in starts:
        for start in positions[1:]:
            try:
                return decoder.raw_decode(candidate[start:])[0]
            except json.JSONDecodeError:
                continue
    raise LLMError("The answer contains no valid JSON")


def _close_open_brackets(text: str) -> str | None:
    """``text`` with its unclosed strings, arrays and objects closed, or None if nothing is open."""
    stack: list[str] = []
    in_string = escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack:
                return None
            stack.pop()
    if not stack and not in_string:
        return None
    return text.rstrip().rstrip(",") + ('"' if in_string else "") + "".join(reversed(stack))


def extract_python(text: str) -> str:
    blocks = [body for lang, body in fenced_blocks(text) if lang in ("python", "py", "")]
    code = max(blocks, key=len) if blocks else text
    code = code.strip("\n") + "\n"
    check_python(code)
    return code


def check_python(code: str) -> None:
    """Syntax check against Blender's Python version."""
    try:
        ast.parse(code, feature_version=BLENDER_PYTHON)
    except SyntaxError as exc:
        raise LLMError(f"The script has a syntax error on line {exc.lineno}: {exc.msg}") from exc


def extract_diff(text: str) -> str:
    blocks = [body for lang, body in fenced_blocks(text) if lang in ("diff", "patch", "udiff", "")]
    body = max(blocks, key=len) if blocks else text
    lines = body.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(("--- ", "@@"))), None)
    if start is None:
        raise LLMError("The answer contains no unified diff")
    return "\n".join(lines[start:]) + "\n"
