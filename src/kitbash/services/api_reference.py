"""Documentation of the ``kitbash_bpy`` helpers, generated from the module itself (single source of truth)."""

import ast
from functools import cache
from pathlib import Path

from kitbash.infra.blender import blender_script


@cache
def blender_api_reference(path: Path | None = None) -> str:
    source = (path or blender_script("kitbash_bpy.py")).read_text(encoding="utf-8")
    lines = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and any(_is_api(d) for d in node.decorator_list):
            doc = " ".join((ast.get_docstring(node) or "").split())
            lines.append(f"- `kb.{node.name}({ast.unparse(node.args)})`: {doc}")
    return "\n".join(lines)


def _is_api(decorator: ast.expr) -> bool:
    return isinstance(decorator, ast.Name) and decorator.id == "api"
