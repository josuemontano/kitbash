"""Entry point executed by Blender: ``blender -b [file.blend] -P kb_runner.py -- <payload.json>``.

Loads the payload, exposes its arguments through ``kitbash_bpy``, runs the target script and always
writes a JSON result (``ok``, ``result``, ``error``, ``traceback``) for kitbash to read.
"""

import json
import runpy
import sys
import traceback


def main() -> int:
    payload_path = sys.argv[sys.argv.index("--") + 1]
    with open(payload_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    sys.path.insert(0, payload["helpers_dir"])
    import kitbash_bpy as kb

    kb._configure(payload["args"])
    try:
        runpy.run_path(payload["script"], run_name="__main__")
        outcome = {"ok": True, "result": kb._result()}
    except SystemExit as exc:
        ok = exc.code in (0, None)
        outcome = {"ok": ok, "result": kb._result(), "error": None if ok else f"SystemExit({exc.code})"}
    except BaseException as exc:
        outcome = {
            "ok": False,
            "result": kb._result(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    with open(payload["result_path"], "w", encoding="utf-8") as handle:
        json.dump(outcome, handle, default=str, indent=1)
    return 0 if outcome["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
