# Task: patch the $phase script

Subject: $subject
What it must be: $description
Style: $style

## Current script (script.py)
```python
$script
```

## Edits to make, most important first
```json
$edits
```

## Error from the last run
$error

## Diff history
Statuses: kept (applied and kept), reverted (made the result worse), rejected (never applied).
Never repeat a reverted or rejected change.
$history

## Why your previous attempt was rejected
$rejection

## Helper API (kitbash_bpy)
$api

## Output
One unified diff against script.py in a single ```diff block:
- start with `--- a/script.py` and `+++ b/script.py`;
- every hunk starts with an `@@ -l,n +l,n @@` header and has at least 3 unchanged context lines copied
  exactly from the current script (line numbers may be approximate);
- keep it small and focused (ideally under 40 changed lines): fix the highest-priority edits first and
  never rewrite the script wholesale; later cycles can take the rest;
- the patched file must be a complete, valid Blender script.
