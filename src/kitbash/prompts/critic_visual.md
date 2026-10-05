# Task: visual critique of the $phase result (cycle $cycle)

Subject: $subject
What it must be: $description
Style: $style

Attachments, in order: reference images ($references), then renders of the current result ($renders).

## Measured facts
```json
$facts
```

## Rubric criteria to score (use these ids)
$criteria

## User feedback to honour
$feedback

## Previous cycles
Do not ask again for changes that were reverted or rejected.
$history

## Output
JSON only:
{"summary": "two or three sentences", "scores": {"<criterion id>": {"score": 0.0-1.0|null, "pass": true|false|null, "notes": "evidence or why unavailable"}}, "edits": [{"target": "what to change", "issue": "what is wrong", "instruction": "exactly what to change in the script", "priority": "high | medium | low"}]}
Include every criterion listed. For assessed criteria, give both a numeric score and a boolean verdict.
If evidence is unavailable, use {"score": null, "pass": null, "notes": "what evidence is missing"}.
Never omit a criterion or invent a pass/fail assessment without evidence.
Give at most 6 edits, each small and concrete; no edits if everything passes.
