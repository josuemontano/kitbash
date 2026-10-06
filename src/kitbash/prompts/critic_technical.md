# Task: technical critique of the $phase result (cycle $cycle)

Subject: $subject
What it must be: $description
Style: $style

## Script
```python
$script
```

## Error from the last run
$error

## Measured facts
```json
$facts
```

## Inspection report
```json
$report
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
{"summary": "two or three sentences", "scores": {"<criterion id>": {"score": 0.0-1.0|null, "pass": true|false|null, "notes": "evidence or why unavailable"}}, "edits": [{"target": "function or line", "issue": "what is wrong", "instruction": "exactly what to change in the script", "priority": "high | medium | low"}]}
Include every criterion listed. For assessed criteria, give both a numeric score and a boolean verdict.
If evidence is unavailable, use {"score": null, "pass": null, "notes": "what evidence is missing"}.
Never omit a criterion or invent a pass/fail assessment without evidence. If the script failed, the first edit must fix the error.
Give at most 6 edits; no edits if everything passes.
