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
{"summary": "two or three sentences", "scores": {"<criterion id>": {"score": 0.0-1.0, "pass": true|false, "notes": "evidence"}}, "edits": [{"target": "function or line", "issue": "what is wrong", "instruction": "exactly what to change in the script", "priority": "high | medium | low"}]}
Score every criterion listed from the evidence. If the script failed, the first edit must fix the error.
Give at most 6 edits; no edits if everything passes.
