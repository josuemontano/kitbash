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
{"summary": "two or three sentences", "scores": {"<criterion id>": {"score": 0.0-1.0, "pass": true|false, "notes": "why"}}, "edits": [{"target": "what to change", "issue": "what is wrong", "instruction": "exactly what to change in the script", "priority": "high | medium | low"}]}
Score every criterion listed. Give at most 6 edits, each small and concrete; no edits if everything passes.
