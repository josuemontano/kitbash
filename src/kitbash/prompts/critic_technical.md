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

## Rubric criteria (use these ids in feedback)
$criteria

## Bounded evaluation results
```json
$scorecard
```
These structured results own scoring. Diagnose failed, uncertain or regressed criteria; do not rescore them.

## User feedback to honour
$feedback

## Previous cycles
Do not ask again for changes that were reverted or rejected.
$history

## Output
JSON only:
{"summary": "two or three sentences", "edits": [{"target": "criterion id and function or line", "issue": "what is wrong", "instruction": "exactly what to change in the script", "priority": "high | medium | low"}]}
Return feedback only: no scores, probabilities or pass/fail verdicts. For uncertain criteria, explain
what evidence is missing rather than inventing a judgment. If the script failed, the first edit must
fix the error. Preserve dimensions that already pass and avoid regressions shown in prior results.
Give at most 6 small, concrete edits.
