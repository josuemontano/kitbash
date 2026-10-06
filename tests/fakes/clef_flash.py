"""Deterministic System One responses at the HTTP boundary; no live model calls."""


def answer(request: dict) -> dict:
    answers = {}
    for criterion_id, question in request["questions"].items():
        if question["type"] == "noul":
            answers[criterion_id] = {"type": "noul", "noul": 0.995}
        else:
            levels = question["criteria"]
            answers[criterion_id] = {
                "type": "score", "score": len(levels) - 1,
                "legend": {str(i): label for i, label in enumerate(levels)},
                "probabilities": {str(i): float(i == len(levels) - 1) for i in range(len(levels))},
                "confidence": 1.0,
            }
    return {"model": request["model"], "answers": answers, "usage": {"input_tokens": 512, "output_tokens": 0}}
