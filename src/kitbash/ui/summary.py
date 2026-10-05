"""Final terminal summary and the dry-run plan."""

from collections.abc import Mapping
from typing import Any

from rich.console import Console

from kitbash.services.plan import Planner
from kitbash.ui.tables import simple_table


def print_summary(console: Console, report: Mapping[str, Any]) -> None:
    totals = report["totals"]
    console.print(
        simple_table(
            "Run summary",
            ("metric", "value"),
            [
                ("wall time", f"{totals['wall_time_s']:,.0f}s"),
                ("compute time", f"{totals['compute_time_s']:,.0f}s"),
                ("user time", f"{totals['user_time_s']:,.0f}s"),
                ("review queue wait", f"{totals['review_queue_wait_s']:,.0f}s"),
                ("LLM calls", int(totals.get("llm_calls", 0))),
                ("tokens in / out", f"{int(totals.get('tokens_in', 0)):,} / {int(totals.get('tokens_out', 0)):,}"),
                ("cost estimate", f"${totals.get('cost_usd', 0.0):,.4f}"),
                ("critic cycles", totals["critic_cycles"]),
                ("user interventions", totals["user_interventions"]),
                ("Trellis time", f"{totals['trellis_time_s']:,.0f}s"),
                ("Retopology time", f"{totals['retopology_time_s']:,.0f}s"),
                ("idle (backpressure)", f"{totals['worker_idle_backpressure_s']:,.0f}s"),
            ],
        )
    )
    console.print(
        simple_table(
            "Phases",
            ("phase", "status", "wall", "user", "LLM calls", "cost", "cycles"),
            [
                (name, p["status"], f"{p['wall_time_s']:,.0f}s", f"{p['user_time_s']:,.0f}s", int(p.get("llm_calls", 0)),
                 f"${p.get('cost_usd', 0.0):,.4f}", p["critic_cycles"])
                for name, p in report["phases"].items()
            ],
        )
    )
    if report["assets"]:
        console.print(
            simple_table(
                "Assets",
                ("asset", "state", "wall", "compute", "queue wait", "cycles", "USD", "round trip"),
                [
                    (asset_id, a["state"] + (" (reused)" if a["reused"] else ""), f"{a['wall_time_s']:,.0f}s", f"{a['compute_time_s']:,.0f}s",
                     f"{a['review_queue_wait_s']:,.0f}s", a["critic_cycles"], a.get("usd_material_mode") or "-",
                     a.get("usd_roundtrip_score") or "-")
                    for asset_id, a in report["assets"].items()
                ],
            )
        )


def print_plan(console: Console, planner: Planner) -> None:
    console.print(simple_table("Plan (dry run: nothing was executed)", ("phase", "step", "estimate"), [(r.phase, r.step, r.estimate) for r in planner.steps()]))
    console.print(simple_table("Models", ("phase", "role", "model"), planner.models()))
    console.print(simple_table("Paths", ("what", "path", "status"), planner.paths()))
