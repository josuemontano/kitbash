"""Analytics: aggregate spans, events and state transitions into analytics.json and analytics.md."""

import json
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from kitbash.analytics.tracker import EventKind, SpanKind
from kitbash.domain.phases import PHASE_ORDER
from kitbash.paths import OutputLayout
from kitbash.store.state import StateDB

USER_KINDS = (SpanKind.GATE, SpanKind.USER_REVIEW, SpanKind.USER_INPUT)
TIMELINE_WIDTH = 64
STATE_GLYPHS = {
    "queued": ".", "referencing": "r", "input_needed": "I", "generating": "g", "building": "b",
    "critiquing": "c", "awaiting_review": "W", "needs_rework": "N", "approved": "A", "skipped": "S",
}


def _duration(span: Mapping[str, Any]) -> float:
    return max(0.0, float(span["ended_at"]) - float(span["started_at"]))


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _r(value: float) -> float:
    return round(value, 3)


class _Totals:
    """Accumulates the usual counters for one grouping (a phase, an asset, an agent, a model...)."""

    KEYS = ("llm_calls", "llm_time_s", "tokens_in", "tokens_out", "cost_usd")

    def __init__(self) -> None:
        self.values: dict[str, float] = defaultdict(float, dict.fromkeys(self.KEYS, 0.0))

    def add_llm(self, span: Mapping[str, Any]) -> None:
        meta = span["meta"]
        self.values["llm_calls"] += 1
        self.values["llm_time_s"] += _duration(span)
        self.values["tokens_in"] += meta.get("tokens_in") or 0
        self.values["tokens_out"] += meta.get("tokens_out") or 0
        self.values["cost_usd"] += meta.get("cost_usd") or 0.0

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 6 if k == "cost_usd" else 3) for k, v in sorted(self.values.items())}


class AnalyticsReport:
    def __init__(self, state: StateDB, layout: OutputLayout) -> None:
        self._state = state
        self._layout = layout

    # -- building ----------------------------------------------------------------------------------

    def build(self) -> dict[str, Any]:
        spans = self._state.spans.spans()
        events = self._state.spans.events()
        transitions = self._state.assets.transitions()
        meta = self._state.meta.all()
        now = time.time()
        started = min([s["started_at"] for s in spans] + [meta.get("run_started_at", now)])
        ended = max([s["ended_at"] for s in spans] + [started])
        user_time = sum(_duration(s) for s in spans if s["kind"] in USER_KINDS)
        worker_busy = sum(_duration(s) for s in spans if s["kind"] == SpanKind.ASSET_WORK)
        phase_spans = [s for s in spans if s["kind"] == SpanKind.PHASE]
        main_thread_compute = sum(_duration(s) for s in phase_spans if s["name"] != "modelling") - sum(
            _duration(s) for s in spans if s["kind"] == SpanKind.GATE and s["phase"] != "modelling"
        )
        llm = [s for s in spans if s["kind"] == SpanKind.LLM]
        totals = _Totals()
        for span in llm:
            totals.add_llm(span)
        cycles = [c for c in self._state.cycles.all_cycles() if c.status != "pending"]
        return {
            "run": {
                "output": str(self._layout.root),
                "input": meta.get("input"),
                "style": meta.get("style"),
                "started_at": started,
                "finished_at": ended,
                "wall_time_s": _r(ended - started),
                "versions": meta.get("versions", {}),
                "blender_capabilities": meta.get("blender_capabilities", {}),
                "models": meta.get("models", {}),
            },
            "totals": {
                **totals.as_dict(),
                "wall_time_s": _r(ended - started),
                "compute_time_s": _r(worker_busy + max(main_thread_compute, 0.0)),
                "worker_busy_s": _r(worker_busy),
                "user_time_s": _r(user_time),
                "review_queue_wait_s": _r(sum(_duration(s) for s in spans if s["kind"] == SpanKind.REVIEW_WAIT)),
                "critic_cycles": len(cycles),
                "retries": sum(1 for e in events if e["kind"] == EventKind.RETRY) + self._trellis_retries(spans),
                "user_interventions": sum(1 for e in events if e["kind"] == EventKind.USER_INTERVENTION),
                "trellis_time_s": _r(sum(_duration(s) for s in spans if s["name"] == "trellis")),
                "retopology_time_s": _r(sum(_duration(s) for s in spans if s["name"] == "retopology")),
                "worker_idle_backpressure_s": _r(sum(_duration(s) for s in spans if s["kind"] == SpanKind.IDLE_BACKPRESSURE)),
                "worker_idle_s": _r(sum(_duration(s) for s in spans if s["kind"] == SpanKind.IDLE)),
            },
            "phases": self._phases(spans, events, cycles),
            "assets": self._assets(spans, events, transitions, now),
            "agents": self._grouped(llm, "agent"),
            "models": self._grouped(llm, lambda s: s["meta"].get("model")),
            "roles": self._grouped(llm, lambda s: s["meta"].get("role")),
            "steps": self._steps(spans),
            "workers": self._workers(spans),
            "concurrency": self._concurrency(spans),
            "llm_calls": [
                {
                    "task": s["name"], "phase": s["phase"], "asset": s["asset_id"], "agent": s["agent"],
                    "model": s["meta"].get("model"), "duration_s": _r(_duration(s)),
                    "tokens_in": s["meta"].get("tokens_in"), "tokens_out": s["meta"].get("tokens_out"),
                    "cost_usd": s["meta"].get("cost_usd"), "error": s["meta"].get("error"),
                }
                for s in llm
            ],
            "scene": meta.get("assembly", {}),
        }

    def _phases(self, spans, events, cycles) -> dict[str, Any]:
        rows: dict[str, Any] = {}
        statuses = {row["name"]: row for row in self._state.phases.rows()}
        for phase in PHASE_ORDER:
            name = phase.value
            totals = _Totals()
            for span in spans:
                if span["phase"] == name and span["kind"] == SpanKind.LLM:
                    totals.add_llm(span)
            rows[name] = {
                "status": statuses.get(name, {}).get("status", "pending"),
                "wall_time_s": _r(sum(_duration(s) for s in spans if s["kind"] == SpanKind.PHASE and s["name"] == name)),
                "user_time_s": _r(sum(_duration(s) for s in spans if s["phase"] == name and s["kind"] in USER_KINDS)),
                "critic_cycles": sum(1 for c in cycles if c.phase == name),
                "user_interventions": sum(1 for e in events if e["phase"] == name and e["kind"] == EventKind.USER_INTERVENTION),
                "retries": sum(1 for e in events if e["phase"] == name and e["kind"] == EventKind.RETRY),
                **totals.as_dict(),
            }
        return rows

    def _assets(self, spans, events, transitions, now) -> dict[str, Any]:
        by_asset: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for transition in transitions:
            by_asset[transition["asset_id"]].append(transition)
        records = {r.id: r for r in self._state.assets.all()}
        cycles = self._state.cycles.all_cycles()
        result = {}
        for asset_id, record in records.items():
            own = [s for s in spans if s["asset_id"] == asset_id]
            totals = _Totals()
            for span in own:
                if span["kind"] == SpanKind.LLM:
                    totals.add_llm(span)
            timeline = self._timeline(by_asset.get(asset_id, []), now)
            facts = self._best_facts(asset_id, record.best_cycle, cycles)
            trellis = [s for s in own if s["name"] == "trellis"]
            retopology = [s for s in own if s["name"] == "retopology"]
            result[asset_id] = {
                "name": record.name,
                "modelling_method": record.modelling_method,
                "state": record.state.value,
                "reused": record.reused,
                "backlot_id": record.backlot_id,
                "score": record.score,
                "wall_time_s": _r(timeline[-1]["end"] - timeline[0]["start"]) if timeline else 0.0,
                "compute_time_s": _r(sum(_duration(s) for s in own if s["kind"] == SpanKind.ASSET_WORK)),
                "review_queue_wait_s": _r(sum(_duration(s) for s in own if s["kind"] == SpanKind.REVIEW_WAIT)),
                "user_time_s": _r(sum(_duration(s) for s in own if s["kind"] in USER_KINDS)),
                "critic_cycles": sum(1 for c in cycles if c.subject == asset_id and c.status != "pending"),
                "trellis_time_s": _r(sum(_duration(s) for s in trellis)),
                "retopology_time_s": _r(sum(_duration(s) for s in retopology)),
                "retopology": record.extra.get("retopology"),
                "trellis_retries": sum(1 for s in trellis if (s["meta"].get("attempt") or 1) > 1),
                "retries": sum(1 for e in events if e["asset_id"] == asset_id and e["kind"] == EventKind.RETRY),
                "user_interventions": sum(1 for e in events if e["asset_id"] == asset_id and e["kind"] == EventKind.USER_INTERVENTION),
                "usd_material_mode": facts.get("usd_material_mode"),
                "usd_roundtrip_score": facts.get("usd_roundtrip_score"),
                **totals.as_dict(),
                "timeline": timeline,
            }
        return result

    @staticmethod
    def _timeline(transitions: Sequence[Mapping[str, Any]], now: float) -> list[dict[str, Any]]:
        intervals = []
        for index, transition in enumerate(transitions):
            end = transitions[index + 1]["at"] if index + 1 < len(transitions) else None
            state = transition["to_state"]
            if end is None and state in ("approved", "skipped"):
                end = transition["at"]
            end = end if end is not None else now
            intervals.append({"state": state, "start": transition["at"], "end": end, "duration_s": _r(end - transition["at"]), "note": transition["note"]})
        return intervals

    @staticmethod
    def _best_facts(asset_id: str, best_cycle: int | None, cycles) -> dict[str, Any]:
        row = next((c for c in cycles if c.subject == asset_id and c.cycle == best_cycle and c.critique_path), None)
        if row is None or not Path(row.critique_path).is_file():
            return {}
        return json.loads(Path(row.critique_path).read_text(encoding="utf-8")).get("scorecard", {}).get("facts", {})

    @staticmethod
    def _grouped(llm: Iterable[Mapping[str, Any]], key) -> dict[str, Any]:
        groups: dict[str, _Totals] = defaultdict(_Totals)
        for span in llm:
            name = key(span) if callable(key) else span.get(key)
            groups[str(name or "unknown")].add_llm(span)
        return {name: totals.as_dict() for name, totals in sorted(groups.items())}

    @staticmethod
    def _steps(spans) -> dict[str, Any]:
        steps: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "time_s": 0.0})
        for span in spans:
            if span["kind"] in (SpanKind.STEP, SpanKind.SUBPROCESS):
                entry = steps[f"{span['phase'] or '-'}:{span['name']}"]
                entry["count"] += 1
                entry["time_s"] = _r(entry["time_s"] + _duration(span))
        return dict(sorted(steps.items()))

    @staticmethod
    def _workers(spans) -> dict[str, Any]:
        workers: dict[str, dict[str, float]] = defaultdict(lambda: {"busy_s": 0.0, "idle_backpressure_s": 0.0, "idle_s": 0.0})
        for span in spans:
            worker = span["worker"] or span["meta"].get("worker")
            if not worker:
                continue
            key = {SpanKind.ASSET_WORK: "busy_s", SpanKind.IDLE_BACKPRESSURE: "idle_backpressure_s", SpanKind.IDLE: "idle_s"}.get(span["kind"])
            if key:
                workers[worker][key] = _r(workers[worker][key] + _duration(span))
        return dict(sorted(workers.items()))

    @staticmethod
    def _concurrency(spans) -> dict[str, Any]:
        """For every asset review: how much generation/critique work ran for other assets meanwhile."""
        reviews = [s for s in spans if s["kind"] == SpanKind.USER_REVIEW]
        work = [s for s in spans if s["kind"] == SpanKind.ASSET_WORK]
        heavy = [s for s in spans if s["kind"] in (SpanKind.SUBPROCESS, SpanKind.LLM) and s["phase"] == "modelling"]
        idle = [s for s in spans if s["kind"] == SpanKind.IDLE_BACKPRESSURE]
        rows = []
        for review in reviews:
            start, end = review["started_at"], review["ended_at"]
            others = [s for s in work if s["asset_id"] != review["asset_id"]]
            rows.append({
                "asset": review["asset_id"],
                "start": start,
                "end": end,
                "duration_s": _r(end - start),
                "other_assets_compute_s": _r(sum(_overlap(start, end, s["started_at"], s["ended_at"]) for s in others)),
                "other_assets_trellis_llm_blender_s": _r(sum(
                    _overlap(start, end, s["started_at"], s["ended_at"]) for s in heavy if s["asset_id"] != review["asset_id"]
                )),
                "worker_idle_backpressure_s": _r(sum(_overlap(start, end, s["started_at"], s["ended_at"]) for s in idle)),
                "assets_in_progress": sorted({s["asset_id"] for s in others if _overlap(start, end, s["started_at"], s["ended_at"]) > 0}),
            })
        review_time = sum(r["duration_s"] for r in rows)
        during = sum(r["other_assets_compute_s"] for r in rows)
        return {
            "reviews": rows,
            "review_time_s": _r(review_time),
            "other_assets_compute_during_review_s": _r(during),
            "mean_parallel_workers_during_review": _r(during / review_time) if review_time else 0.0,
        }

    def _trellis_retries(self, spans) -> int:
        return sum(1 for s in spans if s["name"] == "trellis" and (s["meta"].get("attempt") or 1) > 1)

    # -- writing -----------------------------------------------------------------------------------

    def write(self) -> dict[str, Any]:
        report = self.build()
        self._layout.analytics_dir.mkdir(parents=True, exist_ok=True)
        self._layout.analytics_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        self._layout.analytics_md.write_text(render_markdown(report), encoding="utf-8")
        return report


# -- markdown ----------------------------------------------------------------------------------------


def _md_table(columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join("" if v is None else str(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _money(value: float | None) -> str:
    return "-" if not value else f"${value:,.4f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    run, totals = report["run"], report["totals"]
    sections = [
        "# kitbash analytics",
        f"Output: `{run['output']}`  \nInput: {run.get('input')}  \nStyle: {run.get('style')}  \n"
        f"Versions: {', '.join(f'{k} {v}' for k, v in run.get('versions', {}).items())}",
        "## Totals",
        _md_table(("metric", "value"), [
            ("wall time (s)", totals["wall_time_s"]),
            ("compute time (s)", totals["compute_time_s"]),
            ("user time (s)", totals["user_time_s"]),
            ("review queue wait (s)", totals["review_queue_wait_s"]),
            ("LLM calls", int(totals.get("llm_calls", 0))),
            ("tokens in / out", f"{int(totals.get('tokens_in', 0)):,} / {int(totals.get('tokens_out', 0)):,}"),
            ("cost estimate", _money(totals.get("cost_usd"))),
            ("critic cycles", totals["critic_cycles"]),
            ("retries", totals["retries"]),
            ("user interventions", totals["user_interventions"]),
            ("Trellis time (s)", totals["trellis_time_s"]),
            ("retopology time (s)", totals["retopology_time_s"]),
            ("worker idle from backpressure (s)", totals["worker_idle_backpressure_s"]),
            ("worker idle, no work (s)", totals["worker_idle_s"]),
        ]),
        "## Phases",
        _md_table(
            ("phase", "status", "wall (s)", "user (s)", "LLM calls", "tokens in", "tokens out", "cost", "cycles", "interventions"),
            [
                (name, p["status"], p["wall_time_s"], p["user_time_s"], int(p.get("llm_calls", 0)), int(p.get("tokens_in", 0)),
                 int(p.get("tokens_out", 0)), _money(p.get("cost_usd")), p["critic_cycles"], p["user_interventions"])
                for name, p in report["phases"].items()
            ],
        ),
        "## Assets",
        _md_table(
            ("asset", "state", "wall (s)", "compute (s)", "queue wait (s)", "user (s)", "cycles", "Trellis (s)", "retopology (s)",
             "LLM calls", "cost", "USD materials", "round trip"),
            [
                (asset_id, a["state"] + (" (reused)" if a["reused"] else ""), a["wall_time_s"], a["compute_time_s"],
                 a["review_queue_wait_s"], a["user_time_s"], a["critic_cycles"], a["trellis_time_s"],
                 a["retopology_time_s"], int(a.get("llm_calls", 0)),
                 _money(a.get("cost_usd")), a.get("usd_material_mode") or "-", a.get("usd_roundtrip_score") or "-")
                for asset_id, a in report["assets"].items()
            ],
        ),
        "## Timeline",
        render_timeline(report["assets"], report["concurrency"]["reviews"]),
        "## Review overlap",
        "Work on other assets while the user reviewed an asset (shows generation continuing during review).\n\n"
        + _md_table(
            ("reviewed asset", "review (s)", "other assets compute (s)", "Trellis/LLM/Blender (s)", "idle backpressure (s)", "in progress"),
            [
                (r["asset"], r["duration_s"], r["other_assets_compute_s"], r["other_assets_trellis_llm_blender_s"],
                 r["worker_idle_backpressure_s"], ", ".join(r["assets_in_progress"]))
                for r in report["concurrency"]["reviews"]
            ],
        )
        + f"\n\nMean parallel workers during reviews: {report['concurrency']['mean_parallel_workers_during_review']}",
        "## Agents",
        _md_table(("agent", "calls", "tokens in", "tokens out", "time (s)", "cost"), [
            (name, int(a.get("llm_calls", 0)), int(a.get("tokens_in", 0)), int(a.get("tokens_out", 0)), a.get("llm_time_s", 0), _money(a.get("cost_usd")))
            for name, a in report["agents"].items()
        ]),
        "## Models",
        _md_table(("model", "calls", "tokens in", "tokens out", "time (s)", "cost"), [
            (name, int(m.get("llm_calls", 0)), int(m.get("tokens_in", 0)), int(m.get("tokens_out", 0)), m.get("llm_time_s", 0), _money(m.get("cost_usd")))
            for name, m in report["models"].items()
        ]),
        "## Steps",
        _md_table(("step", "count", "time (s)"), [(name, s["count"], s["time_s"]) for name, s in report["steps"].items()]),
        "## Workers",
        _md_table(("worker", "busy (s)", "idle backpressure (s)", "idle (s)"), [
            (name, w["busy_s"], w["idle_backpressure_s"], w["idle_s"]) for name, w in report["workers"].items()
        ]),
    ]
    if report.get("scene"):
        scene = report["scene"]
        acceptance = scene.get("acceptance", {})
        sections += ["## Scene", f"Acceptance: **{acceptance.get('status', 'not validated')}**  \n"
                     f"Automatic pass: {acceptance.get('automatic_pass', False)}  \nPublished: {acceptance.get('published', False)}  \n"
                     f"Blend: `{scene.get('scene_blend')}`  \nUSD: `{scene.get('scene_usd')}`  \n"
                     f"USD materials: {scene.get('usd_material_mode')}  \nUSD round trip score: {scene.get('usd_roundtrip_score')}"]
        if acceptance.get("issues"):
            sections.append("Validation failures:\n" + "\n".join(f"- {issue}" for issue in acceptance["issues"]))
    return "\n\n".join(sections) + "\n"


def render_timeline(assets: Mapping[str, Any], reviews: Sequence[Mapping[str, Any]]) -> str:
    """ASCII Gantt chart: one row per asset, one glyph per time slice; 'U' marks the user reviewing it."""
    intervals = [i for a in assets.values() for i in a["timeline"] if not a["reused"]]
    if not intervals:
        return "(no modelling timeline)"
    start = min(i["start"] for i in intervals)
    end = max(i["end"] for i in intervals)
    span = max(end - start, 1e-6)
    step = span / TIMELINE_WIDTH
    lines = []
    for asset_id, asset in assets.items():
        if asset["reused"]:
            continue
        row = []
        for column in range(TIMELINE_WIDTH):
            t = start + (column + 0.5) * step
            glyph = " "
            for interval in asset["timeline"]:
                if interval["start"] <= t < interval["end"] or (interval["start"] == interval["end"] == t):
                    glyph = STATE_GLYPHS.get(interval["state"], "?")
            if any(r["asset"] == asset_id and r["start"] <= t < r["end"] for r in reviews):
                glyph = "U"
            row.append(glyph)
        lines.append(f"{asset_id[:20]:<20} |{''.join(row)}|")
    legend = "  ".join(f"{g}={s}" for s, g in STATE_GLYPHS.items()) + "  U=user reviewing"
    return f"```\n0s{' ' * (TIMELINE_WIDTH + 17)}{span:,.0f}s\n" + "\n".join(lines) + f"\n```\n{legend}"
