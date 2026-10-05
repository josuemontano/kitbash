from kitbash.analytics import context
from kitbash.analytics.report import AnalyticsReport, render_timeline
from kitbash.analytics.tracker import EventKind, SpanKind, Tracker
from kitbash.domain.assets import AssetRecord, AssetState
from kitbash.paths import OutputLayout
from kitbash.store.state import StateDB


def test_report_separates_user_and_compute_time_and_shows_overlap(tmp_path):
    state = StateDB(tmp_path / "state.db")
    layout = OutputLayout.at(tmp_path / "out")
    tracker = Tracker(state.spans)
    t0 = 1000.0
    with context.bind(phase="modelling"):
        with context.bind(asset_id="a", worker="worker-1"):
            tracker.record(SpanKind.ASSET_WORK, "a", t0, t0 + 50)
            tracker.record(SpanKind.SUBPROCESS, "trellis", t0 + 5, t0 + 30, attempt=1)
            tracker.record(SpanKind.LLM, "modelling.script", t0 + 31, t0 + 40, tokens_in=100, tokens_out=50, cost_usd=0.01, model="m1", role="code")
        with context.bind(asset_id="b", worker="worker-2"):
            tracker.record(SpanKind.ASSET_WORK, "b", t0 + 10, t0 + 100)
            tracker.record(SpanKind.SUBPROCESS, "trellis", t0 + 55, t0 + 80, attempt=2)
            tracker.event(EventKind.RETRY, "trellis")
        with context.bind(asset_id="a"):
            tracker.record(SpanKind.REVIEW_WAIT, "a", t0 + 50, t0 + 60)
            tracker.record(SpanKind.USER_REVIEW, "a", t0 + 60, t0 + 90)
            tracker.event(EventKind.USER_INTERVENTION, "feedback")
        tracker.record(SpanKind.IDLE_BACKPRESSURE, "worker_idle", t0 + 50, t0 + 70, worker="worker-1")
    tracker.record(SpanKind.PHASE, "modelling", t0, t0 + 110)
    for asset, times in {"a": (t0, t0 + 50, t0 + 95), "b": (t0 + 10, t0 + 100, t0 + 105)}.items():
        state.assets.upsert(AssetRecord(id=asset, name=asset.upper(), state=AssetState.APPROVED))
        for state_name, at in zip(("queued", "awaiting_review", "approved"), times, strict=True):
            state.db.execute(
                "INSERT INTO asset_transitions(asset_id, from_state, to_state, at, note) VALUES (?, NULL, ?, ?, '')", (asset, state_name, at)
            )

    report = AnalyticsReport(state, layout).write()
    totals = report["totals"]
    assert totals["user_time_s"] == 30.0 and totals["review_queue_wait_s"] == 10.0
    assert totals["worker_busy_s"] == 140.0 and totals["worker_idle_backpressure_s"] == 20.0
    assert totals["llm_calls"] == 1 and totals["tokens_in"] == 100 and totals["cost_usd"] == 0.01
    assert totals["retries"] == 2  # one retry event + one Trellis attempt > 1
    assert totals["user_interventions"] == 1

    a = report["assets"]["a"]
    assert a["review_queue_wait_s"] == 10.0 and a["user_time_s"] == 30.0 and a["trellis_time_s"] == 25.0
    assert [i["state"] for i in a["timeline"]] == ["queued", "awaiting_review", "approved"]

    (review,) = report["concurrency"]["reviews"]
    assert review["asset"] == "a" and review["other_assets_compute_s"] == 30.0  # b kept working during the review
    assert review["other_assets_trellis_llm_blender_s"] == 20.0 and review["assets_in_progress"] == ["b"]
    assert report["workers"]["worker-1"]["idle_backpressure_s"] == 20.0

    markdown = layout.analytics_md.read_text()
    assert "## Review overlap" in markdown and "| a | 30.0 | 30.0 |" in markdown
    assert layout.analytics_json.is_file()
    state.close()


def test_timeline_marks_user_review():
    assets = {
        "a": {"reused": False, "timeline": [{"state": "building", "start": 0, "end": 10}, {"state": "awaiting_review", "start": 10, "end": 20}]},
        "b": {"reused": True, "timeline": []},
    }
    chart = render_timeline(assets, [{"asset": "a", "start": 15, "end": 20}])
    row = next(line for line in chart.splitlines() if line.startswith("a "))
    assert "b" in row and "W" in row and "U" in row and not any(line.startswith("b ") for line in chart.splitlines())
