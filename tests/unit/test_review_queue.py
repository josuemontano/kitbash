import random
import threading
import time

import pytest

from kitbash.analytics.tracker import SpanKind, Tracker
from kitbash.pipeline.review_queue import EntryKind, ReviewEntry, ReviewQueue
from kitbash.pipeline.scheduler import Scheduler
from kitbash.store.state import StateDB


@pytest.fixture
def tracker(tmp_path):
    state = StateDB(tmp_path / "state.db")
    yield Tracker(state.spans), state
    state.close()


def test_review_queue_is_fifo_in_completion_order():
    queue = ReviewQueue()
    for asset in ("b", "a", "c"):
        queue.put(ReviewEntry(EntryKind.REVIEW, asset))
    assert [queue.get(0.01).asset_id for _ in range(3)] == ["b", "a", "c"]
    assert queue.get(0.01) is None


def test_review_queue_get_wakes_up_on_put():
    queue = ReviewQueue()
    threading.Timer(0.05, lambda: queue.put(ReviewEntry(EntryKind.INPUT_NEEDED, "x", "need a name"))).start()
    entry = queue.get(timeout=2)
    assert entry.kind is EntryKind.INPUT_NEEDED and queue.current is entry
    queue.done()
    assert queue.current is None


def test_new_assets_wait_for_a_slot_but_rework_does_not():
    scheduler = Scheduler(capacity=2)
    stop = threading.Event()
    for asset in ("a", "b", "c"):
        scheduler.submit(asset)
    assert scheduler.next(stop) == "a"
    assert scheduler.next(stop) == "b"
    assert scheduler.backpressured
    result = []
    waiter = threading.Thread(target=lambda: result.append(scheduler.next(stop)))
    waiter.start()
    time.sleep(0.2)
    assert result == []  # "c" is blocked: the review buffer is full
    scheduler.submit_rework("a")  # a is back from review with feedback; it keeps its slot
    waiter.join(timeout=2)
    assert result == ["a"]
    scheduler.release("b")  # b approved: its slot frees up
    assert scheduler.next(stop) == "c"
    assert not scheduler.backpressured


def test_close_unblocks_workers():
    scheduler = Scheduler(capacity=1)
    stop = threading.Event()
    result = []
    thread = threading.Thread(target=lambda: result.append(scheduler.next(stop)))
    thread.start()
    scheduler.close()
    thread.join(timeout=2)
    assert result == [None]


def test_idle_time_is_recorded_by_reason(tracker):
    tracker, state = tracker
    scheduler = Scheduler(capacity=1, tracker=tracker)
    stop = threading.Event()
    scheduler.submit("a")
    scheduler.submit("b")
    assert scheduler.next(stop, "w1") == "a"
    threading.Timer(0.3, lambda: scheduler.release("a")).start()
    assert scheduler.next(stop, "w1") == "b"
    idle = [s for s in state.spans.spans() if s["kind"] == SpanKind.IDLE_BACKPRESSURE]
    assert idle and idle[0]["meta"]["worker"] == "w1"
    assert idle[0]["ended_at"] - idle[0]["started_at"] >= 0.25


def test_review_buffer_is_never_exceeded_under_concurrency():
    """Producers and a slow consumer: at most `capacity` assets are ever in flight or awaiting review."""
    capacity, workers, assets = 3, 4, 20
    scheduler = Scheduler(capacity=capacity)
    queue = ReviewQueue()
    stop = threading.Event()
    lock = threading.Lock()
    holding, peak = set(), [0]
    for index in range(assets):
        scheduler.submit(f"asset_{index:02d}")

    def worker() -> None:
        while (asset := scheduler.next(stop)) is not None:
            with lock:
                holding.add(asset)
                peak[0] = max(peak[0], len(holding))
            time.sleep(random.uniform(0.001, 0.01))
            queue.put(ReviewEntry(EntryKind.REVIEW, asset))
            scheduler.finished(asset)

    threads = [threading.Thread(target=worker) for _ in range(workers)]
    for thread in threads:
        thread.start()
    reviewed = []
    while len(reviewed) < assets:
        entry = queue.get(timeout=2)
        assert entry is not None, "producers stalled"
        time.sleep(0.005)
        assert len(queue) + 1 <= capacity
        reviewed.append(entry.asset_id)
        with lock:
            holding.discard(entry.asset_id)
        scheduler.release(entry.asset_id)
    scheduler.close()
    for thread in threads:
        thread.join(timeout=2)
    assert sorted(reviewed) == [f"asset_{i:02d}" for i in range(assets)]
    assert peak[0] <= capacity


def test_an_asset_stays_busy_while_any_worker_is_on_it():
    """A reworked asset can be taken again before its previous worker reports done."""
    scheduler = Scheduler(capacity=2)
    stop = threading.Event()
    scheduler.submit("a")
    assert scheduler.next(stop) == "a"      # worker 1
    scheduler.submit_rework("a")            # reviewed and sent back before worker 1 returned
    assert scheduler.next(stop) == "a"      # worker 2
    scheduler.finished("a")                 # worker 1 returns
    assert not scheduler.idle()             # worker 2 is still on it
    scheduler.finished("a")
    assert scheduler.idle()
