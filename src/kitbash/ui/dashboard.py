"""One ``rich.Live`` display for the run. Only the main thread reads input, and it pauses the display
while it does, so prompts never interleave with the live refresh."""

import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from attrs import define, field
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from kitbash.critique.store import CycleResult
from kitbash.domain.assets import AssetState
from kitbash.pipeline.board import AssetBoard
from kitbash.pipeline.review_queue import ReviewQueue
from kitbash.pipeline.scheduler import Scheduler

type View = Callable[[], RenderableType]

STATE_STYLES = {
    AssetState.QUEUED: "dim",
    AssetState.REFERENCING: "cyan",
    AssetState.INPUT_NEEDED: "bold magenta",
    AssetState.GENERATING: "blue",
    AssetState.BUILDING: "yellow",
    AssetState.CRITIQUING: "yellow",
    AssetState.AWAITING_REVIEW: "bold green",
    AssetState.APPROVED: "green",
    AssetState.SKIPPED: "red",
    AssetState.NEEDS_REWORK: "magenta",
}


class Dashboard:
    def __init__(self, console: Console, refresh_per_second: float = 4) -> None:
        self.console = console
        self._refresh = refresh_per_second
        self._live: Live | None = None
        self._lock = threading.RLock()
        self._messages: deque[str] = deque(maxlen=6)
        self._view: View | None = None

    @contextmanager
    def showing(self, view: View) -> Iterator[None]:
        with self._lock:
            self._view = view
            # Transient: pausing for a prompt erases the live region instead of leaving copies behind.
            self._live = Live(get_renderable=self._render, console=self.console, refresh_per_second=self._refresh, transient=True)
            self._live.start()
        try:
            yield
        finally:
            with self._lock:
                if self._live is not None:
                    self._live.stop()
                    self.console.print(self._render())  # keep the final state on screen
                self._live, self._view = None, None

    @contextmanager
    def paused(self) -> Iterator[None]:
        """Stop refreshing while the user is prompted."""
        with self._lock:
            live = self._live
            if live is not None:
                live.stop()
        try:
            yield
        finally:
            with self._lock:
                if live is not None and self._live is live:
                    live.vertical_overflow = "ellipsis"  # Live.stop() switches it to "visible"
                    live.start(refresh=True)

    def log(self, message: str) -> None:
        self._messages.append(f"{time.strftime('%H:%M:%S')} {message}")

    def _render(self) -> RenderableType:
        body = self._view() if self._view else Text("")
        if not self._messages:
            return body
        return Group(body, Panel("\n".join(self._messages), title="events", border_style="dim"))


@define
class PhaseProgress:
    """Progress of a phase-level critic loop (breakdown, layout), also a loop observer."""

    title: str
    status: str = "starting"
    cycle: int = 0
    last_score: float | None = None
    started_at: float = field(factory=time.time)

    def building(self, cycle: int) -> None:
        self.cycle, self.status = cycle, "writing and running the script"

    def critiquing(self, cycle: int) -> None:
        self.cycle, self.status = cycle, "critics reviewing"

    def evaluated(self, result: CycleResult) -> None:
        self.last_score, self.status = result.score, f"cycle {result.cycle:02d} scored {result.score:.2f}"

    def view(self) -> RenderableType:
        score = "n/a" if self.last_score is None else f"{self.last_score:.2f}"
        elapsed = time.time() - self.started_at
        text = f"cycle {self.cycle:02d}  ·  {self.status}  ·  last score {score}  ·  {elapsed:,.0f}s"
        return Panel(text, title=self.title, border_style="cyan")


class ModellingView:
    """Progress table for every asset plus the pinned review panel."""

    def __init__(self, board: AssetBoard, scheduler: Scheduler, queue: ReviewQueue, since: dict[str, float]) -> None:
        self._board = board
        self._scheduler = scheduler
        self._queue = queue
        self._since = since
        self._started = time.time()

    def __call__(self) -> RenderableType:
        table = Table(title="Modelling", expand=True, show_lines=False)
        for column in ("asset", "state", "cycle", "score", "trellis", "in state", "note"):
            table.add_column(column, overflow="fold")
        now = time.time()
        for asset in self._board.all():
            trellis = asset.extra.get("trellis", {})
            table.add_row(
                asset.id,
                Text(asset.state.value, style=STATE_STYLES[asset.state]),
                str(asset.extra.get("cycle", "")),
                f"{asset.score:.2f}" if asset.score is not None else str(asset.extra.get("last_score", "")),
                f"{trellis['duration_s']:.0f}s" if trellis.get("duration_s") else "",
                f"{now - self._since.get(asset.id, now):.0f}s",
                "reused from backlot" if asset.reused else (asset.error or "")[:60],
            )
        snapshot = self._scheduler.snapshot()
        current = self._queue.current
        waiting = len(self._queue)
        lines = [
            f"reviewing: [bold]{current.asset_id}[/bold] ({current.kind.value})" if current else "reviewing: -",
            f"waiting for review: {waiting}   buffer: {snapshot['holding']}/{snapshot['capacity']}"
            + ("   [bold red]backpressure: workers idle[/bold red]" if self._scheduler.backpressured else ""),
            f"workers busy: {snapshot['busy']}   queued: {snapshot['queued']}   rework: {snapshot['rework']}   "
            f"elapsed: {now - self._started:,.0f}s",
        ]
        return Group(table, Panel("\n".join(lines), title="review", border_style="green"))
