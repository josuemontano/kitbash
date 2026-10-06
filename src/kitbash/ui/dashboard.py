"""Thread-safe bridge between synchronous pipeline work and the Textual terminal UI."""

import subprocess
import sys
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from attrs import define, field
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from kitbash.critique.store import CycleResult
from kitbash.domain.assets import AssetState
from kitbash.infra.process import ProcessCancelled, ProcessEvent, observe_processes
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


@define
class JobOutput:
    event: ProcessEvent
    started: float = field(factory=time.monotonic)
    finished: float | None = None
    status: str = "running"
    sequence: int = 0
    output: deque[tuple[int, str, str]] = field(factory=lambda: deque(maxlen=256))


@define
class Question:
    text: str
    choices: tuple[str, ...] | None
    default: str | None
    content: tuple[RenderableType, ...]
    images: tuple[Path, ...]
    answered: threading.Event = field(factory=threading.Event)
    answer: str | None = None


class Dashboard:
    def __init__(self, console: Console, refresh_per_second: float = 4, *, show_previews: bool = True) -> None:
        self.console = console
        self.refresh_per_second = refresh_per_second
        self.show_previews = show_previews
        self.enabled = False
        self._lock = threading.RLock()
        self._messages: deque[str] = deque(maxlen=100)
        self._view: View | None = None
        self._jobs: OrderedDict[str, JobOutput] = OrderedDict()
        self._question: Question | None = None
        self._stopping = threading.Event()

    def run[T](self, operation: Callable[[], T], cancel: Callable[[], None]) -> T:
        if not self.console.is_terminal or not sys.stdin.isatty() or self.console.quiet:
            return operation()
        from kitbash.ui.tui import RunScreen

        self.enabled = True
        self._stopping.clear()
        self._jobs.clear()
        self._messages.clear()
        self._view = None
        screen = RunScreen(self, operation, cancel)
        try:
            try:
                screen.run()
            finally:
                screen.join()
            if screen.error is not None:
                raise screen.error
            return screen.result
        finally:
            self.enabled = False

    @contextmanager
    def observing(self) -> Iterator[None]:
        with observe_processes(self.process_event):
            yield

    @contextmanager
    def showing(self, view: View) -> Iterator[None]:
        with self._lock:
            self._view = view
        try:
            yield
        finally:
            if not self.enabled:
                self.console.print(view())

    def process_event(self, event: ProcessEvent) -> None:
        with self._lock:
            if event.kind == "started":
                self._jobs[event.job_id] = JobOutput(event)
            elif (job := self._jobs.get(event.job_id)) is not None:
                if event.kind == "output":
                    job.sequence += 1
                    job.output.append((job.sequence, event.stream, event.text))
                elif event.kind == "finished":
                    if event.text:
                        job.sequence += 1
                        job.output.append((job.sequence, "stderr", event.text))
                    job.finished = time.monotonic()
                    job.status = (
                        "cancelled" if event.cancelled else "timed out" if event.timed_out
                        else "done" if event.returncode == 0 else f"failed ({event.returncode})"
                    )
            if event.kind == "finished":
                # Keep all running jobs and a bounded history of completed commands.
                completed = [key for key, job in self._jobs.items() if job.finished is not None]
                for key in completed[:-12]:
                    del self._jobs[key]

    def snapshot(self) -> tuple[RenderableType, list[JobOutput], tuple[str, ...], Question | None]:
        with self._lock:
            body = self._view() if self._view else Panel("Checking tools and models…", title="Preflight")
            jobs = [
                JobOutput(job.event, job.started, job.finished, job.status, job.sequence, deque(job.output))
                for job in self._jobs.values()
            ]
            return body, jobs, tuple(self._messages), self._question

    def ask(
        self, question: str, *, choices: Sequence[str] | None = None, default: str | None = None,
        content: Sequence[RenderableType] = (), images: Sequence[Path] = (),
    ) -> str:
        if not self.enabled:
            for item in content:
                self.console.print(item)
            return Prompt.ask(question, choices=list(choices) if choices is not None else None, default=default, console=self.console)
        pending = Question(
            question, tuple(choices) if choices is not None else None, default, tuple(content),
            tuple(images) if self.show_previews else (),
        )
        with self._lock:
            if self._stopping.is_set():
                raise ProcessCancelled("The run is stopping")
            self._question = pending
        pending.answered.wait()
        if pending.answer is None:
            raise ProcessCancelled("The run is stopping")
        return pending.answer

    def answer(self, value: str) -> str | None:
        with self._lock:
            question = self._question
            if question is None:
                return None
            value = value.strip() or question.default
            if value is None or (question.choices is not None and value not in question.choices):
                return "Choose: " + ", ".join(question.choices) if question.choices else "Please type an answer."
            question.answer = value
            self._question = None
            question.answered.set()
        return None

    def stop(self) -> None:
        with self._lock:
            self._stopping.set()
            if self._question is not None:
                self._question.answered.set()
                self._question = None

    def open_images(self, paths: Sequence[Path]) -> None:
        if not self.show_previews:
            return
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        for path in paths:
            self.log(f"Preview: {path}")
            try:
                subprocess.Popen([opener, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError as exc:
                self.log(f"Could not open preview: {exc}")

    def log(self, message: str) -> None:
        if not self.enabled:
            self.console.print(Text(message))
            return
        with self._lock:
            self._messages.append(f"{time.strftime('%H:%M:%S')} {message}")


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
        for column in ("asset", "state", "cycle", "score", "trellis", "retopo", "in state", "note"):
            table.add_column(column, overflow="fold")
        now = time.time()
        for asset in self._board.all():
            trellis = asset.extra.get("trellis", {})
            retopology = asset.extra.get("retopology") or {}
            table.add_row(
                asset.id,
                Text(asset.state.value, style=STATE_STYLES[asset.state]),
                str(asset.extra.get("cycle", "")),
                f"{asset.score:.2f}" if asset.score is not None else str(asset.extra.get("last_score", "")),
                f"{trellis['duration_s']:.0f}s" if trellis.get("duration_s") else "",
                f"{retopology['duration_s']:.0f}s" if retopology.get("duration_s") else "",
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
