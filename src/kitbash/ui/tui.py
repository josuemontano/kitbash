"""Full-screen run monitor. Only this thread touches Textual widgets."""

import shlex
import threading
import time
from collections.abc import Callable
from io import UnsupportedOperation
from typing import Any, ClassVar

from rich.console import Group
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.events import Resize
from textual.widgets import Button, Footer, Input, RichLog, Static

from kitbash.analytics.context import propagate
from kitbash.infra.process import ProcessCancelled, defer_interrupts
from kitbash.ui.dashboard import Dashboard, JobOutput, Question


class _DescriptorlessCapture:
    """Keep Textual output capture without advertising an invalid OS descriptor."""

    def __init__(self, capture: Any) -> None:
        self._capture = capture

    def __getattr__(self, name: str) -> Any:
        return getattr(self._capture, name)

    def fileno(self) -> int:
        raise UnsupportedOperation("Textual output capture has no file descriptor")


class JobCard(Vertical):
    """One command's identity, lifecycle and independently scrollable output."""

    def __init__(self, job: JobOutput) -> None:
        super().__init__(classes="job")
        self.job = job
        self.sequence = 0
        trace = job.event.trace
        owner = trace.asset_id or trace.phase or "preflight"
        worker = trace.worker or "main"
        tool = trace.agent or job.event.args[0].rsplit("/", 1)[-1]
        self.border_title = f"{owner} · {worker} · {tool}"

    def compose(self) -> ComposeResult:
        yield Static(classes="job-status", markup=False)
        yield RichLog(max_lines=300, min_width=1, wrap=True, markup=False, highlight=False, classes="job-output")

    def on_mount(self) -> None:
        output = self.query_one(RichLog)
        output.write(Text("$ " + shlex.join(self.job.event.args)[:500], style="dim"))
        if self.job.event.log_path:
            output.write(Text(f"Log: {self.job.event.log_path}", style="dim"))
        self.update_job(self.job)

    def update_job(self, job: JobOutput) -> None:
        self.job = job
        duration = (job.finished or time.monotonic()) - job.started
        style = "green" if job.status == "done" else "cyan" if job.status == "running" else "red"
        self.query_one(".job-status", Static).update(Text(f"{job.status.upper()}  ·  {duration:.0f}s  ·  #{job.event.job_id[:6]}", style=style))
        self.set_class(job.status == "running", "running")
        self.set_class(job.status not in ("running", "done"), "failed")
        output = self.query_one(RichLog)
        if job.output and job.output[0][0] > self.sequence + 1:
            output.write(Text("… older output omitted from the live view", style="dim"))
        for sequence, stream, text in job.output:
            if sequence <= self.sequence:
                continue
            rendered = Text.from_ansi(text.replace("\r\n", "\n").replace("\r", "\n").removesuffix("\n"))
            if stream == "stderr":
                rendered = Text("stderr: ", style="yellow") + rendered
            output.write(rendered)
            self.sequence = sequence


class RunScreen(App):
    TITLE = "kitbash"
    CSS = """
    Screen { background: $surface; }
    #title { height: 3; padding: 1 2; background: $primary-background; color: $text; text-style: bold; }
    #workspace { height: 1fr; }
    #monitor { width: 1fr; }
    #progress-scroll { height: auto; max-height: 12; }
    #progress { height: auto; padding: 0 1; }
    #jobs-label { height: 1; padding: 0 2; color: $text-muted; }
    #jobs { height: auto; grid-size: 2; grid-columns: 1fr 1fr; grid-rows: 14; grid-gutter: 0 1; padding: 0 1; }
    #jobs.narrow { grid-size: 1; grid-columns: 1fr; }
    .job { height: 14; border: round $primary; padding: 0 1; }
    .job.running { border: round $accent; }
    .job.failed { border: round $error; }
    .job-status { height: 1; }
    .job-output { height: 1fr; scrollbar-size: 1 1; }
    #events { height: auto; max-height: 7; padding: 0 2; color: $text-muted; }
    #review { width: 42%; min-width: 30; border: round $success; padding: 0 1; display: none; }
    #review.visible { display: block; }
    #review-body { height: 1fr; }
    #review-content { height: auto; }
    #preview-paths { height: auto; color: $text-muted; }
    #open-previews { margin: 1 0; }
    #question { height: auto; margin-top: 1; }
    #answer { margin: 1 0; }
    #validation { height: auto; color: $error; }
    #bottom-bar { dock: bottom; height: 2; }
    #status { height: 1; padding: 0 1; background: $primary-background; color: $text; }
    Footer { dock: bottom; }
    Screen.compact #title { height: 1; padding: 0 1; }
    Screen.compact #workspace { layout: vertical; }
    Screen.compact #progress-scroll { max-height: 5; }
    Screen.compact #review { width: 100%; height: 55%; min-height: 8; }
    """
    BINDINGS: ClassVar = [
        Binding("ctrl+c", "stop", "Stop run", priority=True),
        Binding("ctrl+q", "stop", "Stop run", show=False, priority=True),
        Binding("f2", "previews", "Open previews"),
    ]

    def __init__(self, dashboard: Dashboard, operation: Callable[[], Any], cancel: Callable[[], None]) -> None:
        super().__init__()
        # Textual 6 returns -1 from fileno(); multiprocessing's resource tracker
        # includes it in passfds when a lazy dependency first creates a lock.
        # Descriptorless streams must raise instead. Leave all writes captured,
        # without temporarily changing process-global stderr in worker threads.
        self._capture_stdout = _DescriptorlessCapture(self._capture_stdout)
        self._capture_stderr = _DescriptorlessCapture(self._capture_stderr)
        self.dashboard = dashboard
        self.operation = operation
        self.cancel = cancel
        self.result: Any = None
        self.error: BaseException | None = None
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self._cancel_thread: threading.Thread | None = None
        self._stopping = False
        self._started = time.monotonic()
        self._cards: dict[str, JobCard] = {}
        self._question: Question | None = None
        self._last_events: tuple[str, ...] = ()

    def compose(self) -> ComposeResult:
        yield Static("KITBASH  /  scene workshop", id="title")
        with Horizontal(id="workspace"):
            with VerticalScroll(id="monitor"):
                with VerticalScroll(id="progress-scroll"):
                    yield Static(id="progress")
                yield Static("PARALLEL JOBS  ·  live command output", id="jobs-label")
                yield Grid(id="jobs")
                yield Static(id="events", markup=False)
            with Vertical(id="review"):
                with VerticalScroll(id="review-body"):
                    yield Static(id="review-content")
                    yield Static(id="preview-paths", markup=False)
                    yield Button("Open previews (F2)", id="open-previews")
                yield Static(id="question", markup=False)
                yield Static(id="validation", markup=False)
                yield Input(placeholder="Type your answer, then Enter", id="answer")
        with Vertical(id="bottom-bar"):
            yield Static("Starting…", id="status", markup=False)
            yield Footer()

    def on_mount(self) -> None:
        self._thread = threading.Thread(target=propagate(self._execute), name="kitbash-pipeline")
        self._thread.start()
        self.set_interval(1 / max(1, self.dashboard.refresh_per_second), self._refresh_run)

    def on_resize(self, event: Resize) -> None:
        self.screen.set_class(event.size.width < 100, "compact")

    def _execute(self) -> None:
        try:
            with self.dashboard.observing():
                self.result = self.operation()
                if self._stopping:
                    self.error = KeyboardInterrupt()
        except ProcessCancelled:
            self.error = KeyboardInterrupt()
        except BaseException as exc:
            self.error = exc
        finally:
            self._done.set()

    async def _refresh_run(self) -> None:
        body, jobs, messages, question = self.dashboard.snapshot()
        self.query_one("#progress", Static).update(body)
        grid = self.query_one("#jobs", Grid)
        grid.set_class(grid.size.width < 80, "narrow")
        retained = {job.event.job_id for job in jobs}
        for job_id in tuple(self._cards):
            if job_id not in retained:
                await self._cards.pop(job_id).remove()
        for job in jobs:
            key = job.event.job_id
            if key not in self._cards:
                card = self._cards[key] = JobCard(job)
                await grid.mount(card, before=0)
            else:
                self._cards[key].update_job(job)
        if messages != self._last_events:
            self.query_one("#events", Static).update(Text("\n".join(messages[-6:])))
            self._last_events = messages
        if question is not self._question:
            self._show_question(question)
        running = sum(job.finished is None for job in jobs)
        failed = sum(job.status.startswith("failed") or job.status == "timed out" for job in jobs)
        elapsed = int(time.monotonic() - self._started)
        state = "STOPPING · waiting for workers" if self._stopping else "INPUT NEEDED" if question else "RUNNING"
        self.query_one("#status", Static).update(
            f"{state}  |  {running} jobs running  |  {failed} recent failures  |  {elapsed // 60:02d}:{elapsed % 60:02d}"
        )
        if self._done.is_set():
            self.exit()

    def _show_question(self, question: Question | None) -> None:
        self._question = question
        self.query_one("#review").set_class(question is not None, "visible")
        if question is None:
            return
        self.query_one("#review-content", Static).update(Group(*question.content))
        self.query_one("#preview-paths", Static).update(Text("\n".join(str(path) for path in question.images)))
        self.query_one("#open-previews").display = bool(question.images)
        choices = " (" + "/".join(question.choices) + ")" if question.choices else ""
        default = f" [Enter: {question.default}]" if question.default else ""
        self.query_one("#question", Static).update(Text(question.text + choices + default))
        self.query_one("#validation", Static).update("")
        answer = self.query_one("#answer", Input)
        answer.value = ""
        answer.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if self._question is None or self._stopping:
            return
        if error := self.dashboard.answer(event.value):
            self.query_one("#validation", Static).update(Text(error))
        else:
            self._show_question(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "open-previews":
            self.action_previews()

    def action_previews(self) -> None:
        if self._question:
            self.dashboard.open_images(self._question.images)

    def action_stop(self) -> None:
        if self._stopping or self._done.is_set():
            return
        self._stopping = True
        self.dashboard.stop()
        self._cancel_thread = threading.Thread(target=self.cancel, name="kitbash-cancel")
        self._cancel_thread.start()

    def action_quit(self) -> None:
        self.action_stop()

    def join(self) -> None:
        """Also clean up if Textual itself fails or receives a terminal shutdown."""
        with defer_interrupts():
            if self._thread is not None:
                if not self._done.is_set():
                    self.dashboard.stop()
                    self.cancel()
                self._thread.join()
            if self._cancel_thread is not None:
                self._cancel_thread.join()
