"""Subprocess execution with timeouts, logs and clean shutdown."""

import codecs
import io
import os
import selectors
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from pathlib import Path
from typing import BinaryIO, TextIO
from uuid import uuid4

from attrs import field, frozen

from kitbash.analytics import context
from kitbash.analytics.context import TraceContext


@frozen
class ProcessEvent:
    """One invocation's lifecycle or a bounded, not necessarily line-complete output chunk."""

    kind: str
    job_id: str
    args: tuple[str, ...] = ()
    trace: TraceContext = field(factory=TraceContext)
    log_path: Path | None = None
    text: str = ""
    stream: str = "stdout"
    returncode: int | None = None
    timed_out: bool = False
    cancelled: bool = False


_OBSERVER: ContextVar[Callable[[ProcessEvent], None] | None] = ContextVar("process_observer", default=None)
_OUTPUT_CHUNK = 8192


@contextmanager
def observe_processes(callback: Callable[[ProcessEvent], None]) -> Iterator[None]:
    """Observe this context's calls; propagate the context to observe work in other threads.

    Callbacks run synchronously on the launching thread and must not block. A nested observer
    replaces the outer observer until its scope exits. Callback failures unwind and clean up the child.
    """
    token = _OBSERVER.set(callback)
    try:
        yield
    finally:
        _OBSERVER.reset(token)


@frozen
class ProcessResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool
    log_path: Path | None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def tail(self, lines: int = 40) -> str:
        text = (self.stderr.strip() or self.stdout.strip()).splitlines()
        return "\n".join(text[-lines:])


class ProcessCancelled(BaseException):
    """Shutdown, not a tool failure: bypass ordinary error handling and retry loops."""


class _Child:
    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None
        self.timed_out = False
        self._lock = threading.Lock()
        self._stopped = False

    def terminate(self) -> None:
        with self._lock:
            if self.process is not None and not self._stopped:
                _kill_group(self.process)
                self._stopped = True


class ProcessRegistry:
    """Owns one run's children; cancellation closes the launch gate permanently for that run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._children: set[_Child] = set()
        self._cancelled = threading.Event()

    @contextmanager
    def bind(self) -> Iterator[None]:
        """Use this owner for nested tool calls in the current thread."""
        token = _CURRENT.set(self)
        try:
            yield
        finally:
            _CURRENT.reset(token)

    def check_cancelled(self) -> None:
        if self._cancelled.is_set():
            raise ProcessCancelled("The run is stopping")

    def terminate_all(self) -> None:
        with defer_interrupts():
            with self._lock:
                self._cancelled.set()
                children = tuple(self._children)
            for child in children:
                child.terminate()


_CURRENT: ContextVar[ProcessRegistry | None] = ContextVar("process_registry", default=None)
_DEFERRING: ContextVar[bool] = ContextVar("deferring_interrupts", default=False)


def current_registry() -> ProcessRegistry:
    """Return the bound owner, or a fresh owner for a standalone concurrent operation."""
    return _CURRENT.get() or ProcessRegistry()


@contextmanager
def defer_interrupts() -> Iterator[None]:
    """Finish teardown before honoring another Ctrl+C, preserving an error already unwinding."""
    if threading.current_thread() is not threading.main_thread() or _DEFERRING.get():
        yield
        return
    interrupted = False
    unwinding = sys.exception() is not None

    def interrupt(signum, frame) -> None:
        nonlocal interrupted
        interrupted = True

    previous = signal.signal(signal.SIGINT, interrupt)
    token = _DEFERRING.set(True)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
        _DEFERRING.reset(token)
    if interrupted and not unwinding:
        raise KeyboardInterrupt


def run_process(
    args: Sequence[str],
    *,
    timeout_s: float,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    log_path: Path | None = None,
    registry: ProcessRegistry | None = None,
) -> ProcessResult:
    """Run a child in its own session, owning its group until all exit paths are cleaned up.

    With ``log_path`` combined output streams into that file; otherwise streams are captured apart.
    Observers receive both streams separately, including partial lines, before the command finishes.
    A cancelled owner raises ``ProcessCancelled`` rather than returning a retryable tool failure.
    """
    registry = registry or current_registry()
    command, trace, job_id = tuple(args), context.current(), uuid4().hex
    observer = _OBSERVER.get()

    def emit(kind: str, **values) -> None:
        if observer is not None:
            observer(ProcessEvent(kind, job_id, args=command, trace=trace, log_path=log_path, **values))

    started = time.monotonic()
    child = _Child()
    log = None
    cancelled = False
    failure: BaseException | None = None
    try:
        emit("started")
        # Launch and registration share the shutdown gate. Register before spawning so interruption
        # cannot leave a successfully returned Popen outside its owner's cleanup scope.
        with defer_interrupts(), registry._lock:
            registry.check_cancelled()
            registry._children.add(child)
            log = _open_log(log_path, command, cwd) if log_path else None
            child.process = subprocess.Popen(
                command,
                cwd=cwd,
                env={**os.environ, **env} if env else None,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                start_new_session=True,
            )
        process = child.process
        stdout, stderr = _stream_output(child, timeout_s, log, emit)
        child.terminate()
        registry.check_cancelled()
        duration = time.monotonic() - started
        if log is not None:
            status = "TIMEOUT" if child.timed_out else f"exit {process.returncode}"
            log.write(f"\n# {status} after {duration:.1f}s\n")
            log.flush()
            stdout, stderr = log_path.read_text(encoding="utf-8", errors="replace"), ""
        result = ProcessResult(
            args=command,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
            duration_s=duration,
            timed_out=child.timed_out,
            log_path=log_path,
        )
        registry.check_cancelled()
    except BaseException as exc:
        failure = exc
        cancelled = isinstance(exc, (ProcessCancelled, KeyboardInterrupt))
        raise
    finally:
        with defer_interrupts():
            try:
                child.terminate()
            finally:
                process = child.process
                if process is not None:
                    for pipe in (process.stdin, process.stdout, process.stderr):
                        if pipe is not None:
                            pipe.close()
                if log is not None:
                    log.close()
                with registry._lock:
                    if child.process is None or child._stopped:
                        registry._children.discard(child)
                # A broken observer must not replace the failure already being unwound.
                unwinding = sys.exception() is not None
                try:
                    emit(
                        "finished", returncode=process.returncode if process is not None else None,
                        timed_out=child.timed_out, cancelled=cancelled or registry._cancelled.is_set(),
                        text=str(failure) if failure is not None else "",
                    )
                except BaseException:
                    if not unwinding:
                        raise
    registry.check_cancelled()
    return result


def _stream_output(
    child: _Child, timeout_s: float, log: TextIO | None, emit: Callable[..., None],
) -> tuple[str, str]:
    """Multiplex raw pipes without line buffering or reader threads that could outlive the run."""
    process = child.process
    assert process is not None and process.stdout is not None and process.stderr is not None
    captured: dict[str, list[str]] = {"stdout": [], "stderr": []}
    decoders = {
        stream: io.IncrementalNewlineDecoder(codecs.getincrementaldecoder("utf-8")("replace"), translate=True)
        for stream in captured
    }

    def output(stream: str, data: bytes, *, final: bool = False) -> None:
        if log is not None and data:
            # Preserve the file's original bytes, even across split/invalid UTF-8 sequences.
            log.buffer.write(data)
            log.flush()
        text = decoders[stream].decode(data, final=final)
        if text:
            if log is None:
                captured[stream].append(text)
            emit("output", stream=stream, text=text)

    deadline = time.monotonic() + timeout_s
    drain_deadline: float | None = None
    with selectors.DefaultSelector() as selector:
        for stream, pipe in (("stdout", process.stdout), ("stderr", process.stderr)):
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, stream)
        while selector.get_map() or process.poll() is None:
            remaining = deadline - time.monotonic()
            if not child._stopped:
                # Logged calls have always waited for the leader, not pipe-owning descendants.
                if log is not None and process.poll() is not None:
                    child.terminate()
                elif remaining <= 0:
                    child.timed_out = True
                    child.terminate()
            if child._stopped:
                if drain_deadline is None:
                    drain_deadline = time.monotonic() + 1
                elif time.monotonic() >= drain_deadline:
                    break
            ready = selector.select(0 if child._stopped else min(0.1, max(0, remaining)))
            if child._stopped and not ready:
                break
            for key, _ in ready:
                pipe: BinaryIO = key.fileobj
                try:
                    data = os.read(pipe.fileno(), _OUTPUT_CHUNK)
                except BlockingIOError:
                    continue
                output(key.data, data, final=not data)
                if not data:
                    selector.unregister(pipe)
        # An escaped descendant can retain or continually write a pipe after our group is gone.
        # Drain buffered output after cleanup, but never wait indefinitely for that unrelated owner.
        for key in tuple(selector.get_map().values()):
            output(key.data, b"", final=True)
    process.wait()
    return "".join(captured["stdout"]), "".join(captured["stderr"])


def _open_log(path: Path, args: Sequence[str], cwd: Path | None):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8", buffering=1)
    try:
        handle.write(f"$ {' '.join(args)}\n# cwd: {cwd or os.getcwd()}\n\n")
        handle.flush()
        return handle
    except BaseException:
        handle.close()
        raise


def _kill_group(process: subprocess.Popen) -> None:
    # The group can outlive its leader (and keep captured pipes open). Never use poll() as the
    # condition for signalling it. Reap the leader even if the group already disappeared.
    _signal_group(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 5
    while _group_exists(process.pid):
        process.poll()
        if time.monotonic() >= deadline:
            _signal_group(process.pid, signal.SIGKILL)
            break
        time.sleep(0.02)
    process.wait()


def _signal_group(pid: int, sig: int) -> None:
    # Darwin reports EPERM for a group containing only zombie processes.
    with suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, sig)


def _group_exists(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
