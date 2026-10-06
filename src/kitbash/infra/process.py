"""Subprocess execution with timeouts, logs and clean shutdown."""

import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from pathlib import Path

from attrs import frozen


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
    A cancelled owner raises ``ProcessCancelled`` rather than returning a retryable tool failure.
    """
    registry = registry or _CURRENT.get() or ProcessRegistry()
    started = time.monotonic()
    child = _Child()
    log = None
    timed_out = False
    try:
        # Launch and registration share the shutdown gate. Register before spawning so interruption
        # cannot leave a successfully returned Popen outside its owner's cleanup scope.
        with defer_interrupts(), registry._lock:
            registry.check_cancelled()
            registry._children.add(child)
            log = _open_log(log_path, args, cwd) if log_path else None
            child.process = subprocess.Popen(
                list(args),
                cwd=cwd,
                env={**os.environ, **env} if env else None,
                stdin=subprocess.DEVNULL,
                stdout=log or subprocess.PIPE,
                stderr=subprocess.STDOUT if log else subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=True,
            )
        process = child.process
        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            child.terminate()
            stdout, stderr = process.communicate()
        child.terminate()
        registry.check_cancelled()
        duration = time.monotonic() - started
        if log is not None:
            status = "TIMEOUT" if timed_out else f"exit {process.returncode}"
            log.write(f"\n# {status} after {duration:.1f}s\n")
            log.flush()
            stdout, stderr = log_path.read_text(encoding="utf-8", errors="replace"), ""
        result = ProcessResult(
            args=tuple(args),
            returncode=process.returncode,
            stdout=stdout or "",
            stderr=stderr or "",
            duration_s=duration,
            timed_out=timed_out,
            log_path=log_path,
        )
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
    registry.check_cancelled()
    return result


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
