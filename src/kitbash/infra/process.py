"""Subprocess execution with timeouts, logs and clean shutdown."""

import os
import signal
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
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


class ProcessRegistry:
    """Tracks running children so an interrupted run can stop them all."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processes: set[subprocess.Popen] = set()

    def add(self, process: subprocess.Popen) -> None:
        with self._lock:
            self._processes.add(process)

    def discard(self, process: subprocess.Popen) -> None:
        with self._lock:
            self._processes.discard(process)

    def terminate_all(self) -> None:
        with self._lock:
            processes = list(self._processes)
        for process in processes:
            _kill_group(process)


REGISTRY = ProcessRegistry()


def run_process(
    args: Sequence[str],
    *,
    timeout_s: float,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    log_path: Path | None = None,
    registry: ProcessRegistry = REGISTRY,
) -> ProcessResult:
    """Run ``args`` to completion or until ``timeout_s``; the whole process group is killed on timeout.

    With ``log_path`` the combined output streams into that file while the process runs (``tail -f`` it),
    and ``stdout`` of the result holds that combined output. Without it, stdout and stderr are captured apart.
    """
    started = time.monotonic()
    log = _open_log(log_path, args, cwd) if log_path else None
    process = subprocess.Popen(
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
    registry.add(process)
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_group(process)
        stdout, stderr = process.communicate()
    finally:
        registry.discard(process)
    duration = time.monotonic() - started
    if log is not None:
        status = "TIMEOUT" if timed_out else f"exit {process.returncode}"
        log.write(f"\n# {status} after {duration:.1f}s\n")
        log.close()
        stdout, stderr = log_path.read_text(encoding="utf-8", errors="replace"), ""
    return ProcessResult(
        args=tuple(args),
        returncode=process.returncode,
        stdout=stdout or "",
        stderr=stderr or "",
        duration_s=duration,
        timed_out=timed_out,
        log_path=log_path,
    )


def _open_log(path: Path, args: Sequence[str], cwd: Path | None):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8", buffering=1)
    handle.write(f"$ {' '.join(args)}\n# cwd: {cwd or os.getcwd()}\n\n")
    handle.flush()
    return handle


def _kill_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (ProcessLookupError, PermissionError):
        return
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
