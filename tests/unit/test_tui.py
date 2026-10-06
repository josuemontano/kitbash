import errno
import json
import os
import select
import subprocess
import sys
import textwrap
import time

import pytest


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX descriptor inheritance and pseudo-terminal regression")
@pytest.mark.parametrize("fail_operation", [False, True], ids=["success", "pipeline-error"])
def test_interactive_capture_allows_fresh_multiprocessing_spawn(tmp_path, fail_operation):
    import pty
    import termios

    # A fresh interpreter is essential: an already-running resource tracker hides
    # Textual's invalid fileno(), which breaks the first lazy tqdm/model lock.
    script = textwrap.dedent("""\
        import json
        import multiprocessing
        import sys
        import threading
        from pathlib import Path

        from rich.console import Console

        from kitbash.ui import tui
        from kitbash.ui.dashboard import Dashboard

        captured = []

        class CaptureScreen(tui.RunScreen):
            def on_mount(self):
                # Textual dispatches base-class handlers automatically.
                self.begin_capture_print(self)

            def on_print(self, event):
                captured.append((event.stderr, event.text))

        tui.RunScreen = CaptureScreen
        original_stdout, original_stderr = sys.stdout, sys.stderr
        assert sys.stdin.isatty() and sys.stdout.isatty()
        dashboard = Dashboard(Console(), refresh_per_second=20, show_previews=False)
        failure = ValueError("original pipeline failure")
        fail_operation = sys.argv[2] == "True"

        def operation():
            assert dashboard.enabled
            assert threading.current_thread().name == "kitbash-pipeline"
            stdout, stderr = sys.stdout, sys.stderr
            assert stdout is not original_stdout and stderr is not original_stderr
            print("captured stdout", flush=True)
            print("captured stderr", file=sys.stderr, flush=True)
            context = multiprocessing.get_context("spawn")
            # Real named semaphore registration launches the resource tracker.
            # Then exercise descriptor passing and IPC in a spawned child too.
            lock = context.RLock()
            with lock:
                queue = context.Queue()
                child = context.Process(target=queue.put, args=("child completed",))
                try:
                    child.start()
                    assert queue.get(timeout=15) == "child completed"
                    child.join(15)
                    assert child.exitcode == 0
                finally:
                    if child.pid is not None:
                        if child.is_alive():
                            child.kill()
                            child.join()
                        child.close()
                    queue.close()
                    queue.join_thread()
            assert sys.stdout is stdout and sys.stderr is stderr
            if fail_operation:
                raise failure
            return "completed"

        try:
            result = dashboard.run(operation, lambda: None)
        except ValueError as exc:
            assert fail_operation and exc is failure
        else:
            assert not fail_operation and result == "completed"
        assert not dashboard.enabled
        assert sys.stdout is original_stdout and sys.stderr is original_stderr
        assert "".join(text for stderr, text in captured if not stderr) == "captured stdout\\n"
        assert "".join(text for stderr, text in captured if stderr) == "captured stderr\\n"
        Path(sys.argv[1]).write_text(json.dumps({"captured": captured, "restored": True}))
        """)
    report = tmp_path / "capture.json"
    master, slave = pty.openpty()
    try:
        termios.tcsetwinsize(slave, (30, 100))
        process = subprocess.Popen(
            [sys.executable, "-c", script, str(report), str(fail_operation)],
            stdin=slave, stdout=slave, stderr=slave,
            env={**os.environ, "TERM": "xterm-256color"},
        )
    except BaseException:
        os.close(master)
        raise
    finally:
        os.close(slave)
    terminal = bytearray()
    deadline = time.monotonic() + 45
    try:
        # Drain the terminal while running so a full PTY buffer cannot stall the UI.
        while time.monotonic() < deadline:
            if select.select([master], [], [], 0.1)[0]:
                try:
                    chunk = os.read(master, 65536)
                except OSError as exc:
                    if exc.errno != errno.EIO:
                        raise
                    break
                if not chunk:
                    break
                terminal.extend(chunk)
            elif process.poll() is not None:
                break
        process.wait(timeout=max(0.1, deadline - time.monotonic()))
        assert process.returncode == 0, terminal.decode(errors="replace")
        assert json.loads(report.read_text())["restored"]
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        os.close(master)
