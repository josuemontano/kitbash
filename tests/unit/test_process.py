import os
import signal
import subprocess
import sys
import threading
import time

import pytest

from kitbash.analytics import context
from kitbash.infra import process as processes
from kitbash.infra.process import ProcessCancelled, ProcessRegistry, current_registry, observe_processes, run_process
from tests.helpers import process_gone as gone
from tests.helpers import wait_for


@pytest.mark.parametrize("logged", [False, True])
def test_partial_streams_arrive_before_exit_and_decode_incrementally(tmp_path, logged):
    stdout_seen, stderr_seen, unicode_seen = (tmp_path / name for name in ("stdout", "stderr", "unicode"))
    log_path = tmp_path / "stream.log" if logged else None
    code = (
        "import os,time\nfrom pathlib import Path\n"
        "os.write(1, b'out-\\xe2')\nos.write(2, b'err-')\n"
        f"while not (Path({str(stdout_seen)!r}).exists() and Path({str(stderr_seen)!r}).exists()): time.sleep(.01)\n"
        "os.write(1, b'\\x82\\xac\\r')\n"
        f"while not Path({str(unicode_seen)!r}).exists(): time.sleep(.01)\n"
        "os.write(1, b'\\nend\\xff')\nos.write(2, b'done\\n')\n"
    )
    events = []

    def observe(event):
        events.append(event)
        if event.kind == "output":
            if "out-" in event.text:
                if log_path is not None:
                    assert b"out-\xe2" in log_path.read_bytes()
                stdout_seen.touch()
            if "err-" in event.text:
                stderr_seen.touch()
            if "\u20ac" in event.text:
                unicode_seen.touch()

    command = [sys.executable, "-c", code]
    with context.bind(phase="modelling", asset_id="crate", worker="worker-1"), observe_processes(observe):
        result = run_process(command, timeout_s=10, log_path=log_path)
    assert result.ok
    assert events[0].kind == "started" and events[-1].kind == "finished"
    assert len({event.job_id for event in events}) == 1
    assert all(event.args == tuple(command) and event.log_path == log_path for event in events)
    assert all((event.trace.phase, event.trace.asset_id, event.trace.worker) == ("modelling", "crate", "worker-1") for event in events)
    assert "".join(event.text for event in events if event.kind == "output" and event.stream == "stdout") == "out-\u20ac\nend\ufffd"
    assert "".join(event.text for event in events if event.kind == "output" and event.stream == "stderr") == "err-done\n"
    assert events[-1].returncode == 0 and not events[-1].timed_out and not events[-1].cancelled
    if logged:
        assert result.stderr == "" and result.stdout == log_path.read_text(encoding="utf-8", errors="replace")
        assert b"end\xff" in log_path.read_bytes() and "# exit 0" in result.stdout
    else:
        assert (result.stdout, result.stderr) == ("out-\u20ac\nend\ufffd", "err-done\n")


@pytest.mark.parametrize("logged", [False, True])
def test_verbose_streams_are_drained_in_bounded_chunks(tmp_path, logged):
    events = []
    size = 1_048_576
    code = f"import sys; sys.stdout.write('o' * {size}); sys.stdout.flush(); sys.stderr.write('e' * {size}); sys.stderr.flush()"
    with observe_processes(events.append):
        result = run_process([sys.executable, "-c", code], timeout_s=10, log_path=tmp_path / "verbose.log" if logged else None)
    assert result.ok
    chunks = [event for event in events if event.kind == "output"]
    assert all(0 < len(event.text) <= processes._OUTPUT_CHUNK for event in chunks)
    assert "".join(event.text for event in chunks if event.stream == "stdout") == "o" * size
    assert "".join(event.text for event in chunks if event.stream == "stderr") == "e" * size
    if logged:
        # Separate pipes can become readable together: only per-stream order is defined.
        payload = result.stdout.split("\n\n", 1)[1].rsplit("\n#", 1)[0]
        assert len(payload) == 2 * size and payload.count("o") == size and payload.count("e") == size
        assert result.stderr == ""
        assert result.stdout == result.log_path.read_text()
    else:
        assert (result.stdout, result.stderr) == ("o" * size, "e" * size)


@pytest.mark.parametrize("timed_out", [False, True], ids=["failure", "timeout"])
def test_finished_event_reports_failure_and_timeout(tmp_path, timed_out):
    events = []
    code = "import os,time; os.write(2, b'last partial'); " + ("time.sleep(60)" if timed_out else "raise SystemExit(7)")
    with observe_processes(events.append):
        result = run_process([sys.executable, "-c", code], timeout_s=1, log_path=tmp_path / "failed.log")
    assert not result.ok and result.timed_out is timed_out
    assert "last partial" in result.stdout
    finished = events[-1]
    assert finished.kind == "finished" and finished.returncode == result.returncode
    assert finished.timed_out is timed_out and not finished.cancelled
    assert any(event.kind == "output" and event.stream == "stderr" and event.text == "last partial" for event in events)
    if not timed_out:
        assert result.returncode == 7


@pytest.mark.parametrize("repeat_interrupt", [False, True], ids=["term", "repeated-interrupt-and-kill"])
def test_ctrl_c_reaps_child_and_preserves_interrupt(tmp_path, repeat_interrupt):
    pid_path = tmp_path / "pid"
    script = tmp_path / "interrupt.py"
    child_code = "import os,signal,time; from pathlib import Path; "
    if repeat_interrupt:
        child_code += "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
    child_code += f"Path({str(pid_path)!r}).write_text(str(os.getpid())); time.sleep(60)"
    script.write_text(
        "import sys\nfrom kitbash.infra.process import run_process\n"
        "try:\n"
        "    run_process([sys.executable, '-c', "
        + repr(child_code)
        + f"], timeout_s=90, log_path=__import__('pathlib').Path({str(tmp_path / 'child.log')!r}))\n"
        "except KeyboardInterrupt:\n"
        "    import os\n"
        f"    child_pid = int(__import__('pathlib').Path({str(pid_path)!r}).read_text())\n"
        "    try:\n"
        "        os.waitpid(child_pid, os.WNOHANG)\n"
        "    except ChildProcessError:\n"
        "        pass\n"
        "    else:\n"
        "        raise AssertionError('direct child was not reaped')\n"
        "    print('original KeyboardInterrupt', flush=True)\n"
        "    raise SystemExit(130)\n"
    )
    harness = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        wait_for(lambda: pid_path.exists() and pid_path.stat().st_size)
        child_pid = int(pid_path.read_text())
        harness.send_signal(signal.SIGINT)
        if repeat_interrupt:
            time.sleep(0.1)
            harness.send_signal(signal.SIGINT)
        stdout, stderr = harness.communicate(timeout=15)
        assert harness.returncode == 130, stderr
        assert stdout.strip() == "original KeyboardInterrupt"
        wait_for(lambda: gone(child_pid))
    finally:
        if harness.poll() is None:
            harness.kill()
        harness.wait()
        harness.stdout.close()
        harness.stderr.close()
        if pid_path.exists() and not gone(int(pid_path.read_text())):
            os.killpg(int(pid_path.read_text()), signal.SIGKILL)


@pytest.mark.parametrize("logged", [False, True], ids=["captured-timeout", "logged-success"])
def test_exited_leader_does_not_orphan_term_resistant_descendant(tmp_path, logged):
    pid_path = tmp_path / "descendant"
    code = (
        "import os,signal,time\nfrom pathlib import Path\n"
        "if os.fork() == 0:\n"
        "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"    Path({str(pid_path)!r}).write_text(str(os.getpid()))\n"
        "    time.sleep(60)\n"
        "else:\n"
        f"    while not Path({str(pid_path)!r}).exists(): time.sleep(.01)\n"
        "    os._exit(0)\n"
    )
    try:
        result = run_process(
            [sys.executable, "-c", code], timeout_s=0.5,
            log_path=tmp_path / "group.log" if logged else None,
        )
        assert result.timed_out is not logged
        assert result.returncode == 0
        wait_for(lambda: gone(int(pid_path.read_text())))
    finally:
        if pid_path.exists() and not gone(int(pid_path.read_text())):
            os.kill(int(pid_path.read_text()), signal.SIGKILL)


def test_shutdown_racing_spawn_closes_launch_gate_and_reaps(tmp_path, monkeypatch):
    registry = ProcessRegistry()
    spawned = threading.Event()
    release_spawn = threading.Event()
    stopping = threading.Event()
    child = []
    errors = []
    events = []
    popen = subprocess.Popen

    def paused_spawn(*args, **kwargs):
        process = popen(*args, **kwargs)
        child.append(process)
        spawned.set()
        assert release_spawn.wait(10)
        return process

    def run():
        try:
            with observe_processes(events.append):
                run_process([sys.executable, "-c", "import time; time.sleep(60)"], timeout_s=90, registry=registry)
        except BaseException as exc:
            errors.append(exc)

    def stop():
        stopping.set()
        registry.terminate_all()

    monkeypatch.setattr(processes.subprocess, "Popen", paused_spawn)
    runner = threading.Thread(target=run)
    stopper = threading.Thread(target=stop)
    runner.start()
    try:
        assert spawned.wait(10)
        stopper.start()
        assert stopping.wait(10)
        release_spawn.set()
        stopper.join(15)
        runner.join(15)
        assert not stopper.is_alive() and not runner.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], ProcessCancelled)
        assert child[0].returncode is not None
        assert child[0].stdout.closed and child[0].stderr.closed
        assert gone(child[0].pid)
        assert events[0].kind == "started" and events[-1].kind == "finished"
        assert events[-1].cancelled and not events[-1].timed_out
        assert events[-1].returncode == child[0].returncode
        with pytest.raises(ProcessCancelled):
            run_process([sys.executable, "-c", "raise SystemExit(99)"], timeout_s=5, registry=registry)
        monkeypatch.setattr(processes.subprocess, "Popen", popen)
        assert run_process([sys.executable, "-c", "print('fresh run')"], timeout_s=5).stdout.strip() == "fresh run"
    finally:
        release_spawn.set()
        registry.terminate_all()
        runner.join(15)
        if stopper.ident is not None:
            stopper.join(15)


def test_spawn_failure_closes_log(tmp_path, monkeypatch):
    handles = []
    events = []
    open_log = processes._open_log

    def remember_log(*args):
        handle = open_log(*args)
        handles.append(handle)
        return handle

    monkeypatch.setattr(processes, "_open_log", remember_log)
    with observe_processes(events.append), pytest.raises(FileNotFoundError):
        run_process([str(tmp_path / "missing-executable")], timeout_s=5, log_path=tmp_path / "spawn.log")
    assert handles[0].closed
    assert [event.kind for event in events] == ["started", "finished"]
    assert events[-1].returncode is None and not events[-1].cancelled
    assert "missing-executable" in events[-1].text


def test_unexpected_stream_error_reaps_and_closes_pipes(monkeypatch):
    children = []
    original = RuntimeError("communication interrupted")
    popen = subprocess.Popen

    def remember_child(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)
        return child

    def fail(*args, **kwargs):
        raise original

    monkeypatch.setattr(processes.subprocess, "Popen", remember_child)
    monkeypatch.setattr(processes, "_stream_output", fail)
    with pytest.raises(RuntimeError) as caught:
        run_process([sys.executable, "-c", "import time; time.sleep(60)"], timeout_s=90)
    assert caught.value is original
    assert children[0].returncode is not None
    assert children[0].stdout.closed and children[0].stderr.closed


def test_interrupt_during_parallel_preflight_reaps_all_pings(tmp_path):
    executable = tmp_path / "omp"
    executable.write_text(
        f"#!{sys.executable}\nimport os,sys,time\nfrom pathlib import Path\n"
        "model = sys.argv[sys.argv.index('--model') + 1]\n"
        f"(Path({str(tmp_path)!r}) / model).write_text(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    executable.chmod(0o755)
    script = tmp_path / "preflight.py"
    script.write_text(
        "from pathlib import Path\n"
        "from kitbash.config import load_config\n"
        "from kitbash.domain.roles import Role\n"
        "from kitbash.services.preflight import ping_models\n"
        "try:\n"
        f"    ping_models(load_config(None), {str(executable)!r}, {{'first': Role.CODE, 'second': Role.CODE}}, Path({str(tmp_path)!r}))\n"
        "except KeyboardInterrupt:\n"
        "    raise SystemExit(130)\n"
    )
    harness = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    markers = [tmp_path / name for name in ("first", "second")]
    try:
        wait_for(lambda: all(path.exists() and path.stat().st_size for path in markers))
        harness.send_signal(signal.SIGINT)
        _, stderr = harness.communicate(timeout=15)
        assert harness.returncode == 130, stderr
        assert all(gone(int(path.read_text())) for path in markers)
    finally:
        if harness.poll() is None:
            harness.kill()
        harness.wait()
        harness.stdout.close()
        harness.stderr.close()
        for path in markers:
            if path.exists() and path.stat().st_size and not gone(int(path.read_text())):
                os.killpg(int(path.read_text()), signal.SIGKILL)


def test_last_preflight_task_cancellation_cannot_report_success(tmp_path, monkeypatch):
    from kitbash.config import load_config
    from kitbash.domain.roles import Role
    from kitbash.services.preflight import OmpClient, ping_models

    def cancel(self, request):
        current_registry().terminate_all()

    monkeypatch.setattr(OmpClient, "complete", cancel)
    with pytest.raises(ProcessCancelled):
        ping_models(load_config(None), "unused", {"last": Role.CODE}, tmp_path)


def test_cancelled_preflight_does_not_start_another_model_request(tmp_path, monkeypatch):
    from kitbash.config import load_config
    from kitbash.domain.roles import Role
    from kitbash.services.preflight import OmpClient, ping_models

    def unexpected_request(self, request):
        pytest.fail("cancelled preflight invoked the model")

    monkeypatch.setattr(OmpClient, "complete", unexpected_request)
    owner = ProcessRegistry()
    owner.terminate_all()
    with owner.bind(), pytest.raises(ProcessCancelled):
        ping_models(load_config(None), "unused", {"never": Role.CODE}, tmp_path)


def test_preflight_fatal_cancels_earlier_ping_without_masking_original(tmp_path, monkeypatch):
    from kitbash.config import load_config
    from kitbash.domain.roles import Role
    from kitbash.services.preflight import OmpClient, ping_models

    pid_path = tmp_path / "running-ping"
    registry = ProcessRegistry()
    original = FileNotFoundError("configured model executable disappeared")
    # Bound the broken implementation: ordered map masks the completed failure with cancellation.
    watchdog = threading.Timer(5, registry.terminate_all)

    def complete(self, request):
        if request.model == "running":
            run_process([
                sys.executable, "-c",
                f"import os,time; from pathlib import Path; Path({str(pid_path)!r}).write_text(str(os.getpid())); time.sleep(60)",
            ], timeout_s=90)
        else:
            wait_for(lambda: pid_path.exists() and pid_path.stat().st_size)
            watchdog.start()
            raise original

    monkeypatch.setattr(OmpClient, "complete", complete)
    try:
        with registry.bind(), pytest.raises(FileNotFoundError) as caught:
            ping_models(load_config(None), "unused", {"running": Role.CODE, "broken": Role.CODE}, tmp_path)
        assert caught.value is original
        assert gone(int(pid_path.read_text()))
        with pytest.raises(ChildProcessError):
            os.waitpid(int(pid_path.read_text()), os.WNOHANG)
    finally:
        watchdog.cancel()
        if watchdog.ident is not None:
            watchdog.join()
        registry.terminate_all()


def test_parallel_workers_share_observer_and_owner_with_distinct_job_context(tmp_path):
    from kitbash.analytics.tracker import Tracker
    from kitbash.pipeline.scheduler import Scheduler
    from kitbash.pipeline.workers import WorkerPool
    from kitbash.store.state import StateDB

    state = StateDB(tmp_path / "state.db")
    tracker = Tracker(state.spans)
    registry = ProcessRegistry()
    scheduler = Scheduler(2, tracker)
    release = tmp_path / "release"
    completed = threading.Event()
    lock = threading.Lock()
    events, delivered = [], []
    live = set()

    def observe(event):
        with lock:
            events.append(event)
            if event.kind == "output":
                live.add(event.trace.asset_id)
                if len(live) == 2:
                    release.touch()

    def work(asset_id):
        assert current_registry() is registry
        result = run_process([
            sys.executable, "-c",
            "import time\nfrom pathlib import Path\n"
            f"print({asset_id!r}, flush=True)\n"
            f"while not Path({str(release)!r}).exists(): time.sleep(.01)\n",
        ], timeout_s=10)
        assert result.ok and result.stdout.strip() == asset_id
        return asset_id

    def failed(asset_id, error):
        raise error

    def deliver(asset_id):
        with lock:
            delivered.append(asset_id)
            if len(delivered) == 2:
                completed.set()

    for asset_id in ("crate", "mug"):
        scheduler.submit(asset_id)
    with registry.bind(), context.bind(agent="builder"), observe_processes(observe):
        pool = WorkerPool(2, scheduler, work, failed, deliver, tracker, "modelling")
        try:
            pool.start()
            assert completed.wait(15), pool.fatal
            pool.stop(cancel=False)
            pool.join()
            registry.check_cancelled()
            assert pool.fatal is None
            assert sorted(delivered) == ["crate", "mug"]
            started = [event for event in events if event.kind == "started"]
            assert len(started) == 2 and len({event.job_id for event in started}) == 2
            assert {event.trace.worker for event in started} == {"worker-1", "worker-2"}
            for event in started:
                own = [item for item in events if item.job_id == event.job_id]
                assert own[-1].kind == "finished" and own[-1].returncode == 0 and not own[-1].cancelled
                assert all(item.trace == event.trace for item in own)
                assert event.trace.phase == "modelling" and event.trace.agent == "builder"
                assert "".join(item.text for item in own if item.kind == "output") == event.trace.asset_id + "\n"
        finally:
            release.touch()
            pool.stop()
            pool.join()
            state.close()


@pytest.mark.parametrize("during_timeout", [False, True])
def test_observer_failure_reaps_child_and_finished_error_cannot_mask_original(during_timeout):
    original = RuntimeError("output observer failed")
    events = []

    def observe(event):
        events.append(event)
        if event.kind == "output":
            raise original
        if event.kind == "finished":
            raise ValueError("finished observer also failed")

    code = "import os,signal,time\n"
    if during_timeout:
        code += (
            "def stopped(signum, frame):\n"
            "    os.write(2, str(os.getpid()).encode())\n"
            "    raise SystemExit(0)\n"
            "signal.signal(signal.SIGTERM, stopped)\n"
        )
    else:
        code += "os.write(1, str(os.getpid()).encode())\n"
    code += "time.sleep(60)\n"

    with observe_processes(observe), pytest.raises(RuntimeError) as caught:
        run_process([sys.executable, "-c", code], timeout_s=1 if during_timeout else 10)
    assert caught.value is original
    pid = int(next(event.text for event in events if event.kind == "output"))
    assert gone(pid)
    with pytest.raises(ChildProcessError):
        os.waitpid(pid, os.WNOHANG)
    assert events[-1].kind == "finished" and events[-1].returncode is not None
    assert not events[-1].cancelled and events[-1].text == str(original)
    assert events[-1].timed_out is during_timeout
