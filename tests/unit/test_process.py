import os
import signal
import subprocess
import sys
import threading
import time

import pytest

from kitbash.infra import process as processes
from kitbash.infra.process import ProcessCancelled, ProcessRegistry, current_registry, run_process
from tests.helpers import process_gone as gone
from tests.helpers import wait_for


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
    popen = subprocess.Popen

    def paused_spawn(*args, **kwargs):
        process = popen(*args, **kwargs)
        child.append(process)
        spawned.set()
        assert release_spawn.wait(10)
        return process

    def run():
        try:
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
    open_log = processes._open_log

    def remember_log(*args):
        handle = open_log(*args)
        handles.append(handle)
        return handle

    monkeypatch.setattr(processes, "_open_log", remember_log)
    with pytest.raises(FileNotFoundError):
        run_process([str(tmp_path / "missing-executable")], timeout_s=5, log_path=tmp_path / "spawn.log")
    assert handles[0].closed


def test_unexpected_communication_error_reaps_and_closes_pipes(monkeypatch):
    children = []
    original = RuntimeError("communication interrupted")
    popen = subprocess.Popen

    def broken_communication(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)

        def fail(*args, **kwargs):
            raise original

        child.communicate = fail
        return child

    monkeypatch.setattr(processes.subprocess, "Popen", broken_communication)
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
