import json
import os
import signal
import sys
import threading
from types import SimpleNamespace

import pytest
from rich.console import Console

from kitbash.app import Application, create_workspace
from kitbash.domain.assets import AssetRecord, AssetState
from kitbash.domain.run_input import RunInput
from kitbash.errors import LLMAccessError
from kitbash.infra.blender import BlenderRunner
from kitbash.infra.process import run_process
from kitbash.infra.trellis import TrellisRunner
from kitbash.phases.modelling import ModellingPhase
from kitbash.pipeline.board import AssetBoard
from kitbash.store.state import StateDB
from tests.helpers import process_gone as gone
from tests.helpers import requires_blender, wait_for, write_test_config

pytestmark = [pytest.mark.integration, pytest.mark.timeout(60)]


@pytest.mark.parametrize("trigger", ["fatal", "interrupt"])
@pytest.mark.parametrize("tool", ["trellis", pytest.param("blender", marks=[pytest.mark.blender, requires_blender])])
def test_modelling_joins_children_before_application_teardown(tmp_path, monkeypatch, trigger, tool):
    config_path = write_test_config(tmp_path, pipeline={"threads": 3, "review_buffer": 3})
    config, layout, run_input = create_workspace(tmp_path / "out", RunInput.create(None, "shutdown probe"), config_path, None, {})
    application = Application(config, layout, run_input, interactive=False, console=Console(quiet=True))
    board = AssetBoard(application.state.assets)
    assets = ["active", "waiting", "fatal"] if tool == "trellis" else ["active", "fatal"]
    board.ensure(AssetRecord(id=name, name=name, state=AssetState.GENERATING) for name in assets)
    ready = tmp_path / "ready"
    descendant_ready = tmp_path / "descendant-ready"
    pids_path = tmp_path / "pids.json"
    invocations = tmp_path / "invocations"
    child_code = (
        "import signal,time; from pathlib import Path; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(descendant_ready)!r}).touch(); time.sleep(60)"
    )
    tool_script = tmp_path / ("generate.py" if tool == "trellis" else "long_blender.py")
    tool_script.write_text(
        "import json,os,subprocess,time\nfrom pathlib import Path\n"
        f"with Path({str(invocations)!r}).open('a') as handle: handle.write('started\\n')\n"
        f"child = subprocess.Popen([{sys.executable!r}, '-c', {child_code!r}])\n"
        f"Path({str(pids_path)!r}).write_text(json.dumps([os.getpid(), child.pid]))\n"
        f"while not Path({str(descendant_ready)!r}).exists(): time.sleep(.01)\n"
        f"Path({str(ready)!r}).touch()\n"
        "time.sleep(60)\n"
    )
    trellis = TrellisRunner(
        tmp_path, python=sys.executable, steps=1, pipeline_type="test", no_texture=True,
        timeout_s=90, retries=3, max_concurrent=1, recorder=application.tracker,
    )
    blender = BlenderRunner("blender", timeout_s=90, recorder=application.tracker)
    waiting = threading.Event()
    active_finished = threading.Event()
    unwound = []
    failed = []
    delivered = []
    close_observations = []
    original = LLMAccessError("provider rejected model credentials")

    def advance(asset_id):
        if asset_id == "fatal":
            try:
                wait_for(ready.exists, timeout=20)
                if tool == "trellis":
                    assert waiting.wait(10)
            except AssertionError as exc:
                raise LLMAccessError(f"child startup failed: {exc}") from exc
            if trigger == "fatal":
                raise original
            os.kill(os.getpid(), signal.SIGINT)
            assert active_finished.wait(15)
            return board.get(asset_id)
        if asset_id == "waiting":
            # Ensure the active asset owns the single Trellis slot first.
            wait_for(ready.exists, timeout=20)
            waiting.set()
        try:
            if tool == "trellis":
                trellis.generate(tmp_path / "image.png", tmp_path / asset_id, asset_id, seed=0)
            else:
                blender.run(tool_script, args={}, log_path=tmp_path / "blender.log")
            return board.get(asset_id)
        finally:
            unwound.append((asset_id, application._http.is_closed))
            application.state.meta.set(f"unwound_{asset_id}", True)
            if asset_id == "active":
                active_finished.set()

    def fail(asset_id, error):
        failed.append((asset_id, error))
        return board.transition(asset_id, AssetState.INPUT_NEEDED, error=str(error))

    phase = ModellingPhase(
        None, None, None, application.state, application.user, application.dashboard, application.tracker, config,
    )
    route = phase._route

    def record_route(record, scheduler, queue, *, initial=False):
        if not initial:
            delivered.append(record.id)
        route(record, scheduler, queue, initial=initial)

    monkeypatch.setattr(phase, "_route", record_route)
    for resource in (application._http, application.backlot, application.state):
        close = resource.close

        def observed_close(close=close):
            close_observations.append({
                "workers": [thread.name for thread in threading.enumerate() if thread.name.startswith("worker-")],
                "pids_gone": pids_path.exists() and all(gone(pid) for pid in json.loads(pids_path.read_text())),
                "unwound": sorted(asset for asset, _ in unwound),
            })
            close()

        monkeypatch.setattr(resource, "close", observed_close)
    try:
        with pytest.raises(LLMAccessError if trigger == "fatal" else KeyboardInterrupt) as caught:
            try:
                phase._execute(board, SimpleNamespace(advance=advance, fail=fail))
            finally:
                application.close()
        if trigger == "fatal":
            assert caught.value is original
        assert failed == [] and delivered == []
        expected = sorted(set(assets) - {"fatal"})
        assert sorted(asset for asset, _ in unwound) == expected
        assert all(not was_closed for _, was_closed in unwound)
        assert all(observation == {"workers": [], "pids_gone": True, "unwound": expected} for observation in close_observations)
        assert invocations.read_text().splitlines() == ["started"]  # no Trellis retry or launch from its waiting slot
        resumed = StateDB(layout.state_db)
        try:
            assert all(record.state is AssetState.GENERATING and record.error is None for record in resumed.assets.all())
            assert all(resumed.meta.get(f"unwound_{asset}") is True for asset in expected)
        finally:
            resumed.close()
        assert run_process([sys.executable, "-c", "print('next run')"], timeout_s=5).stdout.strip() == "next run"
    finally:
        if pids_path.exists():
            for pid in json.loads(pids_path.read_text()):
                if not gone(pid):
                    os.kill(pid, signal.SIGKILL)
        for thread in threading.enumerate():
            if thread.name.startswith("worker-"):
                thread.join(10)
