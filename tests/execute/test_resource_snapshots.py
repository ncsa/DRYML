"""Proof for bounded Execute process probing and snapshot arithmetic."""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path
from threading import Event
from time import monotonic

from dryml.execute._process import run_bounded


def test_bounded_probe_drains_flood_without_retaining_unbounded_output():
    """A flooding probe is drained while retaining only the configured diagnostic prefix."""
    result = run_bounded([sys.executable, "-c", "import sys; sys.stdout.write('x' * 1000000)"], timeout=2, output_limit=64)
    assert result.returncode == 0
    assert len(result.stdout) == 64
    assert result.stdout_truncated


def test_bounded_probe_deadline_terminates_child_and_reports_timeout():
    """A deadline terminates the Execute-owned child without an unbounded communicate buffer."""
    result = run_bounded([sys.executable, "-c", "import time; time.sleep(5)"], deadline=monotonic() + 0.05, output_limit=64)
    assert result.timed_out
    assert result.returncode is not None


def test_pre_cancelled_probe_never_launches(monkeypatch):
    """A cancellation already observed at the gate does not start a child."""
    cancelled = Event()
    cancelled.set()
    monkeypatch.setattr("dryml.execute._process.subprocess.Popen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("launched")))
    result = run_bounded([sys.executable, "-c", "raise SystemExit"], timeout=1, output_limit=64, cancelled=cancelled)
    assert result.cancelled
    assert result.returncode is None
    assert not result.stdout_complete
    assert not result.stderr_complete


def test_root_exit_cannot_leave_owned_descendant_holding_a_pipe(tmp_path: Path):
    """Owned descendants are cleaned even when the root exits before them."""
    pid_file = tmp_path / "owned.pid"
    program = (
        "import subprocess, sys; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)']); "
        f"open({str(pid_file)!r}, 'w').write(str(child.pid)); sys.stdout.write('root')"
    )
    result = run_bounded([sys.executable, "-c", program], timeout=0.25, output_limit=64, termination_timeout=0.05)
    pid = int(pid_file.read_text())
    assert result.cleanup_complete
    assert result.stdout_complete
    if os.name != "posix":
        return
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pass
    else:
        stat = Path(f"/proc/{pid}/stat")
        assert stat.exists() and stat.read_text().split()[2] == "Z"


def test_termination_escalates_when_an_owned_child_ignores_sigterm(tmp_path: Path):
    """Cancellation escalates after the child confirms its SIGTERM handler."""
    ready = tmp_path / "ignoring-sigterm"

    class CancelWhenReady:
        def is_set(self):
            return ready.exists()

    program = (
        "import signal, time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(ready)!r}).write_text('ready'); time.sleep(30)"
    )
    result = run_bounded(
        [sys.executable, "-c", program], timeout=5, output_limit=64,
        termination_timeout=2, cancelled=CancelWhenReady(),
    )
    try:
        assert ready.exists()
        assert result.cancelled
        assert not result.timed_out
        assert result.cleanup_complete
        if os.name == "posix":
            assert result.returncode == -signal.SIGKILL
    finally:
        if result.cleanup_owner is not None:
            assert result.cleanup_owner.reconcile(deadline=monotonic() + 5)
