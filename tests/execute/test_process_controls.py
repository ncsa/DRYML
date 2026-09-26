"""Regression coverage for Execute-owned bounded process controls."""

from __future__ import annotations

import ctypes
import os
import signal
import sys
from types import SimpleNamespace
from unittest.mock import Mock
from pathlib import Path
from threading import Event, Thread
from time import monotonic, sleep

import pytest

from dryml.execute import _process as process_module
from dryml.execute import subprocess as subprocess_module
from dryml.execute._process import run_bounded


def _os_with_name(name: str) -> SimpleNamespace:
    """Return an isolated OS module copy with one platform name for a module seam."""
    return SimpleNamespace(**{**vars(os), "name": name})


def _running(pid: int) -> bool:
    """Return whether ``pid`` still represents a non-zombie process on Linux."""
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        return stat.read_text(encoding="utf-8").split()[2] != "Z"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _escaped_pipe_holder(pid_file: Path) -> str:
    """Build a root program that leaves a detached child holding its stdout pipe."""
    return (
        "import subprocess, sys; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True); "
        f"open({str(pid_file)!r}, 'w', encoding='utf-8').write(str(child.pid))"
    )


def _stop(pid: int | None) -> None:
    """Terminate a caller-managed fixture process if it is still running."""
    if pid is not None and _running(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.mark.skipif(os.name != "posix", reason="process-group escape fixtures require POSIX")
def test_escaped_pipe_holder_returns_incomplete_output_without_killing_escape(tmp_path: Path):
    """A detached descendant cannot hold bounded discovery beyond its drainage phase."""
    pid_file = tmp_path / "escaped.pid"
    pid: int | None = None
    try:
        started = monotonic()
        result = run_bounded(
            [sys.executable, "-c", _escaped_pipe_holder(pid_file)],
            timeout=1.0,
            termination_timeout=0.05,
            output_limit=64,
        )
        elapsed = monotonic() - started
        assert not result.timed_out
        pid = int(pid_file.read_text(encoding="utf-8"))
        assert elapsed < 1.3
        assert result.cleanup_complete
        assert not result.stdout_complete
        assert _running(pid)
    finally:
        _stop(pid)


@pytest.mark.skipif(os.name != "posix", reason="process-group fixture requires POSIX")
def test_polling_never_sleeps_past_the_remaining_execution_deadline():
    """A large policy polling interval cannot add arbitrary timeout latency."""
    started = monotonic()
    result = run_bounded(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        timeout=0.02,
        termination_timeout=0.02,
        poll_interval=0.5,
        output_limit=64,
    )
    assert result.timed_out
    assert monotonic() - started < 0.2


def _closed_pipes_wait_for_release(marker: Path, release: Path) -> str:
    """Build a child that closes output before a caller-controlled normal exit."""
    return (
        "import os\n"
        "import time\n"
        f"open({str(marker)!r}, 'w', encoding='ascii').write('pipes-closed')\n"
        "os.close(1)\n"
        "os.close(2)\n"
        f"while not os.path.exists({str(release)!r}):\n"
        "    time.sleep(0.001)\n"
    )


def test_closed_pipes_do_not_terminate_a_normally_finishing_root(tmp_path: Path):
    """EOF drains output but waits within the execution bound for the root exit."""
    marker = tmp_path / "pipes-closed"
    release = tmp_path / "release"

    def release_child() -> None:
        while not marker.exists():
            sleep(0.001)
        sleep(0.02)
        release.write_text("release", encoding="ascii")

    thread = Thread(target=release_child, daemon=True)
    thread.start()
    try:
        result = run_bounded(
            [sys.executable, "-c", _closed_pipes_wait_for_release(marker, release)],
            timeout=1,
            termination_timeout=0.05,
            output_limit=64,
        )
    finally:
        thread.join(1)
    assert result.returncode == 0
    assert not result.timed_out
    assert not result.cancelled
    assert result.cleanup_complete


def test_closed_pipes_still_honor_the_execution_deadline(tmp_path: Path):
    """A root that closes output but never exits is terminated at its deadline."""
    marker = tmp_path / "pipes-closed"
    release = tmp_path / "never-release"
    result = run_bounded(
        [sys.executable, "-c", _closed_pipes_wait_for_release(marker, release)],
        timeout=0.05,
        termination_timeout=0.05,
        output_limit=64,
    )
    assert marker.exists()
    assert result.timed_out
    assert result.cleanup_complete


def test_closed_pipes_still_honor_cancellation(tmp_path: Path):
    """Caller cancellation terminates a root after both output pipes reach EOF."""
    marker = tmp_path / "pipes-closed"
    release = tmp_path / "never-release"
    cancelled = Event()

    def cancel_child() -> None:
        while not marker.exists():
            sleep(0.001)
        cancelled.set()

    thread = Thread(target=cancel_child, daemon=True)
    thread.start()
    try:
        result = run_bounded(
            [sys.executable, "-c", _closed_pipes_wait_for_release(marker, release)],
            timeout=1,
            termination_timeout=0.05,
            output_limit=64,
            cancelled=cancelled,
        )
    finally:
        cancelled.set()
        thread.join(1)
    assert result.cancelled
    assert result.cleanup_complete


@pytest.mark.skipif(os.name != "posix", reason="process-group escape fixtures require POSIX")
def test_cancellation_during_post_root_drain_is_bounded(tmp_path: Path):
    """Cancellation remains effective after the root exits but an escape holds output."""
    pid_file = tmp_path / "escaped.pid"
    cancelled = Event()
    pid: int | None = None

    def cancel_after_root_exit() -> None:
        while not pid_file.exists():
            sleep(0.001)
        sleep(0.02)
        cancelled.set()

    thread = Thread(target=cancel_after_root_exit)
    thread.start()
    try:
        result = run_bounded(
            [sys.executable, "-c", _escaped_pipe_holder(pid_file)],
            timeout=1,
            termination_timeout=0.05,
            output_limit=64,
            cancelled=cancelled,
        )
        pid = int(pid_file.read_text(encoding="utf-8"))
        assert result.cancelled
        assert result.cleanup_complete
        assert not result.stdout_complete
    finally:
        cancelled.set()
        thread.join(0.2)
        _stop(pid)


@pytest.mark.skipif(os.name != "posix", reason="process-group fixture requires POSIX")
@pytest.mark.parametrize(
    ("exit_during_signal", "descendant_remains"),
    [(False, False), (True, False), (True, True)],
)
def test_cleanup_reaps_an_exited_root_without_ignoring_descendants(
        monkeypatch, exit_during_signal, descendant_remains):
    """An exited root is reaped, but only absent groups qualify as cleaned up."""
    state = {"exited": not exit_during_signal, "reaped": False}

    def poll():
        if state["exited"]:
            state["reaped"] = True
            return 0
        return None

    def signal_group(pid, sig):
        if exit_during_signal:
            state["exited"] = True
            raise PermissionError("root exited during signalling")
        assert state["reaped"], "reap an exited root before signalling its group"

    monkeypatch.setattr(process_module, "_signal_group", signal_group)
    monkeypatch.setattr(
        process_module, "_group_exists",
        lambda pid: descendant_remains or not state["reaped"],
    )
    owner = process_module.OwnedProcess(SimpleNamespace(pid=12345, poll=poll))
    assert owner.reconcile(deadline=monotonic() + 1) is (not descendant_remains)
    assert state["reaped"]


@pytest.mark.skipif(os.name != "posix", reason="process-group fixture requires POSIX")
def test_cleanup_permission_error_retains_reconciliation_owner(monkeypatch, tmp_path: Path):
    """An unconfirmed group is reported with a narrow owner rather than forgotten."""
    pid_file = tmp_path / "root.pid"
    program = f"import time; open({str(pid_file)!r}, 'w').write('started'); time.sleep(30)"
    monkeypatch.setattr(process_module, "_signal_group", lambda *_args: (_ for _ in ()).throw(PermissionError("denied")))
    result = run_bounded([sys.executable, "-c", program], timeout=0.02, termination_timeout=0.02, output_limit=64)
    owner = result.cleanup_owner
    try:
        assert not result.cleanup_complete
        assert owner is not None
        assert owner.process.poll() is None
    finally:
        if owner is not None:
            os.killpg(owner.process.pid, signal.SIGKILL)
            owner.process.wait(timeout=1)


@pytest.mark.skipif(os.name != "posix", reason="process-group fixture requires POSIX")
@pytest.mark.parametrize("error", [OSError("reader failed"), KeyboardInterrupt()])
def test_post_launch_errors_release_the_owned_process_group(monkeypatch, error: BaseException):
    """Reader failures and interrupts cannot bypass owned-process cleanup."""
    observed = {}

    def fail_after_launch(process):
        observed["process"] = process
        raise error

    monkeypatch.setattr(process_module, "_detach_pipes", fail_after_launch)
    try:
        with pytest.raises(type(error)):
            run_bounded([sys.executable, "-c", "import time; time.sleep(30)"], timeout=1, output_limit=64)
        assert observed["process"].poll() is not None
    finally:
        process = observed.get("process")
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=1)


class _FakeWindowsApi:
    """Minimal checked Windows Job API fixture with pointer-width-safe handles."""

    def __init__(self, *, assign: int = 1, terminate: int = 1, close: int = 1, active: int = 0) -> None:
        self.assign = assign
        self.terminate_result = terminate
        self.close_result = close
        self.active = active
        self.job_handle = 1 << 40
        self.assigned_handles: list[int] = []
        self.closed_handles: list[int] = []
        self.member = True

    def CreateJobObjectW(self, _security, _name):
        return self.job_handle

    def SetInformationJobObject(self, _handle, _kind, _info, _size):
        return 1

    def AssignProcessToJobObject(self, _job, process):
        self.assigned_handles.append(process)
        return self.assign

    def TerminateJobObject(self, _handle, _exit_code):
        return self.terminate_result

    def QueryInformationJobObject(self, _handle, _kind, info, size, returned):
        ctypes.cast(info, ctypes.POINTER(process_module._JOBOBJECT_BASIC_ACCOUNTING_INFORMATION)).contents.ActiveProcesses = self.active
        ctypes.cast(returned, ctypes.POINTER(ctypes.c_uint32)).contents.value = size
        return 1

    def CloseHandle(self, handle):
        self.closed_handles.append(handle)
        return self.close_result

    def OpenProcess(self, _access, _inherit, pid):
        return (1 << 42) + pid

    def IsProcessInJob(self, _process, _job, member):
        ctypes.cast(member, ctypes.POINTER(ctypes.c_int)).contents.value = int(self.member)
        return 1


class _FakeProcess:
    """Process seam carrying a deliberately large native handle."""

    _handle = 1 << 41
    pid = 1

    def poll(self):
        return None

    def kill(self):
        raise AssertionError("a Job-backed process must not use root-only cleanup")


def test_windows_job_preserves_large_handles_and_rejects_assignment_failure(monkeypatch):
    """Typed Job APIs preserve 64-bit handles and fail closed on assignment errors."""
    api = _FakeWindowsApi()
    monkeypatch.setattr(process_module, "_windows_kernel32", lambda: api)
    job = process_module._WindowsJob.assign(_FakeProcess())
    assert job._handle == 1 << 40
    assert api.assigned_handles == [1 << 41]

    failing = _FakeWindowsApi(assign=0)
    monkeypatch.setattr(process_module, "_windows_kernel32", lambda: failing)
    with pytest.raises(OSError, match="AssignProcessToJobObject"):
        process_module._WindowsJob.assign(_FakeProcess())
    assert failing.closed_handles == [1 << 40]

    unclosable = _FakeWindowsApi(assign=0, close=0)
    monkeypatch.setattr(process_module, "_windows_kernel32", lambda: unclosable)
    with pytest.raises(OSError, match="AssignProcessToJobObject") as raised:
        process_module._WindowsJob.assign(_FakeProcess())
    assert not raised.value.windows_job.assigned


def test_windows_job_assignment_failure_preserves_diagnostic_without_add_note(monkeypatch):
    """Python 3.10 keeps supplemental cleanup evidence without ``BaseException.add_note``."""
    class OSErrorWithoutAddNote(OSError):
        add_note = None

    unclosable = _FakeWindowsApi(assign=0, close=0)
    monkeypatch.setattr(process_module, "_windows_kernel32", lambda: unclosable)
    monkeypatch.setattr(process_module, "_windows_error", lambda message: OSErrorWithoutAddNote(message))

    with pytest.raises(OSErrorWithoutAddNote, match="AssignProcessToJobObject") as raised:
        process_module._WindowsJob.assign(_FakeProcess())

    assert raised.value.__notes__ == ["CloseHandle failed while releasing an unassigned owned Job"]
    assert not raised.value.windows_job.assigned


def test_windows_job_checks_termination_and_active_process_evidence(monkeypatch):
    """Termination errors and nonempty Job accounting cannot report cleanup success."""
    failing = _FakeWindowsApi(terminate=0)
    job = process_module._WindowsJob(1 << 40, failing)
    with pytest.raises(OSError, match="TerminateJobObject"):
        job.terminate()

    active = _FakeWindowsApi(active=1)
    monkeypatch.setattr(process_module, "os", _os_with_name("nt"))
    assert not process_module._terminate_owned(_FakeProcess(), deadline=monotonic(), poll_interval=0.001, job=process_module._WindowsJob(1 << 40, active))


def test_windows_job_membership_accepts_wrapped_worker_with_large_handles(monkeypatch):
    """A distinct wrapper child passes only through an exact Job membership proof."""
    api = _FakeWindowsApi()
    job = process_module._WindowsJob(1 << 40, api)
    monkeypatch.setattr(process_module, "os", _os_with_name("nt"))
    monkeypatch.setattr(subprocess_module, "os", _os_with_name("nt"))

    assert job.contains_pid(99)
    owner = process_module.OwnedProcess(_FakeProcess(), job=job)
    assert owner.job is job
    assert subprocess_module.SubProcessBackend._worker_belongs_to_owner(99, owner)
    assert api.closed_handles[-1] == (1 << 42) + 99
    api.member = False
    assert not job.contains_pid(99)


def test_windows_kernel32_binding_is_cached_outside_pipe_polling(monkeypatch):
    """Repeated Windows pipe checks reuse one typed kernel32 binding factory call."""
    kernel32 = Mock()
    kernel32.PeekNamedPipe.return_value = 0
    factory = Mock(return_value=kernel32)
    monkeypatch.setattr(process_module.ctypes, "WinDLL", factory, raising=False)
    monkeypatch.setattr(process_module.ctypes, "get_last_error", lambda: 109, raising=False)
    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(get_osfhandle=lambda _fd: 1 << 40))
    process_module._windows_kernel32.cache_clear()
    try:
        stream = SimpleNamespace(fileno=lambda: 3)
        for _ in range(8):
            assert process_module._windows_pipe_available(stream)
        assert factory.call_count == 1
    finally:
        process_module._windows_kernel32.cache_clear()
