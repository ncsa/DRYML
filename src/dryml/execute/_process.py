"""Bounded subprocess execution used by Execute inspection and bootstrap paths."""

from __future__ import annotations

import ctypes
import math
import os
import selectors
import signal
import subprocess
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import BinaryIO, Protocol, Sequence


class _Cancellation(Protocol):
    """Describe the cancellation signal accepted by bounded process controls."""

    def is_set(self) -> bool:
        """Return whether the caller requested bounded process termination."""
        ...


@dataclass(slots=True)
class OwnedProcess:
    """Retain the narrow ownership boundary for an unconfirmed subprocess cleanup.

    Args:
        process: The launched root process that identifies the POSIX group or
            supplies the Windows process handle.
        job: The assigned Windows Job Object, if one was established.
        root_termination_sufficient: Permit root-only confirmation solely for a
            pre-GO bootstrap that never acquired a Windows Job.

    ``reconcile`` only acts on the group or Job established by ``run_bounded``.
    Escaped descendants are deliberately outside this owner and remain caller
    managed.
    """

    process: subprocess.Popen[bytes]
    job: "_WindowsJob | None" = None
    root_termination_sufficient: bool = False

    def reconcile(self, *, deadline: float, poll_interval: float = 0.005) -> bool:
        """Attempt bounded termination and report whether the owned boundary stopped.

        Args:
            deadline: Absolute monotonic deadline for this reconciliation pass.
            poll_interval: Maximum interval between native status checks.

        Returns:
            ``True`` only when the owned group or Job has been confirmed empty.

        Failure behavior:
            Native permission, status, or termination failures return ``False`` so
            callers retain this owner for a later reconciliation pass.
        """
        return _terminate_owned(self.process, deadline=deadline, poll_interval=poll_interval, job=self.job, root_termination_sufficient=self.root_termination_sufficient)


@dataclass(frozen=True, slots=True)
class BoundedProcessResult:
    """Describe bounded process completion and retained diagnostic prefixes.

    ``stdout`` and ``stderr`` contain at most the supplied byte limit. Output is
    drained after that limit so a verbose child cannot block on a full pipe.
    ``cleanup_owner`` is present only when the owned process boundary could not be
    confirmed stopped; its escaped descendants are never represented by it.
    """

    returncode: int | None
    stdout: bytes
    stderr: bytes
    stdout_truncated: bool
    stderr_truncated: bool
    timed_out: bool
    cancelled: bool
    stdout_complete: bool
    stderr_complete: bool
    cleanup_complete: bool
    cleanup_owner: OwnedProcess | None = None


def minimal_environment(overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Build a minimal child environment without inheriting coordinator secrets.

    Args:
        overrides: Explicit launch values supplied by the owning selector.

    Returns:
        A small executable environment with the supplied overrides.

    Raises:
        TypeError: If an override key or value is not text.
    """
    environment = {"PATH": os.defpath}
    if os.name == "nt":
        for key in ("SYSTEMROOT", "WINDIR", "COMSPEC"):
            if key in os.environ:
                environment[key] = os.environ[key]
    for key, value in (overrides or {}).items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError("child environment keys and values must be strings")
        environment[key] = value
    return environment


def run_bounded(
    command: Sequence[str],
    *,
    timeout: float | None = None,
    deadline: float | None = None,
    output_limit: int,
    cancelled: _Cancellation | None = None,
    cwd: str | os.PathLike[str] | None = None,
    env: dict[str, str] | None = None,
    termination_timeout: float = 5.0,
    read_chunk_bytes: int = 8192,
    poll_interval: float = 0.005,
) -> BoundedProcessResult:
    """Run an argv-only Execute-owned child with bounded retention and cleanup.

    Args:
        command: Non-empty executable argv. Shell parsing is never used.
        timeout: Optional finite positive runtime bound. At least one of this or
            ``deadline`` is required.
        deadline: Optional absolute monotonic execution deadline. Termination and
            final pipe drainage receive their separately configured bounded phase.
        output_limit: Maximum retained bytes for each output stream.
        cancelled: Optional object with ``is_set()`` requesting termination.
        cwd: Optional child working directory.
        env: Explicit child-environment overrides. Ambient coordinator values are
            never inherited.
        termination_timeout: Finite grace period before owned-process escalation.
        read_chunk_bytes: Bounded pipe read size supplied by backend policy.
        poll_interval: Bounded polling interval supplied by backend policy.

    Returns:
        The exit status, bounded retained output, and completion state. Incomplete
        cleanup provides ``cleanup_owner`` for later reconciliation rather than
        silently forgetting an unconfirmed owned boundary.

    Raises:
        TypeError: If command or limits have invalid types.
        ValueError: If command is empty or an operational bound is invalid.
        OSError: If the child cannot be launched or pipe controls fail. When cleanup
            is unconfirmed, the raised exception has a ``cleanup_owner`` attribute.

    Side Effects:
        Starts and terminates only this owned child group or Windows Job. Pipe reads
        are nonblocking in the coordinator thread; no daemon reader is relied on
        for cleanup. Windows establishes its Job after launch, so pre-assignment
        launcher activity is not contained; U4 uses this helper only for probes and
        U5's GO gate must keep workload loading after the assignment succeeds.
    """
    if isinstance(command, str) or not isinstance(command, Sequence) or not command or not all(isinstance(part, str) and part for part in command):
        raise TypeError("command must be a non-empty sequence of non-empty strings")
    if isinstance(output_limit, bool) or not isinstance(output_limit, int):
        raise TypeError("output_limit must be an integer")
    if output_limit <= 0:
        raise ValueError("output_limit must be positive")
    if timeout is None and deadline is None:
        raise ValueError("a finite timeout or deadline is required")
    if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout must be a finite positive duration")
    if deadline is not None and (isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline)):
        raise TypeError("deadline must be a monotonic timestamp")
    if isinstance(termination_timeout, bool) or not isinstance(termination_timeout, (int, float)) or not math.isfinite(termination_timeout) or termination_timeout <= 0:
        raise ValueError("termination_timeout must be a finite positive duration")
    if isinstance(read_chunk_bytes, bool) or not isinstance(read_chunk_bytes, int) or read_chunk_bytes <= 0:
        raise ValueError("read_chunk_bytes must be a positive integer")
    if isinstance(poll_interval, bool) or not isinstance(poll_interval, (int, float)) or not math.isfinite(poll_interval) or poll_interval <= 0:
        raise ValueError("poll_interval must be a finite positive duration")
    now = time.monotonic()
    stop_at = min(value for value in (now + float(timeout) if timeout is not None else None, float(deadline) if deadline is not None else None) if value is not None)
    # ``stop_at`` controls launch/execution; cleanup gets its own bounded phase so
    # a child observed at an execution deadline can still be confirmed stopped.
    finish_at = stop_at + float(termination_timeout)
    if cancelled is not None and cancelled.is_set():
        return BoundedProcessResult(None, b"", b"", False, False, False, True, False, False, True)
    if time.monotonic() >= stop_at:
        return BoundedProcessResult(None, b"", b"", False, False, True, False, False, False, True)
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    process = subprocess.Popen(
        tuple(command), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=cwd, env=minimal_environment(env), start_new_session=(os.name == "posix"), creationflags=creationflags,
    )
    owner = OwnedProcess(process)
    raw_streams: dict[str, BinaryIO] = {}
    selector: selectors.BaseSelector | None = None
    cleanup_complete = False
    try:
        if os.name == "nt":
            try:
                owner.job = _WindowsJob.assign(process)
            except BaseException as error:
                owner.job = getattr(error, "windows_job", None)
                raise
        raw_streams = _detach_pipes(process)
        captured = {"stdout": bytearray(), "stderr": bytearray()}
        truncated = {"stdout": False, "stderr": False}
        complete = {"stdout": True, "stderr": True}
        if os.name == "posix":
            selector = selectors.DefaultSelector()
            for name, stream in raw_streams.items():
                selector.register(stream, selectors.EVENT_READ, name)
        timed_out = cancelled_result = terminated = False
        while raw_streams:
            now = time.monotonic()
            if cancelled is not None and cancelled.is_set() and not cancelled_result:
                cancelled_result = True
                cleanup_complete = owner.reconcile(deadline=_cleanup_deadline(finish_at, termination_timeout), poll_interval=float(poll_interval))
                terminated = True
            elif process.poll() is None and now >= stop_at:
                timed_out = True
                cleanup_complete = owner.reconcile(deadline=_cleanup_deadline(finish_at, termination_timeout), poll_interval=float(poll_interval))
                terminated = True
            elif process.poll() is not None and not terminated:
                # The root can exit while an owned descendant still holds a pipe.
                cleanup_complete = owner.reconcile(deadline=_cleanup_deadline(finish_at, termination_timeout), poll_interval=float(poll_interval))
                terminated = True
            if time.monotonic() >= finish_at:
                for name in raw_streams:
                    complete[name] = False
                break
            drain_deadline = finish_at if terminated or process.poll() is not None else min(stop_at, finish_at)
            _drain_ready(raw_streams, selector, captured, truncated, complete, output_limit, read_chunk_bytes, poll_interval, drain_deadline)
        if not terminated:
            # Both pipes closed before the root did; still release the owned tree.
            cleanup_complete = owner.reconcile(deadline=_cleanup_deadline(finish_at, termination_timeout), poll_interval=float(poll_interval))
        _close_pipes(process, raw_streams.values())
        if selector is not None:
            selector.close()
            selector = None
        if cleanup_complete and owner.job is not None:
            owner.job.close()
        retained_owner = None if cleanup_complete else owner
        return BoundedProcessResult(
            process.poll(), bytes(captured["stdout"]), bytes(captured["stderr"]),
            truncated["stdout"], truncated["stderr"], timed_out, cancelled_result,
            complete["stdout"], complete["stderr"], cleanup_complete, retained_owner,
        )
    except BaseException as error:
        cleanup_complete = owner.reconcile(deadline=_cleanup_deadline(finish_at, termination_timeout), poll_interval=float(poll_interval))
        _close_pipes(process, raw_streams.values())
        if selector is not None:
            selector.close()
        if cleanup_complete and owner.job is not None:
            try:
                owner.job.close()
            except OSError:
                cleanup_complete = False
        if not cleanup_complete:
            _retain_cleanup_owner(error, owner)
        raise


def _cleanup_deadline(finish_at: float, termination_timeout: float) -> float:
    """Return the current cleanup phase bound without extending that phase."""
    return min(finish_at, time.monotonic() + float(termination_timeout))


def _detach_pipes(process: subprocess.Popen[bytes]) -> dict[str, BinaryIO]:
    """Detach unbuffered pipes and make their reads safe for one coordinator loop."""
    streams: dict[str, BinaryIO] = {}
    try:
        for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
            assert stream is not None
            raw = stream.detach()
            streams[name] = raw
            if os.name == "posix":
                os.set_blocking(raw.fileno(), False)
    except BaseException:
        _close_pipes(process, streams.values())
        raise
    return streams


def _drain_ready(
    streams: dict[str, BinaryIO], selector: selectors.BaseSelector | None,
    captured: dict[str, bytearray], truncated: dict[str, bool], complete: dict[str, bool],
    output_limit: int, read_chunk_bytes: int, poll_interval: float, finish_at: float,
) -> None:
    """Drain currently readable streams without sleeping beyond the phase deadline."""
    remaining = max(0.0, finish_at - time.monotonic())
    if selector is None:
        ready = [(name, stream) for name, stream in streams.items() if _windows_pipe_available(stream)]
        if not ready:
            time.sleep(min(float(poll_interval), remaining))
            return
    else:
        ready = [(key.data, key.fileobj) for key, _ in selector.select(min(float(poll_interval), remaining))]
    for name, stream in ready:
        try:
            chunk = os.read(stream.fileno(), read_chunk_bytes)
        except BlockingIOError:
            continue
        except OSError:
            complete[name] = False
            _unregister_stream(streams, selector, name)
            continue
        if not chunk:
            _unregister_stream(streams, selector, name)
            continue
        room = output_limit - len(captured[name])
        if room > 0:
            captured[name].extend(chunk[:room])
        if len(chunk) > room:
            truncated[name] = True


def _unregister_stream(streams: dict[str, BinaryIO], selector: selectors.BaseSelector | None, name: str) -> None:
    """Remove one completed stream from the bounded polling set."""
    stream = streams.pop(name)
    if selector is not None:
        selector.unregister(stream)
    stream.close()


def _terminate_owned(process: subprocess.Popen[bytes], *, deadline: float, poll_interval: float, job: "_WindowsJob | None", root_termination_sufficient: bool = False) -> bool:
    """Release an owned group/Job by a finite deadline and confirm its absence."""
    if os.name == "posix":
        try:
            _signal_group(process.pid, signal.SIGTERM)
            kill_at = time.monotonic() + max(0.0, deadline - time.monotonic()) / 2
            while _group_exists(process.pid) and time.monotonic() < kill_at:
                process.poll()
                time.sleep(min(poll_interval, max(0.0, kill_at - time.monotonic())))
            if _group_exists(process.pid):
                _signal_group(process.pid, signal.SIGKILL)
            while _group_exists(process.pid) and time.monotonic() < deadline:
                process.poll()
                time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))
            process.poll()
            return not _group_exists(process.pid)
        except (OSError, PermissionError):
            return False
    if job is None and root_termination_sufficient:
        # Before GO, Execute's bootstrap performs no workload or descendant work;
        # a confirmed launcher exit therefore proves this narrow unassigned launch.
        try:
            if process.poll() is None:
                process.kill()
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))
            return process.poll() is not None
        except OSError:
            return False
    if job is None or not job.assigned:
        # A failed ownership handshake is never reported as root-only success.
        try:
            if process.poll() is None:
                process.kill()
        except OSError:
            pass
        return False
    try:
        job.terminate()
        while time.monotonic() < deadline:
            if job.active_process_count() == 0:
                process.poll()
                return True
            time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))
        return job.active_process_count() == 0
    except OSError:
        return False


def _group_exists(pid: int) -> bool:
    """Return whether the owned POSIX process group still exists."""
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_group(pid: int, sig: signal.Signals) -> None:
    """Signal only the group created for this owned child."""
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def _close_pipes(process: subprocess.Popen[bytes], streams: object = ()) -> None:
    """Close only coordinator-owned pipe objects after nonblocking reads stop."""
    for stream in (*tuple(streams), process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except (OSError, ValueError):
                pass


def _retain_cleanup_owner(error: BaseException, owner: OwnedProcess) -> None:
    """Attach recovery ownership to an interruption without replacing its cause."""
    setattr(error, "cleanup_owner", owner)
    _add_exception_note(error, "owned process cleanup was not confirmed; reconcile with exception.cleanup_owner")


def _add_exception_note(error: BaseException, message: str) -> None:
    """Attach a supplementary diagnostic on every supported Python version."""
    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        add_note(message)
    else:
        setattr(error, "__notes__", [*getattr(error, "__notes__", ()), message])


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    """Native JOB_OBJECT_BASIC_LIMIT_INFORMATION layout used for kill-on-close."""

    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _IO_COUNTERS(ctypes.Structure):
    """Native IO_COUNTERS layout required by extended Job limit information."""

    _fields_ = [(name, ctypes.c_uint64) for name in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    """Native extended Job limit information with the kill-on-close flag."""

    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
    """Native Job accounting information used to confirm active process count."""

    _fields_ = [
        ("TotalUserTime", ctypes.c_int64),
        ("TotalKernelTime", ctypes.c_int64),
        ("ThisPeriodTotalUserTime", ctypes.c_int64),
        ("ThisPeriodTotalKernelTime", ctypes.c_int64),
        ("TotalPageFaultCount", ctypes.c_uint32),
        ("TotalProcesses", ctypes.c_uint32),
        ("ActiveProcesses", ctypes.c_uint32),
        ("TotalTerminatedProcesses", ctypes.c_uint32),
    ]


@lru_cache(maxsize=1)
def _windows_kernel32() -> object:
    """Return kernel32 with typed Job and pipe-control entry points."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = ctypes.c_void_p
    dword = ctypes.c_uint32
    kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p)
    kernel32.CreateJobObjectW.restype = handle
    kernel32.SetInformationJobObject.argtypes = (handle, ctypes.c_int, ctypes.c_void_p, dword)
    kernel32.SetInformationJobObject.restype = ctypes.c_int
    kernel32.AssignProcessToJobObject.argtypes = (handle, handle)
    kernel32.AssignProcessToJobObject.restype = ctypes.c_int
    kernel32.TerminateJobObject.argtypes = (handle, ctypes.c_uint)
    kernel32.TerminateJobObject.restype = ctypes.c_int
    kernel32.QueryInformationJobObject.argtypes = (handle, ctypes.c_int, ctypes.c_void_p, dword, ctypes.POINTER(dword))
    kernel32.QueryInformationJobObject.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = (handle,)
    kernel32.CloseHandle.restype = ctypes.c_int
    kernel32.OpenProcess.argtypes = (dword, ctypes.c_int, dword)
    kernel32.OpenProcess.restype = handle
    kernel32.IsProcessInJob.argtypes = (handle, handle, ctypes.POINTER(ctypes.c_int))
    kernel32.IsProcessInJob.restype = ctypes.c_int
    kernel32.PeekNamedPipe.argtypes = (handle, ctypes.c_void_p, dword, ctypes.c_void_p, ctypes.POINTER(dword), ctypes.c_void_p)
    kernel32.PeekNamedPipe.restype = ctypes.c_int
    return kernel32


def _windows_error(message: str) -> OSError:
    """Build an actionable native Windows error from the current thread error code."""
    return OSError(getattr(ctypes, "get_last_error", lambda: 0)(), message)


def _windows_pipe_available(stream: BinaryIO) -> bool:
    """Return whether a Windows anonymous pipe has bytes ready without blocking."""
    import msvcrt

    available = ctypes.c_uint32()
    handle = ctypes.c_void_p(msvcrt.get_osfhandle(stream.fileno()))
    if _windows_kernel32().PeekNamedPipe(handle, None, 0, None, ctypes.byref(available), None):
        return available.value > 0
    error = ctypes.get_last_error()
    if error in {109, 232}:  # ERROR_BROKEN_PIPE, ERROR_NO_DATA
        return True
    raise OSError(error, "PeekNamedPipe failed for owned process output")


class _WindowsJob:
    """Own a Windows child tree through a checked kill-on-close Job Object."""

    def __init__(self, handle: object, kernel32: object | None = None, *, assigned: bool = True) -> None:
        """Store an already configured native Job handle and its typed API."""
        self._handle = handle
        self._kernel32 = _windows_kernel32() if kernel32 is None else kernel32
        self.assigned = assigned

    @classmethod
    def assign(cls, process: subprocess.Popen[bytes]) -> "_WindowsJob":
        """Create and assign a kill-on-close Job before Execute exchanges work."""
        kernel32 = _windows_kernel32()
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise _windows_error("CreateJobObjectW failed")
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        try:
            if not kernel32.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
                raise _windows_error("SetInformationJobObject failed for owned Job")
            if not kernel32.AssignProcessToJobObject(handle, process._handle):
                raise _windows_error("AssignProcessToJobObject failed for owned Job")
        except BaseException as error:
            if not kernel32.CloseHandle(handle):
                _add_exception_note(error, "CloseHandle failed while releasing an unassigned owned Job")
                setattr(error, "windows_job", cls(handle, kernel32, assigned=False))
            raise
        return cls(handle, kernel32)

    def terminate(self) -> None:
        """Terminate every process associated with this Job Object or raise OSError."""
        if not self._kernel32.TerminateJobObject(self._handle, 1):
            raise _windows_error("TerminateJobObject failed for owned Job")

    def active_process_count(self) -> int:
        """Return the native Job active-process count or raise OSError."""
        info = _JOBOBJECT_BASIC_ACCOUNTING_INFORMATION()
        returned = ctypes.c_uint32()
        if not self._kernel32.QueryInformationJobObject(self._handle, 1, ctypes.byref(info), ctypes.sizeof(info), ctypes.byref(returned)):
            raise _windows_error("QueryInformationJobObject failed for owned Job")
        if returned.value and returned.value < ctypes.sizeof(info):
            raise OSError("QueryInformationJobObject returned incomplete accounting information")
        return int(info.ActiveProcesses)

    def contains_pid(self, pid: int) -> bool:
        """Return whether an extant process belongs to this exact owned Job.

        Native query/open failures are deliberately reported as ``False`` so a
        wrapped worker cannot pass admission without an exact Job membership proof.
        """
        if not self.assigned or isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            return False
        process = None
        try:
            process = self._kernel32.OpenProcess(0x1000, 0, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not process:
                return False
            member = ctypes.c_int()
            contained = bool(self._kernel32.IsProcessInJob(process, self._handle, ctypes.byref(member)) and member.value)
            if not self._kernel32.CloseHandle(process):
                return False
            process = None
            return contained
        except (AttributeError, OSError, TypeError):
            return False
        finally:
            if process:
                self._kernel32.CloseHandle(process)

    def close(self) -> None:
        """Close a confirmed-empty Job Object or raise OSError on handle failure."""
        if not self._kernel32.CloseHandle(self._handle):
            raise _windows_error("CloseHandle failed for owned Job")


__all__ = ["BoundedProcessResult", "OwnedProcess", "minimal_environment", "run_bounded"]
