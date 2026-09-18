"""Focused proof for the additive real subprocess Execute backend."""

from __future__ import annotations

import os
import subprocess
import sys
from contextlib import contextmanager
from concurrent.futures import CancelledError
from pathlib import Path
from threading import Event
from time import monotonic, sleep
from types import SimpleNamespace

import pytest

from dryml.execute import executor as executor_module
from dryml.execute import subprocess as subprocess_module
from dryml.execute.accounting import ResourceAuthority
from dryml.execute.errors import AdmissionError, CleanupError, ExecutionUncertainError
from dryml.execute.executor import Executor
from dryml.execute.models import WorkerSetup
from dryml.execute.output import ExecutionOutput
from dryml.execute.subprocess import SubProcessConfig, SubProcessFuture
from dryml.environments import CurrentEnvironmentSpec
from dryml.environments import CompatibilityIssue
from dryml.environments.compatibility import report_from_issues
from dryml.worlds import CountConstraint, ResourceRequirement, RoleRequirement, WorldRequirement


def _os_with_name(name: str) -> SimpleNamespace:
    """Return an isolated OS module copy with one platform name for a module seam."""
    return SimpleNamespace(**{**vars(os), "name": name})


def _require_cpu_affinity() -> None:
    """Skip only the real worker proof when native CPU affinity is unavailable."""
    if not all(callable(getattr(os, name, None)) for name in ("sched_getaffinity", "sched_setaffinity")):
        pytest.skip("native CPU affinity is unavailable on this platform")


def _add(left: int, right: int = 0) -> int:
    """Return a simple importable workload value."""
    return left + right


def _emit_and_return(value: str) -> str:
    """Write to both worker descriptors before returning a value."""
    import os

    os.write(1, b"worker stdout\n")
    os.write(2, b"worker stderr\n")
    return value


def _hold_affinity(marker_directory: str, name: str) -> list[int]:
    """Atomically publish affinity before exposing the worker readiness signal."""
    directory = Path(marker_directory)
    affinity = sorted(os.sched_getaffinity(0))
    pending = directory / f"{name}.pending"
    pending.write_text(",".join(map(str, affinity)), encoding="ascii")
    pending.replace(directory / f"{name}.running")
    release = directory / f"{name}.release"
    while not release.exists():
        sleep(0.01)
    return affinity


def test_exact_current_pin_requires_fresh_identity_without_candidate_search(
        tmp_path, monkeypatch):
    """
    A pin overrides config Python and reaches the worker without a requirement.
    """
    monkeypatch.setattr(
        subprocess_module,
        "discover_candidates",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("pinned call must not enumerate candidates")),
    )
    executor = Executor(SubProcessConfig(
        spool_directory=tmp_path,
        python_executable=Path("/definitely/not/the-selected-python"),
        automatic_environment_discovery=False,
    ))
    try:
        assert executor.run(_add,
                            2,
                            3,
                            environment_spec=CurrentEnvironmentSpec()) == 5
    finally:
        executor.close(cancel=True, timeout=5)


def test_exact_pin_identity_mismatch_without_requirement_withholds_payload(
        tmp_path, monkeypatch):
    """
    A fresh pre-GO identity mismatch cannot deserialize or invoke the workload.
    """
    marker = tmp_path / "invoked"
    monkeypatch.setattr(
        subprocess_module,
        "compare_selection",
        lambda *_args: report_from_issues((CompatibilityIssue(
            "selection_identity_mismatch", "error", "mismatch"), ),
                                          policy="strict"),
    )
    executor = Executor(SubProcessConfig(spool_directory=tmp_path))
    try:
        future = executor.submit(_write_sentinel,
                                 str(marker),
                                 environment_spec=CurrentEnvironmentSpec())
        with pytest.raises(AdmissionError, match="identity"):
            future.result(timeout=10)
        future.cleanup(timeout=5)
        assert not marker.exists()
    finally:
        executor.close(cancel=True, timeout=5)


def test_affinity_marker_is_visible_only_after_payload_write(tmp_path, monkeypatch):
    """An observer cannot mistake a created but unwritten marker for readiness."""
    marker = tmp_path / "worker.running"
    (tmp_path / "worker.release").touch()
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {1, 3}, raising=False)
    original_write = Path.write_text

    def observe_write(path, data, **kwargs):
        original_write(path, "", **kwargs)
        assert not marker.exists()
        return original_write(path, data, **kwargs)

    monkeypatch.setattr(Path, "write_text", observe_write)

    assert _hold_affinity(str(tmp_path), "worker") == [1, 3]
    assert marker.read_text(encoding="ascii") == "1,3"
    assert not (tmp_path / "worker.pending").exists()


def _write_sentinel(path: str) -> str:
    """Write a workload marker only if this callable is actually invoked."""
    Path(path).write_text("invoked", encoding="ascii")
    return "invoked"


def _spawn_fd_holder() -> int:
    """Return after a same-group descendant retains the redirected output fds."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    return child.pid


class _UnpickleMarker:
    """Trusted serializer fixture whose reduction writes only during worker loading."""

    def __init__(self, path: str) -> None:
        self.path = path

    def __reduce__(self):
        """Return an importable reconstruction hook with no preflight side effect."""
        return (_mark_unpickled, (self.path,))


def _mark_unpickled(path: str) -> str:
    """Record that the worker deserialized the fixture after receiving payload bytes."""
    Path(path).write_text("unpickled", encoding="ascii")
    return "marker"


def _identity(value: object) -> object:
    """Return a deserialized argument unchanged."""
    return value


@contextmanager
def worker_setup_factory(context, data):
    """Record verified setup entry/exit around a worker payload fixture."""
    path = Path(data["path"])
    assert context.backend == "subprocess"
    assert context.submission_id
    assert context.native_grant["kind"] == "subprocess"
    os.write(1, b"setup entered\n")
    if data.get("output_bytes"):
        os.write(1, b"x" * data["output_bytes"])
    path.write_text("entered", encoding="ascii")
    try:
        yield
    finally:
        os.write(2, b"setup exited\n")
        if data.get("output_bytes"):
            os.write(2, b"y" * data["output_bytes"])
        path.write_text("exited", encoding="ascii")


def _require_setup_entry(path: str) -> str:
    """Require that pre-deserialization setup has entered before invocation."""
    marker = Path(path)
    assert marker.read_text(encoding="ascii") == "entered"
    return "result"


@contextmanager
def failing_worker_setup_factory(_context, _data):
    """Produce a deterministic setup-exit failure after an encoded result exists."""
    yield
    raise RuntimeError("teardown must not replace the result")


@contextmanager
def failing_worker_setup_entry(_context, _data):
    """Fail setup entry before the worker can receive its payload frame."""
    raise RuntimeError("entry failure")
    yield


def _wait_for(path: Path, timeout: float = 10) -> None:
    """Wait for a real worker file marker without serializing a synchronization primitive."""
    deadline = monotonic() + timeout
    while not path.exists():
        if monotonic() >= deadline:
            raise AssertionError(f"worker did not create {path.name}")
        sleep(0.01)


def test_subprocess_future_runs_descriptor_payload_and_retains_output(tmp_path: Path):
    """A concrete Future keeps the real launcher and post-cleanup output."""
    output = ExecutionOutput()
    executor = Executor(SubProcessConfig(spool_directory=tmp_path))
    try:
        future = executor.submit(_emit_and_return, "ok", output=output)

        assert isinstance(future, SubProcessFuture)
        assert future.result(timeout=10) == "ok"
        assert future.process is not None
        assert future.pid is not None
        assert "worker stdout" in output.snapshot().stdout
        assert "worker stderr" in output.snapshot().stderr
        future.cleanup(timeout=5)
        assert "worker stdout" in output.snapshot().stdout
    finally:
        executor.close(cancel=True, timeout=5)


def test_subprocess_worker_setup_runs_before_payload_and_preserves_teardown_output(tmp_path: Path):
    """Post-GO setup receives qualified evidence and unwinds after result encoding."""
    marker = tmp_path / "setup-marker"
    output = ExecutionOutput()
    executor = Executor(SubProcessConfig(spool_directory=tmp_path))
    setup = WorkerSetup(
        factory="tests.execute.test_subprocess_backend:worker_setup_factory",
        data={"path": str(marker)},
    )
    try:
        future = executor.submit(_require_setup_entry, str(marker), worker_setup=setup, output=output)
        assert future.result(timeout=10) == "result"
        assert marker.read_text(encoding="ascii") == "exited"
        snapshot = output.snapshot()
        assert "setup entered" in snapshot.stdout
        assert "setup exited" in snapshot.stderr
        future.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=5)


def test_subprocess_setup_teardown_failure_preserves_result_and_cleanup_evidence(tmp_path: Path):
    """A setup exit failure is independent evidence, not a replacement terminal error."""
    executor = Executor(SubProcessConfig(spool_directory=tmp_path))
    setup = WorkerSetup(
        factory="tests.execute.test_subprocess_backend:failing_worker_setup_factory",
        data={},
    )
    try:
        future = executor.submit(_add, 2, 3, worker_setup=setup)
        assert future.result(timeout=10) == 5
        assert future.snapshot().cleanup_issues[-1].code == "worker_setup_exit_failed"
        assert future.snapshot().cleanup_issues[-1].message == "RuntimeError"
        assert future.snapshot().cleanup_state == "incomplete"
        with pytest.raises(CleanupError):
            future.cleanup(timeout=5)
        assert future.result(timeout=0) == 5
        assert future.snapshot().cleanup_state == "incomplete"
        with pytest.raises(CleanupError):
            executor.close(cancel=True, timeout=5)
    finally:
        if executor._state != "cleanup_incomplete":
            executor.close(cancel=True, timeout=5)


def test_subprocess_setup_entry_failure_withholds_payload_deserialization(tmp_path: Path):
    """A setup-entry failure exits once and never delivers the serialized workload."""
    marker = tmp_path / "unpickled"
    executor = Executor(SubProcessConfig(spool_directory=tmp_path))
    setup = WorkerSetup(
        factory="tests.execute.test_subprocess_backend:failing_worker_setup_entry",
        data={},
    )
    try:
        future = executor.submit(_identity, _UnpickleMarker(str(marker)), worker_setup=setup)
        with pytest.raises(Exception, match="setup failed"):
            future.result(timeout=10)
        assert not marker.exists()
        future.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=5)


def test_subprocess_setup_output_larger_than_a_pipe_drains_before_payload(tmp_path: Path):
    """Setup and teardown output drain concurrently without blocking payload transfer."""
    marker = tmp_path / "setup-marker"
    output = ExecutionOutput()
    executor = Executor(SubProcessConfig(
        spool_directory=tmp_path, output_frame_limit_bytes=1024,
        live_output_queue_limit_bytes=1024, output_limit_bytes=2048,
    ))
    setup = WorkerSetup(
        factory="tests.execute.test_subprocess_backend:worker_setup_factory",
        data={"path": str(marker), "output_bytes": 131_072},
    )
    try:
        future = executor.submit(_require_setup_entry, str(marker), worker_setup=setup, output=output)
        assert future.result(timeout=10) == "result"
        snapshot = output.snapshot()
        assert snapshot.stdout_truncated and snapshot.stderr_truncated
        future.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=5)


def test_subprocess_spool_retry_skips_retired_backend_cleanup(tmp_path: Path, monkeypatch):
    """A failed spool removal retries without re-entering retired native cleanup."""
    from dryml.execute._spooling import SpoolBudget
    from dryml.execute.errors import CleanupError

    executor = Executor(SubProcessConfig(spool_directory=tmp_path))
    future = executor.submit(_add, 1, 2)
    assert future.result(timeout=10) == 3
    backend = executor._backend
    assert backend is not None
    payload = executor._submissions[future].payload
    original_rmdir = Path.rmdir
    failed = False

    def fail_payload_rmdir_once(path: Path) -> None:
        nonlocal failed
        if path == payload.path.parent and not failed:
            failed = True
            raise OSError("busy")
        original_rmdir(path)

    monkeypatch.setattr(Path, "rmdir", fail_payload_rmdir_once)
    with pytest.raises(CleanupError):
        future.cleanup(timeout=5)
    assert failed
    assert future.submission_id not in backend._known
    assert SpoolBudget.snapshot().reserved_bytes > 0

    future.cleanup(timeout=5)
    executor.close(timeout=5)
    assert SpoolBudget.snapshot().reserved_bytes == 0


def test_subprocess_supports_lambda_closure_and_remote_failure(tmp_path: Path):
    """Dill snapshots supported closures once and sanitizes remote failures."""
    offset = 4
    executor = Executor(SubProcessConfig(spool_directory=tmp_path))
    assert executor.run(lambda value: value + offset, 3) == 7
    failed = executor.submit(lambda: (_ for _ in ()).throw(ValueError("private argument text")))
    with pytest.raises(Exception, match="remote"):
        failed.result(timeout=10)
    failed.cleanup(timeout=5)
    executor.close(timeout=5)


def test_subprocess_rejects_oversized_worker_result_without_success(tmp_path: Path):
    """The bounded result channel cannot publish an oversized result."""
    executor = Executor(
        SubProcessConfig(
            spool_directory=tmp_path,
            invocation_limit_bytes=16_384,
            result_limit_bytes=64,
            spool_limit_bytes=16_448,
        )
    )
    future = executor.submit(lambda: "x" * 4096)
    with pytest.raises(Exception):
        future.result(timeout=10)
    future.cleanup(timeout=5)
    executor.close(timeout=5)


def test_real_constrained_workers_are_disjoint_and_release_queued_demand(tmp_path: Path, monkeypatch):
    """Busy real grants reject waiting demand, then release it with exact affinity."""
    _require_cpu_affinity()
    available = sorted(os.sched_getaffinity(0))
    if len(available) < 2:
        pytest.skip("requires at least two available host CPUs for real affinity proof")
    one_cpu_world = WorldRequirement({"main": RoleRequirement(resources=ResourceRequirement(cpus=CountConstraint(1, 1)))})
    busy_world = WorldRequirement({"main": RoleRequirement(resources=ResourceRequirement(cpus=CountConstraint(len(available) - 1, len(available) - 1)))})
    for name in ("first", "second", "queued", "reopened"):
        (tmp_path / name).mkdir()
    first = Executor(SubProcessConfig(spool_directory=tmp_path / "first", admission_timeout=3))
    second = Executor(SubProcessConfig(spool_directory=tmp_path / "second", admission_timeout=3))
    queued = Executor(SubProcessConfig(spool_directory=tmp_path / "queued", admission_timeout=5))
    reopened = Executor(SubProcessConfig(spool_directory=tmp_path / "reopened", admission_timeout=5))
    first.start()
    second.start()
    queued.start()
    reopened.start()
    authority = queued._backend._authority
    wait_started = Event()
    original_wait_for_change = authority.wait_for_change

    def observe_wait_for_change(*args, **kwargs):
        wait_started.set()
        return original_wait_for_change(*args, **kwargs)

    monkeypatch.setattr(authority, "wait_for_change", observe_wait_for_change)
    try:
        one = first.submit(_hold_affinity, str(tmp_path), "one", world=one_cpu_world)
        two = second.submit(_hold_affinity, str(tmp_path), "two", world=one_cpu_world)
        _wait_for(tmp_path / "one.running")
        if two.done():
            two.result(timeout=0)
        _wait_for(tmp_path / "two.running")
        observed = {
            tuple(map(int, (tmp_path / "one.running").read_text(encoding="ascii").split(","))),
            tuple(map(int, (tmp_path / "two.running").read_text(encoding="ascii").split(","))),
        }
        assert len(observed) == 2
        first_snapshot = first.resources(timeout=3)
        second_snapshot = second.resources(timeout=3)
        assert first_snapshot.allocated.cpus == second_snapshot.allocated.cpus == 1.0
        assert first_snapshot.available.cpus == second_snapshot.available.cpus == len(available) - 2
        blocked = queued.submit(_write_sentinel, str(tmp_path / "busy-sentinel"), world=busy_world)
        assert wait_started.wait(timeout=3)
        with pytest.raises(AdmissionError, match="waiting for local resources"):
            blocked.result(timeout=7)
        assert not (tmp_path / "busy-sentinel").exists()
        blocked.cleanup(timeout=3)
        (tmp_path / "one.release").touch()
        assert one.result(timeout=5) in [list(value) for value in observed]
        one.cleanup(timeout=3)
        released = reopened.submit(_hold_affinity, str(tmp_path), "reopened", world=busy_world)
        _wait_for(tmp_path / "reopened.running")
        reopened_affinity = tuple(map(int, (tmp_path / "reopened.running").read_text(encoding="ascii").split(",")))
        assert len(reopened_affinity) == len(available) - 1
        assert len(set(reopened_affinity)) == len(available) - 1
        (tmp_path / "reopened.release").touch()
        assert released.result(timeout=5) == list(reopened_affinity)
        released.cleanup(timeout=3)
        (tmp_path / "two.release").touch()
        assert two.result(timeout=5) in [list(value) for value in observed]
        two.cleanup(timeout=3)
    finally:
        for name in ("one", "two", "reopened"):
            (tmp_path / f"{name}.release").touch()
        for executor in (first, second, queued, reopened):
            executor.close(cancel=True, timeout=5)


def test_rejected_go_never_deserializes_payload_at_actual_worker_boundary(tmp_path: Path, monkeypatch):
    """A worker receiving STOP instead of GO never runs a trusted reduction hook."""
    from dryml.execute._protocol import FrameState, decode_exact_frame, encode_control

    executor = Executor(SubProcessConfig(spool_directory=tmp_path))
    executor.start()
    backend = executor._backend
    assert backend is not None
    original_send = backend._send

    def stop_at_go(connection, data):
        frame = decode_exact_frame(data, header_limit=backend._config.control_header_limit_bytes, payload_limit=backend._limits())
        if frame.state is FrameState.GO:
            data = encode_control(FrameState.STOP, frame.correlation, {"reason": "test-rejection"}, header_limit=backend._config.control_header_limit_bytes)
        original_send(connection, data)

    monkeypatch.setattr(backend, "_send", stop_at_go)
    marker = tmp_path / "unpickled"
    try:
        future = executor.submit(_identity, _UnpickleMarker(str(marker)))
        assert not marker.exists()
        with pytest.raises(ExecutionUncertainError):
            future.result(timeout=5)
        assert not marker.exists()
        future.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=5)


def test_known_result_survives_missing_output_fences_and_group_cleanup(tmp_path: Path):
    """A descendant-held fd leaves capture incomplete without invalidating its result."""
    output = ExecutionOutput()
    executor = Executor(SubProcessConfig(spool_directory=tmp_path, output_final_timeout=0.05))
    try:
        future = executor.submit(_spawn_fd_holder, output=output)
        assert future.result(timeout=5) > 0
        sleep(0.1)
        assert not output.snapshot().complete
        future.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=5)


def test_duplicate_callbacks_are_coordinator_only_and_may_close_executor(tmp_path: Path):
    """Every initial callback receives the exact Future outside worker execution."""
    executor = Executor(SubProcessConfig(spool_directory=tmp_path))
    observed: list[tuple[int, object]] = []

    def callback(future):
        observed.append((os.getpid(), future))
        executor.close(timeout=5)

    future = executor.submit(os.getpid, done_callbacks=(callback, callback))
    worker_pid = future.result(timeout=5)
    deadline = monotonic() + 5
    while len(observed) != 2 and monotonic() < deadline:
        sleep(0.01)
    assert observed == [(os.getpid(), future), (os.getpid(), future)]
    assert worker_pid != os.getpid()
    while executor.state != "closed" and monotonic() < deadline:
        sleep(0.01)
    assert executor.state == "closed"


def test_subprocess_launch_thread_failure_leaves_only_executor_spool_cleanup(tmp_path: Path, monkeypatch):
    """A failed fresh launch thread does not leave a backend cleanup association."""
    original_thread = executor_module.Thread

    def fail_subprocess_thread(*args, **kwargs):
        if kwargs.get("name") == "dryml-execute-subprocess":
            raise RuntimeError("launch thread unavailable")
        return original_thread(*args, **kwargs)

    monkeypatch.setattr("dryml.execute.subprocess.Thread", fail_subprocess_thread)
    executor = Executor(SubProcessConfig(spool_directory=tmp_path))
    try:
        future = executor.submit(_add, 1, 2)
        with pytest.raises(Exception):
            future.result(timeout=5)
        future.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=5)


def test_cancel_before_subprocess_admission_settles_launch_cleanup(tmp_path: Path, monkeypatch):
    """An accepted cancellation before backend admission remains executor-cleanable."""
    executor = Executor(SubProcessConfig(spool_directory=tmp_path))
    executor.start()
    backend = executor._backend
    assert backend is not None
    entered = Event()
    release = Event()
    original_run = backend._run

    def paused_run(call, future):
        entered.set()
        assert release.wait(timeout=2)
        original_run(call, future)

    monkeypatch.setattr(backend, "_run", paused_run)
    try:
        future = executor.submit(_add, 1, 2)
        assert entered.wait(timeout=2)
        assert future.cancel()
        release.set()
        future.cleanup(timeout=2)
        executor.close(timeout=2)
    finally:
        release.set()
        executor.close(cancel=True, timeout=2)


@pytest.mark.parametrize("wrapped", (False, True))
def test_windows_assignment_failure_retains_run_and_charge_until_retry(monkeypatch, tmp_path: Path, wrapped):
    """Unconfirmed pre-GO Windows cleanup retains the launcher and reservation owner."""
    backend = SubProcessConfig(spool_directory=tmp_path, python_executable=Path(sys.executable)).create_backend()
    authority = ResourceAuthority()
    backend._authority = authority
    future = backend.create_future("windows-assignment-failure", ExecutionOutput())
    reservation = authority.reserve(
        future.submission_id, subprocess_module.ResourceAmounts(1.0, None, {}, {}),
        generation="subprocess-v1", attempt="0", total=subprocess_module.ResourceAmounts(1.0, None, {}, {}),
    )
    assert reservation is not None
    process = SimpleNamespace(pid=101, poll=lambda: None)
    calls: list[object] = []

    def reconcile(_owner, **_kwargs):
        calls.append(_owner)
        return len(calls) > 1

    monkeypatch.setattr(subprocess_module, "os", _os_with_name("nt"))
    monkeypatch.setattr(backend, "_reserve", lambda *_args: reservation)
    runtime = ["conda", "run", "python"] if wrapped else [sys.executable]
    monkeypatch.setattr(backend, "_select_environment", lambda _call: (runtime, None))
    monkeypatch.setattr(backend, "_launch", lambda *_args: process)
    monkeypatch.setattr(subprocess_module._WindowsJob, "assign", lambda _process: (_ for _ in ()).throw(OSError("AssignProcessToJobObject failed")))
    monkeypatch.setattr(subprocess_module.OwnedProcess, "reconcile", reconcile)
    with backend._lock:
        backend._known.add(future.submission_id)
        backend._launching[future.submission_id] = Event()
    call = SimpleNamespace(submission_id=future.submission_id, admission_deadline=monotonic() + 1)

    backend._run(call, future)

    assert future.done()
    assert calls[0].root_termination_sufficient is (not wrapped)
    assert future.process is process
    assert future.submission_id in backend._runs
    assert authority.snapshot(total=subprocess_module.ResourceAmounts(1.0, None, {}, {})).allocated.cpus == 1.0
    backend.reconcile_cleanup(future.submission_id, timeout=1)
    assert future.submission_id not in backend._runs
    assert authority.snapshot(total=subprocess_module.ResourceAmounts(1.0, None, {}, {})).allocated.cpus == 0
