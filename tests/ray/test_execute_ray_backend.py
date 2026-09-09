"""Explicit existing-server integration proof for the Execute Ray backend."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from concurrent.futures import CancelledError
from pathlib import Path
from threading import Event, current_thread
from time import monotonic, sleep

import pytest

from dryml.execute.executor import Executor
from dryml.execute.errors import ExecutionDeadlineExceeded
from dryml.execute.output import ExecutionOutput
from dryml.execute.ray import RayBackendConfig, RayFuture
from dryml.worlds import CountConstraint, ResourceRequirement, RoleRequirement, WorldRequirement
from .conftest import require_ray_integration


def _ray_address() -> str:
    """Require an explicit caller-owned Ray endpoint for enabled integration."""
    return require_ray_integration()


def _emit(value: str) -> str:
    """Write both captured streams and return a closure-compatible value."""
    os.write(1, b"ray stdout\n")
    os.write(2, b"ray stderr\n")
    return value


def _one_cpu_world() -> WorldRequirement:
    """Require the Ray task's verified one-logical-CPU worker grant."""
    return WorldRequirement({
        "main": RoleRequirement(resources=ResourceRequirement(cpus=CountConstraint(1, 1))),
    })


def _one_cpu_small_memory_world() -> WorldRequirement:
    """Require Ray's supported one-byte logical memory grant without device claims."""
    return WorldRequirement({
        "main": RoleRequirement(resources=ResourceRequirement(cpus=CountConstraint(1, 1), memory=CountConstraint(1, 1))),
    })


def _wait_for_marker(marker: Path, future, *, timeout: float = 30) -> None:
    """Wait for a user-function marker rather than guessing worker progress."""
    deadline = monotonic() + timeout
    while not marker.exists():
        assert not future.done(), "worker ended before the user function started"
        if monotonic() >= deadline:
            raise TimeoutError("worker did not create its running marker")
        sleep(0.01)


def test_existing_ray_executes_lambda_closure_result_and_output(tmp_path: Path):
    """Attach only to the supplied server and use common Future/output semantics."""
    offset = 3
    output = ExecutionOutput()
    executor = Executor(RayBackendConfig(address=_ray_address(), spool_directory=tmp_path, admission_timeout=90, connect_timeout=60))
    try:
        future = executor.submit(
            lambda value: _emit(str(value + offset)), 4, output=output,
            world=_one_cpu_world(),
        )
        assert isinstance(future, RayFuture)
        assert future.result(timeout=30) == "7"
        assert future.object_ref is not None
        assert future.worker_id is not None
        assert future.node_id is not None
        assert "ray stdout" in output.snapshot().stdout
        assert "ray stderr" in output.snapshot().stderr
        future.cleanup(timeout=30)
        assert "ray stdout" in output.snapshot().stdout
    finally:
        executor.close(cancel=True, timeout=30)


def test_existing_ray_running_cancellation_requires_exact_worker_exit(tmp_path: Path):
    """Confirm cancellation only after the marked Ray worker and task terminate."""
    def blocking_workload(marker: Path) -> None:
        """Signal post-GO execution then retain the exact worker until cancellation."""
        marker.write_text("running", encoding="ascii")
        import time

        time.sleep(30)

    executor = Executor(RayBackendConfig(address=_ray_address(), spool_directory=tmp_path, admission_timeout=90, connect_timeout=60, termination_timeout=10))
    try:
        marker = tmp_path / "cancel-started"
        future = executor.submit(blocking_workload, marker, world=_one_cpu_world())
        _wait_for_marker(marker, future)
        assert future.object_ref is not None
        assert future.worker_id is not None
        assert future.worker_pid is not None
        assert executor.resources(timeout=10).allocated.cpus == 1.0
        assert future.request_cancel()
        with pytest.raises(CancelledError):
            future.result(timeout=20)
        run = executor._backend._runs[future.submission_id]
        assert run.native_cancelled
        assert run.termination_qualified
        # Terminal cancellation does not refund the native charge before cleanup.
        assert executor.resources(timeout=10).allocated.cpus == 1.0
        future.cleanup(timeout=20)
        assert future.object_ref is None
        assert executor.resources(timeout=10).allocated.cpus == 0.0
    finally:
        executor.close(cancel=True, timeout=30)


def test_existing_ray_execution_deadline_requires_confirmed_termination(tmp_path: Path):
    """Apply a workload deadline after GO and retain only a confirmed expiry."""
    def blocking_workload(marker: Path) -> None:
        """Signal post-GO execution then retain the exact worker until cancellation."""
        marker.write_text("running", encoding="ascii")
        import time

        time.sleep(30)

    executor = Executor(RayBackendConfig(address=_ray_address(), spool_directory=tmp_path, admission_timeout=90, connect_timeout=60, termination_timeout=10))
    try:
        marker = tmp_path / "deadline-started"
        future = executor.submit(blocking_workload, marker, world=_one_cpu_world(), execution_timeout=0.5)
        _wait_for_marker(marker, future)
        with pytest.raises(ExecutionDeadlineExceeded):
            future.result(timeout=20)
        assert marker.exists()
        assert executor.resources(timeout=10).allocated.cpus == 1.0
        future.cleanup(timeout=20)
    finally:
        executor.close(cancel=True, timeout=30)


def test_existing_ray_result_before_deadline_is_not_rewritten(tmp_path: Path):
    """Keep a completed user result when the later deadline watcher wakes."""
    def complete() -> str:
        """Return without importing the pytest module in the Ray worker."""
        return "completed-before-deadline"

    callbacks: list[tuple[object, str]] = []
    callback_done = Event()

    def callback(completed) -> None:
        """Record the coordinator Future without serializing the callback to Ray."""
        callbacks.append((completed, current_thread().name))
        callback_done.set()

    executor = Executor(RayBackendConfig(address=_ray_address(), spool_directory=tmp_path, admission_timeout=90, connect_timeout=60, termination_timeout=10))
    try:
        future = executor.submit(complete, world=_one_cpu_world(), execution_timeout=0.5, done_callbacks=(callback,))
        assert future.result(timeout=20) == "completed-before-deadline"
        assert callback_done.wait(5)
        assert callbacks == [(future, "dryml-execute-callback")]
        sleep(0.75)
        assert future.result(timeout=0) == "completed-before-deadline"
        future.cleanup(timeout=20)
    finally:
        executor.close(cancel=True, timeout=30)


def test_existing_ray_small_logical_memory_grant_is_admitted(tmp_path: Path):
    """Exercise Ray's logical memory scheduler option without inventing memory IDs."""
    def complete() -> str:
        """Return from the selected worker without importing the pytest module."""
        return "logical-memory-granted"

    executor = Executor(RayBackendConfig(address=_ray_address(), spool_directory=tmp_path, admission_timeout=90, connect_timeout=60, termination_timeout=10))
    try:
        future = executor.submit(complete, world=_one_cpu_small_memory_world())
        assert future.result(timeout=20) == "logical-memory-granted"
        assert future.snapshot().report is not None
        future.cleanup(timeout=20)
    finally:
        executor.close(cancel=True, timeout=30)


def test_existing_ray_concurrent_logical_cpu_charges_are_not_double_subtracted(tmp_path: Path):
    """Keep two distinct one-CPU grants charged exactly once against four CPUs."""
    def blocking_workload(marker: Path, release: Path, value: str) -> str:
        """Signal post-GO execution and retain distinct captured output per task."""
        marker.write_text("running", encoding="ascii")
        import os
        import time

        os.write(1, f"worker-{value}\n".encode("ascii"))
        while not release.exists():
            time.sleep(0.01)
        return value

    executor = Executor(RayBackendConfig(address=_ray_address(), spool_directory=tmp_path, admission_timeout=90, connect_timeout=60, termination_timeout=10))
    first_output = ExecutionOutput()
    second_output = ExecutionOutput()
    try:
        release = tmp_path / "release"
        first = executor.submit(blocking_workload, tmp_path / "first-started", release, "first", world=_one_cpu_world(), output=first_output)
        second = executor.submit(blocking_workload, tmp_path / "second-started", release, "second", world=_one_cpu_world(), output=second_output)
        _wait_for_marker(tmp_path / "first-started", first)
        _wait_for_marker(tmp_path / "second-started", second)
        resources = executor.resources(timeout=10)
        assert resources.total.cpus == 4.0
        assert resources.allocated.cpus == 2.0
        assert resources.available.cpus == 2.0
        release.write_text("release", encoding="ascii")
        assert first.result(timeout=20) == "first"
        assert second.result(timeout=20) == "second"
        assert "worker-first" in first_output.snapshot().stdout
        assert "worker-second" in second_output.snapshot().stdout
        first.cleanup(timeout=20)
        second.cleanup(timeout=20)
        assert executor.resources(timeout=10).allocated.cpus == 0.0
    finally:
        executor.close(cancel=True, timeout=30)


def test_existing_ray_external_exact_task_loss_is_not_reported_as_success(tmp_path: Path):
    """Isolate an exact native-task loss and retain its unconfirmed charge."""
    marker = tmp_path / "loss-started"
    script = textwrap.dedent(
        f"""
        from pathlib import Path
        from time import monotonic, sleep

        import ray

        from dryml.execute.errors import CleanupError, ExecutionError, ExecutionUncertainError
        from dryml.execute.executor import Executor
        from dryml.execute.ray import RayBackendConfig
        from dryml.worlds import CountConstraint, ResourceRequirement, RoleRequirement, WorldRequirement

        marker = Path({str(marker)!r})

        def workload() -> None:
            marker.write_text("running", encoding="ascii")
            import time

            time.sleep(30)

        world = WorldRequirement({{
            "main": RoleRequirement(resources=ResourceRequirement(cpus=CountConstraint(1, 1))),
        }})
        executor = Executor(RayBackendConfig(
            address={_ray_address()!r}, spool_directory=Path({str(tmp_path)!r}),
            admission_timeout=90, connect_timeout=60, termination_timeout=5,
        ))
        future = executor.submit(workload, world=world)
        deadline = monotonic() + 30
        while not marker.exists():
            if future.done() or monotonic() >= deadline:
                raise RuntimeError("user workload did not start")
            sleep(0.01)
        assert future.object_ref is not None
        ray.cancel(future.object_ref, force=True, recursive=False)
        try:
            future.result(timeout=20)
        except (ExecutionError, ExecutionUncertainError):
            pass
        else:
            raise AssertionError("external task loss was reported as a user result")
        try:
            future.cleanup(timeout=5)
        except CleanupError:
            pass
        else:
            raise AssertionError("unqualified task loss released its resource charge")
        allocations = executor.resources(timeout=10).allocations
        assert len(allocations) == 1 and allocations[0].state == "unconfirmed"
        try:
            executor.close(cancel=True, timeout=5)
        except CleanupError:
            pass
        else:
            raise AssertionError("unqualified task loss closed as if cleanup succeeded")
        """
    )
    completed = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, text=True, capture_output=True, timeout=60)
    assert completed.returncode == 0, completed.stderr[-4000:]


def test_existing_borrowed_ray_driver_survives_close_and_namespace_conflict(tmp_path: Path):
    """Keep a test-owned borrowed driver alive while rejecting another namespace."""
    namespace = "dryml-u6-borrowed-driver"
    script = textwrap.dedent(
        f"""
        from pathlib import Path

        import ray

        from dryml.execute.errors import ExecutionError
        from dryml.execute.executor import Executor
        from dryml.execute.ray import RayBackendConfig

        def workload() -> str:
            return "borrowed-driver-result"

        ray.init(address={_ray_address()!r}, namespace={namespace!r})
        try:
            executor = Executor(RayBackendConfig(
                address={_ray_address()!r}, namespace={namespace!r},
                spool_directory=Path({str(tmp_path)!r}), admission_timeout=90,
                connect_timeout=60,
            ))
            future = executor.submit(workload)
            assert future.result(timeout=20) == "borrowed-driver-result"
            future.cleanup(timeout=20)
            executor.close(cancel=True, timeout=20)
            assert ray.is_initialized()
            conflicting = Executor(RayBackendConfig(
                address={_ray_address()!r}, namespace="dryml-u6-conflict",
                spool_directory=Path({str(tmp_path)!r}) / "conflict",
                admission_timeout=90, connect_timeout=60,
            ))
            try:
                conflicting.start()
            except ExecutionError:
                pass
            else:
                raise AssertionError("a borrowed namespace conflict was accepted")
            assert ray.is_initialized()
        finally:
            ray.shutdown()
        """
    )
    completed = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, text=True, capture_output=True, timeout=60)
    assert completed.returncode == 0, completed.stderr[-4000:]
