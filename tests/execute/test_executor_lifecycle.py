from __future__ import annotations

import inspect
import time
from threading import Event, Thread

import pytest

from dryml.execute._spooling import SpoolBudget
from dryml.execute.errors import CleanupError, ExecutionError

from .test_executor import FakeBackend, config


def test_close_waits_for_work_and_cancel_close_requests_prestart_cancellation(tmp_path):
    """Normal close waits for a running fake call while cancellation asks its Future first."""
    from dryml.execute.executor import Executor

    backend = FakeBackend()
    gate = Event()
    backend.run_gate = gate
    executor = Executor(config(tmp_path, backend))
    future = executor.submit(lambda: 2)
    started = Event()

    def close() -> None:
        started.set()
        executor.close()

    closer = Thread(target=close)
    closer.start()
    assert started.wait(1)
    assert closer.is_alive()
    gate.set()
    closer.join(2)
    assert not closer.is_alive()
    assert future.result() == 2
    assert executor.state == "closed"


def test_normal_close_waits_past_termination_timeout_for_accepted_work(tmp_path):
    """Normal closure preserves its unbounded work wait despite a tiny cleanup bound."""
    from dryml.execute.executor import Executor

    backend = FakeBackend()
    gate = Event()
    backend.run_gate = gate
    executor = Executor(config(tmp_path, backend, termination_timeout=0.01))
    executor.submit(lambda: 2)
    closer = Thread(target=executor.close)
    closer.start()
    time.sleep(0.05)
    assert closer.is_alive()
    gate.set()
    closer.join(1)
    assert executor.state == "closed"


def test_cancel_close_requests_running_cancellation_before_waiting_for_dispatch(tmp_path):
    """Cancellation reaches a blocked backend worker before close waits for admission."""
    from dryml.execute.executor import Executor

    class CancellableBackend(FakeBackend):
        def submit(self, call, *, future):
            future._set_cancel_requester(lambda: gate.set() or cancelled.set())
            entered.set()
            super().submit(call, future=future)

    gate = Event()
    cancelled = Event()
    entered = Event()
    backend = CancellableBackend()
    backend.run_gate = gate
    executor = Executor(config(tmp_path, backend, termination_timeout=1))
    executor.submit(lambda: 2)
    assert entered.wait(1)
    executor.close(cancel=True)
    assert cancelled.is_set()
    assert executor.state == "closed"


def test_normal_close_starts_an_immediately_accepted_pending_call(tmp_path):
    """Close waits for accepted initialization instead of rejecting that call as closing."""
    from dryml.execute.executor import Executor

    backend = FakeBackend()
    backend.start_gate = Event()
    executor = Executor(config(tmp_path, backend, termination_timeout=0.01))
    future = executor.submit(lambda: 4)
    assert backend.start_entered.wait(1)
    closer = Thread(target=executor.close)
    closer.start()
    time.sleep(0.05)
    assert closer.is_alive()
    backend.start_gate.set()
    closer.join(1)
    assert future.result() == 4
    assert executor.state == "closed"


def test_executor_submit_has_only_the_approved_public_controls():
    """Private one-off ownership does not leak a user-callable submit hook."""
    from dryml.execute.executor import Executor

    assert "_on_accepted" not in inspect.signature(Executor.submit).parameters


def test_normal_close_bounds_initializer_after_admission_failed(tmp_path):
    """A terminal call does not turn unresolved initialization into an infinite wait."""
    from dryml.execute.errors import AdmissionError
    from dryml.execute.executor import Executor

    backend = FakeBackend()
    backend.start_gate = Event()
    executor = Executor(config(tmp_path, backend, admission_timeout=0.05,
                               termination_timeout=0.05))
    future = executor.submit(lambda: 1)
    assert backend.start_entered.wait(1)
    with pytest.raises(AdmissionError):
        future.result(timeout=1)
    failures = []
    closer = Thread(target=lambda: _capture(failures, executor.close))
    try:
        closer.start()
        closer.join(1)
        assert not closer.is_alive()
        assert len(failures) == 1 and isinstance(failures[0], CleanupError)
        assert executor.state == "cleanup_incomplete"
        assert not backend.closed
    finally:
        backend.start_gate.set()
        closer.join(1)
        executor.close(timeout=1)


def test_cleanup_failure_retains_spool_charge_until_retry(tmp_path, monkeypatch):
    """An unlink failure leaves a retryable cleanup-incomplete executor, not a false close."""
    from pathlib import Path

    from dryml.execute.executor import Executor

    backend = FakeBackend()
    executor = Executor(config(tmp_path, backend))
    future = executor.submit(lambda: 1)
    assert future.result(timeout=2) == 1
    original = Path.rmdir
    monkeypatch.setattr(Path, "rmdir", lambda path: (_ for _ in ()).throw(OSError("busy")))
    with pytest.raises(CleanupError):
        executor.close(timeout=0.1)
    assert executor.state == "cleanup_incomplete"
    assert SpoolBudget.snapshot().reserved_bytes > 0
    monkeypatch.setattr(Path, "rmdir", original)
    executor.close(timeout=1)
    assert executor.state == "closed"
    assert SpoolBudget.snapshot().leases == 0


def test_idle_lease_conflict_rejects_without_creating_a_future(tmp_path):
    """The first preflight lease remains through idle ownership and rejects another quota."""
    from dryml.execute.executor import Executor

    first_backend = FakeBackend()
    first = Executor(config(tmp_path, first_backend))
    first.submit(lambda: 1).result(timeout=2)
    assert SpoolBudget.snapshot().leases == 1
    second_backend = FakeBackend()
    second = Executor(config(tmp_path, second_backend, spool_limit_bytes=40_001))
    with pytest.raises(ExecutionError, match="spool_configuration_conflict"):
        second.submit(lambda: 2)
    assert second_backend.future_calls == 0
    second.close()
    first.close()


def test_callback_can_close_its_parent_without_waiting_for_callback_delivery(tmp_path):
    """Completion callbacks run outside lifecycle ownership and can close the executor."""
    from dryml.execute.executor import Executor

    backend = FakeBackend()
    executor = Executor(config(tmp_path, backend))
    completed = Event()

    def close_from_callback(_future) -> None:
        executor.close()
        completed.set()

    future = executor.submit(lambda: 1, done_callbacks=(close_from_callback,))
    assert future.result(timeout=2) == 1
    assert completed.wait(2)
    assert executor.state == "closed"


def test_concurrent_close_waits_for_pending_start_and_closes_once(tmp_path):
    """Concurrent close callers share one owner and never publish false closure."""
    from dryml.execute.errors import CleanupError
    from dryml.execute.executor import Executor

    backend = FakeBackend()
    backend.start_gate = Event()
    executor = Executor(config(tmp_path, backend, termination_timeout=0.05))
    executor.submit(lambda: 1)
    assert backend.start_entered.wait(1)
    failures: list[BaseException] = []

    def close() -> None:
        try:
            executor.close(timeout=0.05)
        except BaseException as exc:
            failures.append(exc)

    first = Thread(target=close)
    second = Thread(target=close)
    first.start()
    second.start()
    first.join(1)
    second.join(1)
    assert executor.state == "cleanup_incomplete"
    assert all(isinstance(failure, CleanupError) for failure in failures)
    assert not backend.closed
    backend.start_gate.set()
    executor.close(timeout=1)
    assert len(backend.closed) == 1


def test_close_rejects_a_paused_preflight_and_retries_its_unlink_cleanup(tmp_path, monkeypatch):
    """A rejected paused preflight keeps its spool charge until later cleanup succeeds."""
    from pathlib import Path

    from dryml.execute._spooling import PayloadSpooler
    from dryml.execute.executor import Executor

    paused = Event()
    release = Event()
    original_snapshot = PayloadSpooler.snapshot

    def snapshot(self, *args, **kwargs):
        value = original_snapshot(self, *args, **kwargs)
        paused.set()
        assert release.wait(1)
        return value

    monkeypatch.setattr(PayloadSpooler, "snapshot", snapshot)
    executor = Executor(config(tmp_path, FakeBackend()))
    submission_failure: list[BaseException] = []
    producer = Thread(target=lambda: _capture(submission_failure, lambda: executor.submit(lambda: 1)))
    producer.start()
    assert paused.wait(1)
    original_rmdir = Path.rmdir
    monkeypatch.setattr(Path, "rmdir", lambda _path: (_ for _ in ()).throw(OSError("busy")))
    close_failure: list[BaseException] = []
    closer = Thread(target=lambda: _capture(close_failure, lambda: executor.close(timeout=1)))
    closer.start()
    release.set()
    producer.join(1)
    closer.join(1)
    assert submission_failure and isinstance(submission_failure[0], RuntimeError)
    assert close_failure and isinstance(close_failure[0], CleanupError)
    assert executor.state == "cleanup_incomplete"
    assert SpoolBudget.snapshot().reserved_bytes > 0
    monkeypatch.setattr(Path, "rmdir", original_rmdir)
    executor.close(timeout=1)
    assert executor.state == "closed"


def test_context_body_error_remains_primary_when_close_cleanup_fails(tmp_path, monkeypatch):
    """Context cleanup is chained behind, rather than replacing, a body exception."""
    from pathlib import Path

    from dryml.execute.executor import Executor

    backend = FakeBackend()
    executor = Executor(config(tmp_path, backend))
    original_rmdir = Path.rmdir
    monkeypatch.setattr(Path, "rmdir", lambda _path: (_ for _ in ()).throw(OSError("busy")))
    with pytest.raises(ValueError, match="body") as failure:
        with executor:
            assert executor.submit(lambda: 1).result(timeout=1) == 1
            raise ValueError("body")
    assert isinstance(failure.value.__cause__, CleanupError)
    monkeypatch.setattr(Path, "rmdir", original_rmdir)
    executor.close(timeout=1)


def _capture(failures, call) -> None:
    """Capture a worker-thread exception for event-driven lifecycle assertions."""
    try:
        call()
    except BaseException as exc:
        failures.append(exc)
