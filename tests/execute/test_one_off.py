from __future__ import annotations

from threading import Event, Thread

import pytest

from dryml.execute.errors import CleanupError

from .test_executor import FakeBackend, config


def test_one_off_future_cleanup_closes_its_hidden_owner(tmp_path):
    """One-off cleanup reconciles the one call and then its retained private executor."""
    from dryml.execute.executor import submit

    backend = FakeBackend()
    future = submit(lambda: 9, backend=config(tmp_path, backend))
    assert future.result(timeout=2) == 9
    future.cleanup(timeout=1)
    assert backend.cleaned == [future.submission_id]
    assert backend.closed


def test_blocking_one_off_cleanup_error_preserves_future(tmp_path, monkeypatch):
    """Blocking one-off cleanup failure exposes the completed Future for recovery."""
    from pathlib import Path

    from dryml.execute.executor import run

    backend = FakeBackend()
    original = Path.rmdir
    monkeypatch.setattr(Path, "rmdir", lambda path: (_ for _ in ()).throw(OSError("busy")))
    with pytest.raises(CleanupError) as failure:
        run(lambda: 5, backend=config(tmp_path, backend))
    assert failure.value.execution is not None
    assert failure.value.execution.result() == 5
    monkeypatch.setattr(Path, "rmdir", original)
    failure.value.execution.cleanup(timeout=1)


def test_one_off_workload_error_remains_primary_when_cleanup_also_fails(tmp_path, monkeypatch):
    """Cleanup failure chains behind, rather than replaces, a workload failure."""
    from pathlib import Path

    from dryml.execute.executor import run

    backend = FakeBackend()
    original = Path.rmdir
    monkeypatch.setattr(Path, "rmdir", lambda path: (_ for _ in ()).throw(OSError("busy")))
    with pytest.raises(ValueError, match="workload") as failure:
        run(lambda: (_ for _ in ()).throw(ValueError("workload")), backend=config(tmp_path, backend))
    assert isinstance(failure.value.__cause__, CleanupError)
    assert failure.value.__cause__.execution is not None
    monkeypatch.setattr(Path, "rmdir", original)
    failure.value.__cause__.execution.cleanup(timeout=1)


def test_reusable_future_cleanup_does_not_close_or_cancel_another_submission(tmp_path):
    """Per-call reconciliation leaves the reusable parent and sibling Future intact."""
    from dryml.execute.executor import Executor

    backend = FakeBackend()
    executor = Executor(config(tmp_path, backend))
    first = executor.submit(lambda: "first")
    second = executor.submit(lambda: "second")
    assert first.result(timeout=2) == "first"
    assert second.result(timeout=2) == "second"
    first.cleanup(timeout=1)
    assert backend.cleaned == [first.submission_id]
    assert second.snapshot().cleanup_state == "pending"
    assert executor.state != "closed"
    executor.close()


def test_one_off_terminal_completion_releases_hidden_owner_without_user_cleanup(tmp_path):
    """A dropped one-off Future cannot retain a completed executor indefinitely."""
    from dryml.execute.executor import _one_off_owners, submit

    backend = FakeBackend()
    future = submit(lambda: 3, backend=config(tmp_path, backend))
    assert future.result(timeout=2) == 3
    assert backend.closed_event.wait(1)
    assert backend.closed
    assert not _one_off_owners


def test_conflicting_one_off_preflight_releases_its_hidden_owner(tmp_path):
    """A one-off denied before reservation does not strand a hidden executor."""
    from dryml.execute.executor import Executor, _one_off_owners, submit
    from dryml.execute.errors import ExecutionError

    retained = Executor(config(tmp_path, FakeBackend()))
    assert retained.submit(lambda: 1).result(timeout=1) == 1
    before = set(_one_off_owners)
    with pytest.raises(ExecutionError, match="spool_configuration_conflict"):
        submit(lambda: 2, backend=config(tmp_path, FakeBackend(), spool_limit_bytes=40_001))
    assert _one_off_owners == before
    retained.close()


def test_one_off_paused_preflight_retains_owner_until_failed_unlink_recovers(tmp_path, monkeypatch):
    """A one-off with no returned Future remains inspectable through preflight recovery."""
    from pathlib import Path

    from dryml.execute._spooling import PayloadSpooler, SpoolBudget
    from dryml.execute.executor import _one_off_owners, submit

    paused = Event()
    release = Event()
    original_snapshot = PayloadSpooler.snapshot

    def snapshot(self, *args, **kwargs):
        value = original_snapshot(self, *args, **kwargs)
        paused.set()
        assert release.wait(1)
        return value

    monkeypatch.setattr(PayloadSpooler, "snapshot", snapshot)
    original_rmdir = Path.rmdir
    monkeypatch.setattr(Path, "rmdir", lambda _path: (_ for _ in ()).throw(OSError("busy")))
    before = set(_one_off_owners)
    failures: list[BaseException] = []

    def invoke() -> None:
        try:
            submit(lambda: 1, backend=config(tmp_path, FakeBackend()))
        except BaseException as exc:
            failures.append(exc)

    worker = Thread(target=invoke)
    worker.start()
    assert paused.wait(1)
    release.set()
    worker.join(1)
    owners = set(_one_off_owners) - before
    assert not failures and owners
    assert SpoolBudget.snapshot().reserved_bytes > 0
    monkeypatch.setattr(Path, "rmdir", original_rmdir)
    owner = owners.pop()
    owner._reconcile_owner(cancel=True)
    assert owner not in _one_off_owners


def test_no_future_preflight_cleanup_retries_after_failure_is_lifted(tmp_path, monkeypatch):
    """A retained owner keeps monitoring a failed preflight that returned no Future."""
    from pathlib import Path

    from dryml.execute.executor import Executor, _one_off_owners, submit

    original_get_backend = Executor._get_backend
    original_rmdir = Path.rmdir
    retained_before = set(_one_off_owners)

    def fail_after_spool(self, *args, **kwargs):
        raise ValueError("preflight failed after spool publication")

    monkeypatch.setattr(Executor, "_get_backend", fail_after_spool)
    monkeypatch.setattr(Path, "rmdir", lambda _path: (_ for _ in ()).throw(OSError("busy")))
    with pytest.raises(ValueError, match="preflight failed"):
        submit(
            lambda: 1,
            backend=config(
                tmp_path,
                FakeBackend(),
                termination_timeout=0.02,
                one_off_cleanup_attempts=1,
                one_off_cleanup_retry_interval=0.01,
            ),
        )
    assert set(_one_off_owners) - retained_before
    monkeypatch.setattr(Executor, "_get_backend", original_get_backend)
    monkeypatch.setattr(Path, "rmdir", original_rmdir)
    deadline = __import__("time").monotonic() + 1
    while set(_one_off_owners) - retained_before and __import__("time").monotonic() < deadline:
        __import__("time").sleep(0.01)
    assert not (set(_one_off_owners) - retained_before)
