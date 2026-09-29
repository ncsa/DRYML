"""Core Future adaptation and lifecycle coverage."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Barrier, Event, Thread

import pytest

import dryml.core.execute as execute_module
from dryml.core import Executor, Repo
from dryml.core.execute import (
    CoreAdaptationOutcome,
    CoreExecutionError,
    CoreExecutionFuture,
    CoreOptions,
    CoreOutcomeEvidence,
    PreparedCoreCall,
    _CoreRecovery,
)
from dryml.core.store.dir import DirStore
from dryml.execute.errors import CleanupError, ExecutionUncertainError
from dryml.execute.future import ExecutionFuture
from dryml.execute.subprocess import SubProcessConfig


def _value(value):
    """Return an ordinary result through the core outcome codec."""
    return value


class _ControlledBackend(ExecutionFuture[bytes]):
    """Generic Future whose callback delivery is explicitly controlled by a test."""

    def add_done_callback(self, callback):
        """Retain the bridge without running it in generic callback threads."""
        self.callbacks.append(callback)

    def __init__(self, submission_id="controlled"):
        """Create a generic byte Future with an observable callback bridge list."""
        super().__init__(submission_id)
        self.callbacks = []


class _Storage:
    """Minimal facade-owned recovery resource for deterministic lifecycle tests."""

    cache = "weak"
    update_args = False
    recovery_repo = object()

    def __init__(self, close=None):
        """Retain an optional synchronous close action."""
        self._close = close or (lambda: None)

    def close(self):
        """Run the configured owned-resource close action."""
        self._close()


def _facade(monkeypatch, backend=None, storage=None, recover=None, decode=None, one_off=False):
    """Build a core facade without transport or Store dependencies."""
    backend = backend or _ControlledBackend()
    storage = storage or _Storage()
    decoded = CoreAdaptationOutcome({"success": True}, CoreOutcomeEvidence((), ()))
    monkeypatch.setattr(execute_module, "config", lambda **_: nullcontext())
    monkeypatch.setattr(execute_module, "decode_core_outcome", decode or (lambda *_, **__: decoded))
    monkeypatch.setattr(
        execute_module.SharedDirStoreStrategy,
        "recover",
        recover or (lambda *_, **__: "adapted"),
    )
    return CoreExecutionFuture(
        backend,
        prepared_storage=storage,
        prepared=PreparedCoreCall(b"call", {"repo": {}, "control_store": None}),
        recovery=_CoreRecovery({}),
        return_objects=False,
        result_limit_bytes=1024,
        one_off=one_off,
    ), backend


def test_core_future_projects_backend_lifecycle_and_consumer_recovers_queued_completion(monkeypatch):
    """A consumer starts exactly one adaptation after generic callback delivery is delayed."""
    future, backend = _facade(monkeypatch)

    assert future.snapshot().state == "pending"
    assert backend._begin_admission()
    assert future.snapshot().state == "admitting"
    assert not future.running()
    assert backend._authorize()
    assert future.snapshot().state == "running"
    assert future.running()
    assert backend._publish_result(b"outcome")
    assert future.snapshot().state == "adapting"

    assert future.result() == "adapted"
    assert future.snapshot().state == "succeeded"


def test_core_future_wait_timeout_and_pre_go_cancellation_preserve_later_outcomes(monkeypatch):
    """Wait expiry is local, while pre-GO cancellation remains confirmed through adaptation."""
    future, backend = _facade(monkeypatch)
    with pytest.raises(TimeoutError):
        future.result(timeout=0)
    assert not future.done()
    assert backend._publish_result(b"outcome")
    assert future.result() == "adapted"

    cancelled, cancelled_backend = _facade(monkeypatch, backend=_ControlledBackend("cancelled"))
    assert cancelled.cancel()
    with pytest.raises(BaseException):
        cancelled.result()
    assert cancelled_backend.cancelled()
    assert cancelled.snapshot().state == "cancelled"


def test_core_future_barrier_consumers_join_one_recovery_and_callback(monkeypatch):
    """Concurrent result, exception, await, and callback consumers share one adaptation."""
    entered = Event()
    release = Event()
    callback = Event()
    calls = 0

    def recover(*_, **__):
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(1)
        return "adapted"

    future, backend = _facade(monkeypatch, recover=recover)
    future.add_done_callback(lambda completed: callback.set())
    assert backend._publish_result(b"outcome")
    barrier = Barrier(3)
    observations = []

    def get_result():
        barrier.wait()
        observations.append(future.result())

    def get_exception():
        barrier.wait()
        observations.append(future.exception())

    first = Thread(target=get_result)
    second = Thread(target=get_exception)
    first.start()
    second.start()
    barrier.wait()
    assert entered.wait(1)
    release.set()
    first.join(timeout=1)
    second.join(timeout=1)
    assert not first.is_alive()
    assert not second.is_alive()
    assert sorted(observations, key=str) == [None, "adapted"]
    assert calls == 1
    assert callback.wait(1)

    async def await_completed():
        return await future

    assert asyncio.run(await_completed()) == "adapted"


def test_core_future_preserves_backend_uncertainty_and_core_failure_phase(monkeypatch):
    """Uncertain backends and delivered core phases remain distinct terminal observations."""
    uncertain, backend = _facade(monkeypatch)
    assert backend._publish_uncertain(ExecutionUncertainError("lost"))
    with pytest.raises(ExecutionUncertainError, match="lost"):
        uncertain.result()
    assert uncertain.snapshot().state == "uncertain"
    assert uncertain.snapshot().phase == "backend"

    running, running_backend = _facade(monkeypatch, backend=_ControlledBackend("running"))
    assert running_backend._begin_admission()
    assert running_backend._authorize()
    running_backend._set_cancel_requester(lambda: True)
    assert running.request_cancel()
    assert running_backend._publish_uncertain(ExecutionUncertainError("termination unknown"))
    with pytest.raises(ExecutionUncertainError, match="termination unknown"):
        running.result()
    assert running.snapshot().backend.cancel_requested
    assert running.snapshot().state == "uncertain"

    calls = 0

    def decode(*_, **__):
        nonlocal calls
        calls += 1
        return CoreAdaptationOutcome({"success": True}, CoreOutcomeEvidence((), ()))

    failed_backend = _ControlledBackend("phase")
    failed, _ = _facade(
        monkeypatch,
        backend=failed_backend,
        decode=decode,
        recover=lambda *_, **__: (_ for _ in ()).throw(
            CoreExecutionError("publication failed", phase="publish"),
        ),
    )
    assert failed_backend._publish_result(b"outcome")
    with pytest.raises(CoreExecutionError, match="publication failed"):
        failed.result()
    assert calls == 1
    assert failed.snapshot().phase == "publish"

    startup_backend = _ControlledBackend("startup")
    startup, _ = _facade(
        monkeypatch,
        backend=startup_backend,
        decode=lambda *_, **__: (_ for _ in ()).throw(RuntimeError("decoder unavailable")),
    )
    assert startup_backend._publish_result(b"outcome")
    with pytest.raises(RuntimeError, match="decoder unavailable"):
        startup.result()
    assert startup.snapshot().phase == "recover"


def test_core_future_recovers_when_generic_callback_thread_start_fails(monkeypatch):
    """A queued generic bridge cannot strand core result consumers indefinitely."""
    backend = ExecutionFuture[bytes]("thread-start")
    future, _ = _facade(monkeypatch, backend=backend)
    original_start = __import__("threading").Thread.start
    calls = 0

    def fail_twice(thread):
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise RuntimeError("thread unavailable")
        return original_start(thread)

    monkeypatch.setattr("dryml.execute.future.Thread.start", fail_twice)
    assert backend._publish_result(b"outcome")
    assert future.result(timeout=1) == "adapted"


def test_core_cancelled_awaiter_removes_its_bridge_without_cancelling_backend(monkeypatch):
    """Cancelling one local awaiter releases only its facade callback bridge."""
    future, backend = _facade(monkeypatch)

    async def exercise():
        task = asyncio.create_task(future._await_result())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    assert future._callbacks == []
    assert not backend.cancelled()
    assert not backend.done()


def test_core_adaptation_scopes_the_frozen_recovery_session_without_context_leak(monkeypatch):
    """Recovery callback threads receive only the retained Repo/cache and restore their context."""
    observed = []

    class Scope:
        """Record one scoped frozen recovery configuration."""

        def __enter__(self):
            return None

        def __exit__(self, *args):
            observed.append("restored")

    future, backend = _facade(monkeypatch)
    monkeypatch.setattr(
        execute_module,
        "config",
        lambda *, repo, cache: (observed.append((repo, cache)) or Scope()),
    )
    assert backend._publish_result(b"outcome")
    assert future.result() == "adapted"
    assert observed == [(future._storage.recovery_repo, "weak"), "restored"]


def test_core_cleanup_releases_local_storage_after_generic_uncertainty_with_bounded_issue(monkeypatch):
    """Independent recovery resources close despite remote uncertainty and retain no failure text."""
    released = Event()
    backend = _ControlledBackend("cleanup")
    backend._prepare(diagnostic_text_limit_bytes=3, diagnostic_issue_limit=1)
    backend._set_cleanup_reconciler(lambda timeout: None)
    storage = _Storage(lambda: (released.set(), (_ for _ in ()).throw(RuntimeError("secret")))[1])
    future, _ = _facade(monkeypatch, backend=backend, storage=storage)
    assert backend._publish_result(b"outcome")
    assert future.result() == "adapted"
    backend._record_worker_cleanup_issues(("worker",))

    with pytest.raises(CleanupError):
        future.cleanup(timeout=1)
    assert released.wait(1)
    snapshot = future.snapshot()
    assert snapshot.cleanup_state == "incomplete"
    assert len(snapshot.cleanup_issues) == 1
    assert len(snapshot.cleanup_issues[0].message.encode("utf-8")) <= 3


def test_core_one_off_automatic_cleanup_releases_owned_storage_once(monkeypatch):
    """Automatic one-off cleanup and a later explicit retry do not double-close storage."""
    releases = []
    backend = _ControlledBackend("one-off")
    backend._set_cleanup_reconciler(lambda timeout: None)
    future, _ = _facade(
        monkeypatch,
        backend=backend,
        storage=_Storage(lambda: releases.append("closed")),
        one_off=True,
    )
    assert backend._publish_result(b"outcome")
    assert future.result() == "adapted"
    assert releases == ["closed"]
    future.cleanup(timeout=1)
    assert releases == ["closed"]


def test_core_future_concurrent_result_and_await_join_one_adapted_outcome(tmp_path):
    """All public consumers observe the stored adapted scalar rather than bytes."""
    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    (tmp_path / "spool").mkdir()
    executor = Executor(
        SubProcessConfig(spool_directory=tmp_path / "spool"),
        core=CoreOptions(repo=repo, return_objects=False),
    )
    try:
        future = executor.submit(_value, 9)
        with ThreadPoolExecutor(max_workers=2) as workers:
            first = workers.submit(future.result, 10)
            second = workers.submit(future.result, 10)
            assert first.result() == second.result() == 9
        async def await_future():
            return await future

        assert asyncio.run(await_future()) == 9
        future.cleanup(timeout=5)
        assert future.result() == 9
    finally:
        executor.close(cancel=True, timeout=5)


def test_core_future_cleanup_requires_adaptation_terminality(tmp_path):
    """Core cleanup cannot race a generic byte completion and close recovery early."""
    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    (tmp_path / "spool").mkdir()
    executor = Executor(
        SubProcessConfig(spool_directory=tmp_path / "spool"),
        core=CoreOptions(repo=repo, return_objects=False),
    )
    try:
        future = executor.submit(_value, 3)
        try:
            future.cleanup(timeout=1)
        except RuntimeError:
            pass
        assert future.result(timeout=10) == 3
        future.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=5)


def test_core_future_cleanup_and_executor_close_share_one_snapshot_release(tmp_path, monkeypatch):
    """Concurrent facade cleanup and owner close linearize one recovery-Repo release."""
    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    (tmp_path / "spool").mkdir()
    executor = Executor(
        SubProcessConfig(spool_directory=tmp_path / "spool"),
        core=CoreOptions(repo=repo, return_objects=False),
    )
    calls = []
    try:
        future = executor.submit(_value, 7)
        assert future.result(timeout=10) == 7
        original_close = type(future._storage).close

        def observe_close(storage):
            """Record only this submission's owned snapshot close."""
            if storage is future._storage:
                calls.append(storage)
            return original_close(storage)

        monkeypatch.setattr(type(future._storage), "close", observe_close)
        barrier = Barrier(3)
        failures = []

        def cleanup():
            """Join facade cleanup after all lifecycle contenders are ready."""
            try:
                barrier.wait()
                future.cleanup(timeout=5)
            except BaseException as error:
                failures.append(error)

        def close_owner():
            """Close the owner while other callers reconcile the same future."""
            try:
                barrier.wait()
                executor.close(cancel=True, timeout=5)
            except BaseException as error:
                failures.append(error)

        first = Thread(target=cleanup)
        second = Thread(target=cleanup)
        owner = Thread(target=close_owner)
        first.start()
        second.start()
        owner.start()
        first.join(timeout=10)
        second.join(timeout=10)
        owner.join(timeout=10)
        assert not first.is_alive()
        assert not second.is_alive()
        assert not owner.is_alive()
        assert failures == []
        assert calls == [future._storage]
    finally:
        executor.close(cancel=True, timeout=5)
