from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from types import SimpleNamespace
from typing import Any

import pytest

from dryml.execute._spooling import deserialize_call
from dryml.execute.backend import Backend
from dryml.execute.config import BackendConfig
from dryml.execute.errors import ExecutionError
from dryml.execute.future import ExecutionFuture


class FakeFuture(ExecutionFuture[Any]):
    """Concrete common Future used by the descriptor-only fake backend."""


class FakeBackend(Backend):
    """Execute payload descriptors synchronously after the executor authorizes them."""

    def __init__(self) -> None:
        self.started = 0
        self.calls = []
        self.cleaned: list[str] = []
        self.closed: list[tuple[bool, float | None]] = []
        self.closed_event = Event()
        self.future_calls = 0
        self.run_gate: Event | None = None
        self.start_gate: Event | None = None
        self.start_entered = Event()
        self.known_ids: set[str] = set()

    def start(self) -> None:
        self.started += 1
        self.start_entered.set()
        if self.start_gate is not None:
            assert self.start_gate.wait(2)

    def capabilities(self) -> frozenset[str]:
        return frozenset()

    def create_future(self, submission_id: str, output) -> FakeFuture:
        self.future_calls += 1
        return FakeFuture(submission_id, output=output)

    def submit(self, call, *, future) -> None:
        from dryml.execute.models import PayloadSpool

        assert future.submission_id == call.submission_id
        assert future.output is call.output
        assert isinstance(call.payload, PayloadSpool)
        self.calls.append((call, future))
        self.known_ids.add(call.submission_id)
        assert future._begin_admission()
        if not future._authorize(deadline=call.admission_deadline):
            return
        if self.run_gate is not None:
            assert self.run_gate.wait(2)
        fn, args, kwargs = deserialize_call(
            call.payload.path.read_bytes(), limit_bytes=10_000
        )
        try:
            future._publish_result(fn(*args, **kwargs))
        except BaseException as exc:
            future._publish_exception(exc)

    def discover(self, *, environment=None, world=None, timeout: float):
        raise AssertionError("U4 owns discovery")

    def resources(self, *, timeout: float):
        raise AssertionError("U4 owns resource inspection")

    def reconcile_cleanup(self, submission_id: str, *, timeout: float) -> None:
        if submission_id not in self.known_ids:
            raise AssertionError("unknown cleanup ID")
        self.cleaned.append(submission_id)

    def close(self, *, cancel: bool, timeout: float | None) -> None:
        self.closed.append((cancel, timeout))
        self.closed_event.set()


@dataclass(frozen=True, kw_only=True)
class FakeConfig(BackendConfig):
    """Keep one inert fake backend associated with an executor configuration."""

    backend: FakeBackend = field(repr=False, compare=False)

    def create_backend(self) -> Backend:
        return self.backend


def config(tmp_path, backend: FakeBackend, **overrides: object) -> FakeConfig:
    """Return a small valid configuration with a caller-owned spool parent."""
    values = dict(
        backend=backend,
        spool_directory=tmp_path,
        spool_limit_bytes=100_000,
        invocation_limit_bytes=10_000,
        result_limit_bytes=10_000,
        spool_file_limit=8,
        preflight_limit=2,
    )
    values.update(overrides)
    return FakeConfig(**values)


def test_executor_accepts_one_precreated_concrete_future_and_runs_descriptor(tmp_path):
    """Acceptance keeps the backend's concrete Future identity through dispatch."""
    from dryml.execute.executor import Executor

    backend = FakeBackend()
    executor = Executor(config(tmp_path, backend))
    future = executor.submit(lambda value: value + 1, 2)

    assert isinstance(future, FakeFuture)
    assert future.result(timeout=2) == 3
    assert backend.future_calls == 1
    assert backend.calls[0][1] is future
    assert not hasattr(backend.calls[0][0], "fn")
    executor.close()


def test_invalid_controls_reject_before_quota_serialization_or_future_factory(tmp_path):
    """Malformed callbacks do not create a spool, bind output, or call a backend factory."""
    from dryml.execute._spooling import SpoolBudget
    from dryml.execute.executor import Executor
    from dryml.execute.output import ExecutionOutput

    backend = FakeBackend()
    output = ExecutionOutput()
    executor = Executor(config(tmp_path, backend))
    with pytest.raises(TypeError, match="callable"):
        executor.submit(lambda: 1, done_callbacks=(object(),), output=output)

    assert backend.future_calls == 0
    assert SpoolBudget.snapshot().leases == 0
    assert output._bound_submission_id is None
    executor.close()


@pytest.mark.parametrize("keyword", ("environment", "world"))
def test_invalid_requirement_controls_reject_before_quota_or_factory(tmp_path, keyword):
    """Requirement controls are typed before any preflight side effect."""
    from dryml.execute._spooling import SpoolBudget
    from dryml.execute.executor import Executor

    backend = FakeBackend()
    executor = Executor(config(tmp_path, backend))
    with pytest.raises(TypeError, match=keyword):
        executor.submit(lambda: 1, **{keyword: {"not": "a requirement"}})
    with pytest.raises(TypeError, match=keyword):
        executor.with_options(**{keyword: {"not": "a requirement"}})
    assert backend.future_calls == 0
    assert SpoolBudget.snapshot().leases == 0


def test_non_inert_factory_future_is_rejected_before_acceptance(tmp_path):
    """A factory cannot return a completed, running, or base Future instance."""
    from dryml.execute.executor import Executor

    class CompletedBackend(FakeBackend):
        def create_future(self, submission_id, output):
            future = FakeFuture(submission_id, output=output)
            future._publish_result(1)
            return future

    backend = CompletedBackend()
    executor = Executor(config(tmp_path, backend))
    with pytest.raises(ExecutionError, match="inert"):
        executor.submit(lambda: 1)
    assert not backend.calls
    executor.close()


def test_delayed_start_expires_without_backend_submission_or_unknown_cleanup(tmp_path):
    """Admission expiry wins while the single backend start remains blocked."""
    from dryml.execute.errors import AdmissionError
    from dryml.execute.executor import Executor

    backend = FakeBackend()
    backend.start_gate = Event()
    executor = Executor(config(tmp_path, backend, admission_timeout=0.05))
    future = executor.submit(lambda: 1)
    assert backend.start_entered.wait(1)
    with pytest.raises(AdmissionError):
        future.result(timeout=1)
    future.cleanup(timeout=1)
    assert not backend.calls
    assert not backend.cleaned
    backend.start_gate.set()
    executor.close(timeout=1)


def test_callback_list_is_snapshotted_before_fast_completion(tmp_path):
    """Initial callback registrations preserve order despite caller-list mutation."""
    from dryml.execute.executor import Executor

    backend = FakeBackend()
    callbacks: list[object] = []
    delivered: list[int] = []
    done = Event()

    def first(future: ExecutionFuture[int]) -> None:
        delivered.append(future.result())

    def second(future: ExecutionFuture[int]) -> None:
        delivered.append(future.result())
        done.set()

    callbacks.extend((first, second))
    executor = Executor(config(tmp_path, backend))
    future = executor.submit(lambda: 4, done_callbacks=callbacks)  # type: ignore[arg-type]
    callbacks.clear()
    assert future.result(timeout=2) == 4
    assert done.wait(2)
    assert delivered == [4, 4]
    executor.close()


def test_backend_replacement_or_legacy_future_is_rejected_before_dispatch(tmp_path):
    """A backend cannot replace the prepared concrete Future or its output identity."""
    from dryml.execute.executor import Executor

    class ReplacingBackend(FakeBackend):
        def create_future(self, submission_id: str, output):
            return FakeFuture("another-submission", output=output)

    backend = ReplacingBackend()
    executor = Executor(config(tmp_path, backend))
    with pytest.raises(ExecutionError, match="another submission"):
        executor.submit(lambda: 1)
    assert not backend.calls
    executor.close()


def test_deadline_inheritance_and_stable_relative_spool_parent(tmp_path, monkeypatch):
    """Direct controls resolve per call while a relative spool parent stays executor-stable."""
    from dryml.execute.executor import Executor

    original = tmp_path / "original"
    changed = tmp_path / "changed"
    spool = original / "spool"
    spool.mkdir(parents=True)
    changed.mkdir()
    backend = FakeBackend()
    monkeypatch.chdir(original)
    executor = Executor(config(Path("spool"), backend, execution_timeout=0.2))
    monkeypatch.chdir(changed)
    inherited = executor.submit(lambda: 1)
    disabled = executor.with_options(execution_timeout=None).submit(lambda: 2)
    overridden = executor.submit(lambda: 3, execution_timeout=0.1)
    assert inherited.result(timeout=2) == 1
    assert disabled.result(timeout=2) == 2
    assert overridden.result(timeout=2) == 3
    assert [call.execution_timeout for call, _ in backend.calls] == [0.2, None, 0.1]
    assert all(call.payload.path.parent.parent == spool for call, _ in backend.calls)
    assert executor._config.execution_timeout == 0.2
    executor.close()


def test_factory_uses_captured_working_directory_and_specialized_interpreter(tmp_path, monkeypatch):
    """Lazy factory creation receives normalized controls captured at executor construction."""
    from dryml.execute.executor import Executor

    @dataclass(frozen=True, kw_only=True)
    class CapturingConfig(BackendConfig):
        backend: FakeBackend = field(repr=False, compare=False)
        python_executable: Path | None = None

        def create_backend(self) -> Backend:
            self.backend.factory_config = self
            return self.backend

    original = tmp_path / "original"
    changed = tmp_path / "changed"
    original.mkdir()
    spool = original / "spool"
    spool.mkdir()
    changed.mkdir()
    backend = FakeBackend()
    monkeypatch.chdir(original)
    monkeypatch.setenv("TMPDIR", "spool")
    supplied = CapturingConfig(
        backend=backend,
        spool_directory=None,
        spool_limit_bytes=100_000,
        invocation_limit_bytes=10_000,
        result_limit_bytes=10_000,
        spool_file_limit=8,
        preflight_limit=2,
    )
    executor = Executor(supplied)
    monkeypatch.chdir(changed)
    assert executor.submit(lambda: 1).result(timeout=2) == 1
    assert backend.factory_config is not supplied
    assert backend.factory_config.working_directory == original
    assert backend.factory_config.python_executable == Path(sys.executable)
    assert backend.calls[0][0].payload.path.parent.parent == spool
    executor.close()


@pytest.mark.parametrize(
    ("platform", "environment"),
    (
        ("nt", {"TMPDIR": "spool", "TEMP": "fallback-temp", "TMP": "fallback-tmp"}),
        ("posix", {"TMPDIR": "spool", "TEMP": "ignored-temp"}),
    ),
)
def test_capture_temp_parent_uses_isolated_platform_environment(tmp_path, monkeypatch, platform, environment):
    """Temporary-parent policy honors TMPDIR without mutating the ambient environment."""
    from dryml.execute import executor as executor_module
    from dryml.execute.executor import Executor

    captured = tmp_path / "captured"
    platform_os = SimpleNamespace(**{**vars(os), "name": platform, "environ": environment})
    monkeypatch.setattr(executor_module, "os", platform_os)

    assert Executor._capture_temp_parent(captured) == captured / "spool"


def test_waiters_observe_their_own_failed_backend_start_generation(tmp_path):
    """A retry cannot turn waiters on a failed start generation into later success."""
    from dryml.execute.executor import Executor

    class FailsFirstStart(FakeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.release_failure = Event()

        def start(self) -> None:
            self.started += 1
            self.start_entered.set()
            if self.started == 1:
                assert self.release_failure.wait(1)
                raise RuntimeError("first start failed")

    backend = FailsFirstStart()
    executor = Executor(config(tmp_path, backend))
    first = executor.submit(lambda: 1)
    second = executor.submit(lambda: 2)
    assert backend.start_entered.wait(1)
    backend.release_failure.set()
    with pytest.raises(RuntimeError, match="first start failed"):
        first.result(timeout=1)
    with pytest.raises(RuntimeError, match="first start failed"):
        second.result(timeout=1)
    third = executor.submit(lambda: 3)
    assert third.result(timeout=1) == 3
    executor.close()


def test_waiters_observe_their_own_failed_backend_factory_generation(tmp_path, monkeypatch):
    """A later factory retry cannot revive submitters that joined a failed attempt."""
    from dryml.execute.executor import Executor

    @dataclass(frozen=True, kw_only=True)
    class RetryFactoryConfig(BackendConfig):
        backend: FakeBackend = field(repr=False, compare=False)
        entered: Event = field(repr=False, compare=False)
        release: Event = field(repr=False, compare=False)

        def create_backend(self) -> Backend:
            self.entered.set()
            assert self.release.wait(1)
            attempts = getattr(self.backend, "factory_attempts", 0) + 1
            self.backend.factory_attempts = attempts
            if attempts == 1:
                raise RuntimeError("first factory failed")
            return self.backend

    backend = FakeBackend()
    entered = Event()
    release = Event()
    supplied = RetryFactoryConfig(
        backend=backend,
        entered=entered,
        release=release,
        spool_directory=tmp_path,
        spool_limit_bytes=100_000,
        invocation_limit_bytes=10_000,
        result_limit_bytes=10_000,
        spool_file_limit=8,
        preflight_limit=2,
    )
    executor = Executor(supplied)
    original_get_backend = Executor._get_backend
    entered_get_backend = 0
    entered_lock = Lock()
    joined = Event()

    def count_waiters(self, deadline, *, accepted=False):
        nonlocal entered_get_backend
        with entered_lock:
            entered_get_backend += 1
            if entered_get_backend == 2:
                joined.set()
        return original_get_backend(self, deadline, accepted=accepted)

    monkeypatch.setattr(Executor, "_get_backend", count_waiters)
    failures: list[BaseException] = []
    first = Thread(target=lambda: _capture_failure(failures, lambda: executor.submit(lambda: 1)))
    second = Thread(target=lambda: _capture_failure(failures, lambda: executor.submit(lambda: 2)))
    first.start()
    assert entered.wait(1)
    second.start()
    assert joined.wait(1)
    release.set()
    first.join(1)
    second.join(1)
    assert len(failures) == 2
    assert all(isinstance(failure, RuntimeError) and str(failure) == "first factory failed" for failure in failures)
    assert executor._backend_results == {}
    assert executor._backend_result_waiters == {}
    assert executor.submit(lambda: 3).result(timeout=1) == 3
    executor.close()


def test_failed_backend_generations_do_not_retain_results_or_tracebacks(tmp_path):
    """Completed factory/start failures are retained only for their joined waiters."""
    from dryml.execute.executor import Executor

    @dataclass(frozen=True, kw_only=True)
    class FailingFactoryConfig(BackendConfig):
        def create_backend(self) -> Backend:
            raise RuntimeError("factory failure")

    factory = Executor(FailingFactoryConfig(spool_directory=tmp_path))
    for _ in range(8):
        with pytest.raises(RuntimeError, match="factory failure"):
            factory.start()
        assert factory._backend_results == {}
        assert factory._backend_result_waiters == {}
    factory.close()

    class FailingStartBackend(FakeBackend):
        def start(self) -> None:
            raise RuntimeError("start failure")

    starter = Executor(config(tmp_path / "start", FailingStartBackend()))
    for _ in range(8):
        with pytest.raises(RuntimeError, match="start failure"):
            starter.start()
        assert starter._backend_start_results == {}
        assert starter._backend_start_result_waiters == {}
    starter.close()


def _capture_failure(failures: list[BaseException], call) -> None:
    """Capture one concurrent submit outcome for factory-generation assertions."""
    try:
        call()
    except BaseException as exc:
        failures.append(exc)
