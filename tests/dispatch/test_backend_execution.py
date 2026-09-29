"""Backend-hosted Dispatch execution coverage."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path

import pytest

import dryml.dispatch as dispatch
import dryml.core.execute as core_execute
from dryml.core import ObjectRef, Repo, Serializable, StateRef, function
from dryml.core.execute import CoreExecutionFuture, CoreOptions
from dryml.core.store.dir import DirStore
from dryml.environments import CurrentEnvironmentSpec
from dryml.environments import req as environment_req
from dryml.execute.backend import Backend
from dryml.execute.config import BackendConfig
from dryml.execute.errors import BackendUnavailableError, CleanupError
from dryml.execute.subprocess import SubProcessConfig


def _add(value: int, *, env: str) -> tuple[int, str]:
    """Return ordinary workload data, including a control-named keyword."""

    return value + 1, env


class _StatefulResult(Serializable):
    """Persist one scalar so Dispatch can exercise core reference recovery."""

    def __init__(self, value: int = 0) -> None:
        """Record the value persisted by the state hooks."""

        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec) -> None:
        """Persist the scalar state through the selected core codec."""

        Path(dest_dir, "value").write_text(str(self.value), encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec) -> None:
        """Restore the scalar state published by the worker."""

        self.value = int(Path(src_dir, "value").read_text(encoding="ascii"))


def _aliased_results(value: _StatefulResult):
    """
    Return automatic and explicit reference values with one shared result.
    """

    result = _StatefulResult(value.value + 1)
    return result, result, value.object_ref


@function
def _function_target(value: int) -> int:
    """Supply one importable core-owned function wrapper target."""

    return value + 2


@wraps(_function_target)
def _copied_function_target(value: int) -> int:
    """
    Forward through ordinary copied metadata without changing invocation owner.
    """

    return _function_target(value)


class _RebindableCallable:
    """
    Provide a callable-instance root whose raw descriptor can be replaced.
    """

    def __call__(self) -> None:
        """Fail if backend preparation ever invokes the original workload."""

        raise AssertionError("workload must not run during preparation")


class _UnavailableBackend(Backend):
    """
    Backend fixture that proves Dispatch never tries a second registered route.
    """

    def __init__(self) -> None:
        """Initialize the observable startup counter."""

        self.started = 0

    def start(self) -> None:
        """Reject startup for the exact selected fake backend."""

        self.started += 1
        raise BackendUnavailableError("selected backend is unavailable")

    def capabilities(self) -> frozenset[str]:
        """Expose no workload capability because startup always fails."""

        return frozenset()

    def create_future(self, submission_id, output):
        """
        Reject future creation because unavailable discovery must stop first.
        """

        raise AssertionError("unavailable backend must not create a future")

    def submit(self, call, *, future) -> None:
        """
        Reject workload submission because unavailable discovery must stop
        first.
        """

        raise AssertionError("unavailable backend must not submit")

    def discover(self,
                 *,
                 environment=None,
                 environment_spec=None,
                 world=None,
                 timeout):
        """Reject discovery after the selected backend's failed startup."""

        raise AssertionError("unavailable backend must not discover")

    def resources(self, *, timeout):
        """Reject unrelated resource observation."""

        raise AssertionError("unavailable backend must not inspect resources")

    def reconcile_cleanup(self, submission_id, *, timeout) -> None:
        """Reject submission cleanup because no workload was accepted."""

        raise AssertionError("unavailable backend must not reconcile work")

    def close(self, *, cancel, timeout) -> None:
        """Accept discovery-owner closure after failed startup."""


@dataclass(frozen=True, kw_only=True)
class _UnavailableConfig(BackendConfig):
    """Bind one fake backend to an inert explicit Dispatch configuration."""

    backend: _UnavailableBackend = field(repr=False, compare=False)

    def create_backend(self) -> Backend:
        """Return the exact configured fake without selecting alternatives."""

        return self.backend


@pytest.fixture(autouse=True)
def _clear_dispatch_state() -> None:
    """Keep each backend submission independent of process defaults."""

    dispatch._state._reset_for_testing()
    yield
    dispatch._state._reset_for_testing()


def test_dispatch_submit_and_run_use_the_existing_core_one_off_owner(
        tmp_path) -> None:
    """
    Return CoreExecutionFuture and recovered results through a real subprocess.
    """

    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    spool = tmp_path / "spool"
    spool.mkdir()
    dispatch.set_execute_backend_default(
        SubProcessConfig(spool_directory=spool),
        core=CoreOptions(repo=repo, return_objects=False),
    )

    future = dispatch.submit(_add, 4, env="workload-data")

    assert isinstance(future, CoreExecutionFuture)
    assert future.result(timeout=10) == (5, "workload-data")
    future.cleanup(timeout=5)
    assert dispatch.run(_add, 7, env="still-workload-data") == (
        8,
        "still-workload-data",
    )


def test_dispatch_preserves_core_reference_and_alias_recovery(
        tmp_path) -> None:
    """
    Keep StateRef/ObjectRef and result aliases under the existing core policy.
    """

    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    spool = tmp_path / "spool"
    spool.mkdir()
    dispatch.set_execute_backend_default(
        SubProcessConfig(spool_directory=spool),
        core=CoreOptions(repo=repo, return_objects=False),
    )
    value = _StatefulResult(4, repo=repo)
    repo.save(value, deep_capture=True)

    first, second, original = dispatch.run(_aliased_results, value)

    assert isinstance(first, StateRef)
    assert first is second
    assert isinstance(original, ObjectRef)
    assert original == value.object_ref


def test_dispatch_forwards_one_frozen_exact_selector_without_reresolution(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Resolve the worker pin at entry and give core that exact selection object.
    """

    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    spool = tmp_path / "spool"
    spool.mkdir()
    dispatch.set_execute_backend_default(
        SubProcessConfig(spool_directory=spool),
        core=CoreOptions(repo=repo, return_objects=False),
    )
    resolved = []
    resolve = dispatch._preflight.resolve_environment_spec

    def observe(spec):
        """Record the only Dispatch-level selector resolution."""

        selection = resolve(spec)
        resolved.append(selection)
        return selection

    monkeypatch.setattr(dispatch._preflight, "resolve_environment_spec",
                        observe)

    assert dispatch.with_options(python=CurrentEnvironmentSpec()).run(
        _add, 1, env="data"
    ) == (2, "data")
    assert len(resolved) == 1


def test_unavailable_default_never_falls_back_to_a_registered_backend(
) -> None:
    """
    Stop at the configured default without Ray or another execution attempt.
    """

    unavailable = _UnavailableBackend()
    viable = _UnavailableBackend()
    dispatch.register_backend("viable", _UnavailableConfig(backend=viable))
    dispatch.set_execute_backend_default(
        _UnavailableConfig(backend=unavailable))

    with pytest.raises(dispatch.DispatchError) as raised:
        dispatch.run(lambda: None)

    assert ("dispatch.backend_discovery_unavailable"
            in raised.value.report.diagnostics)
    assert unavailable.started == 1
    assert viable.started == 0


def test_target_drift_during_core_preparation_reclaims_owned_resources(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Turn an observed post-serialization guard mismatch into DispatchError.
    """

    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    spool = tmp_path / "spool"
    spool.mkdir()
    dispatch.set_execute_backend_default(
        SubProcessConfig(spool_directory=spool),
        core=CoreOptions(repo=repo, return_objects=False),
    )
    closed = []
    original_prepare = core_execute.SharedDirStoreStrategy.prepare
    original_close = core_execute._PreparedSharedStorage.close

    def workload() -> None:
        """Provide a captured root whose declarations drift during encoding."""

    def mutate_after_prepare(*args, **kwargs):
        """Change one attached declaration after core owns recovery storage."""

        prepared = original_prepare(*args, **kwargs)
        environment_req(tags=("drift",))(workload)
        return prepared

    def observe_close(storage):
        """Record cleanup of the submission-owned reconstruction."""

        closed.append(storage)
        return original_close(storage)

    monkeypatch.setattr(
        core_execute.SharedDirStoreStrategy, "prepare", mutate_after_prepare
    )
    monkeypatch.setattr(core_execute._PreparedSharedStorage, "close",
                        observe_close)

    with pytest.raises(dispatch.DispatchError) as raised:
        dispatch.submit(workload)

    assert "dispatch.target_changed" in raised.value.report.diagnostics
    assert len(closed) == 1


def test_callable_instance_rebinding_rejects_during_backend_preparation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject rebinding before backend acceptance or workload invocation."""

    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    spool = tmp_path / "spool"
    spool.mkdir()
    dispatch.set_execute_backend_default(
        SubProcessConfig(spool_directory=spool),
        core=CoreOptions(repo=repo, return_objects=False),
    )
    original_prepare = core_execute.SharedDirStoreStrategy.prepare
    original_call = _RebindableCallable.__dict__["__call__"]

    def replacement(self) -> None:
        """
        Fail if final validation incorrectly accepts the rebound workload.
        """

        del self
        raise AssertionError("rebound workload must not run")

    def rebind_after_prepare(*args, **kwargs):
        """Replace the descriptor after core owns preparation state."""

        prepared = original_prepare(*args, **kwargs)
        _RebindableCallable.__call__ = replacement
        return prepared

    monkeypatch.setattr(
        core_execute.SharedDirStoreStrategy, "prepare", rebind_after_prepare
    )
    try:
        with pytest.raises(dispatch.DispatchError) as raised:
            dispatch.submit(_RebindableCallable())
    finally:
        _RebindableCallable.__call__ = original_call

    assert "dispatch.target_changed" in raised.value.report.diagnostics


def test_closed_borrowed_repo_rejects_final_acceptance_without_substitution(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Leave a closed caller Repo as a core ownership failure, not a new route.
    """

    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    spool = tmp_path / "spool"
    spool.mkdir()
    dispatch.set_execute_backend_default(
        SubProcessConfig(spool_directory=spool),
        core=CoreOptions(repo=repo, return_objects=False),
    )
    prepare = dispatch.api._prepare_frozen_core_submission

    def close_after_prepare(*args, **kwargs):
        """
        Close only the borrowed source authority after preparation succeeds.
        """

        frozen = prepare(*args, **kwargs)
        repo.close(flush=False)
        return frozen

    monkeypatch.setattr(
        dispatch.api, "_prepare_frozen_core_submission", close_after_prepare
    )

    with pytest.raises(ValueError,
                       match="shared Store authority is unavailable"):
        dispatch.submit(_add, 1, env="data")


def test_dispatch_preserves_supported_wrapper_ownership(tmp_path) -> None:
    """Keep copied wrapper metadata separate from core's invocation owner."""

    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    spool = tmp_path / "spool"
    spool.mkdir()
    dispatch.set_execute_backend_default(
        SubProcessConfig(spool_directory=spool),
        core=CoreOptions(repo=repo, return_objects=False),
    )
    assert dispatch.run(_copied_function_target, 5) == 7


def test_dispatch_run_keeps_execution_failure_primary_when_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Mirror core one-off failure ordering instead of changing recovery
    semantics.
    """

    dispatch.set_execute_backend_default(SubProcessConfig())
    primary = RuntimeError("workload failed")
    cleanup = CleanupError("cleanup failed")

    class _Future:
        """Minimal core-future-shaped failure fixture."""

        def result(self):
            """Raise the configured primary execution failure."""

            raise primary

        def cleanup(self):
            """Raise the established cleanup error after result failure."""

            raise cleanup

    monkeypatch.setattr(dispatch.api, "_submit_backend",
                        lambda *_args: _Future())

    with pytest.raises(RuntimeError) as raised:
        dispatch.run(lambda: None)

    assert raised.value is primary
    assert raised.value.__cause__ is cleanup
