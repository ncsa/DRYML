"""Public reusable core Execute facade coverage."""

from __future__ import annotations

from threading import Event
from contextlib import contextmanager

import pytest

import dryml.core.execute as execute_module
from dryml.core import Executor, Repo
from dryml.core.execute import (
    CoreOptions,
    _capture_frozen_core_controls,
    _prepare_frozen_core_submission,
    _submit_frozen_core_submission,
)
from dryml.core.session import config as config_module
from dryml.core.store.dir import DirStore
from dryml.execute.errors import CleanupError
from dryml.execute.subprocess import SubProcessConfig
from dryml.environments import CurrentEnvironmentSpec
from dryml.runtime import RuntimeMode


def _add(value, *, offset=1):
    """Return a transportable scalar used by subprocess facade tests."""
    return value + offset


def _keyword_collision(*, core, output):
    """Prove view keywords remain workload data rather than facade controls."""
    return f"{core}:{output}"


def _worker_cache_policy():
    """Return the cache policy actually installed in the worker session."""
    from dryml.core.session import current_cache

    return current_cache()


def test_core_executor_view_constructor_preserves_legacy_shapes_and_optional_selector():  # noqa: E501
    """Core views retain prior positional and keyword construction forms."""
    from dryml.core.execute import ExecutorView

    parent = object()
    legacy_positional = ExecutorView(
        parent, None, None, None, None, None, (), None,
    )
    legacy_keyword = ExecutorView(
        executor=parent,
        core=None,
        environment=None,
        world=None,
        execution_timeout=None,
        stream_output=None,
        done_callbacks=(),
        output=None,
    )
    selector = CurrentEnvironmentSpec()
    selected = ExecutorView(
        executor=parent,
        core=None,
        environment=None,
        world=None,
        execution_timeout=None,
        stream_output=None,
        done_callbacks=(),
        output=None,
        environment_spec=selector,
    )

    assert legacy_positional.environment_spec is None
    assert legacy_keyword.environment_spec is None
    assert selected.environment_spec is selector


def test_core_executor_returns_recovered_value_and_core_callbacks(tmp_path):
    """One generic byte completion adapts before the core callback observes it."""
    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    (tmp_path / "spool").mkdir()
    executor = Executor(
        SubProcessConfig(spool_directory=tmp_path / "spool"),
        core=CoreOptions(repo=repo, return_objects=False),
    )
    called = Event()
    observed = []
    try:
        future = executor.submit(
            _add, 4, done_callbacks=(lambda completed: (observed.append(completed.result()), called.set()),),
        )
        assert future.result(timeout=10) == 5
        assert isinstance(future.backend_future.result(timeout=1), bytes)
        assert called.wait(5)
        assert observed == [5]
        assert future.snapshot().state == "succeeded"
        future.cleanup(timeout=5)
        cache_future = executor.submit(_worker_cache_policy, core=CoreOptions(cache="strong"))
        assert cache_future.result(timeout=10) == "strong"
        cache_future.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=5)


def test_core_executor_view_keeps_control_named_keywords_as_workload_data(tmp_path):
    """A core view owns no second backend and forwards colliding call keywords."""
    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    (tmp_path / "spool").mkdir()
    executor = Executor(
        SubProcessConfig(spool_directory=tmp_path / "spool"),
        core=CoreOptions(repo=repo, return_objects=False),
    )
    try:
        view = executor.with_options()
        future = view.submit(_keyword_collision, core="workload", output="also-workload")
        assert future.result(timeout=10) == "workload:also-workload"
        future.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=5)


def test_core_executor_forwards_exact_selector_without_a_software_requirement(
        tmp_path):
    """
    Core Execute reuses generic exact selection rather than adding a second
    path.
    """
    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    executor = Executor(
        SubProcessConfig(spool_directory=tmp_path),
        core=CoreOptions(repo=repo, return_objects=False),
    )
    try:
        future = executor.submit(_add,
                                 4,
                                 environment_spec=CurrentEnvironmentSpec())
        assert future.result(timeout=10) == 5
        future.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=5)


def test_core_executor_accepts_a_detached_repo_definition_and_closes_only_its_handles(tmp_path, monkeypatch):
    """Detached Repo authority reconstructs once for preparation and recovery."""
    store = DirStore(tmp_path / "store", query_index="none")
    definition = Repo(store).to_definition()
    closed = []
    original_close = DirStore.close

    def observe_close(self):
        """Record coordinator-owned reconstructed handles without changing close behavior."""
        closed.append(self)
        return original_close(self)

    monkeypatch.setattr(DirStore, "close", observe_close)
    spool = tmp_path / "spool"
    spool.mkdir()
    executor = Executor(
        SubProcessConfig(spool_directory=spool),
        core=CoreOptions(repo=definition, return_objects=False),
    )
    try:
        future = executor.submit(_add, 4)
        assert future.result(timeout=10) == 5
        future.cleanup(timeout=5)
        assert store not in closed
        assert closed
    finally:
        executor.close(cancel=True, timeout=5)


def test_core_executor_rejects_invalid_callbacks_before_store_preparation(tmp_path, monkeypatch):
    """Invalid core controls fail before a Repo export or generic acceptance."""
    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    executor = Executor(
        SubProcessConfig(spool_directory=tmp_path), core=CoreOptions(repo=repo),
    )
    monkeypatch.setattr(repo, "to_definition", lambda: (_ for _ in ()).throw(AssertionError("must not export")))
    try:
        with pytest.raises(TypeError, match="callbacks"):
            executor.submit(_add, 1, done_callbacks=(object(),))
    finally:
        executor.close(cancel=True, timeout=5)


def test_frozen_core_preparation_runs_validator_before_opening_borrowed_storage(  # noqa: E501
        tmp_path, monkeypatch):
    """
    A rejected preflight never exports or opens the ambient borrowed Repo.
    """

    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    backend_config = SubProcessConfig(spool_directory=tmp_path)
    monkeypatch.setattr(
        repo, "to_definition", lambda:
        (_ for _ in ()).throw(AssertionError("must not export")))

    with pytest.raises(RuntimeError, match="drift"):
        _prepare_frozen_core_submission(
            backend_config,
            None,
            _add,
            (1, ),
            kwargs=None,
            core=CoreOptions(repo=repo),
            validators=(lambda:
                        (_ for _ in ()).throw(RuntimeError("drift")), ),
        )


def test_frozen_core_preparation_rejects_a_closed_borrowed_repo(tmp_path):
    """
    A frozen call never substitutes a closed ambient Repo with another one.
    """

    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    repo.close(flush=False)

    with pytest.raises(ValueError,
                       match="shared Store authority is unavailable"):
        _prepare_frozen_core_submission(
            SubProcessConfig(spool_directory=tmp_path), None, _add, (1,),
            kwargs=None, core=CoreOptions(repo=repo),
        )


def test_frozen_controls_capture_ambient_core_before_delayed_preparation(
        tmp_path, monkeypatch):
    """
    Delayed preparation uses entry Repo/cache controls and opens no Store at
    capture.
    """

    entry_repo = Repo(DirStore(tmp_path / "entry", query_index="none"))
    later_repo = Repo(DirStore(tmp_path / "later", query_index="none"))
    backend_config = SubProcessConfig(spool_directory=tmp_path)
    monkeypatch.setattr(
        entry_repo,
        "to_definition",
        lambda:
        (_ for _ in ()).throw(AssertionError("capture must not export")),
    )
    with config_module(repo=entry_repo, cache="strong"):
        controls = _capture_frozen_core_controls(None, executor_core=None)
    monkeypatch.undo()
    monkeypatch.setattr(
        later_repo,
        "to_definition",
        lambda:
        (_ for _ in ()).throw(AssertionError("must not use later Repo")),
    )

    with config_module(repo=later_repo, cache="none"):
        frozen = _prepare_frozen_core_submission(
            backend_config, None, _add, (1,), kwargs=None, core=None,
            frozen_controls=controls,
        )
    try:
        assert frozen.effective.repo is entry_repo
        assert frozen.effective.cache == "strong"
    finally:
        frozen.close()


def test_invalid_submit_callbacks_close_unaccepted_frozen_storage(
        tmp_path, monkeypatch):
    """
    Validation after preparation releases only the owned recovery
    reconstruction.
    """

    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    frozen = _prepare_frozen_core_submission(
        SubProcessConfig(spool_directory=tmp_path),
        None,
        _add,
        (1, ),
        kwargs=None,
        core=CoreOptions(repo=repo),
    )
    closed = []
    original_close = type(frozen.storage).close

    def observe_close(storage):
        closed.append(True)
        return original_close(storage)

    monkeypatch.setattr(type(frozen.storage), "close", observe_close)
    with pytest.raises(TypeError, match="callbacks"):
        _submit_frozen_core_submission(
            lambda *args, **kwargs: pytest.fail(
                "invalid callbacks must not submit"),
            SubProcessConfig(spool_directory=tmp_path),
            frozen,
            environment=None,
            environment_spec=None,
            world=None,
            execution_timeout=None,
            stream_output=None,
            done_callbacks=(object(), ),
            output=None,
            one_off=False,
        )

    assert closed == [True]


def test_failed_preparation_closes_its_owned_recovery_repo(
        tmp_path, monkeypatch):
    """
    Serialization failure does not leak the discardable preparation
    reconstruction.
    """

    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    closed = []
    original_close = execute_module._PreparedSharedStorage.close

    def observe_close(storage):
        closed.append(True)
        return original_close(storage)

    monkeypatch.setattr(execute_module._PreparedSharedStorage, "close",
                        observe_close)
    monkeypatch.setattr(
        execute_module.SharedDirStoreStrategy,
        "prepare",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            execute_module.CoreCallCodecError("serialization failed"),
        ),
    )

    with pytest.raises(execute_module.CoreCallCodecError,
                       match="serialization failed"):
        _prepare_frozen_core_submission(
            SubProcessConfig(spool_directory=tmp_path), None, _add, (1,),
            kwargs=None, core=CoreOptions(repo=repo),
        )

    assert closed == [True]


def test_failed_preacceptance_cleanup_retains_the_frozen_recovery_handle(
        tmp_path, monkeypatch):
    """
    A cleanup failure exposes the still-retryable frozen owner rather than
    losing it.
    """

    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    frozen = _prepare_frozen_core_submission(
        SubProcessConfig(spool_directory=tmp_path),
        None,
        _add,
        (1, ),
        kwargs=None,
        core=CoreOptions(repo=repo),
    )
    monkeypatch.setattr(
        type(frozen.storage), "close",
        lambda storage: (_ for _ in ()).throw(RuntimeError("close failed")),
    )

    with pytest.raises(CleanupError) as failure:
        _submit_frozen_core_submission(
            lambda *args, **kwargs: pytest.fail(
                "invalid callbacks must not submit"),
            SubProcessConfig(spool_directory=tmp_path),
            frozen,
            environment=None,
            environment_spec=None,
            world=None,
            execution_timeout=None,
            stream_output=None,
            done_callbacks=(object(), ),
            output=None,
            one_off=False,
        )

    assert failure.value.execution is frozen
    monkeypatch.undo()
    frozen.close()


def test_frozen_controls_keep_the_entry_orchestration_floor_after_relaxation(
        tmp_path, monkeypatch):
    """
    A later relaxed runtime cannot restore live result materialization
    authority.
    """

    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    mode = [RuntimeMode.ORCHESTRATOR]
    monkeypatch.setattr(
        execute_module,
        "active_runtime",
        lambda: type("Runtime", (), {"mode": mode[0]})(),
    )
    with config_module(repo=repo):
        controls = _capture_frozen_core_controls(
            CoreOptions(return_objects="auto"), executor_core=None,
        )
    mode[0] = RuntimeMode.NONE

    frozen = _prepare_frozen_core_submission(
        SubProcessConfig(spool_directory=tmp_path),
        None,
        _add,
        (1, ),
        kwargs=None,
        core=None,
        frozen_controls=controls,
    )
    try:
        assert not frozen.return_objects
    finally:
        frozen.close()


def test_frozen_controls_reject_a_stricter_floor_before_storage_preparation(
        tmp_path, monkeypatch):
    """
    A runtime that becomes stricter during preflight rejects before Repo
    export.
    """

    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    mode = [RuntimeMode.NONE]
    monkeypatch.setattr(
        execute_module,
        "active_runtime",
        lambda: type("Runtime", (), {"mode": mode[0]})(),
    )
    with config_module(repo=repo):
        controls = _capture_frozen_core_controls(
            CoreOptions(return_objects=True), executor_core=None,
        )
    mode[0] = RuntimeMode.ORCHESTRATOR
    monkeypatch.setattr(
        repo,
        "to_definition",
        lambda: (_ for _ in ()).throw(AssertionError("must not export")),
    )

    with pytest.raises(ValueError, match="orchestration"):
        _prepare_frozen_core_submission(
            SubProcessConfig(spool_directory=tmp_path), None, _add, (1,),
            kwargs=None, core=None, frozen_controls=controls,
        )


def test_acceptance_holds_the_lease_and_rechecks_a_closed_borrowed_repo(
        tmp_path, monkeypatch):
    """
    The final storage fence and backend acceptance share one publication lease.
    """

    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    frozen = _prepare_frozen_core_submission(
        SubProcessConfig(spool_directory=tmp_path),
        None,
        _add,
        (1, ),
        kwargs=None,
        core=CoreOptions(repo=repo),
    )
    repo.close(flush=False)
    submitted = []

    with pytest.raises(ValueError,
                       match="shared Store authority is unavailable"):
        _submit_frozen_core_submission(
            lambda *args, **kwargs: submitted.append(True),
            SubProcessConfig(spool_directory=tmp_path), frozen,
            environment=None, environment_spec=None, world=None,
            execution_timeout=None, stream_output=None, done_callbacks=(),
            output=None, one_off=False,
        )

    assert submitted == []
    assert frozen.storage._closed

    repo = Repo(DirStore(tmp_path / "accepted", query_index="none"))
    frozen = _prepare_frozen_core_submission(
        SubProcessConfig(spool_directory=tmp_path),
        None,
        _add,
        (1, ),
        kwargs=None,
        core=CoreOptions(repo=repo),
    )
    held = []

    @contextmanager
    def lease():
        held.append("entered")
        try:
            yield None
        finally:
            held.append("released")

    monkeypatch.setattr(execute_module.publication, "lease", lease)
    monkeypatch.setattr(execute_module, "CoreExecutionFuture",
                        lambda *args, **kwargs: object())

    def submitter(*args, **kwargs):
        assert held == ["entered"]
        return object()

    _submit_frozen_core_submission(
        submitter,
        SubProcessConfig(spool_directory=tmp_path),
        frozen,
        environment=None,
        environment_spec=None,
        world=None,
        execution_timeout=None,
        stream_output=None,
        done_callbacks=(),
        output=None,
        one_off=False,
    )

    assert held == ["entered", "released"]
    with pytest.raises(RuntimeError, match="already been consumed"):
        _submit_frozen_core_submission(
            submitter, SubProcessConfig(spool_directory=tmp_path), frozen,
            environment=None, environment_spec=None, world=None,
            execution_timeout=None, stream_output=None, done_callbacks=(),
            output=None, one_off=False,
        )
    frozen.storage.close()
    repo.close(flush=False)
