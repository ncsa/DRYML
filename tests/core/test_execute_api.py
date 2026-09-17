"""Public reusable core Execute facade coverage."""

from __future__ import annotations

from threading import Event

import pytest

from dryml.core import Executor, Repo
from dryml.core.execute import CoreOptions
from dryml.core.store.dir import DirStore
from dryml.execute.subprocess import SubProcessConfig


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
