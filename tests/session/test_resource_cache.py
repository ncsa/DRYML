"""Standalone ownership and context contracts for Session resource caching."""

import asyncio
import contextvars
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys

import pytest

import dryml
from dryml.core import session as core_session
from dryml import session
from dryml.core import Repo
from dryml.core.store.dir import DirStore
from dryml.core.store.zip import ZipStore


@pytest.fixture(autouse=True)
def reset_core_session():
    """Keep the process-local core selection independent between cache tests."""

    dryml.reset_config()
    try:
        yield
    finally:
        dryml.reset_config()


def test_resource_cache_activation_nesting_and_cleared_snapshots():
    """Activation is opt-in, nests for one owner, and clears on final exit."""

    assert session.current_resource_cache() is None

    with session.resource_cache() as outer:
        assert session.current_resource_cache() is outer
        assert outer.repos == ()
        assert outer.stores == ()
        with session.resource_cache() as inner:
            assert inner is outer
            assert session.current_resource_cache() is outer
        assert session.current_resource_cache() is outer

    assert session.current_resource_cache() is None
    assert outer.repos == ()
    assert outer.stores == ()


def test_resource_cache_borrows_current_repo_and_store_without_closing(tmp_path):
    """Selected resources are inspectable borrowed entries and survive teardown."""

    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(store)
    dryml.configure(repo=repo)

    with session.resource_cache() as cache:
        assert cache.repos == (repo,)
        assert cache.stores == (store,)
        with pytest.raises(RuntimeError, match="resource cache"):
            repo.close(flush=False)
        with pytest.raises(RuntimeError, match="resource cache"):
            store.close()

    assert cache.repos == ()
    assert cache.stores == ()
    repo.close(flush=False)
    store.close()


def test_resource_cache_refuses_raw_zipstore_close_until_outer_exit(tmp_path):
    """The raw close guard applies to buffered as well as direct Store handles."""

    store = ZipStore(tmp_path / "store.zip")
    repo = Repo(store)
    dryml.configure(repo=repo)

    with session.resource_cache():
        with pytest.raises(RuntimeError, match="resource cache"):
            store.close()

    store.close()
    repo.close(flush=False)


def test_resource_cache_rejects_copied_task_and_thread_contexts():
    """Inherited ContextVars cannot use a cache outside its owner task/thread."""

    with session.resource_cache() as cache:
        context = contextvars.copy_context()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(context.run, session.current_resource_cache)
            with pytest.raises(RuntimeError, match="owner"):
                future.result()

        async def copied_task():
            with pytest.raises(RuntimeError, match="owner"):
                session.current_resource_cache()

        asyncio.run(copied_task())
        assert session.current_resource_cache() is cache


def test_resource_cache_uses_independent_noninherited_contexts():
    """A fresh thread may enter a distinct cache without inheriting the caller's."""

    with session.resource_cache() as outer:
        def enter_independent_cache():
            assert session.current_resource_cache() is None
            with session.resource_cache() as inner:
                return inner is outer

        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(enter_independent_cache).result() is False


def test_inactive_copied_context_rejects_lookup():
    with session.resource_cache():
        copied = contextvars.copy_context()
    with pytest.raises(RuntimeError, match="inactive"):
        copied.run(session.current_resource_cache)


@pytest.mark.parametrize("scoped", [False, True])
def test_failed_selection_admission_preserves_original(tmp_path, monkeypatch, scoped):
    original = Repo(DirStore(tmp_path / "original", query_index="none"))
    replacement = Repo(DirStore(tmp_path / "replacement", query_index="none"))
    dryml.configure(repo=original)
    with session.resource_cache() as cache:
        def reject(repo):
            raise RuntimeError("admission rejected")
        monkeypatch.setattr(cache, "_admit_borrowed_repo", reject)
        with pytest.raises(RuntimeError, match="admission rejected"):
            if scoped:
                with dryml.config(repo=replacement):
                    pytest.fail("rejected context entered")
            else:
                dryml.configure(repo=replacement)
        assert core_session.current_repo() is original
    original.close(flush=False)
    replacement.close(flush=False)


def test_resource_cache_activation_stays_standalone_and_backend_lazy():
    """Activation does not initialize execution, managed, or optional backends."""

    script = """
import sys
from dryml import session
with session.resource_cache() as cache:
    assert cache.repos == ()
assert session.current_resource_cache() is None
assert not {'dryml.execute', 'dryml.managed', 'tensorflow', 'torch', 'jax'} & set(sys.modules)
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_cache_tracks_borrowed_selection_changes_without_releasing_old_entries(tmp_path):
    """Changing a borrowed selection retains both cache entries until final exit."""

    first = Repo(DirStore(tmp_path / "first", query_index="none"))
    second = Repo(DirStore(tmp_path / "second", query_index="none"))
    dryml.configure(repo=first)

    with session.resource_cache() as cache:
        dryml.configure(repo=second)
        assert cache.repos == (first, second)
        assert cache.stores == (first.default_store, second.default_store)
        assert dryml.status()["repo"] is second

    first.close(flush=False)
    second.close(flush=False)


def test_close_producing_core_session_transitions_fail_before_selection_changes(tmp_path, monkeypatch):
    """A leased owned selection cannot be replaced or scoped unsafely."""

    original = Repo(DirStore(tmp_path / "original", query_index="none"))
    replacement = Repo(DirStore(tmp_path / "replacement", query_index="none"))
    owned_store = DirStore(tmp_path / "owned", query_index="none")
    dryml.configure(repo=original)

    with session.resource_cache() as cache:
        dryml.configure(repo=owned_store)
        owned = dryml.status()["repo"]
        assert owned in cache.repos

        with pytest.raises(RuntimeError, match="resource cache"):
            dryml.configure(repo=replacement)
        assert dryml.status()["repo"] is owned

        with monkeypatch.context() as context:
            context.setattr(
                core_session,
                "_coerce_repo",
                lambda value: pytest.fail("unsafe temporary selection was coerced"),
            )
            with pytest.raises(RuntimeError, match="cannot safely restore"):
                with dryml.config(repo=object()):
                    raise AssertionError("unsafe temporary selection entered")
        assert dryml.status()["repo"] is owned

    dryml.configure(repo=replacement)
    owned.close(flush=False)
    original.close(flush=False)
    replacement.close(flush=False)
