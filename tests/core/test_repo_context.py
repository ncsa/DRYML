from pathlib import Path

import pytest

from dryml.core import Object, Repo, Serializable
import dryml.core.repo as repo_module
from dryml.core.repo import get_default_repo, manage_repo
from dryml.core.store.dir import DirStore
from dryml.core.store.zip import ZipStore


class ContextAware(Object):
    def __init__(self):
        super().__init__()
        self.repo = get_default_repo()


class OwnedSaveValue(Serializable):
    """Stateful value used to prove explicit save-path ownership."""

    def __init__(self, value="owned"):
        self.value = value


def test_manage_repo_none_is_context_local_and_temporary(monkeypatch):
    assert get_default_repo() is None
    closed = []
    original_close = Repo.close
    monkeypatch.setattr(Repo, "close", lambda self, *args, **kwargs: closed.append(self))

    with manage_repo(None) as repo:
        assert get_default_repo() is repo
        assert ContextAware().repo is repo

    assert closed == [repo]
    assert get_default_repo() is None
    monkeypatch.setattr(Repo, "close", original_close)


def test_repo_closes_opened_handles_once_but_never_borrowed_handles(tmp_path, monkeypatch):
    """Repo owns specification coercions while direct Store handles stay borrowed."""

    root = tmp_path / "owned"
    DirStore(root, query_index="none")
    closed = []
    original_close = DirStore.close

    def observe_close(self):
        closed.append(self)
        return original_close(self)

    monkeypatch.setattr(DirStore, "close", observe_close)
    owned = Repo(root)
    owned_handle = owned.stores[0]
    owned.close(flush=False)
    owned.close(flush=False)

    borrowed_handle = DirStore(tmp_path / "borrowed", query_index="none")
    borrowed = Repo(borrowed_handle)
    borrowed.close(flush=False)

    assert closed.count(owned_handle) == 1
    assert borrowed_handle not in closed


def test_failed_owned_repo_close_keeps_buffered_handle_open_for_retry(tmp_path, monkeypatch):
    """A flush failure does not discard an owned ZipStore's inspectable buffer."""

    repo = Repo(tmp_path / "owned.zip")
    store = repo.stores[0]
    store._archive_dirty = True
    monkeypatch.setattr(store, "commit", lambda: (_ for _ in ()).throw(OSError("commit failed")))

    try:
        with pytest.raises(OSError, match="commit failed"):
            repo.close()
        assert not repo._closed
        assert store._archive_dirty
    finally:
        repo.close(flush=False)


def test_failed_commit_retains_every_owned_buffer_for_retry(tmp_path, monkeypatch):
    """A failed flush leaves all Repo-owned buffered archive handles inspectable."""

    repo = Repo([tmp_path / "first.zip", tmp_path / "second.zip"])
    first, second = repo.stores
    first._archive_dirty = True
    second._archive_dirty = True
    monkeypatch.setattr(first, "commit", lambda: (_ for _ in ()).throw(OSError("commit failed")))

    try:
        with pytest.raises(OSError, match="commit failed"):
            repo.close()
        assert not repo._closed
        assert repo._owned_stores == [first, second]
        assert first._archive_dirty
        assert second._archive_dirty
    finally:
        repo.close(flush=False)


def test_store_registration_returns_none_and_checks_close_before_opening(monkeypatch, tmp_path):
    """Public registration remains void and never opens a closed Repo spec."""

    repo = Repo()
    store = DirStore(tmp_path / "borrowed", query_index="none")
    assert repo.add_store(store) is None
    assert repo.set_default_store(store) is None
    repo.close(flush=False)
    calls = []
    monkeypatch.setattr(
        repo_module, "make_store", lambda spec: calls.append(spec) or ZipStore(spec),
    )

    with pytest.raises(RuntimeError, match="close begins"):
        repo.add_store(tmp_path / "must-not-open.zip")

    assert calls == []


def test_topology_lease_rejects_new_store_spec_before_coercion(tmp_path):
    """A managed topology lease cannot initialize a rejected Store specification."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    absent = tmp_path / "must-not-open.zip"

    with repo.retain_topology():
        with pytest.raises(RuntimeError, match="topology"):
            repo.add_store(absent)

    assert not absent.exists()


def test_owned_registration_failure_rolls_back_and_retains_failed_cleanup(tmp_path, monkeypatch):
    """A failed binding refresh neither publishes nor leaks a fresh Store handle."""

    borrowed = DirStore(tmp_path / "borrowed", query_index="none")
    repo = Repo(borrowed)
    opened = []
    original_make_store = repo_module.make_store

    def observe_open(spec):
        store = original_make_store(spec)
        opened.append(store)
        return store

    monkeypatch.setattr(repo_module, "make_store", observe_open)
    monkeypatch.setattr(
        repo._query_index, "refresh_bindings",
        lambda: (_ for _ in ()).throw(OSError("refresh failed")),
    )
    with pytest.raises(OSError, match="refresh failed"):
        repo.add_store(tmp_path / "owned.zip")

    owned = opened[0]
    assert repo.stores == [borrowed]
    assert owned not in repo._owned_stores
    assert not Path(owned.base_dir).exists()

    original_close = ZipStore.close
    monkeypatch.setattr(
        ZipStore, "close",
        lambda store: (_ for _ in ()).throw(OSError("cleanup failed"))
        if store.archive_path == str(tmp_path / "retained.zip") else original_close(store),
    )
    with pytest.raises(OSError, match="refresh failed"):
        repo.add_store(tmp_path / "retained.zip")
    retained = opened[1]
    with pytest.raises(OSError, match="cleanup failed"):
        repo.close(flush=False)
    assert not repo._closed
    assert retained in repo._owned_stores
    monkeypatch.setattr(ZipStore, "close", original_close)
    repo.close(flush=False)


def test_constructor_and_explicit_save_path_own_only_opened_handles(tmp_path, monkeypatch):
    """Partial construction unwinds and explicit save paths remain Repo-owned."""

    opened = []
    original_make_store = repo_module.make_store

    def observe_open(spec):
        store = original_make_store(spec)
        opened.append(store)
        return store

    monkeypatch.setattr(repo_module, "make_store", observe_open)
    with pytest.raises(ValueError):
        Repo([tmp_path / "partial.zip", object()])
    assert not Path(opened[0].base_dir).exists()

    repo = Repo()
    repo.save(OwnedSaveValue(repo=repo), store=tmp_path / "explicit.zip")
    owned = repo.default_store
    assert owned in repo._owned_stores
    repo.close(flush=False)
    assert not Path(owned.base_dir).exists()
