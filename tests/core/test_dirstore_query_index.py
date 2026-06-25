import os
from pathlib import Path
import pytest

from dryml.core2 import Definition, Object, Repo, SKIP_ARGS
from dryml.core2.query.model import QueryIndexUnavailable
from dryml.core2.query.sqlite import SQLiteQueryIndexConfig, sqlite_available
from dryml.core2.query.sqlite.index import SQLiteStoreQueryIndex
from dryml.core2.store.dir import DirStore


class QueryIndexDirLeaf(Object):
    def __init__(self, name="leaf"):
        super().__init__()
        self.name = name


def test_dirstore_query_index_default_is_auto_and_lazy(tmp_path):
    store = DirStore(tmp_path / "store")

    assert store.query_index_policy == "auto"
    assert Path(store.object_root_dir).exists()
    assert not Path(store.dryml_dir).exists()
    assert store.query_index_path == os.fspath(tmp_path / "store" / ".dryml" / "query-index-v1.sqlite")
    assert store.query_index_dirty_path == os.fspath(tmp_path / "store" / ".dryml" / "query-index.dirty")


def test_dirstore_memory_and_none_do_not_open_query_index(tmp_path):
    memory_store = DirStore(tmp_path / "memory", query_index="memory")
    none_store = DirStore(tmp_path / "none", query_index="none")

    assert memory_store.open_query_index() is None
    assert none_store.open_query_index() is None
    assert not Path(memory_store.dryml_dir).exists()
    assert not Path(none_store.dryml_dir).exists()


def test_dirstore_rejects_unknown_query_index_policy(tmp_path):
    with pytest.raises(ValueError):
        DirStore(tmp_path / "store", query_index="bad-policy")


def test_dirstore_custom_query_index_factory_is_lazy(tmp_path):
    calls = []

    def factory(store):
        calls.append(store.catalog_key())
        return {"source": store.catalog_key()}

    store = DirStore(tmp_path / "store", query_index=factory)

    assert calls == []
    opened = store.open_query_index()
    assert opened == {"source": store.catalog_key()}
    assert calls == [store.catalog_key()]


def test_dirstore_sqlite_policy_opens_skeleton_without_creating_file(tmp_path):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    index = store.open_query_index()

    assert isinstance(index, SQLiteStoreQueryIndex)
    assert index.source_key == store.catalog_key()
    assert index.path == Path(store.query_index_path)
    assert not Path(store.query_index_path).exists()
    assert index.status().state == "missing"


def test_dirstore_auto_uses_sqlite_when_available_without_construction_io(tmp_path):
    store = DirStore(tmp_path / "store", query_index="auto")
    index = store.open_query_index()

    if sqlite_available():
        assert isinstance(index, SQLiteStoreQueryIndex)
        assert index.status().state == "missing"
        assert not Path(store.query_index_path).exists()
    else:
        assert index is None


def test_dirstore_sqlite_config_uses_default_sidecar_path(tmp_path):
    config = SQLiteQueryIndexConfig(journal_mode="delete", busy_timeout=1.5)
    store = DirStore(tmp_path / "store", query_index=config)
    index = store.open_query_index()

    assert isinstance(index, SQLiteStoreQueryIndex)
    assert Path(index.config.path) == Path(store.query_index_path)
    assert index.config.journal_mode == "delete"
    assert index.config.busy_timeout == 1.5


def test_dirstore_sqlite_config_respects_explicit_path(tmp_path):
    explicit = tmp_path / "custom.sqlite"
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(path=explicit, journal_mode="delete"))
    index = store.open_query_index()

    assert isinstance(index, SQLiteStoreQueryIndex)
    assert index.path == explicit
    assert index.config.path == explicit


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_dirstore_sqlite_index_can_initialize_empty_sidecar(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    index = store.open_query_index()

    index.initialize_empty()
    status = index.status()

    assert Path(store.query_index_path).exists()
    assert status.state == "ready"
    assert status.generation == 0
    assert status.store_key == store.catalog_key()


def test_sqlite_policy_unavailable_error_message(monkeypatch, tmp_path):
    import dryml.core2.store.dir as dir_module

    monkeypatch.setattr(dir_module, "sqlite_available", lambda: False)
    store = DirStore(tmp_path / "store", query_index="sqlite")

    with pytest.raises(QueryIndexUnavailable, match="query_index='sqlite'"):
        store.open_query_index()


def test_none_policy_exact_lookup_does_not_hydrate_store(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = QueryIndexDirLeaf("exact", repo=repo)
    repo.save_object(obj)

    reopened_store = DirStore(store.base_dir, query_index="none")
    repo2 = Repo(stores=reopened_store)

    def fail_hydrate():
        raise AssertionError("query_index='none' should not auto-hydrate for exact lookup")

    reopened_store.hydrate_index = fail_hydrate

    assert repo2.query(obj.definition).stored().count() == 1


def test_none_policy_broad_query_does_not_auto_scan(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = QueryIndexDirLeaf("broad", repo=repo)
    repo.save_object(obj)

    reopened_store = DirStore(store.base_dir, query_index="none")
    repo2 = Repo(stores=reopened_store)

    def fail_hydrate():
        raise AssertionError("query_index='none' should not auto-hydrate broad queries")

    reopened_store.hydrate_index = fail_hydrate

    selector = Definition(QueryIndexDirLeaf, SKIP_ARGS)

    assert repo2.query(selector).stored().count() == 0


def test_none_policy_refresh_true_is_explicit_scan(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = QueryIndexDirLeaf("explicit", repo=repo)
    repo.save_object(obj)

    repo2 = Repo(stores=DirStore(store.base_dir, query_index="none"))
    selector = Definition(QueryIndexDirLeaf, SKIP_ARGS)

    assert repo2.query(selector).stored(refresh=True).count() == 1


def test_sqlite_policy_broad_query_uses_index_without_memory_scan(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = QueryIndexDirLeaf("sqlite", repo=repo)
    repo.save_object(obj)

    reopened_store = DirStore(store.base_dir, query_index="sqlite")
    repo2 = Repo(stores=reopened_store)

    selector = Definition(QueryIndexDirLeaf, SKIP_ARGS)

    assert repo2.query(selector).stored().count() == 1
    assert Path(reopened_store.query_index_path).exists()


def test_sqlite_missing_index_builds_once_from_store_roots(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = QueryIndexDirLeaf("build", repo=repo)
    repo.save_object(obj)

    reopened_store = DirStore(store.base_dir, query_index="sqlite")
    calls = []
    original = reopened_store.hydrate_index

    def hydrate_once():
        calls.append(True)
        yield from original()

    reopened_store.hydrate_index = hydrate_once
    repo2 = Repo(stores=reopened_store)
    selector = Definition(QueryIndexDirLeaf, SKIP_ARGS)

    assert repo2.query(selector).stored().count() == 1
    assert len(calls) == 1
    assert repo2.query(selector).stored().count() == 1
    assert len(calls) == 1


def test_sqlite_dirty_index_rebuilds_and_clears_marker(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = QueryIndexDirLeaf("dirty", repo=repo)
    repo.save_object(obj)
    store.mark_query_index_dirty()

    reopened_store = DirStore(store.base_dir, query_index="sqlite")
    repo2 = Repo(stores=reopened_store)
    selector = Definition(QueryIndexDirLeaf, SKIP_ARGS)

    assert reopened_store.query_index_is_dirty()
    assert repo2.index_status(store=reopened_store)[0].state == "dirty"
    assert repo2.query(selector).stored().count() == 1
    assert not reopened_store.query_index_is_dirty()
    assert repo2.index_status(store=reopened_store)[0].state == "ready"


def test_repo_rebuild_index_rebuilds_sqlite_sidecar(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    obj = QueryIndexDirLeaf("rebuild", repo=repo)
    repo.save_object(obj)
    index = store.open_query_index()
    index.remove_stored_roots([obj.definition])
    selector = Definition(QueryIndexDirLeaf, SKIP_ARGS)

    assert repo.query(selector).stored().count() == 0
    repo.rebuild_index(store=store)
    assert repo.query(selector).stored().count() == 1
