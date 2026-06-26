import os
from pathlib import Path
import pytest

from dryml.core2 import Definition, Object, Repo, SKIP_ARGS
from dryml.core2.query.model import QueryIndexUnavailable
from dryml.core2.query.sqlite import SQLiteQueryIndexConfig, require_sqlite, sqlite_available
from dryml.core2.query.sqlite.index import SQLiteStoreQueryIndex
from dryml.core2.store.dir import DirStore


class QueryIndexDirLeaf(Object):
    def __init__(self, name="leaf"):
        super().__init__()
        self.name = name


class FailingSaveDirLeaf(Object):
    def __init__(self, name="fail"):
        super().__init__()
        self.name = name

    def save_state_to_dir_imp(self, dest_dir, revision=None):
        raise RuntimeError("object save failed")


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
    store = DirStore(tmp_path / "store", query_index="memory")
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


def test_sqlite_exact_query_probes_store_without_full_rebuild(tmp_path):
    if not sqlite_available():
        pytest.skip("sqlite3 is unavailable")
    store = DirStore(tmp_path / "store", query_index="memory")
    repo = Repo(stores=store)
    obj = QueryIndexDirLeaf("exact-probe", repo=repo)
    repo.save_object(obj)

    reopened_store = DirStore(store.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    calls = []

    def fail_hydrate():
        calls.append(True)
        raise AssertionError("exact SQLite fallback should not hydrate the Store")

    reopened_store.hydrate_index = fail_hydrate
    repo2 = Repo(stores=reopened_store)

    results = repo2.query(obj.definition).stored().defs()

    assert list(results) == [obj.definition]
    assert calls == []
    status = repo2.index_status(store=reopened_store)[0]
    assert status.generation == 1
    assert status.row_counts["stored_roots"] == 1

    assert list(repo2.query(obj.definition).stored().defs()) == [obj.definition]
    assert calls == []
    assert repo2.index_status(store=reopened_store)[0].generation == 1


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


def test_dirstore_query_index_status_rebuild_and_reconcile(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    obj = QueryIndexDirLeaf("store-admin", repo=repo)
    repo.save_object(obj)

    status = store.query_index_status()
    assert status.backend == "sqlite"
    assert status.state == "ready"
    assert status.row_counts["stored_roots"] == 1

    validate = store.reconcile_query_index()
    assert validate.changed is False
    assert validate.action == "validate"
    assert validate.generation_before == validate.generation_after == 1

    index = store.open_query_index()
    index.remove_stored_roots([obj.definition])
    assert repo.query(Definition(QueryIndexDirLeaf, SKIP_ARGS)).stored().count() == 0

    rebuilt = store.rebuild_query_index()
    assert rebuilt.changed
    assert rebuilt.action == "rebuild"
    assert rebuilt.generation_after == 1
    assert store.query_index_status().row_counts["stored_roots"] == 1


def test_dirstore_reconcile_repairs_dirty_index(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    obj = QueryIndexDirLeaf("dirty-admin", repo=repo)
    repo.save_object(obj)
    store.mark_query_index_dirty()

    assert store.query_index_status().state == "dirty"

    report = store.reconcile_query_index()

    assert report.changed
    assert report.action == "rebuild"
    assert not store.query_index_is_dirty()
    assert store.query_index_status().state == "ready"


def test_failed_object_save_does_not_update_query_index(tmp_path):
    opened = []

    class RecordingIndex:
        def register_stored_roots(self, graph, roots):
            raise AssertionError("failed object save should not register roots")

    def factory(store):
        index = RecordingIndex()
        opened.append(index)
        return index

    store = DirStore(tmp_path / "store", query_index=factory)
    repo = Repo(stores=store)
    obj = FailingSaveDirLeaf("failed", repo=repo)

    with pytest.raises(RuntimeError, match="object save failed"):
        repo.save_object(obj)

    assert opened == []


def test_object_files_publish_before_index_root_activation(tmp_path):
    checks = []

    class AssertingIndex:
        def __init__(self, store):
            self.store = store

        def register_stored_roots(self, graph, roots):
            roots = tuple(roots)
            checks.append(tuple(self.store.has(root) for root in roots))

    store = DirStore(tmp_path / "store", query_index=lambda store: AssertingIndex(store))
    repo = Repo(stores=store)
    obj = QueryIndexDirLeaf("ordering", repo=repo)

    repo.save_object(obj)

    assert checks == [(True,)]


def test_sqlite_stored_root_metadata_records_def_file_stat(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    obj = QueryIndexDirLeaf("metadata", repo=repo)
    repo.save_object(obj)

    sqlite3 = require_sqlite()
    con = sqlite3.connect(store.query_index_path)
    row = con.execute("SELECT relative_def_path, def_size, def_mtime_ns FROM stored_roots").fetchone()
    con.close()

    def_path = Path(store.base_dir) / row[0]
    stat = def_path.stat()
    assert row[1] == stat.st_size
    assert row[2] == stat.st_mtime_ns


def test_repo_validate_index_reports_ready_and_dirty(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    obj = QueryIndexDirLeaf("validate", repo=repo)
    repo.save_object(obj)

    report = repo.validate_index(store=store, thorough=True)[0]
    assert report.ok
    assert report.row_counts["stored_roots"] == 1
    assert report.diagnostics["build_state"] == "ready"
    status = repo.index_status(store=store)[0]
    assert "wal_runtime_known_safe" in status.diagnostics

    store.mark_query_index_dirty()
    dirty = repo.validate_index(store=store)[0]
    assert not dirty.ok
    assert dirty.issues[0].message == "SQLite query index is dirty."


def test_repo_validate_index_reports_stored_root_hash_mismatch(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    obj = QueryIndexDirLeaf("hash-mismatch", repo=repo)
    repo.save_object(obj)

    sqlite3 = require_sqlite()
    con = sqlite3.connect(store.query_index_path)
    con.execute("UPDATE stored_roots SET storage_hash = ?", (bytes([0]) * 32,))
    con.commit()
    con.close()

    report = repo.validate_index(store=store, thorough=True)[0]

    assert not report.ok
    assert any(issue.message == "Stored root storage hash mismatch." for issue in report.issues)


def test_repo_validate_index_reports_missing_root_file(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    obj = QueryIndexDirLeaf("missing-file", repo=repo)
    repo.save_object(obj)

    def_path = Path(store.base_dir) / "objects" / obj.definition.stable_hash()[:2] / obj.definition.stable_hash() / "def.pkl"
    def_path.unlink()

    report = repo.validate_index(store=store, thorough=True)[0]

    assert not report.ok
    assert any(issue.message == "Stored root def.pkl is missing." for issue in report.issues)


def test_auto_policy_uses_sqlite_when_available(tmp_path):
    store = DirStore(tmp_path / "store", query_index="auto")
    repo = Repo(stores=store)
    obj = QueryIndexDirLeaf("auto", repo=repo)
    repo.save_object(obj)

    selector = Definition(QueryIndexDirLeaf, SKIP_ARGS)
    assert repo.query(selector).stored().count() == 1
    if sqlite_available():
        assert Path(store.query_index_path).exists()
        assert repo.index_status(store=store)[0].backend == "sqlite"
    else:
        assert not Path(store.query_index_path).exists()


def test_mixed_memory_and_sqlite_query_merges_sources(tmp_path):
    sqlite_store = DirStore(tmp_path / "sqlite", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    memory_store = DirStore(tmp_path / "memory", query_index="memory")
    repo = Repo(stores=[sqlite_store, memory_store])
    sqlite_obj = QueryIndexDirLeaf("sqlite", repo=repo)
    memory_obj = QueryIndexDirLeaf("memory", repo=repo)
    repo.save_object(sqlite_obj, store=sqlite_store)
    repo.save_object(memory_obj, store=memory_store)

    repo2 = Repo(stores=[
        DirStore(sqlite_store.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete")),
        DirStore(memory_store.base_dir, query_index="memory"),
    ])
    selector = Definition(QueryIndexDirLeaf, SKIP_ARGS)

    results = repo2.query(selector).stored().defs()

    assert set(results) == {sqlite_obj.definition, memory_obj.definition}
    replica_dirs = {
        cdef: tuple(store.base_dir for store in results.replicas(cdef))
        for cdef in results
    }
    assert replica_dirs[sqlite_obj.definition] == (sqlite_store.base_dir,)
    assert replica_dirs[memory_obj.definition] == (memory_store.base_dir,)
