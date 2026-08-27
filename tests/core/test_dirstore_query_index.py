import hashlib
import os
from pathlib import Path
import shutil
import threading
import time
import pytest

from dryml.core import Definition, Missing, Object, Ref, Repo, Selector, SKIP_ARGS
from dryml.core.bound_args import BoundArguments
from dryml.core.cdef_graph import ConcreteDefinitionGraph
from dryml.core.cdef_identity import V2_IDENTITY_VERSION
from dryml.core.definition import ConcreteDefinition
from dryml.core.freeze import FrozenDict, FrozenTuple
from dryml.core.query.model import QueryIndexIncompatible, QueryIndexUnavailable, ValidationIssue, ValidationReport
from dryml.core.query.sqlite import SQLiteQueryIndexConfig, require_sqlite, sqlite_available
import dryml.core.query.sqlite.index as sqlite_index_module
from dryml.core.query.sqlite.index import SQLiteStoreQueryIndex
from dryml.core.store.dir import DirStore
from dryml.core.symbol import ImportRef
from dryml.core.utils.general import pickle_save


class QueryIndexDirLeaf(Object):
    def __init__(self, name="leaf"):
        super().__init__()
        self.name = name


class QueryIndexCfgOwner(Object):
    def __init__(self, name="owner", cfg=None, ref=None):
        super().__init__()
        self.name = name
        self.cfg = {} if cfg is None else cfg
        self.ref = ref


class QueryIndexChainNode(Object):
    def __init__(self, name, child=None, ref=None, width=None):
        super().__init__()
        self.name = name
        self.child = child
        self.ref = ref
        self.width = width


class FailingSaveDirLeaf(Object):
    def __init__(self, name="fail"):
        super().__init__()
        self.name = name

    def save_state_to_dir_imp(self, dest_dir, revision=None):
        raise RuntimeError("object save failed")


def _authority_digest(store) -> str:
    digest = hashlib.sha256()
    root = Path(store.base_dir)
    for path in sorted(root.joinpath("objects").rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


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


def test_sqlite_stored_query_missing_does_not_require_presence(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    root_missing = QueryIndexCfgOwner("root", repo=repo)
    nested_missing = QueryIndexCfgOwner("nested", cfg={}, repo=repo)
    ref_child = QueryIndexCfgOwner("child", repo=repo)
    ref_parent = QueryIndexCfgOwner("parent", ref=ref_child.definition.ref(), repo=repo)

    for obj in (root_missing, nested_missing, ref_parent):
        repo.save_object(obj)

    root_selector = Definition(QueryIndexCfgOwner, "root", missing=Missing())
    nested_selector = Definition(QueryIndexCfgOwner, "nested", cfg={"x": Missing()})
    ref_selector = Definition(
        QueryIndexCfgOwner,
        "parent",
        ref=Ref(Selector(Definition(QueryIndexCfgOwner, "child", missing=Missing()))),
    )

    assert list(repo.query(root_selector).stored().defs()) == [root_missing.definition]
    assert list(repo.query(nested_selector).stored().defs()) == [nested_missing.definition]
    assert list(repo.query(ref_selector).stored().defs()) == [ref_parent.definition]


@pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")
def test_sqlite_stored_query_can_inspect_ref_target_subgraph(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    d = Definition(QueryIndexChainNode, "D", width=512)
    c = Definition(QueryIndexChainNode, "C", ref=d.ref())
    b = Definition(QueryIndexChainNode, "B", child=c.mat())
    a = Definition(QueryIndexChainNode, "A", ref=b.ref()).build(repo=repo)
    repo.save_object(a)

    selector = Definition(
        QueryIndexChainNode,
        "A",
        ref=Ref(Selector(Definition(
            QueryIndexChainNode,
            "B",
            child=Definition(
                QueryIndexChainNode,
                "C",
                ref=Definition(QueryIndexChainNode, "D", width=512).ref(),
            ),
        ))),
    )

    assert list(repo.query(selector).stored().defs()) == [a.definition]


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
    import dryml.core.store.dir as dir_module

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


def test_ready_sqlite_indexes_are_not_opened_during_repo_construction(tmp_path):
    store1 = DirStore(tmp_path / "store1", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    store2 = DirStore(tmp_path / "store2", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=[store1, store2])
    repo.save_object(QueryIndexDirLeaf("one", repo=repo), store=store1)
    repo.save_object(QueryIndexDirLeaf("two", repo=repo), store=store2)

    reopened1 = DirStore(store1.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    reopened2 = DirStore(store2.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    open_calls = []

    def wrap_open(source):
        original = source.open_query_index

        def opened():
            open_calls.append(source.catalog_key())
            return original()

        return opened

    reopened1.open_query_index = wrap_open(reopened1)
    reopened2.open_query_index = wrap_open(reopened2)

    repo2 = Repo(stores=[reopened1, reopened2])

    assert open_calls == []
    assert repo2.query(Definition(QueryIndexDirLeaf, SKIP_ARGS)).stored().count() == 2
    assert sorted(open_calls) == sorted([reopened1.catalog_key(), reopened2.catalog_key()])


def test_ready_sqlite_exact_query_does_not_scan_object_directories(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    obj = QueryIndexDirLeaf("ready-exact", repo=repo)
    repo.save_object(obj)

    reopened_store = DirStore(store.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete"))

    def fail_hydrate():
        raise AssertionError("ready exact SQLite query should not hydrate Store roots")

    reopened_store.hydrate_index = fail_hydrate
    repo2 = Repo(stores=reopened_store)

    assert repo2.query(obj.definition).stored().count() == 1


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


def test_sqlite_concurrent_missing_index_build_claim_scans_once(tmp_path):
    store = DirStore(tmp_path / "store", query_index="memory")
    repo = Repo(stores=store)
    obj = QueryIndexDirLeaf("claim", repo=repo)
    repo.save_object(obj)

    store1 = DirStore(store.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    store2 = DirStore(store.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    calls = []
    lock = threading.Lock()

    def make_hydrate(source):
        original = source.hydrate_index

        def hydrate():
            with lock:
                calls.append(source.catalog_key())
            time.sleep(0.05)
            yield from original()

        return hydrate

    store1.hydrate_index = make_hydrate(store1)
    store2.hydrate_index = make_hydrate(store2)
    selector = Definition(QueryIndexDirLeaf, SKIP_ARGS)
    results = []
    errors = []

    def query_store(source):
        try:
            results.append(Repo(stores=source).query(selector).stored().count())
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=query_store, args=(source,)) for source in (store1, store2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert errors == []
    assert sorted(results) == [1, 1]
    assert len(calls) == 1


def test_sqlite_stale_build_claim_is_recovered(tmp_path):
    store = DirStore(tmp_path / "store", query_index="memory")
    repo = Repo(stores=store)
    obj = QueryIndexDirLeaf("stale-claim", repo=repo)
    repo.save_object(obj)

    reopened_store = DirStore(store.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    index = reopened_store.open_query_index()
    claim_path = index._build_claim_path()
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    claim_path.write_text("stale\n", encoding="utf-8")
    old_time = time.time() - 1000
    os.utime(claim_path, (old_time, old_time))

    count = Repo(stores=reopened_store).query(Definition(QueryIndexDirLeaf, SKIP_ARGS)).stored().count()

    assert count == 1
    assert not claim_path.exists()


def test_sqlite_rebuild_registers_store_roots_in_batches(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="memory")
    repo = Repo(stores=store)
    objs = [QueryIndexDirLeaf(f"batch-{idx}", repo=repo) for idx in range(3)]
    for obj in objs:
        repo.save_object(obj)

    monkeypatch.setattr(sqlite_index_module, "_REBUILD_BATCH_SIZE", 2)
    original_for_query_index_roots = sqlite_index_module.ConcreteDefinitionGraph.for_query_index_roots
    batch_sizes = []

    def spy_for_query_index_roots(cls, cdefs):
        cdefs = tuple(cdefs)
        batch_sizes.append(len(cdefs))
        return original_for_query_index_roots(cdefs)

    monkeypatch.setattr(sqlite_index_module.ConcreteDefinitionGraph, "for_query_index_roots", classmethod(spy_for_query_index_roots))
    reopened_store = DirStore(store.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete"))

    count = Repo(stores=reopened_store).query(Definition(QueryIndexDirLeaf, SKIP_ARGS)).stored().count()

    assert count == 3
    assert batch_sizes == [2, 1]


def test_sqlite_rebuilds_mixed_v1_v2_authoritative_roots(tmp_path):
    store = DirStore(tmp_path / "store", query_index="memory")
    v1 = ConcreteDefinition._from_persisted_record(
        ImportRef("builtins", "dict"),
        FrozenTuple(("v1",)),
        FrozenDict({}),
    )
    v2 = ConcreteDefinition._from_persisted_record(
        ImportRef("builtins", "dict"),
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments((("value", "v2"),)),
    )
    for cdef in (v1, v2):
        def_path = Path(store.object_dir(cdef)) / "def.pkl"
        def_path.parent.mkdir(parents=True)
        pickle_save(cdef, def_path)

    index = DirStore(store.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete")).open_query_index()
    index.rebuild()

    assert index.status().state == "ready"
    assert index.status().row_counts["stored_roots"] == 2
    with index.read_view() as view:
        assert view.exact_ids(v1)
        assert view.exact_ids(v2)


def test_sqlite_interrupted_build_is_not_marked_ready(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="memory")
    repo = Repo(stores=store)
    obj = QueryIndexDirLeaf("interrupted", repo=repo)
    repo.save_object(obj)
    index = DirStore(store.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete")).open_query_index()

    def fail_validation(self, *, roots):
        raise RuntimeError("injected pre-ready validation failure")

    monkeypatch.setattr(SQLiteStoreQueryIndex, "_validate_rebuild_before_ready", fail_validation)

    with pytest.raises(RuntimeError, match="pre-ready"):
        index.rebuild()

    assert index.status().state == "missing"
    assert not list(Path(index.path).parent.glob(f"{Path(index.path).name}.rebuild-*.tmp"))


def test_sqlite_failed_replacement_preserves_ready_sidecar(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    obj = QueryIndexDirLeaf("replacement-failure", repo=repo)
    repo.save_object(obj)
    index = store.open_query_index()
    before = Path(index.path).read_bytes()

    def fail_validation(self, *, roots):
        raise RuntimeError("injected replacement validation failure")

    monkeypatch.setattr(SQLiteStoreQueryIndex, "_validate_rebuild_before_ready", fail_validation)

    with pytest.raises(RuntimeError, match="replacement validation"):
        index.rebuild()

    assert Path(index.path).read_bytes() == before
    assert index.status().state == "ready"
    with index.read_view() as view:
        assert view.exact_ids(obj.definition)


def test_sqlite_rebuild_failure_between_batches_preserves_ready_sidecar_and_authority(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    objects = [QueryIndexDirLeaf(f"between-batches-{index}", repo=repo) for index in range(2)]
    for obj in objects:
        repo.save_object(obj)
    index = store.open_query_index()
    before_sidecar = Path(index.path).read_bytes()
    before_authority = _authority_digest(store)
    original_graph = sqlite_index_module.ConcreteDefinitionGraph.for_query_index_roots
    calls = 0
    monkeypatch.setattr(sqlite_index_module, "_REBUILD_BATCH_SIZE", 1)

    def fail_second_batch(cls, roots):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected failure between rebuild batches")
        return original_graph(roots)

    monkeypatch.setattr(
        sqlite_index_module.ConcreteDefinitionGraph,
        "for_query_index_roots",
        classmethod(fail_second_batch),
    )

    with pytest.raises(OSError, match="between rebuild batches"):
        index.rebuild()

    assert calls == 2
    assert Path(index.path).read_bytes() == before_sidecar
    assert _authority_digest(store) == before_authority
    with index.read_view() as view:
        assert all(view.exact_ids(obj.definition) for obj in objects)
    assert set(repo.query(Definition(QueryIndexDirLeaf, SKIP_ARGS)).stored().defs()) == {
        obj.definition for obj in objects
    }
    assert not list(Path(index.path).parent.glob(f"{Path(index.path).name}.rebuild-*.tmp*"))


def test_sqlite_quarantine_keeps_canonical_sidecar_readable_until_activation(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    obj = QueryIndexDirLeaf("quarantine-read", repo=repo)
    repo.save_object(obj)
    index = store.open_query_index()
    replacement_path = index._replacement_path()
    replacement = SQLiteStoreQueryIndex(
        source_key=index.source_key,
        path=replacement_path,
        config=SQLiteQueryIndexConfig(path=replacement_path, journal_mode="delete"),
    )
    replacement.initialize_empty(generation=index.current_generation() + 1)
    replacement.close()
    original_link = os.link
    observed = {}

    def read_during_quarantine(source, target):
        original_link(source, target)
        with index.read_view() as view:
            observed["present"] = bool(view.exact_ids(obj.definition))

    monkeypatch.setattr(os, "link", read_during_quarantine)

    index._activate_replacement(replacement_path, quarantine_existing=True)

    assert observed == {"present": True}
    assert list(Path(index.path).parent.glob(f"{Path(index.path).name}.quarantine-*"))


def test_sqlite_activation_failure_preserves_ready_sidecar_and_authority(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    obj = QueryIndexDirLeaf("quarantine-activation-failure", repo=repo)
    repo.save_object(obj)
    index = store.open_query_index()
    before = Path(index.path).read_bytes()
    before_authority = _authority_digest(store)
    original_replace = os.replace

    def fail_activation(source, destination):
        if Path(destination) == index.path and ".rebuild-" in Path(source).name:
            raise OSError("injected activation failure")
        return original_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_activation)

    with pytest.raises(OSError, match="activation failure"):
        index.rebuild(quarantine_existing=True)

    assert Path(index.path).read_bytes() == before
    assert _authority_digest(store) == before_authority
    with index.read_view() as view:
        assert view.exact_ids(obj.definition)
    assert list(repo.query(Definition(QueryIndexDirLeaf, SKIP_ARGS)).stored().defs()) == [obj.definition]
    assert list(Path(index.path).parent.glob(f"{Path(index.path).name}.quarantine-*"))
    assert not list(Path(index.path).parent.glob(f"{Path(index.path).name}.rebuild-*.tmp*"))


def test_sqlite_keyboard_interrupt_cleans_staged_replacement(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    obj = QueryIndexDirLeaf("interrupted-cleanup", repo=repo)
    repo.save_object(obj)
    index = store.open_query_index()
    before_sidecar = Path(index.path).read_bytes()
    before_authority = _authority_digest(store)

    def interrupt_validation(self, *, roots):
        raise KeyboardInterrupt("injected rebuild interruption")

    monkeypatch.setattr(SQLiteStoreQueryIndex, "_validate_rebuild_before_ready", interrupt_validation)

    with pytest.raises(KeyboardInterrupt, match="rebuild interruption"):
        index.rebuild()

    assert Path(index.path).read_bytes() == before_sidecar
    assert _authority_digest(store) == before_authority
    with index.read_view() as view:
        assert view.exact_ids(obj.definition)
    assert list(repo.query(Definition(QueryIndexDirLeaf, SKIP_ARGS)).stored().defs()) == [obj.definition]
    assert not list(Path(index.path).parent.glob(f"{Path(index.path).name}.rebuild-*.tmp*"))


def test_sqlite_rebuild_preserves_dirty_marker_from_concurrent_store_save(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    first = QueryIndexDirLeaf("before", repo=repo)
    repo.save_object(first)
    index = store.open_query_index()
    original_validation = SQLiteStoreQueryIndex._validate_rebuild_before_ready
    later = QueryIndexDirLeaf("during", repo=repo)

    def save_during_validation(self, *, roots):
        store.save_object(later)
        return original_validation(self, roots=roots)

    monkeypatch.setattr(SQLiteStoreQueryIndex, "_validate_rebuild_before_ready", save_during_validation)
    index.rebuild()

    assert store.query_index_is_dirty()

    monkeypatch.setattr(SQLiteStoreQueryIndex, "_validate_rebuild_before_ready", original_validation)
    index.refresh("auto")
    with index.read_view() as view:
        assert view.exact_ids(later.definition)
    assert not store.query_index_is_dirty()


def test_sqlite_registration_clears_only_markers_for_registered_roots(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    baseline = QueryIndexDirLeaf("baseline", repo=repo)
    repo.save_object(baseline)
    first = QueryIndexDirLeaf("first-overlap", repo=repo)
    second = QueryIndexDirLeaf("second-overlap", repo=repo)

    store.save_object(first)
    first_markers = set(store._query_index_dirty_markers())
    store.save_object(second)
    index = store.open_query_index()
    index.register_stored_roots(
        ConcreteDefinitionGraph.for_query_index(second.definition),
        [second.definition],
    )

    assert first_markers <= set(store._query_index_dirty_markers())
    assert store.query_index_is_dirty()
    index.refresh("auto")
    with index.read_view() as view:
        assert view.exact_ids(first.definition)
        assert view.exact_ids(second.definition)
    assert not store.query_index_is_dirty()


def test_sqlite_rebuild_keeps_token_for_root_published_after_snapshot(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    repo.save_object(QueryIndexDirLeaf("baseline", repo=repo))
    later = QueryIndexDirLeaf("published-later", repo=repo)
    marker = Path(store.mark_query_index_dirty(later.definition))
    index = store.open_query_index()

    index.rebuild()

    assert marker.exists()
    store.save_object(later)
    index.refresh("auto")
    with index.read_view() as view:
        assert view.exact_ids(later.definition)
    assert not store.query_index_is_dirty()


def test_failed_root_publication_discards_its_dirty_token(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    repo.save_object(QueryIndexDirLeaf("baseline", repo=repo))
    failed = QueryIndexDirLeaf("failed-publication", repo=repo)
    failed_dir = store.object_dir(failed.definition)
    original_replace = os.replace

    def fail_final_replace(source, destination):
        if destination == failed_dir:
            raise OSError("injected root publication failure")
        return original_replace(source, destination)

    monkeypatch.setattr("dryml.core.store.store.os.replace", fail_final_replace)

    with pytest.raises(OSError, match="root publication"):
        store.save_object(failed)

    assert not store.has(failed.definition)
    assert not store.query_index_is_dirty()


@pytest.mark.parametrize("payload", [b"", b"\xff", b"garbage\n"])
def test_successful_rebuild_clears_malformed_dirty_marker(tmp_path, payload):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    repo.save_object(QueryIndexDirLeaf("baseline", repo=repo))
    marker = Path(store.mark_query_index_dirty())
    marker.write_bytes(payload)
    index = store.open_query_index()

    index.rebuild()

    assert not marker.exists()
    assert not store.query_index_is_dirty()
    assert index.status().state == "ready"


def test_sqlite_peer_connection_reopens_after_replacement(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    first = QueryIndexDirLeaf("first")
    store.save_object(first)
    rebuilding = store.open_query_index()
    rebuilding.rebuild()

    peer_store = DirStore(store.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    peer = peer_store.open_query_index()
    with peer.read_view() as view:
        old_generation = view.generation
        assert view.exact_ids(first.definition)

    second = QueryIndexDirLeaf("second")
    store.save_object(second)
    rebuilding.rebuild()

    with peer.read_view() as view:
        assert view.generation > old_generation
        assert view.exact_ids(second.definition)


def test_sqlite_future_sidecar_fails_before_store_scan_or_row_decode(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    obj = QueryIndexDirLeaf("future-sidecar", repo=repo)
    repo.save_object(obj)
    path = Path(store.query_index_path)
    con = require_sqlite().connect(path)
    con.execute("UPDATE catalog_state SET cdef_codec_version = cdef_codec_version + 1")
    con.commit()
    con.close()
    before = path.read_bytes()

    reopened_store = DirStore(store.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    index = reopened_store.open_query_index()
    calls = []

    def fail_hydrate():
        calls.append(True)
        raise AssertionError("future sidecar must fail before Store scanning")

    reopened_store.hydrate_index = fail_hydrate

    with pytest.raises(QueryIndexIncompatible, match="refusing to rebuild"):
        index.rebuild()

    assert calls == []
    assert path.read_bytes() == before
    assert index.status().state == "incompatible"


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
    assert rebuilt.generation_after > rebuilt.generation_before
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


def test_dirstore_reconcile_adds_external_object(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    first = QueryIndexDirLeaf("first", repo=repo)
    repo.save_object(first)

    external_store = DirStore(store.base_dir, query_index="memory")
    external_repo = Repo(stores=external_store)
    second = QueryIndexDirLeaf("second", repo=external_repo)
    external_repo.save_object(second)

    selector = Definition(QueryIndexDirLeaf, SKIP_ARGS)
    assert set(repo.query(selector).stored().defs()) == {first.definition}

    report = store.reconcile_query_index()

    assert report.changed
    assert report.action == "rebuild"
    assert set(repo.query(selector).stored().defs()) == {first.definition, second.definition}


def test_dirstore_reconcile_removes_deleted_object(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    keep = QueryIndexDirLeaf("keep", repo=repo)
    remove = QueryIndexDirLeaf("remove", repo=repo)
    repo.save_object(keep)
    repo.save_object(remove)
    shutil.rmtree(Path(store.object_dir(remove.definition)))

    report = store.reconcile_query_index()

    assert report.changed
    assert report.action == "rebuild"
    assert list(repo.query(Definition(QueryIndexDirLeaf, SKIP_ARGS)).stored().defs()) == [keep.definition]


def test_dirstore_validate_detects_changed_or_misplaced_def_pickle(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    original = QueryIndexDirLeaf("original", repo=repo)
    changed = QueryIndexDirLeaf("changed", repo=repo)
    repo.save_object(original)
    def_path = Path(store.object_dir(original.definition)) / "def.pkl"
    pickle_save(changed.definition, def_path)
    authority_bytes = def_path.read_bytes()

    report = store.validate_query_index(thorough=True)

    reconcile = store.reconcile_query_index()

    assert not report.ok
    assert any(issue.message == "Store root scan failed." for issue in report.issues)
    assert reconcile.changed is False
    assert any(issue.message == "SQLite query-index rebuild failed." for issue in reconcile.issues)
    assert def_path.read_bytes() == authority_bytes
    assert store.query_index_status().state == "dirty"


def test_dirstore_reconcile_quarantines_corrupt_database(tmp_path):
    store = DirStore(tmp_path / "store", query_index="memory")
    repo = Repo(stores=store)
    obj = QueryIndexDirLeaf("corrupt", repo=repo)
    repo.save_object(obj)

    sqlite_store = DirStore(store.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    path = Path(sqlite_store.query_index_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a sqlite database")

    assert sqlite_store.query_index_status().state == "corrupt"
    report = sqlite_store.reconcile_query_index()

    assert report.changed
    assert report.action == "rebuild"
    assert sqlite_store.query_index_status().state == "ready"
    assert list(path.parent.glob(f"{path.name}.quarantine-*"))
    assert Repo(stores=sqlite_store).query(Definition(QueryIndexDirLeaf, SKIP_ARGS)).stored().count() == 1


def test_dirstore_reconcile_foreign_key_failure_triggers_rebuild(tmp_path):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    obj = QueryIndexDirLeaf("fk", repo=repo)
    repo.save_object(obj)

    sqlite3 = require_sqlite()
    con = sqlite3.connect(store.query_index_path)
    con.execute("PRAGMA foreign_keys=OFF")
    con.execute(
        """
        INSERT INTO stored_roots (def_id, storage_hash, relative_def_path, indexed_generation)
        VALUES (?, ?, ?, ?)
        """,
        (9999, bytes([0]) * 32, "objects/00/missing/def.pkl", 1),
    )
    con.commit()
    con.close()

    report = store.reconcile_query_index()

    assert report.changed
    assert report.action == "rebuild"
    assert any(issue.message == "SQLite foreign_key_check failed." for issue in report.issues)
    assert store.validate_query_index(thorough=True).ok


def test_dirstore_reconcile_quick_check_failure_triggers_rebuild(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    repo = Repo(stores=store)
    obj = QueryIndexDirLeaf("quick", repo=repo)
    repo.save_object(obj)
    original_validate = SQLiteStoreQueryIndex.validate
    calls = 0

    def fail_once(self, *, thorough=False):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ValidationReport(
                "sqlite",
                self.source_key,
                False,
                (ValidationIssue("error", "SQLite quick_check failed.", "injected"),),
                row_counts={"stored_roots": 1},
            )
        return original_validate(self, thorough=thorough)

    monkeypatch.setattr(SQLiteStoreQueryIndex, "validate", fail_once)

    report = store.reconcile_query_index()

    assert report.changed
    assert report.action == "rebuild"
    assert any(issue.message == "SQLite quick_check failed." for issue in report.issues)


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
