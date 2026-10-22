"""Current-format discovery, completeness and detached page boundaries."""

from contextlib import contextmanager
from pathlib import Path
import sqlite3

import pytest

from dryml.core import Definition, Object, Repo, Serializable
from dryml.core.query.model import QueryError, QueryIndexError
from dryml.core.store.dir import DirStore

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")


class DiscoveryValue(Serializable):
    def __init__(self, name):
        self.name, self.value = name, 0

    def save_state_to_dir_imp(self, directory, *, codec):
        Path(directory, "value").write_text(str(self.value))


class DiscoveryPair(Object):
    def __init__(self, left, right):
        self.left, self.right = left, right


def saved_values(tmp_path, count=12):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    obj = DiscoveryValue("selected", repo=repo)
    saved = []
    for value in range(count):
        obj.value = value
        saved.append(repo.save(obj))
    return store, repo, saved


def no_inventory(monkeypatch, store):
    monkeypatch.setattr(store, "inspect_states", lambda *a, **k: pytest.fail("variant inventory"))
    monkeypatch.setattr(store, "iter_definition_records", lambda *a, **k: pytest.fail("legacy inventory"))
    import dryml.core.query.authority as authority
    original = authority.capture_store_projection

    def capture(*args, **kwargs):
        assert not kwargs.get("rebuild"), "query rebuilt ready index"
        return original(*args, **kwargs)

    monkeypatch.setattr(authority, "capture_store_projection", capture)


def test_ordinary_current_queries_are_immediately_visible(tmp_path):
    store, _, saved = saved_values(tmp_path, 2)
    reader = Repo(DirStore(store.base_dir, query_index="sqlite"))
    assert reader.query().cdefs().stored().refresh(False).count() == 1
    assert reader.query().saved().stored().refresh(False).count() == 2
    assert reader._num_constructions == 0


def test_unconstrained_concrete_facets_do_not_truncate_recipe_membership(tmp_path):
    store, _, saved = saved_values(tmp_path, 1)
    child = Definition(DiscoveryValue, "selected").concretize()
    bare = Definition(DiscoveryPair, child, "fixed").concretize()
    pinned = Definition(DiscoveryPair, child.with_state(saved[0].state), "fixed").concretize()
    store.publish_node(bare)
    store.publish_node(pinned)
    query = store.query().cdefs().stored().sel(bare).refresh(False)
    assert query.count() == 2
    assert query.require_indexed().count() == 2


def test_exact_full_key_hit_and_miss_are_index_free(tmp_path, monkeypatch):
    from dryml.core import StateKey
    store, repo, saved = saved_values(tmp_path, 1)
    key = saved[0].state.key
    monkeypatch.setattr(store, "open_query_index", lambda: pytest.fail("exact lookup opened SQLite"))
    no_inventory(monkeypatch, store)
    assert repo.query().local_states().state_key(key).refresh(False).one() == saved[0].state
    assert repo.query().local_states().state_key(StateKey(key.cdef_digest, key.codec_hash[:-1] + ("0" if key.codec_hash[-1] != "0" else "1"))).refresh(False).count() == 0


def test_exact_missing_key_neither_prepares_nor_reads_live_cache(tmp_path, monkeypatch):
    from dryml.core import StateKey
    import dryml.core.node_prepare as preparation
    store, repo, saved = saved_values(tmp_path, 1)
    missing = StateKey("0" * 64, saved[0].state.key.codec_hash)
    monkeypatch.setattr(preparation.NodePreparation, "__init__", lambda *a, **k: pytest.fail("exact miss prepared a graph"))
    monkeypatch.setattr(store, "open_query_index", lambda: pytest.fail("exact miss opened SQLite"))
    assert repo.query().local_states().state_key(missing).refresh(False).count() == 0


def test_origin_page_is_selective_and_deduplicates_donors(tmp_path, monkeypatch):
    store, repo, saved = saved_values(tmp_path)
    other = DiscoveryValue("selected", repo=repo)
    repo.save(other, states={"$": saved[0].state})
    for i in range(8):
        repo.save(DiscoveryValue("unrelated-" + str(i), repo=repo))
    no_inventory(monkeypatch, store)
    query = repo.query().local_states(origin=saved[0].state.creation_cdef).refresh(False).require_indexed()
    result = query.take(10)
    assert len(result) == 10
    assert len({state.key for state in result}) == 10
    assert all(state.key.cdef_digest == saved[0].state.key.cdef_digest for state in result)


def test_local_state_identity_constraint_is_domain_error(tmp_path):
    from dryml.core.query.model import QueryDomainError
    _, repo, saved = saved_values(tmp_path, 1)
    with pytest.raises(QueryDomainError, match="donor identity"):
        repo.query().local_states().sel(Definition(DiscoveryValue).with_identity(saved[0].object_id)).count()


def test_annotations_current_capture_and_metadata_query_are_distinct(tmp_path):
    from dryml.core import SaveAnnotations
    from dryml.core.query import field
    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    saved = repo.save(DiscoveryValue("metadata", repo=repo), annotations=SaveAnnotations(
        object={"purpose": "fixture"}, state={"phase": "captured", "nested": [1, (2, 3)]}))
    assert repo.get_metadata(saved)["phase"] == "captured"
    captured = repo.get_snapshot_metadata(saved)
    assert captured.captured_state_annotations["phase"] == "captured"
    assert captured.lineages
    repo.set_metadata(saved, {"phase": "current"})
    assert repo.get_snapshot_metadata(saved).captured_state_annotations["phase"] == "captured"
    query = repo.query().saved().stored().where(field("state", "phase").eq("current")).refresh(False).require_indexed()
    assert query.count() == 1
    assert query.where(field("object", "purpose").eq("other")).count() == 0
    assert repo.delete_metadata(saved)
    assert repo.get_metadata(saved) is None
    assert repo.get_snapshot_metadata(saved).captured_state_annotations["nested"] == [1, (2, 3)]


def test_empty_and_count_require_relation_guards(tmp_path):
    store, repo, _ = saved_values(tmp_path, 1)
    index = store.open_query_index()
    with sqlite3.connect(index.path) as con:
        con.execute("DROP TRIGGER store_projection_delete")
    for query in (repo.query().cdefs(), repo.query().sel(Definition(DiscoveryValue, name="absent"))):
        with pytest.raises(QueryIndexError):
            query.refresh(False).require_indexed().count()


def test_prepare_and_yield_are_outside_fences_and_continuation_refuses(tmp_path, monkeypatch):
    from dryml.core.store.records import NodeNameRecord
    store, repo, saved = saved_values(tmp_path, 4)
    depth = [0]
    original = store.authority_read_fence

    @contextmanager
    def fence():
        with original():
            depth[0] += 1
            try:
                yield
            finally:
                depth[0] -= 1

    import dryml.core.node_codec as codec
    decode = codec._decode_node_graph_views

    def checked(packet, **kwargs):
        assert depth[0] == 0
        return decode(packet, **kwargs)

    monkeypatch.setattr(store, "authority_read_fence", fence)
    monkeypatch.setattr(codec, "_decode_node_graph_views", checked)
    iterator = repo.query().saved().stored().refresh(False).page_size(1).iter()
    first = next(iterator)
    assert depth[0] == 0
    _, (record,), _ = store._prepare_node_roots((saved[0],), "saved")
    name = NodeNameRecord("moved", "saved", record.digest)
    store.compare_and_set_node_name(name, expected=None)
    with pytest.raises(QueryError, match="changed|unstable|continuation"):
        next(iterator)
    assert first is not None and depth[0] == 0


def test_zero_prefix_does_not_acquire_or_prepare_sources(tmp_path, monkeypatch):
    store, repo, _ = saved_values(tmp_path, 1)
    import dryml.core.node_prepare as preparation
    monkeypatch.setattr(store, "authority_read_fence", lambda: pytest.fail("zero prefix acquired source"))
    monkeypatch.setattr(store, "open_query_index", lambda: pytest.fail("zero prefix opened index"))
    monkeypatch.setattr(preparation.NodePreparation, "__init__", lambda *a, **k: pytest.fail("zero prefix prepared Node"))
    assert repo.query().local_states().refresh(True).take(0).count() == 0
    assert list(repo.query().saved().page_size(1).take(0)) == []


def test_origin_batch_decodes_once_and_never_authenticates_payload(tmp_path, monkeypatch):
    store, repo, saved = saved_values(tmp_path)
    import dryml.core.node_codec as codec
    original = codec._decode_node_graph_views
    decodes = []

    def counted(packet, **kwargs):
        decodes.append(len(packet["views"]))
        return original(packet, **kwargs)

    monkeypatch.setattr(codec, "_decode_node_graph_views", counted)
    monkeypatch.setattr(store, "_authenticate_state_data", lambda *a: pytest.fail("discovery read private payload"))
    result = repo.query().local_states(origin=saved[0].state.creation_cdef).page_size(3).refresh(False).take(10)
    assert len(result) == 10 and decodes == [1]


@pytest.mark.parametrize("damage", ["conflict", "missing"])
def test_filtered_out_store_declaration_anchor_is_still_checked(tmp_path, damage):
    from dryml.core import ObjectId
    from dryml.core.query.model import QueryError
    from dryml.core.store.store import StoreCapabilityError
    first, second = (DirStore(tmp_path / name, query_index="sqlite") for name in ("first", "second"))
    identity = ObjectId()
    selected = Definition(DiscoveryValue, name="wanted").concretize().with_identity(identity)
    first.publish_node(selected)
    other = selected if damage == "missing" else Definition(DiscoveryValue, name="filtered").concretize().with_identity(identity)
    second.publish_node(other)
    if damage == "missing":
        Path(second._node_path("declarations/anchors", second._anchor_key(identity))).unlink()
    query = Repo((first, second)).query().sel(Definition(DiscoveryValue, name="wanted").with_identity(identity)).refresh(False).require_indexed()
    with pytest.raises((QueryError, StoreCapabilityError), match="anchor|conflict"):
        query.count()


def test_mixed_facets_are_correlated_to_the_same_occurrences(tmp_path):
    from dryml.core.utils.graph.path import GraphPath, Parameter
    store, repo, saved = saved_values(tmp_path, 2)
    # The same recipe is essential for state compatibility; identity/state pairs
    # are then swapped between otherwise independent occurrences.
    first, second = DiscoveryValue("selected", repo=repo), DiscoveryValue("selected", repo=repo)
    left_path, right_path = GraphPath((Parameter("left"),)), GraphPath((Parameter("right"),))
    a = repo.save(DiscoveryPair(first, second, repo=repo), states={left_path: saved[0].state, right_path: saved[1].state})
    repo.save(DiscoveryPair(first, second, repo=repo), states={left_path: saved[1].state, right_path: saved[0].state})
    left = Definition(DiscoveryValue).with_identity(a.at(left_path).object_id).with_state(saved[0].state)
    right = Definition(DiscoveryValue).with_state(saved[1].state)
    query = repo.query().saved().stored().sel(Definition(DiscoveryPair, left=left, right=right)).refresh(False).require_indexed()
    assert query.count() == 1
    assert query.one().graph_equal(a)


def test_partial_unordered_features_are_only_conservative_prefilters(tmp_path):
    from dryml.core import ObjectId
    store, repo, saved = saved_values(tmp_path, 2)
    first, second = (Definition(DiscoveryValue, "selected").concretize().with_identity(ObjectId()).with_state(value.state)
                     for value in saved)
    graph = Definition(DiscoveryPair, {first, second}, "unordered").concretize()
    store.publish_node(graph, kind="saved")
    pattern = Definition(DiscoveryPair, left={Definition(DiscoveryValue).with_identity(first.object_id),
                                             Definition(DiscoveryValue).with_identity(second.object_id)})
    found = repo.query().saved().stored().sel(pattern).refresh(False).require_indexed().collect()
    assert found.one().graph_equal(graph)


def _fresh_key_reader(path, key, queue):
    store = None
    try:
        store = DirStore.open_existing(path, query_index="none")
        repo = Repo(store)
        query = repo.query().local_states().state_key(key).refresh(False)
        queue.put((query.count(), query.available().count(), repo._num_constructions))
        repo.close(flush=False)
    except BaseException as error:
        queue.put(type(error).__name__)
    finally:
        if store is not None:
            store.close()


def test_clear_and_fresh_process_never_restore_stale_availability(tmp_path):
    import multiprocessing
    from tests.core.test_state_retention import install
    store = DirStore(tmp_path / "store", query_index="sqlite")
    key = install(store)
    repo = Repo(store)
    query = repo.query().local_states().state_key(key).refresh(False)
    assert query.available().count() == 1
    repo.clear_cache(strong=False, weak=False)
    assert store.delete_state(key).status == "removed"
    assert query.count() == 1 and query.available().count() == 0
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_fresh_key_reader, args=(store.base_dir, key, queue))
    process.start()
    try:
        assert queue.get(timeout=15) == (1, 0, 0)
        process.join(15)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        queue.close()
        queue.join_thread()


def test_descriptor_history_available_and_retained_domains_are_independent(tmp_path):
    from tests.core.test_state_retention import install, retained
    store = DirStore(tmp_path / "store", query_index="sqlite")
    key = install(store)
    query = store.query().local_states().refresh(False).require_indexed()
    assert query.count() == query.available().count() == 1
    assert query.retained().count() == 0
    token = retained(store, store, key)
    assert query.retained().count() == 1
    store.retention.release(token, (store,))
    assert store.delete_state(key).status == "removed"
    assert query.count() == 1 and query.available().count() == 0
    assert store.query().cdefs().stored().refresh(False).count() == 0


def test_source_withdrawal_preserves_other_replica_knowledge(tmp_path):
    first, second = (DirStore(tmp_path / name, query_index="sqlite") for name in ("first", "second"))
    node = Definition(DiscoveryValue, name="known").concretize()
    roots = [store.publish_node(node) for store in (first, second)]
    query = Repo((first, second)).query().cdefs().stored().refresh(False)
    results = query.collect()
    value = results.one()
    assert len(results.sources(value)) == 2
    assert first.withdraw_node(roots[0].digest)
    results = query.collect()
    assert results.one().graph_equal(node) and len(results.sources(results.one())) == 1
    assert second.read_node(roots[1].digest).graph_equal(node)


@pytest.mark.parametrize("change", ["sidecar", "coverage", "topology", "availability"])
def test_each_continuation_disposition_rejects_changed_cut(tmp_path, change):
    from tests.core.test_state_retention import install
    store = DirStore(tmp_path / "store", query_index="sqlite")
    for value in range(4):
        install(store, payload=str(value).encode())
    repo = Repo(store)
    query = repo.query().local_states().refresh(False).page_size(1)
    iterator = query.iter()
    first = next(iterator)
    if change == "sidecar":
        store.rebuild_query_index()
    elif change == "coverage":
        with sqlite3.connect(store.open_query_index().path) as con:
            con.execute("UPDATE store_projection_coverage SET dirty=1 WHERE family='edge'")
    elif change == "topology":
        repo.add_store(DirStore(tmp_path / "added", query_index="none"))
    else:
        assert store.delete_state(first.key).status == "removed"
    with pytest.raises(QueryError, match="changed|continuation|coverage"):
        next(iterator)
