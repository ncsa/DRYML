import pytest
import shutil
import threading

from dryml.core2 import Definition, Object, Repo, Serializable, SKIP_ARGS
from dryml.core2.cdef_graph import ConcreteDefinitionGraph
from dryml.core2.definition import ConcreteDefinition
from dryml.core2.freeze import FrozenDict, FrozenTuple
from dryml.core2.query import QueryIndexError, SetMember
from dryml.core2.query.index import OccurrenceTraversalSnapshot
from dryml.core2.query.path import get_subtree
from dryml.core2.store.dir import DirStore
from dryml.core2.store.store import Store
from dryml.core2.utils.general import pickle_load, pickle_save
from dryml.core2.utils.stable_hash import stable_hash_function


class IndexLeaf(Object):
    def __init__(self, name):
        super().__init__()
        self.name = name


class IndexPersistent(Serializable):
    def __init__(self, child, *, state=0):
        super().__init__()
        self.child = child
        self.state = state

    def save_state_to_dir_imp(self, dest_dir, revision=None):
        pickle_save(self.state, f"{dest_dir}/state.pkl")

    def restore_state_from_dir_imp(self, src_dir, revision=None):
        self.state = pickle_load(f"{src_dir}/state.pkl")


class BadIndexStore(Store):
    def __init__(self, values):
        self.values = values

    @property
    def base_dir(self):
        return "bad-index-store"

    @property
    def object_root_dir(self):
        return "bad-index-store/objects"

    def has(self, cdef):
        return False

    def hydrate_index(self):
        return tuple(self.values)

    def _object_dir(self, cdef):
        return "bad-index-store/objects/missing"

    def commit(self):
        pass


def test_catalog_registering_same_definition_is_idempotent():
    repo = Repo()
    cdef = IndexLeaf("x").definition

    first = repo._query_catalog.register_cached(cdef)
    second = repo._query_catalog.register_cached(cdef)

    assert first == second
    assert len(repo._query_catalog.definitions_by_id) == 1
    assert all(list(posting).count(first) == 1 for posting in repo._query_catalog.local_postings.values())


def test_catalog_exposes_only_graph_native_index_state():
    repo = Repo()
    catalog = repo._query_catalog

    assert not hasattr(catalog, "postings")
    assert not hasattr(catalog, "occurrences_by_nested")
    assert not hasattr(catalog, "occurrences_by_owner")
    assert not hasattr(catalog, "occurrence_by_key")


def test_cached_only_definition_is_known_not_stored_then_save_promotes(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = IndexLeaf("cached", repo=repo)
    repo.add_objects(obj)

    assert list(repo.find_defs(None, scope="cached", refresh=False)) == [obj.definition]
    assert len(repo.find_defs(None, scope="stored", refresh=False)) == 0

    repo.save_object(obj)

    assert list(repo.find_defs(None, scope="stored", refresh=False)) == [obj.definition]


def test_same_cdef_in_two_stores_has_one_record_and_two_replicas(tmp_path):
    store1 = DirStore(tmp_path / "store1")
    store2 = DirStore(tmp_path / "store2")
    repo = Repo(stores=[store1, store2])
    obj = IndexPersistent(IndexLeaf("x", repo=repo), repo=repo)
    repo.save_object(obj, store=store1)
    repo.save_object(obj, store=store2)

    repo2 = Repo(stores=[DirStore(store1.base_dir), DirStore(store2.base_dir)])
    results = repo2.find_defs(None)

    assert list(results) == [obj.definition]
    assert len(results.replicas(obj.definition)) == 2


def test_graph_registration_records_direct_edges_once():
    repo = Repo()
    child = IndexLeaf("child", repo=repo)
    parent = IndexPersistent(child, repo=repo)

    repo._query_catalog.register_cached(parent.definition)
    repo._query_catalog.register_cached(parent.definition)

    catalog = repo._query_catalog
    parent_id = catalog.cdef_id(parent.definition)
    child_id = catalog.cdef_id(child.definition)

    assert len(catalog.edge_by_key) == 1
    edge = next(iter(catalog.edge_by_key.values()))
    assert edge.parent_id == parent_id
    assert edge.child_id == child_id
    assert str(edge.path) == "$.args[0]"


def test_repeated_stored_registration_does_not_change_generation(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = IndexLeaf("stored", repo=repo)
    catalog = repo._query_catalog

    catalog.register_stored(obj.definition, store)
    generation = catalog.generation
    catalog.register_stored(obj.definition, store)

    assert catalog.generation == generation


def test_cache_sync_registers_union_graph_once(monkeypatch):
    from dryml.core2.query import index as index_mod

    repo = Repo()
    child = IndexLeaf("shared", repo=repo)
    left = IndexPersistent(child, state=1, repo=repo)
    right = IndexPersistent(child, state=2, repo=repo)
    repo.pin(left)
    repo.pin(right)
    calls = []
    original = index_mod.ConcreteDefinitionGraph.from_roots

    def spy_from_roots(cls, cdefs):
        cdefs = tuple(cdefs)
        calls.append(cdefs)
        return original(cdefs)

    monkeypatch.setattr(index_mod.ConcreteDefinitionGraph, "from_roots", classmethod(spy_from_roots))

    repo._query_catalog.sync_caches()

    assert len(calls) == 1
    assert set(calls[0]) >= {left.definition, right.definition}


def test_multiroot_graph_registration_visits_nodes_and_edges_once(monkeypatch):
    repo = Repo()
    child = IndexLeaf("shared", repo=repo)
    left = IndexPersistent(child, state=1, repo=repo)
    right = IndexPersistent(child, state=2, repo=repo)
    graph = ConcreteDefinitionGraph.from_roots((left.definition, right.definition))
    catalog = repo._query_catalog
    node_calls = []
    edge_visits = []
    original_register = catalog._register_definition_locked
    original_edges = graph.edges

    def spy_register(cdef):
        node_calls.append(cdef)
        return original_register(cdef)

    def spy_edges():
        edges = original_edges()
        edge_visits.extend(edges)
        return edges

    monkeypatch.setattr(catalog, "_register_definition_locked", spy_register)
    monkeypatch.setattr(graph, "edges", spy_edges)

    catalog.register_graph(graph)

    assert len(node_calls) == len(graph.nodes())
    assert len(edge_visits) == len(original_edges())


def test_shared_child_local_fingerprint_compiled_once(monkeypatch):
    from dryml.core2.query import index as index_mod

    repo = Repo()
    child_cdef = ConcreteDefinition(IndexLeaf, FrozenTuple(("shared",)), FrozenDict({}))
    left_cdef = ConcreteDefinition(IndexPersistent, FrozenTuple((child_cdef,)), FrozenDict({"state": 1}))
    right_cdef = ConcreteDefinition(IndexPersistent, FrozenTuple((child_cdef,)), FrozenDict({"state": 2}))
    calls = []
    original = index_mod.target_local_fingerprint

    def spy(cdef):
        calls.append(cdef)
        return original(cdef)

    monkeypatch.setattr(index_mod, "target_local_fingerprint", spy)

    repo._query_catalog.register_cached(left_cdef)
    repo._query_catalog.register_cached(right_cdef)

    assert calls.count(child_cdef) == 1


def test_parent_local_postings_do_not_contain_child_interior_features():
    repo = Repo()
    child = IndexLeaf("needle", repo=repo)
    parent = IndexPersistent(child, repo=repo)

    repo._query_catalog.register_cached(parent.definition)
    parent_id = repo._query_catalog.cdef_id(parent.definition)
    parent_record = repo._query_catalog.definitions_by_id[parent_id]
    child_name_hash = stable_hash_function(child.definition.args[0])

    assert all(
        not (token.kind == "SCALAR_VALUE" and token.payload == child_name_hash)
        for token in parent_record.local_fingerprint.counts
    )


def test_repeated_nested_cdef_keeps_every_owner_path_occurrence(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = IndexLeaf("shared", repo=repo)
    parent = IndexPersistent([child, child], repo=repo)
    repo.save_object(parent)

    repo2 = Repo(stores=DirStore(store.base_dir))
    occurrences = repo2.find_occurrences(Definition(IndexLeaf, SKIP_ARGS))

    assert occurrences.count() == 2
    assert {str(occ.path) for occ in occurrences} == {"$.args[0][0]", "$.args[0][1]"}
    assert occurrences.definitions().count() == 1


def test_occurrence_iteration_is_lazy(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = IndexLeaf("child", repo=repo)
    parent = IndexPersistent(child, repo=repo)
    repo.save_object(parent)

    catalog = repo._query_catalog
    child_id = catalog.cdef_id(child.definition)
    yields = 0
    original = OccurrenceTraversalSnapshot.iter_occurrences

    def spy_iter(self, *, max_occurrences=None):
        nonlocal yields
        for occ in original(self, max_occurrences=max_occurrences):
            yields += 1
            yield occ

    monkeypatch.setattr(OccurrenceTraversalSnapshot, "iter_occurrences", spy_iter)

    iterator = catalog.iter_occurrences_for_nested_ids({child_id})

    assert yields == 0
    assert next(iterator).definition == child.definition
    assert yields == 1


def test_occurrence_limit_stops_before_full_path_enumeration(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = IndexLeaf("shared", repo=repo)
    parent = IndexPersistent([child, child, child], repo=repo)
    repo.save_object(parent)

    catalog = repo._query_catalog
    child_id = catalog.cdef_id(child.definition)
    yields = 0
    original = OccurrenceTraversalSnapshot.iter_occurrences

    def spy_iter(self, *, max_occurrences=None):
        nonlocal yields
        for occ in original(self, max_occurrences=max_occurrences):
            yields += 1
            yield occ

    monkeypatch.setattr(OccurrenceTraversalSnapshot, "iter_occurrences", spy_iter)

    occurrences = tuple(catalog.iter_occurrences_for_nested_ids({child_id}, max_occurrences=1))

    assert len(occurrences) == 1
    assert yields == 1


def test_occurrence_iteration_does_not_hold_catalog_lock(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = IndexLeaf("shared", repo=repo)
    parent = IndexPersistent([child, child], repo=repo)
    repo.save_object(parent)

    catalog = repo._query_catalog
    child_id = catalog.cdef_id(child.definition)
    iterator = catalog.iter_occurrences_for_nested_ids({child_id})
    assert next(iterator).definition == child.definition
    acquired = []

    def try_lock():
        locked = catalog.lock.acquire(timeout=1)
        acquired.append(locked)
        if locked:
            catalog.lock.release()

    thread = threading.Thread(target=try_lock)
    thread.start()
    thread.join(timeout=2)

    assert acquired == [True]


def test_occurrence_result_is_lazy(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = IndexLeaf("wanted", repo=repo)
    repo.save_object(IndexPersistent(child, repo=repo))
    yields = 0
    original = OccurrenceTraversalSnapshot.iter_occurrences

    def spy_iter(self, *, max_occurrences=None):
        nonlocal yields
        for occ in original(self, max_occurrences=max_occurrences):
            yields += 1
            yield occ

    monkeypatch.setattr(OccurrenceTraversalSnapshot, "iter_occurrences", spy_iter)

    occurrences = repo.query(Definition(IndexLeaf, "wanted")).nested(refresh=False).execute()

    assert yields == 0
    assert occurrences.first().definition == child.definition
    assert yields == 1


def test_first_occurrence_does_not_enumerate_remaining_paths(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = IndexLeaf("shared", repo=repo)
    repo.save_object(IndexPersistent([child, child, child], repo=repo))
    yields = 0
    original = OccurrenceTraversalSnapshot.iter_occurrences

    def spy_iter(self, *, max_occurrences=None):
        nonlocal yields
        for occ in original(self, max_occurrences=max_occurrences):
            yields += 1
            yield occ

    monkeypatch.setattr(OccurrenceTraversalSnapshot, "iter_occurrences", spy_iter)

    occurrences = repo.query(Definition(IndexLeaf, "shared")).nested(refresh=False).execute()

    assert occurrences.first().definition == child.definition
    assert yields == 1


def test_query_max_occurrences_stops_path_generation(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = IndexLeaf("shared", repo=repo)
    repo.save_object(IndexPersistent([child, child, child], repo=repo))
    yields = 0
    original = OccurrenceTraversalSnapshot.iter_occurrences

    def spy_iter(self, *, max_occurrences=None):
        nonlocal yields
        for occ in original(self, max_occurrences=max_occurrences):
            yields += 1
            yield occ

    monkeypatch.setattr(OccurrenceTraversalSnapshot, "iter_occurrences", spy_iter)

    occurrences = (
        repo.query(Definition(IndexLeaf, "shared"))
        .nested(refresh=False)
        .max_occurrences(1)
        .execute()
    )

    assert occurrences.count() == 1
    assert yields == 1


def test_selective_occurrence_query_does_not_scan_unrelated_roots(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    wanted = IndexLeaf("wanted", repo=repo)
    repo.save_object(IndexPersistent(wanted, repo=repo))
    for idx in range(6):
        repo.save_object(IndexPersistent(IndexLeaf(f"other-{idx}", repo=repo), repo=repo))

    def fail_root_scan(*args, **kwargs):
        raise AssertionError("selective occurrences should not scan stored roots")

    monkeypatch.setattr(repo._query_catalog, "iter_all_occurrences", fail_root_scan)

    occurrences = repo.query(Definition(IndexLeaf, "wanted")).nested(refresh=False).execute()

    assert occurrences.count() == 1
    assert occurrences.one().definition == wanted.definition


def test_refresh_failure_rolls_back_catalog_atomically(tmp_path):
    good_store = DirStore(tmp_path / "good")
    repo = Repo(stores=good_store)
    good = IndexLeaf("good", repo=repo)
    repo.save_object(good)

    old_results = repo.find_defs(None)
    assert list(old_results) == [good.definition]

    repo.add_store(BadIndexStore([good.definition, "not-a-cdef"]))
    before = catalog_state(repo)

    with pytest.raises(QueryIndexError):
        repo.find_defs(None, refresh=True)

    assert catalog_state(repo) == before
    assert list(repo.find_defs(None, refresh=False)) == [good.definition]

    repo.stores = [good_store]
    assert list(repo.find_defs(None, refresh=True)) == [good.definition]


def test_forced_refresh_builds_replacement_without_snapshot_helper(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = IndexLeaf("stored", repo=repo)
    repo.save_object(obj)
    catalog = repo._query_catalog

    assert not hasattr(catalog, "_snapshot_locked")
    assert list(repo.find_defs(None, refresh=True)) == [obj.definition]


def test_forced_refresh_failure_keeps_live_catalog_without_snapshot(tmp_path):
    good_store = DirStore(tmp_path / "good")
    repo = Repo(stores=good_store)
    good = IndexLeaf("good", repo=repo)
    repo.save_object(good)
    before = catalog_state(repo)

    repo.add_store(BadIndexStore(["not-a-cdef"]))

    with pytest.raises(QueryIndexError):
        repo.find_defs(None, refresh=True)

    assert catalog_state(repo) == before
    assert list(repo.find_defs(None, refresh=False)) == [good.definition]


def test_catalog_snapshot_is_stable_across_registration():
    repo = Repo()
    first = IndexLeaf("first", repo=repo)
    repo.add_objects(first)
    snapshot = repo._query_catalog.snapshot()
    before_ids = snapshot.all_definition_ids()

    second = IndexLeaf("second", repo=repo)
    repo.add_objects(second)
    second_id = repo._query_catalog.cdef_id(second.definition)

    assert snapshot.all_definition_ids() == before_ids
    assert second_id not in snapshot.all_definition_ids()
    assert snapshot.ids_to_cdefs({second_id}) == ()


def test_catalog_snapshot_is_stable_across_forced_refresh(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = IndexLeaf("stored", repo=repo)
    repo.save_object(obj)
    repo.find_defs(None, refresh=True)
    snapshot = repo._query_catalog.snapshot()
    obj_id = snapshot.cdef_id(obj.definition)

    shutil.rmtree(store.object_dir(obj.definition))
    assert repo.find_defs(None, refresh=True).count() == 0

    assert snapshot.ids_to_cdefs({obj_id}) == (obj.definition,)


def test_occurrence_result_does_not_gain_new_owner_after_execute(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = IndexLeaf("child", repo=repo)
    first_parent = IndexPersistent(child, state=1, repo=repo)
    repo.save_object(first_parent)

    occurrences = repo.query(Definition(IndexLeaf, "child")).nested(refresh=False).execute()
    second_parent = IndexPersistent(child, state=2, repo=repo)
    repo.save_object(second_parent)

    assert {occ.owner for occ in occurrences} == {first_parent.definition}


def test_occurrence_result_survives_owner_deletion_after_execute(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = IndexLeaf("child", repo=repo)
    parent = IndexPersistent(child, repo=repo)
    repo.save_object(parent)

    occurrences = repo.query(Definition(IndexLeaf, "child")).nested(refresh=False).execute()
    shutil.rmtree(store.object_dir(parent.definition))
    assert repo.find_defs(None, refresh=True).count() == 0

    assert occurrences.count() == 1
    assert occurrences.one().owner == parent.definition


def test_occurrence_result_repeated_iteration_is_stable(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = IndexLeaf("child", repo=repo)
    parent = IndexPersistent([child, child], repo=repo)
    repo.save_object(parent)

    occurrences = repo.query(Definition(IndexLeaf, "child")).nested(refresh=False).execute()

    first = tuple(occ.path for occ in occurrences)
    second = tuple(occ.path for occ in occurrences)

    assert first == second
    assert len(first) == 2


def test_occurrence_first_count_and_iteration_use_same_snapshot(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = IndexLeaf("child", repo=repo)
    parent = IndexPersistent([child, child], repo=repo)
    repo.save_object(parent)

    occurrences = repo.query(Definition(IndexLeaf, "child")).nested(refresh=False).execute()
    first = occurrences.first()
    repo.save_object(IndexPersistent(child, state=99, repo=repo))

    assert first.owner == parent.definition
    assert occurrences.count() == 2
    assert {occ.owner for occ in occurrences} == {parent.definition}


def test_auto_hydration_failure_leaves_catalog_unchanged_and_retries(tmp_path):
    good_store = DirStore(tmp_path / "good")
    good_repo = Repo(stores=good_store)
    good = IndexLeaf("good", repo=good_repo)
    good_repo.save_object(good)

    bad_store = BadIndexStore(["not-a-cdef"])
    repo = Repo(stores=[DirStore(good_store.base_dir), bad_store])
    before = catalog_state(repo)

    with pytest.raises(QueryIndexError):
        repo.find_defs(None)

    assert catalog_state(repo) == before
    assert repo._query_catalog.hydrated_stores == set()

    repo.stores = [DirStore(good_store.base_dir)]
    assert list(repo.find_defs(None)) == [good.definition]


def test_auto_hydration_fingerprints_each_new_cdef_once(tmp_path, monkeypatch):
    from dryml.core2.query import index as index_mod

    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = IndexLeaf("child", repo=repo)
    parent = IndexPersistent(child, repo=repo)
    repo.save_object(parent)

    calls = []
    original = index_mod.target_local_fingerprint

    def spy(cdef):
        calls.append(cdef)
        return original(cdef)

    monkeypatch.setattr(index_mod, "target_local_fingerprint", spy)
    repo2 = Repo(stores=DirStore(store.base_dir))

    assert list(repo2.find_defs(None)) == [parent.definition]
    assert calls.count(parent.definition) == 1
    assert calls.count(child.definition) == 1


def test_auto_hydration_builds_each_store_graph_once(tmp_path, monkeypatch):
    from dryml.core2.query import index as index_mod

    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    parent = IndexPersistent(IndexLeaf("child", repo=repo), repo=repo)
    repo.save_object(parent)
    calls = []
    original = index_mod.ConcreteDefinitionGraph.from_roots

    def spy_from_roots(cls, cdefs):
        cdefs = tuple(cdefs)
        calls.append(cdefs)
        return original(cdefs)

    monkeypatch.setattr(index_mod.ConcreteDefinitionGraph, "from_roots", classmethod(spy_from_roots))
    repo2 = Repo(stores=DirStore(store.base_dir))

    assert list(repo2.find_defs(None)) == [parent.definition]
    assert len(calls) == 1


def test_nested_definition_inside_set_is_indexed_with_defined_path(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = IndexLeaf("set-child", repo=repo)
    parent = IndexPersistent({child}, repo=repo)
    repo.save_object(parent)

    repo2 = Repo(stores=DirStore(store.base_dir))
    occurrences = repo2.find_occurrences(Definition(IndexLeaf, SKIP_ARGS))

    assert occurrences.count() == 1
    assert occurrences.one().definition == child.definition
    assert isinstance(occurrences.one().path[-1], SetMember)
    assert get_subtree(parent.definition, occurrences.one().path) == child.definition


def test_forced_refresh_removes_deleted_root_and_occurrences(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = IndexLeaf("child", repo=repo)
    parent = IndexPersistent(child, repo=repo)
    repo.save_object(parent)

    repo2 = Repo(stores=DirStore(store.base_dir))
    assert repo2.find_defs(None).count() == 1
    assert repo2.find_occurrences(Definition(IndexLeaf, SKIP_ARGS)).count() == 1

    shutil.rmtree(store.object_dir(parent.definition))

    assert repo2.find_defs(None, refresh=True).count() == 0
    assert repo2.find_occurrences(Definition(IndexLeaf, SKIP_ARGS), refresh=False).count() == 0


def test_exact_store_probe_confirms_persisted_definition_after_hash_hit(tmp_path, monkeypatch):
    monkeypatch.setattr(ConcreteDefinition, "stable_hash", lambda self: "collision")
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    stored = IndexLeaf("stored", repo=repo)
    queried = IndexLeaf("queried", repo=repo)
    repo.save_object(stored)

    repo2 = Repo(stores=DirStore(store.base_dir))

    assert repo2.query(queried.definition).stored().count() == 0
    assert repo2.query(stored.definition).stored().count() == 1


def catalog_state(repo):
    catalog = repo._query_catalog
    return {
        "definitions": tuple(sorted(catalog.definitions_by_id.keys())),
        "replicas": tuple(sorted((k, tuple(sorted(v))) for k, v in catalog.replicas_by_definition.items())),
        "stored_by_store": tuple(sorted((k, tuple(sorted(v))) for k, v in catalog.stored_definitions_by_store.items())),
        "local_postings": tuple(sorted((repr(k), tuple(sorted(v.items()))) for k, v in catalog.local_postings.items())),
        "edges": tuple(sorted((k[0], str(k[1]), k[2]) for k in catalog.edge_by_key.keys())),
        "outgoing_edges": tuple(sorted((k, tuple(sorted((edge[0], str(edge[1]), edge[2]) for edge in v))) for k, v in catalog.outgoing_edges.items())),
        "incoming_edges": tuple(sorted((k, tuple(sorted((edge[0], str(edge[1]), edge[2]) for edge in v))) for k, v in catalog.incoming_edges.items())),
        "child_by_parent_path": tuple(sorted(((k[0], str(k[1])), tuple(sorted(v))) for k, v in catalog.child_by_parent_path.items())),
        "parents_by_child_path": tuple(sorted(((k[0], str(k[1])), tuple(sorted(v))) for k, v in catalog.parents_by_child_path.items())),
        "hydrated": tuple(sorted(catalog.hydrated_stores)),
        "store_by_id": tuple(sorted(catalog.store_by_id.keys())),
        "light_index": tuple(sorted(cdef.stable_hash() for cdef in repo.light_index)),
        "generation": catalog.generation,
    }
