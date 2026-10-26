import pytest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

from dryml.core import Object, Repo, Serializable
from dryml.core.store.dir import DirStore
from dryml.core.utils.general import pickle_load, pickle_save


class MultiLeaf(Object):
    def __init__(self, name):
        super().__init__()
        self.name = name


class MultiState(Serializable):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.value = 0

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        pickle_save(self.value, f"{dest_dir}/value.pkl")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        self.value = pickle_load(f"{src_dir}/value.pkl")


def test_distinct_cdefs_in_two_stores_produce_two_v3_identities(tmp_path):
    store1 = DirStore(tmp_path / "store1")
    store2 = DirStore(tmp_path / "store2")
    repo = Repo(stores=[store1, store2])
    first = MultiLeaf("first", repo=repo)
    second = MultiLeaf("second", repo=repo)
    repo.save_object(first, store=store1)
    repo.save_object(second, store=store2)

    repo2 = Repo(stores=[DirStore(store1.base_dir), DirStore(store2.base_dir)])

    assert repo2.query().cdefs().stored().count() == 2


def test_same_cdef_in_two_stores_deduplicates_and_tracks_sources(tmp_path):
    store1 = DirStore(tmp_path / "store1")
    store2 = DirStore(tmp_path / "store2")
    repo = Repo(stores=[store1, store2])
    obj = MultiLeaf("same", repo=repo)
    repo.save_object(obj, store=store1)
    repo.save_object(obj, store=store2)

    repo2 = Repo(stores=[DirStore(store1.base_dir), DirStore(store2.base_dir)])
    results = repo2.query().cdefs().stored().collect()

    assert results.count() == 1 and results.one().graph_equal(obj.definition)
    assert len(results.sources(obj.definition)) == 2


def test_equivalent_store_instances_share_physical_identity(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = MultiLeaf("same", repo=repo)
    repo.save_object(obj)

    repo2 = Repo(stores=[DirStore(store.base_dir), DirStore(store.base_dir)])
    results = repo2.query().cdefs().stored().collect()

    assert results.count() == 1 and results.one().graph_equal(obj.definition)
    assert len(results.sources(obj.definition)) == 1


def test_v3_capture_deduplicates_equivalent_store_instances(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = MultiLeaf("same", repo=repo)
    repo.save_object(obj)

    from dryml.core.query import authority
    original = authority.capture_store_projection
    calls = []
    def capture(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(authority, "capture_store_projection", capture)
    repo2 = Repo(stores=[
        DirStore(store.base_dir, query_index="memory"),
        DirStore(store.base_dir, query_index="memory"),
    ])

    assert repo2.query().cdefs().stored().count() == 1
    assert calls == [1]


def test_v3_query_uses_replaced_connected_store_reference(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = MultiLeaf("same", repo=repo)
    repo.save_object(obj)

    old_store = DirStore(store.base_dir)
    repo2 = Repo(stores=old_store)
    assert repo2.query().cdefs().stored().count() == 1

    new_store = DirStore(store.base_dir)
    repo2.stores = [new_store]
    refreshed = repo2.query().cdefs().stored().collect()

    assert len(refreshed.sources(obj.definition)) == 1


def test_structural_query_is_fresh_while_exact_state_ref_load_restores_state(tmp_path):
    store1 = DirStore(tmp_path / "store1")
    store2 = DirStore(tmp_path / "store2")
    repo = Repo(stores=[store1, store2])
    obj = MultiState("same", repo=repo)
    obj.value = 1
    first_state = repo.save_object(obj, store=store1)
    obj.value = 2
    second_state = repo.save_object(obj, store=store2, deep_capture=True)

    repo2 = Repo(stores=[DirStore(store2.base_dir), DirStore(store1.base_dir)])
    structural = repo2.load_or_build(obj.definition, cache="none")
    exact = repo2.load_state_ref(second_state, reuse_live="never")

    assert first_state != second_state
    assert structural.value == 0
    assert exact.value == 2


def test_result_order_is_independent_of_store_order(tmp_path):
    store1 = DirStore(tmp_path / "store1")
    store2 = DirStore(tmp_path / "store2")
    repo = Repo(stores=[store1, store2])
    first = MultiLeaf("first", repo=repo)
    second = MultiLeaf("second", repo=repo)
    repo.save_object(first, store=store1)
    repo.save_object(second, store=store2)

    order_a = list(Repo(stores=[DirStore(store1.base_dir), DirStore(store2.base_dir)]).query().cdefs().stored().collect())
    order_b = list(Repo(stores=[DirStore(store2.base_dir), DirStore(store1.base_dir)]).query().cdefs().stored().collect())

    assert len(order_a) == len(order_b) == 2
    assert all(a.graph_equal(b) for a, b in zip(order_a, order_b))


@pytest.mark.timeout(30)
def test_u7_reversed_store_order_acquires_unique_sorted_retained_cuts(tmp_path, monkeypatch):
    """Concurrent reversed producers share canonical lock order and replica proof."""

    from contextlib import contextmanager
    from dataclasses import replace
    import threading
    from dryml.core import Definition

    stores = tuple(DirStore(tmp_path / name, query_index="sqlite") for name in ("a", "b"))
    values = sorted((Definition(MultiLeaf, str(i)).concretize() for i in range(3)),
                    key=lambda value: value.exact_digest())
    for store in stores:
        for value in values:
            store.publish_node(value)
    readiness = threading.Barrier(2, timeout=10)
    local = threading.local()
    orders, results, errors = {}, [], []

    for store in stores:
        key = store.authority_fence_key()
        original_fence = store.authority_read_fence
        index = store.open_query_index()
        original_status = index.status

        @contextmanager
        def fence(key=key, original=original_fence):
            held = getattr(local, "held", [])
            outer = key not in held
            if outer:
                assert not held or key > held[-1]
                orders.setdefault(threading.get_ident(), []).append(key)
            with original():
                local.held = [*held, key]
                try:
                    yield
                finally:
                    local.held = held

        def status(original=original_status):
            assert not getattr(local, "held", [])
            result = original()
            local.ready = getattr(local, "ready", 0) + 1
            if local.ready == len(stores):
                readiness.wait()
            return result

        monkeypatch.setattr(store, "authority_read_fence", fence)
        monkeypatch.setattr(index, "status", status)
        monkeypatch.setattr(index, "_ensure_ready", lambda: pytest.fail("repairing entry inside cut"))
    monkeypatch.setattr("dryml.core.query.authority.capture_store_projection", lambda *a, **kw: pytest.fail("ready replicas enumerated authority"))

    def query(selected):
        try:
            bounded = replace(Repo(selected).query().cdefs().stored().refresh(False).require_indexed().max_verify(1), take_limit=1)
            results.append(bounded._run_terminal(bounded._collect_with_capture, return_capture=True))
        except BaseException as error:
            errors.append(error)
        finally:
            for store in stores:
                store.open_query_index()._connections.close_current()

    duplicate = DirStore(stores[0].base_dir, query_index="sqlite")
    workers = [threading.Thread(target=query, args=(selected,), daemon=True)
               for selected in ((stores[0], duplicate, stores[1]), tuple(reversed(stores)))]
    for worker in workers:
        worker.start()
    try:
        for worker in workers:
            worker.join(10)
    finally:
        readiness.abort()
        for worker in workers:
            worker.join(10)
    assert all(not worker.is_alive() for worker in workers) and not errors
    assert len(orders) == 2
    assert all(set(order) == {store.authority_fence_key() for store in stores} for order in orders.values())
    for result, capture in results:
        assert result.one().graph_equal(values[0]) and len(result.sources(result.one())) == 2
        assert capture.verified_count == 1 and capture.store_scan_count == 0
        assert capture.candidate_rows_read == capture.cdef_blobs_decoded == 4
    assert all(not store.open_query_index()._connections._leases for store in stores)
