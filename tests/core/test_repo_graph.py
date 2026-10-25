import pytest

from dryml.core import Definition, Object, Repo, Serializable
from dryml.core.cdef_identity import cdef_node_key
from dryml.core.object import WorkspaceCapable
from dryml.core.query.path import GraphPath, Index, Key, Parameter, SetMember, get_subtree
from dryml.core.repo import RepoGraphError, RepoSaveError, default_repo, get_default_repo
from dryml.core.repo_plan import GraphApplyResult, collect_runtime_roots
from dryml.core.store.dir import DirStore
from dryml.core.utils.general import pickle_load, pickle_save
from dryml.core.workspaces import WorkspaceManager
from dryml.core._content_facts import _primitive_size
from tests.core.test_repo_aliases import _saved_digest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")


def _stage_test_fact(cut, source, *, context=("test-primitive", 1), prepare=None):
    """Exercise the narrow prepared-content interface without domain shortcuts."""
    import sys
    cut.stage_prepared(context, source, sys.getsizeof(source),
                       (lambda: source) if prepare is None else prepare)


def test_content_pending_budget_nested_cuts_and_admission_bypass():
    """All active cuts share slots, and a denied reservation allocates no copy."""
    repo = Repo()
    facility = repo._validation_evidence
    copies = []
    with repo._read_cut() as outer:
        for i in range(257):
            source = str(i).encode()
            _stage_test_fact(outer, source, prepare=lambda value=source: (copies.append(value), value)[1])
        assert len(copies) == 256
        assert len(facility._pending) == 256
        assert not facility._resident
        with repo._read_cut() as nested:
            _stage_test_fact(nested, b"nested", prepare=lambda: pytest.fail("budget bypass copied"))
        borrower = Repo._for_state_io((), _content_owner=repo)
        with borrower._read_cut() as borrowed:
            _stage_test_fact(borrowed, b"borrowed", prepare=lambda: pytest.fail("borrowed bypass copied"))
        borrower.close(flush=False)
        assert facility.high_water_entries == 256
        assert facility.accounted_bytes <= facility.max_bytes
    assert len(facility._resident) == 256
    assert not facility._pending
    with repo._read_cut() as cut:
        _stage_test_fact(cut, b"new")
    assert len(facility._resident) == 256
    with repo._read_cut() as cut:
        assert cut.prepared(("test-primitive", 1), b"0") is None
        assert cut.prepared(("test-primitive", 1), b"new") == b"new"


def test_content_exact_combined_entry_and_total_byte_limits():
    """Key/source/value/bookkeeping, including pending copies, obey exact limits."""
    import sys
    repo = Repo()
    facility = repo._validation_evidence
    context = ("near-limit", 1)
    source = b""
    overhead = 2048 + _primitive_size(context) + sys.getsizeof(source) + sys.getsizeof(b"")
    value = b"x" * (facility.max_entry_bytes - overhead)
    with repo._read_cut() as cut:
        for i in range(17):
            # Distinct source bytes keep lookup buckets distinct while holding
            # the combined charge exactly at the single-entry ceiling.
            current = bytes([i])
            cut.stage_prepared(context, current, sys.getsizeof(value) - 1,
                               lambda: value[:-1])
        assert len(facility._pending) == 16
        assert facility.accounted_bytes == facility.max_bytes
        assert facility.high_water_bytes == facility.max_bytes
        cut.stage_prepared(context, source, sys.getsizeof(value) + 1,
                           lambda: pytest.fail("oversized fact copied"))
    assert len(facility._resident) == 16
    with repo._read_cut() as cut:
        cut.stage_prepared(context, b"later", sys.getsizeof(b"small"), lambda: b"small")
    assert len(facility._resident) == 16
    assert facility.accounted_bytes < facility.max_bytes


def test_content_lru_touch_and_forced_bucket_collisions(monkeypatch):
    """Four candidates, exact fresh bytes and context, never hash equality alone."""
    import dryml.core.repo as repo_module
    monkeypatch.setattr(repo_module, "_content_hash", lambda source: b"collision")
    repo = Repo()
    for source in (b"one", b"two", b"three", b"four"):
        with repo._read_cut() as cut:
            _stage_test_fact(cut, source)
    with repo._read_cut() as cut:
        assert cut.prepared(("test-primitive", 1), b"one") == b"one"
        assert cut.prepared(("test-primitive", 2), b"one") is None
        assert cut.prepared(("different-kind", 1), b"one") is None
        assert cut.prepared(("test-primitive", 1), b"other") is None
        _stage_test_fact(cut, b"five")
    with repo._read_cut() as cut:
        assert cut.prepared(("test-primitive", 1), b"two") is None
        assert cut.prepared(("test-primitive", 1), b"one") == b"one"
        assert cut.prepared(("test-primitive", 1), b"five") == b"five"
    assert len(repo._validation_evidence._resident) == 4
    assert repo._validation_evidence.high_water_entries == 4


def test_prepared_implementation_identity_does_not_keep_callable_alive():
    """Weak implementation identities cannot substitute a new or expired parser."""
    import weakref
    repo = Repo()
    def first():
        pass
    def second():
        pass
    identity = weakref.ref(first)
    with repo._read_cut() as cut:
        _stage_test_fact(cut, b"same", context=("implementation", identity))
    with repo._read_cut() as cut:
        assert cut.prepared(("implementation", weakref.ref(second)), b"same") is None
    del first
    assert identity() is None
    with repo._read_cut() as cut:
        assert cut.prepared(("implementation", identity), b"same") is None


def test_clear_during_preparation_keeps_charge_and_cannot_repopulate():
    """Epoch clear retires in-flight facts while preserving their reservations."""
    repo = Repo()
    facility = repo._validation_evidence
    with repo._read_cut() as cut:
        def prepare():
            assert not facility._lock._is_owned()
            before = facility.accounted_bytes
            assert before > 0
            repo.clear_cache(strong=False, weak=False)
            assert facility.accounted_bytes == before
            assert not facility._resident
            return b"value"
        _stage_test_fact(cut, b"value", prepare=prepare)
        assert facility.accounted_bytes == 0
        with pytest.raises(RuntimeError, match="cleared"):
            cut.check()
    assert not facility._pending and not facility._resident


def test_concurrent_misses_reserve_before_copy_and_do_not_hold_lock():
    """A second thread can miss during preparation, but cannot exceed the budget."""
    from threading import Event, Thread
    repo = Repo()
    facility = repo._validation_evidence
    facility.max_entries = 1
    entered, release = Event(), Event()
    failures = []

    def worker():
        try:
            with repo._read_cut() as cut:
                def prepare():
                    assert not facility._lock._is_owned()
                    entered.set()
                    assert release.wait(5)
                    return b"same"
                _stage_test_fact(cut, b"same", prepare=prepare)
        except BaseException as error:
            failures.append(error)

    thread = Thread(target=worker)
    thread.start()
    try:
        assert entered.wait(5)
        with repo._read_cut() as cut:
            assert cut.prepared(("test-primitive", 1), b"same") is None
            _stage_test_fact(cut, b"same", prepare=lambda: pytest.fail("concurrent bypass copied"))
        assert len(facility._pending) == 1 and not facility._resident
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and not failures
    assert facility.high_water_entries == 1
    assert len(facility._resident) == 1 and not facility._pending


def test_failed_scope_discards_staged_facts_and_partial_authority():
    """Even a caught substep failure prevents admission from its owning cut."""
    repo = Repo()
    with repo._read_cut() as cut:
        _stage_test_fact(cut, b"first")
        cut.snapshots["partial"] = object()
        cut.resolved_state_refs["partial"] = object()
        cut.validated_state_refs.append(object())
        cut.discard()
        _stage_test_fact(cut, b"later", prepare=lambda: pytest.fail("failed scope prepared"))
        assert not cut.snapshots and not cut.resolved_state_refs and not cut.validated_state_refs
    assert repo._validation_evidence.accounted_bytes == 0
    assert not repo._validation_evidence._resident


@pytest.mark.parametrize("caught", [False, True])
def test_prepare_exception_releases_reservations_and_poisoned_scope(tmp_path, monkeypatch, caught):
    """A failed optional copy releases all charges even if its caller catches it."""
    from contextlib import contextmanager

    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    facility = repo._validation_evidence
    error = RuntimeError("retention copy failed")
    events = []
    original = repo._authority_read_fences

    @contextmanager
    def fences(stores):
        with original(stores):
            events.append("enter")
            try:
                yield
            finally:
                events.append("exit")

    monkeypatch.setattr(repo, "_authority_read_fences", fences)

    def prepare():
        assert not facility._lock._is_owned()
        assert len(facility._pending) == 2 and facility.accounted_bytes > 0
        raise error

    def operation():
        with repo._read_cut() as cut:
            _stage_test_fact(cut, b"first")
            if caught:
                with pytest.raises(RuntimeError) as failed:
                    _stage_test_fact(cut, b"second", prepare=prepare)
                assert failed.value is error and cut._failed
                assert not cut._pending and facility.accounted_bytes == 0
                _stage_test_fact(cut, b"later", prepare=lambda: pytest.fail("failed cut prepared"))
            else:
                _stage_test_fact(cut, b"second", prepare=prepare)

    if caught:
        operation()
    else:
        with pytest.raises(RuntimeError) as failed:
            operation()
        assert failed.value is error
    assert events == ["enter", "exit"]
    assert repo._topology_leases == 0
    assert not facility._resident and not facility._pending
    assert facility.accounted_bytes == 0
    repo.close(flush=False)


def test_borrowed_facility_distinct_cuts_lifetime_clear_and_detach():
    """Borrowers lease origin lifetime; only explicit clear shares the epoch."""
    owner = Repo()
    with owner._read_cut() as cut:
        _stage_test_fact(cut, b"retained")
    facility = owner._validation_evidence
    borrower = Repo._for_state_io((), _content_owner=owner)
    assert borrower._validation_evidence is facility
    assert owner._topology_leases == 1
    with pytest.raises(RuntimeError, match="topology lease"):
        owner.close(flush=False)
    with owner._read_cut() as outer, borrower._read_cut() as inner:
        assert inner.repo is borrower and outer.repo is owner
        assert inner.snapshots is not outer.snapshots
        assert inner.prepared(("test-primitive", 1), b"retained") == b"retained"
        owner.clear_cache(strong=False, weak=False)
        for cut in (inner, outer):
            with pytest.raises(RuntimeError, match="cleared"):
                cut.check()
    with owner._read_cut() as cut:
        _stage_test_fact(cut, b"again")
    borrower.close(flush=False)
    assert borrower._validation_evidence is not facility
    assert owner._topology_leases == 0
    assert len(facility._resident) == 1
    with Repo()._read_cut() as independent:
        assert independent.prepared(("test-primitive", 1), b"again") is None
    owner.close(flush=False)
    assert not facility._resident and facility.accounted_bytes == 0


def test_failed_owner_close_leaves_open_empty_facility(monkeypatch):
    """Accepted close clears before flush; retry remains possible after failure."""
    repo = Repo()
    with repo._read_cut() as cut:
        _stage_test_fact(cut, b"retained")
    original = repo.flush
    monkeypatch.setattr(repo, "flush", lambda: (_ for _ in ()).throw(RuntimeError("flush failed")))
    with pytest.raises(RuntimeError, match="flush failed"):
        repo.close()
    assert not repo._closed and not repo._closing
    assert not repo._validation_evidence._resident
    assert repo._validation_evidence.accounted_bytes == 0
    monkeypatch.setattr(repo, "flush", original)
    repo.close(flush=False)


def test_duplicate_concurrent_misses_share_accounting_not_authority():
    """Concurrent preparation may duplicate, with both candidates charged."""
    from threading import Barrier, Thread
    repo = Repo()
    facility = repo._validation_evidence
    barrier = Barrier(2)
    failures = []
    def worker():
        try:
            with repo._read_cut() as cut:
                def prepare():
                    assert not facility._lock._is_owned()
                    barrier.wait(timeout=5)
                    return b"same"
                _stage_test_fact(cut, b"same", prepare=prepare)
        except BaseException as error:
            failures.append(error)
    threads = [Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert all(not thread.is_alive() for thread in threads) and not failures
    assert len(facility._resident) == 2 and not facility._pending
    assert facility.accounted_bytes == sum(fact.charge for fact in facility._resident.values())
    assert facility.high_water_entries == 2
    assert repo._topology_leases == 0


def test_failed_borrower_close_keeps_owner_lease_and_content(monkeypatch):
    """A borrower cleanup failure cannot clear or release the originating Repo."""
    owner = Repo()
    with owner._read_cut() as cut:
        _stage_test_fact(cut, b"retained")
    facility = owner._validation_evidence
    borrower = Repo._for_state_io((), _content_owner=owner)
    original = borrower._query_index.close
    def fail():
        raise RuntimeError("borrower cleanup failed")
    monkeypatch.setattr(borrower._query_index, "close", fail)
    with pytest.raises(RuntimeError, match="borrower cleanup failed"):
        borrower.close(flush=False)
    assert not borrower._closing and not borrower._closed
    assert owner._topology_leases == 1 and len(facility._resident) == 1
    monkeypatch.setattr(borrower._query_index, "close", original)
    borrower.close(flush=False)
    assert owner._topology_leases == 0 and len(facility._resident) == 1


def test_content_foreign_thread_rejected_before_facility_bookkeeping():
    """Foreign reads, staging and unwind leave owner reservations/resources intact."""
    from threading import Thread
    repo = Repo()
    errors = []
    with repo._read_cut() as cut:
        _stage_test_fact(cut, b"pending")
        def foreign():
            for operation in (
                lambda: cut.prepared(("test-primitive", 1), b"pending"),
                lambda: _stage_test_fact(cut, b"foreign"),
                lambda: cut.__exit__(None, None, None),
            ):
                try:
                    operation()
                except RuntimeError as error:
                    errors.append(str(error))
        thread = Thread(target=foreign)
        thread.start()
        thread.join(5)
        assert not thread.is_alive()
        assert len(errors) == 3 and all("thread" in error for error in errors)
        assert len(repo._validation_evidence._pending) == 1
        assert repo._topology_leases == 1
    assert len(repo._validation_evidence._resident) == 1


def test_content_inherited_owner_and_borrower_rejected_before_locks():
    """Inherited content access cannot acquire either Repo's or the facility lock."""
    import os
    if not hasattr(os, "fork"):
        pytest.skip("requires fork")
    owner = Repo()
    borrower = Repo._for_state_io((), _content_owner=owner)
    with borrower._read_cut() as cut:
        _stage_test_fact(cut, b"pending")
        pid = os.fork()
        if pid == 0:
            class ForbiddenLock:
                def __enter__(self):
                    os._exit(2)
                def __exit__(self, *args):
                    os._exit(3)
            owner._configuration_lock = borrower._configuration_lock = owner._validation_evidence._lock = ForbiddenLock()
            for operation in (
                lambda: cut.prepared(("test-primitive", 1), b"pending"),
                lambda: _stage_test_fact(cut, b"foreign"),
                lambda: cut.__exit__(None, None, None),
                lambda: borrower._read_cut().__enter__(),
                lambda: Repo._for_state_io((), _content_owner=owner),
                borrower.clear_cache, owner._validation_evidence.clear,
                lambda: borrower.close(flush=False),
            ):
                try:
                    operation()
                except RuntimeError:
                    continue
                except BaseException:
                    os._exit(4)
                os._exit(5)
            os._exit(0)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        cut.check()
    borrower.close(flush=False)
    assert owner._topology_leases == 0


class GraphLeaf(Object):
    def __init__(self, name):
        super().__init__()
        self.name = name


class GraphNode(Object):
    def __init__(self, name, *children):
        super().__init__()
        self.name = name
        self.children = children


class WorkspaceLeaf(Object, WorkspaceCapable):
    def __init__(self, name):
        super().__init__()
        self.name = name


class RepoAware(Object):
    def __init__(self):
        super().__init__()
        self.seen_repo = get_default_repo()


class PersistentLeaf(Serializable):
    def __init__(self, name, *, value=0):
        super().__init__()
        self.name = name
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        pickle_save(self.value, f"{dest_dir}/value.pkl")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        self.value = pickle_load(f"{src_dir}/value.pkl")


class PersistentNode(Serializable):
    def __init__(self, name, child):
        super().__init__()
        self.name = name
        self.child = child


def test_iter_graph_walks_rooted_object_graph_postorder():
    repo = Repo()
    left = GraphLeaf("left", repo=repo)
    right = GraphLeaf("right", repo=repo)
    root = GraphNode("root", left, right, repo=repo)

    assert [obj.name for obj in repo.iter_graph(root)] == ["left", "right", "root"]


def test_iter_graph_can_exclude_root_and_dedupes_shared_children():
    repo = Repo()
    leaf = GraphLeaf("leaf", repo=repo)
    root = GraphNode("root", leaf, leaf, repo=repo)

    assert [obj.name for obj in repo.iter_graph(root, include_root=False)] == ["leaf"]


def test_iter_graph_occurrence_mode_visits_shared_child_twice():
    repo = Repo()
    leaf = GraphLeaf("leaf", repo=repo)
    root = GraphNode("root", leaf, leaf, repo=repo)

    assert [obj.name for obj in repo.iter_graph(root, include_root=False, dedupe=False)] == ["leaf", "leaf"]


def test_iter_graph_preorder_visits_parent_before_children():
    repo = Repo()
    left = GraphLeaf("left", repo=repo)
    right = GraphLeaf("right", repo=repo)
    root = GraphNode("root", left, right, repo=repo)

    assert [obj.name for obj in repo.iter_graph(root, order="pre")] == ["root", "left", "right"]


def test_apply_graph_returns_results_by_definition_in_visit_order():
    repo = Repo()
    left = GraphLeaf("left", repo=repo)
    right = GraphLeaf("right", repo=repo)
    root = GraphNode("root", left, right, repo=repo)

    results = repo.apply_graph(root, lambda obj: obj.name)

    assert list(results.values()) == ["left", "right", "root"]


def test_apply_graph_occurrence_mode_preserves_repeated_results():
    repo = Repo()
    leaf = GraphLeaf("leaf", repo=repo)
    root = GraphNode("root", leaf, leaf, repo=repo)

    results = repo.apply_graph(root, lambda obj: obj.name, include_root=False, dedupe=False)

    assert all(isinstance(result, GraphApplyResult) for result in results)
    assert [result.value for result in results] == ["leaf", "leaf"]
    assert [str(result.path) for result in results] == ["$[@param(\"children\")][0]", "$[@param(\"children\")][1]"]


def test_repo_occurrence_path_preserves_arg_segment():
    repo = Repo()
    leaf = GraphLeaf("leaf", repo=repo)
    root = GraphNode("root", leaf, repo=repo)

    result = repo.apply_graph(root, lambda obj: obj.name, include_root=False, dedupe=False)[0]

    assert isinstance(result.path[0], Parameter)
    assert get_subtree(root.definition, result.path) == leaf.definition


def test_repo_occurrence_path_preserves_mapping_key_segment():
    repo = Repo()
    leaf = GraphLeaf("leaf", repo=repo)
    root = GraphNode("root", {5: leaf}, repo=repo)

    result = repo.apply_graph(root, lambda obj: obj.name, include_root=False, dedupe=False)[0]

    assert isinstance(result.path[0], Parameter)
    assert result.path[1] == Index(0)
    assert result.path[2] == Key(5)
    assert get_subtree(root.definition, result.path) == leaf.definition


def test_repo_occurrence_path_preserves_set_member_segment():
    repo = Repo()
    leaf = GraphLeaf("leaf", repo=repo)
    root = GraphNode("root", {leaf}, repo=repo)

    result = repo.apply_graph(root, lambda obj: obj.name, include_root=False, dedupe=False)[0]

    assert isinstance(result.path[0], Parameter)
    assert isinstance(result.path[1], Index)
    assert isinstance(result.path[2], SetMember)
    assert get_subtree(root.definition, result.path) == leaf.definition


def test_runtime_root_path_for_integer_dict_key_is_key_not_index():
    leaf = GraphLeaf("leaf")

    roots = collect_runtime_roots({5: leaf})

    assert roots[0].path == GraphPath((Key(5),))


def test_runtime_root_set_paths_are_stable_set_members():
    leaf = GraphLeaf("leaf")

    first = collect_runtime_roots({leaf})
    second = collect_runtime_roots({leaf})

    assert isinstance(first[0].path[0], SetMember)
    assert first[0].path == second[0].path


def test_add_objects_retains_independent_equal_instances():
    repo = Repo()
    first = GraphLeaf("same", repo=repo)
    second = GraphLeaf("same")
    repo.add_objects(first)

    repo.add_objects(second)

    assert repo.strong_obj_cache.candidates(first.definition) == (first,)
    assert repo.strong_obj_cache.candidates(second.definition) == (second,)


def test_binding_keeps_two_live_instances_with_independent_private_nodes():
    repo = Repo()
    first = GraphLeaf("same")
    second = GraphLeaf("same")

    repo.add_objects([first, second])

    assert first in repo.strong_obj_cache.candidates(first.definition)
    assert second in repo.strong_obj_cache.candidates(second.definition)


def test_workspace_labels_distinguish_private_nodes_without_exposing_tokens(tmp_path):
    repo = Repo()
    repo.workspace_manager = WorkspaceManager(tmp_path / "workspaces")

    direct_first = WorkspaceLeaf("same", repo=repo)
    direct_second = WorkspaceLeaf("same", repo=repo)
    shared = Definition(WorkspaceLeaf, "same")
    aliased = repo.load_or_build(
        Definition(GraphNode, "aliased", shared, shared), cache="none"
    )
    first = Definition(WorkspaceLeaf, "same")
    second = Definition(WorkspaceLeaf, "same")
    distinct = repo.load_or_build(
        Definition(GraphNode, "distinct", first, second), cache="none"
    )

    assert direct_first.workspace != direct_second.workspace
    assert aliased.children[0].workspace == aliased.children[1].workspace
    assert distinct.children[0].workspace != distinct.children[1].workspace
    assert direct_first.workspace != aliased.children[0].workspace
    assert aliased.children[0].workspace != distinct.children[0].workspace
    for obj in (
            direct_first, direct_second, aliased.children[0],
            distinct.children[0], distinct.children[1]
    ):
        assert repr(cdef_node_key(obj.definition)) not in obj.workspace
        assert obj._realization_scope.token not in obj.workspace


def test_get_cached_is_exact_and_rejects_cross_tier_ambiguity():
    repo = Repo()
    first = GraphLeaf("same", repo=repo)
    second = GraphLeaf("same", repo=repo)
    repo.cache_strong(first)
    repo.cache_strong(second)

    assert first.definition.graph_equal(second.definition)
    assert first.definition != second.definition
    assert repo.get_cached(first.definition) is first
    assert repo.get_cached(second.definition) is second

    duplicate = GraphLeaf("same", repo=repo, __cdef__=first.definition)
    repo.cache_weak(duplicate)

    assert repo.get_cached(first.definition) is None
    assert not repo.has_cached(first.definition)


def test_independent_current_nodes_keep_workspace_and_store_bindings(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)

    class WorkspaceHandle:
        def __init__(self, key):
            self.key = key

        def path(self):
            return str(tmp_path / "workspaces" / self.key)

    class WorkspaceManager:
        def alloc(self, key, *, scope=None, node_key=None):
            return WorkspaceHandle(
                f"{key}/{scope.workspace_label}" if scope else key
            )

    repo.workspace_manager = WorkspaceManager()
    first = Definition(WorkspaceLeaf, "same").concretize(repo=repo)
    second = Definition(WorkspaceLeaf, "same").concretize(repo=repo)

    first_obj = repo.load_or_build(first)
    second_obj = repo.load_or_build(second)
    repo.pin(first_obj)
    repo.set_object_store(first, store)
    repo.set_object_store(second, store)

    assert first.graph_equal(second)
    assert first != second
    assert cdef_node_key(first) is not cdef_node_key(second)
    assert repo.strong_obj_cache[first] is first_obj
    assert repo.weak_obj_cache[second] is second_obj
    assert repo.obj_default_store[first] is store
    assert repo.obj_default_store[second] is store
    assert first_obj.workspace != second_obj.workspace


def test_add_objects_assigns_store_to_unique_graph_nodes(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo()
    child = GraphLeaf("child", repo=repo)
    root = GraphNode("root", child, repo=repo)

    repo.add_objects(root, store=store)

    assert repo.obj_default_store[root.definition] is store
    assert repo.obj_default_store[child.definition] is store


def test_iter_graph_missing_cdef_raises_by_default():
    repo = Repo()
    leaf = GraphLeaf("leaf")

    with pytest.raises(RepoGraphError):
        list(repo.iter_graph(leaf.definition))


def test_iter_graph_missing_cdef_can_skip():
    repo = Repo()
    leaf = GraphLeaf("leaf")

    assert list(repo.iter_graph(leaf.definition, missing="skip")) == []


def test_iter_graph_missing_cdef_can_load_from_store(tmp_path):
    store = DirStore(tmp_path / "store")
    repo1 = Repo(stores=store)
    leaf = GraphLeaf("leaf", repo=repo1)
    repo1.save_object(leaf)

    repo2 = Repo(stores=DirStore(store.base_dir))
    loaded = list(repo2.iter_graph(leaf.definition, missing="load"))

    assert len(loaded) == 1
    assert loaded[0].name == "leaf"


def test_object_init_sees_explicit_construction_repo_and_restores_outer_repo():
    outer_repo = Repo()
    construction_repo = Repo()

    with default_repo(outer_repo):
        obj = RepoAware(repo=construction_repo)
        assert obj.seen_repo is construction_repo
        assert get_default_repo() is outer_repo




def test_save_object_uses_direct_state_ref_controls(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo()
    obj = GraphLeaf("saved", repo=repo)

    state = repo.save_object(obj, store=store, alias="saved")

    assert store.read_node(_saved_digest(state), kind="saved").graph_equal(state)
    assert store.read_node_name("saved").root_digest == _saved_digest(state)


def test_save_object_without_store_raises_repo_save_error():
    repo = Repo()
    obj = GraphLeaf("unsaved", repo=repo)

    with pytest.raises(RepoSaveError, match="No Store available"):
        repo.save_object(obj)


def test_save_uses_direct_controls_for_one_loaded_object(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo()
    obj = GraphLeaf("saved", repo=repo)
    repo.pin(obj)

    state = repo.save(obj, store=store)

    assert store.read_node(_saved_digest(state), kind="saved").graph_equal(state)


def test_save_skips_ephemeral_child_under_serializable_parent(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = GraphLeaf("child", repo=repo)
    parent = PersistentNode("parent", child, repo=repo)

    state = repo.save_object(parent)

    assert store.read_node(_saved_digest(state), kind="saved").graph_equal(state)
    assert len(state.state_spec()) == 1


def test_save_traverses_skipped_ephemeral_to_save_serializable_descendant(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    leaf = PersistentLeaf("leaf", repo=repo)
    leaf.value = 42
    child = GraphNode("ephemeral", leaf, repo=repo)
    parent = PersistentNode("parent", child, repo=repo)

    state = repo.save_object(parent)

    assert len(state.state_spec()) == 2
    assert state.at('$[@param("child")][@param("children")][0]').state.key.codec_hash == leaf._last_state_hash


def test_explicit_ephemeral_root_save_creates_object_dir(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = GraphLeaf("root", repo=repo)

    state = repo.save_object(obj)

    assert len(state.state_spec()) == 0
    assert store.read_node(_saved_digest(state), kind="saved").graph_equal(state)


def test_live_ephemeral_alias_forces_save(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = GraphLeaf("aliased", repo=repo)

    state = repo.save_object(obj, alias="leaf")

    assert store.read_node_name("leaf").root_digest == _saved_digest(state)


def test_retired_ephemeral_depth_is_not_a_save_option(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = GraphLeaf("child", repo=repo)
    parent = GraphNode("parent", child, repo=repo)

    with pytest.raises(TypeError):
        repo.save_object(parent, ephemeral_depth=None)
