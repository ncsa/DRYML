"""Focused contracts for inert Repo save-routing configuration."""

import inspect
import sys
import threading
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest

from dryml.core import Object, Repo, SaveRouting, Selector, Serializable, save_object
from dryml.core.cdef_graph import EdgeKind
from dryml.core.links import DefLink
from dryml.core.repo import RepoLoadError, RepoSaveError
from dryml.core.store.dir import DirStore
from dryml.core.store.records import DefinitionRecord, StoredRootRecord


class Routed(Object):
    """Object type used to exercise Selector-based routing decisions."""


class Unrouted(Object):
    """Object type that does not match the routing rules in these tests."""


class RoutedLeaf(Serializable):
    """Stateful child whose hook count proves routed saves capture once."""

    captures = 0

    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        type(self).captures += 1
        Path(dest_dir, "value").write_text(str(self.value), encoding="ascii")


class RoutedRoot(Object):
    """Stateless routed root retaining one materializing child."""

    def __init__(self, child):
        self.child = child


class RoutedPair(Object):
    """Stateless root used to distinguish shared from equal child identities."""

    def __init__(self, left, right):
        self.left = left
        self.right = right


class RoutedBranch(Object):
    """Stateless nested branch which still needs its own StateRef projection."""

    def __init__(self, child):
        self.child = child


class RoutedTree(Object):
    """Stateless root containing independently routed stateless branches."""

    def __init__(self, left, right=None):
        self.left = left
        self.right = right


class SeedRoot(Serializable):
    """Stateful root used to distinguish seed definition routing from its root."""

    def __init__(self, child):
        self.child = child


class MembershipCountingStore(DirStore):
    """DirStore probe that records routed membership reads without allowing scans."""

    def __init__(self, base_dir):
        super().__init__(base_dir, query_index="memory")
        self.membership_reads = 0
        self.stored_root_paths = []

    def _read_file(self, path, record_type):
        if record_type is StoredRootRecord:
            self.stored_root_paths.append(path)
        return super()._read_file(path, record_type)

    def read_stored_root_record(self, digest):
        self.membership_reads += 1
        return super().read_stored_root_record(digest)

    def iter_stored_root_records(self):
        raise AssertionError("routed membership readback must not scan stored roots")


def test_save_routing_normalizes_rules_and_placement_shorthands(tmp_path):
    """Placement shorthands retain the installed ordered all-match policy."""

    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    routing = SaveRouting(
        routes=[(Selector(Routed), second)],
        match_mode="all",
        graph_mode="per-object",
    )
    repo = Repo([first, second], save_routing=routing)

    assert routing.routes == ((Selector(Routed), second),)
    assert repo.save_routing == routing
    repo.set_save_routing("closure")
    assert repo.save_routing == SaveRouting(routing.routes, "all", "closure")
    repo.set_save_routing("per-object")
    assert repo.save_routing == routing

    fresh = Repo(first, save_routing="per-object")
    assert fresh.save_routing == SaveRouting()


def test_internal_selection_uses_first_all_fallback_and_handle_deduplication(tmp_path):
    """Selection retains rule order and never treats one handle as two replicas."""

    fallback = DirStore(tmp_path / "fallback")
    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    repo = Repo(
        [fallback, first, second],
        save_routing=SaveRouting(
            ((Selector(Routed), first), (Selector(Routed), second)),
        ),
    )

    with repo._retain_save_context() as context:
        assert repo._select_save_destinations(context, Routed()) == (first,)
        assert repo._select_save_destinations(context, Unrouted()) == (fallback,)

    repo.set_save_routing(
        SaveRouting(
            ((Selector(Routed), first), (Selector(Routed), first), (Selector(Routed), second)),
            "all",
        )
    )
    with repo._retain_save_context() as context:
        assert repo._select_save_destinations(context, Routed()) == (first, second)


def test_readding_existing_store_without_default_keeps_connected_store_order(tmp_path):
    """Re-registering a handle is idempotent unless explicitly made default."""

    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    repo = Repo([first, second])

    repo.add_store(first)

    assert repo.stores == [first, second]
    assert repo.default_store is first


def test_invalid_routing_and_physical_handles_fail_without_mutating_configuration(tmp_path):
    """Invalid routes and ambiguous built-in handle aliases leave Repo state intact."""

    connected = DirStore(tmp_path / "connected")
    disconnected = DirStore(tmp_path / "disconnected")
    repo = Repo(connected, save_routing="per-object")
    original = repo.save_routing
    stores = tuple(repo.stores)

    with pytest.raises(ValueError, match="connected"):
        repo.set_save_routing(SaveRouting(((Selector(Routed), disconnected),)))
    with pytest.raises(ValueError, match="graph mode"):
        repo.set_save_routing("unknown")
    with pytest.raises(ValueError, match="match_mode"):
        SaveRouting(match_mode="replicate")

    assert repo.save_routing == original
    assert tuple(repo.stores) == stores

    duplicate = DirStore(connected.base_dir)
    unconfigured = Repo(connected)
    unconfigured.add_store(duplicate)
    with pytest.raises(ValueError, match="physical"):
        with unconfigured._retain_save_context():
            pass
    with pytest.raises(ValueError, match="physical"):
        unconfigured.set_save_routing(SaveRouting())
    assert unconfigured.save_routing is None
    assert tuple(unconfigured.stores) == (connected, duplicate)


def test_retained_context_isolated_from_configuration_and_source_changes(tmp_path, monkeypatch):
    """A retained context keeps its policy, source order, and default Store."""

    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    repo = Repo(first, save_routing="per-object")

    ready = threading.Event()
    release = threading.Event()
    observed = []

    def retain_and_observe():
        with repo._retain_save_context() as context:
            ready.set()
            assert release.wait(timeout=5)
            observed.extend((
                context.default_store,
                repo._select_save_destinations(context, Routed()),
            ))

    worker = threading.Thread(target=retain_and_observe)
    worker.start()
    assert ready.wait(timeout=5)
    repo.add_store(second, make_default=True)
    repo.set_save_routing(SaveRouting(((Selector(Routed), second),)))
    release.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert observed == [first, (first,)]


def test_save_context_lease_blocks_repo_close(tmp_path):
    """A context lease protects its Repo resources until the holder exits."""

    repo = Repo(DirStore(tmp_path / "store"))
    retained = repo._retain_save_context()
    retained.__enter__()
    try:
        with pytest.raises(RuntimeError, match="active save context"):
            repo.close(flush=False)
    finally:
        retained.__exit__(None, None, None)
    repo.close(flush=False)


def test_core_import_avoids_managed_policy_in_a_fresh_process():
    """Core imports do not load managed policy in an isolated interpreter."""

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import dryml.core; assert 'dryml.managed' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_empty_routing_does_not_create_a_store_during_selection():
    """Configured fallback without a default Store fails without implicit storage."""

    repo = Repo(save_routing=SaveRouting())

    assert repo.stores == []
    with repo._retain_save_context() as context:
        with pytest.raises(RepoSaveError, match="No Store"):
            repo._select_save_destinations(context, Unrouted())
    assert repo.stores == []


def test_per_object_routing_projects_a_child_state_ref_without_copying_its_payload(tmp_path):
    """Routed descendants retain one external payload and an exact child projection."""
    parent_store = DirStore(tmp_path / "parent")
    child_store = DirStore(tmp_path / "child")
    repo = Repo(
        [parent_store, child_store],
        save_routing=SaveRouting(
            ((Selector(RoutedRoot), parent_store), (Selector(RoutedLeaf), child_store)),
        ),
    )
    root = RoutedRoot(RoutedLeaf(3, repo=repo), repo=repo)

    state, report = repo.save_object(root, deep_capture=True, report_stores=True)
    child_path = next(iter(root.object_ref.objects))
    child_state = state.at(child_path)

    assert child_store.validate_local_state(child_state, ())
    assert parent_store.read_state_ref_record(state.digest()).state_ref == state
    assert child_store.read_state_ref_record(child_state.digest()).state_ref == child_state
    with pytest.raises(Exception):
        parent_store.validate_local_state(child_state, ())
    assert Repo([parent_store, child_store]).load_state_ref(child_state, reuse_live="never").value == 3
    child_snapshot = next(snapshot for snapshot in report.snapshots if snapshot.state_ref == child_state)
    assert Repo(list(child_snapshot.required_stores)).load_state_ref(
        child_snapshot.state_ref, reuse_live="never",
    ).value == 3
    assert list(Repo(child_store).query(child_state.definition).stored().defs()) == [child_state.definition]

    with pytest.raises(RepoLoadError):
        Repo(parent_store).load_state_ref(state, reuse_live="never")


def test_routed_membership_readback_targets_one_root_in_a_growing_catalogue(tmp_path):
    """One routed membership receipt reads only its own marker and definition authority."""
    seed = DirStore(tmp_path / "store", query_index="memory")
    for value in range(64):
        seed.write_definition_record(DefinitionRecord(RoutedLeaf(value).definition))
    store = MembershipCountingStore(seed.base_dir)
    repo = Repo(store, save_routing=SaveRouting())

    state = repo.save_object(RoutedLeaf("new", repo=repo), deep_capture=True)
    expected_path = store._stored_root_path(DefinitionRecord(state.definition).digest)

    assert store.membership_reads == 1
    assert store.stored_root_paths
    assert set(store.stored_root_paths) == {expected_path}


def test_all_matching_routing_captures_once_and_replicates_exact_projections(tmp_path):
    """Every replica receives one captured hash rather than independently sampled state."""
    parents = [DirStore(tmp_path / name) for name in ("parent-a", "parent-b")]
    children = [DirStore(tmp_path / name) for name in ("child-a", "child-b")]
    repo = Repo(
        [*parents, *children],
        save_routing=SaveRouting(
            (
                (Selector(RoutedRoot), parents[0]),
                (Selector(RoutedRoot), parents[1]),
                (Selector(RoutedLeaf), children[0]),
                (Selector(RoutedLeaf), children[1]),
            ),
            match_mode="all",
        ),
    )
    RoutedLeaf.captures = 0
    root = RoutedRoot(RoutedLeaf(4, repo=repo), repo=repo)

    state = repo.save_object(root, deep_capture=True)
    child_path = next(iter(root.object_ref.objects))
    child_state = state.at(child_path)

    assert RoutedLeaf.captures == 1
    assert all(store.read_state_ref_record(state.digest()).state_ref == state for store in parents)
    assert all(store.read_state_ref_record(child_state.digest()).state_ref == child_state for store in children)
    assert root.child.last_state_ref == child_state


def test_routing_keeps_shared_and_structurally_equal_children_distinct(tmp_path):
    """Capture follows live node identity, not structural equality, below stateless roots."""
    root_store = DirStore(tmp_path / "root")
    child_store = DirStore(tmp_path / "child")
    repo = Repo(
        [root_store, child_store],
        save_routing=SaveRouting(
            ((Selector(RoutedPair), root_store), (Selector(RoutedLeaf), child_store)),
        ),
    )
    shared_child = RoutedLeaf(6, repo=repo)
    shared = RoutedPair(shared_child, shared_child, repo=repo)
    independent = RoutedPair(RoutedLeaf(6, repo=repo), RoutedLeaf(6, repo=repo), repo=repo)
    RoutedLeaf.captures = 0

    shared_state = repo.save_object(shared, deep_capture=True)
    independent_state = repo.save_object(independent, deep_capture=True)

    assert len(shared_state.object.objects) == 1
    assert len(independent_state.object.objects) == 2
    assert len(set(independent_state.object.objects.values())) == 2
    assert RoutedLeaf.captures == 3


def test_closure_and_explicit_store_route_the_complete_root_closure(tmp_path):
    """Closure routing and an explicit Store suppress descendant placement rules."""
    parent_store = DirStore(tmp_path / "parent")
    child_store = DirStore(tmp_path / "child")
    explicit_store = DirStore(tmp_path / "explicit")
    repo = Repo(
        [parent_store, child_store, explicit_store],
        save_routing=SaveRouting(
            ((Selector(RoutedRoot), parent_store), (Selector(RoutedLeaf), child_store)),
            graph_mode="closure",
        ),
    )
    root = RoutedRoot(RoutedLeaf(5, repo=repo), repo=repo)

    state = repo.save_object(root, deep_capture=True)
    assert Repo(parent_store).load_state_ref(state, reuse_live="never").child.value == 5

    explicit = repo.save_object(root, store=explicit_store, deep_capture=True)
    assert Repo(explicit_store).load_state_ref(explicit, reuse_live="never").child.value == 5


def test_ordinary_save_entry_points_remove_federated_and_expose_modes(tmp_path):
    """All ordinary save APIs reject the retired control without a compatibility shim."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    obj = RoutedLeaf(13, repo=repo)
    entry_points = (
        (repo.save_object, (obj,)),
        (repo.save, (obj,)),
        (obj.save, (repo,)),
        (save_object, (obj, repo)),
    )

    for entry_point, args in entry_points:
        parameters = inspect.signature(entry_point).parameters
        assert "federated" not in parameters
        assert {"match_mode", "graph_mode"} <= set(parameters)
        with pytest.raises(TypeError):
            entry_point(*args, federated=True)


@pytest.mark.parametrize(
    ("keyword", "value", "error"),
    (
        ("match_mode", 1, TypeError),
        ("match_mode", "replicate", ValueError),
        ("graph_mode", 1, TypeError),
        ("graph_mode", "everywhere", ValueError),
    ),
)
def test_save_mode_overrides_validate_before_explicit_store_routing(tmp_path, keyword, value, error):
    """Mode validation is strict even when an explicit Store bypasses routing."""

    store = DirStore(tmp_path / "store")
    explicit = DirStore(tmp_path / "explicit")
    repo = Repo(store)

    with pytest.raises(error):
        repo.save_object(RoutedLeaf(14, repo=repo), store=explicit, **{keyword: value})

    assert not (Path(explicit.base_dir) / "state-refs").exists()


def test_per_save_modes_do_not_mutate_policy_and_explicit_store_bypasses_routes(tmp_path):
    """Overrides are save-local, while explicit destinations always receive one closure."""

    parent_store = DirStore(tmp_path / "parent")
    child_store = DirStore(tmp_path / "child")
    explicit_store = DirStore(tmp_path / "explicit")
    repo = Repo(
        [parent_store, child_store, explicit_store],
        save_routing=SaveRouting(
            ((Selector(RoutedRoot), parent_store), (Selector(RoutedLeaf), child_store)),
        ),
    )
    original = repo.save_routing
    root = RoutedRoot(RoutedLeaf(15, repo=repo), repo=repo)

    closure = repo.save_object(root, graph_mode="closure", deep_capture=True)
    assert Repo(parent_store).load_state_ref(closure, reuse_live="never").child.value == 15
    assert repo.save_routing == original

    explicit = repo.save_object(
        root, store=explicit_store, match_mode="all", graph_mode="per-object", deep_capture=True,
    )
    assert Repo(explicit_store).load_state_ref(explicit, reuse_live="never").child.value == 15
    assert repo.save_routing == original
    assert child_store.read_state_ref_record(explicit.digest()) is None


def test_unconfigured_save_uses_closure_ledger_and_reports_complete_work(tmp_path):
    """Unconfigured ordinary saves retain default-Store closure and U3 report evidence."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    root = RoutedRoot(RoutedLeaf(16, repo=repo), repo=repo)

    state, report = repo.save_object(root, deep_capture=True, report_stores=True)

    assert report.target_stores == (store,)
    assert report.required_stores == (store,)
    assert len(report.snapshots) == 1
    assert report.snapshots[0].state_ref == state
    assert all(publication.status == "completed" for publication in report.publications)
    assert Repo(store).load_state_ref(state, reuse_live="never").child.value == 16


def test_per_object_routing_projects_distinct_stateless_children_by_live_identity(tmp_path):
    """Stateless branches publish their own projections without becoming payloads."""
    root_store = DirStore(tmp_path / "root")
    branch_store = DirStore(tmp_path / "branch")
    leaf_store = DirStore(tmp_path / "leaf")
    repo = Repo(
        [root_store, branch_store, leaf_store],
        save_routing=SaveRouting(
            (
                (Selector(RoutedTree), root_store),
                (Selector(RoutedBranch), branch_store),
                (Selector(RoutedLeaf), leaf_store),
            ),
        ),
    )
    root = RoutedTree(
        RoutedBranch(RoutedLeaf(1, repo=repo), repo=repo),
        RoutedBranch(RoutedLeaf(1, repo=repo), repo=repo),
        repo=repo,
    )

    state = repo.save_object(root, deep_capture=True)
    branch_paths = tuple(
        path for path, value in root._runtime_projection.items()
        if isinstance(value, RoutedBranch)
    )

    assert len(branch_paths) == 2
    branch_states = tuple(state.at(path) for path in branch_paths)
    assert len({branch_state.digest() for branch_state in branch_states}) == 2
    assert all(
        branch_store.read_state_ref_record(branch_state.digest()).state_ref == branch_state
        for branch_state in branch_states
    )
    assert all(
        leaf_store.validate_local_state(
            branch_state.at(next(iter(branch_state.states))), (),
        )
        for branch_state in branch_states
    )


def test_save_uses_the_disabled_routing_context_retained_for_its_work(tmp_path, monkeypatch):
    """A save cannot mix a prior routing branch with a disabled retained context."""
    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    repo = Repo(
        [first, second],
        save_routing=SaveRouting(((Selector(RoutedLeaf), second),)),
    )
    obj = RoutedLeaf(9, repo=repo)
    original = repo._retain_save_context

    @contextmanager
    def disable_before_retention():
        repo.set_save_routing(None)
        with original() as context:
            yield context

    monkeypatch.setattr(repo, "_retain_save_context", disable_before_retention)

    state = repo.save_object(obj, deep_capture=True)

    assert state.object == obj.object_ref
    assert first.read_state_ref_record(state.digest()).state_ref == state
    assert repo.save_routing is None


def test_save_context_lease_covers_derived_index_registration(tmp_path, monkeypatch):
    """Repo close remains blocked until post-publication index work has finished."""
    store = DirStore(tmp_path / "store")
    repo = Repo(store, save_routing="per-object")
    original = repo._query_index.register_saved_graph

    def assert_active_lease(*args, **kwargs):
        with pytest.raises(RuntimeError, match="active save context"):
            repo.close(flush=False)
        return original(*args, **kwargs)

    monkeypatch.setattr(repo._query_index, "register_saved_graph", assert_active_lease)

    repo.save_object(RoutedLeaf(10, repo=repo), deep_capture=True)


def test_seed_payload_routes_by_its_definition_not_the_stateful_root(tmp_path):
    """An imported StateRef seed uses its leaf selector even when the root is Serializable."""
    source = DirStore(tmp_path / "source")
    source_repo = Repo(source)
    imported = source_repo.save_object(RoutedLeaf(11, repo=source_repo), deep_capture=True)
    root_store = DirStore(tmp_path / "root")
    leaf_store = DirStore(tmp_path / "leaf")
    repo = Repo(
        [root_store, leaf_store, source],
        save_routing=SaveRouting(
            ((Selector(SeedRoot), root_store), (Selector(RoutedLeaf), leaf_store)),
        ),
    )
    root = SeedRoot(imported, repo=repo)

    state = repo.save_object(root, deep_capture=True)
    seed_path = next(path for path in state.object.objects if path)
    seed_definition = state.object.at(seed_path).definition
    seed_hash = state.states[seed_path]

    leaf_store.validate_local_state(state.at(seed_path), ())
    with pytest.raises(Exception):
        root_store.validate_local_state(state.at(seed_path), ())


def test_ref_only_import_does_not_create_a_routed_seed_payload(tmp_path):
    """A Ref-only exact reference remains a value and is not independently routed."""
    source = DirStore(tmp_path / "source")
    source_repo = Repo(source)
    imported = source_repo.save_object(RoutedLeaf(12, repo=source_repo), deep_capture=True)
    root_store = DirStore(tmp_path / "root")
    leaf_store = DirStore(tmp_path / "leaf")
    repo = Repo(
        [root_store, leaf_store, source],
        save_routing=SaveRouting(
            ((Selector(SeedRoot), root_store), (Selector(RoutedLeaf), leaf_store)),
        ),
    )
    root = SeedRoot(DefLink.finalized(EdgeKind.REF, imported), repo=repo)

    state = repo.save_object(root, deep_capture=True)

    assert len(state.object.objects) == 1
    with pytest.raises(Exception):
        leaf_store.validate_local_state(imported, ())
