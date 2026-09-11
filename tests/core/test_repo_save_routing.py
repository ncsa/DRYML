"""Focused contracts for inert Repo save-routing configuration."""

import sys
import threading
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest

from dryml.core import Object, Ref, Repo, SaveRouting, Selector, Serializable
from dryml.core.repo import RepoSaveError
from dryml.core.store.dir import DirStore


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
        unconfigured.set_save_routing(SaveRouting())
    assert unconfigured.save_routing is None
    assert tuple(unconfigured.stores) == (connected, duplicate)


def test_retained_context_isolated_from_configuration_and_source_changes(tmp_path, monkeypatch):
    """A retained context keeps its policy, source order, and default Store."""

    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    repo = Repo(first, save_routing="per-object")
    monkeypatch.setattr(first, "validate_local_state", lambda *_: "first-source")
    monkeypatch.setattr(second, "validate_local_state", lambda *_: "second-source")

    ready = threading.Event()
    release = threading.Event()
    observed = []

    def retain_and_observe():
        with repo._retain_save_context() as context:
            ready.set()
            assert release.wait(timeout=5)
            observed.extend((
                context.default_store,
                context.find_local_state(Routed().definition, "pkl-" + "a" * 64),
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
    assert observed == [first, first, (first,)]


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

    assert child_store.validate_local_state(
        child_state.definition, child_state.states[next(iter(child_state.states))]
    )
    assert parent_store.read_state_ref_record(state.digest()).state_ref == state
    assert child_store.read_state_ref_record(child_state.digest()).state_ref == child_state
    with pytest.raises(Exception):
        parent_store.validate_local_state(
            child_state.definition, child_state.states[next(iter(child_state.states))]
        )
    assert Repo([parent_store, child_store]).load_state_ref(child_state, reuse_live="never").value == 3
    child_snapshot = next(snapshot for snapshot in report.snapshots if snapshot.state_ref == child_state)
    assert Repo(list(child_snapshot.required_stores)).load_state_ref(
        child_snapshot.state_ref, reuse_live="never",
    ).value == 3


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
            branch_state.object.at(next(iter(branch_state.states))).definition,
            branch_state.states[next(iter(branch_state.states))],
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

    leaf_store.validate_local_state(seed_definition, seed_hash)
    with pytest.raises(Exception):
        root_store.validate_local_state(seed_definition, seed_hash)


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
    root = SeedRoot(Ref(imported), repo=repo)

    state = repo.save_object(root, deep_capture=True)

    assert len(state.object.objects) == 1
    with pytest.raises(Exception):
        leaf_store.validate_local_state(
            imported.definition, next(iter(imported.states.values()))
        )
