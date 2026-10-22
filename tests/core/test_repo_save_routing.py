"""Focused contracts for inert Repo save-routing configuration."""

import inspect
import sys
import threading
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

from dryml.core import Object, Repo, SaveRouting, Selector, Serializable, save_object
from dryml.core.cdef_graph import EdgeKind
from dryml.core.links import DefLink
from dryml.core.repo import RepoLoadError, RepoSaveError
from dryml.core.store.dir import DirStore
from dryml.core.store.records import DefinitionRecord, StoredRootRecord
from dryml.core.node_prepare import NodePreparation
from dryml.core.utils.graph.path import GraphPath, Parameter


def saved_digest(node):
    return NodePreparation((node,)).digest(node, domain="dryml-saved-root/v1")


def assert_published(store, node):
    assert store.read_node(saved_digest(node), kind="saved").graph_equal(node)


def payload_value(store, node):
    with store.read_state(node.state.key) as source:
        return Path(source.data_directory, "value").read_text()


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

    captures = 0

    def __init__(self, child):
        self.child = child

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        type(self).captures += 1
        Path(dest_dir, "value").write_text("middle", encoding="ascii")


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

    def _read_membership(self, digest, kind):
        self.membership_reads += 1
        self.stored_root_paths.append((kind, digest))
        return super()._read_membership(digest, kind)

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

    assert len(routing.routes) == 1 and routing.routes[0][1] is second
    assert routing.routes[0][0].matches(Routed().definition)
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

    assert payload_value(child_store, child_state) == "3"
    assert_published(parent_store, state)
    assert_published(child_store, child_state)
    with pytest.raises(Exception):
        with parent_store.read_state(child_state.state.key):
            pass
    child_snapshot = next(snapshot for snapshot in report.snapshots if snapshot.state_ref == child_state)
    assert child_store in child_snapshot.required_stores

    from dryml.core.repo import RepoLoadError
    with pytest.raises(RepoLoadError, match="unavailable"):
        Repo(parent_store).load(state)
    loaded = Repo([parent_store, child_store]).load(state)
    assert loaded.child.value == 3
    assert loaded.child.object_id == root.child.object_id


def test_routed_membership_readback_targets_one_root_in_a_growing_catalogue(tmp_path):
    """One routed membership receipt reads only its own marker and definition authority."""
    seed = DirStore(tmp_path / "store", query_index="memory")
    for value in range(64):
        seed.publish_node(RoutedLeaf(value).definition)
    store = MembershipCountingStore(seed.base_dir)
    repo = Repo(store, save_routing=SaveRouting())

    state = repo.save_object(RoutedLeaf("new", repo=repo), deep_capture=True)
    expected_path = ("saved", saved_digest(state))

    assert store.membership_reads > 0
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
    for store in parents:
        assert_published(store, state)
    for store in children:
        assert_published(store, child_state)
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

    assert len(shared_state.identity_spec()) == 1
    assert len(independent_state.identity_spec()) == 2
    assert len(set(independent_state.identity_spec().values())) == 2
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
    assert payload_value(parent_store, state.at(next(iter(state.state_spec())))) == "5"

    explicit = repo.save_object(root, store=explicit_store, deep_capture=True)
    assert payload_value(explicit_store, explicit.at(next(iter(explicit.state_spec())))) == "5"


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

    assert not tuple(explicit.iter_state_ref_records())


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
    assert payload_value(parent_store, closure.at(next(iter(closure.state_spec())))) == "15"
    assert repo.save_routing == original

    explicit = repo.save_object(
        root, store=explicit_store, match_mode="all", graph_mode="per-object", deep_capture=True,
    )
    assert payload_value(explicit_store, explicit.at(next(iter(explicit.state_spec())))) == "15"
    assert repo.save_routing == original
    assert child_store.read_node(saved_digest(explicit), kind="saved") is None


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
    assert payload_value(store, state.at(next(iter(state.state_spec())))) == "16"


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
    assert len({saved_digest(branch_state) for branch_state in branch_states}) == 2
    for branch_state in branch_states:
        assert_published(branch_store, branch_state)
        assert payload_value(leaf_store, branch_state.at(next(iter(branch_state.state_spec())))) == "1"


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

    assert dict(state.identity_spec()) == dict(obj.object_ref.objects)
    assert_published(first, state)
    assert repo.save_routing is None


def test_save_context_lease_covers_derived_index_registration(tmp_path, monkeypatch):
    """Repo close remains blocked until post-publication index work has finished."""
    store = DirStore(tmp_path / "store")
    repo = Repo(store, save_routing="per-object")
    original = store._complete_index_mutation

    def assert_active_lease(*args, **kwargs):
        with pytest.raises(RuntimeError, match="active save context"):
            repo.close(flush=False)
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "_complete_index_mutation", assert_active_lease)

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
    saved = repo.save(root)
    child_state = saved.at(GraphPath((Parameter("child"),)))
    assert payload_value(leaf_store, child_state) == "11"
    assert child_state.state.key == imported.state.key
    assert_published(root_store, saved)
    assert_published(leaf_store, child_state)


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

    assert len(state.identity_spec()) == 1
    assert leaf_store.read_node(saved_digest(imported), kind="saved") is None


def test_failed_child_publication_leaves_dependent_parent_unattempted(tmp_path, monkeypatch):
    parent = DirStore(tmp_path / "parent", query_index="none")
    child = DirStore(tmp_path / "child", query_index="none")
    repo = Repo([parent, child], save_routing=SaveRouting(
        ((Selector(RoutedRoot), parent), (Selector(RoutedLeaf), child))))
    root = RoutedRoot(RoutedLeaf(3, repo=repo), repo=repo)
    RoutedLeaf.captures = 0
    def fail(batch):
        raise OSError("child publication unavailable")
    monkeypatch.setattr(child, "publish_save_batch", fail)
    with pytest.raises(RepoSaveError) as caught:
        repo.save_object(root)
    assert [(p.store, p.status) for p in caught.value.report.publications] == [
        (child, "failed"), (parent, "unattempted")]
    assert root.last_state_ref is None
    assert RoutedLeaf.captures == 1
    assert not Path(parent.base_dir, "graphs").exists()


def test_partial_replica_retry_reuses_confirmed_state_without_capture(tmp_path, monkeypatch):
    first = DirStore(tmp_path / "first", query_index="none")
    second = DirStore(tmp_path / "second", query_index="none")
    repo = Repo([first, second], save_routing=SaveRouting(
        ((Selector(RoutedLeaf), first), (Selector(RoutedLeaf), second)), "all", "closure"))
    root = RoutedLeaf(4, repo=repo)
    original = second.publish_save_batch
    monkeypatch.setattr(second, "publish_save_batch", lambda batch: (_ for _ in ()).throw(OSError("replica unavailable")))
    RoutedLeaf.captures = 0
    with pytest.raises(RepoSaveError) as caught:
        repo.save_object(root)
    confirmed = caught.value.report.snapshots[0].state_ref
    assert root.last_state_ref is confirmed
    monkeypatch.setattr(second, "publish_save_batch", original)
    repaired = repo.save_object(root, states=confirmed.state_spec(), source_store=first)
    assert repaired.graph_equal(confirmed)
    assert RoutedLeaf.captures == 1
    assert_published(second, repaired)


def test_cancellation_keeps_identity_and_names_unattempted_targets(tmp_path, monkeypatch):
    first = DirStore(tmp_path / "first", query_index="none")
    second = DirStore(tmp_path / "second", query_index="none")
    repo = Repo([first, second], save_routing=SaveRouting(
        ((Selector(RoutedLeaf), first), (Selector(RoutedLeaf), second)), "all", "closure"))
    obj = RoutedLeaf(3, repo=repo)
    def interrupt(batch):
        raise KeyboardInterrupt()
    monkeypatch.setattr(first, "publish_save_batch", interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        repo.save_object(obj)
    assert [(p.store, p.status) for p in caught.value.report.publications] == [
        (first, "failed"), (second, "unattempted")]
    assert obj.last_state_ref is None


def test_alternating_store_routes_publish_a_valid_live_dag(tmp_path):
    """A live A-B-A DAG needs one coalesced indexed batch per physical Store."""
    first = DirStore(tmp_path / "first", query_index="sqlite")
    second = DirStore(tmp_path / "second", query_index="sqlite")
    repo = Repo([first, second], save_routing=SaveRouting((
        (Selector(RoutedRoot), first),
        (Selector(SeedRoot), second),
        (Selector(RoutedLeaf), first),
    )))
    root = RoutedRoot(SeedRoot(RoutedLeaf(3, repo=repo), repo=repo), repo=repo)
    statements = [[], []]
    for store, trace in zip((first, second), statements):
        index = store.open_query_index()
        index.maintain_store()
        index._connections.connection().set_trace_callback(trace.append)
    SeedRoot.captures = RoutedLeaf.captures = 0
    saved, report = repo.save_object(root, report_stores=True)
    assert_published(first, saved)
    assert len(saved.state_spec()) == 2
    assert SeedRoot.captures == RoutedLeaf.captures == 1
    assert [sum(sql.strip().upper() == "COMMIT" for sql in trace)
            for trace in statements] == [1, 1]
    assert all(p.status == "completed" for p in report.publications)
    assert report.required_stores == (first, second)
    assert root.last_state_ref is saved
    for snapshot in report.snapshots:
        assert snapshot.required_stores
    for publication in report.publications:
        assert publication.protected_state_keys == publication.state_keys
        assert not publication.store.authority_cut().pending
        for key in publication.state_keys:
            assert publication.store.delete_state(key).status == "retained"


@pytest.mark.parametrize("failure_boundary", ["creation", "grant"])
def test_alternating_store_failure_qualifies_recovery_only_after_protection(
        tmp_path, monkeypatch, failure_boundary):
    """Partial local metadata is not a usable receipt for an unprotected closure."""
    first = DirStore(tmp_path / "first", query_index="sqlite")
    second = DirStore(tmp_path / "second", query_index="sqlite")
    repo = Repo([first, second], save_routing=SaveRouting((
        (Selector(RoutedRoot), first),
        (Selector(SeedRoot), second),
        (Selector(RoutedLeaf), first),
    )))
    root = RoutedRoot(SeedRoot(RoutedLeaf(3, repo=repo), repo=repo), repo=repo)
    original = second._atomic_write

    def fail_before_protection(path, payload):
        path = str(path)
        if ((failure_boundary == "creation" and path.endswith("/def.pkl"))
                or (failure_boundary == "grant" and "/grants/" in path)):
            raise OSError("middle publication unavailable")
        return original(path, payload)

    monkeypatch.setattr(second, "_atomic_write", fail_before_protection)
    SeedRoot.captures = RoutedLeaf.captures = 0
    with pytest.raises(RepoSaveError) as caught:
        repo.save_object(root)
    report = caught.value.report
    assert [(p.store, p.status) for p in report.publications] == [
        (first, "completed"), (second, "uncertain")]
    local, failed = report.publications
    assert len(local.state_keys) == len(local.planned_state_keys) == 1
    assert len(failed.planned_state_keys) == 1
    assert failed.state_keys == (failed.planned_state_keys if failure_boundary == "grant" else ())
    assert failed.protected_state_keys == ()
    assert local.protected_state_keys == local.state_keys
    assert report.required_stores == ()
    assert root.last_state_ref is None
    assert root.child.last_state_ref is None
    planned = local.state_ref
    root_snapshot = next(s for s in report.snapshots if s.state_ref is planned)
    assert root_snapshot.stores == (first,)
    assert root_snapshot.required_stores == ()
    leaf_snapshot = next(s for s in report.snapshots if s.state_ref.state is not None)
    assert leaf_snapshot.required_stores == (first,)
    assert set(report.state_stores) == {
        path for path, state in planned.state_spec().items()
        if state.key in local.state_keys + failed.state_keys}
    assert_published(first, planned)
    assert first.delete_state(local.state_keys[0]).status == "retained"
    assert second.retention.read(failed.retention_token).disposition == "prepared"
    if failure_boundary == "grant":
        with second.read_state(failed.planned_state_keys[0]) as source:
            assert source.state.key == failed.planned_state_keys[0]
    monkeypatch.setattr(second, "_atomic_write", original)
    second.repair_save_batch(failed.mutation_id, failed.retention_token)
    assert not second.authority_cut().pending
    assert second.retention.read(failed.retention_token).disposition == "adopted"
    assert second.delete_state(failed.planned_state_keys[0]).status == "retained"
    assert SeedRoot.captures == RoutedLeaf.captures == 1
    assert root.last_state_ref is None
    repaired, repaired_report = repo.save_object(
        root, states=planned.state_spec(), report_stores=True)
    assert repaired.graph_equal(planned)
    assert repaired_report.required_stores == (first, second)
    assert root.last_state_ref is repaired
    assert SeedRoot.captures == RoutedLeaf.captures == 1


def test_alternating_routes_choose_a_protected_replica_after_target_failure(tmp_path, monkeypatch):
    """Recovery chooses an available protected provider, not every routed replica."""
    first, second, replica = [DirStore(tmp_path / name, query_index="none")
                              for name in ("first", "second", "replica")]
    repo = Repo([first, second, replica], save_routing=SaveRouting((
        (Selector(RoutedRoot), first),
        (Selector(SeedRoot), second),
        (Selector(SeedRoot), replica),
        (Selector(RoutedLeaf), first),
    ), "all"))
    root = RoutedRoot(SeedRoot(RoutedLeaf(3, repo=repo), repo=repo), repo=repo)

    def fail(batch):
        raise OSError("middle replica unavailable")

    monkeypatch.setattr(second, "publish_save_batch", fail)
    SeedRoot.captures = RoutedLeaf.captures = 0
    with pytest.raises(RepoSaveError) as caught:
        repo.save_object(root)
    report = caught.value.report
    assert [(p.store, p.status) for p in report.publications] == [
        (first, "completed"), (second, "failed"), (replica, "completed")]
    assert report.required_stores == (first, replica)
    assert root.last_state_ref is report.publications[0].state_ref
    assert all(second not in s.required_stores for s in report.snapshots)
    assert SeedRoot.captures == RoutedLeaf.captures == 1


def test_failed_provider_leaves_transitive_dependents_unattempted(tmp_path, monkeypatch):
    """Prepared-batch fallback cannot publish through a failed dependency chain."""
    parent, middle, leaf = [DirStore(tmp_path / name, query_index="none")
                            for name in ("parent", "middle", "leaf")]
    repo = Repo([parent, middle, leaf], save_routing=SaveRouting((
        (Selector(RoutedRoot), parent),
        (Selector(SeedRoot), middle),
        (Selector(RoutedLeaf), leaf),
    )))
    root = RoutedRoot(SeedRoot(RoutedLeaf(3, repo=repo), repo=repo), repo=repo)

    def fail(batch):
        raise OSError("leaf unavailable")

    monkeypatch.setattr(leaf, "publish_save_batch", fail)
    with pytest.raises(RepoSaveError) as caught:
        repo.save_object(root)
    assert [(p.store, p.status) for p in caught.value.report.publications] == [
        (leaf, "failed"), (parent, "unattempted"), (middle, "unattempted")]
    assert caught.value.report.snapshots == ()
    assert caught.value.report.required_stores == ()
    assert root.last_state_ref is None
    assert not Path(parent.base_dir, "graphs").exists()
    assert not Path(middle.base_dir, "graphs").exists()


@pytest.mark.parametrize("prior_failure", [False, True])
def test_alternating_routes_reconcile_local_protection_before_reraising_cancellation(
        tmp_path, monkeypatch, prior_failure):
    """An interruption after local publication preserves identity and partial grants."""
    first, second, replica, spare = [DirStore(tmp_path / name, query_index="none")
                                     for name in ("first", "second", "replica", "spare")]
    repo = Repo([first, second, replica, spare], save_routing=SaveRouting((
        (Selector(RoutedRoot), first),
        (Selector(SeedRoot), second),
        (Selector(SeedRoot), replica),
        (Selector(SeedRoot), spare),
        (Selector(RoutedLeaf), first),
    ), "all"))
    root = RoutedRoot(SeedRoot(RoutedLeaf(3, repo=repo), repo=repo), repo=repo)
    interruption = KeyboardInterrupt()

    def interrupt(batch):
        assert root.last_state_ref is None
        raise interruption

    if prior_failure:
        def fail(batch):
            raise OSError("earlier replica unavailable")
        monkeypatch.setattr(second, "publish_save_batch", fail)
    monkeypatch.setattr(replica if prior_failure else second, "publish_save_batch", interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        repo.save_object(root)
    assert caught.value is interruption
    report = caught.value.report
    assert [(p.store, p.status) for p in report.publications] == [
        (first, "completed"), (second, "failed"),
        (replica, "failed" if prior_failure else "unattempted"), (spare, "unattempted")]
    assert root.last_state_ref is None
    assert report.required_stores == ()
    local = report.publications[0]
    assert first.delete_state(local.state_keys[0]).status == "retained"
    assert first.retention.read(local.retention_token).disposition == "adopted"
