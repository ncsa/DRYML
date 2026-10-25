"""Managed routing on current durable authority, without legacy snapshot events.

Buffered retained publication is unsupported. Archive commit/reopen/cleanup fault
proofs belong to Core's ZipStore tests; managed must refuse before capture rather
than emulate those retired publication events. Direct routing, independent replica
authority, source pinning, interruption, partial failure and cleanup remain here.
"""

from contextlib import contextmanager
from pathlib import Path
import shutil

import pytest

from dryml.core import Object, Repo, SaveRouting, Selector, Ref, SavedNode
from dryml.core.node_prepare import NodePreparation
from dryml.core.object import Pickleable
from dryml.core.repo import RepoLoadError, RepoSaveError
from dryml.core.store.dir import DirStore
from dryml.core.store.store import StoreCapabilityError
from dryml.core.store.zip import ZipStore
from dryml.managed import ManagedConfig, ManagedInterrupted, ManagedResultError, managed_operation
from dryml.managed import storage as storage_module

pytestmark = pytest.mark.usefixtures("fixed_managed_snapshot_environment")


class RoutedValue(Pickleable):
    """Stateful receiver exposing checkpoint, final, resume and resource facts."""

    captures = 0

    def __init__(self, value=0):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        type(self).captures += 1

    @managed_operation(resumable=True)
    def checkpoint_then_finish(self, *, managed) -> Ref[SavedNode]:
        self.value = 1
        checkpoint = managed.checkpoint(durable=True)
        self.value = 2
        return checkpoint

    @managed_operation(store_parameter="store")
    def finish(self, *, store=None, managed):
        self.value += 1
        return self.value

    @managed_operation()
    def inspect_state_repo(self, *, managed):
        # Return detached facts: a live Repo is not a durable result.
        return len(managed.state_repo.stores)

    @managed_operation()
    def return_live_repo(self, *, managed):
        """Deliberately return a resource to exercise closed-result refusal."""
        return managed.state_repo

    @managed_operation(resumable=True)
    def interrupt_then_finish(self, *, managed):
        if managed.is_resuming:
            self.value += 1
            return self.value
        self.value = 1
        managed.interrupt()


class RoutedRoot(Object):
    """Stateless operation root with an independently routed stateful child."""

    def __init__(self, child):
        self.child = child

    @managed_operation()
    def checkpoint_then_finish(self, *, managed) -> Ref[SavedNode]:
        self.child.value = 1
        checkpoint = managed.checkpoint(durable=True)
        self.child.value = 2
        return checkpoint


def _saved(store, node):
    digest = NodePreparation((node,)).digest(node, domain="dryml-saved-root/v1")
    return store.read_node(digest, kind="saved")


def test_checkpoint_and_final_preserve_per_object_replica_closures(tmp_path):
    """Every routed root and child replica retains its exact published selection."""
    roots = [DirStore(tmp_path / name) for name in ("root-a", "root-b")]
    children = [DirStore(tmp_path / name) for name in ("child-a", "child-b")]
    repo = Repo([*roots, *children], save_routing=SaveRouting(
        tuple((Selector(RoutedRoot), s) for s in roots)
        + tuple((Selector(RoutedValue), s) for s in children),
        match_mode="all", graph_mode="per-object"))
    root = RoutedRoot(RoutedValue(repo=repo), repo=repo)
    result = root.checkpoint_then_finish(managed=ManagedConfig(state_repo=repo))
    checkpoint, final = result.value, result.final_state_ref
    assert not checkpoint.graph_equal(final)
    path, = checkpoint.state_spec()
    for selection, expected in ((checkpoint, 1), (final, 2)):
        assert all(_saved(s, selection).graph_equal(selection) for s in roots)
        child = selection.at(path)
        assert all(_saved(s, child).graph_equal(child) for s in children)
        assert all(s.inspect_states(child.state.key.cdef_digest)[0].available for s in children)
        assert repo.load_state_ref(selection, reuse_live="never").child.value == expected


def test_checkpoint_and_final_preserve_first_match_closure_child_authority(tmp_path):
    """Closure routes keep all selected child keys in the first root destination."""
    first, child = (DirStore(tmp_path / name) for name in ("first", "child"))
    repo = Repo([first, child], save_routing=SaveRouting(
        ((Selector(RoutedRoot), first), (Selector(RoutedValue), child)), graph_mode="closure"))
    root = RoutedRoot(RoutedValue(repo=repo), repo=repo)
    result = root.checkpoint_then_finish(managed=ManagedConfig(state_repo=repo))
    for selection, expected in ((result.value, 1), (result.final_state_ref, 2)):
        assert Repo(first).load_state_ref(selection, reuse_live="never").child.value == expected
        assert _saved(child, selection) is None


def test_resume_uses_retained_checkpoint_authority_after_routes_change(tmp_path):
    """Resume uses the original full key; new publication follows changed routes."""
    old, new = (DirStore(tmp_path / name) for name in ("old", "new"))
    repo = Repo([old, new], save_routing=SaveRouting(((Selector(RoutedValue), old),)))
    value = RoutedValue(repo=repo)
    config = ManagedConfig(state_repo=repo)
    with pytest.raises(ManagedInterrupted):
        value.interrupt_then_finish(managed=config)
    before = value.interrupt_then_finish.status(state_repo=repo)
    repo.set_save_routing(SaveRouting(((Selector(RoutedValue), new),)))
    value.value = 99
    result = value.interrupt_then_finish(managed=config)
    assert result.value == 2
    assert result.execution.run_id == before.run_id
    assert result.execution.invocation_id != before.invocation_id
    assert _saved(old, before.checkpoint).graph_equal(before.checkpoint)
    assert _saved(new, result.final_state_ref).graph_equal(result.final_state_ref)


@pytest.mark.parametrize("failure", ("replica", "index", "cancel"))
def test_replica_or_index_failure_never_associates_partial_state(tmp_path, monkeypatch, failure):
    """Actual Store-batch faults preserve confirmed replicas without a checkpoint."""
    first, second = (DirStore(tmp_path / name, query_index="sqlite") for name in ("first", "second"))
    repo = Repo([first, second], save_routing=SaveRouting(
        ((Selector(RoutedValue), first), (Selector(RoutedValue), second)), match_mode="all"))
    value = RoutedValue(repo=repo)
    error_type = KeyboardInterrupt if failure == "cancel" else OSError
    seam = "_complete_index_mutation" if failure == "index" else "publish_save_batch"
    monkeypatch.setattr(second, seam, lambda *a, **k: (_ for _ in ()).throw(error_type("second failed")))
    expected = ManagedInterrupted if failure == "cancel" else RepoSaveError
    with pytest.raises(expected) as caught:
        value.checkpoint_then_finish(managed=ManagedConfig(state_repo=repo))
    assert tuple(Path(first.base_dir, "membership", "saved").glob("*/*.record"))
    status = value.checkpoint_then_finish.status(state_repo=repo)
    assert status.checkpoint is None and status.final is None
    assert status.state != "completed"
    if failure != "cancel":
        assert caught.value.report is not None
    if failure == "index":
        assert second.authority_cut().pending


@pytest.mark.parametrize("placement", ("only", "replica", "override"))
def test_buffered_retained_publication_refuses_before_capture_or_commit(tmp_path, monkeypatch, placement):
    """Managed cannot report extracted buffered bytes as durable retained state."""
    direct = DirStore(tmp_path / "direct")
    buffered = ZipStore(tmp_path / "buffered.zip")
    control = DirStore(tmp_path / "control")
    stores = [buffered] if placement == "only" else [direct, buffered]
    targets = [direct] if placement == "override" else stores
    repo = Repo(stores, save_routing=SaveRouting(
        tuple((Selector(RoutedValue), s) for s in targets), match_mode="all"))
    value = RoutedValue(repo=repo)
    calls = []
    monkeypatch.setattr(buffered, "commit", lambda: calls.append("commit"))
    before = RoutedValue.captures
    try:
        with pytest.raises(StoreCapabilityError):
            value.finish(**({"store": buffered} if placement == "override" else {}),
                         managed=ManagedConfig(state_repo=repo, control_store=control))
        assert RoutedValue.captures == before
        assert calls == []
        assert not Path(buffered.archive_path).exists()
        assert not tuple(Path(direct.base_dir, "membership", "saved").glob("*/*.record"))
        assert value.finish.status(state_repo=repo, control_store=control).final is None
    finally:
        buffered.close()


def test_store_shorthand_context_exposes_repo_without_closing_borrowed_store(tmp_path):
    """The invocation borrows an explicit direct Store and returns detached facts."""
    state = DirStore(tmp_path / "state")
    value = RoutedValue()
    result = value.inspect_state_repo(managed=ManagedConfig(state_repo=state))
    assert result.value == 1
    assert _saved(state, result.final_state_ref).graph_equal(result.final_state_ref)


def test_live_repo_result_is_rejected_before_capture(tmp_path):
    """Resource transport is not a second durable-result grammar."""
    repo = Repo(DirStore(tmp_path / "state"))
    value = RoutedValue()
    before = RoutedValue.captures
    with pytest.raises(ManagedResultError):
        value.return_live_repo(managed=ManagedConfig(state_repo=repo))
    assert RoutedValue.captures == before
    assert value.return_live_repo.status(state_repo=repo).final is None


@pytest.mark.parametrize("extra_store", (False, True), ids=("matching", "unequal"))
def test_direct_replica_lifecycle_keeps_required_coverage_independent(tmp_path, extra_store):
    """Unselected connected Stores neither publish nor substitute for replicas."""
    replicas = [DirStore(tmp_path / name) for name in ("first", "second")]
    stores = replicas + ([DirStore(tmp_path / "unselected")] if extra_store else [])
    repo = Repo(stores, save_routing=SaveRouting(
        tuple((Selector(RoutedValue), s) for s in replicas), match_mode="all"))
    result = RoutedValue().checkpoint_then_finish(managed=ManagedConfig(state_repo=repo))
    for selection, expected in ((result.value, 1), (result.final_state_ref, 2)):
        for store in replicas:
            assert _saved(store, selection).graph_equal(selection)
            assert Repo(store).load_state_ref(selection, reuse_live="never").value == expected
        if extra_store:
            assert _saved(stores[-1], selection) is None


def test_warm_status_and_resume_reject_deleted_associated_checkpoint(tmp_path):
    """Status is metadata-only; actual unfinished resume refuses missing data."""
    store = DirStore(tmp_path / "state")
    repo = Repo(store)
    value = RoutedValue()
    config = ManagedConfig(state_repo=repo)
    with pytest.raises(ManagedInterrupted):
        value.interrupt_then_finish(managed=config)
    status = value.interrupt_then_finish.status(state_repo=repo)
    checkpoint = status.checkpoint
    shutil.rmtree(store.state_data_path(checkpoint.state.key))
    assert value.interrupt_then_finish.status(state_repo=repo).checkpoint.graph_equal(checkpoint)
    with pytest.raises(RepoLoadError):
        value.interrupt_then_finish(managed=config)
    assert value.value == 1


@pytest.mark.parametrize("failure", ("reader", "corrupt"))
def test_overridden_direct_reader_preserves_independent_publication_validation(tmp_path, monkeypatch, failure):
    """Fresh selected reader overrides and byte corruption cannot borrow old proof."""
    store = DirStore(tmp_path / "state")
    repo = Repo(store)
    value = RoutedValue()
    result = value.finish(managed=ManagedConfig(state_repo=repo))
    final = result.final_state_ref
    if failure == "reader":
        @contextmanager
        def refuse(*args, **kwargs):
            raise OSError("selected reader unavailable")
            yield
        monkeypatch.setattr(store, "read_states", refuse)
        expected = RepoLoadError
    else:
        path, = Path(store.state_data_path(final.state.key)).iterdir()
        path.write_bytes(b"corrupt selected payload")
        expected = RepoLoadError
    with pytest.raises(expected):
        storage_module._validate_selection(repo, value, final, final, source_store=store)
    assert result.value == 1


def test_retention_cleanup_reports_every_routed_child_authority(tmp_path):
    """Public cleanup uses exact routed keys, preserving result and saved metadata."""
    from tests.managed.test_retention_cleanup import CleanupGroup, routed
    control, root, stores = routed(tmp_path)
    CleanupGroup.action = lambda obj, ctx: {"done": True}
    result = root.run(managed=ManagedConfig(state_repo=control.state_repo))
    outcomes = control.cleanup_checkpoints(result.execution.operation_id)
    assert {item.domain for item in outcomes} == {store.retention_domain for store in stores}
    assert len(outcomes) == 3 and all(item.status == "removed" for item in outcomes)
    assert result.value == {"done": True}
    assert control.status(result.execution.operation_id).final.graph_equal(result.final_state_ref)
