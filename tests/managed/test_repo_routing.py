"""U8 managed publication coverage for routed and buffered state authority."""

from __future__ import annotations

import traceback
import shutil
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("fixed_managed_snapshot_environment")

from dryml.core import Object, Repo, SaveRouting, Selector
from dryml.core.object import Pickleable
from dryml.core.reference_values import StateRef
from dryml.core.repo import RepoSaveError
from dryml.core.signatures import Ref
from dryml.core.store.dir import DirStore
from dryml.core.store.zip import ZipStore
from dryml.managed import (
    ManagedConfig,
    ManagedInterrupted,
    ManagedPublicationError,
    managed_operation,
)
from dryml.managed.control import ManagedControlStore
from dryml.managed.errors import ManagedRecoveryError
from dryml.managed import storage as storage_module


class RoutedValue(Pickleable):
    """Stateful managed receiver used to inspect exact routed receipts."""

    def __init__(self, value=0):
        self.value = value

    @managed_operation(resumable=True)
    def checkpoint_then_finish(self, *, managed) -> Ref[StateRef]:
        """Publish a checkpoint before changing the final payload."""

        self.value = 1
        checkpoint = managed.checkpoint()
        self.value = 2
        return checkpoint

    @managed_operation()
    def finish(self, *, managed):
        """Publish a final payload without an intermediate safe point."""

        self.value += 1
        return self.value

    @managed_operation()
    def inspect_state_repo(self, *, managed):
        """Return the resolved state Repo exposed to managed method code."""
        return managed.state_repo

    @managed_operation()
    def interrupt_during_checkpoint(self, *, managed):
        """Publish one checkpoint through an interruptible state boundary."""
        self.value = 1
        managed.checkpoint()

    @managed_operation(resumable=True)
    def interrupt_then_finish(self, *, managed):
        """Leave a checkpoint on first entry and finish from its exact restore."""

        if managed.is_resuming:
            self.value += 1
            return self.value
        self.value = 1
        managed.interrupt()


class RoutedRoot(Object):
    """Stateless managed root whose child has independently routed state."""

    def __init__(self, child):
        self.child = child

    @managed_operation()
    def checkpoint_then_finish(self, *, managed) -> Ref[StateRef]:
        """Publish distinct child checkpoint and final values."""

        self.child.value = 1
        checkpoint = managed.checkpoint()
        self.child.value = 2
        return checkpoint


class SeedDependency(Pickleable):
    """Persisted payload reused through a materializing exact StateRef seed."""

    def __init__(self, value=0):
        self.value = value


class SeededRoutedRoot(Object):
    """Managed stateless root retaining a StateRef seed without recapturing it."""

    def __init__(self, dependency):
        self.dependency = dependency

    @managed_operation(resumable=True)
    def checkpoint_then_interrupt(self, *, managed):
        """Mutate the materialized seed, checkpoint it, and terminate the attempt."""
        self.dependency.value += 1
        managed.interrupt()


def test_checkpoint_and_final_preserve_per_object_replica_closures(tmp_path):
    """Both managed save boundaries retain exact child authority in every replica."""

    roots = [DirStore(tmp_path / name) for name in ("root-a", "root-b")]
    children = [DirStore(tmp_path / name) for name in ("child-a", "child-b")]
    repo = Repo(
        [*roots, *children],
        save_routing=SaveRouting(
            (
                (Selector(RoutedRoot), roots[0]),
                (Selector(RoutedRoot), roots[1]),
                (Selector(RoutedValue), children[0]),
                (Selector(RoutedValue), children[1]),
            ),
            match_mode="all",
            graph_mode="per-object",
        ),
    )
    root = RoutedRoot(RoutedValue(repo=repo), repo=repo)

    checkpoint = root.checkpoint_then_finish(managed=ManagedConfig(state_repo=repo))
    status = root.checkpoint_then_finish.status(state_repo=repo)

    assert status.final_state_ref is not None
    assert checkpoint != status.final_state_ref
    child_path = next(iter(checkpoint.object.objects))
    checkpoint_child = checkpoint.at(child_path)
    final_child = status.final_state_ref.at(child_path)
    assert all(store.read_state_ref_record(checkpoint.digest()).state_ref == checkpoint for store in roots)
    assert all(store.read_state_ref_record(checkpoint_child.digest()).state_ref == checkpoint_child for store in children)
    assert all(store.read_state_ref_record(status.final_state_ref.digest()).state_ref == status.final_state_ref for store in roots)
    assert all(store.read_state_ref_record(final_child.digest()).state_ref == final_child for store in children)
    assert all(
        list(Repo(store).query().sel(checkpoint_child.definition).cdefs().stored().collect())
        == [checkpoint_child.definition]
        for store in children
    )
    recovered = Repo._for_state_io([*roots, *children])
    try:
        assert recovered.load_state_ref(checkpoint, reuse_live="never").child.value == 1
        assert recovered.load_state_ref(status.final_state_ref, reuse_live="never").child.value == 2
    finally:
        recovered.close(flush=False)


def test_managed_commits_selected_zip_replicas_but_not_unrelated_dirty_store(tmp_path):
    """Managed publication durably commits only its selected required archives."""

    control = DirStore(tmp_path / "control")
    selected = [ZipStore(tmp_path / name) for name in ("selected-a.zip", "selected-b.zip")]
    unrelated = ZipStore(tmp_path / "unrelated.zip")
    repo = Repo(
        [*selected, unrelated],
        save_routing=SaveRouting(
            ((Selector(RoutedValue), selected[0]), (Selector(RoutedValue), selected[1])),
            match_mode="all",
        ),
    )
    value = RoutedValue(repo=repo)
    unrelated._archive_dirty = True

    assert value.finish(managed=ManagedConfig(state_repo=repo, control_store=control)) == 1
    final = value.finish.status(state_repo=repo, control_store=control).final_state_ref

    assert all(not store._archive_dirty for store in selected)
    assert unrelated._archive_dirty
    for store in selected:
        reopened = ZipStore.open_existing(store.archive_path)
        try:
            recovered = Repo._for_state_io((reopened,))
            try:
                assert recovered.load_state_ref(final, reuse_live="never").value == 1
            finally:
                recovered.close(flush=False)
        finally:
            reopened.close()
    unrelated.close()
    assert not Path(unrelated.archive_path).exists()


def test_checkpoint_and_final_preserve_first_match_closure_child_authority(tmp_path):
    """First-match closure routing keeps checkpoint/final child queries at the root route."""

    root_store = DirStore(tmp_path / "root")
    child_store = DirStore(tmp_path / "child")
    repo = Repo(
        [root_store, child_store],
        save_routing=SaveRouting(
            ((Selector(RoutedRoot), root_store), (Selector(RoutedValue), child_store)),
            graph_mode="closure",
        ),
    )
    root = RoutedRoot(RoutedValue(repo=repo), repo=repo)

    checkpoint = root.checkpoint_then_finish(managed=ManagedConfig(state_repo=repo))
    final = root.checkpoint_then_finish.status(state_repo=repo).final_state_ref
    child_path = next(iter(checkpoint.object.objects))

    assert Repo(root_store).load_state_ref(checkpoint, reuse_live="never").child.value == 1
    assert Repo(root_store).load_state_ref(final, reuse_live="never").child.value == 2
    assert root_store.validate_local_state(checkpoint, child_path).state_hash == checkpoint.states[child_path]
    assert child_store.read_state_ref_record(checkpoint.digest()) is None


def test_resume_uses_retained_checkpoint_authority_after_routes_change(tmp_path):
    """Resume reads the old exact checkpoint even when future saves select a new route."""

    old = DirStore(tmp_path / "old")
    new = DirStore(tmp_path / "new")
    repo = Repo([old, new], save_routing=SaveRouting(((Selector(RoutedValue), old),)))
    value = RoutedValue(repo=repo)
    with pytest.raises(ManagedInterrupted):
        value.interrupt_then_finish(managed=ManagedConfig(state_repo=repo))
    checkpoint = value.interrupt_then_finish.status(state_repo=repo).checkpoint_state_ref
    repo.set_save_routing(SaveRouting(((Selector(RoutedValue), new),)))
    value.value = 99

    assert value.interrupt_then_finish(managed=ManagedConfig(state_repo=repo)) == 2
    status = value.interrupt_then_finish.status(state_repo=repo)

    assert status.final_state_ref is not None
    assert old.read_state_ref_record(checkpoint.digest()).state_ref == checkpoint
    assert new.read_state_ref_record(status.final_state_ref.digest()).state_ref == status.final_state_ref


def test_zip_commit_failure_has_report_and_never_associates_checkpoint(tmp_path, monkeypatch):
    """A failed required archive commit leaves state report evidence but no checkpoint."""

    state = ZipStore(tmp_path / "state.zip")
    control = DirStore(tmp_path / "control")
    repo = Repo(state)
    value = RoutedValue(repo=repo)
    monkeypatch.setattr(state, "commit", lambda: (_ for _ in ()).throw(OSError("commit failed")))

    with pytest.raises(RepoSaveError, match="managed state publication") as caught:
        value.checkpoint_then_finish(
            managed=ManagedConfig(state_repo=repo, control_store=control),
        )

    assert caught.value.report is not None
    assert any(item.phase == "commit" and item.status == "failed" for item in caught.value.report.publications)
    assert isinstance(caught.value.__cause__, ManagedPublicationError)
    assert isinstance(caught.value.__cause__.__cause__, RepoSaveError)
    assert "OSError: commit failed" in "".join(traceback.format_exception(caught.value))
    assert value.checkpoint_then_finish.status(
        state_repo=repo, control_store=control,
    ).checkpoint_state_ref is None


def test_interrupted_completed_zip_commit_preserves_prior_checkpoint(tmp_path, monkeypatch):
    """A completed orphan never replaces an earlier managed checkpoint after interruption."""

    state = ZipStore(tmp_path / "state.zip")
    control = DirStore(tmp_path / "control")
    repo = Repo(state)
    value = RoutedValue(repo=repo)
    with pytest.raises(ManagedInterrupted):
        value.interrupt_then_finish(
            managed=ManagedConfig(state_repo=repo, control_store=control),
        )
    prior = value.interrupt_then_finish.status(
        state_repo=repo, control_store=control,
    ).checkpoint_state_ref
    original = state.commit

    def commit_then_interrupt():
        original()
        raise KeyboardInterrupt("commit interruption")

    monkeypatch.setattr(state, "commit", commit_then_interrupt)
    with pytest.raises(KeyboardInterrupt, match="commit interruption") as caught:
        value.interrupt_then_finish(
            managed=ManagedConfig(state_repo=repo, control_store=control),
        )

    commits = [item for item in caught.value.report.publications if item.phase == "commit"]
    assert [(item.store, item.status) for item in commits] == [(state, "completed")]
    assert value.interrupt_then_finish.status(
        state_repo=repo, control_store=control,
    ).checkpoint_state_ref == prior


def test_control_failure_after_committed_state_leaves_reopenable_orphan(tmp_path, monkeypatch):
    """A control association fault never makes durable state appear associated."""

    state = ZipStore(tmp_path / "state.zip")
    control = DirStore(tmp_path / "control")
    repo = Repo(state)
    value = RoutedValue(repo=repo)
    original = ManagedControlStore._transition_running_fields

    calls = 0

    def fail_association(self, operation_id, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("control association failed")
        return original(self, operation_id, **kwargs)

    monkeypatch.setattr(ManagedControlStore, "_transition_running_fields", fail_association)
    with pytest.raises(OSError, match="control association failed") as caught:
        value.checkpoint_then_finish(
            managed=ManagedConfig(state_repo=repo, control_store=control),
        )

    assert caught.value.report is not None
    assert value.checkpoint_then_finish.status(
        state_repo=repo, control_store=control,
    ).checkpoint_state_ref is None
    reopened = ZipStore.open_existing(state.archive_path)
    try:
        assert tuple(reopened.iter_state_ref_records())
    finally:
        reopened.close()


def test_replica_or_index_failure_never_associates_partial_state(tmp_path, monkeypatch):
    """A durable first replica or index cannot become a managed checkpoint alone."""

    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    repo = Repo(
        [first, second],
        save_routing=SaveRouting(
            ((Selector(RoutedValue), first), (Selector(RoutedValue), second)),
            match_mode="all",
        ),
    )
    value = RoutedValue(repo=repo)
    original_write = second.publish_snapshot

    def fail_second_replica(*args, **kwargs):
        raise OSError("second replica failed")

    monkeypatch.setattr(second, "publish_snapshot", fail_second_replica)
    with pytest.raises(RepoSaveError, match="managed state publication"):
        value.checkpoint_then_finish(managed=ManagedConfig(state_repo=repo))
    assert tuple(first.iter_state_ref_records())
    assert value.checkpoint_then_finish.status(state_repo=repo).checkpoint_state_ref is None
    monkeypatch.setattr(second, "publish_snapshot", original_write)

    indexed_repo = Repo(first)
    indexed = RoutedValue(repo=indexed_repo)
    monkeypatch.setattr(
        indexed_repo._query_index,
        "register_saved_graph",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("index failed")),
    )
    with pytest.raises(RepoSaveError, match="managed state publication"):
        indexed.checkpoint_then_finish(managed=ManagedConfig(state_repo=indexed_repo))
    assert indexed.checkpoint_then_finish.status(state_repo=indexed_repo).checkpoint_state_ref is None


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("boundary", ["commit", "reopened_validation"])
def test_managed_publication_preserves_control_flow_and_report(
        tmp_path, monkeypatch, error_type, boundary,
):
    """Managed save control flow retains identity and the available core report."""

    state = ZipStore(tmp_path / "state.zip")
    control = DirStore(tmp_path / "control")
    repo = Repo(state)
    value = RoutedValue(repo=repo)
    if boundary == "commit":
        monkeypatch.setattr(
            state, "commit",
            lambda: (_ for _ in ()).throw(error_type("commit interruption")),
        )
    else:
        from dryml.managed import storage as storage_module

        monkeypatch.setattr(
            storage_module, "_validate_reopened_managed_state",
            lambda *args: (_ for _ in ()).throw(error_type("reopened interruption")),
        )

    with pytest.raises(error_type) as caught:
        value.checkpoint_then_finish(
            managed=ManagedConfig(state_repo=repo, control_store=control),
        )

    assert caught.value.report is not None
    assert value.checkpoint_then_finish.status(
        state_repo=repo, control_store=control,
    ).checkpoint_state_ref is None


def test_missing_committed_replica_cannot_be_hidden_by_a_valid_replica(tmp_path, monkeypatch):
    """Reopened validation rejects a missing selected archive replica independently."""

    first = ZipStore(tmp_path / "first.zip")
    second = ZipStore(tmp_path / "second.zip")
    control = DirStore(tmp_path / "control")
    repo = Repo(
        [first, second],
        save_routing=SaveRouting(
            ((Selector(RoutedValue), first), (Selector(RoutedValue), second)),
            match_mode="all",
        ),
    )
    value = RoutedValue(repo=repo)
    original_commit = second.commit

    def commit_then_remove_archive():
        original_commit()
        Path(second.archive_path).unlink()

    monkeypatch.setattr(second, "commit", commit_then_remove_archive)
    with pytest.raises(RepoSaveError, match="managed state publication") as caught:
        value.checkpoint_then_finish(
            managed=ManagedConfig(state_repo=repo, control_store=control),
        )

    assert caught.value.report is not None
    assert Path(first.archive_path).is_file()
    assert value.checkpoint_then_finish.status(
        state_repo=repo, control_store=control,
    ).checkpoint_state_ref is None


def test_reopened_nested_payloads_are_validated_once_per_durable_closure(tmp_path, monkeypatch):
    """Replica record checks plus one root closure avoid repeated payload rehashing."""

    state = ZipStore(tmp_path / "state.zip")
    control = DirStore(tmp_path / "control")
    repo = Repo(
        (state,),
        save_routing=SaveRouting(
            ((Selector(RoutedRoot), state), (Selector(RoutedValue), state)),
            graph_mode="per-object",
        ),
    )
    value = RoutedRoot(RoutedValue(repo=repo), repo=repo)
    reopened = set()
    validations = []
    accelerated = []
    original_open = ZipStore.open_existing
    previous = sys.getprofile()

    def open_existing(path):
        store = original_open(path)
        reopened.add(id(store))
        return store

    def observe(frame, event, arg):
        if id(frame.f_locals.get("self")) not in reopened:
            return
        if event == "call" and frame.f_code is DirStore._validate_local_state_dir.__code__:
            validations.append(frame.f_locals["directory"])
        if event == "return" and frame.f_code is DirStore._can_reuse_exact_validation.__code__:
            accelerated.append(arg)

    monkeypatch.setattr(ZipStore, "open_existing", staticmethod(open_existing))
    sys.setprofile(observe)
    try:
        value.checkpoint_then_finish(managed=ManagedConfig(state_repo=repo, control_store=control))
    finally:
        sys.setprofile(previous)

    assert len(validations) == len(set(validations)) == 2
    assert accelerated and all(accelerated)


@pytest.mark.parametrize("missing", ("record", "payload"))
def test_reopened_root_closure_rejects_missing_nested_replica_authority(
        tmp_path, monkeypatch, missing):
    """One root closure still rejects an independently projected child's loss."""

    state = ZipStore(tmp_path / "state.zip")
    repo = Repo(
        (state,),
        save_routing=SaveRouting(
            ((Selector(RoutedRoot), state), (Selector(RoutedValue), state)),
            graph_mode="per-object",
        ),
    )
    root = RoutedRoot(RoutedValue(repo=repo), repo=repo)
    state_ref, report = repo.save_object(root, deep_capture=True, report_stores=True)
    state.commit()
    child_ref = next(snapshot.state_ref for snapshot in report.snapshots if snapshot.state_ref != state_ref)
    reopened = set()
    original_open = ZipStore.open_existing
    original_read = ZipStore.read_state_ref_record
    original_validate = ZipStore.validate_local_state

    def open_existing(path):
        store = original_open(path)
        reopened.add(id(store))
        return store

    def read_state_ref_record(self, digest):
        if missing == "record" and id(self) in reopened and digest == child_ref.digest():
            return None
        return original_read(self, digest)

    def validate_local_state(self, reference, path):
        if missing == "payload" and id(self) in reopened and reference == child_ref:
            raise FileNotFoundError("nested payload missing")
        return original_validate(self, reference, path)

    monkeypatch.setattr(ZipStore, "open_existing", staticmethod(open_existing))
    monkeypatch.setattr(ZipStore, "read_state_ref_record", read_state_ref_record)
    monkeypatch.setattr(ZipStore, "validate_local_state", validate_local_state)
    with pytest.raises(ManagedRecoveryError, match="missing_state_reference"):
        storage_module._validate_reopened_managed_state(repo, state_ref, report, (state,))


def test_dirty_reused_zip_seed_is_required_durable_and_commit_failure_keeps_checkpoint(
        tmp_path, monkeypatch):
    """An embedded original seed remains a dirty dependency after child recapture.

    The managed save writes the mutated live child to the root archive while its
    definition still embeds the original StateRef.  Recovery therefore needs the
    original dependency record and payload, without writing to that archive.
    """

    dependency = ZipStore(tmp_path / "dependency.zip")
    root_store = ZipStore(tmp_path / "root.zip")
    control = DirStore(tmp_path / "control")
    dependency_repo = Repo(dependency)
    seed = dependency_repo.save_object(SeedDependency(41, repo=dependency_repo), deep_capture=True)
    assert dependency._archive_dirty

    routing = SaveRouting(
        (
            (Selector(SeededRoutedRoot), root_store),
            (Selector(SeedDependency), root_store),
        ),
        graph_mode="per-object",
    )
    repo = Repo([dependency, root_store], save_routing=routing)
    value = SeededRoutedRoot(seed, repo=repo)
    reports = []
    from dryml.managed import context as context_module
    from dryml.managed.storage import _publish_managed_state_result

    def capture_publication():
        def publish(*args, **kwargs):
            result = _publish_managed_state_result(*args, **kwargs)
            reports.append(result.report)
            return result

        return publish

    def no_new_dependency_write(*args, **kwargs):
        raise AssertionError("reused dependency must not receive a managed-save write")

    monkeypatch.setattr(context_module, "_publish_managed_state", capture_publication)
    for name in (
            "create_local_state_staging", "prepare_local_state", "prepare_rebound_local_state",
            "publish_snapshot", "write_definition_record",
    ):
        monkeypatch.setattr(dependency, name, no_new_dependency_write)

    with pytest.raises(ManagedInterrupted):
        value.checkpoint_then_interrupt(managed=ManagedConfig(state_repo=repo, control_store=control))
    checkpoint = value.checkpoint_then_interrupt.status(
        state_repo=repo, control_store=control,
    ).checkpoint_state_ref

    assert checkpoint is not None
    assert len(reports) == 1
    assert dependency in reports[0].required_stores
    assert root_store in reports[0].required_stores
    assert not dependency._archive_dirty
    assert not root_store._archive_dirty

    repo.close(flush=False)
    dependency_repo.close(flush=False)
    dependency.close()
    root_store.close()
    reopened_dependency = ZipStore.open_existing(tmp_path / "dependency.zip")
    reopened_root = ZipStore.open_existing(tmp_path / "root.zip")
    reopened_routing = SaveRouting(
        (
            (Selector(SeededRoutedRoot), reopened_root),
            (Selector(SeedDependency), reopened_root),
        ),
        graph_mode="per-object",
    )
    reopened_repo = Repo([reopened_dependency, reopened_root], save_routing=reopened_routing)
    try:
        restored = reopened_repo.load_state_ref(checkpoint, reuse_live="never")
        assert restored.dependency.value == 42
        reopened_dependency_repo = Repo(reopened_dependency)
        reopened_dependency_repo.save_object(
            SeedDependency(-1, repo=reopened_dependency_repo), deep_capture=True,
        )
        assert reopened_dependency._archive_dirty
        for name in (
                "create_local_state_staging", "prepare_local_state", "prepare_rebound_local_state",
                "publish_snapshot", "write_definition_record",
        ):
            monkeypatch.setattr(reopened_dependency, name, no_new_dependency_write)
        monkeypatch.setattr(
            reopened_dependency,
            "commit",
            lambda: (_ for _ in ()).throw(OSError("dependency commit failed")),
        )
        with pytest.raises(RepoSaveError, match="managed state publication") as caught:
            restored.checkpoint_then_interrupt(
                managed=ManagedConfig(state_repo=reopened_repo, control_store=control),
            )

        report = caught.value.report
        commits = [item for item in report.publications if item.phase == "commit"]
        assert any(
            item.store is reopened_dependency and item.status in {"failed", "uncertain"}
            for item in commits
        )
        assert restored.checkpoint_then_interrupt.status(
            state_repo=reopened_repo, control_store=control,
        ).checkpoint_state_ref == checkpoint
    finally:
        reopened_repo.close(flush=False)
        reopened_dependency.close()
        reopened_root.close()


def test_closure_seed_replica_excludes_dirty_historical_source_from_required_commits(
        tmp_path, monkeypatch):
    """A complete closure replica recovers without committing its dirty seed source."""

    dependency = ZipStore(tmp_path / "dependency.zip")
    root_store = ZipStore(tmp_path / "root.zip")
    control = DirStore(tmp_path / "control")
    dependency_repo = Repo(dependency)
    seed = dependency_repo.save_object(SeedDependency(41, repo=dependency_repo), deep_capture=True)
    repo = Repo(
        [dependency, root_store],
        save_routing=SaveRouting(
            ((Selector(SeededRoutedRoot), root_store),), graph_mode="closure",
        ),
    )
    value = SeededRoutedRoot(seed, repo=repo)
    reports = []
    from dryml.managed import context as context_module
    from dryml.managed.storage import _publish_managed_state_result

    def capture_publication():
        def publish(*args, **kwargs):
            result = _publish_managed_state_result(*args, **kwargs)
            reports.append(result.report)
            return result

        return publish

    def no_new_dependency_write(*args, **kwargs):
        raise AssertionError("closure replication must not republish the seed source")

    monkeypatch.setattr(context_module, "_publish_managed_state", capture_publication)
    for name in (
            "create_local_state_staging", "prepare_local_state", "prepare_rebound_local_state",
            "publish_snapshot", "write_definition_record",
    ):
        monkeypatch.setattr(dependency, name, no_new_dependency_write)

    with pytest.raises(ManagedInterrupted):
        value.checkpoint_then_interrupt(managed=ManagedConfig(state_repo=repo, control_store=control))
    checkpoint = value.checkpoint_then_interrupt.status(
        state_repo=repo, control_store=control,
    ).checkpoint_state_ref

    assert checkpoint is not None
    assert reports[0].required_stores == (root_store,)
    assert dependency._archive_dirty
    assert not root_store._archive_dirty
    repo.close(flush=False)
    dependency_repo.close(flush=False)
    dependency.close()
    root_store.close()
    reopened_root = ZipStore.open_existing(tmp_path / "root.zip")
    try:
        assert Repo(reopened_root).load_state_ref(
            checkpoint, reuse_live="never",
        ).dependency.value == 42
    finally:
        reopened_root.close()


def test_store_shorthand_context_exposes_repo_without_closing_borrowed_store(tmp_path):
    """A Store shorthand returns a Repo and leaves its buffered handle available."""

    state = ZipStore(tmp_path / "state.zip")
    control = DirStore(tmp_path / "control")
    value = RoutedValue(repo=Repo(state))

    resolved = value.inspect_state_repo(
        managed=ManagedConfig(state_repo=state, control_store=control),
    )
    final = value.inspect_state_repo.status(state_repo=state, control_store=control).final_state_ref

    assert isinstance(resolved, Repo)
    assert tuple(resolved.stores) == (state,)
    assert state.read_state_ref_record(final.digest()).state_ref == final


@pytest.mark.parametrize("failure", [None, "stale", "validation", "cleanup"])
def test_zip_event_follows_independent_durable_cut_and_resource_cleanup(tmp_path, monkeypatch, failure):
    """Even failed durable verification closes every temporary view before delivery."""

    control = DirStore(tmp_path / "control")
    first, second = (ZipStore(tmp_path / name) for name in ("first.zip", "second.zip"))
    repo = Repo(
        [first, second], save_routing=SaveRouting(
            ((Selector(RoutedValue), first), (Selector(RoutedValue), second)),
            match_mode="all",
        ),
    )
    value = RoutedValue(repo=repo)
    repo.save_object(value, deep_capture=True)
    first.commit()
    second.commit()
    stale = tmp_path / "stale.zip"
    shutil.copyfile(second.archive_path, stale)
    opened, temporary, minted, closure_cuts = [], [], [], []
    durable_payloads = []
    original_open = ZipStore.open_existing
    original_closure = storage_module._validate_publication_closure
    original_event = storage_module._PublicationEvent.__init__
    previous = sys.getprofile()

    def observe(frame, event, arg):
        if event == "return" and frame.f_code is Repo._for_state_io.__func__.__code__:
            assert arg._validation_evidence is repo._validation_evidence
            temporary.append(arg)
        if event == "call" and frame.f_code is DirStore._validate_local_state_dir.__code__:
            if any(frame.f_locals["self"] is store for store in opened):
                durable_payloads.append(frame.f_locals["directory"])
        if previous is not None:
            previous(frame, event, arg)

    def open_existing(path):
        store = original_open(stale if failure == "stale" and path == second.archive_path else path)
        opened.append(store)
        if failure == "cleanup" and path == second.archive_path:
            original_close = store.close

            def close_then_fail():
                original_close()
                raise OSError("durable cleanup failed")

            monkeypatch.setattr(store, "close", close_then_fail)
        return store

    def closure(selected_repo, reference, cut):
        closure_cuts.append(cut)
        if failure == "validation" and selected_repo is not repo:
            raise SystemExit("durable validation stopped")
        return original_closure(selected_repo, reference, cut)

    def mint(self, *args):
        assert opened and all(store._closed_handle for store in opened)
        assert temporary and all(item._closed for item in temporary)
        assert all(not cut.active for cut in closure_cuts)
        minted.append(self)
        original_event(self, *args)

    monkeypatch.setattr(ZipStore, "open_existing", staticmethod(open_existing))
    monkeypatch.setattr(storage_module, "_validate_publication_closure", closure)
    monkeypatch.setattr(storage_module._PublicationEvent, "__init__", mint)
    sys.setprofile(observe)
    try:
        if failure is None:
            assert value.finish(managed=ManagedConfig(state_repo=repo, control_store=control)) == 1
        else:
            error_type = SystemExit if failure == "validation" else RepoSaveError
            with pytest.raises(error_type) as caught:
                value.finish(managed=ManagedConfig(state_repo=repo, control_store=control))
            assert caught.value.report is not None
    finally:
        sys.setprofile(previous)
    assert len(opened) == 2 and all(store._closed_handle for store in opened)
    assert all(not Path(store.base_dir).exists() for store in opened)
    assert temporary and all(item._closed for item in temporary)
    assert all(not cut.active for cut in closure_cuts)
    assert len(minted) == (1 if failure is None else 0)
    if failure is None:
        assert len(durable_payloads) == len(set(durable_payloads)) == 2
        assert repo._validation_evidence._resident
    assert not repo._validation_evidence._pending
    assert repo._validation_evidence.accounted_bytes == sum(
        fact.charge for fact in repo._validation_evidence._resident.values())
    if failure != "stale":
        assert len(closure_cuts) == (3 if failure is None else 2)
        assert closure_cuts[0] is not closure_cuts[1]
        if failure is None:
            assert closure_cuts[2] not in closure_cuts[:2]
    status = value.finish.status(state_repo=repo, control_store=control)
    assert (status.state, status.final_state_ref is not None) == (
        "completed" if failure is None else "failed", failure is None,
    )


def test_uncertain_zip_commit_mints_no_event_and_retains_prior_checkpoint(tmp_path, monkeypatch):
    """Indeterminate commit evidence cannot be upgraded to completed publication."""

    state = ZipStore(tmp_path / "state.zip")
    control = DirStore(tmp_path / "control")
    repo = Repo(state)
    value = RoutedValue(repo=repo)
    config = ManagedConfig(state_repo=repo, control_store=control)
    with pytest.raises(ManagedInterrupted):
        value.interrupt_then_finish(managed=config)
    prior = value.interrupt_then_finish.status(state_repo=repo, control_store=control).checkpoint_state_ref
    original_commit = state.commit

    def uncertain_commit():
        original_commit()
        monkeypatch.setattr(state, "_archive_identity", lambda: (_ for _ in ()).throw(OSError("commit evidence unavailable")))

    def no_event(*args, **kwargs):
        raise AssertionError("uncertain commit cannot mint an event")

    monkeypatch.setattr(state, "commit", uncertain_commit)
    monkeypatch.setattr(storage_module._PublicationEvent, "__init__", no_event)
    with pytest.raises(RepoSaveError) as caught:
        value.interrupt_then_finish(managed=config)
    assert isinstance(caught.value.__cause__, ManagedPublicationError)
    assert caught.value.__cause__.outcome == "indeterminate"
    assert any(item.phase == "commit" and item.status == "uncertain" for item in caught.value.report.publications)
    assert value.interrupt_then_finish.status(state_repo=repo, control_store=control).checkpoint_state_ref == prior


def test_missing_required_embedded_seed_cannot_reuse_connected_proof(tmp_path, monkeypatch):
    """A smaller required view cannot borrow an otherwise valid connected closure."""

    dependency, root_store = (DirStore(tmp_path / name) for name in ("dependency", "root"))
    dependency_repo = Repo(dependency)
    seed = dependency_repo.save_object(SeedDependency(41, repo=dependency_repo), deep_capture=True)
    repo = Repo(
        [dependency, root_store], save_routing=SaveRouting(
            ((Selector(SeededRoutedRoot), root_store), (Selector(SeedDependency), root_store)),
            graph_mode="per-object",
        ),
    )
    value = SeededRoutedRoot(seed, repo=repo)
    monkeypatch.setattr(storage_module, "_required_managed_publication_stores", lambda report: (root_store,))
    with pytest.raises(RepoSaveError) as caught:
        value.checkpoint_then_interrupt(managed=ManagedConfig(state_repo=repo))
    assert caught.value.report is not None
    assert value.checkpoint_then_interrupt.status(state_repo=repo).checkpoint_state_ref is None


def test_overridden_direct_reader_preserves_independent_publication_validation(tmp_path, monkeypatch):
    """Reader overrides retain their authority behavior and disable proof sharing."""

    store = DirStore(tmp_path / "state")
    repo = Repo(store)
    value = RoutedValue(repo=repo)
    original = store.read_state_ref_record
    durable_calls = []
    original_durable = storage_module._validate_reopened_managed_state

    def reader(digest):
        return original(digest)

    def durable(*args):
        durable_calls.append(True)
        return original_durable(*args)

    monkeypatch.setattr(store, "read_state_ref_record", reader)
    monkeypatch.setattr(storage_module, "_validate_reopened_managed_state", durable)
    assert not store._can_reuse_exact_validation()
    assert value.finish(managed=ManagedConfig(state_repo=repo)) == 1
    assert durable_calls == [True]


@pytest.mark.parametrize("extra_store", [False, True], ids=["matching", "unequal"])
def test_direct_replica_lifecycle_keeps_required_coverage_independent(tmp_path, extra_store):
    """An unselected connected view restores the two separate durable obligations."""
    from dryml.core.materialization import _build_exact_state_load_plan

    replicas = [DirStore(tmp_path / name, query_index="none") for name in ("first", "second")]
    stores = [*replicas, DirStore(tmp_path / "unselected", query_index="none")] if extra_store else replicas
    repo = Repo(stores, save_routing=SaveRouting(
        tuple((Selector(RoutedValue), store) for store in replicas), match_mode="all",
    ))
    value = RoutedValue(repo=repo)
    closures, durable_repos, accelerated = [], [], []
    previous = sys.getprofile()

    def observe(frame, event, arg):
        if event == "call" and frame.f_code is _build_exact_state_load_plan.__code__:
            closures.append((frame.f_locals["repo"], frame.f_locals["state_ref"]))
        if event == "return" and frame.f_code is Repo._for_state_io.__func__.__code__:
            durable_repos.append(arg)
        if event == "return" and frame.f_code is DirStore._can_reuse_exact_validation.__code__:
            accelerated.append(arg)
        if previous is not None:
            previous(frame, event, arg)

    sys.setprofile(observe)
    try:
        checkpoint = value.checkpoint_then_finish(managed=ManagedConfig(state_repo=repo))
    finally:
        sys.setprofile(previous)
    final = value.last_state_ref
    assert len(closures) == (8 if extra_store else 6)
    assert accelerated and all(accelerated)
    assert len(durable_repos) == (2 if extra_store else 0)
    for durable in durable_repos:
        assert durable._closed and durable._topology_leases == 0
        assert tuple(durable.stores) == tuple(replicas)
    assert sum(selected is not repo for selected, _ in closures) == (2 if extra_store else 0)
    assert repo._topology_leases == 0 and not repo._validation_evidence._pending
    for reference in (checkpoint, final):
        for store in replicas:
            assert store.read_state_ref_record(reference.digest()).state_ref == reference
            assert Repo(store).load_state_ref(reference, reuse_live="never").value == (1 if reference == checkpoint else 2)
    if extra_store:
        assert stores[-1].read_state_ref_record(checkpoint.digest()) is None


def test_warm_status_and_resume_reject_deleted_associated_checkpoint(tmp_path):
    """Independent checkpoint recovery never trusts publication or warm content facts."""
    from tests.core.test_exact_load_metadata_evidence import exact_work_counts

    store = DirStore(tmp_path / "state", query_index="none")
    control = DirStore(tmp_path / "control", query_index="none")
    repo = Repo(store)
    value = RoutedValue(repo=repo)
    config = ManagedConfig(state_repo=repo, control_store=control)
    with pytest.raises(ManagedInterrupted):
        value.interrupt_then_finish(managed=config)
    status = value.interrupt_then_finish.status(state_repo=repo, control_store=control)
    checkpoint = status.checkpoint_state_ref
    assert status.state == "interrupted" and checkpoint is not None
    assert repo._validation_evidence._resident
    shutil.rmtree(store._snapshot_path(checkpoint.digest()))
    for operation in (
        lambda: value.interrupt_then_finish.status(state_repo=repo, control_store=control),
        lambda: value.interrupt_then_finish(managed=config),
    ):
        with exact_work_counts() as counts:
            with pytest.raises(ManagedRecoveryError, match="missing_state_reference"):
                operation()
        assert counts["_read_snapshot"] > 0
        assert value.value == 1 and value.last_state_ref == checkpoint
        assert repo._topology_leases == 0 and not repo._validation_evidence._pending


@pytest.mark.parametrize("backend", ["direct", "reopened"])
def test_reopened_selected_replica_payload_corruption_cannot_hide_behind_first(tmp_path, monkeypatch, backend):
    """A valid first replica cannot authorize a corrupt selected physical view."""
    control = DirStore(tmp_path / "control", query_index="none")
    if backend == "direct":
        first, second = (DirStore(tmp_path / name, query_index="none") for name in ("first", "second"))
    else:
        first, second = (ZipStore(tmp_path / name) for name in ("first.zip", "second.zip"))
    repo = Repo([first, second], save_routing=SaveRouting(
        ((Selector(RoutedValue), first), (Selector(RoutedValue), second)), match_mode="all",
    ))
    value = RoutedValue(repo=repo)
    original_open = ZipStore.open_existing
    opened = []

    def corrupt(store):
        reference = value.last_state_ref
        local = store.open_local_state(reference, next(iter(reference.states)))
        files = tuple(Path(local.handle, "data").iterdir())
        assert files
        files[0].write_bytes(b"corrupt selected replica")
        assert store._can_reuse_exact_validation()

    def open_existing(path):
        store = original_open(path)
        opened.append(store)
        if path == second.archive_path:
            corrupt(store)
        assert store._can_reuse_exact_validation()
        return store

    if backend == "reopened":
        monkeypatch.setattr(ZipStore, "open_existing", staticmethod(open_existing))
    else:
        commit = storage_module._commit_save_report

        def commit_then_corrupt(*args, **kwargs):
            report = commit(*args, **kwargs)
            corrupt(second)
            return report

        monkeypatch.setattr(storage_module, "_commit_save_report", commit_then_corrupt)
    try:
        with pytest.raises(RepoSaveError, match="managed state publication") as caught:
            value.finish(managed=ManagedConfig(state_repo=repo, control_store=control))
        assert caught.value.report is not None
        assert len(opened) == (2 if backend == "reopened" else 0)
        assert all(store._closed_handle for store in opened)
        assert not repo._validation_evidence._pending and repo._topology_leases == 0
        status = value.finish.status(state_repo=repo, control_store=control)
        assert status.state == "failed" and status.final_state_ref is None
    finally:
        repo.close(flush=False)
        first.close()
        second.close()
