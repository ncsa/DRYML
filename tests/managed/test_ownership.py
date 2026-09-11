"""U5 lifetime ownership coverage for selected managed state Stores."""

from __future__ import annotations

import multiprocessing
import os
import shutil
from threading import Thread

import pytest

from dryml.core import Object, Repo, Serializable
from dryml.core.reference_values import ObjectId
from dryml.core.store.dir import DirStore
from dryml.core.store.zip import ZipStore
from dryml.locking import FileLock, LockError
from dryml.managed.control import ManagedControlStore
from dryml.managed.errors import ManagedConflictError, ManagedRecoveryError, ManagedStoreError
from dryml.managed import storage as storage_module
from dryml.managed.storage import (
    _acquire_state_ownership,
    _acquire_state_locks,
    _probe_state_ownership,
    _state_lock_path,
    ownership_evidence,
)


class OwnedValue(Serializable):
    """Minimal stateful node used to exercise ObjectId lifetime leases."""

    def __init__(self, value):
        self.value = value


class OwnedRoot(Object):
    """Stateless root retaining one or more stateful descendants."""

    def __init__(self, *children):
        self.children = children


def _object_ids(root):
    """Return the exact stateful identities of one retained live graph."""

    return tuple(root.object_ref.objects.values())


def _zip_lock_owner(archive, object_ids, ready, release):
    """Retain a path-backed ZipStore lifetime lock until deliberate child exit."""

    lease = _acquire_state_locks(Repo((ZipStore(archive),)), object_ids)
    ready.set()
    assert release.wait(10)
    os._exit(0)


def _bootstrap_lock_owner(root, object_id, ready, start, queue):
    """Race a first disjoint lifetime-lock admission in a real child process."""

    ready.set()
    assert start.wait(10)
    try:
        lease = _acquire_state_locks(Repo((DirStore(root),)), (object_id,))
    except BaseException as error:  # pragma: no cover - asserted through parent result.
        queue.put(type(error).__name__)
    else:
        lease.release()
        queue.put("ok")


def test_ownership_composes_graph_reservation_and_state_repo_lock_set(tmp_path):
    """A stateless root retains every descendant lock through the active lease."""

    store = DirStore(tmp_path / "state")
    duplicate_handle = DirStore(tmp_path / "state")
    repo = Repo((store,))
    root = OwnedRoot(OwnedValue(1, repo=repo), repo=repo)

    duplicate_repo = Repo((duplicate_handle,))
    control_one = ManagedControlStore(DirStore(tmp_path / "control-one"), repo)
    control_two = ManagedControlStore(DirStore(tmp_path / "control-two"), duplicate_repo)
    with _acquire_state_ownership(repo, root) as owner:
        assert owner.active
        assert owner.reservation.active
        assert owner.object_ids == _object_ids(root)
        assert _state_lock_path(store, owner.object_ids[0]).endswith(".lock")
        assert not control_one.probe_state_ownership(owner.ownership)
        assert not control_two.probe_state_ownership(owner.ownership)
        with pytest.raises(ManagedConflictError, match="already reserved"):
            _acquire_state_ownership(duplicate_repo, root)

    assert _probe_state_ownership(duplicate_repo, owner.ownership)


def test_partial_lock_acquisition_releases_earlier_state_leases(tmp_path):
    """A contested later ObjectId leaves an earlier acquired ObjectId available."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    root = OwnedRoot(OwnedValue(1, repo=repo), OwnedValue(2, repo=repo), repo=repo)
    object_ids = tuple(sorted(_object_ids(root), key=lambda value: _state_lock_path(store, value)))
    first_path = _state_lock_path(store, object_ids[0])
    ManagedControlStore(store, Repo((store,))).initialize()
    held = FileLock(_state_lock_path(store, object_ids[1]))
    assert held.acquire()
    try:
        with pytest.raises(ManagedConflictError, match="state lock"):
            _acquire_state_ownership(repo, root)
        released = FileLock(first_path)
        assert released.acquire(blocking=False)
        released.release()
    finally:
        held.release()


def test_disjoint_graphs_can_overlap_but_thread_cannot_use_or_release_foreign_owner(tmp_path):
    """Only overlapping live graphs conflict, and a token is thread-bound."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    first = OwnedValue(1, repo=repo)
    second = OwnedValue(2, repo=repo)
    owner = _acquire_state_ownership(repo, first)
    result = []

    def use_from_other_thread():
        result.append(owner.release())
        with pytest.raises(ManagedConflictError, match="owner"):
            owner.require_owner()

    thread = Thread(target=use_from_other_thread)
    thread.start()
    thread.join(timeout=5)
    try:
        assert not thread.is_alive()
        assert result == [False]
        assert owner.active
        with _acquire_state_ownership(repo, second):
            pass
        with pytest.raises(ManagedConflictError, match="already reserved"):
            _acquire_state_ownership(repo, first)
    finally:
        owner.release()


def test_pending_claim_in_another_state_repo_is_rejected_without_abandoning_it(tmp_path):
    """Managed state selection rejects claims before generic core save can consume them."""

    declaration_store = DirStore(tmp_path / "declaration")
    selected_store = DirStore(tmp_path / "selected")
    repo = Repo((declaration_store, selected_store))
    reference = repo.declare_object(OwnedValue(1).definition, store=declaration_store)
    live = repo.build_object_ref(reference, store=declaration_store)
    claim = declaration_store.read_claim_record(reference.digest())

    with _acquire_state_ownership(repo, live) as owner:
        assert len(owner.ownership["store_keys"]) == 2

    assert declaration_store.read_claim_record(reference.digest()) == claim
    assert live._claim_lease is not None


def test_probe_releases_partial_leases_and_does_not_classify_contention_as_owner_loss(tmp_path):
    """The U6 probe reports availability only when it acquired every requested lease."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    root = OwnedRoot(OwnedValue(1, repo=repo), OwnedValue(2, repo=repo), repo=repo)
    object_ids = tuple(sorted(_object_ids(root), key=lambda value: _state_lock_path(store, value)))
    ManagedControlStore(store, Repo((store,))).initialize()
    seeded = _acquire_state_locks(repo, object_ids)
    seeded.release()
    held = FileLock(_state_lock_path(store, object_ids[1]))
    assert held.acquire()
    try:
        assert not _probe_state_ownership(repo, ownership_evidence(repo, object_ids))
        first = FileLock(_state_lock_path(store, object_ids[0]))
        assert first.acquire(blocking=False)
        first.release()
    finally:
        held.release()
    assert _probe_state_ownership(repo, ownership_evidence(repo, object_ids))


def test_probe_propagates_native_lock_failure_instead_of_calling_it_owner_loss(tmp_path, monkeypatch):
    """A capability error remains actionable rather than becoming inconclusive contention."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    obj = OwnedValue(1, repo=repo)
    ManagedControlStore(store, Repo((store,))).initialize()
    seeded = _acquire_state_locks(repo, _object_ids(obj))
    seeded.release()

    class BrokenLock:
        def __init__(self, path):
            self.path = path

        def acquire(self, *, blocking):
            raise LockError("native failure")

    monkeypatch.setattr(storage_module, "FileLock", BrokenLock)
    with pytest.raises(ManagedStoreError, match="acquire"):
        _probe_state_ownership(repo, ownership_evidence(repo, _object_ids(obj)))


def test_read_only_probe_never_recreates_missing_lock_namespace(tmp_path):
    """Missing retained lock evidence is inconclusive and cannot be bootstrapped by a probe."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    value = OwnedValue(1, repo=repo)
    evidence = ownership_evidence(repo, _object_ids(value))
    lease = _acquire_state_locks(repo, _object_ids(value))
    lease.release()
    namespace = os.path.dirname(os.path.dirname(_state_lock_path(store, _object_ids(value)[0])))
    shutil.rmtree(namespace)

    with pytest.raises(ManagedRecoveryError, match="state_lock_namespace_missing"):
        _probe_state_ownership(repo, evidence)

    assert not os.path.exists(namespace)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is unavailable on this host")
def test_first_disjoint_process_owners_never_observe_a_partial_namespace(tmp_path, monkeypatch):
    """A staged first namespace keeps a concurrent disjoint owner admissible."""

    context = multiprocessing.get_context("fork")
    root = os.fspath(tmp_path / "state")
    first_ready, second_ready = context.Event(), context.Event()
    first_start, second_start = context.Event(), context.Event()
    format_entered, release_format = context.Event(), context.Event()
    results = context.Queue()
    original_write = storage_module._write_state_namespace_format

    def pause_before_format(path):
        format_entered.set()
        assert release_format.wait(10)
        original_write(path)

    monkeypatch.setattr(storage_module, "_write_state_namespace_format", pause_before_format)
    first = context.Process(
        target=_bootstrap_lock_owner,
        args=(root, ObjectId(), first_ready, first_start, results),
    )
    second = context.Process(
        target=_bootstrap_lock_owner,
        args=(root, ObjectId(), second_ready, second_start, results),
    )
    first.start()
    try:
        assert first_ready.wait(10)
        first_start.set()
        assert format_entered.wait(10)
        second.start()
        assert second_ready.wait(10)
        second_start.set()
        release_format.set()
        first.join(20)
        second.join(20)
        assert first.exitcode == second.exitcode == 0
        assert sorted((results.get(timeout=2), results.get(timeout=2))) == ["ok", "ok"]
    finally:
        release_format.set()
        for process in (first, second):
            if process.is_alive():
                process.terminate()
                process.join(5)


def test_failed_initial_namespace_format_leaves_no_final_partial_authority(tmp_path, monkeypatch):
    """A short format write cleans only unpublished bootstrap staging before retry."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    value = OwnedValue(1, repo=repo)
    root = os.path.join(store.base_dir, "managed")

    monkeypatch.setattr(storage_module.os, "write", lambda fd, payload: len(payload) - 1)
    with pytest.raises(ManagedStoreError, match="namespace"):
        _acquire_state_locks(repo, _object_ids(value))

    assert not os.path.exists(root)


def test_probe_validates_the_full_closed_evidence_before_filesystem_effects(tmp_path):
    """Malformed v1 evidence, including bool versions, cannot create lock artifacts."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    malformed = {"version": True, "store_keys": ["a" * 64], "object_keys": ["b" * 64]}

    with pytest.raises(ManagedRecoveryError, match="invalid_ownership"):
        _probe_state_ownership(repo, malformed)

    assert not os.path.exists(os.path.join(store.base_dir, "managed"))


def test_overlapping_store_object_pairs_conflict_while_disjoint_repos_proceed(tmp_path):
    """Ownership is the physical Store/ObjectId Cartesian product, not control scope."""

    shared = DirStore(tmp_path / "shared")
    other = DirStore(tmp_path / "other")
    first_repo = Repo((shared,))
    overlapping_repo = Repo((DirStore(tmp_path / "shared"),))
    disjoint_repo = Repo((other,))
    value = OwnedValue(1, repo=first_repo)

    object_ids = _object_ids(value)
    held = _acquire_state_locks(first_repo, object_ids)
    try:
        with pytest.raises(ManagedConflictError, match="state lock"):
            _acquire_state_locks(overlapping_repo, object_ids)
        lease = _acquire_state_locks(disjoint_repo, object_ids)
        lease.release()
    finally:
        held.release()


def test_topology_lease_blocks_set_changes_but_allows_default_reorder(tmp_path):
    """An invocation freezes Store membership while later default policy can change."""

    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    repo = Repo((first, second))
    value = OwnedValue(1, repo=repo)

    with _acquire_state_ownership(repo, value):
        repo.set_default_store(second)
        assert repo.default_store is second
        with pytest.raises(RuntimeError, match="topology"):
            repo.add_store(DirStore(tmp_path / "third"))
        with pytest.raises(RuntimeError, match="topology"):
            repo.close()


def test_ownership_freezes_topology_before_one_graph_capture(tmp_path, monkeypatch):
    """A concurrent Store addition cannot split captured evidence from acquired locks."""

    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    added = DirStore(tmp_path / "added")
    repo = Repo((first, second))
    value = OwnedValue(1, repo=repo)
    expected = ownership_evidence(repo, _object_ids(value))
    entered, release = multiprocessing.Event(), multiprocessing.Event()
    original = repo._state_graph_evidence
    calls, outcome = [], []

    def capture_once(obj):
        calls.append(obj)
        entered.set()
        assert release.wait(10)
        return original(obj)

    def acquire():
        try:
            with _acquire_state_ownership(repo, value) as owner:
                outcome.append(owner.ownership)
        except BaseException as error:  # pragma: no cover - asserted below.
            outcome.append(error)

    monkeypatch.setattr(repo, "_state_graph_evidence", capture_once)
    worker = Thread(target=acquire)
    worker.start()
    try:
        assert entered.wait(10)
        repo.set_default_store(second)
        with pytest.raises(RuntimeError, match="topology"):
            repo.add_store(added)
    finally:
        release.set()
        worker.join(10)

    assert not worker.is_alive()
    assert calls == [value]
    assert outcome == [expected]
    assert repo.stores == [second, first]
    assert not os.path.exists(os.path.join(added.base_dir, "managed"))


def test_zip_lifetime_lock_is_process_visible_and_releases_after_abrupt_exit(tmp_path):
    """Zip ownership uses a sibling path lock, never extraction or archive commit locks."""

    archive = os.fspath(tmp_path / "state.zip")
    repo = Repo((ZipStore(archive),))
    value = OwnedValue(1, repo=repo)
    object_ids = _object_ids(value)
    path = _state_lock_path(repo.default_store, object_ids[0])
    assert ".dryml-managed" in path
    assert ".dryml.lock" not in path
    assert repo.default_store._tmp.name not in path
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    child = context.Process(target=_zip_lock_owner, args=(archive, object_ids, ready, release))
    child.start()
    try:
        assert ready.wait(10)
        with pytest.raises(ManagedConflictError, match="state lock"):
            _acquire_state_locks(repo, object_ids)
        release.set()
        child.join(10)
        assert child.exitcode == 0
        lease = _acquire_state_locks(repo, object_ids)
        lease.release()
    finally:
        release.set()
        if child.is_alive():
            child.terminate()
            child.join(5)
