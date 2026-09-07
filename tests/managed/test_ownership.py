"""U5 lifetime ownership coverage for selected managed state Stores."""

from __future__ import annotations

from threading import Event, Thread

import pytest

from dryml.core import Object, Repo, Serializable
from dryml.core.store.dir import DirStore
from dryml.locking import FileLock, LockError
from dryml.managed.control import ManagedControlStore
from dryml.managed.errors import ManagedConflictError, ManagedStoreError
from dryml.managed import storage as storage_module
from dryml.managed.storage import (
    _acquire_state_ownership,
    _probe_state_ownership,
    _state_lock_path,
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


def test_ownership_composes_graph_reservation_and_state_store_lock_set(tmp_path):
    """A stateless root retains every descendant lock through the active lease."""

    store = DirStore(tmp_path / "state")
    duplicate_handle = DirStore(tmp_path / "state")
    repo = Repo((store,))
    root = OwnedRoot(OwnedValue(1, repo=repo), repo=repo)

    control_one = ManagedControlStore(DirStore(tmp_path / "control-one"), store)
    control_two = ManagedControlStore(DirStore(tmp_path / "control-two"), duplicate_handle)
    with _acquire_state_ownership(repo, root, store) as owner:
        assert owner.active
        assert owner.reservation.active
        assert owner.object_ids == _object_ids(root)
        assert _state_lock_path(store, owner.object_ids[0]).endswith(".lock")
        assert not control_one.probe_state_ownership(owner.object_ids)
        assert not control_two.probe_state_ownership(owner.object_ids)
        with pytest.raises(ManagedConflictError, match="already reserved"):
            _acquire_state_ownership(repo, root, duplicate_handle)

    assert _probe_state_ownership(duplicate_handle, _object_ids(root))


def test_partial_lock_acquisition_releases_earlier_state_leases(tmp_path):
    """A contested later ObjectId leaves an earlier acquired ObjectId available."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    root = OwnedRoot(OwnedValue(1, repo=repo), OwnedValue(2, repo=repo), repo=repo)
    object_ids = tuple(sorted(_object_ids(root), key=lambda value: _state_lock_path(store, value)))
    first_path = _state_lock_path(store, object_ids[0])
    ManagedControlStore(store, store).initialize()
    held = FileLock(_state_lock_path(store, object_ids[1]))
    assert held.acquire()
    try:
        with pytest.raises(ManagedConflictError, match="state lock"):
            _acquire_state_ownership(repo, root, store)
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
    owner = _acquire_state_ownership(repo, first, store)
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
        with _acquire_state_ownership(repo, second, store):
            pass
        with pytest.raises(ManagedConflictError, match="already reserved"):
            _acquire_state_ownership(repo, first, store)
    finally:
        owner.release()


def test_pending_claim_in_another_state_store_is_rejected_without_abandoning_it(tmp_path):
    """Managed state selection rejects claims before generic core save can consume them."""

    declaration_store = DirStore(tmp_path / "declaration")
    selected_store = DirStore(tmp_path / "selected")
    repo = Repo((declaration_store, selected_store))
    reference = repo.declare_object(OwnedValue(1).definition, store=declaration_store)
    live = repo.build_object_ref(reference, store=declaration_store)
    claim = declaration_store.read_claim_record(reference.digest())

    with pytest.raises(ManagedStoreError, match="initial state"):
        _acquire_state_ownership(repo, live, selected_store)

    assert declaration_store.read_claim_record(reference.digest()) == claim
    assert live._claim_lease is not None


def test_probe_releases_partial_leases_and_does_not_classify_contention_as_owner_loss(tmp_path):
    """The U6 probe reports availability only when it acquired every requested lease."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    root = OwnedRoot(OwnedValue(1, repo=repo), OwnedValue(2, repo=repo), repo=repo)
    object_ids = tuple(sorted(_object_ids(root), key=lambda value: _state_lock_path(store, value)))
    ManagedControlStore(store, store).initialize()
    held = FileLock(_state_lock_path(store, object_ids[1]))
    assert held.acquire()
    try:
        assert not _probe_state_ownership(store, object_ids)
        first = FileLock(_state_lock_path(store, object_ids[0]))
        assert first.acquire(blocking=False)
        first.release()
    finally:
        held.release()
    assert _probe_state_ownership(store, object_ids)


def test_probe_propagates_native_lock_failure_instead_of_calling_it_owner_loss(tmp_path, monkeypatch):
    """A capability error remains actionable rather than becoming inconclusive contention."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    obj = OwnedValue(1, repo=repo)
    ManagedControlStore(store, store).initialize()

    class BrokenLock:
        def __init__(self, path):
            self.path = path

        def acquire(self, *, blocking):
            raise LockError("native failure")

    monkeypatch.setattr(storage_module, "FileLock", BrokenLock)
    with pytest.raises(ManagedStoreError, match="acquire"):
        _probe_state_ownership(store, _object_ids(obj))
