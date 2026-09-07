"""U5 spawned-process and defensive-fork managed ownership coverage."""

from __future__ import annotations

import multiprocessing
import os

import pytest

from dryml.core import Repo, Serializable
from dryml.core.reference_values import ObjectId
from dryml.core.store.dir import DirStore
from dryml.managed.errors import ManagedConflictError
from dryml.managed.storage import _acquire_state_locks, _acquire_state_ownership


class ProcessValue(Serializable):
    """Minimal live value used when a fork test needs a graph reservation too."""

    def __init__(self, value):
        self.value = value


def _hold_locks_then_exit(root, object_ids, ready, release):
    """Spawn target retaining state leases until deliberate process exit."""

    lease = _acquire_state_locks(DirStore(root), object_ids)
    assert lease.active
    ready.set()
    assert release.wait(10)
    os._exit(0)


def test_spawned_owner_exit_releases_all_managed_state_locks(tmp_path):
    """Kernel ownership ends on spawned owner exit without a timeout takeover."""

    context = multiprocessing.get_context("spawn")
    root = os.fspath(tmp_path / "state")
    object_ids = (ObjectId(), ObjectId())
    ready = context.Event()
    release = context.Event()
    owner = context.Process(target=_hold_locks_then_exit, args=(root, object_ids, ready, release))
    owner.start()
    try:
        assert ready.wait(10)
        with pytest.raises(Exception):
            _acquire_state_locks(DirStore(root), object_ids)
        release.set()
        owner.join(10)
        assert owner.exitcode == 0
        lease = _acquire_state_locks(DirStore(root), object_ids)
        lease.release()
    finally:
        release.set()
        if owner.is_alive():
            owner.terminate()
            owner.join(5)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is unavailable on this host")
def test_forked_managed_token_cannot_release_parent_or_bypass_parent_lock(tmp_path):
    """Forked children invalidate domain tokens while shared locking retains parent ownership."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    obj = ProcessValue(1, repo=repo)
    owner = _acquire_state_ownership(repo, obj, store)
    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read_fd)
        try:
            try:
                contender = _acquire_state_locks(store, owner.object_ids)
            except ManagedConflictError:
                acquired = False
            else:
                acquired = contender.active
                contender.release()
            os.write(write_fd, f"{owner.active}:{owner.release()}:{acquired}".encode("ascii"))
        finally:
            os.close(write_fd)
        os._exit(0)

    os.close(write_fd)
    try:
        assert os.read(read_fd, 32) == b"False:False:False"
        _, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        assert owner.active
    finally:
        os.close(read_fd)
        owner.release()
