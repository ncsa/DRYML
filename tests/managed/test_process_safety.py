"""U5 spawned-process and defensive-fork managed ownership coverage."""

from __future__ import annotations

import multiprocessing
import os
from threading import Thread

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

    lease = _acquire_state_locks(Repo((DirStore(root),)), object_ids)
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
            _acquire_state_locks(Repo((DirStore(root),)), object_ids)
        release.set()
        owner.join(10)
        assert owner.exitcode == 0
        lease = _acquire_state_locks(Repo((DirStore(root),)), object_ids)
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
    owner = _acquire_state_ownership(repo, obj)
    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read_fd)
        try:
            try:
                contender = _acquire_state_locks(repo, owner.object_ids)
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


def test_control_reference_proof_rejects_foreign_thread_and_expired_cut(tmp_path, fixed_snapshot_environment):
    """A populated control memo is usable only by the cut's live owning caller."""

    from dryml.managed.storage import _state_ref_for_digest, _validate_control_state_ref

    repo = Repo(DirStore(tmp_path / "state"))
    reference = repo.save_object(ProcessValue(1, repo=repo), deep_capture=True)
    errors = []
    with repo._read_cut() as cut:
        assert _state_ref_for_digest(repo, reference.digest(), cut=cut) == reference
        _validate_control_state_ref(repo, reference, cut)

        def foreign():
            for operation in (
                lambda: _state_ref_for_digest(repo, reference.digest(), cut=cut),
                lambda: _validate_control_state_ref(repo, reference, cut),
                lambda: cut.__exit__(None, None, None),
            ):
                with pytest.raises(RuntimeError, match="another thread"):
                    operation()
                errors.append(True)

        worker = Thread(target=foreign)
        worker.start()
        worker.join(10)
        assert not worker.is_alive() and errors == [True] * 3
        assert cut.active and len(cut.validated_state_refs) == 1
    assert cut.resolved_state_refs == {} and cut.validated_state_refs == []
    with pytest.raises(RuntimeError, match="inactive"):
        _validate_control_state_ref(repo, reference, cut)
    repo.close(flush=False)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is unavailable on this host")
def test_inherited_control_cut_cannot_use_or_release_parent_proof(tmp_path, fixed_snapshot_environment):
    """Fork rejection occurs before inherited locks or successful-proof lookup."""

    from dryml.managed.storage import _state_ref_for_digest, _validate_control_state_ref

    repo = Repo(DirStore(tmp_path / "state"))
    reference = repo.save_object(ProcessValue(1, repo=repo), deep_capture=True)
    with repo._read_cut() as cut:
        _state_ref_for_digest(repo, reference.digest(), cut=cut)
        _validate_control_state_ref(repo, reference, cut)
        child = os.fork()
        if child == 0:  # pragma: no cover - checked through process exit status.
            try:
                for operation in (
                    lambda: _state_ref_for_digest(repo, reference.digest(), cut=cut),
                    lambda: _validate_control_state_ref(repo, reference, cut),
                    lambda: cut.__exit__(None, None, None),
                ):
                    try:
                        operation()
                    except RuntimeError:
                        continue
                    os._exit(1)
                os._exit(0)
            except BaseException:
                os._exit(2)
        _, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        assert cut.active and len(cut.validated_state_refs) == 1
    assert repo._topology_leases == 0
    repo.close(flush=False)
