"""U6 resume coverage using a deliberately seeded associated checkpoint."""

from __future__ import annotations

from uuid import uuid4

from dryml.core import Repo
from dryml.core.object import Pickleable
from dryml.core.store.dir import DirStore
import pytest

from dryml.managed import ManagedConfig, ManagedRecoveryError, ManagedRerunRequiredError, managed_operation
from dryml.managed.control import ControlSnapshot, ManagedControlStore
from dryml.managed.identity import argument_digest, operation_digest
from dryml.managed.storage import ownership_evidence


class ResumeValue(Pickleable):
    """Stateful receiver whose method proves exact restoration before re-entry."""

    def __init__(self, value=0):
        self.value = value

    @managed_operation(resumable=True)
    def add(self, amount, *, managed):
        """Return scalar resume metadata after mutating restored state."""

        was_resuming = managed.is_resuming
        self.value += amount
        return was_resuming, self.value


def test_resume_restores_the_supplied_live_target_and_keeps_attempt_identity(tmp_path):
    """A compatible unfinished authority restores then re-enters under a new owner."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    value = ResumeValue(4, repo=repo)
    checkpoint = repo.save_object(value, store=store, deep_capture=True)
    operation = value.add
    operation_id = operation_digest(value.object_ref, "add")
    arguments = argument_digest(ResumeValue.add, value, (3,), {"managed": None})
    attempt_id = uuid4().hex
    ManagedControlStore(store, repo).create_initial(ControlSnapshot(
        operation_id, value.object_ref.digest(), arguments, "add", attempt_id, None,
        1, "interrupted", None, checkpoint.digest(), None, None,
        ownership_evidence(repo, value.object_ref.objects.values()),
    ))
    value.value = 99

    was_resuming, result = operation(3, managed=ManagedConfig(state_repo=store))
    status = operation.status(state_repo=store)
    assert (was_resuming, result, value.value) == (True, 7, 7)
    assert status.checkpoint_state_ref == checkpoint
    assert status.attempt_id == attempt_id
    assert status.state == "completed"


def test_resume_requires_matching_arguments_and_an_associated_checkpoint(tmp_path):
    """Incomplete or incompatible current authority cannot silently enter workload."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    value = ResumeValue(4, repo=repo)
    operation_id = operation_digest(value.object_ref, "add")
    arguments = argument_digest(ResumeValue.add, value, (3,), {"managed": None})
    authority = ManagedControlStore(store, repo)
    authority.create_initial(ControlSnapshot(
        operation_id, value.object_ref.digest(), arguments, "add", uuid4().hex, None,
        1, "interrupted", None, None, None, None,
        ownership_evidence(repo, value.object_ref.objects.values()),
    ))

    with pytest.raises(ManagedRerunRequiredError, match="rerun_required"):
        value.add(4, managed=ManagedConfig(state_repo=store))
    assert value.value == 4


class RestoreFailureValue(Pickleable):
    """Receiver with a controlled restore-hook failure for invalid-target recovery."""

    fail_restore = False

    def __init__(self, value=0):
        self.value = value

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        """Fail before the default payload hook when the class seam requests it."""

        if type(self).fail_restore:
            raise RuntimeError("restore failed")

    @managed_operation(resumable=True)
    def add(self, amount, *, managed):
        """Mutate restored state after a successful managed resume."""

        self.value += amount
        return self.value


def test_restore_failure_invalidates_old_target_but_fresh_load_can_resume(tmp_path):
    """Failed in-place restore retains checkpoint authority while retiring old live state."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    value = RestoreFailureValue(4, repo=repo)
    checkpoint = repo.save_object(value, store=store, deep_capture=True)
    operation_id = operation_digest(value.object_ref, "add")
    arguments = argument_digest(RestoreFailureValue.add, value, (3,), {"managed": None})
    attempt_id = uuid4().hex
    ManagedControlStore(store, repo).create_initial(ControlSnapshot(
        operation_id, value.object_ref.digest(), arguments, "add", attempt_id, None,
        1, "interrupted", None, checkpoint.digest(), None, None,
        ownership_evidence(repo, value.object_ref.objects.values()),
    ))
    RestoreFailureValue.fail_restore = True
    try:
        with pytest.raises(Exception, match="restore failed"):
            value.add(3, managed=ManagedConfig(state_repo=store))
        with pytest.raises(ManagedRecoveryError, match="fresh exact load"):
            value.add(3, managed=ManagedConfig(state_repo=store, rerun=True))
        assert value.add.status(state_repo=store).checkpoint_state_ref == checkpoint
        RestoreFailureValue.fail_restore = False
        fresh = Repo._for_state_io((store,))
        try:
            recovered = fresh.load_state_ref(checkpoint, reuse_live="never")
            assert recovered.add(3, managed=ManagedConfig(state_repo=store)) == 7
            assert recovered.add.status(state_repo=store).attempt_id == attempt_id
        finally:
            fresh.close()
    finally:
        RestoreFailureValue.fail_restore = False
