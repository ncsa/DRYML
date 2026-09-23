"""U6 explicit-rerun current-state coverage."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.usefixtures("fixed_managed_snapshot_environment")

from dryml.core import Repo
from dryml.core.object import Pickleable
from dryml.core.store.dir import DirStore
from dryml.managed import ManagedConfig, ManagedRerunRequiredError, managed_operation
from dryml.managed import control as control_module
from dryml.managed.control import ControlSnapshot, ManagedControlStore
from dryml.managed.identity import argument_digest, operation_digest
from dryml.managed.storage import ownership_evidence


class RerunValue(Pickleable):
    """Stateful receiver used to prove rerun does not reset live state."""

    def __init__(self, value=0):
        self.value = value

    @managed_operation()
    def add(self, amount, *, managed):
        """Mutate current state for completion and rerun checks."""

        self.value += amount
        return self.value


def test_completed_default_requires_rerun_and_rerun_uses_current_state(tmp_path):
    """Rerun creates a new attempt while retaining the caller's current Object state."""

    store = DirStore(tmp_path / "state")
    value = RerunValue(1, repo=Repo((store,)))
    config = ManagedConfig(state_repo=store)
    assert value.add(2, managed=config) == 3
    first_attempt = value.add.status(state_repo=store).attempt_id
    with pytest.raises(ManagedRerunRequiredError, match="already_completed"):
        value.add(2, managed=config)
    assert value.add(4, managed=ManagedConfig(state_repo=store, rerun=True)) == 7
    status = value.add.status(state_repo=store)
    assert status.attempt_id != first_attempt
    assert status.state == "completed"


def test_invocation_reconciles_a_pending_completed_authority_before_deciding(tmp_path):
    """A recovered completed replacement is not mistaken for an orphan final state."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    value = RerunValue(1, repo=repo)
    final_state = repo.save_object(value, store=store, deep_capture=True)
    operation_id = operation_digest(value.object_ref, "add")
    arguments = argument_digest(RerunValue.add, value, (2,), {"managed": None})
    control = ManagedControlStore(store, repo)
    initial = control.create_initial(ControlSnapshot(
        operation_id, value.object_ref.digest(), arguments, "add", "a" * 32,
        "b" * 32, 1, "running", None, None, None, None,
        ownership_evidence(repo, value.object_ref.objects.values()),
    ))
    completed = ControlSnapshot(
        operation_id, value.object_ref.digest(), arguments, "add", initial.attempt_id,
        None, 2, "completed", None, None, final_state.digest(), None,
        initial.ownership,
    )
    operation = control._operation_path(operation_id)
    control._replace_file(control._current_path(operation), completed.to_bytes())
    intent = control_module._PendingIntent(
        operation_id, 1, control_module._payload_digest(initial.to_bytes()),
        2, control_module._payload_digest(completed.to_bytes()),
    )
    control._write_new(control._pending_path(operation), intent.to_bytes())

    with pytest.raises(ManagedRerunRequiredError, match="already_completed"):
        value.add(2, managed=ManagedConfig(state_repo=store))
    assert control.inspect(operation_id) == completed


class RerunResumeValue(Pickleable):
    """Resumable receiver used to retain a synthetic U7 checkpoint after rerun."""

    fail = True

    def __init__(self, value=0):
        self.value = value

    @managed_operation(resumable=True)
    def add(self, amount, *, managed):
        """Fail once so the rerun authority can be inspected before resume."""

        if type(self).fail:
            raise ValueError("rerun failure")
        self.value += amount
        return managed.is_resuming, self.value


def test_rerun_replaces_argument_digest_for_a_later_checkpoint_resume(tmp_path):
    """A rerun attempt records its new arguments rather than inheriting old ones."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    value = RerunResumeValue(4, repo=repo)
    checkpoint = repo.save_object(value, store=store, deep_capture=True)
    operation_id = operation_digest(value.object_ref, "add")
    old_arguments = argument_digest(RerunResumeValue.add, value, (2,), {"managed": None})
    new_arguments = argument_digest(RerunResumeValue.add, value, (3,), {"managed": None})
    control = ManagedControlStore(store, repo)
    control.create_initial(ControlSnapshot(
        operation_id, value.object_ref.digest(), old_arguments, "add", "a" * 32,
        None, 1, "interrupted", None, checkpoint.digest(), None, None,
        ownership_evidence(repo, value.object_ref.objects.values()),
    ))

    with pytest.raises(ValueError, match="rerun failure"):
        value.add(3, managed=ManagedConfig(state_repo=store, rerun=True))
    failed = control.inspect(operation_id)
    assert failed.argument_digest == new_arguments

    seeded = ControlSnapshot(
        operation_id, value.object_ref.digest(), new_arguments, "add", failed.attempt_id,
        None, failed.generation + 1, "failed", None, checkpoint.digest(), None, "method_error",
        failed.ownership,
    )
    control.transition(operation_id, seeded, expected_generation=failed.generation)
    value.value = 99
    RerunResumeValue.fail = False
    try:
        assert value.add(3, managed=ManagedConfig(state_repo=store)) == (True, 7)
    finally:
        RerunResumeValue.fail = True
