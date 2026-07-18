from __future__ import annotations

import pytest

from dryml.core2.store.dir import DirStore
from dryml.formats.refs import format_cdef_id
from dryml.managed import (
    ManagedInputValidationRequiredError,
    ManagedOperationStore,
    ManagedRerunRequiredError,
    ManagedStateError,
    OperationKey,
    StaleManagedResultError,
)


KEY = OperationKey(format_cdef_id("7" * 64), "compute")
FP = "managed-declaration-v1-" + "8" * 64


def _operation(path):
    return ManagedOperationStore(DirStore(path)).operation(KEY, FP)


def _complete_and_activate(operation):
    with operation.acquire() as lease:
        decision = lease.prepare(resumable=True)
        realization_id = decision.realization.realization_id
        lease._complete_control_only(realization_id)
        lease._activate_control_only(realization_id)
    return realization_id


def test_pending_resumes_normally_and_explicit_rerun_retains_abandoned_history(tmp_path):
    operation = _operation(tmp_path / "store")
    with operation.acquire() as lease:
        first = lease.prepare(resumable=True)
        lease.interrupt(first.realization.realization_id, checkpoint_head="checkpoint-v1-1")

    with operation.acquire() as lease:
        resumed = lease.prepare(resumable=True)
        assert resumed.action == "resume"
        assert resumed.realization.realization_id == first.realization.realization_id
        assert len(resumed.realization.attempt_ids) == 2
        rerun = lease.prepare(resumable=False, rerun=True)
        assert rerun.action == "rerun"
        assert rerun.realization.realization_id != first.realization.realization_id

    history = operation.history()
    assert [item.status for item in history] == ["abandoned", "running"]


def test_non_resumable_pending_requires_explicit_rerun(tmp_path):
    operation = _operation(tmp_path / "store")
    with operation.acquire() as lease:
        first = lease.prepare(resumable=False)
        lease.fail(first.realization.realization_id, diagnostic="worker failed")

    with operation.acquire() as lease:
        with pytest.raises(ManagedRerunRequiredError, match="explicit rerun"):
            lease.prepare(resumable=False)


def test_acquisition_recovers_one_orphaned_resumable_realization(tmp_path, monkeypatch):
    operation = _operation(tmp_path / "store")
    lease = operation.acquire()
    original = lease._write_control_for

    def fail_after_realization_write(state):
        raise OSError("simulated control write interruption")

    monkeypatch.setattr(lease, "_write_control_for", fail_after_realization_write)
    with pytest.raises(OSError, match="simulated"):
        lease.prepare(resumable=True)
    orphan_id = operation.history()[0].realization_id
    monkeypatch.setattr(lease, "_write_control_for", original)
    lease.release()

    with operation.acquire() as recovered:
        decision = recovered.prepare(resumable=True)

    assert decision.action == "resume"
    assert decision.realization.realization_id == orphan_id


def test_explicit_realization_id_cannot_overwrite_retained_history(tmp_path):
    operation = _operation(tmp_path / "store")
    with operation.acquire() as lease:
        first = lease.prepare(resumable=True)
        lease.interrupt(first.realization.realization_id)
        with pytest.raises(ManagedStateError, match="already exists"):
            lease.prepare(
                resumable=True,
                rerun=True,
                realization_id=first.realization.realization_id,
            )

    assert operation.history()[0].status == "interrupted"


def test_completed_reuse_requires_explicit_stable_input_validation_hook(tmp_path):
    operation = _operation(tmp_path / "store")
    active_id = _complete_and_activate(operation)

    with operation.acquire() as lease:
        with pytest.raises(ManagedInputValidationRequiredError):
            lease.prepare(resumable=True)
        with pytest.raises(StaleManagedResultError):
            lease.prepare(resumable=True, active_inputs_valid=False)
        reused = lease.prepare(resumable=True, active_inputs_valid=True)

    assert reused.action == "reuse"
    assert reused.realization.realization_id == active_id


def test_completed_reuse_can_run_later_stable_input_validator_under_lease(tmp_path):
    operation = _operation(tmp_path / "store")
    active_id = _complete_and_activate(operation)
    observed = []

    with operation.acquire() as lease:
        reused = lease.prepare(
            resumable=True,
            active_inputs_valid=lambda active: observed.append(active.realization_id) or True,
        )

    assert reused.action == "reuse"
    assert observed == [active_id]


@pytest.mark.parametrize("outcome", ["failed", "interrupted"])
def test_prior_active_survives_incomplete_rerun(tmp_path, outcome):
    operation = _operation(tmp_path / outcome)
    old_id = _complete_and_activate(operation)

    with operation.acquire() as lease:
        rerun = lease.prepare(resumable=True, rerun=True)
        assert operation.active().realization_id == old_id
        if outcome == "failed":
            lease.fail(rerun.realization.realization_id)
        else:
            lease.interrupt(rerun.realization.realization_id)

    assert operation.active().realization_id == old_id


def test_activation_is_pointer_last_supports_rollback_and_rebuild(tmp_path):
    operation = _operation(tmp_path / "store")
    first_id = _complete_and_activate(operation)
    with operation.acquire() as lease:
        second = lease.prepare(resumable=False, rerun=True)
        second_id = second.realization.realization_id
        lease._complete_control_only(second_id)
        lease._activate_control_only(second_id)
        assert operation.active().realization_id == second_id
        lease._activate_control_only(first_id)

    assert operation.active().realization_id == first_id
    operation.active_pointer_path.unlink()
    assert operation.rebuild_active_pointer().realization_id == first_id
    assert operation.active().realization_id == first_id

    reopened = _operation(tmp_path / "store")
    assert [item.realization_id for item in reopened.history()] == [first_id, second_id]
    assert reopened.active().realization_id == first_id
