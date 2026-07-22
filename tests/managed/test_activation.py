from __future__ import annotations

import json
from pathlib import Path

import pytest

from dryml.core.store.dir import DirStore
from dryml.formats.refs import format_cdef_id
from dryml.managed import (
    ManagedActivationIndeterminateError,
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

    def fail_after_realization_write(state, *, reset_progress=False):
        raise OSError("simulated control write interruption")

    monkeypatch.setattr(lease, "_write_control_for", fail_after_realization_write)
    with pytest.raises(OSError, match="simulated"):
        lease.prepare(resumable=True)
    orphan_id = operation.history()[0].realization_id
    monkeypatch.setattr(lease, "_write_control_for", original)
    lease.release()

    def fail_history_scan():
        raise AssertionError("sequence-aware recovery must not scan retained history")

    monkeypatch.setattr(operation, "history", fail_history_scan)

    with operation.acquire() as recovered:
        decision = recovered.prepare(resumable=True)

    assert decision.action == "resume"
    assert decision.realization.realization_id == orphan_id


def test_rerun_sequence_uses_bounded_control_after_scaling_and_reopen(
    tmp_path, monkeypatch
):
    path = tmp_path / "store"
    operation = _operation(path)
    for expected_sequence in range(1, 33):
        with operation.acquire() as lease:
            decision = lease.prepare(
                resumable=False, rerun=expected_sequence > 1
            )
            assert decision.realization.sequence == expected_sequence
            lease._complete_control_only(decision.realization.realization_id)
            lease._activate_control_only(decision.realization.realization_id)

    reopened = _operation(path)

    def fail_history_scan():
        raise AssertionError("rerun startup must not scan retained history")

    monkeypatch.setattr(reopened, "history", fail_history_scan)
    with reopened.acquire() as lease:
        rerun = lease.prepare(resumable=False, rerun=True)

    assert rerun.realization.sequence == 33


def test_missing_reserved_realization_recovers_without_history_scan(
    tmp_path, monkeypatch
):
    path = tmp_path / "store"
    operation = _operation(path)
    lease = operation.acquire()
    original = operation._write_realization

    def interrupt_before_realization_write(state):
        raise KeyboardInterrupt("simulated process interruption")

    monkeypatch.setattr(
        operation, "_write_realization", interrupt_before_realization_write
    )
    with pytest.raises(KeyboardInterrupt, match="simulated"):
        lease.prepare(resumable=True)
    monkeypatch.setattr(operation, "_write_realization", original)
    lease.release()

    reopened = _operation(path)

    def fail_history_scan():
        raise AssertionError("reservation recovery must not scan retained history")

    monkeypatch.setattr(reopened, "history", fail_history_scan)
    with reopened.acquire() as recovered:
        decision = recovered.prepare(resumable=True)

    assert decision.action == "start"
    assert decision.realization.sequence == 2


@pytest.mark.parametrize("mismatched", [False, True])
def test_realization_directory_fsync_failure_preserves_recovery_reservation(
    tmp_path, monkeypatch, mismatched
):
    import dryml.managed.store as store_module

    path = tmp_path / "store"
    operation = _operation(path)
    lease = operation.acquire()
    original_fsync_directory = store_module._fsync_directory
    injected = False

    def fail_realization_directory_fsync(directory):
        nonlocal injected
        if Path(directory) == operation._realizations_dir and not injected:
            injected = True
            raise OSError("simulated realization directory fsync failure")
        return original_fsync_directory(directory)

    monkeypatch.setattr(
        store_module, "_fsync_directory", fail_realization_directory_fsync
    )
    with pytest.raises(ManagedStateError, match="durably written"):
        lease.prepare(resumable=True)

    control = operation._read_control()
    reserved_id = control.reserved_realization_id
    assert reserved_id is not None
    realization_path = operation._realization_path(reserved_id)
    assert realization_path.exists()
    with pytest.raises(ManagedStateError, match="unreconciled"):
        lease.prepare(resumable=True)

    monkeypatch.setattr(
        store_module, "_fsync_directory", original_fsync_directory
    )
    lease.release()

    if mismatched:
        payload = json.loads(realization_path.read_text())
        payload["sequence"] += 1
        realization_path.write_text(json.dumps(payload))
        with pytest.raises(ManagedStateError, match="reserved realization sequence"):
            operation.acquire()
        assert operation._read_control().reserved_realization_id == reserved_id
        return

    def fail_history_scan():
        raise AssertionError("reservation recovery must not scan retained history")

    monkeypatch.setattr(operation, "history", fail_history_scan)
    with operation.acquire() as recovered:
        decision = recovered.prepare(resumable=True)

    assert decision.action == "resume"
    assert decision.realization.realization_id == reserved_id


def test_legacy_control_scans_once_then_persists_sequence_cursor(
    tmp_path, monkeypatch
):
    path = tmp_path / "store"
    operation = _operation(path)
    _complete_and_activate(operation)
    control = json.loads(operation.control_path.read_text())
    for field in (
        "next_realization_sequence",
        "latest_realization_id",
        "reserved_realization_id",
    ):
        control.pop(field)
    operation.control_path.write_text(json.dumps(control))

    reopened = _operation(path)
    original_history = reopened.history
    scans = 0

    def count_history_scan():
        nonlocal scans
        scans += 1
        return original_history()

    monkeypatch.setattr(reopened, "history", count_history_scan)
    with reopened.acquire():
        pass

    assert scans == 1
    assert reopened._read_control().next_realization_sequence == 2

    migrated = _operation(path)
    monkeypatch.setattr(
        migrated,
        "history",
        lambda: (_ for _ in ()).throw(
            AssertionError("migrated control must not rescan retained history")
        ),
    )
    with migrated.acquire():
        pass


@pytest.mark.parametrize("value", [0, True, "2"])
def test_malformed_realization_sequence_cursor_fails_closed(tmp_path, value):
    operation = _operation(tmp_path / "store")
    _complete_and_activate(operation)
    control = json.loads(operation.control_path.read_text())
    control["next_realization_sequence"] = value
    operation.control_path.write_text(json.dumps(control))

    with pytest.raises(ManagedStateError, match="next realization sequence"):
        operation.acquire()


def test_incomplete_realization_sequence_markers_fail_closed(tmp_path):
    operation = _operation(tmp_path / "store")
    _complete_and_activate(operation)
    control = json.loads(operation.control_path.read_text())
    control.pop("reserved_realization_id")
    operation.control_path.write_text(json.dumps(control))

    with pytest.raises(ManagedStateError, match="sequence markers are incomplete"):
        operation.acquire()


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


def test_activation_event_failure_before_publication_preserves_old_active(
    tmp_path, monkeypatch
):
    import dryml.managed.store as store_module

    operation = _operation(tmp_path / "store")
    first_id = _complete_and_activate(operation)
    original_write_json = store_module._write_json
    injected = False

    def fail_event_publication(path, data, *, immutable=False):
        nonlocal injected
        if Path(path).parent == operation._activations_dir and not injected:
            injected = True
            raise ManagedStateError("simulated activation event publication failure")
        return original_write_json(path, data, immutable=immutable)

    monkeypatch.setattr(store_module, "_write_json", fail_event_publication)
    with operation.acquire() as lease:
        rerun = lease.prepare(resumable=False, rerun=True)
        rerun_id = rerun.realization.realization_id
        lease._complete_control_only(rerun_id)
        with pytest.raises(ManagedStateError, match="event publication failure"):
            lease._activate_control_only(rerun_id)

    assert operation.active().realization_id == first_id
    assert operation.rebuild_active_pointer().realization_id == first_id


@pytest.mark.parametrize("boundary", ["event", "pointer"])
def test_activation_reconciles_post_replace_directory_fsync_failure(
    tmp_path, monkeypatch, boundary
):
    import dryml.managed.store as store_module

    operation = _operation(tmp_path / boundary)
    _complete_and_activate(operation)
    original_fsync_directory = store_module._fsync_directory
    target = (
        operation._activations_dir
        if boundary == "event"
        else operation.generation_dir
    )
    injected = False

    def fail_once_after_replace(directory):
        nonlocal injected
        if Path(directory) == target and not injected:
            injected = True
            raise OSError(f"simulated {boundary} directory fsync failure")
        return original_fsync_directory(directory)

    with operation.acquire() as lease:
        rerun = lease.prepare(resumable=False, rerun=True)
        rerun_id = rerun.realization.realization_id
        lease._complete_control_only(rerun_id)
        monkeypatch.setattr(
            store_module, "_fsync_directory", fail_once_after_replace
        )
        activated = lease._activate_control_only(rerun_id)

    assert injected
    assert activated.realization_id == rerun_id
    assert operation.active().realization_id == rerun_id
    assert len(operation._activation_events()) == 2
    assert operation._activation_events()[-1] == operation.active_event()


def test_pointer_publication_failure_retries_authoritative_event(
    tmp_path, monkeypatch
):
    import dryml.managed.store as store_module

    operation = _operation(tmp_path / "store")
    _complete_and_activate(operation)
    original_write_json = store_module._write_json
    injected = False

    def fail_first_pointer_write(path, data, *, immutable=False):
        nonlocal injected
        if Path(path) == operation.active_pointer_path and not injected:
            injected = True
            raise ManagedStateError("simulated pointer publication failure")
        return original_write_json(path, data, immutable=immutable)

    monkeypatch.setattr(store_module, "_write_json", fail_first_pointer_write)
    with operation.acquire() as lease:
        rerun = lease.prepare(resumable=False, rerun=True)
        rerun_id = rerun.realization.realization_id
        lease._complete_control_only(rerun_id)
        activated = lease._activate_control_only(rerun_id)

    assert injected
    assert activated.realization_id == rerun_id
    assert operation.active().realization_id == rerun_id
    assert len(operation._activation_events()) == 2
    assert operation._activation_events()[-1] == operation.active_event()


@pytest.mark.parametrize("boundary", ["event", "pointer"])
def test_transient_post_commit_validation_read_is_retried(
    tmp_path, monkeypatch, boundary
):
    import dryml.managed.store as store_module

    operation = _operation(tmp_path / boundary)
    _complete_and_activate(operation)
    original_write_json = store_module._write_json
    original_read_json = store_module._read_json
    published_path = None
    injected = False

    def track_publication(path, data, *, immutable=False):
        nonlocal published_path
        result = original_write_json(path, data, immutable=immutable)
        path = Path(path)
        if boundary == "event" and path.parent == operation._activations_dir:
            published_path = path
        elif boundary == "pointer" and path == operation.active_pointer_path:
            published_path = path
        return result

    def fail_first_post_commit_read(path, name):
        nonlocal injected
        if (
            published_path is not None
            and Path(path) == published_path
            and not injected
        ):
            injected = True
            raise OSError(f"simulated {boundary} validation read failure")
        return original_read_json(path, name)

    monkeypatch.setattr(store_module, "_write_json", track_publication)
    monkeypatch.setattr(store_module, "_read_json", fail_first_post_commit_read)
    with operation.acquire() as lease:
        rerun = lease.prepare(resumable=False, rerun=True)
        rerun_id = rerun.realization.realization_id
        lease._complete_control_only(rerun_id)
        activated = lease._activate_control_only(rerun_id)

    assert injected
    assert activated.realization_id == rerun_id
    assert operation.active().realization_id == rerun_id


def test_unreadable_event_after_uncertain_write_reports_indeterminate(
    tmp_path, monkeypatch
):
    import dryml.managed.store as store_module

    operation = _operation(tmp_path / "store")
    first_id = _complete_and_activate(operation)
    original_write_json = store_module._write_json
    original_read_json = store_module._read_json
    published_event = None

    def publish_then_raise(path, data, *, immutable=False):
        nonlocal published_event
        result = original_write_json(path, data, immutable=immutable)
        if Path(path).parent == operation._activations_dir:
            published_event = Path(path)
            raise OSError("simulated lost activation write acknowledgement")
        return result

    def fail_event_reads(path, name):
        if published_event == Path(path):
            raise OSError("simulated unavailable activation event")
        return original_read_json(path, name)

    monkeypatch.setattr(store_module, "_write_json", publish_then_raise)
    monkeypatch.setattr(store_module, "_read_json", fail_event_reads)
    with operation.acquire() as lease:
        rerun = lease.prepare(resumable=False, rerun=True)
        rerun_id = rerun.realization.realization_id
        lease._complete_control_only(rerun_id)
        with pytest.raises(
            ManagedActivationIndeterminateError,
            match="publication is indeterminate",
        ):
            lease._activate_control_only(rerun_id)

    assert operation.active().realization_id == first_id
    monkeypatch.setattr(store_module, "_write_json", original_write_json)
    monkeypatch.setattr(store_module, "_read_json", original_read_json)
    assert operation.rebuild_active_pointer().realization_id == rerun_id


def test_mismatched_published_activation_event_fails_closed(tmp_path, monkeypatch):
    import dryml.managed.store as store_module

    operation = _operation(tmp_path / "store")
    first_id = _complete_and_activate(operation)
    original_write_json = store_module._write_json
    injected = False

    def publish_mismatched_event(path, data, *, immutable=False):
        nonlocal injected
        if Path(path).parent == operation._activations_dir and not injected:
            injected = True
            mismatched = dict(data)
            mismatched["realization_id"] = first_id
            original_write_json(path, mismatched, immutable=immutable)
            raise ManagedStateError("simulated uncertain event publication")
        return original_write_json(path, data, immutable=immutable)

    monkeypatch.setattr(store_module, "_write_json", publish_mismatched_event)
    with operation.acquire() as lease:
        rerun = lease.prepare(resumable=False, rerun=True)
        rerun_id = rerun.realization.realization_id
        lease._complete_control_only(rerun_id)
        with pytest.raises(ManagedStateError, match="does not match the proposed"):
            lease._activate_control_only(rerun_id)

    assert operation.active().realization_id == first_id
