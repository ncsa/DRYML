"""U6 authority-only managed-status coverage."""

from __future__ import annotations

import multiprocessing
import os
import shutil
from uuid import uuid4

import pytest

pytestmark = pytest.mark.usefixtures("fixed_managed_snapshot_environment")

from dryml.core import Repo
from dryml.managed.storage import ownership_evidence
from dryml.core.object import Pickleable
from dryml.core.store.dir import DirStore
from dryml.managed import ManagedConfig, managed_operation
from dryml.managed.control import ControlSnapshot, ManagedControlStore
from dryml.managed.identity import argument_digest, operation_digest
from dryml.managed.storage import _acquire_state_locks, _acquire_state_ownership
from dryml.managed.errors import ManagedRecoveryError, ManagedStoreError


def _spawn_lock_owner_then_exit(root, object_ids, ready, exit_now):
    """Retain selected state locks in a spawned process until deliberate exit."""

    _acquire_state_locks(Repo((DirStore(root),)), object_ids)
    ready.set()
    assert exit_now.wait(10)
    os._exit(0)


class StatusValue(Pickleable):
    """Stateful receiver used to inspect uninitialized selected authority."""

    @managed_operation()
    def run(self, *, managed):
        """Provide a checked operation without running it during status inspection."""

        raise AssertionError("status must not execute the managed workload")


def test_status_is_not_started_for_the_selected_absent_authority_without_hooks(tmp_path):
    """Status reads only the requested control Store and leaves workload untouched."""

    state = DirStore(tmp_path / "state")
    other_control = DirStore(tmp_path / "other-control")
    value = StatusValue(repo=Repo((state,)))

    status = value.run.status(state_repo=state, control_store=other_control)
    assert (status.state, status.attempt_id, status.generation) == ("not_started", None, 0)
    assert status.operation_id


class SeparateControlValue(Pickleable):
    """Receiver used to demonstrate selected-control-store responsibility."""

    def __init__(self, value=0):
        self.value = value

    @managed_operation()
    def run(self, *, managed):
        """Complete normally in whichever explicit control Store was selected."""

        self.value += 1
        return self.value


def _running_snapshot(repo, value, arguments, *, generation=1, attempt_id=None, owner_id=None):
    """Build retained v2 ownership evidence from the real selected Repo graph."""

    return ControlSnapshot(
        operation_digest(value.object_ref, "run"), value.object_ref.digest(), arguments,
        "run", attempt_id or uuid4().hex, owner_id or uuid4().hex, generation,
        "running", None, None, None, None,
        ownership_evidence(repo, value.object_ref.objects.values()),
    )


def test_wrong_selected_control_store_has_independent_not_started_metadata(tmp_path):
    """Managed intentionally does not search for lifecycle authority in other Stores."""

    state = DirStore(tmp_path / "state")
    selected = DirStore(tmp_path / "selected-control")
    wrong = DirStore(tmp_path / "wrong-control")
    value = SeparateControlValue(repo=Repo((state,)))

    assert value.run(managed=ManagedConfig(state_repo=state, control_store=selected)) == 1
    assert value.run.status(state_repo=state, control_store=selected).state == "completed"
    assert value.run.status(state_repo=state, control_store=wrong).state == "not_started"


def test_owner_loss_status_and_request_are_read_only_after_full_free_probe(tmp_path):
    """A dead recorded owner is not overwritten by status or request handling."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    value = SeparateControlValue(repo=repo)
    operation_id = operation_digest(value.object_ref, "run")
    arguments = argument_digest(SeparateControlValue.run, value, (), {"managed": None})
    initial = _running_snapshot(repo, value, arguments)
    control = ManagedControlStore(store, repo)
    control.create_initial(initial)
    seeded = _acquire_state_locks(repo, value.object_ref.objects.values())
    seeded.release()

    observed = value.run.status(state_repo=store)
    requested = value.run.request_interrupt(state_repo=store)
    assert (observed.state, observed.failure_code) == ("failed", "owner_lost")
    assert requested.outcome == "not_running"
    assert control.inspect(operation_id) == initial


def test_request_validates_stale_attempt_and_merges_only_the_current_owner(tmp_path):
    """An owner still holding state locks accepts only a current exact request."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    value = SeparateControlValue(repo=repo)
    operation_id = operation_digest(value.object_ref, "run")
    arguments = argument_digest(SeparateControlValue.run, value, (), {"managed": None})
    attempt_id, owner_id = uuid4().hex, uuid4().hex
    control = ManagedControlStore(store, repo)
    control.create_initial(_running_snapshot(repo, value, arguments, attempt_id=attempt_id, owner_id=owner_id))

    with _acquire_state_ownership(repo, value):
        stale = value.run.request_interrupt(state_repo=store, expected_attempt_id=uuid4().hex)
        requested = value.run.request_interrupt(state_repo=store, expected_attempt_id=attempt_id)
        repeated = value.run.request_interrupt(state_repo=store, expected_attempt_id=attempt_id)
    assert stale.outcome == "stale_attempt"
    assert (requested.outcome, requested.generation) == ("requested", 2)
    assert (repeated.outcome, repeated.generation) == ("already_requested", 2)


def test_spawned_owner_exit_then_request_does_not_write_running_authority(tmp_path):
    """A free complete lock set after spawn exit produces read-only not-running."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    value = SeparateControlValue(repo=repo)
    operation_id = operation_digest(value.object_ref, "run")
    arguments = argument_digest(SeparateControlValue.run, value, (), {"managed": None})
    initial = _running_snapshot(repo, value, arguments)
    control = ManagedControlStore(store, repo)
    control.create_initial(initial)
    _, _, object_ids = repo._state_graph_evidence(value)
    context = multiprocessing.get_context("spawn")
    ready, exit_now = context.Event(), context.Event()
    owner = context.Process(
        target=_spawn_lock_owner_then_exit,
        args=(os.fspath(store.base_dir), object_ids, ready, exit_now),
    )
    owner.start()
    try:
        assert ready.wait(10)
        exit_now.set()
        owner.join(10)
        assert owner.exitcode == 0
        assert value.run.request_interrupt(state_repo=store).outcome == "not_running"
        assert control.inspect(operation_id) == initial
    finally:
        exit_now.set()
        if owner.is_alive():
            owner.terminate()
            owner.join(5)


def test_owner_probe_checks_stale_attempt_before_probe_and_propagates_adapter_errors(tmp_path, monkeypatch):
    """A stale request does not probe, while unavailable probes are not owner loss."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    value = SeparateControlValue(repo=repo)
    operation_id = operation_digest(value.object_ref, "run")
    arguments = argument_digest(SeparateControlValue.run, value, (), {"managed": None})
    control = ManagedControlStore(store, repo)
    control.create_initial(_running_snapshot(repo, value, arguments))

    def unavailable_probe(self, object_ids):
        raise ManagedStoreError("probe_unavailable", "state probe adapter failed")

    monkeypatch.setattr(ManagedControlStore, "probe_state_ownership", unavailable_probe)
    assert value.run.request_interrupt(
        state_repo=store, expected_attempt_id=uuid4().hex,
    ).outcome == "stale_attempt"
    with pytest.raises(ManagedStoreError, match="probe_unavailable"):
        value.run.status(state_repo=store)
    with pytest.raises(ManagedStoreError, match="probe_unavailable"):
        value.run.request_interrupt(state_repo=store)


def test_locked_owner_probe_marks_generation_replacement_inconclusive(tmp_path, monkeypatch):
    """A generation replacement during probing is never classified as owner loss."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    value = SeparateControlValue(repo=repo)
    operation_id = operation_digest(value.object_ref, "run")
    arguments = argument_digest(SeparateControlValue.run, value, (), {"managed": None})
    control = ManagedControlStore(store, repo)
    initial = control.create_initial(_running_snapshot(repo, value, arguments))
    replacement = _running_snapshot(
        repo, value, arguments, generation=2, attempt_id=initial.attempt_id,
    )

    def replace_during_probe(self, ownership):
        self._replace_file(self._current_path(self._operation_path(operation_id)), replacement.to_bytes())
        return True

    monkeypatch.setattr(ManagedControlStore, "probe_state_ownership", replace_during_probe)
    observed, ownerless, stable = control.observe_running_owner(
        operation_id,
    )
    assert (observed, ownerless, stable) == (replacement, False, False)


def test_status_never_substitutes_live_receiver_keys_for_retained_ownership(tmp_path):
    """A reduced Repo cannot declare an owner dead by probing only its own graph."""

    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    control_store = DirStore(tmp_path / "control")
    full_repo = Repo((first, second))
    value = SeparateControlValue(repo=full_repo)
    arguments = argument_digest(SeparateControlValue.run, value, (), {"managed": None})
    control = ManagedControlStore(control_store, full_repo)
    control.create_initial(_running_snapshot(full_repo, value, arguments))

    with pytest.raises(ManagedRecoveryError, match="ownership_store_missing"):
        value.run.status(state_repo=Repo((first,)), control_store=control_store)


def test_reopened_operation_rejects_different_state_store_before_workload_or_locks(tmp_path):
    """Retained control evidence cannot transfer an operation to a different Repo."""

    original = DirStore(tmp_path / "original")
    replacement = DirStore(tmp_path / "replacement")
    control_store = DirStore(tmp_path / "control")
    original_repo = Repo((original,))
    value = SeparateControlValue(repo=original_repo)
    arguments = argument_digest(SeparateControlValue.run, value, (), {"managed": None})
    control = ManagedControlStore(control_store, original_repo)
    initial = control.create_initial(_running_snapshot(original_repo, value, arguments))
    with pytest.raises(ManagedRecoveryError, match="ownership_mismatch"):
        value.run(managed=ManagedConfig(state_repo=Repo((replacement,)), control_store=control_store))

    assert value.value == 0
    assert control.inspect(initial.operation_id) == initial
    assert not os.path.exists(os.path.join(replacement.base_dir, "managed"))


def test_missing_lock_namespace_never_becomes_owner_loss_or_recovery_evidence(tmp_path):
    """Status/request probing cannot recreate removed state-lock evidence for a running owner."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    value = SeparateControlValue(repo=repo)
    operation_id = operation_digest(value.object_ref, "run")
    arguments = argument_digest(SeparateControlValue.run, value, (), {"managed": None})
    initial = _running_snapshot(repo, value, arguments)
    control = ManagedControlStore(store, repo)
    control.create_initial(initial)
    lease = _acquire_state_locks(repo, value.object_ref.objects.values())
    lease.release()
    lock_path = os.path.join(store.base_dir, "managed", "locks", "v1")
    shutil.rmtree(lock_path)

    with pytest.raises(ManagedRecoveryError, match="state_lock_namespace_missing"):
        value.run.status(state_repo=store)
    with pytest.raises(ManagedRecoveryError, match="state_lock_namespace_missing"):
        value.run.request_interrupt(state_repo=store)

    assert not os.path.exists(lock_path)
    assert control.inspect(operation_id) == initial
