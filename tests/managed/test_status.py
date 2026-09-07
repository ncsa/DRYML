"""U6 authority-only managed-status coverage."""

from __future__ import annotations

import multiprocessing
import os
from uuid import uuid4

import pytest

from dryml.core import Repo
from dryml.core.object import Pickleable
from dryml.core.store.dir import DirStore
from dryml.managed import ManagedConfig, managed_operation
from dryml.managed.control import ControlSnapshot, ManagedControlStore
from dryml.managed.identity import argument_digest, operation_digest
from dryml.managed.storage import _acquire_state_locks, _acquire_state_ownership
from dryml.managed.errors import ManagedStoreError


def _spawn_lock_owner_then_exit(root, object_ids, ready, exit_now):
    """Retain selected state locks in a spawned process until deliberate exit."""

    _acquire_state_locks(DirStore(root), object_ids)
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

    status = value.run.status(state_store=state, control_store=other_control)
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


def test_wrong_selected_control_store_has_independent_not_started_metadata(tmp_path):
    """Managed intentionally does not search for lifecycle authority in other Stores."""

    state = DirStore(tmp_path / "state")
    selected = DirStore(tmp_path / "selected-control")
    wrong = DirStore(tmp_path / "wrong-control")
    value = SeparateControlValue(repo=Repo((state,)))

    assert value.run(managed=ManagedConfig(state_store=state, control_store=selected)) == 1
    assert value.run.status(state_store=state, control_store=selected).state == "completed"
    assert value.run.status(state_store=state, control_store=wrong).state == "not_started"


def test_owner_loss_status_and_request_are_read_only_after_full_free_probe(tmp_path):
    """A dead recorded owner is not overwritten by status or request handling."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    value = SeparateControlValue(repo=repo)
    operation_id = operation_digest(value.object_ref, "run")
    arguments = argument_digest(SeparateControlValue.run, value, (), {"managed": None})
    initial = ControlSnapshot(
        operation_id, value.object_ref.digest(), arguments, "run", uuid4().hex,
        uuid4().hex, 1, "running", None, None, None, None,
    )
    control = ManagedControlStore(store, store)
    control.create_initial(initial)

    observed = value.run.status(state_store=store)
    requested = value.run.request_interrupt(state_store=store)
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
    control = ManagedControlStore(store, store)
    control.create_initial(ControlSnapshot(
        operation_id, value.object_ref.digest(), arguments, "run", attempt_id,
        owner_id, 1, "running", None, None, None, None,
    ))

    with _acquire_state_ownership(repo, value, store):
        stale = value.run.request_interrupt(state_store=store, expected_attempt_id=uuid4().hex)
        requested = value.run.request_interrupt(state_store=store, expected_attempt_id=attempt_id)
        repeated = value.run.request_interrupt(state_store=store, expected_attempt_id=attempt_id)
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
    initial = ControlSnapshot(
        operation_id, value.object_ref.digest(), arguments, "run", uuid4().hex,
        uuid4().hex, 1, "running", None, None, None, None,
    )
    control = ManagedControlStore(store, store)
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
        assert value.run.request_interrupt(state_store=store).outcome == "not_running"
        assert control.inspect(operation_id) == initial
    finally:
        exit_now.set()
        if owner.is_alive():
            owner.terminate()
            owner.join(5)


def test_owner_probe_checks_stale_attempt_before_probe_and_propagates_adapter_errors(tmp_path, monkeypatch):
    """A stale request does not probe, while unavailable probes are not owner loss."""

    store = DirStore(tmp_path / "state")
    value = SeparateControlValue(repo=Repo((store,)))
    operation_id = operation_digest(value.object_ref, "run")
    arguments = argument_digest(SeparateControlValue.run, value, (), {"managed": None})
    control = ManagedControlStore(store, store)
    control.create_initial(ControlSnapshot(
        operation_id, value.object_ref.digest(), arguments, "run", uuid4().hex,
        uuid4().hex, 1, "running", None, None, None, None,
    ))

    def unavailable_probe(self, object_ids):
        raise ManagedStoreError("probe_unavailable", "state probe adapter failed")

    monkeypatch.setattr(ManagedControlStore, "probe_state_ownership", unavailable_probe)
    assert value.run.request_interrupt(
        state_store=store, expected_attempt_id=uuid4().hex,
    ).outcome == "stale_attempt"
    with pytest.raises(ManagedStoreError, match="probe_unavailable"):
        value.run.status(state_store=store)
    with pytest.raises(ManagedStoreError, match="probe_unavailable"):
        value.run.request_interrupt(state_store=store)


def test_locked_owner_probe_marks_generation_replacement_inconclusive(tmp_path, monkeypatch):
    """A generation replacement during probing is never classified as owner loss."""

    store = DirStore(tmp_path / "state")
    value = SeparateControlValue(repo=Repo((store,)))
    operation_id = operation_digest(value.object_ref, "run")
    arguments = argument_digest(SeparateControlValue.run, value, (), {"managed": None})
    control = ManagedControlStore(store, store)
    initial = control.create_initial(ControlSnapshot(
        operation_id, value.object_ref.digest(), arguments, "run", uuid4().hex,
        uuid4().hex, 1, "running", None, None, None, None,
    ))
    replacement = ControlSnapshot(
        operation_id, initial.object_ref_digest, initial.argument_digest, "run",
        initial.attempt_id, uuid4().hex, 2, "running", None, None, None, None,
    )

    def replace_during_probe(self, object_ids):
        self._replace_file(self._current_path(self._operation_path(operation_id)), replacement.to_bytes())
        return True

    monkeypatch.setattr(ManagedControlStore, "probe_state_ownership", replace_during_probe)
    observed, ownerless, stable = control.observe_running_owner(
        operation_id, tuple(value.object_ref.objects.values()),
    )
    assert (observed, ownerless, stable) == (replacement, False, False)
