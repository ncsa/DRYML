"""U6 synchronous managed-operation lifecycle coverage."""

from __future__ import annotations

import pytest
from threading import Event, Thread

from dryml.core import Object, Repo
from dryml.core.object import Pickleable
from dryml.core.store.dir import DirStore
from dryml.managed import ManagedConfig, managed_operation
from dryml.managed import control as control_module
from dryml.managed.control import ManagedControlStore
from dryml.managed.errors import ManagedControlError, ManagedPublicationError
from dryml.runtime.errors import RuntimeTransitionError
from dryml import session


class LifecycleValue(Pickleable):
    """Small stateful managed receiver used for lifecycle publication tests."""

    def __init__(self, value=0):
        self.value = value

    @managed_operation(resumable=True)
    def add(self, amount, *, managed):
        """Mutate the value and return the ordinary method result."""

        assert managed.active
        assert managed.is_resuming is False
        self.value += amount
        return self.value


def test_start_deep_saves_completes_and_returns_the_ordinary_result(tmp_path):
    """A fresh synchronous invocation commits completion before its return value."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    value = LifecycleValue(2, repo=repo)

    assert value.add(3, managed=ManagedConfig(state_store=store)) == 5
    status = value.add.status(state_store=store)
    assert (status.state, status.generation) == ("completed", 2)
    assert status.attempt_id is not None
    assert status.final_state_ref == value.last_state_ref
    assert status.checkpoint_state_ref is None


class StatelessRoot(Object):
    """Root with transient fields and a stateful descendant managed as one graph."""

    def __init__(self, child):
        self.child = child
        self.transient = "not-state"

    @managed_operation()
    def add_child(self, amount, *, managed):
        """Mutate only the retained stateful descendant."""

        self.child.value += amount
        return self.child.value


class FailingValue(Pickleable):
    """Receiver whose ordinary exception must be recorded then preserved."""

    @managed_operation()
    def fail(self, *, managed):
        """Raise the exact application exception after lifecycle entry."""

        raise ValueError("application failure")


class LateRequestValue(Pickleable):
    """Receiver whose method waits while another caller publishes a request."""

    entered = Event()
    release = Event()

    def __init__(self, value=0):
        self.value = value

    @managed_operation()
    def finish(self, *, managed):
        """Return normally after a concurrent request without a safe point."""

        type(self).entered.set()
        assert type(self).release.wait(10)
        self.value += 1
        return self.value

    @managed_operation()
    def fail_after_request(self, *, managed):
        """Preserve the original application failure after a concurrent request."""

        type(self).entered.set()
        assert type(self).release.wait(10)
        raise ValueError("late requested failure")


def test_stateless_root_deep_saves_its_stateful_descendant(tmp_path):
    """A root without local state still receives final graph authority."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    root = StatelessRoot(LifecycleValue(1, repo=repo), repo=repo)

    assert root.add_child(4, managed=ManagedConfig(state_store=store)) == 5
    assert root.last_state_ref is not None
    assert root.child.last_state_ref is None


def test_method_error_is_recorded_without_replacing_the_application_exception(tmp_path):
    """Best-effort terminal failure preserves the exact ordinary method error."""

    store = DirStore(tmp_path / "state")
    value = FailingValue(repo=Repo((store,)))

    with pytest.raises(ValueError, match="application failure"):
        value.fail(managed=ManagedConfig(state_store=store))
    status = value.fail.status(state_store=store)
    assert (status.state, status.failure_code) == ("failed", "method_error")


def test_indeterminate_failure_recording_is_chained_from_the_method_error(tmp_path, monkeypatch):
    """Failure publication uncertainty preserves the application exception as cause."""

    class IndeterminateFailure(Pickleable):
        """Receiver that exposes its exact ordinary error for chain assertions."""

        def __init__(self):
            self.error = ValueError("application failure")

        @managed_operation()
        def fail(self, *, managed):
            """Raise the retained application error after lifecycle entry."""

            raise self.error

    store = DirStore(tmp_path / "state")
    value = IndeterminateFailure(repo=Repo((store,)))
    original = ManagedControlStore._clear_intent_and_acknowledge

    def leave_failed_pending(self, operation, snapshot, digest):
        if snapshot.state == "failed":
            raise ManagedPublicationError("indeterminate", "failure acknowledgement is uncertain")
        return original(self, operation, snapshot, digest)

    monkeypatch.setattr(ManagedControlStore, "_clear_intent_and_acknowledge", leave_failed_pending)
    with pytest.raises(ManagedPublicationError) as caught:
        value.fail(managed=ManagedConfig(state_store=store))
    assert (caught.value.outcome, caught.value.__cause__) == ("indeterminate", value.error)


@pytest.mark.parametrize("member, expected", [("finish", "completed"), ("fail_after_request", "failed")])
def test_late_request_does_not_stale_owner_completion_or_failure(tmp_path, member, expected):
    """A same-owner request advances generation without invalidating terminal cleanup."""

    LateRequestValue.entered, LateRequestValue.release = Event(), Event()
    store = DirStore(tmp_path / "state")
    value = LateRequestValue(repo=Repo((store,)))
    operation = getattr(value, member)
    outcome = []

    def invoke():
        try:
            outcome.append(operation(managed=ManagedConfig(state_store=store)))
        except BaseException as error:
            outcome.append(error)

    worker = Thread(target=invoke)
    worker.start()
    try:
        assert LateRequestValue.entered.wait(10)
        assert operation.request_interrupt(state_store=store).outcome == "requested"
        LateRequestValue.release.set()
        worker.join(10)
        assert not worker.is_alive()
    finally:
        LateRequestValue.release.set()
        worker.join(10)

    if member == "finish":
        assert outcome == [1]
    else:
        assert len(outcome) == 1
        assert type(outcome[0]) is ValueError
        assert str(outcome[0]) == "late requested failure"
    assert operation.status(state_store=store).state == expected


def test_indeterminate_completion_preserves_pending_authority_and_withholds_result(tmp_path, monkeypatch):
    """An uncertain final association is not replaced by an invented failure."""

    store = DirStore(tmp_path / "state")
    value = LifecycleValue(2, repo=Repo((store,)))
    original = ManagedControlStore._clear_intent_and_acknowledge

    def leave_completed_pending(self, operation, snapshot, digest):
        if snapshot.state == "completed":
            raise ManagedPublicationError("indeterminate", "completion acknowledgement is uncertain")
        return original(self, operation, snapshot, digest)

    monkeypatch.setattr(ManagedControlStore, "_clear_intent_and_acknowledge", leave_completed_pending)
    with pytest.raises(ManagedPublicationError) as caught:
        value.add(3, managed=ManagedConfig(state_store=store))
    assert caught.value.outcome == "indeterminate"
    control = ManagedControlStore(store, store)
    # Inspection must remain blocked until exact pending reconciliation completes.
    with pytest.raises(ManagedControlError, match="pending_reconciliation"):
        value.add.status(state_store=store)
    # Recover through the adapter without treating the ordinary return as delivered.
    from dryml.managed.identity import operation_digest

    monkeypatch.setattr(ManagedControlStore, "_clear_intent_and_acknowledge", original)
    control.reconcile(operation_digest(value.object_ref, "add"))
    assert value.add.status(state_store=store).state == "completed"


def test_final_save_failure_records_failed_without_a_completion_association(tmp_path, monkeypatch):
    """A pre-association final publication error preserves the original failure."""

    store = DirStore(tmp_path / "state")
    value = LifecycleValue(2, repo=Repo((store,)))

    def fail_final_save(*args, **kwargs):
        raise OSError("state publication failed")

    monkeypatch.setattr(Repo, "save_object", fail_final_save)
    with pytest.raises(OSError, match="state publication failed"):
        value.add(3, managed=ManagedConfig(state_store=store))
    status = value.add.status(state_store=store)
    assert (status.state, status.failure_code, status.final_state_ref) == ("failed", "publication_error", None)


class AdmissionValue(Pickleable):
    """Receiver proving runtime strict mode blocks managed workload entry."""

    calls = 0

    @managed_operation()
    def run(self, *, managed):
        """Record workload execution only after managed runtime admission."""

        type(self).calls += 1


def test_managed_workload_requires_a_materialized_receiver_and_runtime_admission(tmp_path):
    """Bad receivers are typed errors and strict orchestration cannot run methods."""

    from dryml.managed.runtime import invoke
    from dryml.managed import ManagedConfigError

    with pytest.raises(ManagedConfigError, match="materialized Object"):
        invoke(AdmissionValue.run, object(), (), None, {})

    store = DirStore(tmp_path / "state")
    value = AdmissionValue(repo=Repo((store,)))
    AdmissionValue.calls = 0
    session.set_mode("orchestrator")
    try:
        with pytest.raises(RuntimeTransitionError, match="prohibits Object materialization"):
            value.run(managed=ManagedConfig(state_store=store))
        assert AdmissionValue.calls == 0
    finally:
        session.reset()
