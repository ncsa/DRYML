"""Managed lifecycle integration with the shared signature boundary."""

from __future__ import annotations

from uuid import uuid4

import pytest

pytestmark = pytest.mark.usefixtures("fixed_managed_snapshot_environment")

from dryml.core import Repo
from dryml.core.object import Pickleable
from dryml.core.repo import RepoLoadError
from dryml.core.reference_values import ObjectRef, StateRef
from dryml.core.signatures import Mat, Ref, SignatureError
from dryml.core.store.dir import DirStore
from dryml.managed import ManagedConfig, managed_operation
from dryml.managed.control import ControlSnapshot, ManagedControlStore
from dryml.managed.identity import _argument_digest_bound, operation_digest
from dryml.managed.storage import ownership_evidence
from dryml.managed import runtime as runtime_module


class SignatureValue(Pickleable):
    """Small stateful value used as both a receiver and a materialized argument."""

    def __init__(self, value=0):
        self.value = value
        self.calls = 0

    @managed_operation(resumable=True)
    def receive(self, value: Mat[StateRef], *, managed):
        """Record the delivered live snapshot value and return its live identity."""

        self.calls += 1
        return value

    @managed_operation()
    def opposing(self, value: Mat[StateRef], *, managed):
        """Expose input wrapper validation before lifecycle publication."""

        self.calls += 1
        return value

    @managed_operation()
    def return_conflict(self, *, managed) -> Mat[StateRef]:
        """Return an opposing reference assertion after an explicit checkpoint."""

        self.calls += 1
        managed.checkpoint()
        return Ref(self.last_state_ref)

    @managed_operation()
    def excluded(self: Mat[StateRef], *, managed: Mat[StateRef]):
        """Return the receiver while leaving excluded controls unconverted."""

        return self

    @managed_operation()
    def unannotated_return(self, *, managed):
        """Return retained structural authority through the default Mat slot."""

        return self.last_state_ref

    @managed_operation()
    def swallow_interrupt(self, *, managed) -> Mat[StateRef]:
        """Catch the safe-point exception without clearing terminal interruption."""

        try:
            managed.interrupt()
        except Exception:
            return self.last_state_ref

    @managed_operation()
    def receive_reference(self, value: Mat[StateRef], *, managed):
        """Expose the exact snapshot selected before managed argument delivery."""

        return value

    @managed_operation()
    def adopt(self, value: Mat[ObjectRef], *, managed):
        """Attach a newly materialized child for final-publication claim transfer."""

        self.child = value
        return value


def test_managed_mat_state_ref_digests_selected_snapshot_and_delivers_live_state(tmp_path):
    """Annotated StateRefs retain exact identity evidence before live Mat delivery."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    receiver = SignatureValue(repo=repo)
    value = SignatureValue(1, repo=repo)
    first = repo.save_object(value, deep_capture=True)
    value.value = 2
    second = repo.save_object(value, deep_capture=True)

    delivered = receiver.receive(first, managed=ManagedConfig(state_repo=repo))
    second_receiver = SignatureValue(repo=repo)
    second_receiver.receive(second, managed=ManagedConfig(state_repo=repo))
    control = ManagedControlStore(store, repo)
    first_digest = control.inspect(operation_digest(receiver.object_ref, "receive")).argument_digest
    second_digest = control.inspect(operation_digest(second_receiver.object_ref, "receive")).argument_digest

    assert delivered.value == 1
    assert delivered is not value
    assert first != second
    assert first_digest != second_digest
    assert receiver.receive.status(state_repo=repo).state == "completed"


def test_managed_signature_rejects_opposing_input_before_body_or_control_creation(tmp_path):
    """A Ref assertion cannot enter a Mat argument slot or publish a lifecycle."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    receiver = SignatureValue(repo=repo)
    snapshot = repo.save_object(receiver, deep_capture=True)

    with pytest.raises(SignatureError):
        receiver.opposing(Ref(snapshot), managed=ManagedConfig(state_repo=store))

    assert receiver.calls == 0
    assert receiver.opposing.status(state_repo=store).state == "not_started"


def test_managed_signature_preflight_closes_only_its_private_repo_wrapper(tmp_path, monkeypatch):
    """Every post-resolution preflight exit deterministically closes the wrapper."""

    store = DirStore(tmp_path / "state")
    external_repo = Repo((store,))
    receiver = SignatureValue(repo=external_repo)
    snapshot = external_repo.save_object(receiver, deep_capture=True)
    closed = []
    original_close = Repo.close

    def observe_close(self, *args, **kwargs):
        closed.append((self, kwargs.get("flush", True)))
        return original_close(self, *args, **kwargs)

    monkeypatch.setattr(Repo, "close", observe_close)
    monkeypatch.setattr(
        store, "close", lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("borrowed Store must not close")
        ),
    )

    with pytest.raises(SignatureError):
        receiver.opposing(Ref(snapshot), managed=ManagedConfig(state_repo=store))

    assert len(closed) == 1
    assert closed[0][0] is not external_repo
    assert closed[0][1] is False
    assert external_repo.default_store is store


def test_managed_return_normalization_records_failure_without_completion(tmp_path):
    """Return normalization follows the terminal guard and precedes final publication."""

    store = DirStore(tmp_path / "state")
    receiver = SignatureValue(repo=Repo((store,)))

    with pytest.raises(SignatureError):
        receiver.return_conflict(managed=ManagedConfig(state_repo=store))

    status = receiver.return_conflict.status(state_repo=store)
    assert receiver.calls == 1
    assert (status.state, status.failure_code, status.final_state_ref) == ("failed", "method_error", None)
    assert status.checkpoint_state_ref is not None


def test_managed_receiver_and_injected_control_are_not_signature_values(tmp_path):
    """Only ordinary authored slots participate in shared signature conversion."""

    store = DirStore(tmp_path / "state")
    receiver = SignatureValue(repo=Repo((store,)))

    result = receiver.excluded(managed=ManagedConfig(state_repo=store))

    assert result is receiver


def test_managed_unannotated_structural_return_uses_the_default_mat_boundary(tmp_path):
    """An unannotated StateRef return materializes before final publication."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    receiver = SignatureValue(repo=repo)
    repo.save_object(receiver, deep_capture=True)

    assert receiver.unannotated_return(managed=ManagedConfig(state_repo=repo)) is receiver


def test_managed_terminal_interrupt_suppresses_return_realization(tmp_path, monkeypatch):
    """Catching a terminal safe-point signal cannot enter the return Mat seam."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    receiver = SignatureValue(repo=repo)
    calls = []
    original = Repo.materialize_boundary

    def observe(self, roots, **kwargs):
        calls.append(roots)
        return original(self, roots, **kwargs)

    monkeypatch.setattr(Repo, "materialize_boundary", observe)
    with pytest.raises(Exception, match="interrupted"):
        receiver.swallow_interrupt(managed=ManagedConfig(state_repo=repo))

    assert calls == []
    assert receiver.swallow_interrupt.status(state_repo=repo).state == "interrupted"


def test_managed_argument_delivery_keeps_the_snapshot_pinned_at_digest_time(tmp_path, monkeypatch):
    """Later Store metadata cannot replace the exact StateRef selected for identity."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    receiver = SignatureValue(repo=repo)
    source = SignatureValue(1, repo=repo)
    first = repo.save_object(source, deep_capture=True)
    original = runtime_module._argument_digest_bound

    def mutate_after_digest(values):
        digest = original(values)
        source.value = 2
        repo.save_object(source, deep_capture=True)
        return digest

    monkeypatch.setattr(runtime_module, "_argument_digest_bound", mutate_after_digest)
    delivered = receiver.receive_reference(first.object, managed=ManagedConfig(state_repo=repo))
    control = ManagedControlStore(store, repo).inspect(operation_digest(receiver.object_ref, "receive_reference"))

    assert delivered.value == 1
    assert control.argument_digest == _argument_digest_bound((("value", first),))


@pytest.mark.parametrize("conflicting", (False, True))
def test_managed_resume_preflights_checkpoint_with_overlapping_mat_state_ref(tmp_path, conflicting):
    """Compatible overlap reuses ownership; conflicting overlap restores nothing."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    receiver = SignatureValue(4, repo=repo)
    checkpoint = repo.save_object(receiver, deep_capture=True)
    argument = checkpoint
    if conflicting:
        receiver.value = 7
        argument = repo.save_object(receiver, deep_capture=True)
    receiver.value = 99
    operation_id = operation_digest(receiver.object_ref, "receive")
    arguments = _argument_digest_bound((("value", argument),))
    ManagedControlStore(store, repo).create_initial(ControlSnapshot(
        operation_id,
        receiver.object_ref.digest(),
        arguments,
        "receive",
        uuid4().hex,
        None,
        1,
        "interrupted",
        None,
        checkpoint.digest(),
        None,
        None,
        ownership_evidence(repo, receiver.object_ref.objects.values()),
    ))

    if conflicting:
        with pytest.raises(RepoLoadError, match="incompatible effective"):
            receiver.receive(argument, managed=ManagedConfig(state_repo=repo))
        assert receiver.value == 99
        assert receiver.calls == 0
    else:
        delivered = receiver.receive(argument, managed=ManagedConfig(state_repo=repo))
        assert delivered is receiver
        assert receiver.value == 4
        assert receiver.calls == 1


def test_managed_argument_claim_transfer_survives_later_final_publication_failure(tmp_path, monkeypatch):
    """A delivered argument retains its claim when final control association fails."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    receiver = SignatureValue(repo=repo)
    reference = repo.declare_object(SignatureValue(7, repo=repo).definition)
    original = runtime_module._completion_boundary

    def fail_after_state(stage):
        if stage == "final_state_published":
            raise OSError("final association failed")
        return original(stage)

    monkeypatch.setattr(runtime_module, "_completion_boundary", fail_after_state)
    with pytest.raises(OSError, match="final association failed"):
        receiver.adopt(reference, managed=ManagedConfig(state_repo=repo))

    assert store.read_claim_record(reference.digest()).status == "claimed"
    assert receiver.child.object_ref == reference
    assert receiver.child._claim_lease is not None
    assert receiver.adopt.status(state_repo=repo).state == "failed"
