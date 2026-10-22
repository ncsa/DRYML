"""Detached historical status, targeted requests and producer/caller distinction."""

from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

from dryml.core.query.model import QueryIndexUnavailable
from dryml.core.utils.graph.path import Parameter
from dryml.managed import ManagedConfig
from dryml.managed.results import ResultReference
from dryml.managed.errors import ManagedControlError
from tests.managed.test_control_codec import saved_receiver
from tests.managed.test_u14_authority import Experiment, adapter, facts, receiver, running


def test_absent_status_read_only_and_selected_domain_specific(tmp_path):
    control, identity = adapter(tmp_path), facts(receiver())
    status = control.status(identity.operation_id)
    assert (status.state, status.run_id, status.invocation_id, status.generation) == ("not_started", None, None, 0)
    assert not Path(control.root).exists()


def test_detached_status_and_replay_admission_after_checkpoint_reclamation(tmp_path, monkeypatch):
    control = adapter(tmp_path)
    node = saved_receiver(control)
    identity = facts(node)
    with control.admission(identity) as producer:
        completed = replace(running(identity, producer), state="completed", checkpoint=node,
                            final=node, producer_invocation_id=producer.invocation_id, result_reference=ResultReference("f" * 64))
        control.create_initial(completed, owner=producer)
    key = node.at((Parameter("dataset"),)).state.key
    assert control.control_store.delete_state(key).status == "removed"
    def forbidden(*args, **kwargs): pytest.fail("metadata path opened reclaimed payload")
    monkeypatch.setattr(control.control_store, "read_state", forbidden)
    monkeypatch.setattr(control.control_store, "_authenticate_state_data", forbidden)
    status = control.status(identity.operation_id)
    assert status.state == "completed" and status.final.graph_equal(node)
    with control.admission(identity) as caller:
        assert caller.run_id == producer.run_id
        assert caller.invocation_id != status.producer_invocation_id == producer.invocation_id
        assert caller.current.final.graph_equal(node)


def test_external_request_preconditions_and_dead_owner_are_read_only(tmp_path):
    control, identity = adapter(tmp_path), facts(receiver())
    with control.admission(identity) as owner:
        initial = control.create_initial(running(identity, owner), owner=owner)
        for kwargs, outcome in (({"expected_run_id": uuid4().hex}, "stale_run"),
                                ({"expected_invocation_id": uuid4().hex}, "stale_invocation"),
                                ({"expected_generation": 2}, "stale_generation")):
            assert control.request_interrupt_if_running(identity.operation_id, **kwargs) == (outcome, initial)
    assert control.status(identity.operation_id).failure_code == "owner_lost"
    assert control.request_interrupt_if_running(identity.operation_id) == ("not_running", initial)
    assert control.inspect(identity.operation_id) == initial


def test_actual_failed_body_retains_detached_failure_status(tmp_path):
    control = adapter(tmp_path)
    from tests.managed.test_u14_authority import Dataset
    obj = Experiment(Dataset(repo=control.state_repo), repo=control.state_repo)
    with pytest.raises(AssertionError, match="foundation") as error:
        obj.run(managed=ManagedConfig(state_repo=control.state_repo))
    assert error.value.failed_result.status.state == "failed"
    assert obj.run.status(state_repo=control.state_repo).state == "failed"
    assert obj.run.request_interrupt(state_repo=control.state_repo,
        expected_run_id=uuid4().hex).outcome == "not_running"


def test_lifecycle_projection_has_complete_registered_owner_coverage(tmp_path):
    control = adapter(tmp_path, index="sqlite")
    index = control.control_store.open_query_index()
    assert index.projection_rows("lifecycle", refresh=True) == ()


def test_completion_projection_error_retains_completion_and_mutation_cut(tmp_path, monkeypatch):
    from dryml.core.store.authority import StoreMutationError
    control = adapter(tmp_path, index="sqlite")
    node = saved_receiver(control)
    identity = facts(node)
    with control.admission(identity) as owner:
        control.create_initial(running(identity, owner), owner=owner)
        index = control.control_store.open_query_index()
        original = index.maintain_store
        def fail_after_cas(*args, **kwargs):
            if control.control_store.authority_cut().pending:
                raise OSError("controlled completion projection failure")
            return original(*args, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(index, "maintain_store", fail_after_cas)
            with pytest.raises(StoreMutationError) as error:
                control.transition_running_owner(identity.operation_id, owner=owner,
                    run_id=owner.run_id, invocation_id=owner.invocation_id, expected_generation=1,
                    build=lambda current: replace(current, generation=2, state="completed",
                        final=node, producer_invocation_id=owner.invocation_id, result_reference=ResultReference("f" * 64)))
        assert error.value.mutation_id and error.value.cut.pending
        assert control.inspect(identity.operation_id).state == "completed"
    with control.admission(identity) as recovered:
        assert recovered.reconciled and recovered.current.state == "completed"


@pytest.mark.parametrize("state", ["running", "finalizing"])
def test_accepted_interrupt_preserves_phase_and_fences_completion(tmp_path, state):
    from dryml.managed.errors import ManagedConflictError
    control = adapter(tmp_path)
    node = saved_receiver(control)
    identity = facts(node)
    with control.admission(identity) as owner:
        initial = replace(running(identity, owner), state=state)
        control.create_initial(initial, owner=owner)
        outcome, requested = control.request_interrupt_if_running(identity.operation_id)
        assert outcome == "requested" and requested.state == state
        with pytest.raises(ManagedConflictError, match="interrupted"):
            control.transition(identity.operation_id, replace(requested, state="completed",
                generation=3, interrupt_request=None, final=node, producer_invocation_id=owner.invocation_id,
                result_reference=ResultReference("f" * 64)),
                owner=owner, expected_generation=2, expected_run_id=owner.run_id,
                expected_invocation_id=owner.invocation_id)
