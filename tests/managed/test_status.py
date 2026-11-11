"""Detached historical status, targeted requests and producer/caller distinction."""

from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import pytest

from dryml.core.utils.graph.path import Parameter
from dryml.managed import ManagedConfig
from dryml.managed.results import ResultReference
from tests.managed.test_control_codec import saved_receiver
from tests.managed.test_u14_authority import Experiment, adapter, facts, receiver, running


def test_public_reference_status_selects_repo_without_namespace_or_payload_io(tmp_path, monkeypatch):
    """A query reference can be inspected directly without constructing its class."""
    from dryml.core import Definition
    from dryml.core.session import config
    from dryml.managed import ManagedControlStore, status
    from tests.managed.test_u14_authority import Dataset

    control, node = adapter(tmp_path), receiver()
    def forbidden(*args, **kwargs):
        pytest.fail("inspection constructed, loaded, initialized, or closed a resource")
    monkeypatch.setattr(Experiment, "__init__", forbidden)
    monkeypatch.setattr(Dataset, "__init__", forbidden)
    monkeypatch.setattr(control.control_store, "read_state", forbidden)
    monkeypatch.setattr(control.control_store, "close", forbidden)
    monkeypatch.setattr(control.state_repo, "close", forbidden)
    monkeypatch.setattr(ManagedControlStore, "initialize", forbidden)
    with config(repo=control.state_repo):
        selected = ManagedControlStore.from_repo()
        assert selected.state_repo is control.state_repo
        assert selected.control_store is control.control_store
        assert status(node, "run").state == "not_started"
        assert status(node, "other_member").state == "not_started"
    assert selected.status_for(node, "run").state == "not_started"
    assert not Path(control.root).exists()
    # Missing stateful identities cannot stand in for independent instances.
    from dryml.managed import ManagedConfigError
    with pytest.raises(ManagedConfigError):
        status(Definition(Dataset).concretize(), "run", repo=control.state_repo)


def test_public_status_store_selection_matches_invocation_and_never_federates(tmp_path):
    from dryml.core import Repo
    from dryml.core.store.dir import DirStore
    from dryml.managed import ManagedControlStore, status
    from dryml.managed.defaults import _control_store_defaults

    control = adapter(tmp_path)
    node = saved_receiver(control)
    identity = facts(node)
    with control.admission(identity) as owner:
        current = replace(running(identity, owner), state="completed", final=node,
                          producer_invocation_id=owner.invocation_id,
                          result_reference=ResultReference("f" * 64))
        control.create_initial(current, owner=owner)
    first = DirStore(tmp_path / "empty")
    repo = Repo((first, control.control_store))
    assert status(node, "run", repo=repo).state == "not_started"
    assert status(node, "run", repo=repo, control_store=control.control_store).state == "completed"
    with _control_store_defaults(control.control_store):
        assert ManagedControlStore.from_repo(repo).control_store is control.control_store
        assert status(node, "run", repo=repo).state == "completed"
        assert status(node, "run", repo=repo, control_store=first).state == "not_started"


def test_public_status_complete_history_survives_reclamation_and_matches_live_view(tmp_path, monkeypatch):
    from dryml.managed import status
    from tests.managed.u16_fixtures import Value, setup

    control, live = setup(tmp_path)
    monkeypatch.setattr(Value, "action", lambda obj, managed: None)
    result = live.run(managed=ManagedConfig(state_repo=control.state_repo))
    saved = result.final_state_ref
    control.cleanup_checkpoints(result.execution.operation_id)
    assert not Path(control.control_store.state_data_path(saved.state.key)).exists()
    def forbidden(*args, **kwargs):
        pytest.fail("metadata inspection resolved or restored unavailable receiver")
    monkeypatch.setattr(control.control_store, "read_state", forbidden)
    monkeypatch.setattr(Value, "__init__", forbidden)
    monkeypatch.setattr(Value, "restore_state_from_dir_imp", forbidden)
    expected = live.run.status(state_repo=control.state_repo)
    for ref in (saved, saved.object_view(), live):
        actual = status(ref, "run", repo=control.state_repo)
        assert actual.state == "completed"
        assert (actual.operation_id, actual.run_id, actual.invocation_id, actual.generation,
                actual.producer_invocation_id) == (
                    expected.operation_id, expected.run_id, expected.invocation_id,
                    expected.generation, expected.producer_invocation_id)
        assert actual.final.graph_equal(saved)


def test_public_status_running_failed_and_interrupted_are_not_completed(tmp_path):
    from dryml.managed import status

    control, node = adapter(tmp_path), receiver()
    identity = facts(node)
    with control.admission(identity) as owner:
        current = control.create_initial(running(identity, owner), owner=owner)
        assert status(node, "run", repo=control.state_repo).state == "running"
        interrupted = control.transition(identity.operation_id,
            replace(current, generation=2, state="interrupted"), owner=owner,
            expected_generation=1, expected_run_id=owner.run_id,
            expected_invocation_id=owner.invocation_id)
        assert status(node, "run", repo=control.state_repo).state == "interrupted"
        control.transition(identity.operation_id,
            replace(interrupted, generation=3, state="failed", failure_code="body_error"),
            owner=owner, expected_generation=2, expected_run_id=owner.run_id,
            expected_invocation_id=owner.invocation_id)
    actual = status(node, "run", repo=control.state_repo)
    assert actual.state == "failed" and actual.failure_code == "body_error"


def test_public_status_reports_owner_loss_and_propagates_pending_authority(tmp_path, monkeypatch):
    from dryml.managed import ManagedControlError, status
    from dryml.managed.control import ManagedControlStore

    control, node = adapter(tmp_path), receiver()
    with control.admission(facts(node)) as owner:
        control.create_initial(running(facts(node), owner), owner=owner)
    actual = status(node, "run", repo=control.state_repo)
    assert (actual.state, actual.failure_code) == ("failed", "owner_lost")
    def pending(*args):
        raise ManagedControlError("pending_reconciliation", "controlled pending authority")
    monkeypatch.setattr(ManagedControlStore, "inspect", pending)
    with pytest.raises(ManagedControlError, match="pending_reconciliation"):
        status(node, "run", repo=control.state_repo)


def test_public_status_verifies_full_receiver_evidence_and_resource_types(tmp_path, monkeypatch):
    from dryml.managed import ManagedConfigError, ManagedConflictError, ManagedControlStore, ManagedStoreError, status

    control, node = adapter(tmp_path), receiver()
    with control.admission(facts(receiver())) as owner:
        different = running(owner.identity, owner)
    monkeypatch.setattr(control, "observe_running_owner", lambda *args: (different, False, True))
    with pytest.raises(ManagedConflictError, match="operation_collision"):
        control.status_for(node, "run")
    for invalid in (None, object(), "not metadata"):
        with pytest.raises(ManagedConfigError):
            status(invalid, "run", repo=control.state_repo)
    with pytest.raises(ManagedConfigError):
        status(node, "not a member", repo=control.state_repo)
    with pytest.raises(ManagedStoreError, match="invalid_state_repo"):
        ManagedControlStore.from_repo(control.control_store)
    with pytest.raises(ManagedStoreError, match="invalid_control_store"):
        status(node, "run", repo=control.state_repo, control_store=object())


def test_query_candidate_can_restart_fresh_with_the_same_operation_identity(tmp_path, monkeypatch):
    """Retry a failed independent object instead of creating a replacement copy."""
    from dryml.core import Definition, Repo
    from dryml.managed import status
    from tests.managed.u16_fixtures import Value, setup

    control, live = setup(tmp_path)
    def fail_after_checkpoint(obj, context):
        obj.value = 17
        context.checkpoint()
        raise ValueError("controlled first-attempt failure")
    monkeypatch.setattr(Value, "action", fail_after_checkpoint)
    with pytest.raises(ValueError):
        live.run(managed=ManagedConfig(state_repo=control.state_repo))
    candidate = control.state_repo.query().sel(Definition(Value)).object_projection().one()
    failed = status(candidate, "run", repo=control.state_repo)
    assert failed.state == "failed" and failed.checkpoint is not None
    restart = Definition(Value).concretize().with_realization(identities=candidate.identity_spec())
    assert restart.object_projection_equal(candidate)
    fresh_repo = Repo(control.control_store)
    try:
        fresh = fresh_repo.build(restart, cache="none")
        assert fresh.value == 0
        monkeypatch.setattr(Value, "action", lambda obj, context: obj.value)
        result = fresh.run(managed=ManagedConfig(state_repo=fresh_repo, rerun=True))
        assert result.value == 0
        assert result.execution.operation_id == failed.operation_id
        assert result.execution.run_id != failed.run_id
        assert result.final_state_ref.object_id == candidate.object_id
        assert status(candidate, "run", repo=fresh_repo).state == "completed"
        assert fresh_repo.query().sel(Definition(Value)).object_projection().count() == 1
    finally:
        fresh_repo.close(flush=False)


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


@pytest.mark.parametrize("incomplete", [False, True], ids=["missing-owner", "incomplete-owner"])
def test_cold_core_refuses_actual_managed_authority_without_complete_owner(tmp_path, incomplete):
    """Persisted lifecycle rows cannot substitute for an available owner reader."""
    control = adapter(tmp_path, index="sqlite")
    identity = facts(receiver())
    with control.admission(identity) as owner:
        current = control.create_initial(running(identity, owner), owner=owner)
    store = control.control_store
    index = store.open_query_index()
    rows = index.projection_rows("lifecycle", refresh=False)
    assert any(key == "current" for _, key, _ in rows)
    cut = store.authority_cut()
    generation = index.status().generation
    file_identity = index._connections._path_identity()
    script = f'''
from contextlib import closing
import sys
import sqlite3
from dryml.core import Definition
from dryml.core.store.dir import DirStore
from dryml.core.store.store import StoreAuthorityError
from dryml.core.query.authority import register_control_projection
from dryml.core.query.model import QueryIndexDirty, QueryIndexUnavailable
assert "dryml.managed" not in sys.modules
if {incomplete!r}:
    register_control_projection("managed", "managed", lambda path: None)
store = DirStore({store.base_dir!r}, query_index="sqlite")
index = store.open_query_index()
assert index.status().state != "ready"
# An older/incomplete owner feed may have explicitly marked lifecycle unsupported.
with closing(sqlite3.connect(index.path)) as con, con:
    con.execute("UPDATE store_projection_coverage SET supported=0 WHERE family='lifecycle'")
assert index.status().state != "ready"
for query in (store.query().cdefs().stored(), store.query().cdefs().stored().sel(Definition(int))):
    try:
        query.refresh(False).count()
    except QueryIndexDirty:
        pass
    else:
        raise AssertionError("Missing owner certified a count")
try:
    index.projection_rows("lifecycle", refresh=False)
except QueryIndexUnavailable:
    pass
else:
    raise AssertionError("Missing owner certified lifecycle rows")
try:
    index.maintain_store(rebuild=True)
except StoreAuthorityError:
    pass
else:
    raise AssertionError("Missing owner rebuilt incomplete coverage")
assert "dryml.managed" not in sys.modules
'''
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
    assert store.authority_cut() == cut
    assert control.inspect(identity.operation_id) == current
    assert index.status().generation == generation
    assert index._connections._path_identity() == file_identity
    # A refused rebuild marks derived state dirty; the loaded owner can repair it.
    index.maintain_store()
    assert index.projection_rows("lifecycle", refresh=False) == rows


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
