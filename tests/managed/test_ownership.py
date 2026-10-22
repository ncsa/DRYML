"""Logical ownership scope and retained Core state-IO exclusivity."""

from dataclasses import replace
from threading import Event, Thread
from uuid import uuid4

import pytest

from dryml.core import ObjectId, Repo
from dryml.core.state import is_reserved
from dryml.core.store.dir import DirStore
from dryml.managed.control import ControlIntent, ManagedControlStore
from dryml.managed.errors import ManagedConflictError
from tests.managed.test_u14_authority import Dataset, adapter, facts, receiver, running


def test_duplicate_handles_same_domain_conflict_but_selected_domains_independent(tmp_path):
    control, identity = adapter(tmp_path), facts(receiver())
    second_handle = DirStore(control.control_store.base_dir, query_index="none")
    same_domain = ManagedControlStore(second_handle, Repo(second_handle))
    other_store = DirStore(tmp_path / "other", query_index="none")
    other_domain = ManagedControlStore(other_store, Repo(other_store))
    with control.admission(identity):
        with pytest.raises(ManagedConflictError, match="already_running"):
            same_domain.admission(identity)
        with other_domain.admission(identity): pass


def test_distinct_shared_live_dataset_operations_overlap_in_threads(tmp_path):
    control = adapter(tmp_path)
    shared = Dataset(repo=control.state_repo)
    first = facts(receiver(shared.definition.with_identity(shared.object_id)))
    second = facts(first.receiver.with_identity(ObjectId()))
    entered, release, results = Event(), Event(), []
    def sibling():
        try:
            with control.admission(second):
                assert not is_reserved(shared)
                entered.set()
                assert release.wait(10)
                results.append("ok")
        except BaseException as error: results.append(error); entered.set()
    with control.admission(first):
        thread = Thread(target=sibling); thread.start()
        try:
            assert entered.wait(10)
            assert not results
        finally:
            release.set(); thread.join(10)
    assert results == ["ok"]


def test_state_io_reservations_still_exclude_actual_shared_live_mutation(tmp_path):
    from dryml.core.repo import RepoSaveError
    control = adapter(tmp_path)
    shared = Dataset(repo=control.state_repo)
    with control.state_repo.reserve_state_graph(shared):
        assert is_reserved(shared)
        with pytest.raises(RepoSaveError, match="reserved"):
            control.state_repo.reserve_state_graph(shared)
        with control.admission(facts(receiver(shared.definition.with_identity(shared.object_id)))):
            pass
    assert not is_reserved(shared)


def test_stale_owner_cannot_release_new_lock_or_another_invocation_grant_intent(tmp_path):
    control, identity = adapter(tmp_path), facts(receiver())
    with control.admission(identity) as first:
        old_intent = ControlIntent(uuid4().hex, first.run_id, first.invocation_id, 1, "release_pending")
        initial = control.create_initial(replace(running(identity, first), intents=(old_intent,)), owner=first)
    with control.admission(identity) as second:
        current = control.transition(identity.operation_id,
            replace(initial, invocation_id=second.invocation_id, generation=2), owner=second,
            expected_generation=1, expected_run_id=initial.run_id, expected_invocation_id=initial.invocation_id)
        assert not first.release() and second.active
        current = control.transition_running_owner(identity.operation_id, owner=second,
            run_id=second.run_id, invocation_id=second.invocation_id, expected_generation=2,
            build=lambda current: replace(current, generation=3, intents=(old_intent,)))
        with pytest.raises(ManagedConflictError):
            control.authorize_release(identity.operation_id, old_intent.token, owner=second,
                run_id=second.run_id, invocation_id=second.invocation_id, expected_generation=3)
        with pytest.raises(ManagedConflictError):
            control.authorize_release(identity.operation_id, old_intent.token, owner=first,
                run_id=first.run_id, invocation_id=first.invocation_id, expected_generation=1)
        with pytest.raises(ManagedConflictError, match="intent_conflict"):
            control.transition_running_owner(identity.operation_id, owner=second,
                run_id=second.run_id, invocation_id=second.invocation_id, expected_generation=3,
                build=lambda current: replace(current, generation=4,
                    intents=(replace(old_intent, disposition="released"),)))
