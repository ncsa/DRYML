"""U15 selected-start precedence, recovery seeding and retained physical roots."""

from dataclasses import replace
from uuid import UUID, uuid4

import pytest

from dryml.core.repo import RepoLoadError
from dryml.managed import ManagedConfig
from tests.managed.u15_fixtures import Value, enter, identity, seed, setup


def test_override_b_never_opens_unavailable_unselected_initial_a(tmp_path, monkeypatch):
    control, live, a = setup(tmp_path)
    live.value = 2
    b = control.state_repo.save(live)
    old = seed(control, live, b, state="completed", initial=a)
    Value.captures = 0
    original = control.control_store.read_states
    opened = []
    def selected(keys):
        keys = tuple(keys)
        assert a.state.key not in keys
        opened.extend(keys)
        return original(keys)
    monkeypatch.setattr(control.control_store, "read_states", selected)
    new, value, resuming = enter(control, live, config=ManagedConfig(rerun=True, start_from=b))
    assert value == 2 and not resuming and opened and set(opened) == {b.state.key}
    assert new.initial.graph_equal(a) and new.checkpoint.graph_equal(b)
    assert new.operation_id == old.operation_id and new.run_id != old.run_id
    assert (Value.restores, Value.captures, Value.bodies) == (1, 0, 0)
    # The override did not replace the immutable default: the next no-override
    # rerun selects unavailable A and cannot fall back to its usable B recovery.
    with pytest.raises(RepoLoadError):
        enter(control, live, config=ManagedConfig(rerun=True))
    assert control.inspect(old.operation_id) == new and Value.restores == 1


def test_invalid_selected_b_never_falls_back_to_initial_a(tmp_path, monkeypatch):
    control, live, a = setup(tmp_path)
    live.value = 2
    b = control.state_repo.save(live)
    old = seed(control, live, a, initial=a)
    original = control.control_store.read_states
    def selected(keys):
        keys = tuple(keys)
        assert a.state.key not in keys
        raise RepoLoadError("B unavailable")
    monkeypatch.setattr(control.control_store, "read_states", selected)
    with pytest.raises(RepoLoadError):
        enter(control, live, config=ManagedConfig(rerun=True, start_from=b))
    assert control.inspect(old.operation_id) == old and live.value == 2
    monkeypatch.setattr(control.control_store, "read_states", original)
    recovered = enter(control, live)
    assert recovered.disposition == "preparation_abandoned"
    assert (Value.restores, Value.bodies) == (0, 0)


def test_rerun_initial_seeds_recovery_and_resume_does_not_resave(tmp_path):
    control, live, a = setup(tmp_path)
    old = seed(control, live, a, state="completed", initial=a)
    live.value = 81
    def fail(*args):
        raise ValueError("execution seam failure")
    with pytest.raises(ValueError):
        enter(control, live, config=ManagedConfig(rerun=True), amount=3, execute=fail)
    failed = control.inspect(old.operation_id)
    assert failed.run_id != old.run_id and failed.checkpoint.graph_equal(a)
    live.value = 92
    resumed, value, resuming = enter(control, live, amount=3)
    assert value == 1 and resuming and resumed.run_id == failed.run_id
    assert resumed.initial.graph_equal(a)
    assert (Value.restores, Value.captures, Value.bodies) == (2, 0, 0)


def test_selected_prior_start_has_new_grant_before_old_root_release(tmp_path, monkeypatch):
    control, live, saved = setup(tmp_path)
    first, _, _ = enter(control, live, config=ManagedConfig(rerun=True, start_from=saved))
    old_root = first.intents[-1]
    calls = []
    # retention is a property returning wrappers, so observe the owning class.
    from dryml.core.store.retention import RetentionAuthority
    release = RetentionAuthority.release
    def observed(self, token, stores):
        current = control.inspect(first.operation_id)
        assert current.run_id != first.run_id
        new_root = current.intents[-1]
        assert self.read(str(UUID(new_root.token))).disposition == "adopted"
        calls.append(token)
        return release(self, token, stores)
    monkeypatch.setattr(RetentionAuthority, "release", observed)
    second, _, _ = enter(control, live, config=ManagedConfig(rerun=True, start_from=saved))
    assert calls == [str(UUID(old_root.token))]
    assert control.control_store.retention.read(calls[0]).disposition == "released"
    assert second.intents[-1].token != old_root.token
    from dryml.core.store.retention import grants
    assert grants(control.control_store, saved.state.key)


def test_retired_invocation_cannot_reconcile_or_release_new_run_grants(tmp_path, monkeypatch):
    from dryml.managed import runtime
    from dryml.managed.errors import ManagedConflictError
    from dryml.core.store.retention import RetentionAuthority
    control, live, saved = setup(tmp_path)
    owners = []
    enter(control, live, config=ManagedConfig(rerun=True, start_from=saved),
          execute=lambda owner, *args: owners.append(owner))
    new, _, _ = enter(control, live, config=ManagedConfig(rerun=True, start_from=saved))
    preparation = control.read_preparation(new.operation_id)
    root = new.intents[-1]
    ledger = control.control_store.retention.read(str(UUID(root.token)))
    monkeypatch.setattr(RetentionAuthority, "release", lambda *args, **kwargs: pytest.fail("stale physical release"))
    with pytest.raises(ManagedConflictError, match="invalid_operation_owner"):
        runtime._settle_preparation(control, owners[0], preparation)
    with pytest.raises(ManagedConflictError):
        control.authorize_release(new.operation_id, root.token, owner=owners[0],
            run_id=new.run_id, invocation_id=new.invocation_id, expected_generation=new.generation)
    assert control.control_store.retention.read(ledger.token) == ledger
