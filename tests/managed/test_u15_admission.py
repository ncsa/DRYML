"""Proof-first U15 metadata/start policy and two-phase failure boundaries."""

from dataclasses import replace
import os
from uuid import uuid4

import pytest

from dryml.managed import ManagedConfig, ManagedConfigError
from dryml.managed.errors import ManagedRerunRequiredError
from tests.managed.u15_fixtures import Value, enter, identity, seed, setup
from dryml.core import Object
from dryml.managed import managed_operation


@pytest.mark.parametrize("phase", [None, "completed", "failed"])
def test_start_without_rerun_rejects_all_phases_before_realization(tmp_path, phase):
    control, live, saved = setup(tmp_path)
    old = None if phase is None else seed(control, live, saved, state=phase)
    with pytest.raises(ManagedConfigError, match="start_requires_rerun"):
        enter(control, live, config=ManagedConfig(start_from=saved),
              realize=lambda _: pytest.fail("realized invalid call"))
    assert control.inspect(identity(control, live).operation_id) == old
    assert (Value.restores, Value.captures, Value.bodies) == (0, 0, 0)


def test_completed_replay_is_detached_and_never_opens_receiver(tmp_path, monkeypatch):
    control, live, saved = setup(tmp_path)
    old = seed(control, live, saved, state="completed")
    def forbidden(*args, **kwargs):
        pytest.fail("replay performed payload or realization work")
    monkeypatch.setattr(control.state_repo, "prepare_restore", forbidden)
    monkeypatch.setattr(control.control_store, "read_states", forbidden)
    first = enter(control, live, realize=forbidden, execute=forbidden)
    second = enter(control, live, realize=forbidden, execute=forbidden)
    assert first.disposition == second.disposition == "replay"
    assert first.run_id == second.run_id == old.run_id
    assert first.invocation_id != second.invocation_id != old.invocation_id
    assert first.current.producer_invocation_id == old.invocation_id


@pytest.mark.parametrize("phase", [None, "completed", "failed"])
def test_live_start_does_not_fabricate_recovery(tmp_path, phase):
    control, live, saved = setup(tmp_path)
    old = None if phase is None else seed(control, live, saved, state=phase)
    live.value = 93
    current, value, resuming = enter(control, live, config=ManagedConfig(rerun=True))
    assert value == 93 and not resuming and current.checkpoint is None
    assert old is None or current.run_id != old.run_id
    assert (Value.restores, Value.captures, Value.bodies) == (0, 0, 0)


def test_saved_start_seeds_recovery_without_capture(tmp_path):
    control, live, saved = setup(tmp_path)
    old = seed(control, live, saved, state="completed")
    live.value = 99
    current, value, resuming = enter(control, live,
        config=ManagedConfig(rerun=True, start_from=saved))
    assert value == 1 and not resuming and current.checkpoint.graph_equal(saved)
    assert current.run_id != old.run_id
    assert (Value.restores, Value.captures, Value.bodies) == (1, 0, 0)
    assert any(intent.kind == "start" for intent in current.intents)


@pytest.mark.parametrize("descriptor", [Value.run, Value.fixed])
def test_unfinished_without_latest_requires_explicit_rerun(tmp_path, descriptor):
    control, live, saved = setup(tmp_path)
    old = seed(control, live, saved, checkpoint=False, descriptor=descriptor)
    with pytest.raises(ManagedRerunRequiredError):
        enter(control, live, descriptor=descriptor)
    assert control.inspect(old.operation_id) == old
    assert Value.restores == 0


@pytest.mark.parametrize("phase", [None, "completed", "failed"])
def test_ordinary_without_start_selects_first_replay_or_resume(tmp_path, phase):
    control, live, saved = setup(tmp_path)
    old = None if phase is None else seed(control, live, saved, state=phase)
    live.value = 81
    result = enter(control, live)
    if phase == "completed":
        assert result.disposition == "replay" and Value.restores == 0
    else:
        current, value, resuming = result
        assert resuming == (phase == "failed")
        assert value == (1 if resuming else 81)
        assert current.checkpoint is None if phase is None else current.checkpoint.graph_equal(saved)
    assert (Value.captures, Value.bodies) == (0, 0)


def test_direct_call_normalizes_arguments_before_any_control_or_restore(tmp_path):
    control, live, saved = setup(tmp_path)
    config = ManagedConfig(state_repo=control.state_repo, control_store=control.control_store)
    with pytest.raises(ManagedConfigError):
        live.run(object(), managed=config)
    assert not os.path.exists(control.root)
    assert (Value.restores, Value.captures, Value.bodies) == (0, 0, 0)


def test_public_call_now_enters_authored_body_and_preserves_failure(tmp_path):
    """U16 replaces U15's pre-body stop; this admission fixture raises in its body."""
    control, live, saved = setup(tmp_path)
    with pytest.raises(AssertionError, match="U15 must not execute an authored body"):
        live.run(managed=ManagedConfig(state_repo=control.state_repo))
    current = control.inspect(identity(control, live).operation_id)
    assert current.state == "failed" and current.checkpoint is None
    assert (Value.restores, Value.captures, Value.bodies) == (0, 0, 1)


def test_reclaimed_receiver_checkpoint_does_not_block_detached_replay(tmp_path):
    from dryml.managed.identity import AdmissionIdentity
    from dryml.managed.runtime import admit_execution
    from dryml.core.store.retention import grants
    control, live, saved = setup(tmp_path)
    old = seed(control, live, saved, state="completed")
    # Remove only the explicitly identified user-save grant, then reclaim payload.
    for grant in grants(control.control_store, saved.state.key):
        control.control_store.retention.release(grant.token, (control.control_store,))
    assert control.control_store.delete_state(saved.state.key).status == "removed"
    result = admit_execution(AdmissionIdentity(saved, old.member, old.argument_digest),
        Value.run, ManagedConfig().snapshot(), control,
        realize=lambda _: pytest.fail("replay reconstructed receiver"),
        execute=lambda *args: pytest.fail("replay executed"))
    assert result.disposition == "replay" and result.current.final.graph_equal(saved)


def test_realization_and_execution_keep_same_nonblocking_owner_and_unpinned_metadata(tmp_path):
    from dryml.managed.errors import ManagedConflictError
    control, live, saved = setup(tmp_path)
    facts = identity(control, live)
    owners = []
    def realize(node):
        assert node.state is None
        with pytest.raises(ManagedConflictError, match="already_running"):
            control.admission(facts)
        return live
    def execute(owner, current, obj, resuming):
        assert owner.active and owner.invocation_id == current.invocation_id
        with pytest.raises(ManagedConflictError, match="already_running"):
            control.admission(facts)
        owners.append(owner)
        return current
    enter(control, live, config=ManagedConfig(rerun=True, start_from=saved), realize=realize, execute=execute)
    assert not owners[0].active and Value.restores == 1


def test_signature_defaults_and_explicit_values_have_same_compatibility(tmp_path):
    from dryml.managed.identity import AdmissionIdentity, _argument_digest_bound
    control, live, saved = setup(tmp_path)
    descriptor = Value.run
    def digest(args, kwargs):
        plan = descriptor.signature_plan().prepare_bound(
            descriptor.bind_arguments(live, args, kwargs), repo=control.state_repo)
        return _argument_digest_bound(tuple(plan.authority.items()))
    assert digest((), {}) == digest((2,), {}) == digest((), {"amount": 2})
    assert digest((3,), {}) != digest((), {})


def test_sqlite_selected_start_runs_hooks_and_preparation_outside_fences(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from dryml.core.store.retention import _writers
    from dryml.core.node_prepare import NodePreparation
    control, live, saved = setup(tmp_path, index="sqlite")
    held = []
    original_lock = control._control_lock
    original_hook = Value.restore_state_from_dir_imp
    original_prepare = NodePreparation.__init__
    @contextmanager
    def short():
        with original_lock():
            held.append("control")
            try:
                yield
            finally:
                held.pop()
    def hook(self, *args, **kwargs):
        assert not held and not _writers()
        return original_hook(self, *args, **kwargs)
    def prepare(self, *args, **kwargs):
        assert not held and not _writers()
        return original_prepare(self, *args, **kwargs)
    monkeypatch.setattr(control, "_control_lock", short)
    monkeypatch.setattr(Value, "restore_state_from_dir_imp", hook)
    monkeypatch.setattr(NodePreparation, "__init__", prepare)
    new, value, resuming = enter(control, live, config=ManagedConfig(rerun=True, start_from=saved))
    assert new.checkpoint.graph_equal(saved) and Value.restores == 1
    assert not control.control_store.pending_mutations()
    from dryml.core.query.authority import capture_store_projection
    rebuilt = capture_store_projection(control.control_store, rebuild=True)
    assert control.control_store._relative(control._preparation_path(new.operation_id)) in rebuilt.sources


def test_old_nested_root_intent_and_start_versions_preserve_source_bytes(tmp_path):
    import json
    from pathlib import Path
    from dryml.managed.control import ControlSnapshot
    from dryml.managed.errors import ManagedControlError
    control, live, saved = setup(tmp_path)
    old = seed(control, live, saved, initial=saved)
    data = old.to_data()
    data["intents"][0]["version"] = 1
    payload = json.dumps(data).encode()
    with pytest.raises(ManagedControlError):
        ControlSnapshot.from_bytes(payload)
    assert control.inspect(old.operation_id) == old
    path = Path(control._preparation_path(old.operation_id))
    path.parent.mkdir(parents=True)
    payload = b'{"schema":"dryml-managed-start","version":0}'
    path.write_bytes(payload)
    with pytest.raises(ManagedControlError):
        enter(control, live)
    assert path.read_bytes() == payload and control.inspect(old.operation_id) == old
    assert (Value.restores, Value.captures, Value.bodies) == (0, 0, 0)


class UnorderedRun(Object):
    def __init__(self, children):
        self.children = children

    @managed_operation(resumable=True)
    def run(self, *, managed):
        raise AssertionError("U15 must not enter the authored body")


def unordered_run(tmp_path):
    from dryml.core import Repo
    from dryml.core.store.dir import DirStore
    from dryml.managed.control import ManagedControlStore
    from dryml.managed.identity import AdmissionIdentity

    store = DirStore(tmp_path / "unordered", query_index="none")
    repo = Repo(store)
    live = UnorderedRun({Value(value=index, repo=repo) for index in range(5)}, repo=repo)
    saved = repo.save(live)
    control = ManagedControlStore(store, repo)
    facts = AdmissionIdentity.from_call(repo.describe(live), UnorderedRun.run, (), {})
    Value.restores = Value.captures = Value.bodies = 0
    return control, live, saved, facts


def test_unordered_saved_rerun_then_resume_keeps_root_evidence(tmp_path, monkeypatch):
    from dryml.managed import runtime
    from dryml.managed import ResultReference
    from dryml.managed.control import ControlIntent, ControlSnapshot
    from tests.managed.test_control_codec import force_fresh_handle_order

    flip = force_fresh_handle_order(monkeypatch)
    control, live, saved, facts = unordered_run(tmp_path)
    with control.admission(facts) as owner:
        initial = ControlIntent(uuid4().hex, owner.run_id, owner.invocation_id, 1,
                                "adopted", kind="initial", selection=saved)
        old = control.create_initial(ControlSnapshot(facts.receiver, facts.member, facts.argument_digest,
            owner.run_id, owner.invocation_id, 1, "completed", final=saved,
            producer_invocation_id=owner.invocation_id, intents=(initial,),
            result_reference=ResultReference("f" * 64)), owner=owner)
    def fail(owner, current, obj, resuming):
        assert not resuming and current.initial.graph_equal(saved)
        raise ValueError("execution seam")
    flip()
    with pytest.raises(ValueError, match="execution seam"):
        runtime.admit_execution(facts, UnorderedRun.run, ManagedConfig(rerun=True).snapshot(),
            control, realize=lambda _: live, execute=fail)
    failed = control.inspect(facts.operation_id)
    assert failed.state == "failed" and failed.run_id != old.run_id
    assert failed.checkpoint.graph_equal(saved) and failed.history[0].final.graph_equal(saved)
    flip()
    current = runtime.admit_execution(facts, UnorderedRun.run, ManagedConfig().snapshot(), control,
        realize=lambda _: live, execute=lambda owner, current, obj, resuming: (current, resuming))
    assert current[1] and current[0].run_id == failed.run_id
    assert current[0].initial.graph_equal(saved) and current[0].checkpoint.graph_equal(saved)
    assert (Value.restores, Value.captures, Value.bodies) == (10, 0, 0)


@pytest.mark.parametrize("admitted", [False, True])
def test_u15_unordered_restart_settles_exact_preparation_without_hooks(tmp_path, monkeypatch, admitted):
    from dryml.managed import runtime
    from dryml.managed.control import ManagedControlStore
    from tests.managed.test_control_codec import force_fresh_handle_order

    flip = force_fresh_handle_order(monkeypatch)
    control, live, saved, facts = unordered_run(tmp_path)
    with control.admission(facts) as owner:
        prepared = runtime._prepare_run(control, owner, None, None, False)
        ready = control.write_preparation(replace(prepared, phase="ready"), owner=owner, expected=prepared)
        if admitted:
            control.admit_prepared(ready, owner=owner)
    flip()
    restarted = ManagedControlStore(control.control_store, control.state_repo)
    def forbidden(*args):
        pytest.fail("restart reconciliation executed or realized receiver")
    outcome = runtime.admit_execution(facts, UnorderedRun.run, ManagedConfig().snapshot(),
        restarted, realize=forbidden, execute=forbidden)
    assert outcome.disposition == ("run_admitted" if admitted else "preparation_abandoned")
    settled = restarted.read_preparation(facts.operation_id)
    assert settled.settled and settled.phase == ("admitted" if admitted else "abandoned")
    assert (outcome.current is not None) == admitted
    assert (Value.restores, Value.captures, Value.bodies) == (0, 0, 0)
