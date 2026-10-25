"""Public retained-root cleanup, historical replay and federated failure proofs."""

from pathlib import Path

import pytest

from dryml.core import Definition, Object, Repo, SaveRouting, Selector
from dryml.core.store.dir import DirStore
from dryml.managed import ManagedConfig, ManagedConflictError
from tests.managed.u16_fixtures import Value, setup
from dryml.managed import managed_operation
from dryml.managed.control import ManagedControlStore


class CleanupGroup(Object):
    """Route independent private children through a real managed checkpoint."""

    action = None

    def __init__(self, children):
        self.children = children

    @managed_operation(resumable=True)
    def run(self, *, managed):
        return type(self).action(self, managed)


def routed(tmp_path):
    stores = [DirStore(tmp_path / name, query_index="none") for name in ("a", "b", "c")]
    repo = Repo(stores, save_routing=SaveRouting((
        (Selector(CleanupGroup), stores[0]),
        *((Selector(Definition(Value, value=i)), store) for i, store in enumerate(stores)),
    ), graph_mode="per-object"))
    live = CleanupGroup([Value(value=i, repo=repo) for i in range(3)], repo=repo)
    return ManagedControlStore(stores[0], repo), live, stores


def test_public_retain_release_does_not_capture_and_preserves_latest(tmp_path):
    control, live = setup(tmp_path)
    seen = []

    def body(obj, ctx):
        first = ctx.checkpoint()
        ctx.retain(first)
        before = Value.captures
        obj.value = 2
        latest = ctx.checkpoint()
        assert Value.captures == before + 1
        assert control.control_store.delete_state(first.state.key).status == "retained"
        outcomes = ctx.release(first)
        assert [item.status for item in outcomes] == ["removed"]
        assert Value.captures == before + 1
        assert control.control_store.delete_state(latest.state.key).status == "retained"
        seen.append(first)
        ctx.final_state(latest)
        return first

    Value.action = body
    result = live.run(managed=ManagedConfig(state_repo=control.state_repo))
    assert result.value.graph_equal(seen[0])
    inventory = control.state_repo.inspect_states(seen[0].state.key.cdef_digest)
    assert not next(item.descriptor.available for item in inventory if item.descriptor.state.key == seen[0].state.key)
    assert Value.captures == 2


def test_completed_cleanup_keeps_result_reference_status_and_replay(tmp_path):
    control, live = setup(tmp_path, index="sqlite")
    Value.action = lambda obj, ctx: {"value": obj.value}
    config = ManagedConfig(state_repo=control.state_repo)
    result = live.run(managed=config)
    operation = result.execution.operation_id
    roots = control.list_checkpoints(operation)
    assert [root.kind for root in roots] == ["final"]
    outcomes = control.cleanup_checkpoints(operation, tokens=[root.token for root in roots])
    assert [item.status for item in outcomes] == ["removed"]
    assert control.list_checkpoints(operation) == ()
    assert control.status(operation).state == "completed"
    assert control.status(operation).final.graph_equal(result.final_state_ref)
    with pytest.raises(Exception, match="unavailable|missing"):
        control.state_repo.load(result.final_state_ref, cache="none")
    counts = Value.bodies, Value.captures, Value.restores
    assert live.run(managed=config).value == {"value": 0}
    assert (Value.bodies, Value.captures, Value.restores) == counts
    index = control.control_store.open_query_index()
    index.close()
    index.path.unlink()
    assert control.control_store.open_query_index().projection_rows("lifecycle", refresh=True)
    assert result.value == {"value": 0}


def test_cleanup_reports_same_operation_active_use_and_never_queues(tmp_path):
    control, live = setup(tmp_path)

    def body(obj, ctx):
        first = ctx.checkpoint(durable=True)
        obj.value = 1
        latest = ctx.checkpoint()
        with control.control_store.read_state(first.state.key):
            assert [item.status for item in ctx.release(first)] == ["in-use"]
        assert control.state_repo.load(first, cache="none").value == 0
        assert [item.status for item in ctx.release(first)] == ["removed"]
        ctx.final_state(latest)

    Value.action = body
    assert live.run(managed=ManagedConfig(state_repo=control.state_repo)).status.state == "completed"


def test_external_cleanup_refuses_running_owner_without_waiting(tmp_path):
    control, live = setup(tmp_path)

    def body(obj, ctx):
        saved = ctx.checkpoint(durable=True)
        with pytest.raises(ManagedConflictError, match="already_running"):
            control.cleanup_checkpoints(ctx.operation_id)
        assert control.control_store.delete_state(saved.state.key).status == "retained"
        ctx.final_state(saved)

    Value.action = body
    live.run(managed=ManagedConfig(state_repo=control.state_repo))


def test_garbage_failure_is_successful_with_exact_retry_evidence(tmp_path, monkeypatch):
    import dryml.core.store.retention as retention
    control, live = setup(tmp_path, index="sqlite")
    old = []

    def body(obj, ctx):
        old.append(ctx.checkpoint())
        obj.value = 1
        ctx.final_state(ctx.checkpoint())
        return 7

    Value.action = body
    original = retention.remove_private_data
    def fail_old(path):
        if old and path == control.control_store.state_data_path(old[0].state.key):
            raise OSError("physical garbage denied")
        return original(path)
    with monkeypatch.context() as patch:
        patch.setattr(retention, "remove_private_data", fail_old)
        result = live.run(managed=ManagedConfig(state_repo=control.state_repo))
    assert result.value == 7 and result.status.state == "completed"
    domain, key, outcome = result.cleanup_report[0]
    assert key == old[0].state.key and outcome.status == "cleanup-pending"
    assert control.control_store.retry_state_removal(key, outcome.incarnation).status == "removed"


@pytest.mark.parametrize("phase", ["release", "withdraw"])
def test_authority_or_index_failure_never_becomes_successful_cleanup(tmp_path, monkeypatch, phase):
    control, live = setup(tmp_path, index="sqlite")
    old = []
    original = control.control_store._prepare_and_apply

    def body(obj, ctx):
        old.append(ctx.checkpoint())
        obj.value = 1
        ctx.checkpoint()

    def fail(prepare, **kwargs):
        def checked():
            result = prepare()
            paths = [change.path for change in result[0] if change is not None]
            if old and any(("/grants/" in path if phase == "release" else "/removals/" in path)
                           and old[0].state.key.codec_hash in path for path in paths):
                raise OSError("controlled " + phase + " failure")
            return result
        return original(checked, **kwargs)
    Value.action = body
    monkeypatch.setattr(control.control_store, "_prepare_and_apply", fail)
    with pytest.raises(OSError, match="controlled"):
        live.run(managed=ManagedConfig(state_repo=control.state_repo))
    assert live.run.status(state_repo=control.state_repo).state != "completed"
    assert Path(control.control_store.state_data_path(old[0].state.key)).is_dir()


def test_cleanup_unknown_token_is_rejected_before_releasing_any_root(tmp_path):
    from uuid import uuid4
    control, live = setup(tmp_path)
    Value.action = lambda obj, ctx: None
    result = live.run(managed=ManagedConfig(state_repo=control.state_repo))
    root, = control.list_checkpoints(result.execution.operation_id)
    with pytest.raises(ManagedConflictError, match="unknown"):
        control.cleanup_checkpoints(result.execution.operation_id, tokens=[root.token, uuid4().hex])
    assert control.control_store.delete_state(result.final_state_ref.state.key).status == "retained"


@pytest.mark.parametrize("phase", ["release", "removal"])
def test_public_checkpoint_federated_failure_keeps_A_and_continues_C(tmp_path, monkeypatch, phase):
    control, live, stores = routed(tmp_path)
    old = []

    def body(obj, ctx):
        saved = ctx.checkpoint()
        old.extend(saved.state_spec().values())
        for child in obj.children:
            child.value += 10
        if phase == "removal":
            original = stores[1].delete_state
            def fail(key):
                if key in {state.key for state in old}:
                    raise OSError("B removal failed")
                return original(key)
            monkeypatch.setattr(stores[1], "delete_state", fail)
        else:
            original = stores[1]._prepare_and_apply
            def fail(prepare, **kwargs):
                def checked():
                    result = prepare()
                    if any(change is not None and "/grants/" in change.path and change.after is None
                           and any(state.key.codec_hash in change.path for state in old) for change in result[0]):
                        raise OSError("B release failed")
                    return result
                return original(checked, **kwargs)
            monkeypatch.setattr(stores[1], "_prepare_and_apply", fail)
        ctx.checkpoint()

    CleanupGroup.action = body
    with pytest.raises(OSError, match="B .* failed") as caught:
        live.run(managed=ManagedConfig(state_repo=control.state_repo))
    outcomes = {item.domain: item for item in caught.value.cleanup_outcomes}
    assert [outcomes[store.retention_domain].status for store in stores] == ["removed", "failed", "removed"]
    assert outcomes[stores[1].retention_domain].phase == phase
    for index in (0, 2):
        key = outcomes[stores[index].retention_domain].key
        assert not Path(stores[index].state_data_path(key)).exists()
    assert live.run.status(state_repo=control.state_repo).state != "completed"


def test_disconnected_cleanup_is_unattempted_and_retry_reconnects_exact_targets(tmp_path):
    control, live, stores = routed(tmp_path)
    CleanupGroup.action = lambda obj, ctx: 9
    result = live.run(managed=ManagedConfig(state_repo=control.state_repo))
    operation = result.execution.operation_id
    reduced = ManagedControlStore(stores[0], Repo([stores[0], stores[2]]))
    with pytest.raises(Exception, match="pending") as caught:
        reduced.cleanup_checkpoints(operation)
    outcomes = {item.domain: item.status for item in caught.value.cleanup_outcomes}
    assert [outcomes[store.retention_domain] for store in stores] == ["removed", "unattempted", "removed"]
    assert control.list_checkpoints(operation)[0].disposition == "release_pending"
    assert control.cleanup_checkpoints(operation)
    assert control.list_checkpoints(operation) == ()
    assert result.value == 9 and control.status(operation).state == "completed"


def test_public_cleanup_preserves_foreign_reader_until_explicit_retry(tmp_path):
    from threading import Event, Thread
    control, live = setup(tmp_path)
    Value.action = lambda obj, ctx: None
    result = live.run(managed=ManagedConfig(state_repo=control.state_repo))
    key = result.final_state_ref.state.key
    ready, release = Event(), Event()
    errors = []

    def reader():
        try:
            store = DirStore(control.control_store.base_dir, query_index="none")
            with store.read_state(key) as source:
                ready.set()
                assert release.wait(10)
                assert Path(source.data_directory).is_dir()
            store.close()
        except BaseException as error:
            errors.append(error)

    worker = Thread(target=reader)
    worker.start()
    try:
        assert ready.wait(10)
        outcomes = control.cleanup_checkpoints(result.execution.operation_id)
        assert [item.status for item in outcomes] == ["in-use"]
        assert control.list_checkpoints(result.execution.operation_id) == ()
    finally:
        release.set()
        worker.join(10)
    assert not worker.is_alive() and not errors
    assert Path(control.control_store.state_data_path(key)).is_dir()
    assert [item.status for item in control.state_repo.delete_states([key])] == ["removed"]


def test_cleanup_association_projection_failure_preserves_grants_and_completed_history(tmp_path, monkeypatch):
    from dryml.core.store.authority import StoreMutationError
    from dryml.core.store.retention import grants
    control, live = setup(tmp_path, index="sqlite")
    Value.action = lambda obj, ctx: 42
    result = live.run(managed=ManagedConfig(state_repo=control.state_repo))
    operation = result.execution.operation_id
    key = result.final_state_ref.state.key
    index = control.control_store.open_query_index()
    original = index.maintain_store

    def fail_after_cas(*args, **kwargs):
        if control.list_checkpoints(operation)[0].disposition == "release_pending":
            raise OSError("cleanup association projection failed")
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(index, "maintain_store", fail_after_cas)
        with pytest.raises(StoreMutationError) as caught:
            control.cleanup_checkpoints(operation)
    assert [item.status for item in caught.value.cleanup_outcomes] == ["unattempted"]
    assert control.status(operation).state == "completed"
    assert grants(control.control_store, key)
    assert control.control_store.authority_cut().pending
    assert Path(control.control_store.state_data_path(key)).is_dir()
    assert [item.status for item in control.cleanup_checkpoints(operation)] == ["removed"]
    assert result.value == 42


def test_interrupted_public_cleanup_keeps_every_target_outcome_and_can_retry(tmp_path, monkeypatch):
    control, live, stores = routed(tmp_path)
    CleanupGroup.action = lambda obj, ctx: 5
    result = live.run(managed=ManagedConfig(state_repo=control.state_repo))

    def interrupt(key):
        raise KeyboardInterrupt("cleanup interrupted")
    with monkeypatch.context() as patch:
        patch.setattr(stores[1], "delete_state", interrupt)
        with pytest.raises(KeyboardInterrupt) as caught:
            control.cleanup_checkpoints(result.execution.operation_id)
    outcomes = caught.value.cleanup_outcomes
    assert len(outcomes) == 3
    assert {item.domain for item in outcomes} == {store.retention_domain for store in stores}
    assert next(item.status for item in outcomes if item.domain == stores[1].retention_domain) == "failed"
    assert control.status(result.execution.operation_id).state == "completed"
    assert all(item.status == "removed" for item in control.cleanup_checkpoints(result.execution.operation_id))
    assert result.value == 5


def test_cleanup_keeps_another_operations_equal_full_key_protected(tmp_path):
    control, first = setup(tmp_path)
    second = Value(repo=control.state_repo)
    Value.action = lambda obj, ctx: 11
    config = ManagedConfig(state_repo=control.state_repo)
    a = first.run(managed=config)
    def reuse(obj, ctx):
        ctx.final_state(ctx.checkpoint(states=a.final_state_ref.state_spec()))
        return 11
    Value.action = reuse
    b = second.run(managed=config)
    assert a.execution.operation_id != b.execution.operation_id
    assert a.final_state_ref.state.key == b.final_state_ref.state.key
    assert [item.status for item in control.cleanup_checkpoints(a.execution.operation_id)] == ["retained"]
    assert control.state_repo.load(b.final_state_ref, cache="none").value == 0
    assert [item.status for item in control.cleanup_checkpoints(b.execution.operation_id)] == ["retained", "removed"]
    assert a.value == b.value == 11


def test_public_rerun_preserves_durable_and_explicit_roots_until_release(tmp_path):
    control, live = setup(tmp_path)
    old = []

    def first(obj, ctx):
        old.append(ctx.checkpoint(durable=True))
        ctx.retain(old[0])
        ctx.final_state(old[0])
    Value.action = first
    config = ManagedConfig(state_repo=control.state_repo)
    before = live.run(managed=config)

    def rerun(obj, ctx):
        assert control.control_store.delete_state(old[0].state.key).status == "retained"
        obj.value = 2
        ctx.final_state(ctx.checkpoint())
        assert [item.status for item in ctx.release(old[0])] == ["retained", "removed"]
    Value.action = rerun
    after = live.run(managed=ManagedConfig(state_repo=control.state_repo, rerun=True))
    assert before.execution.run_id != after.execution.run_id
    assert before.status.state == after.status.state == "completed"
