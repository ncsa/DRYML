"""Explicit checkpoint callback ordering and caught-error policy (U16)."""

import pytest

from dryml.core.store.retention import _writers
from dryml.managed import ManagedConfig, ManagedContextError, ManagedInterrupted
from tests.managed.u16_fixtures import Value, enter, setup


def test_callbacks_receive_committed_latest_in_order_outside_state_reservations(tmp_path):
    control, live = setup(tmp_path, index="sqlite")
    observed = []
    def first(obj, ctx):
        current = control.inspect(ctx.operation_id)
        assert current.checkpoint.graph_equal(ctx.latest_checkpoint)
        assert not _writers() and not control.control_store.pending_mutations()
        assert control.state_repo._topology_leases == 0
        observed.append(("first", obj is live, obj.value, ctx.latest_checkpoint))
    def second(obj, ctx):
        observed.append(("second", obj is live, obj.value, ctx.latest_checkpoint))
    def body(obj, ctx):
        obj.value = 4
        checkpoint = ctx.checkpoint()
        obj.value = 5
        ctx.final_state(checkpoint)
        return checkpoint
    Value.action = body
    raw, final = enter(control, live, config=ManagedConfig(callbacks=[first, second]))
    assert [v[:3] for v in observed] == [("first", True, 4), ("second", True, 4)]
    assert all(v[3].graph_equal(raw) for v in observed) and final is raw


@pytest.mark.parametrize("caught", [False, True])
def test_callback_error_preserves_C_and_caught_error_permits_finalization(tmp_path, caught):
    control, live = setup(tmp_path)
    calls = []
    error = RuntimeError("observer failed")
    def fail(obj, ctx):
        calls.append(ctx.latest_checkpoint)
        obj.value = 99
        raise error
    def body(obj, ctx):
        obj.value = 4
        if not caught:
            ctx.checkpoint()
        else:
            with pytest.raises(RuntimeError) as raised:
                ctx.checkpoint()
            assert raised.value is error
            ctx.final_state(ctx.latest_checkpoint)
            return "handled"
    Value.action = body
    config = ManagedConfig(callbacks=[fail, lambda *args: pytest.fail("later callback")])
    if caught:
        raw, final = enter(control, live, config=config)
        assert raw == "handled" and final.graph_equal(calls[0])
    else:
        with pytest.raises(RuntimeError) as raised:
            enter(control, live, config=config)
        assert raised.value is error
        status = live.run.status(state_repo=control.state_repo)
        assert (status.state, status.failure_code) == ("failed", "callback_error")
        assert status.checkpoint.graph_equal(calls[0]) and status.final is None
    assert len(calls) == 1 and Value.captures == 1 and live.value == 99
    assert control.state_repo.load(calls[0], cache="none").value == 4


def test_recursive_checkpoint_is_rejected_but_association_survives(tmp_path):
    control, live = setup(tmp_path)
    Value.action = lambda obj, ctx: ctx.checkpoint()
    with pytest.raises(ManagedContextError, match="recursive_checkpoint"):
        enter(control, live, config=ManagedConfig(callbacks=[lambda obj, ctx: ctx.checkpoint()]))
    assert live.run.status(state_repo=control.state_repo).checkpoint is not None


def test_callback_can_run_distinct_operation_sharing_same_live_object(tmp_path):
    control, live = setup(tmp_path)
    from dryml.managed import managed_operation
    class Shared(Value):
        @managed_operation()
        def other(self, *, managed):
            managed.final_state(managed.checkpoint())
    live = Shared(repo=control.state_repo)
    Shared.action = lambda obj, ctx: ctx.checkpoint()
    calls = []
    def callback(obj, ctx):
        # Distinct logical operations may share a read-only live graph. Their
        # state IO reservations do not last through the callback or body.
        enter(control, obj, descriptor=Shared.other)
        calls.append(True)
    enter(control, live, config=ManagedConfig(callbacks=[callback]))
    assert calls == [True]


def test_request_during_callbacks_is_honored_at_same_checkpoint(tmp_path):
    control, live = setup(tmp_path)
    Value.action = lambda obj, ctx: ctx.checkpoint()
    outcomes = []
    def callback(obj, ctx):
        outcomes.append(control.request_interrupt_if_running(ctx.operation_id)[0])
    with pytest.raises(ManagedInterrupted):
        enter(control, live, config=ManagedConfig(callbacks=[callback]))
    status = live.run.status(state_repo=control.state_repo)
    assert outcomes == ["requested"] and status.state == "interrupted" and status.checkpoint is not None


def test_caught_confirmed_checkpoint_association_error_can_select_prior_final(tmp_path, monkeypatch):
    control, live = setup(tmp_path)
    original = control.admit_prepared
    previous = []
    def body(obj, ctx):
        first = ctx.checkpoint()
        previous.append(first)
        def fail(*args, **kwargs):
            raise OSError("association refused before CAS")
        monkeypatch.setattr(control, "admit_prepared", fail)
        obj.value = 7
        with pytest.raises(OSError):
            ctx.checkpoint()
        monkeypatch.setattr(control, "admit_prepared", original)
        ctx.final_state(first)
        return "handled"
    Value.action = body
    raw, final = enter(control, live)
    assert raw == "handled" and final is previous[0]
    current = control.inspect(live.run.status(state_repo=control.state_repo).operation_id)
    assert current.checkpoint.graph_equal(final) and Value.captures == 2
