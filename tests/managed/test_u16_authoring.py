"""U16 proof-first authored-body and final-state handoff checks."""

import pytest

from dryml.managed import ManagedConfig, ManagedConflictError, ManagedControlError
from tests.managed.u16_fixtures import Value, enter, setup
from dryml.core import function
from dryml.managed import managed_operation


@function
def independent(amount: int):
    return amount + 1


class Wrapped(Value):
    @managed_operation()
    @function
    def run(self, amount: int = 2, *, managed):
        self.value = independent(amount)
        return self.value


def test_explicit_checkpoint_final_has_one_capture_one_callback_no_initial(tmp_path):
    control, live = setup(tmp_path)
    seen = []
    def body(obj, context):
        saved = context.checkpoint()
        assert context.initial_state_ref is None
        assert context.latest_checkpoint.graph_equal(saved)
        context.final_state(saved)
        obj.value = 99
        return saved
    Value.action = body
    raw, final = enter(control, live, config=ManagedConfig(callbacks=[lambda obj, ctx: seen.append(ctx.latest_checkpoint)]))
    assert final is raw and len(seen) == 1 and seen[0].graph_equal(raw)
    assert (Value.captures, Value.restores, Value.bodies, live.value) == (1, 0, 1, 99)


def test_automatic_final_has_no_checkpoint_callback(tmp_path):
    control, live = setup(tmp_path)
    Value.action = lambda obj, ctx: 42
    seen = []
    raw, final = enter(control, live, config=ManagedConfig(callbacks=[lambda *args: seen.append(args)]))
    assert raw == 42 and final.state is not None
    assert seen == [] and Value.captures == 1


def test_initial_is_explicit_exact_idempotent_and_immutable(tmp_path):
    control, live = setup(tmp_path)
    def body(obj, ctx):
        first = ctx.checkpoint(initial=True)
        ctx.initial_state(first)
        obj.value = 2
        other = ctx.checkpoint()
        with pytest.raises(ManagedConflictError, match="initial_conflict"):
            ctx.initial_state(other)
        assert ctx.initial_state_ref.graph_equal(first)
        ctx.final_state(first)
        return first
    Value.action = body
    raw, final = enter(control, live)
    assert raw is final and Value.captures == 2


def test_public_body_returns_durable_invocation_result(tmp_path):
    control, live = setup(tmp_path)
    def body(obj, ctx):
        saved = ctx.checkpoint()
        ctx.final_state(saved)
        return saved
    Value.action = body
    result = live.run(managed=ManagedConfig(state_repo=control.state_repo))
    status = live.run.status(state_repo=control.state_repo)
    assert status.state == "completed" and status.checkpoint is not None
    assert result.value.graph_equal(status.checkpoint)
    assert Value.bodies == Value.captures == 1


def test_direct_arguments_normalize_once_and_independent_nested_function_keeps_boundary(tmp_path, monkeypatch):
    from dryml.core.signatures import SignaturePlan
    control, live = setup(tmp_path, cls=Wrapped)
    calls = []
    bound, args = SignaturePlan.prepare_bound, SignaturePlan._prepare_args
    def prepare_bound(self, *a, **kw):
        calls.append(("managed", self))
        return bound(self, *a, **kw)
    def prepare_args(self, *a, **kw):
        calls.append(("nested", self))
        return args(self, *a, **kw)
    monkeypatch.setattr(SignaturePlan, "prepare_bound", prepare_bound)
    monkeypatch.setattr(SignaturePlan, "_prepare_args", prepare_args)
    result = live.run(managed=ManagedConfig(state_repo=control.state_repo))
    assert result.value == 3
    assert live.value == 3
    assert [kind for kind, plan in calls] == ["managed", "nested"]
    assert calls[0][1] is Wrapped.run.signature_plan() and calls[1][1] is not calls[0][1]
