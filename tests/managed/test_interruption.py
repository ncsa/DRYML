"""Independent interruption and completion-generation fence qualification."""

from threading import Event, Thread
from uuid import uuid4

import pytest

from dryml.managed import ManagedConfig, ManagedContextError, ManagedInterrupted
from dryml.managed import context as context_module
from tests.managed.u16_fixtures import Value, enter, setup


@pytest.mark.parametrize("where", ["body", "capture", "selected_final"])
def test_interrupt_is_responsive_under_lifetime_owner_and_cannot_be_swallowed(tmp_path, monkeypatch, where):
    control, live = setup(tmp_path)
    reached, release = Event(), Event()
    outcomes, errors = [], []
    original = Value.save_state_to_dir_imp
    def capture(obj, directory, *, codec):
        reached.set()
        assert release.wait(15)
        return original(obj, directory, codec=codec)
    if where == "capture":
        monkeypatch.setattr(Value, "save_state_to_dir_imp", capture)
    def body(obj, ctx):
        if where == "selected_final":
            ctx.final_state(ctx.checkpoint())
        if where != "capture":
            reached.set()
            assert release.wait(15)
        try:
            ctx.checkpoint()
        except Exception as error:
            errors.append(error)
        return "caught and returned"
    Value.action = body
    def run():
        try:
            enter(control, live)
        except BaseException as error:
            outcomes.append(error)
    worker = Thread(target=run)
    worker.start()
    try:
        assert reached.wait(15)
        status = live.run.status(state_repo=control.state_repo)
        assert status.state == "running"
        request = live.run.request_interrupt(state_repo=control.state_repo,
            expected_run_id=status.run_id, expected_invocation_id=status.invocation_id,
            expected_generation=status.generation)
        assert request.outcome == "requested"
    finally:
        release.set(); worker.join(20)
    assert not worker.is_alive() and len(outcomes) == 1 and isinstance(outcomes[0], ManagedInterrupted)
    assert live.run.status(state_repo=control.state_repo).state == "interrupted"


def test_explicit_interrupt_retains_checkpoint_and_exact_cause(tmp_path):
    control, live = setup(tmp_path)
    cause = ValueError("stop now")
    Value.action = lambda obj, ctx: ctx.interrupt(cause=cause)
    with pytest.raises(ManagedInterrupted) as raised:
        enter(control, live)
    status = live.run.status(state_repo=control.state_repo)
    assert raised.value.__cause__ is cause and status.state == "interrupted" and status.checkpoint is not None


def test_invalid_explicit_interrupt_cause_fails_before_checkpoint_mutation(tmp_path):
    control, live = setup(tmp_path)
    Value.action = lambda obj, ctx: ctx.interrupt(cause="invalid")
    with pytest.raises(ManagedContextError, match="invalid_cause"):
        enter(control, live)
    assert Value.captures == 0 and live.run.status(state_repo=control.state_repo).checkpoint is None


@pytest.mark.parametrize("before", [True, False])
def test_interrupt_immediately_before_or_after_completion_CAS(tmp_path, monkeypatch, before):
    control, live = setup(tmp_path)
    Value.action = lambda obj, ctx: ctx.final_state(ctx.checkpoint())
    requests = []
    from dryml.managed import runtime
    def boundary(stage):
        if stage == ("before_completion" if before else "completion_recorded"):
            operation_id = live.run.status(state_repo=control.state_repo).operation_id
            requests.append(control.request_interrupt_if_running(operation_id)[0])
    monkeypatch.setattr(runtime, "_completion_boundary", boundary)
    if before:
        with pytest.raises(ManagedInterrupted):
            live.run(managed=ManagedConfig(state_repo=control.state_repo))
        assert requests == ["requested"]
    else:
        assert live.run(managed=ManagedConfig(state_repo=control.state_repo)).value is None
        assert requests == ["not_running"]


@pytest.mark.parametrize("field,outcome", [("expected_run_id", "stale_run"),
    ("expected_invocation_id", "stale_invocation"), ("expected_generation", "stale_generation")])
def test_targeted_stale_request_has_no_effect(tmp_path, field, outcome):
    control, live = setup(tmp_path)
    def body(obj, ctx):
        before = control.inspect(ctx.operation_id).to_bytes()
        expected = 999 if field == "expected_generation" else uuid4().hex
        assert control.request_interrupt_if_running(ctx.operation_id, **{field: expected})[0] == outcome
        assert control.inspect(ctx.operation_id).to_bytes() == before
        ctx.final_state(ctx.checkpoint())
    Value.action = body
    enter(control, live)


def test_caught_explicit_interrupt_rejects_later_mutations_and_return(tmp_path):
    control, live = setup(tmp_path)
    def body(obj, ctx):
        try:
            ctx.interrupt()
        except Exception:
            for call in (ctx.checkpoint, lambda: ctx.initial_state(ctx.latest_checkpoint),
                         lambda: ctx.final_state(ctx.latest_checkpoint)):
                with pytest.raises(ManagedContextError, match="inactive_context"):
                    call()
        return "success is forbidden"
    Value.action = body
    with pytest.raises(ManagedInterrupted):
        enter(control, live)


def test_request_after_callback_decision_is_caught_at_return(tmp_path, monkeypatch):
    control, live = setup(tmp_path)
    def boundary(stage):
        if stage == "callbacks_finished":
            current = live.run.status(state_repo=control.state_repo)
            assert control.request_interrupt_if_running(current.operation_id)[0] == "requested"
    monkeypatch.setattr(context_module, "_checkpoint_boundary", boundary)
    Value.action = lambda obj, ctx: ctx.checkpoint()
    with pytest.raises(ManagedInterrupted):
        enter(control, live)


def test_stale_context_generation_cannot_capture_select_or_release(tmp_path):
    from dataclasses import replace
    from dryml.managed import ManagedConflictError
    control, live = setup(tmp_path)
    def body(obj, ctx):
        saved = ctx.checkpoint()
        old = ctx._snapshot
        control.transition(ctx.operation_id, replace(old, generation=old.generation + 1),
            owner=ctx._admission, expected_generation=old.generation,
            expected_run_id=ctx.run_id, expected_invocation_id=ctx.invocation_id)
        captures = Value.captures
        for call in (ctx.checkpoint, lambda: ctx.initial_state(saved), lambda: ctx.final_state(saved),
            lambda: control.authorize_release(ctx.operation_id, old.intents[0].token,
                owner=ctx._admission, run_id=ctx.run_id, invocation_id=ctx.invocation_id,
                expected_generation=old.generation)):
            with pytest.raises(ManagedConflictError):
                call()
        assert Value.captures == captures
        raise ValueError("end stale execution")
    Value.action = body
    with pytest.raises(ValueError):
        enter(control, live)


@pytest.mark.parametrize("with_checkpoint", [False, True])
def test_keyboard_interrupt_preserves_latest_without_extra_capture(tmp_path, with_checkpoint):
    control, live = setup(tmp_path)
    def body(obj, ctx):
        obj.value = 1
        if with_checkpoint:
            ctx.checkpoint()
        obj.value = 2
        raise KeyboardInterrupt()
    Value.action = body
    with pytest.raises(ManagedInterrupted) as raised:
        enter(control, live)
    status = live.run.status(state_repo=control.state_repo)
    assert isinstance(raised.value.__cause__, KeyboardInterrupt)
    assert status.state == "interrupted" and (status.checkpoint is not None) == with_checkpoint
    assert Value.captures == int(with_checkpoint) and live.value == 2
    if not with_checkpoint:
        from dryml.managed import ManagedRerunRequiredError
        with pytest.raises(ManagedRerunRequiredError, match="rerun_required"):
            enter(control, live)


def test_disjoint_nested_interruption_fails_only_the_enclosing_method_as_ordinary_error(tmp_path):
    from dryml.managed import managed_operation
    class Nested(Value):
        @managed_operation()
        def stop(self, *, managed):
            managed.interrupt()
    control, live = setup(tmp_path)
    nested = Nested(repo=control.state_repo)
    def body(obj, ctx):
        nested.stop(managed=ManagedConfig(state_repo=ctx.state_repo))
    Value.action = body
    with pytest.raises(ManagedInterrupted):
        enter(control, live)
    outer = live.run.status(state_repo=control.state_repo)
    inner = nested.stop.status(state_repo=control.state_repo)
    assert (outer.state, outer.failure_code, outer.checkpoint) == ("failed", "method_error", None)
    assert inner.state == "interrupted" and inner.checkpoint is not None
