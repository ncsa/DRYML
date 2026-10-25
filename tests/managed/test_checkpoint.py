"""Current SavedNode checkpoint/selection/protection qualification (U16)."""

import os
from threading import Thread

import pytest

from dryml.core import Definition
from dryml.core.object import Object
from dryml.core.node import require_saved
from dryml.core.node_codec import encode_node_graph, decode_node_graph
from dryml.core.node_prepare import NodePreparation
from dryml.core.store.retention import grants
from dryml.managed import ManagedConfig, ManagedConflictError, ManagedContextError
from dryml.managed.context import _selection_equal
from tests.managed.u16_fixtures import Value, enter, setup


def test_checkpoint_saves_exact_intermediate_state_and_associates_it(tmp_path):
    control, live = setup(tmp_path)
    def body(obj, ctx):
        obj.value = 1
        first = ctx.checkpoint(durable=True)
        obj.value = 2
        ctx.final_state(first)
        return first
    Value.action = body
    first, final = enter(control, live)
    assert first is final and live.value == 2 and Value.captures == 1
    assert control.state_repo.load(first, cache="none").value == 1


def test_partial_explicit_state_reuse_captures_remaining_nodes_once(tmp_path):
    from dryml.core import Repo
    from dryml.core.store.dir import DirStore
    class Pair(Value):
        def __init__(self, left, right):
            self.left, self.right = left, right
    store = DirStore(tmp_path / "state", query_index="none")
    repo = Repo(store)
    child = Value(repo=repo)
    live = Pair(child, child, repo=repo)
    from dryml.managed.control import ManagedControlStore
    control = ManagedControlStore(store, repo)
    saved = repo.save(live)
    states = {path: state for path, state in saved.state_spec().items() if path.segments}
    assert len(states) == 1
    Pair.captures = Value.captures = 0
    def body(obj, ctx):
        obj.left.value = 7
        result = ctx.checkpoint(states=states)
        ctx.final_state(result)
        return result
    Pair.action = body
    result, final = enter(control, live)
    assert final is result and Pair.captures == 1 and Value.captures == 0
    assert result.state_spec().values() and live.left is live.right


def test_initial_may_first_be_designated_during_resume(tmp_path):
    control, live = setup(tmp_path)
    seen = []
    def body(obj, ctx):
        seen.append((ctx.run_id, ctx.invocation_id, ctx.is_resuming))
        if not ctx.is_resuming:
            obj.value = 3
            ctx.checkpoint()
            raise ValueError("resume me")
        assert ctx.initial_state_ref is None and obj.value == 3
        ctx.initial_state(ctx.latest_checkpoint)
        ctx.initial_state(ctx.initial_state_ref)
        ctx.final_state(ctx.initial_state_ref)
    Value.action = body
    with pytest.raises(ValueError, match="resume me"):
        enter(control, live)
    live.value = 9
    enter(control, live)
    assert seen[0][0] == seen[1][0] and seen[0][1] != seen[1][1]
    assert seen[1][2] and Value.restores == 1 and Value.captures == 1


def test_initial_durable_latest_and_user_grants_are_independent(tmp_path):
    control, live = setup(tmp_path)
    user = control.state_repo.save(live)
    user_tokens = {g.token for g in grants(control.control_store, user.state.key)}
    evidence = []
    def body(obj, ctx):
        first = ctx.checkpoint(initial=True, durable=True)
        before = ctx._snapshot
        obj.value = 5
        ctx.checkpoint()
        after = ctx._snapshot
        evidence.extend((first, before, after))
        ctx.final_state(first)
    Value.action = body
    enter(control, live)
    first, before, after = evidence
    assert {v.kind for v in before.intents} == {"latest", "initial", "durable"}
    assert len({v.token for v in before.intents}) == 3
    assert user_tokens <= {g.token for g in grants(control.control_store, user.state.key)}
    assert next(v for v in before.intents if v.kind == "latest").token not in {v.token for v in after.intents}
    assert {v.token for v in before.intents if v.kind != "latest"} <= {v.token for v in after.intents}
    assert control.control_store.retention.read(next(g.token for g in grants(control.control_store, first.state.key))).disposition == "adopted"


@pytest.mark.parametrize("selection", ["initial", "final"])
def test_wrong_original_projection_is_rejected_before_hooks(tmp_path, monkeypatch, selection):
    control, live = setup(tmp_path)
    foreign = control.state_repo.save(Value(repo=control.state_repo))
    Value.captures = Value.restores = 0
    monkeypatch.setattr(control.control_store, "read_states", lambda *args, **kwargs:
        pytest.fail("wrong projection reached a state reader"))
    def body(obj, ctx):
        with pytest.raises(ManagedConflictError, match="incompatible_selection"):
            getattr(ctx, selection + "_state")(foreign)
        assert ctx.initial_state_ref is None and ctx.final_state_ref is None
        raise ValueError("stop")
    Value.action = body
    with pytest.raises(ValueError):
        enter(control, live)
    assert Value.captures == Value.restores == 0


@pytest.mark.parametrize("reason", ["initial", "final"])
@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_selection_reader_cancellation_preserves_exact_exception_before_hooks(tmp_path, monkeypatch, reason, error_type):
    control, live = setup(tmp_path)
    saved = control.state_repo.save(live)
    Value.captures = Value.restores = 0
    original = control.control_store.read_states
    error = error_type("cancelled selection reader")
    def body(obj, ctx):
        def refuse(*args, **kwargs):
            raise error
        monkeypatch.setattr(control.control_store, "read_states", refuse)
        with pytest.raises(error_type) as raised:
            getattr(ctx, reason + "_state")(saved)
        assert raised.value is error
        assert Value.captures == Value.restores == 0
        monkeypatch.setattr(control.control_store, "read_states", original)
        ctx.final_state(saved)
    Value.action = body
    enter(control, live)


def test_context_rejects_foreign_and_retired_use_without_retiring_owner(tmp_path):
    control, live = setup(tmp_path)
    captured = []
    def body(obj, ctx):
        import pickle
        with pytest.raises(TypeError, match="cannot be serialized"):
            pickle.dumps(ctx)
        captured.append(ctx)
        errors = []
        def foreign():
            for call in (ctx.checkpoint, lambda: ctx.initial_state(None), lambda: ctx.final_state(None)):
                with pytest.raises(ManagedContextError, match="inactive_context"):
                    call()
            errors.append(True)
        worker = Thread(target=foreign)
        worker.start(); worker.join(10)
        assert not worker.is_alive() and errors == [True] and ctx.active
        ctx.final_state(ctx.checkpoint())
    Value.action = body
    enter(control, live)
    with pytest.raises(ManagedContextError, match="inactive_context"):
        captured[0].checkpoint()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_forked_child_cannot_use_parent_context(tmp_path):
    control, live = setup(tmp_path)
    def body(obj, ctx):
        child = os.fork()
        if child == 0:
            try:
                ctx.checkpoint()
            except ManagedContextError:
                os._exit(0)
            os._exit(1)
        assert os.waitstatus_to_exitcode(os.waitpid(child, 0)[1]) == 0
        ctx.final_state(ctx.checkpoint())
    Value.action = body
    enter(control, live)


def test_selection_equality_preserves_retained_construction_not_packet_order(tmp_path):
    control, live = setup(tmp_path)
    saved = control.state_repo.save(live)
    assert _selection_equal(saved, decode_node_graph(encode_node_graph(NodePreparation((saved,)))))
    # Retained construction is independently observable even if realization and
    # all exact Node facets are otherwise identical.
    changed = saved.copy_graph()
    from dryml.core.state_values import LocalState, StateKey
    pin = LocalState(StateKey(saved.state.key.cdef_digest, "pkl-" + "f" * 64), saved.state.creation_cdef)
    object.__setattr__(changed, "_construction", saved.construction_view.with_state(pin))
    require_saved(changed)
    assert saved.graph_equal(changed) and not _selection_equal(saved, changed)
    from dryml.managed.control import ControlIntent
    from dataclasses import replace
    from uuid import uuid4
    intent = ControlIntent(uuid4().hex, uuid4().hex, uuid4().hex, 1,
        "adopted", kind="initial", selection=saved)
    assert intent != replace(intent, selection=changed)


def test_empty_stateless_graph_is_a_valid_checkpoint_and_final(tmp_path):
    from dryml.managed import managed_operation
    class Stateless(Object):
        @managed_operation()
        def run(self, *, managed):
            saved = managed.checkpoint(initial=True)
            assert not saved.state_spec()
            managed.final_state(saved)
            return saved
    control, live = setup(tmp_path, cls=Stateless)
    raw, final = enter(control, live)
    assert require_saved(raw) is raw and final is raw


@pytest.mark.parametrize("reason", ["latest", "initial", "final"])
def test_fresh_reader_override_errors_are_preserved(tmp_path, monkeypatch, reason):
    control, live = setup(tmp_path)
    original = control.control_store.read_states
    failure = ValueError("fresh external reader refusal")
    def body(obj, ctx):
        saved = ctx.checkpoint()
        def refuse(*args, **kwargs):
            raise failure
        monkeypatch.setattr(control.control_store, "read_states", refuse)
        from dryml.core.repo import RepoLoadError
        with pytest.raises((ValueError, RepoLoadError)) as caught:
            if reason == "latest":
                ctx.checkpoint(states=saved.state_spec())
            else:
                getattr(ctx, reason + "_state")(saved)
        assert caught.value is failure or caught.value.__cause__ is failure
        monkeypatch.setattr(control.control_store, "read_states", original)
        ctx.final_state(saved)
    Value.action = body
    enter(control, live)


def test_selected_final_survives_later_nondurable_latest_replacement(tmp_path):
    control, live = setup(tmp_path)
    def body(obj, ctx):
        first = ctx.checkpoint()
        ctx.final_state(first)
        obj.value = 12
        ctx.checkpoint()
        return first
    Value.action = body
    first, final = enter(control, live)
    assert first is final and Value.captures == 2 and Value.restores == 0
    assert control.state_repo.load(final, cache="none").value == 0


def test_mixed_earlier_child_and_later_progress_final_preserved_without_IO(tmp_path):
    from dryml.core import Repo
    from dryml.core.store.dir import DirStore
    from dryml.managed.control import ManagedControlStore
    class Pair(Value):
        def __init__(self, left, right):
            self.left, self.right, self.value = left, right, 0
    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    live = Pair(Value(1, repo=repo), Value(2, repo=repo), repo=repo)
    earlier = repo.save(live)
    live.value, live.left.value, live.right.value = 5, 11, 12
    later = repo.save(live)
    from dryml.core import Parameter
    left = (Parameter("left"),)
    mixed = later.replace_bindings(states={left: earlier.at(left).state})
    Pair.captures = Value.captures = Pair.restores = Value.restores = 0
    def body(obj, ctx):
        ctx.final_state(mixed)
        assert ctx.final_state_ref is mixed
        return mixed
    Pair.action = body
    raw, final = enter(ManagedControlStore(repo.default_store, repo), live)
    assert raw is final is mixed and _selection_equal(final, mixed)
    assert (Pair.captures, Value.captures, Pair.restores, Value.restores) == (0, 0, 0, 0)
    assert (live.value, live.left.value, live.right.value) == (5, 11, 12)


def test_unordered_initial_and_final_accept_independent_decodes_and_keep_wire_evidence(tmp_path, monkeypatch):
    from dryml.core import Repo
    from dryml.core.store.dir import DirStore
    from dryml.managed.control import ManagedControlStore
    from tests.managed.test_control_codec import force_fresh_handle_order
    class Unordered(Value):
        def __init__(self, children):
            self.children = children
    flip = force_fresh_handle_order(monkeypatch)
    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    live = Unordered({Value(i, repo=repo) for i in range(5)}, repo=repo)
    saved = repo.save(live)
    packet = encode_node_graph(NodePreparation((saved,)))
    flip()
    independent = decode_node_graph(packet)
    def body(obj, ctx):
        ctx.initial_state(saved)
        ctx.initial_state(independent)
        ctx.final_state(independent)
        return independent
    Unordered.action = body
    Unordered.captures = Value.captures = Unordered.restores = Value.restores = 0
    control = ManagedControlStore(repo.default_store, repo)
    raw, final = enter(control, live)
    assert raw is final is independent
    current = control.inspect(live.run.status(state_repo=repo).operation_id)
    assert _selection_equal(current.initial, saved)
    assert _selection_equal(next(v.selection for v in current.intents if v.kind == "final"), independent)
    assert (Unordered.captures, Value.captures, Unordered.restores, Value.restores) == (0, 0, 0, 0)
