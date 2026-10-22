"""Actual public completion, its own index cut, and metadata-only repair."""

import json

import pytest

from dryml.core.query.model import QueryIndexDirty
from dryml.core.store.authority import StoreMutationError
from dryml.managed import ManagedConfig
from dryml.managed.control import ManagedControlStore, ControlSnapshot
from tests.managed.test_completion_result import CompletionValue
from tests.managed.u16_fixtures import Value, setup


@pytest.mark.parametrize("phase", ["cold", "warm", "reopened"])
@pytest.mark.parametrize("selected", [False, True])
def test_public_checkpoint_final_work_budget(tmp_path, phase, selected):
    """Current equivalents of the six old closure budgets count actual hooks."""
    control, live = setup(tmp_path, index="sqlite")
    store = control.control_store
    if phase != "cold": store.open_query_index().maintain_store()
    if phase == "reopened":
        from dryml.core import Repo
        from dryml.core.store.dir import DirStore
        control = ManagedControlStore(DirStore(store.base_dir, query_index="sqlite"), Repo(store))
    callbacks = []
    def body(obj, ctx):
        obj.value = 1
        saved = ctx.checkpoint()
        if selected: ctx.final_state(saved)
        obj.value = 2
        return {"value": 2}
    Value.action = body
    result = live.run(managed=ManagedConfig(state_repo=control.state_repo,
        control_store=control.control_store, callbacks=[lambda obj, ctx: callbacks.append(ctx.latest_checkpoint)]))
    assert result.value == {"value": 2} and result.status.state == "completed"
    assert (Value.bodies, Value.captures, Value.restores, len(callbacks)) == (1, 1 if selected else 2, 0, 1)
    assert control.state_repo.load(result.final_state_ref, cache="none").value == (1 if selected else 2)


def test_completion_own_projection_failure_withholds_success_and_repairs_without_effects(tmp_path, monkeypatch):
    from dryml.core import Repo
    from dryml.core.store.dir import DirStore
    store = DirStore(tmp_path / "state", query_index="sqlite")
    repo = Repo(store)
    live = CompletionValue(repo=repo)
    CompletionValue.bodies = CompletionValue.captures = CompletionValue.restores = CompletionValue.handoffs = 0
    control = ManagedControlStore(store, repo)
    index = store.open_query_index()
    original = index.maintain_store
    committed = []
    def fail_after_completion(*args, **kwargs):
        from dryml.managed.identity import operation_digest
        operation_id = operation_digest(repo.describe(live), "advance")
        raw = control._read_optional(control._current_path(control._operation_path(operation_id)))
        if raw is not None and ControlSnapshot.from_bytes(raw).state == "completed" and store.authority_cut().pending:
            committed.append(ControlSnapshot.from_bytes(raw))
            raise OSError("completion SQL phase failed")
        return original(*args, **kwargs)
    monkeypatch.setattr(index, "maintain_store", fail_after_completion)
    with pytest.raises(StoreMutationError) as raised:
        live.advance(managed=ManagedConfig(state_repo=repo))
    assert committed and raised.value.mutation_id and raised.value.cut.pending
    current = control.inspect(committed[0].operation_id)
    assert current.state == "completed" and current.result_reference is not None
    assert live.advance.status(state_repo=repo).state == "completed"
    pending = store.pending_mutations()
    assert any("managed-current:" + current.operation_id in r.intent["scopes"] for r in pending)
    for terminal in (bool, len, tuple):
        with pytest.raises(QueryIndexDirty): terminal(index.projection_rows("lifecycle", refresh=False))
    assert (CompletionValue.bodies, CompletionValue.captures, CompletionValue.restores, CompletionValue.handoffs) == (1, 1, 0, 0)
    monkeypatch.setattr(index, "maintain_store", original)
    counts = CompletionValue.bodies, CompletionValue.captures, CompletionValue.restores, CompletionValue.handoffs
    recovered = live.advance(managed=ManagedConfig(state_repo=repo))
    assert recovered.disposition == "completion_recorded"
    assert (CompletionValue.bodies, CompletionValue.captures, CompletionValue.restores, CompletionValue.handoffs) == counts
    assert not store.pending_mutations()
    rows = index.projection_rows("lifecycle", refresh=False)
    assert any(key == "current" and json.loads(payload)["state"] == "completed" for _, key, payload in rows)
    assert any(key == "result" for _, key, _ in rows)
    assert any(key == "result-content" for _, key, _ in rows)
    assert live.advance(managed=ManagedConfig(state_repo=repo)).value == {"value": 3}
    assert (CompletionValue.bodies, CompletionValue.captures, CompletionValue.restores, CompletionValue.handoffs) == counts


def test_consumer_failure_preserves_completion_and_is_never_replayed(tmp_path, monkeypatch):
    from dryml.core import Repo
    from dryml.core.store.dir import DirStore
    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    live = CompletionValue(repo=repo)
    calls = []
    def consumer(*args):
        calls.append(args)
        raise RuntimeError("consumer failed after completion")
    monkeypatch.setattr(CompletionValue, "_managed_post_publication", consumer)
    with pytest.raises(RuntimeError, match="consumer failed"):
        live.advance(managed=ManagedConfig(state_repo=repo))
    assert live.advance.status(state_repo=repo).state == "completed"
    assert live.advance(managed=ManagedConfig(state_repo=repo)).value == {"value": 3}
    assert len(calls) == 1


@pytest.mark.parametrize("after", [False, True])
@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_cancelled_completion_CAS_keeps_exact_cut_and_reconciles_without_effects(tmp_path, monkeypatch, after, error_type):
    from dryml.core import Repo
    from dryml.core.store.dir import DirStore
    repo = Repo(DirStore(tmp_path / "state", query_index="sqlite"))
    live = CompletionValue(repo=repo)
    CompletionValue.bodies = CompletionValue.captures = CompletionValue.restores = CompletionValue.handoffs = 0
    original = ManagedControlStore._replace_file
    def replace(path, payload):
        if path.endswith("current.json") and json.loads(payload).get("state") == "completed":
            if after: original(path, payload)
            raise error_type()
        return original(path, payload)
    monkeypatch.setattr(ManagedControlStore, "_replace_file", staticmethod(replace))
    with pytest.raises(BaseException): live.advance(managed=ManagedConfig(state_repo=repo))
    from dryml.managed.identity import operation_digest
    operation_id = operation_digest(repo.describe(live), "advance")
    control = ManagedControlStore(repo.default_store, repo)
    raw = control._read_optional(control._current_path(control._operation_path(operation_id)))
    current = ControlSnapshot.from_bytes(raw)
    assert current.state == ("completed" if after else "finalizing")
    from dryml.managed.control import _PendingIntent
    import hashlib
    pending = _PendingIntent.from_bytes(control._read_optional(control._pending_path(control._operation_path(operation_id))), operation_id=operation_id)
    assert hashlib.sha256(raw).hexdigest() == (pending.proposed_digest if after else pending.previous_digest)
    counts = CompletionValue.bodies, CompletionValue.captures, CompletionValue.restores, CompletionValue.handoffs
    assert counts == (1, 1, 0, 0)
    monkeypatch.setattr(ManagedControlStore, "_replace_file", staticmethod(original))
    assert live.advance(managed=ManagedConfig(state_repo=repo)).disposition == "completion_recorded"
    assert live.advance(managed=ManagedConfig(state_repo=repo)).value == {"value": 3}
    assert (CompletionValue.bodies, CompletionValue.captures, CompletionValue.restores, CompletionValue.handoffs) == counts


def test_mismatched_provisional_receipt_cannot_release_an_independent_user_root(tmp_path):
    from dataclasses import replace
    from dryml.managed import ManagedResultError
    from dryml.managed.runtime import _retire_publications
    from dryml.core.store.retention import grants
    control, live = setup(tmp_path)
    saved = control.state_repo.save(live)
    user = next(v for v in grants(control.control_store, saved.state.key) if control.control_store.retention.read(v.token).disposition == "adopted")
    def body(obj, ctx):
        ctx.final_state(ctx.checkpoint())
        root = next(v for v in ctx._snapshot.intents if v.kind == "latest")
        assert root.publications
        mismatched = replace(root.publications[0], retention_token=user.token)
        with pytest.raises(ManagedResultError, match="another publication"):
            _retire_publications(control, replace(root, publications=(mismatched,)))
        assert control.control_store.retention.read(user.token).disposition == "adopted"
        return 1
    Value.action = body
    assert live.run(managed=ManagedConfig(state_repo=control.state_repo)).value == 1
