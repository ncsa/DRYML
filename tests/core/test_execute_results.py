"""Result-publication coverage for the core Execute strategy boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from dryml.core import ConcreteDefinition, Object, ObjectRef, Repo, SaveRouting, Selector, Serializable, StateRef
from dryml.core.execute import (
    CoreExecutionError, ExecutionContext, SharedDirStoreStrategy, current_context, decode_core_outcome,
    worker_context,
)
from dryml.core.execute_codec import CoreCallCodecError, _outcome
from dryml.core.signatures import Ref
from dryml.core.store.dir import DirStore


class ResultValue(Serializable):
    """Small stateful result fixture with visible durable payload."""

    def __init__(self, value=0):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "value").write_text(str(self.value), encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        self.value = int(Path(src_dir, "value").read_text(encoding="ascii"))


class StructuralResult(Object):
    """Stateless result fixture whose definition is sufficient authority."""


def _state_result(value):
    """Return a new stateful worker-produced result."""
    return ResultValue(value)


def _structural_result():
    """Return a new stateless worker-produced result."""
    return StructuralResult()


def _nested(reference: Ref[ObjectRef]):
    """Return repeated live and pre-existing reference leaves in nested values."""
    value = ResultValue(3)
    return {"items": (value, [value, reference])}


def _reference_result(reference: Ref[ObjectRef]) -> Ref[ObjectRef]:
    """Return explicitly reference-valued authority without materializing it."""
    return reference


def _cycle():
    """Return an unsupported cyclic ordinary result graph."""
    value = []
    value.append(value)
    return value


def _cycle_after_update(value):
    """Return a rejected cycle after mutating an otherwise selected update root."""
    value.value += 1
    result = []
    result.append(result)
    return result


def _live_repo_after_update(value):
    """Return a rejected worker Repo after mutating a selected update root."""
    value.value += 1
    return current_context().repo


def _worker_store_count():
    """Return ordinary data through the framework-owned worker context getter."""
    return len(current_context().repo.stores)


def _independent_same_identity_results(reference: Ref[StateRef]):
    """Return distinct live graphs that deliberately share one ObjectRef identity."""
    from dryml.core.execute import current_context

    repo = current_context().repo
    first = repo.load_state_ref(reference, reuse_live="never")
    second = repo.load_state_ref(reference, reuse_live="never")
    first.value = 11
    second.value = 12
    return first, second


def _multiple_roots():
    """Return separate live roots whose full evidence cannot fit a small outcome."""
    return ResultValue(1), ResultValue(2)


def _invoke(strategy, fn, args, repo, *, return_objects=False):
    """Execute and recover a worker result using one shared Repo authority."""
    prepared = strategy.prepare(fn, args, {}, repo=repo, control_store=None, update_args=False)
    output = strategy.invoke(prepared.invocation, repo=repo, update_args=False)
    return strategy.recover(
        output, prepared, repo=repo, args=args, kwargs={},
        return_objects=return_objects, update_args=False,
    )


def test_live_results_publish_before_automatic_reference_selection(tmp_path):
    """Stateful and stateless live results become StateRef and CDef after saving."""
    repo = Repo(DirStore(tmp_path / "store"))
    strategy = SharedDirStoreStrategy()

    state = _invoke(strategy, _state_result, (7,), repo)
    structural = _invoke(strategy, _structural_result, (), repo)

    assert isinstance(state, StateRef)
    assert Repo(DirStore.open_existing(tmp_path / "store")).load_state_ref(state, reuse_live="never").value == 7
    assert isinstance(structural, ConcreteDefinition)


def test_nested_results_preserve_reference_leaves_and_repeated_live_aliases(tmp_path):
    """Recursive publication keeps ordinary shape, aliases, and existing references."""
    repo = Repo(DirStore(tmp_path / "store"))
    value = ResultValue(7, repo=repo)
    reference = repo.save(value, deep_capture=True)

    result = _invoke(SharedDirStoreStrategy(), _nested, (reference.object,), repo)

    first, repeated = result["items"]
    assert isinstance(first, StateRef)
    assert first is repeated[0]
    assert repeated[1] == reference.object
    assert _invoke(SharedDirStoreStrategy(), _reference_result, (reference.object,), repo) == reference.object


def test_return_objects_materializes_only_auto_results_once_and_never_refreshes_inputs(tmp_path):
    """Auto-published leaves share one fresh graph while Ref leaves and inputs stay inert."""
    repo = Repo(DirStore(tmp_path / "store"))
    original = ResultValue(2, repo=repo)
    state = repo.save(original, deep_capture=True)
    strategy = SharedDirStoreStrategy()
    prepared = strategy.prepare(_nested, (state.object,), {}, repo=repo, control_store=None, update_args=False)

    result = strategy.recover(
        strategy.invoke(prepared.invocation, repo=repo, update_args=False), prepared,
        repo=repo, args=(state.object,), kwargs={}, return_objects=True, update_args=False,
    )

    first, repeated = result["items"]
    assert first is repeated[0]
    assert isinstance(repeated[1], ObjectRef)
    assert original.value == 2


def test_result_cycles_fail_without_claiming_a_successful_outcome(tmp_path):
    """Cycles are rejected before result transport instead of being pickled."""
    repo = Repo(DirStore(tmp_path / "store"))
    strategy = SharedDirStoreStrategy()
    prepared = strategy.prepare(_cycle, (), {}, repo=repo, control_store=None, update_args=False)

    output = strategy.invoke(prepared.invocation, repo=repo, update_args=False)
    with pytest.raises(CoreExecutionError, match="CoreCallCodecError|cyclic"):
        strategy.recover(
            output, prepared, repo=repo, args=(), kwargs={},
            return_objects=False, update_args=False,
        )


def test_rejected_result_graph_does_not_publish_selected_updates(tmp_path, monkeypatch):
    """Cycle validation rejects the complete result/update graph before any save."""
    repo = Repo(DirStore(tmp_path / "store"))
    value = ResultValue(2, repo=repo)
    repo.save(value, deep_capture=True)
    saves = []
    save = repo.save
    monkeypatch.setattr(repo, "save", lambda *args, **kwargs: saves.append(args[0]) or save(*args, **kwargs))
    strategy = SharedDirStoreStrategy()
    prepared = strategy.prepare(
        _cycle_after_update, (value,), {}, repo=repo, control_store=None, update_args=True,
    )

    output = strategy.invoke(prepared.invocation, repo=repo, update_args=True)

    assert not decode_core_outcome(output, repo=repo).value["success"]
    assert saves == []


def test_live_resource_result_does_not_publish_selected_updates(tmp_path, monkeypatch):
    """Live resources are rejected before an update save can make side effects."""
    repo = Repo(DirStore(tmp_path / "store"))
    value = ResultValue(2, repo=repo)
    repo.save(value, deep_capture=True)
    saves = []
    save = repo.save
    monkeypatch.setattr(repo, "save", lambda *args, **kwargs: saves.append(args[0]) or save(*args, **kwargs))
    strategy = SharedDirStoreStrategy()
    prepared = strategy.prepare(
        _live_repo_after_update, (value,), {}, repo=repo, control_store=None, update_args=True,
    )

    with worker_context(ExecutionContext(repo, None)):
        output = strategy.invoke(prepared.invocation, repo=repo, update_args=True)

    assert not decode_core_outcome(output, repo=repo).value["success"]
    assert saves == []


def test_framework_current_context_dependency_resolves_in_worker(tmp_path):
    """A stable DRYML context getter resolves against worker-local state."""
    repo = Repo(DirStore(tmp_path / "store"))
    strategy = SharedDirStoreStrategy()
    prepared = strategy.prepare(
        _worker_store_count, (), {}, repo=repo, control_store=None,
        update_args=False,
    )

    with worker_context(ExecutionContext(repo, None)):
        output = strategy.invoke(prepared.invocation, repo=repo, update_args=False)

    outcome = decode_core_outcome(output, repo=repo).value
    assert outcome["success"]
    assert outcome["result"] == 1


def test_same_identity_independent_result_graphs_do_not_share_a_snapshot(tmp_path):
    """Only actual runtime aliases, never matching ObjectRef digests, reuse snapshots."""
    repo = Repo(DirStore(tmp_path / "store"))
    state = repo.save(ResultValue(2, repo=repo), deep_capture=True)
    strategy = SharedDirStoreStrategy()
    prepared = strategy.prepare(
        _independent_same_identity_results, (state,), {}, repo=repo,
        control_store=None, update_args=False,
    )

    with worker_context(ExecutionContext(repo, None)):
        output = strategy.invoke(prepared.invocation, repo=repo, update_args=False)
    first, second = decode_core_outcome(output, repo=repo).value["result"]

    assert isinstance(first, StateRef)
    assert isinstance(second, StateRef)
    assert first != second
    assert Repo(DirStore.open_existing(tmp_path / "store")).load_state_ref(first, reuse_live="never").value == 11
    assert Repo(DirStore.open_existing(tmp_path / "store")).load_state_ref(second, reuse_live="never").value == 12


def test_small_outcome_limit_rejects_multiple_roots_before_publication(tmp_path, monkeypatch):
    """Evidence reservation rejects a multi-root result before Execute-owned saves."""
    repo = Repo(DirStore(tmp_path / "store"))
    saves = []
    save = repo.save
    monkeypatch.setattr(repo, "save", lambda *args, **kwargs: saves.append(args[0]) or save(*args, **kwargs))
    strategy = SharedDirStoreStrategy()
    prepared = strategy.prepare(_multiple_roots, (), {}, repo=repo, control_store=None, update_args=False)

    outcome = decode_core_outcome(
        strategy.invoke(
            prepared.invocation, repo=repo, update_args=False, result_limit_bytes=1_000,
        ),
        repo=repo,
    )

    assert not outcome.value["success"]
    assert outcome.value["reason"] == "CoreCallCodecError"
    assert outcome.evidence.publications == ()
    assert saves == []


def test_post_publication_overflow_retains_exact_evidence_only(tmp_path):
    """The bounded fallback drops only result bytes and retains readable StateRefs."""
    repo = Repo(DirStore(tmp_path / "store"))
    state = repo.save(ResultValue(4, repo=repo), deep_capture=True)
    publications = [{
        "state": state.to_data(), "store": 0, "phase": "snapshot",
        "status": "completed", "path": "$",
    }]
    failure_size = len(_outcome(
        False, publications=publications, reason="result outcome exceeds configured bound after publication",
        limit_bytes=1_000_000,
    ))

    outcome = decode_core_outcome(_outcome(
        True, result=b"x" * 8_192, publications=publications,
        limit_bytes=failure_size + 32,
    ), repo=repo)

    assert not outcome.value["success"]
    assert outcome.evidence.publications[0].state_ref == state
    assert Repo(DirStore.open_existing(tmp_path / "store")).load_state_ref(
        outcome.evidence.publications[0].state_ref, reuse_live="never",
    ).value == 4


def test_outcome_evidence_uses_exact_refs_and_store_table_indexes_only(tmp_path):
    """Publication evidence retains authority without serializing StoreReport handles."""
    repo = Repo(DirStore(tmp_path / "store"))
    strategy = SharedDirStoreStrategy()
    prepared = strategy.prepare(_state_result, (5,), {}, repo=repo, control_store=None, update_args=False)

    outcome = decode_core_outcome(
        strategy.invoke(prepared.invocation, repo=repo, update_args=False), repo=repo,
    )

    assert isinstance(outcome.value["result"], StateRef)
    assert outcome.evidence.publications
    assert all(item.state_ref == outcome.value["result"] for item in outcome.evidence.publications)
    assert {item.store_index for item in outcome.evidence.publications} == {0}


def test_partial_replica_publication_is_a_failed_outcome_with_exact_evidence(tmp_path, monkeypatch):
    """A failed replica does not erase the first Store's completed authority evidence."""
    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    repo = Repo(
        [first, second],
        save_routing=SaveRouting(((Selector(ResultValue), first), (Selector(ResultValue), second)), "all"),
    )
    strategy = SharedDirStoreStrategy()
    prepared = strategy.prepare(_state_result, (6,), {}, repo=repo, control_store=None, update_args=False)
    worker = Repo.from_definition(repo.to_definition())
    monkeypatch.setattr(
        worker.stores[1], "publish_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("replica failure")),
    )
    try:
        output = strategy.invoke(prepared.invocation, repo=worker, update_args=False)
    finally:
        worker.close(flush=False)

    decoded = decode_core_outcome(output, repo=repo)
    assert not decoded.value["success"]
    assert {(item.store_index, item.status) for item in decoded.evidence.publications if item.phase == "snapshot"} == {
        (0, "completed"), (1, "failed"),
    }
    with pytest.raises(CoreExecutionError) as raised:
        strategy.recover(
            output, prepared, repo=repo, args=(6,), kwargs={},
            return_objects=False, update_args=False,
        )
    assert raised.value.evidence == decoded.evidence
