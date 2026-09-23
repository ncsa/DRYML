"""Argument-publication and in-place refresh coverage for core Execute."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

from dryml.core import Repo, Serializable, StateRef
from dryml.core.execute import CoreExecutionError, SharedDirStoreStrategy
from dryml.core.execute_codec import decode_outcome
from dryml.core.signatures import Ref
from dryml.core.store.dir import DirStore


class UpdatedValue(Serializable):
    """Stateful argument fixture whose worker mutation is restored into caller identity."""

    def __init__(self, value=0):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "value").write_text(str(self.value), encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        self.value = int(Path(src_dir, "value").read_text(encoding="ascii"))


class UpdatedRoot(Serializable):
    """Graph root fixture retaining an independently stateful child argument."""

    def __init__(self, child):
        self.child = child


def _mutate(value):
    """Mutate one selected worker argument and return it as an alias."""
    value.value += 1
    return value


def _mutate_and_return(value):
    """Mutate one delivered graph to prove fresh result recovery is independent."""
    value.value += 1
    return value


def _mutate_pair(first, second):
    """Mutate two independently selected roots for refresh-ledger failure coverage."""
    first.value += 1
    second.value += 1
    return None


def _mutate_child(root) -> Ref[StateRef]:
    """Mutate and return a descendant shared with its selected update root."""
    child = root.child
    child.value += 1
    return child


def test_update_args_publishes_once_and_restores_the_original_argument(tmp_path, monkeypatch):
    """Opt-in refresh restores the original caller object from its exact saved state."""
    repo = Repo(DirStore(tmp_path / "store"))
    original = UpdatedValue(2, repo=repo)
    repo.save(original, deep_capture=True)
    saves = []
    save = repo.save
    monkeypatch.setattr(repo, "save", lambda *args, **kwargs: saves.append(args[0]) or save(*args, **kwargs))
    strategy = SharedDirStoreStrategy()
    prepared = strategy.prepare(_mutate, (original,), {}, repo=repo, control_store=None, update_args=True)

    output = strategy.invoke(prepared.invocation, repo=repo, update_args=True)
    result = strategy.recover(
        output, prepared, repo=repo, args=(original,), kwargs={},
        return_objects=False, update_args=True,
    )

    assert isinstance(result, StateRef)
    assert original.value == 3
    assert original.last_state_ref == result
    assert len(saves) == 1
    assert saves[0] is not original


def test_returned_input_without_updates_is_a_fresh_graph_and_does_not_mutate_original(tmp_path):
    """Fresh auto-result recovery never reuses or refreshes the caller input instance."""
    repo = Repo(DirStore(tmp_path / "store"))
    original = UpdatedValue(2, repo=repo)
    repo.save(original, deep_capture=True)
    strategy = SharedDirStoreStrategy()
    prepared = strategy.prepare(_mutate_and_return, (original,), {}, repo=repo, control_store=None, update_args=False)

    returned = strategy.recover(
        strategy.invoke(prepared.invocation, repo=repo, update_args=False), prepared,
        repo=repo, args=(original,), kwargs={}, return_objects=True, update_args=False,
    )

    assert returned is not original
    assert returned.value == 3
    assert original.value == 2


def test_refresh_failure_retains_ordered_ledger_and_never_retries_earlier_targets(tmp_path, monkeypatch):
    """A middle refresh failure records completed authority without replaying restores."""
    repo = Repo(DirStore(tmp_path / "store"))
    first = UpdatedValue(1, repo=repo)
    second = UpdatedValue(2, repo=repo)
    repo.save(first, deep_capture=True)
    repo.save(second, deep_capture=True)
    strategy = SharedDirStoreStrategy()
    prepared = strategy.prepare(_mutate_pair, (first, second), {}, repo=repo, control_store=None, update_args=True)
    recovery = strategy.bind_recovery(prepared, args=(first, second), kwargs={})
    output = strategy.invoke(prepared.invocation, repo=repo, update_args=True)
    restore = repo.restore_state_ref_into
    calls = []

    def fail_second(target, state):
        calls.append(target)
        if target is second:
            raise RuntimeError("second restore failed")
        return restore(target, state)

    monkeypatch.setattr(repo, "restore_state_ref_into", fail_second)
    with pytest.raises(CoreExecutionError) as raised:
        strategy.recover(output, prepared, repo=repo, args=(first, second), kwargs={}, return_objects=False, update_args=True, _recovery=recovery)

    assert first.value == 2
    assert [entry.status for entry in raised.value.evidence.refreshes] == ["applied", "preflight_failed"]
    with pytest.raises(CoreExecutionError) as repeated:
        strategy.recover(output, prepared, repo=repo, args=(first, second), kwargs={}, return_objects=False, update_args=True, _recovery=recovery)
    assert repeated.value is raised.value
    assert calls == [first, second]


def test_result_descendant_reuses_the_coalesced_update_snapshot_once(tmp_path, monkeypatch):
    """A returned update descendant is the exact projection of its saved root."""
    repo = Repo(DirStore(tmp_path / "store"))
    child = UpdatedValue(2, repo=repo)
    repo.save(child, deep_capture=True)
    root = UpdatedRoot(child, repo=repo)
    repo.save(root, deep_capture=True)
    saves = []
    original_save = Repo.save
    monkeypatch.setattr(
        Repo, "save", lambda self, *args, **kwargs: saves.append(args[0]) or original_save(self, *args, **kwargs),
    )
    strategy = SharedDirStoreStrategy()
    prepared = strategy.prepare(_mutate_child, (root,), {}, repo=repo, control_store=None, update_args=True)
    worker = Repo.from_definition(repo.to_definition())
    try:
        output = strategy.invoke(prepared.invocation, repo=worker, update_args=True)
    finally:
        worker.close(flush=False)
    outcome = decode_outcome(output, repo=repo)
    assert outcome["success"], outcome["reason"]
    result = outcome["result"]
    update = StateRef.from_data(outcome["updates"][0]["state"])

    assert result == update.at(next(path for path, value in root._runtime_projection.items() if value is child))
    strategy.recover(
        output, prepared, repo=repo, args=(root,), kwargs={},
        return_objects=False, update_args=True,
    )
    assert root.child is child
    assert child.value == 3
    assert len(saves) == 1
