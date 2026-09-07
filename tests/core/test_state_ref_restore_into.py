from pathlib import Path

import pytest

from dryml.core import Object, Repo, Serializable
from dryml.core.object import Pickleable
from dryml.core.repo import RepoLoadError, RepoSaveError
from dryml.core.store.dir import DirStore


class RestoreIntoValue(Serializable):
    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "value").write_text(str(self.value), encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        self.value = int(Path(src_dir, "value").read_text(encoding="ascii"))


class RestoreIntoPair(Object):
    def __init__(self, child):
        self.first = child
        self.second = child


class RestoreIntoIndependentPair(Object):
    def __init__(self, first, second):
        self.first = first
        self.second = second


class FailingRestoreIntoValue(RestoreIntoValue):
    fail = False

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        super().restore_state_from_dir_imp(src_dir, codec=codec)
        if type(self).fail:
            raise RuntimeError("restore hook failed")


class SelectivelyFailingRestoreValue(RestoreIntoValue):
    failing_target = None

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        super().restore_state_from_dir_imp(src_dir, codec=codec)
        if self is type(self).failing_target:
            raise RuntimeError("selected restore hook failed")


class RestoreIntoPickleable(Pickleable):
    def __init__(self, value):
        self.value = value


class RestoreIntoPickleableParent(Pickleable):
    def __init__(self, child, value):
        self.child = child
        self.value = value


class CountingEmbeddedValue(RestoreIntoValue):
    restores = 0

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        type(self).restores += 1
        super().restore_state_from_dir_imp(src_dir, codec=codec)


class CountingEmbeddedParent(Serializable):
    observed_child_value = None

    def __init__(self, child):
        self.child = child

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        type(self).observed_child_value = self.child[0].value


def test_targeted_restore_keeps_the_supplied_live_identity_and_forces_hook(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    obj = RestoreIntoValue(1, repo=repo)
    state = repo.save_object(obj)
    obj.value = 99

    with repo.reserve_state_graph(obj) as reservation:
        restored = repo.restore_state_ref_into(obj, state, reservation=reservation)

    assert restored is state
    assert obj.value == 1
    assert obj.last_state_ref is state


def test_targeted_restore_preserves_alias_identity_and_invalidates_after_hook_failure(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    child = FailingRestoreIntoValue(2, repo=repo)
    root = RestoreIntoPair(child, repo=repo)
    state = repo.save_object(root)
    child.value = 99

    repo.restore_state_ref_into(root, state)
    assert root.first is root.second is child
    assert child.value == 2

    FailingRestoreIntoValue.fail = True
    try:
        with pytest.raises(RepoLoadError, match="Targeted exact restore failed"):
            repo.restore_state_ref_into(root, state)
    finally:
        FailingRestoreIntoValue.fail = False

    assert root._restore_failed
    assert child._restore_failed
    assert root.last_state_ref is None
    assert child._last_state_hash is None
    with pytest.raises(RepoSaveError, match="invalidated"):
        repo.save_object(root)
    with pytest.raises(RepoLoadError, match="invalidated"):
        repo.restore_state_ref_into(root, state)
    assert child not in repo._all_live_candidates()

    fresh = Repo(DirStore(tmp_path / "store")).load_state_ref(state, reuse_live="never")
    assert not fresh._restore_failed
    assert fresh.first.value == 2


def test_restore_failure_invalidates_equal_definition_siblings(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    first = SelectivelyFailingRestoreValue(2, repo=repo)
    second = SelectivelyFailingRestoreValue(2, repo=repo)
    root = RestoreIntoIndependentPair(first, second, repo=repo)
    state = repo.save_object(root)
    SelectivelyFailingRestoreValue.failing_target = second

    try:
        with pytest.raises(RepoLoadError, match="Targeted exact restore failed"):
            repo.restore_state_ref_into(root, state)
    finally:
        SelectivelyFailingRestoreValue.failing_target = None

    assert root._restore_failed
    assert first._restore_failed
    assert second._restore_failed
    with pytest.raises(RepoSaveError, match="invalidated"):
        repo.save_object(second)


def test_targeted_restore_preflight_failure_leaves_target_valid_and_pickle_payload_replaces(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    obj = RestoreIntoPickleable(1, repo=repo)
    state = repo.save_object(obj)
    obj.extra = "discard me"
    repo.restore_state_ref_into(obj, state)
    assert not hasattr(obj, "extra")
    assert obj.value == 1

    path, state_hash = next(iter(state.states.items()))
    definition = state.object.at(path).definition
    Path(store._local_state_path(definition.graph_hash(), state_hash), "data", "heavy.pkl").unlink()
    with pytest.raises(RepoLoadError, match="preflight is incomplete"):
        repo.restore_state_ref_into(obj, state)
    assert not obj._restore_failed


def test_targeted_restore_walks_embedded_reference_closure_once_before_parent_hook(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    child = CountingEmbeddedValue(1, repo=repo)
    seed = repo.save_object(child)
    root = CountingEmbeddedParent([seed, seed], repo=repo)
    state = repo.save_object(root)
    child = root.child[0]
    child.value = 99
    CountingEmbeddedValue.restores = 0
    CountingEmbeddedParent.observed_child_value = None

    repo.restore_state_ref_into(root, state)

    assert root.child[0] is root.child[1] is child
    assert child.value == 1
    assert CountingEmbeddedValue.restores == 1
    assert CountingEmbeddedParent.observed_child_value == 1


def test_pickleable_targeted_restore_preserves_live_graph_bindings(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    child = RestoreIntoValue(2, repo=repo)
    root = RestoreIntoPickleableParent(child, 1, repo=repo)
    state = repo.save_object(root)
    root.extra = "discard me"
    root.value = 99

    repo.restore_state_ref_into(root, state)

    assert root.child is child
    assert root.child.object_id == child.object_id
    assert root.value == 1
    assert not hasattr(root, "extra")
    assert root.object_ref == state.object
