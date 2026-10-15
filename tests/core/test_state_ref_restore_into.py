from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

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

    path = next(iter(state.states))
    Path(store.open_local_state(state, path).handle, "data", "heavy.pkl").unlink()
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


@pytest.mark.speed_heavy
@pytest.mark.parametrize("kind", ("xgb-regressor", "stats-logit"))
@pytest.mark.parametrize("store_kind", ("dir", "zip"))
def test_native_graph_restore_unfitted_fitted_invalidation_and_fresh_recovery(
        tmp_path, monkeypatch, kind, store_kind):
    """Native candidate isolation cannot rescue a Core-invalidated aliased graph."""
    import numpy as np
    from dryml.core.store.zip import ZipStore
    from tests.models.test_estimator_experiments import _arrays, _recipe

    destination = tmp_path / ("store" if store_kind == "dir" else "store.zip")
    store_cls = DirStore if store_kind == "dir" else ZipStore
    repo = Repo(store_cls(destination))
    model, _ = _recipe(kind)
    root = RestoreIntoPair(model, repo=repo)
    x, y = _arrays(kind)
    try:
        empty = repo.save_object(root, deep_capture=True)
        model.fit(x, y)
        fitted = repo.save_object(root, deep_capture=True)
        repo.default_store.commit()
        payload, metadata = model._accepted_fit.payload, model._accepted_fit.metadata
        expected = model(x)
        object_id = model.object_id
        repo.restore_state_ref_into(root, empty)
        assert root.first is root.second is model and model.object_id == object_id
        assert model._accepted_fit is None
        with pytest.raises(ValueError, match="unfitted"):
            model(x)
        repo.restore_state_ref_into(root, fitted)
        assert root.first is root.second is model and model.object_id == object_id
        assert model._accepted_fit.payload == payload
        assert model._accepted_fit.metadata == metadata
        np.testing.assert_array_equal(model(x), expected)
        prior = model._accepted_fit
        assert root.last_state_ref == fitted
        assert model._last_state_hash is not None
        for node in (root, model):
            assert any(candidate is node for candidate in repo._all_live_candidates())

        def reject(candidate):
            raise ValueError("entered native restore validation")

        monkeypatch.setattr(model, "_validate_native", reject)
        with pytest.raises(RepoLoadError, match="Targeted exact restore failed"):
            repo.restore_state_ref_into(root, fitted)
        assert model._accepted_fit is prior
        for node in (root, model):
            assert node._restore_failed
            assert node.last_state_ref is None and node._last_state_hash is None
            assert all(candidate is not node for candidate in repo._all_live_candidates())
            with pytest.raises(RepoSaveError, match="invalidated"):
                repo.save_object(node)
        with pytest.raises(RepoLoadError, match="invalidated"):
            repo.restore_state_ref_into(root, fitted)
        fresh_repo = Repo(store_cls(destination))
        try:
            fresh_empty = fresh_repo.load_state_ref(empty, reuse_live="never", cache="none")
            fresh = fresh_repo.load_state_ref(fitted, reuse_live="never", cache="none")
            assert fresh_empty.first._accepted_fit is None
            assert fresh is not root and fresh.first is not model
            assert fresh.first is fresh.second
            assert fresh.first.object_id == object_id and fresh.object_ref == fitted.object
            assert not fresh._restore_failed and not fresh.first._restore_failed
            assert fresh.last_state_ref == fitted
            assert fresh.first._accepted_fit.payload == payload
            assert fresh.first._accepted_fit.metadata == metadata
            np.testing.assert_array_equal(fresh.first(x), expected)
        finally:
            fresh_repo.close(flush=False)
    finally:
        repo.close(flush=False)
