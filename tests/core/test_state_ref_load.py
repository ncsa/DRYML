from pathlib import Path

import pytest

from dryml.core import Object, Repo, Serializable
from dryml.core.repo import RepoLoadError
from dryml.core.store.dir import DirStore


class RestoredValue(Serializable):
    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "value").write_text(str(self.value), encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        self.value = int(Path(src_dir, "value").read_text(encoding="ascii"))


class Pair(Object):
    def __init__(self, left, right):
        self.left = left
        self.right = right


class RestoredParent(Object):
    observed_child_values = []

    def __init__(self, child):
        type(self).observed_child_values.append(child.value)
        self.child = child


class FailedRestoreValue(RestoredValue):
    constructed = None

    def __init__(self, value):
        super().__init__(value)
        type(self).constructed = self

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        raise RuntimeError("incompatible payload")


def test_load_state_ref_restores_saved_identity_and_shared_topology(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    leaf = RestoredValue(1, repo=repo)
    leaf.value = 9
    state = repo.save_object(Pair(leaf, leaf, repo=repo))

    loaded = Repo(DirStore(tmp_path / "store")).load_state_ref(state, reuse_live="never")

    assert loaded.left is loaded.right
    assert loaded.left.value == 9
    assert loaded.object_ref == state.object
    assert loaded.left.object_id == leaf.object_id


def test_exact_load_restores_dependencies_before_parent_construction(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    child = RestoredValue(1, repo=repo)
    child.value = 9
    state = repo.save_object(RestoredParent(child, repo=repo))
    RestoredParent.observed_child_values = []

    loaded = Repo(DirStore(tmp_path / "store")).load_state_ref(state, reuse_live="never")

    assert RestoredParent.observed_child_values == [9]
    assert loaded.child.value == 9


def test_failed_fresh_exact_restore_leaves_no_cache_or_state_hash(tmp_path):
    source = DirStore(tmp_path / "store")
    state = Repo(source).save_object(FailedRestoreValue(1))
    reopened = Repo(DirStore(tmp_path / "store"))
    FailedRestoreValue.constructed = None

    with pytest.raises(RepoLoadError, match="codec 'pkl'.*incompatible payload") as error:
        reopened.load_state_ref(state, reuse_live="never", cache="strong")

    assert isinstance(error.value.__cause__, RuntimeError)
    assert FailedRestoreValue.constructed._last_state_hash is None
    assert reopened._all_live_candidates() == ()
