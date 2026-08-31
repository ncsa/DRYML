from pathlib import Path

import pytest

from dryml.core import Object, Repo, Serializable
from dryml.core.repo import RepoLoadError
from dryml.core.store.dir import DirStore


class SeedValue(Serializable):
    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "value").write_text(str(self.value), encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        self.value = int(Path(src_dir, "value").read_text(encoding="ascii"))


class SeedParent(Object):
    def __init__(self, child):
        self.child = child


class FailingSeedParent(SeedParent):
    fail_construction = False

    def __init__(self, child):
        if type(self).fail_construction:
            raise RuntimeError("parent rejected restored seed")
        super().__init__(child)


def test_exact_load_resolves_materializing_state_ref_before_parent(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    child = SeedValue(1, repo=repo)
    child.value = 7
    seed = repo.save_object(child)
    parent = SeedParent(seed, repo=repo)
    state = repo.save_object(parent)

    loaded = Repo(DirStore(tmp_path / "store")).load_state_ref(state, reuse_live="never")

    assert loaded.child.value == 7


def test_exact_load_restores_a_repeated_materializing_state_ref_once(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    child = SeedValue(1, repo=repo)
    child.value = 7
    seed = repo.save_object(child)
    state = repo.save_object(SeedParent([seed, seed], repo=repo))

    loaded = Repo(DirStore(tmp_path / "store")).load_state_ref(state, reuse_live="never")

    assert loaded.child[0] is loaded.child[1]
    assert loaded.child[0].value == 7


def test_parent_failure_evicts_a_greedily_restored_materializing_seed(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    seed = repo.save_object(SeedValue(1, repo=repo))
    state = repo.save_object(FailingSeedParent(seed, repo=repo))
    reopened = Repo(DirStore(tmp_path / "store"))
    candidate = reopened.load_state_ref(seed, reuse_live="never")
    candidate._last_state_hash = "pkl-" + "0" * 64
    FailingSeedParent.fail_construction = True

    try:
        with pytest.raises(RepoLoadError, match="parent rejected restored seed") as error:
            reopened.load_state_ref(state, reuse_live="greedy")
    finally:
        FailingSeedParent.fail_construction = False

    assert "mutated and evicted candidates" in str(error.value)
    assert candidate._last_state_hash is None
    assert candidate not in reopened._all_live_candidates()
