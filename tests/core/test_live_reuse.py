from pathlib import Path

import pytest

from dryml.core import Object, Repo, Serializable
from dryml.core.repo import RepoLoadError
from dryml.core.store.dir import DirStore


class ReuseValue(Serializable):
    restores = 0

    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "value").write_text(str(self.value), encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        type(self).restores += 1
        self.value = int(Path(src_dir, "value").read_text(encoding="ascii"))


def test_matching_reuses_current_checkpoint_without_restoring(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    saved = ReuseValue(1, repo=repo)
    state = repo.save_object(saved)
    ReuseValue.restores = 0

    loaded = repo.load_state_ref(state, reuse_live="matching")

    assert loaded is saved
    assert ReuseValue.restores == 0


def test_never_constructs_a_new_live_object_under_the_saved_id(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    saved = ReuseValue(1, repo=repo)
    state = repo.save_object(saved)

    loaded = repo.load_state_ref(state, reuse_live="never")

    assert loaded is not saved
    assert loaded.object_id == saved.object_id


def test_greedy_restores_one_unique_stale_candidate_in_place(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    saved = ReuseValue(1, repo=repo)
    saved.value = 4
    state = repo.save_object(saved)
    saved.value = 99
    saved._last_state_hash = "pkl-" + "0" * 64
    ReuseValue.restores = 0

    loaded = repo.load_state_ref(state, reuse_live="greedy")

    assert loaded is saved
    assert loaded.value == 4
    assert loaded._last_state_hash == next(iter(state.states.values()))
    assert ReuseValue.restores == 1


def test_matching_ignores_stale_candidates_after_exact_hash_filtering(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    saved = ReuseValue(1, repo=repo)
    state = repo.save_object(saved)
    stale = repo.load_state_ref(state, reuse_live="never")
    stale._last_state_hash = "pkl-" + "0" * 64
    ReuseValue.restores = 0

    loaded = repo.load_state_ref(state, reuse_live="matching")

    assert loaded is saved
    assert ReuseValue.restores == 0


@pytest.mark.parametrize("reuse_live", ["matching", "greedy"])
def test_ambiguous_eligible_candidates_build_fresh(tmp_path, reuse_live):
    repo = Repo(DirStore(tmp_path / "store"))
    saved = ReuseValue(1, repo=repo)
    state = repo.save_object(saved)
    duplicate = repo.load_state_ref(state, reuse_live="never")
    if reuse_live == "greedy":
        saved._last_state_hash = duplicate._last_state_hash = "pkl-" + "0" * 64
    ReuseValue.restores = 0

    loaded = repo.load_state_ref(state, reuse_live=reuse_live)

    assert loaded is not saved
    assert loaded is not duplicate
    assert ReuseValue.restores == 1


class ReusePair(Object):
    def __init__(self, first, second):
        self.first = first
        self.second = second


def test_never_preserves_independent_nodes_with_equal_state_hashes(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    first = ReuseValue(1, repo=repo)
    second = ReuseValue(1, repo=repo)
    state = repo.save_object(ReusePair(first, second, repo=repo))

    loaded = Repo(DirStore(tmp_path / "store")).load_state_ref(state, reuse_live="never")

    assert len(set(state.states.values())) == 1
    assert loaded.first is not loaded.second
    assert loaded.first.object_id != loaded.second.object_id


class FailingGreedyValue(ReuseValue):
    def restore_state_from_dir_imp(self, src_dir, *, codec):
        raise RuntimeError("restore rejected")


def test_failed_greedy_restore_evicts_and_clears_the_visible_candidate(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    saved = FailingGreedyValue(1, repo=repo)
    state = repo.save_object(saved)
    saved._last_state_hash = "pkl-" + "0" * 64

    with pytest.raises(RepoLoadError, match="Exact restore"):
        repo.load_state_ref(state, reuse_live="greedy")

    assert saved._last_state_hash is None
    assert saved not in repo._all_live_candidates()


class MultiFailureValue(ReuseValue):
    restore_calls = 0
    fail_second_restore = False

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        type(self).restore_calls += 1
        if type(self).fail_second_restore and type(self).restore_calls == 2:
            raise RuntimeError("second restore rejected")
        super().restore_state_from_dir_imp(src_dir, codec=codec)


def test_later_greedy_failure_evicts_every_previously_mutated_candidate(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    first = MultiFailureValue(1, repo=repo)
    second = MultiFailureValue(2, repo=repo)
    state = repo.save_object(ReusePair(first, second, repo=repo))
    repo._evict_live(first)
    repo._evict_live(second)
    candidates = repo.load_state_ref(state, reuse_live="never")
    candidates.first._last_state_hash = "pkl-" + "0" * 64
    candidates.second._last_state_hash = "pkl-" + "0" * 64
    MultiFailureValue.restore_calls = 0
    MultiFailureValue.fail_second_restore = True

    try:
        with pytest.raises(RepoLoadError, match="mutated and evicted candidates") as error:
            repo.load_state_ref(state, reuse_live="greedy")
    finally:
        MultiFailureValue.fail_second_restore = False

    assert str(next(iter(state.states))) in str(error.value)
    assert candidates.first._last_state_hash is None
    assert candidates.second._last_state_hash is None
    assert candidates.first not in repo._all_live_candidates()
    assert candidates.second not in repo._all_live_candidates()
