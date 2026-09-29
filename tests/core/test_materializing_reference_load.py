from pathlib import Path

import pytest

from dryml.core import Definition, Object, Repo, Serializable
from dryml.core.cdef_graph import EdgeKind
from dryml.core.links import DefLink
from dryml.core.repo import RepoLoadError
from dryml.core.store.dir import DirStore

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")


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


class ExactOrderSeed(SeedValue):
    """Serializable exact-load child that records dependency construction order."""

    events = []

    def __init__(self, value):
        type(self).events.append(value)
        super().__init__(value)


class ExactOrderParent(Object):
    """Shared-child root retaining a reference-only definition as inert data."""

    def __init__(self, left, right, ref):
        assert left is right
        ExactOrderSeed.events.append("parent")
        self.left = left
        self.right = right
        self.ref = ref


def test_exact_load_resolves_materializing_state_ref_before_parent(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    child = SeedValue(1, repo=repo)
    child.value = 7
    seed = repo.save_object(child)
    parent = SeedParent(seed, repo=repo)
    state = repo.save_object(parent)

    loaded = Repo(DirStore(tmp_path / "store")).load_state_ref(state, reuse_live="never")

    assert loaded.child.value == 7


@pytest.mark.parametrize("reuse_live", ["matching", "greedy", "never"])
def test_enclosing_state_overrides_materializing_seed_state(tmp_path, reuse_live):
    repo = Repo(DirStore(tmp_path / "store"))
    child = SeedValue(1, repo=repo)
    child.value = 7
    seed = repo.save_object(child)
    parent = SeedParent(seed, repo=repo)
    parent.child.value = 9
    state = repo.save_object(parent, deep_capture=True)

    loaded = Repo(DirStore(tmp_path / "store")).load_state_ref(
        state, reuse_live=reuse_live
    )

    assert loaded.child.value == 9


def test_exact_load_restores_a_repeated_materializing_state_ref_once(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    child = SeedValue(1, repo=repo)
    child.value = 7
    seed = repo.save_object(child)
    state = repo.save_object(SeedParent([seed, seed], repo=repo))

    loaded = Repo(DirStore(tmp_path / "store")).load_state_ref(state, reuse_live="never")

    assert loaded.child[0] is loaded.child[1]
    assert loaded.child[0].value == 7


def test_exact_load_orders_shared_dependencies_before_parent_and_keeps_ref_terminal(tmp_path):
    """Occurrence reversal remains dependency-first without materializing REF targets."""

    repo = Repo(DirStore(tmp_path / "store"))
    child = ExactOrderSeed(7, repo=repo)
    hidden = Definition(ExactOrderSeed, 8).concretize(repo=repo)
    state = repo.save_object(ExactOrderParent(
        child,
        child,
        DefLink.finalized(EdgeKind.REF, hidden),
        repo=repo,
    ))
    ExactOrderSeed.events.clear()

    loaded = Repo(DirStore(tmp_path / "store")).load_state_ref(state, reuse_live="never")

    assert ExactOrderSeed.events == [7, "parent"]
    assert loaded.left is loaded.right
    assert loaded.ref == hidden


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
