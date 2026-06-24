import pytest

from dryml.core2 import Definition, Object, Repo, Serializable
from dryml.core2.materialization import build_materialization_plan
from dryml.core2.policies import RepoLoadOptions
from dryml.core2.repo import RepoLoadError


class MaterialLeaf(Object):
    constructed = []

    def __init__(self, name):
        super().__init__()
        self.name = name
        type(self).constructed.append(name)


class MaterialParent(Object):
    def __init__(self, left, right=None):
        super().__init__()
        self.left = left
        self.right = right


class MaterialSerializable(Serializable):
    def __init__(self, name):
        super().__init__()
        self.name = name


def test_materialization_plan_does_not_construct():
    repo = Repo()
    cdef = Definition(MaterialLeaf, "planned").concretize(repo=repo)
    MaterialLeaf.constructed.clear()

    plan = build_materialization_plan(
        repo,
        cdef,
        RepoLoadOptions(build_missing=True),
        memo={},
        path=[""],
    )

    assert plan.order == (cdef,)
    assert MaterialLeaf.constructed == []
    assert repo._num_constructions == 0


def test_materialization_shared_child_constructed_once():
    repo = Repo()
    child_def = Definition(MaterialLeaf, "shared")
    parent_def = Definition(MaterialParent, child_def, right=child_def)
    MaterialLeaf.constructed.clear()

    parent = repo.load_or_build(parent_def)

    assert parent.left is parent.right
    assert MaterialLeaf.constructed.count("shared") == 1
    assert repo._num_constructions == 2


def test_materialization_new_still_shares_within_one_pass():
    repo = Repo()
    child_def = Definition(MaterialLeaf, "new-shared")
    parent_def = Definition(MaterialParent, child_def, right=child_def).concretize(repo=repo)

    parent = repo.load_object(parent_def, instance="new", cache="none", build_missing=True)

    assert parent.left is parent.right
    assert parent.definition not in repo.strong_obj_cache


def test_materialization_new_requires_cache_none():
    repo = Repo()
    cdef = Definition(MaterialLeaf, "x").concretize(repo=repo)

    with pytest.raises(ValueError, match="cache='none'"):
        repo.load_object(cdef, instance="new", cache="weak", build_missing=True)


def test_cached_parent_restore_state_false_prunes_missing_child():
    repo = Repo()
    child = MaterialSerializable("missing", repo=repo)
    parent = MaterialParent(child, repo=repo)
    repo.pin(parent)
    repo.weak_obj_cache.pop(child.definition, None)
    repo.strong_obj_cache.pop(child.definition, None)

    assert repo.load_object(parent.definition, restore_state=False) is parent


def test_cached_parent_restore_state_true_processes_missing_child():
    repo = Repo()
    child = MaterialSerializable("missing", repo=repo)
    parent = MaterialParent(child, repo=repo)
    repo.pin(parent)
    repo.weak_obj_cache.pop(child.definition, None)
    repo.strong_obj_cache.pop(child.definition, None)

    with pytest.raises(RepoLoadError, match="Missing stored state"):
        repo.load_object(parent.definition, restore_state=True, build_missing=False)
