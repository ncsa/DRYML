import pytest

from dryml.core import ConcreteDefinition, Definition, Object, Repo, Serializable
from dryml.core.freeze import FrozenDict, FrozenTuple
from dryml.core.materialization import MaterializationAction, build_materialization_plan, execute_materialization_plan, from_canonical_local
from dryml.core.policies import RepoLoadOptions
from dryml.core.repo import RepoLoadError
from dryml.core.store.dir import DirStore


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


class MaterialChainNode(Object):
    constructed = []

    def __init__(self, name, child=None, ref=None):
        super().__init__()
        self.name = name
        self.child = child
        self.ref = ref
        type(self).constructed.append(name)


class MaterialSerializable(Serializable):
    def __init__(self, name):
        super().__init__()
        self.name = name


class BadRestoreSerializable(Serializable):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def save_state_to_dir_imp(self, dest_dir, revision=None):
        pass

    def restore_state_from_dir_imp(self, src_dir, revision=None):
        raise RuntimeError("restore boom")


class FailingMaterial(Object):
    def __init__(self, child=None):
        super().__init__()
        self.child = child
        raise RuntimeError("construct boom")


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


def test_materialization_action_captures_realization_policy(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = MaterialSerializable("stored", repo=repo)
    repo.save_object(obj)
    cdef = obj.definition
    repo.clear_cache(strong=True, weak=True)

    plan = build_materialization_plan(
        repo,
        cdef,
        RepoLoadOptions(restore_state=True, build_missing=False, cache="strong"),
        revision={cdef: "requested"},
        memo={},
        path=[""],
    )

    action = plan.actions[cdef]
    assert action.kind == "construct"
    assert action.restore_state is True
    assert action.store is store
    assert action.revision == "requested"
    assert action.build_missing is False
    assert action.cache == "strong"


def test_executor_uses_planned_store_for_cached_reuse(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = MaterialSerializable("cached", repo=repo)
    repo.save_object(obj)
    repo.pin(obj)
    cdef = obj.definition
    plan = build_materialization_plan(
        repo,
        cdef,
        RepoLoadOptions(restore_state=True),
        memo={},
        path=[""],
    )

    def fail_store_lookup(_):
        raise AssertionError("executor should use MaterializationAction.store")

    monkeypatch.setattr(repo, "_first_store_with", fail_store_lookup)

    assert execute_materialization_plan(repo, plan, memo={}, revision={}, root=cdef) is obj


def test_materialization_shared_child_constructed_once():
    repo = Repo()
    child_def = Definition(MaterialLeaf, "shared")
    parent_def = Definition(MaterialParent, child_def, right=child_def)
    MaterialLeaf.constructed.clear()

    parent = repo.load_or_build(parent_def)

    assert parent.left is parent.right
    assert MaterialLeaf.constructed.count("shared") == 1
    assert repo._num_constructions == 2


def test_materialization_stops_at_ref_edge_before_materialized_subgraph():
    repo = Repo()
    d_def = Definition(MaterialChainNode, "D")
    c_def = Definition(MaterialChainNode, "C", ref=d_def.ref())
    b_def = Definition(MaterialChainNode, "B", child=c_def.mat())
    a_def = Definition(MaterialChainNode, "A", ref=b_def.ref())
    a_cdef = a_def.concretize(repo=repo)
    b_cdef = a_cdef.kwargs["ref"].target

    MaterialChainNode.constructed.clear()
    plan = build_materialization_plan(
        repo,
        a_cdef,
        RepoLoadOptions(build_missing=True),
        memo={},
        path=[""],
    )
    obj = repo.load_or_build(a_cdef)

    assert plan.order == (a_cdef,)
    assert MaterialChainNode.constructed == ["A"]
    assert obj.name == "A"
    assert obj.ref == b_cdef
    assert isinstance(obj.ref, ConcreteDefinition)
    assert repo._num_constructions == 1


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


def test_restore_failure_does_not_pollute_memo(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    obj = BadRestoreSerializable("bad", repo=repo)
    repo.save_object(obj)
    cdef = obj.definition
    repo.clear_cache(strong=True, weak=True)
    memo = {}

    with pytest.raises(RepoLoadError, match="Error restoring state"):
        repo._materialize_cdef(cdef, memo=memo)

    assert cdef not in memo
    assert cdef not in repo.strong_obj_cache
    assert cdef not in repo.weak_obj_cache


def test_constructor_failure_does_not_publish_cache_entry():
    repo = Repo()
    cdef = Definition(FailingMaterial).concretize(repo=repo)

    with pytest.raises(RepoLoadError, match="Error constructing"):
        repo.load_object(cdef, build_missing=True, cache="strong")

    assert cdef not in repo.strong_obj_cache
    assert cdef not in repo.weak_obj_cache


def test_cached_reuse_does_not_resolve_backend_class(monkeypatch):
    from dryml.core import materialization as materialization_mod

    repo = Repo()
    obj = MaterialLeaf("cached", repo=repo)
    repo.pin(obj)

    def fail_resolve(_):
        raise AssertionError("resolve_symbol should not be called for cached reuse")

    monkeypatch.setattr(materialization_mod, "resolve_symbol", fail_resolve)

    assert repo.load_object(obj.definition, restore_state=False) is obj


def test_executor_honors_materialization_action_kind():
    repo = Repo()
    cdef = Definition(MaterialLeaf, "planned").concretize(repo=repo)
    memo = {}
    plan = build_materialization_plan(
        repo,
        cdef,
        RepoLoadOptions(build_missing=True),
        memo=memo,
        path=[""],
    )
    plan.actions[cdef] = MaterializationAction(cdef, "reuse", "$", reuse_source="cache")

    with pytest.raises(RepoLoadError, match="cached reuse"):
        execute_materialization_plan(repo, plan, memo=memo, revision={}, root=cdef)


def test_parent_failure_leaves_successful_child_cached():
    repo = Repo()
    child_def = Definition(MaterialLeaf, "child")
    parent_def = Definition(FailingMaterial, child_def).concretize(repo=repo)
    child_cdef = parent_def.args[0]

    with pytest.raises(RepoLoadError, match="Error constructing"):
        repo.load_object(parent_def, build_missing=True, cache="strong")

    assert child_cdef in repo.strong_obj_cache
    assert parent_def not in repo.strong_obj_cache


def test_from_canonical_local_uses_shared_decoder_resolver():
    repo = Repo()
    child = MaterialLeaf("child")
    replacement = object()
    value = FrozenDict({"child": child.definition, "items": FrozenTuple((child.definition,))})

    decoded = from_canonical_local(value, resolve_cdef=lambda cdef: replacement, repo=repo)

    assert decoded == {"child": replacement, "items": (replacement,)}
