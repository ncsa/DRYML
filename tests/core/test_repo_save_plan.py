from dryml.core import Object, Repo, Serializable
from dryml.core.repo_plan import build_save_plan
from dryml.core.store.dir import DirStore


class SavePlanLeaf(Serializable):
    def __init__(self, name):
        self.name = name

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        pass


class SavePlanNode(Object):
    def __init__(self, child):
        self.child = child


def test_save_plan_uses_retained_primary_bindings_once(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    child = SavePlanLeaf("child", repo=repo)
    parent = SavePlanNode([child, child], repo=repo)

    plan = build_save_plan(repo, parent, store=store)

    assert len(plan.actions) == 1
    assert plan.actions[0].obj is child
    assert plan.actions[0].path in parent.object_ref.objects


def test_save_plan_keeps_independent_equal_nodes_distinct(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    first = SavePlanLeaf("same", repo=repo)
    second = SavePlanLeaf("same", repo=repo)
    parent = SavePlanNode([first, second], repo=repo)

    plan = build_save_plan(repo, parent, store=store)

    assert {action.obj for action in plan.actions} == {first, second}
