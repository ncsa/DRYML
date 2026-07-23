from dryml.core import Object, Repo, Serializable
from dryml.core.repo_plan import build_save_plan, execute_save_plan
from dryml.core.store.dir import DirStore
from dryml.core.utils.general import pickle_save


class SavePlanLeaf(Object):
    def __init__(self, name):
        super().__init__()
        self.name = name


class SavePlanNode(Object):
    def __init__(self, child, *, extra=None):
        super().__init__()
        self.child = child
        self.extra = extra


class SavePlanSerializable(Serializable):
    def __init__(self, child=None, *, name="serial"):
        super().__init__()
        self.child = child
        self.name = name

    def save_state_to_dir_imp(self, dest_dir, revision=None):
        pickle_save(self.name, f"{dest_dir}/name.pkl")


def test_save_plan_order_is_child_before_parent(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = SavePlanSerializable(name="child", repo=repo)
    parent = SavePlanSerializable(child, name="parent", repo=repo)
    repo.add_objects(parent)

    plan = build_save_plan(repo, parent, store=store, revision={}, ephemeral_depth=0)

    assert [action.definition for action in plan.actions] == [child.definition, parent.definition]


def test_save_plan_ephemeral_depth_uses_cdef_edges_not_container_depth(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = SavePlanLeaf("child", repo=repo)
    parent = SavePlanNode([[child]], repo=repo)
    repo.add_objects(parent)

    depth_zero = build_save_plan(repo, parent, store=store, revision={}, ephemeral_depth=0)
    depth_one = build_save_plan(repo, parent, store=store, revision={}, ephemeral_depth=1)

    assert [action.definition for action in depth_zero.actions] == [parent.definition]
    assert [action.definition for action in depth_one.actions] == [child.definition, parent.definition]


def test_save_plan_shared_child_saved_once_per_store(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = SavePlanSerializable(name="child", repo=repo)
    parent = SavePlanSerializable([child, child], name="parent", repo=repo)
    repo.add_objects(parent)

    plan = build_save_plan(repo, parent, store=store, revision={}, ephemeral_depth=0)
    execute_save_plan(repo, plan)

    assert [action.definition for action in plan.actions].count(child.definition) == 1
    assert store.has(child.definition)
    assert store.has(parent.definition)


def test_save_execution_registers_graph_structure_once(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    child = SavePlanSerializable(name="child", repo=repo)
    parent = SavePlanSerializable(child, name="parent", repo=repo)
    repo.add_objects(parent)
    calls = []
    original = repo._query_catalog.register_graph

    def spy_register_graph(graph):
        calls.append(graph)
        return original(graph)

    monkeypatch.setattr(repo._query_catalog, "register_graph", spy_register_graph)

    plan = build_save_plan(repo, parent, store=store, revision={}, ephemeral_depth=0)
    execute_save_plan(repo, plan)

    assert calls == [plan.graph]


def test_save_plan_explicit_store_wins(tmp_path):
    default_store = DirStore(tmp_path / "default")
    explicit_store = DirStore(tmp_path / "explicit")
    repo = Repo(stores=default_store)
    obj = SavePlanLeaf("root", repo=repo)
    repo.add_objects(obj)

    plan = build_save_plan(repo, obj, store=explicit_store, revision={}, ephemeral_depth=0)

    assert plan.actions[0].store is explicit_store
