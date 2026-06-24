from dryml.core2 import Definition, Object, Repo, SKIP_ARGS
from dryml.core2.store.dir import DirStore


class PlannerLeaf(Object):
    def __init__(self, name="leaf"):
        super().__init__()
        self.name = name


class PlannerParent(Object):
    def __init__(self, child=None, *, name="parent"):
        super().__init__()
        self.child = child
        self.name = name


def test_planner_can_choose_nested_exact_anchor():
    repo = Repo()
    rare = PlannerLeaf("rare", repo=repo)
    owners = []
    for idx in range(8):
        child = rare if idx in {2, 5} else PlannerLeaf(f"common-{idx}", repo=repo)
        owners.append(PlannerParent(child=child, name=f"owner-{idx}", repo=repo))
    repo.add_objects(*owners)

    selector = Definition(PlannerParent, SKIP_ARGS, child=rare.definition)
    results = repo.query(selector).known(refresh=False).defs()

    assert list(results) == sorted([owners[2].definition, owners[5].definition], key=lambda cdef: (cdef.stable_hash(), repr(cdef)))
    assert str(results.explanation.graph_anchor_path) == "$.child"
    assert results.explanation.graph_anchor_mode == "exact"
    assert results.explanation.verified_count == 2


def test_planner_propagates_child_scalar_candidates_to_parent():
    repo = Repo()
    wanted = PlannerParent(child=PlannerLeaf(name="wanted", repo=repo), repo=repo)
    for idx in range(6):
        repo.add_objects(PlannerParent(child=PlannerLeaf(name=f"other-{idx}", repo=repo), repo=repo))
    repo.add_objects(wanted)

    selector = Definition(PlannerParent, SKIP_ARGS, child=Definition(PlannerLeaf, SKIP_ARGS, name="wanted"))
    results = repo.query(selector).known(refresh=False).defs()

    assert list(results) == [wanted.definition]
    assert results.explanation.graph_candidate_count == 1
    assert results.explanation.verified_count == 1


def test_nested_owner_query_uses_graph_edges_without_materializing(tmp_path):
    repo = Repo(stores=DirStore(tmp_path / "store"))
    child = PlannerLeaf(name="child", repo=repo)
    parent = PlannerParent(child=child, repo=repo)
    repo.save_object(parent)

    owners = repo.query(Definition(PlannerLeaf, SKIP_ARGS, name="child")).nested(refresh=False).owners().defs()

    assert list(owners) == [parent.definition]
    assert repo._num_constructions == 0
