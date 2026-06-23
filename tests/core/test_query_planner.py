from dryml.core2 import Definition, Object, Repo, SKIP_ARGS
from dryml.core2.definition import selector_match
from dryml.core2.query.query import _exact_constraints_match
from dryml.core2.query.fingerprint import collect_exact_constraints


class PlanBase(Object):
    def __init__(self, child=None, *, name="base"):
        super().__init__()
        self.child = child
        self.name = name


class PlanSub(PlanBase):
    pass


class PlanRoot(Object):
    def __init__(self, child=None, *, name="root"):
        super().__init__()
        self.child = child
        self.name = name


def brute_force(selector, candidates):
    constraints = collect_exact_constraints(selector) if selector is not None else ()
    out = []
    for cdef in candidates:
        if selector is None or selector_match(selector, cdef, strict=False):
            if _exact_constraints_match(cdef, constraints):
                out.append(cdef)
    return sorted(out, key=lambda cdef: (cdef.stable_hash(), repr(cdef)))


def test_indexed_results_equal_brute_force_corpus():
    repo = Repo()
    objs = [
        PlanRoot(PlanBase(name="a"), name="one", repo=repo),
        PlanRoot(PlanBase(name="b"), name="two", repo=repo),
        PlanRoot(PlanSub(name="a"), name="three", repo=repo),
        PlanBase(name="standalone", repo=repo),
    ]
    repo.add_objects(*objs)
    candidates = tuple(repo.strong_obj_cache.keys())
    selector = Definition(PlanRoot, SKIP_ARGS, child=Definition(PlanBase, SKIP_ARGS, name="a"))

    indexed = list(repo.query(selector).known(refresh=False).defs())

    assert indexed == brute_force(selector, candidates)


def test_empty_feature_selector_falls_back_to_full_universe():
    repo = Repo()
    for idx in range(5):
        repo.add_objects(PlanBase(name=f"n{idx}", repo=repo))

    explanation = repo.query(Definition(PlanBase, lambda child: True)).known(refresh=False).explain()

    assert explanation.universe_size == 5
    assert explanation.candidate_count == 5
    assert explanation.verified_count == 5


def test_selective_fingerprint_query_verifies_fewer_candidates_than_domain():
    repo = Repo()
    for idx in range(10):
        repo.add_objects(PlanBase(name=f"n{idx}", repo=repo))

    result = repo.query(Definition(PlanBase, SKIP_ARGS, name="n3")).known(refresh=False).defs()

    assert result.count() == 1
    assert result.explanation.universe_size == 10
    assert result.explanation.verified_count < result.explanation.universe_size


def test_refine_exact_class_matches_one_shot_query():
    repo = Repo()
    exact_child = PlanBase(name="child", repo=repo)
    sub_child = PlanSub(name="child", repo=repo)
    exact_parent = PlanRoot(child=exact_child, repo=repo)
    sub_parent = PlanRoot(child=sub_child, repo=repo)
    repo.add_objects(exact_parent, sub_parent)

    selector = Definition(PlanRoot, SKIP_ARGS, child=Definition(PlanBase, SKIP_ARGS))
    one_shot = repo.query(selector).class_match("exact").known(refresh=False).defs()
    broad = repo.query(Definition(PlanRoot, SKIP_ARGS)).known(refresh=False).defs()
    refined = broad.query(selector).class_match("exact").defs()

    assert list(one_shot) == [exact_parent.definition]
    assert list(refined) == list(one_shot)
