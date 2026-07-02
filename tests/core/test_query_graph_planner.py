from collections.abc import Mapping

from dryml.core2 import Definition, Object, Repo, SKIP_ARGS, Satisfies
from dryml.core2.definition import ConcreteDefinition
from dryml.core2.query.graph_plan import graph_candidate_ids
from dryml.core2.query.index import MemoryDefinitionGraphReadView
from dryml.core2.query.model import FeatureRequirement, FeatureToken, QueryStats
from dryml.core2.query.path import DefinitionPath, Kwarg
from dryml.core2.query.query import _query_match
from dryml.core2.query.selector_graph import SelectorGraph, SelectorGraphEdge, SelectorGraphNode
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


class PlannerSetParent(Object):
    def __init__(self, members=None, *, name="set-parent"):
        super().__init__()
        self.members = set() if members is None else members
        self.name = name


def brute_force_known(repo, selector):
    return sorted(
        [cdef for cdef in repo.strong_obj_cache if _query_match(selector, cdef, strict=False, class_match="selector")],
        key=lambda cdef: (cdef.stable_hash(), repr(cdef)),
    )


def nested_cdefs_by_owner(owner):
    out = []
    seen = set()

    def visit(value):
        if isinstance(value, ConcreteDefinition):
            if value in seen:
                return
            seen.add(value)
            out.append(value)
            visit(value.args)
            visit(value.kwargs)
            return
        if isinstance(value, Mapping):
            for child in value.values():
                visit(child)
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            for child in value:
                visit(child)

    visit(owner.args)
    visit(owner.kwargs)
    return tuple(out)


def brute_force_nested_definitions(owners, selector):
    matches = {
        nested
        for owner in owners
        for nested in nested_cdefs_by_owner(owner)
        if _query_match(selector, nested, strict=False, class_match="selector")
    }
    return sorted(matches, key=lambda cdef: (cdef.stable_hash(), repr(cdef)))


def brute_force_owners(owners, selector):
    matches = {
        owner
        for owner in owners
        if any(_query_match(selector, nested, strict=False, class_match="selector") for nested in nested_cdefs_by_owner(owner))
    }
    return sorted(matches, key=lambda cdef: (cdef.stable_hash(), repr(cdef)))


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


def test_nested_exact_anchor_does_not_enumerate_full_root_domain(monkeypatch):
    repo = Repo()
    rare = PlannerLeaf("rare", repo=repo)
    owners = []
    for idx in range(12):
        child = rare if idx in {3, 9} else PlannerLeaf(f"common-{idx}", repo=repo)
        owners.append(PlannerParent(child=child, name=f"owner-{idx}", repo=repo))
    repo.add_objects(*owners)

    catalog = repo._query_catalog
    full_domain_size = len(catalog.definitions_by_id)
    local_candidate_universe_sizes = []
    original = MemoryDefinitionGraphReadView.local_candidates

    def spy_local_candidates(self, requirements, *, within=None, domain=None, stats=None):
        if within is not None:
            local_candidate_universe_sizes.append(len(within))
        return original(self, requirements, within=within, domain=domain, stats=stats)

    monkeypatch.setattr(MemoryDefinitionGraphReadView, "local_candidates", spy_local_candidates)

    selector = Definition(PlannerParent, SKIP_ARGS, child=rare.definition)
    results = repo.query(selector).known(refresh=False).defs()

    assert list(results) == sorted([owners[3].definition, owners[9].definition], key=lambda cdef: (cdef.stable_hash(), repr(cdef)))
    assert str(results.explanation.graph_anchor_path) == "$.child"
    assert local_candidate_universe_sizes
    assert max(local_candidate_universe_sizes) < full_domain_size


def test_exact_root_query_does_not_construct_stored_universe(tmp_path, monkeypatch):
    repo = Repo(stores=DirStore(tmp_path / "store"))
    obj = PlannerLeaf("exact", repo=repo)
    repo.save_object(obj)

    def fail_stored_ids():
        raise AssertionError("all_stored_ids should not be called for exact-root queries")

    monkeypatch.setattr(repo._query_catalog, "all_stored_ids", fail_stored_ids)

    results = repo.query(obj.definition).stored(refresh=False).defs()

    assert list(results) == [obj.definition]


def test_selective_stored_query_never_constructs_stored_id_set(tmp_path, monkeypatch):
    repo = Repo(stores=DirStore(tmp_path / "store"))
    wanted = PlannerParent(child=PlannerLeaf(name="wanted", repo=repo), repo=repo)
    repo.save_object(wanted)
    for idx in range(6):
        repo.save_object(PlannerParent(child=PlannerLeaf(name=f"other-{idx}", repo=repo), repo=repo))

    monkeypatch.setattr(
        repo._query_catalog,
        "all_stored_ids",
        lambda: (_ for _ in ()).throw(AssertionError("eager stored universe")),
    )
    monkeypatch.setattr(
        repo._query_catalog,
        "stored_count",
        lambda: (_ for _ in ()).throw(AssertionError("eager stored count")),
    )

    selector = Definition(PlannerParent, SKIP_ARGS, child=Definition(PlannerLeaf, SKIP_ARGS, name="wanted"))
    results = repo.query(selector).stored(refresh=False).defs()

    assert list(results) == [wanted.definition]
    assert results.explanation.universe_size is None


def test_selective_known_query_never_constructs_known_id_set(monkeypatch):
    repo = Repo()
    wanted = PlannerParent(child=PlannerLeaf(name="wanted", repo=repo), repo=repo)
    repo.add_objects(wanted)
    for idx in range(6):
        repo.add_objects(PlannerParent(child=PlannerLeaf(name=f"other-{idx}", repo=repo), repo=repo))

    monkeypatch.setattr(
        repo._query_catalog,
        "all_known_ids",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("eager known universe")),
    )
    monkeypatch.setattr(
        repo._query_catalog,
        "known_count",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("eager known count")),
    )

    selector = Definition(PlannerParent, SKIP_ARGS, child=Definition(PlannerLeaf, SKIP_ARGS, name="wanted"))
    results = repo.query(selector).known(refresh=False).defs()

    assert list(results) == [wanted.definition]
    assert results.explanation.universe_size is None


def test_selective_cached_query_never_constructs_cached_id_set(monkeypatch):
    repo = Repo()
    wanted = PlannerParent(child=PlannerLeaf(name="wanted", repo=repo), repo=repo)
    repo.add_objects(wanted)
    for idx in range(6):
        repo.add_objects(PlannerParent(child=PlannerLeaf(name=f"other-{idx}", repo=repo), repo=repo))

    monkeypatch.setattr(
        repo._query_catalog,
        "all_cached_ids",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("eager cached universe")),
    )
    monkeypatch.setattr(
        repo._query_catalog,
        "cached_count",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("eager cached count")),
    )

    selector = Definition(PlannerParent, SKIP_ARGS, child=Definition(PlannerLeaf, SKIP_ARGS, name="wanted"))
    results = repo.query(selector).cached(refresh=False).defs()

    assert list(results) == [wanted.definition]
    assert results.explanation.universe_size is None


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


def test_nonexact_nested_anchor_does_not_enumerate_all_definition_ids(monkeypatch):
    repo = Repo()
    wanted = PlannerParent(child=PlannerLeaf(name="wanted", repo=repo), repo=repo)
    for idx in range(20):
        repo.add_objects(PlannerParent(child=PlannerLeaf(name=f"other-{idx}", repo=repo), repo=repo))
    repo.add_objects(wanted)

    catalog = repo._query_catalog
    full_domain_size = len(catalog.definitions_by_id)
    bounded_universe_sizes = []
    original = MemoryDefinitionGraphReadView.local_candidates

    def spy_local_candidates(self, requirements, *, within=None, domain=None, stats=None):
        if within is not None:
            bounded_universe_sizes.append(len(within))
        return original(self, requirements, within=within, domain=domain, stats=stats)

    monkeypatch.setattr(MemoryDefinitionGraphReadView, "local_candidates", spy_local_candidates)

    selector = Definition(PlannerParent, SKIP_ARGS, child=Definition(PlannerLeaf, SKIP_ARGS, name="wanted"))
    results = repo.query(selector).known(refresh=False).defs()

    assert list(results) == [wanted.definition]
    assert results.explanation.graph_anchor_mode == "local-posting"
    assert bounded_universe_sizes
    assert max(bounded_universe_sizes) < full_domain_size


def test_anchor_selection_estimates_every_node_and_materializes_only_chosen(monkeypatch):
    repo = Repo()
    wanted = PlannerParent(child=PlannerLeaf(name="wanted", repo=repo), name="wanted-root", repo=repo)
    for idx in range(8):
        repo.add_objects(PlannerParent(child=PlannerLeaf(name=f"other-{idx}", repo=repo), name="common-root", repo=repo))
    repo.add_objects(wanted)

    catalog = repo._query_catalog
    estimate_calls = []
    materialize_calls = []
    original_estimate = MemoryDefinitionGraphReadView.estimate_local_candidates
    original_materialize = MemoryDefinitionGraphReadView.local_candidates

    def spy_estimate(self, requirements):
        estimate_calls.append(requirements)
        if len(estimate_calls) == 1:
            return 1000
        return original_estimate(self, requirements)

    def spy_materialize(self, requirements, *, within=None, domain=None, stats=None):
        if within is None and domain is None:
            materialize_calls.append(requirements)
        return original_materialize(self, requirements, within=within, domain=domain, stats=stats)

    monkeypatch.setattr(MemoryDefinitionGraphReadView, "estimate_local_candidates", spy_estimate)
    monkeypatch.setattr(MemoryDefinitionGraphReadView, "local_candidates", spy_materialize)

    selector = Definition(
        PlannerParent,
        SKIP_ARGS,
        child=Definition(PlannerLeaf, SKIP_ARGS, name="wanted"),
        name="wanted-root",
    )
    results = repo.query(selector).known(refresh=False).defs()

    assert list(results) == [wanted.definition]
    assert len(estimate_calls) >= 2
    assert len(materialize_calls) == 1


def test_planner_runs_against_backend_without_lock_or_internal_dicts():
    root_req = (FeatureRequirement(FeatureToken("ROOT", DefinitionPath(), "root")),)
    child_req = (FeatureRequirement(FeatureToken("CHILD", DefinitionPath(), "child")),)
    selector_graph = SelectorGraph(
        root=0,
        nodes=(
            SelectorGraphNode(0, DefinitionPath(), Definition(PlannerParent, SKIP_ARGS), root_req),
            SelectorGraphNode(1, DefinitionPath((Kwarg("child"),)), Definition(PlannerLeaf, SKIP_ARGS), child_req),
        ),
        edges=(SelectorGraphEdge(0, DefinitionPath((Kwarg("child"),)), 1),),
    )

    class FakeIndex:
        def __init__(self):
            self.materialized = []

        def all_definition_ids(self):
            raise AssertionError("planner should not request all IDs when an anchor exists")

        def estimate_exact_ids(self, cdef):
            return 0

        def estimate_local_candidates(self, requirements):
            return 10 if requirements == root_req else 1

        def exact_ids(self, cdef):
            return set()

        def local_candidates(self, requirements, *, within=None, domain=None, stats=None):
            if within is None and domain is None:
                self.materialized.append(requirements)
            candidates = {"child"} if requirements == child_req else {"root"}
            if within is not None:
                candidates &= set(within)
            return candidates

        def parents(self, child_ids, path, *, unordered, within=None):
            out = {"root"} if "child" in child_ids else set()
            return out if within is None else out & set(within)

        def children(self, parent_ids, path, *, unordered, within=None):
            out = {"child"} if "root" in parent_ids else set()
            return out if within is None else out & set(within)

    index = FakeIndex()
    stats = QueryStats()

    assert graph_candidate_ids(index, selector_graph, None, stats=stats) == {"root"}
    assert index.materialized == [child_req]
    assert stats.graph_anchor_mode == "local-posting"


def test_graph_planner_matches_independent_bruteforce_matrix():
    repo = Repo()
    shared = PlannerLeaf("shared", repo=repo)
    rare = PlannerLeaf("rare", repo=repo)
    objects = [
        PlannerParent(child=shared, name="shared-a", repo=repo),
        PlannerParent(child=shared, name="shared-b", repo=repo),
        PlannerParent(child=rare, name="rare", repo=repo),
        PlannerParent(child=[shared, shared], name="repeated", repo=repo),
        PlannerSetParent(members={rare}, name="set", repo=repo),
    ]
    repo.add_objects(*objects)
    selectors = [
        Definition(PlannerParent, SKIP_ARGS, child=Definition(PlannerLeaf, "shared")),
        Definition(PlannerParent, SKIP_ARGS, child=rare.definition),
        Definition(PlannerParent, SKIP_ARGS, child=[shared.definition, shared.definition]),
        Definition(PlannerSetParent, SKIP_ARGS, members={rare.definition}),
        Definition(PlannerParent, SKIP_ARGS, name=Satisfies(lambda value: value.startswith("shared"), name="starts-shared")),
    ]

    for selector in selectors:
        assert list(repo.query(selector).known(refresh=False).defs()) == brute_force_known(repo, selector)


def test_graph_query_matches_bruteforce_across_domains_and_replicas(tmp_path):
    store1 = DirStore(tmp_path / "store1")
    store2 = DirStore(tmp_path / "store2")
    repo = Repo(stores=[store1, store2])
    shared = PlannerLeaf("shared", repo=repo)
    rare = PlannerLeaf("rare", repo=repo)
    cached_only = PlannerParent(child=shared, name="cached-only", repo=repo)
    stored_a = PlannerParent(child=shared, name="stored-a", repo=repo)
    stored_b = PlannerParent(child=rare, name="stored-b", repo=repo)
    stored_set = PlannerSetParent(members={rare}, name="stored-set", repo=repo)
    repo.add_objects(cached_only)
    repo.save_object(stored_a, store=store1)
    repo.save_object(stored_a, store=store2)
    repo.save_object(stored_b, store=store1)
    repo.save_object(stored_set, store=store2)
    stored_roots = [stored_a.definition, stored_b.definition, stored_set.definition]

    selectors = [
        Definition(PlannerParent, SKIP_ARGS, child=Definition(PlannerLeaf, SKIP_ARGS, name="shared")),
        Definition(PlannerParent, SKIP_ARGS, child=rare.definition),
        Definition(PlannerSetParent, SKIP_ARGS, members={rare.definition}),
        repo.query(stored_a.definition).categorical(path="child", recursive=True).selector,
        repo.query(stored_a.definition).categorical(recursive=True).selector,
    ]

    for selector in selectors:
        expected_known = brute_force_known(repo, selector)
        assert list(repo.query(selector).known(refresh=False).defs()) == expected_known

        expected_stored = [cdef for cdef in expected_known if cdef in stored_roots]
        assert list(repo.query(selector).stored(refresh=False).defs()) == expected_stored

    child_selector = Definition(PlannerLeaf, "shared")
    expected_nested = brute_force_nested_definitions(stored_roots, child_selector)
    expected_owners = brute_force_owners(stored_roots, child_selector)

    assert list(repo.query(child_selector).nested(refresh=False).definitions().defs()) == expected_nested
    owners = repo.query(child_selector).nested(refresh=False).owners().defs()
    assert list(owners) == expected_owners
    assert owners.replicas(stored_a.definition) == (store1, store2)

    occurrences = tuple(repo.query(child_selector).nested(refresh=False).execute())
    assert {occ.owner for occ in occurrences} == set(expected_owners)
    assert {occ.definition for occ in occurrences} == set(expected_nested)


def test_nested_owner_query_uses_graph_edges_without_materializing(tmp_path):
    repo = Repo(stores=DirStore(tmp_path / "store"))
    child = PlannerLeaf(name="child", repo=repo)
    parent = PlannerParent(child=child, repo=repo)
    repo.save_object(parent)

    owners = repo.query(Definition(PlannerLeaf, SKIP_ARGS, name="child")).nested(refresh=False).owners().defs()

    assert list(owners) == [parent.definition]
    assert repo._num_constructions == 0


def test_nested_definitions_does_not_enumerate_occurrence_paths(tmp_path, monkeypatch):
    repo = Repo(stores=DirStore(tmp_path / "store"))
    child = PlannerLeaf(name="child", repo=repo)
    parent = PlannerParent(child=child, repo=repo)
    repo.save_object(parent)

    def fail_occurrences(*args, **kwargs):
        raise AssertionError("nested definitions should not enumerate occurrence paths")

    monkeypatch.setattr(repo._query_catalog, "occurrences_for_nested_ids", fail_occurrences)
    monkeypatch.setattr(repo._query_catalog, "all_occurrences", fail_occurrences)

    defs = repo.query(Definition(PlannerLeaf, SKIP_ARGS, name="child")).nested(refresh=False).definitions().defs()

    assert list(defs) == [child.definition]


def test_nested_exact_anchor_does_not_construct_nested_universe(tmp_path, monkeypatch):
    repo = Repo(stores=DirStore(tmp_path / "store"))
    child = PlannerLeaf(name="child", repo=repo)
    parent = PlannerParent(child=child, repo=repo)
    repo.save_object(parent)

    def fail_nested_ids():
        raise AssertionError("nested_ids should not be called for exact nested anchors")

    monkeypatch.setattr(repo._query_catalog, "nested_ids", fail_nested_ids)

    defs = repo.query(child.definition).nested(refresh=False).definitions().defs()

    assert list(defs) == [child.definition]


def test_nested_definition_filter_does_not_call_stored_ids(tmp_path, monkeypatch):
    repo = Repo(stores=DirStore(tmp_path / "store"))
    child = PlannerLeaf(name="child", repo=repo)
    parent = PlannerParent(child=child, repo=repo)
    repo.save_object(parent)

    monkeypatch.setattr(
        repo._query_catalog,
        "all_stored_ids",
        lambda: (_ for _ in ()).throw(AssertionError("nested filter scanned stored roots")),
    )

    defs = repo.query(child.definition).nested(refresh=False).definitions().defs()

    assert list(defs) == [child.definition]


def test_owner_projection_does_not_call_stored_ids(tmp_path, monkeypatch):
    repo = Repo(stores=DirStore(tmp_path / "store"))
    child = PlannerLeaf(name="child", repo=repo)
    parent = PlannerParent(child=child, repo=repo)
    repo.save_object(parent)

    monkeypatch.setattr(
        repo._query_catalog,
        "all_stored_ids",
        lambda: (_ for _ in ()).throw(AssertionError("owner projection scanned stored roots")),
    )

    owners = repo.query(child.definition).nested(refresh=False).owners().defs()

    assert list(owners) == [parent.definition]


def test_nested_filter_keeps_definition_that_is_also_stored_root(tmp_path):
    repo = Repo(stores=DirStore(tmp_path / "store"))
    child = PlannerLeaf(name="child", repo=repo)
    parent = PlannerParent(child=child, repo=repo)
    repo.save_object(child)
    repo.save_object(parent)

    defs = repo.query(child.definition).nested(refresh=False).definitions().defs()

    assert list(defs) == [child.definition]


def test_nested_owners_uses_reverse_edges_without_occurrence_expansion(tmp_path, monkeypatch):
    repo = Repo(stores=DirStore(tmp_path / "store"))
    child = PlannerLeaf(name="child", repo=repo)
    parent = PlannerParent(child=child, repo=repo)
    repo.save_object(parent)

    def fail_occurrences(*args, **kwargs):
        raise AssertionError("nested owners should not enumerate occurrence paths")

    monkeypatch.setattr(repo._query_catalog, "occurrences_for_nested_ids", fail_occurrences)
    monkeypatch.setattr(repo._query_catalog, "all_occurrences", fail_occurrences)

    owners = repo.query(Definition(PlannerLeaf, SKIP_ARGS, name="child")).nested(refresh=False).owners().defs()

    assert list(owners) == [parent.definition]
