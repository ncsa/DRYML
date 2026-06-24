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
    original = catalog.local_candidate_ids

    def spy_local_candidate_ids(universe_ids, requirements, *, stats=None):
        local_candidate_universe_sizes.append(len(universe_ids))
        return original(universe_ids, requirements, stats=stats)

    monkeypatch.setattr(catalog, "local_candidate_ids", spy_local_candidate_ids)

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
        raise AssertionError("stored_ids should not be called for exact-root queries")

    monkeypatch.setattr(repo._query_catalog, "stored_ids", fail_stored_ids)

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
        "stored_ids",
        lambda: (_ for _ in ()).throw(AssertionError("eager stored universe")),
    )

    selector = Definition(PlannerParent, SKIP_ARGS, child=Definition(PlannerLeaf, SKIP_ARGS, name="wanted"))
    results = repo.query(selector).stored(refresh=False).defs()

    assert list(results) == [wanted.definition]


def test_selective_known_query_never_constructs_known_id_set(monkeypatch):
    repo = Repo()
    wanted = PlannerParent(child=PlannerLeaf(name="wanted", repo=repo), repo=repo)
    repo.add_objects(wanted)
    for idx in range(6):
        repo.add_objects(PlannerParent(child=PlannerLeaf(name=f"other-{idx}", repo=repo), repo=repo))

    monkeypatch.setattr(
        repo._query_catalog,
        "known_ids",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("eager known universe")),
    )

    selector = Definition(PlannerParent, SKIP_ARGS, child=Definition(PlannerLeaf, SKIP_ARGS, name="wanted"))
    results = repo.query(selector).known(refresh=False).defs()

    assert list(results) == [wanted.definition]


def test_selective_cached_query_never_constructs_cached_id_set(monkeypatch):
    repo = Repo()
    wanted = PlannerParent(child=PlannerLeaf(name="wanted", repo=repo), repo=repo)
    repo.add_objects(wanted)
    for idx in range(6):
        repo.add_objects(PlannerParent(child=PlannerLeaf(name=f"other-{idx}", repo=repo), repo=repo))

    monkeypatch.setattr(
        repo._query_catalog,
        "cached_ids",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("eager cached universe")),
    )

    selector = Definition(PlannerParent, SKIP_ARGS, child=Definition(PlannerLeaf, SKIP_ARGS, name="wanted"))
    results = repo.query(selector).cached(refresh=False).defs()

    assert list(results) == [wanted.definition]


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
    original = catalog.local_candidate_ids

    def spy_local_candidate_ids(universe_ids, requirements, *, stats=None):
        bounded_universe_sizes.append(len(universe_ids))
        return original(universe_ids, requirements, stats=stats)

    monkeypatch.setattr(catalog, "local_candidate_ids", spy_local_candidate_ids)

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
    original_estimate = catalog.estimate_local_candidates
    original_materialize = catalog.local_candidate_ids_unbounded

    def spy_estimate(requirements):
        estimate_calls.append(requirements)
        if len(estimate_calls) == 1:
            return 1000
        return original_estimate(requirements)

    def spy_materialize(requirements, *, stats=None):
        materialize_calls.append(requirements)
        return original_materialize(requirements, stats=stats)

    monkeypatch.setattr(catalog, "estimate_local_candidates", spy_estimate)
    monkeypatch.setattr(catalog, "local_candidate_ids_unbounded", spy_materialize)

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
        "stored_ids",
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
        "stored_ids",
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
