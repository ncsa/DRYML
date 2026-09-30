"""Query integration coverage for exact template support selectors."""

from __future__ import annotations

from itertools import chain, repeat
import warnings

import pytest

from dryml.core import Definition, Generator, Object, Ref, Repo, Template
from dryml.core.cdef_graph import EdgeKind
from dryml.core.domains import UniformFromSet
from dryml.core.errors import ParameterizationLimitError
from dryml.core.links import DefLink
from dryml.core.query.model import QueryDomainError, QueryIndexError
from dryml.core.template import Par
from dryml.core.params import AnyValue
from dryml.core.factory import FactorySpec


class QueryTemplateLeaf(Object):
    """Small owned node used to distinguish equal query graph witnesses."""

    def __init__(self, width):
        self.width = width


class QueryTemplateParent(Object):
    """Root object retaining an ordered collection of query graph nodes."""

    def __init__(self, children):
        self.children = children


class FactoryQueryOwner(Object):
    """Owner whose inert factory call exercises partial selector lowering."""

    def __init__(self, factory):
        self.factory = factory


class RecipeQueryOwner(Object):
    """Owner carrying an inert recipe through an explicit Ref boundary."""

    def __init__(self, recipe: Template):
        self.recipe = recipe


def _selector(*, shared: bool):
    child = Definition(QueryTemplateLeaf, Par("width"))
    children = [child, child] if shared else [child, Definition(QueryTemplateLeaf, Par("width"))]
    return Generator(
        Definition(QueryTemplateParent, children),
        {"width": UniformFromSet((64,))},
    ).support_selector()


def test_template_selector_query_retains_matching_graph_witness_before_deduplication():
    """Exact support returns the matching topology even when CDefs compare equally."""

    repo = Repo()
    shared_leaf = QueryTemplateLeaf(64, repo=repo)
    shared = QueryTemplateParent([shared_leaf, shared_leaf], repo=repo)
    independent = QueryTemplateParent(
        [QueryTemplateLeaf(64, repo=repo), QueryTemplateLeaf(64, repo=repo)],
        repo=repo,
    )
    repo.add_objects(shared, independent)

    results = tuple(repo.query(_selector(shared=True)).cached().defs())

    assert results == (shared.definition,)


def test_template_selector_query_rejects_unsupported_structural_rewrites():
    """Residual-bearing queries fail before a structural rewrite can drop support."""

    query = Repo().query(_selector(shared=True))

    with pytest.raises(QueryDomainError):
        query.references()
    with pytest.raises(TypeError):
        Repo().references().definition(_selector(shared=True))
    with pytest.raises(QueryDomainError):
        query.categorical()
    with pytest.raises(QueryDomainError):
        query.restore()
    with pytest.raises(QueryDomainError):
        query.exact()


def test_template_selector_rejects_malformed_store_root_enumeration(tmp_path, monkeypatch):
    """Exact terminals fail closed when a Store emits non-CDef authority."""

    from dryml.core.store.dir import DirStore

    store = DirStore(tmp_path / "store")
    monkeypatch.setattr(
        store,
        "iter_authoritative_root_definitions",
        lambda: iter((Definition(QueryTemplateLeaf, 64),)),
    )

    with pytest.raises((QueryDomainError, QueryIndexError), match="not ConcreteDefinition|non-CDef"):
        Repo(store).query(_selector(shared=True)).stored().defs()


def test_template_selector_query_witness_budget_never_returns_partial_results():
    """Every witness visit consumes the terminal residual budget."""

    repo = Repo()
    repo.add_objects(
        QueryTemplateParent([QueryTemplateLeaf(64, repo=repo)] * 2, repo=repo),
        QueryTemplateParent([QueryTemplateLeaf(64, repo=repo)] * 2, repo=repo),
    )

    with pytest.raises(ParameterizationLimitError):
        repo.query(_selector(shared=True)).cached().max_witnesses(1).defs()

    assert len(repo.query(_selector(shared=True)).cached().max_witnesses(None).defs()) == 1


def test_nested_template_selector_honors_reference_aware_path_policies(tmp_path):
    """Exact nested support uses each literal retained containment edge policy."""

    from dryml.core.store.dir import DirStore

    repo = Repo(DirStore(tmp_path / "store"))
    material_leaf = QueryTemplateLeaf(64, repo=repo)
    referenced_leaf = QueryTemplateLeaf(64, repo=repo)
    mixed_leaf = QueryTemplateLeaf(64, repo=repo)
    material = QueryTemplateParent(material_leaf, repo=repo)
    reference = QueryTemplateParent(
        DefLink.finalized(EdgeKind.REF, referenced_leaf.definition), repo=repo,
    )
    mixed = QueryTemplateParent(
        DefLink.finalized(
            EdgeKind.REF,
            QueryTemplateParent(mixed_leaf, repo=repo).definition,
        ),
        repo=repo,
    )
    repo.save_object(material)
    repo.save_object(reference)
    repo.save_object(mixed)

    selector = Generator(
        Definition(QueryTemplateLeaf, Par("width")),
        {"width": UniformFromSet((64,))},
    ).support_selector()

    assert len(tuple(repo.query(selector).nested().execute())) == 1
    assert len(tuple(repo.query(selector).nested(edges="ref").execute())) == 1
    assert len(tuple(repo.query(selector).nested(edges="all").execute())) == 3
    assert len(
        tuple(repo.query(selector).nested(edges="all", contains_ref=True).execute())
    ) == 2
    assert len(
        repo.query(selector).nested(edges="all").max_occurrences(1).definitions().defs()
    ) == 1
    assert len(
        tuple(repo.query(selector).nested(edges="all").max_occurrences(1).execute())
    ) == 1


def test_nested_template_selector_preserves_shared_topology_witnesses(tmp_path):
    """Exact nested support distinguishes a shared child graph from equal nodes."""

    from dryml.core.store.dir import DirStore

    repo = Repo(DirStore(tmp_path / "store"))
    shared_leaf = QueryTemplateLeaf(64, repo=repo)
    shared = QueryTemplateParent([shared_leaf, shared_leaf], repo=repo)
    independent = QueryTemplateParent(
        [QueryTemplateLeaf(64, repo=repo), QueryTemplateLeaf(64, repo=repo)],
        repo=repo,
    )
    root = QueryTemplateParent([shared.definition, independent.definition], repo=repo)
    repo.save_object(root)

    results = repo.query(_selector(shared=True)).nested().definitions().defs()

    assert tuple(results) == (shared.definition,)


def test_nested_template_selector_drains_before_occurrence_cap(tmp_path):
    """Raw caps cannot suppress exact-support assignment or witness failures."""

    from dryml.core.store.dir import DirStore

    repo = Repo(DirStore(tmp_path / "store"))
    leaf = QueryTemplateLeaf(64, repo=repo)
    repo.save_object(QueryTemplateParent([leaf, leaf], repo=repo))
    selector = Generator(
        Definition(QueryTemplateLeaf, Par("width")),
        {"width": UniformFromSet((64,))},
    ).support_selector(max_assignments=1)
    query = repo.query(selector).nested().max_occurrences(1)

    with pytest.raises(ParameterizationLimitError, match="assignment limit"):
        query.execute()
    with pytest.raises(ParameterizationLimitError, match="witness limit"):
        query.max_witnesses(1).execute()


def test_capped_nested_template_result_cannot_regrow_fixed_witnesses(tmp_path):
    """Fixed requery and projections retain only a capped raw result's paths."""

    from dryml.core.store.dir import DirStore

    repo = Repo(DirStore(tmp_path / "store"))
    first = QueryTemplateParent(QueryTemplateLeaf(64, repo=repo), repo=repo)
    second = QueryTemplateParent(QueryTemplateLeaf(128, repo=repo), repo=repo)
    repo.save_object(first)
    repo.save_object(second)
    selector = Generator(
        Definition(QueryTemplateLeaf, Par("width")),
        {"width": UniformFromSet((64, 128))},
    ).support_selector()

    raw = repo.query(selector).nested().max_occurrences(1).execute()
    visible = raw.one()

    assert raw._containment.bounded
    assert tuple(raw.query().nested().execute()) == (visible,)
    assert tuple(raw.owners().query().nested().owners().defs()) == (visible.owner,)


def test_fixed_complete_cdef_containment_supports_generator_refinement(tmp_path):
    """Complete direct CDef evidence retains context across exact refinements."""

    from dryml.core.store.dir import DirStore

    repo = Repo(DirStore(tmp_path / "store"))
    first = QueryTemplateLeaf(64, repo=repo)
    second = QueryTemplateLeaf(128, repo=repo)
    repo.save_object(QueryTemplateParent(first, repo=repo))
    repo.save_object(QueryTemplateParent(second, repo=repo))
    direct_selector = Definition(QueryTemplateLeaf, AnyValue())
    generator = Generator(
        Definition(QueryTemplateLeaf, Par("width")),
        {"width": UniformFromSet((64,))},
    ).support_selector()

    complete = repo.query(direct_selector).nested(edges="all").definitions().defs()
    refined = complete.refine(generator)

    assert complete._containment.complete
    assert tuple(refined) == (first.definition,)
    assert refined._containment == complete._containment
    assert refined._containment_carrier == "target"
    assert tuple(item.target for item in refined._containment_witnesses) == (first.definition,)
    assert tuple(refined.refine(generator)) == (first.definition,)

    indexed = repo.query(direct_selector).nested().definitions().defs()
    assert not indexed._containment.complete
    with pytest.raises(QueryDomainError, match="complete retained witness evidence"):
        indexed.refine(generator)


@pytest.mark.parametrize("carrier", ["direct", "raw", "owner"])
@pytest.mark.parametrize(
    ("budget", "message"),
    [("assignment", "assignment limit"), ("witness", "witness limit")],
)
def test_fixed_complete_containment_shares_template_budgets(
        tmp_path, monkeypatch, carrier, budget, message):
    """Fixed containment candidates share exact selector assignment and visit caps."""

    from dryml.core.store.dir import DirStore

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    first = QueryTemplateParent(QueryTemplateLeaf(64, repo=repo), repo=repo)
    second = QueryTemplateParent(QueryTemplateLeaf(128, repo=repo), repo=repo)
    repo.save_object(first)
    repo.save_object(second)
    direct_selector = Definition(QueryTemplateLeaf, AnyValue())
    generator = Generator(
        Definition(
            QueryTemplateLeaf if carrier != "owner" else QueryTemplateParent,
            Par("width") * 1 if carrier != "owner"
            else Definition(QueryTemplateLeaf, Par("width") * 1),
        ),
        {"width": UniformFromSet((64, 128))},
    ).support_selector(max_assignments=2)

    if carrier == "direct":
        fixed = repo.query(direct_selector).nested(edges="all").definitions().defs()
    elif carrier == "raw":
        fixed = repo.query(direct_selector).nested(edges="all").execute().definitions()
    else:
        fixed = repo.query(direct_selector).nested(edges="all").owners().defs()

    assert fixed._containment.complete
    assert len(fixed) == 2
    assert all(generator.matches(candidate) for candidate in fixed)
    monkeypatch.setattr(
        store,
        "iter_authoritative_root_definitions",
        lambda: pytest.fail("fixed containment refinement scanned Store authority"),
    )

    query = fixed.query(generator)
    if budget == "witness":
        query = query.max_witnesses(1)
    with pytest.raises(ParameterizationLimitError, match=message):
        query.defs()


def test_nested_template_selector_rejects_forbidden_shared_residual_scan(tmp_path, monkeypatch):
    """Reference-aware exact support fails before entering authority traversal."""

    from dryml.core.store.dir import DirStore

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    repo.save_object(QueryTemplateParent(QueryTemplateLeaf(64, repo=repo), repo=repo))

    monkeypatch.setattr(
        store,
        "iter_authoritative_root_definitions",
        lambda: pytest.fail("non-analyzing explanation scanned authority"),
    )
    explanation = repo.query(_selector(shared=True)).nested(edges="all").require_indexed().explain()

    assert explanation.scan_required

    with pytest.raises(QueryDomainError, match="complete witness scanning"):
        repo.query(_selector(shared=True)).nested(edges="all").require_indexed().execute()


@pytest.mark.parametrize("policy", ["allow", "warn", "require_indexed", "forbid"])
@pytest.mark.parametrize("terminal", ["count", "exists", "execute", "explain"])
def test_nested_template_selector_materialize_ref_filter_is_empty_without_authority(
        tmp_path, monkeypatch, policy, terminal):
    """Impossible exact containment does not require Store enumeration or a scan."""

    from dryml.core.store.dir import DirStore

    store = DirStore(tmp_path / "store")
    repo = Repo(store)

    def fail_authority_access(*args, **kwargs):
        pytest.fail("deterministic empty exact containment accessed Store authority")

    monkeypatch.setattr(store, "iter_authoritative_root_definitions", None)
    monkeypatch.setattr(store, "authoritative_root_definitions", fail_authority_access)
    query = repo.query(_selector(shared=True)).nested(contains_ref=True)
    query = query.require_indexed() if policy == "require_indexed" else query.scan_policy(policy)

    with warnings.catch_warnings(record=True) as caught:
        result = getattr(query, terminal)()

    assert not caught
    if terminal == "count":
        assert result == 0
    elif terminal == "exists":
        assert result is False
    elif terminal == "execute":
        assert tuple(result) == ()
    else:
        assert result.scan_required is False
        assert result.result_count == 0


def test_nested_template_selector_rejects_incomplete_fixed_witnesses():
    """Fixed raw results cannot claim exact support without complete evidence."""

    from dryml.core.query.result import OccurrenceResultSet

    repo = Repo()
    incomplete = OccurrenceResultSet(repo, (), witness_complete=False)

    with pytest.raises(QueryDomainError, match="complete immutable witness evidence"):
        incomplete.refine(_selector(shared=True))


def test_template_selector_witness_budget_counts_prefilter_rejections_and_duplicates(
        tmp_path, monkeypatch):
    """The default cap charges every authority visit before safe prefiltering."""

    import dryml.core.query.query as query_module
    from dryml.core.store.dir import DirStore

    assert query_module._DEFAULT_GENERATOR_WITNESS_LIMIT == 65_536
    monkeypatch.setattr(query_module, "_DEFAULT_GENERATOR_WITNESS_LIMIT", 2)

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    rejected = QueryTemplateLeaf(64, repo=repo)
    leaf = QueryTemplateLeaf(64, repo=repo)
    matching = QueryTemplateParent([leaf, leaf], repo=repo)
    repo.save_object(rejected)
    repo.save_object(matching)

    def authority():
        return chain(repeat(rejected.definition, 2), (matching.definition,))

    monkeypatch.setattr(store, "iter_authoritative_root_definitions", authority)

    query = repo.query(_selector(shared=True)).stored()
    with pytest.raises(ParameterizationLimitError, match="witness limit"):
        query.defs()

    assert tuple(query.max_witnesses(3).defs()) == (matching.definition,)


def test_nested_template_witness_budget_stops_authority_streaming(tmp_path, monkeypatch):
    """Nested witness exhaustion does not pre-accumulate later Store roots."""

    from dryml.core.store.dir import DirStore

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    first_leaf = QueryTemplateLeaf(64, repo=repo)
    first = QueryTemplateParent([first_leaf, first_leaf], repo=repo)
    second_leaf = QueryTemplateLeaf(64, repo=repo)
    second = QueryTemplateParent([second_leaf, second_leaf], repo=repo)
    visited = []

    def authority():
        visited.append(first.definition)
        yield first.definition
        visited.append(second.definition)
        yield second.definition

    monkeypatch.setattr(store, "iter_authoritative_root_definitions", authority)

    with pytest.raises(ParameterizationLimitError, match="witness limit"):
        repo.query(_selector(shared=True)).nested().definitions().max_witnesses(1).defs()

    assert visited == [first.definition]


def test_template_selector_query_scans_stored_authority_and_controls_refinement(tmp_path):
    """Stored and complete fixed universes retain exact graph evidence."""

    from dryml.core.store.dir import DirStore

    store = DirStore(tmp_path / "store")
    repo = Repo(stores=store)
    independent = QueryTemplateParent(
        [QueryTemplateLeaf(64, repo=repo), QueryTemplateLeaf(64, repo=repo)],
        repo=repo,
    )
    shared_leaf = QueryTemplateLeaf(64, repo=repo)
    shared = QueryTemplateParent([shared_leaf, shared_leaf], repo=repo)
    repo.save_object(independent)
    repo.save_object(shared)

    selected = repo.query(_selector(shared=True)).stored().defs()

    assert tuple(selected) == (shared.definition,)
    assert tuple(selected.refine(_selector(shared=True))) == (shared.definition,)
    with pytest.raises(QueryDomainError):
        repo.query().stored().defs().refine(_selector(shared=True))

    leaf_selector = Generator(
        Definition(QueryTemplateLeaf, Par("width")),
        {"width": UniformFromSet((64,))},
    ).support_selector()
    nested = repo.query(leaf_selector).nested().definitions().defs()
    assert len(nested) == 1

    from dryml.core import SaveRouting

    fallback = DirStore(tmp_path / "fallback")
    routed = Repo(stores=(store, fallback), save_routing=SaveRouting(((_selector(shared=True), fallback),)))
    with routed._retain_save_context() as context:
        assert routed._select_save_destinations(context, shared.definition) == (fallback,)
        assert routed._select_save_destinations(context, independent.definition) == (store,)
    assert routed.to_definition().to_data()["routing"]["routes"][0]["selector"]["kind"] == "template-selector"


def test_template_selector_uses_index_prefilter_without_hydrate_index(tmp_path, monkeypatch):
    """Stored exact queries retain topology without bypassing index candidates."""

    from dryml.core.store.dir import DirStore

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    leaf = QueryTemplateLeaf(64, repo=repo)
    owner = QueryTemplateParent([leaf, leaf], repo=repo)
    repo.save_object(owner)
    monkeypatch.setattr(
        store,
        "hydrate_index",
        lambda: pytest.fail("exact query bypassed the index prefilter"),
    )

    assert tuple(repo.query(_selector(shared=True)).stored().defs()) == (
        owner.definition,
    )


def test_traversed_recipe_ref_has_direct_and_query_support_parity():
    """Loose index projection cannot reject a recipe accepted by exact support."""

    recipe = Definition(QueryTemplateLeaf, Par("width"))
    selector = Generator(
        Definition(RecipeQueryOwner, Ref(recipe)),
        {"width": UniformFromSet((64,))},
        traverse_refs=True,
    ).support_selector()
    repo = Repo()
    owner = RecipeQueryOwner(recipe.sub(width=64), repo=repo)
    repo.add_objects(owner)

    assert selector.matches(owner.definition)
    assert tuple(repo.query(selector).cached().defs()) == (owner.definition,)


def test_bare_template_recipe_has_direct_and_query_support_parity():
    """Receiving-role projection quotes natural Template authoring for support."""

    recipe = Definition(QueryTemplateLeaf, Par("width"))
    selector = Generator(
        Definition(RecipeQueryOwner, recipe),
        {"width": UniformFromSet((64,))},
    ).support_selector()
    repo = Repo()
    owner = RecipeQueryOwner(recipe.sub(width=64), repo=repo)
    repo.add_objects(owner)

    assert selector.matches(owner.definition)
    assert tuple(repo.query(selector).cached().defs()) == (owner.definition,)


@pytest.mark.parametrize("shared_first", [False, True])
def test_template_selector_preserves_graph_witnesses_across_stores(tmp_path, shared_first):
    """Store partition and insertion order cannot choose the wrong topology witness."""

    from dryml.core.store.dir import DirStore

    independent_store = DirStore(tmp_path / "independent")
    shared_store = DirStore(tmp_path / "shared")
    independent_repo = Repo(independent_store)
    independent = QueryTemplateParent(
        [
            QueryTemplateLeaf(64, repo=independent_repo),
            QueryTemplateLeaf(64, repo=independent_repo),
        ],
        repo=independent_repo,
    )
    independent_repo.save_object(independent)

    shared_repo = Repo(shared_store)
    shared_leaf = QueryTemplateLeaf(64, repo=shared_repo)
    shared = QueryTemplateParent([shared_leaf, shared_leaf], repo=shared_repo)
    shared_repo.save_object(shared)

    stores = (shared_store, independent_store) if shared_first else (independent_store, shared_store)
    results = Repo(stores=stores).query(_selector(shared=True)).stored().defs()

    assert tuple(results) == (shared.definition,)
    assert results.replicas(shared.definition) == (shared_store,)


def test_occurrence_refinement_preserves_owner_replica_authority(tmp_path):
    """Exact nested refinement and projection retain each owner's source Store."""

    from dryml.core.store.dir import DirStore

    first_store = DirStore(tmp_path / "first")
    second_store = DirStore(tmp_path / "second")
    first_repo = Repo(first_store)
    second_repo = Repo(second_store)
    first_leaf = QueryTemplateLeaf(64, repo=first_repo)
    second_leaf = QueryTemplateLeaf(128, repo=second_repo)
    first = QueryTemplateParent([first_leaf], repo=first_repo)
    second = QueryTemplateParent([second_leaf], repo=second_repo)
    first_repo.save_object(first)
    second_repo.save_object(second)

    selector = Generator(
        Definition(QueryTemplateLeaf, Par("width")),
        {"width": UniformFromSet((64, 128))},
    ).support_selector()
    repo = Repo(stores=(first_store, second_store))
    occurrences = repo.query(selector).nested().execute().refine(selector)
    owners = occurrences.owners()

    assert owners.replicas(first.definition) == (first_store,)
    assert owners.replicas(second.definition) == (second_store,)
    loaded = tuple(owners.objects().values())
    assert len(loaded) == 2
    assert {item.children[0].width: item._store_affinity for item in loaded} == {
        64: first_store,
        128: second_store,
    }


def test_occurrence_projections_retain_exact_witnesses_for_refinement(tmp_path):
    """Definitions and owners projected from exact occurrences remain refinable."""

    from dryml.core.store.dir import DirStore

    repo = Repo(DirStore(tmp_path / "store"))
    leaf = QueryTemplateLeaf(64, repo=repo)
    owner = QueryTemplateParent([leaf], repo=repo)
    repo.save_object(owner)
    leaf_selector = Generator(
        Definition(QueryTemplateLeaf, Par("width")),
        {"width": UniformFromSet((64,))},
    ).support_selector()
    owner_selector = Generator(
        Definition(QueryTemplateParent, [Definition(QueryTemplateLeaf, Par("width"))]),
        {"width": UniformFromSet((64,))},
    ).support_selector()

    occurrences = repo.query(leaf_selector).nested().execute()

    assert tuple(occurrences.definitions().refine(leaf_selector)) == (leaf.definition,)
    assert tuple(occurrences.owners().refine(owner_selector)) == (owner.definition,)


def test_template_query_enforces_one_cumulative_assignment_budget():
    """Per-witness exact proofs cannot multiply beyond the terminal hard cap."""

    repo = Repo()
    repo.add_objects(
        QueryTemplateParent(
            [QueryTemplateLeaf(64, repo=repo), QueryTemplateLeaf(64, repo=repo)],
            repo=repo,
        ),
        QueryTemplateParent(
            [QueryTemplateLeaf(64, repo=repo), QueryTemplateLeaf(64, repo=repo)],
            repo=repo,
        ),
    )
    child = Definition(QueryTemplateLeaf, Par("width") * 1)
    selector = Generator(
        Definition(QueryTemplateParent, [child, Definition(QueryTemplateLeaf, Par("width") * 1)]),
        {"width": UniformFromSet((64,))},
    ).support_selector(max_assignments=1)

    with pytest.raises(ParameterizationLimitError, match="assignment limit"):
        repo.query(selector).cached().defs()


def test_query_backed_results_reject_exact_refinement_without_witness_evidence():
    """Paged structural representatives fail explicitly rather than invent witnesses."""

    from dryml.core.query.result import QueryBackedDefinitionResultSet

    repo = Repo()
    leaf = QueryTemplateLeaf(64, repo=repo)
    parent = QueryTemplateParent([leaf, leaf], repo=repo)
    results = QueryBackedDefinitionResultSet(
        repo,
        lambda: iter(((parent.definition, ()),)),
        materializable=False,
    )

    with pytest.raises(QueryDomainError, match="witness"):
        results.refine(_selector(shared=True))


@pytest.mark.parametrize("limit", [True, 0, -1, 1.5])
def test_template_selector_query_rejects_invalid_witness_limits(limit):
    """Witness limits accept only positive exact integer controls or None."""

    with pytest.raises(ValueError):
        Repo().query(_selector(shared=True)).max_witnesses(limit)


def test_partial_factory_pattern_uses_scan_verification_not_an_atomic_hash():
    """Wildcard factory arguments retain all matching targets through lowering."""

    repo = Repo()
    first = FactoryQueryOwner(FactorySpec("factory", 1), repo=repo)
    second = FactoryQueryOwner(FactorySpec("factory", 2), repo=repo)
    repo.add_objects(first, second)

    selector = Definition(FactoryQueryOwner, FactorySpec("factory", AnyValue()))

    from dryml.core.query.selector_graph import compile_selector_graph

    graph = compile_selector_graph(selector)
    assert graph is not None and graph.requires_scan
    assert set(repo.query(selector).cached().defs()) == {first.definition, second.definition}
