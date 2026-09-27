"""Query integration coverage for exact template support selectors."""

from __future__ import annotations

import pytest

from dryml.core import Definition, Object, Repo
from dryml.core.domains import UniformFromSet
from dryml.core.errors import TemplateLimitError
from dryml.core.query.model import QueryDomainError
from dryml.core.template import Par, Template
from dryml.core.template_selector import TemplateGenerator
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


def _selector(*, shared: bool):
    child = Definition(QueryTemplateLeaf, Par("width"))
    children = [child, child] if shared else [child, Definition(QueryTemplateLeaf, Par("width"))]
    return TemplateGenerator(
        Template(QueryTemplateParent, children),
        width=UniformFromSet((64,)),
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


def test_template_selector_query_witness_budget_never_returns_partial_results():
    """Every witness visit consumes the terminal residual budget."""

    repo = Repo()
    repo.add_objects(
        QueryTemplateParent([QueryTemplateLeaf(64, repo=repo)] * 2, repo=repo),
        QueryTemplateParent([QueryTemplateLeaf(64, repo=repo)] * 2, repo=repo),
    )

    with pytest.raises(TemplateLimitError):
        repo.query(_selector(shared=True)).cached().max_witnesses(1).defs()

    assert len(repo.query(_selector(shared=True)).cached().max_witnesses(None).defs()) == 1


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

    leaf_selector = TemplateGenerator(
        Template(QueryTemplateLeaf, Par("width")),
        width=UniformFromSet((64,)),
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
