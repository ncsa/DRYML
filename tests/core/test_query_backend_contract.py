from dataclasses import dataclass

import pytest

from dryml.core2 import Definition, Object, Repo, SKIP_ARGS
from dryml.core2.cdef_graph import ConcreteDefinitionGraph
from dryml.core2.query.domain import StoredDomain
from dryml.core2.query.graph_plan import graph_candidate_ids
from dryml.core2.query.model import IndexWriteResult
from dryml.core2.query.selector_graph import compile_selector_graph
from dryml.core2.query.sqlite import SQLiteQueryIndexConfig, sqlite_available
from dryml.core2.query.sqlite.index import SQLiteStoreQueryIndex
from dryml.core2.store.store import Store


class ContractLeaf(Object):
    def __init__(self, name="leaf"):
        super().__init__()
        self.name = name


class ContractParent(Object):
    def __init__(self, child=None, *, name="parent"):
        super().__init__()
        self.child = child
        self.name = name


class ContractPair(Object):
    def __init__(self, left=None, right=None, *, name="pair"):
        super().__init__()
        self.left = left
        self.right = right
        self.name = name


class ContractStore(Store):
    def __init__(self, key="contract-memory-store"):
        self.key = key

    @property
    def base_dir(self):
        return self.key

    @property
    def object_root_dir(self):
        return self.key

    def has(self, cdef):
        return False

    def hydrate_index(self):
        return ()

    def _object_dir(self, cdef):
        return self.key

    def commit(self):
        return None

    def catalog_key(self):
        return self.key


class MemoryContractIndex:
    source_key = "contract-memory-index"

    def __init__(self):
        self.repo = Repo(stores=ContractStore())
        self.store = self.repo.default_store

    def read_view(self, *, include_cached=True):
        return self.repo._query_catalog.read_view(include_cached=include_cached)

    def current_generation(self):
        return self.repo._query_catalog.current_generation()

    def register_stored_roots(self, graph, roots):
        before = self.current_generation()
        self.repo._query_catalog.register_stored_roots(tuple(roots), self.store)
        generation = self.current_generation()
        return IndexWriteResult(generation=generation, changed=generation != before)

    def refresh(self, policy, *, stats=None):
        return None

    def close(self):
        return None


@dataclass(frozen=True)
class BackendCase:
    name: str
    index: object


@pytest.fixture(params=["memory", "sqlite"])
def backend_case(request, tmp_path):
    if request.param == "memory":
        return BackendCase("memory", MemoryContractIndex())
    if not sqlite_available():
        pytest.skip("sqlite3 is unavailable")
    return BackendCase(
        "sqlite",
        SQLiteStoreQueryIndex(
            source_key="contract-sqlite-store",
            path=tmp_path / "index.sqlite",
            config=SQLiteQueryIndexConfig(journal_mode="delete", busy_timeout=1.0),
        ),
    )


def test_contract_register_exact_lookup_and_generation(backend_case):
    index = backend_case.index
    obj = ContractLeaf(name="exact")
    graph = ConcreteDefinitionGraph.from_root(obj.definition)

    result = index.register_stored_roots(graph, [obj.definition])
    repeat = index.register_stored_roots(graph, [obj.definition])

    assert result.changed
    assert repeat.changed is False
    assert index.current_generation() == result.generation
    with index.read_view(include_cached=False) as view:
        dids = view.exact_ids(obj.definition)
        assert len(dids) == 1
        did = next(iter(dids))
        assert view.generation == result.generation
        assert view.is_stored_id(did)
        assert view.cdefs_by_id(dids) == {did: obj.definition}


def test_contract_local_candidates_and_planner(backend_case):
    index = backend_case.index
    wanted = ContractParent(child=ContractLeaf(name="wanted"), name="root")
    other = ContractParent(child=ContractLeaf(name="other"), name="root")
    graph = ConcreteDefinitionGraph.from_roots([wanted.definition, other.definition])
    index.register_stored_roots(graph, [wanted.definition, other.definition])

    selector = Definition(ContractParent, SKIP_ARGS, child=Definition(ContractLeaf, SKIP_ARGS, name="wanted"))
    selector_graph = compile_selector_graph(selector)

    with index.read_view(include_cached=False) as view:
        domain = StoredDomain(view)
        candidate_ids = graph_candidate_ids(view, selector_graph, domain)
        cdefs = tuple(view.cdefs_by_id(candidate_ids).values())

    assert cdefs == (wanted.definition,)


def test_contract_parent_child_and_nested_semantics(backend_case):
    index = backend_case.index
    leaf = ContractLeaf(name="nested")
    owner = ContractPair(left=leaf, right=ContractLeaf(name="other"), name="owner")
    graph = ConcreteDefinitionGraph.from_root(owner.definition)
    index.register_stored_roots(graph, [owner.definition])
    left_path = next(edge.path for edge in graph.edges() if edge.child == leaf.definition)

    with index.read_view(include_cached=False) as view:
        owner_id = next(iter(view.exact_ids(owner.definition)))
        leaf_id = next(iter(view.exact_ids(leaf.definition)))
        assert view.children({owner_id}, left_path, unordered=False) == {leaf_id}
        assert view.parents({leaf_id}, left_path, unordered=False) == {owner_id}
        assert view.children({owner_id}, left_path.parent, unordered=True) >= {leaf_id}
        assert leaf_id in view.nested_ids()
        assert view.filter_nested_ids({owner_id, leaf_id}) == {leaf_id}
        assert view.has_stored_ancestor(leaf_id)


def test_contract_owner_projection_and_occurrence_capture(backend_case):
    index = backend_case.index
    leaf = ContractLeaf(name="shared")
    owner1 = ContractParent(child=leaf, name="owner1")
    owner2 = ContractParent(child=leaf, name="owner2")
    graph = ConcreteDefinitionGraph.from_roots([owner1.definition, owner2.definition])
    index.register_stored_roots(graph, [owner1.definition, owner2.definition])

    with index.read_view(include_cached=False) as view:
        leaf_id = next(iter(view.exact_ids(leaf.definition)))
        projection = view.project_owners({leaf_id})
        occurrences = tuple(view.occurrence_snapshot_for_nested_ids({leaf_id}).iter_occurrences())

    assert set(projection.cdefs) == {owner1.definition, owner2.definition}
    assert {occ.owner for occ in occurrences} == {owner1.definition, owner2.definition}
    assert {occ.definition for occ in occurrences} == {leaf.definition}
    assert all(str(occ.path) == "$.child" for occ in occurrences)


def test_contract_remove_stored_roots_when_supported(backend_case):
    index = backend_case.index
    if not hasattr(index, "remove_stored_roots"):
        pytest.skip("backend does not expose removal yet")
    obj = ContractLeaf(name="remove")
    graph = ConcreteDefinitionGraph.from_root(obj.definition)
    index.register_stored_roots(graph, [obj.definition])

    result = index.remove_stored_roots([obj.definition])

    assert result.changed
    with index.read_view(include_cached=False) as view:
        did = next(iter(view.exact_ids(obj.definition)))
        assert not view.is_stored_id(did)
