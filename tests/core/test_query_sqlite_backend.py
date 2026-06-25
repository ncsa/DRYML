import pytest

from dryml.core2 import Definition, Object, SKIP_ARGS
from dryml.core2.cdef_graph import ConcreteDefinitionGraph
from dryml.core2.query.domain import StoredDomain
from dryml.core2.query.graph_plan import graph_candidate_ids
from dryml.core2.query.selector_graph import compile_selector_graph
from dryml.core2.query.sqlite import SQLiteQueryIndexConfig, sqlite_available
from dryml.core2.query.sqlite.index import SQLiteStoreQueryIndex


pytestmark = pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")


class SQLiteLeaf(Object):
    def __init__(self, name="leaf"):
        super().__init__()
        self.name = name


class SQLiteParent(Object):
    def __init__(self, child=None, *, name="parent"):
        super().__init__()
        self.child = child
        self.name = name


class SQLitePair(Object):
    def __init__(self, left=None, right=None, *, name="pair"):
        super().__init__()
        self.left = left
        self.right = right
        self.name = name


def sqlite_index(tmp_path):
    return SQLiteStoreQueryIndex(
        source_key="sqlite-test-store",
        path=tmp_path / "index.sqlite",
        config=SQLiteQueryIndexConfig(journal_mode="delete", busy_timeout=1.0),
    )


def test_register_stored_roots_persists_graph_and_is_idempotent(tmp_path):
    index = sqlite_index(tmp_path)
    leaf = SQLiteLeaf("rare")
    owner = SQLiteParent(child=leaf, name="owner")
    graph = ConcreteDefinitionGraph.from_root(owner.definition)

    result = index.register_stored_roots(graph, [owner.definition])
    repeat = index.register_stored_roots(graph, [owner.definition])

    assert result.changed
    assert result.generation == 1
    assert result.definitions_added == 2
    assert result.edges_added == 1
    assert result.roots_added == 1
    assert repeat.changed is False
    assert repeat.generation == 1
    assert index.status().generation == 1

    with index.read_view() as view:
        owner_id = view.cdef_id(owner.definition)
        leaf_id = view.cdef_id(leaf.definition)
        assert owner_id is not None
        assert leaf_id is not None
        assert view.is_stored_id(owner_id)
        assert not view.is_stored_id(leaf_id)
        assert view.has_stored_ancestor(leaf_id)
        assert view.cdefs_by_id({owner_id, leaf_id}) == {
            owner_id: owner.definition,
            leaf_id: leaf.definition,
        }


def test_exact_and_local_candidate_lookup(tmp_path):
    index = sqlite_index(tmp_path)
    wanted = SQLiteParent(child=SQLiteLeaf(name="wanted"), name="target")
    other = SQLiteParent(child=SQLiteLeaf(name="other"), name="target")
    graph = ConcreteDefinitionGraph.from_roots([wanted.definition, other.definition])
    index.register_stored_roots(graph, [wanted.definition, other.definition])

    selector = Definition(SQLiteParent, SKIP_ARGS, child=Definition(SQLiteLeaf, SKIP_ARGS, name="wanted"))
    selector_graph = compile_selector_graph(selector)

    with index.read_view() as view:
        domain = StoredDomain(view)
        ids = graph_candidate_ids(view, selector_graph, domain)
        cdefs = tuple(view.cdefs_by_id(ids).values())

    assert cdefs == (wanted.definition,)


def test_parent_child_relations_and_nested_projection(tmp_path):
    index = sqlite_index(tmp_path)
    leaf = SQLiteLeaf("nested")
    owner = SQLitePair(left=leaf, right=SQLiteLeaf("other"), name="owner")
    graph = ConcreteDefinitionGraph.from_root(owner.definition)
    index.register_stored_roots(graph, [owner.definition])

    # Use concrete graph paths for the exact relation checks.
    left_path = next(edge.path for edge in graph.edges() if edge.child == leaf.definition)
    with index.read_view() as view:
        owner_id = view.cdef_id(owner.definition)
        leaf_id = view.cdef_id(leaf.definition)
        assert view.children({owner_id}, left_path, unordered=False) == {leaf_id}
        assert view.parents({leaf_id}, left_path, unordered=False) == {owner_id}
        assert view.children({owner_id}, left_path.parent, unordered=True) >= {leaf_id}
        assert leaf_id in view.nested_ids()
        assert view.filter_nested_ids({owner_id, leaf_id}) == {leaf_id}


def test_owner_projection_and_occurrence_capture(tmp_path):
    index = sqlite_index(tmp_path)
    leaf = SQLiteLeaf("shared")
    owner1 = SQLiteParent(child=leaf, name="owner1")
    owner2 = SQLiteParent(child=leaf, name="owner2")
    graph = ConcreteDefinitionGraph.from_roots([owner1.definition, owner2.definition])
    index.register_stored_roots(graph, [owner1.definition, owner2.definition])

    with index.read_view() as view:
        leaf_id = view.cdef_id(leaf.definition)
        projection = view.project_owners({leaf_id})
        occurrences = tuple(view.occurrence_snapshot_for_nested_ids({leaf_id}).iter_occurrences())

    assert set(projection.cdefs) == {owner1.definition, owner2.definition}
    assert {occ.owner for occ in occurrences} == {owner1.definition, owner2.definition}
    assert {occ.definition for occ in occurrences} == {leaf.definition}
    assert all(str(occ.path) == "$.child" for occ in occurrences)


def test_remove_stored_roots_updates_generation_and_stored_scope(tmp_path):
    index = sqlite_index(tmp_path)
    obj = SQLiteLeaf("stored")
    graph = ConcreteDefinitionGraph.from_root(obj.definition)
    index.register_stored_roots(graph, [obj.definition])

    removed = index.remove_stored_roots([obj.definition])
    repeat = index.remove_stored_roots([obj.definition])

    assert removed.changed
    assert removed.roots_removed == 1
    assert removed.generation == 2
    assert repeat.changed is False
    assert repeat.generation == 2
    with index.read_view() as view:
        did = view.cdef_id(obj.definition)
        assert did is not None
        assert not view.is_stored_id(did)
