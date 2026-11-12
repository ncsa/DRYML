"""Bounded metadata work for selective nested queries and scalar path counts."""

import pytest

from dryml.core import Definition, Object, ObjectId, Repo, Selector, Serializable
from dryml.core.query.identity import IdentitySet
from dryml.core.query.query import IdentityQuery
from dryml.core.query.relationships import RelationshipDepthExceeded
from dryml.core.store.dir import DirStore


class NestedWorkModel(Object):
    """Provide a class-only model pattern without scalar posting anchors."""

    def __init__(self):
        raise AssertionError("nested query constructed a model")


class NestedWorkDerived(NestedWorkModel):
    """Exercise selectors whose class policy includes subclasses."""


class NestedWorkBranch(Object):
    """Retain labeled materializing DAG edges without private state."""

    def __init__(self, label, children):
        raise AssertionError("nested query constructed a branch")


class NestedWorkState(Serializable):
    """Carry identity declarations whose conflicts must survive prefiltering."""

    def __init__(self, width):
        raise AssertionError("nested query constructed a stateful object")


def test_nested_exact_class_filters_unrelated_blocks_before_hydration(tmp_path, monkeypatch):
    from dryml.core.query import source

    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    try:
        target = Definition(NestedWorkModel).concretize()
        owner = Definition(NestedWorkBranch, "selected", [target, target]).concretize()
        selected = store.publish_node(owner)
        for index in range(12):
            store.publish_node(Definition(NestedWorkBranch, str(index), [Definition(NestedWorkBranch, "leaf", [])]).concretize())
        observed = []
        original = source._NodePage.root
        def read(page, selected_store, record):
            observed.append(record.block_digest)
            return original(page, selected_store, record)
        monkeypatch.setattr(source._NodePage, "root", read)
        monkeypatch.setattr(store, "read_states", lambda *a, **k: pytest.fail("nested query read payloads"))
        selector = Selector(Definition(NestedWorkModel), cls_policy="exact")
        assert repo.query().nested().sel(selector).count() == 2
        assert observed == []
        assert repo._num_constructions == 0
    finally:
        repo.close(flush=False)


def test_unbounded_occurrence_count_counts_diamond_paths_without_enumerating_them(monkeypatch):
    from dryml.core.query import relationships
    from dryml.core.node_prepare import NodePreparation

    leaf = Definition(NestedWorkModel).concretize()
    layer = (leaf, leaf)
    nodes = {id(leaf)}
    levels = 10
    for depth in range(levels):
        layer = tuple(Definition(NestedWorkBranch, f"{depth}-{side}", layer).concretize() for side in (0, 1))
        nodes.update(map(id, layer))
    root = Definition(NestedWorkBranch, "root", layer).concretize()
    nodes.add(id(root))
    visits = []
    original = relationships._ordered_direct_edges
    def read(value):
        visits.append(id(value))
        assert len(visits) <= 4 * len(nodes), "count expanded path multiplicity into repeated graph walks"
        return original(value)
    monkeypatch.setattr(relationships, "_ordered_direct_edges", read)
    query = IdentityQuery.from_set(IdentitySet((root,))).nested(leaf)
    preparations, prepare = [], NodePreparation.__init__

    def prepare_once(owner, *args, **kwargs):
        preparations.append(True)
        return prepare(owner, *args, **kwargs)

    monkeypatch.setattr(NodePreparation, "__init__", prepare_once)
    assert query.count() == 2 ** (levels + 1)
    assert len(visits) <= len(nodes)
    assert len(preparations) <= 16, "count rebuilt a graph preparation for each target admission"


def test_nested_prefilter_preserves_subclasses_caps_and_depth_failures(tmp_path):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    try:
        child = Definition(NestedWorkDerived).concretize()
        store.publish_node(Definition(NestedWorkBranch, "root", [child, child]).concretize())
        query = repo.query().nested().sel(Selector(Definition(NestedWorkModel)))
        assert query.count() == query.collect().count() == 2
        assert query.max_occurrences(1).count() == 1
        assert query.max_occurrences(0).count() == 0
        with pytest.raises(RelationshipDepthExceeded):
            query.max_depth(0).count()
        bounded = repo.query().take(1).query().nested().sel(Selector(Definition(NestedWorkModel)))
        assert bounded.count() == bounded.collect().count()
    finally:
        repo.close(flush=False)


@pytest.mark.parametrize("reverse", (False, True))
def test_occurrence_count_keeps_path_sensitive_cycle_protection(monkeypatch, reverse):
    from dryml.core.query import relationships
    from dryml.core.utils.graph.path import GraphPath, Parameter

    root, first, middle, earlier = (
        Definition(NestedWorkBranch, name, []).concretize() for name in ("root", "first", "middle", "earlier")
    )
    leaf = Definition(NestedWorkModel).concretize()
    graph = {id(root): (middle, first) if reverse else (first, middle), id(first): (middle,), id(middle): (earlier,), id(earlier): (leaf,), id(leaf): ()}
    original_key = relationships._node_key
    shared_guard = object()
    monkeypatch.setattr(relationships, "_node_key", lambda node: shared_guard if node is first or node is earlier else original_key(node))
    def edges(node):
        return tuple(relationships.DirectRelationship(relationships.RelationshipKind.MATERIALIZE,
                     GraphPath((Parameter(str(index)),)), child) for index, child in enumerate(graph[id(node)]))
    monkeypatch.setattr(relationships, "_ordered_direct_edges", edges)
    query = IdentityQuery.from_set(IdentitySet((root, middle))).nested(leaf)
    assert query.count() == query.collect().count() == 2


def test_nested_seek_upgrade_is_optional_without_changing_authority(tmp_path):
    from dryml.core.query.model import QueryWouldScanError

    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    try:
        child = Definition(NestedWorkModel).concretize()
        store.publish_node(Definition(NestedWorkBranch, "root", child).concretize())
        index = store.open_query_index()
        selector = Selector(Definition(NestedWorkModel), cls_policy="exact")
        assert repo.query().sel(selector).require_indexed().count() == 1
        assert repo.query().require_indexed().nested(selector).count() == 1
        con = index._connections.connection(readonly=False)
        con.execute("DROP INDEX IF EXISTS store_facet_class")
        before = store.authority_cut()
        with pytest.raises(QueryWouldScanError):
            repo.query().refresh(False).scan_policy("forbid").nested(selector).count()
        assert not con.execute("SELECT 1 FROM sqlite_master WHERE name='store_facet_class'").fetchone()
        assert repo.query().nested(selector).count() == 1
        assert con.execute("SELECT 1 FROM sqlite_master WHERE name='store_facet_class'").fetchone()
        assert store.authority_cut() == before
    finally:
        repo.close(flush=False)


def test_nested_prefilter_does_not_hide_connected_declaration_conflicts(tmp_path):
    from dryml.core.query import QueryError

    stores = [DirStore(tmp_path / name, query_index="sqlite") for name in ("first", "second")]
    repo = Repo(stores)
    try:
        identity = ObjectId()
        for width, store in zip((32, 64), stores):
            model = Definition(NestedWorkState, width).concretize().with_identity(identity)
            store.publish_node(Definition(NestedWorkBranch, "root", model).concretize())
        selector = Selector(Definition(NestedWorkState, 32), cls_policy="exact")
        with pytest.raises(QueryError, match="conflict"):
            repo.query().nested(selector).count()
    finally:
        repo.close(flush=False)


def test_snapshot_candidates_preserve_associations_references_and_duplicate_contributions(tmp_path, fixed_snapshot_environment):
    from dryml.core.query import EdgePolicy, RelationshipKind
    from tests.core.test_query_decode_work import DecodeState, DecodeValue

    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    try:
        model = DecodeState(DecodeValue("target", repo=repo), repo=repo)
        parent = DecodeState([model, model], repo=repo)
        for progress in range(3):
            parent.progress = progress
            repo.save(parent)
    finally:
        repo.close(flush=False)
    repo = Repo(DirStore.open_existing(tmp_path / "store"))
    try:
        roots = repo.query().collect().query()
        selector = Selector(Definition(DecodeState, Definition(DecodeValue, "target")), cls_policy="exact")
        for edges in (EdgePolicy.ALL, EdgePolicy.OWNED, EdgePolicy.ASSOCIATIONS,
                      EdgePolicy(frozenset((RelationshipKind.REFERENCE,)))):
            reference = roots.nested(selector, edges=edges).collect()
            query = repo.query().page_size(1).nested(selector, edges=edges)
            result = query.collect()
            assert {item.key for item in result} == {item.key for item in reference}
            assert query.count() == reference.count()
        assert repo._num_constructions == 0
    finally:
        repo.close(flush=False)


@pytest.mark.parametrize("change", ("authority", "index", "interrupt"))
def test_clustered_root_capture_releases_its_address_connection_on_retry_or_interrupt(tmp_path, monkeypatch, change):
    import sqlite3
    from dryml.core.query import source

    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    connections, changed = [], []
    original_open, original_expand = sqlite3.connect, source._NodePage.expand

    def observe(*args, **kwargs):
        con = original_open(*args, **kwargs)
        if args[0] == ":memory:":
            connections.append(con)
        return con

    def expand(page, *args, **kwargs):
        result = original_expand(page, *args, **kwargs)
        if not changed:
            changed.append(True)
            if change == "authority":
                store.publish_node(Definition(NestedWorkBranch, "after", Definition(NestedWorkModel)).concretize())
            elif change == "index":
                store.rebuild_query_index()
            else:
                raise KeyboardInterrupt("controlled nested capture interruption")
        return result

    try:
        store.publish_node(Definition(NestedWorkBranch, "before", Definition(NestedWorkModel)).concretize())
        monkeypatch.setattr(sqlite3, "connect", observe)
        monkeypatch.setattr(source._NodePage, "expand", expand)
        query = repo.query().page_size(1).nested(Selector(Definition(NestedWorkModel), cls_policy="exact")).max_occurrences(100)
        if change == "interrupt":
            with pytest.raises(KeyboardInterrupt, match="controlled nested"):
                query.count()
        else:
            assert query.count() == (2 if change == "authority" else 1)
        assert connections
        for con in connections:
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                con.execute("SELECT 1")
        assert store.open_query_index().status().state == "ready"
    finally:
        repo.close(flush=False)


def test_owner_cache_eviction_is_byte_bounded_and_oversized_entries_are_not_retained():
    from dryml.core.query.source import _NodeOwnerCache

    cache = _NodeOwnerCache(8, 8)
    cache.retain("first", object(), 3)
    cache.retain("second", object(), 3)
    first = cache.get("first")
    cache.retain("third", object(), 3)
    assert cache.get("first") is first and cache.get("second") is None
    cache.retain("large", object(), 9)
    assert cache.get("large") is None and cache.bytes == 6
    cache = _NodeOwnerCache(8, 8, protect_blocks=True)
    cache.retain(b"current block", first, 5)
    cache.retain(("projection", 1), object(), 3)
    cache.retain(("projection", 2), object(), 4)
    assert cache.get(b"current block") is first and cache.bytes == 5
    cache.retain(b"next block", object(), 5)
    assert cache.get(b"current block") is None and cache.bytes == 5


def test_optional_class_index_does_not_block_an_existing_scalar_seek(tmp_path):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    try:
        store.publish_node(Definition(NestedWorkBranch, "target", []).concretize())
        index = store.open_query_index()
        con = index._connections.connection(readonly=False)
        con.execute("DROP INDEX store_facet_class")
        query = store.query().sel(Selector(Definition(NestedWorkBranch, "target"), cls_policy="exact"))
        assert query.scan_policy("forbid").count() == 1
        assert query.require_indexed().count() == 1
        assert not index._has_class_seek()
    finally:
        store.close()


def test_clustered_capture_keeps_alias_distinctions_under_digest_collisions(tmp_path, monkeypatch):
    from dryml.core.node_prepare import NodePreparation

    original = NodePreparation.digest

    def collision(prepared, node, *, domain="dryml-node-graph/v1"):
        return "0" * 64 if domain == "dryml-node-graph/v1" else original(prepared, node, domain=domain)

    monkeypatch.setattr(NodePreparation, "digest", collision)
    store = DirStore(tmp_path / "store", query_index="sqlite")
    try:
        leaf = Definition(NestedWorkModel).concretize()
        shared = Definition(NestedWorkBranch, "same", [leaf, leaf]).concretize()
        separate = Definition(NestedWorkBranch, "same", [Definition(NestedWorkModel), Definition(NestedWorkModel)]).concretize()
        assert not shared.graph_equal(separate)
        store.publish_node(shared)
        store.publish_node(separate)
        query = store.query().page_size(1).nested(Selector(Definition(NestedWorkModel), cls_policy="exact"))
        assert query.count() == query.collect().count() == 4
        assert query.owners().count() == 2
    finally:
        store.close()


def test_runtime_matchers_keep_per_path_streaming_evaluation(monkeypatch):
    from dryml.core.params import Satisfies
    from dryml.core.query import relationships

    leaf = Definition(NestedWorkBranch, "leaf", []).concretize()
    branch = Definition(NestedWorkBranch, "branch", [leaf, leaf]).concretize()
    root = Definition(NestedWorkBranch, "root", [branch, branch]).concretize()
    calls = []

    def alternating(value):
        calls.append(value)
        return value == "leaf" and len(calls) % 2 == 0

    query = IdentitySet((root,)).query().nested(Definition(NestedWorkBranch, Satisfies(alternating)))
    monkeypatch.setattr(relationships, "count_relationship_occurrences",
                        lambda *a, **k: pytest.fail("runtime matcher entered pure DAG counting"))
    expected = query.collect().count()
    expected_calls = list(calls)
    calls.clear()
    assert query.count() == expected and calls == expected_calls and len(calls) == 6


def test_pure_factory_patterns_keep_nested_candidate_planning(tmp_path):
    from dryml.core import F, Par
    from dryml.core.query.query import _V3Restriction
    from dryml.core.query.source import SourceCapture

    pattern = Definition(NestedWorkBranch, "model", [F("Dense", units=Par("width"))]).loose_selector()
    assert not _V3Restriction("sel", pattern).prepared.runtime_matchers
    store = DirStore(tmp_path / "store", query_index="sqlite")
    try:
        query = store.query().nested(pattern)
        candidates = query._candidate_roots(SourceCapture(query.roots))
        assert any(item.kind == "nested_candidates" for item in candidates.restrictions)
    finally:
        store.close()
