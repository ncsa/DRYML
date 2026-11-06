"""Exact postings counts, bounded graph work and fresh authority obligations."""

import json

import pytest

from dryml.core import Definition, F, Object, Par, Repo, Selector
from dryml.core.store.dir import DirStore


class PostingModel(Object):
    """Carry an ordered factory configuration without invoking a backend."""

    def __init__(self, layers, label="model"):
        self.layers, self.label = layers, label


class PostingOwner(Object):
    """Preserve shared children and distinguish independent graph roots."""

    def __init__(self, children, label="owner"):
        self.children, self.label = children, label


def model(width=8, output=2):
    return Definition(PostingModel, [F("Dense", units=width, activation="relu"),
                                     F("Dense", units=output, activation="linear")])


def test_factory_postings_count_never_decodes_nodes_and_matches_reference(tmp_path, monkeypatch):
    import dryml.core.node_codec as codec
    from dryml.core.query.indexed_count import _ProofReader

    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    try:
        selected, rejected = [], []
        for width, output in ((8, 2), (16, 2), (32, 3)):
            child = model(width, output).concretize()
            record = store.publish_node(Definition(PostingOwner, [child, child], str(width)).concretize())
            (selected if output == 2 else rejected).append(record.block_digest)
        predicate = model(Par("width")).loose_selector()
        expected = repo.query().collect().query().nested(predicate).collect().count()
        blocks, original = [], _ProofReader.graph

        def graph(reader, selected_store, digest):
            blocks.append(digest)
            return original(reader, selected_store, digest)

        monkeypatch.setattr(_ProofReader, "graph", graph)
        monkeypatch.setattr(codec, "_decode_node_graph_views", lambda *a, **k: pytest.fail("count decoded a Node graph"))
        monkeypatch.setattr(store, "read_states", lambda *a, **k: pytest.fail("count read state payloads"))
        assert repo.query().refresh(False).nested(predicate).count() == expected == 4
        assert set(blocks) == set(selected)
        assert not set(blocks).intersection(rejected)
        assert repo._num_constructions == 0
    finally:
        repo.close(flush=False)


def test_snapshot_count_reuses_certified_state_and_named_view_validation(tmp_path, monkeypatch, fixed_snapshot_environment):
    from tests.core.test_query_decode_work import DecodeState, DecodeValue
    from dryml.core.state_values import LocalState
    import dryml.core.node_codec as codec

    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    try:
        child = DecodeState(DecodeValue("target", repo=repo), repo=repo)
        parent = DecodeState([child, child], repo=repo)
        for progress in range(3):
            parent.progress = progress
            repo.save(parent)
    finally:
        repo.close(flush=False)
    repo = Repo(DirStore.open_existing(tmp_path / "store"))
    try:
        predicate = Selector(Definition(DecodeState, Definition(DecodeValue, "target")), cls_policy="exact")
        expected = repo.query().collect().query().nested(predicate).collect().count()
        monkeypatch.setattr(codec, "_decode_node_graph_views", lambda *a, **k: pytest.fail("reconstructed snapshot metadata"))
        monkeypatch.setattr(LocalState, "validate_for", lambda *a, **k: pytest.fail("repeated immutable state compatibility"))
        assert repo.query().refresh(False).nested(predicate).count() == expected
    finally:
        repo.close(flush=False)


def test_stateless_saved_presentations_do_not_multiply_occurrences(tmp_path):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    try:
        child = model().concretize()
        store.publish_node(Definition(PostingOwner, [child, child]).concretize(), kind="saved")
        query = store.query().nested(model(Par("width")).loose_selector())
        assert query.collect().count() == query.count() == 2
        assert store.query().collect().query().nested(model(Par("width")).loose_selector()).count() == 2
    finally:
        store.close()


def test_local_state_roots_keep_state_creation_traversal(tmp_path, fixed_snapshot_environment):
    from tests.core.test_query_decode_work import DecodeState, DecodeValue

    repo = Repo(DirStore(tmp_path / "store", query_index="sqlite"))
    try:
        repo.save(DecodeState(DecodeValue("target", repo=repo), repo=repo))
        selector = Selector(Definition(DecodeValue, "target"), cls_policy="exact")
        query = repo.query().local_states().nested(selector)
        assert query.count() == query.collect().count() > 0
    finally:
        repo.close(flush=False)


@pytest.mark.parametrize("pattern", ("shared", "separate", "factory", "partial", "missing", "choice", "boolean"))
def test_primitive_matching_agrees_with_node_residual(tmp_path, pattern):
    from dryml.core import AnyValue, Missing, Choice
    from dryml.core.selector import QueryPredicate
    from dryml.core.query.query import _V3Restriction
    from dryml.core.query.postings import _compile, _read_graph

    child = model().concretize()
    values = [Definition(PostingOwner, [child, child]).concretize(),
              Definition(PostingOwner, [child, child.copy_graph()]).concretize(),
              model().concretize(), model(32, 3).concretize()]
    shared = model(Par("width")).loose_selector().root
    patterns = {
        "shared": Definition(PostingOwner, [shared, shared]),
        "separate": Definition(PostingOwner, [shared, shared.copy_graph()]),
        "factory": model(Par("width")).loose_selector().root,
        "partial": Definition(PostingModel, label=AnyValue()),
        "missing": Definition(PostingModel, missing=Missing()),
        "choice": Definition(PostingModel, label=Choice(["model", "other"])),
        "boolean": QueryPredicate(Definition(PostingModel)) | QueryPredicate(Definition(PostingOwner)),
    }
    selected = patterns[pattern]
    if pattern == "boolean":
        # Concrete leaves establish their own exact class semantics in Boolean plans.
        selected = QueryPredicate(values[0]) | QueryPredicate(values[2])
    else:
        selected = Selector(selected, cls_policy="exact")
    prepared = _V3Restriction("sel", selected).prepared
    plan = _compile(prepared)
    assert plan.residual is None
    store = DirStore(tmp_path / "store", query_index="sqlite")
    try:
        for value in values:
            record = store.publish_node(value)
            path = store._relative(store._node_path("definitions/blocks", record.block_digest))
            con = store.open_query_index()._connections.connection(readonly=True)
            envelope = json.loads(con.execute("SELECT payload FROM store_projection_rows WHERE family='query_graph' AND source=?", (path,)).fetchone()[0])
            graph = _read_graph(envelope)
            slot = graph["locators"][record.view["table"] + ":" + str(record.view["slot"])]
            assert plan.matches(graph, slot) == prepared.matches(value)
    finally:
        store.close()


@pytest.mark.parametrize("damage", ("block", "membership", "declaration"))
def test_count_detects_same_epoch_authority_damage_after_a_success(tmp_path, damage):
    from pathlib import Path
    from dryml.core import ObjectId
    from dryml.core.query.model import QueryError
    from dryml.core.store.store import StoreCapabilityError, StoreAuthorityError

    child = model().concretize().with_identity(ObjectId())
    store = DirStore(tmp_path / "store", query_index="sqlite")
    try:
        root = store.publish_node(Definition(PostingOwner, [child]).concretize())
        query = store.query().refresh(False).nested(model(Par("width")).loose_selector())
        assert query.count() > 0
        if damage == "block":
            path = store._node_path("definitions/blocks", root.block_digest)
        elif damage == "membership":
            path = store._node_path("membership/recipe", root.digest)
        else:
            path = store._node_path("declarations/required", store._anchor_key(child.object_id))
        before = store.authority_cut()
        Path(path).unlink()
        assert store.authority_cut() == before
        with pytest.raises((QueryError, StoreCapabilityError, StoreAuthorityError)):
            query.count()
    finally:
        store.close()


@pytest.mark.parametrize("change", ("authority", "index", "interrupt"))
def test_postings_count_recaptures_changes_and_releases_readers(tmp_path, monkeypatch, change):
    from dryml.core.query.indexed_count import _ProofReader

    store = DirStore(tmp_path / "store", query_index="sqlite")
    original, changed = _ProofReader.membership, []

    def membership(reader, *args):
        result = original(reader, *args)
        if not changed:
            changed.append(True)
            if change == "authority":
                store.publish_node(Definition(PostingOwner, [model()], "later").concretize())
            elif change == "index":
                store.rebuild_query_index()
            else:
                raise KeyboardInterrupt("controlled count interruption")
        return result

    try:
        store.publish_node(Definition(PostingOwner, [model()]).concretize())
        monkeypatch.setattr(_ProofReader, "membership", membership)
        query = store.query().nested(model(Par("width")).loose_selector())
        if change == "interrupt":
            with pytest.raises(KeyboardInterrupt, match="controlled count"):
                query.count()
        else:
            assert query.count() == (2 if change == "authority" else 1)
        assert store.open_query_index().status().state == "ready"
        assert not store.open_query_index()._connections.connection(readonly=True).in_transaction
    finally:
        store.close()


def test_compact_dag_count_shares_suffixes_without_expanding_paths():
    from dryml.core.query.postings import _count_paths

    nodes = [{"guard": 0, "edges": []}]
    previous = [0, 0]
    for level in range(20):
        start = len(nodes)
        nodes.extend({"guard": start + side, "edges": [("materialize", child) for child in previous]} for side in range(2))
        previous = [start, start + 1]
    root = len(nodes)
    nodes.append({"guard": root, "edges": [("materialize", child) for child in previous]})
    calls = []

    def matches(slot):
        calls.append(slot)
        assert len(calls) < 200, "compact count enumerated shared paths"
        return slot == 0

    assert _count_paths({"nodes": nodes}, root, matches, {"materialize"}) == 2 ** 21


@pytest.mark.parametrize("envelope", ({"version": True, "supported": False},
                                     {"version": 2, "supported": False},
                                     {"version": 1, "supported": True, "size": 2 ** 40, "data": ""},
                                     {"version": 1, "supported": True, "size": 12, "data": "invalid"}))
def test_invalid_graph_envelopes_refuse_without_unbounded_decompression(envelope):
    from dryml.core.query.postings import _read_graph
    from dryml.core.query.model import QueryIndexDirty

    with pytest.raises(QueryIndexDirty, match="Malformed certified"):
        _read_graph(envelope)


def test_empty_postings_count_checks_all_coverage_families(tmp_path):
    from dryml.core.query.model import QueryIndexDirty

    store = DirStore(tmp_path / "store", query_index="sqlite")
    try:
        store.publish_node(Definition(PostingOwner, []).concretize())
        query = store.query().refresh(False).nested(model(Par("width")).loose_selector())
        assert query.count() == 0
        con = store.open_query_index()._connections.connection(readonly=False)
        con.execute("DELETE FROM store_projection_rows WHERE family='query_graph'")
        with pytest.raises(QueryIndexDirty):
            query.count()
    finally:
        store.close()


@pytest.mark.timeout(10)
def test_quoted_factory_graph_keeps_residual_without_unbounded_projection(tmp_path):
    from dryml.core.query.postings import _compile
    from dryml.core.query.query import _V3Restriction

    nested = Definition(PostingModel, [model().quote()])
    pattern = nested.loose_selector()
    assert _compile(_V3Restriction("sel", pattern).prepared).residual is not None
    store = DirStore(tmp_path / "store", query_index="sqlite")
    try:
        store.publish_node(Definition(PostingOwner, [nested]).concretize())
        query = store.query().nested(pattern)
        assert query.count() == query.collect().count() == 1
    finally:
        store.close()


def test_nonpersistent_scan_does_not_build_unused_certificates(tmp_path, monkeypatch):
    import dryml.core.query.postings as postings

    store = DirStore(tmp_path / "store", query_index="none")
    try:
        store.publish_node(Definition(PostingOwner, [model()]).concretize())
        monkeypatch.setattr(postings, "_build_graph", lambda *a, **k: pytest.fail("scan built unused certificates"))
        assert store.query().nested(model(Par("width")).loose_selector()).count() == 1
    finally:
        store.close()
