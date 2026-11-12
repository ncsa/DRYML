"""Exact postings counts, bounded graph work and fresh authority obligations."""

import json

import pytest

from dryml.core import Definition, F, Node, Object, Par, Ref, Repo, Selector
from dryml.core.store.dir import DirStore
from tests.core.test_query_decode_work import DecodeState


class PostingModel(Object):
    """Carry an ordered factory configuration without invoking a backend."""

    def __init__(self, layers, label="model"):
        self.layers, self.label = layers, label


class PostingOwner(Object):
    """Preserve shared children and distinguish independent graph roots."""

    def __init__(self, children, label="owner"):
        self.children, self.label = children, label


class PostingQuotedState(DecodeState):
    """Retain a quoted Node recipe as data in state creation evidence."""

    def __init__(self, value: Ref[Node]):
        super().__init__(value)


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
def test_unordered_quoted_graph_keeps_residual_without_unbounded_projection(tmp_path):
    from dryml.core.query.postings import _compile
    from dryml.core.query.query import _V3Restriction

    nested = Definition(PostingModel, [Definition(PostingOwner, {model().concretize()}).quote()])
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


def test_direct_factory_identity_count_uses_no_node_decoding(tmp_path, monkeypatch):
    import dryml.core.node_codec as codec
    from dryml.core.query.indexed_count import _ProofReader

    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    try:
        wanted_blocks, excluded_blocks = set(), set()
        for number, (width, output) in enumerate(((8, 2), (8, 2), (16, 2), (32, 3))):
            value = model(width, output).concretize()
            record = store.publish_node(Definition(PostingOwner, [value, value], str(number)).concretize())
            (wanted_blocks if output == 2 else excluded_blocks).add(record.block_digest)
        selector = model(Par("width")).loose_selector()
        query = repo.query().sel(selector)
        expected = query.collect().count()
        observed, original = [], _ProofReader.graph

        def graph(reader, selected_store, digest):
            observed.append(digest)
            return original(reader, selected_store, digest)

        monkeypatch.setattr(_ProofReader, "graph", graph)
        monkeypatch.setattr(codec, "_decode_node_graph_views", lambda *a, **k: pytest.fail("identity count decoded a Node graph"))
        assert query.refresh(False).require_indexed().count() == expected == 2
        assert set(observed) == wanted_blocks
        assert not set(observed).intersection(excluded_blocks)
        assert repo._num_constructions == 0
    finally:
        repo.close(flush=False)


@pytest.mark.parametrize("domain", ("default", "cdefs", "object_refs", "saved"))
def test_direct_identity_count_preserves_saved_domain_presentations(tmp_path, monkeypatch, fixed_snapshot_environment, domain):
    from tests.core.test_query_decode_work import DecodeState, DecodeValue
    import dryml.core.node_codec as codec

    repo = Repo(DirStore(tmp_path / "store", query_index="sqlite"))
    try:
        child = DecodeState(DecodeValue("target", repo=repo), repo=repo)
        parent = DecodeState([child, child], repo=repo)
        for progress in range(2):
            child.progress, parent.progress = progress, progress
            repo.save(parent)
    finally:
        repo.close(flush=False)
    repo = Repo(DirStore.open_existing(tmp_path / "store"))
    try:
        selector = Selector(Definition(DecodeState, Definition(DecodeValue, "target")), cls_policy="exact")
        query = repo.query() if domain == "default" else getattr(repo.query(), domain)()
        query = query.sel(selector)
        expected = query.collect().count()
        monkeypatch.setattr(codec, "_decode_node_graph_views", lambda *a, **k: pytest.fail("direct domain count decoded a Node"))
        assert query.refresh(False).count() == expected > 0
    finally:
        repo.close(flush=False)


def test_direct_identity_count_merges_cached_and_stored_members(tmp_path, monkeypatch):
    from tests.core.test_query_decode_work import DecodeValue
    import dryml.core.node_codec as codec

    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    try:
        cached = [DecodeValue("duplicate", repo=repo), DecodeValue("cache-only", repo=repo)]
        store.publish_node(Definition(DecodeValue, "duplicate").concretize())
        selector = Selector(Definition(DecodeValue), cls_policy="exact")
        query = repo.query().sel(selector)
        expected = query.collect().count()
        assert cached
        monkeypatch.setattr(codec, "_decode_node_graph_views", lambda *a, **k: pytest.fail("cached identity count decoded Store Nodes"))
        assert query.count() == expected
    finally:
        repo.close(flush=False)


@pytest.mark.parametrize("damage", ("block", "membership", "declaration"))
def test_direct_identity_count_retains_fresh_authority_validation(tmp_path, damage):
    from pathlib import Path
    from dryml.core import ObjectId
    from dryml.core.query.model import QueryError
    from dryml.core.store.store import StoreCapabilityError, StoreAuthorityError

    store = DirStore(tmp_path / "store", query_index="sqlite")
    try:
        child = model().concretize().with_identity(ObjectId())
        root = store.publish_node(Definition(PostingOwner, [child]).concretize())
        query = store.query().refresh(False).sel(model(Par("width")).loose_selector())
        assert query.count() > 0
        path = (store._node_path("definitions/blocks", root.block_digest) if damage == "block" else
                store._node_path("membership/recipe", root.digest) if damage == "membership" else
                store._node_path("declarations/required", store._anchor_key(child.object_id)))
        before = store.authority_cut()
        Path(path).unlink()
        assert store.authority_cut() == before
        with pytest.raises((QueryError, StoreCapabilityError, StoreAuthorityError)):
            query.count()
    finally:
        store.close()


@pytest.mark.parametrize("change", ("authority", "index", "interrupt"))
def test_direct_identity_count_recaptures_and_releases_readers(tmp_path, monkeypatch, change):
    from dryml.core.query.indexed_count import _ProofReader

    store = DirStore(tmp_path / "store", query_index="sqlite")
    original, changed = _ProofReader.membership, []

    def membership(reader, *args):
        result = original(reader, *args)
        if not changed:
            changed.append(True)
            if change == "authority":
                store.publish_node(Definition(PostingOwner, [model(16)], "later").concretize())
            elif change == "index":
                store.rebuild_query_index()
            else:
                raise KeyboardInterrupt("controlled identity count interruption")
        return result

    try:
        store.publish_node(Definition(PostingOwner, [model()]).concretize())
        monkeypatch.setattr(_ProofReader, "membership", membership)
        query = store.query().sel(model(Par("width")).loose_selector())
        if change == "interrupt":
            with pytest.raises(KeyboardInterrupt, match="controlled identity"):
                query.count()
        else:
            assert query.count() == (2 if change == "authority" else 1)
        assert not store.open_query_index()._connections.connection(readonly=True).in_transaction
    finally:
        store.close()


def test_direct_identity_count_keeps_verification_bounds_and_callback_fallback(tmp_path, monkeypatch):
    from dryml.core import Satisfies
    from dryml.core.query.model import QueryVerifyBudgetExceeded
    import dryml.core.query.indexed_count as indexed

    store = DirStore(tmp_path / "store", query_index="sqlite")
    try:
        store.publish_node(model().concretize())
        query = store.query().sel(model(Par("width")).loose_selector())
        with pytest.raises(QueryVerifyBudgetExceeded):
            query.max_verify(0).count()
        monkeypatch.setattr(indexed, "_count", lambda *a, **k: pytest.fail("unsupported identity predicate used compact count"))
        calls = []
        runtime = Selector(Definition(PostingModel, label=Satisfies(lambda value: calls.append(value) or value == "model")), cls_policy="exact")
        assert store.query().sel(runtime).count() == 1 and calls == ["model"]
    finally:
        store.close()


def test_direct_identity_count_keeps_digest_collisions_separate(tmp_path, monkeypatch):
    from dryml.core.node_prepare import NodePreparation
    import dryml.core.node_codec as codec

    original = NodePreparation.digest

    def digest(prepared, node, *, domain="dryml-node-graph/v1"):
        return "0" * 64 if domain == "dryml-node-graph/v1" else original(prepared, node, domain=domain)

    monkeypatch.setattr(NodePreparation, "digest", digest)
    store = DirStore(tmp_path / "store", query_index="sqlite")
    try:
        store.publish_node(model(8).concretize())
        store.publish_node(model(16).concretize())
        query = store.query().sel(model(Par("width")).loose_selector())
        expected = query.collect().count()
        monkeypatch.setattr(codec, "_decode_node_graph_views", lambda *a, **k: pytest.fail("collision count decoded Nodes"))
        assert query.count() == expected == 2
    finally:
        store.close()


def test_direct_identity_count_retains_distinct_stateless_presentation_kinds(tmp_path, monkeypatch):
    import dryml.core.node_codec as codec

    store = DirStore(tmp_path / "store", query_index="sqlite")
    try:
        store.publish_node(model().concretize(), kind="saved")
        selector = model(Par("width")).loose_selector()
        query = store.query().sel(selector)
        expected = query.collect()
        assert {key.kind for key in expected._entries} == {"cdef", "object_ref", "state_ref"}
        monkeypatch.setattr(codec, "_decode_node_graph_views", lambda *a, **k: pytest.fail("presentation count decoded Nodes"))
        assert query.count() == expected.count() == 3
        assert store.query().cdefs().sel(selector).count() == 1
        assert store.query().object_refs().sel(selector).count() == 1
        assert store.query().saved().sel(selector).count() == 1
    finally:
        store.close()


@pytest.mark.parametrize("damage", ("none", "conflict", "missing"))
def test_direct_identity_count_checks_declarations_in_nonmatching_stores(tmp_path, monkeypatch, damage):
    from pathlib import Path
    from dryml.core import ObjectId
    from dryml.core.query.model import QueryError
    from dryml.core.store.store import StoreCapabilityError
    import dryml.core.node_codec as codec

    first, second = [DirStore(tmp_path / name, query_index="sqlite") for name in ("first", "second")]
    identity = ObjectId()
    selected = model().concretize().with_identity(identity)
    first.publish_node(selected)
    other = model(output=3).concretize().with_identity(identity) if damage == "conflict" else selected
    second.publish_node(other)
    repo = Repo((first, second))
    try:
        if damage == "missing":
            Path(second._node_path("declarations/anchors", second._anchor_key(identity))).unlink()
        query = repo.query().sel(model(Par("width")).loose_selector())
        if damage == "none":
            expected = query.collect().count()
            monkeypatch.setattr(codec, "_decode_node_graph_views", lambda *a, **k: pytest.fail("replicated count decoded Nodes"))
            assert query.count() == expected
        else:
            with pytest.raises((QueryError, StoreCapabilityError), match="anchor|conflict"):
                query.count()
    finally:
        repo.close(flush=False)


def test_direct_empty_identity_count_checks_coverage_before_answering(tmp_path):
    from dryml.core.query.model import QueryIndexDirty

    store = DirStore(tmp_path / "store", query_index="sqlite")
    try:
        store.publish_node(model(output=3).concretize())
        query = store.query().refresh(False).sel(model(Par("width")).loose_selector())
        assert query.count() == 0
        con = store.open_query_index()._connections.connection(readonly=False)
        con.execute("DELETE FROM store_projection_rows WHERE family='query_graph'")
        with pytest.raises(QueryIndexDirty):
            query.count()
    finally:
        store.close()


def test_pinned_child_state_counts_without_decoding_and_keeps_other_versions_out(tmp_path, monkeypatch, fixed_snapshot_environment):
    from tests.core.test_query_decode_work import DecodeState, DecodeValue
    from dryml.core.query.postings import _compile
    from dryml.core.query.query import _V3Restriction
    import dryml.core.node_codec as codec

    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    try:
        dataset = DecodeState(DecodeValue("large-test-data", repo=repo), repo=repo)
        first = repo.save(dataset)
        dataset.progress = 1
        second = repo.save(dataset)
        assert first.object_id == second.object_id and first.state.key != second.state.key
        for index, saved in enumerate((first, second)):
            store.publish_node(Definition(PostingOwner, [model(8), saved], str(index)).concretize())
        pattern = Definition(PostingOwner, [model(Par("width")), first]).loose_selector()
        assert _compile(_V3Restriction("sel", pattern).prepared).residual is None
        direct = repo.query().cdefs().sel(pattern)
        nested = repo.query().nested(first)
        expected_direct, expected_nested = direct.collect().count(), nested.collect().count()
        assert expected_direct == 1 and expected_nested > 0
        monkeypatch.setattr(codec, "_decode_node_graph_views", lambda *a, **k: pytest.fail("pinned state count decoded a Node graph"))
        monkeypatch.setattr(store, "read_states", lambda *a, **k: pytest.fail("pin comparison read private state"))
        assert direct.count() == expected_direct
        assert nested.count() == expected_nested
        assert repo.query().sel(first).count() == repo.query().sel(second).count() == 1
    finally:
        repo.close(flush=False)


def test_state_pin_matches_full_creation_evidence_not_just_the_key(tmp_path, monkeypatch, fixed_snapshot_environment):
    from tests.core.test_query_decode_work import DecodeState, DecodeValue
    from dryml.core.state_values import LocalState
    import dryml.core.node_codec as codec

    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    try:
        saved = repo.save(DecodeState(DecodeValue("original", repo=repo), repo=repo))
        copied = saved.with_state(LocalState(saved.state.key, saved.state.creation_cdef.copy_graph()))
        other_creation = Definition(DecodeState, Definition(DecodeValue, "different")).concretize()
        conflicting = saved.without_state().with_state(LocalState(saved.state.key, other_creation))
        assert Selector(copied).matches(saved)
        assert not Selector(conflicting).matches(saved)
        monkeypatch.setattr(codec, "_decode_node_graph_views", lambda *a, **k: pytest.fail("state creation proof decoded Nodes"))
        assert repo.query().sel(copied).count() == 1
        assert repo.query().sel(conflicting).count() == 0
    finally:
        repo.close(flush=False)


def test_loose_experiment_pattern_retains_fixed_dataset_sizes_and_state_pin(tmp_path, monkeypatch, fixed_snapshot_environment):
    from tests.core.test_query_decode_work import DecodeState, DecodeValue
    import dryml.core.node_codec as codec

    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    try:
        large = repo.save(DecodeState(DecodeValue(20000, repo=repo), repo=repo))
        small = repo.save(DecodeState(DecodeValue(2000, repo=repo), repo=repo))
        train_large, train_small = [Definition(DecodeValue, size).concretize() for size in (100000, 10000)]
        for index, (train, test, width) in enumerate(((train_large, large, 8), (train_large, large, 16),
                                                    (train_small, small, 8), (train_small, large, 8),
                                                    (train_large, small, 8))):
            store.publish_node(Definition(PostingOwner, [model(width), train, test], str(index)).concretize())
        pattern = Definition(PostingOwner, [model(Par("width")), train_large, large]).loose_selector()
        query = repo.query().cdefs().sel(pattern)
        assert pattern.root.parameters["children"][2] is large
        assert query.collect().count() == 2
        monkeypatch.setattr(codec, "_decode_node_graph_views", lambda *a, **k: pytest.fail("experiment dataset pin decoded Nodes"))
        assert query.count() == 2
    finally:
        repo.close(flush=False)


@pytest.mark.parametrize("case", ("soft_same", "soft_extra", "soft_omission", "concrete_same", "concrete_identity", "node_kind"))
def test_posted_quotation_upgrades_earlier_alias_proofs_exactly(tmp_path, case):
    from dryml.core import ObjectId, SKIP_ARGS
    from dryml.core.query.postings import _compile, _read_graph
    from dryml.core.query.query import _V3Restriction

    if case.startswith("concrete"):
        expected = model().concretize()
        actual = expected.copy_graph()
        if case == "concrete_identity":
            actual = actual.with_identity(ObjectId())
    elif case == "soft_omission":
        expected = Definition(PostingModel, SKIP_ARGS, label="model")
        actual = Definition(PostingModel, label="model")
    else:
        expected = Definition(PostingModel, label="model")
        actual = expected.copy_graph() if case == "soft_same" else Definition(PostingModel, layers=[], label="model")
        if case == "node_kind":
            actual = model().concretize()
    pattern = Selector(Definition(PostingOwner, children=[expected, expected.quote()]), cls_policy="exact")
    candidate = Definition(PostingOwner, children=[actual, actual.quote()])
    prepared = _V3Restriction("sel", pattern).prepared
    plan = _compile(prepared)
    assert plan.residual is None
    reference = prepared.matches(candidate)
    assert reference == (case in {"soft_same", "concrete_same"})
    store = DirStore(tmp_path / "store", query_index="sqlite")
    try:
        record = store.publish_node(candidate)
        path = store._relative(store._node_path("definitions/blocks", record.block_digest))
        con = store.open_query_index()._connections.connection(readonly=True)
        envelope = json.loads(con.execute("SELECT payload FROM store_projection_rows WHERE family='query_graph' AND source=?", (path,)).fetchone()[0])
        graph = _read_graph(envelope)
        slot = graph["locators"][record.view["table"] + ":" + str(record.view["slot"])]
        assert plan.matches(graph, slot) == reference
    finally:
        store.close()


def test_ordered_quote_and_state_pin_execute_together_without_decoding(tmp_path, monkeypatch, fixed_snapshot_environment):
    from tests.core.test_query_decode_work import DecodeState, DecodeValue
    import dryml.core.node_codec as codec

    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    try:
        source = Definition(DecodeValue, {"n": 20000}).concretize()
        stateful = PostingQuotedState(source, repo=repo)
        saved = repo.save(stateful)
        stateful.progress = 1
        other_state = repo.save(stateful)
        for index, pinned in enumerate((saved, other_state)):
            store.publish_node(Definition(PostingOwner, [pinned, source.quote()], str(index)).concretize())
        pattern = Definition(PostingOwner, [saved, source.quote()]).loose_selector()
        query = repo.query().cdefs().sel(pattern)
        expected = query.collect().count()
        assert expected == 1
        monkeypatch.setattr(codec, "_decode_node_graph_views", lambda *a, **k: pytest.fail("quoted dataset state count decoded Nodes"))
        assert query.count() == expected
    finally:
        repo.close(flush=False)


def test_quoted_parameters_remain_literal_data_not_loose_wildcards(tmp_path, monkeypatch):
    import dryml.core.node_codec as codec

    store = DirStore(tmp_path / "store", query_index="sqlite")
    try:
        wanted = Definition(PostingModel, label=Par("this.test_data"))
        for index, value in enumerate((wanted, Definition(PostingModel, label=Par("this.train_data")),
                                       Definition(PostingModel, label="literal"))):
            store.publish_node(Definition(PostingOwner, [value.quote()], str(index)).concretize())
        selector = Definition(PostingOwner, [wanted.quote()]).loose_selector()
        query = store.query().cdefs().sel(selector)
        expected = query.collect().count()
        assert expected == 1
        monkeypatch.setattr(codec, "_decode_node_graph_views", lambda *a, **k: pytest.fail("quoted parameter comparison decoded Nodes"))
        assert query.count() == expected
    finally:
        store.close()


def test_certified_collection_decodes_only_matching_blocks_and_merges_evidence(tmp_path, monkeypatch, fixed_snapshot_environment):
    import dryml.core.node_codec as codec
    import dryml.core.query.indexed_count as indexed

    first, second = [DirStore(tmp_path / name, query_index="sqlite") for name in ("first", "second")]
    writer = Repo(first)
    selected = writer.save(DecodeState(model(), repo=writer))
    first.publish_node(Definition(PostingOwner, [selected, selected], "selected").concretize())
    second.publish_node(selected, kind="saved")
    first.publish_node(Definition(PostingOwner, [model(output=3)], "unrelated").concretize())
    writer.close(flush=False)
    repo = Repo((first, second))
    try:
        query = repo.query().sel(Selector(Definition(DecodeState), cls_policy="exact"))
        with monkeypatch.context() as patch:
            patch.setattr(indexed, "_try_identity_collect", lambda *a: None)
            expected = query.collect()
        calls, original = [], codec._decode_node_graph_views
        def decode(packet, **kwargs):
            calls.append(packet["views"])
            return original(packet, **kwargs)
        monkeypatch.setattr(codec, "_decode_node_graph_views", decode)
        from dryml.core.state_values import LocalState
        monkeypatch.setattr(LocalState, "validate_for", lambda *a, **k: pytest.fail("repeated certified state validation"))
        monkeypatch.setattr(first, "read_states", lambda *a: pytest.fail("collection loaded payloads"))
        actual = query.collect()
        assert actual._entries.keys() == expected._entries.keys()
        for key in actual._entries:
            assert actual._entries[key][1] == expected._entries[key][1]
        assert len(calls) <= actual.count()
        assert repo._num_constructions == 0
        assert len(actual.evidence_for(selected).sources) == 2
    finally:
        repo.close(flush=False)


@pytest.mark.parametrize("damage", ("block", "membership", "declaration"))
def test_certified_collection_rechecks_source_authority_after_success(tmp_path, damage):
    from pathlib import Path
    from dryml.core import ObjectId
    from dryml.core.query.model import QueryError
    from dryml.core.store.store import StoreCapabilityError, StoreAuthorityError

    store = DirStore(tmp_path / "store", query_index="sqlite")
    child = model().concretize().with_identity(ObjectId())
    try:
        root = store.publish_node(Definition(PostingOwner, [child]).concretize())
        query = store.query().refresh(False).object_refs().sel(model(Par("width")).loose_selector())
        assert query.collect().count() == 1
        paths = {
            "block": store._node_path("definitions/blocks", root.block_digest),
            "membership": store._node_path("membership/recipe", root.digest),
            "declaration": store._node_path("declarations/anchors", store._anchor_key(child.object_id)),
        }
        Path(paths[damage]).unlink()
        with pytest.raises((QueryError, StoreCapabilityError, StoreAuthorityError)):
            query.collect()
    finally:
        store.close()


@pytest.mark.parametrize("change", ("authority", "index", "interrupt"))
def test_certified_collection_revalidates_after_decoding(tmp_path, monkeypatch, change):
    import dryml.core.query.indexed_count as indexed

    store = DirStore(tmp_path / "store", query_index="sqlite")
    original, changed = indexed._collect_selected, False
    def decode(*args):
        nonlocal changed
        result = original(*args)
        if not changed:
            changed = True
            if change == "authority":
                store.publish_node(model(16).concretize())
            elif change == "index":
                store.open_query_index().maintain_store(rebuild=True)
            else:
                raise KeyboardInterrupt("controlled collection interruption")
        return result
    try:
        store.publish_node(model().concretize())
        monkeypatch.setattr(indexed, "_collect_selected", decode)
        query = store.query().sel(model(Par("width")).loose_selector())
        if change == "interrupt":
            with pytest.raises(KeyboardInterrupt, match="controlled collection"):
                query.collect()
        else:
            assert query.collect().count() == (2 if change == "authority" else 1)
        assert not store.open_query_index()._connections.connection(readonly=True).in_transaction
    finally:
        store.close()
