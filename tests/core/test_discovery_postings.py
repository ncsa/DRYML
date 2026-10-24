"""Occurrence-local discovery storage, portable locators and selective SQL work."""

import json
import sqlite3

import pytest

from dryml.core import Definition, ObjectId
from dryml.core.query.relationships import _discovery_rows
from dryml.core.query.sqlite.lowering import _store_discovery_relation, _store_discovery_page_sql
from dryml.core.store.dir import DirStore
from tests.core.test_local_state_discovery import DiscoveryPair, DiscoveryValue

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

def _sql_work(store, pattern, *, kind="object_refs", limit=10):
    query = getattr(store.query(), kind)().sel(pattern).refresh(False).require_indexed()
    sql, params = _store_discovery_page_sql(_store_discovery_relation(query), page_size=limit)
    with sqlite3.connect(store.open_query_index().path) as con:
        plan = tuple(row[-1] for row in con.execute("EXPLAIN QUERY PLAN " + sql, params))
        steps = [0]
        def progress():
            steps[0] += 1
        con.set_progress_handler(progress, 1)
        rows = con.execute(sql, params).fetchall()
    return steps[0], plan, rows


def test_discovery_chain_prepares_shared_views_and_stores_local_features(tmp_path, monkeypatch):
    from dryml.core.node_prepare import NodePreparation
    store = DirStore(tmp_path / "store", query_index="none")
    root = Definition(DiscoveryValue, "leaf").concretize()
    for i in range(15):
        root = Definition(DiscoveryPair, root, i).concretize()
    _, (record,), _ = store._prepare_node_roots((root,), "recipe")
    owners, original = [], NodePreparation.__init__
    def counted(owner, *args, **kwargs):
        original(owner, *args, **kwargs)
        owners.append(owner)
    monkeypatch.setattr(NodePreparation, "__init__", counted)
    rows = list(_discovery_rows(root, record))
    features = sum(len(row.get("features", ())) for row in rows)
    discoveries = sum(owner.counts["discovery"] for owner in owners)
    print("chain", len(rows), "rows", len(owners), "owners", discoveries, "discoveries", features, "features")
    assert discoveries <= 3 * 16
    assert features <= 3 * 16


def test_selective_posting_hit_and_miss_do_not_scan_unrelated_payloads(tmp_path):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    identity = ObjectId()
    selected = Definition(DiscoveryValue, "selected").concretize().with_identity(identity)
    store.publish_node(selected)
    patterns = [Definition(DiscoveryValue).with_identity(identity),
                Definition(DiscoveryValue).with_identity(ObjectId()),
                Definition(DiscoveryValue, name="selected"),
                Definition(DiscoveryValue, name="missing")]
    small = [_sql_work(store, pattern) for pattern in patterns]
    for i in range(32):
        store.publish_node(Definition(DiscoveryValue, "unrelated-" + str(i)).concretize().with_identity(ObjectId()))
    large = [_sql_work(store, pattern) for pattern in patterns]
    print("selective VM", [(a[0], b[0]) for a, b in zip(small, large)], "plans", [b[1] for b in large])
    assert [len(result[2]) for result in large] == [1, 0, 1, 0]
    assert all(b[0] <= a[0] + 100 for a, b in zip(small, large))
    assert all(any("store_discovery_token" in line and "<expr>=?" in line for line in result[1]) for result in large)
    assert all(any("store_discovery_fact" in line and "<expr>=?" in line for line in result[1]) for result in large)


def _force_tied_order(monkeypatch, reverse):
    import dryml.core.utils.graph.value as values
    from dryml.core.utils.graph.path import SetMember
    original = values.iter_set_members
    def ordered(items):
        members = original(items)
        buckets = {}
        for segment, value in members:
            buckets.setdefault(segment.fingerprint, []).append(value)
        return tuple((SetMember(fp, i), value) for fp, bucket in sorted(buckets.items())
                     for i, value in enumerate(sorted(bucket, key=lambda v: json.dumps(v.object_id.to_data()),
                                                       reverse=reverse[0])))
    monkeypatch.setattr(values, "iter_set_members", ordered)


@pytest.mark.parametrize("policy", ["sqlite", "none"])
@pytest.mark.parametrize("facets", ["identity", "state"])
def test_discovery_locator_survives_forced_tied_set_order_flip(tmp_path, monkeypatch, policy, facets):
    reverse = [False]
    _force_tied_order(monkeypatch, reverse)
    store = DirStore(tmp_path / "store", query_index=policy)
    children = [Definition(DiscoveryValue, "same").concretize().with_identity(ObjectId()) for _ in range(2)]
    if facets == "state":
        from tests.core.test_local_state_discovery import saved_values
        _, _, donors = saved_values(tmp_path / "donors", 2)
        # Equal creation recipes, distinct full StateKeys, no payload in this Store.
        children = [Definition(DiscoveryValue, "selected").concretize().with_identity(child.object_id).with_state(donor.state)
                    for child, donor in zip(children, donors)]
    root = Definition(DiscoveryPair, set(children), "set").concretize()
    store.publish_node(root, kind="saved" if facets == "state" else "recipe")
    # Disabled indexing also captures discovery and resolves it in separate decodes.
    import dryml.core.query.authority as authority
    capture = authority.capture_store_projection
    def captured(*args, **kwargs):
        reverse[0] = False
        result = capture(*args, **kwargs)
        reverse[0] = True
        return result
    monkeypatch.setattr(authority, "capture_store_projection", captured)
    reverse[0] = True
    for child in children:
        pattern = Definition(DiscoveryValue).with_identity(child.object_id)
        if facets == "state":
            pattern = pattern.with_state(child.state)
        query = getattr(store.query(), "saved" if facets == "state" else "object_refs")().sel(pattern).refresh(False)
        result = query.one()
        assert result.object_id == child.object_id
        assert result.graph_equal(child)
        if facets == "state":
            assert result.state.key == child.state.key


@pytest.mark.parametrize("shape,size", [("chain", 8), ("chain", 16), ("tree", 8), ("tree", 16),
                                        ("alias", 8), ("alias", 16), ("diamond", 3), ("diamond", 6),
                                        ("crossjoin", 3), ("crossjoin", 6)])
def test_publication_exports_local_facts_once_and_reports_hard_join_work(tmp_path, monkeypatch, shape, size):
    from dryml.core.node_prepare import NodePreparation
    from tests.core.test_node_key_contract import _workload
    store = DirStore(tmp_path / "store", query_index="sqlite")
    store.publish_node(Definition(DiscoveryValue, "seed").concretize())
    root = _workload(shape, size).concretize()
    owners, original = [], NodePreparation.__init__
    def counted(owner, *args, **kwargs):
        original(owner, *args, **kwargs)
        owners.append(owner)
    monkeypatch.setattr(NodePreparation, "__init__", counted)
    # Ready synchronous publication, including all producers' preparation/export.
    record = store.publish_node(root)
    counts = {key: sum(owner.counts.get(key, 0) for owner in owners) for key in owners[0].counts}
    index = store.open_query_index()
    rows = [json.loads(row[2]) for row in index.projection_rows("discovery", refresh=False)
            if json.loads(row[2])["root"]["digest"] == record.digest]
    facts = [json.loads(row[2]) for row in index.projection_rows("discovery_feature", refresh=False)
             if record.block_digest in json.loads(row[2])["node"]]
    links = [json.loads(row[2]) for row in index.projection_rows("discovery_link", refresh=False)
             if record.block_digest in json.loads(row[2])["parent"]]
    assert all("features" not in row and "path" not in row for row in rows)
    assert len(facts) == len({(f["node"], f["token"]) for f in facts})
    n = len(rows)
    assert len(facts) <= 3 * n and len(links) <= 2 * n
    assert counts["discovery"] <= 6 * n
    assert counts["export_rows"] <= 3 * n
    if shape in {"chain", "tree", "alias"}:
        assert counts["fallback_rows"] == 0
    else:
        assert counts["fallback_rows"] > 0
    before = len(owners), index.current_generation()
    store.publish_node(root)
    assert index.current_generation() == before[1]
    print("DISCOVERY-WORK", shape, size, "nodes", n, "owners", before[0], "features", len(facts), "links", len(links), json.dumps(counts, sort_keys=True))


def test_unanchored_shape_admits_scan_before_inventory_or_recovery(tmp_path, monkeypatch):
    from dryml.core.query.model import QueryIndexUnavailable, QueryWouldScanError
    store = DirStore(tmp_path / "store", query_index="sqlite")
    store.publish_node(Definition(DiscoveryValue, "value").concretize())
    monkeypatch.setattr(store, "open_query_index", lambda: pytest.fail("unsupported plan opened index before scan admission"))
    query = store.query().sel(Definition(DiscoveryValue)).refresh(True)
    with pytest.raises(QueryIndexUnavailable, match="posting anchor"):
        query.require_indexed().count()
    with pytest.raises(QueryWouldScanError, match="posting anchor"):
        query.scan_policy("forbid").count()


def test_full_key_and_ten_match_vm_work_is_independent_of_unrelated_history(tmp_path, monkeypatch):
    from dryml.core import StateKey, LocalState
    from tests.core.test_local_state_discovery import saved_values, no_inventory
    store, repo, saved = saved_values(tmp_path, 10)
    old = saved[0].state
    missing = LocalState(StateKey(old.key.cdef_digest, "pkl-" + "0" * 64), old.creation_cdef)
    patterns = [Definition(DiscoveryValue).with_state(old), Definition(DiscoveryValue).with_state(missing),
                Definition(DiscoveryValue, name="selected")]
    small = [_sql_work(store, p, kind="saved") for p in patterns]
    for number in range(16):
        repo.save(DiscoveryValue("unrelated-" + str(number), repo=repo))
    no_inventory(monkeypatch, store)
    monkeypatch.setattr(store, "_authenticate_state_data", lambda *a: pytest.fail("discovery authenticated payload"))
    large = [_sql_work(store, p, kind="saved") for p in patterns]
    assert [len(r[2]) for r in large] == [1, 0, 10]
    assert all(b[0] <= a[0] + 100 for a, b in zip(small, large))
    for pattern, count in zip(patterns, (1, 0, 10)):
        query = store.query().saved().sel(pattern).refresh(False).require_indexed()
        assert len(list(query.take(10))) == count
        assert len(list(query.take(1))) == min(count, 1)
        assert len(list(query.take(0))) == 0
    print("FULL-KEY-VM", [(a[0], b[0]) for a, b in zip(small, large)], "plans", [r[1] for r in large])


def _fresh_locator_reader(path, children, policy, queue):
    try:
        from pytest import MonkeyPatch
        with MonkeyPatch.context() as patch:
            _force_tied_order(patch, [True])
            store = DirStore.open_existing(path, query_index=policy)
            result = []
            patch.setattr(store, "_authenticate_state_data", lambda *a: pytest.fail("payload IO"))
            for child in children:
                pattern = Definition(DiscoveryValue).with_identity(child.object_id).with_state(child.state)
                found = store.query().saved().sel(pattern).refresh(False).one()
                result.append((found.object_id == child.object_id, found.state.key == child.state.key, found.graph_equal(child)))
            store.close()
            queue.put(result)
    except BaseException as error:
        queue.put((type(error).__name__, str(error)))


@pytest.mark.parametrize("policy", ["sqlite", "none"])
def test_locator_fresh_process_reads_exact_own_identity_and_full_key(tmp_path, monkeypatch, policy):
    import multiprocessing
    from tests.core.test_local_state_discovery import saved_values
    _, _, donors = saved_values(tmp_path / "donors", 2)
    children = [Definition(DiscoveryValue, "selected").concretize().with_identity(ObjectId()).with_state(d.state) for d in donors]
    _force_tied_order(monkeypatch, [False])
    store = DirStore(tmp_path / "store", query_index=policy)
    store.publish_node(Definition(DiscoveryPair, set(children), "set").concretize(), kind="saved")
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    worker = context.Process(target=_fresh_locator_reader, args=(store.base_dir, children, policy, queue))
    worker.start()
    try:
        assert queue.get(timeout=20) == [(True, True, True), (True, True, True)]
        worker.join(20)
        assert worker.exitcode == 0
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(10)
        queue.close()
        queue.join_thread()


@pytest.mark.parametrize("family", ["discovery_feature", "discovery_link"])
@pytest.mark.parametrize("damage", ["coverage", "row", "seek"])
def test_posting_coverage_and_seek_loss_refuse_negative_answers_then_repair(tmp_path, monkeypatch, family, damage):
    from dryml.core.query.model import QueryIndexError
    from tests.core.test_store_incremental_index import authority_bytes
    store = DirStore(tmp_path / "store", query_index="sqlite")
    child = Definition(DiscoveryValue, "child").concretize().with_identity(ObjectId())
    store.publish_node(Definition(DiscoveryPair, child, "pair").concretize())
    index = store.open_query_index()
    before = authority_bytes(store)
    with sqlite3.connect(index.path) as con:
        if damage == "coverage":
            con.execute("DELETE FROM store_projection_coverage WHERE family=?", (family,))
        elif damage == "row":
            con.execute("DELETE FROM store_projection_rows WHERE family=?", (family,))
        else:
            con.execute("DROP INDEX " + ("store_discovery_token" if family == "discovery_feature" else "store_discovery_child"))
    for pattern in (Definition(DiscoveryValue).with_identity(child.object_id), Definition(DiscoveryValue).with_identity(ObjectId())):
        with pytest.raises(QueryIndexError):
            store.query().object_refs().sel(pattern).refresh(False).require_indexed().count()
    monkeypatch.setattr(store, "_authenticate_state_data", lambda *a: pytest.fail("repair read payload"))
    index.maintain_store()
    assert authority_bytes(store) == before
    assert index.validate(thorough=True).ok
    assert store.query().object_refs().sel(Definition(DiscoveryValue).with_identity(child.object_id)).refresh(False).require_indexed().count() == 1


def test_correlated_edges_retain_alias_roles_and_exact_quoted_residual(tmp_path):
    from dryml.core.links import DefLink
    from dryml.core.cdef_graph import EdgeKind
    store = DirStore(tmp_path / "store", query_index="sqlite")
    child = Definition(DiscoveryValue, "child").concretize().with_identity(ObjectId())
    roots = [Definition(DiscoveryPair, child, child).concretize(),
             Definition(DiscoveryPair, child, child.copy_graph()).concretize(),
             Definition(DiscoveryPair, DefLink.finalized(EdgeKind.REF, child), child).concretize()]
    for root in roots:
        store.publish_node(root)
    shared = Definition(DiscoveryValue).with_identity(child.object_id)
    query = store.query().cdefs().stored().sel(Definition(DiscoveryPair, left=shared, right=shared)).refresh(False).require_indexed()
    assert query.one().graph_equal(roots[0])
    reference = Definition(DiscoveryPair, left=DefLink.assertion(EdgeKind.REF, shared), right=shared)
    assert store.query().cdefs().stored().sel(reference).refresh(False).require_indexed().one().graph_equal(roots[2])
    quoted = Definition(DiscoveryPair, {"quote": child.quote()}, "quoted").concretize()
    other = Definition(DiscoveryPair, {"quote": Definition(DiscoveryValue, "other").concretize().quote()}, "quoted").concretize()
    for root in (quoted, other):
        store.publish_node(root)
    pattern = Definition(DiscoveryPair, left={"quote": child.quote()}, right="quoted")
    assert store.query().cdefs().stored().sel(pattern).refresh(False).require_indexed().one().graph_equal(quoted)


def test_shared_discovery_owner_warm_clear_and_eviction_change_only_work(tmp_path, monkeypatch):
    from dryml.core.node_prepare import NodePreparation
    from dryml.core.node_codec import encode_node_graph, _decode_node_graph_views
    from dryml.core.query.relationships import _discovery_preparation
    root = Definition(DiscoveryPair, Definition(DiscoveryValue, "child"), "root").concretize()
    store = DirStore(tmp_path / "store", query_index="none")
    _, (header,), _ = store._prepare_node_roots((root,), "recipe")
    (root,), owner = _decode_node_graph_views(encode_node_graph(NodePreparation((root,))), with_owner=True)
    p = _discovery_preparation(owner, header.block_digest)
    first = list(_discovery_rows(root, header, preparation=p))
    before = dict(p.owner.counts)
    assert _discovery_preparation(owner, header.block_digest) is p
    assert list(_discovery_rows(root, header, preparation=p)) == first
    assert p.owner.counts == before
    assert owner._cache_bytes <= owner.max_cached_bytes
    owner.clear()
    new = _discovery_preparation(owner, header.block_digest)
    assert new is not p and list(_discovery_rows(root, header, preparation=new)) == first
    owner.clear()
    owner.max_cached_bytes = 1
    one = _discovery_preparation(owner, header.block_digest)
    two = _discovery_preparation(owner, header.block_digest)
    assert one is not two and owner._cache_bytes == 0
    assert list(_discovery_rows(root, header, preparation=two)) == first


def test_mixed_facet_sql_joins_seek_same_occurrences_on_hit_and_miss(tmp_path):
    from tests.core.test_local_state_discovery import saved_values
    store, repo, donors = saved_values(tmp_path, 2)
    children = [Definition(DiscoveryValue, "selected").concretize().with_identity(ObjectId()) for _ in range(2)]
    roots = [Definition(DiscoveryPair, children[0].with_state(a.state), children[1].with_state(b.state)).concretize()
             for a, b in (donors, donors[::-1])]
    for root in roots:
        store.publish_node(root, kind="saved")
    left = Definition(DiscoveryValue).with_identity(children[0].object_id).with_state(donors[0].state)
    patterns = [Definition(DiscoveryPair, left=left, right=Definition(DiscoveryValue).with_state(donors[1].state)),
                Definition(DiscoveryPair, left=left, right=Definition(DiscoveryValue).with_state(donors[0].state)),
                Definition(DiscoveryPair, left=Definition(DiscoveryValue).with_identity(ObjectId()))]
    small = [_sql_work(store, p, kind="saved") for p in patterns]
    for number in range(16):
        repo.save(DiscoveryValue("unrelated-" + str(number), repo=repo))
    large = [_sql_work(store, p, kind="saved") for p in patterns]
    assert [len(r[2]) for r in large] == [1, 0, 0]
    assert all(b[0] <= a[0] + 100 for a, b in zip(small, large))
    for result in large:
        assert any("store_discovery_child" in line and "<expr>=?" in line for line in result[1])
    assert store.query().saved().stored().sel(patterns[0]).refresh(False).require_indexed().one().graph_equal(roots[0])
    assert store.query().saved().stored().sel(patterns[1]).refresh(False).require_indexed().count() == 0
    print("MIXED-VM", [(a[0], b[0]) for a, b in zip(small, large)], "plans", [r[1] for r in large])


@pytest.mark.parametrize("shape", ["alias", "crossjoin"])
def test_saved_graph_discovery_counts_shared_facets_and_contribution_output(tmp_path, monkeypatch, shape):
    from dryml.core.node_prepare import NodePreparation
    from tests.core.test_local_state_discovery import saved_values
    _, _, donors = saved_values(tmp_path / "donors", 1)
    store = DirStore(tmp_path / "store", query_index="sqlite")
    store.publish_node(Definition(DiscoveryValue, "seed").concretize())
    leaf = donors[0]
    root = leaf
    pair = lambda a, b: Definition(DiscoveryPair, a, b).concretize()
    for _ in range(3):
        root = pair(root, root) if shape == "alias" else pair(pair(root, "scalar"), pair(root, root))
    owners, original = [], NodePreparation.__init__
    def counted(owner, *args, **kwargs):
        original(owner, *args, **kwargs)
        owners.append(owner)
    monkeypatch.setattr(NodePreparation, "__init__", counted)
    record = store.publish_node(root, kind="saved")
    index = store.open_query_index()
    contributions = [json.loads(r[2]) for r in index.projection_rows("discovery", refresh=False)
                     if json.loads(r[2])["root"]["digest"] == record.digest]
    facts = [json.loads(r[2]) for r in index.projection_rows("discovery_feature", refresh=False)
             if record.block_digest in json.loads(r[2])["node"]]
    assert {r["kind"] for r in contributions} == {"cdef", "object_ref", "state_ref"}
    counts = {key: sum(o.counts.get(key, 0) for o in owners) for key in owners[0].counts}
    assert counts["discovery"] <= 6 * len(contributions)
    assert all("features" not in r for r in contributions)
    assert len(facts) == len({(r["node"], r["token"]) for r in facts})
    assert store.query().local_states().count() == 0  # graph knowledge is not installed state
    assert store.query().saved().sel(Definition(DiscoveryValue).with_state(leaf.state)).refresh(False).require_indexed().one().state.key == leaf.state.key
    print("SAVED-DISCOVERY-WORK", shape, "contributions", len(contributions), "features", len(facts), json.dumps(counts, sort_keys=True))


def test_membership_delta_after_tied_order_flip_joins_existing_block_facts(tmp_path, monkeypatch):
    reverse = [False]
    _force_tied_order(monkeypatch, reverse)
    store = DirStore(tmp_path / "store", query_index="sqlite")
    children = [Definition(DiscoveryValue, "same").concretize().with_identity(ObjectId()) for _ in range(2)]
    root = Definition(DiscoveryPair, set(children), "set").concretize()
    record = store.publish_node(root)
    index = store.open_query_index()
    before = index.projection_rows("discovery_feature", refresh=False)
    assert store.withdraw_node(record.digest)
    reverse[0] = True
    assert store.publish_node(root) == record
    assert index.projection_rows("discovery_feature", refresh=False) == before
    for child in children:
        pattern = Definition(DiscoveryValue).with_identity(child.object_id)
        assert store.query().object_refs().sel(pattern).refresh(False).require_indexed().one().object_id == child.object_id


def test_boolean_prefilters_and_unanchored_scan_warning_keep_exact_results(tmp_path):
    from dryml.core.selector import QueryPredicate
    from dryml.core.query.model import QueryIndexUnavailable
    store = DirStore(tmp_path / "store", query_index="sqlite")
    for name in ("first", "second", "third"):
        store.publish_node(Definition(DiscoveryValue, name).concretize())
    a, b = (QueryPredicate(Definition(DiscoveryValue, name=name)) for name in ("first", "second"))
    base = store.query().cdefs().stored().refresh(False)
    assert base.sel(a | b).require_indexed().count() == 2
    assert base.sel(a & ~b).require_indexed().count() == 1
    unanchored = a | ~b
    with pytest.raises(QueryIndexUnavailable, match="posting anchor"):
        base.sel(unanchored).require_indexed().count()
    with pytest.warns(RuntimeWarning, match="posting anchor") as warnings:
        assert base.sel(unanchored).scan_policy("warn").count() == 2
    assert len(warnings) == 1


def test_soft_discovery_retains_independent_roots_and_authenticated_local_facts(tmp_path, monkeypatch):
    """Unknown positional names neither resolve quotes nor hide independent roots."""
    from dryml.core import Par
    from dryml.core.node_codec import _decode_node_graph_views
    from dryml.core.query.relationships import _discovery_preparation
    from dryml.core.symbol import ImportRef

    child = Definition(DiscoveryValue, "owned").concretize().with_identity(ObjectId())
    soft = Definition(ImportRef("_dryml_unavailable_recipe", "Future"), Par("name"), child, label="soft")
    root = Definition(DiscoveryPair, {"quote": soft.quote()}, child).concretize()
    store = DirStore(tmp_path / "store", query_index="none")
    block, records, _ = store._prepare_node_roots((root, soft, child), "recipe")
    packet = block.blocks
    packet["views"] = [record.view for record in records]
    roots, owner = _decode_node_graph_views(packet, with_owner=True)
    monkeypatch.setattr(ImportRef, "resolve", lambda *a, **k: pytest.fail("discovery resolved a target"))

    prepared = _discovery_preparation(owner, block.digest)
    rows = [list(_discovery_rows(node, record, preparation=prepared)) for node, record in zip(roots, records)]
    assert [sum(row["is_root"] and row["kind"] == "cdef" for row in contribution) for contribution in rows] == [1, 1, 1]
    assert len(rows[0]) == 4  # root and owned child, each with recipe and identity views
    assert len(rows[1]) == 4  # independently published soft root and its real child
    assert all(row["locator"] in owner._transport_locators.values() for contribution in rows for row in contribution)
    for node, contribution in zip(roots, rows):
        selected = next(row for row in contribution if row["is_root"] and row["kind"] == "cdef")
        locator = owner._transport_locators[id(node)]
        base = block.digest + ":" + locator["table"] + ":" + str(locator["slot"])
        assert selected["fact"] in {base, base + ":creation"}
    links = [value for family, _, value in prepared.facts() if family == "discovery_link"]
    assert not any(link["parent"] == prepared.fact_key(roots[0]) and
                   link["child"] == prepared.fact_key(roots[1]) for link in links)
    assert any(link["parent"] == prepared.fact_key(roots[1]) and
               link["child"] == prepared.fact_key(roots[2]) for link in links)
    assert any(json.loads(value["token"]) == ["scalar", "label", "str", "soft"]
               for family, _, value in prepared.facts() if family == "discovery_feature")


def _fresh_soft_recipe_reader(path, expected, soft, queue):
    try:
        from pytest import MonkeyPatch
        from dryml.core import EdgePolicy
        from dryml.core.symbol import ImportRef
        from tests.core.test_store_incremental_index import authority_bytes

        with MonkeyPatch.context() as patch:
            patch.setattr(ImportRef, "resolve", lambda *a, **k: pytest.fail("metadata resolved a target"))
            store = DirStore.open_existing(path, query_index="sqlite")
            patch.setattr(store, "_authenticate_state_data", lambda *a, **k: pytest.fail("metadata read payload"))
            before = authority_bytes(store)
            index = store.open_query_index()
            index.rebuild()
            assert index.validate(thorough=True).ok
            assert authority_bytes(store) == before
            pattern = Definition().with_identity(expected.object_id).with_state(expected.state)
            found = store.query().saved().sel(pattern).refresh(False).require_indexed().one()
            assert found.graph_equal(expected)
            assert store.query().cdefs().stored().refresh(False).count() == 2
            assert any(node.graph_equal(soft) for node in store.query().cdefs().stored().refresh(False).collect())
            assert store.query().cdefs().stored().sel(Definition(DiscoveryPair, right="independent")).refresh(False).require_indexed().one().graph_equal(soft)
            target = Definition(DiscoveryValue, "never-constructed").concretize()
            mixed = Definition(left=Definition().with_identity(expected.object_id))
            assert store.query().cdefs().stored().sel(mixed).refresh(False).require_indexed().count() == 1
            assert store.query().cdefs().stored().sel(mixed).nested(target, edges=EdgePolicy.ALL).owners().count() == 0
            queue.put("ok")
            store.close()
    except BaseException as error:
        queue.put((type(error).__name__, str(error)))


def test_quoted_soft_recipe_with_owned_anchor_survives_fresh_process_rebuild(tmp_path):
    """Public publication and fresh metadata rebuild preserve soft and saved roots."""
    import multiprocessing
    from dryml.core import Par
    from tests.core.test_local_state_discovery import saved_values
    from tests.core.test_repo_query_nested import QueryLeaf, QueryRecipeOwner

    _, _, (child,) = saved_values(tmp_path / "donor", 1)
    recipe = Definition(QueryLeaf, Par("name"))
    quote_owner = Definition(QueryRecipeOwner, recipe).concretize()
    root = Definition(DiscoveryPair, child, quote_owner).concretize()
    soft = Definition(DiscoveryPair, Par("left"), right="independent")
    store = DirStore(tmp_path / "store", query_index="sqlite")
    store.publish_node(root, kind="saved")
    store.publish_node(root)
    store.publish_node(soft)
    assert store.query().saved().sel(Definition().with_identity(child.object_id)).refresh(False).require_indexed().one().graph_equal(child)
    store.close()
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    worker = context.Process(target=_fresh_soft_recipe_reader, args=(store.base_dir, child, soft, queue))
    worker.start()
    try:
        assert queue.get(timeout=30) == "ok"
        worker.join(20)
        assert worker.exitcode == 0
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(10)
        queue.close()
        queue.join_thread()
