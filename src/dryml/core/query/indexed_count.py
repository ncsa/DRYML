"""Postings-first scalar counts over certified Store graph projections.

Pure semantic proofs come from coverage-versioned projection records. Membership,
required declarations, immutable source bytes, connected-Store agreement, and the
captured authority/index generation are checked independently on every terminal.
Unsupported predicates or execution controls retain the ordinary Node consumer.
No private payload, runtime Object, symbol resolution or source mutation is used.
"""

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os

from .postings import _compile, _read_graph, _atom, _count_paths, _ResidualRequired


class _ProofReader:
    """Terminal-local primitive evidence backed by fresh detached byte witnesses."""

    def __init__(self, stores, indexes, tokens):
        from .source import _NodePage, _NodeOwnerCache
        self.stores, self.indexes, self.tokens = stores, indexes, tokens
        self.page = _NodePage(stores, tokens, indexes)
        self.graphs = _NodeOwnerCache(32 * 1024 * 1024, 32 * 1024 * 1024)
        self.checked_blocks, self.anchors, self.declarations = set(), {}, {}

    @contextmanager
    def connection(self, store):
        from .source import _node_index_token, _SourceCutChanged
        from .model import QueryIndexDirty
        domain = store.authority_fence_key()
        index = self.indexes[domain]
        with index._connections._read_lease() as (con, identity):
            try:
                with store.authority_read_fence():
                    con.execute("BEGIN")
                    if _node_index_token(index, con, identity, store.authority_cut()) != self.tokens[domain]:
                        raise _SourceCutChanged()
                yield con
            except QueryIndexDirty as error:
                raise _SourceCutChanged() from error
            finally:
                if con.in_transaction:
                    con.execute("ROLLBACK")

    def read(self, store, path, reader, required=False):
        from .source import _SourceCutChanged
        with store.authority_read_fence():
            if store.authority_cut() != self.tokens[store.authority_fence_key()][0]:
                raise _SourceCutChanged()
            return self.page.read(store, path, reader, required)

    def graph(self, store, digest):
        from ..store import authority
        from ..store.store import StoreAuthorityError
        from .model import QueryIndexDirty
        key = store.authority_fence_key(), digest
        existing = self.graphs.get(key)
        if existing is not None:
            return existing
        path = store._relative(store._node_path("definitions/blocks", digest))
        with self.connection(store) as con:
            evidence = dict(con.execute("SELECT family,payload FROM store_projection_rows "
                                        "WHERE source=? AND family IN ('record_bytes','query_graph')", (path,)))
            if set(evidence) != {"record_bytes", "query_graph"}:
                raise QueryIndexDirty("Missing certified graph evidence; rebuild derived coverage.")
            if key not in self.checked_blocks:
                from .source import _SourceCutChanged
                with store.authority_read_fence():
                    if store.authority_cut() != self.tokens[key[0]][0]:
                        raise _SourceCutChanged()
                    raw = authority.raw(store, path)
                fingerprint = json.loads(evidence["record_bytes"])
                if (raw is None or len(raw) != fingerprint["size"]
                        or hashlib.sha256(raw).hexdigest() != fingerprint["sha256"]):
                    raise StoreAuthorityError("Query graph source bytes are missing or changed; repair Store metadata.")
                self.page.witnesses[key[0]].file(os.path.join(store.base_dir, path), raw)
                self.checked_blocks.add(key)
        envelope = json.loads(evidence["query_graph"])
        graph = _read_graph(envelope)
        if graph is None:
            raise _ResidualRequired("graph evidence exceeds compact execution budget")
        self.graphs.retain(key, graph, 4096 + 8 * envelope["size"])
        return graph

    @staticmethod
    def root(graph, record):
        from ..store.store import StoreAuthorityError
        try:
            view = record.view
            slot = graph["locators"][view["table"] + ":" + str(view["slot"])]
            row = graph["nodes"][slot]
            valid = row["keys"][view["view"]] == view["expected_key"]
            if record.root_kind == "projection":
                valid &= row["exact"] == graph["nodes"][row["object"]]["exact"]
            elif record.root_kind == "construction":
                valid &= row["creation"] is not None and row["exact"] == graph["nodes"][row["creation"]]["exact"]
            elif record.root_kind == "saved":
                valid &= row["saved"]
            if not valid:
                raise ValueError()
            return slot
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise StoreAuthorityError("Named root disagrees with certified immutable graph evidence.") from error

    def declaration(self, identity):
        from ..store import records as r
        from .source import InventoryCapabilityError
        from .model import QueryError
        if identity in self.declarations:
            return self.declarations[identity]
        certificates = {}
        for store in self.stores:
            domain = store.authority_fence_key()
            key = store._anchor_key(identity)
            anchor = self.read(store, store._relative(store._node_path("declarations/anchors", key)), r.DeclarationAnchorRecord)
            required = self.read(store, store._relative(store._node_path("declarations/required", key)), r.DeclarationRequirementRecord)
            if anchor is None and required is None:
                continue
            if (anchor is None or required is None or anchor.object_id != identity or required.object_id != identity
                    or anchor.projection.digest != required.projection_digest):
                raise InventoryCapabilityError("Missing or incompatible required declaration anchor.")
            graph = self.graph(store, anchor.projection.block_digest)
            row = graph["nodes"][self.root(graph, anchor.projection)]
            if _atom(row["shape"]["identity"]) != identity:
                raise InventoryCapabilityError("Declaration projection names another ObjectId.")
            certificates[domain] = row["exact"]
            self.anchors[domain, identity] = anchor
        if len(set(certificates.values())) > 1:
            raise QueryError("Query object-reference authority conflicts across connected Stores.")
        self.declarations[identity] = certificates
        return certificates

    def membership(self, store, source, record, graph):
        from ..store import records as r
        from .source import InventoryCapabilityError
        domain = store.authority_fence_key()
        member = self.read(store, source, r.NodeMembershipRecord, True)
        if (member.root_kind, member.root_digest) != (record.root_kind, record.digest):
            raise InventoryCapabilityError("Derived candidate has no matching source membership.")
        path = store._relative(store._node_path("graphs" if record.root_kind == "saved" else "definitions/roots", record.digest))
        if self.read(store, path, r.NodeRootRecord, True) != record:
            raise InventoryCapabilityError("Derived candidate disagrees with current root authority.")
        root = self.root(graph, record)
        for identity in member.identities:
            self.declaration(identity)
        pending, reachable = [root], set()
        while pending:
            slot = pending.pop()
            if slot in reachable:
                continue
            reachable.add(slot)
            row = graph["nodes"][slot]
            identity = _atom(row["shape"]["identity"])
            if identity is not None:
                anchor = self.anchors.get((domain, identity))
                if (identity not in member.identities or anchor is None
                        or (record.root_kind, record.digest) not in anchor.sources):
                    raise InventoryCapabilityError("Candidate has no required declaration contribution.")
                if self.declarations[identity][domain] != graph["nodes"][row["object"]]["exact"]:
                    raise InventoryCapabilityError("Candidate conflicts with its declaration anchor.")
            pending.extend(child for kind, child in row["edges"] if kind in {"materialize", "reference"})
        return reachable


def _try_count(occurrence, capture):
    """Return an exact postings count, or None to retain the ordinary residual.

    The supported path is an unbounded ordinary Store/Repo source with optional
    kind restrictions and pure closed target predicates. Every candidate root
    contribution is freshly authenticated, including declarations in nonmatching
    connected Stores.
    """
    from .query import _V3Restriction
    from .source import RepoSource, StoreSource

    query = occurrence.roots
    if (occurrence.fixed is not None or occurrence.occurrence_limit is not None
            or occurrence.path_filter is not None or occurrence.through_kinds or occurrence.metadata_restrictions
            or query.take_limit is not None or capture.max_depth_limit is not None or capture.max_verify_limit is not None
            or not isinstance(query.source, (RepoSource, StoreSource))
            or any(item.kind != "kind" or item.value == "local_state" for item in query.restrictions)):
        return None
    selectors = ((occurrence.selector,) if occurrence.selector is not None else ()) + occurrence.additional_selectors
    if not selectors:
        return None
    predicates = [_compile(_V3Restriction("sel", value).prepared) for value in selectors]
    if any(item.residual is not None for item in predicates):
        return None
    candidates = replace(query, restrictions=tuple(_V3Restriction("sel", value) for value in selectors))
    ready = _ready_indexes(query, candidates, capture)
    if ready is None:
        return None
    stores, indexes = ready
    try:
        return _count(occurrence, capture, stores, indexes, candidates, predicates)
    except _ResidualRequired:
        return None


def _try_identity_count(query, capture):
    """Count covered unbounded identity selections without building Node results.

    Unlike occurrences, identities retain their explicit discovery domain kind.
    Every contributing source is authenticated before deduplication, including
    replicated sources and declarations in nonmatching connected Stores. Scoped,
    composed, bounded and unsupported predicates retain the ordinary consumer.
    """
    from .source import RepoSource, StoreSource

    if (query.take_limit is not None or capture.max_depth_limit is not None or capture.max_verify_limit is not None
            or not isinstance(query.source, (RepoSource, StoreSource))
            or any(item.kind not in {"kind", "sel"} or item.kind == "kind" and item.value == "local_state"
                   for item in query.restrictions)):
        return None
    predicates = [_compile(item.prepared) for item in query.restrictions if item.kind == "sel"]
    if not predicates or any(item.residual is not None for item in predicates):
        return None
    ready = _ready_indexes(query, query, capture)
    if ready is None:
        return None
    stores, indexes = ready
    try:
        return _count(None, capture, stores, indexes, query, predicates, identity_query=query)
    except _ResidualRequired:
        return None


def _ready_indexes(query, candidates, capture):
    """Apply shared scan/refresh policy and select current certified backends."""
    from .source import node_query_stores, _node_discovery_plan
    from .model import QueryIndexDirty, QueryIndexUnavailable, QueryStats

    stores = node_query_stores(query, capture)
    if stores is None:
        return None
    _node_discovery_plan(candidates, capture, class_seek=True)
    targets = capture.resolve_index_targets()
    indexes = {item.store.authority_fence_key(): target.index for target in targets for item in target.presentations}
    if any(store.authority_fence_key() not in indexes for store in stores):
        return None
    for store in stores:
        index = indexes[store.authority_fence_key()]
        state = index.status().state
        if state != "ready":
            if state == "incompatible":
                index._assert_sidecar_rebuildable()
            if capture.refresh_policy is False:
                raise (QueryIndexDirty if state == "dirty" else QueryIndexUnavailable)(
                    "Current Store query coverage is not ready; refresh(False) forbids maintenance.")
            stats = QueryStats()
            try:
                index.refresh("auto", stats=stats, scan_admission=capture.require_scan)
            finally:
                capture.merge_maintenance_stats(stats)
        if not index._has_class_seek():
            return None
    return stores, indexes


def _count(occurrence, capture, stores, indexes, candidates, predicates, *, identity_query=None):
    from ..store import records as r
    from .source import _node_index_token, _SourceCutChanged, _node_discovery_plan, RepoSource
    from .identity import _identity_kind
    from ..node_prepare import NodePreparation
    from .relationships import count_relationship_occurrences

    query = identity_query if identity_query is not None else occurrence.roots
    tokens = {}
    for store in stores:
        domain = store.authority_fence_key()
        index = indexes[domain]
        with store.authority_read_fence(), index._connections._read_lease() as (con, identity):
            con.execute("BEGIN")
            try:
                tokens[domain] = _node_index_token(index, con, identity, store.authority_cut())
            finally:
                con.execute("ROLLBACK")
        capture._record_cut(domain)
        if not hasattr(capture, "_node_query_tokens"):
            capture._node_query_tokens = {}
        if capture._node_query_tokens.setdefault(domain, tokens[domain]) != tokens[domain]:
            raise _SourceCutChanged()
    reader = _ProofReader(stores, indexes, tokens)
    family, where, args, seek = _node_discovery_plan(candidates, capture, class_seek=True)
    table = "store_projection_rows p" + (" INDEXED BY " + seek if seek else "")
    clause = " WHERE p.family=? AND (" + where + ")"
    sql = "SELECT DISTINCT p.source FROM " + table + clause
    roots_seen, total = set(), 0
    kinds = None if identity_query is not None else {kind.value for kind in occurrence.edges.kinds}
    for store in stores:
        rows = []
        if identity_query is not None:
            with reader.connection(store) as con:
                cursor = con.execute("SELECT p.source,p.payload FROM " + table + clause, (family, *args))
                while page := cursor.fetchmany(query.candidate_page_size):
                    rows.extend((source, json.loads(payload)) for source, payload in page)
                    capture.pages_fetched += 1
        else:
            with reader.connection(store) as con:
                sources = [row[0] for row in con.execute(sql, (family, *args))]
            capture.pages_fetched += 1
            for offset in range(0, len(sources), 128):
                batch = sources[offset:offset + 128]
                selected = []
                with reader.connection(store) as con:
                    cursor = con.execute("SELECT source,payload FROM store_projection_rows WHERE family='discovery' "
                                         "AND source IN (" + ",".join("?" for _ in batch) + ")", batch)
                    while page := cursor.fetchmany(query.candidate_page_size):
                        selected.extend(page)
                        capture.pages_fetched += 1
                rows.extend((source, json.loads(payload)) for source, payload in selected)
        rows.sort(key=lambda item: (item[1]["root"]["block_digest"], item[0]))
        capture.candidate_rows_read += len(rows)
        current_block, graph, memberships, matched = None, None, {}, {}
        records = {}
        for source, item in rows:
            if any(restriction.value != item["kind"] for restriction in query.restrictions if restriction.kind == "kind"):
                continue
            if source not in records:
                records[source] = r.NodeRootRecord.from_data(item["root"])
            record = records[source]
            if current_block != record.block_digest:
                current_block = record.block_digest
                graph = reader.graph(store, current_block)
                memberships, matched = {}, {}
            if source not in memberships:
                memberships[source] = reader.membership(store, source, record, graph)
            raw_slot = graph["locators"][item["locator"]["table"] + ":" + str(item["locator"]["slot"])]
            if raw_slot not in memberships[source]:
                from .source import InventoryCapabilityError
                raise InventoryCapabilityError("Candidate occurrence has no matching source-root contribution.")
            slot = graph["facts"][item["fact"]]
            node = graph["nodes"][slot]
            if node["digest"] != item["digest"]:
                from .source import InventoryCapabilityError
                raise InventoryCapabilityError("Candidate semantic digest disagrees with certified authority.")
            key = (item["kind"] if identity_query is not None else node["kind"]), node["digest"], node["exact"]
            if key in roots_seen:
                continue
            roots_seen.add(key)

            def matches(target):
                if target not in matched:
                    value = graph["nodes"][target]
                    matched[target] = ((identity_query is not None or occurrence.target_kind_filter is None
                                        or value["kind"] in occurrence.target_kind_filter)
                                        and all(predicate.matches(graph, target) for predicate in predicates))
                return matched[target]

            total += bool(matches(slot)) if identity_query is not None else _count_paths(graph, slot, matches, kinds)
    if isinstance(query.source, RepoSource):
        source = query.source
        cached = capture.cache_knowledge(source.repo, weak=source.weak)
        for value in cached:
            owner = NodePreparation((value,))
            key = _identity_kind(value), owner.digest(value), owner._flat(owner._slot(value)).decode("ascii")
            if key in roots_seen or any(item.value != key[0] for item in query.restrictions if item.kind == "kind"):
                continue
            roots_seen.add(key)
            if identity_query is not None:
                total += all(predicate.prepared.matches(value) for predicate in predicates)
            else:
                total += count_relationship_occurrences((value,), lambda target: (
                    (occurrence.target_kind_filter is None or _identity_kind(target) in occurrence.target_kind_filter)
                    and all(predicate.prepared.matches(target) for predicate in predicates)), edges=occurrence.edges)
        with source.repo.retain_topology(allow_physical_duplicates=True):
            if tuple(map(id, source.repo.stores)) != tuple(map(id, capture.repo_stores(source.repo))):
                from .model import QueryError
                raise QueryError("Repo topology changed during query continuation.")
    reader.page.validate()
    capture.verified_count += len(roots_seen)
    return total
