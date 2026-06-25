from __future__ import annotations

from contextlib import contextmanager
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ...cdef_graph import ConcreteDefinitionGraph
from ...definition import ConcreteDefinition
from ..codecs import (
    decode_cdef,
    decode_feature_token,
    decode_graph_path,
    digest_blob,
    encode_cdef,
    encode_feature_token,
    encode_graph_path,
)
from ..fingerprint import canonical_class_key, target_local_fingerprint
from ..index import OccurrenceTraversalSnapshot, OwnerProjection, AllOccurrenceTraversalSnapshot
from ..model import (
    DefinitionEdgeRecord,
    DefinitionId,
    DefinitionRecord,
    FeatureRequirement,
    IndexWriteResult,
    QueryIndexError,
    QueryIndexStatus,
    QueryIndexUnavailable,
    QueryStats,
)
from ..utils import cdef_equal, stable_hash_to_blob
from . import SQLiteQueryIndexConfig, require_sqlite
from .connection import SQLiteConnectionManager
from .schema import SQLITE_QUERY_INDEX_SCHEMA_VERSION, initialize_schema, validate_schema


class SQLiteStoreQueryIndex:
    def __init__(
            self,
            *,
            source_key: str,
            path: str | Path,
            config: SQLiteQueryIndexConfig | None = None,
            canonical_version: int = 1,
            store=None,
            dirty_path: str | Path | None = None):
        if config is None:
            config = SQLiteQueryIndexConfig(path=path)
        elif config.path is None:
            config = SQLiteQueryIndexConfig(
                path=path,
                journal_mode=config.journal_mode,
                durability=config.durability,
                busy_timeout=config.busy_timeout,
                max_write_retries=config.max_write_retries,
            )
        self._source_key = source_key
        self.path = Path(path)
        self.config = config
        self.canonical_version = canonical_version
        self.store = store
        self.dirty_path = None if dirty_path is None else Path(dirty_path)
        self._connections = SQLiteConnectionManager(self.config)

    @property
    def source_key(self) -> str:
        return self._source_key

    def initialize_empty(self) -> None:
        with self._write_transaction() as con:
            initialize_schema(con, store_key=self.source_key, canonical_version=self.canonical_version)
        con = self._connections.connection(readonly=False)
        con.execute("PRAGMA optimize")

    def current_generation(self) -> int:
        self._ensure_ready()
        con = self._connections.connection(readonly=True)
        return _read_generation(con)

    def status(self) -> QueryIndexStatus:
        if self._is_dirty():
            return QueryIndexStatus(
                backend="sqlite",
                store_key=self.source_key,
                generation=None,
                schema_version=None,
                semantic_versions={},
                state="dirty",
                path=str(self.path),
            )
        if not self.path.exists():
            return QueryIndexStatus(
                backend="sqlite",
                store_key=self.source_key,
                generation=None,
                schema_version=None,
                semantic_versions={},
                state="missing",
                path=str(self.path),
            )
        sqlite3 = require_sqlite()
        try:
            con = self._connections.connection(readonly=True)
            validate_schema(con, store_key=self.source_key, canonical_version=self.canonical_version)
            row = con.execute(
                """
                SELECT generation, schema_version, graph_schema_version, path_schema_version,
                       fingerprint_version, cdef_codec_version, feature_codec_version,
                       canonical_version, build_state
                FROM catalog_state
                WHERE singleton = 1
                """
            ).fetchone()
            journal_mode = con.execute("PRAGMA journal_mode").fetchone()[0]
        except QueryIndexUnavailable:
            raise
        except Exception as exc:
            return QueryIndexStatus(
                backend="sqlite",
                store_key=self.source_key,
                generation=None,
                schema_version=SQLITE_QUERY_INDEX_SCHEMA_VERSION,
                semantic_versions={},
                state="incompatible",
                sqlite_version=sqlite3.sqlite_version_info,
                path=str(self.path),
            )
        if row is None:
            state = "incompatible"
            generation = None
            schema_version = None
            semantic_versions = {}
        else:
            generation = row[0]
            schema_version = row[1]
            semantic_versions = {
                "graph_schema_version": row[2],
                "path_schema_version": row[3],
                "fingerprint_version": row[4],
                "cdef_codec_version": row[5],
                "feature_codec_version": row[6],
                "canonical_version": row[7],
            }
            state = row[8]
        return QueryIndexStatus(
            backend="sqlite",
            store_key=self.source_key,
            generation=generation,
            schema_version=schema_version,
            semantic_versions=semantic_versions,
            state=state,
            journal_mode=journal_mode,
            sqlite_version=sqlite3.sqlite_version_info,
            path=str(self.path),
        )

    @contextmanager
    def read_view(self, *, include_cached: bool = True):
        self._ensure_ready()
        con = self._connections.connection(readonly=True)
        con.execute("BEGIN")
        view = None
        try:
            generation = _read_generation(con)
            view = SQLiteQueryIndexReadView(con, source_key=self.source_key, generation=generation)
            yield view
        finally:
            if view is not None:
                view.close()
            con.execute("ROLLBACK")

    def refresh(self, policy, *, stats=None) -> None:
        if policy is False:
            return
        if policy is True:
            self.rebuild(stats=stats)
            return
        if not self.path.exists() or self._is_dirty():
            self.rebuild(stats=stats)
            return
        self._ensure_ready()

    def rebuild(self, *, stats: QueryStats | None = None) -> None:
        if self.store is None or not hasattr(self.store, "hydrate_index"):
            raise QueryIndexUnavailable("SQLite query-index rebuild requires an owning Store with hydrate_index().")
        cdefs = tuple(self.store.hydrate_index())
        for cdef in cdefs:
            if not isinstance(cdef, ConcreteDefinition):
                raise QueryIndexError(f"Store {self.store!r} yielded {type(cdef).__name__}, not ConcreteDefinition.")

        self._connections.close_all_current_process()
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        self.initialize_empty()
        if cdefs:
            graph = ConcreteDefinitionGraph.from_roots(cdefs)
            self.register_stored_roots(graph, cdefs)
        self._clear_dirty()
        con = self._connections.connection(readonly=False)
        con.execute("PRAGMA optimize")
        if stats is not None:
            stats.store_scan_count += 1
            stats.refresh_action = "sqlite-rebuild"

    def register_stored_roots(self, graph, roots):
        roots = tuple(dict.fromkeys(roots))
        if not roots:
            return IndexWriteResult(generation=self.current_generation(), changed=False)
        encoded_nodes = tuple(_EncodedNode.from_cdef(node.definition) for node in graph.nodes())
        encoded_edges = tuple(_EncodedEdge.from_edge(edge) for edge in graph.edges())

        with self._write_transaction() as con:
            initialize_schema(con, store_key=self.source_key, canonical_version=self.canonical_version)
            generation = _read_generation(con)
            next_generation = generation + 1
            counters = _WriteCounters()
            cdef_to_id: dict[ConcreteDefinition, int] = {}
            for encoded in encoded_nodes:
                did, added = _resolve_definition_id(con, encoded, generation=next_generation)
                cdef_to_id[encoded.cdef] = did
                counters.definitions_added += int(added)
                counters.changed = counters.changed or added
                for feature in encoded.features:
                    feature_id, _ = _resolve_feature_id(con, feature.token_blob)
                    cur = con.execute(
                        """
                        INSERT OR IGNORE INTO postings (feature_id, def_id, multiplicity)
                        VALUES (?, ?, ?)
                        """,
                        (feature_id, did, feature.multiplicity),
                    )
                    if cur.rowcount:
                        counters.postings_added += 1
                        counters.changed = True
                        con.execute(
                            "UPDATE feature_tokens SET document_frequency = document_frequency + 1 WHERE feature_id = ?",
                            (feature_id,),
                        )

            for encoded in encoded_edges:
                parent_id = cdef_to_id[encoded.parent]
                child_id = cdef_to_id[encoded.child]
                cur = con.execute(
                    """
                    INSERT OR IGNORE INTO definition_edges (parent_def_id, path_hash, path_blob, child_def_id)
                    VALUES (?, ?, ?, ?)
                    """,
                    (parent_id, encoded.path_hash, encoded.path_blob, child_id),
                )
                if cur.rowcount:
                    counters.edges_added += 1
                    counters.changed = True

            for root in roots:
                root_id = cdef_to_id[root]
                stable_hash = root.stable_hash()
                cur = con.execute(
                    """
                    INSERT OR IGNORE INTO stored_roots (
                        def_id, storage_hash, relative_def_path, def_size, def_mtime_ns, indexed_generation
                    ) VALUES (?, ?, ?, NULL, NULL, ?)
                    """,
                    (root_id, stable_hash_to_blob(stable_hash), _relative_def_path(stable_hash), next_generation),
                )
                if cur.rowcount:
                    counters.roots_added += 1
                    counters.changed = True

            if counters.changed:
                generation = _bump_generation(con, next_generation)
            return IndexWriteResult(
                generation=generation,
                changed=counters.changed,
                definitions_added=counters.definitions_added,
                edges_added=counters.edges_added,
                postings_added=counters.postings_added,
                roots_added=counters.roots_added,
            )

    def remove_stored_roots(self, roots):
        roots = tuple(dict.fromkeys(roots))
        if not roots:
            return IndexWriteResult(generation=self.current_generation(), changed=False)
        with self._write_transaction() as con:
            self._ensure_schema_in_transaction(con)
            generation = _read_generation(con)
            removed = 0
            for root in roots:
                for did in _exact_ids_for_cdef(con, root):
                    cur = con.execute("DELETE FROM stored_roots WHERE def_id = ?", (did,))
                    removed += cur.rowcount
            if removed:
                generation = _bump_generation(con, generation + 1)
            return IndexWriteResult(generation=generation, changed=bool(removed), roots_removed=removed)

    def ensure_exact_stored(self, cdef, *, stats=None) -> None:
        raise QueryIndexUnavailable("SQLite exact stored lookup is not implemented yet.")

    def sync_caches(self, *, reuse_weak: bool = True) -> None:
        return None

    def close(self) -> None:
        self._connections.close_all_current_process()

    def _ensure_ready(self) -> None:
        if self._is_dirty():
            self.rebuild()
            return
        if not self.path.exists():
            self.initialize_empty()
            return
        con = self._connections.connection(readonly=True)
        validate_schema(con, store_key=self.source_key, canonical_version=self.canonical_version)

    def _ensure_schema_in_transaction(self, con) -> None:
        initialize_schema(con, store_key=self.source_key, canonical_version=self.canonical_version)

    def _is_dirty(self) -> bool:
        return self.dirty_path is not None and self.dirty_path.exists()

    def _clear_dirty(self) -> None:
        if self.dirty_path is None:
            return
        try:
            self.dirty_path.unlink()
        except FileNotFoundError:
            pass

    @contextmanager
    def _write_transaction(self):
        con = self._connections.connection(readonly=False)
        con.execute("BEGIN IMMEDIATE")
        try:
            yield con
        except Exception:
            try:
                con.execute("ROLLBACK")
            except Exception:
                self._connections.close_current()
            raise
        else:
            con.execute("COMMIT")


class SQLiteQueryIndexReadView:
    def __init__(self, con, *, source_key: str, generation: int):
        self._con = con
        self.source_key = source_key
        self.generation = generation
        self._active = True

    def close(self) -> None:
        self._active = False

    def _check_active(self) -> None:
        if not self._active:
            raise QueryIndexError("SQLite query index read view is closed.")

    def all_definition_ids(self) -> set[DefinitionId]:
        self._check_active()
        return {row[0] for row in self._con.execute("SELECT def_id FROM definitions")}

    def exact_ids(self, cdef: ConcreteDefinition) -> set[DefinitionId]:
        self._check_active()
        return set(_exact_ids_for_cdef(self._con, cdef))

    def estimate_exact_ids(self, cdef: ConcreteDefinition) -> int:
        self._check_active()
        return self._con.execute(
            "SELECT COUNT(*) FROM definitions WHERE stable_hash = ?",
            (stable_hash_to_blob(cdef.stable_hash()),),
        ).fetchone()[0]

    def estimate_local_candidates(self, requirements: tuple[FeatureRequirement, ...]) -> int:
        self._check_active()
        if not requirements:
            return self._con.execute("SELECT COUNT(*) FROM definitions").fetchone()[0]
        estimates = []
        for req in requirements:
            feature_id = self._feature_id(req.token)
            if feature_id is None:
                return 0
            row = self._con.execute(
                "SELECT document_frequency FROM feature_tokens WHERE feature_id = ?",
                (feature_id,),
            ).fetchone()
            estimates.append(0 if row is None else row[0])
        return min(estimates) if estimates else 0

    def local_candidates(
            self,
            requirements: tuple[FeatureRequirement, ...],
            *,
            within: set[DefinitionId] | None = None,
            domain=None,
            stats: QueryStats | None = None) -> set[DefinitionId]:
        self._check_active()
        if not requirements:
            if domain is not None:
                candidates = domain.all_ids()
            else:
                candidates = self.all_definition_ids() if within is None else set(within)
            if stats is not None:
                stats.candidate_count = len(candidates)
            return candidates

        posting_sets = []
        selected_requirements = []
        posting_sizes = []
        for req in requirements:
            feature_id = self._feature_id(req.token)
            if feature_id is None:
                if stats is not None:
                    stats.selected_features = requirements
                    stats.posting_sizes = tuple(0 for _ in requirements)
                    stats.candidate_count = 0
                return set()
            ids = {
                row[0]
                for row in self._con.execute(
                    "SELECT def_id FROM postings WHERE feature_id = ? AND multiplicity >= ?",
                    (feature_id, req.count),
                )
            }
            posting_sets.append(ids)
            selected_requirements.append(req)
            posting_sizes.append(len(ids))
        order = sorted(range(len(posting_sets)), key=lambda idx: posting_sizes[idx])
        candidates = set(posting_sets[order[0]])
        for idx in order[1:]:
            candidates &= posting_sets[idx]
        if within is not None:
            candidates &= set(within)
        if domain is not None:
            candidates = domain.filter(candidates)
        if stats is not None:
            stats.selected_features = tuple(selected_requirements[idx] for idx in order)
            stats.posting_sizes = tuple(posting_sizes[idx] for idx in order)
            stats.candidate_count = len(candidates)
        return candidates

    def is_stored_id(self, did: DefinitionId) -> bool:
        self._check_active()
        return self._con.execute("SELECT 1 FROM stored_roots WHERE def_id = ?", (did,)).fetchone() is not None

    def filter_stored_ids(self, ids) -> set[DefinitionId]:
        self._check_active()
        return {did for did in ids if self.is_stored_id(did)}

    def all_stored_ids(self) -> set[DefinitionId]:
        self._check_active()
        return {row[0] for row in self._con.execute("SELECT def_id FROM stored_roots")}

    def is_cached_id(self, did: DefinitionId, *, reuse_weak: bool = True) -> bool:
        return False

    def all_cached_ids(self, *, reuse_weak: bool = True) -> set[DefinitionId]:
        return set()

    def all_known_ids(self, *, reuse_weak: bool = True) -> set[DefinitionId]:
        return self.all_stored_ids()

    def nested_ids(self) -> set[DefinitionId]:
        self._check_active()
        nested = set()
        for owner_id in self.all_stored_ids():
            nested.update(self._descendant_ids(owner_id))
        return nested

    def filter_nested_ids(self, ids) -> set[DefinitionId]:
        self._check_active()
        return {did for did in ids if self.has_stored_ancestor(did)}

    def has_stored_ancestor(self, did: DefinitionId) -> bool:
        self._check_active()
        seen = set()
        stack = list(self._parent_edges(did))
        while stack:
            edge = stack.pop()
            parent_id = edge.parent_id
            if parent_id in seen:
                continue
            if self.is_stored_id(parent_id):
                return True
            seen.add(parent_id)
            stack.extend(self._parent_edges(parent_id))
        return False

    def cdefs_by_id(self, ids) -> dict[DefinitionId, ConcreteDefinition]:
        self._check_active()
        out = {}
        for did in ids:
            row = self._con.execute("SELECT cdef_blob FROM definitions WHERE def_id = ?", (did,)).fetchone()
            if row is not None:
                out[did] = decode_cdef(row[0])
        return out

    def ids_to_cdefs(self, ids) -> tuple[ConcreteDefinition, ...]:
        return tuple(self.cdefs_by_id(ids).values())

    def definition_for_id(self, did: DefinitionId) -> ConcreteDefinition:
        return self.cdefs_by_id({did})[did]

    def cdef_id(self, cdef: ConcreteDefinition) -> DefinitionId | None:
        ids = self.exact_ids(cdef)
        return next(iter(ids), None)

    def record_for_cdef(self, cdef: ConcreteDefinition) -> DefinitionRecord | None:
        did = self.cdef_id(cdef)
        if did is None:
            return None
        return DefinitionRecord(did, cdef, canonical_class_key(cdef.cls), target_local_fingerprint(cdef))

    def replica_map(self, ids) -> dict[ConcreteDefinition, tuple[str, ...]]:
        self._check_active()
        stored = self.filter_stored_ids(ids)
        return {cdef: (self.source_key,) for cdef in self.cdefs_by_id(stored).values()}

    def parents(
            self,
            child_ids: set[DefinitionId],
            path,
            *,
            unordered: bool,
            within: set[DefinitionId] | frozenset[DefinitionId] | None = None) -> set[DefinitionId]:
        self._check_active()
        if not child_ids or (within is not None and not within):
            return set()
        path_blob = encode_graph_path(path)
        path_hash = digest_blob(path_blob)
        out = set()
        for child_id in child_ids:
            rows = self._con.execute(
                """
                SELECT parent_def_id, path_blob FROM definition_edges
                WHERE child_def_id = ? AND path_hash = ?
                """,
                (child_id, path_hash),
            ) if not unordered else self._con.execute(
                "SELECT parent_def_id, path_blob FROM definition_edges WHERE child_def_id = ?",
                (child_id,),
            )
            for parent_id, row_path_blob in rows:
                row_path = decode_graph_path(row_path_blob)
                if row_path == path or (unordered and row_path.startswith(path)):
                    if within is None or parent_id in within:
                        out.add(parent_id)
        return out

    def children(
            self,
            parent_ids: set[DefinitionId],
            path,
            *,
            unordered: bool,
            within: set[DefinitionId] | frozenset[DefinitionId] | None = None) -> set[DefinitionId]:
        self._check_active()
        if not parent_ids or (within is not None and not within):
            return set()
        path_blob = encode_graph_path(path)
        path_hash = digest_blob(path_blob)
        out = set()
        for parent_id in parent_ids:
            rows = self._con.execute(
                """
                SELECT child_def_id, path_blob FROM definition_edges
                WHERE parent_def_id = ? AND path_hash = ?
                """,
                (parent_id, path_hash),
            ) if not unordered else self._con.execute(
                "SELECT child_def_id, path_blob FROM definition_edges WHERE parent_def_id = ?",
                (parent_id,),
            )
            for child_id, row_path_blob in rows:
                row_path = decode_graph_path(row_path_blob)
                if row_path == path or (unordered and row_path.startswith(path)):
                    if within is None or child_id in within:
                        out.add(child_id)
        return out

    def project_owners(self, ids) -> OwnerProjection:
        self._check_active()
        owner_ids = self._owner_ids_for_nested_ids(set(ids))
        owners = tuple(self.cdefs_by_id(owner_ids).values())
        replicas = self.replica_map(owner_ids)
        return OwnerProjection(
            owner_ids=frozenset(owner_ids),
            cdefs=owners,
            replicas={owner: replicas.get(owner, ()) for owner in owners},
        )

    def occurrence_snapshot_for_nested_ids(self, target_ids) -> OccurrenceTraversalSnapshot:
        self._check_active()
        incoming = defaultdict(list)
        cdefs = self.cdefs_by_id(set(target_ids))
        stored_ids = set()
        seen = set()
        stack = [did for did in target_ids if did in cdefs]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for edge in self._parent_edges(cur):
                incoming[edge.child_id].append(edge)
                cdefs.update(self.cdefs_by_id({edge.child_id, edge.parent_id}))
                if self.is_stored_id(edge.parent_id):
                    stored_ids.add(edge.parent_id)
                if edge.parent_id not in seen:
                    stack.append(edge.parent_id)
        return OccurrenceTraversalSnapshot(
            targets=set(target_ids),
            cdefs=cdefs,
            stored_ids=stored_ids,
            incoming={child_id: tuple(edges) for child_id, edges in incoming.items()},
            owner_replicas=self.replica_map(stored_ids),
        )

    def all_occurrence_snapshot(self) -> AllOccurrenceTraversalSnapshot:
        outgoing = defaultdict(list)
        for parent_id, path_blob, child_id in self._con.execute(
                "SELECT parent_def_id, path_blob, child_def_id FROM definition_edges"):
            edge = DefinitionEdgeRecord((parent_id, decode_graph_path(path_blob), child_id), parent_id, decode_graph_path(path_blob), child_id)
            outgoing[parent_id].append(edge)
        return AllOccurrenceTraversalSnapshot(
            cdefs=self.cdefs_by_id(self.all_definition_ids()),
            stored_ids=self.all_stored_ids(),
            outgoing={parent_id: tuple(edges) for parent_id, edges in outgoing.items()},
        )

    def _feature_id(self, token) -> int | None:
        token_blob = encode_feature_token(token)
        token_hash = digest_blob(token_blob)
        for feature_id, row_blob in self._con.execute(
                "SELECT feature_id, token_blob FROM feature_tokens WHERE token_hash = ?",
                (token_hash,)):
            if decode_feature_token(row_blob) == token:
                return feature_id
        return None

    def _parent_edges(self, child_id: DefinitionId) -> tuple[DefinitionEdgeRecord, ...]:
        rows = self._con.execute(
            "SELECT parent_def_id, path_blob, child_def_id FROM definition_edges WHERE child_def_id = ?",
            (child_id,),
        )
        return tuple(
            DefinitionEdgeRecord((parent_id, decode_graph_path(path_blob), row_child_id), parent_id, decode_graph_path(path_blob), row_child_id)
            for parent_id, path_blob, row_child_id in rows
        )

    def _owner_ids_for_nested_ids(self, ids: set[DefinitionId]) -> set[DefinitionId]:
        owners = set()
        seen = set(ids)
        stack = list(ids)
        while stack:
            cur = stack.pop()
            for edge in self._parent_edges(cur):
                if self.is_stored_id(edge.parent_id):
                    owners.add(edge.parent_id)
                if edge.parent_id not in seen:
                    seen.add(edge.parent_id)
                    stack.append(edge.parent_id)
        return owners

    def _descendant_ids(self, owner_id: DefinitionId) -> set[DefinitionId]:
        descendants = set()
        stack = [row[0] for row in self._con.execute(
            "SELECT child_def_id FROM definition_edges WHERE parent_def_id = ?",
            (owner_id,),
        )]
        while stack:
            cur = stack.pop()
            if cur in descendants:
                continue
            descendants.add(cur)
            stack.extend(row[0] for row in self._con.execute(
                "SELECT child_def_id FROM definition_edges WHERE parent_def_id = ?",
                (cur,),
            ))
        return descendants


class _WriteCounters:
    def __init__(self):
        self.changed = False
        self.definitions_added = 0
        self.edges_added = 0
        self.postings_added = 0
        self.roots_added = 0


class _EncodedFeature:
    def __init__(self, token_blob: bytes, multiplicity: int):
        self.token_blob = token_blob
        self.multiplicity = multiplicity


class _EncodedNode:
    def __init__(self, cdef, stable_hash, class_key, cdef_blob, features):
        self.cdef = cdef
        self.stable_hash = stable_hash
        self.class_key = class_key
        self.cdef_blob = cdef_blob
        self.features = features

    @classmethod
    def from_cdef(cls, cdef: ConcreteDefinition):
        fingerprint = target_local_fingerprint(cdef)
        return cls(
            cdef=cdef,
            stable_hash=stable_hash_to_blob(cdef.stable_hash()),
            class_key=stable_hash_to_blob(canonical_class_key(cdef.cls)),
            cdef_blob=encode_cdef(cdef),
            features=tuple(
                _EncodedFeature(encode_feature_token(token), count)
                for token, count in fingerprint.counts.items()
            ),
        )


class _EncodedEdge:
    def __init__(self, parent, child, path_blob, path_hash):
        self.parent = parent
        self.child = child
        self.path_blob = path_blob
        self.path_hash = path_hash

    @classmethod
    def from_edge(cls, edge):
        path_blob = encode_graph_path(edge.path)
        return cls(edge.parent, edge.child, path_blob, digest_blob(path_blob))


def _resolve_definition_id(con, encoded: _EncodedNode, *, generation: int) -> tuple[int, bool]:
    max_ordinal = -1
    for did, cdef_blob, ordinal in con.execute(
            "SELECT def_id, cdef_blob, collision_ordinal FROM definitions WHERE stable_hash = ? ORDER BY collision_ordinal",
            (encoded.stable_hash,)):
        max_ordinal = max(max_ordinal, ordinal)
        if cdef_equal(decode_cdef(cdef_blob), encoded.cdef):
            return did, False
    ordinal = max_ordinal + 1
    cur = con.execute(
        """
        INSERT INTO definitions (stable_hash, collision_ordinal, class_key, cdef_blob, created_generation)
        VALUES (?, ?, ?, ?, ?)
        """,
        (encoded.stable_hash, ordinal, encoded.class_key, encoded.cdef_blob, generation),
    )
    return cur.lastrowid, True


def _resolve_feature_id(con, token_blob: bytes) -> tuple[int, bool]:
    token = decode_feature_token(token_blob)
    token_hash = digest_blob(token_blob)
    max_ordinal = -1
    for feature_id, row_blob, ordinal in con.execute(
            "SELECT feature_id, token_blob, collision_ordinal FROM feature_tokens WHERE token_hash = ? ORDER BY collision_ordinal",
            (token_hash,)):
        max_ordinal = max(max_ordinal, ordinal)
        if decode_feature_token(row_blob) == token:
            return feature_id, False
    cur = con.execute(
        """
        INSERT INTO feature_tokens (token_hash, collision_ordinal, token_blob, document_frequency)
        VALUES (?, ?, ?, 0)
        """,
        (token_hash, max_ordinal + 1, token_blob),
    )
    return cur.lastrowid, True


def _exact_ids_for_cdef(con, cdef: ConcreteDefinition) -> tuple[int, ...]:
    out = []
    for did, cdef_blob in con.execute(
            "SELECT def_id, cdef_blob FROM definitions WHERE stable_hash = ?",
            (stable_hash_to_blob(cdef.stable_hash()),)):
        if cdef_equal(decode_cdef(cdef_blob), cdef):
            out.append(did)
    return tuple(out)


def _read_generation(con) -> int:
    row = con.execute("SELECT generation FROM catalog_state WHERE singleton = 1").fetchone()
    if row is None:
        raise QueryIndexError("SQLite query index is missing catalog_state generation.")
    return row[0]


def _bump_generation(con, generation: int) -> int:
    con.execute(
        "UPDATE catalog_state SET generation = ?, updated_at = ? WHERE singleton = 1",
        (generation, datetime.now(timezone.utc).isoformat()),
    )
    return generation


def _relative_def_path(stable_hash: str) -> str:
    return f"objects/{stable_hash[:2]}/{stable_hash}/def.pkl"
