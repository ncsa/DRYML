from __future__ import annotations

from contextlib import contextmanager
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import sys
import time
from uuid import uuid4

from ...cdef_graph import ConcreteDefinitionGraph, EdgeKind, as_query_index_graph
from ...definition import ConcreteDefinition
from ..codecs import (
    QueryIndexCodec,
    digest_blob,
    encode_reference,
)
from ..fingerprint import canonical_class_key, target_local_fingerprint
from ..model import (
    AllOccurrenceTraversalSnapshot,
    DefinitionEdgeRecord,
    DefinitionId,
    DefinitionRecord,
    FeatureRequirement,
    IndexWriteResult,
    OccurrenceTraversalSnapshot,
    OwnerProjection,
    QueryIndexBusy,
    QueryIndexDirty,
    QueryIndexError,
    QueryIndexIncompatible,
    QueryIndexStatus,
    QueryIndexUnavailable,
    QueryStats,
    ReconcileReport,
    ValidationIssue,
    ValidationReport,
    CANONICAL_QUERY_SEMANTICS_VERSION,
)
from ..lowering import CandidateRelation, LoweredQueryPlan, LoweringDiagnostics, PagedResultCursor, PhysicalRelationPlan, QueryTerminal, ScanPolicy
from ..utils import cdef_equal, chunked, feature_token_equal, stable_hash_from_blob, stable_hash_to_blob
from . import SQLiteQueryIndexConfig, require_sqlite
from .connection import SQLiteConnectionManager
from .schema import SQLITE_QUERY_INDEX_SCHEMA_VERSION, initialize_schema, stored_compatibility_decision, validate_schema
from .utils import is_sqlite_busy_error, wal_runtime_is_known_safe
from .lowering import SQLiteOptimizerPolicy, SQLiteRelationCompiler


try:
    import fcntl
except ImportError:  # pragma: no cover - selected only on POSIX hosts.
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - selected only on Windows hosts.
    msvcrt = None


_CODEC = QueryIndexCodec()
_REBUILD_BATCH_SIZE = 500
_BUILD_CLAIM_STALE_SECONDS = 300.0
_BUILD_CLAIM_WAIT_SECONDS = 30.0


def _claim_lock_backend() -> str:
    """Return the platform-selected primitive used for rebuild claim locking."""

    return "windows" if os.name == "nt" else "posix"


def _try_lock_claim_file(fd: int) -> bool:
    """Try to hold the cross-process lock for a query-index claim file."""

    if _claim_lock_backend() == "posix":
        if fcntl is None:
            raise QueryIndexUnavailable("POSIX claim locking is unavailable for the SQLite query index.")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True

    if msvcrt is None:
        raise QueryIndexUnavailable("Windows claim locking is unavailable for the SQLite query index.")
    if os.fstat(fd).st_size == 0:
        os.write(fd, b"\0")
    os.lseek(fd, 0, os.SEEK_SET)
    try:
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    except OSError:
        return False
    return True


def _unlock_claim_file(fd: int) -> None:
    """Release a query-index claim file lock held by this process."""

    if _claim_lock_backend() == "posix":
        if fcntl is None:
            raise QueryIndexUnavailable("POSIX claim locking is unavailable for the SQLite query index.")
        fcntl.flock(fd, fcntl.LOCK_UN)
        return
    if msvcrt is None:
        raise QueryIndexUnavailable("Windows claim locking is unavailable for the SQLite query index.")
    os.lseek(fd, 0, os.SEEK_SET)
    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


class SQLiteStoreQueryIndex:
    """Persistent query-index backend for one physical Store.

    The backend stores graph nodes, local feature postings, direct definition
    edges, and active stored-root membership in a SQLite sidecar. Store object
    files remain authoritative: this index can be rebuilt or reconciled from the
    owning Store and never owns object state.
    """

    def __init__(
            self,
            *,
            source_key: str,
            path: str | Path,
            config: SQLiteQueryIndexConfig | None = None,
            canonical_version: int = CANONICAL_QUERY_SEMANTICS_VERSION,
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

    def initialize_empty(self, *, build_state: str = "ready", generation: int = 0) -> None:
        """Initialize an empty sidecar with an explicit state and generation.

        Args:
            build_state: Catalog state exposed during initialization.
            generation: Initial non-negative generation. Replacement rebuilds
                use this to remain monotonic across atomic sidecar publication.

        Raises:
            QueryIndexError: If the sidecar schema cannot be initialized.
            QueryIndexUnavailable: If SQLite is unavailable.
        """

        def initialize(con):
            initialize_schema(
                con,
                store_key=self.source_key,
                canonical_version=self.canonical_version,
                build_state=build_state,
            )
            if generation:
                _bump_generation(con, generation)

        self._run_write_transaction(initialize)
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
                row_counts=_empty_row_counts(),
                diagnostics={"dirty_marker": str(self.dirty_path) if self.dirty_path is not None else None},
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
                row_counts=_empty_row_counts(),
                diagnostics={"exists": False},
            )
        sqlite3 = require_sqlite()
        try:
            con = self._connections.connection(readonly=True)
            validate_schema(con, store_key=self.source_key, canonical_version=self.canonical_version, require_ready=False)
            row = con.execute(
                """
                SELECT index_uuid, generation, schema_version, graph_schema_version, path_schema_version,
                       fingerprint_version, cdef_codec_version, feature_codec_version,
                       query_index_codec_version, canonical_version, build_state, dirty
                FROM catalog_state
                WHERE singleton = 1
                """
            ).fetchone()
            journal_mode = con.execute("PRAGMA journal_mode").fetchone()[0]
            row_counts = _row_counts(con)
            diagnostics = _status_diagnostics(con, journal_mode)
        except QueryIndexUnavailable:
            raise
        except Exception as exc:
            return QueryIndexStatus(
                backend="sqlite",
                store_key=self.source_key,
                generation=None,
                schema_version=SQLITE_QUERY_INDEX_SCHEMA_VERSION,
                semantic_versions={},
                state="corrupt" if _is_sqlite_corrupt_exception(exc) else "incompatible",
                sqlite_version=sqlite3.sqlite_version_info,
                path=str(self.path),
                row_counts=_empty_row_counts(),
                diagnostics={"error": repr(exc)},
            )
        if row is None:
            state = "incompatible"
            generation = None
            schema_version = None
            semantic_versions = {}
        else:
            diagnostics["index_uuid"] = row[0]
            diagnostics["dirty_flag"] = bool(row[11])
            generation = row[1]
            schema_version = row[2]
            semantic_versions = {
                "graph_schema_version": row[3],
                "path_schema_version": row[4],
                "fingerprint_version": row[5],
                "cdef_codec_version": row[6],
                "feature_codec_version": row[7],
                "query_index_codec_version": row[8],
                "canonical_version": row[9],
            }
            state = "dirty" if row[11] else row[10]
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
            row_counts=row_counts,
            diagnostics=diagnostics,
        )

    def validate(self, *, thorough: bool = False) -> ValidationReport:
        issues: list[ValidationIssue] = []
        diagnostics = {}
        if self._is_dirty():
            issues.append(ValidationIssue("error", "SQLite query index is dirty."))
            return ValidationReport("sqlite", self.source_key, False, tuple(issues), row_counts=_empty_row_counts(), diagnostics={"dirty_marker": str(self.dirty_path) if self.dirty_path is not None else None})
        if not self.path.exists():
            issues.append(ValidationIssue("error", "SQLite query index is missing."))
            return ValidationReport("sqlite", self.source_key, False, tuple(issues), row_counts=_empty_row_counts(), diagnostics={"exists": False})
        con = None
        try:
            con = self._connections.connection(readonly=True)
            validate_schema(con, store_key=self.source_key, canonical_version=self.canonical_version, require_ready=False)
            build_state = _read_build_state(con)
            diagnostics["build_state"] = build_state
            journal_mode = con.execute("PRAGMA journal_mode").fetchone()[0]
            diagnostics.update(_status_diagnostics(con, journal_mode))
            if build_state != "ready":
                issues.append(ValidationIssue("error", "SQLite query index is not ready.", f"build_state={build_state!r}"))
            _validate_sqlite_integrity(con, issues)
            if thorough:
                self._validate_decodable_rows(con, issues)
                self._validate_stored_roots(con, issues)
                self._validate_store_roots(con, issues)
            counts = _row_counts(con)
        except Exception as exc:
            issues.append(ValidationIssue("error", "SQLite query index validation failed.", repr(exc)))
            try:
                counts = _row_counts(con) if con is not None else _empty_row_counts()
            except Exception:
                counts = _empty_row_counts()
        return ValidationReport("sqlite", self.source_key, not any(issue.severity == "error" for issue in issues), tuple(issues), row_counts=counts, diagnostics=diagnostics)

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
            self.rebuild(stats=stats, force=True)
            return
        status = self.status()
        if status.state in {"missing", "dirty", "building", "incompatible", "corrupt"}:
            self.rebuild(stats=stats, quarantine_existing=status.state in {"corrupt", "incompatible"}, force=False)
            return
        self._ensure_ready()

    def rebuild(self, *, stats: QueryStats | None = None, quarantine_existing: bool = False, force: bool = True) -> None:
        """Recreate this SQLite index from the owning Store's root definitions.

        Rebuilds acquire a sidecar build claim so concurrent callers do not all
        scan the Store. They preflight all authoritative roots, validate a
        complete sibling sidecar while it is `building`, then atomically replace
        the active sidecar. Store roots are never changed by rebuild or recovery.

        Args:
            stats: Optional query statistics updated with the refresh action.
            quarantine_existing: Preserve an incompatible or corrupt previous
                sidecar after successful replacement when possible.
            force: Acquire the build claim even when another ready index exists.

        Raises:
            QueryIndexError: If authoritative roots are invalid, staging fails,
                or the replacement sidecar does not validate.
            QueryIndexUnavailable: If SQLite is unavailable.
        """

        with self._build_claim(force=force) as acquired:
            if not acquired:
                if stats is not None:
                    stats.refresh_action = "sqlite-rebuild-wait"
                return
            self._rebuild_owned(stats=stats, quarantine_existing=quarantine_existing)

    def reconcile(self) -> ReconcileReport:
        """Validate this Store index against object files and rebuild if needed.

        SQLite v1 uses an exclusive rebuild policy for missing, dirty, corrupt,
        incompatible, stale, or divergent indexes. Object files remain the source
        of truth. If Store files are themselves inconsistent, reconciliation
        reports validation issues and leaves the existing index unmodified.
        """

        before = self.status()
        if before.state in {"missing", "dirty", "building", "incompatible", "corrupt"}:
            return self._reconcile_by_rebuild(before, (), quarantine_existing=before.state in {"corrupt", "incompatible"})
        report = self.validate(thorough=True)
        if report.ok:
            return ReconcileReport(
                backend=before.backend,
                store_key=before.store_key,
                changed=False,
                action="validate",
                generation_before=before.generation,
                generation_after=before.generation,
                definitions_scanned=(report.row_counts or {}).get("stored_roots", 0),
                validated=True,
                issues=report.issues,
                diagnostics=report.diagnostics,
            )
        return self._reconcile_by_rebuild(before, report.issues)

    def _reconcile_by_rebuild(
            self,
            before: QueryIndexStatus,
            issues: tuple[ValidationIssue, ...],
            *,
            quarantine_existing: bool = False) -> ReconcileReport:
        try:
            force = before.state not in {"missing", "dirty", "building", "incompatible", "corrupt"}
            self.rebuild(quarantine_existing=quarantine_existing, force=force)
        except Exception as exc:
            return ReconcileReport(
                backend=before.backend,
                store_key=before.store_key,
                changed=False,
                action="validate",
                generation_before=before.generation,
                generation_after=before.generation,
                validated=True,
                issues=(*issues, ValidationIssue("error", "SQLite query-index rebuild failed.", repr(exc))),
            )
        after = self.status()
        return ReconcileReport(
            backend=after.backend,
            store_key=after.store_key,
            changed=True,
            action="rebuild",
            generation_before=before.generation,
            generation_after=after.generation,
            definitions_scanned=(after.row_counts or {}).get("stored_roots", 0),
            validated=True,
            issues=issues,
            diagnostics=after.diagnostics,
        )

    def _rebuild_owned(self, *, stats: QueryStats | None = None, quarantine_existing: bool = False) -> None:
        if self.store is None or not hasattr(self.store, "iter_definition_records"):
            raise QueryIndexUnavailable("SQLite query-index rebuild requires an owning Store with iter_definition_records().")
        compatibility = self._assert_sidecar_rebuildable()

        dirty_markers = self._dirty_markers()
        generation_seed = self._replacement_generation_seed()
        try:
            roots = self._preflight_store_roots()
        except Exception:
            self._mark_dirty()
            raise

        replacement_path = self._replacement_path()
        replacement = SQLiteStoreQueryIndex(
            source_key=self.source_key,
            path=replacement_path,
            config=replace(self.config, path=replacement_path),
            canonical_version=self.canonical_version,
            store=self.store,
        )
        scanned = 0
        visible_progress = compatibility == "rebuild"
        if visible_progress:
            print("DRYML query index metadata is older; rebuilding derived index from Store authority.", file=sys.stderr, flush=True)
        try:
            replacement.initialize_empty(build_state="building", generation=generation_seed)
            for cdefs in chunked(roots, _REBUILD_BATCH_SIZE):
                scanned += len(cdefs)
                if visible_progress:
                    print(f"DRYML query index rebuild progress: {scanned}/{len(roots)} roots", file=sys.stderr, flush=True)
                graph = ConcreteDefinitionGraph.for_query_index_roots(cdefs)
                replacement._register_stored_roots(graph, cdefs, require_ready=False)
            replacement._register_reference_authority(require_ready=False)
            replacement._validate_rebuild_before_ready(roots=roots)
            replacement._set_build_state("ready")
            con = replacement._connections.connection(readonly=False)
            con.execute("PRAGMA optimize")
            replacement.close()
            self._checkpoint_and_cleanup_sidecars(replacement_path, label="staged")
            SQLiteConnectionManager._close_current_thread_for_path(self.path)
            self._checkpoint_and_cleanup_sidecars(
                self.path,
                label="canonical",
                allow_corrupt=quarantine_existing,
            )
            self._activate_replacement(replacement_path, quarantine_existing=quarantine_existing)
            self._cleanup_replacement(replacement_path)
        except BaseException:
            try:
                replacement.close()
            finally:
                self._cleanup_replacement(replacement_path)
            raise
        self._clear_dirty(
            dirty_markers, roots=roots, clear_unscoped=True,
            clear_scoped=True,
        )
        if stats is not None:
            stats.store_scan_count += 1
            stats.refresh_action = "sqlite-rebuild"
            stats.result_count = scanned

    def _preflight_store_roots(self) -> tuple[ConcreteDefinition, ...]:
        """Read and type-check authoritative root membership before rebuild."""

        roots: list[ConcreteDefinition] = []
        by_hash: dict[str, list[ConcreteDefinition]] = defaultdict(list)

        def add(cdef) -> None:
            if not isinstance(cdef, ConcreteDefinition):
                raise QueryIndexError(
                    f"Store {self.store!r} yielded {type(cdef).__name__}, "
                    "not ConcreteDefinition."
                )
            bucket = by_hash[cdef.graph_hash()]
            if any(existing.graph_equal(cdef) for existing in bucket):
                return
            bucket.append(cdef)
            roots.append(cdef)

        for definition in self.store.authoritative_root_definitions():
            add(definition)
        return tuple(roots)

    def _register_reference_authority(self, *, require_ready: bool) -> None:
        """Cache immutable reference facts while leaving Store records authoritative.

        The rows are rebuilt as one replacement-sidecar unit. They accelerate no
        terminal yet: reference queries re-read Store authority so a stale or
        corrupted derived projection can never omit a result or hide a conflict.
        """

        self._register_reference_rows(
            tuple(_reference_authority_rows(self.store)),
            require_ready=require_ready,
            replace=True,
        )

    def _register_reference_rows(
            self, rows, *, require_ready: bool, replace: bool) -> None:
        """Register complete or incremental advisory reference projections."""

        def operation(con):
            validate_schema(
                con,
                store_key=self.source_key,
                canonical_version=self.canonical_version,
                require_ready=require_ready,
            )
            if replace:
                con.execute("DELETE FROM reference_object_ids")
                con.execute("DELETE FROM reference_records")
            for row in rows:
                con.execute(
                    """
                    INSERT OR IGNORE INTO reference_records (
                        source_kind, source_digest, owner_kind, owner_digest,
                        path_blob, reference_kind, reference_digest,
                        reference_blob, state_hashes_blob, alias
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    row[:10],
                )
                reference = row[10]
                state_hashes = row[11]
                object_ref = reference.object if hasattr(reference, "object") else reference
                for path, object_id in object_ref.objects.items():
                    state_hash = None if state_hashes is None else state_hashes.get(path)
                    con.execute(
                        """
                        INSERT OR IGNORE INTO reference_object_ids (
                            reference_kind, reference_digest, object_id_blob,
                            namespace_blob, path_blob, state_hash
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            row[5], row[6], encode_reference(object_id),
                            "/".join(object_id.namespace).encode("ascii"),
                            _CODEC.encode_graph_path(path), state_hash,
                        ),
                    )

        self._run_write_transaction(operation)

    def reference_candidate_sources(
            self, *, object_id=None, namespace=None, object_ref=None,
            state_hash=None):
        """Return indexed authority sources for selective reference filters.

        Args:
            object_id: Optional complete ObjectId containment filter.
            namespace: Optional ObjectId namespace-prefix filter.
            object_ref: Optional exact ObjectRef filter.
            state_hash: Optional local-state hash containment filter.

        Returns:
            Sorted ``(source_kind, source_digest)`` candidates, or ``None`` when
            no supplied filter can narrow authority safely.

        Raises:
            QueryIndexError: If the ready derived sidecar cannot be queried.

        Side Effects:
            None. Callers must verify every candidate against Store authority.
        """
        namespace_filter = namespace if namespace else None
        if (
                object_id is None and namespace_filter is None
                and object_ref is None and state_hash is None
        ):
            return None
        joins = ""
        conditions = [
            "rr.source_kind IN ('definition', 'declaration', 'state-ref')"
        ]
        parameters = []
        if object_id is not None or namespace_filter is not None or state_hash is not None:
            joins = " JOIN reference_object_ids oi USING (reference_kind, reference_digest)"
        if object_id is not None:
            conditions.append("oi.object_id_blob = ?")
            parameters.append(encode_reference(object_id))
        if namespace_filter is not None:
            encoded = "/".join(namespace_filter)
            conditions.append(
                "(CAST(oi.namespace_blob AS TEXT) = ? "
                "OR CAST(oi.namespace_blob AS TEXT) LIKE ?)"
            )
            parameters.extend((encoded, f"{encoded}/%"))
        if object_ref is not None:
            conditions.append("rr.reference_kind = 'object'")
            conditions.append("rr.reference_digest = ?")
            parameters.append(object_ref.digest())
        if state_hash is not None:
            conditions.append("oi.state_hash = ?")
            parameters.append(state_hash)
        try:
            con = self._connections.connection(readonly=True)
            validate_schema(
                con,
                store_key=self.source_key,
                canonical_version=self.canonical_version,
                require_ready=True,
            )
            rows = con.execute(
                "SELECT DISTINCT rr.source_kind, rr.source_digest "
                f"FROM reference_records rr{joins} WHERE "
                + " AND ".join(conditions)
                + " ORDER BY rr.source_kind, rr.source_digest",
                tuple(parameters),
            )
        except Exception as error:
            raise QueryIndexError(
                f"Reference candidate lookup failed: {error}"
            ) from error
        return tuple((row[0], row[1]) for row in rows)

    def _assert_sidecar_rebuildable(self) -> str:
        """Reject future sidecars before scanning Store authority or index rows."""

        if not self.path.exists():
            return "missing"
        try:
            con = self._connections.connection(readonly=True)
            decision = stored_compatibility_decision(
                con,
                store_key=self.source_key,
                canonical_version=self.canonical_version,
            )
        except Exception as exc:
            if _is_sqlite_corrupt_exception(exc):
                return
            raise QueryIndexError("Could not inspect SQLite query-index compatibility before rebuild.") from exc
        if decision == "future-unsupported":
            raise QueryIndexIncompatible(
                "SQLite query-index metadata is future, missing, or malformed; refusing to rebuild over it."
            )
        return decision

    def register_stored_roots(self, graph, roots):
        return self._register_stored_roots(graph, roots, require_ready=True)

    def register_graph(self, graph):
        return self._register_stored_roots(graph, (), require_ready=True)

    def activate_stored_roots(self, graph, roots):
        return self._register_stored_roots(graph, roots, require_ready=True)

    def register_saved_graph(self, graph, roots, state_refs=()):
        """Incrementally register roots and StateRefs from one completed save.

        Args:
            graph: Complete query graph for the publication.
            roots: Definitions made independently queryable as stored roots.
            state_refs: Newly published exact StateRefs whose reference rows are
                advisory acceleration only.

        Returns:
            IndexWriteResult describing graph and root changes.

        Raises:
            QueryIndexError: If the ready sidecar cannot accept the update.

        Side Effects:
            Mutates the SQLite sidecar and clears captured publication markers
            according to the documented deferred dirty-marker policy.
        """
        dirty_markers = self._dirty_markers()
        result = self.register_stored_roots(graph, roots)
        rows = tuple(
            row
            for state_ref in state_refs
            for row in _state_reference_authority_rows(state_ref)
        )
        if rows:
            self._register_reference_rows(
                rows, require_ready=True, replace=False
            )
        self._clear_dirty(
            dirty_markers, roots=tuple(roots), clear_unscoped=True,
            clear_scoped=True,
        )
        return result

    def _register_stored_roots(self, graph, roots, *, require_ready: bool):
        if require_ready and self._build_claim_path().exists():
            self._mark_dirty()
            raise QueryIndexBusy(
                "Cannot register Store roots while a SQLite query-index rebuild is active."
            )
        dirty_markers = self._dirty_markers()
        roots = tuple(dict.fromkeys(roots))
        graph = as_query_index_graph(graph, roots if roots else graph.roots)
        graph_nodes = graph.nodes()
        node_hash_blobs = {node.definition: stable_hash_to_blob(node.stable_hash) for node in graph_nodes}
        encoded_edges = tuple(_EncodedEdge.from_edge(edge) for edge in graph.edges())
        if not roots and not graph_nodes and not encoded_edges:
            return IndexWriteResult(generation=self.current_generation(), changed=False)
        existing_node_ids, missing_nodes = self._preflight_graph_nodes(
            graph_nodes,
            node_hash_blobs,
            require_ready=require_ready,
        )
        encoded_missing_nodes = tuple(
            _EncodedNode.from_cdef(cdef, stable_hash_blob=node_hash_blobs[cdef])
            for cdef in missing_nodes
        )

        def operation(con):
            initialize_schema(con, store_key=self.source_key, canonical_version=self.canonical_version)
            validate_schema(
                con,
                store_key=self.source_key,
                canonical_version=self.canonical_version,
                require_ready=require_ready,
            )
            generation = _read_generation(con)
            next_generation = generation + 1
            counters = _WriteCounters()
            cdef_to_id: dict[ConcreteDefinition, int] = dict(existing_node_ids)

            for encoded in encoded_missing_nodes:
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
                    INSERT OR IGNORE INTO definition_edges (parent_def_id, path_hash, path_blob, unordered, edge_kind, child_def_id)
                    VALUES (?, ?, ?, 0, ?, ?)
                    """,
                    (parent_id, encoded.path_hash, encoded.path_blob, encoded.edge_kind.value, child_id),
                )
                if cur.rowcount:
                    counters.edges_added += 1
                    counters.changed = True

            for root in roots:
                root_id = cdef_to_id[root]
                storage_hash, relative_def_path, def_size, def_mtime_ns = self._root_record_metadata(root)
                cur = con.execute(
                    """
                    INSERT OR IGNORE INTO stored_roots (
                        def_id, storage_hash, relative_def_path, def_size, def_mtime_ns, indexed_generation
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (root_id, storage_hash, relative_def_path, def_size, def_mtime_ns, next_generation),
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

        result = self._run_write_transaction(operation)
        self._clear_dirty(dirty_markers, roots=roots)
        return result

    def _preflight_graph_nodes(self, graph_nodes, node_hash_blobs, *, require_ready: bool):
        missing_nodes = [node.definition for node in graph_nodes]
        if not self.path.exists():
            return {}, tuple(missing_nodes)

        con = self._connections.connection(readonly=True)
        validate_schema(
            con,
            store_key=self.source_key,
            canonical_version=self.canonical_version,
            require_ready=require_ready,
        )
        existing_node_ids: dict[ConcreteDefinition, int] = {}
        missing_nodes = []
        for node in graph_nodes:
            did = _existing_definition_id(con, node.definition, stable_hash_blob=node_hash_blobs[node.definition])
            if did is None:
                missing_nodes.append(node.definition)
            else:
                existing_node_ids[node.definition] = did
        return existing_node_ids, tuple(missing_nodes)

    def remove_stored_roots(self, roots):
        roots = tuple(dict.fromkeys(roots))
        if not roots:
            return IndexWriteResult(generation=self.current_generation(), changed=False)
        def operation(con):
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

        return self._run_write_transaction(operation)

    def ensure_exact_stored(self, cdef, *, stats=None) -> bool:
        self._ensure_ready()
        with self.read_view(include_cached=False) as view:
            if view.filter_stored_ids(view.exact_ids(cdef)):
                if stats is not None:
                    stats.fast_path = "exact-root-index"
                return True

        try:
            is_stored_root = any(
                root.graph_equal(cdef)
                for root in self.store.authoritative_root_definitions()
            )
        except (AttributeError, NotImplementedError):
            return False
        if not is_stored_root:
            return False

        reader = getattr(self.store, "read_definition", None)
        if reader is None:
            return False
        persisted = reader(cdef)
        if persisted is None or not cdef_equal(persisted, cdef):
            return False

        self.register_stored_roots(ConcreteDefinitionGraph.from_root(persisted), [persisted])
        if stats is not None:
            stats.fast_path = "exact-root-store-has"
        return True

    def sync_caches(self, *, reuse_weak: bool = True) -> None:
        return None

    def close(self) -> None:
        self._connections.close_all_current_process()

    def _ensure_ready(self) -> None:
        if self._is_dirty():
            self.rebuild()
            return
        if not self.path.exists():
            if self._build_claim_path().exists():
                # A missing active path can be a staged rebuild, not a new index.
                self.rebuild(force=False)
                return
            with self._build_claim(force=False) as acquired:
                if acquired and not self.path.exists():
                    self.initialize_empty()
            return
        con = self._connections.connection(readonly=True)
        try:
            validate_schema(con, store_key=self.source_key, canonical_version=self.canonical_version)
        except QueryIndexDirty:
            if self.store is None or not hasattr(self.store, "iter_definition_records"):
                raise
            self.rebuild()

    def _ensure_schema_in_transaction(self, con) -> None:
        initialize_schema(con, store_key=self.source_key, canonical_version=self.canonical_version)
        validate_schema(con, store_key=self.source_key, canonical_version=self.canonical_version)

    def _is_dirty(self) -> bool:
        return bool(self._dirty_markers())

    def _dirty_markers(self) -> tuple[Path, ...]:
        if self.dirty_path is None:
            return ()
        markers = tuple(self.dirty_path.parent.glob(f"{self.dirty_path.name}.*"))
        if self.dirty_path.exists():
            markers = (*markers, self.dirty_path)
        return markers

    def _clear_dirty(
            self,
            markers: tuple[Path, ...], *,
            roots: tuple[ConcreteDefinition, ...],
            clear_unscoped: bool = False,
            clear_scoped: bool = False) -> None:
        root_keys = {self._root_marker_key(root) for root in roots}
        for marker in markers:
            try:
                mutation = marker.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError):
                mutation = ""
            is_scoped = (
                len(mutation) == 64
                and all(char in "0123456789abcdef" for char in mutation)
            )
            if not (
                    (is_scoped and (clear_scoped or mutation in root_keys))
                    or (clear_unscoped and not is_scoped)
            ):
                continue
            try:
                marker.unlink()
            except FileNotFoundError:
                pass

    def _mark_dirty(self) -> None:
        if self.dirty_path is None:
            return
        self.dirty_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path = self.dirty_path.with_name(f"{self.dirty_path.name}.{uuid4().hex}")
        with open(marker_path, "x", encoding="utf-8") as file:
            file.write("dirty\n")

    def _replacement_generation_seed(self) -> int:
        current = -1
        if self.path.exists():
            try:
                current = _read_generation(self._connections.connection(readonly=True))
            except Exception:
                pass
        return max(current + 1, time.time_ns())

    def _root_marker_key(self, cdef: ConcreteDefinition) -> str:
        """Return the dirty-token identity for a stored root.

        Current DirStores publish DefinitionRecords, so their token must name
        that record rather than the retired object-root stable hash. Standalone
        SQLite indexes retain the stable-hash key used by their public API.
        """
        metadata = getattr(self.store, "query_index_record_metadata", None)
        if metadata is not None:
            result = metadata(cdef)
            if result is not None:
                return result[0]
        return cdef.stable_hash()

    def _root_record_metadata(self, cdef: ConcreteDefinition) -> tuple[bytes, str, int | None, int | None]:
        """Return derived root-row metadata without treating the row as authority."""
        metadata = getattr(self.store, "query_index_record_metadata", None)
        if metadata is not None:
            result = metadata(cdef)
            if result is None:
                raise QueryIndexError("Stored root has no matching authoritative DefinitionRecord.")
            record_digest, relative_path, size, mtime_ns = result
            return stable_hash_to_blob(record_digest), relative_path, size, mtime_ns
        stable_hash = cdef.stable_hash()
        return stable_hash_to_blob(stable_hash), _relative_def_path(stable_hash), None, None

    def _validate_rebuild_before_ready(self, *, roots: tuple[ConcreteDefinition, ...]) -> None:
        con = self._connections.connection(readonly=True)
        issues: list[ValidationIssue] = []
        validate_schema(con, store_key=self.source_key, canonical_version=self.canonical_version, require_ready=False)
        _validate_sqlite_integrity(con, issues)
        self._validate_decodable_rows(con, issues)
        self._validate_stored_roots(con, issues)
        self._validate_reference_rows(con, issues)
        self._validate_store_roots(con, issues, roots=roots)
        errors = tuple(issue for issue in issues if issue.severity == "error")
        if errors:
            detail = "; ".join(issue.message for issue in errors[:3])
            raise QueryIndexError(f"SQLite query-index rebuild validation failed before ready: {detail}")

    def _validate_decodable_rows(self, con, issues: list[ValidationIssue]) -> None:
        for did, cdef_blob in con.execute("SELECT def_id, cdef_blob FROM definitions"):
            try:
                cdef = _CODEC.decode_cdef(cdef_blob)
                if stable_hash_to_blob(cdef.stable_hash()) != con.execute(
                        "SELECT stable_hash FROM definitions WHERE def_id = ?", (did,)).fetchone()[0]:
                    issues.append(ValidationIssue("error", "Definition stable hash mismatch.", str(did)))
            except Exception as exc:
                issues.append(ValidationIssue("error", "Definition row failed to decode.", f"{did}: {exc!r}"))
        for feature_id, token_blob in con.execute("SELECT feature_id, token_blob FROM feature_tokens"):
            try:
                _CODEC.decode_feature_token(token_blob)
            except Exception as exc:
                issues.append(ValidationIssue("error", "Feature token row failed to decode.", f"{feature_id}: {exc!r}"))
        for parent_id, path_blob, child_id in con.execute("SELECT parent_def_id, path_blob, child_def_id FROM definition_edges"):
            try:
                _CODEC.decode_graph_path(path_blob)
            except Exception as exc:
                issues.append(ValidationIssue("error", "Definition edge path failed to decode.", f"{parent_id}->{child_id}: {exc!r}"))

    def _validate_stored_roots(self, con, issues: list[ValidationIssue]) -> None:
        rows = con.execute(
            """
            SELECT stored_roots.def_id, stored_roots.storage_hash, stored_roots.relative_def_path,
                   stored_roots.def_size, stored_roots.def_mtime_ns, definitions.cdef_blob
            FROM stored_roots
            JOIN definitions
              ON definitions.def_id = stored_roots.def_id
            """
        )
        for did, storage_hash, relative_def_path, def_size, def_mtime_ns, cdef_blob in rows:
            try:
                cdef = _CODEC.decode_cdef(cdef_blob)
            except Exception as exc:
                issues.append(ValidationIssue("error", "Stored root CDef failed to decode.", f"{did}: {exc!r}"))
                continue
            try:
                stored_hash = stable_hash_from_blob(storage_hash)
            except Exception as exc:
                issues.append(ValidationIssue("error", "Stored root storage hash is invalid.", f"{did}: {exc!r}"))
                continue
            try:
                expected_hash, expected_path, expected_size, expected_mtime_ns = self._root_record_metadata(cdef)
                expected_key = stable_hash_from_blob(expected_hash)
            except Exception as exc:
                issues.append(ValidationIssue("error", "Stored DefinitionRecord is missing or invalid.", f"{did}: {exc!r}"))
                continue
            if stored_hash != expected_key:
                issues.append(ValidationIssue("error", "Stored root authority key mismatch.", f"{did}: stored={stored_hash}, expected={expected_key}"))
            if relative_def_path != expected_path:
                issues.append(ValidationIssue("error", "Stored DefinitionRecord relative path mismatch.", f"{did}: stored={relative_def_path}, expected={expected_path}"))
            if def_size is not None and expected_size is not None and def_size != expected_size:
                issues.append(ValidationIssue("error", "Stored DefinitionRecord size mismatch.", f"{did}: stored={def_size}, actual={expected_size}"))
            if def_mtime_ns is not None and expected_mtime_ns is not None and def_mtime_ns != expected_mtime_ns:
                issues.append(ValidationIssue("warning", "Stored DefinitionRecord mtime changed.", f"{did}: stored={def_mtime_ns}, actual={expected_mtime_ns}"))

    def _validate_store_roots(
            self,
            con,
            issues: list[ValidationIssue],
            *,
            roots: tuple[ConcreteDefinition, ...] | None = None) -> None:
        if self.store is None or not hasattr(self.store, "iter_definition_records"):
            return
        if roots is None:
            try:
                roots = self._preflight_store_roots()
            except Exception as exc:
                issues.append(ValidationIssue("error", "Store root scan failed.", repr(exc)))
                return
        actual_roots = set()
        for cdef in roots:
            if not isinstance(cdef, ConcreteDefinition):
                issues.append(ValidationIssue("error", "Store scan yielded non-CDef root.", type(cdef).__name__))
                continue
            actual_roots.add(cdef)
            root_ids = _exact_ids_for_cdef(con, cdef)
            if not root_ids:
                issues.append(ValidationIssue("error", "Store root is missing from SQLite query index.", cdef.stable_hash()))
                continue
            stored = con.execute(
                f"SELECT 1 FROM stored_roots WHERE def_id IN ({', '.join('?' for _ in root_ids)}) LIMIT 1",
                root_ids,
            ).fetchone()
            if stored is None:
                issues.append(ValidationIssue("error", "Store root is indexed but not active as a stored root.", cdef.stable_hash()))
        indexed_roots = {
            _CODEC.decode_cdef(row[0])
            for row in con.execute(
                "SELECT definitions.cdef_blob FROM stored_roots JOIN definitions ON definitions.def_id = stored_roots.def_id"
            )
        }
        if indexed_roots != actual_roots:
            issues.append(ValidationIssue("error", "SQLite stored roots differ from authoritative Store roots."))

    def _validate_reference_rows(self, con, issues: list[ValidationIssue]) -> None:
        """Validate reference blobs before a staged index becomes ready."""

        for row in con.execute("SELECT reference_kind, reference_digest, reference_blob, path_blob FROM reference_records"):
            kind, digest, blob, path_blob = row
            try:
                reference = _CODEC.decode_reference(blob)
                if kind == "object" and reference.digest() != digest:
                    raise ValueError("ObjectRef digest mismatch")
                if kind == "state" and reference.digest() != digest:
                    raise ValueError("StateRef digest mismatch")
                _CODEC.decode_graph_path(path_blob)
            except Exception as exc:
                issues.append(ValidationIssue("error", "Reference row failed to decode.", repr(exc)))

    @contextmanager
    def _build_claim(self, *, force: bool = False):
        """Acquire a process-held claim for a staged sidecar rebuild.

        Args:
            force: Continue to rebuild a ready sidecar when ``True``.

        Yields:
            ``True`` for the process that owns the claim, otherwise ``False``
            when another process already published a ready sidecar.

        Raises:
            QueryIndexBusy: If another live rebuild does not finish before the
                configured wait deadline.
        """

        claim_path = self._build_claim_path()
        start = time.monotonic()
        saw_existing_claim = False
        while True:
            if not force and self.path.exists() and not self._is_dirty():
                self._connections.close_all_current_process()
                if self.status().state == "ready":
                    yield False
                    return
            if saw_existing_claim and self.path.exists() and not self._is_dirty():
                self._connections.close_all_current_process()
                if self.status().state == "ready":
                    yield False
                    return
            try:
                claim_path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(str(claim_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            except FileExistsError:
                saw_existing_claim = True
                if self._claim_is_stale(claim_path):
                    continue
                self._connections.close_all_current_process()
                if self.path.exists() and not self._is_dirty() and self.status().state == "ready":
                    yield False
                    return
                if time.monotonic() - start > _BUILD_CLAIM_WAIT_SECONDS:
                    raise QueryIndexBusy("Timed out waiting for another SQLite query-index rebuild to finish.")
                time.sleep(0.01)
                continue
            locked = False
            owner_identity = None
            try:
                locked = _try_lock_claim_file(fd)
                if not locked:
                    raise QueryIndexBusy("Could not lock a new SQLite query-index rebuild claim.")
                owner_identity = os.fstat(fd)
                owner_token = uuid4().hex
                os.lseek(fd, 0, os.SEEK_SET)
                os.write(
                    fd,
                    f"pid={os.getpid()}\ntoken={owner_token}\ncreated={datetime.now(timezone.utc).isoformat()}\n".encode(),
                )
                if not force and self.path.exists() and not self._is_dirty():
                    self._connections.close_all_current_process()
                    if self.status().state == "ready":
                        yield False
                        return
                yield True
            finally:
                try:
                    if locked:
                        _unlock_claim_file(fd)
                finally:
                    os.close(fd)
                if locked:
                    self._unlink_claim_if_owned(claim_path, owner_identity)
            return

    def _build_claim_path(self) -> Path:
        return self.path.with_name(f"{self.path.name}.building")

    def _claim_is_stale(self, claim_path: Path) -> bool:
        """Remove an old claim only when no process currently holds its lock."""

        try:
            if time.time() - claim_path.stat().st_mtime <= _BUILD_CLAIM_STALE_SECONDS:
                return False
        except OSError:
            return False
        try:
            fd = os.open(str(claim_path), os.O_RDWR)
        except FileNotFoundError:
            return False
        except OSError:
            return False
        locked = False
        owner_identity = None
        stale = False
        try:
            locked = _try_lock_claim_file(fd)
            if not locked:
                return False
            owner_identity = os.fstat(fd)
            try:
                current = claim_path.stat()
            except FileNotFoundError:
                return False
            if not os.path.samestat(current, owner_identity):
                return False
            if time.time() - current.st_mtime <= _BUILD_CLAIM_STALE_SECONDS:
                return False
            stale = True
        except OSError:
            return False
        finally:
            if locked:
                _unlock_claim_file(fd)
            os.close(fd)
        if stale:
            self._unlink_claim_if_owned(claim_path, owner_identity)
        return stale

    @staticmethod
    def _unlink_claim_if_owned(claim_path: Path, owner_identity) -> None:
        """Remove a closed claim only when its path still identifies its owner."""

        try:
            current = claim_path.stat()
        except FileNotFoundError:
            return
        if os.path.samestat(current, owner_identity):
            try:
                claim_path.unlink()
            except FileNotFoundError:
                pass

    def _replacement_path(self) -> Path:
        return self.path.with_name(f"{self.path.name}.rebuild-{uuid4().hex}.tmp")

    def _activate_replacement(self, replacement_path: Path, *, quarantine_existing: bool) -> None:
        """Publish a complete staged sidecar without removing the canonical path early."""

        SQLiteConnectionManager._close_current_thread_for_path(self.path)
        if quarantine_existing and self.path.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            target = self.path.with_name(f"{self.path.name}.quarantine-{stamp}")
            try:
                os.link(self.path, target)
            except OSError:
                shutil.copy2(self.path, target)
        # A quarantine copy can admit a final canonical read after the first
        # close barrier; Windows requires that pooled handle closed as well.
        SQLiteConnectionManager._close_current_thread_for_path(self.path)
        os.replace(replacement_path, self.path)

    @staticmethod
    def _checkpoint_and_cleanup_sidecars(
            path: Path,
            *,
            label: str,
            allow_corrupt: bool = False) -> None:
        """Checkpoint and remove sidecars before a SQLite main-file replacement.

        Args:
            path: Main database path whose sidecars must be made inactive.
            label: Diagnostic name identifying staged or canonical sidecars.
            allow_corrupt: Permit a recognized corrupt main file to skip its
                impossible checkpoint before sidecar cleanup and quarantine.

        Raises:
            QueryIndexBusy: If a live reader prevents a complete WAL checkpoint
                or sidecar cleanup.
        """

        if not path.exists():
            return
        sqlite3 = require_sqlite()
        con = sqlite3.connect(str(path), timeout=0, isolation_level=None)
        try:
            row = con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        except Exception as exc:
            if is_sqlite_busy_error(exc):
                raise QueryIndexBusy(f"SQLite {label} sidecar checkpoint is blocked by an active reader.") from exc
            if not (allow_corrupt and _is_sqlite_corrupt_exception(exc)):
                raise QueryIndexError(f"SQLite {label} sidecar checkpoint failed.") from exc
            row = (0,)
        finally:
            con.close()
        if row is None or row[0] != 0:
            raise QueryIndexBusy(f"SQLite {label} sidecar checkpoint is blocked by an active reader.")
        for sidecar in (
                path.with_name(f"{path.name}-wal"),
                path.with_name(f"{path.name}-shm")):
            try:
                sidecar.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise QueryIndexBusy(f"SQLite {label} sidecar cleanup is blocked by an active reader.") from exc

    @staticmethod
    def _cleanup_replacement(replacement_path: Path) -> None:
        for path in (
                replacement_path,
                replacement_path.with_name(f"{replacement_path.name}-journal"),
                replacement_path.with_name(f"{replacement_path.name}-wal"),
                replacement_path.with_name(f"{replacement_path.name}-shm")):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def _set_build_state(self, state: str) -> None:
        if state not in {"building", "ready", "dirty"}:
            raise ValueError("state must be 'building', 'ready', or 'dirty'.")

        def operation(con):
            validate_schema(con, store_key=self.source_key, canonical_version=self.canonical_version, require_ready=False)
            con.execute(
                "UPDATE catalog_state SET build_state = ?, dirty = ?, updated_at = ? WHERE singleton = 1",
                (state, 1 if state == "dirty" else 0, datetime.now(timezone.utc).isoformat()),
            )

        self._run_write_transaction(operation)

    def _run_write_transaction(self, operation):
        max_retries = max(0, int(self.config.max_write_retries))
        delay = 0.005
        for attempt in range(max_retries + 1):
            began = False
            try:
                con = self._connections.connection(readonly=False)
                con.execute("BEGIN IMMEDIATE")
                began = True
                result = operation(con)
                con.execute("COMMIT")
                return result
            except Exception as exc:
                if began:
                    try:
                        con.execute("ROLLBACK")
                    except Exception:
                        self._connections.close_current()
                if is_sqlite_busy_error(exc):
                    if attempt < max_retries:
                        time.sleep(delay)
                        delay = min(delay * 2, 0.25)
                        continue
                    raise QueryIndexBusy(f"SQLite query index is busy after {max_retries + 1} write attempts.") from exc
                raise
        raise QueryIndexBusy(f"SQLite query index is busy after {max_retries + 1} write attempts.")


class SQLiteQueryIndexReadView:
    """Generation-stable SQLite read transaction for query planning.

    Store-local integer IDs returned by this view are valid only while the view
    is active and only for its captured generation. Callers must resolve IDs to
    CDefs before leaving the read view.
    """

    optimizer_policy = SQLiteOptimizerPolicy()

    def __init__(self, con, *, source_key: str, generation: int):
        self._con = con
        self.source_key = source_key
        self.generation = generation
        self._active = True
        self._temp_relations: list[str] = []
        self._temp_relation_counter = 0
        self._relation_counter = 0
        self._relation_plans: dict[str, LoweredQueryPlan] = {}

    def close(self) -> None:
        self.drop_temp_relations()
        self._active = False

    def _check_active(self) -> None:
        if not self._active:
            raise QueryIndexError("SQLite query index read view is closed.")

    @property
    def supports_lowering(self) -> bool:
        return True

    def lower_selector_graph(
            self,
            selector_graph,
            domain,
            *,
            terminal: QueryTerminal,
            scan_policy: ScanPolicy,
            diagnostics: LoweringDiagnostics | None = None,
            within_relation: str | None = None) -> LoweredQueryPlan:
        self._check_active()
        if within_relation is not None and within_relation not in self._temp_relations:
            raise QueryIndexError("within_relation must be a temp relation owned by this read view.")
        compiler = SQLiteRelationCompiler(
            self._con,
            source_key=self.source_key,
            generation=self.generation,
            codec=_CODEC,
            cdef_loader=self.cdefs_by_id,
        )
        plan = compiler.lower_selector_graph(
            selector_graph,
            domain,
            terminal=terminal,
            scan_policy=scan_policy,
            diagnostics=diagnostics,
            within_relation=within_relation,
        )
        self._relation_plans[plan.relation_id] = plan
        return plan

    def iter_candidate_cdef_batches(
            self,
            plan: LoweredQueryPlan,
            *,
            after: PagedResultCursor | None = None,
            batch_size: int):
        self._check_active()
        compiler = SQLiteRelationCompiler(
            self._con,
            source_key=self.source_key,
            generation=self.generation,
            codec=_CODEC,
            cdef_loader=self.cdefs_by_id,
        )
        return compiler.iter_candidate_cdef_batches(plan, after=after, batch_size=batch_size)

    def iter_relation_cdef_batches(
            self,
            relation: CandidateRelation,
            *,
            after: PagedResultCursor | None = None,
            batch_size: int):
        self._check_active()
        if relation.source_key != self.source_key or relation.generation != self.generation:
            raise QueryIndexError("CandidateRelation is not compatible with this read view.")
        plan = self._relation_plans.get(relation.relation_id)
        if plan is None:
            raise QueryIndexError("CandidateRelation is not owned by this read view.")
        return self.iter_candidate_cdef_batches(plan, after=after, batch_size=batch_size)

    def relation_exact_stored(self, cdef: ConcreteDefinition) -> CandidateRelation:
        self._check_active()
        ids = self.filter_stored_ids(self.exact_ids(cdef))
        name = self.create_temp_relation(ids)
        relation_id = self._next_relation_id("exact_stored")
        plan = LoweredQueryPlan(
            source_key=self.source_key,
            generation=self.generation,
            domain="stored",
            terminal="count",
            candidate_sql=f"SELECT def_id FROM temp.{name}",
            strategy="exact-safe-count",
            relation_id=relation_id,
            relation_kind="temp",
            estimated_size=len(ids),
            exact_safe=True,
            debug_label="exact-stored",
        )
        self._relation_plans[relation_id] = plan
        return plan.relation()

    def relation_from_ids(
            self,
            ids,
            *,
            domain: str = "known",
            debug_label: str = "id-relation") -> CandidateRelation:
        self._check_active()
        unique_ids = tuple(dict.fromkeys(ids))
        name = self.create_temp_relation(unique_ids)
        relation_id = self._next_relation_id("ids")
        diagnostics = LoweringDiagnostics(
            strategy="sqlite-relation-ids",
            relation_strategy="temp-relation",
            materialized_relations=(name,),
            estimated_rows=len(unique_ids),
            relations_created=1,
            temp_rows_inserted=len(unique_ids),
            physical_plan=PhysicalRelationPlan(
                strategy="temp-relation",
                root_relation_kind="temp",
                materialized_relations=(name,),
                fallback_reason="explicit-id-relation",
            ),
        )
        plan = LoweredQueryPlan(
            source_key=self.source_key,
            generation=self.generation,
            domain=domain,
            terminal="page",
            candidate_sql=f"SELECT def_id FROM temp.{name}",
            strategy="sqlite-relation-ids",
            relation_id=relation_id,
            relation_kind="temp",
            estimated_size=len(unique_ids),
            debug_label=debug_label,
            diagnostics=diagnostics,
        )
        self._relation_plans[relation_id] = plan
        return plan.relation()

    def relation_filter_domain(self, relation: CandidateRelation, domain) -> CandidateRelation:
        plan = self._plan_for_relation(relation)
        compiler = SQLiteRelationCompiler(self._con, source_key=self.source_key, generation=self.generation, codec=_CODEC)
        candidate_sql = compiler._apply_domain_sql(plan.candidate_sql, domain.name)
        return self._register_derived_relation(
            plan,
            candidate_sql,
            params=plan.params,
            relation_kind="cte",
            exact_safe=plan.exact_safe,
            debug_label=f"{relation.debug_label}:domain:{domain.name}",
        )

    def relation_parents(self, relation: CandidateRelation, path, *, unordered: bool = False, edge_kind: EdgeKind = EdgeKind.MATERIALIZE) -> CandidateRelation:
        plan = self._plan_for_relation(relation)
        params = list(plan.params)
        path_sql = self._relation_path_predicate("e", path, unordered, params, edge_kind=edge_kind)
        candidate_sql = f"""
        SELECT DISTINCT e.parent_def_id AS def_id
        FROM definition_edges e
        JOIN ({plan.candidate_sql}) child_relation
          ON child_relation.def_id = e.child_def_id
        WHERE {path_sql}
        """
        return self._register_derived_relation(
            plan,
            candidate_sql,
            params=tuple(params),
            exact_safe=False,
            debug_label=f"{relation.debug_label}:parents",
        )

    def relation_children(self, relation: CandidateRelation, path, *, unordered: bool = False, edge_kind: EdgeKind = EdgeKind.MATERIALIZE) -> CandidateRelation:
        plan = self._plan_for_relation(relation)
        params = list(plan.params)
        path_sql = self._relation_path_predicate("e", path, unordered, params, edge_kind=edge_kind)
        candidate_sql = f"""
        SELECT DISTINCT e.child_def_id AS def_id
        FROM definition_edges e
        JOIN ({plan.candidate_sql}) parent_relation
          ON parent_relation.def_id = e.parent_def_id
        WHERE {path_sql}
        """
        return self._register_derived_relation(
            plan,
            candidate_sql,
            params=tuple(params),
            exact_safe=False,
            debug_label=f"{relation.debug_label}:children",
        )

    def relation_semijoin_child_exists(
            self,
            parent_relation: CandidateRelation,
            child_relation: CandidateRelation,
            path,
            *,
            unordered: bool = False,
            edge_kind: EdgeKind = EdgeKind.MATERIALIZE) -> CandidateRelation:
        parent_plan = self._plan_for_relation(parent_relation)
        child_plan = self._plan_for_relation(child_relation)
        params = [*parent_plan.params, *child_plan.params]
        path_sql = self._relation_path_predicate("e", path, unordered, params, edge_kind=edge_kind)
        candidate_sql = f"""
        SELECT DISTINCT parent_relation.def_id
        FROM ({parent_plan.candidate_sql}) parent_relation
        WHERE EXISTS (
            SELECT 1
            FROM definition_edges e
            JOIN ({child_plan.candidate_sql}) child_relation
              ON child_relation.def_id = e.child_def_id
            WHERE e.parent_def_id = parent_relation.def_id
              AND {path_sql}
        )
        """
        return self._register_derived_relation(
            parent_plan,
            candidate_sql,
            params=tuple(params),
            exact_safe=False,
            debug_label=f"{parent_relation.debug_label}:semijoin-child-exists",
        )

    def relation_intersect(self, left: CandidateRelation, right: CandidateRelation) -> CandidateRelation:
        left_plan = self._plan_for_relation(left)
        right_plan = self._plan_for_relation(right)
        candidate_sql = f"""
        SELECT def_id FROM ({left_plan.candidate_sql})
        INTERSECT
        SELECT def_id FROM ({right_plan.candidate_sql})
        """
        relation = self._register_derived_relation(
            left_plan,
            candidate_sql,
            params=(*left_plan.params, *right_plan.params),
            exact_safe=left.exact_safe and right.exact_safe,
            debug_label=f"{left.debug_label}:intersect:{right.debug_label}",
        )
        return relation

    def relation_union(self, left: CandidateRelation, right: CandidateRelation) -> CandidateRelation:
        left_plan = self._plan_for_relation(left)
        right_plan = self._plan_for_relation(right)
        candidate_sql = f"""
        SELECT def_id FROM ({left_plan.candidate_sql})
        UNION
        SELECT def_id FROM ({right_plan.candidate_sql})
        """
        relation = self._register_derived_relation(
            left_plan,
            candidate_sql,
            params=(*left_plan.params, *right_plan.params),
            exact_safe=left.exact_safe and right.exact_safe,
            debug_label=f"{left.debug_label}:union:{right.debug_label}",
        )
        return relation

    def relation_materialize(self, relation: CandidateRelation, *, reason: str | None = None) -> CandidateRelation:
        plan = self._plan_for_relation(relation)
        name = self._create_empty_temp_relation()
        self._con.execute(
            f"INSERT OR IGNORE INTO temp.{name} (def_id) SELECT DISTINCT def_id FROM ({plan.candidate_sql})",
            plan.params,
        )
        row_count = self._con.execute(f"SELECT COUNT(*) FROM temp.{name}").fetchone()[0]
        relation_id = self._next_relation_id("materialized")
        diagnostics = plan.diagnostics.copy()
        diagnostics.relation_strategy = "temp-relation"
        diagnostics.materialized_relations = (*diagnostics.materialized_relations, name)
        diagnostics.relations_created += 1
        diagnostics.temp_rows_inserted += row_count
        diagnostics.physical_plan = PhysicalRelationPlan(
            strategy="temp-relation",
            root_relation_kind="temp",
            inline_relations=diagnostics.inline_relations,
            materialized_relations=diagnostics.materialized_relations,
            fallback_reason=reason or diagnostics.anchor_fallback_reason,
        )
        materialized = replace(
            plan,
            candidate_sql=f"SELECT def_id FROM temp.{name}",
            params=(),
            relation_id=relation_id,
            relation_kind="temp",
            estimated_size=row_count,
            debug_label=f"{relation.debug_label}:materialized",
            diagnostics=diagnostics,
        )
        self._relation_plans[relation_id] = materialized
        return materialized.relation()

    def relation_project_owners(self, relation: CandidateRelation) -> CandidateRelation:
        plan = self._plan_for_relation(relation)
        candidate_sql = f"""
        WITH RECURSIVE
            targets(def_id) AS ({plan.candidate_sql}),
            ancestors(current_id) AS (
                SELECT definition_edges.parent_def_id
                FROM targets
                JOIN definition_edges
                  ON definition_edges.child_def_id = targets.def_id
                 AND definition_edges.edge_kind = 'materialize'
                UNION
                SELECT definition_edges.parent_def_id
                FROM ancestors
                JOIN definition_edges
                  ON definition_edges.child_def_id = ancestors.current_id
                 AND definition_edges.edge_kind = 'materialize'
            )
        SELECT DISTINCT stored_roots.def_id
        FROM ancestors
        JOIN stored_roots
          ON stored_roots.def_id = ancestors.current_id
        """
        derived = self._register_derived_relation(
            plan,
            candidate_sql,
            params=plan.params,
            exact_safe=False,
            debug_label=f"{relation.debug_label}:owner-projection",
        )
        return derived

    def relation_count_estimate(self, relation: CandidateRelation) -> int | None:
        return relation.estimated_rows

    def relation_exact_safe_count(self, relation: CandidateRelation) -> int | None:
        plan = self._plan_for_relation(relation)
        if not relation.exact_safe:
            return None
        return self._con.execute(f"SELECT COUNT(*) FROM ({plan.candidate_sql})", plan.params).fetchone()[0]

    def relation_diagnostics(self, relation: CandidateRelation) -> LoweringDiagnostics:
        return self._plan_for_relation(relation).diagnostics

    def relation_optimize(
            self,
            relation: CandidateRelation,
            *,
            terminal: QueryTerminal | None = None,
            use_count: int = 1,
            recursive: bool = False) -> CandidateRelation:
        plan = self._plan_for_relation(relation)
        policy = self.optimizer_policy
        reason = self._materialization_reason(
            plan,
            relation,
            terminal=terminal,
            use_count=use_count,
            recursive=recursive,
            policy=policy,
        )
        if reason is None:
            diagnostics = plan.diagnostics
            if diagnostics.physical_plan is not None:
                diagnostics.physical_plan = replace(
                    diagnostics.physical_plan,
                    strategy="inline-cte",
                    materialized_relations=diagnostics.materialized_relations,
                )
            return relation
        return self.relation_materialize(relation, reason=reason)

    def explain_lowered_plan(self, plan: LoweredQueryPlan) -> tuple[str, ...]:
        self._check_active()
        compiler = SQLiteRelationCompiler(self._con, source_key=self.source_key, generation=self.generation, codec=_CODEC)
        return compiler.explain_query_plan(plan)

    def _plan_for_relation(self, relation: CandidateRelation) -> LoweredQueryPlan:
        self._check_active()
        if relation.source_key != self.source_key or relation.generation != self.generation:
            raise QueryIndexError("CandidateRelation is not compatible with this read view.")
        plan = self._relation_plans.get(relation.relation_id)
        if plan is None:
            raise QueryIndexError("CandidateRelation is not owned by this read view.")
        return plan

    def _register_derived_relation(
            self,
            plan: LoweredQueryPlan,
            candidate_sql: str,
            *,
            params: tuple = (),
            relation_kind: str = "cte",
            exact_safe: bool = False,
            debug_label: str = "derived-relation") -> CandidateRelation:
        relation_id = self._next_relation_id("relation")
        diagnostics = plan.diagnostics.copy()
        diagnostics.inline_relations = tuple(dict.fromkeys((*diagnostics.inline_relations, relation_id)))
        if diagnostics.physical_plan is not None:
            diagnostics.physical_plan = replace(
                diagnostics.physical_plan,
                inline_relations=diagnostics.inline_relations,
            )
        derived = replace(
            plan,
            candidate_sql=candidate_sql,
            params=params,
            relation_id=relation_id,
            relation_kind=relation_kind,
            estimated_size=plan.estimated_size,
            exact_safe=exact_safe,
            debug_label=debug_label,
            diagnostics=diagnostics,
        )
        self._relation_plans[relation_id] = derived
        return derived.relation()

    def _materialization_reason(
            self,
            plan: LoweredQueryPlan,
            relation: CandidateRelation,
            *,
            terminal: QueryTerminal | None,
            use_count: int,
            recursive: bool,
            policy: SQLiteOptimizerPolicy) -> str | None:
        if relation.relation_kind == "temp":
            return None
        if terminal == "page" and not policy.materialize_page_relations:
            return None
        if recursive and policy.materialize_recursive_owner_inputs:
            return "recursive-input"
        if policy.materialize_if_reused and use_count > 1:
            return "reused-relation"
        estimate = relation.estimated_rows if relation.estimated_rows is not None else plan.estimated_size
        if estimate is not None and estimate > policy.materialize_if_estimate_gt:
            return "estimated-large"
        if len(plan.candidate_sql) > policy.materialize_if_sql_length_gt:
            return "sql-length-large"
        return None

    def _next_relation_id(self, prefix: str) -> str:
        self._relation_counter += 1
        return f"{prefix}_{self._relation_counter}"

    def _relation_path_predicate(self, alias: str, path, unordered: bool, params: list, *, edge_kind: EdgeKind) -> str:
        params.append(edge_kind.value)
        if unordered:
            return f"{alias}.edge_kind = ?"
        path_blob = _CODEC.encode_graph_path(path)
        params.extend((digest_blob(path_blob), path_blob))
        return f"{alias}.edge_kind = ? AND {alias}.path_hash = ? AND {alias}.path_blob = ?"

    def create_temp_relation(self, ids) -> str:
        self._check_active()
        name = self._create_empty_temp_relation()
        unique_ids = tuple(dict.fromkeys(ids))
        if unique_ids:
            self._con.executemany(f"INSERT OR IGNORE INTO {name} (def_id) VALUES (?)", ((did,) for did in unique_ids))
        return name

    def _create_empty_temp_relation(self) -> str:
        self._temp_relation_counter += 1
        name = f"dryml_rel_{self._temp_relation_counter}"
        self._con.execute(f"CREATE TEMP TABLE {name} (def_id INTEGER PRIMARY KEY) WITHOUT ROWID")
        self._temp_relations.append(name)
        return name

    def drop_temp_relations(self) -> None:
        while self._temp_relations:
            name = self._temp_relations.pop()
            try:
                self._con.execute(f"DROP TABLE IF EXISTS temp.{name}")
            except Exception:
                pass

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

        resolved = []
        for req in requirements:
            feature_id = self._feature_id(req.token)
            if feature_id is None:
                if stats is not None:
                    stats.selected_features = requirements
                    stats.posting_sizes = tuple(0 for _ in requirements)
                    stats.candidate_count = 0
                return set()
            row = self._con.execute(
                "SELECT document_frequency FROM feature_tokens WHERE feature_id = ?",
                (feature_id,),
            ).fetchone()
            resolved.append((req, feature_id, 0 if row is None else row[0]))
        resolved.sort(key=lambda item: item[2])
        candidates = _posting_ids(self._con, resolved[0][1], resolved[0][0].count, within=within)
        posting_sizes = [resolved[0][2]]
        for req, feature_id, document_frequency in resolved[1:]:
            if not candidates:
                break
            candidates = _posting_ids(self._con, feature_id, req.count, within=candidates)
            posting_sizes.append(min(document_frequency, len(candidates)))
        if within is not None:
            candidates &= set(within)
        if domain is not None:
            candidates = domain.filter(candidates)
        if stats is not None:
            stats.selected_features = tuple(req for req, _, _ in resolved)
            stats.posting_sizes = tuple(posting_sizes)
            stats.candidate_count = len(candidates)
        return candidates

    def is_stored_id(self, did: DefinitionId) -> bool:
        self._check_active()
        return self._con.execute("SELECT 1 FROM stored_roots WHERE def_id = ?", (did,)).fetchone() is not None

    def filter_stored_ids(self, ids) -> set[DefinitionId]:
        self._check_active()
        ids = tuple(dict.fromkeys(ids))
        out = set()
        for batch in chunked(ids, 500):
            placeholders = ", ".join("?" for _ in batch)
            out.update(row[0] for row in self._con.execute(
                f"SELECT def_id FROM stored_roots WHERE def_id IN ({placeholders})",
                batch,
            ))
        return out

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
        ids = tuple(dict.fromkeys(ids))
        out = set()
        for batch in chunked(ids, 250):
            values = ", ".join("(?)" for _ in batch)
            out.update(row[0] for row in self._con.execute(
                f"""
                WITH RECURSIVE
                    candidate_ids(def_id) AS (VALUES {values}),
                    ancestors(start_id, current_id) AS (
                        SELECT candidate_ids.def_id, definition_edges.parent_def_id
                        FROM candidate_ids
                        JOIN definition_edges
                          ON definition_edges.child_def_id = candidate_ids.def_id
                         AND definition_edges.edge_kind = 'materialize'
                        UNION
                        SELECT ancestors.start_id, definition_edges.parent_def_id
                        FROM ancestors
                        JOIN definition_edges
                          ON definition_edges.child_def_id = ancestors.current_id
                         AND definition_edges.edge_kind = 'materialize'
                    )
                SELECT DISTINCT ancestors.start_id
                FROM ancestors
                JOIN stored_roots
                  ON stored_roots.def_id = ancestors.current_id
                """,
                batch,
            ))
        return out

    def filter_domain(self, domain, ids) -> set[DefinitionId]:
        self._check_active()
        return domain.with_catalog(self).filter(ids)

    def has_stored_ancestor(self, did: DefinitionId) -> bool:
        self._check_active()
        return did in self.filter_nested_ids({did})

    def cdefs_by_id(self, ids) -> dict[DefinitionId, ConcreteDefinition]:
        self._check_active()
        out = {}
        for batch in chunked(tuple(dict.fromkeys(ids)), 500):
            placeholders = ", ".join("?" for _ in batch)
            for did, cdef_blob in self._con.execute(
                    f"SELECT def_id, cdef_blob FROM definitions WHERE def_id IN ({placeholders}) ORDER BY def_id",
                    batch):
                out[did] = _CODEC.decode_cdef(cdef_blob)
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
            edge_kind: EdgeKind = EdgeKind.MATERIALIZE,
            within: set[DefinitionId] | frozenset[DefinitionId] | None = None) -> set[DefinitionId]:
        self._check_active()
        if not child_ids or (within is not None and not within):
            return set()
        path_blob = _CODEC.encode_graph_path(path)
        path_hash = digest_blob(path_blob)
        out = set()
        for child_id in child_ids:
            rows = self._con.execute(
                """
                SELECT parent_def_id, path_blob FROM definition_edges
                WHERE child_def_id = ? AND edge_kind = ? AND path_hash = ?
                """,
                (child_id, edge_kind.value, path_hash),
            ) if not unordered else self._con.execute(
                "SELECT parent_def_id, path_blob FROM definition_edges WHERE child_def_id = ? AND edge_kind = ?",
                (child_id, edge_kind.value),
            )
            for parent_id, row_path_blob in rows:
                row_path = _CODEC.decode_graph_path(row_path_blob)
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
            edge_kind: EdgeKind = EdgeKind.MATERIALIZE,
            within: set[DefinitionId] | frozenset[DefinitionId] | None = None) -> set[DefinitionId]:
        self._check_active()
        if not parent_ids or (within is not None and not within):
            return set()
        path_blob = _CODEC.encode_graph_path(path)
        path_hash = digest_blob(path_blob)
        out = set()
        for parent_id in parent_ids:
            rows = self._con.execute(
                """
                SELECT child_def_id, path_blob FROM definition_edges
                WHERE parent_def_id = ? AND edge_kind = ? AND path_hash = ?
                """,
                (parent_id, edge_kind.value, path_hash),
            ) if not unordered else self._con.execute(
                "SELECT child_def_id, path_blob FROM definition_edges WHERE parent_def_id = ? AND edge_kind = ?",
                (parent_id, edge_kind.value),
            )
            for child_id, row_path_blob in rows:
                row_path = _CODEC.decode_graph_path(row_path_blob)
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

    def capture_occurrences(self, target_ids=None, *, max_occurrences: int | None = None):
        self._check_active()
        if target_ids is None:
            snapshot = self.all_occurrence_snapshot()
        else:
            snapshot = self.occurrence_snapshot_for_nested_ids(target_ids)
        return tuple(snapshot.iter_occurrences(max_occurrences=max_occurrences))

    def all_occurrence_snapshot(self) -> AllOccurrenceTraversalSnapshot:
        outgoing = defaultdict(list)
        for parent_id, path_blob, child_id, edge_kind in self._con.execute(
                "SELECT parent_def_id, path_blob, child_def_id, edge_kind FROM definition_edges"):
            path = _CODEC.decode_graph_path(path_blob)
            kind = EdgeKind(edge_kind)
            edge = DefinitionEdgeRecord((parent_id, path, child_id, kind), parent_id, path, child_id, kind)
            outgoing[parent_id].append(edge)
        return AllOccurrenceTraversalSnapshot(
            cdefs=self.cdefs_by_id(self.all_definition_ids()),
            stored_ids=self.all_stored_ids(),
            outgoing={parent_id: tuple(edges) for parent_id, edges in outgoing.items()},
        )

    def _feature_id(self, token) -> int | None:
        token_blob = _CODEC.encode_feature_token(token)
        token_hash = digest_blob(token_blob)
        for feature_id, row_blob in self._con.execute(
                "SELECT feature_id, token_blob FROM feature_tokens WHERE token_hash = ?",
                (token_hash,)):
            if feature_token_equal(_CODEC.decode_feature_token(row_blob), token):
                return feature_id
        for feature_id, row_blob in self._con.execute(
                "SELECT feature_id, token_blob FROM feature_tokens"):
            if feature_token_equal(_CODEC.decode_feature_token(row_blob), token):
                return feature_id
        return None

    def _parent_edges(self, child_id: DefinitionId) -> tuple[DefinitionEdgeRecord, ...]:
        rows = self._con.execute(
            "SELECT parent_def_id, path_blob, child_def_id, edge_kind FROM definition_edges WHERE child_def_id = ? AND edge_kind = 'materialize'",
            (child_id,),
        )
        return tuple(
            DefinitionEdgeRecord((parent_id, _CODEC.decode_graph_path(path_blob), row_child_id, EdgeKind(edge_kind)), parent_id, _CODEC.decode_graph_path(path_blob), row_child_id, EdgeKind(edge_kind))
            for parent_id, path_blob, row_child_id, edge_kind in rows
        )

    def _owner_ids_for_nested_ids(self, ids: set[DefinitionId]) -> set[DefinitionId]:
        owners = set()
        for batch in chunked(tuple(dict.fromkeys(ids)), 250):
            values = ", ".join("(?)" for _ in batch)
            owners.update(row[0] for row in self._con.execute(
                f"""
                WITH RECURSIVE
                    target_ids(def_id) AS (VALUES {values}),
                    ancestors(current_id) AS (
                        SELECT definition_edges.parent_def_id
                        FROM target_ids
                        JOIN definition_edges
                          ON definition_edges.child_def_id = target_ids.def_id
                         AND definition_edges.edge_kind = 'materialize'
                        UNION
                        SELECT definition_edges.parent_def_id
                        FROM ancestors
                        JOIN definition_edges
                          ON definition_edges.child_def_id = ancestors.current_id
                         AND definition_edges.edge_kind = 'materialize'
                    )
                SELECT DISTINCT stored_roots.def_id
                FROM ancestors
                JOIN stored_roots
                  ON stored_roots.def_id = ancestors.current_id
                """,
                batch,
            ))
        return owners

    def _descendant_ids(self, owner_id: DefinitionId) -> set[DefinitionId]:
        descendants = set()
        stack = [row[0] for row in self._con.execute(
            "SELECT child_def_id FROM definition_edges WHERE parent_def_id = ? AND edge_kind = 'materialize'",
            (owner_id,),
        )]
        while stack:
            cur = stack.pop()
            if cur in descendants:
                continue
            descendants.add(cur)
            stack.extend(row[0] for row in self._con.execute(
                "SELECT child_def_id FROM definition_edges WHERE parent_def_id = ? AND edge_kind = 'materialize'",
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
    def from_cdef(cls, cdef: ConcreteDefinition, *, stable_hash_blob: bytes | None = None):
        if stable_hash_blob is None:
            stable_hash_blob = stable_hash_to_blob(cdef.stable_hash())
        fingerprint = target_local_fingerprint(cdef)
        return cls(
            cdef=cdef,
            stable_hash=stable_hash_blob,
            class_key=stable_hash_to_blob(canonical_class_key(cdef.cls)),
            cdef_blob=_CODEC.encode_cdef(cdef),
            features=tuple(
                _EncodedFeature(_CODEC.encode_feature_token(token), count)
                for token, count in fingerprint.counts.items()
            ),
        )


def _existing_definition_id(con, cdef: ConcreteDefinition, *, stable_hash_blob: bytes) -> int | None:
    for did, cdef_blob in con.execute(
            "SELECT def_id, cdef_blob FROM definitions WHERE stable_hash = ? ORDER BY collision_ordinal",
            (stable_hash_blob,)):
        if cdef_equal(_CODEC.decode_cdef(cdef_blob), cdef):
            return did
    return None


class _EncodedEdge:
    def __init__(self, parent, child, path_blob, path_hash, edge_kind: EdgeKind):
        self.parent = parent
        self.child = child
        self.path_blob = path_blob
        self.path_hash = path_hash
        self.edge_kind = edge_kind

    @classmethod
    def from_edge(cls, edge):
        path_blob = _CODEC.encode_graph_path(edge.path)
        return cls(edge.parent, edge.child, path_blob, digest_blob(path_blob), edge.kind)


def _resolve_definition_id(con, encoded: _EncodedNode, *, generation: int) -> tuple[int, bool]:
    max_ordinal = -1
    for did, cdef_blob, ordinal in con.execute(
            "SELECT def_id, cdef_blob, collision_ordinal FROM definitions WHERE stable_hash = ? ORDER BY collision_ordinal",
            (encoded.stable_hash,)):
        max_ordinal = max(max_ordinal, ordinal)
        if cdef_equal(_CODEC.decode_cdef(cdef_blob), encoded.cdef):
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
    token = _CODEC.decode_feature_token(token_blob)
    token_hash = digest_blob(token_blob)
    max_ordinal = -1
    for feature_id, row_blob, ordinal in con.execute(
            "SELECT feature_id, token_blob, collision_ordinal FROM feature_tokens WHERE token_hash = ? ORDER BY collision_ordinal",
            (token_hash,)):
        max_ordinal = max(max_ordinal, ordinal)
        if feature_token_equal(_CODEC.decode_feature_token(row_blob), token):
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
        if cdef_equal(_CODEC.decode_cdef(cdef_blob), cdef):
            out.append(did)
    return tuple(out)


def _posting_ids(con, feature_id: int, min_count: int, *, within=None) -> set[int]:
    if within is None:
        return {
            row[0]
            for row in con.execute(
                "SELECT def_id FROM postings WHERE feature_id = ? AND multiplicity >= ? ORDER BY def_id",
                (feature_id, min_count),
            )
        }
    ids = tuple(dict.fromkeys(within))
    out = set()
    for batch in chunked(ids, 500):
        placeholders = ", ".join("?" for _ in batch)
        out.update(row[0] for row in con.execute(
            f"""
            SELECT def_id
            FROM postings
            WHERE feature_id = ?
              AND multiplicity >= ?
              AND def_id IN ({placeholders})
            ORDER BY def_id
            """,
            (feature_id, min_count, *batch),
        ))
    return out


def _read_generation(con) -> int:
    row = con.execute("SELECT generation FROM catalog_state WHERE singleton = 1").fetchone()
    if row is None:
        raise QueryIndexError("SQLite query index is missing catalog_state generation.")
    return row[0]


def _read_build_state(con) -> str:
    row = con.execute("SELECT build_state FROM catalog_state WHERE singleton = 1").fetchone()
    if row is None:
        raise QueryIndexError("SQLite query index is missing catalog_state build_state.")
    return row[0]


def _status_diagnostics(con, journal_mode: str) -> dict[str, object]:
    sqlite3 = require_sqlite()
    diagnostics: dict[str, object] = {
        "wal_runtime_known_safe": wal_runtime_is_known_safe(sqlite3.sqlite_version_info),
    }
    if journal_mode.lower() == "wal":
        try:
            diagnostics["wal_checkpoint_passive"] = tuple(con.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone())
        except Exception as exc:
            diagnostics["wal_checkpoint_error"] = repr(exc)
    return diagnostics


def _validate_sqlite_integrity(con, issues: list[ValidationIssue]) -> None:
    quick = con.execute("PRAGMA quick_check").fetchall()
    for row in quick:
        if row[0] != "ok":
            issues.append(ValidationIssue("error", "SQLite quick_check failed.", str(row[0])))
    fk_rows = con.execute("PRAGMA foreign_key_check").fetchall()
    for row in fk_rows:
        issues.append(ValidationIssue("error", "SQLite foreign_key_check failed.", repr(row)))


def _is_sqlite_corrupt_exception(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "file is not a database" in text or "database disk image is malformed" in text


def _bump_generation(con, generation: int) -> int:
    con.execute(
        "UPDATE catalog_state SET generation = ?, updated_at = ? WHERE singleton = 1",
        (generation, datetime.now(timezone.utc).isoformat()),
    )
    return generation


def _relative_def_path(stable_hash: str) -> str:
    return f"objects/{stable_hash[:2]}/{stable_hash}/def.pkl"


def _reference_authority_rows(store):
    """Yield complete derived reference rows from current Store authority.

    Each row retains its authority source and canonical owner/path/reference
    identity.  This extraction is intentionally shared by replacement rebuilds
    only; callers must still verify query answers against Store records.
    """

    from ...store.records import DefinitionRecord
    from ..reference import ReferenceOccurrence, _iter_embedded_references
    from ...reference_values import ObjectRef, StateRef
    from ...utils.graph.path import GraphPath

    def owner_key(value):
        if isinstance(value, StateRef):
            return "state", value.digest()
        if isinstance(value, ObjectRef):
            return "object", value.digest()
        return "definition", DefinitionRecord(value).digest

    def encode(source_kind, source_digest, occurrence, *, alias=""):
        value = occurrence.value
        kind = "state" if isinstance(value, StateRef) else "object"
        owner_kind, owner_digest = owner_key(occurrence.owner)
        states = value.states if isinstance(value, StateRef) else None
        return (
            source_kind, source_digest, owner_kind, owner_digest,
            _CODEC.encode_graph_path(occurrence.path), kind, value.digest(),
            encode_reference(value),
            None if states is None else repr(tuple(sorted(states.items(), key=lambda item: str(item[0])))).encode("ascii"),
            alias, value, states,
        )

    for record in store.iter_definition_records():
        for occurrence in _iter_embedded_references(record.definition, owner=record.definition):
            yield encode("definition", record.digest, occurrence)
    for record in store.iter_declaration_records():
        reference = record.object_ref
        yield encode("declaration", record.digest, ReferenceOccurrence(reference, GraphPath(), reference))
    for record in store.iter_state_ref_records():
        state = record.state_ref
        yield encode("state-ref", record.digest, ReferenceOccurrence(state, GraphPath(), state))
        yield encode("state-ref", record.digest, ReferenceOccurrence(state, GraphPath(), state.object))
    for record in store.iter_object_alias_records():
        reference = record.object_ref
        yield encode(
            "object-alias", f"{record.alias}:{reference.digest()}",
            ReferenceOccurrence(reference, GraphPath(), reference), alias=record.alias,
        )
    for record in store.iter_state_alias_records():
        state_record = store.read_state_ref_record(record.state_ref_digest)
        if state_record is None or state_record.state_ref.object != record.object_ref:
            raise QueryIndexError("State alias points to missing or incompatible StateRef authority.")
        state = state_record.state_ref
        yield encode(
            "state-alias", f"{record.alias}:{record.state_ref_digest}",
            ReferenceOccurrence(state, GraphPath(), state), alias=record.alias,
        )


def _state_reference_authority_rows(state):
    """Yield incremental advisory rows for one published StateRef."""
    from ...reference_values import StateRef
    from ...utils.graph.path import GraphPath

    if not isinstance(state, StateRef):
        raise TypeError("Incremental reference registration requires a StateRef.")
    source_digest = state.digest()
    for value in (state, state.object):
        states = value.states if isinstance(value, StateRef) else None
        kind = "state" if isinstance(value, StateRef) else "object"
        yield (
            "state-ref", source_digest, "state", source_digest,
            _CODEC.encode_graph_path(GraphPath()), kind, value.digest(),
            encode_reference(value),
            None if states is None else repr(
                tuple(sorted(states.items(), key=lambda item: str(item[0])))
            ).encode("ascii"),
            "", value, states,
        )


def _row_counts(con) -> dict[str, int]:
    return {
        table: con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in (
            "definitions", "feature_tokens", "postings", "definition_edges",
            "stored_roots", "reference_records", "reference_object_ids",
        )
    }


def _empty_row_counts() -> dict[str, int]:
    return {
        table: 0
        for table in (
            "definitions", "feature_tokens", "postings", "definition_edges",
            "stored_roots", "reference_records", "reference_object_ids",
        )
    }
