"""Provide derived Store query indexes and process-local SQLite read views.

Store records remain authoritative. Index contexts retain transactions and reader
leases through cleanup, reject inherited view use, and report readiness or SQLite
failures to callers; rebuild and mutation affect only derived sidecar state.
Empty dirty-token captures require no cleanup-only authority reads; indexed
content writes, readiness checks, and staged validation remain required.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shutil
import sys
import time
from uuid import uuid4

from ...cdef_graph import EdgeKind, as_query_index_graph
from ...definition import ConcreteDefinition
from ..codecs import (
    QueryIndexCodec,
    digest_blob,
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
    QueryIndexCorrupt,
    QueryIndexDirty,
    QueryIndexError,
    QueryIndexIncompatible,
    QueryIndexStatus,
    QueryIndexUnavailable,
    QueryStats,
    ReconcileReport,
    ValidationIssue,
    ValidationReport,
    V3ProjectionCoverage,
    CANONICAL_QUERY_SEMANTICS_VERSION,
)
from ..lowering import CandidateRelation, LoweredQueryPlan, LoweringDiagnostics, PagedResultCursor, PhysicalRelationPlan, QueryTerminal, ScanPolicy
from ..identity import identity_key
from ..relationships import iter_direct_relationships, relationship_closure
from ..utils import cdef_equal, chunked, feature_token_equal, stable_hash_from_blob, stable_hash_to_blob
from . import SQLiteQueryIndexConfig, require_sqlite
from .connection import SQLiteConnectionManager
from .schema import (SQLITE_QUERY_INDEX_SCHEMA_VERSION, STORE_COVERAGE_VERSION,
                     STORE_PROJECTION_FAMILIES, STORE_UNAVAILABLE_FAMILIES,
                     initialize_schema, stored_compatibility_decision, validate_schema)
from .utils import is_sqlite_busy_error, wal_runtime_is_known_safe
from .lowering import SQLiteOptimizerPolicy, SQLiteRelationCompiler
from .... import filesystem
from ....locking import LockError, try_lock_file, unlock_file


_CODEC = QueryIndexCodec()
_BUILD_CLAIM_STALE_SECONDS = 300.0
_BUILD_CLAIM_WAIT_SECONDS = 30.0


def _store_family_supported(family, store):
    from ..authority import lifecycle_projection_supported
    return family not in STORE_UNAVAILABLE_FAMILIES or lifecycle_projection_supported(store)


def _store_row_digest(row):
    from ...store.records import _json_bytes
    family, source, key, payload = row
    return int.from_bytes(hashlib.sha256(b"dryml-store-projection-row/v1\0" +
        _json_bytes([family, source, key]) + b"\0" + payload).digest(), "big")


def _store_intent_digest(record):
    return _store_intent_data_digest(record.intent)


def _store_intent_data_digest(data):
    from ...store.records import _json_bytes
    # SQL certifies the observed cut, not native durability. Lifecycle progress
    # must not invalidate that cut's receipt; Store acknowledgement gates it.
    data["status"] = "classified"
    data["persistence_confirmed"] = True
    return hashlib.sha256(b"dryml-store-projection-receipt/v1\0" + _json_bytes(data)).digest()


@dataclass(frozen=True, slots=True)
class _StoredIdentityCursor:
    """Private graph-complete cursor for one stored-CDef projection page."""

    source_key: str
    generation: int
    graph_hash: str
    root_blob: bytes


@dataclass(frozen=True, slots=True)
class _StoredIdentityBatch:
    """Private stored-CDef page with its continuation cursor."""

    cdefs: tuple[ConcreteDefinition, ...]
    next_cursor: _StoredIdentityCursor


def _try_lock_claim_file(fd: int) -> bool:
    """Try to hold the cross-process lock for a query-index claim file."""

    try:
        return try_lock_file(fd)
    except LockError as error:
        raise QueryIndexUnavailable("SQLite query-index claim locking is unavailable.") from error


def _unlock_claim_file(fd: int) -> None:
    """Release a query-index claim file lock held by this process."""

    try:
        unlock_file(fd)
    except LockError as error:
        raise QueryIndexUnavailable("SQLite query-index claim unlocking is unavailable.") from error


class SQLiteStoreQueryIndex:
    """Persistent query-index backend for one physical Store.

    The backend stores graph nodes, local feature postings, direct definition
    edges, and active stored-root membership in a SQLite sidecar. Store object
    files remain authoritative: this index can be rebuilt or reconciled from the
    owning Store and never owns object state. Read views lease cached handles;
    ordinary entry may repair derived state, while bounded callers use protected
    non-repairing entry under their retained Store fences.

    Args:
        source_key: Stable string identifying the owning Store's index domain.
        path: String or Path for the derived SQLite sidecar.
        config: Optional SQLite connection, durability, and contention settings.
        canonical_version: Expected integer canonical query semantics version.
        store: Optional owning Store supplying rebuild and authority records.
        dirty_path: Optional string or Path for durable publication dirty tokens.

    Construction records configuration without opening SQLite. Read/mutation
    operations report QueryIndexError subclasses for readiness, compatibility,
    contention, or backend failures; rebuild changes only derived files.
    Direct explicit refresh calls each reconcile one fresh authority cut; ready
    validation preserves the sidecar, and repair succeeds only after activation
    and captured-token acknowledgement. Report-oriented reconcile retains errors.

    With a format-4 owning Store, schema-10 coverage is maintained through
    ``maintain_store`` and read through ``projection_rows``. Legacy Query V3 views
    are refused. Ready refresh uses the current authority certificate; explicit
    rebuild inventories metadata without payload IO. Public Store mutations own
    projection completion and exact journal acknowledgement synchronously. Coverage
    is Store-local: absent owner namespaces need no process-local adapter; present
    owner authority requires its complete reader before readiness or maintenance.
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

    @property
    def _store_projection(self):
        return self.store is not None and hasattr(self.store, "store_uuid")

    def preflight_store(self):
        """Reject unavailable, foreign or future SQLite before authority effects.

        Uses only backend/owner metadata. Missing and owned old/corrupt sidecars
        remain eligible for visible metadata-only recovery. Returns None; SQLite
        and QueryIndexError failures propagate without changing Store authority.
        """
        import math
        if not self._store_projection:
            raise QueryIndexUnavailable("Store projection preflight requires a format-4 owning Store.")
        from ....paths import local_path_key
        self.store._assert_state_owner_open()
        if (local_path_key(self.path) != local_path_key(self.config.path)
                or local_path_key(self._connections.path) != local_path_key(self.path)
                or replace(self._connections.config, path=self.config.path) != self.config
                or self.source_key != self.store.catalog_key()):
            raise QueryIndexIncompatible("Store projection destination/source configuration disagrees with its owner.")
        if (self.config.journal_mode not in {"auto", "wal", "delete"}
                or self.config.durability not in {"normal", "full"}
                or type(self.config.max_write_retries) is not int or self.config.max_write_retries < 0
                or type(self.config.busy_timeout) not in {int, float}
                or not math.isfinite(self.config.busy_timeout) or self.config.busy_timeout < 0):
            raise QueryIndexError("Invalid SQLite persistent index configuration.")
        require_sqlite()
        decision = self._assert_sidecar_rebuildable()
        if decision in {"compatible", "rebuild"}:
            with self._connections._read_lease() as (con, _):
                if con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='store_projection_cut'").fetchone():
                    certificate = con.execute("SELECT contract_version, store_uuid FROM store_projection_cut").fetchone()
                    if certificate is not None and (type(certificate[0]) is not int
                                                    or certificate[0] > STORE_COVERAGE_VERSION
                                                    or certificate[1] != self.store.store_uuid):
                        raise QueryIndexIncompatible("Future or foreign Store coverage certificate; refusing maintenance.")
        self.store._bind_query_index(self)

    def _store_certificate(self, con, cut, *, receipts=(), allow_pending=False, thorough=False):
        from ...store import authority as a
        if a.raw(self.store, ".dryml/query-index-destination.record") != self.store._query_index_binding(self):
            raise QueryIndexDirty("Store projection destination binding is missing or changed.")
        validate_schema(con, store_key=self.source_key, canonical_version=self.canonical_version)
        certificate = con.execute(
            "SELECT contract_version, store_uuid, epoch, index_uuid FROM store_projection_cut"
        ).fetchall()
        index_uuid = con.execute("SELECT index_uuid FROM catalog_state WHERE singleton=1").fetchone()[0]
        if certificate != [(STORE_COVERAGE_VERSION, cut.store_uuid, cut.epoch, index_uuid)]:
            raise QueryIndexDirty("SQLite coverage does not certify this Store authority cut.")
        coverage = con.execute(
            "SELECT family, supported, row_count, row_digest, dirty FROM store_projection_coverage"
        ).fetchall()
        if ({row[0] for row in coverage} != set(STORE_PROJECTION_FAMILIES)
                or any(row[1] != 1 or not _store_family_supported(row[0], self.store)
                       or row[2] < 0 or len(row[3]) != 32 or row[4] for row in coverage)):
            raise QueryIndexDirty("Closed Store coverage contract is incomplete or changed.")
        guards = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='trigger' "
                    "AND name IN ('store_projection_insert','store_projection_update','store_projection_delete')")}
        if len(guards) != 3:
            raise QueryIndexDirty("Store projection publication guards are missing.")
        seeks = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='index' "
                    "AND name IN ('store_discovery_fact','store_discovery_token','store_discovery_parent','store_discovery_child')")}
        if len(seeks) != 4:
            raise QueryIndexDirty("Store discovery seek indexes are missing; rebuild derived state.")
        if cut.pending:
            if not allow_pending:
                raise QueryIndexDirty("Store publication has unacknowledged projection work.")
            by_id = {r.intent["mutation_id"]: r for r in receipts}
            for identity in cut.pending:
                record = by_id.get(identity)
                row = con.execute("SELECT epoch, evidence FROM store_projection_receipts WHERE mutation_id=?",
                                  (identity,)).fetchone()
                if record is None or row != (record.intent["epoch"], _store_intent_digest(record)):
                    raise QueryIndexDirty("SQLite lacks the exact captured mutation receipt.")
        if thorough:
            actual = {family: (0, 0) for family in STORE_PROJECTION_FAMILIES}
            for row in con.execute("SELECT family, source, row_key, payload FROM store_projection_rows"):
                count, digest = actual[row[0]]
                actual[row[0]] = count + 1, digest ^ _store_row_digest(row)
            if any(actual[f] != (count, int.from_bytes(digest, "big"))
                   for f, _, count, digest, _ in coverage):
                raise QueryIndexDirty("Store projection row witnesses disagree.")

    def _store_status(self):
        """Inspect current authority and coverage without repair or graph inventory."""
        state, diagnostics, generation = "missing", {}, None
        if self.path.exists():
            try:
                with self.store.authority_read_fence(), self._connections._read_lease() as (con, identity):
                    con.execute("BEGIN")
                    try:
                        cut = self.store.authority_cut()
                        validate_schema(con, store_key=self.source_key,
                                        canonical_version=self.canonical_version, require_ready=False)
                        generation = _read_generation(con)
                        self._store_certificate(con, cut)
                        if identity != self._connections._path_identity() or self._is_dirty():
                            raise QueryIndexDirty("SQLite sidecar changed or has dirty tokens.")
                        diagnostics = {"store_uuid": cut.store_uuid, "epoch": cut.epoch,
                                       "coverage_contract": STORE_COVERAGE_VERSION,
                                       "index_uuid": con.execute(
                                           "SELECT index_uuid FROM catalog_state WHERE singleton=1").fetchone()[0]}
                        state = "ready"
                    finally:
                        con.execute("ROLLBACK")
            except QueryIndexBusy:
                raise
            except QueryIndexDirty as error:
                state, diagnostics = "dirty", {"error": str(error)}
            except QueryIndexIncompatible as error:
                state, diagnostics = "incompatible", {"error": str(error)}
            except Exception as error:
                state = "corrupt" if _is_sqlite_corrupt_exception(error) else "incompatible"
                diagnostics = {"error": str(error)}
        return QueryIndexStatus("sqlite", self.source_key, generation, SQLITE_QUERY_INDEX_SCHEMA_VERSION,
                                {"coverage_contract": STORE_COVERAGE_VERSION}, state,
                                path=str(self.path), diagnostics=diagnostics)

    def projection_rows(self, family, *, refresh=False):
        """Read one certified format-4 relation as detached correlated byte rows.

        ``family`` must be in coverage contract 2. ``refresh=False`` performs no
        maintenance; True permits replay/rebuild before the short read cut. Returns
        sorted (source path, correlation key, canonical JSON bytes) tuples. Missing,
        stale, pending or incomplete coverage raises QueryIndexDirty. Lifecycle
        reads require a complete registered owner reader even for an absent owner
        namespace; otherwise they raise QueryIndexUnavailable. General certificate
        checks require adapters only for owner namespaces actually present in the
        Store. This is an index-level seam, not a Repo terminal.
        """
        if type(family) is not str or type(refresh) is not bool:
            raise TypeError("Projection reads require a string family and boolean refresh.")
        if family not in STORE_PROJECTION_FAMILIES:
            raise ValueError("Unknown Store projection family.")
        if family in STORE_UNAVAILABLE_FAMILIES:
            from ..authority import lifecycle_projection_supported
            if not lifecycle_projection_supported():
                raise QueryIndexUnavailable("Store projection owner is not integrated: " + family)
        if not self._store_projection:
            raise QueryIndexUnavailable("Format-4 projection requires its owning Store.")
        if refresh:
            self.maintain_store()
        with self.store.authority_read_fence():
            if not self.path.exists() or self._is_dirty():
                raise QueryIndexDirty("Store sidecar is missing or dirty; explicit maintenance required.")
            with self._connections._read_lease() as (con, identity):
                con.execute("BEGIN")
                try:
                    self._store_certificate(con, self.store.authority_cut())
                    rows = tuple(con.execute("SELECT source, row_key, payload FROM store_projection_rows "
                                             "WHERE family=? ORDER BY source, row_key", (family,)))
                    if identity != self._connections._path_identity():
                        raise QueryIndexDirty("Store sidecar changed during projection read.")
                    return rows
                finally:
                    con.execute("ROLLBACK")

    def maintain_store(self, *, rebuild=False, stats=None, quarantine_existing=False, scan_admission=None):
        """Complete Store projection replay, SQL coverage commit and exact ack.

        Replays classified journal records without application capture or payload
        IO. A missing/stale/owned incompatible index uses visible staged metadata-
        only rebuild; ``rebuild=True`` requests unconditional replacement. Returns
        None only after captured work is acknowledged. Later writers cause bounded
        retry or StoreCutChanged and are never acknowledged by an earlier cut.
        A declined rebuild claim is revalidated against its captured authority
        cut; a claimant's subsequent publication is retryable cut drift, not a
        claim-contention failure for the completed replacement.
        SQL ambiguity is resolved by rereading exact committed receipts before ack.
        Store-owned native persistence recovery precedes projection reuse/ack;
        visible classification or a SQL commit alone cannot confirm durability.
        Unresolved native barriers raise StoreMutationError with pending evidence.
        Foreign/future destinations and incomplete authority are preserved/refused.
        ``quarantine_existing=True`` preserves the owned old sidecar on replacement;
        ``stats`` optionally receives started inventory work and refresh action.
        ``scan_admission`` is an optional zero-argument query policy callback,
        invoked outside Store writer fences before any required rebuild inventory.
        Its refusal propagates before inventory; explicit maintenance without a
        callback retains its ordinary rebuild permission.
        """
        from ..authority import capture_store_projection
        from ...store import authority as a
        if type(rebuild) is not bool or type(quarantine_existing) is not bool:
            raise TypeError("Store projection rebuild/quarantine controls must be boolean.")
        if not self._store_projection:
            raise QueryIndexUnavailable("Store projection maintenance requires a format-4 owning Store.")
        self.preflight_store()
        a.recover(self.store)
        if not rebuild and self.status().state == "ready":
            return
        # Only maintenance ownership may inventory. The clean replay path checks
        # epoch continuity using bounded pending records, not historical inventory.
        cut, records = self.store.authority_cut(), self.store.pending_mutations()
        replace_index = rebuild or not self.path.exists() or bool(self._dirty_markers())
        base = None
        if not replace_index:
            try:
                with self._connections._read_lease() as (con, _):
                    validate_schema(con, store_key=self.source_key, canonical_version=self.canonical_version)
                    base = con.execute("SELECT epoch, store_uuid FROM store_projection_cut").fetchone()
                    coverage = con.execute("SELECT family, dirty FROM store_projection_coverage").fetchall()
                    replace_index = (base is None or base[1] != cut.store_uuid
                                     or {r[0] for r in coverage} != set(STORE_PROJECTION_FAMILIES)
                                     or any(r[1] for r in coverage))
                    if not replace_index:
                        self._store_certificate(con, replace(cut, epoch=base[0], pending=()))
                epoch = -1 if base is None else base[0]
                for record in records:
                    data = record.intent
                    if data["base_epoch"] > epoch:
                        replace_index = True
                    epoch = max(epoch, data["epoch"])
                replace_index |= epoch != cut.epoch
            except QueryIndexBusy:
                raise
            except Exception as error:
                if is_sqlite_busy_error(error):
                    raise QueryIndexBusy("SQLite projection inspection is busy.") from error
                replace_index = True
        if replace_index:
            if scan_admission is not None:
                scan_admission()
            with self._build_claim(force=True) as acquired:
                if not acquired:
                    if self.status().state == "ready":
                        return
                    if self.store.authority_cut() != cut:
                        raise a.StoreCutChanged("Store changed after the competing projection rebuild; retry preparation.")
                    raise QueryIndexBusy("Store projection replacement belongs to another maintainer; retry.")
                self.preflight_store()
                if stats is not None:
                    stats.store_scan_count += 1
                    stats.refresh_action = "sqlite-rebuild"
                print("DRYML Store query index: metadata-only rebuild of missing/stale/older derived coverage.",
                      file=sys.stderr, flush=True)
                try:
                    prepared = capture_store_projection(self.store, rebuild=True)
                except a.StoreCutChanged:
                    raise
                except Exception:
                    self._mark_dirty()
                    raise
                markers = self._dirty_markers()
                replacement_path = self._replacement_path()
                replacement = SQLiteStoreQueryIndex(source_key=self.source_key, path=replacement_path,
                    config=replace(self.config, path=replacement_path), canonical_version=self.canonical_version)
                try:
                    replacement.initialize_empty(build_state="building", generation=self._replacement_generation_seed())
                    replacement._run_write_transaction(lambda con: self._write_store_projection(con, prepared, initialize=True))
                    con = replacement._connections.connection(readonly=True)
                    self._store_certificate(con, prepared.cut, receipts=prepared.receipts,
                                            allow_pending=True, thorough=True)
                    if con.execute("PRAGMA quick_check").fetchone() != ("ok",):
                        raise QueryIndexCorrupt("Staged Store projection failed SQLite validation.")
                    replacement.close()
                    self._checkpoint_and_cleanup_sidecars(replacement_path, label="staged")
                    with self.store.writer_lock():
                        prepared.witness.validate()
                        self.preflight_store()
                        SQLiteConnectionManager._close_idle_current_process_for_path(self.path)
                        self._checkpoint_and_cleanup_sidecars(self.path, label="canonical", allow_corrupt=True)
                        self._activate_replacement(replacement_path, quarantine_existing=quarantine_existing)
                        self._confirm_store_projection(prepared)
                        for marker in markers:
                            filesystem.remove_file(marker, cleanup_directory=marker.parent, missing_ok=True)
                        self._ack_store_projection(prepared)
                finally:
                    replacement.close()
                    self._cleanup_replacement(replacement_path)
            return
        prepared = capture_store_projection(self.store)
        with self.store.writer_lock():
            prepared.witness.validate()
            try:
                self._confirm_store_projection(prepared)
            except (QueryIndexDirty, QueryIndexIncompatible):
                try:
                    self._run_write_transaction(lambda con: self._write_store_projection(con, prepared))
                except Exception:
                    # COMMIT may have succeeded before its result was lost.
                    self._confirm_store_projection(prepared)
                self._confirm_store_projection(prepared)
            self._ack_store_projection(prepared)

    def acknowledge_store_mutation(self, identity, epoch):
        """Acknowledge only an exact durable SQLite receipt under the Store fence.

        ``identity``/``epoch`` name a classified Store mutation. Returns the current
        authority cut; missing/mismatched receipts raise QueryIndexDirty before
        consuming any pending work. Later pending mutations remain untouched.
        Store-owned persistence confirmation is required independently of the SQL
        receipt; unresolved native barriers raise StoreMutationError with pending
        evidence. No graph preparation, SQL write or application work runs here.
        """
        from ...store import authority as a
        from ...store.records import MutationIntentRecord
        with self.store.writer_lock():
            record = self.store._read_file(a._path(self.store, identity), MutationIntentRecord)
            if record is None or record.intent["epoch"] != epoch:
                raise QueryIndexDirty("Acknowledgement requires an exact classified mutation receipt.")
            with self._connections._read_lease() as (con, file_identity):
                con.execute("BEGIN")
                try:
                    validate_schema(con, store_key=self.source_key, canonical_version=self.canonical_version)
                    certificate = con.execute("SELECT store_uuid, epoch, index_uuid FROM store_projection_cut").fetchone()
                    index_uuid = con.execute("SELECT index_uuid FROM catalog_state WHERE singleton=1").fetchone()[0]
                    receipt = con.execute("SELECT epoch, evidence FROM store_projection_receipts WHERE mutation_id=?",
                                          (identity,)).fetchone()
                    if (certificate is None or certificate[0] != self.store.store_uuid
                            or certificate[1] < epoch or certificate[2] != index_uuid
                            or receipt != (epoch, _store_intent_digest(record))
                            or file_identity != self._connections._path_identity()):
                        raise QueryIndexDirty("SQLite has not incorporated this exact mutation/cut.")
                    self._store_certificate(con, replace(a.read_cut(self.store), epoch=certificate[1], pending=()))
                finally:
                    con.execute("ROLLBACK")
            return a.acknowledge(self.store, identity, epoch)

    def _confirm_store_projection(self, prepared):
        with self._connections._read_lease() as (con, identity):
            con.execute("BEGIN")
            try:
                self._store_certificate(con, prepared.cut, receipts=prepared.receipts, allow_pending=True)
                if identity != self._connections._path_identity():
                    raise QueryIndexDirty("SQLite projection destination changed before acknowledgement.")
            finally:
                con.execute("ROLLBACK")

    def _ack_store_projection(self, prepared):
        from ...store import authority as a
        for record in prepared.receipts:
            data = record.intent
            a.acknowledge(self.store, data["mutation_id"], data["epoch"])

    def _write_store_projection(self, con, prepared, *, initialize=False):
        if not initialize:
            validate_schema(con, store_key=self.source_key, canonical_version=self.canonical_version)
            previous = con.execute("SELECT epoch FROM store_projection_cut").fetchone()
            if previous is None:
                raise QueryIndexDirty("Missing Store coverage requires explicit rebuild.")
            self._store_certificate(con, replace(prepared.cut, epoch=previous[0], pending=()))
            epoch = previous[0]
            for record in prepared.receipts:
                data = record.intent
                if data["base_epoch"] > epoch:
                    raise QueryIndexDirty("Projection replay lost epoch continuity; retry metadata rebuild.")
                epoch = max(epoch, data["epoch"])
            if epoch != prepared.cut.epoch:
                raise QueryIndexDirty("Projection replay cut changed; retry metadata rebuild.")
        else:
            con.executemany("INSERT INTO store_projection_coverage VALUES (?, ?, 0, ?, 0)",
                ((family, int(_store_family_supported(family, self.store)), bytes(32))
                 for family in STORE_PROJECTION_FAMILIES))
        witnesses = {f: [n, int.from_bytes(d, "big")] for f, n, d in con.execute(
            "SELECT family, row_count, row_digest FROM store_projection_coverage")}
        desired = defaultdict(dict)
        for row in prepared.rows:
            desired[row[1]][row[0], row[2]] = row
        for source in prepared.sources:
            old = {(r[0], r[2]): r for r in con.execute(
                "SELECT family, source, row_key, payload FROM store_projection_rows WHERE source=?", (source,))}
            new = desired[source]
            for key, row in old.items():
                if new.get(key) == row:
                    continue
                con.execute("DELETE FROM store_projection_rows WHERE family=? AND source=? AND row_key=?", row[:3])
                witnesses[row[0]][0] -= 1
                witnesses[row[0]][1] ^= _store_row_digest(row)
            for key, row in new.items():
                if old.get(key) == row:
                    continue
                con.execute("INSERT INTO store_projection_rows VALUES (?, ?, ?, ?)", row)
                witnesses[row[0]][0] += 1
                witnesses[row[0]][1] ^= _store_row_digest(row)
        con.executemany("UPDATE store_projection_coverage SET row_count=?, row_digest=?, dirty=0 WHERE family=?",
                        ((n, d.to_bytes(32, "big"), f) for f, (n, d) in witnesses.items()))
        index_uuid = con.execute("SELECT index_uuid FROM catalog_state WHERE singleton=1").fetchone()[0]
        con.execute("INSERT OR REPLACE INTO store_projection_cut VALUES (1, ?, ?, ?, ?)",
                    (STORE_COVERAGE_VERSION, prepared.cut.store_uuid, prepared.cut.epoch, index_uuid))
        con.executemany("INSERT OR REPLACE INTO store_projection_receipts VALUES (?, ?, ?)",
                        ((data["mutation_id"], data["epoch"], _store_intent_data_digest(data))
                         for r in prepared.receipts for data in (r.intent,)))
        con.execute("UPDATE catalog_state SET generation=generation+1, build_state='ready', dirty=0, updated_at=?",
                    (datetime.now(timezone.utc).isoformat(),))

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
        """Return the ready generation under a short teardown-protected read lease.

        Raises readiness/backend errors before reading. May maintain derived
        state through the owning readiness policy; retains no connection lease.
        """
        self._ensure_ready()
        with self._connections._read_lease() as (con, _):
            return _read_generation(con)

    def status(self) -> QueryIndexStatus:
        """Inspect current sidecar readiness without repair or graph inventory.

        Returns:
            QueryIndexStatus with current readiness and validation diagnostics.
            Missing, dirty, incompatible, and corrupt evidence remains explicit.

        Raises:
            QueryIndexBusy: If an owned Store's status validation encounters typed
                reader contention. Retry after the active reader closes; contention
                does not establish incompatibility or authorize a rebuild.

        Side Effects:
            May open a short-lived leased SQLite read transaction, released before
            returning or raising. Does not replace the index or mutate authority.
        """
        if self._store_projection:
            return self._store_status()
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
            guard = con.execute(
                "SELECT dirty FROM v3_projection_guard WHERE family = 'stored_root'"
            ).fetchone()
            guard_dirty = guard is None or guard[0] != 0
            journal_mode = con.execute("PRAGMA journal_mode").fetchone()[0]
            row_counts = _row_counts(con)
            diagnostics = _status_diagnostics(con, journal_mode)
            diagnostics["stored_root_guard_dirty"] = guard_dirty
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
            state = "dirty" if row[11] or guard_dirty else row[10]
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
        """Inspect derived consistency, returning issues rather than repairing.

        ``thorough=True`` on an owned index retains a Store fence and leased
        transaction through graph-exact authority comparison and final identity
        checks. Returns a ValidationReport; failures become error issues. This
        operation changes neither authority nor the index generation/file.
        """
        if self._store_projection:
            try:
                with self.store.authority_read_fence(), self._connections._read_lease() as (con, _):
                    con.execute("BEGIN")
                    try:
                        self._store_certificate(con, self.store.authority_cut(), thorough=thorough)
                    finally:
                        con.execute("ROLLBACK")
                return ValidationReport("sqlite", self.source_key, True, ())
            except Exception as error:
                return ValidationReport("sqlite", self.source_key, False, (ValidationIssue("error", str(error)),))
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
            guard = con.execute(
                "SELECT dirty FROM v3_projection_guard WHERE family = 'stored_root'"
            ).fetchone()
            if guard is None or guard[0] != 0:
                issues.append(ValidationIssue(
                    "error", "SQLite stored-root projection changed outside managed publication."
                ))
            _validate_sqlite_integrity(con, issues)
            if thorough:
                self._validate_decodable_rows(con, issues)
                self._validate_stored_roots(con, issues)
                if not _v3_projection_is_complete(con):
                    issues.append(ValidationIssue("error", "Query V3 projection coverage is incomplete."))
            counts = _row_counts(con)
        except Exception as exc:
            issues.append(ValidationIssue("error", "SQLite query index validation failed.", repr(exc)))
            try:
                counts = _row_counts(con) if con is not None else _empty_row_counts()
            except Exception:
                counts = _empty_row_counts()
        return ValidationReport("sqlite", self.source_key, not any(issue.severity == "error" for issue in issues), tuple(issues), row_counts=counts, diagnostics=diagnostics)

    def v3_projection_coverage(self) -> V3ProjectionCoverage:
        """Return V3 projection integrity without promoting it to authority.

        A ready sidecar can prove that its own graph-exact rows have not been
        omitted only through the derived row-set witnesses.  It cannot yet prove
        complete V3 Store inventory coverage: aliases, metadata and the mutable
        main reference require authority-cut evidence, and ``write_main_ref``
        intentionally has no dirty token.  Indexed-only execution is therefore
        explicitly unavailable.
        """

        if self._store_projection:
            return V3ProjectionCoverage(backend="sqlite", store_key=self.source_key,
                generation=None, projection_complete=False, authority_complete=False, indexed_only=False,
                reason="Format-4 coverage is separate from the unintegrated legacy Query V3 consumer.")
        status = self.status()
        if status.state != "ready":
            return V3ProjectionCoverage(
                backend="sqlite", store_key=self.source_key,
                generation=status.generation, projection_complete=False,
                authority_complete=False, indexed_only=False,
                reason="derived sidecar is not ready",
            )
        try:
            con = self._connections.connection(readonly=True)
            complete = _v3_projection_is_complete(con)
            roots = con.execute(
                "SELECT COUNT(DISTINCT root_graph_hash) FROM v3_identity_projection"
            ).fetchone()[0]
            identities = con.execute("SELECT COUNT(*) FROM v3_identity_projection").fetchone()[0]
            relationships = con.execute("SELECT COUNT(*) FROM v3_relationship_projection").fetchone()[0]
        except Exception:
            complete = False
            roots = identities = relationships = 0
        return V3ProjectionCoverage(
            backend="sqlite", store_key=self.source_key,
            generation=status.generation, projection_complete=complete,
            authority_complete=False, indexed_only=False,
            source_roots=roots, identities=identities, relationships=relationships,
            reason=(
                "V3 authority coverage is unavailable; use Store authority fallback "
                "or reject indexed-only execution"
            ),
        )

    @contextmanager
    def read_view(self, *, include_cached: bool = True):
        """Yield a leased generation-stable view, repairing readiness if needed.

        ``include_cached`` is accepted for backend compatibility. Entry may
        initialize/rebuild derived state or wait on a build claim; callers must
        not retain Store authority fences during entry. SQLite/index failures
        propagate. The transaction and lease release on failure or cancellation.
        Views are process-local: inherited child use raises QueryIndexError,
        and child unwind leaves the parent's SQLite handle untouched. Open a new
        view in the child for an independent transaction.
        """

        if self._store_projection:
            raise QueryIndexUnavailable("Legacy query read views do not cover format-4 Nodes; use projection_rows.")
        self._ensure_ready()
        with self._leased_read_view() as view:
            yield view

    @contextmanager
    def _retained_read_view(self):
        """Enter a non-repairing stored-root view inside a retained authority cut.

        Reject dirty tokens, schema/build incompatibility, projection mutation,
        or a changed sidecar identity before yielding, including empty indexes.
        Never initialize, rebuild, or wait on a build claim. The transaction's
        actual UUID/generation and file identity, rather than preflight status,
        identify this view. All resources release on entry failure/cancellation.
        """

        if self._store_projection:
            raise QueryIndexUnavailable("Legacy query paging does not cover format-4 Nodes.")
        if self._is_dirty() or not self.path.exists():
            raise QueryIndexDirty("The query index is not ready inside the authority cut.")
        try:
            with self._leased_read_view(require_ready=True) as view:
                yield view
        except require_sqlite().Error as error:
            if is_sqlite_busy_error(error):
                raise QueryIndexBusy("SQLite retained transaction is busy; retry.") from error
            raise QueryIndexDirty("The query index could not retain a ready SQLite transaction.") from error

    @contextmanager
    def _leased_read_view(self, *, require_ready=False):
        pid = os.getpid()
        with self._connections._read_lease() as (con, identity):
            con.execute("BEGIN")
            view = None
            try:
                if require_ready:
                    validate_schema(con, store_key=self.source_key, canonical_version=self.canonical_version)
                    _require_clean_stored_root_guard(con)
                    if self._is_dirty() or identity != self._connections._path_identity():
                        raise QueryIndexDirty("The query index changed during retained view entry.")
                generation = _read_generation(con)
                view = SQLiteQueryIndexReadView(con, source_key=self.source_key, generation=generation)
                view.index_uuid = con.execute(
                    "SELECT index_uuid FROM catalog_state WHERE singleton = 1"
                ).fetchone()[0]
                view.file_identity = identity
                yield view
            finally:
                try:
                    if view is not None:
                        view.close()
                finally:
                    if os.getpid() == pid:
                        con.execute("ROLLBACK")

    def refresh(self, policy, *, stats=None, scan_admission=None) -> None:
        """Perform one backend lifecycle call, with no cross-call completion cache.

        Args:
            policy: False for no work, auto for readiness recovery, or True/always
                for an explicit validation cycle. Ready Store-4 cycles check compact
                authority/coverage certificates. Explicit reconciliation requires
                an owning format-4 Store; standalone indexes are populated manually.
            stats: Optional QueryStats retaining started inventories even on error.
            scan_admission: Optional zero-argument inventory admission callback for
                the current-format query consumer; refusal precedes rebuild IO.

        Returns:
            None after the requested work succeeds. Ready explicit cycles validate
            without replacing the sidecar; repair requires staged validation,
            activation and captured-token acknowledgement.

        Raises:
            QueryIndexError: On authority, compatibility, contention, validation,
                or publication failure. A foreign-owned destination is refused.
            QueryIndexUnavailable: If SQLite or Store authority is unavailable.

        Side Effects:
            May replace derived files and acknowledge captured dirty tokens.
            Tokens published after the maintenance cut survive.
        """
        if policy is False:
            return
        if self._store_projection:
            self.maintain_store(stats=stats, scan_admission=scan_admission)
            if stats is not None and (policy is True or policy == "always") and stats.refresh_action == "none":
                stats.refresh_action = "sqlite-validate"
            return
        if policy is True or policy == "always":
            raise QueryIndexUnavailable("Explicit reconciliation requires an owning format-4 Store.")
        self._assert_sidecar_rebuildable()
        status = self.status()
        if status.state in {"missing", "dirty", "building", "incompatible", "corrupt"}:
            self.rebuild(stats=stats, quarantine_existing=status.state in {"corrupt", "incompatible"}, force=False)
            return
        self._ensure_ready()

    def rebuild(self, *, stats: QueryStats | None = None, quarantine_existing: bool = False, force: bool = True) -> None:
        """Recreate this SQLite index from the owning format-4 Store's authority.

        Rebuilds acquire a sidecar build claim so concurrent callers do not all
        scan the Store. They preflight all authoritative roots, validate a
        complete sibling sidecar while it is `building`, then atomically replace
        the active sidecar. Store roots are never changed by rebuild or recovery.
        Standalone manually populated indexes have no authoritative source and
        cannot be rebuilt. Authority capture validates current Node memberships,
        state buckets and lifecycle records without payload IO.

        Args:
            stats: QueryStats or None, updated before authority capture,
                retaining started inventory expenditure even when rebuild fails.
            quarantine_existing: bool; preserve an incompatible or corrupt previous
                sidecar after successful replacement when possible.
            force: bool; acquire the build claim even when another ready index exists.

        Returns:
            None after owned replacement and captured-token acknowledgement, or
            after a declined build claim. A declined claim alone is not strict
            reconciliation success; use ``refresh(True)`` for that contract.

        Raises:
            QueryIndexError: If authoritative roots are invalid, staging fails,
                or the replacement sidecar does not validate.
            QueryIndexUnavailable: If SQLite is unavailable.
            QueryIndexIncompatible: If a valid persisted foreign owner or
                unsupported future metadata prevents replacement.
            QueryIndexBusy: If an active reader prevents checkpoint or replacement.
                On Windows, close retained readers before retrying replacement.

        Side Effects:
            May replace derived files and acknowledge captured covered tokens.
            Later tokens survive. Failure after activation does not imply rollback.
        """

        if self._store_projection:
            self.maintain_store(rebuild=force, stats=stats, quarantine_existing=quarantine_existing)
            return
        raise QueryIndexUnavailable("SQLite query-index rebuild requires an owning format-4 Store.")

    def reconcile(self) -> ReconcileReport:
        """Report one strict reconciliation attempt against Store authority.

        Ready graph-exact agreement preserves the sidecar; repair requires staged
        validation, activation, and captured-token acknowledgement. Use
        ``refresh(True)`` when ordinary failure must raise rather than return a
        report. Neither method retains cross-call completion state.

        Returns:
            ReconcileReport retaining historical issues and actual phase outcomes.
            Ordinary failures become error issues. ``validated`` records work
            performed, not success; ``changed`` records this attempt's visible
            activation even if acknowledgement later fails. Normal report return
            alone is not strict maintenance completion evidence.

        Raises:
            BaseException: Cancellation/interruption propagates.

        Side Effects:
            May replace derived files and acknowledge captured covered dirty
            tokens, preserving Store authority and later/unrelated tokens.
        """
        if self._store_projection:
            before = self.status()
            before_identity = self._connections._path_identity()
            stats, issues = QueryStats(), ()
            try:
                self.maintain_store(stats=stats)
            except Exception as error:
                issues = (ValidationIssue("error", str(error)),)
            after = self.status()
            return ReconcileReport("sqlite", self.source_key,
                before.generation != after.generation or before_identity != self._connections._path_identity(),
                "rebuild" if stats.store_scan_count else "replay", before.generation, after.generation,
                validated=not issues, issues=issues,
                diagnostics={"store_scan_count": stats.store_scan_count, "refresh_action": stats.refresh_action})
        return ReconcileReport(
            backend="sqlite",
            store_key=self.source_key,
            changed=False,
            action="none",
            generation_before=None,
            generation_after=None,
            validated=False,
            issues=(ValidationIssue("error", "Reconciliation requires an owning format-4 Store."),),
        )

    def _assert_sidecar_rebuildable(self) -> str:
        """Reject future sidecars before scanning Store authority or index rows."""

        if not self.path.exists():
            return "missing"
        try:
            with self._connections._read_lease() as (con, _):
                decision = stored_compatibility_decision(
                    con,
                    store_key=self.source_key,
                    canonical_version=self.canonical_version,
                )
        except Exception as exc:
            if is_sqlite_busy_error(exc):
                raise QueryIndexBusy("SQLite compatibility inspection is busy; retry.") from exc
            if _is_sqlite_corrupt_exception(exc):
                return
            raise QueryIndexError("Could not inspect SQLite query-index compatibility before rebuild.") from exc
        if decision == "future-unsupported":
            raise QueryIndexIncompatible(
                "SQLite query-index metadata is future, missing, or malformed; refusing to rebuild over it."
            )
        if decision == "foreign-owned":
            raise QueryIndexIncompatible("SQLite query-index destination has a foreign Store owner; refusing maintenance.")
        return decision

    def register_stored_roots(self, graph, roots):
        """Register query graph content and independently stored root definitions.

        Args:
            graph: ConcreteDefinitionGraph or supported query graph adapter.
            roots: Iterable of ConcreteDefinitions to activate as stored roots.

        Returns:
            IndexWriteResult with generation and changed-row counts. Empty graph
            and root input performs the existing readiness check only.

        Raises:
            QueryIndexError: On readiness, authority, contention, or write failure.

        Side Effects:
            Writes derived rows and acknowledges only captured covered tokens.
            Empty captures skip cleanup-only node eligibility and root-key reads.
            The publication caller retains the Store authority fence.
        """
        return self._register_stored_roots(graph, roots, require_ready=True)

    def register_graph(self, graph):
        """Register graph content without activating independent stored roots.

        ``graph`` is a ConcreteDefinitionGraph or supported query graph adapter.
        Returns IndexWriteResult; empty input still checks readiness. Writes only
        derived rows, acknowledging captured eligible closure-node tokens under
        the caller's Store publication fence. Empty captures need no cleanup-only
        authority reads. QueryIndexError reports readiness or mutation failure.
        """
        return self._register_stored_roots(graph, (), require_ready=True)

    def activate_stored_roots(self, graph, roots):
        """Activate root definitions with their graph content in the derived index.

        Accepts a ConcreteDefinitionGraph or supported adapter and an iterable of
        ConcreteDefinitions. Returns IndexWriteResult; empty input still checks
        readiness. Under the caller's Store publication fence, writes graph/root
        rows and acknowledges only captured covered tokens, with no cleanup-only
        authority reads for empty captures. QueryIndexError reports readiness,
        authority, contention, or mutation failure.
        """
        return self._register_stored_roots(graph, roots, require_ready=True)

    def _register_stored_roots(self, graph, roots, *, require_ready: bool):
        if self._store_projection:
            raise QueryIndexIncompatible("Format-4 projections must be published through Store mutation authority.")
        if require_ready and self._build_claim_path().exists():
            self._mark_dirty()
            raise QueryIndexBusy(
                "Cannot register Store roots while a SQLite query-index rebuild is active."
            )
        roots = _unique_graph_roots(roots)
        graph = as_query_index_graph(graph, roots if roots else graph.roots)
        graph_nodes = graph.nodes()
        node_hash_blobs = {node.definition: stable_hash_to_blob(node.stable_hash) for node in graph_nodes}
        encoded_edges = tuple(_EncodedEdge.from_edge(edge) for edge in graph.edges())
        encoded_roots = tuple((root.graph_hash(), _CODEC.encode_cdef(root)) for root in roots)
        v3_identities, v3_relationships = _v3_projection_rows(roots)
        if not roots and not graph_nodes and not encoded_edges:
            return IndexWriteResult(generation=self.current_generation(), changed=False)
        dirty_markers = self._dirty_markers()
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
            _require_clean_stored_root_guard(con)
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
                        def_id, root_graph_hash, storage_hash, relative_def_path,
                        def_size, def_mtime_ns, indexed_generation
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        root_id, root.graph_hash(), storage_hash,
                        relative_def_path, def_size, def_mtime_ns,
                        next_generation,
                    ),
                )
                counters.changed = counters.changed or bool(cur.rowcount)

            for root_hash, root_blob in encoded_roots:
                cur = con.execute(
                    "INSERT OR IGNORE INTO v3_stored_root_projection "
                    "(root_graph_hash, root_blob) VALUES (?, ?)",
                    (root_hash, root_blob),
                )
                if cur.rowcount:
                    counters.roots_added += 1
                    counters.changed = True

            added_identities = []
            added_relationships = []
            for row in v3_identities:
                cur = con.execute(
                    "INSERT OR IGNORE INTO v3_identity_projection "
                    "(root_graph_hash, identity_kind, identity_digest) VALUES (?, ?, ?)",
                    row,
                )
                counters.changed = counters.changed or bool(cur.rowcount)
                if cur.rowcount:
                    added_identities.append(row)
            for row in v3_relationships:
                cur = con.execute(
                    "INSERT OR IGNORE INTO v3_relationship_projection "
                    "(root_graph_hash, owner_kind, owner_digest, relationship_kind, "
                    "path_blob, target_kind, target_digest) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    row,
                )
                counters.changed = counters.changed or bool(cur.rowcount)
                if cur.rowcount:
                    added_relationships.append(row)
            _update_v3_projection_coverage(
                con, added_identities=added_identities,
                added_relationships=added_relationships,
            )
            con.execute(
                "UPDATE v3_projection_guard SET dirty = 0 "
                "WHERE family = 'stored_root'"
            )

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

        max_retries = max(0, int(self.config.max_write_retries))
        delay = 0.005
        for attempt in range(max_retries + 1):
            try:
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
                    did = _existing_definition_id(
                        con,
                        node.definition,
                        stable_hash_blob=node_hash_blobs[node.definition],
                    )
                    if did is None:
                        missing_nodes.append(node.definition)
                    else:
                        existing_node_ids[node.definition] = did
                return existing_node_ids, tuple(missing_nodes)
            except Exception as exc:
                if is_sqlite_busy_error(exc):
                    if attempt < max_retries:
                        time.sleep(delay)
                        delay = min(delay * 2, 0.25)
                        continue
                    raise QueryIndexBusy(
                        f"SQLite query index is busy after {max_retries + 1} preflight attempts."
                    ) from exc
                raise
        raise QueryIndexBusy(
            f"SQLite query index is busy after {max_retries + 1} preflight attempts."
        )

    def remove_stored_roots(self, roots):
        roots = _unique_graph_roots(roots)
        if not roots:
            return IndexWriteResult(generation=self.current_generation(), changed=False)
        def operation(con):
            self._ensure_schema_in_transaction(con)
            _require_clean_stored_root_guard(con)
            generation = _read_generation(con)
            removed = 0
            projection_changed = False
            removed_identities = []
            removed_relationships = []
            for root in roots:
                root_hash = root.graph_hash()
                root_blob = _CODEC.encode_cdef(root)
                cur = con.execute(
                    "DELETE FROM v3_stored_root_projection "
                    "WHERE root_graph_hash = ? AND root_blob = ?",
                    (root_hash, root_blob),
                )
                removed += cur.rowcount
                remaining_roots = tuple(
                    _CODEC.decode_cdef(row[0])
                    for row in con.execute(
                        "SELECT root_blob FROM v3_stored_root_projection"
                    )
                )
                if not any(cdef_equal(candidate, root) for candidate in remaining_roots):
                    for did in _exact_ids_for_cdef(con, root):
                        con.execute("DELETE FROM stored_roots WHERE def_id = ?", (did,))
                removed_identities.extend(con.execute(
                    "SELECT root_graph_hash, identity_kind, identity_digest "
                    "FROM v3_identity_projection WHERE root_graph_hash = ?",
                    (root_hash,),
                ).fetchall())
                removed_relationships.extend(con.execute(
                    "SELECT root_graph_hash, owner_kind, owner_digest, relationship_kind, "
                    "path_blob, target_kind, target_digest "
                    "FROM v3_relationship_projection WHERE root_graph_hash = ?",
                    (root_hash,),
                ).fetchall())
                cur = con.execute(
                    "DELETE FROM v3_identity_projection WHERE root_graph_hash = ?",
                    (root_hash,),
                )
                projection_removed = cur.rowcount
                cur = con.execute(
                    "DELETE FROM v3_relationship_projection WHERE root_graph_hash = ?",
                    (root_hash,),
                )
                projection_removed += cur.rowcount
                if projection_removed:
                    projection_changed = True
            if removed or projection_changed:
                _update_v3_projection_coverage(
                    con, removed_identities=removed_identities,
                    removed_relationships=removed_relationships,
                )
                con.execute(
                    "UPDATE v3_projection_guard SET dirty = 0 "
                    "WHERE family = 'stored_root'"
                )
                generation = _bump_generation(con, generation + 1)
            return IndexWriteResult(
                generation=generation,
                changed=bool(removed or projection_changed),
                roots_removed=removed,
            )

        return self._run_write_transaction(operation)

    def ensure_exact_stored(self, cdef, *, stats=None) -> bool:
        """Check manually registered CDef membership without consulting Store authority.

        Accepts a ConcreteDefinition and optional QueryStats; returns bool. May
        initialize a standalone derived index and raises QueryIndexError on
        readiness/read failures. Format-4 Stores use Node projection queries.
        """
        self._ensure_ready()
        with self.read_view(include_cached=False) as view:
            if view.filter_stored_ids(view.exact_ids(cdef)):
                if stats is not None:
                    stats.fast_path = "exact-root-index"
                return True

        return False

    def sync_caches(self, *, reuse_weak: bool = True) -> None:
        return None

    def close(self) -> None:
        """Close this process's unleased cached handles, preserving active views.

        Returns None. SQLite close errors propagate and retain the failing handle
        for recovery. Active leased readers finish their own transaction cleanup;
        direct unleased handle users must coordinate teardown with this call.
        """

        self._connections.close_all_current_process()

    def _ensure_ready(self) -> None:
        if self._store_projection:
            self.maintain_store()
            return
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
        validate_schema(con, store_key=self.source_key, canonical_version=self.canonical_version)

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
            clear_scoped: bool = False,
            metadata_keys=()) -> None:
        """Durably remove only captured dirty tokens covered by indexed work.

        Args:
            markers: Token paths captured before the completed index mutation.
            roots: Root definitions whose scoped tokens may be acknowledged.
            clear_unscoped: Whether unscoped tokens are covered.
            clear_scoped: Whether every root-scoped token is covered.
            metadata_keys: Exact metadata token keys covered by the mutation.

        Raises:
            QueryIndexError: If logical removal or its persistence barrier fails.

        Side Effects:
            Removes only matching captured tokens. Tokens published concurrently
            after the capture remain visible and keep the index dirty.
            An empty capture returns without computing cleanup-only root keys.
        """

        if not markers:
            return
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
                    or mutation in metadata_keys
            ):
                continue
            try:
                filesystem.remove_file(
                    marker,
                    cleanup_directory=marker.parent,
                    missing_ok=True,
                )
            except OSError as exc:
                raise QueryIndexError(
                    "SQLite query-index dirty-marker removal failed."
                ) from exc

    def _mark_dirty(self) -> None:
        """Durably publish one unscoped dirty token before index mutation.

        Raises:
            QueryIndexError: If parent creation or token publication fails.

        Side Effects:
            Creates a unique private token containing ``dirty``, then publishes
            it through the public filesystem persistence boundary.
        """

        if self.dirty_path is None:
            return
        try:
            filesystem.ensure_directory(self.dirty_path.parent)
        except OSError as exc:
            raise QueryIndexError(
                "SQLite query-index dirty-marker parent creation failed."
            ) from exc
        marker_path = self.dirty_path.with_name(f"{self.dirty_path.name}.{uuid4().hex}")
        temporary_path = marker_path.with_name(
            f".query-index-dirty-{uuid4().hex}.tmp"
        )
        try:
            with open(temporary_path, "x", encoding="utf-8") as file:
                file.write("dirty\n")
            filesystem.publish_file(
                temporary_path, marker_path, replace=False,
            )
        except BaseException as exc:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            if isinstance(exc, OSError):
                raise QueryIndexError(
                    "SQLite query-index dirty-marker publication failed."
                ) from exc
            raise

    def _replacement_generation_seed(self) -> int:
        current = -1
        if self.path.exists():
            try:
                with self._connections._read_lease() as (con, _):
                    current = _read_generation(con)
            except Exception:
                pass
        return max(current + 1, time.time_ns())

    def _root_marker_key(self, cdef: ConcreteDefinition) -> str:
        """Return the stable-hash token for a manually registered CDef root."""
        return cdef.stable_hash()

    def _root_record_metadata(self, cdef: ConcreteDefinition) -> tuple[bytes, str, int | None, int | None]:
        """Return derived root-row metadata without treating the row as authority."""
        stable_hash = cdef.stable_hash()
        return stable_hash_to_blob(stable_hash), _relative_def_path(stable_hash), None, None

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
        """Durably publish a staged sidecar without removing authority early.

        Args:
            replacement_path: Complete validated sibling SQLite database.
            quarantine_existing: Preserve a best-effort copy of old authority.

        Raises:
            QueryIndexError: If replacement or its persistence barrier fails.

        Side Effects:
            Closes canonical handles, may create a quarantine copy, and atomically
            replaces the canonical sidecar through the active platform policy.
        """

        if SQLiteConnectionManager._close_idle_current_process_for_path(self.path) and os.name == "nt":
            raise QueryIndexBusy("SQLite query-index replacement is blocked by an active reader; retry after it closes.")
        if quarantine_existing and self.path.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            target = self.path.with_name(f"{self.path.name}.quarantine-{stamp}")
            try:
                os.link(self.path, target)
            except OSError:
                shutil.copy2(self.path, target)
        # Windows replacement also requires idle handles on other threads closed.
        # A quarantine copy can admit a final read after the first close barrier.
        if SQLiteConnectionManager._close_idle_current_process_for_path(self.path) and os.name == "nt":
            raise QueryIndexBusy("SQLite query-index replacement is blocked by an active reader; retry after it closes.")
        try:
            filesystem.publish_file(
                replacement_path, self.path, replace=True,
            )
        except OSError as exc:
            raise QueryIndexError(
                "SQLite query-index replacement activation failed."
            ) from exc

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
            except BaseException as exc:
                if began:
                    try:
                        con.execute("ROLLBACK")
                    except Exception:
                        self._connections.close_current()
                if isinstance(exc, Exception) and is_sqlite_busy_error(exc):
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
    CDefs before leaving the read view. Index-owned contexts lease the underlying
    handle through cleanup, preventing build-claim teardown from closing an active
    same-process reader. ``index_uuid`` and ``file_identity`` identify the actual
    pinned sidecar when supplied by those contexts; ``generation`` is derived
    SQLite state, not a Store authority generation.
    Views belong to their creating process. Inherited child operations reject
    with QueryIndexError; closing an inherited view only invalidates the child's
    Python view without accessing its parent's SQLite handle.

    Args:
        con: Open SQLite connection with an owner-retained read transaction.
        source_key: String identifying this Store-local index domain.
        generation: Integer generation read from this transaction's catalog.

    Methods decode detached CDefs or Store-local IDs and may create temporary
    planning relations. Invalid/closed views raise QueryIndexError; SQLite IO
    failures propagate to the owning context, which releases the transaction.
    """

    optimizer_policy = SQLiteOptimizerPolicy()

    def __init__(self, con, *, source_key: str, generation: int):
        self._con = con
        self._pid = os.getpid()
        self.source_key = source_key
        self.generation = generation
        self._active = True
        self._temp_relations: list[str] = []
        self._temp_relation_counter = 0
        self._relation_counter = 0
        self._relation_plans: dict[str, LoweredQueryPlan] = {}

    def close(self) -> None:
        """Invalidate this view, dropping temporary relations only in its owner.

        Returns:
            None. Repeated closure is harmless.

        Side Effects:
            Attempts temporary-table cleanup in the creating process, retaining
            the existing best-effort SQLite-error handling. In a forked child,
            only the Python view is invalidated; the inherited handle is untouched.
            Transaction rollback remains the owning index context's responsibility.
        """

        if os.getpid() == self._pid:
            self.drop_temp_relations()
        self._active = False

    def _check_process(self) -> None:
        if os.getpid() != self._pid:
            raise QueryIndexError(
                "SQLite query index read view belongs to another process; "
                "open a new read view in this process."
            )

    def _check_active(self) -> None:
        self._check_process()
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

    def iter_stored_identity_cdef_batches(
            self,
            *,
            after: _StoredIdentityCursor | None = None,
            batch_size: int,
            stats: QueryStats | None = None):
        """Yield stored CDefs in Query V3 canonical digest order.

        Args:
            after: Optional cursor returned by the preceding page from this
                source generation.
            batch_size: Positive maximum rows decoded for this page.
            stats: Optional terminal counters updated for rows and pages read.

        Yields:
            At most one detached private batch. Reinvoke with its cursor to
            continue keyset paging.

        Raises:
            ValueError: If the page size or cursor is invalid.
            QueryIndexError: If this read view is closed, belongs to another
                process, or its rows fail to decode as complete CDefs.

        Side Effects:
            Reads only the current SQLite transaction and updates ``stats`` when
            supplied; it does not consult or mutate Store authority.
        """

        self._check_active()
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive exact int.")
        _require_clean_stored_root_guard(self._con)
        params = []
        cursor_sql = ""
        if after is not None:
            if (
                    after.source_key != self.source_key
                    or after.generation != self.generation):
                raise QueryIndexDirty(
                    "Stored-CDef paging was invalidated by a query-index generation change."
                )
            cursor_sql = """
                WHERE root_graph_hash > ?
                   OR (root_graph_hash = ? AND root_blob > ?)
            """
            params.extend((after.graph_hash, after.graph_hash, after.root_blob))
        params.append(batch_size)
        rows = self._con.execute(
            f"""
            SELECT root_graph_hash, root_blob
            FROM v3_stored_root_projection
            {cursor_sql}
            ORDER BY root_graph_hash, root_blob
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        if not rows:
            return
        try:
            cdefs = tuple(_CODEC.decode_cdef(row[1]) for row in rows)
        except Exception as error:
            raise QueryIndexError(
                "Stored identity page contains an invalid graph-complete CDef."
            ) from error
        last_digest, last_blob = rows[-1]
        cursor = _StoredIdentityCursor(
            source_key=self.source_key,
            generation=self.generation,
            graph_hash=last_digest,
            root_blob=last_blob,
        )
        if stats is not None:
            stats.candidate_rows_read += len(rows)
            stats.cdef_blobs_decoded += len(cdefs)
            stats.pages_fetched += 1
        yield _StoredIdentityBatch(cdefs, cursor)

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
        """Drop this view's temporary tables in its creating process.

        Returns:
            None. Individual SQLite cleanup errors remain best-effort.

        Raises:
            QueryIndexError: If the view was inherited from another process,
                before accessing SQLite or changing its temporary-table list.

        Side Effects:
            Removes tracked temporary relations from the owner-retained handle.
        """

        self._check_process()
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
        """Detach the complete occurrence graph from this process-local view.

        Returns:
            An AllOccurrenceTraversalSnapshot of definitions, stored IDs, and
            typed outgoing edges for the captured generation.

        Raises:
            QueryIndexError: If the view is closed or belongs to another process.
            sqlite3.Error: If reading the retained transaction fails.

        Side Effects:
            Reads SQLite rows without modifying Store authority.
        """

        self._check_active()
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
    legacy_reference_blob_false_positive = False
    reference_blob_error = "NULL value in reference_records.reference_blob"
    if (
            require_sqlite().sqlite_version_info < (3, 41, 0)
            and any(row[0] == reference_blob_error for row in quick)
    ):
        # Older SQLite mischecks non-key columns in WITHOUT ROWID tables.
        legacy_reference_blob_false_positive = all(
            row[0] is not None
            for row in con.execute("SELECT reference_blob FROM reference_records")
        )
    for row in quick:
        if row[0] != "ok" and not (
                row[0] == reference_blob_error
                and legacy_reference_blob_false_positive
        ):
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


def _unique_graph_roots(roots) -> tuple[ConcreteDefinition, ...]:
    """Deduplicate root inputs by complete graph identity."""

    unique: list[ConcreteDefinition] = []
    seen = set()
    for root in roots:
        if not isinstance(root, ConcreteDefinition):
            raise TypeError("Query-index roots must be ConcreteDefinition values.")
        key = identity_key(root)
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return tuple(unique)


def _v3_projection_rows(roots):
    """Build graph-exact typed candidate rows for newly registered roots.

    The rows deliberately contain only derived identity hashes and typed direct
    relationship facts.  They never serialize a new authoritative identity or
    make a Store completeness claim.
    """

    identities = set()
    relationships = set()
    for root in roots:
        root_hash = root.graph_hash()
        for value in relationship_closure((root,)):
            owner = identity_key(value)
            identities.add((root_hash, owner.kind, owner.digest))
            for edge in iter_direct_relationships(value):
                target = identity_key(edge.target)
                relationships.add((
                    root_hash, owner.kind, owner.digest, edge.kind.value,
                    _CODEC.encode_graph_path(edge.path), target.kind, target.digest,
                ))
    return tuple(sorted(identities)), tuple(sorted(relationships))


def _v3_row_digest(row) -> bytes:
    """Hash one unambiguous typed projection row for commutative coverage."""

    digest = hashlib.sha256()
    for value in row:
        payload = value if isinstance(value, bytes) else str(value).encode("ascii")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.digest()


def _v3_projection_digest(con, table: str, columns: str) -> tuple[int, bytes]:
    """Verify a row-set witness only on explicit integrity checks."""

    digest = bytearray(32)
    count = 0
    for row in con.execute(f"SELECT {columns} FROM {table} ORDER BY {columns}"):
        count += 1
        for index, byte in enumerate(_v3_row_digest(row)):
            digest[index] ^= byte
    return count, bytes(digest)


def _update_v3_projection_coverage(
    con, *, added_identities=(), added_relationships=(),
    removed_identities=(), removed_relationships=(),
) -> None:
    """Update row-set witnesses with only rows changed by this transaction."""

    families = (
        ("identity", added_identities, removed_identities),
        ("relationship", added_relationships, removed_relationships),
    )
    for family, additions, removals in families:
        if not additions and not removals:
            continue
        stored = con.execute(
            "SELECT row_count, row_digest FROM v3_projection_coverage WHERE family = ?",
            (family,),
        ).fetchone()
        if stored is None or len(stored[1]) != 32:
            raise QueryIndexCorrupt("Query V3 derived coverage witness is unavailable.")
        count = stored[0] + len(additions) - len(removals)
        if count < 0:
            raise QueryIndexCorrupt("Query V3 derived coverage count is inconsistent.")
        digest = bytearray(stored[1])
        for row in (*additions, *removals):
            for index, byte in enumerate(_v3_row_digest(row)):
                digest[index] ^= byte
        con.execute(
            "UPDATE v3_projection_coverage SET row_count = ?, row_digest = ? WHERE family = ?",
            (count, bytes(digest), family),
        )


def _v3_projection_is_complete(con) -> bool:
    """Validate derived row-set witnesses without consulting Store authority."""

    families = (
        ("identity", "v3_identity_projection", "root_graph_hash, identity_kind, identity_digest"),
        (
            "relationship", "v3_relationship_projection",
            "root_graph_hash, owner_kind, owner_digest, relationship_kind, path_blob, target_kind, target_digest",
        ),
    )
    for family, table, columns in families:
        stored = con.execute(
            "SELECT row_count, row_digest FROM v3_projection_coverage WHERE family = ?",
            (family,),
        ).fetchone()
        if stored is None or stored != _v3_projection_digest(con, table, columns):
            return False
    return True


def _require_clean_stored_root_guard(con) -> None:
    """Reject stored-root pages or deltas after unmanaged sidecar mutation."""

    row = con.execute(
        "SELECT dirty FROM v3_projection_guard WHERE family = 'stored_root'"
    ).fetchone()
    if row is None or row[0] != 0:
        raise QueryIndexDirty(
            "SQLite stored-root projection changed outside managed index publication."
        )


def _row_counts(con) -> dict[str, int]:
    return {
        table: con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in (
            "definitions", "feature_tokens", "postings", "definition_edges",
            "stored_roots", "v3_stored_root_projection",
            "reference_records", "reference_object_ids",
            "metadata_records",
        )
    }


def _empty_row_counts() -> dict[str, int]:
    return {
        table: 0
        for table in (
            "definitions", "feature_tokens", "postings", "definition_edges",
            "stored_roots", "v3_stored_root_projection",
            "reference_records", "reference_object_ids",
            "metadata_records",
        )
    }
