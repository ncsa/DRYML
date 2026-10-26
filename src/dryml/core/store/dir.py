"""Format-4 Node metadata, CDef-owned state buckets and owner-scoped readers.

Legacy publication entry points explicitly refuse V2/V3 records; they never
reinterpret that authority as identity-3 metadata.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
import threading
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from uuid import uuid4
from pathlib import Path
from typing import Literal

from ... import filesystem
from ...locking import FileLock, interprocess_lock, supports_advisory_locking
from ...paths import local_path_key
from .records import (
    StoreFormatRecord, StoreRecordError,
    StateKeyRecord, NodeBlockRecord, NodeRootRecord, StateVariantManifest,
    StateAvailabilityRecord, NodeMembershipRecord, DeclarationAnchorRecord, NodeMetadataRecord,
    DeclarationRequirementRecord, _json_bytes, _digest_value, _digest, _MAX_RECORD_BYTES,
    AuthorityRecord, MutationIntentRecord, NodeNameRecord, RetiredDeclarationAnchorRecord,
)
from . import authority
from . import retention as state_retention
from .authority import PreparedChange, StoreCutChanged
from .store import (
    Store, StoreAuthorityError, StoreCapabilityError,
    StorePublicationCapabilities,
    StateDescriptor, StateInstallation, StateReadSource,
)
from ..query.model import QueryIndexStatus, QueryIndexUnavailable, ReconcileReport
from ..query.sqlite import SQLiteQueryIndexConfig
from ..query.sqlite.index import SQLiteStoreQueryIndex


QueryIndexPolicy = Literal["auto", "sqlite", "memory", "none"]
_REMOVED_ENTRY_PREFIX = ".store-removed-"


class DirStore(Store):
    """Persist current logical authority directly beneath one local filesystem root.

    Shared Node blocks and compact root records are digest sharded. Each full
    StateKey owns one variant beneath its creation-CDef bucket; bucket headers,
    historical manifests, payload availability and independent root memberships
    remain distinct. Native publication persists staged content and changed
    directory entries under a cooperating-writer fence. Initial root creation
    and format-gate publication use a derived
    sibling advisory lock, so simultaneous trusted constructors never mistake a
    live temporary format file for old authority.

    Supported mutations use durable intents, byte CAS and one committed authority
    cut. Graph validation runs outside writer-blocking fences. Legacy publication
    consumers fail explicitly; enabled SQLite projection completes synchronously.
    Full-key native leases and physical grants fence private-data reclamation;
    their handshake policy lives in the Store retention adapter.

    Args:
        base_dir (str or PathLike): Direct Store root to open or initialize.
        query_index: Derived-index policy string or SQLiteQueryIndexConfig; it
            never changes which files constitute authoritative contents.

    Raises:
        ValueError: If the derived-index policy is unsupported.
        StoreAuthorityError: If existing authority is malformed/incompatible.
        StoreCapabilityError: If required filesystem guarantees are unavailable.

    Side Effects:
        May initialize the root and persistent Store UUID. Public metadata reads
        validate current authority. A caller-owned bounded content scope may share
        interpretation of identical immutable bytes across detached batches; every
        batch still reads and revalidates its own current cut and byte witnesses.
        Existing incompatible Stores are
        refused without conversion, deletion or replacement.
    """

    _supports_deferred_local_state = True

    def __init__(
            self,
            base_dir: str | os.PathLike[str],
            *,
            query_index: QueryIndexPolicy | SQLiteQueryIndexConfig = "auto",
            _existing_only: bool = False):
        self._base_dir = os.path.abspath(os.fspath(base_dir))
        if isinstance(query_index, str):
            if query_index not in {"auto", "sqlite", "memory", "none"}:
                raise ValueError("DirStore query_index must be 'auto', 'sqlite', 'memory', 'none', or SQLiteQueryIndexConfig.")
            self._query_index_policy: QueryIndexPolicy = query_index
            self._query_index_config = None
        elif isinstance(query_index, SQLiteQueryIndexConfig):
            self._query_index_policy = "sqlite"
            self._query_index_config = query_index
        else:
            raise ValueError("DirStore query_index must be 'auto', 'sqlite', 'memory', 'none', or SQLiteQueryIndexConfig.")
        self.query_index = query_index
        self._query_index_instance: SQLiteStoreQueryIndex | None = None
        self._query_index_unavailable = False
        self._initialize_format(existing_only=_existing_only)
        self._store_uuid = self._read_file(self.store_format_path, StoreFormatRecord).store_uuid
        self._state_owner_closed = False
        self._staging_leases = {}
        self._staging_threads = {}
        self._staging_owner_uuid = str(uuid4())
        self._prepared_fact_owner = object()
        evidence = os.stat(self._base_dir)
        self._authority_evidence = (evidence.st_dev, evidence.st_ino)
        authority.read_cut(self)

    @property
    def store_uuid(self):
        """Return the persistent format-4 authority UUID (str), without payload IO."""
        return self._store_uuid

    @property
    def retention_domain(self):
        """Return stable Store UUID for key ordering/owner evidence (str).

        During bootstrap only, a root-local provisional name tracks writer order.
        No IO, graph preparation or grant acquisition occurs.
        """
        domain = getattr(self, "_store_uuid", None)
        return domain if domain is not None else "bootstrap:" + os.path.realpath(self.base_dir)

    @property
    def retention(self):
        """Return durable RetentionAuthority primitives; association policy is caller-owned."""
        return state_retention.RetentionAuthority(self)

    def _state_lease_path(self, key):
        return os.path.join(self.base_dir, ".state-leases", key.cdef_digest[:2],
                            key.cdef_digest, key.codec_hash + ".lock")

    def state_leases(self, keys, *, exclusive=False, blocking=True):
        """Protect exact keys with native leases before writer fences; see Store.state_leases."""
        return state_retention.state_leases(((self, key) for key in keys), exclusive=exclusive, blocking=blocking)

    def delete_state(self, key):
        """Try immediate full-key cleanup, preserving history; see Store.delete_state."""
        return state_retention.delete_state(self, key)

    def retry_state_removal(self, key, incarnation):
        """Retry exact withdrawn incarnation safely; see Store.retry_state_removal."""
        return state_retention.delete_state(self, key, incarnation=incarnation)

    def _assert_state_owner_open(self):
        if getattr(self, "_state_owner_closed", False) or getattr(self, "_closed_handle", False):
            raise RuntimeError("Store state source owner is closed.")
        if hasattr(self, "_store_uuid"):
            record = self._read_file(self.store_format_path, StoreFormatRecord)
            if record is None or record.store_uuid != self._store_uuid:
                raise StoreAuthorityError("Store authority UUID changed; reopen its owner.")

    def preflight_publication(self, operation, *, local_state=False, record_family=None):
        """Require backend guarantees and a current-format publication operation.

        operation is a diagnostic name, local_state requests staging guarantees,
        and record_family must explicitly name dryml-node-graph/v1.
        Returns None or raises StoreCapabilityError before legacy writer effects.
        QueryIndexError reports unavailable/conflicting persistent destinations
        before authority effects. May bind derived index configuration under the
        Store fence; no product publication or graph preparation occurs here.
        V2/V3 Repo/reference writers await their owning consumer cutover; changing
        the Store marker never admits their old nested records into format 4.
        """
        self._assert_state_owner_open()
        super().preflight_publication(operation, local_state=local_state, record_family=record_family)
        if record_family != "dryml-node-graph/v1":
            raise StoreCapabilityError(
                f"{operation} uses a legacy Store consumer; format 4 requires Node/state-bucket APIs.")
        self.open_query_index()

    def authority_cut(self):
        """Return a coherent AuthorityRecord (UUID/epoch/pending), without graph IO.

        Missing/incompatible authority raises StoreAuthorityError. This describes
        buffered authority for ZipStore, not durable archive completion.
        """
        self._assert_state_owner_open()
        with self.authority_read_fence():
            return authority.read_cut(self)

    def recover_authority(self):
        """Classify interrupted work and reestablish its native persistence barriers.

        Recovery is metadata-only, preserves authoritative files/staging, and
        never captures/restores Objects or completes unexecuted transitions.
        Returns replayable intent records without executing missing transitions.
        Missing/corrupt evidence raises StoreAuthorityError. Unresolved barriers
        raise StoreMutationError with the mutation/cut receipt and preserve pending
        classification. Recovery flushes affected entries and ancestor directories,
        not private payloads. Unsupported standalone native barriers fail explicitly.
        Classification is independent of SQLite availability or compatibility.
        """
        # Authority classification must remain usable independently of a missing,
        # unavailable or incompatible derived destination.
        self._assert_state_owner_open()
        super().preflight_publication("recover authority", record_family="dryml-node-graph/v1")
        return authority.recover(self)

    def _persist_authority_projection(self, record):
        data = record.intent
        directories = set()
        for row in data["projection"]:
            if authority._observed(self, row) != authority.decode(row["value"]):
                raise StoreCutChanged("Authority changed before native persistence recovery.")
            path = os.path.join(self.base_dir, row["path"])
            if row["kind"] == "file" and row["value"] is not None:
                self._filesystem(filesystem.sync_file, path)
            directories.add(os.path.dirname(path))
        for row in data["transitions"]:
            if row["staging"] is not None:
                directories.add(os.path.dirname(os.path.join(self.base_dir, row["staging"])))
        # A failed mkdir barrier can leave an ancestor visible before the target
        # exists. Repeat the entire affected chain, and both rename parents. Child
        # content crossed its tree barriers before the native visibility move.
        ancestors = set()
        for directory in directories:
            while True:
                if os.path.isdir(directory):
                    ancestors.add(directory)
                if directory == self.base_dir:
                    break
                directory = os.path.dirname(directory)
        for directory in sorted(ancestors, key=lambda path: (-path.count(os.sep), path)):
            self._filesystem(filesystem.sync_directory, directory)

    def acknowledge_mutation(self, mutation_id, epoch):
        """Acknowledge exactly a captured classified UUID/cut; return current cut.

        The owner must first incorporate its exact replay work and coverage.
        Native persistence confirmation remains Store-owned even with committed
        SQL coverage. Unresolved barriers raise StoreMutationError with pending
        mutation/cut evidence. Mismatched evidence raises StoreAuthorityError.
        Later mutations are never acknowledged by this call.
        """
        self.preflight_publication("acknowledge mutation", record_family="dryml-node-graph/v1")
        try:
            index = self.open_query_index()
            if index is not None:
                return index.acknowledge_store_mutation(mutation_id, epoch)
            return authority.acknowledge(self, mutation_id, epoch)
        except (StoreAuthorityError, StoreRecordError, TypeError, ValueError):
            raise
        except BaseException as error:
            cut, refs = None, ()
            try:
                cut = self.authority_cut()
                record = self._read_file(os.path.join(self.base_dir, "mutations", mutation_id + ".record"), MutationIntentRecord)
                if record is not None:
                    refs = tuple(record.intent["confirmed"])
            except Exception:
                pass
            if not isinstance(error, Exception):
                error.mutation_id, error.cut = mutation_id, cut
                error.confirmed_refs, error.index_disposition = refs, "repair_pending"
                raise
            raise authority.StoreMutationError(str(error), mutation_id, cut, refs) from error

    @state_retention._protect_replay
    def replay_mutation(self, mutation_id):
        """Complete captured partial transitions without graph or lifecycle work.

        mutation_id must name a persisted classified intent in this Store, even
        if its partial projection was already acknowledged. Returns
        the repaired AuthorityRecord. Exact projection witnesses must still match;
        later conflicting writers cause StoreCutChanged, never overwrite. Missing
        owned staging or invalid staged payload fails explicitly. All validation
        and payload authentication precede the writer fence. The repair is itself
        journaled. An already acknowledged original receipt is preserved; pending
        original work is acknowledged only at its exact captured cut. Repeating
        repair of the exact final bytes is a no-op, including incarnation checks.
        State repair acquires independent nonblocking exclusive key leases and
        refuses another live allocator's staging. Retention owner-ledger repair
        serializes on its intent token; grant repair requires the owning
        protect/release handshake instead of generic replay.
        """
        return self._replay_mutation_impl(mutation_id)

    def _repair_mutation_record(self, mutation_id):
        from .records import _uuid_value
        _uuid_value(mutation_id)
        with self.authority_read_fence():
            cut = authority.read_cut(self)
            record = self._read_file(authority._path(self, mutation_id), MutationIntentRecord)
            if (record is None or record.intent["mutation_id"] != mutation_id
                    or record.intent["store_uuid"] != cut.store_uuid
                    or record.intent["status"] == "pending"
                    or not record.intent["persistence_confirmed"]):
                raise StoreAuthorityError("Repair requires an exact persisted classified, persistence-confirmed mutation.")
            return record

    def _replay_mutation_impl(self, mutation_id):
        from .records import _uuid_value, _decode_authority_bytes, StateVariantManifest
        import json
        _uuid_value(mutation_id)
        self.preflight_publication("replay mutation", record_family="dryml-node-graph/v1")
        self.recover_authority()
        captured = self._repair_mutation_record(mutation_id)
        data = captured.intent
        def prepare():
            observed, final = {}, {}
            for row in data["projection"]:
                observed[(row["path"], row["kind"])] = _decode_authority_bytes(row["value"])
            for row in data["transitions"]:
                final[(row["path"], row["kind"])] = _decode_authority_bytes(row["after"])
            current = {}
            for row in data["transitions"]:
                key = (row["path"], row["kind"])
                if key not in current:
                    current[key] = authority._observed(self, row)
            if current == final:
                return [], (), None
            if authority.read_cut(self).epoch != data["epoch"]:
                raise StoreCutChanged("Repair cut is stale; later Store authority must be reconciled explicitly.")
            if current != observed:
                raise StoreCutChanged("Later authority conflicts with captured repair evidence.")
            done = {key for key in current if current[key] == final[key]}
            changes = []
            initial = {}
            for row in data["transitions"]:
                initial.setdefault((row["path"], row["kind"]), row)
            # Visibility after an interrupted native barrier is not a durable
            # acknowledgement. Re-persist exact metadata and parent entries even
            # when all desired bytes are already visible. Staged child content
            # crossed its file/tree barriers before the native visibility move.
            for key in current:
                if key not in done:
                    continue
                path, kind = key
                if kind == "file":
                    value = current[key]
                    if value is None:
                        before = _decode_authority_bytes(initial[key]["before"])
                        if before is not None:
                            changes.append(PreparedChange(path, None, before))
                            changes.append(PreparedChange(path, before, None))
                    else:
                        changes.append(PreparedChange(path, value, value))
                elif kind == "variant":
                    evidence = json.loads(current[key])
                    manifest = StateVariantManifest.from_bytes(_decode_authority_bytes(evidence["manifest"]))
                    self._authenticate_state_data(manifest, os.path.join(self.base_dir, path, "data"))
                    for file_path in (path + "/manifest.record", path + "/availability.record",
                                      path.rsplit("/", 1)[0] + "/def.pkl"):
                        value = authority.raw(self, file_path)
                        if value is None:
                            raise StoreAuthorityError("Missing metadata required to persist recovered state.")
                        changes.append(PreparedChange(file_path, value, value))
            for row in data["transitions"]:
                key = (row["path"], row["kind"])
                if key in done:
                    continue
                before, after = (_decode_authority_bytes(row[n]) for n in ("before", "after"))
                if current[key] == after:
                    continue
                if current[key] != before:
                    raise StoreAuthorityError("Partial authority does not match a replayable transition.")
                if row["kind"] != "file":
                    stage = os.path.join(self.base_dir, row["staging"])
                    if not os.path.isdir(stage):
                        raise StoreAuthorityError("Recovery staging is missing; preserve intent and restage explicitly.")
                    if row["kind"] == "variant":
                        evidence = json.loads(after)
                        manifest = StateVariantManifest.from_bytes(_decode_authority_bytes(evidence["manifest"]))
                        if authority.variant_evidence(self, row["staging"]) != after:
                            raise StoreAuthorityError("Recovery staging metadata changed.")
                        self._authenticate_state_data(manifest, os.path.join(stage, "data"))
                    else:
                        path = os.path.join(self.base_dir, row["path"], "..", "manifest.record")
                        manifest = self._read_file(os.path.normpath(path), StateVariantManifest)
                        if manifest is None:
                            raise StoreAuthorityError("Recovery data has no authoritative manifest.")
                        self._authenticate_state_data(manifest, stage)
                changes.append(PreparedChange(row["path"], before, after, row["staging"], row["kind"]))
                current[key] = after
            return changes, tuple(data["scopes"]) + ("replay:" + mutation_id,), None
        # Conflict is not an invitation to recapture or reinterpret the old intent.
        self._prepare_and_apply(prepare, retry_stale=False)
        cut = self.authority_cut()
        if mutation_id not in cut.pending:
            return cut
        return self.acknowledge_mutation(mutation_id, data["epoch"])

    @state_retention._protect_prepared
    def apply_prepared_mutation(self, facts):
        """Apply Store-owned PreparedMutation facts; return (mutation UUID, cut).

        Facts contain validated immutable bytes, affected scopes and exact CAS
        witnesses; no callbacks or live graph values. Stale facts raise
        StoreCutChanged before effects. Failed publication raises StoreMutationError
        with confirmed references/cut and repair disposition. New work remains
        pending in nonpersistent mode until its owner acknowledges it. Enabled
        persistent projection commits and acknowledges before return. Callers can
        obtain replay data with pending_mutations. This seam is owned
        by this Store handle and refuses another handle's prepared facts.
        State installs retain shared full-key leases. Manual retention, removal
        and availability-record changes raise StoreCapabilityError: use their
        owning lifecycle APIs so owner dispositions and key protection participate.
        """
        if type(facts) is not authority.PreparedMutation:
            raise TypeError("Store mutations require PreparedMutation facts.")
        if facts.owner is not self._prepared_fact_owner:
            raise StoreAuthorityError("Prepared facts belong to a different Store owner.")
        self.preflight_publication("apply prepared mutation", record_family="dryml-node-graph/v1")
        index = self.open_query_index()
        if index is not None:
            self._validate_projection_changes(facts)
            if not self.authority_cut().pending:
                index.maintain_store()
        identity, cut = authority.apply(self, facts)
        if index is not None:
            self._complete_index_mutation(index, identity, cut)
            cut = self.authority_cut()
        return identity, cut

    @staticmethod
    def _validate_projection_changes(facts):
        from ..query.authority import projection_record_type
        for change in facts.changes:
            if change.kind == "data" and not change.path.endswith("/data"):
                raise StoreAuthorityError("State data transition requires its exact data/ address.")
            paths = (change.path,) if change.kind == "file" else (
                (change.path[:-5] if change.kind == "data" else change.path) + "/manifest.record",
                (change.path[:-5] if change.kind == "data" else change.path) + "/availability.record")
            for path in paths:
                reader = projection_record_type(path)
                if change.kind == "file" and change.after is not None:
                    try:
                        reader.from_bytes(change.after)
                    except StoreRecordError:
                        if reader is not DeclarationAnchorRecord:
                            raise
                        RetiredDeclarationAnchorRecord.from_bytes(change.after)

    def _complete_index_mutation(self, index, identity, cut):
        try:
            index.maintain_store()
        except BaseException as error:
            refs = ()
            try:
                if identity is not None:
                    record = self._read_file(os.path.join(self.base_dir, "mutations", identity + ".record"), MutationIntentRecord)
                    if record is not None:
                        refs = record.intent["confirmed"]
                cut = self.authority_cut()
            except Exception:
                pass
            if not isinstance(error, Exception):
                error.mutation_id, error.cut = identity, cut
                error.confirmed_refs, error.index_disposition = tuple(refs), "repair_pending"
                error.recovery_attempt = "journal-replay-or-metadata-rebuild"
                raise
            outcome = authority.StoreMutationError("SQLite projection completion failed: " + str(error), identity, cut, refs)
            outcome.recovery_attempt = "journal-replay-or-metadata-rebuild"
            raise outcome from error

    def pending_mutations(self):
        """Return exact intent records captured under one short authority cut.

        Records retain base/result cuts, scopes, expected transitions, confirmed
        relative references and exact projection bytes for batch delta/coverage
        consumers. Pending unclassified work is not a completeness certificate.
        Missing/mismatched records raise StoreAuthorityError. No graph, payload,
        SQL, application hook or repair work runs here.
        """
        self._assert_state_owner_open()
        with self.authority_read_fence():
            cut = authority.read_cut(self)
            records = []
            for identity in cut.pending:
                record = self._read_file(os.path.join(self.base_dir, "mutations", identity + ".record"), MutationIntentRecord)
                if (record is None or record.intent["mutation_id"] != identity
                        or record.intent["store_uuid"] != cut.store_uuid):
                    raise StoreAuthorityError("Missing or mismatched pending intent.")
                records.append(record)
            return tuple(records)

    def _metadata_read(self):
        return authority.detached_read(self)

    def _relative(self, path):
        return os.path.relpath(path, self.base_dir).replace(os.sep, "/")

    def _prepared_file(self, path, record, *, mutable=False):
        before = authority.raw(self, self._relative(path))
        after = None if record is None else record.to_bytes()
        if before == after:
            return None
        if before is not None and not mutable:
            raise StoreAuthorityError("Immutable Node metadata collision.")
        return PreparedChange(self._relative(path), before, after)

    def _prepare_and_apply(self, prepare, *, retry_stale=True, return_receipt=False):
        # The preparation callable is strictly outside the Store writer fence.
        # A bounded retry admits concurrent siblings without merging stale CAS.
        for _ in range(8):
            try:
                self.recover_authority()
                index = self.open_query_index()
                if index is not None and not self.authority_cut().pending:
                    index.maintain_store()
                with self._metadata_read() as captured:
                    changes, scopes, result = prepare()
                    unique = list(dict.fromkeys(change for change in changes if change is not None))
                    facts = captured.batch(unique, scopes)
                if index is not None:
                    self._validate_projection_changes(facts)
                identity, cut = authority.apply(self, facts, _acknowledge_nonpersistent=index is None)
                if index is not None:
                    self._complete_index_mutation(index, identity, cut)
                return (identity, self.authority_cut()) if return_receipt else result
            except StoreCutChanged:
                if not retry_stale:
                    raise
                continue
        raise StoreCutChanged("Store changed repeatedly; retry publication explicitly.")

    def _query_v3_inventory_capabilities(self):
        raise StoreCapabilityError("Format 4 Node authority requires the new query coverage contract.")

    def preflight_save_batch(self):
        """Admit durable format-4 retained saves before capture; return None.

        Requires native grants/staging and the configured projection destination.
        Buffered owners and unavailable explicit indexes raise before hooks.
        """
        self.publication_capabilities.require_durable("user retained graph save")
        self.preflight_publication("Node save batch", local_state=True,
                                   record_family="dryml-node-graph/v1")
        index = self.open_query_index()
        if index is not None:
            index.maintain_store()

    def repair_save_batch(self, mutation_id, token):
        """Replay one failed save under its owner-token and exclusive-key fences.

        mutation_id and token are UUIDs from SavePublication. Only the exact
        prepared/adopted local owner intent and matching per-key grants are
        admitted; abandonment/retirement fences repair permanently. Returns the
        repaired cut, without capture/hooks. Missing staging, newer conflicting
        authority or active use raises StoreAuthorityError with evidence intact.
        Projection acknowledgement does not consume repair eligibility. Missing
        transitions get a new journaled mutation; the original acknowledged facts
        remain unchanged. Repeated repair verifies exact final bytes without writes.
        Index-only failures may instead use the index's maintain_store method.
        """
        from .records import _decode_authority_bytes
        from .retention import RetentionIntentRecord, RetentionGrantRecord
        self.publication_capabilities.require_durable("repair retained graph save")
        self.preflight_publication("repair Node save batch", record_family="dryml-node-graph/v1")
        with self.retention._fence(token):
            self.recover_authority()
            record = self._repair_mutation_record(mutation_id)
            changes = record.intent["transitions"]
            path = "retention/intents/" + token + ".record"
            intents = [RetentionIntentRecord.from_bytes(_decode_authority_bytes(row["after"]))
                       for row in changes if row["path"] == path and row["after"] is not None]
            if not intents or intents[0].owner_domain != self.retention_domain:
                raise StoreAuthorityError("Mutation does not belong to this save owner token.")
            intended = intents[0]
            actual = self.retention.read(token)
            if actual is not None and (actual.disposition not in {"prepared", "adopted"}
                                       or actual.resources != intended.resources or actual.graph_digest != intended.graph_digest):
                raise StoreAuthorityError("Save repair was abandoned, retired or replaced.")
            if len(intended.resources) != 1 or intended.resources[0].domain != self.retention_domain:
                raise StoreAuthorityError("Save repair requires its exact target-local resource ledger.")
            resource = intended.resources[0]
            for allocation, owner in resource.staging:
                if (os.path.isdir(os.path.join(self._staging_root, allocation))
                        and self._staging_owner(allocation) != owner):
                    raise StoreAuthorityError("Save repair staging no longer matches its resource owner.")
            grant = RetentionGrantRecord(self.retention_domain, token, intended.graph_digest, resource.keys) if resource.keys else None
            for row in changes:
                if row["path"].startswith("retention/intents/") and row["path"] != path:
                    raise StoreAuthorityError("Save repair cannot acquire another owner intent.")
                if "/grants/" in row["path"]:
                    if row["after"] is None or RetentionGrantRecord.from_bytes(_decode_authority_bytes(row["after"])) != grant:
                        raise StoreAuthorityError("Save repair contains a foreign or retiring grant.")
            with self._replay_staging(changes):
                with self.state_leases(resource.keys, exclusive=True, blocking=False) as acquired:
                    if not acquired:
                        raise StoreAuthorityError("Save repair is in-use; retry after active leases end.")
                    return self._replay_mutation_impl(mutation_id)

    @contextmanager
    def read_states(self, keys):
        """Borrow authenticated available full keys, grouping headers at one cut.

        Returns a context yielding a key/StateReadSource dict. Missing/withdrawn
        keys are omitted; malformed evidence or payloads raise StoreAuthorityError.
        Sources retain ordered leases and expire on exit. No graph preparation or
        payload authentication runs under the short metadata validation fence.
        """
        keys = tuple(dict.fromkeys(keys))
        sources = {}
        with self.state_leases(keys):
            try:
                with self._metadata_read() as captured:
                    origins = {digest: self.read_creation(digest) for digest in dict.fromkeys(k.cdef_digest for k in keys)}
                    for key in keys:
                        origin = origins[key.cdef_digest]
                        descriptor = self._read_variant(key, origin)
                        if descriptor is None or not descriptor.available:
                            continue
                        data = self.state_data_path(key)
                        self._authenticate_state_data(descriptor.manifest, data)
                        sources[key] = StateReadSource(self, descriptor.state, descriptor.manifest, data,
                                                       _metadata_evidence=captured, _metadata_owner=self._prepared_fact_owner)
                yield sources
            finally:
                for source in sources.values():
                    source._active = False

    def publish_save_batch(self, batch):
        """Install a NodeSaveBatch in one journal/projection commit.

        Full-key leases must already cover batch.states in global Store/key order.
        Metadata comparison, owned staging validation and payload authentication
        run in detached preparation, never in writer fences. Returns mutation UUID
        and acknowledged cut. Each new key gets a physical provisional grant
        before membership, then an adopted user root unless batch.provisional.
        That owner root covers only this Store's exact resources, not external
        payload availability; Repo qualifies complete recovery after all targets.
        Partial writes preserve prepared intents/grants and exact replay evidence;
        no failed or ambiguous publication authorizes automatic retirement.
        Withdrawn keys explicitly require a separate exclusive recovery phase.
        """
        from ..repo_plan import NodeSaveBatch
        from ..node_prepare import NodePreparation
        from ..node_views import CREATION_VIEW, OBJECT_PROJECTION_VIEW, view_equal
        from .retention import (RetentionIntentRecord, RetentionResource,
                                RetentionGrantRecord, RetentionPendingRecord, _grant_path, _held, _key_address)
        if not isinstance(batch, NodeSaveBatch):
            raise TypeError("Node publication requires NodeSaveBatch.")
        self.preflight_save_batch()
        keys = tuple(sorted((m.key for m, _ in batch.states), key=lambda k: (k.cdef_digest, k.codec_hash)))
        held = dict(_held())
        if any(_key_address(self, key) not in held for key in keys):
            raise StoreAuthorityError("Node save requires active ordered full-key leases.")
        seen_keys = set()
        for manifest, stage in batch.states:
            if manifest.key in seen_keys:
                raise StoreAuthorityError("Node batch repeats a full StateKey.")
            seen_keys.add(manifest.key)
            if stage is not None:
                self._require_staging_owner(stage)
                self._authenticate_state_data(manifest, os.path.join(stage, "data"))
                self._write_state_stage_record(Path(stage, "manifest.record"), manifest.to_bytes())
                availability = StateAvailabilityRecord(manifest.key, str(uuid4()))
                self._write_state_stage_record(Path(stage, "availability.record"), availability.to_bytes())
            availability = self._read_file(os.path.join(os.path.dirname(self.state_data_path(manifest.key)), "availability.record"), StateAvailabilityRecord)
            if availability is not None and (availability.status != "published" or not os.path.isdir(self.state_data_path(manifest.key))):
                if not dict(_held()).get(_key_address(self, manifest.key)):
                    raise StoreAuthorityError("Withdrawn state requires an independent exclusive publication phase.")
                state_retention.settle_interrupted_publication(self, manifest.key)
                state_retention.detach_withdrawn_data(self, manifest.key)
        staging = tuple((self._require_staging_owner(stage), self._staging_owner(self._require_staging_owner(stage)))
                        for _, stage in batch.states if stage is not None)
        resource = RetentionResource(self.retention_domain, keys, staging)
        intent = RetentionIntentRecord(self.retention_domain, batch.token, batch.graph_digest, (resource,))
        grant = RetentionGrantRecord(self.retention_domain, batch.token, batch.graph_digest, keys) if keys else None
        intent_path = os.path.join(self.base_dir, "retention", "intents", batch.token + ".record")
        pending_path = os.path.join(self.base_dir, "retention", "pending.record")

        def prepare():
            previous_intent = self._read_file(intent_path, RetentionIntentRecord)
            if previous_intent is not None:
                if (previous_intent.disposition != "adopted" or previous_intent.graph_digest != intent.graph_digest
                        or len(previous_intent.resources) != 1 or previous_intent.resources[0].domain != self.retention_domain
                        or previous_intent.resources[0].keys != keys):
                    raise StoreAuthorityError("Save publication token already exists; reconcile its owning intent.")
            pending = self._read_file(pending_path, RetentionPendingRecord) or RetentionPendingRecord()
            prepared_pending = RetentionPendingRecord(pending.tokens + (batch.token,))
            changes = [] if previous_intent is not None else [self._prepared_file(pending_path, prepared_pending, mutable=True),
                       self._prepared_file(intent_path, intent)]
            changes.append(self._prepared_file(self._node_path("definitions/blocks", batch.block.digest), batch.block))
            origins = {}
            evidence = {source.state.key.cdef_digest: source for source in batch.creation_sources
                        if source.store is self and source._metadata_owner is self._prepared_fact_owner}
            validated = set()
            current_cut = authority.read_cut(self)
            for header, node in batch.creations:
                source = evidence.get(header.digest)
                proof = None if source is None else source._metadata_evidence
                if proof is not None and proof.cut == current_cut:
                    if id(proof) not in validated:
                        with self.authority_read_fence():
                            proof.validate()
                        validated.add(id(proof))
                    old = source.state.creation_cdef
                else:
                    old = self.read_creation(header.digest)
                if old is not None and not view_equal(old, node, CREATION_VIEW):
                    raise StoreAuthorityError("Creation bucket header collision.")
                origins[header.digest] = old or node
                if old is None:
                    changes.append(self._prepared_file(os.path.join(self._bucket_path(header.digest), "def.pkl"), header))
            for manifest, stage in batch.states:
                key = manifest.key
                origin = origins.get(key.cdef_digest)
                if origin is None:
                    raise StoreAuthorityError("State has no prepared creation header.")
                target = os.path.dirname(self.state_data_path(key))
                old = self._read_variant(key, origin)
                if old is not None:
                    if old.manifest != manifest:
                        raise StoreAuthorityError("State manifest collision.")
                    if old.available:
                        self._authenticate_state_data(manifest, self.state_data_path(key))
                    else:
                        if stage is None or not dict(_held()).get(_key_address(self, key)):
                            raise StoreAuthorityError("Withdrawn state requires owned staging and independent exclusive recovery.")
                        availability_path = os.path.join(target, "availability.record")
                        incarnation = StateAvailabilityRecord(key, str(uuid4()))
                        first = self._prepared_file(availability_path, replace(incarnation, status="prepared"), mutable=True)
                        changes.append(first)
                        changes.append(PreparedChange(self._relative(self.state_data_path(key)), None, b"directory",
                                                       self._relative(os.path.join(stage, "data")), "data"))
                        changes.append(PreparedChange(self._relative(availability_path), first.after, incarnation.to_bytes()))
                else:
                    if stage is None:
                        raise StoreAuthorityError("Selected installed state is unavailable.")
                    changes.append(PreparedChange(self._relative(target), None,
                                   authority.variant_evidence(self, self._relative(stage)), self._relative(stage), "variant"))
                if grant is not None:
                    changes.append(self._prepared_file(_grant_path(self, key, grant), grant))
            # Merge all source contributions before writing each identity anchor.
            contributions = {}
            for record, node in batch.roots:
                ids = tuple(n.object_id for n in self._publication_identities(node))
                ids = tuple(dict.fromkeys(ids))
                for identity in ids:
                    contributions.setdefault(identity, []).append((record.root_kind, record.digest))
            for projection, node in batch.projections:
                identity = node.object_id
                if identity not in contributions:
                    continue
                old = self.read_declaration_anchor(identity)
                if old is not None:
                    if not view_equal(self._expand_node_root(old.projection), node, OBJECT_PROJECTION_VIEW):
                        raise StoreAuthorityError("ObjectId declaration projection conflicts with existing anchor.")
                    projection = old.projection
                sources = tuple(sorted(set((() if old is None else old.sources) + tuple(contributions[identity]))))
                anchor = DeclarationAnchorRecord(identity, projection, sources)
                key = self._anchor_key(identity)
                changes.append(self._prepared_file(self._node_path("declarations/anchors", key), anchor, mutable=True))
                changes.append(self._prepared_file(self._node_path("declarations/required", key),
                               DeclarationRequirementRecord(identity, projection.digest)))
            for record, node in batch.roots:
                kind = record.root_kind
                root_path = self._node_path("graphs" if kind == "saved" else "definitions/roots", record.digest)
                old = self._read_file(root_path, NodeRootRecord)
                if old is not None:
                    previous = self._expand_node_root(old)
                    if not NodePreparation((previous,)).exact_equal(previous, NodePreparation((node,)), node):
                        raise StoreAuthorityError("Node root semantic collision.")
                else:
                    changes.append(self._prepared_file(root_path, record))
                identities = tuple(dict.fromkeys(n.object_id for n in self._publication_identities(node)))
                changes.append(self._prepared_file(self._node_path("membership/" + kind, record.digest),
                               NodeMembershipRecord(kind, record.digest, identities)))
            for name in batch.names:
                changes.append(self._prepared_file(os.path.join(self.base_dir, "names", name.name + ".record"), name, mutable=True))
            from .records import NodeMetadataRecord
            from ..metadata import _captured_annotation_data, _node_snapshot_metadata, encode_metadata_mapping
            import json
            for root_record, node, content in batch.captures:
                data = json.loads(content)
                for lineage in data["lineages"]:
                    if lineage["identity"] is None:
                        continue
                    digest = lineage["projection"]
                    path = self._node_metadata_path("lineage", digest)
                    old = self._read_file(path, NodeMetadataRecord)
                    if old is None:
                        record = NodeMetadataRecord("lineage", digest, _json_bytes(lineage))
                        changes.append(self._prepared_file(path, record))
                    else:
                        self._validate_node_metadata_record(old, "lineage", digest)
                        lineage.update(json.loads(old.content))
                for scope, digest, field in (("current_object", node.object_projection_digest(), "object"),
                                              ("current_state", root_record.digest, "state")):
                    path = self._node_metadata_path(scope, digest)
                    old = self._read_file(path, NodeMetadataRecord)
                    if old is not None:
                        self._validate_node_metadata_record(old, scope, digest)
                    values = None if old is None else json.loads(old.content)["values"]
                    explicit = None if batch.annotations is None else getattr(batch.annotations, field)
                    if explicit is not None:
                        values = encode_metadata_mapping(explicit)
                        changes.append(self._prepared_file(path, NodeMetadataRecord(scope, digest, _json_bytes({"values": values})), mutable=True))
                    from ..metadata import decode_metadata_mapping
                    data["captured_annotations"][field] = _captured_annotation_data(None if values is None else decode_metadata_mapping(values))
                _node_snapshot_metadata(node, data)
                path = self._node_metadata_path("capture", root_record.digest)
                old = self._read_file(path, NodeMetadataRecord)
                if old is None:
                    changes.append(self._prepared_file(path, NodeMetadataRecord("capture", root_record.digest, _json_bytes(data))))
                else:
                    self._validate_node_metadata_record(old, "capture", root_record.digest)
                    _node_snapshot_metadata(node, json.loads(old.content))
            if not batch.provisional and previous_intent is None:
                changes.append(PreparedChange(self._relative(intent_path), intent.to_bytes(),
                                              replace(intent, disposition="adopted").to_bytes()))
                changes.append(PreparedChange(self._relative(pending_path), prepared_pending.to_bytes(), pending.to_bytes()))
            return changes, ("state", "creation", "root", "membership", "declarations", "names", "retention", "metadata"), None
        return self._prepare_and_apply(prepare, return_receipt=True)

    def save_batch_needs_recovery(self, batch):
        """Return whether exact availability requires a separate exclusive phase.

        Reads only full-key availability/presence under a detached revalidated
        metadata cut. Does not hash graphs, authenticate data, acquire key leases
        or mutate authority. Publication freshly checks this scheduling evidence.
        """
        with self._metadata_read():
            for manifest, _ in batch.states:
                key = manifest.key
                record = self._read_file(os.path.join(os.path.dirname(self.state_data_path(key)), "availability.record"), StateAvailabilityRecord)
                if record is not None and (record.status != "published" or not os.path.isdir(self.state_data_path(key))):
                    return True
            return False

    def query(self):
        """Return a deferred current-format identity/discovery query.

        Reads no source until a terminal. Enabled ready indexes supply bounded
        candidates; missing/stale coverage follows refresh/scan controls. Memory
        and disabled indexing require an explicitly permitted metadata-only scan.
        Malformed authority and changed continuations fail visibly; no payload IO
        or Object construction occurs during discovery.
        """
        from ..query.query import IdentityQuery
        return IdentityQuery.from_store(self)

    def _node_path(self, area, digest):
        _digest_value(digest, area + " digest")
        return self._digest_path(os.path.join(self.base_dir, area), digest)

    def _bucket_path(self, digest):
        _digest_value(digest, "creation digest")
        return os.path.join(self.base_dir, "states", digest[:2], digest)

    def state_data_path(self, key):
        """Derive data/ from StateKey with zero hashing/discovery; asserts no availability.

        key must be a StateKey or TypeError is raised. The returned str is private
        addressing; consumers borrow data through read_state. Performs no IO.
        """
        from ..state_values import StateKey
        if not isinstance(key, StateKey):
            raise TypeError("State addressing requires a StateKey.")
        return os.path.join(self._bucket_path(key.cdef_digest), key.codec_hash, "data")

    @staticmethod
    def _prepare_node_roots(roots, kind):
        from ..node_prepare import NodePreparation, EXACT_VIEW
        from ..node_views import CREATION_VIEW, OBJECT_PROJECTION_VIEW
        from ..node_codec import encode_node_graph
        policy = {"construction": CREATION_VIEW, "projection": OBJECT_PROJECTION_VIEW}.get(kind, EXACT_VIEW)
        prepared = NodePreparation(roots, view=policy)
        for node in prepared.nodes:
            selected = getattr(node, "state", None)
            if selected is not None:
                try:
                    selected.validate_for(node, codec=selected.codec)
                except (ValueError, TypeError) as error:
                    raise StoreAuthorityError("Invalid selected state creation/compatibility evidence.") from error
        packet = encode_node_graph(prepared)
        views = packet.pop("views")
        block = NodeBlockRecord(_json_bytes(packet))
        block_digest = block.digest
        records = tuple(NodeRootRecord(block_digest, kind, _json_bytes(view)) for view in views)
        block.to_bytes()
        for record in records:
            record.to_bytes()
        return block, records, prepared.roots

    def _read_node_block(self, digest):
        record = self._read_file(self._node_path("definitions/blocks", digest), NodeBlockRecord)
        if record is None or record.digest != digest:
            raise StoreAuthorityError("Missing or mismatched authenticated Node blocks.")
        return record

    def _expand_node_root(self, record):
        from .._content_facts import _current_content_facts
        from ..node_codec import decode_node_graph
        block = self._read_node_block(record.block_digest)
        facts = _current_content_facts()
        if facts is None:
            return self._decode_node_root(record, block)
        return facts.decode(("store-node-root", record.view_bytes, record.root_kind,
                             decode_node_graph, type(self)._validate_node_selections),
                            block.content, lambda: self._decode_node_root(record, block))

    def _decode_node_root(self, record, block):
        """Interpret freshly read immutable bytes; no Store validity is retained."""
        from ..node_codec import decode_node_graph, NodeCodecError
        from ..node_prepare import NodePreparation
        from ..node_views import creation_cdef, object_projection
        from ..node import require_saved
        detached = authority.detached_owner(self)
        cache_key = block.content, record.view_bytes, record.root_kind
        if detached is not None and cache_key in detached.node_views:
            return detached.node_views[cache_key]
        packet = block.blocks
        packet["views"] = [record.view]
        try:
            root = decode_node_graph(packet, expected_view=record.view["view"])
            if record.root_kind in {"construction", "projection"}:
                normalized = (creation_cdef(root) if record.root_kind == "construction" else object_projection(root))
                a, b = NodePreparation((root,)), NodePreparation((normalized,))
                if not a.exact_equal(root, b, normalized):
                    raise ValueError("Noncanonical semantic root policy.")
            if record.root_kind == "saved":
                require_saved(root)
            self._validate_node_selections(root)
            if detached is not None:
                from .._content_facts import _node_block_owner_size
                charge = _node_block_owner_size(block.content, packet)
                if len(detached.node_views) < 16 and detached.node_view_bytes + charge <= 8 * 1024 * 1024:
                    detached.node_views[cache_key] = root
                    detached.node_view_bytes += charge
            return root
        except (NodeCodecError, TypeError, ValueError) as error:
            raise StoreAuthorityError("Malformed or incompatible Node root authority.") from error

    def _check_node_record(self, path, record):
        existing = self._read_file(path, type(record))
        if existing is not None and existing != record:
            raise StoreAuthorityError("Immutable Node metadata collision.")
        return existing

    def install_creations(self, roots):
        """Install concrete roots using one shared metadata table; return digest tuple.

        roots is an iterable of concrete creation views. Full preparation precedes
        the writer fence. Existing headers are compared semantically (including
        collision evidence), never replaced for different block placement. Invalid
        input/authority raises StoreAuthorityError; publication IO errors propagate
        with any already installed metadata intact. No state/membership is implied.
        """
        from ..node_views import CREATION_VIEW, view_equal
        self.preflight_publication("install creation metadata", record_family="dryml-node-graph/v1")
        try:
            block, records, prepared = self._prepare_node_roots(tuple(roots), "construction")
        except (ValueError, TypeError) as error:
            raise StoreAuthorityError("Invalid creation authority.") from error
        block_digest = block.digest
        def prepare():
            missing = {}
            for record, root in zip(records, prepared):
                existing = self.read_creation(record.digest)
                if existing is None:
                    previous = missing.get(record.digest)
                    if previous is not None and not view_equal(previous[1], root, CREATION_VIEW):
                        raise StoreAuthorityError("Creation bucket header collision within batch.")
                    missing.setdefault(record.digest, (record, root))
                elif not view_equal(existing, root, CREATION_VIEW):
                    raise StoreAuthorityError("Creation bucket header collision.")
            changes = []
            if missing:
                changes.append(self._prepared_file(self._node_path("definitions/blocks", block_digest), block))
                for record, _ in missing.values():
                    changes.append(self._prepared_file(os.path.join(self._bucket_path(record.digest), "def.pkl"), record))
            return changes, ("creation",), None
        self._prepare_and_apply(prepare)
        return tuple(record.digest for record in records)

    def install_creation(self, root):
        """Install one concrete creation header, returning its digest; see install_creations."""
        return self.install_creations((root,))[0]

    def _read_creation_record(self, digest):
        record = self._read_file(os.path.join(self._bucket_path(digest), "def.pkl"), NodeRootRecord)
        if record is not None and (record.root_kind != "construction" or record.digest != digest):
            raise StoreAuthorityError("Creation bucket header collision or path mismatch.")
        if record is None and os.path.lexists(self._bucket_path(digest)):
            raise StoreAuthorityError("Missing required creation bucket header.")
        return record

    def read_creation(self, cdef_digest):
        """Read detached creation metadata or None, validating closure and carried digest.

        cdef_digest is a lowercase creation key. Malformed authority raises
        StoreAuthorityError without rewriting it. Reads no state data or Objects;
        metadata survives Store closure, unlike borrowed read sources.
        """
        self._assert_state_owner_open()
        with self._metadata_read():
            record = self._read_creation_record(cdef_digest)
            return None if record is None else self._expand_node_root(record)

    @staticmethod
    def _state_inventory(data_dir):
        files = []
        def failed(error):
            raise error
        for current, dirs, names in os.walk(data_dir, followlinks=False, onerror=failed):
            if current != data_dir and not dirs and not names:
                raise StoreAuthorityError("State data has an empty nested directory.")
            for name in dirs + names:
                mode = os.lstat(os.path.join(current, name)).st_mode
                if not (stat.S_ISDIR(mode) if name in dirs else stat.S_ISREG(mode)):
                    raise StoreAuthorityError("State data contains an unsupported entry.")
            for name in names:
                path = os.path.join(current, name)
                digest = hashlib.sha256()
                with open(path, "rb") as source:
                    for part in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(part)
                files.append((os.path.relpath(path, data_dir).replace(os.sep, "/"),
                              os.path.getsize(path), digest.hexdigest()))
        return tuple(sorted(files))

    def install_state(self, root, staging, *, codec, codec_version, interpretation):
        """Install/reuse one verified full-key variant from owned staging/data.

        Inputs and return follow Store.install_state. Hashes codec version,
        interpretation, inventory and bytes outside the fence. Validates all
        preexisting header/manifest evidence before effects, publishes each variant
        independently with native filesystem durability, and never replaces a
        populated bucket. Missing reclaimed data may be reinstalled with a fresh
        incarnation. Failures preserve historical authority and owned staging.
        staging must belong to the allocating handle/thread. Ordinary publication
        retains a shared full-key lease; withdrawn/partial recovery releases that
        lease and separately tries exclusive recovery without upgrading. No durable
        root is created by this low-level installation receipt. Matching visible
        installations are reused only after Store-owned pending persistence
        recovery; unresolved barriers raise StoreMutationError with their receipt.
        """
        from ..state_values import StateKey, LocalState
        from ..node_views import CREATION_VIEW, view_equal
        self.preflight_publication("install state variant", local_state=True, record_family="dryml-node-graph/v1")
        self._require_staging_owner(staging)
        stage = os.fspath(staging)
        data = os.path.join(stage, "data")
        try:
            entries = set(os.listdir(stage))
            if entries not in ({"data"}, {"data", "manifest.record", "availability.record"}) or not stat.S_ISDIR(os.lstat(data).st_mode):
                raise StoreAuthorityError("State staging must contain private data and only complete Store-prepared metadata.")
            block, (header,), (origin,) = self._prepare_node_roots((root,), "construction")
            files = self._state_inventory(data)
            codec_hash = StateVariantManifest.content_hash(codec, codec_version, interpretation, files)
            key = StateKey(header.digest, codec_hash)
            manifest = StateVariantManifest(key, codec_version, _json_bytes(interpretation), files)
            manifest.validate_payload(data)
            manifest_bytes = manifest.to_bytes()
            incarnation = None
            if entries != {"data"}:
                staged_manifest = self._read_file(os.path.join(stage, "manifest.record"), StateVariantManifest)
                incarnation = self._read_file(os.path.join(stage, "availability.record"), StateAvailabilityRecord)
                if staged_manifest != manifest or incarnation is None or incarnation.key != key or incarnation.status != "published":
                    raise StoreAuthorityError("Retry staging does not match exact prepared state evidence.")
        except (ValueError, TypeError) as error:
            raise StoreAuthorityError("Invalid codec/state/creation evidence.") from error
        target = os.path.dirname(self.state_data_path(key))
        incarnation = incarnation or StateAvailabilityRecord(key, str(uuid4()))
        # Fully stage the complete immutable child outside the writer fence.
        self._write_state_stage_record(Path(stage, "manifest.record"), manifest_bytes)
        self._write_state_stage_record(Path(stage, "availability.record"), incarnation.to_bytes())
        def prepare():
            old_origin = self.read_creation(header.digest)
            if old_origin is not None and not view_equal(old_origin, origin, CREATION_VIEW):
                raise StoreAuthorityError("Creation bucket header collision.")
            old = self._read_variant(key, old_origin) if os.path.lexists(target) else None
            if old is not None:
                if old.manifest != manifest:
                    raise StoreAuthorityError("State variant manifest collision.")
                if old.available:
                    self._authenticate_state_data(old.manifest, self.state_data_path(key))
                    return [], ("state",), StateInstallation(old.state, False, self._state_durability)
            changes = []
            if old_origin is None:
                changes.append(self._prepared_file(self._node_path("definitions/blocks", block.digest), block))
                changes.append(self._prepared_file(os.path.join(self._bucket_path(key.cdef_digest), "def.pkl"), header))
            if old is None:
                expected = authority.variant_evidence(self, self._relative(stage))
                changes.append(PreparedChange(self._relative(target), None, expected,
                                              self._relative(stage), "variant"))
            else:
                # Reclamation leaves immutable manifest/header history. Reinstall
                # only private data, then replace its publication incarnation.
                restored = os.path.join(target, "data")
                if os.path.lexists(restored):
                    self._authenticate_state_data(manifest, restored)
                availability = os.path.join(target, "availability.record")
                first = self._prepared_file(availability, replace(incarnation, status="prepared"), mutable=True)
                changes.append(first)
                if not os.path.lexists(restored):
                    changes.append(PreparedChange(self._relative(restored), None, b"directory",
                                                  self._relative(data), "data"))
                changes.append(PreparedChange(self._relative(availability), first.after,
                                              incarnation.to_bytes()))
            return changes, ("state", "creation"), StateInstallation(LocalState(key, origin), True, self._state_durability)
        # An unavailable key needs a separate exclusive recovery phase. Never
        # upgrade this ordinary publication's shared lease in place.
        class RecoveryRequired(Exception):
            pass
        original_prepare = prepare
        def shared_prepare():
            available = self._read_file(os.path.join(target, "availability.record"), StateAvailabilityRecord)
            if available is not None and (available.status != "published" or not os.path.isdir(self.state_data_path(key))):
                raise RecoveryRequired()
            return original_prepare()
        try:
            with self.state_leases((key,)):
                result = self._prepare_and_apply(shared_prepare)
        except RecoveryRequired:
            with self.state_leases((key,), exclusive=True, blocking=False) as acquired:
                if not acquired:
                    raise StoreAuthorityError("State recovery is in-use; release publication/read leases before retry.")
                state_retention.settle_interrupted_publication(self, key)
                with self._metadata_read():
                    old_origin = self.read_creation(key.cdef_digest)
                    if old_origin is not None and not view_equal(old_origin, origin, CREATION_VIEW):
                        raise StoreAuthorityError("Creation bucket header collision.")
                    old = self._read_variant(key, old_origin)
                    if old is None or old.manifest != manifest:
                        raise StoreAuthorityError("State recovery requires identical manifest/header history.")
                state_retention.detach_withdrawn_data(self, key)
                result = self._prepare_and_apply(original_prepare)
        self.discard_local_state_staging(stage)
        return result

    @property
    def _state_durability(self):
        return self.publication_capabilities.durability

    @staticmethod
    def _write_state_stage_record(path, payload):
        with path.open("wb") as target:
            if target.write(payload) != len(payload):
                raise OSError("State metadata staging write was incomplete.")

    @staticmethod
    def _authenticate_state_data(manifest, data):
        try:
            manifest.validate_payload(data)
        except StoreRecordError as error:
            raise StoreAuthorityError("State private data does not match its codec manifest.") from error

    def _read_variant(self, key, origin):
        from ..state_values import LocalState
        variant = os.path.dirname(self.state_data_path(key))
        if not os.path.lexists(variant):
            return None
        if not stat.S_ISDIR(os.lstat(variant).st_mode):
            raise StoreAuthorityError("State variant is not a real directory.")
        manifest = self._read_file(os.path.join(variant, "manifest.record"), StateVariantManifest)
        if manifest is None or manifest.key != key or origin is None:
            raise StoreAuthorityError("Missing or mismatched state manifest/creation authority.")
        incarnation = self._read_file(os.path.join(variant, "availability.record"), StateAvailabilityRecord)
        if incarnation is None or incarnation.key != key:
            raise StoreAuthorityError("Missing or mismatched state availability incarnation.")
        path = self.state_data_path(key)
        present = os.path.lexists(path)
        if present and not stat.S_ISDIR(os.lstat(path).st_mode):
            raise StoreAuthorityError("Private state data is not a real directory.")
        return StateDescriptor(LocalState(key, origin), manifest,
                               present and incarnation.status == "published", incarnation.incarnation)

    def inspect_states(self, cdef_digest, *, available_only=False):
        """Return bucket history/availability without payload IO or Object construction.

        Interprets the creation header once outside short writer-blocking scopes;
        exact cut/byte witnesses are revalidated before returning. available_only must
        be bool. Malformed/incompatible records raise StoreAuthorityError intact.
        This bounded-to-one-bucket inventory is not the later indexed paging API.
        """
        from ..state_values import StateKey
        self._assert_state_owner_open()
        if type(available_only) is not bool:
            raise TypeError("available_only must be bool.")
        with self._metadata_read() as captured:
            origin = self.read_creation(cdef_digest)
            if origin is None:
                return ()
            result = []
            for name in captured.directory(self._relative(self._bucket_path(cdef_digest))):
                child = Path(self._bucket_path(cdef_digest), name)
                if child.name == "def.pkl":
                    continue
                try:
                    key = StateKey(cdef_digest, child.name)
                except ValueError as error:
                    raise StoreAuthorityError("Malformed state variant address.") from error
                descriptor = self._read_variant(key, origin)
                if descriptor is None:
                    raise StoreAuthorityError("State variant vanished during fenced inspection.")
                if not available_only or descriptor.available:
                    result.append(descriptor)
            return tuple(result)

    @contextmanager
    def read_state(self, key, *, destination=None, codec=None, codec_version=None):
        """Borrow authenticated private data/ in an owner-checked context.

        key must be StateKey. Optional destination/codec/codec_version validate LocalState
        compatibility before payload IO. Raises StoreAuthorityError for missing,
        incompatible or damaged bytes; TypeError for incomplete options. Releases
        metadata fence before yielding. A native shared full-key lease spans
        validation and borrowing; cleanup immediately reports in-use. Context/Store
        closure expires the source. No durable retention root is created.
        """
        self._assert_state_owner_open()
        StateKeyRecord(key)
        options = (destination is not None, codec is not None, codec_version is not None)
        if any(options) and (not all(options) or type(codec_version) is not int or codec_version < 1):
            raise TypeError("destination, codec and positive integer codec_version must be supplied together.")
        with self.state_leases((key,)):
            with self._read_state_source(key, destination, codec, codec_version) as source:
                yield source

    @contextmanager
    def _read_state_source(self, key, destination, codec, codec_version):
        with self._metadata_read():
            origin = self.read_creation(key.cdef_digest)
            descriptor = self._read_variant(key, origin)
            if descriptor is None or not descriptor.available:
                raise StoreAuthorityError("Selected state is unavailable.")
            if destination is not None:
                if descriptor.manifest.codec_version != codec_version:
                    raise StoreAuthorityError("Selected state codec version is incompatible.")
                try:
                    descriptor.state.validate_for(destination, codec=codec)
                except (TypeError, ValueError) as error:
                    raise StoreAuthorityError("Selected state codec/configuration is incompatible.") from error
            self._authenticate_state_data(descriptor.manifest, self.state_data_path(key))
            source = StateReadSource(self, descriptor.state, descriptor.manifest, self.state_data_path(key))
        try:
            yield source
        finally:
            source._active = False

    @staticmethod
    def _anchor_key(object_id):
        from ..reference_values import ObjectId
        if not isinstance(object_id, ObjectId):
            raise TypeError("Declaration anchors require an ObjectId.")
        return _digest("dryml-object-id-address/v1", object_id.to_data())

    def read_declaration_anchor(self, object_id):
        """Read and validate canonical ObjectId projection and source contributions.

        Returns DeclarationAnchorRecord or None for an undeclared identity.
        Established missing anchors/projections/witnesses raise StoreAuthorityError
        instead of proving absence. Reads shared metadata only; no private bytes.
        """
        self._assert_state_owner_open()
        key = self._anchor_key(object_id)
        with self._metadata_read():
            anchor = self._read_file(self._node_path("declarations/anchors", key), DeclarationAnchorRecord)
            required = self._read_file(self._node_path("declarations/required", key), DeclarationRequirementRecord)
            if anchor is None and required is None:
                return None
            if (anchor is None or required is None or anchor.object_id != object_id
                    or required.object_id != object_id or required.projection_digest != anchor.projection.digest):
                raise StoreAuthorityError("Missing or conflicting required declaration anchor.")
            projection = self._expand_node_root(anchor.projection)
            if projection.object_id != object_id:
                raise StoreAuthorityError("Declaration anchor projection names another ObjectId.")
            return anchor

    @staticmethod
    def _publication_identities(root):
        from ..node import _all_nodes
        return tuple(node for node in _all_nodes(root) if node.object_id is not None)

    @staticmethod
    def _validate_node_selections(root):
        from ..node_prepare import NodePreparation
        for node in NodePreparation((root,)).nodes:
            if getattr(node, "state", None) is not None:
                try:
                    node.state.validate_for(node, codec=node.state.codec)
                except (ValueError, TypeError) as error:
                    raise StoreAuthorityError("Invalid selected state creation/compatibility evidence.") from error

    def _prepare_node_publication(self, root, kind):
        from ..node import _as_node_view, require_saved
        from ..node_views import OBJECT_PROJECTION_VIEW, view_equal
        if kind not in {"recipe", "saved"}:
            raise TypeError("Root membership kind must be recipe or saved.")
        root = _as_node_view(root)
        if kind == "saved":
            require_saved(root)
        block, (record,), _ = self._prepare_node_roots((root,), kind)
        nodes = {}
        for node in self._publication_identities(root):
            old = nodes.get(node.object_id)
            if old is not None and not view_equal(old, node, OBJECT_PROJECTION_VIEW):
                raise StoreAuthorityError("Conflicting declarations within publication.")
            nodes[node.object_id] = node
        identities = sorted(nodes, key=self._anchor_key)
        pblock, projections, projected = self._prepare_node_roots(tuple(nodes[i] for i in identities), "projection")
        return block, record, pblock, tuple(zip(identities, projections, projected))

    def publish_node(self, root, *, kind="recipe"):
        """Install shared Node metadata and validated ObjectId anchors, membership last.

        root/kind and errors follow Store.publish_node. Immutable preparation is
        outside the writer fence. All applicable declarations are checked before
        effects, using canonical projection evidence, not exact state-pinned graph
        equality. Journal intent precedes effects; partial IO errors carry confirmed
        references and repair evidence. Enabled SQLite projection and exact
        acknowledgement complete synchronously before return.
        """
        from ..node_views import OBJECT_PROJECTION_VIEW, view_equal
        self.preflight_publication("publish Node metadata", record_family="dryml-node-graph/v1")
        block, record, pblock, projections = self._prepare_node_publication(root, kind)
        source = (kind, record.digest)
        membership = NodeMembershipRecord(kind, record.digest, tuple(i for i, _, _ in projections))
        membership.to_bytes()
        block_digest, pblock_digest = block.digest, pblock.digest
        def prepare():
            existing = self._read_membership(record.digest, kind)
            if existing is not None:
                # Direct reading checks every required anchor even for a no-op.
                self.read_node(record.digest, kind=kind)
            updates = []
            for identity, projection, node in projections:
                old = self.read_declaration_anchor(identity)
                if old is not None:
                    old_node = self._expand_node_root(old.projection)
                    if not view_equal(old_node, node, OBJECT_PROJECTION_VIEW):
                        raise StoreAuthorityError("ObjectId declaration projection conflicts with existing anchor.")
                    projection = old.projection
                sources = tuple(sorted(set((() if old is None else old.sources) + (source,))))
                update = DeclarationAnchorRecord(identity, projection, sources)
                update.to_bytes()
                updates.append(update)
            root_path = self._node_path("graphs" if kind == "saved" else "definitions/roots", record.digest)
            old_record = self._read_file(root_path, NodeRootRecord)
            if old_record is not None:
                a = self._expand_node_root(old_record)
                from ..node_prepare import NodePreparation
                b = NodePreparation((root,))
                if old_record.root_kind != kind or not NodePreparation((a,)).exact_equal(a, b, root):
                    raise StoreAuthorityError("Node root semantic collision.")
            self._check_node_record(self._node_path("definitions/blocks", block_digest), block)
            new_projections = any(anchor.projection.block_digest == pblock_digest for anchor in updates)
            if new_projections:
                self._check_node_record(self._node_path("definitions/blocks", pblock_digest), pblock)
            changes = []
            if old_record is None:
                changes.append(self._prepared_file(self._node_path("definitions/blocks", block_digest), block))
            if new_projections and existing is None:
                changes.append(self._prepared_file(self._node_path("definitions/blocks", pblock_digest), pblock))
            if old_record is None:
                changes.append(self._prepared_file(root_path, record))
            for anchor in updates:
                key = self._anchor_key(anchor.object_id)
                witness = DeclarationRequirementRecord(anchor.object_id, anchor.projection.digest)
                old = self._read_file(self._node_path("declarations/anchors", key), DeclarationAnchorRecord)
                changes.append(self._prepared_file(self._node_path("declarations/anchors", key), anchor, mutable=True))
                changes.append(self._prepared_file(self._node_path("declarations/required", key), witness))
            changes.append(self._prepared_file(self._node_path("membership/" + kind, record.digest), membership))
            return changes, ("root", "membership", "declarations"), old_record or record
        return self._prepare_and_apply(prepare)

    def _read_membership(self, digest, kind):
        if kind not in {"recipe", "saved"}:
            raise TypeError("Root membership kind must be recipe or saved.")
        membership = self._read_file(self._node_path("membership/" + kind, digest), NodeMembershipRecord)
        if membership is not None and (membership.root_kind != kind or membership.root_digest != digest):
            raise StoreAuthorityError("Root membership path mismatch.")
        return membership

    def read_node(self, digest, *, kind="recipe"):
        """Read one published root with complete declaration-source validation.

        Returns detached Node or None when membership is absent. Missing graph,
        malformed closure, incompatible projection or missing required anchors
        raises StoreAuthorityError intact. Reads no payloads or query sidecars.
        """
        from ..node_views import OBJECT_PROJECTION_VIEW, view_equal
        self._assert_state_owner_open()
        with self._metadata_read():
            membership = self._read_membership(digest, kind)
            if membership is None:
                return None
            record = self._read_file(self._node_path("graphs" if kind == "saved" else "definitions/roots", digest), NodeRootRecord)
            if record is None or record.digest != digest or record.root_kind != kind:
                raise StoreAuthorityError("Missing or mismatched published Node root.")
            root = self._expand_node_root(record)
            nodes = self._publication_identities(root)
            if set(membership.identities) != {node.object_id for node in nodes}:
                raise StoreAuthorityError("Membership declaration requirements mismatch.")
            for node in nodes:
                anchor = self.read_declaration_anchor(node.object_id)
                if anchor is None or (kind, digest) not in anchor.sources:
                    raise StoreAuthorityError("Missing required declaration anchor/source.")
                if not view_equal(node, self._expand_node_root(anchor.projection), OBJECT_PROJECTION_VIEW):
                    raise StoreAuthorityError("Published Node declaration conflicts with anchor.")
            return root

    def _name_path(self, name):
        from .records import _alias
        _alias(name)
        return os.path.join(self.base_dir, "names", name + ".record")

    def read_node_name(self, name):
        """Read one format-4 name and validate its explicit root membership.

        name is a bounded path-safe string. Returns NodeNameRecord or None;
        malformed/dangling names raise StoreAuthorityError. Validation is detached
        and cut-revalidated; no payloads, Objects or index are accessed.
        """
        self._assert_state_owner_open()
        with self._metadata_read():
            record = self._read_file(self._name_path(name), NodeNameRecord)
            if record is not None:
                if record.name != name or self.read_node(record.root_digest, kind=record.root_kind) is None:
                    raise StoreAuthorityError("Dangling or mismatched Node name authority.")
            return record

    def compare_and_set_node_name(self, record, *, expected):
        """Journal a bounded name CAS to an already published Node root.

        record and expected are NodeNameRecords (expected may be None for absent).
        Returns the installed record. Mismatched CAS raises StoreAliasConflictError
        before effects; malformed/dangling authority raises StoreAuthorityError.
        An equal current name is an authoritative no-op. IO failures retain intent,
        actual cut and confirmed publication evidence, without rollback.
        """
        from .store import StoreAliasConflictError
        if not isinstance(record, NodeNameRecord) or (expected is not None and
                (not isinstance(expected, NodeNameRecord) or expected.name != record.name)):
            raise TypeError("Name CAS requires matching NodeNameRecords or expected=None.")
        self.preflight_publication("name Node root", record_family="dryml-node-graph/v1")
        def prepare():
            if self.read_node_name(record.name) != expected:
                raise StoreAliasConflictError("Node name CAS did not match current authority.")
            if self.read_node(record.root_digest, kind=record.root_kind) is None:
                raise StoreAuthorityError("Node name requires published root membership.")
            return [self._prepared_file(self._name_path(record.name), record, mutable=True)], ("names",), record
        return self._prepare_and_apply(prepare)

    def delete_node_name(self, name, *, expected):
        """Journal removal of one exact current name; return whether it existed.

        name is path-safe; expected is its NodeNameRecord or None. CAS mismatch
        raises StoreAliasConflictError. Only the name is removed; graph, state and
        declaration history remain intact. IO errors preserve partial evidence.
        """
        from .store import StoreAliasConflictError
        self._name_path(name)
        if expected is not None and (not isinstance(expected, NodeNameRecord) or expected.name != name):
            raise TypeError("Name deletion requires its expected NodeNameRecord.")
        self.preflight_publication("delete Node name", record_family="dryml-node-graph/v1")
        def prepare():
            old = self.read_node_name(name)
            if old != expected:
                raise StoreAliasConflictError("Node name deletion CAS did not match.")
            changes = [] if old is None else [self._prepared_file(self._name_path(name), None, mutable=True)]
            return changes, ("names",), old is not None
        return self._prepare_and_apply(prepare)

    def rename_node_name(self, name, new_name, *, expected, expected_target=None):
        """Journal exact source/target name CAS without deleting graph authority.

        Names are path-safe strings; expected is the source NodeNameRecord and
        expected_target is the destination record or None. Returns the new record.
        Mismatches raise StoreAliasConflictError before effects. Interrupted rename
        may leave both complete names; the journal classifies that honest cut.
        """
        from .store import StoreAliasConflictError
        self._name_path(name)
        self._name_path(new_name)
        if not isinstance(expected, NodeNameRecord) or expected.name != name:
            raise TypeError("Rename requires the exact expected source name.")
        if expected_target is not None and (not isinstance(expected_target, NodeNameRecord)
                                            or expected_target.name != new_name):
            raise TypeError("Rename requires a matching expected target name.")
        self.preflight_publication("rename Node name", record_family="dryml-node-graph/v1")
        renamed = replace(expected, name=new_name)
        def prepare():
            if self.read_node_name(name) != expected:
                raise StoreAliasConflictError("Rename source CAS did not match.")
            if name == new_name:
                return [], ("names",), expected
            if self.read_node_name(new_name) != expected_target:
                raise StoreAliasConflictError("Rename target CAS did not match.")
            return [self._prepared_file(self._name_path(new_name), renamed, mutable=True),
                    self._prepared_file(self._name_path(name), None, mutable=True)], ("names",), renamed
        return self._prepare_and_apply(prepare)

    def withdraw_node(self, digest, *, kind="recipe"):
        """Withdraw explicit membership and only its declaration contributions.

        digest/kind identify a published root. Returns True if removed, False for
        absent or still-named roots. Immutable graphs, declaration requirements,
        canonical anchors (including zero-source history), and all state data are
        preserved. No retention/reclamation is performed. Corrupt authority raises
        StoreAuthorityError; failed IO carries honest partial mutation evidence.
        """
        self.preflight_publication("withdraw Node membership", record_family="dryml-node-graph/v1")
        self._read_membership(digest, kind)
        def prepare():
            membership = self._read_membership(digest, kind)
            if membership is None:
                return [], ("membership", "declarations"), False
            self.read_node(digest, kind=kind)
            with self._metadata_read() as captured:
                for name in captured.directory("names"):
                    if name.startswith(".dryml-removed-") and not name.endswith(".record"):
                        continue
                    if not name.endswith(".record"):
                        raise StoreAuthorityError("Malformed Node name inventory.")
                    record = self.read_node_name(name[:-7])
                    if (record.root_kind, record.root_digest) == (kind, digest):
                        return [], ("membership", "declarations"), False
            changes = []
            for identity in membership.identities:
                anchor = self.read_declaration_anchor(identity)
                sources = tuple(s for s in anchor.sources if s != (kind, digest))
                cls = DeclarationAnchorRecord if sources else RetiredDeclarationAnchorRecord
                updated = cls(anchor.object_id, anchor.projection, sources)
                changes.append(self._prepared_file(self._node_path("declarations/anchors", self._anchor_key(identity)),
                                                   updated, mutable=True))
            changes.append(self._prepared_file(self._node_path("membership/" + kind, digest), None, mutable=True))
            return changes, ("membership", "declarations"), True
        return self._prepare_and_apply(prepare)

    @staticmethod
    def _filesystem(operation, *args, **kwargs):
        """Translate only missing filesystem guarantees into Store policy errors."""

        try:
            return operation(*args, **kwargs)
        except filesystem.FilesystemCapabilityError as error:
            raise StoreCapabilityError(str(error)) from error

    @classmethod
    def open_existing(
            cls,
            base_dir: str | os.PathLike[str],
            *,
            query_index: QueryIndexPolicy | SQLiteQueryIndexConfig = "auto") -> "DirStore":
        """Open validated current authority without creating or repairing it.

        Args:
            base_dir: Existing direct Store root.
            query_index: Derived-index policy for the fresh handle.

        Returns:
            A matching Session-cached Store handle when resource caching is active,
            otherwise a new caller-owned handle over current-format authority.

        Raises:
            StoreAuthorityError: If the root or required format record is absent,
                malformed, or not the required filesystem type.
        """

        path = os.path.abspath(os.fspath(base_dir))
        if cls is not DirStore or not isinstance(query_index, str):
            # Custom SQLite configuration has no portable descriptor grammar and
            # retains the established fresh-handle behavior.
            cls._validate_existing_root(path)
            return cls(path, query_index=query_index, _existing_only=True)

        from ..repo_definition import _open_store_descriptor

        return _open_store_descriptor({
            "kind": "dir",
            "path": path,
            "query_index": query_index,
        })

    @property
    def base_dir(self) -> str:
        """Return the direct-layout Store root directory."""
        return self._base_dir

    @property
    def query_index_policy(self) -> QueryIndexPolicy:
        """Return the configured derived-index policy without making it authority."""
        return self._query_index_policy

    @property
    def mutation_index_disposition(self):
        """Return the explicit current publication/index policy (str), without IO.

        none/memory have no persistent maintenance obligation. auto uses SQLite
        when available, otherwise declares optional-backend fallback. Explicit
        sqlite never falls back. This policy is not itself readiness evidence.
        """
        if self._query_index_policy in {"none", "memory"}:
            return {"none": "disabled", "memory": "memory"}[self._query_index_policy]
        from ..query.sqlite import sqlite_available
        return ("nonpersistent:auto-backend-unavailable" if self._query_index_policy == "auto"
                and (self._query_index_unavailable or not sqlite_available()) else "persistent:sqlite-synchronous")

    @property
    def dryml_dir(self) -> str:
        """Return the derived-sidecar directory, which is never Store authority."""
        return os.path.join(self.base_dir, ".dryml")

    @property
    def query_index_path(self) -> str:
        """Return the canonical SQLite sidecar path without creating it."""
        return os.path.join(self.dryml_dir, "query-index-v1.sqlite")

    @property
    def query_index_dirty_path(self) -> str:
        """Return the prefix used for durable derived-index dirty markers."""
        return os.path.join(self.dryml_dir, "query-index.dirty")

    def open_query_index(self):
        """Bind one persistent destination per authority domain; return its backend.

        none/memory return None. auto may declare nonpersistent fallback only when
        optional SQLite is unavailable; explicit sqlite propagates that failure.
        Destination/configuration conflicts raise QueryIndexIncompatible before
        mutation. Binding persists derived configuration under the Store fence;
        it never initializes an empty ready index over nonempty authority.
        """
        self._assert_state_owner_open()
        if self._query_index_instance is not None:
            try:
                self._query_index_instance.preflight_store()
            except QueryIndexUnavailable:
                if self._query_index_policy != "auto":
                    raise
                self._query_index_unavailable = True
                return None
            self._query_index_unavailable = False
            return self._query_index_instance
        if self._query_index_policy in {"memory", "none"}:
            return None
        from ..query.sqlite import require_sqlite
        try:
            require_sqlite()
        except QueryIndexUnavailable:
            if self._query_index_policy == "auto":
                self._query_index_unavailable = True
                return None
            raise
        self._query_index_unavailable = False
        config = self._query_index_config or SQLiteQueryIndexConfig()
        path = local_path_key(config.path or self.query_index_path)
        config = replace(config, path=path)
        index = SQLiteStoreQueryIndex(source_key=self.catalog_key(), path=path, config=config,
                                      store=self, dirty_path=self.query_index_dirty_path)
        index.preflight_store()
        self._query_index_instance = index
        return index

    def _query_index_binding(self, index):
        config = index.config
        return _json_bytes({"kind": "dryml-store-index-destination", "version": 1,
                              "store_uuid": self.store_uuid, "path": local_path_key(index.path),
                              "journal_mode": config.journal_mode, "durability": config.durability})

    def _bind_query_index(self, index):
        from ..query.model import QueryIndexIncompatible
        binding = self._query_index_binding(index)
        relative = ".dryml/query-index-destination.record"
        # The destination is write-once. Readers of an existing atomic record
        # need no writer lock, including refresh waiters joining a fenced owner.
        previous = authority.raw(self, relative)
        if previous == binding:
            return
        if previous is not None:
            raise QueryIndexIncompatible("Conflicting persistent index destination/configuration for this Store authority domain.")
        with self.writer_lock():
            previous = authority.raw(self, relative)
            if previous is not None and previous != binding:
                raise QueryIndexIncompatible("Conflicting persistent index destination/configuration for this Store authority domain.")
            if previous is None:
                self._atomic_write(os.path.join(self.base_dir, relative), binding)

    def _query_index_dirty_markers(self) -> tuple[str, ...]:
        """Return all current dirty tokens without interpreting them as authority."""
        root = Path(self.dryml_dir)
        if not root.exists():
            return ()
        markers = tuple(os.fspath(path) for path in root.glob("query-index.dirty.*"))
        if os.path.exists(self.query_index_dirty_path):
            markers = (*markers, self.query_index_dirty_path)
        return markers

    def clear_query_index_dirty(self) -> None:
        """Durably remove every current derived query-index dirty marker.

        Returns:
            ``None`` after all markers visible at scan time are logically absent.

        Raises:
            OSError: If durable removal of any marker fails.
            StoreCapabilityError: If the platform cannot provide the required
                publication guarantee.

        Side Effects:
            Removes each derived token independently without modifying
            Node/state authority. Concurrently added tokens remain dirty.
        """
        for marker in self._query_index_dirty_markers():
            self._filesystem(
                filesystem.remove_file,
                marker,
                cleanup_directory=os.path.dirname(marker) or ".",
                missing_ok=True,
            )

    def query_index_is_dirty(self) -> bool:
        """Return whether a published definition may not be represented by SQLite."""
        return bool(self._query_index_dirty_markers())

    def query_index_status(self) -> QueryIndexStatus:
        """Return the configured derived-index status without reading authority."""
        if self._query_index_policy == "none":
            return QueryIndexStatus("none", self.catalog_key(), None, None, {}, "disabled")
        if self._query_index_policy == "memory":
            return QueryIndexStatus("memory", self.catalog_key(), None, None, {}, "disabled")
        index = self.open_query_index()
        if index is None:
            return QueryIndexStatus("memory", self.catalog_key(), None, None, {}, "ready")
        return index.status()

    def _open_rebuildable_query_index(self) -> SQLiteStoreQueryIndex:
        index = self.open_query_index()
        if index is None:
            raise QueryIndexUnavailable(
                f"DirStore query_index={self._query_index_policy!r} does not provide a rebuildable persistent index."
            )
        return index

    def rebuild_query_index(self) -> ReconcileReport:
        """Request unconditional replacement of the configured SQLite sidecar.

        Scans validated Store authority and uses the backend's default forced
        build claim, unlike refresh(True)/refresh("always"), which may only
        validate a correct ready sidecar. Replacement is staged and validated;
        Store records are never rewritten.

        Returns:
            ReconcileReport with the rebuild action and observed generations.
            The backend may decline a build claim; this report alone is not
            strict reconciliation evidence. Use backend ``refresh(True)`` when
            successful maintenance must be proven.

        Raises:
            QueryIndexUnavailable: If no persistent backend is available.
            QueryIndexError: If authority capture, compatibility, contention,
                staging, activation, or captured-token acknowledgement fails.
                Failure after visible activation does not imply rollback.

        Side Effects:
            An owned rebuild replaces derived index files and acknowledges only
            captured covered dirty tokens. Later tokens remain for recovery.
        """
        index = self._open_rebuildable_query_index()
        before = index.status()
        index.rebuild()
        after = index.status()
        return ReconcileReport(
            backend=after.backend,
            store_key=after.store_key,
            changed=True,
            action="rebuild",
            generation_before=before.generation,
            generation_after=after.generation,
            definitions_scanned=(after.row_counts or {}).get("stored_roots", 0),
            validated=True,
        )

    def reconcile_query_index(self) -> ReconcileReport:
        """Report one reconciliation of the configured persistent query index.

        A correct ready sidecar is validated against graph-exact Store authority
        without replacement; missing, divergent, or recoverable damaged derived
        state is rebuilt. This method retains no cross-call completion state.

        Returns:
            ReconcileReport with historical issues and actual maintenance phases.
            Ordinary backend failures become error issues. ``validated`` records
            validation work, not success; ``changed`` records visible activation,
            even when acknowledgement fails. Neither implies a successful cycle.

        Raises:
            QueryIndexUnavailable: If no persistent backend can be opened.
            QueryIndexError: If opening the configured backend fails.
            BaseException: Cancellation/interruption propagates rather than
                returning a successful report.

        Side Effects:
            May replace derived files and acknowledge captured covered dirty
            tokens. Store authority and later/unrelated tokens are preserved.
        """
        return self._open_rebuildable_query_index().reconcile()

    def validate_query_index(self, *, thorough: bool = False):
        """Validate the configured index while leaving Node/state authority intact."""
        index = self.open_query_index()
        if index is None:
            return super().validate_query_index(thorough=thorough)
        return index.validate(thorough=thorough)

    @property
    def publication_capabilities(self) -> StorePublicationCapabilities:
        """Return local-filesystem publication guarantees after format validation."""
        return StorePublicationCapabilities(
            True, True, True, supports_advisory_locking(self._writer_lock_path), True, "durable"
        )

    @property
    def _writer_lock_path(self) -> str:
        return os.path.join(self.base_dir, ".writer.lock")

    def writer_lock(self):
        """Serialize a multi-record Store authority transition.

        Only bounded byte/association checks, prepared installs and authority cuts
        belong inside. Prepare graphs, invoke hooks/providers and authenticate
        payloads before entry. StateKey protection must precede this fence;
        callers must not wrap high-level graph APIs in an externally held fence.
        """
        @contextmanager
        def fenced():
            with interprocess_lock(self._writer_lock_path):
                with state_retention.writer_scope(self):
                    yield
        return fenced()

    @property
    def store_format_path(self) -> str:
        """Return the only Store-wide authority format record path."""
        return os.path.join(self.base_dir, "store-format.record")

    @property
    def _bootstrap_lock_path(self) -> str:
        """Return the derived sibling lock for this canonical Store root.

        The durable lock file is named from the SHA-256 digest of the ``os.fsencode``
        bytes of the normalized real root path and is never authority or removed
        by a Store handle.
        Keeping it outside the root lets bootstrap reject arbitrary nonempty
        roots without first adding a Store-owned entry to them.
        """
        root = local_path_key(self.base_dir)
        identity = hashlib.sha256(os.fsencode(root)).hexdigest()
        return os.path.join(os.path.dirname(root), f".dryml-bootstrap-{identity}.lock")

    @classmethod
    def _validate_existing_root(cls, base_dir: str) -> None:
        """Validate existing root/type evidence without creating Store state."""

        try:
            root_mode = os.lstat(base_dir).st_mode
        except OSError as error:
            raise StoreAuthorityError("DirStore root is missing or inaccessible.") from error
        if not stat.S_ISDIR(root_mode):
            raise StoreAuthorityError("DirStore root is not a directory.")
        format_path = os.path.join(base_dir, "store-format.record")
        try:
            format_mode = os.lstat(format_path).st_mode
        except OSError as error:
            raise StoreAuthorityError("DirStore lacks current store-format.record.") from error
        if not stat.S_ISREG(format_mode):
            raise StoreAuthorityError("DirStore format record is not a regular file.")
        try:
            with open(format_path, "rb") as source:
                StoreFormatRecord.from_bytes(source.read(_MAX_RECORD_BYTES + 1))
        except (OSError, StoreRecordError) as error:
            raise StoreAuthorityError("DirStore format record is malformed or inaccessible.") from error

    def _initialize_format(self, *, existing_only: bool = False) -> None:
        """Validate or durably initialize this direct Store's format gate.

        Args:
            existing_only: Require existing current authority without creating it.

        Raises:
            StoreAuthorityError: If existing authority is absent, malformed, or
                mixed with an unmarked nonempty root.
            OSError: If bootstrap locking, directory creation, or gate publication
                fails.

        Side Effects:
            May durably create missing root components and ``store-format.record``
            while holding the derived bootstrap lock.
        """

        root = Path(self.base_dir)
        if existing_only:
            self._validate_existing_root(self.base_dir)
            return
        if root.exists() and not root.is_dir():
            raise StoreAuthorityError(f"DirStore root is not a directory: {self.base_dir!r}.")
        if os.path.lexists(self.store_format_path):
            self._read_file(self.store_format_path, StoreFormatRecord)
            return
        if not root.exists():
            self._filesystem(
                filesystem.ensure_directory, os.path.dirname(self.base_dir),
            )
        # This lease covers root creation through final marker replacement.  It
        # remains outside authority so an unmarked root can still be rejected
        # without treating a Store-created lock as permission to overwrite it.
        with interprocess_lock(self._bootstrap_lock_path):
            if root.exists() and not root.is_dir():
                raise StoreAuthorityError(f"DirStore root is not a directory: {self.base_dir!r}.")
            if not root.exists():
                self._filesystem(filesystem.ensure_directory, self.base_dir)
            if os.path.lexists(self.store_format_path):
                self._read_file(self.store_format_path, StoreFormatRecord)
                return
            # A non-empty root without the current gate is retired/incompatible;
            # do not create a new format marker over unknown authority.
            entries = {entry.name for entry in root.iterdir()}
            if entries == {"authority.record"}:
                initial = self._read_file(os.path.join(self.base_dir, "authority.record"), AuthorityRecord)
                if initial.epoch != 0 or initial.pending:
                    raise StoreAuthorityError("Incomplete Store initialization has noninitial authority.")
                self._atomic_write(self.store_format_path, StoreFormatRecord(store_uuid=initial.store_uuid).to_bytes())
                return
            if entries:
                raise StoreAuthorityError("Store lacks current store-format.record; old or mixed authority is unsupported.")
            format_record = StoreFormatRecord()
            self._atomic_write(os.path.join(self.base_dir, "authority.record"),
                               AuthorityRecord(format_record.store_uuid).to_bytes())
            self._atomic_write(self.store_format_path, format_record.to_bytes())

    def _read_file(self, path: str, record_type):
        try:
            mode = os.lstat(path).st_mode
        except FileNotFoundError:
            authority.observe(self, path, None)
            return None
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise StoreAuthorityError(f"Store record is not a regular file: {path!r}.")
        try:
            with open(path, "rb") as source:
                payload = source.read(getattr(record_type, "max_bytes", _MAX_RECORD_BYTES) + 1)
            authority.observe(self, path, payload)
            if record_type is DeclarationAnchorRecord and payload.startswith(
                    b"DRYML-METADATA/dryml-declaration-retired-anchor/1\n"):
                record_type = RetiredDeclarationAnchorRecord
            from .._content_facts import _current_content_facts
            facts = _current_content_facts()
            reader = record_type.from_bytes
            if facts is not None and record_type in (NodeBlockRecord, NodeRootRecord, NodeMetadataRecord):
                return facts.decode((record_type, getattr(reader, "__func__", reader), record_type.max_bytes),
                                    payload, lambda: reader(payload))
            return reader(payload)
        except StoreRecordError as error:
            raise StoreAuthorityError(f"Malformed Store record {path!r}: {error}") from error

    def _atomic_write(self, path: str, payload: bytes) -> None:
        """Flush and atomically publish complete bytes at one Store path.

        Args:
            path: Final direct Store path.
            payload: Complete record bytes to publish.

        Raises:
            OSError: If writing, flushing, or durable replacement fails.
            StoreCapabilityError: If the platform cannot provide the required
                publication guarantee.

        Side Effects:
            Creates durable parent directories, publishes one sibling temporary,
            and removes an unpublished temporary after failure when possible.
        """

        parent = os.path.dirname(path)
        self._filesystem(filesystem.ensure_directory, parent)
        fd, temporary = tempfile.mkstemp(prefix=".store-", dir=parent)
        try:
            with os.fdopen(fd, "wb") as target:
                written = target.write(payload)
                if written != len(payload):
                    raise OSError("Store authority temporary write was incomplete.")
            self._filesystem(
                filesystem.publish_file, temporary, path, replace=True,
            )
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _digest_path(directory: str, digest: str) -> str:
        return os.path.join(directory, digest[:2], f"{digest}.record")

    @property
    def _staging_root(self) -> str:
        return os.path.join(self.base_dir, ".staging")

    def _local_state_staging_id(self, handle: object) -> str:
        """Return the opaque ID of one existing direct local-state allocation.

        The helper validates Store containment and returns no path information;
        callers retain all cache-specific marker and retention policy.  It does
        not allocate, delete, or otherwise transfer ownership of ``handle``.
        """

        try:
            path = os.path.abspath(os.fspath(handle))
        except TypeError as error:
            raise StoreAuthorityError("local-state staging handle is invalid.") from error
        token = os.path.basename(path)
        if (os.path.dirname(path) != self._staging_root or len(token) != 32
                or any(char not in "0123456789abcdef" for char in token)):
            raise StoreAuthorityError("local-state staging handle is foreign or malformed.")
        try:
            mode = os.lstat(path).st_mode
        except FileNotFoundError as error:
            raise StoreAuthorityError("local-state staging handle is missing.") from error
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise StoreAuthorityError("local-state staging handle is not a real directory.")
        return token

    def _resolve_local_state_staging_id(self, work_id: object) -> str | None:
        """Resolve one opaque local-state ID without allocating or deleting state.

        ``None`` denotes a missing direct child.  Malformed IDs and non-directory
        entries are rejected rather than silently treated as a resumable prefix.
        """

        if type(work_id) is not str or len(work_id) != 32 or any(
                char not in "0123456789abcdef" for char in work_id):
            raise StoreAuthorityError("local-state staging ID is malformed.")
        path = os.path.join(self._staging_root, work_id)
        try:
            mode = os.lstat(path).st_mode
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise StoreAuthorityError("local-state staging ID is not a real directory.")
        return path

    def create_local_state_staging(self) -> str:
        """Allocate staging/data with an exact owner marker and lifetime native lease.

        Returns a path (str). The allocating handle owns it through successful
        publication/discard or explicit lease release/close. Cooperating recovery
        cannot remove live work. Capability/IO errors propagate before hooks.
        """
        self.preflight_publication("create local-state staging", local_state=True, record_family="dryml-node-graph/v1")
        allocation = uuid4().hex
        path = os.path.join(self._staging_root, allocation)
        lease = FileLock(os.path.join(self._staging_root, ".leases", allocation + ".lock"))
        lease.acquire()
        self._staging_leases[allocation] = lease
        self._staging_threads[allocation] = (os.getpid(), threading.get_ident())
        try:
            os.makedirs(os.path.join(path, "data"))
            self._atomic_write(path + ".owner", self._staging_owner_uuid.encode("ascii"))
        except BaseException:
            self._staging_leases.pop(allocation).release()
            self._staging_threads.pop(allocation)
            raise
        return path

    def _staging_owner(self, allocation):
        from .records import _uuid_value
        self._resolve_local_state_staging_id(allocation)
        payload = authority.raw(self, ".staging/" + allocation + ".owner")
        if payload is None:
            raise StoreAuthorityError("Staging owner evidence is missing.")
        try:
            return _uuid_value(payload.decode("ascii"))
        except (UnicodeError, ValueError) as error:
            raise StoreAuthorityError("Malformed staging owner evidence.") from error

    def _require_staging_owner(self, handle):
        allocation = self._local_state_staging_id(handle)
        if (allocation not in self._staging_leases
                or self._staging_threads.get(allocation) != (os.getpid(), threading.get_ident())
                or self._staging_owner(allocation) != self._staging_owner_uuid):
            raise StoreAuthorityError("Staging must belong to this live allocating handle/thread.")
        return allocation

    @contextmanager
    def _replay_staging(self, changes):
        with ExitStack() as stack:
            allocations = set()
            for change in changes:
                relative = change["staging"]
                if relative is None or not os.path.isdir(os.path.join(self.base_dir, relative)):
                    continue
                parts = relative.split("/")
                if len(parts) not in {2, 3} or parts[0] != ".staging" or (len(parts) == 3 and parts[2] != "data"):
                    raise StoreAuthorityError("Replay staging must name an exact owned allocation.")
                allocations.add(parts[1])
            for allocation in sorted(allocations):
                owner = self._staging_owner(allocation)
                if allocation in self._staging_leases:
                    self._require_staging_owner(os.path.join(self._staging_root, allocation))
                    continue
                lease = FileLock(os.path.join(self._staging_root, ".leases", allocation + ".lock"))
                if not lease.acquire(blocking=False):
                    raise StoreAuthorityError("Replay staging is in-use by its live allocator.")
                stack.callback(lease.release)
                if self._staging_owner(allocation) != owner:
                    raise StoreAuthorityError("Replay staging owner changed; preserve it.")
            yield

    def release_local_state_staging(self, handle):
        """Release this handle's allocation lease while preserving exact recovery work.

        handle is its staging path. Returns None; foreign owners raise
        StoreAuthorityError. Release is explicit handoff, never abandonment proof.
        The resource ledger must already name work needed after interruption.
        """
        allocation = os.path.basename(os.fspath(handle))
        if os.path.abspath(os.fspath(handle)) != os.path.join(self._staging_root, allocation):
            raise StoreAuthorityError("Staging handle is foreign.")
        lease = self._staging_leases.get(allocation)
        if lease is None or self._staging_threads.get(allocation) != (os.getpid(), threading.get_ident()):
            raise StoreAuthorityError("Staging lease belongs to another owner.")
        lease.release()
        self._staging_leases.pop(allocation)
        self._staging_threads.pop(allocation)

    def _discard_abandoned_staging(self, allocation, owner):
        path = self._resolve_local_state_staging_id(allocation)
        marker = os.path.join(self._staging_root, allocation) + ".owner"
        if not os.path.exists(marker):
            if path is not None:
                raise StoreAuthorityError("Staging has no owner evidence; preserve it.")
            return True
        if self._staging_owner(allocation) != owner:
            raise StoreAuthorityError("Staging owner changed; preserve another allocation.")
        lease = FileLock(os.path.join(self._staging_root, ".leases", allocation + ".lock"))
        if not lease.acquire(blocking=False):
            return False
        try:
            if self._staging_owner(allocation) != owner:
                raise StoreAuthorityError("Staging owner changed before cleanup.")
            if path is not None:
                shutil.rmtree(path)
            self._filesystem(filesystem.remove_file, marker, cleanup_directory=self._staging_root, missing_ok=True)
            return True
        finally:
            lease.release()

    @staticmethod
    def _node_metadata_address(target, scope=None):
        from ..node import Node, require_saved
        from ..definition import ConcreteDefinition
        if not isinstance(target, Node) or not isinstance(target, ConcreteDefinition):
            raise TypeError("Format-4 metadata requires a concrete Node.")
        if scope is None:
            scope = "current_state" if target.state_spec() else "current_object"
        if scope in {"current_state", "capture"}:
            require_saved(target)
            digest = hashlib.sha256(b"dryml-saved-root/v1\0" + target.exact_key()).hexdigest()
        elif scope in {"current_object", "lineage"}:
            digest = target.object_projection_digest()
        else:
            raise ValueError("Unsupported Node metadata scope.")
        return scope, digest

    def _node_metadata_path(self, scope, digest):
        if scope not in {"current_object", "current_state", "lineage", "capture"}:
            raise ValueError("Unsupported Node metadata scope.")
        return self._node_path("metadata/" + scope, digest)

    @staticmethod
    def _validate_node_metadata_record(record, scope, digest):
        if record.scope != scope or record.target_digest != digest:
            raise StoreAuthorityError("Node metadata target/scope does not match authority address.")

    def _holds_node_metadata(self, target, scope=None):
        scope, digest = self._node_metadata_address(target, scope)
        if scope in {"current_state", "capture"}:
            node = self.read_node(digest, kind="saved")
            if node is None:
                return False
            if not node.graph_equal(target):
                raise StoreAuthorityError("Saved metadata target has incompatible exact authority.")
            return True
        if target.object_id is None:
            return self.query().object_refs().sel(target.object_view()).exists()
        anchor = self.read_declaration_anchor(target.object_id)
        if anchor is None:
            return False
        if not target.object_projection_equal(self._expand_node_root(anchor.projection)):
            raise StoreAuthorityError("Object metadata target conflicts with declaration authority.")
        return True

    def _read_node_metadata(self, target, scope=None):
        from .records import NodeMetadataRecord
        scope, digest = self._node_metadata_address(target, scope)
        with self._metadata_read():
            if not self._holds_node_metadata(target, scope):
                raise KeyError(digest)
            record = self._read_file(self._node_metadata_path(scope, digest), NodeMetadataRecord)
            if record is not None:
                self._validate_node_metadata_record(record, scope, digest)
                if scope == "lineage":
                    import json
                    from ..reference_values import ObjectId
                    if ObjectId.from_data(json.loads(record.content)["identity"]) != target.object_id:
                        raise StoreAuthorityError("Lineage metadata names another ObjectId.")
            return record

    def _read_node_capture(self, target):
        import json
        from ..metadata import _node_snapshot_metadata
        record = self._read_node_metadata(target, "capture")
        if record is None:
            raise StoreAuthorityError("Published saved graph lacks captured metadata authority.")
        return _node_snapshot_metadata(target, json.loads(record.content))

    def _read_node_capture_for_selection(self, target, prepared):
        """Read capture after an owning active-source destination preflight.

        The private caller has validated every selected StateKey/destination.
        Decode fresh captured block bytes, compare exact graph evidence with that
        prepared target, and check every declaration contribution. This reuses the
        destination validation instead of repeating it for an equal saved graph.
        The detached owner/final cut witness remains operation-local. Custom root
        readers retain their public validation path.
        """
        import json
        from ..node_codec import _decode_node_graph_views
        from ..metadata import _node_snapshot_metadata
        from .records import NodeMetadataRecord
        if (getattr(self.read_node, "__func__", None) is not DirStore.read_node
                or getattr(self._expand_node_root, "__func__", None) is not DirStore._expand_node_root):
            if not self._holds_node_metadata(target, "capture"):
                return None
            record = self._read_node_metadata(target, "capture")
            return None if record is None else (record, self._read_node_capture(target))
        digest = prepared.digest(target, domain="dryml-saved-root/v1")
        with self._metadata_read():
            member = self._read_membership(digest, "saved")
            if member is None:
                return None
            record = self._read_file(self._node_path("graphs", digest), NodeRootRecord)
            if record is None or record.root_kind != "saved" or record.digest != digest:
                raise StoreAuthorityError("Missing or mismatched selected saved-graph authority.")
            block = self._read_node_block(record.block_digest)
            packet = block.blocks
            packet["views"] = [record.view]
            (root,), owner = _decode_node_graph_views(packet, with_owner=True)
            if not owner.exact_equal(root, prepared, target):
                raise StoreAuthorityError("Selected saved graph conflicts with captured authority.")
            nodes = self._publication_identities(root)
            if set(member.identities) != {node.object_id for node in nodes}:
                raise StoreAuthorityError("Selected membership declaration requirements mismatch.")
            for node in nodes:
                anchor = self.read_declaration_anchor(node.object_id)
                if anchor is None or ("saved", digest) not in anchor.sources:
                    raise StoreAuthorityError("Missing selected declaration anchor/source.")
                if not node.object_projection_equal(self._expand_node_root(anchor.projection)):
                    raise StoreAuthorityError("Selected declaration conflicts with its anchor.")
            capture = self._read_file(self._node_metadata_path("capture", digest), NodeMetadataRecord)
            if capture is None:
                raise StoreAuthorityError("Published saved graph lacks captured metadata authority.")
            self._validate_node_metadata_record(capture, "capture", digest)
            return capture, _node_snapshot_metadata(target, json.loads(capture.content))

    def read_metadata(self, target, *, scope=None):
        """Read a fresh detached current mapping for a concrete Node target.

        scope is object/state, or None to infer state when selections exist and
        object otherwise. Stateless saved graphs need explicit scope='state' to
        distinguish that question from an equal object projection. Returns dict
        or None (absent). Unknown targets raise KeyError; invalid scope/authority
        raises ValueError/StoreAuthorityError. Old exact-reference inputs raise
        StoreCapabilityError. Reads metadata only, under a revalidated cut, without
        payload IO, environment observation, construction or mutation.
        """
        from ..node import Node
        if not isinstance(target, Node):
            raise StoreCapabilityError("Legacy reference metadata is not format-4 authority.")
        import json
        from ..metadata import decode_metadata_mapping, _node_annotation_scope
        area = _node_annotation_scope(scope)
        record = self._read_node_metadata(target, area)
        return None if record is None else decode_metadata_mapping(json.loads(record.content)["values"])

    def write_metadata(self, target, values, *, scope=None) -> None:
        """Journal one whole current annotation replacement and index its cut.

        Args:
            target: Concrete Node object projection or saved graph.
            values: Valid complete metadata mapping to install.
            scope: object/state or None, with the inference in read_metadata.

        Returns:
            ``None``.

        Raises:
            TypeError: If inputs are unsupported.
            ValueError: If metadata values violate codec bounds.
            StoreCapabilityError: For legacy reference targets or unavailable
                filesystem publication guarantees.

        Side Effects:
            Validates fresh target authority in detached preparation, publishes
            intent before effects, and synchronously maintains enabled indexing.
            Captured metadata and payloads are not changed. Partial publication
            failures retain the Store mutation's confirmed cut/repair evidence.
        """

        from ..node import Node
        if isinstance(target, Node):
            from .records import NodeMetadataRecord
            from ..metadata import encode_metadata_mapping, _node_annotation_scope
            scope, digest = self._node_metadata_address(target, _node_annotation_scope(scope))
            record = NodeMetadataRecord(scope, digest, _json_bytes({"values": encode_metadata_mapping(values)}))
            self.preflight_publication("replace Node annotations", record_family="dryml-node-graph/v1")
            def prepare():
                if not self._holds_node_metadata(target, scope):
                    raise KeyError(digest)
                old = self._read_file(self._node_metadata_path(scope, digest), NodeMetadataRecord)
                if old is not None:
                    self._validate_node_metadata_record(old, scope, digest)
                return [self._prepared_file(self._node_metadata_path(scope, digest), record, mutable=True)], ("metadata",), None
            self._prepare_and_apply(prepare)
            return
        raise StoreCapabilityError("Legacy reference metadata is not format-4 authority.")

    def delete_metadata(self, target, *, scope=None) -> bool:
        """Journal removal of one current mapping without removing capture history.

        Args:
            target: Concrete Node object projection or saved graph.
            scope: object/state or None, with the inference in read_metadata.

        Returns:
            ``True`` when the sidecar existed and was removed, otherwise ``False``.

        Raises:
            TypeError: If ``target`` is unsupported.
            OSError: If durable logical removal fails.
            StoreCapabilityError: For legacy reference targets or unavailable
                filesystem publication guarantees.

        Side Effects:
            Revalidates the target/cut before mutation and synchronously updates
            enabled indexing. Only current annotations are removed; lineage,
            captured values and private state remain. Failures retain classified
            mutation/repair evidence rather than claiming rollback.
        """

        from ..node import Node
        if isinstance(target, Node):
            from .records import NodeMetadataRecord
            from ..metadata import _node_annotation_scope
            scope, digest = self._node_metadata_address(target, _node_annotation_scope(scope))
            self.preflight_publication("delete Node annotations", record_family="dryml-node-graph/v1")
            def prepare():
                if not self._holds_node_metadata(target, scope):
                    raise KeyError(digest)
                path = self._node_metadata_path(scope, digest)
                old = self._read_file(path, NodeMetadataRecord)
                if old is not None:
                    self._validate_node_metadata_record(old, scope, digest)
                return [self._prepared_file(path, None, mutable=True)], ("metadata",), old is not None
            return self._prepare_and_apply(prepare)
        raise StoreCapabilityError("Legacy reference metadata is not format-4 authority.")

    def discard_local_state_staging(self, handle: object) -> None:
        """Discard only this allocating handle's staging, after releasing its live lease.

        handle is a direct staging path. Missing consumed allocations are idempotent;
        foreign owners raise StoreAuthorityError. IO errors propagate and preserve
        ownership evidence. This never authorizes cleanup of unrelated live work.
        """
        path = os.path.abspath(os.fspath(handle))
        allocation = os.path.basename(path)
        if path != os.path.join(self._staging_root, allocation):
            raise StoreAuthorityError("local-state staging is not owned by this Store.")
        lease = self._staging_leases.get(allocation)
        if lease is None:
            if not os.path.exists(path) and not os.path.exists(path + ".owner"):
                return
            raise StoreAuthorityError("Staging belongs to another live/recovery owner.")
        owner = self._staging_owner(allocation)
        self.release_local_state_staging(path)
        if not self._discard_abandoned_staging(allocation, owner):
            raise StoreAuthorityError("Staging is in use; preserve it for retry.")

    def catalog_key(self) -> str:
        """Return the stable path-backed identity used only by derived indexes."""
        return f"{type(self).__module__}.{type(self).__qualname__}:{self.base_dir}"

    def close(self) -> None:
        """Release SQLite connections/staging leases, preserving authority and work.

        Raises:
            RuntimeError: If an active Session resource cache retains this Store.
            LockError: If staging leases are closed by a different thread.
        Closing expires borrowed sources. Active key leases last until their
        contexts exit. Unconsumed staging bytes/markers remain for exact recovery;
        owner death/close does not authorize retention-grant release.
        """
        super().close()
        for lease in self._staging_leases.values():
            lease.release()
        self._staging_leases.clear()
        self._staging_threads.clear()
        self._state_owner_closed = True
        if self._query_index_instance is not None:
            self._query_index_instance.close()
            self._query_index_instance = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.base_dir!r})"
