"""Format-4 Node metadata, CDef-owned state buckets and owner-scoped readers.

Legacy consumer methods are fenced off until their separately owned cutovers;
they never reinterpret V2/V3 authority as identity-3 metadata.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from collections.abc import Mapping
from uuid import uuid4
from pathlib import Path
from typing import Iterable, Literal

from ... import filesystem
from ...locking import interprocess_lock, supports_advisory_locking
from ...paths import local_path_key
from .records import (
    ClaimRecord, DeclarationRecord, DefinitionRecord, LocalStateManifest,
    MainRefRecord, ObjectAliasRecord, StateAliasRecord, StateRefRecord,
    StoredRootRecord, StoreFormatRecord, StoreRecordError,
    StateKeyRecord, NodeBlockRecord, NodeRootRecord, StateVariantManifest,
    StateAvailabilityRecord, NodeMembershipRecord, DeclarationAnchorRecord,
    DeclarationRequirementRecord, _json_bytes, _digest_value, _digest, _MAX_RECORD_BYTES,
    AuthorityRecord, MutationIntentRecord, NodeNameRecord, RetiredDeclarationAnchorRecord,
)
from . import authority
from .authority import PreparedChange, StoreCutChanged
from .store import (
    LocalStateSource, Store, StoreAuthorityError, StoreCapabilityError,
    StorePublicationCapabilities, _DELEGATED_LOCAL_STATE,
    StateDescriptor, StateInstallation, StateReadSource,
)
from ..query.model import QueryIndexStatus, QueryIndexUnavailable, ReconcileReport
from ..query.sqlite import SQLiteQueryIndexConfig
from ..query.sqlite.index import SQLiteStoreQueryIndex
from ..reference_values import ObjectRef, StateRef


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
    cut. Graph validation runs outside writer-blocking fences. Legacy Repo/query
    consumers still fail explicitly; enabled SQLite projection completes synchronously.
    Active-use leases are separate.

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
        validate current authority; same-bucket inspection shares interpretation
        only within its explicit fenced batch. Existing incompatible Stores are
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
        self._prepared_fact_owner = object()
        evidence = os.stat(self._base_dir)
        self._authority_evidence = (evidence.st_dev, evidence.st_ino)
        authority.read_cut(self)

    @property
    def store_uuid(self):
        """Return the persistent format-4 authority UUID (str), without payload IO."""
        return self._store_uuid

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
        """Classify persisted interrupted work; return replayable intent records.

        Recovery is metadata-only, preserves authoritative files/staging, and
        never captures/restores Objects or completes unexecuted transitions.
        Missing/corrupt evidence or IO failure raises rather than claiming success.
        Classification is independent of SQLite availability or compatibility.
        """
        # Authority classification must remain usable independently of a missing,
        # unavailable or incompatible derived destination.
        self._assert_state_owner_open()
        super().preflight_publication("recover authority", record_family="dryml-node-graph/v1")
        return authority.recover(self)

    def acknowledge_mutation(self, mutation_id, epoch):
        """Acknowledge exactly a captured classified UUID/cut; return current cut.

        The owner must first incorporate its exact replay work and coverage.
        Mismatched evidence raises StoreAuthorityError; IO failures propagate.
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

    def replay_mutation(self, mutation_id):
        """Complete captured partial transitions without graph or lifecycle work.

        mutation_id must name a classified pending intent in this Store. Returns
        the repaired AuthorityRecord. Exact projection witnesses must still match;
        later conflicting writers cause StoreCutChanged, never overwrite. Missing
        owned staging or invalid staged payload fails explicitly. All validation
        and payload authentication precede the writer fence. The repair is itself
        journaled, then acknowledges only the original captured mutation/cut.
        """
        from .records import _uuid_value, _decode_authority_bytes, StateVariantManifest
        import json
        _uuid_value(mutation_id)
        self.preflight_publication("replay mutation", record_family="dryml-node-graph/v1")
        self.recover_authority()
        captured = next((r for r in self.pending_mutations() if r.intent["mutation_id"] == mutation_id), None)
        if captured is None or captured.intent["status"] == "pending":
            raise StoreAuthorityError("Replay requires an exact classified pending mutation.")
        data = captured.intent
        def prepare():
            if authority.read_cut(self).epoch != data["epoch"]:
                raise StoreCutChanged("Repair cut is stale; later Store authority must be reconciled explicitly.")
            observed, final = {}, {}
            for row in data["projection"]:
                observed[(row["path"], row["kind"])] = _decode_authority_bytes(row["value"])
            for row in data["transitions"]:
                final[(row["path"], row["kind"])] = _decode_authority_bytes(row["after"])
            current = {}
            for row in data["transitions"]:
                key = (row["path"], row["kind"])
                if key not in current:
                    actual = authority._observed(self, row)
                    if actual != observed[key]:
                        raise StoreCutChanged("Later authority conflicts with captured repair evidence.")
                    current[key] = actual
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
        return self.acknowledge_mutation(mutation_id, data["epoch"])

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

    def _prepare_and_apply(self, prepare, *, retry_stale=True):
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
                return result
            except StoreCutChanged:
                if not retry_stale:
                    raise
                continue
        raise StoreCutChanged("Store changed repeatedly; retry publication explicitly.")

    def _query_v3_inventory_capabilities(self):
        raise StoreCapabilityError("Format 4 Node authority requires the new query coverage contract.")

    def query(self):
        """Refuse legacy query coverage until the format-4 query consumer cutover.

        Raises StoreCapabilityError without inventory or index IO; no empty or
        current-index claim is made for the new root/state authority families.
        """
        raise StoreCapabilityError("Format 4 Node queries are not integrated yet.")

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
        from ..node_codec import decode_node_graph, NodeCodecError
        from ..node_prepare import NodePreparation
        from ..node_views import creation_cdef, object_projection
        from ..node import require_saved
        block = self._read_node_block(record.block_digest)
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
        """
        from ..state_values import StateKey, LocalState
        from ..node_views import CREATION_VIEW, view_equal
        self.preflight_publication("install state variant", local_state=True, record_family="dryml-node-graph/v1")
        self._local_state_staging_id(staging)
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
        result = self._prepare_and_apply(prepare)
        if os.path.isdir(stage):
            shutil.rmtree(stage)
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
        metadata fence before yielding. Context/Store closure expires the source;
        U09 active-use/cleanup protection is deliberately not claimed here.
        """
        self._assert_state_owner_open()
        StateKeyRecord(key)
        options = (destination is not None, codec is not None, codec_version is not None)
        if any(options) and (not all(options) or type(codec_version) is not int or codec_version < 1):
            raise TypeError("destination, codec and positive integer codec_version must be supplied together.")
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
        from ..node import _all_nodes, _as_node_view
        from ..reference_values import ObjectRef, StateRef
        from ..utils.graph.value import iter_value_edges
        seen = set()
        nodes = list(_all_nodes(root))
        def references(value):
            if id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, (ObjectRef, StateRef)):
                ref = value.object if isinstance(value, StateRef) else value
                adapted = _as_node_view(ref.definition).with_realization(identities=ref.objects)
                nodes.extend(_all_nodes(adapted))
            for edge in iter_value_edges(value):
                references(edge.value)
        references(root)
        return tuple(node for node in nodes if node.object_id is not None)

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
        with self.writer_lock():
            previous = authority.raw(self, relative)
            if previous is not None and previous != binding:
                raise QueryIndexIncompatible("Conflicting persistent index destination/configuration for this Store authority domain.")
            if previous is None:
                self._atomic_write(os.path.join(self.base_dir, relative), binding)

    def mark_query_index_dirty(self, cdef=None, *, metadata_target=None) -> str | None:
        """Publish a durable marker for an authoritative queryable mutation.

        Args:
            cdef: Optional definition whose immutable DefinitionRecord changed.
            metadata_target: Optional ObjectRef or StateRef whose metadata changed,
                mutually exclusive with ``cdef``. Omitted targets mark an
                unscoped mutation requiring full recovery.

        Returns:
            The marker path, or ``None`` when this Store has no SQLite sidecar.

        Raises:
            TypeError: If the metadata target is not an exact reference.
            ValueError: If both scopes are supplied.
            OSError: If durable marker publication fails.

        Side Effects:
            Writes, flushes, and durably publishes a unique derived token before
            metadata publication.
        """
        from ..reference_values import ObjectRef, StateRef

        if metadata_target is not None:
            if cdef is not None:
                raise ValueError("Dirty markers require one scope only.")
            if not isinstance(metadata_target, (ObjectRef, StateRef)):
                raise TypeError("Metadata dirty targets must be ObjectRef or StateRef.")
        if self._query_index_policy not in {"auto", "sqlite"}:
            return None
        self._filesystem(filesystem.ensure_directory, self.dryml_dir)
        key = "dirty"
        if cdef is not None:
            key = DefinitionRecord(cdef).digest
        elif metadata_target is not None:
            kind = "state" if isinstance(metadata_target, StateRef) else "object"
            key = f"metadata:{kind}:{metadata_target.digest()}"
        marker_path = f"{self.query_index_dirty_path}.{uuid4().hex}"
        fd, temporary_path = tempfile.mkstemp(prefix=".query-index-dirty-", dir=self.dryml_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as target:
                target.write(f"{key}\n")
            self._filesystem(
                filesystem.publish_file,
                temporary_path,
                marker_path,
                replace=True,
            )
        except BaseException:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
            raise
        return marker_path

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
            DefinitionRecord authority. Concurrently added tokens remain dirty.
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
        """Validate the configured index while leaving DefinitionRecord authority intact."""
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
        payloads before entry. Future key protection must precede this fence;
        callers must not wrap high-level graph APIs in an externally held fence.
        """
        return interprocess_lock(self._writer_lock_path)

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

    def _read_file(self, path: str, record_type, *, preparation=None):
        try:
            mode = os.lstat(path).st_mode
        except FileNotFoundError:
            authority.observe(self, path, None)
            return None
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise StoreAuthorityError(f"Store record is not a regular file: {path!r}.")
        if record_type in {DefinitionRecord, StoredRootRecord, LocalStateManifest, StateRefRecord,
                           DeclarationRecord, ClaimRecord, MainRefRecord, ObjectAliasRecord, StateAliasRecord}:
            raise StoreAuthorityError("Legacy nested Store record is incompatible with format 4.")
        try:
            with open(path, "rb") as source:
                payload = source.read(_MAX_RECORD_BYTES + 1)
            authority.observe(self, path, payload)
            if record_type is DeclarationAnchorRecord and payload.startswith(
                    b"DRYML-METADATA/dryml-declaration-retired-anchor/1\n"):
                record_type = RetiredDeclarationAnchorRecord
            if preparation is not None:
                return record_type._from_bytes_prepared(payload, preparation)
            return record_type.from_bytes(payload)
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

    def _definition_path(self, digest: str) -> str:
        return self._digest_path(os.path.join(self.base_dir, "definitions"), digest)

    def _stored_root_path(self, digest: str) -> str:
        return self._digest_path(os.path.join(self.base_dir, "stored-roots"), digest)

    def _snapshot_path(self, digest: str) -> str:
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise StoreAuthorityError("snapshot digest is malformed.")
        return os.path.join(self.base_dir, "snapshots", digest[:2], digest)

    def _declaration_path(self, digest: str) -> str:
        return self._digest_path(os.path.join(self.base_dir, "declarations"), digest)

    def _claim_path(self, digest: str) -> str:
        return self._digest_path(os.path.join(self.base_dir, "claims"), digest)

    def _metadata_path(self, target) -> str:
        from ..reference_values import ObjectRef, StateRef

        if isinstance(target, ObjectRef):
            scope = "object"
        elif isinstance(target, StateRef):
            scope = "state"
        else:
            raise TypeError("metadata target must be an ObjectRef or StateRef.")
        return self._digest_path(os.path.join(self.base_dir, "metadata", scope), target.digest())

    def _lineage_path(self, digest: str) -> str:
        return self._digest_path(os.path.join(self.base_dir, "lineage"), digest)

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
        """Create one Store-owned empty staging directory with an empty ``data`` root."""
        self.preflight_publication("create local-state staging", local_state=True, record_family="dryml-node-graph/v1")
        path = os.path.join(self._staging_root, uuid4().hex)
        os.makedirs(os.path.join(path, "data"))
        return path

    def _existing_immutable(self, path: str, record, record_type):
        """Return compatible existing authority or reject a collision."""

        existing = self._read_file(path, record_type)
        if existing is None or existing == record:
            return existing
        if (
            isinstance(existing, DefinitionRecord)
            and isinstance(record, DefinitionRecord)
            and existing._matches_definition(record.definition)
        ):
            # Private CDef node allocations are runtime-local and do not
            # distinguish immutable graph authority.
            return existing
        raise StoreAuthorityError(
            f"Immutable {record_type.schema} collision at {path!r}."
        )

    def read_definition_record(self, digest: str) -> DefinitionRecord | None:
        """Fully validate current definition bytes and their direct path association.

        Args:
            digest (str): DefinitionRecord digest naming the direct record path.
        Returns:
            Fresh DefinitionRecord, or None when absent.
        Raises:
            StoreAuthorityError: For malformed bytes, file shape or path mismatch.
        Side Effects:
            Reads and freshly decodes trusted dill; public reads retain no facts.
        """
        return self._read_definition_record(digest)

    def _read_definition_record(self, digest, preparation=None):
        if preparation is None:
            record = self._read_file(self._definition_path(digest), DefinitionRecord)
        else:
            record = self._read_file(self._definition_path(digest), DefinitionRecord, preparation=preparation)
        if record is not None and record.digest != digest:
            raise StoreAuthorityError("DefinitionRecord digest does not match its direct path.")
        return record

    def _definition_record_for_cut(self, cut, digest):
        if not self._can_reuse_exact_validation():
            return super()._definition_record_for_cut(cut, digest)
        from .records import _RecordPreparation
        cut.check(self)
        try:
            return self._read_definition_record(digest, _RecordPreparation(cut))
        except BaseException:
            cut.discard()
            raise

    def write_definition_record(
            self, record: DefinitionRecord, *, stored_root: bool = True
    ) -> DefinitionRecord:
        """Refuse legacy definition publication on format 4 before authority effects.

        Raises StoreCapabilityError; use publish_node for identity-3 membership.
        The arguments below retain the unmigrated consumer interface.

        Args:
            record: Complete immutable graph authority to install.
            stored_root: Whether this definition is independently queryable as a
                stored root rather than closure-only graph authority.

        Returns:
            The installed record. A new publication leaves a durable derived
            index marker; idempotent publication does not rewrite that marker.
        """
        if not isinstance(record, DefinitionRecord):
            raise TypeError("record must be a DefinitionRecord.")
        self.preflight_publication(f"write {DefinitionRecord.schema}")
        path = self._definition_path(record.digest)
        with interprocess_lock(self._writer_lock_path):
            existing = self._existing_immutable(
                path, record, DefinitionRecord,
            )
            root_record = StoredRootRecord(record.digest)
            root_path = self._stored_root_path(record.digest)
            existing_root = None
            if stored_root:
                existing_root = self._existing_immutable(
                    root_path, root_record, StoredRootRecord,
                )
            if existing is not None and (not stored_root or existing_root is not None):
                return existing
            self.mark_query_index_dirty(record.definition)
            if existing is None:
                self._atomic_write(path, record.to_bytes())
            if stored_root and existing_root is None:
                self._atomic_write(root_path, root_record.to_bytes())
            return existing if existing is not None else record

    def iter_definition_records(self) -> Iterable[DefinitionRecord]:
        """Yield all validated direct-layout DefinitionRecords in digest order."""
        root = Path(self.base_dir, "definitions")
        if not root.exists():
            return ()
        records = []
        for path in sorted(root.glob("*/*.record")):
            record = self._read_file(os.fspath(path), DefinitionRecord)
            if record is None or path.name != f"{record.digest}.record" or path.parent.name != record.digest[:2]:
                raise StoreAuthorityError(f"DefinitionRecord is stored under an invalid digest path: {path!s}.")
            records.append(record)
        return tuple(records)

    def iter_stored_root_records(self) -> Iterable[StoredRootRecord]:
        """Yield validated stored-root membership in digest order.

        Raises:
            StoreAuthorityError: If a marker path or DefinitionRecord target is
                malformed or missing.

        Returns:
            Complete validated marker records in digest order. Standalone reads
            validate each target directly; fenced inventory capture reuses its
            already validated definitions instead. Neither path reads payloads
            or changes authority.
        """
        return self._read_stored_root_inventory()

    def _validated_definition_root_inventory(
        self,
    ) -> tuple[tuple[DefinitionRecord, ...], tuple[StoredRootRecord, ...]]:
        """Reuse one complete definition decode pass to validate root targets."""

        definitions = tuple(self.iter_definition_records())
        return definitions, self._read_stored_root_inventory(
            {record.digest: record for record in definitions},
        )

    def _read_stored_root_inventory(self, definitions=None):
        root = Path(self.base_dir, "stored-roots")
        if not root.exists():
            return ()
        records = []
        for path in sorted(root.glob("*/*.record")):
            record = self._read_file(os.fspath(path), StoredRootRecord)
            if (
                    record is None
                    or path.name != f"{record.definition_digest}.record"
                    or path.parent.name != record.definition_digest[:2]
                    or (self.read_definition_record(record.definition_digest)
                        if definitions is None else definitions.get(record.definition_digest)) is None
            ):
                raise StoreAuthorityError(
                    f"StoredRootRecord is stored under an invalid digest path: {path!s}."
                )
            records.append(record)
        return tuple(records)

    def read_stored_root_record(self, digest: str) -> StoredRootRecord | None:
        """Read one validated digest-addressed stored-root membership record.

        Args:
            digest: Derived DefinitionRecord digest named by the requested
                StoredRootRecord path.

        Returns:
            The matching StoredRootRecord, or ``None`` when its direct marker
            path is absent.

        Raises:
            StoreAuthorityError: If ``digest`` is malformed, the direct marker is
                malformed or stored at the wrong digest, or its DefinitionRecord
                target is missing or malformed.

        Side Effects:
            None. This reads only the requested membership marker and referenced
            DefinitionRecord; it does not consult the derived query index or
            validate unrelated stored-root authority.
        """
        try:
            expected = StoredRootRecord(digest)
        except Exception as error:
            raise StoreAuthorityError("Stored-root digest is malformed.") from error
        record = self._read_file(self._stored_root_path(digest), StoredRootRecord)
        if record is None:
            return None
        if record != expected:
            raise StoreAuthorityError(
                "StoredRootRecord digest does not match its direct path."
            )
        if self.read_definition_record(digest) is None:
            raise StoreAuthorityError(
                "Stored-root membership targets a missing DefinitionRecord."
            )
        return record

    def read_definition(self, cdef):
        """Read direct definition authority using exact V2 value-scope matching.

        Args:
            cdef (ConcreteDefinition): V2-representable candidate construction graph.
        Returns:
            ConcreteDefinition | None: Matching authority with fresh handles, or
            None if the direct record is absent or has different exact evidence.
        Raises:
            StoreAuthorityError: If a present record is invalid.
            CDefGraphCodecError: If Node facets or retained construction cannot be
                represented by this legacy boundary.
            OSError: If the direct authority file cannot be read.
        Side Effects:
            Reads the deterministic DefinitionRecord path without consulting
            retired object roots, constructing Objects or changing Store contents.
        """
        record = self.read_definition_record(DefinitionRecord(cdef).digest)
        if record is not None and record._matches_definition(cdef):
            return record.definition
        return None

    def query_index_record_metadata(self, cdef) -> tuple[str, str, int, int] | None:
        """Return direct-record metadata used only to validate a SQLite sidecar.

        Args:
            cdef: Definition represented by a prospective stored-root row.

        Returns:
            ``(record_digest, relative_path, size, mtime_ns)`` when a
            V2-scope graph-equivalent DefinitionRecord or declaration, complete
            snapshot, or object-alias record supplies root authority, otherwise
            ``None``.

        Raises:
            StoreAuthorityError: If a present authority record is invalid.
            CDefGraphCodecError: If the candidate carries identity-3 evidence that
                this V2 authority boundary cannot retain.
            OSError: If a selected authority file cannot be inspected.

        Side Effects:
            Reads authority and file metadata without modifying it. Falls back
            to reference enumeration only when the direct definition is absent.
        """
        record = self.read_definition_record(DefinitionRecord(cdef).digest)
        if record is not None and record._matches_definition(cdef):
            path = self._definition_path(record.digest)
        else:
            candidates = (
                (self.iter_declaration_records, lambda item: item.object_ref.definition,
                 lambda item: self._declaration_path(item.digest)),
                (self.iter_state_ref_records, lambda item: item.state_ref.definition,
                 lambda item: os.path.join(self._snapshot_path(item.digest), "state-ref.record")),
                (self.iter_object_alias_records, lambda item: item.object_ref.definition,
                 lambda item: self._ref_path("objects", f"{item.alias}.record")),
            )
            selected = next((
                (item, record_path(item))
                for records, definition, record_path in candidates
                for item in records() if DefinitionRecord(definition(item))._matches_definition(cdef)
            ), None)
            if selected is None:
                return None
            record, path = selected
        metadata = os.stat(path)
        return (
            record.object_ref.digest() if isinstance(record, ObjectAliasRecord) else record.digest,
            os.path.relpath(path, self.base_dir).replace(os.sep, "/"),
            metadata.st_size,
            metadata.st_mtime_ns,
        )

    def _validate_local_state_dir(
            self, directory: str, manifest: LocalStateManifest | None,
            *, defer_payload: bool = False, preparation=None):
        """Validate a physical payload once and return its manifest/definition.

        A supplied manifest is additionally matched to on-disk authority. Passing
        None reads that authority here, avoiding a redundant manifest decode.
        """
        expected = {"data", "def.pkl", "manifest.record"}
        try:
            root_mode = os.lstat(directory).st_mode
            entries = {entry.name: entry for entry in os.scandir(directory)}
        except FileNotFoundError as error:
            raise StoreAuthorityError("local state staging directory is missing.") from error
        if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
            raise StoreAuthorityError("local state directory must be a real directory.")
        if set(entries) != expected:
            raise StoreAuthorityError("local state directory must contain exactly data/, def.pkl, and manifest.record.")
        if not entries["data"].is_dir(follow_symlinks=False):
            raise StoreAuthorityError("local state data entry must be a real directory.")
        if not entries["def.pkl"].is_file(follow_symlinks=False) or not entries["manifest.record"].is_file(follow_symlinks=False):
            raise StoreAuthorityError("local state metadata entries must be regular files.")
        stored_manifest = self._read_file(os.path.join(directory, "manifest.record"), LocalStateManifest)
        if stored_manifest is None or (manifest is not None and stored_manifest != manifest):
            raise StoreAuthorityError("local state manifest bytes do not match requested manifest authority.")
        manifest = stored_manifest
        if defer_payload:
            self._require_deferred_local_state_support(manifest)
        definition_bytes = Path(directory, "def.pkl").read_bytes()
        if hashlib.sha256(definition_bytes).hexdigest() != manifest.definition_file_digest:
            raise StoreAuthorityError("local state definition file bytes do not match manifest authority.")
        if preparation is None:
            definition = self._read_file(os.path.join(directory, "def.pkl"), DefinitionRecord)
        else:
            definition = self._read_file(os.path.join(directory, "def.pkl"), DefinitionRecord, preparation=preparation)
        if definition is None or definition.digest != manifest.definition_digest or definition.graph_hash != manifest.graph_hash:
            raise StoreAuthorityError("local state definition does not match manifest graph and definition digests.")
        try:
            manifest.validate_payload(
                os.path.join(directory, "data"), defer_payload=defer_payload,
            )
        except StoreRecordError as error:
            raise StoreAuthorityError(f"local state payload is invalid: {error}") from error
        return manifest, definition

    def iter_state_ref_records(self) -> Iterable[StateRefRecord]:
        """Enumerate complete v3 snapshot StateRef authority in digest order.

        Returns:
            Iterable of StateRefRecords whose complete association siblings have
            validated, or an empty tuple when no snapshot directories exist.

        Raises:
            StoreAuthorityError: If catalogue shape or a complete snapshot is
                invalid, including a directory disappearing during capture.

        Side Effects:
            Reads every bundle without opening payload bytes or changing Store
            authority. Scoped internal enumeration uses the same validator.
        """
        return self._iter_state_ref_records(self._read_snapshot)

    def _iter_state_ref_records(self, read_snapshot):
        """Enumerate with the same shape validation for public and scoped reads."""
        root = Path(self.base_dir, "snapshots")
        if not root.exists():
            return ()
        records = []
        for path in sorted(root.glob("*/*")):
            if not path.is_dir() or path.parent.name != path.name[:2]:
                raise StoreAuthorityError(f"Snapshot is stored under an invalid digest path: {path!s}.")
            snapshot = read_snapshot(path.name)
            if snapshot is None:
                raise StoreAuthorityError("Snapshot disappeared during authority capture.")
            records.append(snapshot[0])
        return tuple(records)

    @staticmethod
    def _read_json(path: str, *, cut=None, kind=None):
        from dryml.formats import canonical_json_load_bytes
        from dryml.formats import canonical as parser
        from .._content_facts import _primitive_size, _freeze_json_fact, _thaw_json_fact
        from weakref import ref

        try:
            mode = os.lstat(path).st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise StoreAuthorityError("snapshot metadata is not a regular file.")
            with open(path, "rb") as source:
                payload = source.read()
            bounds = dict(max_depth=64, max_nodes=131072, max_entries=65536, max_string=4096, max_int_bits=4096)
            if cut is not None:
                if kind not in {"placement.json", "metadata.json", "snapshot.json"}:
                    raise ValueError("Only snapshot JSON siblings support retained primitives.")
                implementations = (
                    canonical_json_load_bytes, parser.canonical_json_loads,
                    parser.deep_freeze_json, parser.json.loads,
                    parser._reject_constant, parser._reject_duplicate_keys,
                    parser.check_string, _freeze_json_fact, _thaw_json_fact,
                    _primitive_size,
                )
                try:
                    context = ("snapshot-json", kind, 1, "canonical-json", 1,
                               tuple(bounds.items()), "exact-primitives", 1,
                               tuple((ref(value), ref(getattr(value, "__code__", value)))
                                     for value in implementations))
                except TypeError:
                    context = None
                if context is not None:
                    prepared = cut.prepared(context, payload)
                    if prepared is not None:
                        return _thaw_json_fact(prepared)
            value = canonical_json_load_bytes(payload, **bounds)
            if cut is not None and context is not None:
                try:
                    owned_size = _primitive_size(value)
                except TypeError:
                    return value
                cut.stage_prepared(context, payload, owned_size, lambda: _freeze_json_fact(value))
            return value
        except FileNotFoundError:
            return None
        except StoreAuthorityError:
            raise
        except Exception as error:
            raise StoreAuthorityError("snapshot JSON authority is malformed.") from error

    @staticmethod
    def _write_json(path: str, value) -> None:
        from dryml.formats import canonical_json_bytes

        payload = canonical_json_bytes(value, max_depth=64, max_nodes=131072, max_entries=65536, max_string=4096, max_int_bits=4096)
        Path(path).write_bytes(payload)

    def read_metadata(self, target):
        """Refuse legacy exact-reference metadata until its consumer cutover.

        target is the old ObjectRef/StateRef scope. Raises StoreCapabilityError
        without reading or admitting a V2 nested reference in format 4.
        """
        raise StoreCapabilityError("Legacy reference metadata is not format-4 authority.")

    def write_metadata(self, target, values) -> None:
        """Refuse unmigrated reference-metadata publication on format 4.

        StoreCapabilityError precedes authority effects. Remaining arguments
        describe the legacy consumer boundary, not supported format-4 publication.

        Args:
            target: Exact ObjectRef or StateRef attachment scope.
            values: Valid complete metadata mapping to install.

        Returns:
            ``None``.

        Raises:
            TypeError: If inputs are unsupported.
            ValueError: If metadata values violate codec bounds.
            StoreCapabilityError: If this filesystem cannot provide publication.

        Side Effects:
            Serializes cooperating writers and atomically replaces one sidecar.
            It does not alter snapshots, lineage metadata, or payload files.
        """

        from dryml.formats import canonical_json_bytes
        from ..metadata import encode_current_annotations

        record = encode_current_annotations(target, values)
        payload = canonical_json_bytes(
            record, max_depth=64, max_nodes=131072, max_entries=65536,
            max_string=4096, max_int_bits=4096,
        )
        self.preflight_publication("write current metadata")
        with self.writer_lock():
            # The derived index must become observably stale before authority can
            # expose a new whole mapping.
            self.mark_query_index_dirty(metadata_target=target)
            self._atomic_write(self._metadata_path(target), payload)

    def delete_metadata(self, target) -> bool:
        """Refuse legacy reference-metadata removal before effects on format 4.

        Raises StoreCapabilityError pending the reference consumer cutover.

        Args:
            target: Exact ObjectRef or StateRef attachment scope.

        Returns:
            ``True`` when the sidecar existed and was removed, otherwise ``False``.

        Raises:
            TypeError: If ``target`` is unsupported.
            OSError: If durable logical removal fails.
            StoreCapabilityError: If this filesystem cannot provide publication.

        Side Effects:
            Serializes cooperating writers and durably removes only the current
            metadata name. Windows may retain a non-authoritative tombstone after
            best-effort cleanup. Snapshot, lineage, and payload authority remain.
        """

        self.preflight_publication("delete current metadata")
        path = self._metadata_path(target)
        with self.writer_lock():
            try:
                os.stat(path)
            except FileNotFoundError:
                return False
            try:
                # Absence is indexed metadata too, so invalidate before removal.
                self.mark_query_index_dirty(metadata_target=target)
                return self._filesystem(
                    filesystem.remove_file,
                    path,
                    cleanup_directory=os.path.dirname(path) or ".",
                    missing_ok=True,
                )
            except FileNotFoundError:
                return False

    def read_lineage_metadata(self, target):
        """Refuse old ObjectRef lineage without reading or rewriting authority.

        target is a legacy ObjectRef. Raises StoreCapabilityError pending its
        separately owned identity-3 consumer codec; does not claim absence.
        """
        raise StoreCapabilityError("Legacy reference lineage is not format-4 authority.")

    def write_lineage_metadata(self, value):
        """Refuse legacy ObjectRef lineage publication before format-4 effects.

        Raises StoreCapabilityError pending the reference consumer cutover.

        Args:
            value: Valid LineageMetadata to install.

        Returns:
            The installed fact, or an equal existing fact.

        Raises:
            TypeError: If ``value`` is unsupported.
            ValueError: If its lineage fields are invalid.
            StoreAuthorityError: If unequal evidence already exists.

        Side Effects:
            Serializes cooperating writers and writes at most once. It does not
            change current annotations, snapshot captures, or payload files.
        """

        from dryml.formats import canonical_json_bytes
        from ..metadata import encode_lineage_metadata

        record = encode_lineage_metadata(value)
        payload = canonical_json_bytes(
            record, max_depth=64, max_nodes=131072, max_entries=65536,
            max_string=4096, max_int_bits=4096,
        )
        self.preflight_publication("write lineage metadata")
        path = self._lineage_path(value.object_ref.digest())
        with self.writer_lock():
            existing = self.read_lineage_metadata(value.object_ref)
            if existing is not None:
                if existing != value:
                    raise StoreAuthorityError("lineage write-once authority conflicts with existing evidence.")
                return existing
            self.mark_query_index_dirty(metadata_target=value.object_ref)
            self._atomic_write(path, payload)
            return value

    def _read_snapshot(
            self, digest: str, *, payloads: bool = False,
            directory: str | None = None, cut=None):
        """Validate one v3 snapshot association without opening payloads by default."""

        from dryml.core.metadata import decode_snapshot_metadata
        from dryml.records import decode_record
        from ..utils.graph.path import GraphPath

        directory = self._snapshot_path(digest) if directory is None else directory
        try:
            mode = os.lstat(directory).st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise StoreAuthorityError("snapshot authority is not a real directory.")
            entries = {entry.name: entry for entry in os.scandir(directory)}
        except FileNotFoundError:
            return None
        except StoreAuthorityError:
            raise
        except OSError as error:
            raise StoreAuthorityError("snapshot authority is inaccessible.") from error
        required = {"state-ref.record", "placement.json", "metadata.json", "snapshot.json"}
        if not required.issubset(entries) or any(
                not entries[name].is_file(follow_symlinks=False) for name in required):
            raise StoreAuthorityError("snapshot lacks required complete metadata siblings.")
        if cut is None:
            record = self._read_file(os.path.join(directory, "state-ref.record"), StateRefRecord)
        else:
            from .records import _RecordPreparation
            record = self._read_file(os.path.join(directory, "state-ref.record"), StateRefRecord,
                                     preparation=_RecordPreparation(cut))
        if record is None or record.digest != digest:
            raise StoreAuthorityError("snapshot StateRef record does not match its directory.")

        def sibling(name):
            path = os.path.join(directory, name)
            return self._read_json(path) if cut is None else self._read_json(path, cut=cut, kind=name)

        placement_envelope = sibling("placement.json")
        metadata_envelope = sibling("metadata.json")
        snapshot_envelope = sibling("snapshot.json")
        try:
            placement_record = decode_record(placement_envelope)
            snapshot_record = decode_record(snapshot_envelope)
            if placement_record.kind != "dryml.core.snapshot_placement" or placement_record.version != 1:
                raise ValueError("unsupported placement record")
            if snapshot_record.kind != "dryml.core.snapshot" or snapshot_record.version != 1:
                raise ValueError("unsupported snapshot association")
            placement = placement_record.data
            association = snapshot_record.data
            if set(placement) != {"state_ref_digest", "local", "children"} or set(association) != {"state_ref_digest", "placement_id", "metadata_id"}:
                raise ValueError("snapshot records are not closed")
            if placement["state_ref_digest"] != digest or association["state_ref_digest"] != digest:
                raise ValueError("snapshot records target another StateRef")
            if association["placement_id"] != placement_envelope.get("id") or association["metadata_id"] != metadata_envelope.get("id"):
                raise ValueError("snapshot association ids do not match sibling records")
            metadata = decode_snapshot_metadata(metadata_envelope, record.state_ref)
        except Exception as error:
            raise StoreAuthorityError("snapshot metadata association is malformed.") from error
        if metadata.state_ref != record.state_ref:
            raise StoreAuthorityError("snapshot metadata does not match StateRef authority.")
        if not isinstance(placement["local"], list) or not isinstance(placement["children"], list):
            raise StoreAuthorityError("snapshot placement is malformed.")
        local = {}
        for item in placement["local"]:
            if not isinstance(item, dict) or set(item) != {"path", "state_hash", "directory"}:
                raise StoreAuthorityError("snapshot local placement entry is malformed.")
            path = GraphPath.from_data(item["path"])
            if path in local or record.state_ref.states.get(path) != item["state_hash"]:
                raise StoreAuthorityError("snapshot local placement does not match StateRef states.")
            directory_name = item["directory"]
            if not isinstance(directory_name, str) or not directory_name.startswith("local-state/") or ".." in directory_name.split("/"):
                raise StoreAuthorityError("snapshot local placement directory is malformed.")
            local[path] = directory_name
        covered = dict(local)
        for item in placement["children"]:
            if not isinstance(item, dict) or set(item) != {"path", "state_ref_digest"}:
                raise StoreAuthorityError("snapshot child placement entry is malformed.")
            try:
                path = GraphPath.from_data(item["path"])
                child = record.state_ref.at(path)
            except Exception as error:
                raise StoreAuthorityError("snapshot child placement path is malformed.") from error
            if child.digest() != item["state_ref_digest"]:
                raise StoreAuthorityError("snapshot child placement does not match its projection.")
            for child_path, state_hash in child.states.items():
                target_path = path.join(child_path)
                if target_path in covered or record.state_ref.states.get(target_path) != state_hash:
                    raise StoreAuthorityError("snapshot placement overlaps or does not match StateRef states.")
                covered[target_path] = None
        if set(covered) != set(record.state_ref.states):
            raise StoreAuthorityError("snapshot placement does not cover every stateful path.")
        if payloads:
            self._validate_snapshot_locals((record, metadata, local), directory)
        return record, metadata, local

    def _validate_snapshot_locals(
            self, snapshot, directory, *, paths=None, defer_payload=False,
            validated=None, preparation=None):
        """Share physical work while checking every placement/definition binding."""
        record, _, local = snapshot
        validated = {} if validated is None else validated
        reusable = self._can_reuse_exact_validation()
        results = {}
        for path in local if paths is None else paths:
            relative = local.get(path)
            if relative is None:
                raise KeyError(path)
            target = os.path.normpath(os.path.join(directory, relative))
            key = (target, defer_payload)
            evidence = (validated.get((target, False)) or validated.get(key)) if reusable else None
            if evidence is None:
                # Preserve the old supplied-manifest contract for custom lower
                # validators, including overrides returning None on success.
                manifest = None
                if not reusable:
                    manifest = self._read_file(
                        os.path.join(target, "manifest.record"), LocalStateManifest,
                    )
                    if manifest is None:
                        raise StoreAuthorityError("snapshot payload manifest does not match placement.")
                options = {} if preparation is None else {"preparation": preparation}
                if defer_payload:
                    options["defer_payload"] = True
                evidence = self._validate_local_state_dir(
                    target, manifest, **options,
                )
                if evidence is None:
                    evidence = (manifest, self._read_file(
                        os.path.join(target, "def.pkl"), DefinitionRecord,
                    ))
            manifest, definition = evidence
            if manifest.state_hash != record.state_ref.states[path]:
                raise StoreAuthorityError("snapshot payload manifest does not match placement.")
            if defer_payload:
                self._require_deferred_local_state_support(manifest)
            expected = record.state_ref.object.at(path).definition
            if definition is None or not definition._matches_definition(expected):
                raise StoreAuthorityError("snapshot payload definition does not match StateRef path.")
            if reusable:
                validated[key] = evidence
            results[path] = manifest
        return results

    def _can_reuse_exact_validation(self):
        """Opt in only for effective, unmodified built-in authority readers."""
        expected = _EXACT_READERS.get(type(self))
        return expected is not None and all(
            getattr(getattr(self, name), "__func__", getattr(self, name)) is value
            and name not in self.__dict__
            for name, value in expected.items()
        ) and LocalStateManifest.validate_payload is _EXACT_PAYLOAD_VALIDATOR and all(
            getattr(getattr(owner, name), "__func__", getattr(owner, name)) is value
            for owner, name, value in _EXACT_RECORD_VALIDATORS
        )

    def _cut_snapshot_key(self, cut, digest):
        cut.check(self)
        return (self, self.base_dir, self._authority_evidence, self._snapshot_path(digest))

    def _read_snapshot_for_cut(self, cut, digest):
        if not self._can_reuse_exact_validation():
            return super()._read_snapshot_for_cut(cut, digest)
        key = self._cut_snapshot_key(cut, digest)
        snapshot = cut.snapshots.get(key)
        if snapshot is None:
            snapshot = self._read_snapshot(digest, cut=cut)
            if snapshot is not None:
                cut.check(self)
                cut.snapshots[key] = snapshot
        return snapshot

    def _state_ref_records_for_cut(self, cut):
        if not self._can_reuse_exact_validation():
            return super()._state_ref_records_for_cut(cut)
        cut.check(self)
        return self._iter_state_ref_records(
            lambda digest: self._read_snapshot_for_cut(cut, digest),
        )

    def _snapshot_metadata_for_cut(self, cut, digest):
        if not self._can_reuse_exact_validation():
            return super()._snapshot_metadata_for_cut(cut, digest)
        snapshot = self._read_snapshot_for_cut(cut, digest)
        return None if snapshot is None else snapshot[1]

    def _open_local_state_for_cut(self, cut, reference, path, *, defer_payload=False):
        """Promote local siblings once; report validated delegation without failure."""
        if not self._can_reuse_exact_validation():
            return super()._open_local_state_for_cut(
                cut, reference, path, defer_payload=defer_payload,
            )
        from ..utils.graph.path import normalize_path

        path = normalize_path(path)
        snapshot = self._read_snapshot_for_cut(cut, reference.digest())
        if snapshot is None or snapshot[0].state_ref != reference:
            raise KeyError(reference.digest())
        key = self._cut_snapshot_key(cut, reference.digest())
        directory = key[-1]
        validated = cut.locals.setdefault(key, {})
        from .records import _RecordPreparation
        preparation = _RecordPreparation(cut)
        if not defer_payload:
            if key not in cut.eager_snapshots:
                manifests = self._validate_snapshot_locals(snapshot, directory, validated=validated,
                                                            preparation=preparation)
                cut.check(self)
                cut.eager_snapshots[key] = manifests
            manifests = cut.eager_snapshots[key]
        if path not in snapshot[2]:
            if path not in reference.states:
                raise KeyError(path)
            cut.check(self)
            return _DELEGATED_LOCAL_STATE
        if defer_payload:
            manifests = self._validate_snapshot_locals(
                snapshot, directory, paths=(path,), defer_payload=True,
                validated=validated,
                preparation=preparation,
            )
        manifest = manifests[path]
        cut.check(self)
        return LocalStateSource(
            self, os.path.join(directory, snapshot[2][path]), manifest,
            eagerly_validated=not (defer_payload and manifest.deferred_paths),
        )

    def read_state_ref_record(self, digest: str) -> StateRefRecord | None:
        """Read one StateRef only when its complete v3 snapshot validates.

        Args:
            digest (str): Exact digest naming a snapshot directory.

        Returns:
            StateRefRecord, or None for an absent snapshot.

        Raises:
            StoreAuthorityError: If record, metadata, or placement is incomplete
                or incompatible. Payload bytes are not opened by this reader.

        Side Effects:
            Reads complete association siblings without retaining validity.
        """

        snapshot = self._read_snapshot(digest)
        return None if snapshot is None else snapshot[0]

    def get_snapshot_directory(self, target):
        """Return the validated direct directory for one exact snapshot.

        Args:
            target: Exact StateRef to locate.

        Returns:
            Persistent absolute Path for the complete snapshot directory.

        Raises:
            TypeError: If ``target`` is not a StateRef.
            KeyError: If the snapshot is absent.
            StoreAuthorityError: If its association is malformed or incompatible.

        Side Effects:
            Validates metadata and placement records without opening payload bytes,
            materializing Objects, or changing authority.
        """

        from ..reference_values import StateRef

        if not isinstance(target, StateRef):
            raise TypeError("target must be a StateRef.")
        snapshot = self._read_snapshot(target.digest())
        if snapshot is None:
            raise KeyError(target.digest())
        if snapshot[0].state_ref != target:
            raise StoreAuthorityError("snapshot directory has incompatible StateRef authority.")
        return Path(self._snapshot_path(target.digest()))

    def read_snapshot_metadata(self, digest: str):
        """Return captured metadata for one validated complete snapshot.

        Args:
            digest: Exact StateRef digest naming the snapshot directory.

        Returns:
            Detached SnapshotMetadata, or ``None`` when the directory is absent.

        Raises:
            StoreAuthorityError: If the snapshot association is malformed.

        Side Effects:
            Reads metadata association without opening payload bytes or changing
            Store authority.
        """

        snapshot = self._read_snapshot(digest)
        return None if snapshot is None else snapshot[1]

    def discard_local_state_staging(self, handle: object) -> None:
        """Idempotently remove only Store-owned unpublished local-state staging."""

        try:
            path = os.path.abspath(os.fspath(handle))
            root = os.path.realpath(self._staging_root)
            if os.path.commonpath((root, os.path.realpath(path))) != root:
                raise StoreAuthorityError("local-state staging is not owned by this Store.")
            shutil.rmtree(path)
        except FileNotFoundError:
            return

    def prepare_local_state(self, source: object, manifest: LocalStateManifest) -> LocalStateSource:
        """Validate owned completed staging before it is copied into a snapshot."""

        path = os.path.abspath(os.fspath(source))
        root = os.path.realpath(self._staging_root)
        if os.path.commonpath((root, os.path.realpath(path))) != root:
            raise StoreAuthorityError("local-state staging must be created by the selected Store.")
        self._require_deferred_local_state_support(manifest)
        self._validate_local_state_dir(path, manifest)
        return LocalStateSource(self, path, manifest)

    def prepare_rebound_local_state(self, source: LocalStateSource, target_definition) -> LocalStateSource:
        """Copy validated bytes into this Store's staging and bind a fork definition."""

        if not isinstance(source, LocalStateSource):
            raise TypeError("source must be a LocalStateSource.")
        if not source.eagerly_validated:
            raise StoreAuthorityError("cannot rebind a metadata-only local-state source.")
        stage = self.create_local_state_staging()
        try:
            shutil.rmtree(stage)
            shutil.copytree(os.fspath(source.handle), stage)
            record = DefinitionRecord(target_definition)
            definition_bytes = record.to_bytes()
            Path(stage, "def.pkl").write_bytes(definition_bytes)
            manifest = LocalStateManifest(
                source.manifest.codec, record.graph_hash, record.digest,
                hashlib.sha256(definition_bytes).hexdigest(), source.manifest.files,
                source.manifest.deferred_paths, source.manifest.version,
            )
            if manifest.state_hash != source.manifest.state_hash:
                raise StoreAuthorityError("rebound local state changed its payload identity.")
            Path(stage, "manifest.record").write_bytes(manifest.to_bytes())
            return self.prepare_local_state(stage, manifest)
        except BaseException:
            self.discard_local_state_staging(stage)
            raise

    def validate_local_state(self, reference, path) -> LocalStateManifest:
        """Fully validate one snapshot's local payloads and return a selected manifest.

        Args:
            reference (StateRef): Exact snapshot to inspect.
            path (GraphPath or path data): Requested locally placed state.

        Returns:
            LocalStateManifest for the selected path, with all local siblings
            eagerly checked as well.

        Raises:
            TypeError: If the reference or path type is unsupported.
            KeyError: If snapshot authority or local placement is absent.
            StoreAuthorityError: If association, manifest, definition, or payload
                bytes are inconsistent.

        Side Effects:
            Reads and hashes local bytes without restoring Objects, retaining
            cross-call validity, or mutating authority.
        """

        from ..reference_values import StateRef
        from ..utils.graph.path import normalize_path

        if not isinstance(reference, StateRef):
            raise TypeError("reference must be a StateRef.")
        path = normalize_path(path)
        snapshot = self._read_snapshot(reference.digest(), payloads=True)
        if snapshot is None or snapshot[0].state_ref != reference:
            raise KeyError(reference.digest())
        relative = snapshot[2].get(path)
        if relative is None:
            raise KeyError(path)
        manifest = self._read_file(os.path.join(self._snapshot_path(reference.digest()), relative, "manifest.record"), LocalStateManifest)
        assert manifest is not None
        return manifest

    def open_local_state(self, reference, path) -> LocalStateSource:
        """Open one eagerly validated borrowed source for restoration or copy.

        Args:
            reference (StateRef): Complete exact snapshot authority.
            path (GraphPath or path data): Selected snapshot-local state path.

        Returns:
            LocalStateSource borrowing a directory from this Store. Expiration of
            Repo preflight evidence does not shorten the source handle lifetime.

        Raises:
            TypeError: If reference or path has an unsupported type.
            KeyError: If snapshot or local placement is absent.
            StoreAuthorityError: If the bundle, any local sibling payload, or a
                path's definition association is invalid.

        Side Effects:
            Hashes all locally placed payloads eagerly; no Objects are restored
            or constructed and no authority is written.
        """

        from ..utils.graph.path import normalize_path

        path = normalize_path(path)
        manifest = self.validate_local_state(reference, path)
        snapshot = self._read_snapshot(reference.digest())
        assert snapshot is not None
        return LocalStateSource(self, os.path.join(self._snapshot_path(reference.digest()), snapshot[2][path]), manifest)

    def _open_local_state_for_exact_load(self, reference, path) -> LocalStateSource:
        """Open one exact-restore payload after metadata-only deferred validation.

        Non-deferred payloads retain the eager ``(directory, manifest)`` validator
        call, including custom validators returning None on success.
        """

        from ..reference_values import StateRef
        from ..utils.graph.path import normalize_path

        if not isinstance(reference, StateRef):
            raise TypeError("reference must be a StateRef.")
        path = normalize_path(path)
        snapshot = self._read_snapshot(reference.digest())
        if snapshot is None or snapshot[0].state_ref != reference:
            raise KeyError(reference.digest())
        relative = snapshot[2].get(path)
        if relative is None:
            raise KeyError(path)
        directory = os.path.join(self._snapshot_path(reference.digest()), relative)
        manifest = self._read_file(
            os.path.join(directory, "manifest.record"), LocalStateManifest,
        )
        if manifest is None or manifest.state_hash != reference.states[path]:
            raise StoreAuthorityError("snapshot payload manifest does not match placement.")
        self._require_deferred_local_state_support(manifest)
        if manifest.deferred_paths:
            self._validate_local_state_dir(directory, manifest, defer_payload=True)
        else:
            self._validate_local_state_dir(directory, manifest)
        expected = reference.object.at(path).definition
        definition = self._read_file(os.path.join(directory, "def.pkl"), DefinitionRecord)
        if definition is None or not definition._matches_definition(expected):
            raise StoreAuthorityError("snapshot payload definition does not match StateRef path.")
        return LocalStateSource(
            self, directory, manifest, eagerly_validated=not manifest.deferred_paths,
        )

    def _require_deferred_local_state_support(self, manifest: LocalStateManifest) -> None:
        """Reject deferred manifests on backends without durable payload paths."""

        if manifest.deferred_paths and not self._supports_deferred_local_state:
            raise StoreCapabilityError(
                "Store backend does not support deferred local-state payload validation."
            )

    def _validate_local_state_source_for_publication(
            self, source: LocalStateSource) -> None:
        """Require full verification and supported deferred paths before publish."""

        if not isinstance(source, LocalStateSource):
            raise TypeError("source must be a LocalStateSource.")
        self._require_deferred_local_state_support(source.manifest)
        if not source.eagerly_validated:
            raise StoreAuthorityError("snapshot source must be fully payload-validated.")

    def publish_snapshot(
            self, reference, *, evidence, annotations=None, local_states,
            children=None, _before_annotation_write=None):
        """Refuse V3 snapshot publication on format 4 before authority effects.

        Raises StoreCapabilityError pending U10/U11's Node consumer cutover. The
        arguments below describe the unmigrated snapshot interface; no legacy
        snapshot directory is installed by current built-in Stores.

        Args:
            reference: Exact StateRef to install.
            evidence: Fresh SnapshotCapture or matching copied SnapshotMetadata.
            annotations: Optional SaveAnnotations current-map replacements for root
                ObjectRef and StateRef scopes.
            local_states: Mapping of locally owned paths to validated sources.
                Source definitions must exactly match the target's V2 value
                scopes, including sharing within each reference/quotation scope.
                Identity-3 facets or retained construction are not representable.
            children: Optional mapping of child projection paths to StateRefs.
            _before_annotation_write: Internal phase callback invoked immediately
                before each requested current-metadata write.

        Returns:
            Detached SnapshotMetadata for the installed or equal existing snapshot.

        Raises:
            TypeError: If publication inputs are unsupported.
            StoreAuthorityError: If records, sources, coverage, or immutable
                existing snapshot evidence conflict.
            CDefGraphCodecError: If a source or target definition carries Node
                facets or retained construction unsupported by the V2 metadata.
            OSError: If staged content or durable activation fails.
            StoreCapabilityError: If direct publication guarantees are unavailable.

        Side Effects:
            Atomically installs a complete immutable directory. Explicit
            annotations are applied as current LWW sidecars after installation;
            existing captured annotations are never refreshed. Cooperating writers
            are serialized and staging is removed after success or failure.
        """

        from ..metadata import SaveAnnotations, SnapshotCapture, SnapshotMetadata, encode_snapshot_metadata
        from ..reference_values import StateRef
        from ..utils.graph.path import GraphPath
        from dryml.records import GenericRecord, encode_record

        if children is None:
            children = {}
        if not isinstance(reference, StateRef) or not isinstance(local_states, dict) or not isinstance(children, Mapping):
            raise TypeError("snapshot publication requires a StateRef, local-state mapping, and child projection mapping.")
        if annotations is not None and not isinstance(annotations, SaveAnnotations):
            raise TypeError("annotations must be a SaveAnnotations or None.")
        self.preflight_publication("publish snapshot", local_state=True)
        covered = set(local_states)
        child_entries = []
        for path, child in children.items():
            if not isinstance(path, GraphPath) or child != reference.at(path):
                raise StoreAuthorityError("snapshot child projection does not match its parent StateRef.")
            for child_path in child.states:
                target_path = path.join(child_path)
                if target_path in covered:
                    raise StoreAuthorityError("snapshot child projection overlaps local payload authority.")
                covered.add(target_path)
            child_entries.append({"path": path.to_data(), "state_ref_digest": child.digest()})
        if covered != set(reference.states):
            raise StoreAuthorityError("snapshot publication must cover every state path locally or by child projection.")
        os.makedirs(self._staging_root, exist_ok=True)
        stage = tempfile.mkdtemp(prefix="snapshot-", dir=self._staging_root)
        try:
            placement_entries = []
            for path, source in sorted(local_states.items(), key=lambda item: str(item[0])):
                if not isinstance(path, GraphPath) or not isinstance(source, LocalStateSource):
                    raise TypeError("snapshot local states must map GraphPath to LocalStateSource.")
                if source.manifest.state_hash != reference.states[path]:
                    raise StoreAuthorityError("snapshot source does not match StateRef state hash.")
                self._validate_local_state_source_for_publication(source)
                target_definition = reference.object.at(path).definition
                source_definition = self._read_file(os.path.join(os.fspath(source.handle), "def.pkl"), DefinitionRecord)
                if source_definition is None or not source_definition._matches_definition(target_definition):
                    raise StoreAuthorityError("snapshot source definition does not match StateRef path.")
                relative = f"local-state/{source.manifest.graph_hash}/{source.manifest.state_hash}"
                destination = os.path.join(stage, relative)
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                if not os.path.exists(destination):
                    shutil.copytree(os.fspath(source.handle), destination)
                self._validate_local_state_dir(destination, source.manifest)
                placement_entries.append({"path": path.to_data(), "state_hash": source.manifest.state_hash, "directory": relative})
            target = self._snapshot_path(reference.digest())
            with self.writer_lock():
                if isinstance(evidence, SnapshotCapture):
                    object_values = self.read_metadata(reference.object)
                    state_values = self.read_metadata(reference)
                    if annotations is not None:
                        if annotations.object is not None:
                            object_values = annotations.object
                        if annotations.state is not None:
                            state_values = annotations.state
                    metadata = SnapshotMetadata(
                        reference, evidence.lineages, evidence.saved_at,
                        evidence.environment, evidence.environment_status,
                        evidence.requirements, evidence.requirements_status,
                        evidence.requirements_coverage, evidence.diagnostics,
                        object_values, state_values,
                    )
                elif isinstance(evidence, SnapshotMetadata) and evidence.state_ref == reference:
                    metadata = evidence
                else:
                    raise StoreAuthorityError("snapshot evidence does not match its StateRef.")
                Path(stage, "state-ref.record").write_bytes(StateRefRecord(reference).to_bytes())
                placement = encode_record(GenericRecord("dryml.core.snapshot_placement", 1, {"state_ref_digest": reference.digest(), "local": placement_entries, "children": child_entries}))
                metadata_record = encode_snapshot_metadata(metadata)
                association = encode_record(GenericRecord("dryml.core.snapshot", 1, {"state_ref_digest": reference.digest(), "placement_id": placement["id"], "metadata_id": metadata_record["id"]}))
                self._write_json(os.path.join(stage, "placement.json"), placement)
                self._write_json(os.path.join(stage, "metadata.json"), metadata_record)
                self._write_json(os.path.join(stage, "snapshot.json"), association)
                self._read_snapshot_from_directory(stage, reference, payloads=True)
                existing = self._read_snapshot(reference.digest())
                if existing is not None:
                    if existing[0].state_ref != reference:
                        raise StoreAuthorityError("snapshot write-once authority conflicts with existing evidence.")
                    if isinstance(evidence, SnapshotMetadata) and existing[1] != metadata:
                        raise StoreAuthorityError("snapshot write-once authority conflicts with existing evidence.")
                    if annotations is not None:
                        if annotations.object is not None:
                            if _before_annotation_write is not None:
                                _before_annotation_write("object")
                            self.write_metadata(reference.object, annotations.object)
                        if annotations.state is not None:
                            if _before_annotation_write is not None:
                                _before_annotation_write("state")
                            self.write_metadata(reference, annotations.state)
                    return existing[1]
                self.mark_query_index_dirty(metadata_target=reference)
                self._filesystem(
                    filesystem.ensure_directory, os.path.dirname(target),
                )
                self._filesystem(filesystem.publish_directory, stage, target)
                installed = self._read_snapshot(reference.digest(), payloads=True)
                if installed is None:
                    raise StoreAuthorityError("snapshot install did not survive read-back.")
                if annotations is not None:
                    if annotations.object is not None:
                        if _before_annotation_write is not None:
                            _before_annotation_write("object")
                        self.write_metadata(reference.object, annotations.object)
                    if annotations.state is not None:
                        if _before_annotation_write is not None:
                            _before_annotation_write("state")
                        self.write_metadata(reference, annotations.state)
                return installed[1]
        finally:
            if os.path.isdir(stage):
                shutil.rmtree(stage, ignore_errors=True)

    def _read_snapshot_from_directory(self, directory: str, reference, *, payloads: bool):
        """Validate candidate staging using the same path-independent snapshot checks."""

        return self._read_snapshot(
            reference.digest(), payloads=payloads, directory=directory,
        )

    def read_declaration_record(self, digest: str) -> DeclarationRecord | None:
        record = self._read_file(self._declaration_path(digest), DeclarationRecord)
        if record is not None and record.digest != digest:
            raise StoreAuthorityError("DeclarationRecord digest does not match its direct path.")
        return record

    def write_declaration_record(self, record: DeclarationRecord) -> DeclarationRecord:
        """Refuse legacy DeclarationRecord with StoreCapabilityError before effects.

        Identity-3 publication uses publish_node and ObjectId projection anchors.
        """
        if not isinstance(record, DeclarationRecord):
            raise TypeError("record must be a DeclarationRecord.")
        self.preflight_publication(f"write {DeclarationRecord.schema}")
        path = self._declaration_path(record.digest)
        with interprocess_lock(self._writer_lock_path):
            existing = self._existing_immutable(
                path, record, DeclarationRecord,
            )
            if existing is not None:
                return existing
            self.mark_query_index_dirty()
            self._atomic_write(path, record.to_bytes())
            return record

    def iter_declaration_records(self) -> Iterable[DeclarationRecord]:
        """Yield validated declaration records in deterministic direct-path order."""
        root = Path(self.base_dir, "declarations")
        if not root.exists():
            return ()
        records = []
        for path in sorted(root.glob("*/*.record")):
            record = self._read_file(os.fspath(path), DeclarationRecord)
            if record is None or path.name != f"{record.digest}.record" or path.parent.name != record.digest[:2]:
                raise StoreAuthorityError(f"DeclarationRecord is stored under an invalid digest path: {path!s}.")
            records.append(record)
        return tuple(records)

    def read_claim_record(self, digest: str) -> ClaimRecord | None:
        record = self._read_file(self._claim_path(digest), ClaimRecord)
        if record is not None and record.object_digest != digest:
            raise StoreAuthorityError("ClaimRecord object digest does not match its direct path.")
        return record

    def write_claim_record(self, record: ClaimRecord) -> ClaimRecord:
        """Refuse legacy ClaimRecord with StoreCapabilityError before effects."""
        if not isinstance(record, ClaimRecord):
            raise TypeError("record must be a ClaimRecord.")
        self.preflight_publication("write claim")
        with interprocess_lock(self._writer_lock_path):
            self._atomic_write(self._claim_path(record.object_digest), record.to_bytes())
        return record

    def _ref_path(self, *parts: str) -> str:
        return os.path.join(self.base_dir, "refs", *parts)

    def read_main_ref(self) -> MainRefRecord | None:
        """Read the current main-definition mutable reference."""
        return self._read_file(self._ref_path("main.record"), MainRefRecord)

    def write_main_ref(self, record: MainRefRecord) -> MainRefRecord:
        """Refuse legacy main references with StoreCapabilityError before effects."""
        if not isinstance(record, MainRefRecord):
            raise TypeError("record must be a MainRefRecord.")
        self.preflight_publication("write main reference")
        with interprocess_lock(self._writer_lock_path):
            self._atomic_write(self._ref_path("main.record"), record.to_bytes())
        return record

    def read_object_alias(self, alias: str) -> ObjectAliasRecord | None:
        """Read one direct mutable object alias record."""
        return self._read_file(self._ref_path("objects", f"{alias}.record"), ObjectAliasRecord)

    def write_object_alias(self, record: ObjectAliasRecord) -> ObjectAliasRecord:
        """Refuse legacy ObjectRef aliases with StoreCapabilityError before effects."""
        if not isinstance(record, ObjectAliasRecord):
            raise TypeError("record must be an ObjectAliasRecord.")
        self.preflight_publication("write object alias")
        path = self._ref_path("objects", f"{record.alias}.record")
        with interprocess_lock(self._writer_lock_path):
            if self._read_file(path, ObjectAliasRecord) == record:
                return record
            self.mark_query_index_dirty()
            self._atomic_write(path, record.to_bytes())
        return record

    def read_state_alias(self, object_digest: str, alias: str) -> StateAliasRecord | None:
        """Read one direct mutable StateRef alias record."""
        return self._read_file(self._ref_path("states", object_digest[:2], object_digest, f"{alias}.record"), StateAliasRecord)

    def write_state_alias(self, record: StateAliasRecord) -> StateAliasRecord:
        """Refuse legacy StateRef aliases with StoreCapabilityError before effects."""
        if not isinstance(record, StateAliasRecord):
            raise TypeError("record must be a StateAliasRecord.")
        self.preflight_publication("write state alias")
        digest = record.object_ref.digest()
        path = self._ref_path(
            "states", digest[:2], digest, f"{record.alias}.record",
        )
        with interprocess_lock(self._writer_lock_path):
            if self._read_file(path, StateAliasRecord) == record:
                return record
            self.mark_query_index_dirty()
            self._atomic_write(path, record.to_bytes())
        return record

    def iter_object_alias_records(self) -> Iterable[ObjectAliasRecord]:
        """Yield validated object aliases in deterministic path order."""
        root = Path(self.base_dir, "refs", "objects")
        if not root.exists():
            return ()
        return tuple(
            record for path in sorted(root.glob("*.record"))
            if (record := self._read_file(os.fspath(path), ObjectAliasRecord)) is not None
        )

    def iter_state_alias_records(self) -> Iterable[StateAliasRecord]:
        """Yield validated scoped state aliases in deterministic path order."""
        root = Path(self.base_dir, "refs", "states")
        if not root.exists():
            return ()
        return tuple(
            record for path in sorted(root.glob("*/*/*.record"))
            if (record := self._read_file(os.fspath(path), StateAliasRecord)) is not None
        )

    def catalog_key(self) -> str:
        """Return the stable path-backed identity used only by derived indexes."""
        return f"{type(self).__module__}.{type(self).__qualname__}:{self.base_dir}"

    def close(self) -> None:
        """Release this handle's SQLite connections without touching authority.

        Raises:
            RuntimeError: If an active Session resource cache retains this Store.
        """
        super().close()
        self._state_owner_closed = True
        if self._query_index_instance is not None:
            self._query_index_instance.close()
            self._query_index_instance = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.base_dir!r})"


DirStore.publish_snapshot._dryml_annotation_phase_callback = True

# These are capability identities, not validity facts. Observers must leave them
# intact; unknown readers and subclasses use the Store compatibility seam.
_EXACT_READER_NAMES = (
    "read_state_ref_record", "read_snapshot_metadata", "iter_state_ref_records",
    "open_local_state", "validate_local_state", "_open_local_state_for_exact_load",
    "read_definition_record", "_read_definition_record", "_read_snapshot", "_validate_local_state_dir",
    "_validate_snapshot_locals", "_read_file", "_read_json", "_snapshot_path",
    "_require_deferred_local_state_support", "authority_fence_key",
    "authority_read_fence", "writer_lock", "_cut_snapshot_key",
    "_iter_state_ref_records", "_can_reuse_exact_validation",
    "_supports_deferred_local_state",
)
_EXACT_READERS = {DirStore: {name: getattr(DirStore, name) for name in _EXACT_READER_NAMES}}
_EXACT_PAYLOAD_VALIDATOR = LocalStateManifest.validate_payload
_EXACT_RECORD_VALIDATORS = tuple(
    (owner, name, getattr(owner, name).__func__)
    for owner in (StateRefRecord, DefinitionRecord, LocalStateManifest)
    for name in ("from_bytes", "from_data", "_decode_framed")
) + tuple((owner, "from_data", owner.from_data.__func__) for owner in (ObjectRef, StateRef))
