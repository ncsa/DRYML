"""Immutable content-addressed results and detached, freshly reopened read scopes."""

from dataclasses import dataclass
import hashlib
import os
import re
from uuid import UUID

from dryml.core.store.dir import DirStore
from dryml.core.store import authority as journal
from dryml.paths import local_path_key
from .errors import ManagedResultError
from .result_codec import RESULT_VERSION, MAX_RESULT_BYTES, decode_result


@dataclass(frozen=True, slots=True)
class StoreAcknowledgement:
    """Detached exact publication/ack evidence, never cached Store authority.

    store_uuid and optional mutation_id/retention_token are canonical UUID strings;
    epoch is a nonnegative int. Invalid values raise ManagedResultError. The token
    names a provisional Core publication when present. Confirmation freshly reads
    the owning Store; construction grants no authority or retention and does no IO.
    """
    store_uuid: str
    mutation_id: str | None
    epoch: int
    retention_token: str | None = None

    def __post_init__(self):
        try:
            if (type(self.epoch) is not int or self.epoch < 0 or
                    any(type(v) is not str or str(UUID(v)) != v for v in
                        (self.store_uuid, *(v for v in (self.mutation_id, self.retention_token) if v is not None)))):
                raise ValueError()
        except (ValueError, TypeError, AttributeError) as error:
            raise ManagedResultError("invalid_store_ack", "invalid publication acknowledgement evidence") from error

    def to_data(self):
        """Return closed acknowledgement v1 primitives without checking a Store."""
        return {"schema": "dryml-managed-store-ack", "version": 1,
            "store_uuid": self.store_uuid, "mutation_id": self.mutation_id,
            "epoch": self.epoch, "retention_token": self.retention_token}

    @classmethod
    def from_data(cls, data):
        """Decode closed acknowledgement metadata or raise ManagedResultError; no IO."""
        if (type(data) is not dict or set(data) != {"schema", "version", "store_uuid", "mutation_id", "epoch", "retention_token"}
                or data["schema"] != "dryml-managed-store-ack" or type(data["version"]) is not int or data["version"] != 1):
            raise ManagedResultError("invalid_store_ack", "unsupported acknowledgement schema")
        return cls(data["store_uuid"], data["mutation_id"], data["epoch"], data["retention_token"])

    def confirm(self, stores, *, expected_path=None):
        """Return the freshly confirmed connected Store or raise ManagedResultError.

        stores is an iterable of borrowed Stores. Exact mutation receipt status,
        UUID and epoch must agree and no pending mutation can stand in for ack.
        expected_path optionally requires that exact Store-relative contribution.
        A provisional token must belong to that receipt's prepared publication,
        never an independent adopted user root.
        No hooks, capture, restore or acknowledgement writes occur here.
        """
        from dryml.core.store.records import MutationIntentRecord
        store = next((v for v in stores if v.store_uuid == self.store_uuid), None)
        if store is None:
            raise ManagedResultError("ack_source_unavailable", "publication Store is disconnected")
        cut = store.authority_cut()
        if cut.epoch < self.epoch or self.mutation_id in cut.pending:
            raise ManagedResultError("ack_pending", "publication is not acknowledged")
        if self.mutation_id is not None:
            record = store._read_file(os.path.join(store.base_dir, "mutations", self.mutation_id + ".record"), MutationIntentRecord)
            if (record is None or record.intent["store_uuid"] != self.store_uuid
                    or record.intent["epoch"] != self.epoch or record.intent["status"] != "acknowledged"):
                raise ManagedResultError("ack_mismatch", "exact publication acknowledgement is absent")
            transitions = record.intent["transitions"]
            if expected_path is not None and not any(v["path"] == expected_path and v["after"] is not None for v in transitions):
                raise ManagedResultError("ack_mismatch", "acknowledgement belongs to another contribution")
            if self.retention_token is not None:
                from dryml.core.store.records import _decode_authority_bytes
                from dryml.core.store.retention import RetentionIntentRecord
                path = "retention/intents/" + self.retention_token + ".record"
                matches = [v for v in transitions if v["path"] == path and v["after"] is not None]
                if len(matches) != 1:
                    raise ManagedResultError("ack_mismatch", "provisional token belongs to another publication")
                intended = RetentionIntentRecord.from_bytes(_decode_authority_bytes(matches[0]["after"]))
                observed = store.retention.read(self.retention_token)
                if (intended.disposition != "prepared" or observed is None
                        or intended.resources != observed.resources or intended.graph_digest != observed.graph_digest):
                    raise ManagedResultError("ack_mismatch", "provisional publication evidence differs")
        return store


@dataclass(frozen=True, slots=True)
class ResultReference:
    """Typed immutable value address; no IO, resources or availability proof.

    key is lowercase SHA-256; kind/version select the sole supported result codec.
    Invalid fields raise ManagedResultError. This address grants no private state.
    """

    key: str
    kind: str = "dryml-managed-result"
    version: int = RESULT_VERSION

    def __post_init__(self):
        if (type(self.key) is not str or re.fullmatch(r"[0-9a-f]{64}", self.key) is None
                or self.kind != "dryml-managed-result" or type(self.version) is not int
                or self.version != RESULT_VERSION):
            raise ManagedResultError("invalid_result_reference", "unsupported result address")

    def to_data(self):
        """Return closed primitive reference fields, without reading the source."""
        return {"kind": self.kind, "version": self.version, "key": self.key}

    @classmethod
    def from_data(cls, data):
        """Decode the exact reference mapping or raise ManagedResultError; no IO."""
        if type(data) is not dict or set(data) != {"kind", "version", "key"}:
            raise ManagedResultError("invalid_result_reference", "malformed result reference")
        return cls(data["key"], data["kind"], data["version"])


@dataclass(frozen=True, slots=True)
class ResultSource:
    """Reopenable local DirStore path and UUID, never an open resource or proof.

    path is an absolute canonical directory and store_uuid its persistent authority
    UUID. Invalid descriptors raise ManagedResultError. Each read opens a fresh
    supported Store, verifies its UUID, protects the byte read and checks the hash;
    unavailable/replaced sources or missing/corrupt bytes raise ManagedResultError.
    Neither construction nor ordinary metadata inspection performs IO.
    """

    path: str
    store_uuid: str

    def __post_init__(self):
        try:
            valid = (type(self.path) is str and os.path.isabs(self.path)
                and type(self.store_uuid) is str and str(UUID(self.store_uuid)) == self.store_uuid)
        except ValueError:
            valid = False
        if not valid:
            raise ManagedResultError("invalid_result_source", "unsupported result source descriptor")

    def read(self, reference):
        """Return a newly decoded value from ResultReference under a fresh lifetime.

        Every access reopens and validates authority and bytes, even after prior
        successful decoding. Raises ManagedResultError on any unavailable, invalid
        or replaced source; never executes workload or serializes live resources.
        """
        if type(reference) is not ResultReference:
            raise ManagedResultError("invalid_result_reference", "expected typed result address")
        store = None
        try:
            if not os.path.isfile(os.path.join(self.path, "authority.record")):
                raise ManagedResultError("result_source_unavailable", "result source authority is absent")
            store = DirStore(self.path, query_index="none")
            if store.store_uuid != self.store_uuid:
                raise ManagedResultError("result_source_changed", "result source authority differs")
            payload = read_result_bytes(store, reference)
            return decode_result(payload)
        except ManagedResultError:
            raise
        except Exception as error:
            raise ManagedResultError("result_source_unavailable", "could not read the selected result source") from error
        finally:
            if store is not None:
                store.close()


def result_path(store, key):
    """Return the immutable result byte path for DirStore/key; performs no IO.

    Invalid keys raise ManagedResultError. Bytes live outside private StateKey
    storage and carry no state-retention grant.
    """
    ResultReference(key)
    return os.path.join(store.base_dir, "managed", "results", "v1", key[:2], key, "current.json")


def read_result_bytes(store, reference):
    """Read/hash-check typed result bytes under the selected Store read fence.

    store is borrowed DirStore, reference exact ResultReference. Returns bounded
    bytes; missing/corrupt content raises ManagedResultError. No decode/hooks run
    inside the fence. The open file protects the immutable read lifetime.
    """
    if type(reference) is not ResultReference:
        raise ManagedResultError("invalid_result_reference", "expected typed result reference")
    ResultReference(reference.key, reference.kind, reference.version)
    try:
        with store.authority_read_fence():
            with open(result_path(store, reference.key), "rb") as stream:
                payload = stream.read(MAX_RESULT_BYTES + 1)
        if len(payload) > MAX_RESULT_BYTES or hashlib.sha256(payload).hexdigest() != reference.key:
            raise ManagedResultError("result_corrupt", "result bytes differ from their content address")
        return payload
    except OSError as error:
        raise ManagedResultError("result_unavailable", "ordinary result bytes are unavailable") from error


def publish_result(store, payload, *, return_ack=False):
    """Install immutable encoded bytes via Store journal/projection/ack; return ref.

    store is the selected control DirStore; payload is already encoded/validated
    bytes. Collisions and publication/index failures propagate with mutation/cut
    evidence. return_ack=True returns (reference, StoreAcknowledgement) for the
    finalization intent. Preparation/decoding occur outside Store writers/control.
    """
    decode_result(payload)
    reference = ResultReference(hashlib.sha256(payload).hexdigest())
    path = result_path(store, reference.key)
    def prepare():
        old = journal.raw(store, store._relative(path))
        if old is not None and old != payload:
            raise ManagedResultError("result_collision", "immutable result address already contains different bytes")
        return ([journal.PreparedChange(store._relative(path), old, payload)] if old is None else [],
                ("managed-result:" + reference.key,), reference)
    identity, cut = store._prepare_and_apply(prepare, return_receipt=True)
    ack = publication_ack(store, identity)
    return (reference, ack) if return_ack else reference


def publication_ack(store, mutation_id, *, retention_token=None):
    """Detach the exact acknowledged journal epoch, never a later observed epoch.

    store is borrowed, mutation_id its UUID or None for no-op reuse, optional
    retention_token identifies provisional capture. Returns StoreAcknowledgement;
    missing or unacknowledged receipt raises ManagedResultError. No writes occur.
    """
    from dryml.core.store.records import MutationIntentRecord
    if mutation_id is None:
        epoch = store.authority_cut().epoch
    else:
        record = store._read_file(os.path.join(store.base_dir, "mutations", mutation_id + ".record"), MutationIntentRecord)
        if record is None:
            raise ManagedResultError("ack_missing", "publication mutation receipt is absent")
        epoch = record.intent["epoch"]
    ack = StoreAcknowledgement(store.store_uuid, mutation_id, epoch, retention_token)
    ack.confirm((store,))
    return ack


def source_descriptor(store):
    """Detach canonical path/Store UUID from an open selected DirStore; no new IO."""
    return ResultSource(local_path_key(store.base_dir), store.store_uuid)


class _ResultRecord:
    @classmethod
    def from_bytes(cls, payload):
        decode_result(payload)
        result = cls()
        result.key = hashlib.sha256(payload).hexdigest()
        return result

    def to_data(self):
        return ResultReference(self.key).to_data()
