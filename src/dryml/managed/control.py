"""Managed format-2 metadata authority and independently fenced short control CAS.

Logical ownership precedes state/Store work. Short control sections contain only
control byte checks/publication, never payload validation, hooks, lock probes,
Store acquisition, SQL, projection maintenance or acknowledgement.
"""

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import wraps
import hashlib
import json
import os
import re
import shutil
import tempfile
from uuid import UUID, uuid4

from dryml.core.node import Node, require_saved
from dryml.core._content_facts import _current_content_facts
from dryml.core.node_codec import decode_node_graph, encode_node_graph
from dryml.core.node_prepare import NodePreparation
from dryml.core.repo import Repo
from dryml.core.store import authority as journal
from dryml.core.store.dir import DirStore
from dryml.formats import canonical_json_bytes, canonical_json_load_bytes
from dryml.locking import FileLock, interprocess_lock
from dryml.paths import local_path_key

from .errors import ManagedConfigError, ManagedConflictError, ManagedControlError, ManagedResultError
from .identity import AdmissionIdentity, operation_digest, _receiver
from .storage import OperationAdmission, OperationOwnership
from .results import ResultReference, StoreAcknowledgement

_MAX_BYTES = 1024 * 1024
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_MAX_GENERATION = 2**63 - 1
_STATES = frozenset({"running", "interrupted", "failed", "finalizing", "completed"})


def _decoded_record(reader):
    @wraps(reader)
    def decode(cls, payload, *args, **kwargs):
        facts = _current_content_facts()
        build = lambda: reader(cls, payload, *args, **kwargs)
        operation_id = kwargs.get("operation_id")
        if (facts is None or type(payload) is not bytes or args or set(kwargs) - {"operation_id"}
                or operation_id is not None and type(operation_id) is not str):
            return build()
        return facts.decode((cls, reader, tuple(kwargs.items())), payload, build)
    return decode


def _digest(value, label):
    if type(value) is not str or not _HEX.fullmatch(value):
        raise ManagedControlError("invalid_current", label + " must be a lowercase SHA-256 digest")


def _uuid(value, label):
    try:
        valid = type(value) is str and UUID(value).hex == value
    except (ValueError, AttributeError):
        valid = False
    if not valid:
        raise ManagedControlError("invalid_current", label + " must be lowercase UUID hex")


def _generation(value):
    if type(value) is not int or not 1 <= value <= _MAX_GENERATION:
        raise ManagedControlError("invalid_current", "generation is outside its supported range")


def _bytes(data):
    try:
        payload = canonical_json_bytes(data, max_depth=96, max_nodes=65536,
            max_entries=16384, max_string=_MAX_BYTES, max_int_bits=64)
    except (TypeError, ValueError, ManagedConfigError) as error:
        raise ManagedControlError("invalid_control", "control metadata exceeds its closed grammar") from error
    if len(payload) > _MAX_BYTES:
        raise ManagedControlError("snapshot_too_large", "control metadata exceeds 1 MiB")
    return payload


def _data(payload):
    if type(payload) is not bytes or len(payload) > _MAX_BYTES:
        raise ManagedControlError("invalid_control", "control bytes exceed their support bound")
    try:
        def validate():
            canonical_json_load_bytes(payload, max_depth=96, max_nodes=65536,
                max_entries=16384, max_string=_MAX_BYTES, max_int_bits=64)
        facts = _current_content_facts()
        if facts is None:
            validate()
        else:
            facts.decode(("control-json", canonical_json_load_bytes), payload, validate)
        return json.loads(payload)
    except (ValueError, TypeError) as error:
        raise ManagedControlError("invalid_control", "control metadata is malformed") from error


def _closed(data, schema, version, fields):
    if (not isinstance(data, Mapping) or set(data) != set(fields) | {"schema", "version"}
            or data["schema"] != schema or type(data["version"]) is not int or data["version"] != version):
        raise ManagedControlError("unsupported_control", "unsupported " + schema + " schema/version")


def _node_data(node):
    if node is None:
        return None
    facts = _current_content_facts()
    if facts is not None:
        encoded = facts.decode(("control-node-encoding", node, encode_node_graph, NodePreparation), b"",
            lambda: _bytes({"schema": "dryml-managed-node", "version": 1,
                            "graph": encode_node_graph(NodePreparation((node,)))}))
        return json.loads(encoded)
    return {"schema": "dryml-managed-node", "version": 1,
            "graph": encode_node_graph(NodePreparation((node,)))}


def _node(data, *, saved=False):
    if data is None:
        return None
    facts = _current_content_facts()
    if facts is not None:
        return facts.decode(("node", saved), _bytes(data), lambda: _decode_node(data, saved=saved))
    return _decode_node(data, saved=saved)


def _decode_node(data, *, saved):
    _closed(data, "dryml-managed-node", 1, {"graph"})
    try:
        node = decode_node_graph(data["graph"])
        _receiver(node)
        if saved:
            require_saved(node)
        return node
    except (TypeError, ValueError, ManagedConfigError) as error:
        raise ManagedControlError("invalid_node", "incompatible nested managed Node metadata") from error


def _node_equal(left, right):
    if left is right:
        return True
    if left is None or right is None or not left.graph_equal(right):
        return False
    # Retained construction is separate from realized exact Node facets. Root
    # selections must keep both, including ties across unordered occurrences.
    from dryml.core import Definition
    def evidence(node):
        # This is a comparison projection, not author composition: realization
        # and retained construction may be conflicting views of one occurrence.
        # The prepared constructor preserves both instead of coalescing facets.
        pair = Definition._from_prepared_parameters
        nodes = NodePreparation((node,)).nodes
        pairs = frozenset(pair(None, {"node": n, "construction": getattr(n, "construction_view", n)})
                          for n in nodes)
        return pair(None, {"selection": node, "evidence": pairs})
    return evidence(left).graph_equal(evidence(right))


def _result_reference(data):
    try:
        return None if data is None else ResultReference.from_data(data)
    except ManagedResultError as error:
        raise ManagedControlError("invalid_result_reference", "invalid nested result association") from error


def _acknowledgement(data):
    try:
        return None if data is None else StoreAcknowledgement.from_data(data)
    except ManagedResultError as error:
        raise ManagedControlError("invalid_store_ack", "invalid nested publication acknowledgement") from error


def _same_snapshot_bytes(left, right):
    # Logical graph equality cannot authenticate a persisted packet's placement.
    return left is right or left is not None and right is not None and left.to_bytes() == right.to_bytes()


@dataclass(frozen=True, slots=True, eq=False)
class ControlHistory:
    """Retained metadata-only run/invocation outcome, independent of private data.

    run_id/invocation_id are UUID hex, generation is a positive fence, state is a
    lifecycle string, and checkpoint/final are optional SavedNodes. Invalid fields
    raise ManagedControlError. This immutable value performs no payload IO, grants
    no retention and never recursively embeds earlier history. Equality compares
    exact logical Node graphs and retained construction, independent of newly
    allocated handles/packet labels. result_reference is typed ResultReference,
    required for completed history, and does not promise current value availability.
    """
    run_id: str
    invocation_id: str
    generation: int
    state: str
    checkpoint: Node | None = None
    final: Node | None = None
    result_reference: ResultReference | None = None

    def __post_init__(self):
        _uuid(self.run_id, "run_id"); _uuid(self.invocation_id, "invocation_id")
        _generation(self.generation)
        if type(self.state) is not str or self.state not in _STATES:
            raise ManagedControlError("invalid_history", "unknown history state")
        if (self.state == "completed") != (self.final is not None):
            raise ManagedControlError("invalid_history", "only completed history requires final metadata")
        if (self.result_reference is not None and type(self.result_reference) is not ResultReference
                or self.state == "completed" and self.result_reference is None):
            raise ManagedControlError("invalid_history", "completion history requires typed result metadata")
        for node in (self.checkpoint, self.final):
            if node is not None:
                try:
                    _receiver(node)
                    require_saved(node)
                except (ValueError, TypeError, ManagedConfigError) as error:
                    raise ManagedControlError("invalid_history", "history requires saved metadata") from error

    def to_data(self):
        """Return closed history schema v2 primitives, without state reads/effects."""
        return {"schema": "dryml-managed-history", "version": 2,
                "run_id": self.run_id, "invocation_id": self.invocation_id,
                "generation": self.generation, "state": self.state,
                "checkpoint": _node_data(self.checkpoint), "final": _node_data(self.final),
                "result_reference": None if self.result_reference is None else self.result_reference.to_data()}

    def __eq__(self, other):
        return (type(other) is ControlHistory
                and (self.run_id, self.invocation_id, self.generation, self.state) ==
                    (other.run_id, other.invocation_id, other.generation, other.state)
                and _node_equal(self.checkpoint, other.checkpoint)
                 and _node_equal(self.final, other.final)
                 and self.result_reference == other.result_reference)

    __hash__ = None

    @classmethod
    def from_data(cls, data):
        """Decode history mapping; incompatible nested metadata raises ManagedControlError."""
        _closed(data, "dryml-managed-history", 2,
                {"run_id", "invocation_id", "generation", "state", "checkpoint", "final", "result_reference"})
        return cls(data["run_id"], data["invocation_id"], data["generation"], data["state"],
                   _node(data["checkpoint"], saved=True), _node(data["final"], saved=True),
                   _result_reference(data["result_reference"]))


@dataclass(frozen=True, slots=True, eq=False)
class ControlIntent:
    """Fenced root/publication disposition and exact affected-resource evidence.

    token/run_id/invocation_id are UUID hex, generation is its preparation fence,
    disposition is prepared/adopted/abandoned/release_pending/released, resources
    is a tuple of Core RetentionResource values. kind is root/initial/start/latest/
    final/durable; typed reasons require selection SavedNode metadata. Root intent
    v3 explicitly rejects v1/v2 nested input. publications contains exact typed
    provisional-token/Store acknowledgement evidence. Validation raises ManagedControlError.
    Equality compares full resource values and exact logical selection graphs,
    including retained construction evidence,
    not packet labels or digests alone. This ledger authorizes no physical release
    on its own and performs no IO.
    """
    token: str
    run_id: str
    invocation_id: str
    generation: int
    disposition: str
    resources: tuple = ()
    kind: str = "root"
    selection: Node | None = None
    publications: tuple[StoreAcknowledgement, ...] = ()

    def __post_init__(self):
        from dryml.core.store.retention import RetentionResource
        for value in (self.token, self.run_id, self.invocation_id): _uuid(value, "intent identity")
        _generation(self.generation)
        if (type(self.disposition) is not str or self.disposition not in
                {"prepared", "adopted", "abandoned", "release_pending", "released"}
                or type(self.resources) is not tuple or len(self.resources) > 256
                or any(type(value) is not RetentionResource for value in self.resources)):
            raise ManagedControlError("invalid_intent", "invalid managed root intent evidence")
        if type(self.kind) is not str or self.kind not in {"root", "initial", "start", "latest", "final", "durable"}:
            raise ManagedControlError("invalid_intent", "unknown root reason")
        if self.selection is not None:
            try:
                _receiver(self.selection)
                require_saved(self.selection)
            except (TypeError, ValueError, ManagedConfigError) as error:
                raise ManagedControlError("invalid_intent", "root selection requires saved metadata") from error
        if self.kind != "root" and self.selection is None:
            raise ManagedControlError("invalid_intent", "typed roots require saved selection metadata")
        if (type(self.publications) is not tuple or len(self.publications) > 256
                or any(type(v) is not StoreAcknowledgement or v.retention_token is None for v in self.publications)):
            raise ManagedControlError("invalid_intent", "invalid provisional publication evidence")

    def to_data(self):
        """Return closed intent v3 selection/publication evidence, without grant effects."""
        return {"schema": "dryml-managed-root-intent", "version": 3, "token": self.token,
                "run_id": self.run_id, "invocation_id": self.invocation_id,
                "generation": self.generation, "disposition": self.disposition,
                "resources": [value.to_data() for value in self.resources],
                "kind": self.kind, "selection": _node_data(self.selection),
                "publications": [v.to_data() for v in self.publications]}

    def __eq__(self, other):
        return (type(other) is ControlIntent
                and (self.token, self.run_id, self.invocation_id, self.generation,
                     self.disposition, self.resources, self.kind) ==
                    (other.token, other.run_id, other.invocation_id, other.generation,
                     other.disposition, other.resources, other.kind)
                 and _node_equal(self.selection, other.selection) and self.publications == other.publications)

    __hash__ = None

    @classmethod
    def from_data(cls, data):
        """Decode closed root intent mapping, rejecting incompatible resource versions."""
        from dryml.core.store.retention import RetentionResource
        _closed(data, "dryml-managed-root-intent", 3,
                {"token", "run_id", "invocation_id", "generation", "disposition", "resources", "kind", "selection", "publications"})
        if type(data["resources"]) is not list or type(data["publications"]) is not list:
            raise ManagedControlError("invalid_intent", "resources must be a list")
        try:
            return cls(data["token"], data["run_id"], data["invocation_id"], data["generation"],
                       data["disposition"], tuple(RetentionResource.from_data(v) for v in data["resources"]),
                       data["kind"], _node(data["selection"], saved=True),
                       tuple(StoreAcknowledgement.from_data(v) for v in data["publications"]))
        except (ValueError, TypeError, ManagedConfigError, ManagedResultError) as error:
            raise ManagedControlError("invalid_intent", "invalid affected resource metadata") from error


@dataclass(frozen=True, slots=True, eq=False)
class ControlSnapshot:
    """Closed current-record v4 authority with detached receiver, state and result metadata.

    receiver is a bound concrete Node, member a declared identifier, argument_digest
    separate compatibility evidence, run_id stable run UUID, invocation_id current
    caller UUID, generation a monotonic fence, state a lifecycle string. Optional
    checkpoint/final SavedNodes carry metadata, not availability promises;
    producer_invocation_id identifies completion separately from later callers.
    interrupt_request is an exact (run, invocation) pair. history/intents are bounded
    immutable tuples. Invalid schemas/cross-fields raise ManagedControlError.
    Construction validates and retains the immutable operation digest; replacement
    recomputes it. Equality compares exact logical Node graphs and their retained
    construction evidence, not packet labels
    or digest-only evidence. Decoding retains bounded immutable source bytes as a
    private, nonserialized fact, not proof of external authority. Replacement drops
    that fact; every CAS still checks fresh persisted bytes and ownership/fences.
    Codec methods read no payloads and grant no retention.
    result_reference is optional typed ResultReference and mandatory on completed
    records. Finalizing retains it before automatic capture; failed/interrupted
    records may preserve this evidence without claiming an ordinary success value.
    result_ack is optional StoreAcknowledgement, confirmed freshly before completion.
    """
    receiver: Node
    member: str
    argument_digest: str
    run_id: str
    invocation_id: str
    generation: int
    state: str
    interrupt_request: tuple[str, str] | None = None
    checkpoint: Node | None = None
    final: Node | None = None
    producer_invocation_id: str | None = None
    failure_code: str | None = None
    history: tuple[ControlHistory, ...] = ()
    intents: tuple[ControlIntent, ...] = ()
    result_reference: ResultReference | None = None
    result_ack: StoreAcknowledgement | None = None
    _operation_id: str = field(init=False, repr=False)
    _source_bytes: bytes | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        try: object.__setattr__(self, "_operation_id", operation_digest(self.receiver, self.member))
        except (ValueError, TypeError, ManagedConfigError) as error:
            raise ManagedControlError("invalid_current", "invalid operation projection/member") from error
        _digest(self.argument_digest, "argument_digest")
        _uuid(self.run_id, "run_id"); _uuid(self.invocation_id, "invocation_id")
        _generation(self.generation)
        if self.result_reference is not None and type(self.result_reference) is not ResultReference:
            raise ManagedControlError("invalid_current", "result association requires typed metadata")
        if self.result_ack is not None and type(self.result_ack) is not StoreAcknowledgement:
            raise ManagedControlError("invalid_current", "result requires typed acknowledgement evidence")
        if type(self.state) is not str or self.state not in _STATES:
            raise ManagedControlError("invalid_current", "unknown lifecycle state")
        if self.interrupt_request is not None and (type(self.interrupt_request) is not tuple
                or self.interrupt_request != (self.run_id, self.invocation_id)
                or self.state not in {"running", "finalizing"}):
            raise ManagedControlError("invalid_current", "request must address active run/invocation")
        if self.failure_code is not None and (type(self.failure_code) is not str
                or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", self.failure_code) is None):
            raise ManagedControlError("invalid_current", "failure code must be static and bounded")
        if (self.state == "failed") != (self.failure_code is not None):
            raise ManagedControlError("invalid_current", "only failed current requires a failure code")
        if self.state == "completed":
            if self.final is None or self.producer_invocation_id != self.invocation_id or self.result_reference is None:
                raise ManagedControlError("invalid_current", "completion requires final and producer invocation")
        elif self.final is not None or self.producer_invocation_id is not None:
            raise ManagedControlError("invalid_current", "noncompletion cannot claim final/producer")
        for node in (self.checkpoint, self.final):
            if node is not None:
                try:
                    _receiver(node)
                    require_saved(node)
                    if not self.receiver.object_projection_equal(node):
                        raise ValueError("projection mismatch")
                except (ValueError, TypeError, ManagedConfigError) as error:
                    raise ManagedControlError("invalid_current", "saved association does not match receiver") from error
        for values, cls in ((self.history, ControlHistory), (self.intents, ControlIntent)):
            if type(values) is not tuple or len(values) > 256 or any(type(v) is not cls for v in values):
                raise ManagedControlError("invalid_current", "invalid bounded history/intent ledger")
        for record in self.history:
            if record.generation > self.generation or any(
                    node is not None and not self.receiver.object_projection_equal(node)
                    for node in (record.checkpoint, record.final)):
                raise ManagedControlError("invalid_history", "history does not belong to this operation generation/projection")
        if any(value.generation > self.generation for value in self.intents):
            raise ManagedControlError("invalid_intent", "intent preparation fence is newer than authority")
        if len({v.token for v in self.intents}) != len(self.intents):
            raise ManagedControlError("invalid_current", "duplicate managed intent tokens")
        initials = [v for v in self.intents if v.kind == "initial"]
        if len(initials) > 1 or any(v.selection is not None and not self.receiver.object_projection_equal(v.selection)
                                  for v in self.intents):
            raise ManagedControlError("invalid_intent", "root selection does not belong to operation")

    @property
    def initial(self):
        """Return optional immutable initial SavedNode metadata, without payload IO."""
        return next((v.selection for v in self.intents if v.kind == "initial"), None)

    @property
    def operation_id(self):
        """Return the constructor-validated digest without preparation, Store or payload access."""
        return self._operation_id

    def to_data(self):
        """Return detached current v4 primitives, preserving validated source packets.

        Decoded records retain their original nested storage labels; new records
        encode valid Node packets. The returned mutable mapping is not retained.
        No private data or external authority is read.
        """
        if self._source_bytes is not None:
            return _data(self._source_bytes)
        return {"schema": "dryml-managed-current", "version": 4,
                "operation_id": self.operation_id, "receiver": _node_data(self.receiver),
                "member": self.member, "argument_digest": self.argument_digest,
                "run_id": self.run_id, "invocation_id": self.invocation_id,
                "generation": self.generation, "state": self.state,
                "interrupt_request": None if self.interrupt_request is None else list(self.interrupt_request),
                "checkpoint": _node_data(self.checkpoint), "final": _node_data(self.final),
                "producer_invocation_id": self.producer_invocation_id, "failure_code": self.failure_code,
                "history": [v.to_data() for v in self.history], "intents": [v.to_data() for v in self.intents],
                "result_reference": None if self.result_reference is None else self.result_reference.to_data(),
                "result_ack": None if self.result_ack is None else self.result_ack.to_data()}

    def to_bytes(self):
        """Return validated source bytes or bounded canonical JSON for a new value.

        Source bytes are immutable data, not a cached authority check. New values
        may use different Node packet labels for logically equal unordered graphs.
        Oversized new records raise ManagedControlError; no IO occurs.
        """
        if self._source_bytes is not None:
            return self._source_bytes
        return _bytes(self.to_data())

    @classmethod
    @_decoded_record
    def from_bytes(cls, payload, *, operation_id=None):
        """Decode bytes and optional path digest, rejecting every incompatible nested codec.

        Returns ControlSnapshot; malformed/old records raise ManagedControlError
        without modifying source bytes or inspecting any referenced private data.
        payload is exact bytes; operation_id is an optional path-bound digest.
        Within the caller's admission, identical bytes/path constraints may reuse a
        bounded immutable decode. This is never a current-authority or payload
        witness; no IO occurs and callers must freshly read their source bytes.
        """
        data = _data(payload)
        fields = {"operation_id", "receiver", "member", "argument_digest", "run_id", "invocation_id",
                  "generation", "state", "interrupt_request", "checkpoint", "final",
                  "producer_invocation_id", "failure_code", "history", "intents", "result_reference", "result_ack"}
        _closed(data, "dryml-managed-current", 4, fields)
        request = data["interrupt_request"]
        if request is not None and (type(request) is not list or len(request) != 2):
            raise ManagedControlError("invalid_current", "malformed interruption request")
        if type(data["history"]) is not list or type(data["intents"]) is not list:
            raise ManagedControlError("invalid_current", "malformed history/intent list")
        result = cls(receiver=_node(data["receiver"]), member=data["member"], argument_digest=data["argument_digest"],
                     run_id=data["run_id"], invocation_id=data["invocation_id"], generation=data["generation"],
                     state=data["state"], interrupt_request=None if request is None else tuple(request),
                     checkpoint=_node(data["checkpoint"], saved=True), final=_node(data["final"], saved=True),
                     producer_invocation_id=data["producer_invocation_id"], failure_code=data["failure_code"],
                     history=tuple(ControlHistory.from_data(v) for v in data["history"]),
                      intents=tuple(ControlIntent.from_data(v) for v in data["intents"]),
                      result_reference=_result_reference(data["result_reference"]),
                      result_ack=_acknowledgement(data["result_ack"]))
        if result.operation_id != data["operation_id"] or (operation_id is not None and result.operation_id != operation_id):
            raise ManagedControlError("invalid_current", "operation projection does not match record/path identity")
        object.__setattr__(result, "_source_bytes", payload)
        return result

    def __eq__(self, other):
        return (type(other) is ControlSnapshot
                and (self.member, self.argument_digest, self.run_id, self.invocation_id,
                     self.generation, self.state, self.interrupt_request,
                     self.producer_invocation_id, self.failure_code, self.history, self.intents) ==
                    (other.member, other.argument_digest, other.run_id, other.invocation_id,
                     other.generation, other.state, other.interrupt_request,
                     other.producer_invocation_id, other.failure_code, other.history, other.intents)
                and _node_equal(self.receiver, other.receiver)
                and _node_equal(self.checkpoint, other.checkpoint)
                 and _node_equal(self.final, other.final)
                 and self.result_reference == other.result_reference and self.result_ack == other.result_ack)

    __hash__ = None


@dataclass(frozen=True, slots=True)
class _PendingIntent:
    operation_id: str
    previous_generation: int | None
    previous_digest: str | None
    proposed_generation: int
    proposed_digest: str
    token: str

    def __post_init__(self):
        _digest(self.operation_id, "operation_id"); _digest(self.proposed_digest, "proposed_digest")
        _uuid(self.token, "pending token"); _generation(self.proposed_generation)
        if (self.previous_generation is None) != (self.previous_digest is None):
            raise ManagedControlError("invalid_pending", "incomplete previous authority")
        if self.previous_generation is not None:
            _generation(self.previous_generation); _digest(self.previous_digest, "previous_digest")
        if self.proposed_generation != (self.previous_generation or 0) + 1:
            raise ManagedControlError("invalid_pending", "pending generations are not adjacent")

    def to_bytes(self):
        return _bytes({"schema": "dryml-managed-pending", "version": 2,
                       **{name: getattr(self, name) for name in self.__dataclass_fields__}})

    @classmethod
    def from_bytes(cls, payload, *, operation_id):
        data = _data(payload)
        _closed(data, "dryml-managed-pending", 2, cls.__dataclass_fields__)
        result = cls(**{name: data[name] for name in cls.__dataclass_fields__})
        if result.operation_id != operation_id:
            raise ManagedControlError("invalid_pending", "pending operation differs from path")
        return result


@dataclass(frozen=True, slots=True)
class StartPreparation:
    """Durable two-phase run/root association, never a transferable restore receipt.

    previous/proposed are exact current-record fingerprints encoded as snapshots;
    token names the Core affected-resource ledger. phase is prepared, ready,
    admitted or abandoned; settled acknowledges resource reconciliation. Decoding
    rejects incompatible nested metadata without opening state. No hooks run here.
    Validated source bytes are a bounded immutable nonserialized fact, discarded
    by replacement. Nested snapshots preserve their own exact byte evidence. The
    fact confers no authority: publication freshly checks sidecar/current bytes.
    """

    token: str
    previous: ControlSnapshot | None
    proposed: ControlSnapshot
    phase: str = "prepared"
    settled: bool = False
    _source_bytes: bytes | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        _uuid(self.token, "preparation token")
        if (type(self.proposed) is not ControlSnapshot
                or self.previous is not None and type(self.previous) is not ControlSnapshot
                or type(self.phase) is not str or self.phase not in {"prepared", "ready", "admitted", "abandoned"}
                or type(self.settled) is not bool
                or self.settled and self.phase not in {"admitted", "abandoned"}):
            raise ManagedControlError("invalid_preparation", "invalid start preparation")
        old = self.previous
        if (self.proposed.generation != (0 if old is None else old.generation) + 1
                or self.proposed.state not in {"running", "finalizing"}
                or old is not None and self.proposed.operation_id != old.operation_id):
            raise ManagedControlError("invalid_preparation", "preparation does not name adjacent operation authority")

    def to_data(self):
        """Return closed start-intent v1 primitives for metadata-only projection."""
        return {"schema": "dryml-managed-start", "version": 1, "token": self.token,
            "previous": None if self.previous is None else self.previous.to_data(),
            "proposed": self.proposed.to_data(), "phase": self.phase, "settled": self.settled}

    def to_bytes(self):
        """Return bounded closed start-intent v1 bytes, including exact CAS evidence."""
        if self._source_bytes is not None:
            return self._source_bytes
        return _bytes(self.to_data())

    @classmethod
    @_decoded_record
    def from_bytes(cls, payload, *, operation_id):
        """Decode exact bounded start evidence for a required path-bound operation.

        payload is bytes and operation_id a lowercase digest. Returns immutable
        StartPreparation; malformed/version-incompatible or mismatched nested
        evidence raises ManagedControlError without changing source bytes. A
        caller-local bounded content scope may reuse identical-byte decodes, never
        current authority. No Store IO, hooks, restoration or capture occurs.
        """
        data = _data(payload)
        _closed(data, "dryml-managed-start", 1, {"token", "previous", "proposed", "phase", "settled"})
        result = cls(data["token"], None if data["previous"] is None else
            ControlSnapshot.from_bytes(_bytes(data["previous"]), operation_id=operation_id),
            ControlSnapshot.from_bytes(_bytes(data["proposed"]), operation_id=operation_id),
            data["phase"], data["settled"])
        object.__setattr__(result, "_source_bytes", payload)
        return result


class ManagedControlStore:
    """Borrow one exact DirStore and Repo for detached managed control authority.

    control_store owns format-2 managed records; state_repo is borrowed for later
    execution integration, never read by status or admission. Invalid handles raise
    ManagedControlError. Construction neither opens/closes resources nor creates
    a namespace. Mutation journals/invalidation precede short CAS and projection/
    SQL/ack follow it. The lifetime operation lock is separate from short control.
    """

    def __init__(self, control_store, state_repo):
        if type(control_store) is not DirStore or not isinstance(state_repo, Repo):
            raise ManagedControlError("invalid_store", "control requires exact DirStore and borrowed Repo")
        self.control_store, self.state_repo = control_store, state_repo
        self._domain = control_store.store_uuid, local_path_key(control_store.base_dir)

    @property
    def root(self):
        """Return managed namespace path (str), without creating or reading it."""
        return os.path.join(self.control_store.base_dir, "managed")

    def initialize(self):
        """Publish fresh format-2 area atomically; return None.

        Old/malformed existing authority raises ManagedControlError unchanged.
        Namespace bootstrap takes only a Store writer, never short control or an
        operation lock. Existing valid areas require no writer acquisition.
        """
        if self._namespace_present(): return
        self.control_store.preflight_publication("managed area", record_family="dryml-node-graph/v1")
        with self.control_store.writer_lock():
            if self._namespace_present(): return
            staging = tempfile.mkdtemp(prefix=".managed-staging-", dir=self.control_store.base_dir)
            try:
                self._replace_file(os.path.join(staging, "format.json"), _bytes({"schema": "dryml-managed", "version": 2}))
                os.makedirs(os.path.join(staging, "owners", "v2"))
                from dryml import filesystem
                filesystem.publish_directory(staging, self.root)
            finally:
                if os.path.isdir(staging): shutil.rmtree(staging)

    def admission(self, identity, *, check_arguments=True):
        """Acquire nonblocking metadata-only logical ownership and return its lifetime.

        identity is AdmissionIdentity; check_arguments defaults True and compares
        separate compatibility evidence, False is reserved for explicit higher-
        level new-run preparation. Returns OperationAdmission with stable run and
        fresh invocation UUID. Duplicate calls raise ManagedConflictError immediately;
        incompatible/pending authority raises ManagedControlError. No receiver or
        checkpoint is materialized. Caller must release/use the context manager.
        """
        if type(identity) is not AdmissionIdentity or type(check_arguments) is not bool:
            raise ManagedControlError("invalid_admission", "admission requires closed detached identity facts")
        self.initialize()
        path = self._owner_path(identity.operation_id)
        from dryml import filesystem
        filesystem.ensure_directory(os.path.dirname(path))
        owner = OperationOwnership(self._domain, identity.operation_id, path)
        try:
            with self._control_lock():
                _, pending = self._capture(identity.operation_id)
            recovered = pending is not None or self._pending_mutations(identity.operation_id)
            current = self.reconcile(identity.operation_id, owner=owner)
            if current is not None:
                if not current.receiver.object_projection_equal(identity.receiver) or current.member != identity.member:
                    raise ManagedConflictError("operation_collision", "operation projection evidence differs")
                if check_arguments and current.argument_digest != identity.argument_digest:
                    raise ManagedConflictError("argument_mismatch", "ordinary arguments require explicit new-run preparation")
            return OperationAdmission(identity, owner, current, reconciled=bool(recovered))
        except BaseException:
            owner.release()
            raise

    def inspect(self, operation_id):
        """Return ControlSnapshot or None, using only detached historical metadata.

        operation_id is a lowercase digest. Reads take only the independent short
        control guard; old/pending/corrupt metadata raises ManagedControlError.
        No owner probe, payload IO, Store/StateKey lock or repair occurs.
        """
        _digest(operation_id, "operation_id")
        if not self._namespace_present(): return None
        with self._control_lock():
            current, pending = self._capture(operation_id)
        if pending is not None:
            _PendingIntent.from_bytes(pending, operation_id=operation_id)
            raise ManagedControlError("pending_reconciliation", "managed current has a pending replacement")
        return None if current is None else ControlSnapshot.from_bytes(current, operation_id=operation_id)

    def read_preparation(self, operation_id):
        """Return optional detached StartPreparation; no realization or cleanup occurs.

        operation_id is a digest. This retained intent is separate from current so
        first-start/rerun preparation cannot displace old authority. Invalid bytes
        raise ManagedControlError and remain intact.
        """
        _digest(operation_id, "operation_id")
        payload = self._read_optional(self._preparation_path(operation_id))
        return None if payload is None else StartPreparation.from_bytes(payload, operation_id=operation_id)

    def write_preparation(self, proposed, *, owner, expected):
        """Journal an exact start-intent CAS under lifetime ownership, returning it.

        proposed/expected are StartPreparation/optional old intent. Graph encoding
        happens before Store fences. Prepared-to-ready validation permits only
        root-intent changes, comparing unchanged fields logically when needed;
        the persisted sidecar CAS still uses exact validated source bytes.
        Stale ownership/evidence fails closed; IO or
        projection ambiguity preserves the affected-resource ledger. No hooks or
        state leases are acquired under a writer or short control section.
        """
        operation_id = proposed.proposed.operation_id
        self._require_owner(owner, operation_id)
        if expected is None or expected.settled:
            if (proposed.phase != "prepared" or proposed.settled
                    or proposed.proposed.invocation_id != owner.invocation_id
                    or expected is not None and expected.token == proposed.token):
                raise ManagedConflictError("preparation_conflict", "fresh work requires a fresh preparation token")
        else:
            phases = {"prepared": {"ready", "abandoned"}, "ready": {"admitted", "abandoned"},
                      "admitted": {"admitted"}, "abandoned": {"abandoned"}}
            same_proposal = _same_snapshot_bytes(proposed.proposed, expected.proposed)
            if expected.phase == "prepared" and proposed.phase == "ready" and not same_proposal:
                # Replacement drops the source witness: this is an invariant
                # comparison, not a manufactured persisted-byte fingerprint.
                same_proposal = replace(proposed.proposed, intents=expected.proposed.intents) == expected.proposed
            if (proposed.token != expected.token or not _same_snapshot_bytes(proposed.previous, expected.previous)
                    or proposed.phase not in phases[expected.phase] or not same_proposal):
                raise ManagedConflictError("preparation_conflict", "preparation dispositions are irreversible")
        path = self._preparation_path(operation_id)
        payload = proposed.to_bytes()
        old = None if expected is None else expected.to_bytes()
        def prepare():
            if self._read_optional(path) != old:
                raise ManagedConflictError("preparation_conflict", "start intent changed")
            return [journal.PreparedChange(self.control_store._relative(path), old, payload)], ("managed-start:" + operation_id,), proposed
        # Generic Store publication owns journaling, index invalidation and ack;
        # the operation owner is the only writer of this preparation sidecar.
        self.control_store._prepare_and_apply(prepare)
        return proposed

    def admit_prepared(self, preparation, *, owner):
        """Admit a ready run/root association under its invocation and exact byte CAS.

        preparation is persisted StartPreparation. Required grant work and any
        selected-start restoration run outside fences before phase ready. The
        exact previous current (including interrupt generation) must still match.
        Returns committed ControlSnapshot; uncertain writes retain both sides for
        reconciliation. Checkpoint/initial/final roots reuse this same two-phase
        primitive without restoring or allocating a new run.
        """
        proposed = preparation.proposed
        self._require_owner(owner, proposed.operation_id)
        persisted = self.read_preparation(proposed.operation_id)
        if (preparation.phase != "ready" or preparation.settled
                or proposed.invocation_id != owner.invocation_id
                or persisted is None or persisted.to_bytes() != preparation.to_bytes()):
            raise ManagedConflictError("preparation_conflict", "start was not prepared by this invocation")
        if not _same_snapshot_bytes(self.inspect(proposed.operation_id), preparation.previous):
            raise ManagedConflictError("generation_conflict", "authority or interrupt fence changed during preparation")
        self._publish(preparation.previous, proposed)
        owner.run_id = proposed.run_id
        return proposed

    def _preparation_path(self, operation_id):
        return os.path.join(self.root, "operations", "v2", "starts", "v1",
                            operation_id[:2], operation_id, "current.json")

    def status(self, operation_id):
        """Return metadata-only ManagedStatus, without taking/waiting on ownership.

        operation_id selects this control domain. Proven owner loss is projected
        as failed/owner_lost, without writing current. Invalid authority/probe IO
        raises ManagedControlError; saved payload availability is irrelevant.
        """
        from .model import ManagedStatus
        snapshot, lost, stable = self.observe_running_owner(operation_id)
        if not stable:
            raise ManagedConflictError("generation_conflict", "authority changed during owner observation")
        if snapshot is None:
            return ManagedStatus("not_started", operation_id, None, None, 0, None, None, False, None, None)
        return ManagedStatus("failed" if lost else snapshot.state, operation_id, snapshot.run_id,
            snapshot.invocation_id, snapshot.generation, snapshot.checkpoint, snapshot.final,
            snapshot.interrupt_request is not None, "owner_lost" if lost else snapshot.failure_code,
            snapshot.producer_invocation_id)

    def list_checkpoints(self, operation_id, *, include_released=False):
        """Return detached root associations for one operation, without payload IO.

        operation_id is a digest; include_released is an exact bool, default False.
        Returns tuple[ControlIntent], including kind, saved selection, token and
        affected full-key resources. Released metadata remains inspectable with
        include_released=True; it confers no availability. Missing operations return
        (). Corrupt/pending authority raises ManagedControlError. No owner wait,
        serialization, historical inventory scan or retention change occurs.
        """
        if type(include_released) is not bool:
            raise ManagedControlError("invalid_cleanup", "include_released must be bool")
        current = self.inspect(operation_id)
        return () if current is None else tuple(root for root in current.intents
            if include_released or root.disposition != "released")

    def cleanup_checkpoints(self, operation_id, *, tokens=None):
        """Explicitly retire selected operation roots, preserving completion/history.

        operation_id is a digest. tokens is an optional iterable of root UUID hex
        strings from list_checkpoints(); None selects every remaining association,
        including durable/initial/latest/final roots. Returns per-Store/full-key
        StateCleanupOutcomes, continuing independently safe targets after failure.
        Unknown tokens fail before effects. Active owners raise ManagedConflictError
        immediately; this method never interrupts or waits for them. Grant/control/
        index failures during root retirement propagate with cleanup_outcomes and
        durable release-pending evidence. Deletion-phase errors are failed outcomes.
        Only physical garbage failure returns cleanup-pending. Ordinary
        results, historical references and lifecycle state are never deleted or
        rewritten as unfinished; subsequent loads revalidate availability.
        """
        current = self.inspect(operation_id)
        if current is None:
            if tokens is not None and tuple(tokens):
                raise ManagedConflictError("unknown_checkpoint", "unknown checkpoint token")
            return ()
        with self.state_repo.retain_topology():
            with OperationOwnership(self._domain, operation_id, self._owner_path(operation_id)) as owner:
                self.reconcile(operation_id, owner=owner)
                from .storage import _cleanup_checkpoints
                return _cleanup_checkpoints(self, owner, operation_id, tokens=tokens)

    def _retire_checkpoint_roots(self, current, tokens, disposition, *, owner):
        self._require_owner(owner, current.operation_id)
        if not _same_snapshot_bytes(self.inspect(current.operation_id), current):
            raise ManagedConflictError("generation_conflict", "cleanup authority changed")
        proposed = replace(current, generation=current.generation + 1,
            intents=tuple(replace(root, disposition=disposition) if root.token in tokens else root
                          for root in current.intents))
        # Cleanup owns only root disposition. Producer/run/result/recovery metadata
        # stays historical, including when this owner is not the producer invocation.
        self._publish(current, proposed)
        return proposed

    def create_initial(self, snapshot, *, owner):
        """Commit generation-one snapshot under its exact admission token.

        snapshot is ControlSnapshot; owner is its active OperationAdmission.
        Returns committed metadata. Conflicts raise ManagedConflictError; partial
        journal/CAS/projection errors retain evidence and propagate. No body/hooks
        run, and no state-availability/retention policy is implied by this primitive.
        """
        self._require_owner(owner, snapshot.operation_id)
        if (snapshot.generation != 1 or snapshot.run_id != owner.run_id
                or snapshot.invocation_id != owner.invocation_id):
            raise ManagedConflictError("owner_conflict", "initial record differs from admitted identities")
        self._publish(None, snapshot)
        return snapshot

    def transition(self, operation_id, proposed, *, owner, expected_generation,
                   expected_run_id, expected_invocation_id):
        """CAS an exact next-generation owner snapshot; return committed ControlSnapshot.

        operation_id/proposed select immutable operation evidence. owner is active
        admission; expected_generation/run_id/invocation_id are mandatory fences.
        Stale inputs raise ManagedConflictError before effects; malformed authority
        raises ManagedControlError. Journal/projection failures retain mutation/cut
        evidence. This policy-free primitive does not implement rerun preparation.
        """
        current = self.inspect(operation_id)
        self._require_owner(owner, operation_id)
        _uuid(expected_run_id, "expected_run_id")
        _uuid(expected_invocation_id, "expected_invocation_id")
        _generation(expected_generation)
        if (current is None or (current.run_id, current.invocation_id, current.generation) !=
                (expected_run_id, expected_invocation_id, expected_generation)):
            raise ManagedConflictError("owner_conflict", "selected run/invocation/generation changed")
        if (proposed.operation_id != operation_id or proposed.generation != current.generation + 1
                or proposed.invocation_id != owner.invocation_id
                or not proposed.receiver.object_projection_equal(current.receiver)):
            raise ManagedConflictError("generation_conflict", "replacement is not the next exact operation generation")
        if current.interrupt_request is not None and proposed.state == "completed":
            raise ManagedConflictError("interrupted", "accepted interruption fences completion")
        self._validate_intent_transition(current, proposed)
        self._publish(current, proposed)
        if isinstance(owner, OperationAdmission):
            owner.run_id = proposed.run_id
        return proposed

    def transition_running_owner(self, operation_id, *, owner, run_id, invocation_id,
                                 expected_generation, build):
        """Build and CAS one fenced running update, with build outside short locks.

        owner and run/invocation/generation must all match current before build is
        called. build(ControlSnapshot) returns its replacement or None. Returns that
        committed snapshot/None. Stale calls raise ManagedConflictError, including
        an accepted interrupt. Builders may perform no physical resource release;
        use authorize_release first. CAS rechecks exact bytes after builder effects.
        """
        current = self.inspect(operation_id)
        self._fence(current, operation_id, owner, run_id, invocation_id, expected_generation)
        if current.state != "running" or current.interrupt_request is not None:
            raise ManagedConflictError("owner_conflict", "running invocation has been fenced")
        proposed = build(current)
        if proposed is None: return None
        return self.transition(operation_id, proposed, owner=owner, expected_generation=expected_generation,
                               expected_run_id=run_id, expected_invocation_id=invocation_id)

    def complete_recovered(self, expected, final, *, owner):
        """Finish proven result/final evidence under a new recovery-only owner.

        expected is an exact finalizing ControlSnapshot and final its associated
        SavedNode, confirmed by the result owner before entry. Returns completed
        metadata preserving original producer invocation. Fresh current bytes and
        interrupt generation are CAS-checked; ambiguity/index failure propagates.
        No hooks, payloads, grants or graph preparation run under short control.
        """
        self._require_owner(owner, expected.operation_id)
        if (expected.state != "finalizing" or expected.interrupt_request is not None
                or expected.result_reference is None
                or not _same_snapshot_bytes(self.inspect(expected.operation_id), expected)):
            raise ManagedConflictError("generation_conflict", "recovered completion was fenced")
        proposed = replace(expected, generation=expected.generation + 1, state="completed",
            final=final, producer_invocation_id=expected.invocation_id)
        self._publish(expected, proposed)
        return proposed

    def recover_final_association(self, preparation, *, owner):
        """CAS a fully proven finalizing preparation under recovery ownership.

        preparation is an exact persisted ready StartPreparation with result and
        adopted final evidence freshly confirmed by its owning runtime. Returns
        proposed metadata, preserving producer identity. Changed bytes/interrupt
        fence refuse; journal/index errors retain exact evidence. No hooks run.
        """
        proposed = preparation.proposed
        self._require_owner(owner, proposed.operation_id)
        persisted = self.read_preparation(proposed.operation_id)
        if (preparation.phase != "ready" or preparation.settled or persisted is None
                or persisted.to_bytes() != preparation.to_bytes()
                or preparation.previous is None or preparation.previous.state != "finalizing"
                or preparation.previous.interrupt_request is not None or proposed.state != "finalizing"
                or proposed.result_reference is None
                or not _same_snapshot_bytes(self.inspect(proposed.operation_id), preparation.previous)):
            raise ManagedConflictError("generation_conflict", "recovered final association was fenced")
        self._publish(preparation.previous, proposed)
        return proposed

    def authorize_release(self, operation_id, token, *, owner, run_id, invocation_id, expected_generation):
        """Return exact release-pending ControlIntent after active owner/fence checks.

        token is a UUID intent key. All run/invocation/generation preconditions are
        mandatory. A stale actor or another invocation's intent raises
        ManagedConflictError. This is metadata authorization only: no grant removal
        occurs, and higher-level release must retain ownership across its handshake.
        """
        current = self.inspect(operation_id)
        self._fence(current, operation_id, owner, run_id, invocation_id, expected_generation)
        for intent in current.intents:
            if (intent.token == token and intent.run_id == run_id and intent.invocation_id == invocation_id
                    and intent.disposition == "release_pending"):
                return intent
        raise ManagedConflictError("intent_conflict", "intent is not this invocation's release authorization")

    def probe_operation_owner(self, operation_id):
        """Return bool for a nonblocking lifetime-lock availability probe, before control.

        operation_id names retained ownership evidence. Missing evidence or native
        IO errors raise ManagedControlError, never imply death. A free probe is
        not owner-loss proof until run/invocation/generation are reread unchanged.
        No lock file is created/unlinked by this inspection.
        """
        _digest(operation_id, "operation_id")
        path = self._owner_path(operation_id)
        if not os.path.isfile(path):
            raise ManagedControlError("missing_owner_evidence", "operation lifetime lock evidence is absent")
        lock = FileLock(path)
        acquired = lock.acquire(blocking=False)
        if acquired: lock.release()
        return acquired

    def observe_running_owner(self, operation_id):
        """Return (snapshot, ownerless, stable) from an outside-control operation probe.

        A free probe followed by unchanged run/invocation/generation proves loss;
        changed authority returns stable=False. Never waits for a lifetime lock or
        infers death from elapsed time. Read/probe errors propagate without repair.
        """
        initial = self.inspect(operation_id)
        if initial is None or initial.state not in {"running", "finalizing"}:
            return initial, False, True
        free = self.probe_operation_owner(operation_id)
        observed = self.inspect(operation_id)
        stable = observed is not None and (observed.run_id, observed.invocation_id, observed.generation, observed.state) == (
            initial.run_id, initial.invocation_id, initial.generation, initial.state)
        return observed, free and stable, stable

    def request_interrupt_if_running(self, operation_id, *, expected_run_id=None,
                                     expected_invocation_id=None, expected_generation=None):
        """Accept an external targeted interruption without acquiring lifetime ownership.

        Optional expected_run_id/invocation_id UUIDs and generation fence current
        caller intent. Returns (outcome, snapshot): requested, already_requested,
        stale_run, stale_invocation, stale_generation, not_running, or changed.
        A request advances generation and fences the old owner's checkpoint/success.
        Probe occurs before control; journal/index phases never overlap control.
        Malformed inputs/authority raise ManagedControlError; publication failures
        preserve exact mutation/cut evidence rather than report acceptance.
        """
        if expected_run_id is not None: _uuid(expected_run_id, "expected_run_id")
        if expected_invocation_id is not None: _uuid(expected_invocation_id, "expected_invocation_id")
        if expected_generation is not None: _generation(expected_generation)
        initial = self.inspect(operation_id)
        if initial is None or initial.state not in {"running", "finalizing"}: return "not_running", initial
        for expected, actual, label in ((expected_run_id, initial.run_id, "run"),
                (expected_invocation_id, initial.invocation_id, "invocation"),
                (expected_generation, initial.generation, "generation")):
            if expected is not None and expected != actual: return "stale_" + label, initial
        observed, lost, stable = self.observe_running_owner(operation_id)
        if not stable: return "changed", observed
        if lost: return "not_running", observed
        if observed.interrupt_request is not None: return "already_requested", observed
        proposed = replace(observed, generation=observed.generation + 1,
                           interrupt_request=(observed.run_id, observed.invocation_id))
        try: self._publish(observed, proposed)
        except ManagedConflictError: return "changed", self.inspect(operation_id)
        return "requested", proposed

    def reconcile(self, operation_id, *, owner):
        """Resolve only exact retained pending bytes under operation ownership.

        operation_id selects current, owner is its active admission/ownership token.
        Returns settled metadata/None; unsupported/ambiguous authority raises
        ManagedControlError unchanged. Reconciliation never realizes, captures or
        executes. Generic journal recovery/projection precedes/follows short control,
        and no process-local restore receipt is created or transferred.
        """
        self._require_owner(owner, operation_id)
        if not self._namespace_present(): return None
        with self._control_lock(): current, pending = self._capture(operation_id)
        if pending is None:
            snapshot = None if current is None else ControlSnapshot.from_bytes(current, operation_id=operation_id)
            if self._pending_mutations(operation_id):
                self._finish_projection()
            return snapshot
        intent = _PendingIntent.from_bytes(pending, operation_id=operation_id)
        if current is None:
            raise ManagedControlError("ambiguous_pending", "pending authority has no current metadata")
        snapshot = ControlSnapshot.from_bytes(current, operation_id=operation_id)
        fingerprint = hashlib.sha256(current).hexdigest()
        if (snapshot.generation, fingerprint) not in {
                (intent.previous_generation, intent.previous_digest), (intent.proposed_generation, intent.proposed_digest)}:
            raise ManagedControlError("ambiguous_pending", "pending intent does not identify exact current")
        self.control_store.recover_authority()
        with self._control_lock():
            if self._capture(operation_id) != (current, pending):
                raise ManagedConflictError("generation_conflict", "authority changed during reconciliation")
            from dryml import filesystem
            operation = self._operation_path(operation_id)
            filesystem.remove_file(self._pending_path(operation), cleanup_directory=operation)
        self._finish_projection()
        return snapshot

    def _require_owner(self, owner, operation_id):
        owner.require_owner()
        lease = owner.ownership if isinstance(owner, OperationAdmission) else owner
        if lease.authority != self._domain or lease.operation_id != operation_id:
            raise ManagedConflictError("owner_conflict", "owner belongs to another control domain/operation")

    def _pending_mutations(self, operation_id):
        scopes = {"managed-current:" + operation_id, "managed-start:" + operation_id}
        return any(scopes.intersection(record.intent["scopes"]) for record in self.control_store.pending_mutations())

    @staticmethod
    def _validate_intent_transition(current, proposed):
        previous = {value.token: value for value in current.intents}
        following = {value.token: value for value in proposed.intents}
        if not previous.keys() <= following.keys():
            raise ManagedConflictError("intent_conflict", "retained resource evidence cannot be discarded by a current CAS")
        allowed = {"prepared": {"adopted", "abandoned"}, "adopted": {"release_pending"},
                   "abandoned": {"release_pending"}, "release_pending": {"released"}, "released": set()}
        for token, value in following.items():
            old = previous.get(token)
            if old == value:
                continue
            if (value.run_id, value.invocation_id) != (proposed.run_id, proposed.invocation_id):
                raise ManagedConflictError("intent_conflict", "cannot mutate another invocation's resource evidence")
            if old is not None and (replace(old, disposition=value.disposition) != value
                    or value.disposition not in allowed[old.disposition]):
                raise ManagedConflictError("intent_conflict", "resource intent disposition is irreversible")

    def _fence(self, current, operation_id, owner, run_id, invocation_id, generation):
        self._require_owner(owner, operation_id)
        _uuid(run_id, "run_id"); _uuid(invocation_id, "invocation_id"); _generation(generation)
        if (current is None or (current.run_id, current.invocation_id, current.generation) != (run_id, invocation_id, generation)
                or owner.invocation_id != invocation_id):
            raise ManagedConflictError("owner_conflict", "run/invocation/generation fence changed")

    def _publish(self, current, proposed):
        store = self.control_store
        operation_id = proposed.operation_id
        old, new = None if current is None else current.to_bytes(), proposed.to_bytes()
        operation = self._operation_path(operation_id)
        path = self._current_path(operation)
        pending = self._pending_path(operation)
        intent = _PendingIntent(operation_id, None if current is None else current.generation,
            None if old is None else hashlib.sha256(old).hexdigest(), proposed.generation,
            hashlib.sha256(new).hexdigest(), uuid4().hex)
        self._namespace_present()
        store.preflight_publication("managed control CAS", record_family="dryml-node-graph/v1")
        store.recover_authority()
        index = store.open_query_index()
        if index is not None: index.maintain_store()
        relative = os.path.relpath(path, store.base_dir).replace(os.sep, "/")
        mutation_id = str(uuid4())
        stage_name = ".managed-operation-staging-" + mutation_id
        scopes = ("managed-current:" + operation_id,)
        if old is None:
            scopes += ("managed-control-staging:" + stage_name,)
        with store.writer_lock():
            cut = journal.read_cut(store)
            if journal.raw(store, relative) != old:
                raise ManagedConflictError("generation_conflict", "exact current changed before control CAS")
            facts = journal.PreparedMutation(cut, ((relative, old),), (),
                (journal.PreparedChange(relative, old, new),), scopes, store._prepared_fact_owner)
            record = journal._intent(facts, mutation_id)
            staging = None
            bridge = os.path.join(store.base_dir, "mutation-prepare.record")
            if os.path.exists(bridge):
                raise ManagedControlError("pending_journal", "Store intent registration requires recovery")
            try:
                store._atomic_write(bridge, record.to_bytes())
                store._atomic_write(journal._path(store, mutation_id), record.to_bytes())
                cut = replace(cut, pending=cut.pending + (mutation_id,))
                store._atomic_write(os.path.join(store.base_dir, "authority.record"), cut.to_bytes())
                from dryml import filesystem
                filesystem.remove_file(bridge, cleanup_directory=store.base_dir)
                if old is None:
                    staged_path = os.path.join(store.base_dir, stage_name)
                    os.mkdir(staged_path, mode=0o700)
                    staging = staged_path
                    self._replace_file(self._pending_path(staging), intent.to_bytes())
                    self._replace_file(self._current_path(staging), new)
                    filesystem.ensure_directory(os.path.dirname(operation))
                with self._control_lock():
                    actual, existing_pending = self._capture(operation_id)
                    if actual != old or existing_pending is not None:
                        raise ManagedConflictError("generation_conflict", "current/pending changed before short CAS")
                    if staging is not None:
                        filesystem.publish_directory(staging, operation)
                        staging = None
                    else:
                        self._replace_file(pending, intent.to_bytes())
                        self._replace_file(path, new)
                    filesystem.remove_file(pending, cleanup_directory=operation)
                journal._classify(store, record, cut, persistence_confirmed=True)
            except BaseException as error:
                # Classification records actual bytes, never success assumptions.
                try: _, cut = journal._classify(store, record, journal.read_cut(store))
                except BaseException: pass
                if not isinstance(error, Exception):
                    error.mutation_id, error.cut = mutation_id, cut
                    raise
                raise journal.StoreMutationError("managed control publication requires recovery", mutation_id, cut) from error
            finally:
                if staging is not None and os.path.isdir(staging):
                    try: shutil.rmtree(staging)
                    except OSError: pass
        self._finish_projection(mutation_id)

    def _finish_projection(self, mutation_id=None):
        store = self.control_store
        index = store.open_query_index()
        if index is not None:
            store._complete_index_mutation(index, mutation_id, store.authority_cut())
        else:
            for record in store.recover_authority():
                store.acknowledge_mutation(record.intent["mutation_id"], record.intent["epoch"])

    def _capture(self, operation_id):
        operation = self._operation_path(operation_id)
        current = self._read_optional(self._current_path(operation))
        pending = self._read_optional(self._pending_path(operation))
        if os.path.exists(operation) and current is None:
            raise ManagedControlError("missing_current", "existing operation lacks current authority")
        return current, pending

    def _namespace_present(self):
        if not os.path.lexists(self.root): return False
        payload = self._read_optional(os.path.join(self.root, "format.json"))
        if payload is None:
            raise ManagedControlError("invalid_managed_namespace", "managed namespace lacks format.json")
        _closed(_data(payload), "dryml-managed", 2, ())
        if os.path.lexists(os.path.join(self.root, "operations", "v1")):
            raise ManagedControlError("unsupported_control", "legacy operation namespace cannot be admitted by area format 2")
        return True

    def _owner_path(self, operation_id):
        return os.path.join(self.root, "owners", "v2", operation_id[:2], operation_id + ".lock")

    def _operation_path(self, operation_id):
        return os.path.join(self.root, "operations", "v2", operation_id[:2], operation_id)

    @staticmethod
    def _current_path(operation): return os.path.join(operation, "current.json")

    @staticmethod
    def _pending_path(operation): return os.path.join(operation, "pending.json")

    @contextmanager
    def _control_lock(self):
        # Store recovery must not classify a still-live registered mutation in
        # the writer-to-CAS gap. Keep that writer through a *nonblocking* short
        # acquisition; contention fails with retained journal evidence, never
        # waits for control while holding an earlier-order fence.
        from dryml.core.store.retention import _writers
        path = os.path.join(self.root, "control.lock")
        if not _writers():
            with interprocess_lock(path):
                yield
            return
        lock = FileLock(path)
        if not lock.acquire(blocking=False):
            raise ManagedConflictError("control_busy", "short control is busy; release Store fences and reconcile before retry")
        try:
            yield
        finally:
            lock.release()

    @staticmethod
    def _read_optional(path):
        try:
            with open(path, "rb") as stream: payload = stream.read(_MAX_BYTES + 1)
        except FileNotFoundError: return None
        except OSError as error:
            raise ManagedControlError("control_unavailable", "could not read managed control metadata") from error
        if len(payload) > _MAX_BYTES:
            raise ManagedControlError("snapshot_too_large", "control metadata exceeds 1 MiB")
        return payload

    @staticmethod
    def _replace_file(path, payload):
        parent = os.path.dirname(path)
        fd, staging = tempfile.mkstemp(prefix=".control-", dir=parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload); stream.flush(); os.fsync(stream.fileno())
            from dryml import filesystem
            filesystem.publish_file(staging, path, replace=True)
        finally:
            if os.path.exists(staging): os.unlink(staging)


def _projection_reader(path):
    from .result_codec import RESULT_VERSION
    result = re.fullmatch(rf"managed/results/v{RESULT_VERSION}/([0-9a-f]{{2}})/([0-9a-f]{{64}})/current\.json", path)
    if result is not None:
        from .results import _ResultRecord
        if result[1] != result[2][:2]:
            raise ManagedControlError("invalid_result_path", "result shard differs")
        class _SelectedResult:
            @staticmethod
            def from_bytes(payload):
                record = _ResultRecord.from_bytes(payload)
                if record.key != result[2]:
                    raise ManagedControlError("invalid_result_path", "result address differs from bytes")
                return record
        return _SelectedResult
    start = re.fullmatch(r"managed/operations/v2/starts/v1/([0-9a-f]{2})/([0-9a-f]{64})/current\.json", path)
    if start is not None:
        if start[1] != start[2][:2]:
            raise ManagedControlError("invalid_control_path", "unsupported managed start projection path")
        class _SelectedStart:
            @staticmethod
            def from_bytes(payload):
                return StartPreparation.from_bytes(payload, operation_id=start[2])
        return _SelectedStart
    match = re.fullmatch(r"managed/operations/v2/([0-9a-f]{2})/([0-9a-f]{64})/current\.json", path)
    if match is None or match[1] != match[2][:2]:
        raise ManagedControlError("invalid_control_path", "unsupported managed projection path")
    class _SelectedCurrent:
        @staticmethod
        def from_bytes(payload):
            return ControlSnapshot.from_bytes(payload, operation_id=match[2])
    return _SelectedCurrent


def _validate_projection_namespace(store):
    root = os.path.join(store.base_dir, "managed")
    if not os.path.lexists(root):
        return
    gate = journal.raw(store, "managed/format.json")
    if gate is None:
        raise ManagedControlError("invalid_managed_namespace", "managed projection namespace lacks format.json")
    _closed(_data(gate), "dryml-managed", 2, ())
    if os.path.lexists(os.path.join(root, "operations", "v1")):
        raise ManagedControlError("unsupported_control", "legacy operation namespace cannot be projected")
    if os.path.lexists(os.path.join(root, "results", "v1")):
        raise ManagedControlError("unsupported_result", "legacy result namespace cannot be projected")


# Core accepts an owner-supplied reader, but never imports this lifecycle package.
# The owner supplies complete current/history/root/preparation/result contributions.
from dryml.core.query.authority import register_control_projection
register_control_projection("managed", "managed", _projection_reader,
                            validate=_validate_projection_namespace, relations=lambda record: _lifecycle_rows(record))


def _lifecycle_rows(record):
    if isinstance(record, ControlSnapshot):
        yield "current", record.to_data()
        for ordinal, history in enumerate(record.history):
            yield "history:" + str(ordinal), history.to_data()
        for intent in record.intents:
            yield "root:" + intent.token, intent.to_data()
        if record.result_reference is not None:
            yield "result", record.result_reference.to_data()
    elif isinstance(record, StartPreparation):
        yield "preparation", record.to_data()
    else:
        yield "result-content", record.to_data()


__all__ = ["ControlSnapshot", "ControlHistory", "ControlIntent", "ManagedControlStore"]
