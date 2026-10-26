"""Closed format-4 Node/state authority records.

The records in this module are logical values.  Backends choose where bytes are
stored, but every backend must use these validators before authority becomes
visible. The format marker retains its framed dill representation. Node/state
families use canonical JSON frames and explicit kind/version/domain/identity
checks. Record identities derive from validated fields, not serializer output.
Decoders reject incompatible, malformed, oversized, or trailing data with
StoreRecordError and perform no Store mutation or object construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import json
import os
import re
import stat
from uuid import UUID, uuid4
from collections.abc import Mapping
from typing import Any, ClassVar

import dill



_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_CODEC_RE = re.compile(r"^[A-Za-z0-9]{1,32}$")
_FORMAT_MAGIC = b"DRYML-STORE-RECORD/"
_MAX_RECORD_BYTES = 16 * 1024 * 1024
_MAX_JOURNAL_BYTES = 64 * 1024 * 1024
_MAX_FILES = 100_000
_MAX_PATH_BYTES = 4096


class StoreRecordError(ValueError):
    """Raised when current Store record bytes or values are not authoritative."""


def _digest(domain: str, data: Any) -> str:
    """Return a canonical SHA-256 digest for validated record identity data."""

    try:
        payload = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    except (TypeError, ValueError) as error:
        raise StoreRecordError(f"{domain} identity data is not canonical JSON.") from error
    return hashlib.sha256(domain.encode("ascii") + b"\0" + payload).hexdigest()


def _require_exact(data: Any, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping) or set(data) != fields:
        raise StoreRecordError(f"{name} fields must be exactly {sorted(fields)!r}.")
    return data


def _digest_value(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
        raise StoreRecordError(f"{name} must be a 64-character lowercase SHA-256 digest.")
    return value


def _uuid_value(value):
    try:
        if type(value) is not str or str(UUID(value)) != value:
            raise ValueError()
    except (TypeError, ValueError, AttributeError) as error:
        raise StoreRecordError("Expected a canonical UUID string.") from error
    return value


def _codec(value: Any) -> str:
    if not isinstance(value, str) or not _CODEC_RE.fullmatch(value):
        raise StoreRecordError("codec must match [A-Za-z0-9]{1,32}.")
    return value


def _alias(value: Any, name: str = "alias") -> str:
    if not isinstance(value, str) or not value or len(value) > 255 or "/" in value or "\\" in value or value in {".", ".."}:
        raise StoreRecordError(f"{name} must be a non-empty, path-safe string no longer than 255 characters.")
    return value


def _normalized_file_path(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > _MAX_PATH_BYTES:
        raise StoreRecordError("payload path must be a bounded non-empty string.")
    if "\\" in value or value.startswith("/") or value.endswith("/"):
        raise StoreRecordError(f"payload path is not normalized: {value!r}.")
    parts = value.split("/")
    if any(not part or part in {".", ".."} for part in parts):
        raise StoreRecordError(f"payload path is not normalized: {value!r}.")
    return value


def _state_hash(value: Any) -> str:
    if not isinstance(value, str) or "-" not in value:
        raise StoreRecordError("local state hash must be '<codec>-<digest>'.")
    codec, digest = value.split("-", 1)
    _codec(codec)
    _digest_value(digest, "local state digest")
    return value


class _Record:
    """Base class for a closed, framed Store record codec."""

    schema: ClassVar[str]
    version: ClassVar[int] = 1

    def to_data(self) -> dict[str, Any]:
        """Return this record's closed current-version logical payload."""
        raise NotImplementedError

    @classmethod
    def from_data(cls, data: Any):
        """Decode and validate one closed current-version logical payload."""
        raise NotImplementedError

    def to_bytes(self) -> bytes:
        """Encode this record with a schema frame suitable for backend storage."""
        payload = dill.dumps(self.to_data(), protocol=5)
        return _FORMAT_MAGIC + self.schema.encode("ascii") + b"/1\n" + payload

    @classmethod
    def from_bytes(cls, data: bytes):
        """Fully validate a framed record, freshly decoding its dill payload.

        Args:
            data (bytes): Complete current-schema frame and trusted dill payload.
        Returns:
            A fresh record with fresh graph tokens and caller-owned values.
        Raises:
            StoreRecordError: For oversized, malformed, trailing, incompatible,
                or logically inconsistent authority.
        Side Effects:
            Executes trusted dill decoding; retains no cross-call content.
        """
        return cls.from_data(cls._decode_framed(data))

    @classmethod
    def _decode_framed(cls, data):
        if not isinstance(data, bytes) or len(data) > _MAX_RECORD_BYTES:
            raise StoreRecordError(f"{cls.schema} record bytes are missing or oversized.")
        prefix = _FORMAT_MAGIC + cls.schema.encode("ascii") + b"/1\n"
        if not data.startswith(prefix):
            raise StoreRecordError(f"Unsupported or malformed {cls.schema} record frame.")
        stream = io.BytesIO(data[len(prefix):])
        try:
            payload = dill.load(stream)
        except Exception as error:
            raise StoreRecordError(f"Malformed {cls.schema} record payload.") from error
        if stream.read(1):
            raise StoreRecordError(f"{cls.schema} record has trailing data.")
        return payload

@dataclass(frozen=True, slots=True)
class StoreFormatRecord(_Record):
    """Format-4 gate with persistent UUID and explicit CDef identity version 3.

    format_version and identity_version are exact supported integers; store_uuid
    is a canonical UUID string, freshly generated for a new Store and preserved
    on reopen. Invalid or missing fields raise StoreRecordError. Constructing a
    record performs no IO and never upgrades an existing authority tree.
    """

    schema: ClassVar[str] = "store-format"
    format_version: int = 4
    store_uuid: str = field(default_factory=lambda: str(uuid4()))
    identity_version: int = 3

    def __post_init__(self) -> None:
        if type(self.format_version) is not int or self.format_version != 4:
            raise StoreRecordError(f"Unsupported Store format version {self.format_version!r}.")
        _uuid_value(self.store_uuid)
        if type(self.identity_version) is not int or self.identity_version != 3:
            raise StoreRecordError("Unsupported Store CDef identity version.")

    def to_data(self) -> dict[str, Any]:
        """Return closed format/UUID/identity authority, without Store IO."""
        return {"schema": self.schema, "version": self.version, "format_version": self.format_version,
                "store_uuid": self.store_uuid, "identity_version": self.identity_version}

    @classmethod
    def from_data(cls, data: Any) -> "StoreFormatRecord":
        """Decode a current gate; reject incompatible/missing fields without conversion."""
        if isinstance(data, Mapping) and data.get("format_version") != 4:
            raise StoreRecordError(f"Unsupported Store format version {data.get('format_version')!r}.")
        data = _require_exact(data, {"schema", "version", "format_version", "store_uuid", "identity_version"}, cls.schema)
        if data["schema"] != cls.schema or type(data["version"]) is not int or data["version"] != cls.version:
            raise StoreRecordError("Unsupported Store format record version.")
        return cls(data["format_version"], data["store_uuid"], data["identity_version"])


def _validate_inventory(files):
    """Validate the exhaustive, sorted regular-file inventory grammar."""
    if not isinstance(files, tuple) or len(files) > _MAX_FILES:
        raise StoreRecordError("manifest files must be a bounded tuple.")
    previous = None
    for path, size, digest in files:
        _normalized_file_path(path)
        if type(size) is not int or size < 0:
            raise StoreRecordError(f"payload size for {path!r} must be a non-negative integer.")
        _digest_value(digest, f"payload digest for {path!r}")
        if previous is not None and path <= previous:
            raise StoreRecordError("manifest payload paths must be unique and sorted.")
        previous = path


def _validate_payload(files, data_dir):
    """Authenticate the exact regular-file tree, including every payload byte."""
    root = os.fspath(data_dir)
    try:
        root_stat = os.lstat(root)
    except FileNotFoundError as error:
        raise StoreRecordError("local state data directory is missing.") from error
    if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
        raise StoreRecordError("local state data root must be a real directory.")
    found = {}
    for current, directories, names in os.walk(root, followlinks=False):
        relative_dir = os.path.relpath(current, root)
        if relative_dir != "." and not directories and not names:
            raise StoreRecordError(f"local state data contains an empty nested directory: {relative_dir!r}.")
        for directory in directories:
            path = os.path.join(current, directory)
            mode = os.lstat(path).st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise StoreRecordError(f"local state data contains an unsupported directory entry: {path!r}.")
        for name in names:
            path = os.path.join(current, name)
            mode = os.lstat(path).st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise StoreRecordError(f"local state data contains an unsupported file entry: {path!r}.")
            rel = _normalized_file_path(os.path.relpath(path, root).replace(os.sep, "/"))
            size = os.path.getsize(path)
            digest = hashlib.sha256()
            with open(path, "rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            found[rel] = (size, digest.hexdigest())
    if found != {path: (size, digest) for path, size, digest in files}:
        raise StoreRecordError("local state data tree does not exactly match manifest files.")


class _NodeRecord:
    """Closed identity-3 metadata; never admits the legacy dill record family."""

    version = 1
    max_bytes = _MAX_RECORD_BYTES
    fields = frozenset()

    def to_data(self):
        """Return detached closed family/version/domain/identity-3 authority."""
        return {"kind": self.schema, "version": self.version, "domain": self.schema + f"/v{self.version}",
                "identity_version": 3, **self._payload()}

    @classmethod
    def from_data(cls, data):
        """Validate exact nested family fields; return fresh metadata or raise StoreRecordError."""
        data = _require_exact(data, set(cls.fields) | {"kind", "version", "domain", "identity_version"}, cls.schema)
        if (data["kind"] != cls.schema or type(data["version"]) is not int or data["version"] != cls.version
                or data["domain"] != cls.schema + f"/v{cls.version}" or type(data["identity_version"]) is not int
                or data["identity_version"] != 3):
            raise StoreRecordError(f"Incompatible {cls.schema} kind/version/domain/identity.")
        try:
            return cls._from_payload(data)
        except StoreRecordError:
            raise
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
            raise StoreRecordError(f"Malformed {cls.schema} metadata.") from error

    def to_bytes(self):
        """Return canonical framed JSON bytes; performs no IO or symbol resolution."""
        from ...formats.envelope import encode_metadata_frame
        try:
            encoded = encode_metadata_frame(self.schema, self.version, self.to_data())
        except ValueError as error:
            raise StoreRecordError(f"Invalid {self.schema} JSON metadata.") from error
        if len(encoded) > self.max_bytes:
            raise StoreRecordError(f"Oversized {self.schema} record.")
        return encoded

    @classmethod
    def from_bytes(cls, data):
        """Decode exact bounded JSON authority, refusing incompatible/trailing records."""
        from ...formats.envelope import decode_metadata_frame
        from ...formats.errors import EnvelopeError
        try:
            return cls.from_data(decode_metadata_frame(data, schema=cls.schema, version=cls.version, max_bytes=cls.max_bytes))
        except EnvelopeError as error:
            raise StoreRecordError(f"Malformed or incompatible {cls.schema} frame.") from error


def _json_bytes(data):
    from ..node_prepare import _bytes
    from ...formats.canonical import json_ready
    return _bytes(json_ready(data, max_depth=128, max_nodes=1_000_000,
                             max_entries=100_000, max_string=16 * 1024 * 1024))


@dataclass(frozen=True, slots=True)
class StateKeyRecord(_NodeRecord):
    """Version-1 compound address; key is a StateKey, not an availability assertion.

    Invalid components raise StoreRecordError. Encoding/decoding performs no IO,
    graph discovery, or hashing and does not allocate ObjectIds.
    """

    schema = "dryml-state-key"
    fields = frozenset({"cdef_digest", "codec_hash"})
    key: object

    def __post_init__(self):
        from ..state_values import StateKey
        if not isinstance(self.key, StateKey):
            raise StoreRecordError("StateKeyRecord requires a StateKey.")

    def _payload(self):
        return {"cdef_digest": self.key.cdef_digest, "codec_hash": self.key.codec_hash}

    @classmethod
    def _from_payload(cls, data):
        from ..state_values import StateKey
        return cls(StateKey(data["cdef_digest"], data["codec_hash"]))


@dataclass(frozen=True, slots=True)
class NodeBlockRecord(_NodeRecord):
    """Immutable shared Node tables/values, encoded once for all selected roots.

    content is canonical JSON bytes for the identity-3 packet without views.
    Structural/semantic closure is validated by node_codec when a root is read;
    this record authenticates placement-independent block content. Invalid packet
    fields raise StoreRecordError. No Objects, symbols or private data are loaded.
    """

    schema = "dryml-node-blocks"
    fields = frozenset({"digest", "blocks"})
    content: bytes

    def __post_init__(self):
        from ..node_codec import NODE_GRAPH_VERSION
        if type(self.content) is not bytes:
            raise StoreRecordError("Node block content must be immutable bytes.")
        data = self.blocks
        _require_exact(data, {"format", "identity_version", "tables", "values"}, self.schema)
        if (data["format"] != NODE_GRAPH_VERSION or type(data["identity_version"]) is not int
                or data["identity_version"] != 3 or type(data["tables"]) is not dict
                or type(data["values"]) is not dict or self.content != _json_bytes(data)):
            raise StoreRecordError("Incompatible Node block format/identity/content.")

    @property
    def blocks(self):
        """Return detached primitive blocks (dict); does not expand the graph."""
        return json.loads(self.content)

    @property
    def digest(self):
        """Return the authenticated storage-block digest, distinct from root identity."""
        return _digest(self.schema + "/v1", self.blocks)

    def _payload(self):
        return {"digest": self.digest, "blocks": self.blocks}

    @classmethod
    def _from_payload(cls, data):
        result = cls(_json_bytes(data["blocks"]))
        if result.digest != data["digest"]:
            raise StoreRecordError("Node blocks authentication mismatch.")
        return result


_ROOT_POLICIES = {"construction": "dryml-creation-cdef/v1", "recipe": "exact",
                  "saved": "exact", "projection": "dryml-object-projection/v2"}


@dataclass(frozen=True, slots=True)
class NodeRootRecord(_NodeRecord):
    """Compact authenticated root view, including bucket def.pkl authority.

    Args:
        block_digest: Shared NodeBlockRecord address.
        root_kind: construction, recipe, saved, or projection semantic purpose.
        view_bytes: Canonical JSON for node_codec's compact view.
    Invalid kinds, versions, domains or keys raise StoreRecordError. The digest
    covers semantic identity, not storage placement; expansion validates actual
    blocks and policy at the Store boundary. No graph is embedded in this record.
    """

    schema = "dryml-node-root"
    fields = frozenset({"block_digest", "root_kind", "view", "digest"})
    block_digest: str
    root_kind: str
    view_bytes: bytes

    def __post_init__(self):
        from ..node_prepare import EXACT_KEY_VERSION
        _digest_value(self.block_digest, "shared block digest")
        if self.root_kind not in _ROOT_POLICIES or type(self.view_bytes) is not bytes:
            raise StoreRecordError("Invalid root kind or immutable view bytes.")
        view = self.view
        _require_exact(view, {"table", "slot", "view", "key_format", "expected_key"}, self.schema)
        _digest_value(view["table"], "root table")
        if (view["view"] != _ROOT_POLICIES[self.root_kind] or view["key_format"] != EXACT_KEY_VERSION
                or type(view["slot"]) is not int or view["slot"] < 0 or self.view_bytes != _json_bytes(view)):
            raise StoreRecordError("Incompatible root view policy/key format/slot.")
        try:
            key = json.loads(view["expected_key"])
            if (type(key) is not list or len(key) != 5 or key[:2] != [EXACT_KEY_VERSION, view["view"]]
                    or key[3] not in {"compose", "adjacency"}):
                raise ValueError()
            _digest_value(key[2], "root value key")
            _digest_value(key[4], "root topology key")
            if _json_bytes(key).decode("ascii") != view["expected_key"]:
                raise ValueError()
        except (ValueError, TypeError) as error:
            raise StoreRecordError("Invalid semantic root key.") from error

    @property
    def view(self):
        """Return the detached compact view (dict), without graph expansion or IO."""
        return json.loads(self.view_bytes)

    @property
    def digest(self):
        """Return the root's domain-separated semantic address (str)."""
        domain = (_ROOT_POLICIES[self.root_kind] if self.root_kind in {"construction", "projection"}
                  else "dryml-" + self.root_kind + "-root/v1")
        return hashlib.sha256(domain.encode("ascii") + b"\0" + self.view["expected_key"].encode("ascii")).hexdigest()

    def _payload(self):
        return {"block_digest": self.block_digest, "root_kind": self.root_kind,
                "view": self.view, "digest": self.digest}

    @classmethod
    def _from_payload(cls, data):
        result = cls(data["block_digest"], data["root_kind"], _json_bytes(data["view"]))
        if data["digest"] != result.digest:
            raise StoreRecordError("Root view digest mismatch.")
        return result


@dataclass(frozen=True, slots=True)
class StateVariantManifest(_NodeRecord):
    """Version-1 full-StateKey manifest with exhaustive payload authentication.

    key is a StateKey; codec_version is a positive integer; interpretation_bytes
    contains canonical JSON codec metadata; files is a sorted exhaustive tuple of
    (normalized relative path, size, SHA-256). Content identity authenticates all
    codec interpretation/inventory evidence, excluding creation/producer identity.
    Invalid fields/hash mismatches raise StoreRecordError before publication.
    """

    schema = "dryml-state-variant"
    fields = frozenset({"key", "codec_version", "interpretation", "files"})
    key: object
    codec_version: int
    interpretation_bytes: bytes
    files: tuple

    def __post_init__(self):
        StateKeyRecord(self.key)
        if type(self.codec_version) is not int or self.codec_version < 1:
            raise StoreRecordError("Codec version must be a positive exact integer.")
        if (type(self.interpretation_bytes) is not bytes or type(self.interpretation) is not dict
                or self.interpretation_bytes != _json_bytes(self.interpretation)):
            raise StoreRecordError("Codec interpretation must be canonical JSON mapping bytes.")
        if self.key.codec_hash != self.content_hash(self.codec, self.codec_version, self.interpretation, self.files):
            raise StoreRecordError("Codec/content hash does not match manifest evidence.")

    @staticmethod
    def content_hash(codec, codec_version, interpretation, files):
        """Return codec-qualified v1 hash of interpretation and exhaustive inventory.

        codec is a string, codec_version a positive exact int, interpretation a
        JSON mapping and files a validated inventory tuple. Invalid inputs raise
        ValueError/StoreRecordError. This pure hash performs no payload IO and
        excludes framework ObjectIds, CDefs, Store paths, timestamps and producers.
        """
        _codec(codec)
        if type(codec_version) is not int or codec_version < 1:
            raise StoreRecordError("Codec version must be a positive exact integer.")
        interpretation = json.loads(_json_bytes(interpretation))
        if type(interpretation) is not dict:
            raise StoreRecordError("Codec interpretation must be a JSON mapping.")
        _validate_inventory(files)
        return codec + "-" + _digest("dryml-codec-content/v1", {
            "codec": codec, "codec_version": codec_version,
            "interpretation": interpretation, "files": files})

    @property
    def codec(self):
        """Return the explicitly authenticated codec identity (str)."""
        return self.key.codec_hash.split("-", 1)[0]

    @property
    def interpretation(self):
        """Return detached codec interpretation metadata (dict), without payload IO."""
        return json.loads(self.interpretation_bytes)

    @property
    def deferred_paths(self):
        """Return the empty v1 deferral policy; all use authenticates payload bytes."""
        return ()

    def validate_payload(self, data_dir):
        """Authenticate exactly the regular-file data tree or raise StoreRecordError.

        data_dir is a backend-owned directory path. Reads every inventoried byte;
        does not inspect framework siblings, rewrite authority or run codec hooks.
        """
        _validate_payload(self.files, data_dir)

    def _payload(self):
        return {"key": StateKeyRecord(self.key).to_data(), "codec_version": self.codec_version,
                "interpretation": self.interpretation, "files": [list(item) for item in self.files]}

    @classmethod
    def _from_payload(cls, data):
        if type(data["files"]) is not list or any(type(row) is not list or len(row) != 3 for row in data["files"]):
            raise StoreRecordError("Malformed variant inventory.")
        return cls(StateKeyRecord.from_data(data["key"]).key, data["codec_version"],
                   _json_bytes(data["interpretation"]), tuple(tuple(row) for row in data["files"]))


@dataclass(frozen=True, slots=True)
class StateAvailabilityRecord(_NodeRecord):
    """Publication incarnation for one full StateKey; presence is checked separately.

    key is a StateKey and incarnation is a fresh canonical UUID per installation.
    status is prepared, published or withdrawn; only published advertises data.
    Invalid input raises StoreRecordError. This metadata alone promises neither
    data integrity nor active-use protection and has no side effects.
    """

    schema = "dryml-state-availability"
    fields = frozenset({"key", "incarnation", "status"})
    key: object
    incarnation: str
    status: str = "published"

    def __post_init__(self):
        StateKeyRecord(self.key)
        _uuid_value(self.incarnation)
        if self.status not in {"prepared", "published", "withdrawn"}:
            raise StoreRecordError("Invalid state availability status.")

    def _payload(self):
        return {"key": StateKeyRecord(self.key).to_data(), "incarnation": self.incarnation, "status": self.status}

    @classmethod
    def _from_payload(cls, data):
        return cls(StateKeyRecord.from_data(data["key"]).key, data["incarnation"], data["status"])


@dataclass(frozen=True, slots=True)
class NodeMembershipRecord(_NodeRecord):
    """Independent recipe/saved-root membership, never implied by bucket metadata.

    root_kind is recipe or saved; root_digest names the matching NodeRootRecord;
    identities lists required ObjectIds with declaration anchors. Invalid values
    raise StoreRecordError. Construction/encoding has no IO or publication effect.
    """

    schema = "dryml-node-membership"
    fields = frozenset({"root_kind", "root_digest", "identities"})
    root_kind: str
    root_digest: str
    identities: tuple = ()

    def __post_init__(self):
        from ..reference_values import ObjectId
        if self.root_kind not in {"recipe", "saved"}:
            raise StoreRecordError("Invalid membership kind.")
        _digest_value(self.root_digest, "membership root")
        if (type(self.identities) is not tuple or any(not isinstance(i, ObjectId) for i in self.identities)
                or len(set(self.identities)) != len(self.identities)):
            raise StoreRecordError("Invalid membership identity requirements.")

    def _payload(self):
        return {"root_kind": self.root_kind, "root_digest": self.root_digest,
                "identities": [i.to_data() for i in self.identities]}

    @classmethod
    def _from_payload(cls, data):
        from ..reference_values import ObjectId
        if type(data["identities"]) is not list:
            raise StoreRecordError("Invalid membership identities.")
        return cls(data["root_kind"], data["root_digest"], tuple(ObjectId.from_data(i) for i in data["identities"]))


@dataclass(frozen=True, slots=True)
class DeclarationAnchorRecord(_NodeRecord):
    """ObjectId-addressed canonical projection and contributing publication sources.

    object_id is an ObjectId, projection is a projection-kind NodeRootRecord, and
    sources is a sorted unique tuple of (recipe|saved, root digest). State pins
    are excluded by the owning object_projection policy, not by hash equality.
    Malformed fields raise StoreRecordError. No Objects/payloads are opened.
    """

    schema = "dryml-declaration-anchor"
    fields = frozenset({"object_id", "projection", "sources"})
    object_id: object
    projection: NodeRootRecord
    sources: tuple

    def __post_init__(self):
        from ..reference_values import ObjectId
        if not isinstance(self.object_id, ObjectId) or not isinstance(self.projection, NodeRootRecord) or self.projection.root_kind != "projection":
            raise StoreRecordError("Invalid declaration anchor identity/projection.")
        if type(self.sources) is not tuple or not self.sources or tuple(sorted(set(self.sources))) != self.sources:
            raise StoreRecordError("Anchor sources must be nonempty, unique and sorted.")
        for kind, digest in self.sources:
            NodeMembershipRecord(kind, digest)

    def _payload(self):
        return {"object_id": self.object_id.to_data(), "projection": self.projection.to_data(),
                "sources": [list(source) for source in self.sources]}

    @classmethod
    def _from_payload(cls, data):
        from ..reference_values import ObjectId
        if type(data["sources"]) is not list or any(type(row) is not list or len(row) != 2 for row in data["sources"]):
            raise StoreRecordError("Invalid anchor sources.")
        return cls(ObjectId.from_data(data["object_id"]), NodeRootRecord.from_data(data["projection"]),
                   tuple(tuple(source) for source in data["sources"]))


@dataclass(frozen=True, slots=True)
class DeclarationRequirementRecord(_NodeRecord):
    """Persistent ObjectId witness making a missing established anchor detectable.

    object_id and projection_digest identify immutable declaration authority.
    Invalid fields raise StoreRecordError. This witness is not a query index or
    independent graph membership; it is retained with declaration history.
    """

    schema = "dryml-declaration-required"
    fields = frozenset({"object_id", "projection_digest"})
    object_id: object
    projection_digest: str

    def __post_init__(self):
        from ..reference_values import ObjectId
        if not isinstance(self.object_id, ObjectId):
            raise StoreRecordError("Invalid required declaration identity.")
        _digest_value(self.projection_digest, "declaration projection")

    def _payload(self):
        return {"object_id": self.object_id.to_data(), "projection_digest": self.projection_digest}

    @classmethod
    def _from_payload(cls, data):
        from ..reference_values import ObjectId
        return cls(ObjectId.from_data(data["object_id"]), data["projection_digest"])


__all__ = ["StoreFormatRecord", "StoreRecordError", "StateKeyRecord", "NodeBlockRecord", "NodeRootRecord", "StateVariantManifest",
             "StateAvailabilityRecord", "NodeMembershipRecord", "DeclarationAnchorRecord",
             "DeclarationRequirementRecord"]


@dataclass(frozen=True, slots=True)
class AuthorityRecord(_NodeRecord):
    """Store UUID, committed epoch and exact unacknowledged intent UUIDs.

    Epoch is a nonnegative exact integer; pending is a unique tuple of canonical
    UUID strings. Invalid values/versions raise StoreRecordError. This closed v1
    record contains no graph and performs no IO.
    """

    schema = "dryml-store-authority"
    fields = frozenset({"store_uuid", "epoch", "pending"})
    store_uuid: str
    epoch: int = 0
    pending: tuple = ()

    def __post_init__(self):
        _uuid_value(self.store_uuid)
        if type(self.epoch) is not int or self.epoch < 0:
            raise StoreRecordError("Invalid authority epoch.")
        if type(self.pending) is not tuple or len(set(self.pending)) != len(self.pending):
            raise StoreRecordError("Invalid pending mutation identities.")
        for identity in self.pending:
            _uuid_value(identity)

    def _payload(self):
        return {"store_uuid": self.store_uuid, "epoch": self.epoch, "pending": list(self.pending)}

    @classmethod
    def _from_payload(cls, data):
        if type(data["pending"]) is not list:
            raise StoreRecordError("Invalid pending mutation list.")
        return cls(data["store_uuid"], data["epoch"], tuple(data["pending"]))


@dataclass(frozen=True, slots=True)
class MutationIntentRecord(_NodeRecord):
    """Replayable v2 mutation evidence, independent of the Store format marker.

    content is immutable canonical JSON with a stable mutation/store UUID, base
    cut, affected scopes, expected byte transitions, confirmed references, exact
    observed projection work and classification/acknowledgement state. The boolean
    persistence_confirmed distinguishes observed visibility from completed native
    barriers; acknowledgement requires it. Invalid
    fields raise StoreRecordError; decoding never runs graph or lifecycle hooks.
    Complete frames have a journal-only 64 MiB bound, including base64 transition
    and projection copies. Product records retain their separate 16 MiB bound.
    Older intent versions are refused without conversion; Store format remains 4.
    """

    schema = "dryml-store-mutation-intent"
    version = 2
    max_bytes = _MAX_JOURNAL_BYTES
    fields = frozenset({"intent"})
    content: bytes

    def __post_init__(self):
        if type(self.content) is not bytes:
            raise StoreRecordError("Intent content must be immutable bytes.")
        data = self.intent
        _require_exact(data, {"mutation_id", "store_uuid", "base_epoch", "scopes", "transitions",
                              "confirmed", "projection", "epoch", "status", "persistence_confirmed"}, self.schema)
        _uuid_value(data["mutation_id"])
        _uuid_value(data["store_uuid"])
        if type(data["base_epoch"]) is not int or data["base_epoch"] < 0:
            raise StoreRecordError("Invalid intent base cut.")
        if data["epoch"] is not None and (type(data["epoch"]) is not int or data["epoch"] < data["base_epoch"]):
            raise StoreRecordError("Invalid classified cut.")
        if data["status"] not in {"pending", "classified", "acknowledged"}:
            raise StoreRecordError("Invalid intent disposition.")
        if (type(data["persistence_confirmed"]) is not bool
                or (data["status"] == "pending" and data["persistence_confirmed"])
                or (data["status"] == "acknowledged" and not data["persistence_confirmed"])):
            raise StoreRecordError("Invalid intent persistence confirmation.")
        for field_name in ("scopes", "transitions", "confirmed", "projection"):
            if type(data[field_name]) is not list:
                raise StoreRecordError("Invalid intent work list.")
        if any(type(scope) is not str or not scope for scope in data["scopes"]):
            raise StoreRecordError("Invalid affected scope.")
        if (data["status"] == "pending" and (data["epoch"] is not None or data["confirmed"] or data["projection"])):
            raise StoreRecordError("Pending intent cannot certify a resulting cut.")
        if data["status"] != "pending" and data["epoch"] is None:
            raise StoreRecordError("Classified intent requires a resulting cut.")
        for transition in data["transitions"]:
            _require_exact(transition, {"path", "before", "after", "staging", "kind"}, "transition")
            _normalized_file_path(transition["path"])
            if transition["kind"] not in {"file", "variant", "data"}:
                raise StoreRecordError("Invalid transition kind.")
            for name in ("before", "after"):
                _decode_authority_bytes(transition[name])
            if transition["staging"] is not None:
                _normalized_file_path(transition["staging"])
            if transition["kind"] == "file" and transition["staging"] is not None:
                raise StoreRecordError("File transitions cannot borrow directory staging.")
            if transition["kind"] != "file":
                if (transition["before"] is not None or transition["after"] is None
                        or transition["staging"] is None or not re.fullmatch(
                            r"\.staging/[0-9a-f]{32}(?:/data)?", transition["staging"])):
                    raise StoreRecordError("Child installation requires owned staging and absent destination.")
        for row in data["projection"]:
            _require_exact(row, {"path", "kind", "value"}, "projection work")
            _normalized_file_path(row["path"])
            if row["kind"] not in {"file", "variant", "data"}:
                raise StoreRecordError("Invalid projection kind.")
            _decode_authority_bytes(row["value"])
        keys = [(row["path"], row["kind"]) for row in data["projection"]]
        if data["status"] != "pending" and (len(set(keys)) != len(keys)
                or set(keys) != {(row["path"], row["kind"]) for row in data["transitions"]}):
            raise StoreRecordError("Classified projection must cover exactly the affected transitions.")
        transition_paths = {t["path"] for t in data["transitions"]}
        if any(type(ref) is not str or ref not in transition_paths
               for ref in data["confirmed"]):
            raise StoreRecordError("Invalid confirmed publication.")
        if self.content != _json_bytes(data):
            raise StoreRecordError("Noncanonical intent content.")

    @property
    def intent(self):
        """Return detached primitive evidence (dict), with no Store or graph access."""
        return json.loads(self.content)

    def _payload(self):
        return {"intent": self.intent}

    @classmethod
    def _from_payload(cls, data):
        return cls(_json_bytes(data["intent"]))


def _encode_authority_bytes(value):
    import base64
    return None if value is None else base64.b64encode(value).decode("ascii")


def _decode_authority_bytes(value):
    import base64
    try:
        if value is None:
            return None
        if type(value) is not str:
            raise ValueError()
        result = base64.b64decode(value, validate=True)
        if _encode_authority_bytes(result) != value:
            raise ValueError()
        return result
    except (ValueError, TypeError) as error:
        raise StoreRecordError("Invalid intent byte evidence.") from error


@dataclass(frozen=True, slots=True)
class NodeNameRecord(_NodeRecord):
    """One format-4 mutable name targeting an explicitly published Node root.

    name is a bounded path-safe string; root_kind is recipe/saved and root_digest
    its semantic address. Invalid values raise StoreRecordError. A name carries
    no payload-availability guarantee and construction performs no Store IO.
    """

    schema = "dryml-node-name"
    fields = frozenset({"name", "root_kind", "root_digest"})
    name: str
    root_kind: str
    root_digest: str

    def __post_init__(self):
        _alias(self.name)
        NodeMembershipRecord(self.root_kind, self.root_digest)

    def _payload(self):
        return {"name": self.name, "root_kind": self.root_kind, "root_digest": self.root_digest}

    @classmethod
    def _from_payload(cls, data):
        return cls(data["name"], data["root_kind"], data["root_digest"])


__all__ += ["AuthorityRecord", "MutationIntentRecord", "NodeNameRecord"]


@dataclass(frozen=True, slots=True)
class RetiredDeclarationAnchorRecord(DeclarationAnchorRecord):
    """Zero-source canonical declaration history after journaled withdrawal.

    Uses a separate v1 schema at the same ObjectId anchor path, so an emptied
    active-anchor record remains corruption. Fields/returns match the active
    anchor, but sources must be (). Projection history and requirement remain
    authoritative; malformed identity/projection raises StoreRecordError.
    """

    schema = "dryml-declaration-retired-anchor"

    def __post_init__(self):
        from ..reference_values import ObjectId
        if (not isinstance(self.object_id, ObjectId) or not isinstance(self.projection, NodeRootRecord)
                or self.projection.root_kind != "projection" or self.sources != ()):
            raise StoreRecordError("Invalid retired declaration anchor.")


__all__ += ["RetiredDeclarationAnchorRecord"]


@dataclass(frozen=True, slots=True)
class NodeMetadataRecord(_NodeRecord):
    """Closed format-4 current annotations, lineage, or saved capture evidence.

    scope selects current_object/current_state/lineage/capture; target_digest is
    the owning object projection or saved-root address. content is canonical JSON
    validated by core.metadata. Invalid schemas, values or associations raise
    StoreRecordError before publication. No graph or private state is embedded.
    """

    schema = "dryml-node-metadata"
    fields = frozenset({"scope", "target_digest", "content"})
    scope: str
    target_digest: str
    content: bytes

    def __post_init__(self):
        from ..metadata import _validate_node_metadata
        if self.scope not in {"current_object", "current_state", "lineage", "capture"}:
            raise StoreRecordError("Unsupported Node metadata scope.")
        _digest_value(self.target_digest, "metadata target digest")
        if type(self.content) is not bytes:
            raise StoreRecordError("Metadata content must be immutable bytes.")
        try:
            data = json.loads(self.content)
            _validate_node_metadata(self.scope, data)
            if self.scope == "lineage" and data["projection"] != self.target_digest:
                raise ValueError("Lineage projection does not match its metadata target.")
            if self.content != _json_bytes(data):
                raise ValueError("Noncanonical metadata content.")
        except (TypeError, ValueError, KeyError) as error:
            raise StoreRecordError("Malformed Node metadata content.") from error

    def _payload(self):
        return {"scope": self.scope, "target_digest": self.target_digest, "content": json.loads(self.content)}

    @classmethod
    def _from_payload(cls, data):
        return cls(data["scope"], data["target_digest"], _json_bytes(data["content"]))


__all__ += ["NodeMetadataRecord"]
