"""Closed codecs for current-version Store authority records.

The records in this module are logical values.  Backends choose where bytes are
stored, but every backend must use these validators before authority becomes
visible.  Record bytes use a framed dill payload so arbitrary trailing bytes
cannot be accepted as authority; record identity is always derived from the
validated logical fields rather than from serializer output.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
import re
import stat
from collections.abc import Mapping
from typing import Any, ClassVar

import dill

from ..cdef_codec import decode_cdef_graph, encode_cdef_graph
from ..definition import ConcreteDefinition
from ..reference_values import ObjectRef, StateRef
from ..utils.graph.path import GraphPath, graph_path_sort_key


_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_CODEC_RE = re.compile(r"^[A-Za-z0-9]{1,32}$")
_FORMAT_MAGIC = b"DRYML-STORE-RECORD/"
_MAX_RECORD_BYTES = 16 * 1024 * 1024
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


def _role_paths(definition: ConcreteDefinition) -> tuple[dict[str, Any], ...]:
    """Return canonical primary paths and recorded role bits for a CDef graph."""

    from ..cdef_graph import ConcreteDefinitionGraph

    graph = ConcreteDefinitionGraph.from_root(definition, expand_ref_targets=True)
    paths: dict[object, GraphPath] = {id(definition): GraphPath()}
    stack = [(definition, GraphPath())]
    while stack:
        parent, parent_path = stack.pop()
        for edge in graph.outgoing(parent):
            candidate = parent_path.join(edge.path)
            key = id(edge.child)
            previous = paths.get(key)
            if previous is None or graph_path_sort_key(candidate) < graph_path_sort_key(previous):
                paths[key] = candidate
                stack.append((edge.child, candidate))
    nodes = sorted(graph.nodes(), key=lambda node: graph_path_sort_key(paths[id(node.definition)]))
    return tuple(
        {"path": paths[id(node.definition)].to_data(), "stateful": node.definition._stateful_role}
        for node in nodes
    )


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
        """Decode one complete framed record and reject trailing or wrong bytes."""
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
        return cls.from_data(payload)


@dataclass(frozen=True, slots=True)
class StoreFormatRecord(_Record):
    """The sole Store-wide format gate for current direct-layout authority."""

    schema: ClassVar[str] = "store-format"
    format_version: int = 1

    def __post_init__(self) -> None:
        if self.format_version != 1:
            raise StoreRecordError(f"Unsupported Store format version {self.format_version!r}.")

    def to_data(self) -> dict[str, Any]:
        return {"schema": self.schema, "version": self.version, "format_version": self.format_version}

    @classmethod
    def from_data(cls, data: Any) -> "StoreFormatRecord":
        data = _require_exact(data, {"schema", "version", "format_version"}, cls.schema)
        if data["schema"] != cls.schema or data["version"] != cls.version:
            raise StoreRecordError("Unsupported Store format record version.")
        return cls(data["format_version"])


@dataclass(frozen=True, slots=True)
class DefinitionRecord(_Record):
    """Immutable graph-aware CDef authority grouped by structural identity."""

    schema: ClassVar[str] = "definition"
    definition: ConcreteDefinition

    def __post_init__(self) -> None:
        if not isinstance(self.definition, ConcreteDefinition):
            raise StoreRecordError("DefinitionRecord definition must be a ConcreteDefinition.")

    @property
    def graph_hash(self) -> str:
        return self.definition.graph_hash()

    @property
    def structural_hash(self) -> str:
        return self.definition.stable_hash()

    @property
    def roles(self) -> tuple[dict[str, Any], ...]:
        return _role_paths(self.definition)

    @property
    def digest(self) -> str:
        return _digest("dryml-definition-record-v1", {"graph_hash": self.graph_hash, "structural_hash": self.structural_hash, "roles": self.roles})

    def to_data(self) -> dict[str, Any]:
        return {
            "schema": self.schema, "version": self.version, "digest": self.digest,
            "graph_hash": self.graph_hash, "structural_hash": self.structural_hash,
            "definition": encode_cdef_graph(self.definition), "roles": list(self.roles),
        }

    @classmethod
    def from_data(cls, data: Any) -> "DefinitionRecord":
        data = _require_exact(data, {"schema", "version", "digest", "graph_hash", "structural_hash", "definition", "roles"}, cls.schema)
        if data["schema"] != cls.schema or data["version"] != cls.version:
            raise StoreRecordError("Unsupported DefinitionRecord version.")
        _digest_value(data["digest"], "definition digest")
        _digest_value(data["graph_hash"], "definition graph hash")
        _digest_value(data["structural_hash"], "definition structural hash")
        try:
            definition = decode_cdef_graph(data["definition"])
        except Exception as error:
            raise StoreRecordError("DefinitionRecord graph authority is invalid.") from error
        result = cls(definition)
        if data["graph_hash"] != result.graph_hash or data["structural_hash"] != result.structural_hash or data["digest"] != result.digest:
            raise StoreRecordError("DefinitionRecord digest or hash fields do not match graph authority.")
        if not isinstance(data["roles"], list) or tuple(data["roles"]) != result.roles:
            raise StoreRecordError("DefinitionRecord role paths do not match graph authority.")
        return result


@dataclass(frozen=True, slots=True)
class LocalStateManifest(_Record):
    """Exhaustive immutable payload manifest for one codec-specific local state.

    ``definition_digest`` names validated logical graph authority while
    ``definition_file_digest`` authenticates the exact framed ``def.pkl`` bytes
    stored beside the payload. Both are required before a local state can be
    reused, copied, or restored.
    """

    schema: ClassVar[str] = "local-state-manifest"
    codec: str
    graph_hash: str
    definition_digest: str
    definition_file_digest: str
    files: tuple[tuple[str, int, str], ...]

    def __post_init__(self) -> None:
        _codec(self.codec)
        _digest_value(self.graph_hash, "graph hash")
        _digest_value(self.definition_digest, "definition digest")
        _digest_value(self.definition_file_digest, "definition file digest")
        if not isinstance(self.files, tuple) or len(self.files) > _MAX_FILES:
            raise StoreRecordError("manifest files must be a bounded tuple.")
        previous = None
        for path, size, digest in self.files:
            _normalized_file_path(path)
            if type(size) is not int or size < 0:
                raise StoreRecordError(f"payload size for {path!r} must be a non-negative integer.")
            _digest_value(digest, f"payload digest for {path!r}")
            if previous is not None and path <= previous:
                raise StoreRecordError("manifest payload paths must be unique and sorted.")
            previous = path

    @property
    def local_digest(self) -> str:
        return _digest("dryml-local-state-manifest-v1", {"codec": self.codec, "graph_hash": self.graph_hash, "definition_digest": self.definition_digest, "definition_file_digest": self.definition_file_digest, "files": self.files})

    @property
    def state_hash(self) -> str:
        return f"{self.codec}-{self.local_digest}"

    def to_data(self) -> dict[str, Any]:
        return {
            "schema": self.schema, "version": self.version, "codec": self.codec,
            "graph_hash": self.graph_hash, "definition_digest": self.definition_digest,
            "definition_file_digest": self.definition_file_digest,
            "local_digest": self.local_digest,
            "files": [{"path": path, "size": size, "digest": digest} for path, size, digest in self.files],
        }

    @classmethod
    def from_data(cls, data: Any) -> "LocalStateManifest":
        data = _require_exact(data, {"schema", "version", "codec", "graph_hash", "definition_digest", "definition_file_digest", "local_digest", "files"}, cls.schema)
        if data["schema"] != cls.schema or data["version"] != cls.version or not isinstance(data["files"], list):
            raise StoreRecordError("Unsupported LocalStateManifest version or files.")
        files = []
        for entry in data["files"]:
            entry = _require_exact(entry, {"path", "size", "digest"}, "manifest file")
            files.append((entry["path"], entry["size"], entry["digest"]))
        result = cls(data["codec"], data["graph_hash"], data["definition_digest"], data["definition_file_digest"], tuple(files))
        if data["local_digest"] != result.local_digest:
            raise StoreRecordError("LocalStateManifest digest does not match its content.")
        return result

    def validate_payload(self, data_dir: str | os.PathLike[str]) -> None:
        """Verify that ``data_dir`` is exactly this manifest's regular-file tree."""
        root = os.fspath(data_dir)
        try:
            root_stat = os.lstat(root)
        except FileNotFoundError as error:
            raise StoreRecordError("local state data directory is missing.") from error
        if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
            raise StoreRecordError("local state data root must be a real directory.")
        found: list[tuple[str, int, str]] = []
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
                digest = hashlib.sha256()
                with open(path, "rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(block)
                found.append((rel, os.path.getsize(path), digest.hexdigest()))
        if tuple(sorted(found)) != self.files:
            raise StoreRecordError("local state data tree does not exactly match manifest files.")


@dataclass(frozen=True, slots=True)
class StateRefRecord(_Record):
    """Immutable complete StateRef authority record."""

    schema: ClassVar[str] = "state-ref"
    state_ref: StateRef

    def __post_init__(self) -> None:
        if not isinstance(self.state_ref, StateRef):
            raise StoreRecordError("StateRefRecord state_ref must be a StateRef.")

    @property
    def digest(self) -> str:
        return self.state_ref.digest()

    def to_data(self) -> dict[str, Any]:
        return {"schema": self.schema, "version": self.version, "digest": self.digest, "state_ref": self.state_ref.to_data()}

    @classmethod
    def from_data(cls, data: Any) -> "StateRefRecord":
        data = _require_exact(data, {"schema", "version", "digest", "state_ref"}, cls.schema)
        if data["schema"] != cls.schema or data["version"] != cls.version:
            raise StoreRecordError("Unsupported StateRefRecord version.")
        try:
            result = cls(StateRef.from_data(data["state_ref"]))
        except Exception as error:
            raise StoreRecordError("StateRefRecord state reference is invalid.") from error
        if data["digest"] != result.digest:
            raise StoreRecordError("StateRefRecord digest does not match its state reference.")
        return result


@dataclass(frozen=True, slots=True)
class DeclarationRecord(_Record):
    """Immutable registered ObjectRef declaration authority."""

    schema: ClassVar[str] = "declaration"
    object_ref: ObjectRef

    def __post_init__(self) -> None:
        if not isinstance(self.object_ref, ObjectRef) or not self.object_ref.objects:
            raise StoreRecordError("DeclarationRecord requires a non-empty ObjectRef.")

    @property
    def digest(self) -> str:
        return self.object_ref.digest()

    def to_data(self) -> dict[str, Any]:
        return {"schema": self.schema, "version": self.version, "digest": self.digest, "object_ref": self.object_ref.to_data()}

    @classmethod
    def from_data(cls, data: Any) -> "DeclarationRecord":
        data = _require_exact(data, {"schema", "version", "digest", "object_ref"}, cls.schema)
        if data["schema"] != cls.schema or data["version"] != cls.version:
            raise StoreRecordError("Unsupported DeclarationRecord version.")
        try:
            result = cls(ObjectRef.from_data(data["object_ref"]))
        except Exception as error:
            raise StoreRecordError("DeclarationRecord ObjectRef is invalid.") from error
        if data["digest"] != result.digest:
            raise StoreRecordError("DeclarationRecord digest does not match its ObjectRef.")
        return result


@dataclass(frozen=True, slots=True)
class ClaimRecord(_Record):
    """Mutable claim fence shape, validated before U6 lease behavior uses it."""

    schema: ClassVar[str] = "claim"
    object_digest: str
    generation: int
    status: str
    owner: str | None = None
    lease_until: float | None = None
    state_ref_digest: str | None = None

    def __post_init__(self) -> None:
        _digest_value(self.object_digest, "claim object digest")
        if type(self.generation) is not int or self.generation < 0:
            raise StoreRecordError("claim generation must be a non-negative integer.")
        if self.status not in {"available", "claimed", "completed"}:
            raise StoreRecordError("claim status must be available, claimed, or completed.")
        if self.status == "available" and any(value is not None for value in (self.owner, self.lease_until, self.state_ref_digest)):
            raise StoreRecordError("available claims cannot carry owner, lease, or StateRef fields.")
        if self.status == "claimed":
            if not isinstance(self.owner, str) or not self.owner or not isinstance(self.lease_until, (int, float)) or self.state_ref_digest is not None:
                raise StoreRecordError("claimed records require owner and lease only.")
        if self.status == "completed":
            if self.owner is not None or self.lease_until is not None or self.state_ref_digest is None:
                raise StoreRecordError("completed claims require only a StateRef digest.")
            _digest_value(self.state_ref_digest, "claim StateRef digest")

    def to_data(self) -> dict[str, Any]:
        return {"schema": self.schema, "version": self.version, "object_digest": self.object_digest, "generation": self.generation, "status": self.status, "owner": self.owner, "lease_until": self.lease_until, "state_ref_digest": self.state_ref_digest}

    @classmethod
    def from_data(cls, data: Any) -> "ClaimRecord":
        data = _require_exact(data, {"schema", "version", "object_digest", "generation", "status", "owner", "lease_until", "state_ref_digest"}, cls.schema)
        if data["schema"] != cls.schema or data["version"] != cls.version:
            raise StoreRecordError("Unsupported ClaimRecord version.")
        return cls(data["object_digest"], data["generation"], data["status"], data["owner"], data["lease_until"], data["state_ref_digest"])


@dataclass(frozen=True, slots=True)
class MainRefRecord(_Record):
    """Mutable Store main reference to an existing immutable DefinitionRecord."""

    schema: ClassVar[str] = "main-ref"
    definition_digest: str

    def __post_init__(self) -> None:
        _digest_value(self.definition_digest, "main definition digest")

    def to_data(self) -> dict[str, Any]:
        return {"schema": self.schema, "version": self.version, "definition_digest": self.definition_digest}

    @classmethod
    def from_data(cls, data: Any) -> "MainRefRecord":
        data = _require_exact(data, {"schema", "version", "definition_digest"}, cls.schema)
        if data["schema"] != cls.schema or data["version"] != cls.version:
            raise StoreRecordError("Unsupported MainRefRecord version.")
        return cls(data["definition_digest"])


@dataclass(frozen=True, slots=True)
class ObjectAliasRecord(_Record):
    """Mutable Store-local object alias targeting existing ObjectRef authority."""

    schema: ClassVar[str] = "object-alias"
    alias: str
    object_ref: ObjectRef

    def __post_init__(self) -> None:
        _alias(self.alias)
        if not isinstance(self.object_ref, ObjectRef):
            raise StoreRecordError("object alias target must be an ObjectRef.")

    def to_data(self) -> dict[str, Any]:
        return {"schema": self.schema, "version": self.version, "alias": self.alias, "object_ref": self.object_ref.to_data()}

    @classmethod
    def from_data(cls, data: Any) -> "ObjectAliasRecord":
        data = _require_exact(data, {"schema", "version", "alias", "object_ref"}, cls.schema)
        if data["schema"] != cls.schema or data["version"] != cls.version:
            raise StoreRecordError("Unsupported ObjectAliasRecord version.")
        try:
            return cls(data["alias"], ObjectRef.from_data(data["object_ref"]))
        except Exception as error:
            raise StoreRecordError("ObjectAliasRecord target is invalid.") from error


@dataclass(frozen=True, slots=True)
class StateAliasRecord(_Record):
    """Mutable alias scoped to one exact ObjectRef and targeting one StateRef."""

    schema: ClassVar[str] = "state-alias"
    alias: str
    object_ref: ObjectRef
    state_ref_digest: str

    def __post_init__(self) -> None:
        _alias(self.alias)
        if not isinstance(self.object_ref, ObjectRef):
            raise StoreRecordError("state alias scope must be an ObjectRef.")
        _digest_value(self.state_ref_digest, "state alias StateRef digest")

    def to_data(self) -> dict[str, Any]:
        return {"schema": self.schema, "version": self.version, "alias": self.alias, "object_ref": self.object_ref.to_data(), "state_ref_digest": self.state_ref_digest}

    @classmethod
    def from_data(cls, data: Any) -> "StateAliasRecord":
        data = _require_exact(data, {"schema", "version", "alias", "object_ref", "state_ref_digest"}, cls.schema)
        if data["schema"] != cls.schema or data["version"] != cls.version:
            raise StoreRecordError("Unsupported StateAliasRecord version.")
        try:
            return cls(data["alias"], ObjectRef.from_data(data["object_ref"]), data["state_ref_digest"])
        except Exception as error:
            raise StoreRecordError("StateAliasRecord target is invalid.") from error


__all__ = [
    "ClaimRecord", "DeclarationRecord", "DefinitionRecord", "LocalStateManifest",
    "MainRefRecord", "ObjectAliasRecord", "StateAliasRecord", "StateRefRecord",
    "StoreFormatRecord", "StoreRecordError",
]
