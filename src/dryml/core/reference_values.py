"""Import-free immutable identities for realized DRYML construction graphs.

The values in this module deliberately describe graph topology and exact state
identity without constructing Objects, resolving classes, or loading state.
Store authority and runtime materialization are separate later boundaries.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import builtins
import hashlib
import json
import re
from typing import Any, Iterator
from uuid import UUID, uuid4

from .freeze import FrozenDict
from .utils.graph.path import GraphPath, GraphPathLike, graph_path_sort_key, normalize_path
from .utils.stable_hash import stable_int_hash


_NAMESPACE = ContextVar("dryml_object_namespace", default=())
_NAMESPACE_PART = re.compile(r"[A-Za-z0-9]+\Z")
_STATE_HASH = re.compile(r"[A-Za-z0-9]{1,32}-[0-9a-f]{64}\Z")
_MAX_NAMESPACE_DEPTH = 16
_MAX_NAMESPACE_PART_LENGTH = 64
_MAX_NAMESPACE_LENGTH = 256
_REFERENCE_CODEC_VERSION = 1


def _normalize_namespace(namespace: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if not isinstance(namespace, (tuple, list)):
        raise TypeError("ObjectId namespace must be a tuple or list of strings.")
    if len(namespace) > _MAX_NAMESPACE_DEPTH:
        raise ValueError(f"ObjectId namespace has more than {_MAX_NAMESPACE_DEPTH} parts.")
    parts = []
    for part in namespace:
        if not isinstance(part, str):
            raise TypeError("ObjectId namespace parts must be strings.")
        if not _NAMESPACE_PART.fullmatch(part):
            raise ValueError(
                "ObjectId namespace parts must be non-empty ASCII alphanumeric strings."
            )
        if len(part) > _MAX_NAMESPACE_PART_LENGTH:
            raise ValueError(
                f"ObjectId namespace parts must be at most {_MAX_NAMESPACE_PART_LENGTH} characters."
            )
        parts.append(part)
    if sum(len(part) for part in parts) > _MAX_NAMESPACE_LENGTH:
        raise ValueError(
            f"ObjectId namespace must be at most {_MAX_NAMESPACE_LENGTH} characters."
        )
    return tuple(parts)


@dataclass(frozen=True, slots=True, init=False)
class ObjectId:
    """Durable identity of one stateful node in a realized graph.

    Args:
        namespace: Optional namespace for a new identity. ``None`` inherits the
            current :func:`object_namespace` scope, while ``()`` is explicitly
            empty.

    Normal construction always allocates a fresh framework nonce. Decoders use
    :meth:`from_data` to preserve an already-authoritative full UUID.
    """

    namespace: tuple[str, ...]
    nonce: UUID

    def __init__(self, namespace: tuple[str, ...] | list[str] | None = None):
        selected = _NAMESPACE.get() if namespace is None else namespace
        object.__setattr__(self, "namespace", _normalize_namespace(selected))
        object.__setattr__(self, "nonce", uuid4())

    @classmethod
    def _trusted(cls, namespace: tuple[str, ...] | list[str], nonce: UUID) -> "ObjectId":
        if not isinstance(nonce, UUID):
            raise TypeError("ObjectId nonce must be a UUID.")
        result = object.__new__(cls)
        object.__setattr__(result, "namespace", _normalize_namespace(namespace))
        object.__setattr__(result, "nonce", nonce)
        return result

    @classmethod
    def from_data(cls, data: Any) -> "ObjectId":
        """Decode one trusted full ObjectId record.

        Args:
            data: Mapping with exactly ``namespace`` and ``nonce`` fields.

        Returns:
            The validated identity with the persisted UUID preserved.

        Raises:
            TypeError: If the record fields have unsupported types.
            ValueError: If the record is malformed.
        """

        if not isinstance(data, Mapping) or set(data) != {"namespace", "nonce"}:
            raise ValueError("ObjectId records require exactly namespace and nonce fields.")
        try:
            nonce = UUID(data["nonce"])
        except (TypeError, ValueError, AttributeError) as error:
            raise ValueError("ObjectId nonce must be a full UUID string.") from error
        return cls._trusted(data["namespace"], nonce)

    def to_data(self) -> dict[str, Any]:
        """Return the lossless machine-readable ObjectId record."""

        return {"namespace": list(self.namespace), "nonce": str(self.nonce)}

    def __stable_leaf_bytes__(self) -> bytes:
        return _canonical_bytes(("object-id", self.namespace, self.nonce.hex))

    def __str__(self) -> str:
        prefix = "/".join(self.namespace)
        return f"{prefix + '~' if prefix else '~'}{self.nonce.hex[:12]}..."


@contextmanager
def object_namespace(*parts: str) -> Iterator[tuple[str, ...]]:
    """Append namespace parts for identities created inside the context.

    Args:
        *parts: Normalized namespace components to append to the current scope.

    Yields:
        The effective immutable namespace tuple.

    Raises:
        TypeError: If a component is not a string.
        ValueError: If components violate namespace bounds or normalization.

    Side Effects:
        Installs a task- and thread-local ContextVar value that is restored on
        normal or exceptional exit.
    """

    namespace = _normalize_namespace(_NAMESPACE.get() + tuple(parts))
    token = _NAMESPACE.set(namespace)
    try:
        yield namespace
    finally:
        _NAMESPACE.reset(token)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _path_data(path: GraphPath) -> dict[str, Any]:
    return path.to_data()


def _normalize_mapping(mapping: Mapping[Any, Any], value_type: type, name: str) -> FrozenDict:
    if not isinstance(mapping, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    values: dict[GraphPath, Any] = {}
    for path, value in mapping.items():
        normalized = normalize_path(path)
        if normalized in values:
            raise ValueError(f"{name} contains duplicate path {normalized!s}.")
        if not isinstance(value, value_type):
            raise TypeError(f"{name} values must be {value_type.__name__} instances.")
        values[normalized] = value
    return FrozenDict(dict(sorted(values.items(), key=lambda item: graph_path_sort_key(item[0]))))


def _validate_state_hash(value: str) -> None:
    if not isinstance(value, str) or not _STATE_HASH.fullmatch(value):
        raise ValueError(
            "State hashes must be '<codec>-<64 lowercase hexadecimal digest>' values."
        )


def _reference_types():
    return ObjectRef, StateRef


def _collect_occurrences(definition: Any) -> dict[object, list[GraphPath]]:
    """Collect every owned stateful occurrence under a CDef without imports."""

    from .definition import ConcreteDefinition
    from .links import DefLink
    from .cdef_graph import EdgeKind
    from .utils.graph.value import iter_value_edges

    occurrences: dict[object, list[GraphPath]] = {}
    active: set[object] = set()

    def visit_cdef(cdef: ConcreteDefinition, path: GraphPath) -> None:
        key = ("cdef", cdef._node_id)
        if cdef._stateful_role:
            occurrences.setdefault(key, []).append(path)
        if key in active:
            raise ValueError(f"Materializing CDef cycle at {path!s}.")
        active.add(key)
        try:
            for edge in iter_value_edges(cdef):
                visit_value(edge.value, path.child(edge.segment))
        finally:
            active.remove(key)

    def visit_imported(ref: ObjectRef | StateRef, path: GraphPath) -> None:
        imported = ref.object if isinstance(ref, StateRef) else ref
        inner = _collect_occurrences(imported.definition)
        primary = _primary_paths(inner)
        for key, primary_path in primary.items():
            object_id = imported.objects[primary_path]
            for occurrence in inner[key]:
                occurrences.setdefault(("object-id", object_id), []).append(path.join(occurrence))

    def visit_value(value: Any, path: GraphPath) -> None:
        if isinstance(value, ConcreteDefinition):
            visit_cdef(value, path)
            return
        if isinstance(value, DefLink):
            if value.kind is EdgeKind.MATERIALIZE:
                visit_value(value.target, path)
            return
        if isinstance(value, _reference_types()):
            visit_imported(value, path)
            return
        for edge in iter_value_edges(value):
            visit_value(edge.value, path.child(edge.segment))

    visit_cdef(definition, GraphPath())
    return occurrences


def _primary_paths(occurrences: Mapping[object, list[GraphPath]]) -> dict[object, GraphPath]:
    return {
        key: min(paths, key=graph_path_sort_key)
        for key, paths in occurrences.items()
    }


def _expected_paths(definition: Any) -> tuple[dict[object, GraphPath], dict[GraphPath, object]]:
    by_key = _primary_paths(_collect_occurrences(definition))
    return by_key, {path: key for key, path in by_key.items()}


@dataclass(frozen=True, slots=True, init=False, eq=False)
class ObjectRef:
    """Exact graph topology plus one ObjectId for each owned stateful node.

    Args:
        definition: Root canonical CDef preserving the complete graph topology.
        objects: Mapping from each required canonical primary path to ObjectId.

    Raises:
        TypeError: If inputs are not canonical CDef/path/ObjectId values.
        ValueError: If paths, identities, aliases, or imported materializing
            references do not form the exact complete owned-stateful mapping.
            Imported occurrences must retain their embedded ObjectIds.
    """

    definition: Any
    objects: Mapping[GraphPath, ObjectId]

    def __init__(self, definition: Any, objects: Mapping[Any, ObjectId]):
        from .definition import ConcreteDefinition

        if not isinstance(definition, ConcreteDefinition):
            raise TypeError("ObjectRef definition must be a ConcreteDefinition.")
        expected_by_key, expected_by_path = _expected_paths(definition)
        normalized = _normalize_mapping(objects, ObjectId, "ObjectRef objects")
        if set(normalized) != set(expected_by_path):
            missing = sorted(set(expected_by_path) - set(normalized), key=graph_path_sort_key)
            extra = sorted(set(normalized) - set(expected_by_path), key=graph_path_sort_key)
            raise ValueError(
                f"ObjectRef objects must contain exactly canonical primary paths; "
                f"missing={[str(path) for path in missing]!r}, extra={[str(path) for path in extra]!r}."
            )
        values = tuple(normalized.values())
        if len(values) != len(set(values)):
            raise ValueError("ObjectRef cannot assign one ObjectId to independent stateful nodes.")
        for key, primary_path in expected_by_key.items():
            if key[:1] == ("object-id",) and normalized[primary_path] != key[1]:
                raise ValueError(
                    f"ObjectRef imported primary path {primary_path!s} must retain "
                    "its embedded ObjectId."
                )
        object.__setattr__(self, "definition", definition)
        object.__setattr__(self, "objects", normalized)

    @property
    def object_id(self) -> ObjectId | None:
        """Return the root identity, or ``None`` when the root is ephemeral."""

        return self.objects.get(GraphPath())

    def state(self, alias: str) -> "StateSelectorRef":
        """Return a soft selector for a state alias scoped to this exact graph."""

        return StateSelectorRef(self, alias)

    def at(self, path: GraphPathLike = "$") -> "ObjectRef":
        """Project a closed, rebased materializing subtree.

        Args:
            path: Canonical path to a CDef or materializing exact-reference edge,
                including paths nested inside imported exact references.

        Returns:
            The selected subtree with every owned ObjectId path rebased to root.

        Raises:
            QueryPathError: If the path does not select a materializing subtree.
            ValueError: If the path selects a Ref-only edge.
        """

        return _project_object_ref(self, normalize_path(path))

    def _identity_data(self) -> dict[str, Any]:
        return {
            "definition_graph": self.definition.graph_hash(),
            "objects": [
                [_path_data(path), object_id.to_data()]
                for path, object_id in self.objects.items()
            ],
        }

    def digest(self) -> str:
        """Return a deterministic digest for this complete exact identity."""

        return hashlib.sha256(b"dryml-object-ref-v1\x00" + _canonical_bytes(self._identity_data())).hexdigest()

    def __stable_leaf_bytes__(self) -> bytes:
        return b"dryml-object-ref-v1\x00" + self.digest().encode("ascii")

    def __hash__(self) -> int:
        return stable_int_hash(self.digest())

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ObjectRef)
            and self.definition.graph_equal(other.definition)
            and self.objects == other.objects
        )

    def to_data(self) -> dict[str, Any]:
        """Encode this reference with token-neutral CDef graph authority."""

        from .cdef_codec import encode_cdef_graph

        return {
            "codec_version": _REFERENCE_CODEC_VERSION,
            "definition": encode_cdef_graph(self.definition),
            "objects": [
                {"path": _path_data(path), "object_id": object_id.to_data()}
                for path, object_id in self.objects.items()
            ],
        }

    @classmethod
    def from_data(cls, data: Any) -> "ObjectRef":
        """Decode a token-neutral exact ObjectRef authority record."""

        from .cdef_codec import decode_cdef_graph

        if not isinstance(data, Mapping) or set(data) != {"codec_version", "definition", "objects"}:
            raise ValueError("ObjectRef records require codec_version, definition, and objects fields.")
        if data["codec_version"] != _REFERENCE_CODEC_VERSION or not isinstance(data["objects"], list):
            raise ValueError("Unsupported or malformed ObjectRef record.")
        objects = {}
        for entry in data["objects"]:
            if not isinstance(entry, Mapping) or set(entry) != {"path", "object_id"}:
                raise ValueError("ObjectRef object entries require path and object_id fields.")
            path = GraphPath.from_data(entry["path"])
            if path in objects:
                raise ValueError(f"ObjectRef record repeats path {path!s}.")
            objects[path] = ObjectId.from_data(entry["object_id"])
        return cls(decode_cdef_graph(data["definition"]), objects)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class StateRef:
    """Exact immutable local-state hashes for one complete ObjectRef graph.

    Args:
        object: Complete exact ObjectRef graph identity.
        states: Mapping with exactly the ObjectRef primary paths and local hashes.

    Raises:
        TypeError: If the object or state mapping has unsupported types.
        ValueError: If state paths differ from the ObjectRef or hashes are invalid.
    """

    object: ObjectRef
    states: Mapping[GraphPath, str]

    def __init__(self, object: ObjectRef, states: Mapping[Any, str]):
        if not isinstance(object, ObjectRef):
            raise TypeError("StateRef object must be an ObjectRef.")
        normalized = _normalize_mapping(states, str, "StateRef states")
        for state_hash in normalized.values():
            _validate_state_hash(state_hash)
        if set(normalized) != set(object.objects):
            raise ValueError("StateRef states must have exactly the ObjectRef primary paths.")
        builtins.object.__setattr__(self, "object", object)
        builtins.object.__setattr__(self, "states", normalized)

    @property
    def definition(self) -> Any:
        """Return the complete underlying CDef topology without loading state."""

        return self.object.definition

    @property
    def object_id(self) -> ObjectId | None:
        """Return the identified root ObjectId, or ``None`` for ephemeral roots."""

        return self.object.object_id

    def at(self, path: GraphPathLike = "$") -> "StateRef":
        """Project a closed subtree with enclosing authoritative state hashes.

        Args:
            path: Canonical path to a materializing subtree, including paths
                nested inside imported exact references.

        Returns:
            An exact StateRef whose paths are rebased to the selected subtree.

        Raises:
            QueryPathError: If the path does not resolve to a subtree.
            ValueError: If the path selects a Ref-only or non-materializing edge.
        """

        normalized = normalize_path(path)
        projected = self.object.at(normalized)
        states = _project_mapping(self.object.definition, self.object.objects, self.states, normalized)
        return StateRef(projected, states)

    def digest(self) -> str:
        """Return a deterministic digest for this complete state identity."""

        data = {
            "object": self.object.digest(),
            "states": [[_path_data(path), state_hash] for path, state_hash in self.states.items()],
        }
        return hashlib.sha256(b"dryml-state-ref-v1\x00" + _canonical_bytes(data)).hexdigest()

    def __stable_leaf_bytes__(self) -> bytes:
        return b"dryml-state-ref-v1\x00" + self.digest().encode("ascii")

    def __hash__(self) -> int:
        return stable_int_hash(self.digest())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, StateRef) and self.object == other.object and self.states == other.states

    def to_data(self) -> dict[str, Any]:
        """Encode this reference with token-neutral CDef graph authority."""

        return {
            "codec_version": _REFERENCE_CODEC_VERSION,
            "object": self.object.to_data(),
            "states": [
                {"path": _path_data(path), "state": state_hash}
                for path, state_hash in self.states.items()
            ],
        }

    @classmethod
    def from_data(cls, data: Any) -> "StateRef":
        """Decode a token-neutral exact StateRef authority record."""

        if not isinstance(data, Mapping) or set(data) != {"codec_version", "object", "states"}:
            raise ValueError("StateRef records require codec_version, object, and states fields.")
        if data["codec_version"] != _REFERENCE_CODEC_VERSION or not isinstance(data["states"], list):
            raise ValueError("Unsupported or malformed StateRef record.")
        states = {}
        for entry in data["states"]:
            if not isinstance(entry, Mapping) or set(entry) != {"path", "state"}:
                raise ValueError("StateRef state entries require path and state fields.")
            path = GraphPath.from_data(entry["path"])
            if path in states:
                raise ValueError(f"StateRef record repeats path {path!s}.")
            states[path] = entry["state"]
        return cls(ObjectRef.from_data(data["object"]), states)


@dataclass(frozen=True, slots=True)
class StateSelectorRef:
    """Soft state-alias selector that must resolve before CDef finalization.

    Args:
        object: Exact ObjectRef scope for the selected alias.
        alias: Non-empty Store-managed state alias text.

    The value carries no state authority itself. A managing Repo resolves it to
    one StateRef during canonicalization.
    """

    object: ObjectRef
    alias: str

    def __post_init__(self) -> None:
        if not isinstance(self.object, ObjectRef):
            raise TypeError("StateSelectorRef object must be an ObjectRef.")
        if not isinstance(self.alias, str) or not self.alias:
            raise ValueError("StateSelectorRef alias must be a non-empty string.")

    def __stable_leaf_bytes__(self) -> bytes:
        return _canonical_bytes(("state-selector-ref", self.object.digest(), self.alias))


def _unwrap_materializing(value: Any) -> Any:
    from .links import DefLink
    from .cdef_graph import EdgeKind

    if isinstance(value, DefLink):
        if value.kind is not EdgeKind.MATERIALIZE:
            raise ValueError("Reference projection cannot select a Ref-only edge.")
        return value.target
    return value


def _project_mapping(
    definition: Any,
    objects: Mapping[GraphPath, ObjectId],
    values: Mapping[GraphPath, Any],
    path: GraphPath,
) -> FrozenDict:
    from .definition import ConcreteDefinition

    selected = _projection_target(definition, path)
    if isinstance(selected, StateRef):
        selected = selected.object
    if isinstance(selected, ObjectRef):
        selected = selected.definition
    if not isinstance(selected, ConcreteDefinition):
        raise ValueError(f"Path {path!s} does not select a materializing CDef subtree.")

    occurrences = _collect_occurrences(definition)
    primary = _primary_paths(occurrences)
    occurrence_values = {
        occurrence_path: values[primary[key]]
        for key, paths in occurrences.items()
        for occurrence_path in paths
    }
    selected_occurrences = _collect_occurrences(selected)
    out: dict[GraphPath, Any] = {}
    for key, local_paths in selected_occurrences.items():
        local_primary = min(local_paths, key=graph_path_sort_key)
        try:
            out[local_primary] = occurrence_values[path.join(local_primary)]
        except KeyError as error:
            raise ValueError(f"Path {path!s} is not an owned subtree of this ObjectRef.") from error
    return FrozenDict(dict(sorted(out.items(), key=lambda item: graph_path_sort_key(item[0]))))


def _projection_target(definition: Any, path: GraphPath) -> Any:
    """Resolve a materializing path while entering imported exact references."""

    from .utils.graph.value import get_subtree

    selected = definition
    for index, segment in enumerate(path):
        selected = _unwrap_materializing(get_subtree(selected, GraphPath((segment,))))
        if isinstance(selected, StateRef):
            selected = selected.object
        if isinstance(selected, ObjectRef) and index != len(path) - 1:
            selected = selected.definition
    return selected


def _project_object_ref(ref: ObjectRef, path: GraphPath) -> ObjectRef:
    if not path:
        return ref
    selected = _projection_target(ref.definition, path)
    if isinstance(selected, StateRef):
        selected = selected.object
    if isinstance(selected, ObjectRef):
        return selected
    from .definition import ConcreteDefinition

    if not isinstance(selected, ConcreteDefinition):
        raise ValueError(f"Path {path!s} does not select a materializing CDef subtree.")
    projected = _project_mapping(ref.definition, ref.objects, ref.objects, path)
    return ObjectRef(selected, projected)
