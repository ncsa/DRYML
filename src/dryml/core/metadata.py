"""Core-owned Node metadata values and bounded annotation codecs.

The module is intentionally independent of Store publication. It defines the
typed user-value codec, immutable creation/capture models, and closed format-4
Node metadata validation consumed by Store publication. Historical compound
reference metadata and snapshot-directory codecs are retired.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
import re
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

from dryml.environments import EnvironmentRecord, EnvironmentRequirement
from dryml.formats import CanonicalJSONError, EnvelopeError

from .node import Node
from .utils.graph.path import GraphPath, graph_path_sort_key

MetadataScalar: TypeAlias = None | bool | int | float | str
MetadataValue: TypeAlias = MetadataScalar | list["MetadataValue"] | tuple["MetadataValue", ...] | Mapping[str, "MetadataValue"]
MetadataMapping: TypeAlias = Mapping[str, MetadataValue]
MetadataTarget: TypeAlias = Node
MetadataDiagnostic: TypeAlias = tuple[str, str]
EnvironmentStatus: TypeAlias = Literal["known", "incomplete", "unavailable"]
RequirementStatus: TypeAlias = Literal["empty", "value", "conflict", "unavailable"]
EvidenceCoverage: TypeAlias = Literal["complete", "incomplete"]

_METADATA_MAX_DEPTH = 8
_METADATA_MAX_NODES = 1024
_METADATA_MAX_ENTRIES = 64
_METADATA_MAX_STRING = 4096
_METADATA_MAX_INT_BITS = 4096
_DIAGNOSTIC_MAX_ENTRIES = 64
_DIAGNOSTIC_MAX_STRING = 512
_INTEGER = re.compile(r"(?:0|-?[1-9][0-9]*)\Z")


@dataclass(frozen=True, slots=True)
class NodeLineageMetadata:
    """Detached creation evidence for an identified Node object projection.

    object_ref is the exact state-weakened Node; creation_status and created_at
    distinguish unknown evidence from an aware UTC allocation time. Values are
    constructed by the validated format-4 metadata reader, with no payload IO.
    Invalid Node/timestamp types raise TypeError; inconsistent status raises
    ValueError. Construction performs no external reads or publication.
    """

    object_ref: Node
    creation_status: str
    created_at: datetime | None

    def __post_init__(self):
        from .node import Node
        if not isinstance(self.object_ref, Node):
            raise TypeError("Node lineage requires a Node projection.")
        _validate_optional_timestamp(self.created_at)
        if self.creation_status not in {"known", "unknown"} or (self.created_at is not None) != (self.creation_status == "known"):
            raise ValueError("Node lineage timestamp/status is inconsistent.")
        if self.object_ref.object_id is None and self.creation_status != "unknown":
            raise ValueError("A Node without an ObjectId has unknown creation evidence.")


@dataclass(frozen=True, slots=True)
class NodeSnapshotMetadata:
    """Detached format-4 saved-graph capture, independent of current annotations.

    state_ref is the exact saved Node; lineages maps primary owned paths to
    NodeLineageMetadata. Remaining fields retain observation, requirement,
    timestamp and annotation evidence. Readers validate closed
    record/graph associations and return fresh nested annotation containers.
    Invalid saved completeness, lineage coverage, timestamp/status, environment,
    requirement or annotation values raise TypeError/ValueError before effects.
    """

    state_ref: Node
    lineages: Mapping[GraphPath, NodeLineageMetadata]
    saved_at: datetime
    environment: EnvironmentRecord | None
    environment_status: str
    requirements: EnvironmentRequirement | None
    requirements_status: str
    requirements_coverage: str
    diagnostics: tuple
    captured_object_annotations: MetadataMapping | None
    captured_state_annotations: MetadataMapping | None

    def __post_init__(self):
        from .node import require_saved, _owned_nodes
        require_saved(self.state_ref)
        lineages, diagnostics = _validate_capture_fields(self.lineages, self.saved_at, self.environment,
            self.environment_status, self.requirements, self.requirements_status, self.requirements_coverage, self.diagnostics)
        expected = {path: node for path, node in _owned_nodes(self.state_ref) if node.object_id is not None}
        expected.setdefault(GraphPath(), self.state_ref)
        if set(expected) != set(lineages) or any(not lineages[path].object_ref.object_projection_equal(node)
                                                for path, node in expected.items()):
            raise ValueError("Node capture lineage coverage does not match the saved graph.")
        object.__setattr__(self, "lineages", lineages)
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "captured_object_annotations", _copy_mapping_or_none(self.captured_object_annotations))
        object.__setattr__(self, "captured_state_annotations", _copy_mapping_or_none(self.captured_state_annotations))


def _node_annotation_scope(scope):
    if scope is None:
        return None
    if type(scope) is not str or scope not in {"object", "state"}:
        raise ValueError("Node annotation scope must be 'object', 'state', or None.")
    return "current_" + scope


def _validate_node_metadata(scope, data):
    """Validate the closed format-4 metadata payload without graph expansion."""
    from .reference_values import ObjectId
    from .store.records import _digest_value
    if scope in {"current_object", "current_state"}:
        _closed(data, {"values"}, "Node annotations")
        decode_metadata_mapping(data["values"])
        return
    if scope in {"lineage", "capture_lineage"}:
        _closed(data, {"identity", "projection", "creation_status", "created_at"}, "Node lineage")
        if data["identity"] is None:
            if scope != "capture_lineage" or data["creation_status"] != "unknown" or data["created_at"] is not None:
                raise ValueError("Only a captured stateless root may have no lineage identity.")
        else:
            ObjectId.from_data(data["identity"])
        _digest_value(data["projection"], "lineage projection")
        instant = timestamp_from_seconds(data["created_at"])
        if data["creation_status"] not in {"known", "unknown"} or (instant is not None) != (data["creation_status"] == "known"):
            raise ValueError("Invalid lineage timestamp/status.")
        return
    if scope != "capture":
        raise ValueError("Unsupported Node metadata scope.")
    _closed(data, {"lineages", "saved_at", "environment", "environment_status", "requirements",
                  "requirements_status", "requirements_coverage", "diagnostics", "captured_annotations"}, "Node capture")
    environment = None if data["environment"] is None else EnvironmentRecord.from_data(data["environment"])
    requirements = None if data["requirements"] is None else EnvironmentRequirement.from_data(data["requirements"])
    SnapshotCapture({}, timestamp_from_seconds(data["saved_at"]), environment, data["environment_status"],
                    requirements, data["requirements_status"], data["requirements_coverage"],
                    tuple(tuple(item) for item in data["diagnostics"]))
    if type(data["lineages"]) is not list:
        raise ValueError("Capture lineages must be a list.")
    identities = set()
    for lineage in data["lineages"]:
        _validate_node_metadata("capture_lineage", lineage)
        identity = None if lineage["identity"] is None else ObjectId.from_data(lineage["identity"])
        if identity in identities:
            raise ValueError("Duplicate capture lineage.")
        identities.add(identity)
    _closed(data["captured_annotations"], {"object", "state"}, "Node captured annotations")
    for value in data["captured_annotations"].values():
        _decode_captured_annotation(value)


def _node_capture_data(capture, node):
    """Rebase a single fresh save observation onto exact owned Node projections."""
    from .node import _owned_nodes
    by_identity = {fact.object_ref.object_id: fact for fact in capture.lineages.values()}
    lineages = []
    if node.object_id is None:
        lineages.append({"identity": None, "projection": node.object_projection_digest(),
                         "creation_status": "unknown", "created_at": None})
    for _, child in _owned_nodes(node):
        if child.object_id is None:
            continue
        old = by_identity.get(child.object_id)
        instant = None if old is None else old.created_at
        lineages.append({"identity": child.object_id.to_data(), "projection": child.object_projection_digest(),
                         "creation_status": "unknown" if instant is None else "known",
                         "created_at": None if instant is None else timestamp_to_seconds(instant)})
    return {"lineages": lineages, "saved_at": timestamp_to_seconds(capture.saved_at),
            "environment": None if capture.environment is None else capture.environment.to_data(),
            "environment_status": capture.environment_status,
            "requirements": None if capture.requirements is None else capture.requirements.to_data(),
            "requirements_status": capture.requirements_status, "requirements_coverage": capture.requirements_coverage,
            "diagnostics": [list(item) for item in capture.diagnostics],
            "captured_annotations": {"object": _captured_annotation_data(None), "state": _captured_annotation_data(None)}}


def _node_snapshot_metadata(node, data):
    """Validate every captured lineage against the supplied exact saved graph."""
    from .node import _owned_nodes, require_saved
    from .reference_values import ObjectId
    require_saved(node)
    _validate_node_metadata("capture", data)
    by_identity = {None if row["identity"] is None else ObjectId.from_data(row["identity"]): row for row in data["lineages"]}
    lineages = {}
    identities = set()
    if node.object_id is None:
        row = by_identity.get(None)
        if row is None or row["projection"] != node.object_projection_digest():
            raise ValueError("Captured stateless-root lineage does not match the saved graph.")
        identities.add(None)
        lineages[GraphPath()] = NodeLineageMetadata(node.object_view(), "unknown", None)
    for path, child in _owned_nodes(node):
        if child.object_id is None:
            continue
        identities.add(child.object_id)
        row = by_identity.get(child.object_id)
        if row is None or row["projection"] != child.object_projection_digest():
            raise ValueError("Captured lineage does not match the saved graph.")
        lineages[path] = NodeLineageMetadata(child.object_view(), row["creation_status"], timestamp_from_seconds(row["created_at"]))
    if identities != set(by_identity):
        raise ValueError("Captured lineage coverage does not match the saved graph.")
    return NodeSnapshotMetadata(node, MappingProxyType(lineages), timestamp_from_seconds(data["saved_at"]),
        None if data["environment"] is None else EnvironmentRecord.from_data(data["environment"]), data["environment_status"],
        None if data["requirements"] is None else EnvironmentRequirement.from_data(data["requirements"]),
        data["requirements_status"], data["requirements_coverage"], tuple(tuple(item) for item in data["diagnostics"]),
        _decode_captured_annotation(data["captured_annotations"]["object"]),
        _decode_captured_annotation(data["captured_annotations"]["state"]))


@dataclass(frozen=True, slots=True)
class SaveAnnotations:
    """Explicit current-mapping values to apply and capture during a save.

    Args:
        object: Optional current Node object-projection mapping. ``None`` means no explicit
            replacement was requested.
        state: Optional current saved-Node mapping. ``None`` has the same meaning.

    Raises:
        TypeError: If a supplied mapping is not a string-keyed metadata mapping.
        ValueError: If a supplied mapping violates metadata value bounds.

    Inputs are copied so later caller mutation cannot alter a save request.
    """

    object: MetadataMapping | None = None
    state: MetadataMapping | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "object", _copy_mapping_or_none(self.object))
        object.__setattr__(self, "state", _copy_mapping_or_none(self.state))


@dataclass(frozen=True, slots=True)
class SnapshotCapture:
    """Core capture inputs excluding publication-time annotation reads.

    Args:
        lineages: Creation evidence indexed by canonical primary graph paths.
        saved_at: Timezone-aware UTC save instant.
        environment: Existing environment-domain value, if usable.
        environment_status: Known, incomplete, or unavailable observation state.
        requirements: Existing environment-requirement value, if usable.
        requirements_status: Empty, valued, conflicting, or unavailable result.
        requirements_coverage: Whether all materializing classes were inspected.
        diagnostics: Bounded existing diagnostic code/message pairs.

    Raises:
        TypeError: If supplied values have unsupported types.
        ValueError: If status, coverage, values, timestamps, or diagnostics are
            inconsistent.

    The model copies mappings and never performs environment observation.
    """

    lineages: Mapping[GraphPath, NodeLineageMetadata]
    saved_at: datetime
    environment: EnvironmentRecord | None
    environment_status: EnvironmentStatus
    requirements: EnvironmentRequirement | None
    requirements_status: RequirementStatus
    requirements_coverage: EvidenceCoverage
    diagnostics: tuple[MetadataDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        lineages, diagnostics = _validate_capture_fields(
            self.lineages,
            self.saved_at,
            self.environment,
            self.environment_status,
            self.requirements,
            self.requirements_status,
            self.requirements_coverage,
            self.diagnostics,
        )
        object.__setattr__(self, "lineages", lineages)
        object.__setattr__(self, "diagnostics", diagnostics)


def encode_metadata_mapping(values: MetadataMapping) -> list[Any]:
    """Encode one bounded metadata mapping as fully tagged JSON-compatible data.

    Args:
        values: String-keyed user mapping containing supported scalar, list,
            tuple, and nested-mapping values.

    Returns:
        A detached tagged node preserving tuple, integer, and float distinctions.

    Raises:
        TypeError: If an input type is unsupported.
        CanonicalJSONError: If the mapping is cyclic, non-finite, malformed, or
            exceeds logical metadata bounds.
    """

    if not isinstance(values, Mapping):
        raise TypeError("metadata values must be a mapping")
    return _encode_metadata_value(values, 0, [0], set())


def decode_metadata_mapping(data: Any) -> dict[str, MetadataValue]:
    """Decode and validate one tagged metadata mapping.

    Args:
        data: Tagged metadata node produced by :func:`encode_metadata_mapping`.

    Returns:
        A newly allocated mutable mapping with original list/tuple distinctions.

    Raises:
        CanonicalJSONError: If tags, ordering, numeric spellings, or logical
            bounds are malformed.
    """

    value = _decode_metadata_value(data, 0, [0])
    if not isinstance(value, dict):
        raise CanonicalJSONError("metadata root must be a tagged mapping")
    return value


def timestamp_to_seconds(value: datetime) -> int | float:
    """Convert an aware UTC datetime to finite Unix seconds for persistence.

    Args:
        value: A timezone-aware UTC :class:`datetime.datetime`.

    Returns:
        Integer seconds when exact, otherwise fractional floating-point seconds.

    Raises:
        TypeError: If ``value`` is not a datetime.
        ValueError: If it is naive, non-UTC, non-finite, or outside supported
            timestamp conversion range.
    """

    _validate_timestamp(value)
    try:
        seconds = value.timestamp()
    except (OverflowError, OSError, ValueError) as error:
        raise ValueError("timestamp is outside the supported range") from error
    if not math.isfinite(seconds):
        raise ValueError("timestamp must be finite")
    return int(seconds) if seconds.is_integer() else seconds


def timestamp_from_seconds(value: int | float | None) -> datetime | None:
    """Decode numeric Unix seconds to an aware UTC datetime.

    Args:
        value: Integer or finite floating-point Unix seconds, or ``None`` for a
            supported unknown lifecycle timestamp.

    Returns:
        A UTC datetime, or ``None`` for unknown input.

    Raises:
        ValueError: If the numeric value is boolean, non-finite, unsupported, or
            outside Python's datetime range.
    """

    if value is None:
        return None
    if type(value) not in {int, float} or isinstance(value, bool):
        raise ValueError("timestamp must be an integer or finite float")
    if not math.isfinite(value):
        raise ValueError("timestamp must be finite")
    try:
        return datetime.fromtimestamp(value, tz=timezone.utc)
    except (OverflowError, OSError, ValueError) as error:
        raise ValueError("timestamp is outside the supported range") from error


def _encode_metadata_value(value: Any, depth: int, nodes: list[int], active: set[int]) -> list[Any]:
    nodes[0] += 1
    if nodes[0] > _METADATA_MAX_NODES:
        raise CanonicalJSONError("metadata value exceeds node bound")
    if depth > _METADATA_MAX_DEPTH:
        raise CanonicalJSONError("metadata value exceeds depth bound")
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        if value.bit_length() > _METADATA_MAX_INT_BITS:
            raise CanonicalJSONError("metadata integer exceeds bit bound")
        return ["int", str(value)]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalJSONError("metadata floats must be finite")
        return ["float", value.hex()]
    if isinstance(value, str):
        _check_metadata_string(value)
        return ["str", value]
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise CanonicalJSONError("metadata values cannot contain cycles")
        if len(value) > _METADATA_MAX_ENTRIES:
            raise CanonicalJSONError("metadata mapping exceeds entry bound")
        keys = list(value)
        for key in keys:
            if not isinstance(key, str):
                raise TypeError("metadata mapping keys must be strings")
            _check_metadata_string(key)
        active.add(identity)
        try:
            return ["map", [[key, _encode_metadata_value(value[key], depth + 1, nodes, active)] for key in sorted(keys)]]
        finally:
            active.remove(identity)
    if isinstance(value, list) or isinstance(value, tuple):
        identity = id(value)
        if identity in active:
            raise CanonicalJSONError("metadata values cannot contain cycles")
        if len(value) > _METADATA_MAX_ENTRIES:
            raise CanonicalJSONError("metadata sequence exceeds entry bound")
        active.add(identity)
        try:
            tag = "list" if isinstance(value, list) else "tuple"
            return [tag, [_encode_metadata_value(item, depth + 1, nodes, active) for item in value]]
        finally:
            active.remove(identity)
    raise TypeError(f"metadata value type {type(value).__name__} is unsupported")


def _decode_metadata_value(value: Any, depth: int, nodes: list[int]) -> MetadataValue:
    nodes[0] += 1
    if nodes[0] > _METADATA_MAX_NODES:
        raise CanonicalJSONError("metadata value exceeds node bound")
    if depth > _METADATA_MAX_DEPTH:
        raise CanonicalJSONError("metadata value exceeds depth bound")
    if not isinstance(value, list) or not value or not isinstance(value[0], str):
        raise CanonicalJSONError("metadata node must be a tagged list")
    tag = value[0]
    if tag == "null" and len(value) == 1:
        return None
    if tag == "bool" and len(value) == 2 and isinstance(value[1], bool):
        return value[1]
    if tag == "str" and len(value) == 2 and isinstance(value[1], str):
        _check_metadata_string(value[1])
        return value[1]
    if tag == "int" and len(value) == 2:
        if not isinstance(value[1], str) or _INTEGER.fullmatch(value[1]) is None:
            raise CanonicalJSONError("metadata integer spelling is not canonical")
        result = int(value[1])
        if result.bit_length() > _METADATA_MAX_INT_BITS:
            raise CanonicalJSONError("metadata integer exceeds bit bound")
        return result
    if tag == "float" and len(value) == 2:
        if not isinstance(value[1], str):
            raise CanonicalJSONError("metadata float spelling is not canonical")
        try:
            result = float.fromhex(value[1])
        except ValueError as error:
            raise CanonicalJSONError("metadata float spelling is malformed") from error
        if not math.isfinite(result) or result.hex() != value[1]:
            raise CanonicalJSONError("metadata float spelling is not canonical")
        return result
    if tag in {"list", "tuple"} and len(value) == 2 and isinstance(value[1], list):
        if len(value[1]) > _METADATA_MAX_ENTRIES:
            raise CanonicalJSONError("metadata sequence exceeds entry bound")
        result = [_decode_metadata_value(item, depth + 1, nodes) for item in value[1]]
        return result if tag == "list" else tuple(result)
    if tag == "map" and len(value) == 2 and isinstance(value[1], list):
        if len(value[1]) > _METADATA_MAX_ENTRIES:
            raise CanonicalJSONError("metadata mapping exceeds entry bound")
        result: dict[str, MetadataValue] = {}
        previous: str | None = None
        for pair in value[1]:
            if not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[0], str):
                raise CanonicalJSONError("metadata map entries must be key/value pairs")
            key = pair[0]
            _check_metadata_string(key)
            if key in result or (previous is not None and key <= previous):
                raise CanonicalJSONError("metadata map keys must be sorted and unique")
            result[key] = _decode_metadata_value(pair[1], depth + 1, nodes)
            previous = key
        return result
    raise CanonicalJSONError("metadata node tag or arity is unsupported")


def _check_metadata_string(value: str) -> None:
    if len(value) > _METADATA_MAX_STRING:
        raise CanonicalJSONError("metadata string exceeds length bound")


def _copy_mapping_or_none(value: MetadataMapping | None) -> dict[str, MetadataValue] | None:
    if value is None:
        return None
    return decode_metadata_mapping(encode_metadata_mapping(value))


def _validate_timestamp(value: datetime) -> None:
    if not isinstance(value, datetime):
        raise TypeError("timestamp must be a datetime")
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("timestamp must be timezone-aware UTC")


def _validate_optional_timestamp(value: datetime | None) -> None:
    if value is not None:
        _validate_timestamp(value)


def _validate_capture_fields(lineages: Mapping[GraphPath, NodeLineageMetadata], saved_at: datetime, environment: EnvironmentRecord | None, environment_status: str, requirements: EnvironmentRequirement | None, requirements_status: str, requirements_coverage: str, diagnostics: tuple[MetadataDiagnostic, ...]) -> tuple[Mapping[GraphPath, NodeLineageMetadata], tuple[MetadataDiagnostic, ...]]:
    if not isinstance(lineages, Mapping):
        raise TypeError("snapshot lineages must be a mapping")
    normalized: dict[GraphPath, NodeLineageMetadata] = {}
    for path, lineage in lineages.items():
        if not isinstance(path, GraphPath):
            raise TypeError("snapshot lineage paths must be GraphPath values")
        if path in normalized:
            raise ValueError("snapshot lineages contain duplicate paths")
        if not isinstance(lineage, NodeLineageMetadata):
            raise TypeError("snapshot lineage values must be NodeLineageMetadata")
        normalized[path] = lineage
    _validate_timestamp(saved_at)
    if environment is not None and not isinstance(environment, EnvironmentRecord):
        raise TypeError("snapshot environment must be an EnvironmentRecord or None")
    if environment_status not in {"known", "incomplete", "unavailable"}:
        raise ValueError("snapshot environment_status is unsupported")
    if environment_status == "known" and environment is None:
        raise ValueError("known snapshot environment requires an EnvironmentRecord")
    if environment_status == "unavailable" and environment is not None:
        raise ValueError("unavailable snapshot environment cannot carry a value")
    if requirements is not None and not isinstance(requirements, EnvironmentRequirement):
        raise TypeError("snapshot requirements must be an EnvironmentRequirement or None")
    if requirements_status not in {"empty", "value", "conflict", "unavailable"}:
        raise ValueError("snapshot requirements_status is unsupported")
    if requirements_coverage not in {"complete", "incomplete"}:
        raise ValueError("snapshot requirements_coverage is unsupported")
    if (requirements_status == "value") != (requirements is not None):
        raise ValueError("valued snapshot requirements require a value and other outcomes require null")
    if requirements_status == "unavailable" and requirements_coverage != "incomplete":
        raise ValueError("unavailable snapshot requirements require incomplete coverage")
    return MappingProxyType(dict(sorted(normalized.items(), key=lambda item: graph_path_sort_key(item[0])))), _validate_diagnostics(diagnostics)


def _validate_diagnostics(value: tuple[MetadataDiagnostic, ...]) -> tuple[MetadataDiagnostic, ...]:
    if not isinstance(value, tuple):
        raise TypeError("snapshot diagnostics must be a tuple")
    if len(value) > _DIAGNOSTIC_MAX_ENTRIES:
        raise ValueError("snapshot diagnostics exceed entry bound")
    result: list[MetadataDiagnostic] = []
    for entry in value:
        if not isinstance(entry, tuple) or len(entry) != 2 or not all(isinstance(item, str) for item in entry):
            raise TypeError("snapshot diagnostics must contain string code/message pairs")
        if any(len(item) > _DIAGNOSTIC_MAX_STRING for item in entry):
            raise ValueError("snapshot diagnostic text exceeds length bound")
        result.append(entry)
    return tuple(result)


def _captured_annotation_data(value: MetadataMapping | None) -> dict[str, Any]:
    return {"present": value is not None, "values": None if value is None else encode_metadata_mapping(value)}


def _decode_captured_annotation(data: Any) -> dict[str, MetadataValue] | None:
    if not isinstance(data, Mapping):
        raise EnvelopeError("captured annotation entry must be a mapping")
    _closed(data, {"present", "values"}, "captured annotation entry")
    if type(data["present"]) is not bool:
        raise EnvelopeError("captured annotation presence must be a boolean")
    if not data["present"]:
        if data["values"] is not None:
            raise EnvelopeError("absent captured annotations require null values")
        return None
    if data["values"] is None:
        raise EnvelopeError("present captured annotations require tagged values")
    return _decode_as_envelope(decode_metadata_mapping, data["values"], "captured annotation values")


def _closed(data: Mapping[str, Any], fields: set[str], name: str) -> None:
    observed = set(data)
    if observed != fields:
        raise EnvelopeError(f"{name} fields are closed", context={"unknown": sorted(observed - fields), "missing": sorted(fields - observed)})


def _decode_as_envelope(function: Any, value: Any, name: str) -> Any:
    try:
        return function(value)
    except (TypeError, ValueError, CanonicalJSONError) as error:
        raise EnvelopeError(f"{name} is malformed") from error


__all__ = [
    "EvidenceCoverage",
    "EnvironmentStatus",
    "NodeLineageMetadata",
    "NodeSnapshotMetadata",
    "MetadataDiagnostic",
    "MetadataMapping",
    "MetadataScalar",
    "MetadataTarget",
    "MetadataValue",
    "RequirementStatus",
    "SaveAnnotations",
    "SnapshotCapture",
    "decode_metadata_mapping",
    "encode_metadata_mapping",
    "timestamp_from_seconds",
    "timestamp_to_seconds",
]
