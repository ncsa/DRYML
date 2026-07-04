"""Typed helpers for DRYML representation specs and requirements."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from dryml.formats import CanonicalJSONError, deep_freeze_json, json_ready
from dryml.formats.errors import ContentIDError
from dryml.formats.ids import parse_content_id

from .errors import SpecValidationError
from .specs import attach_spec_id, make_spec, validate_spec


CompatibilityStatus = Literal["compatible", "incompatible"]


@dataclass(frozen=True, slots=True)
class RepresentationSpec:
    """Wrapper over a normal ``family='representation'`` spec envelope.

    The wrapper does not introduce a new spec family. It validates and exposes
    the semantic payload fields used by record resolution: version, parameters,
    traits, and storage kinds.
    """

    envelope: Mapping[str, Any]

    def __post_init__(self) -> None:
        validated = validate_representation_spec(self.envelope)
        object.__setattr__(self, "envelope", deep_freeze_json(validated))

    @classmethod
    def create(
        cls,
        kind: str,
        *,
        version: str | None = None,
        parameters: Mapping[str, Any] | None = None,
        traits: tuple[str, ...] | list[str] = (),
        storage_kinds: tuple[str, ...] | list[str] = (),
        payload: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "RepresentationSpec":
        """Build a representation spec and attach its stable ``repr-v1`` ID."""

        return cls(make_representation_spec(kind, version=version, parameters=parameters, traits=traits, storage_kinds=storage_kinds, payload=payload, metadata=metadata))

    @property
    def id(self) -> str:
        """Return the attached representation spec ID."""

        return self.envelope["id"]  # type: ignore[return-value]

    @property
    def kind(self) -> str:
        """Return the representation kind."""

        return self.envelope["kind"]  # type: ignore[return-value]

    @property
    def payload(self) -> Mapping[str, Any]:
        """Return the frozen semantic payload."""

        return self.envelope["payload"]  # type: ignore[return-value]

    @property
    def version(self) -> str | None:
        """Return the optional representation version."""

        version = self.payload.get("version")
        return None if version is None else str(version)

    @property
    def parameters(self) -> Mapping[str, Any]:
        """Return representation parameters as frozen JSON."""

        value = self.payload.get("parameters") or {}
        if not isinstance(value, Mapping):
            raise SpecValidationError("representation parameters must be a mapping")
        return value

    @property
    def traits(self) -> tuple[str, ...]:
        """Return declared representation traits."""

        return _string_tuple(self.payload.get("traits") or (), "traits")

    @property
    def storage_kinds(self) -> tuple[str, ...]:
        """Return storage kinds accepted by this representation.

        ``storage_kinds`` is canonical. Legacy/default payloads that use the
        singular ``storage_kind`` are accepted for compatibility.
        """

        if "storage_kinds" in self.payload:
            return _string_tuple(self.payload.get("storage_kinds") or (), "storage_kinds")
        if "storage_kind" in self.payload:
            return (str(self.payload["storage_kind"]),)
        return ()

    def to_envelope(self) -> dict[str, Any]:
        """Return the JSON-ready spec envelope."""

        return json_ready(self.envelope)


@dataclass(frozen=True, slots=True)
class RepresentationRequirement:
    """Conservative representation query used by resolution and adapters."""

    kind: str | None = None
    representation_id: str | None = None
    version: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    required_traits: tuple[str, ...] = ()
    storage_kinds: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind is not None and (not isinstance(self.kind, str) or not self.kind):
            raise SpecValidationError("representation requirement kind must be a non-empty string")
        if self.representation_id is not None:
            _validate_repr_id(self.representation_id)
        if self.version is not None and not isinstance(self.version, str):
            raise SpecValidationError("representation requirement version must be a string")
        object.__setattr__(self, "parameters", _freeze_mapping(self.parameters, "parameters"))
        object.__setattr__(self, "required_traits", _string_tuple(self.required_traits, "required_traits"))
        object.__setattr__(self, "storage_kinds", _string_tuple(self.storage_kinds, "storage_kinds"))

    def to_json(self) -> dict[str, Any]:
        """Return a compact JSON-ready requirement mapping."""

        data: dict[str, Any] = {}
        if self.kind is not None:
            data["kind"] = self.kind
        if self.representation_id is not None:
            data["representation_id"] = self.representation_id
        if self.version is not None:
            data["version"] = self.version
        if self.parameters:
            data["parameters"] = json_ready(self.parameters)
        if self.required_traits:
            data["required_traits"] = list(self.required_traits)
        if self.storage_kinds:
            data["storage_kinds"] = list(self.storage_kinds)
        return data

    @classmethod
    def from_json(cls, data: Mapping[str, Any] | None) -> "RepresentationRequirement":
        """Build a requirement from JSON-ready data."""

        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise SpecValidationError("representation requirement must be a mapping", context={"type": type(data).__name__})
        unknown = set(data) - {"kind", "representation_id", "version", "parameters", "required_traits", "storage_kinds"}
        if unknown:
            raise SpecValidationError("representation requirement has unknown fields", context={"fields": sorted(unknown)})
        return cls(
            kind=data.get("kind"),
            representation_id=data.get("representation_id"),
            version=data.get("version"),
            parameters=data.get("parameters") or {},
            required_traits=tuple(data.get("required_traits") or ()),
            storage_kinds=tuple(data.get("storage_kinds") or ()),
        )


@dataclass(frozen=True, slots=True)
class RepresentationCompatibilityReport:
    """Structured result for deterministic representation compatibility checks."""

    status: CompatibilityStatus
    requirement: RepresentationRequirement
    representation_id: str | None = None
    representation_kind: str | None = None
    issues: tuple[str, ...] = ()

    @property
    def compatible(self) -> bool:
        """Return whether the requirement is satisfied."""

        return self.status == "compatible"

    def to_json(self) -> dict[str, Any]:
        """Return JSON-ready report data."""

        return {
            "status": self.status,
            "requirement": self.requirement.to_json(),
            "representation_id": self.representation_id,
            "representation_kind": self.representation_kind,
            "issues": list(self.issues),
        }


def make_representation_spec(
    kind: str,
    *,
    version: str | None = None,
    parameters: Mapping[str, Any] | None = None,
    traits: tuple[str, ...] | list[str] = (),
    storage_kinds: tuple[str, ...] | list[str] = (),
    payload: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and attach a canonical ``dryml.representation.v1`` spec."""

    if payload is None:
        spec_payload: dict[str, Any] = {}
    else:
        spec_payload = dict(_freeze_mapping(payload, "payload"))
    if version is not None:
        spec_payload["version"] = version
    if parameters is not None:
        spec_payload["parameters"] = _freeze_mapping(parameters, "parameters")
    if traits:
        spec_payload["traits"] = tuple(_string_tuple(traits, "traits"))
    if storage_kinds:
        spec_payload["storage_kinds"] = tuple(_string_tuple(storage_kinds, "storage_kinds"))
    return attach_spec_id(make_spec(family="representation", kind=kind, payload=spec_payload, metadata=metadata), family="representation")


def validate_representation_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a representation spec envelope with an attached ID."""

    attached = attach_spec_id(spec, family="representation")
    validate_spec(attached, family="representation")
    if attached.get("schema") != "dryml.representation.v1":
        raise SpecValidationError("representation spec schema mismatch", context={"schema": attached.get("schema")})
    if not isinstance(attached.get("kind"), str) or not attached["kind"]:
        raise SpecValidationError("representation kind must be a non-empty string")
    payload = attached.get("payload")
    if not isinstance(payload, Mapping):
        raise SpecValidationError("representation payload must be a mapping", context={"type": type(payload).__name__})
    if "version" in payload and not isinstance(payload["version"], str):
        raise SpecValidationError("representation version must be a string")
    if "parameters" in payload and not isinstance(payload["parameters"], Mapping):
        raise SpecValidationError("representation parameters must be a mapping")
    if "traits" in payload:
        _string_tuple(payload["traits"], "traits")
    if "storage_kinds" in payload:
        _string_tuple(payload["storage_kinds"], "storage_kinds")
    if "storage_kind" in payload and not isinstance(payload["storage_kind"], str):
        raise SpecValidationError("representation storage_kind must be a string")
    return attached


def representation_satisfies(spec: RepresentationSpec | Mapping[str, Any], requirement: RepresentationRequirement | Mapping[str, Any] | None) -> RepresentationCompatibilityReport:
    """Check conservative exact compatibility between a spec and requirement.

    Storage kind semantics are subset based: every requested storage kind must
    be declared by the spec. If the requirement has no storage kinds, storage is
    not considered.
    """

    wrapped = spec if isinstance(spec, RepresentationSpec) else RepresentationSpec(spec)
    req = requirement if isinstance(requirement, RepresentationRequirement) else RepresentationRequirement.from_json(requirement)
    issues: list[str] = []
    if req.representation_id is not None and wrapped.id != req.representation_id:
        issues.append("representation_id")
    if req.kind is not None and wrapped.kind != req.kind:
        issues.append("kind")
    if req.version is not None and wrapped.version != req.version:
        issues.append("version")
    for key, value in req.parameters.items():
        if wrapped.parameters.get(key) != value:
            issues.append(f"parameter:{key}")
    missing_traits = sorted(set(req.required_traits) - set(wrapped.traits))
    issues.extend(f"trait:{trait}" for trait in missing_traits)
    missing_storage = sorted(set(req.storage_kinds) - set(wrapped.storage_kinds))
    issues.extend(f"storage_kind:{kind}" for kind in missing_storage)
    return RepresentationCompatibilityReport(
        status="compatible" if not issues else "incompatible",
        requirement=req,
        representation_id=wrapped.id,
        representation_kind=wrapped.kind,
        issues=tuple(issues),
    )


def _validate_repr_id(value: str) -> None:
    try:
        parts = parse_content_id(value)
    except ContentIDError as exc:
        raise SpecValidationError("invalid representation ID", context=exc.context) from exc
    if parts.prefix != "repr" or parts.schema_version != 1:
        raise SpecValidationError("representation ID must use repr-v1 prefix", context={"value": value})


def _freeze_mapping(value: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpecValidationError("representation JSON field must be a mapping", context={"path": path, "type": type(value).__name__})
    try:
        frozen = deep_freeze_json(value)
    except CanonicalJSONError as exc:
        raise SpecValidationError("representation JSON field is not canonical JSON", context={"path": path, **exc.context}) from exc
    assert isinstance(frozen, Mapping)
    return frozen


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise SpecValidationError(f"representation {field_name} must be a list", context={"type": type(value).__name__})
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise SpecValidationError(f"representation {field_name} entries must be non-empty strings")
    return result


__all__ = [
    "RepresentationCompatibilityReport",
    "RepresentationRequirement",
    "RepresentationSpec",
    "make_representation_spec",
    "representation_satisfies",
    "validate_representation_spec",
]
