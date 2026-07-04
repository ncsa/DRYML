"""Provider probe request models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from dryml.formats import CanonicalJSONError, deep_freeze_json, json_ready
from dryml.formats.ids import content_id
from dryml.operations import attach_operation_id, compute_operation_id, validate_operation_spec
from dryml.operations.errors import OperationSpecError

from .errors import ProviderValidationError


PROBE_POLICY_SCHEMA_VERSION = 1
REQUEST_SCHEMA_VERSION = 1
REQUEST_ID_PREFIX = "providerreq"


@dataclass(frozen=True, slots=True)
class ProbePolicy:
    """Policy governing what provider probes may do in the target process."""

    allow_materialization: bool = False
    device_visibility: str = "none"
    allow_workload_allocation: bool = False
    strict_preimport: bool = False
    timeout: float | None = 30.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = PROBE_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.allow_materialization:
            raise ProviderValidationError("allow_materialization is not supported by provider probes yet", context={"code": "unsupported_probe_policy"})
        if self.allow_workload_allocation:
            raise ProviderValidationError("provider probes cannot hold workload allocations", context={"code": "unsupported_probe_policy"})
        if self.device_visibility not in {"none", "inherit", "explicit"}:
            raise ProviderValidationError("unsupported provider probe device visibility", context={"device_visibility": self.device_visibility})
        if self.timeout is not None and (not isinstance(self.timeout, int | float) or self.timeout < 0):
            raise ProviderValidationError("probe timeout must be non-negative or None", context={"timeout": self.timeout})
        object.__setattr__(self, "timeout", None if self.timeout is None else float(self.timeout))
        object.__setattr__(self, "metadata", _freeze_json_mapping(self.metadata, "probe_policy.metadata"))
        if self.schema_version != PROBE_POLICY_SCHEMA_VERSION:
            raise ProviderValidationError("unsupported probe policy schema version", context={"schema_version": self.schema_version})

    def to_data(self) -> dict[str, Any]:
        """Return JSON-ready policy data."""

        return {
            "schema_version": self.schema_version,
            "allow_materialization": self.allow_materialization,
            "device_visibility": self.device_visibility,
            "allow_workload_allocation": self.allow_workload_allocation,
            "strict_preimport": self.strict_preimport,
            "timeout": self.timeout,
            "metadata": json_ready(self.metadata),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any] | None) -> "ProbePolicy":
        """Build a probe policy from JSON-ready data."""

        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ProviderValidationError("probe policy must be a mapping", context={"type": type(data).__name__})
        unknown = set(data) - {"schema_version", "allow_materialization", "device_visibility", "allow_workload_allocation", "strict_preimport", "timeout", "metadata"}
        if unknown:
            raise ProviderValidationError("probe policy has unknown fields", context={"fields": sorted(unknown)})
        return cls(
            allow_materialization=bool(data.get("allow_materialization", False)),
            device_visibility=data.get("device_visibility", "none"),
            allow_workload_allocation=bool(data.get("allow_workload_allocation", False)),
            strict_preimport=bool(data.get("strict_preimport", False)),
            timeout=data.get("timeout", 30.0),
            metadata=data.get("metadata") or {},
            schema_version=data.get("schema_version", PROBE_POLICY_SCHEMA_VERSION),
        )


@dataclass(frozen=True, slots=True)
class ProviderRequest:
    """Base request shape shared by provider operations."""

    request_kind: ClassVar[str] = "provider_request"
    schema_version: int = REQUEST_SCHEMA_VERSION
    request_key: str | None = None
    environment_spec: Mapping[str, Any] | None = None
    runtime_spec: Mapping[str, Any] | None = None
    provider_names: tuple[str, ...] = ()
    provider_options: Mapping[str, Any] = field(default_factory=dict)
    annotation_context: Mapping[str, Any] = field(default_factory=dict)
    probe_policy: ProbePolicy = field(default_factory=ProbePolicy)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != REQUEST_SCHEMA_VERSION:
            raise ProviderValidationError("unsupported provider request schema version", context={"schema_version": self.schema_version})
        object.__setattr__(self, "environment_spec", None if self.environment_spec is None else _freeze_json_mapping(self.environment_spec, "environment_spec"))
        object.__setattr__(self, "runtime_spec", None if self.runtime_spec is None else _freeze_json_mapping(self.runtime_spec, "runtime_spec"))
        object.__setattr__(self, "provider_names", tuple(str(name) for name in self.provider_names))
        object.__setattr__(self, "provider_options", _freeze_json_mapping(self.provider_options, "provider_options"))
        object.__setattr__(self, "annotation_context", _freeze_json_mapping(self.annotation_context, "annotation_context"))
        object.__setattr__(self, "metadata", _freeze_json_mapping(self.metadata, "metadata"))
        if self.request_key is not None and not isinstance(self.request_key, str):
            raise ProviderValidationError("request_key must be a string", context={"type": type(self.request_key).__name__})

    @property
    def key(self) -> str:
        """Return a stable content key for this request."""

        return self.request_key or content_id(REQUEST_ID_PREFIX, self.schema_version, self._data(include_request_key=False))

    def to_data(self) -> dict[str, Any]:
        """Return canonical JSON-ready request data."""

        return self._data(include_request_key=True)

    def _data(self, *, include_request_key: bool) -> dict[str, Any]:
        data = {
            "schema_version": self.schema_version,
            "request_kind": self.request_kind,
            "environment_spec": None if self.environment_spec is None else json_ready(self.environment_spec),
            "runtime_spec": None if self.runtime_spec is None else json_ready(self.runtime_spec),
            "provider_names": list(self.provider_names),
            "provider_options": json_ready(self.provider_options),
            "annotation_context": json_ready(self.annotation_context),
            "probe_policy": self.probe_policy.to_data(),
            "metadata": json_ready(self.metadata),
        }
        if include_request_key:
            data["request_key"] = self.key
        return data

    @classmethod
    def _common_kwargs(cls, data: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": data.get("schema_version", REQUEST_SCHEMA_VERSION),
            "request_key": data.get("request_key"),
            "environment_spec": data.get("environment_spec"),
            "runtime_spec": data.get("runtime_spec"),
            "provider_names": tuple(data.get("provider_names") or ()),
            "provider_options": data.get("provider_options") or {},
            "annotation_context": data.get("annotation_context") or {},
            "probe_policy": ProbePolicy.from_data(data.get("probe_policy")),
            "metadata": data.get("metadata") or {},
        }


@dataclass(frozen=True, slots=True)
class OperationInspectionRequest(ProviderRequest):
    """Request asking providers to inspect a canonical operation spec."""

    request_kind: ClassVar[str] = "operation_inspection"
    operation_spec: Mapping[str, Any] = field(default_factory=dict)
    operation_id: str | None = None

    def __post_init__(self) -> None:
        ProviderRequest.__post_init__(self)
        try:
            operation_spec = attach_operation_id(validate_operation_spec(self.operation_spec))
        except OperationSpecError as exc:
            raise ProviderValidationError("invalid operation inspection request operation_spec", context=exc.context) from exc
        object.__setattr__(self, "operation_spec", _freeze_json_mapping(operation_spec, "operation_spec"))
        operation_id = self.operation_id or compute_operation_id(operation_spec)
        if operation_id != operation_spec["id"]:
            raise ProviderValidationError("operation_id does not match operation_spec", context={"expected": operation_spec["id"], "observed": operation_id})
        object.__setattr__(self, "operation_id", operation_id)

    def _data(self, *, include_request_key: bool) -> dict[str, Any]:
        data = ProviderRequest._data(self, include_request_key=include_request_key)
        data["operation_spec"] = json_ready(self.operation_spec)
        data["operation_id"] = self.operation_id
        return data

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "OperationInspectionRequest":
        """Build an operation-inspection request from JSON-ready data."""

        _validate_request_data(data, cls.request_kind, {"operation_spec", "operation_id"})
        return cls(operation_spec=data.get("operation_spec") or {}, operation_id=data.get("operation_id"), **cls._common_kwargs(data))


@dataclass(frozen=True, slots=True)
class RepresentationInspectionRequest(ProviderRequest):
    """Structural request placeholder for representation discovery."""

    request_kind: ClassVar[str] = "representation_inspection"

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "RepresentationInspectionRequest":
        _validate_request_data(data, cls.request_kind, set())
        return cls(**cls._common_kwargs(data))


@dataclass(frozen=True, slots=True)
class AdapterPlanningRequest(ProviderRequest):
    """Structural request placeholder for adapter planning."""

    request_kind: ClassVar[str] = "adapter_planning"

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "AdapterPlanningRequest":
        _validate_request_data(data, cls.request_kind, set())
        return cls(**cls._common_kwargs(data))


@dataclass(frozen=True, slots=True)
class CompatibilityCheckRequest(ProviderRequest):
    """Structural request placeholder for compatibility checks."""

    request_kind: ClassVar[str] = "compatibility_check"

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "CompatibilityCheckRequest":
        _validate_request_data(data, cls.request_kind, set())
        return cls(**cls._common_kwargs(data))


@dataclass(frozen=True, slots=True)
class LoweringRequest(OperationInspectionRequest):
    """Request asking providers whether an operation can be lowered."""

    request_kind: ClassVar[str] = "lowering"

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "LoweringRequest":
        _validate_request_data(data, cls.request_kind, {"operation_spec", "operation_id"})
        return cls(operation_spec=data.get("operation_spec") or {}, operation_id=data.get("operation_id"), **cls._common_kwargs(data))


REQUEST_TYPES = {
    OperationInspectionRequest.request_kind: OperationInspectionRequest,
    RepresentationInspectionRequest.request_kind: RepresentationInspectionRequest,
    AdapterPlanningRequest.request_kind: AdapterPlanningRequest,
    CompatibilityCheckRequest.request_kind: CompatibilityCheckRequest,
    LoweringRequest.request_kind: LoweringRequest,
}


def request_from_data(data: Mapping[str, Any]) -> ProviderRequest:
    """Deserialize any provider request by its ``request_kind``."""

    if not isinstance(data, Mapping):
        raise ProviderValidationError("provider request must be a mapping", context={"type": type(data).__name__})
    request_kind = data.get("request_kind")
    try:
        request_type = REQUEST_TYPES[request_kind]
    except KeyError as exc:
        raise ProviderValidationError("unknown provider request kind", context={"request_kind": request_kind}) from exc
    return request_type.from_data(data)


def _validate_request_data(data: Mapping[str, Any], request_kind: str, extra_fields: set[str]) -> None:
    if not isinstance(data, Mapping):
        raise ProviderValidationError("provider request must be a mapping", context={"type": type(data).__name__})
    allowed = {"schema_version", "request_kind", "request_key", "environment_spec", "runtime_spec", "provider_names", "provider_options", "annotation_context", "probe_policy", "metadata"} | extra_fields
    unknown = set(data) - allowed
    if unknown:
        raise ProviderValidationError("provider request has unknown fields", context={"fields": sorted(unknown)})
    if data.get("request_kind") != request_kind:
        raise ProviderValidationError("provider request kind mismatch", context={"expected": request_kind, "observed": data.get("request_kind")})


def _freeze_json_mapping(value: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProviderValidationError("provider request payload must be a mapping", context={"path": path, "type": type(value).__name__})
    try:
        frozen = deep_freeze_json(value)
    except CanonicalJSONError as exc:
        raise ProviderValidationError("provider request payload is not JSON-ready", context={"path": path, **exc.context}) from exc
    assert isinstance(frozen, Mapping)
    return frozen


__all__ = [
    "AdapterPlanningRequest",
    "CompatibilityCheckRequest",
    "LoweringRequest",
    "OperationInspectionRequest",
    "ProbePolicy",
    "ProviderRequest",
    "RepresentationInspectionRequest",
    "request_from_data",
]
