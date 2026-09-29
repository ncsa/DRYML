"""Immutable observed-environment records in the closed v1.1 format."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from dryml.formats import canonical_json_bytes, deep_freeze_json, json_ready, make_envelope, semantic_id, validate_envelope

from .errors import EnvironmentSerializationError
from .utils import normalize_distribution_name

_SCHEMA = "dryml.environment_record.v1.1"
_KIND = "environment_record"
_PREFIX = "envrec"
_RECORD_BOUNDS = {"max_entries": 4096, "max_nodes": 65536}


def _closed(data: Mapping[str, Any], fields: set[str], name: str) -> None:
    unknown, missing = set(data) - fields, fields - set(data)
    if unknown or missing:
        raise EnvironmentSerializationError(f"{name} fields are closed", context={"unknown": sorted(unknown), "missing": sorted(missing)})


@dataclass(frozen=True, slots=True)
class PackageRecord:
    """Observed distribution facts, with location facts excluded from identity."""

    name: str
    version: str | None
    normalized_name: str | None = None
    metadata_name: str | None = None
    location: str | None = None
    installer: str | None = None
    editable: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "normalized_name", normalize_distribution_name(self.normalized_name or self.name))
        deep_freeze_json(self.to_payload())

    def to_payload(self) -> dict[str, Any]:
        """Return the complete closed distribution payload fragment."""

        return {"name": self.name, "normalized_name": self.normalized_name, "version": self.version, "metadata_name": self.metadata_name, "location": self.location, "installer": self.installer, "editable": self.editable}

    @classmethod
    def from_payload(cls, data: Mapping[str, Any]) -> "PackageRecord":
        """Build one distribution record from its closed payload fragment."""

        _closed(data, {"name", "normalized_name", "version", "metadata_name", "location", "installer", "editable"}, "distribution")
        return cls(**dict(data))


@dataclass(frozen=True, slots=True)
class PythonRecord:
    """Observed interpreter facts, where executable and prefixes are display-only."""

    version: str
    implementation: str
    executable: str | None = None
    prefix: str | None = None
    base_prefix: str | None = None

    def __post_init__(self) -> None:
        deep_freeze_json(self.to_payload())

    def to_payload(self) -> dict[str, Any]:
        """Return the complete closed interpreter payload fragment."""

        return {"version": self.version, "implementation": self.implementation, "executable": self.executable, "prefix": self.prefix, "base_prefix": self.base_prefix}

    @classmethod
    def from_payload(cls, data: Mapping[str, Any]) -> "PythonRecord":
        """Build an interpreter record from a closed payload fragment."""

        _closed(data, {"version", "implementation", "executable", "prefix", "base_prefix"}, "python")
        return cls(**dict(data))


@dataclass(frozen=True, slots=True)
class PlatformRecord:
    """Observed platform facts that all contribute to record identity."""

    system: str
    release: str
    version: str
    machine: str
    platform: str
    os_name: str | None = None
    sys_platform: str | None = None
    implementation_name: str | None = None
    implementation_version: str | None = None
    platform_python_implementation: str | None = None

    def __post_init__(self) -> None:
        deep_freeze_json(self.to_payload())

    def to_payload(self) -> dict[str, Any]:
        """Return the complete closed platform payload fragment."""

        return {"system": self.system, "release": self.release, "version": self.version, "machine": self.machine, "platform": self.platform, "os_name": self.os_name, "sys_platform": self.sys_platform, "implementation_name": self.implementation_name, "implementation_version": self.implementation_version, "platform_python_implementation": self.platform_python_implementation}

    @classmethod
    def from_payload(cls, data: Mapping[str, Any]) -> "PlatformRecord":
        """Build platform facts from a closed payload fragment."""

        _closed(data, set(cls.__dataclass_fields__), "platform")
        return cls(**dict(data))


@dataclass(frozen=True, slots=True)
class DrymlRuntimeRecord:
    """Observed DRYML protocol/schema facts, with git revision non-identifying."""

    version: str | None = None
    git_revision: str | None = None
    execution_protocol: str | None = None
    schema_versions: Mapping[str, str] = field(default_factory=dict)
    features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_versions", deep_freeze_json(self.schema_versions))
        object.__setattr__(self, "features", tuple(sorted(set(self.features))))
        deep_freeze_json(self.to_payload())

    def to_payload(self) -> dict[str, Any]:
        """Return the complete closed DRYML payload fragment."""

        return {"version": self.version, "git_revision": self.git_revision, "execution_protocol": self.execution_protocol, "schema_versions": json_ready(self.schema_versions), "features": list(self.features)}

    @classmethod
    def from_payload(cls, data: Mapping[str, Any]) -> "DrymlRuntimeRecord":
        """Build DRYML facts from a closed payload fragment."""

        _closed(data, set(cls.__dataclass_fields__), "dryml")
        return cls(**dict(data))


@dataclass(frozen=True, slots=True)
class EnvironmentRecord:
    """Typed immutable observation of one software environment.

    ``to_data`` emits a closed v1.1 envelope.  Interpreter paths, distribution
    installation facts, git revision, and ``details`` remain inspectable but do
    not affect :attr:`semantic_id`.
    """

    python: PythonRecord
    platform: PlatformRecord
    distributions: Mapping[str, PackageRecord] = field(default_factory=dict)
    dryml: DrymlRuntimeRecord | None = None
    kind: str = "unknown"
    tags: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        packages = {}
        for key, value in self.distributions.items():
            package = value if isinstance(value, PackageRecord) else PackageRecord.from_payload(value)
            packages[normalize_distribution_name(key)] = package
        if len(packages) > 4096:
            raise EnvironmentSerializationError("environment record exceeds distribution bound", context={"limit": 4096})
        object.__setattr__(self, "distributions", MappingProxyType({key: packages[key] for key in sorted(packages)}))
        object.__setattr__(self, "tags", tuple(sorted(set(self.tags))))
        object.__setattr__(self, "details", deep_freeze_json(self.details))
        object.__setattr__(self, "metadata", deep_freeze_json(self.metadata))
        deep_freeze_json({"kind": self.kind, "tags": self.tags})

    @property
    def semantic_id(self) -> str:
        """Return this record's stable v1.1 semantic identifier."""

        return semantic_id(_PREFIX, _SCHEMA, _KIND, self._identifying_payload(), **_RECORD_BOUNDS)

    @property
    def id(self) -> str:
        """Alias for :attr:`semantic_id` retained for typed environment users."""

        return self.semantic_id

    def _payload(self) -> dict[str, Any]:
        return {"python": self.python.to_payload(), "platform": self.platform.to_payload(), "distributions": {key: value.to_payload() for key, value in self.distributions.items()}, "dryml": None if self.dryml is None else self.dryml.to_payload(), "kind": self.kind, "tags": list(self.tags), "details": json_ready(self.details, **_RECORD_BOUNDS)}

    def _identifying_payload(self) -> dict[str, Any]:
        python = self.python.to_payload()
        python = {key: python[key] for key in ("version", "implementation")}
        distributions = {key: {name: value for name, value in package.to_payload().items() if name not in {"location", "installer", "editable"}} for key, package in self.distributions.items()}
        dryml = None if self.dryml is None else {key: value for key, value in self.dryml.to_payload().items() if key != "git_revision"}
        return {"python": python, "platform": self.platform.to_payload(), "distributions": distributions, "dryml": dryml, "kind": self.kind, "tags": list(self.tags)}

    def to_data(self) -> dict[str, Any]:
        """Return a complete bounded v1.1 environment-record envelope."""

        return make_envelope(schema=_SCHEMA, kind=_KIND, prefix=_PREFIX, payload=self._payload(), semantic_id=self.semantic_id, identifying_payload=self._identifying_payload(), metadata=self.metadata, max_bytes=16_777_216, **_RECORD_BOUNDS)

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "EnvironmentRecord":
        """Validate and decode a closed v1.1 environment-record envelope."""

        raw = dict(data)
        attached = raw.pop("id", None)
        envelope = validate_envelope(raw, schema=_SCHEMA, kind=_KIND, prefix=_PREFIX, identifying_payload=raw.get("payload", {}), max_bytes=16_777_216, **_RECORD_BOUNDS)
        payload = envelope["payload"]
        _closed(payload, {"python", "platform", "distributions", "dryml", "kind", "tags", "details"}, "environment record payload")
        if not isinstance(payload["distributions"], Mapping):
            raise EnvironmentSerializationError("environment record distributions must be a mapping")
        value = cls(python=PythonRecord.from_payload(payload["python"]), platform=PlatformRecord.from_payload(payload["platform"]), distributions={key: PackageRecord.from_payload(item) for key, item in payload["distributions"].items()}, dryml=None if payload["dryml"] is None else DrymlRuntimeRecord.from_payload(payload["dryml"]), kind=payload["kind"], tags=tuple(payload["tags"]), details=payload["details"], metadata=envelope.get("metadata", {}))
        if attached is not None and attached != value.semantic_id:
            raise EnvironmentSerializationError("environment record attached ID does not match payload", context={"expected": value.semantic_id, "observed": attached})
        return value


class EnvironmentInternTable:
    """In-memory interning table keyed by immutable v1.1 semantic identifiers."""

    def __init__(self) -> None:
        self._records: dict[str, EnvironmentRecord] = {}
        self._requirements: dict[str, Any] = {}

    def intern_record(self, record: EnvironmentRecord) -> EnvironmentRecord:
        """Return the canonical local instance for an environment record."""

        return self._records.setdefault(record.semantic_id, record)

    def intern_requirement(self, requirement: Any) -> Any:
        """Return the canonical local instance for an environment requirement."""

        return self._requirements.setdefault(requirement.semantic_id, requirement)


__all__ = ["DrymlRuntimeRecord", "EnvironmentInternTable", "EnvironmentRecord", "PackageRecord", "PlatformRecord", "PythonRecord"]
