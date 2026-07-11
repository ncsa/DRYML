"""Serializable records describing observed Python/software environments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .ids import content_id
from .schema import ENVIRONMENT_RECORD_SCHEMA_VERSION
from .serialization import deep_freeze_json, freeze_mapping, json_ready
from .utils import coerce_tuple, normalize_distribution_name


def _mapping(data: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return data


def _required_string(data: Mapping[str, Any], name: str) -> str:
    value = data[name]
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _optional_string(data: Mapping[str, Any], name: str) -> str | None:
    value = data.get(name)
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None")
    return value


def _schema_version(data: Mapping[str, Any]) -> int:
    value = data.get("schema_version", ENVIRONMENT_RECORD_SCHEMA_VERSION)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("schema_version must be an integer")
    return value


def _string_list(data: Mapping[str, Any], name: str) -> tuple[str, ...]:
    value = data.get(name, ())
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{name} must be a sequence of strings")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class PackageRecord:
    """Observed installed Python distribution metadata.

    Package records are built from distribution metadata, not by importing the
    package runtime module.
    """

    name: str
    version: str | None
    normalized_name: str | None = None
    metadata_name: str | None = None
    location: str | None = None
    installer: str | None = None
    editable: bool | None = None
    schema_version: int = ENVIRONMENT_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        normalized = self.normalized_name or normalize_distribution_name(self.name)
        object.__setattr__(self, "normalized_name", normalized)

    def to_data(self) -> dict[str, Any]:
        """Return JSON-compatible package metadata."""

        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "normalized_name": self.normalized_name,
            "version": self.version,
            "metadata_name": self.metadata_name,
            "location": self.location,
            "installer": self.installer,
            "editable": self.editable,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "PackageRecord":
        """Build a package record from serialized data."""

        data = _mapping(data, "package record")
        editable = data.get("editable")
        if editable is not None and not isinstance(editable, bool):
            raise TypeError("editable must be a boolean or None")
        return cls(
            name=_required_string(data, "name"),
            normalized_name=_optional_string(data, "normalized_name"),
            version=_optional_string(data, "version"),
            metadata_name=_optional_string(data, "metadata_name"),
            location=_optional_string(data, "location"),
            installer=_optional_string(data, "installer"),
            editable=editable,
            schema_version=_schema_version(data),
        )


@dataclass(frozen=True, slots=True)
class PythonRecord:
    """Observed Python interpreter facts for an environment."""

    version: str
    implementation: str
    executable: str | None = None
    prefix: str | None = None
    base_prefix: str | None = None
    schema_version: int = ENVIRONMENT_RECORD_SCHEMA_VERSION

    def to_data(self) -> dict[str, Any]:
        """Return JSON-compatible Python interpreter metadata."""

        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "implementation": self.implementation,
            "executable": self.executable,
            "prefix": self.prefix,
            "base_prefix": self.base_prefix,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "PythonRecord":
        """Build a Python record from serialized data."""

        data = _mapping(data, "Python record")
        return cls(
            version=_required_string(data, "version"),
            implementation=_required_string(data, "implementation"),
            executable=_optional_string(data, "executable"),
            prefix=_optional_string(data, "prefix"),
            base_prefix=_optional_string(data, "base_prefix"),
            schema_version=_schema_version(data),
        )


@dataclass(frozen=True, slots=True)
class PlatformRecord:
    """Observed operating-system and platform metadata."""

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
    schema_version: int = ENVIRONMENT_RECORD_SCHEMA_VERSION

    def to_data(self) -> dict[str, Any]:
        """Return JSON-compatible platform metadata."""

        return {
            "schema_version": self.schema_version,
            "system": self.system,
            "release": self.release,
            "version": self.version,
            "machine": self.machine,
            "platform": self.platform,
            "os_name": self.os_name,
            "sys_platform": self.sys_platform,
            "implementation_name": self.implementation_name,
            "implementation_version": self.implementation_version,
            "platform_python_implementation": self.platform_python_implementation,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "PlatformRecord":
        """Build a platform record from serialized data."""

        data = _mapping(data, "platform record")
        return cls(
            system=_required_string(data, "system"),
            release=_required_string(data, "release"),
            version=_required_string(data, "version"),
            machine=_required_string(data, "machine"),
            platform=_required_string(data, "platform"),
            os_name=_optional_string(data, "os_name"),
            sys_platform=_optional_string(data, "sys_platform"),
            implementation_name=_optional_string(data, "implementation_name"),
            implementation_version=_optional_string(data, "implementation_version"),
            platform_python_implementation=_optional_string(data, "platform_python_implementation"),
            schema_version=_schema_version(data),
        )


@dataclass(frozen=True, slots=True)
class DrymlRuntimeRecord:
    """Observed DRYML runtime metadata and supported environment features."""

    version: str | None = None
    git_revision: str | None = None
    execution_protocol: str | None = None
    schema_versions: Mapping[str, int] = field(default_factory=dict)
    features: tuple[str, ...] = ("dryml.environments.v1",)
    schema_version: int = ENVIRONMENT_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_versions", deep_freeze_json({} if self.schema_versions is None else self.schema_versions))
        object.__setattr__(self, "features", tuple(sorted(str(item) for item in coerce_tuple(self.features))))

    def to_data(self) -> dict[str, Any]:
        """Return JSON-compatible DRYML runtime metadata."""

        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "git_revision": self.git_revision,
            "execution_protocol": self.execution_protocol,
            "schema_versions": json_ready(self.schema_versions),
            "features": list(self.features),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "DrymlRuntimeRecord":
        """Build a DRYML runtime record from serialized data."""

        data = _mapping(data, "DRYML runtime record")
        schema_versions = _mapping(data.get("schema_versions", {}), "schema_versions")
        if not all(isinstance(name, str) and not isinstance(version, bool) and isinstance(version, int) for name, version in schema_versions.items()):
            raise TypeError("schema_versions must map strings to integers")
        return cls(
            version=_optional_string(data, "version"),
            git_revision=_optional_string(data, "git_revision"),
            execution_protocol=_optional_string(data, "execution_protocol"),
            schema_versions=schema_versions,
            features=_string_list(data, "features"),
            schema_version=_schema_version(data),
        )


@dataclass(frozen=True, slots=True)
class EnvironmentRecord:
    """Observed facts about a concrete Python/software environment.

    The content ID intentionally includes interpreter path, prefixes, platform,
    installed distribution metadata, tags, and details. This makes the ID a
    precise observed-environment provenance key. Future Store side records can
    add coarser equivalence classes if needed.
    """

    python: PythonRecord
    platform: PlatformRecord
    distributions: Mapping[str, PackageRecord] = field(default_factory=dict)
    dryml: DrymlRuntimeRecord | None = None
    kind: str = "unknown"
    tags: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = ENVIRONMENT_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        packages = {}
        for key, record in self.distributions.items():
            pkg = record if isinstance(record, PackageRecord) else PackageRecord.from_data(record)
            packages[normalize_distribution_name(key)] = pkg
        object.__setattr__(self, "distributions", freeze_mapping(packages))
        object.__setattr__(self, "tags", tuple(sorted(str(item) for item in coerce_tuple(self.tags))))
        object.__setattr__(self, "details", deep_freeze_json({} if self.details is None else self.details))

    @property
    def id(self) -> str:
        """Stable content-addressed ID for this observed environment record."""

        return content_id("envrec", self.schema_version, self.to_data())

    def to_data(self) -> dict[str, Any]:
        """Return JSON-compatible environment metadata."""

        return {
            "schema_version": self.schema_version,
            "python": self.python.to_data(),
            "platform": self.platform.to_data(),
            "distributions": {
                key: self.distributions[key].to_data() for key in sorted(self.distributions)
            },
            "dryml": None if self.dryml is None else self.dryml.to_data(),
            "kind": self.kind,
            "tags": list(self.tags),
            "details": json_ready(self.details),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "EnvironmentRecord":
        """Build an environment record from serialized data."""

        data = _mapping(data, "environment record")
        distributions = _mapping(data.get("distributions", {}), "distributions")
        if not all(isinstance(name, str) and isinstance(value, Mapping) for name, value in distributions.items()):
            raise TypeError("distributions must map strings to package-record mappings")
        dryml = data.get("dryml")
        if dryml is not None and not isinstance(dryml, Mapping):
            raise TypeError("dryml must be a mapping or None")
        kind = data.get("kind", "unknown")
        if not isinstance(kind, str):
            raise TypeError("kind must be a string")
        details = _mapping(data.get("details", {}), "details")
        return cls(
            python=PythonRecord.from_data(data["python"]),
            platform=PlatformRecord.from_data(data["platform"]),
            distributions={
                key: PackageRecord.from_data(value)
                for key, value in distributions.items()
            },
            dryml=(
                None
                if dryml is None
                else DrymlRuntimeRecord.from_data(dryml)
            ),
            kind=kind,
            tags=_string_list(data, "tags"),
            details=details,
            schema_version=_schema_version(data),
        )


class EnvironmentInternTable:
    """In-memory interning table for environment records and requirements."""

    def __init__(self) -> None:
        self._records: dict[str, EnvironmentRecord] = {}
        self._requirements: dict[str, Any] = {}

    def intern_record(self, record: EnvironmentRecord) -> EnvironmentRecord:
        """Return the canonical in-memory instance for an environment record ID."""

        return self._records.setdefault(record.id, record)

    def intern_requirement(self, requirement: Any) -> Any:
        """Return the canonical in-memory instance for an environment requirement ID."""

        return self._requirements.setdefault(requirement.id, requirement)


__all__ = [
    "PackageRecord",
    "PythonRecord",
    "PlatformRecord",
    "DrymlRuntimeRecord",
    "EnvironmentRecord",
    "EnvironmentInternTable",
]
