"""Serializable records describing observed Python/software environments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .ids import content_id
from .schema import ENVIRONMENT_RECORD_SCHEMA_VERSION
from .serialization import freeze_mapping
from .utils import coerce_tuple, normalize_distribution_name


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

        return cls(
            name=data["name"],
            normalized_name=data.get("normalized_name"),
            version=data.get("version"),
            metadata_name=data.get("metadata_name"),
            location=data.get("location"),
            installer=data.get("installer"),
            editable=data.get("editable"),
            schema_version=data.get("schema_version", ENVIRONMENT_RECORD_SCHEMA_VERSION),
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

        return cls(
            version=data["version"],
            implementation=data["implementation"],
            executable=data.get("executable"),
            prefix=data.get("prefix"),
            base_prefix=data.get("base_prefix"),
            schema_version=data.get("schema_version", ENVIRONMENT_RECORD_SCHEMA_VERSION),
        )


@dataclass(frozen=True, slots=True)
class PlatformRecord:
    """Observed operating-system and platform metadata."""

    system: str
    release: str
    version: str
    machine: str
    platform: str
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
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "PlatformRecord":
        """Build a platform record from serialized data."""

        return cls(
            system=data["system"],
            release=data["release"],
            version=data["version"],
            machine=data["machine"],
            platform=data["platform"],
            schema_version=data.get("schema_version", ENVIRONMENT_RECORD_SCHEMA_VERSION),
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
        object.__setattr__(self, "schema_versions", freeze_mapping(self.schema_versions))
        object.__setattr__(self, "features", tuple(sorted(str(item) for item in coerce_tuple(self.features))))

    def to_data(self) -> dict[str, Any]:
        """Return JSON-compatible DRYML runtime metadata."""

        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "git_revision": self.git_revision,
            "execution_protocol": self.execution_protocol,
            "schema_versions": dict(self.schema_versions),
            "features": list(self.features),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "DrymlRuntimeRecord":
        """Build a DRYML runtime record from serialized data."""

        return cls(
            version=data.get("version"),
            git_revision=data.get("git_revision"),
            execution_protocol=data.get("execution_protocol"),
            schema_versions=data.get("schema_versions", {}),
            features=tuple(data.get("features", ())),
            schema_version=data.get("schema_version", ENVIRONMENT_RECORD_SCHEMA_VERSION),
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
        object.__setattr__(self, "details", freeze_mapping(self.details))

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
            "details": dict(self.details),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "EnvironmentRecord":
        """Build an environment record from serialized data."""

        return cls(
            python=PythonRecord.from_data(data["python"]),
            platform=PlatformRecord.from_data(data["platform"]),
            distributions={
                key: PackageRecord.from_data(value)
                for key, value in data.get("distributions", {}).items()
            },
            dryml=(
                None
                if data.get("dryml") is None
                else DrymlRuntimeRecord.from_data(data["dryml"])
            ),
            kind=data.get("kind", "unknown"),
            tags=tuple(data.get("tags", ())),
            details=data.get("details", {}),
            schema_version=data.get("schema_version", ENVIRONMENT_RECORD_SCHEMA_VERSION),
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
