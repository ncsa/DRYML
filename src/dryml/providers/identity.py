"""Provider identity metadata and import-path references."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from dryml.formats import CanonicalJSONError, deep_freeze_json, json_ready
from dryml.formats.ids import content_id

from .errors import ProviderValidationError


PROVIDER_ID_SCHEMA_VERSION = 1
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_DOTTED_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")


@dataclass(frozen=True, slots=True)
class ProviderIdentity:
    """Stable JSON-ready identity for one provider implementation."""

    name: str
    version: str | None = None
    module: str | None = None
    qualname: str | None = None
    capabilities: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = PROVIDER_ID_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _validate_name(self.name))
        if self.version is not None:
            object.__setattr__(self, "version", str(self.version))
        if self.module is not None:
            object.__setattr__(self, "module", _validate_dotted(self.module, "module"))
        if self.qualname is not None:
            object.__setattr__(self, "qualname", _validate_dotted(self.qualname, "qualname"))
        object.__setattr__(self, "capabilities", _coerce_string_tuple(self.capabilities, "capabilities", sort_unique=True))
        object.__setattr__(self, "metadata", _freeze_json_mapping(self.metadata, "metadata"))
        if self.schema_version != PROVIDER_ID_SCHEMA_VERSION:
            raise ProviderValidationError("unsupported provider identity schema version", context={"schema_version": self.schema_version})

    @property
    def id(self) -> str:
        """Return the stable ``provider-v1-*`` content ID."""

        return content_id("provider", self.schema_version, self.to_data(include_id=False))

    def to_data(self, *, include_id: bool = True) -> dict[str, Any]:
        """Return canonical JSON-ready identity data."""

        data = {
            "schema_version": self.schema_version,
            "name": self.name,
            "version": self.version,
            "module": self.module,
            "qualname": self.qualname,
            "capabilities": list(self.capabilities),
            "metadata": json_ready(self.metadata),
        }
        if include_id:
            data["id"] = self.id
        return data

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ProviderIdentity":
        """Build an identity from JSON-ready data."""

        if not isinstance(data, Mapping):
            raise ProviderValidationError("provider identity must be a mapping", context={"type": type(data).__name__})
        unknown = set(data) - {"schema_version", "id", "name", "version", "module", "qualname", "capabilities", "metadata"}
        if unknown:
            raise ProviderValidationError("provider identity has unknown fields", context={"fields": sorted(unknown)})
        identity = cls(
            name=data.get("name"),
            version=data.get("version"),
            module=data.get("module"),
            qualname=data.get("qualname"),
            capabilities=_json_string_sequence(data.get("capabilities"), "capabilities"),
            metadata=data.get("metadata") or {},
            schema_version=data.get("schema_version", PROVIDER_ID_SCHEMA_VERSION),
        )
        if data.get("id") not in (None, identity.id):
            raise ProviderValidationError("provider identity ID does not match payload", context={"expected": identity.id, "observed": data.get("id")})
        return identity


@dataclass(frozen=True, slots=True)
class ProviderRef:
    """Import-path reference to a provider class or factory.

    Constructing this object validates strings only; it never imports the target
    module. Loading is reserved for the probe worker.
    """

    name: str
    module: str
    qualname: str = "Provider"
    version_hint: str | None = None
    capabilities: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = PROVIDER_ID_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _validate_name(self.name))
        object.__setattr__(self, "module", _validate_dotted(self.module, "module"))
        object.__setattr__(self, "qualname", _validate_dotted(self.qualname, "qualname"))
        if self.version_hint is not None:
            object.__setattr__(self, "version_hint", str(self.version_hint))
        object.__setattr__(self, "capabilities", _coerce_string_tuple(self.capabilities, "capabilities", sort_unique=True))
        object.__setattr__(self, "metadata", _freeze_json_mapping(self.metadata, "metadata"))
        if self.schema_version != PROVIDER_ID_SCHEMA_VERSION:
            raise ProviderValidationError("unsupported provider ref schema version", context={"schema_version": self.schema_version})

    def fallback_identity(self) -> ProviderIdentity:
        """Return identity metadata available before importing the provider."""

        return ProviderIdentity(self.name, self.version_hint, self.module, self.qualname, self.capabilities, self.metadata)

    def to_data(self) -> dict[str, Any]:
        """Return canonical JSON-ready provider reference data."""

        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "module": self.module,
            "qualname": self.qualname,
            "version_hint": self.version_hint,
            "capabilities": list(self.capabilities),
            "metadata": json_ready(self.metadata),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ProviderRef":
        """Build an import-path reference from JSON-ready data."""

        if not isinstance(data, Mapping):
            raise ProviderValidationError("provider ref must be a mapping", context={"type": type(data).__name__})
        unknown = set(data) - {"schema_version", "name", "module", "qualname", "version_hint", "capabilities", "metadata"}
        if unknown:
            raise ProviderValidationError("provider ref has unknown fields", context={"fields": sorted(unknown)})
        return cls(
            name=data.get("name"),
            module=data.get("module"),
            qualname=data.get("qualname", "Provider"),
            version_hint=data.get("version_hint"),
            capabilities=_json_string_sequence(data.get("capabilities"), "capabilities"),
            metadata=data.get("metadata") or {},
            schema_version=data.get("schema_version", PROVIDER_ID_SCHEMA_VERSION),
        )


def _validate_name(value: Any) -> str:
    if not isinstance(value, str) or _NAME_RE.fullmatch(value) is None:
        raise ProviderValidationError("provider name must be non-empty and deterministic", context={"name": value})
    return value


def _validate_dotted(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _DOTTED_RE.fullmatch(value) is None:
        raise ProviderValidationError(f"provider {field_name} must be a dotted Python path", context={field_name: value})
    return value


def _coerce_string_tuple(value: Any, field_name: str, *, sort_unique: bool = False) -> tuple[str, ...]:
    if value is None:
        items: tuple[Any, ...] = ()
    elif isinstance(value, str | bytes | bytearray):
        raise ProviderValidationError(f"provider {field_name} must be a sequence of strings, not a string", context={field_name: value})
    elif isinstance(value, Iterable):
        items = tuple(value)
    else:
        raise ProviderValidationError(f"provider {field_name} must be a sequence of strings", context={"type": type(value).__name__})
    result = tuple(str(item) for item in items)
    return tuple(sorted(set(result))) if sort_unique else result


def _json_string_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        raise ProviderValidationError(f"provider {field_name} must be a JSON array", context={field_name: value, "type": type(value).__name__})
    if any(not isinstance(item, str) for item in value):
        raise ProviderValidationError(f"provider {field_name} values must be strings", context={field_name: value})
    return tuple(value)


def _freeze_json_mapping(value: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProviderValidationError("provider metadata must be a mapping", context={"path": path, "type": type(value).__name__})
    try:
        frozen = deep_freeze_json(value)
    except CanonicalJSONError as exc:
        raise ProviderValidationError("provider metadata is not JSON-ready", context={"path": path, **exc.context}) from exc
    assert isinstance(frozen, Mapping)
    return frozen


__all__ = ["PROVIDER_ID_SCHEMA_VERSION", "ProviderIdentity", "ProviderRef"]
