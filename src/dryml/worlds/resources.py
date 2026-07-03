"""Resource constraints and concrete resource requests for DRYML worlds."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .errors import ResourceValidationError

ByteSize = int

_BYTE_RE = re.compile(r"^(0|[1-9][0-9]*)(B|MiB|GiB)$")
_UNIT_FACTORS = {"B": 1, "MiB": 1024**2, "GiB": 1024**3}


def parse_byte_size(value: str | int | None) -> int | None:
    """Parse a byte size from an integer or canonical string.

    Accepted string units are raw bytes (``B``), ``MiB``, and ``GiB``. Bare
    strings and decimal/ambiguous units are rejected so serialized specs remain
    stable and explicit.
    """

    if value is None:
        return None
    if isinstance(value, bool):
        raise ResourceValidationError("byte size must not be bool", context={"value": value})
    if isinstance(value, int):
        if value < 0:
            raise ResourceValidationError("byte size must be >= 0", context={"value": value})
        return value
    if not isinstance(value, str):
        raise ResourceValidationError("byte size must be int or string", context={"type": type(value).__name__})
    match = _BYTE_RE.match(value)
    if not match:
        raise ResourceValidationError("invalid or ambiguous byte-size unit", context={"value": value, "accepted_units": sorted(_UNIT_FACTORS)})
    number, unit = match.groups()
    return int(number) * _UNIT_FACTORS[unit]


def canonical_byte_size(value: str | int | None) -> str | None:
    """Return the canonical JSON string for *value*, or ``None``."""

    size = parse_byte_size(value)
    if size is None:
        return None
    if size and size % _UNIT_FACTORS["GiB"] == 0:
        return f"{size // _UNIT_FACTORS['GiB']}GiB"
    if size and size % _UNIT_FACTORS["MiB"] == 0:
        return f"{size // _UNIT_FACTORS['MiB']}MiB"
    return f"{size}B"


def _as_nonneg_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ResourceValidationError(f"{name} must be an integer >= 0", context={"path": name, "value": value})
    if value < 0:
        raise ResourceValidationError(f"{name} must be >= 0", context={"path": name, "value": value})
    return value


@dataclass(frozen=True, slots=True)
class CountConstraint:
    """Inclusive ``min``/``max`` count constraint with ``exact`` shorthand."""

    min: int | None = None
    max: int | None = None

    def __post_init__(self) -> None:
        if self.min is not None:
            _as_nonneg_int("min", self.min)
        if self.max is not None:
            _as_nonneg_int("max", self.max)
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ResourceValidationError("count constraint min exceeds max", context={"min": self.min, "max": self.max})

    @classmethod
    def from_data(cls, data: Mapping[str, Any] | int | None, *, path: str = "constraint") -> "CountConstraint":
        """Build a constraint from JSON data."""

        if data is None:
            return cls()
        if isinstance(data, int) and not isinstance(data, bool):
            return cls(min=data, max=data)
        if not isinstance(data, Mapping):
            raise ResourceValidationError("count constraint must be a mapping", context={"path": path, "type": type(data).__name__})
        unknown = set(data) - {"min", "max", "exact"}
        if unknown:
            raise ResourceValidationError("count constraint has unknown fields", context={"path": path, "fields": sorted(unknown)})
        if "exact" in data:
            if "min" in data or "max" in data:
                raise ResourceValidationError("exact cannot be combined with min/max", context={"path": path})
            exact = _as_nonneg_int(f"{path}.exact", data["exact"])
            return cls(min=exact, max=exact)
        return cls(
            min=_as_nonneg_int(f"{path}.min", data["min"]) if "min" in data else None,
            max=_as_nonneg_int(f"{path}.max", data["max"]) if "max" in data else None,
        )

    def to_data(self) -> dict[str, int]:
        """Return canonical JSON data for this constraint."""

        if self.min is not None and self.max is not None and self.min == self.max:
            return {"exact": self.min}
        data: dict[str, int] = {}
        if self.min is not None:
            data["min"] = self.min
        if self.max is not None:
            data["max"] = self.max
        return data

    def satisfied_by(self, value: int) -> bool:
        """Return whether *value* satisfies this constraint."""

        _as_nonneg_int("value", value)
        if self.min is not None and value < self.min:
            return False
        if self.max is not None and value > self.max:
            return False
        return True

    def merge(self, other: "CountConstraint", *, path: str = "constraint") -> "CountConstraint":
        """Merge two constraints, raising on conflicts."""

        min_value = max(v for v in (self.min, other.min) if v is not None) if self.min is not None or other.min is not None else None
        max_value = min(v for v in (self.max, other.max) if v is not None) if self.max is not None or other.max is not None else None
        try:
            return CountConstraint(min=min_value, max=max_value)
        except ResourceValidationError as exc:
            raise ResourceValidationError("count constraint merge conflict", context={"path": path, "left": self.to_data(), "right": other.to_data()}) from exc


@dataclass(frozen=True, slots=True)
class ResourceRequirement:
    """Hard resource constraints for one role or process."""

    cpus: CountConstraint = field(default_factory=CountConstraint)
    memory: CountConstraint = field(default_factory=CountConstraint)
    accelerators: Mapping[str, CountConstraint] = field(default_factory=dict)
    devices: Mapping[str, CountConstraint] = field(default_factory=dict)
    named: Mapping[str, CountConstraint] = field(default_factory=dict)

    @classmethod
    def from_data(cls, data: Mapping[str, Any] | None) -> "ResourceRequirement":
        """Build a resource requirement from JSON-ready data."""

        data = data or {}
        if not isinstance(data, Mapping):
            raise ResourceValidationError("resource requirement must be a mapping", context={"type": type(data).__name__})
        unknown = set(data) - {"cpus", "memory", "accelerators", "devices", "named"}
        if unknown:
            raise ResourceValidationError("resource requirement has unknown fields", context={"fields": sorted(unknown)})
        memory = data.get("memory")
        if isinstance(memory, Mapping):
            memory_constraint = _byte_constraint_from_data(memory, path="memory")
        else:
            memory_constraint = CountConstraint.from_data(memory, path="memory") if memory is not None else CountConstraint()
        return cls(
            cpus=CountConstraint.from_data(data.get("cpus"), path="cpus"),
            memory=memory_constraint,
            accelerators=_constraint_map(data.get("accelerators"), path="accelerators"),
            devices=_constraint_map(data.get("devices"), path="devices"),
            named=_constraint_map(data.get("named"), path="named"),
        )

    def to_data(self) -> dict[str, Any]:
        """Return canonical JSON-ready resource requirement data."""

        data: dict[str, Any] = {}
        if self.cpus.to_data():
            data["cpus"] = self.cpus.to_data()
        if self.memory.to_data():
            data["memory"] = _byte_constraint_to_data(self.memory)
        if self.accelerators:
            data["accelerators"] = {key: value.to_data() for key, value in sorted(self.accelerators.items())}
        if self.devices:
            data["devices"] = {key: value.to_data() for key, value in sorted(self.devices.items())}
        if self.named:
            data["named"] = {key: value.to_data() for key, value in sorted(self.named.items())}
        return data

    def merge(self, other: "ResourceRequirement", *, path: str = "resources") -> "ResourceRequirement":
        """Merge this requirement with another hard requirement."""

        return ResourceRequirement(
            cpus=self.cpus.merge(other.cpus, path=f"{path}.cpus"),
            memory=self.memory.merge(other.memory, path=f"{path}.memory"),
            accelerators=_merge_constraint_maps(self.accelerators, other.accelerators, path=f"{path}.accelerators"),
            devices=_merge_constraint_maps(self.devices, other.devices, path=f"{path}.devices"),
            named=_merge_constraint_maps(self.named, other.named, path=f"{path}.named"),
        )


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    """Concrete requested/default resources for a process."""

    cpus: int = 0
    memory: int | None = None
    accelerators: Mapping[str, int] = field(default_factory=dict)
    devices: Mapping[str, Any] = field(default_factory=dict)
    named: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _as_nonneg_int("cpus", self.cpus)
        if self.memory is not None:
            parse_byte_size(self.memory)
        for key, value in self.accelerators.items():
            _validate_name(key, "accelerator")
            _as_nonneg_int(f"accelerators.{key}", value)

    @classmethod
    def from_data(cls, data: Mapping[str, Any] | None) -> "ResourceSpec":
        """Build a concrete resource request from JSON-ready data."""

        data = data or {}
        if not isinstance(data, Mapping):
            raise ResourceValidationError("resource spec must be a mapping", context={"type": type(data).__name__})
        unknown = set(data) - {"cpus", "memory", "accelerators", "devices", "named"}
        if unknown:
            raise ResourceValidationError("resource spec has unknown fields", context={"fields": sorted(unknown)})
        accelerators = data.get("accelerators") or {}
        if not isinstance(accelerators, Mapping):
            raise ResourceValidationError("accelerators must be a mapping")
        return cls(
            cpus=_as_nonneg_int("cpus", data.get("cpus", 0)),
            memory=parse_byte_size(data.get("memory")),
            accelerators={str(key): _as_nonneg_int(f"accelerators.{key}", value) for key, value in accelerators.items()},
            devices=dict(data.get("devices") or {}),
            named=dict(data.get("named") or {}),
        )

    def to_data(self) -> dict[str, Any]:
        """Return canonical JSON-ready requested resource data."""

        data: dict[str, Any] = {"cpus": self.cpus}
        if self.memory is not None:
            data["memory"] = canonical_byte_size(self.memory)
        if self.accelerators:
            data["accelerators"] = {key: self.accelerators[key] for key in sorted(self.accelerators)}
        if self.devices:
            data["devices"] = {key: self.devices[key] for key in sorted(self.devices)}
        if self.named:
            data["named"] = {key: self.named[key] for key in sorted(self.named)}
        return data


def _byte_constraint_from_data(data: Mapping[str, Any], *, path: str) -> CountConstraint:
    converted: dict[str, int] = {}
    for key, value in data.items():
        if key not in {"min", "max", "exact"}:
            raise ResourceValidationError("byte constraint has unknown fields", context={"path": path, "field": key})
        parsed = parse_byte_size(value)
        assert parsed is not None
        converted[key] = parsed
    return CountConstraint.from_data(converted, path=path)


def _byte_constraint_to_data(constraint: CountConstraint) -> dict[str, str]:
    return {key: canonical_byte_size(value) for key, value in constraint.to_data().items()}  # type: ignore[misc]


def _constraint_map(data: Any, *, path: str) -> dict[str, CountConstraint]:
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ResourceValidationError("constraint map must be a mapping", context={"path": path})
    return {str(key): CountConstraint.from_data(value, path=f"{path}.{key}") for key, value in data.items()}


def _merge_constraint_maps(left: Mapping[str, CountConstraint], right: Mapping[str, CountConstraint], *, path: str) -> dict[str, CountConstraint]:
    result = dict(left)
    for key, value in right.items():
        result[key] = result[key].merge(value, path=f"{path}.{key}") if key in result else value
    return result


def _validate_name(name: str, kind: str) -> None:
    if not isinstance(name, str) or not name:
        raise ResourceValidationError(f"{kind} name must be a non-empty string", context={"name": name})


__all__ = [
    "ByteSize",
    "CountConstraint",
    "ResourceRequirement",
    "ResourceSpec",
    "parse_byte_size",
]
