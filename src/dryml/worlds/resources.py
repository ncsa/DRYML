"""Immutable resource constraints and concrete resource requests for worlds."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .errors import ResourceValidationError

_BYTES = {"B": 1, "MiB": 1024**2, "GiB": 1024**3}
_BYTE_RE = re.compile(r"^(0|[1-9][0-9]*)(B|MiB|GiB)$")
_MAX_MAP = 256
_MAX_BITS = 4096


def parse_byte_size(value: int | str | None) -> int | None:
    """Parse an explicit byte count without accepting ambiguous units.

    Args:
        value: A non-negative integer, or a canonical ``B``, ``MiB``, or
            ``GiB`` string.

    Returns:
        The byte count, or ``None`` when no value was supplied.
    """

    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        _integer("byte size", value)
        return value
    if not isinstance(value, str) or not (match := _BYTE_RE.match(value)):
        raise ResourceValidationError("byte size must be a non-negative integer or canonical B/MiB/GiB string")
    result = int(match.group(1)) * _BYTES[match.group(2)]
    _integer("byte size", result)
    return result


def canonical_byte_size(value: int | str | None) -> str | None:
    """Return the shortest canonical unit representation for a byte count."""

    parsed = parse_byte_size(value)
    if parsed is None:
        return None
    if parsed and parsed % _BYTES["GiB"] == 0:
        return f"{parsed // _BYTES['GiB']}GiB"
    if parsed and parsed % _BYTES["MiB"] == 0:
        return f"{parsed // _BYTES['MiB']}MiB"
    return f"{parsed}B"


@dataclass(frozen=True, slots=True)
class CountConstraint:
    """Inclusive immutable count range used by hard resource requirements."""

    min: int | None = None
    max: int | None = None

    def __post_init__(self) -> None:
        if self.min is not None:
            _integer("constraint min", self.min)
        if self.max is not None:
            _integer("constraint max", self.max)
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ResourceValidationError("count constraint min exceeds max")

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "CountConstraint":
        """Decode an exact closed v1.1 ``min``/``max`` declaration."""

        if not isinstance(data, Mapping) or set(data) != {"min", "max"}:
            raise ResourceValidationError("count constraint fields must be exactly min and max")
        return cls(data["min"], data["max"])

    def to_data(self) -> dict[str, int | None]:
        """Return a canonical JSON-compatible constraint declaration."""

        return {"min": self.min, "max": self.max}

    def satisfied_by(self, value: int) -> bool:
        """Return whether a non-negative concrete value satisfies this range."""

        _integer("concrete value", value)
        return (self.min is None or value >= self.min) and (self.max is None or value <= self.max)

    def merge(self, other: "CountConstraint") -> "CountConstraint":
        """Return the intersected range or raise for an empty intersection."""

        if not isinstance(other, CountConstraint):
            raise ResourceValidationError("count constraint merge requires a CountConstraint")
        return CountConstraint(maximum((self.min, other.min)), minimum((self.max, other.max)))


@dataclass(frozen=True, slots=True)
class ResourceRequirement:
    """Hard per-process resource constraints for one role replica."""

    cpus: CountConstraint = field(default_factory=CountConstraint)
    memory: CountConstraint = field(default_factory=CountConstraint)
    accelerators: Mapping[str, CountConstraint] = field(default_factory=dict)
    accelerator_memory: Mapping[str, CountConstraint] = field(default_factory=dict)
    devices: Mapping[str, CountConstraint] = field(default_factory=dict)
    named: Mapping[str, CountConstraint] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "accelerators", _constraint_map(self.accelerators, "accelerators"))
        object.__setattr__(self, "accelerator_memory", _constraint_map(self.accelerator_memory, "accelerator_memory", byte_values=True))
        object.__setattr__(self, "devices", _constraint_map(self.devices, "devices"))
        object.__setattr__(self, "named", _constraint_map(self.named, "named"))

    @classmethod
    def from_data(cls, data: Mapping[str, Any] | None) -> "ResourceRequirement":
        """Decode closed resource constraints from a JSON mapping."""

        data = {} if data is None else data
        fields = {"cpus", "memory", "accelerators", "accelerator_memory", "devices", "named"}
        if not isinstance(data, Mapping) or set(data) - fields:
            raise ResourceValidationError("resource requirement fields are closed")
        return cls(
            _coerce_constraint(data.get("cpus")),
            _byte_constraint(data.get("memory")),
            _constraint_map(data.get("accelerators", {}), "accelerators"),
            _constraint_map(data.get("accelerator_memory", {}), "accelerator_memory", byte_values=True),
            _constraint_map(data.get("devices", {}), "devices"),
            _constraint_map(data.get("named", {}), "named"),
        )

    def to_data(self) -> dict[str, Any]:
        """Return the compact canonical resource-constraint payload."""

        result: dict[str, Any] = {}
        for name, value, bytes_ in (("cpus", self.cpus, False), ("memory", self.memory, True)):
            if _is_constrained(value):
                result[name] = _constraint_data(value, bytes_)
        for name, value, bytes_ in (("accelerators", self.accelerators, False), ("accelerator_memory", self.accelerator_memory, True), ("devices", self.devices, False), ("named", self.named, False)):
            if value:
                result[name] = {key: _constraint_data(item, bytes_) for key, item in value.items()}
        return result

    def merge(self, other: "ResourceRequirement") -> "ResourceRequirement":
        """Intersect every hard resource constraint with ``other``."""

        return ResourceRequirement(self.cpus.merge(other.cpus), self.memory.merge(other.memory), _merge_maps(self.accelerators, other.accelerators), _merge_maps(self.accelerator_memory, other.accelerator_memory), _merge_maps(self.devices, other.devices), _merge_maps(self.named, other.named))


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    """Concrete requested per-process resources, separate from assignment."""

    cpus: int = 0
    memory: int | None = None
    accelerators: Mapping[str, int] = field(default_factory=dict)
    accelerator_memory: Mapping[str, Sequence[int | str]] = field(default_factory=dict)
    devices: Mapping[str, Any] = field(default_factory=dict)
    named: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _integer("cpus", self.cpus)
        object.__setattr__(self, "memory", parse_byte_size(self.memory))
        accelerators = _concrete_map(self.accelerators, "accelerators")
        memory = {}
        for kind, values in self.accelerator_memory.items():
            if kind not in accelerators or isinstance(values, str | bytes) or len(values) != accelerators[kind]:
                raise ResourceValidationError("accelerator_memory must align with requested accelerators")
            parsed = tuple(parse_byte_size(value) for value in values)
            if any(value is None or value <= 0 for value in parsed):
                raise ResourceValidationError("accelerator_memory limits must be positive")
            memory[kind] = parsed
        object.__setattr__(self, "accelerators", MappingProxyType(accelerators))
        object.__setattr__(self, "accelerator_memory", MappingProxyType({key: memory[key] for key in sorted(memory)}))
        object.__setattr__(self, "devices", _frozen_map(self.devices, "devices"))
        object.__setattr__(self, "named", _frozen_map(self.named, "named"))

    @classmethod
    def from_data(cls, data: Mapping[str, Any] | None) -> "ResourceSpec":
        """Decode a closed concrete resource request mapping."""

        data = {} if data is None else data
        fields = {"cpus", "memory", "accelerators", "accelerator_memory", "devices", "named"}
        if not isinstance(data, Mapping) or set(data) - fields:
            raise ResourceValidationError("resource spec fields are closed")
        return cls(data.get("cpus", 0), data.get("memory"), data.get("accelerators", {}), data.get("accelerator_memory", {}), data.get("devices", {}), data.get("named", {}))

    def to_data(self) -> dict[str, Any]:
        """Return the canonical concrete resource request payload."""

        result: dict[str, Any] = {"cpus": self.cpus}
        if self.memory is not None:
            result["memory"] = canonical_byte_size(self.memory)
        for name, values in (("accelerators", self.accelerators), ("devices", self.devices), ("named", self.named)):
            if values:
                result[name] = dict(values)
        if self.accelerator_memory:
            result["accelerator_memory"] = {kind: [canonical_byte_size(value) for value in values] for kind, values in self.accelerator_memory.items()}
        return result


def _integer(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value.bit_length() > _MAX_BITS:
        raise ResourceValidationError(f"{name} must be a bounded non-negative integer")


def _constraint_map(data: Mapping[str, Any], name: str, *, byte_values: bool = False) -> Mapping[str, CountConstraint]:
    if not isinstance(data, Mapping) or len(data) > _MAX_MAP:
        raise ResourceValidationError(f"{name} resource mapping exceeds the bounded entry limit")
    result = {}
    for key, value in data.items():
        if not isinstance(key, str) or not key:
            raise ResourceValidationError(f"{name} resource name must be a non-empty string")
        result[key] = _byte_constraint(value) if byte_values else _coerce_constraint(value)
    return MappingProxyType({key: result[key] for key in sorted(result)})


def _byte_constraint(data: Any) -> CountConstraint:
    if data is None:
        return CountConstraint()
    if isinstance(data, CountConstraint):
        return data
    if isinstance(data, Mapping):
        converted = {key: parse_byte_size(value) for key, value in data.items()}
        return CountConstraint.from_data(converted)
    value = parse_byte_size(data)
    return CountConstraint(value, value)


def _coerce_constraint(data: Any) -> CountConstraint:
    if data is None:
        return CountConstraint()
    if isinstance(data, CountConstraint):
        return data
    if isinstance(data, int) and not isinstance(data, bool):
        return CountConstraint(data, data)
    return CountConstraint.from_data(data)


def _is_constrained(value: CountConstraint) -> bool:
    return value.min is not None or value.max is not None


def _constraint_data(value: CountConstraint, bytes_: bool) -> dict[str, Any]:
    return {key: canonical_byte_size(item) if bytes_ else item for key, item in value.to_data().items()}


def _concrete_map(data: Mapping[str, Any], name: str) -> dict[str, int]:
    if not isinstance(data, Mapping) or len(data) > _MAX_MAP:
        raise ResourceValidationError(f"{name} resource mapping exceeds the bounded entry limit")
    result = {}
    for key, value in data.items():
        if not isinstance(key, str) or not key:
            raise ResourceValidationError(f"{name} resource name must be a non-empty string")
        _integer(f"{name}.{key}", value)
        result[key] = value
    return {key: result[key] for key in sorted(result)}


def _frozen_map(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping) or len(data) > _MAX_MAP or any(not isinstance(key, str) or not key for key in data):
        raise ResourceValidationError(f"{name} resources must be a bounded string-keyed mapping")
    return MappingProxyType({key: data[key] for key in sorted(data)})


def _merge_maps(left: Mapping[str, CountConstraint], right: Mapping[str, CountConstraint]) -> Mapping[str, CountConstraint]:
    return MappingProxyType({key: (left[key].merge(right[key]) if key in left and key in right else left.get(key, right[key])) for key in sorted(set(left) | set(right))})


def maximum(values: tuple[int | None, int | None]) -> int | None:
    """Return the greatest defined range lower bound."""
    return max((value for value in values if value is not None), default=None)


def minimum(values: tuple[int | None, int | None]) -> int | None:
    """Return the least defined range upper bound."""
    return min((value for value in values if value is not None), default=None)
