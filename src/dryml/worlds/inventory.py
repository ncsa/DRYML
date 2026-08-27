"""Framework-free inherited local resource inventory discovery."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from dryml.formats import deep_freeze_json, json_ready

from .errors import ResourceValidationError
from .resources import parse_byte_size


@dataclass(frozen=True, slots=True)
class LocalResourceInventory:
    """Immutable local capacity evidence; injected instances are authoritative."""

    cpus: tuple[int, ...]
    accelerators: Mapping[str, tuple[str | int, ...]] = field(default_factory=dict)
    memory: int | None = None
    accelerator_memory: Mapping[str, Mapping[str | int, int]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        cpus = tuple(sorted(self.cpus))
        if not cpus or len(cpus) > 4096 or any(isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 for cpu in cpus) or len(cpus) != len(set(cpus)):
            raise ResourceValidationError("inventory CPUs must be unique bounded non-negative identifiers")
        accelerators = {}
        if not isinstance(self.accelerators, Mapping) or len(self.accelerators) > 256:
            raise ResourceValidationError("inventory accelerators must be a bounded mapping")
        for kind, values in self.accelerators.items():
            if not isinstance(kind, str) or not kind or isinstance(values, str | bytes):
                raise ResourceValidationError("inventory accelerator IDs must be string-keyed sequences")
            values = tuple(sorted(values, key=str))
            if len(values) > 4096 or len(values) != len(set(values)) or any(not isinstance(value, str | int) for value in values):
                raise ResourceValidationError("inventory accelerator IDs must be unique strings or integers")
            accelerators[kind] = values
        parsed_memory = parse_byte_size(self.memory)
        per_device = {}
        for kind, values in self.accelerator_memory.items():
            if kind not in accelerators or not isinstance(values, Mapping) or set(values) - set(accelerators[kind]):
                raise ResourceValidationError("inventory accelerator memory must map known device IDs")
            parsed = {device: parse_byte_size(value) for device, value in values.items()}
            if any(value is None or value <= 0 for value in parsed.values()):
                raise ResourceValidationError("inventory accelerator memory must be positive")
            per_device[kind] = MappingProxyType(parsed)
        object.__setattr__(self, "cpus", cpus)
        object.__setattr__(self, "accelerators", MappingProxyType({key: accelerators[key] for key in sorted(accelerators)}))
        object.__setattr__(self, "memory", parsed_memory)
        object.__setattr__(self, "accelerator_memory", MappingProxyType({key: per_device[key] for key in sorted(per_device)}))
        object.__setattr__(self, "metadata", deep_freeze_json(self.metadata))

    @property
    def visibility_identity(self) -> tuple[Any, ...]:
        """Return stable visibility facts, excluding volatile available memory."""
        return (self.cpus, tuple((kind, self.accelerators[kind], tuple(sorted(self.accelerator_memory.get(kind, {}).items(), key=lambda item: str(item[0])))) for kind in self.accelerators))

    def summary(self) -> dict[str, Any]:
        """Return bounded inspection data without creating a reservation."""
        return {"cpus": list(self.cpus), "accelerators": {kind: list(values) for kind, values in self.accelerators.items()}, "memory": self.memory, "accelerator_memory": {kind: dict(values) for kind, values in self.accelerator_memory.items()}, "metadata": json_ready(self.metadata)}

    def to_data(self) -> dict[str, Any]:
        """Return a JSON-compatible inventory representation."""
        return self.summary()

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "LocalResourceInventory":
        """Decode an injected authoritative inventory mapping."""
        fields = {"cpus", "accelerators", "memory", "accelerator_memory", "metadata"}
        if not isinstance(data, Mapping) or set(data) - fields:
            raise ResourceValidationError("inventory fields are closed")
        return cls(tuple(data.get("cpus", ())), data.get("accelerators", {}), data.get("memory"), data.get("accelerator_memory", {}), data.get("metadata", {}))


def local_inventory(*, policy: Literal["lightweight"] = "lightweight", environ: Mapping[str, str] | None = None, device_root: str | os.PathLike[str] | None = "/dev") -> LocalResourceInventory:
    """Observe inherited local facts without commands or framework imports.

    Args:
        policy: Only ``"lightweight"`` is supported.
        environ: Optional explicit visibility source for tests and callers.
        device_root: Optional conservative character-device evidence root.

    Returns:
        A best-effort immutable inventory. Unknown facts never create capacity.
    """
    if policy != "lightweight":
        raise ResourceValidationError("only lightweight local inventory is supported")
    diagnostics: list[str] = []
    try:
        cpus = tuple(sorted(os.sched_getaffinity(0)))
        source = "affinity"
    except (AttributeError, OSError):
        count = os.cpu_count() or 1
        cpus, source = tuple(range(min(count, 4096))), "cpu_count"
    if not cpus:
        raise ResourceValidationError("inherited CPU affinity contains no CPUs")
    environment = os.environ if environ is None else environ
    accelerators = _explicit_accelerators(environment, diagnostics)
    if not accelerators and device_root is not None:
        accelerators = _device_root_accelerators(environment, device_root, diagnostics)
    return LocalResourceInventory(cpus, accelerators, _memory(), {}, {"policy": policy, "cpu_source": source, "diagnostics": diagnostics})


def _memory() -> int | None:
    if sys.platform.startswith("linux"):
        try:
            lines = Path("/proc/meminfo").read_text().splitlines()
            available = next((line for line in lines if line.startswith("MemAvailable:")), None)
            host = int(available.split()[1]) * 1024 if available else None
        except (OSError, StopIteration, ValueError, IndexError):
            host = None
        cgroup = _cgroup_memory()
        return min(value for value in (host, cgroup) if value is not None) if host is not None or cgroup is not None else None
    return _platform_memory()[0]


def _platform_memory() -> tuple[int | None, str]:
    """Return an injectable non-Linux memory seam without claiming capacity."""
    return None, "unknown"


def _cgroup_memory() -> int | None:
    """Return the readable cgroup v2 allowance without treating errors as capacity."""
    try:
        limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        usage = Path("/sys/fs/cgroup/memory.current").read_text().strip()
        if limit == "max":
            return None
        return max(int(limit) - int(usage), 0)
    except (OSError, ValueError):
        return None


def _explicit_accelerators(environment: Mapping[str, str], diagnostics: list[str]) -> dict[str, tuple[str | int, ...]]:
    raw = environment.get("DRYML_LOCAL_ACCELERATORS", "").strip()
    if not raw:
        return {}
    result = {}
    try:
        for group in raw.split(";"):
            kind, ids = group.split("=", 1)
            values = tuple(int(value) if value.isdigit() else value for value in ids.split(","))
            if not kind or not values or any(not value or value.lower() in {"all", "none", "void"} or any(character.isspace() for character in value) for value in values if isinstance(value, str)) or len(values) != len(set(values)) or kind in result:
                raise ValueError
            result[kind] = values
    except ValueError:
        diagnostics.append("explicit accelerator visibility was malformed")
        return {}
    return result


def _device_root_accelerators(environment: Mapping[str, str], root: str | os.PathLike[str], diagnostics: list[str]) -> dict[str, tuple[int, ...]]:
    allowed: set[int] | None = None
    for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        visible = environment.get(name, "all").strip().lower()
        if visible in {"all", ""}:
            continue
        if visible in {"none", "void"}:
            return {}
        if any(not value.isdigit() for value in visible.split(",")):
            diagnostics.append(f"{name} visibility was ambiguous")
            return {}
        values = {int(value) for value in visible.split(",")}
        allowed = values if allowed is None else allowed & values
    try:
        values = tuple(sorted(int(path.name[6:]) for path in Path(root).iterdir() if path.name.startswith("nvidia") and path.name[6:].isdigit() and path.is_char_device()))
    except OSError:
        return {}
    values = tuple(value for value in values if allowed is None or value in allowed)
    return {"gpu": values} if values else {}
