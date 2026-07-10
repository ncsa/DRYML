"""Import-safe local resource inventory discovery for requested worlds."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .errors import ResourceValidationError


_MAX_METADATA_DEPTH = 8
_MAX_METADATA_ITEMS = 64
_MAX_METADATA_STRING = 4096
_MAX_EXTERNAL_OUTPUT_CHARS = 64 * 1024
_MAX_EXTERNAL_DEVICE_IDS = 128
_MAX_EXPLICIT_ACCELERATOR_CHARS = 64 * 1024
_MAX_EXPLICIT_ACCELERATOR_IDS = 128


@dataclass(frozen=True, slots=True)
class LocalResourceInventory:
    """Deterministic local CPU, memory, and accelerator capacity facts.

    The model describes discoverable host capacity only.  It does not allocate
    resources, initialize frameworks, or alter process visibility.
    """

    cpus: tuple[int, ...]
    accelerators: Mapping[str, tuple[str | int, ...]] = field(default_factory=dict)
    memory: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cpus = tuple(sorted(_nonneg_int(cpu, "cpu") for cpu in self.cpus))
        if not cpus:
            raise ResourceValidationError("local resource inventory requires at least one CPU")
        if len(set(cpus)) != len(cpus):
            raise ResourceValidationError("local resource inventory CPUs must be unique", context={"cpus": cpus})
        if not isinstance(self.accelerators, Mapping):
            raise ResourceValidationError("local resource inventory accelerators must be a mapping")
        accelerators: dict[str, tuple[str | int, ...]] = {}
        for name, values in self.accelerators.items():
            if not isinstance(name, str) or not name:
                raise ResourceValidationError("accelerator inventory name must be a non-empty string")
            if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
                raise ResourceValidationError("accelerator inventory values must be sequences", context={"accelerator": name})
            normalized = tuple(values)
            if any(isinstance(value, bool) or not isinstance(value, (str, int)) for value in normalized):
                raise ResourceValidationError("accelerator identifiers must be strings or integers", context={"accelerator": name})
            if len(set(normalized)) != len(normalized):
                raise ResourceValidationError("accelerator inventory identifiers must be unique", context={"accelerator": name})
            accelerators[name] = tuple(sorted(normalized, key=lambda value: (str(type(value)), str(value))))
        if self.memory is not None:
            _nonneg_int(self.memory, "memory")
        if not isinstance(self.metadata, Mapping):
            raise ResourceValidationError("local resource inventory metadata must be a mapping")
        object.__setattr__(self, "cpus", cpus)
        object.__setattr__(self, "accelerators", MappingProxyType({name: accelerators[name] for name in sorted(accelerators)}))
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))

    @classmethod
    def local(cls) -> "LocalResourceInventory":
        """Discover a lightweight inventory using the current process environment."""

        return local_inventory()

    def to_data(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible inventory data."""

        return {
            "cpus": list(self.cpus),
            "accelerators": {name: list(values) for name, values in self.accelerators.items()},
            "memory": self.memory,
            "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "LocalResourceInventory":
        """Build an inventory from serialized data."""

        if not isinstance(data, Mapping):
            raise ResourceValidationError("local resource inventory must be a mapping")
        unknown = set(data) - {"cpus", "accelerators", "memory", "metadata"}
        if unknown:
            raise ResourceValidationError("local resource inventory has unknown fields", context={"fields": sorted(unknown)})
        return cls(
            cpus=tuple(data.get("cpus", ())),
            accelerators={} if "accelerators" not in data else data["accelerators"],
            memory=data.get("memory"),
            metadata={} if "metadata" not in data else data["metadata"],
        )

    def summary(self) -> dict[str, Any]:
        """Return a bounded deterministic reporting summary."""

        return {
            "cpu_count": len(self.cpus),
            "accelerator_counts": {name: len(values) for name, values in self.accelerators.items()},
            "memory": self.memory,
            "metadata": _thaw_json(self.metadata),
        }


def local_inventory(
    policy: str = "lightweight",
    *,
    environ: Mapping[str, str] | None = None,
    device_root: str | os.PathLike[str] | None = None,
    command_runner: Callable[..., Any] | None = None,
    timeout: float = 2.0,
) -> LocalResourceInventory:
    """Discover local capacity without importing or initializing ML frameworks.

    ``lightweight`` reads standard OS facts and an explicit accelerator override.
    ``external`` additionally accepts an injected bounded command runner for
    optional device discovery.  The injectable arguments make host-independent
    tests possible and are never mutated.
    """

    if policy not in {"lightweight", "external"}:
        raise ResourceValidationError("unsupported local inventory policy", context={"policy": policy})
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ResourceValidationError("inventory timeout must be positive", context={"timeout": timeout})
    environment = os.environ if environ is None else environ
    diagnostics: list[str] = []
    try:
        cpus = tuple(sorted(int(cpu) for cpu in os.sched_getaffinity(0)))
        cpu_source = "affinity"
    except Exception:
        count = os.cpu_count() or 1
        cpus = tuple(range(count))
        cpu_source = "cpu_count"
    if not cpus:
        raise ResourceValidationError("local CPU affinity contains no executable CPUs")
    memory, memory_source = _local_memory(diagnostics)
    accelerators = _accelerators_from_env(environment, diagnostics)
    if not accelerators and device_root is not None:
        accelerators = _accelerators_from_device_root(environment, device_root, diagnostics)
    if policy == "external" and command_runner is not None:
        _merge_external_accelerators(accelerators, command_runner, timeout, diagnostics)
    metadata: dict[str, Any] = {"policy": policy, "cpu_source": cpu_source, "memory_source": memory_source, "diagnostics": diagnostics[:8]}
    return LocalResourceInventory(cpus=cpus, accelerators=accelerators, memory=memory, metadata=metadata)


def _local_memory(diagnostics: list[str]) -> tuple[int | None, str]:
    available = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
                break
    except Exception as exc:
        diagnostics.append(f"memory discovery unavailable: {type(exc).__name__}")
    cgroup_available, cgroup_source, cgroup_readable = _cgroup_memory_available(diagnostics)
    if not cgroup_readable:
        return None, "unknown"
    if available is None and cgroup_available is None:
        return None, "unknown"
    if available is None:
        return cgroup_available, cgroup_source
    if cgroup_available is None:
        return available, "proc_meminfo"
    return min(available, cgroup_available), f"proc_meminfo+{cgroup_source}"


def _cgroup_memory_available(diagnostics: list[str]) -> tuple[int | None, str, bool]:
    """Return an effective cgroup memory allowance when it is explicit."""

    for limit_path, usage_path, source in (
        (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current"), "cgroup_v2"),
        (Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"), Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"), "cgroup_v1"),
    ):
        if not limit_path.exists():
            continue
        try:
            raw_limit = limit_path.read_text(encoding="utf-8").strip()
            if raw_limit == "max":
                return None, source, True
            limit = int(raw_limit)
            if limit <= 0 or limit >= 1 << 60:
                return None, source, True
            usage = int(usage_path.read_text(encoding="utf-8").strip())
            return max(0, limit - usage), source, True
        except Exception as exc:
            diagnostics.append(f"cgroup memory discovery unavailable: {type(exc).__name__}")
            return None, source, False
    return None, "none", True


def _accelerators_from_env(environ: Mapping[str, str], diagnostics: list[str]) -> dict[str, tuple[str | int, ...]]:
    raw = str(environ.get("DRYML_LOCAL_ACCELERATORS", "")).strip()
    if not raw:
        return {}
    if len(raw) > _MAX_EXPLICIT_ACCELERATOR_CHARS:
        raise ResourceValidationError("DRYML_LOCAL_ACCELERATORS exceeds the bounded limit")
    result: dict[str, tuple[str | int, ...]] = {}
    for group in raw.split(";"):
        if not group:
            continue
        if "=" not in group:
            raise ResourceValidationError("malformed DRYML_LOCAL_ACCELERATORS entry", context={"entry": group})
        name, values = group.split("=", 1)
        name = name.strip()
        parsed = tuple(int(value) if value.isdigit() else value for value in (item.strip() for item in values.split(",")) if value)
        if not name or not parsed:
            raise ResourceValidationError("malformed DRYML_LOCAL_ACCELERATORS entry", context={"entry": group})
        if len(parsed) > _MAX_EXPLICIT_ACCELERATOR_IDS:
            raise ResourceValidationError("DRYML_LOCAL_ACCELERATORS has too many accelerator identifiers", context={"accelerator": name})
        result[name] = parsed
    return result


def _accelerators_from_device_root(environ: Mapping[str, str], device_root: str | os.PathLike[str] | None, diagnostics: list[str]) -> dict[str, tuple[str | int, ...]]:
    """Discover only numeric GPU device files, optionally intersected with visibility."""

    root = Path(device_root)
    try:
        device_ids = tuple(sorted(
            int(path.name[6:])
            for path in root.iterdir()
            if path.name.startswith("nvidia") and path.name[6:].isdigit()
        ))
    except (OSError, ValueError) as exc:
        diagnostics.append(f"device-file accelerator discovery unavailable: {type(exc).__name__}")
        return {}
    if not device_ids:
        return {}
    visible = environ.get("CUDA_VISIBLE_DEVICES", environ.get("NVIDIA_VISIBLE_DEVICES"))
    if visible is None:
        return {"gpu": device_ids[:_MAX_EXPLICIT_ACCELERATOR_IDS]}
    values = tuple(item.strip() for item in visible.split(",") if item.strip())
    if any(value.lower() in {"none", "void", "all"} or not value.isdigit() for value in values):
        diagnostics.append("accelerator visibility was ambiguous")
        return {}
    allowed = {int(value) for value in values}
    return {"gpu": tuple(identifier for identifier in device_ids if identifier in allowed)}


def _merge_external_accelerators(accelerators: dict[str, tuple[str | int, ...]], runner: Callable[..., Any], timeout: float, diagnostics: list[str]) -> None:
    try:
        output = runner(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], timeout=timeout)
        if getattr(output, "returncode", 0) != 0:
            raise RuntimeError("external inventory command returned non-zero status")
        text = getattr(output, "stdout", output)
        if not isinstance(text, str):
            raise TypeError("runner output is not text")
        if len(text) > _MAX_EXTERNAL_OUTPUT_CHARS:
            diagnostics.append("external accelerator output was truncated")
            text = text[:_MAX_EXTERNAL_OUTPUT_CHARS]
            text = text.rsplit("\n", 1)[0] if "\n" in text else ""
        values = tuple(int(line.strip()) for line in text.splitlines() if line.strip())
        if any(value < 0 for value in values):
            raise ValueError("external accelerator identifiers must be non-negative")
        if len(values) > _MAX_EXTERNAL_DEVICE_IDS:
            diagnostics.append("external accelerator identifiers were truncated")
            values = values[:_MAX_EXTERNAL_DEVICE_IDS]
        if values:
            existing = accelerators.get("gpu", ())
            accelerators["gpu"] = tuple(sorted(set(existing) | set(values)))
    except Exception as exc:
        diagnostics.append(f"external accelerator discovery unavailable: {type(exc).__name__}")


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResourceValidationError(f"{name} must be an integer >= 0", context={"value": value})
    return value


def _freeze_json(value: Any, *, depth: int = 0) -> Any:
    if depth > _MAX_METADATA_DEPTH:
        raise ResourceValidationError("inventory metadata nesting exceeds the bounded limit")
    if value is None or isinstance(value, (int, bool)):
        return value
    if isinstance(value, str):
        if len(value) > _MAX_METADATA_STRING:
            raise ResourceValidationError("inventory metadata string exceeds the bounded limit")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ResourceValidationError("inventory metadata floats must be finite")
        return value
    if isinstance(value, Mapping):
        if len(value) > _MAX_METADATA_ITEMS:
            raise ResourceValidationError("inventory metadata mapping exceeds the bounded limit")
        result = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            normalized_key = str(key)
            if normalized_key in result:
                raise ResourceValidationError("inventory metadata keys collide after JSON normalization", context={"key": normalized_key})
            if len(normalized_key) > _MAX_METADATA_STRING:
                raise ResourceValidationError("inventory metadata key exceeds the bounded limit")
            result[normalized_key] = _freeze_json(item, depth=depth + 1)
        return MappingProxyType(result)
    if isinstance(value, (tuple, list)):
        if len(value) > _MAX_METADATA_ITEMS:
            raise ResourceValidationError("inventory metadata sequence exceeds the bounded limit")
        return tuple(_freeze_json(item, depth=depth + 1) for item in value)
    raise ResourceValidationError("inventory metadata must be JSON-compatible", context={"type": type(value).__name__})


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


__all__ = ["LocalResourceInventory", "local_inventory"]
