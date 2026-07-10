"""Import-safe local resource inventory discovery for requested worlds."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ResourceValidationError


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
        object.__setattr__(self, "cpus", cpus)
        object.__setattr__(self, "accelerators", {name: accelerators[name] for name in sorted(accelerators)})
        object.__setattr__(self, "metadata", dict(self.metadata))

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
            "metadata": dict(self.metadata),
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
            accelerators=data.get("accelerators") or {},
            memory=data.get("memory"),
            metadata=data.get("metadata") or {},
        )

    def summary(self) -> dict[str, Any]:
        """Return a bounded deterministic reporting summary."""

        return self.to_data()


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
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
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
        cpus = (0,)
        diagnostics.append("empty CPU affinity fell back to CPU 0")
    memory = _local_memory(diagnostics)
    accelerators = _accelerators_from_env(environment, diagnostics)
    if policy == "external" and command_runner is not None:
        _merge_external_accelerators(accelerators, command_runner, timeout, diagnostics)
    # Device names only supplement an explicitly visible, unambiguous inventory.
    if not accelerators and device_root is not None:
        diagnostics.append(f"device root {Path(device_root)} was not used without explicit visibility")
    metadata: dict[str, Any] = {"policy": policy, "cpu_source": cpu_source, "memory_source": "proc_meminfo", "diagnostics": diagnostics[:8]}
    return LocalResourceInventory(cpus=cpus, accelerators=accelerators, memory=memory, metadata=metadata)


def _local_memory(diagnostics: list[str]) -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                value = int(line.split()[1]) * 1024
                return value
    except Exception as exc:
        diagnostics.append(f"memory discovery unavailable: {type(exc).__name__}")
    return None


def _accelerators_from_env(environ: Mapping[str, str], diagnostics: list[str]) -> dict[str, tuple[str | int, ...]]:
    raw = str(environ.get("DRYML_LOCAL_ACCELERATORS", "")).strip()
    if not raw:
        return {}
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
        result[name] = parsed
    return result


def _merge_external_accelerators(accelerators: dict[str, tuple[str | int, ...]], runner: Callable[..., Any], timeout: float, diagnostics: list[str]) -> None:
    try:
        output = runner(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], timeout=timeout)
        text = getattr(output, "stdout", output)
        if not isinstance(text, str):
            raise TypeError("runner output is not text")
        values = tuple(int(line.strip()) for line in text.splitlines() if line.strip())
        if values:
            existing = accelerators.get("gpu", ())
            accelerators["gpu"] = tuple(sorted(set(existing) | set(values)))
    except Exception as exc:
        diagnostics.append(f"external accelerator discovery unavailable: {type(exc).__name__}")


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResourceValidationError(f"{name} must be an integer >= 0", context={"value": value})
    return value


__all__ = ["LocalResourceInventory", "local_inventory"]
