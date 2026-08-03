"""Bounded, effect-free normalizers for public session configuration values."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from dryml.environments import EnvironmentRequirement, EnvironmentSpec, spec_from_data
from dryml.runtime import RequirementAxes, normalize_requirement_axes
from dryml.worlds import (
    LocalResourceInventory,
    ResourceSpec,
    WorldAllocation,
    WorldSpec,
    validate_world_allocation_spec,
)
from dryml.worlds.errors import WorldError

from .errors import SessionConfigurationError
from .model import _default_requirement_axes, SelectedWorldAllocation, SessionConfiguration

_MODES = frozenset({"python", "managed", "orchestrator"})
_MAX_DEPTH = 8
_MAX_ITEMS = 64
_MAX_NODES = 1024
_MAX_STRING = 4096
_ENVIRONMENT_FIELDS = frozenset({"requirements", "python", "excludes", "capabilities"})
_RESOURCE_FIELDS = frozenset({"cpus", "memory", "gpus", "accelerator_memory"})


def normalize_configuration(
    *,
    mode: str | None,
    resources: Mapping[str, Any] | None = None,
    allocation: Mapping[str, Any] | None = None,
    requested_environment: Mapping[str, Any] | None = None,
    requested_world: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
    requirement_axes: Mapping[str, bool] | None = None,
    **extra: Any,
) -> SessionConfiguration:
    """Normalize one complete replacement candidate without process effects.

    ``requirement_axes`` must be a complete mapping of the three canonical axis
    names to exact booleans. Omitting it selects the requested mode's default.
    """

    if extra:
        raise SessionConfigurationError("session configuration has unknown fields", context={"fields": sorted(extra)})
    if not isinstance(mode, str) or mode not in _MODES:
        raise SessionConfigurationError("session configuration requires a valid mode", context={"mode": mode})
    if resources is not None and allocation is not None:
        raise SessionConfigurationError("resources and allocation are mutually exclusive")
    try:
        normalized_resources = _normalize_resources(resources) if resources is not None else (ResourceSpec() if mode == "managed" else None)
        normalized_allocation = _normalize_allocation_section(allocation) if allocation is not None else None
        normalized_requested_environment = _normalize_requested_environment(requested_environment)
        normalized_world = _normalize_world(requested_world)
        normalized_environment = _normalize_environment(environment)
        normalized_axes = _normalize_requirement_axes(requirement_axes, mode=mode)
    except SessionConfigurationError:
        raise
    except Exception as exc:
        raise SessionConfigurationError(str(exc), context=_bounded_context(getattr(exc, "context", {}))) from exc
    controls = {
        "memory": "declarative" if (normalized_resources and normalized_resources.memory is not None) or (normalized_allocation and normalized_allocation.process.memory is not None) else "undeclared",
        "accelerator_memory": "declarative" if _has_accelerator_memory(normalized_resources, normalized_allocation) else "undeclared",
    }
    return SessionConfiguration(mode, normalized_resources, normalized_allocation, normalized_requested_environment, normalized_world, normalized_environment, controls, normalized_axes)


def _normalize_requirement_axes(value: Mapping[str, bool] | None, *, mode: str) -> RequirementAxes:
    """Return a validated explicit axis mask or one public mode default."""

    if value is None:
        return _default_requirement_axes(mode)
    try:
        return normalize_requirement_axes(value)
    except ValueError as exc:
        raise SessionConfigurationError(str(exc)) from exc


def select_world_allocation(
    value: WorldAllocation | Mapping[str, Any],
    *,
    role: str | None = None,
    replica: int | None = None,
    inventory: LocalResourceInventory | None = None,
) -> SelectedWorldAllocation:
    """Select one exact allocation process only when identity is unambiguous."""

    if (role is None) != (replica is None):
        raise SessionConfigurationError("exact allocation role and replica selectors must appear together")
    allocation = _coerce_world_allocation(value)
    candidates = [(role_name, process) for role_name in sorted(allocation.roles) for process in allocation.roles[role_name]]
    if role is None:
        if len(candidates) != 1:
            raise SessionConfigurationError("multi-process exact allocations require role and replica selectors")
        selected_role, process = candidates[0]
    else:
        if not isinstance(role, str) or not role or isinstance(replica, bool) or not isinstance(replica, int) or replica < 0:
            raise SessionConfigurationError("exact allocation selectors are invalid")
        matches = [(role_name, process) for role_name, process in candidates if role_name == role and process.replica == replica]
        if len(matches) != 1:
            raise SessionConfigurationError("exact allocation role/replica selector was not found", context={"role": role, "replica": replica})
        selected_role, process = matches[0]
    if inventory is not None:
        _validate_inherited_bounds(process, inventory)
    return SelectedWorldAllocation(selected_role, process)


def _normalize_resources(value: Mapping[str, Any]) -> ResourceSpec:
    frozen = _bounded_value(value, path="resources")
    if not isinstance(frozen, Mapping) or set(frozen) - _RESOURCE_FIELDS:
        raise SessionConfigurationError("session resources has unknown fields", context={"fields": sorted(set(frozen) - _RESOURCE_FIELDS) if isinstance(frozen, Mapping) else []})
    cpus = frozen.get("cpus", 0)
    gpus = frozen.get("gpus", 0)
    if isinstance(cpus, bool) or not isinstance(cpus, int) or ("cpus" in frozen and cpus <= 0):
        raise SessionConfigurationError("session resource cpus must be a positive integer")
    if isinstance(gpus, bool) or not isinstance(gpus, int) or gpus < 0:
        raise SessionConfigurationError("session resource gpus must be a nonnegative integer")
    data: dict[str, Any] = {"cpus": cpus}
    if "memory" in frozen:
        data["memory"] = frozen["memory"]
    if gpus:
        data["accelerators"] = {"gpu": gpus}
    if "accelerator_memory" in frozen:
        memory = frozen["accelerator_memory"]
        if not gpus:
            raise SessionConfigurationError("accelerator_memory requires gpus")
        if isinstance(memory, (str, int)) and not isinstance(memory, bool):
            memory = [memory] * gpus
        if isinstance(memory, (str, bytes)) or not hasattr(memory, "__iter__"):
            raise SessionConfigurationError("accelerator_memory must be a byte size or non-string sequence")
        data["accelerator_memory"] = {"gpu": list(memory)}
    try:
        resource = ResourceSpec.from_data(data)
    except WorldError as exc:
        raise SessionConfigurationError(str(exc), context=_bounded_context(exc.context)) from exc
    if resource.memory is not None and resource.memory <= 0:
        raise SessionConfigurationError("session resource memory must be positive")
    return resource


def _normalize_allocation_section(value: Mapping[str, Any]) -> SelectedWorldAllocation:
    if not isinstance(value, Mapping) or set(value) - {"value", "role", "replica"} or "value" not in value:
        raise SessionConfigurationError("allocation must contain value plus optional paired role and replica selectors")
    selectors = _bounded_value(
        {key: value[key] for key in ("role", "replica") if key in value},
        path="allocation",
    )
    return select_world_allocation(value["value"], role=selectors.get("role"), replica=selectors.get("replica"))


def _normalize_world(value: Mapping[str, Any] | None) -> WorldSpec | None:
    if value is None:
        return None
    frozen = _bounded_value(value, path="requested_world")
    if not isinstance(frozen, Mapping):
        raise SessionConfigurationError("requested_world must be a mapping")
    try:
        return WorldSpec.from_data(frozen)
    except WorldError as exc:
        raise SessionConfigurationError(str(exc), context=_bounded_context(exc.context)) from exc


def _normalize_requested_environment(value: Mapping[str, Any] | None) -> EnvironmentSpec | None:
    """Parse one bounded concrete future-worker environment candidate."""

    if value is None:
        return None
    frozen = _bounded_value(value, path="requested_environment")
    if not isinstance(frozen, Mapping):
        raise SessionConfigurationError("requested_environment must be an environment spec mapping")
    try:
        return spec_from_data(frozen)
    except Exception as exc:
        raise SessionConfigurationError(str(exc), context=_bounded_context(getattr(exc, "context", {}))) from exc


def _normalize_environment(value: Mapping[str, Any] | None) -> EnvironmentRequirement:
    if value is None:
        return EnvironmentRequirement()
    frozen = _bounded_value(value, path="environment")
    if not isinstance(frozen, Mapping) or set(frozen) - _ENVIRONMENT_FIELDS:
        raise SessionConfigurationError("environment has unknown fields")
    try:
        return EnvironmentRequirement(
            requirements=tuple(frozen.get("requirements", ())),
            python=frozen.get("python"),
            excludes=tuple(frozen.get("excludes", ())),
            capabilities=tuple(frozen.get("capabilities", ())),
        )
    except Exception as exc:
        raise SessionConfigurationError(str(exc), context=_bounded_context(getattr(exc, "context", {}))) from exc


def _coerce_world_allocation(value: WorldAllocation | Mapping[str, Any]) -> WorldAllocation:
    if isinstance(value, WorldAllocation):
        return value
    frozen = _bounded_value(value, path="world_allocation")
    if not isinstance(frozen, Mapping):
        raise SessionConfigurationError("exact allocation must be a WorldAllocation or canonical world_allocation envelope")
    try:
        if set(frozen) == {"schema", "payload"}:
            if frozen.get("schema") != "dryml.world_allocation.v1":
                raise SessionConfigurationError("exact allocation has an unsupported schema")
        else:
            validate_world_allocation_spec(frozen)
        return WorldAllocation.from_data(frozen["payload"])
    except WorldError as exc:
        raise SessionConfigurationError(str(exc), context=_bounded_context(exc.context)) from exc


def _validate_inherited_bounds(process, inventory: LocalResourceInventory) -> None:
    if not set(process.cpus).issubset(inventory.cpus):
        raise SessionConfigurationError("exact allocation CPU set broadens inherited inventory")
    if process.memory is not None and inventory.memory is not None and process.memory > inventory.memory:
        raise SessionConfigurationError("exact allocation memory broadens inherited inventory")
    for kind, devices in process.accelerators.items():
        if not set(devices).issubset(inventory.accelerators.get(kind, ())):
            raise SessionConfigurationError("exact allocation accelerators broaden inherited inventory", context={"accelerator": kind})
        for device, limit in process.accelerator_memory.get(kind, {}).items():
            capacity = inventory.accelerator_memory.get(kind, {}).get(device)
            if capacity is not None and limit > capacity:
                raise SessionConfigurationError("exact allocation accelerator memory broadens inherited inventory", context={"accelerator": kind})
    if process.devices:
        raise SessionConfigurationError("exact allocation device facts cannot broaden inherited inventory")
    for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        value = process.env.get(name)
        if value is None:
            continue
        allowed = {str(device) for device in inventory.accelerators.get("gpu", ())}
        tokens = {token.strip() for token in value.split(",") if token.strip()}
        if value.strip().lower() == "all" or not tokens.issubset(allowed):
            raise SessionConfigurationError("exact allocation visibility environment broadens inherited inventory", context={"variable": name})


def _bounded_value(value: Any, *, path: str, depth: int = 0, budget: list[int] | None = None, ancestors: set[int] | None = None) -> Any:
    if depth > _MAX_DEPTH:
        raise SessionConfigurationError("session configuration nesting exceeds the bounded limit", context={"path": path, "limit": _MAX_DEPTH})
    budget = [_MAX_NODES] if budget is None else budget
    if budget[0] <= 0:
        raise SessionConfigurationError("session configuration exceeds the aggregate bounded limit", context={"path": path, "limit": _MAX_NODES})
    budget[0] -= 1
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value.bit_length() > 4096:
            raise SessionConfigurationError("session configuration integer exceeds the bounded limit", context={"path": path})
        return value
    if isinstance(value, str):
        if len(value) > _MAX_STRING:
            raise SessionConfigurationError("session configuration string exceeds the bounded limit", context={"path": path, "limit": _MAX_STRING})
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SessionConfigurationError("session configuration floats must be finite", context={"path": path})
        return value
    ancestors = set() if ancestors is None else ancestors
    if isinstance(value, Mapping):
        if len(value) > _MAX_ITEMS or any(not isinstance(key, str) or not key or len(key) > _MAX_STRING for key in value):
            raise SessionConfigurationError("session configuration mapping exceeds the bounded grammar", context={"path": path})
        if id(value) in ancestors:
            raise SessionConfigurationError("session configuration must not contain cycles", context={"path": path})
        ancestors.add(id(value))
        try:
            return {key: _bounded_value(item, path=f"{path}.{key}", depth=depth + 1, budget=budget, ancestors=ancestors) for key, item in value.items()}
        finally:
            ancestors.remove(id(value))
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_ITEMS:
            raise SessionConfigurationError("session configuration sequence exceeds the bounded limit", context={"path": path, "limit": _MAX_ITEMS})
        if id(value) in ancestors:
            raise SessionConfigurationError("session configuration must not contain cycles", context={"path": path})
        ancestors.add(id(value))
        try:
            return [_bounded_value(item, path=f"{path}[{index}]", depth=depth + 1, budget=budget, ancestors=ancestors) for index, item in enumerate(value)]
        finally:
            ancestors.remove(id(value))
    raise SessionConfigurationError("session configuration must contain JSON-compatible values", context={"path": path, "type": type(value).__name__})


def _has_accelerator_memory(resources: ResourceSpec | None, allocation: SelectedWorldAllocation | None) -> bool:
    return bool((resources and resources.accelerator_memory) or (allocation and allocation.process.accelerator_memory))


def _bounded_context(context: Any) -> dict[str, Any]:
    try:
        value = _bounded_value(context if isinstance(context, Mapping) else {}, path="error")
        return value if isinstance(value, dict) else {}
    except SessionConfigurationError:
        return {"diagnostic": "redacted"}


__all__ = ["normalize_configuration", "select_world_allocation"]
