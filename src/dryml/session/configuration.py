"""Bounded, effect-free normalizers for public session declarations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dryml.environments import EnvironmentRequirement
from dryml.formats import deep_freeze_json, json_ready
from dryml.worlds import LocalResourceInventory, ProcessAllocation, ResourceSpec, WorldAllocation
from dryml.worlds.errors import WorldError

from .errors import SessionConfigurationError
from .model import SelectedWorldAllocation, SessionConfiguration, default_requirement_axes, freeze_requirement_axes

_MODES = frozenset({"python", "managed", "orchestrator"})
_RESOURCE_FIELDS = frozenset({"cpus", "memory", "gpus", "accelerator_memory"})
_ENVIRONMENT_FIELDS = frozenset({"requirements", "python", "excludes", "capabilities", "tags", "dryml_protocol", "schema_versions"})


def normalize_configuration(
    *,
    mode: str,
    resources: Mapping[str, Any] | None = None,
    allocation: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
    requirement_axes: Mapping[str, bool] | None = None,
    **extra: Any,
) -> SessionConfiguration:
    """Normalize a complete v1.1 replacement candidate without side effects.

    Args:
        mode: Required ``python``, ``managed``, or ``orchestrator`` mode.
        resources: Optional concise managed resource declaration.
        allocation: Optional exact-allocation envelope with paired selectors.
        environment: Optional concise current-process software requirement.
        requirement_axes: Optional exact compatibility-axis replacement.
        extra: Rejected unknown fields, including deprecated source-v1 fields.

    Returns:
        A pure deeply immutable configuration candidate.

    Raises:
        SessionConfigurationError: If any field is unknown, malformed, unbounded,
            or inconsistent with the selected mode.

    Side Effects:
        None. This function never observes host state or imports optional
        frameworks.
    """

    if extra:
        raise SessionConfigurationError("session configuration fields are closed; source-v1 and unknown fields are unsupported", context={"fields": sorted(extra)})
    if not isinstance(mode, str) or mode not in _MODES:
        raise SessionConfigurationError("session configuration requires mode python, managed, or orchestrator")
    if resources is not None and allocation is not None:
        raise SessionConfigurationError("resources and allocation are mutually exclusive")
    try:
        normalized_resources = _normalize_resources(resources) if resources is not None else None
        normalized_allocation = _normalize_allocation_section(allocation) if allocation is not None else None
        normalized_environment = _normalize_environment(environment)
        axes = default_requirement_axes(mode) if requirement_axes is None else freeze_requirement_axes(_bounded_mapping(requirement_axes, "requirement_axes"))
        if mode != "managed" and (normalized_resources is not None or normalized_allocation is not None):
            raise SessionConfigurationError("resources and allocation require managed mode")
        controls = _controls(normalized_resources, normalized_allocation)
        if mode == "managed":
            return SessionConfiguration(mode, normalized_resources, normalized_allocation, normalized_environment, axes, controls)
        return SessionConfiguration(mode, None, None, normalized_environment, axes, controls)
    except SessionConfigurationError:
        raise
    except Exception as exc:
        raise SessionConfigurationError(str(exc), context=_error_context(exc)) from exc


def select_world_allocation(
    value: WorldAllocation | Mapping[str, Any],
    *,
    role: str | None = None,
    replica: int | None = None,
    inventory: LocalResourceInventory | None = None,
) -> SelectedWorldAllocation:
    """Select exactly one role-qualified process from an exact allocation.

    Args:
        value: A ``WorldAllocation`` or its self-validating v1.1 envelope.
        role: Role name paired with ``replica`` for multi-process allocations.
        replica: Non-negative role replica paired with ``role``.
        inventory: Optional inherited capacity evidence that selection may not
            broaden.

    Returns:
        The only selected exact process.

    Raises:
        SessionConfigurationError: If selectors are incomplete, ambiguous,
            absent, duplicate, malformed, or broader than inventory.

    Side Effects:
        None. Inventory is only inspected when explicitly supplied.
    """

    if (role is None) != (replica is None):
        raise SessionConfigurationError("exact allocation role and replica selectors must appear together")
    allocation = _coerce_allocation(value)
    candidates = [(name, process) for name in sorted(allocation.roles) for process in allocation.roles[name]]
    if role is None:
        if len(candidates) != 1:
            raise SessionConfigurationError("multi-process exact allocations require role and replica selectors")
        selected_role, process = candidates[0]
    else:
        if not isinstance(role, str) or not role or type(replica) is not int or replica < 0:
            raise SessionConfigurationError("exact allocation selectors are invalid")
        matches = [(name, process) for name, process in candidates if name == role and process.replica == replica]
        if len(matches) != 1:
            raise SessionConfigurationError("exact allocation role/replica selector was not found", context={"role": role, "replica": replica})
        selected_role, process = matches[0]
    if inventory is not None:
        if not isinstance(inventory, LocalResourceInventory):
            raise SessionConfigurationError("session inventory must be LocalResourceInventory")
        _validate_inventory_bounds(process, inventory)
    return SelectedWorldAllocation(selected_role, process)


def _normalize_resources(value: Mapping[str, Any]) -> ResourceSpec:
    data = _bounded_mapping(value, "resources")
    unknown = set(data) - _RESOURCE_FIELDS
    if unknown:
        raise SessionConfigurationError("session resources fields are closed", context={"fields": sorted(unknown)})
    cpus = data.get("cpus", 0)
    gpus = data.get("gpus", 0)
    if type(cpus) is not int or ("cpus" in data and cpus <= 0):
        raise SessionConfigurationError("session resource cpus must be a positive integer")
    if type(gpus) is not int or gpus < 0:
        raise SessionConfigurationError("session resource gpus must be a non-negative integer")
    resource: dict[str, Any] = {"cpus": cpus}
    if "memory" in data:
        resource["memory"] = data["memory"]
    if gpus:
        resource["accelerators"] = {"gpu": gpus}
    if "accelerator_memory" in data:
        if not gpus:
            raise SessionConfigurationError("accelerator_memory requires gpus")
        limits = data["accelerator_memory"]
        if isinstance(limits, (str, int)) and not isinstance(limits, bool):
            limits = [limits] * gpus
        if isinstance(limits, (str, bytes)) or not isinstance(limits, list):
            raise SessionConfigurationError("accelerator_memory must be a byte size or non-string sequence")
        resource["accelerator_memory"] = {"gpu": limits}
    try:
        result = ResourceSpec.from_data(resource)
    except WorldError as exc:
        raise SessionConfigurationError(str(exc), context=_error_context(exc)) from exc
    if result.memory is not None and result.memory <= 0:
        raise SessionConfigurationError("session resource memory must be positive")
    return result


def _normalize_allocation_section(value: Mapping[str, Any]) -> SelectedWorldAllocation:
    if not isinstance(value, Mapping) or set(value) - {"value", "role", "replica"} or "value" not in value:
        raise SessionConfigurationError("allocation must contain value plus optional paired role and replica selectors")
    # ``WorldAllocation`` is already an immutable typed value; freezing the
    # outer declaration must not reject that supported exact-allocation path.
    selectors = _bounded_mapping({key: item for key, item in value.items() if key != "value"}, "allocation")
    return select_world_allocation(value["value"], role=selectors.get("role"), replica=selectors.get("replica"))


def _normalize_environment(value: Mapping[str, Any] | None) -> EnvironmentRequirement:
    if value is None:
        return EnvironmentRequirement()
    data = _bounded_mapping(value, "environment")
    unknown = set(data) - _ENVIRONMENT_FIELDS
    if unknown:
        raise SessionConfigurationError("session environment fields are closed", context={"fields": sorted(unknown)})
    try:
        return EnvironmentRequirement(**dict(data))
    except Exception as exc:
        raise SessionConfigurationError(str(exc), context=_error_context(exc)) from exc


def _coerce_allocation(value: WorldAllocation | Mapping[str, Any]) -> WorldAllocation:
    if isinstance(value, WorldAllocation):
        return value
    if not isinstance(value, Mapping):
        raise SessionConfigurationError("exact allocation must be WorldAllocation or a v1.1 envelope")
    try:
        return WorldAllocation.from_data(_bounded_mapping(value, "world_allocation"))
    except Exception as exc:
        raise SessionConfigurationError(str(exc), context=_error_context(exc)) from exc


def _validate_inventory_bounds(process: ProcessAllocation, inventory: LocalResourceInventory) -> None:
    if not set(process.cpus).issubset(inventory.cpus):
        raise SessionConfigurationError("exact allocation CPU set broadens inherited inventory")
    if process.memory is not None:
        if inventory.memory is None:
            raise SessionConfigurationError("exact allocation memory capacity is unknown")
        if process.memory > inventory.memory:
            raise SessionConfigurationError("exact allocation memory broadens inherited inventory")
    for kind, devices in process.accelerators.items():
        if not set(devices).issubset(inventory.accelerators.get(kind, ())):
            raise SessionConfigurationError("exact allocation accelerators broaden inherited inventory", context={"accelerator": kind})
        for device, amount in process.accelerator_memory.get(kind, {}).items():
            capacity = inventory.accelerator_memory.get(kind, {}).get(device)
            if capacity is None:
                raise SessionConfigurationError("exact allocation accelerator memory capacity is unknown", context={"accelerator": kind, "device": str(device)})
            if amount > capacity:
                raise SessionConfigurationError("exact allocation accelerator memory broadens inherited inventory", context={"accelerator": kind})
    if process.devices or process.named:
        raise SessionConfigurationError("exact allocation unsupported device facts cannot broaden inherited inventory")
    for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        visible = process.env.get(name)
        if visible is None:
            continue
        allowed = {str(item) for item in inventory.accelerators.get("gpu", ())}
        values = {item.strip() for item in visible.split(",") if item.strip()}
        if visible.strip().lower() == "all" or not values.issubset(allowed):
            raise SessionConfigurationError("exact allocation visibility environment broadens inherited inventory", context={"variable": name})


def _controls(resources: ResourceSpec | None, allocation: SelectedWorldAllocation | None) -> dict[str, str]:
    process = None if allocation is None else allocation.process
    return {
        "memory": "declarative" if (resources and resources.memory is not None) or (process and process.memory is not None) else "undeclared",
        "accelerator_memory": "declarative" if (resources and resources.accelerator_memory) or (process and process.accelerator_memory) else "undeclared",
    }


def _bounded_mapping(value: Any, path: str) -> Mapping[str, Any]:
    try:
        frozen = deep_freeze_json(value)
    except Exception as exc:
        raise SessionConfigurationError("session configuration must be bounded JSON-compatible data", context={"path": path}) from exc
    if not isinstance(frozen, Mapping):
        raise SessionConfigurationError("session configuration field must be a mapping", context={"path": path})
    return json_ready(frozen)


def _error_context(exc: BaseException) -> dict[str, Any]:
    context = getattr(exc, "context", {})
    try:
        return dict(_bounded_mapping(context if isinstance(context, Mapping) else {}, "error"))
    except SessionConfigurationError:
        return {"diagnostic": "redacted"}


__all__ = ["normalize_configuration", "select_world_allocation"]
