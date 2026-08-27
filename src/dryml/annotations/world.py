"""World requirement and default annotation sugar."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dryml.worlds import WorldRequirement, WorldSpec

from .decorators import default as declare_default
from .decorators import require
from .namespaces import WORLD


def req(*, role: str = "main", roles: Mapping[str, Any] | None = None, replicas: Any = None, cpus: Any = None, memory: Any = None, accelerators: Mapping[str, Any] | None = None, accelerator_memory: Mapping[str, Any] | None = None, devices: Mapping[str, Any] | None = None, named: Mapping[str, Any] | None = None, topology: Mapping[str, Any] | None = None, **kwargs: Any):
    """Declare a hard world requirement without allocating or activating it.

    Args:
        role: Default role name when ``roles`` is omitted.
        roles: Complete role requirement mapping.
        replicas: Replica count constraint for the default role.
        cpus: CPU count constraint for the default role.
        memory: Memory count constraint for the default role.
        accelerators: Accelerator count constraints.
        accelerator_memory: Minimum bytes required for every assigned device.
        devices: Device count constraints.
        named: Named resource count constraints.
        topology: Preserved topology declaration.
        **kwargs: Annotation metadata ``source``, ``priority``, and
            ``merge_policy``.

    Returns:
        An identity-preserving declaration decorator.
    """

    source, priority, merge_policy = _metadata(kwargs)
    if kwargs:
        raise TypeError(f"unknown world requirement fields: {', '.join(sorted(kwargs))}")
    resources = {name: value for name, value in {"cpus": cpus, "memory": memory, "accelerators": accelerators, "accelerator_memory": accelerator_memory, "devices": devices, "named": named}.items() if value is not None}
    role_data = dict(roles) if roles is not None else {role: {"replicas": replicas if replicas is not None else {"min": 1, "max": 1}, "resources": resources, "topology": dict(topology or {})}}
    value = WorldRequirement.from_payload({"roles": role_data})
    return require(namespace=WORLD, fragment=value.to_data(), source=source, priority=priority, merge_policy=merge_policy)


def default(*, role: str = "main", roles: Mapping[str, Any] | None = None, replicas: int = 1, cpus: int | None = None, memory: str | int | None = None, accelerators: Mapping[str, int] | None = None, devices: Mapping[str, Any] | None = None, named: Mapping[str, Any] | None = None, environment: str | None = None, runtime: str | None = None, env: Mapping[str, str] | None = None, metadata: Mapping[str, Any] | None = None, backend: Mapping[str, Any] | None = None, **kwargs: Any):
    """Declare a world selection default without planning or reserving it.

    Args:
        role: Default role name when ``roles`` is omitted.
        roles: Complete requested-role mapping.
        replicas: Positive replica count for the default role.
        cpus: Concrete CPU count for the default role.
        memory: Concrete memory quantity for the default role.
        accelerators: Concrete accelerator counts.
        devices: Concrete device assignments.
        named: Concrete named resources.
        environment: Optional environment selector ID.
        runtime: Optional runtime context ID.
        env: Process environment declarations.
        metadata: Process diagnostic metadata.
        backend: Requested backend declaration.
        **kwargs: Annotation metadata ``source``, ``priority``, and
            ``merge_policy``.

    Returns:
        An identity-preserving declaration decorator.
    """

    source, priority, merge_policy = _metadata(kwargs)
    if kwargs:
        raise TypeError(f"unknown world default fields: {', '.join(sorted(kwargs))}")
    if roles is None:
        resources = {name: value for name, value in {"cpus": cpus, "memory": memory, "accelerators": accelerators, "devices": devices, "named": named}.items() if value is not None}
        process = {"resources": resources}
        if environment is not None:
            process["environment"] = environment
        if runtime is not None:
            process["runtime"] = runtime
        if env is not None:
            process["env"] = dict(env)
        if metadata is not None:
            process["metadata"] = dict(metadata)
        roles = {role: {"replicas": replicas, "process": process}}
    value = WorldSpec.from_payload({"roles": dict(roles), "backend": backend or {"kind": "local", "parameters": {}}}).to_data()
    return declare_default(namespace=WORLD, fragment=value, source=source, priority=priority, merge_policy=merge_policy)


def _metadata(kwargs: dict[str, Any]) -> tuple[Any, int, str | None]:
    return kwargs.pop("source", None), kwargs.pop("priority", 0), kwargs.pop("merge_policy", None)


__all__ = ["default", "req"]
