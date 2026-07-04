"""World requirement/default annotation namespace helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dryml.worlds.specs import WorldRequirement, WorldSpec

from .decorators import default as default_fragment
from .decorators import require
from .namespaces import WORLD


def requirement_fragment(
    *,
    role: str = "main",
    roles: Mapping[str, Any] | None = None,
    replicas: Any = None,
    cpus: Any = None,
    memory: Any = None,
    accelerators: Mapping[str, Any] | None = None,
    devices: Mapping[str, Any] | None = None,
    named: Mapping[str, Any] | None = None,
    topology: Mapping[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Normalize hard world requirement kwargs into a WorldRequirement payload."""

    if extra:
        raise TypeError(f"unknown world requirement fields: {', '.join(sorted(extra))}")
    role_replicas = replicas if replicas is not None else {"exact": 1}
    payload = {"roles": dict(roles)} if roles is not None else {"roles": {role: {"replicas": role_replicas, "resources": _resources(cpus, memory, accelerators, devices, named), "topology": dict(topology or {})}}}
    return WorldRequirement.from_data(payload).to_data()


def default_fragment_data(
    *,
    role: str = "main",
    roles: Mapping[str, Any] | None = None,
    replicas: int = 1,
    cpus: int | None = None,
    memory: str | int | None = None,
    accelerators: Mapping[str, int] | None = None,
    devices: Mapping[str, Any] | None = None,
    named: Mapping[str, Any] | None = None,
    environment: str | None = None,
    runtime: str | None = None,
    env: Mapping[str, str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    backend: Mapping[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Normalize world default kwargs into a WorldSpec payload."""

    if extra:
        raise TypeError(f"unknown world default fields: {', '.join(sorted(extra))}")
    if roles is not None:
        payload = {"roles": dict(roles), "backend": backend or {"kind": "local", "parameters": {}}}
    else:
        process: dict[str, Any] = {"resources": _spec_resources(cpus, memory, accelerators, devices, named)}
        if environment is not None:
            process["environment"] = environment
        if runtime is not None:
            process["runtime"] = runtime
        if env is not None:
            process["env"] = dict(env)
        if metadata is not None:
            process["metadata"] = dict(metadata)
        payload = {"roles": {role: {"replicas": replicas, "process": process}}, "backend": backend or {"kind": "local", "parameters": {}}}
    return WorldSpec.from_data(payload).to_data()


def req(**kwargs: Any):
    """Decorate a target with a hard world/resource requirement."""

    source = kwargs.pop("source", None)
    return require(namespace=WORLD, fragment=requirement_fragment(**kwargs), source=source)


def default(**kwargs: Any):
    """Decorate a target with an overrideable default world spec."""

    source = kwargs.pop("source", None)
    return default_fragment(namespace=WORLD, fragment=default_fragment_data(**kwargs), source=source)


def _resources(cpus: Any, memory: Any, accelerators: Mapping[str, Any] | None, devices: Mapping[str, Any] | None, named: Mapping[str, Any] | None) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if cpus is not None:
        data["cpus"] = cpus
    if memory is not None:
        data["memory"] = memory
    if accelerators is not None:
        data["accelerators"] = dict(accelerators)
    if devices is not None:
        data["devices"] = dict(devices)
    if named is not None:
        data["named"] = dict(named)
    return data


def _spec_resources(cpus: int | None, memory: str | int | None, accelerators: Mapping[str, int] | None, devices: Mapping[str, Any] | None, named: Mapping[str, Any] | None) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if cpus is not None:
        data["cpus"] = cpus
    if memory is not None:
        data["memory"] = memory
    if accelerators is not None:
        data["accelerators"] = dict(accelerators)
    if devices is not None:
        data["devices"] = dict(devices)
    if named is not None:
        data["named"] = dict(named)
    return data


__all__ = ["default", "default_fragment_data", "req", "requirement_fragment"]
