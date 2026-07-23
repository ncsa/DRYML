"""Legacy lowering helpers from old context resource dictionaries.

This module exists only as a compatibility bridge. New code should construct
``ResourceRequirement``/``ResourceSpec`` directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .resources import ResourceRequirement, ResourceSpec


def lower_legacy_resource_spec(value: Mapping[str, Any] | None) -> ResourceSpec:
    """Lower old ``num_cpus``/``num_gpus`` style input to ``ResourceSpec``."""

    value = dict(value or {})
    return ResourceSpec.from_data(
        {
            "cpus": value.get("num_cpus", 0),
            "memory": value.get("memory_bytes"),
            "accelerators": {"gpu": value.get("num_gpus", 0)} if value.get("num_gpus", 0) else {},
        }
    )


def lower_legacy_resource_requirement(value: Mapping[str, Any] | None) -> ResourceRequirement:
    """Lower old ``num_cpus``/``num_gpus`` style input to ``ResourceRequirement``."""

    spec = lower_legacy_resource_spec(value)
    payload: dict[str, Any] = {"cpus": {"min": spec.cpus}}
    if spec.memory is not None:
        payload["memory"] = {"min": f"{spec.memory}B"}
    if spec.accelerators.get("gpu", 0):
        payload["accelerators"] = {"gpu": {"min": spec.accelerators["gpu"]}}
    return ResourceRequirement.from_data(payload)


__all__ = ["lower_legacy_resource_requirement", "lower_legacy_resource_spec"]
