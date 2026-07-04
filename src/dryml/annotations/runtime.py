"""Runtime default annotation namespace helpers."""

from __future__ import annotations

from typing import Any

from dryml.runtime.specs import RuntimeContextSpec

from .decorators import default as default_fragment
from .namespaces import RUNTIME

_RESERVED = {"device_visibility", "limits", "env", "metadata", "world_allocation_id", "mode", "source"}


def runtime_default_fragment(**kwargs: Any) -> dict[str, Any]:
    """Normalize runtime default kwargs into a RuntimeContextSpec payload."""

    reserved = {key: kwargs.pop(key) for key in tuple(kwargs) if key in _RESERVED and key != "source"}
    frameworks = {str(key): value for key, value in kwargs.items()}
    payload = {
        "mode": reserved.get("mode", "orchestrator"),
        "device_visibility": reserved.get("device_visibility") or {},
        "frameworks": frameworks,
        "limits": reserved.get("limits") or {},
        "env": reserved.get("env") or {},
        "metadata": reserved.get("metadata") or {},
    }
    if reserved.get("world_allocation_id") is not None:
        payload["world_allocation_id"] = reserved["world_allocation_id"]
    return RuntimeContextSpec.from_data(payload).to_data()


def default(**kwargs: Any):
    """Decorate a target with an overrideable process-local runtime default."""

    source = kwargs.pop("source", None)
    return default_fragment(namespace=RUNTIME, fragment=runtime_default_fragment(**kwargs), source=source)


__all__ = ["default", "runtime_default_fragment"]
