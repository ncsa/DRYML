"""Dependency-light JAX/JAXLIB group adapter without eager framework imports."""

from __future__ import annotations

import os
import sys

from dryml.runtime.errors import FrameworkImportSafetyError
from dryml.runtime.frameworks import FrameworkImportPlan, FrameworkPostResult

_VISIBILITY_ENV = {
    "CUDA_VISIBLE_DEVICES": "gpu",
    "HIP_VISIBLE_DEVICES": "rocm",
    "ROCR_VISIBLE_DEVICES": "rocm",
    "XLA_VISIBLE_DEVICES": "xla",
}


class JaxRuntimeAdapter:
    """Coordinate JAX and JAXLIB under one visibility and status group."""

    def plan(self, runtime, visibility):
        """Return JAX/XLA visibility, allocator, and memory controls."""
        config = getattr(getattr(runtime, "spec", None), "framework", {}).get("jax", {})
        allocation = runtime.allocation
        updates = dict(visibility.env_updates)
        if not visibility.visible_devices.get("gpu", ()):
            updates.setdefault("JAX_PLATFORMS", str(config.get("platform", "cpu")))
        elif "platform" in config:
            updates["JAX_PLATFORMS"] = str(config["platform"])
        if config.get("allocator"):
            updates["XLA_PYTHON_CLIENT_ALLOCATOR"] = str(config["allocator"])
        if "preallocate" in config:
            updates["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if config["preallocate"] else "false"
        return FrameworkImportPlan(
            updates,
            visibility.visible_devices,
            accelerator_memory=getattr(allocation, "accelerator_memory", {}),
            accelerator_capacity=getattr(allocation, "metadata", {}).get("accelerator_memory_capacity", {}),
            allocator_policy=config.get("allocator"),
            process_memory=getattr(allocation, "memory", None),
        )

    def validate_before_import(self, plan) -> None:
        """Require one immutable group plan before either root executes."""
        if not isinstance(plan.adapter_plan, FrameworkImportPlan):
            raise FrameworkImportSafetyError("JAX import has no immutable runtime plan")

    def post_import(self, plan, module_name):
        """Validate direct JAX/JAXLIB imports and report group outcomes."""
        if module_name.partition(".")[0] == "jaxlib":
            _prove_pre_import_visibility(plan)
            return FrameworkPostResult({"visibility": "visibility-enforced", "threading": "pending-import", "allocator": "pending-import", "process_memory": "declarative", "accelerator_memory": "pending-import"})
        module = sys.modules[module_name]
        expected = plan.visible_devices.get("gpu", ())
        devices = getattr(module, "devices", None)
        if devices is None:
            raise FrameworkImportSafetyError("JAX cannot prove mandatory device visibility")
        try:
            actual = tuple(devices("gpu"))
        except RuntimeError as exc:
            if expected:
                raise FrameworkImportSafetyError("JAX GPU backend is unavailable for the assigned allocation") from exc
            actual = ()
        except TypeError:
            actual = tuple(device for device in devices() if getattr(device, "platform", None) == "gpu")
        if len(actual) != len(expected):
            raise FrameworkImportSafetyError("JAX visible devices do not match the assigned allocation")
        return FrameworkPostResult({"visibility": "visibility-enforced", "threading": "unsupported", "allocator": "framework-configured" if plan.allocator_policy else "unsupported", "process_memory": "declarative", "accelerator_memory": "unsupported"})


def _prove_pre_import_visibility(plan) -> None:
    """Verify that direct JAXLIB import observed every planned visibility value."""
    for variable, kind in _VISIBILITY_ENV.items():
        expected = ",".join(plan.visible_devices.get(kind, ()))
        if plan.env_updates.get(variable) != expected or os.environ.get(variable) != expected:
            raise FrameworkImportSafetyError("JAX cannot prove pre-import visibility for direct jaxlib import")


def adapter() -> JaxRuntimeAdapter:
    """Construct the lazy JAX/JAXLIB group adapter without importing either root."""
    return JaxRuntimeAdapter()
