"""Dependency-light TensorFlow adapter used only after controlled root import."""

from __future__ import annotations

import sys

from dryml.runtime.errors import FrameworkImportSafetyError
from dryml.runtime.frameworks import FrameworkImportPlan, FrameworkPostResult


class TensorFlowRuntimeAdapter:
    """Apply conservative TensorFlow post-import controls without eager imports."""

    def plan(self, runtime, visibility):
        """Return immutable visibility, threading, allocator, and memory controls."""
        config = getattr(getattr(runtime, "spec", None), "framework", {}).get("tensorflow", {})
        allocation = runtime.allocation
        threads = config.get("threads", config.get("num_threads"))
        if threads is None:
            threads = len(getattr(allocation, "cpus", ())) or None
        return FrameworkImportPlan(
            visibility.env_updates,
            visibility.visible_devices,
            threads,
            config.get("interop_threads", config.get("num_interop_threads")),
            getattr(allocation, "accelerator_memory", {}),
            getattr(allocation, "metadata", {}).get("accelerator_memory_capacity", {}),
            config.get("allocator"),
            getattr(allocation, "memory", None),
        )

    def validate_before_import(self, plan) -> None:
        """Require one immutable plan before TensorFlow module execution."""
        if not isinstance(plan.adapter_plan, FrameworkImportPlan):
            raise FrameworkImportSafetyError("TensorFlow import has no immutable runtime plan")

    def post_import(self, plan, module_name):
        """Configure TensorFlow APIs and return truthful independent outcomes."""
        module = sys.modules[module_name]
        config = getattr(module, "config", None)
        if config is None:
            raise FrameworkImportSafetyError("TensorFlow cannot prove mandatory device visibility")
        expected = plan.visible_devices.get("gpu", ())
        physical = _devices(config, "get_physical_devices")
        setter = getattr(config, "set_visible_devices", None)
        if setter is None or (expected and len(physical) != len(expected)):
            raise FrameworkImportSafetyError("TensorFlow cannot configure mandatory visible devices")
        setter(physical if expected else [], "GPU")
        if len(_devices(config, "get_visible_devices")) != len(expected):
            raise FrameworkImportSafetyError("TensorFlow did not enforce mandatory visible devices")
        statuses = {"visibility": "visibility-enforced", "process_memory": "declarative"}
        threading_api = getattr(config, "threading", None)
        if plan.threads is not None and threading_api is not None and hasattr(threading_api, "set_intra_op_parallelism_threads"):
            threading_api.set_intra_op_parallelism_threads(plan.threads)
            if plan.interop_threads is not None and hasattr(threading_api, "set_inter_op_parallelism_threads"):
                threading_api.set_inter_op_parallelism_threads(plan.interop_threads)
            statuses["threading"] = "framework-configured"
        else:
            statuses["threading"] = "unsupported"
        statuses["allocator"] = "unsupported"
        statuses["accelerator_memory"] = "unsupported"
        return FrameworkPostResult(statuses)


def _devices(config, method_name):
    """Return TensorFlow GPU devices through its supported configuration API."""
    method = getattr(config, method_name, None)
    if method is None and method_name == "get_physical_devices":
        method = getattr(config, "list_physical_devices", None)
    if method is None:
        raise FrameworkImportSafetyError("TensorFlow device API is unavailable")
    return tuple(method("GPU"))


def adapter() -> TensorFlowRuntimeAdapter:
    """Construct the lazy TensorFlow adapter without importing TensorFlow."""
    return TensorFlowRuntimeAdapter()
