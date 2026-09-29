"""Dependency-light PyTorch adapter used only after controlled root import."""

from __future__ import annotations

import sys

from dryml.runtime.errors import FrameworkImportSafetyError
from dryml.runtime.frameworks import FrameworkImportPlan, FrameworkPostResult


class TorchRuntimeAdapter:
    """Report PyTorch controls without importing PyTorch during planning."""

    def plan(self, runtime, visibility):
        """Return visibility, threading, allocator, and memory controls."""
        updates = dict(visibility.env_updates)
        config = getattr(getattr(runtime, "spec", None), "framework", {}).get("torch", {})
        if config.get("allocator"):
            updates["PYTORCH_CUDA_ALLOC_CONF"] = str(config["allocator"])
        allocation = runtime.allocation
        threads = config.get("threads", config.get("num_threads"))
        if threads is None:
            threads = len(getattr(allocation, "cpus", ())) or None
        return FrameworkImportPlan(
            updates,
            visibility.visible_devices,
            threads,
            config.get("interop_threads", config.get("num_interop_threads")),
            getattr(allocation, "accelerator_memory", {}),
            getattr(allocation, "metadata", {}).get("accelerator_memory_capacity", {}),
            config.get("allocator"),
            getattr(allocation, "memory", None),
        )

    def validate_before_import(self, plan) -> None:
        """Require one immutable plan before PyTorch module execution."""
        if not isinstance(plan.adapter_plan, FrameworkImportPlan):
            raise FrameworkImportSafetyError("PyTorch import has no immutable runtime plan")

    def post_import(self, plan, module_name):
        """Apply supported controls and verify mandatory CUDA visibility."""
        module = sys.modules[module_name]
        cuda = getattr(module, "cuda", None)
        count = getattr(cuda, "device_count", None)
        if count is None:
            raise FrameworkImportSafetyError("PyTorch cannot prove mandatory CUDA visibility")
        expected = plan.visible_devices.get("gpu", ())
        if count() != len(expected):
            raise FrameworkImportSafetyError("PyTorch visible devices do not match the assigned allocation")
        if plan.threads is None:
            threading = "unsupported"
        elif hasattr(module, "set_num_threads"):
            module.set_num_threads(plan.threads)
            if plan.interop_threads is not None and hasattr(module, "set_num_interop_threads"):
                module.set_num_interop_threads(plan.interop_threads)
            threading = "framework-configured"
        else:
            threading = "unsupported"
        statuses = {
            "visibility": "visibility-enforced",
            "threading": threading,
            "allocator": "framework-configured" if plan.allocator_policy else "unsupported",
            "process_memory": "declarative",
        }
        _configure_memory(cuda, expected, plan, statuses)
        return FrameworkPostResult(statuses)


def _configure_memory(cuda, expected, plan, statuses) -> None:
    """Apply supported per-device memory fractions and report each outcome."""
    limits = plan.accelerator_memory.get("gpu", {})
    capacities = plan.accelerator_capacity.get("gpu", {})
    setter = getattr(cuda, "set_per_process_memory_fraction", None)
    if not limits or setter is None:
        statuses["accelerator_memory"] = "unsupported"
        return
    outcomes = []
    for ordinal, device in enumerate(expected):
        limit = limits.get(device)
        if limit is None:
            continue
        capacity = capacities.get(device)
        key = f"accelerator_memory:gpu:{device}"
        if capacity is None or limit > capacity:
            statuses[key] = "unsupported"
            outcomes.append("unsupported")
            continue
        try:
            setter(limit / capacity, ordinal)
        except (RuntimeError, ValueError):
            if cuda.device_count() != len(expected):
                raise FrameworkImportSafetyError("PyTorch allocator failure invalidated mandatory visibility")
            statuses[key] = "failed"
            outcomes.append("failed")
        else:
            statuses[key] = "framework-configured"
            outcomes.append("framework-configured")
    statuses["accelerator_memory"] = "framework-configured" if outcomes and all(value == "framework-configured" for value in outcomes) else ("failed" if "failed" in outcomes else "unsupported")


def adapter() -> TorchRuntimeAdapter:
    """Construct the lazy PyTorch adapter without importing PyTorch."""
    return TorchRuntimeAdapter()
