"""Lightweight PyTorch import adapter; it never imports PyTorch itself."""

from __future__ import annotations

import sys

from dryml.runtime.errors import FrameworkImportSafetyError
from dryml.runtime.frameworks import FrameworkBootstrapResult, FrameworkPostResult, _LazyFrameworkAdapter


class TorchRuntimeAdapter(_LazyFrameworkAdapter):
    """Apply supported PyTorch controls after ordinary module execution."""

    name = "torch"
    module_name = "torch"

    def build_plan(self, runtime_spec, allocation_view, visibility_plan) -> FrameworkBootstrapResult:
        """Plan PyTorch's documented allocator environment policy lazily."""

        result = super().build_plan(runtime_spec, allocation_view, visibility_plan)
        updates = dict(result.env_updates)
        if result.allocator_policy:
            updates["PYTORCH_CUDA_ALLOC_CONF"] = result.allocator_policy
        return FrameworkBootstrapResult(
            env_updates=updates,
            post_import_threads=result.post_import_threads,
            post_import_interop_threads=result.post_import_interop_threads,
            visible_devices=result.visible_devices,
            accelerator_memory=result.accelerator_memory,
            accelerator_capacity=result.accelerator_capacity,
            allocator_policy=result.allocator_policy,
            process_memory=result.process_memory,
            cpu_affinity=result.cpu_affinity,
            memory_limit=result.memory_limit,
        )

    def validate_before_import(self, result: FrameworkBootstrapResult) -> None:
        """Validate the immutable pre-import plan without mutating the process."""

        if not isinstance(result, FrameworkBootstrapResult):
            raise FrameworkImportSafetyError("PyTorch import has no immutable runtime plan")

    def validate_before_activation(self, result: FrameworkBootstrapResult) -> None:
        """Reject a PyTorch root loaded before the transition barrier."""

        self.validate_before_import(result)
        if self.module_name in sys.modules:
            raise FrameworkImportSafetyError("framework was already imported before runtime bootstrap", context={"framework": self.name, "fix": "apply runtime bootstrap before importing framework modules"})

    def apply_pre_import(self, result: FrameworkBootstrapResult, *, environ: dict[str, str] | None = None) -> None:
        """Apply allocator environment variables already selected by planning."""

        target = environ if environ is not None else __import__("os").environ
        target.update(result.env_updates)

    def post_import(self, result: FrameworkBootstrapResult, module_name: str) -> FrameworkPostResult:
        """Configure PyTorch's supported thread and allocator controls."""

        module = sys.modules[module_name]
        cuda = getattr(module, "cuda", None)
        count = getattr(cuda, "device_count", None)
        if count is None:
            raise FrameworkImportSafetyError("PyTorch cannot prove mandatory CUDA visibility")
        expected = result.visible_devices.get("gpu", ())
        actual = count()
        if actual != len(expected):
            raise FrameworkImportSafetyError("PyTorch visible devices do not match the assigned allocation", context={"expected": len(expected), "actual": actual})
        statuses = {"visibility": "visibility-enforced", "process_memory": "declarative"}
        threads = getattr(result, "post_import_threads", {}).get(self.name)
        if threads is None:
            statuses["threads"] = "unsupported"
        elif hasattr(module, "set_num_threads"):
            module.set_num_threads(threads)
            interop = result.post_import_interop_threads.get(self.name)
            if interop is not None and hasattr(module, "set_num_interop_threads"):
                module.set_num_interop_threads(interop)
            statuses["threads"] = "framework-configured"
        else:
            statuses["threads"] = "unsupported"
        _configure_memory(cuda, expected, result, statuses)
        statuses["allocator"] = "framework-configured" if result.allocator_policy else "unsupported"
        return FrameworkPostResult(module_name, statuses)

    def apply_post_import(self, result: FrameworkBootstrapResult) -> None:
        """Retain the legacy bootstrap hook without bypassing module-aware setup."""

        self.post_import(result, self.module_name)


def _configure_memory(cuda, expected, result, statuses) -> None:
    limits = result.accelerator_memory.get("gpu", {})
    capacities = result.accelerator_capacity.get("gpu", {})
    setter = getattr(cuda, "set_per_process_memory_fraction", None)
    if not limits or setter is None:
        statuses["accelerator_memory"] = "unsupported"
        return
    outcomes = []
    for ordinal, device in enumerate(expected):
        limit = limits.get(device)
        capacity = capacities.get(device)
        if limit is None:
            continue
        if capacity is None or capacity <= 0 or limit > capacity:
            statuses[f"accelerator_memory:gpu:{device}"] = "unsupported"
            outcomes.append("unsupported")
            continue
        try:
            setter(limit / capacity, ordinal)
        except (RuntimeError, ValueError):
            # PyTorch documents these as ordinary allocator rejections. CUDA is
            # rechecked before reporting a recoverable best-effort failure.
            if cuda.device_count() != len(expected):
                raise FrameworkImportSafetyError("PyTorch allocator rejection left visibility unusable")
            statuses[f"accelerator_memory:gpu:{device}"] = "failed"
            outcomes.append("failed")
        else:
            statuses[f"accelerator_memory:gpu:{device}"] = "framework-configured"
            outcomes.append("framework-configured")
    statuses["accelerator_memory"] = "framework-configured" if outcomes and all(outcome == "framework-configured" for outcome in outcomes) else ("failed" if "failed" in outcomes else "unsupported")


def adapter() -> TorchRuntimeAdapter:
    """Construct the lightweight PyTorch adapter."""

    return TorchRuntimeAdapter()
