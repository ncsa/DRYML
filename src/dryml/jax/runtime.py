"""Lightweight JAX/JAXLIB group adapter without eager JAX imports."""

from __future__ import annotations

import sys

from dryml.runtime.errors import FrameworkImportSafetyError
from dryml.runtime.frameworks import FrameworkBootstrapResult, FrameworkPostResult, _LazyFrameworkAdapter


class JaxRuntimeAdapter(_LazyFrameworkAdapter):
    """Report the JAX group controls available from each independently loaded root."""

    name = "jax"
    module_name = "jax"

    def build_plan(self, runtime_spec, allocation_view, visibility_plan) -> FrameworkBootstrapResult:
        """Plan JAX/XLA environment controls without importing JAX or jaxlib."""

        result = super().build_plan(runtime_spec, allocation_view, visibility_plan)
        config = runtime_spec.frameworks.get(self.name, {})
        updates = dict(result.env_updates)
        if not result.visible_devices.get("gpu", ()):
            updates.setdefault("JAX_PLATFORMS", str(config.get("platform", "cpu")))
        elif "platform" in config:
            updates["JAX_PLATFORMS"] = str(config["platform"])
        if result.allocator_policy:
            updates["XLA_PYTHON_CLIENT_ALLOCATOR"] = result.allocator_policy
        if "preallocate" in config:
            updates["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if config["preallocate"] else "false"
        fraction = _uniform_fraction(result)
        if fraction is not None:
            updates["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(fraction)
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
        """Validate the immutable pre-import plan without changing process state."""

        if not isinstance(result, FrameworkBootstrapResult):
            raise FrameworkImportSafetyError("JAX import has no immutable runtime plan")

    def validate_before_activation(self, result: FrameworkBootstrapResult) -> None:
        """Reject either JAX group root loaded before the transition barrier."""

        self.validate_before_import(result)
        loaded = tuple(root for root in ("jax", "jaxlib") if root in sys.modules)
        if loaded:
            raise FrameworkImportSafetyError("framework was already imported before runtime bootstrap", context={"framework": self.name, "loaded": loaded, "fix": "apply runtime bootstrap before importing framework modules"})

    def apply_pre_import(self, result: FrameworkBootstrapResult, *, environ: dict[str, str] | None = None) -> None:
        """Apply only JAX/XLA controls selected during transition planning."""

        target = environ if environ is not None else __import__("os").environ
        target.update(result.env_updates)

    def post_import(self, result: FrameworkBootstrapResult, module_name: str) -> FrameworkPostResult:
        """Validate JAX platform visibility and publish JAX-level outcomes."""

        module = sys.modules[module_name]
        expected = result.visible_devices.get("gpu", ())
        devices = _gpu_devices(module, expected)
        if len(devices) != len(expected):
            raise FrameworkImportSafetyError("JAX visible devices do not match the assigned allocation", context={"expected": len(expected), "actual": len(devices)})
        statuses = {
            "visibility": "visibility-enforced",
            "process_memory": "declarative",
        }
        # jaxlib can prove visibility but cannot finalize JAX-level settings.
        if module_name.partition(".")[0] == "jaxlib":
            statuses.update({"threads": "pending-import", "allocator": "pending-import", "accelerator_memory": "pending-import"})
            return FrameworkPostResult(module_name, statuses)
        statuses["threads"] = "unsupported"
        fraction = _uniform_fraction(result)
        limits = result.accelerator_memory.get("gpu", {})
        if not limits:
            statuses["accelerator_memory"] = "unsupported"
        elif fraction is None:
            statuses["accelerator_memory"] = "unsupported"
        else:
            for device in limits:
                statuses[f"accelerator_memory:gpu:{device}"] = "framework-configured"
            statuses["accelerator_memory"] = "framework-configured"
        statuses["allocator"] = "framework-configured" if result.allocator_policy or "XLA_PYTHON_CLIENT_PREALLOCATE" in result.env_updates else "unsupported"
        return FrameworkPostResult(module_name, statuses)

    def apply_post_import(self, result: FrameworkBootstrapResult) -> None:
        """Retain the legacy bootstrap hook without bypassing module-aware setup."""

        self.post_import(result, self.module_name)


def _gpu_devices(module, expected):
    devices = getattr(module, "devices", None)
    if devices is None:
        raise FrameworkImportSafetyError("JAX cannot prove mandatory device visibility")
    try:
        return tuple(devices("gpu"))
    except RuntimeError as exc:
        if not expected:
            return ()
        raise FrameworkImportSafetyError("JAX GPU backend is unavailable for the assigned allocation") from exc
    except TypeError:
        return tuple(device for device in devices() if getattr(device, "platform", None) == "gpu")


def _uniform_fraction(result: FrameworkBootstrapResult) -> float | None:
    limits = result.accelerator_memory.get("gpu", {})
    if not limits:
        return None
    capacities = result.accelerator_capacity.get("gpu", {})
    fractions = []
    for device, limit in limits.items():
        capacity = capacities.get(device)
        if capacity is None or capacity <= 0 or limit > capacity:
            return None
        fractions.append(limit / capacity)
    return fractions[0] if fractions and all(fraction == fractions[0] for fraction in fractions) else None


def adapter() -> JaxRuntimeAdapter:
    """Construct the lightweight JAX group adapter."""

    return JaxRuntimeAdapter()
