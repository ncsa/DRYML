"""Lightweight TensorFlow import adapter; it never imports TensorFlow itself."""

from __future__ import annotations

import sys

from dryml.runtime.errors import FrameworkImportSafetyError
from dryml.runtime.frameworks import FrameworkBootstrapResult, FrameworkPostResult, _LazyFrameworkAdapter


class TensorFlowRuntimeAdapter(_LazyFrameworkAdapter):
    """Apply supported TensorFlow controls after ordinary module execution."""

    name = "tensorflow"
    module_name = "tensorflow"

    def validate_before_import(self, result: FrameworkBootstrapResult) -> None:
        """Validate the immutable pre-import plan without changing process state."""

        if not isinstance(result, FrameworkBootstrapResult):
            raise FrameworkImportSafetyError("TensorFlow import has no immutable runtime plan")

    def validate_before_activation(self, result: FrameworkBootstrapResult) -> None:
        """Reject a TensorFlow root loaded before the transition barrier."""

        self.validate_before_import(result)
        if self.module_name in sys.modules:
            raise FrameworkImportSafetyError("framework was already imported before runtime bootstrap", context={"framework": self.name, "fix": "apply runtime bootstrap before importing framework modules"})

    def apply_pre_import(self, result: FrameworkBootstrapResult, *, environ: dict[str, str] | None = None) -> None:
        """Apply only transition-planned environment controls before import."""

        target = environ if environ is not None else __import__("os").environ
        target.update(result.env_updates)

    def post_import(self, result: FrameworkBootstrapResult, module_name: str) -> FrameworkPostResult:
        """Configure TensorFlow APIs and return adapter-local control outcomes."""

        module = sys.modules[module_name]
        config = getattr(module, "config", None)
        if config is None:
            raise FrameworkImportSafetyError("TensorFlow cannot prove mandatory device visibility", context={"module": module_name})
        expected = result.visible_devices.get("gpu", ())
        physical = _devices(config, "get_physical_devices", "GPU")
        if expected:
            if len(physical) != len(expected):
                raise FrameworkImportSafetyError("TensorFlow visible devices do not match the assigned allocation", context={"expected": len(expected), "actual": len(physical)})
            setter = getattr(config, "set_visible_devices", None)
            if setter is None:
                raise FrameworkImportSafetyError("TensorFlow cannot configure mandatory visible devices")
            setter(physical, "GPU")
        else:
            setter = getattr(config, "set_visible_devices", None)
            if setter is None:
                raise FrameworkImportSafetyError("TensorFlow cannot hide mandatory visible devices")
            setter([], "GPU")
        visible = _devices(config, "get_visible_devices", "GPU")
        if len(visible) != len(expected):
            raise FrameworkImportSafetyError("TensorFlow did not enforce mandatory visible devices", context={"expected": len(expected), "actual": len(visible)})

        statuses = {"visibility": "visibility-enforced", "process_memory": "declarative"}
        threads = getattr(result, "post_import_threads", {}).get(self.name)
        if threads is None:
            statuses["threads"] = "unsupported"
        elif hasattr(module, "config") and hasattr(module.config, "threading"):
            threading = module.config.threading
            threading.set_intra_op_parallelism_threads(threads)
            interop = result.post_import_interop_threads.get(self.name)
            if interop is not None and hasattr(threading, "set_inter_op_parallelism_threads"):
                threading.set_inter_op_parallelism_threads(interop)
            statuses["threads"] = "framework-configured"
        else:
            statuses["threads"] = "unsupported"
        _configure_memory(config, physical, expected, result, statuses)
        _configure_allocator(config, physical, result, statuses)
        return FrameworkPostResult(module_name, statuses)

    def apply_post_import(self, result: FrameworkBootstrapResult) -> None:
        """Retain the legacy bootstrap hook without bypassing module-aware setup."""

        self.post_import(result, self.module_name)


def _devices(config, method_name: str, kind: str):
    method = getattr(config, method_name, None)
    if method is None and method_name == "get_physical_devices":
        method = getattr(config, "list_physical_devices", None)
    if method is None:
        raise FrameworkImportSafetyError("TensorFlow device API is unavailable", context={"method": method_name})
    return tuple(method(kind))


def _configure_memory(config, physical, expected, result, statuses) -> None:
    limits = result.accelerator_memory.get("gpu", {})
    if not limits:
        statuses["accelerator_memory"] = "unsupported"
        return
    configure = getattr(config, "set_logical_device_configuration", None)
    logical = getattr(config, "LogicalDeviceConfiguration", None)
    if configure is None or logical is None or len(physical) != len(expected):
        statuses["accelerator_memory"] = "unsupported"
        return
    for ordinal, device in enumerate(expected):
        limit = limits.get(device)
        if limit is None:
            continue
        try:
            configure(physical[ordinal], [logical(memory_limit=limit / (1024 * 1024))])
        except (RuntimeError, ValueError):
            statuses[f"accelerator_memory:gpu:{device}"] = "failed"
        else:
            statuses[f"accelerator_memory:gpu:{device}"] = "framework-configured"
    statuses["accelerator_memory"] = "framework-configured" if all(statuses.get(f"accelerator_memory:gpu:{device}") == "framework-configured" for device in limits) else "failed"


def _configure_allocator(config, physical, result, statuses) -> None:
    if result.allocator_policy != "memory_growth":
        statuses["allocator"] = "unsupported"
        return
    setter = getattr(getattr(config, "experimental", None), "set_memory_growth", None)
    if setter is None:
        statuses["allocator"] = "unsupported"
        return
    for device in physical:
        setter(device, True)
    statuses["allocator"] = "framework-configured"


def adapter() -> TensorFlowRuntimeAdapter:
    """Construct the lightweight TensorFlow adapter."""

    return TensorFlowRuntimeAdapter()
