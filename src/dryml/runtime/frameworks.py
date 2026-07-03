"""Lightweight framework bootstrap adapter protocols and adapters."""

from __future__ import annotations

import os
import resource
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from .allocation import RuntimeAllocationView
from .devices import DeviceVisibilityPlan
from .errors import FrameworkImportSafetyError
from .specs import RuntimeContextSpec


@dataclass(frozen=True, slots=True)
class FrameworkBootstrapResult:
    """Framework-specific environment and post-import actions."""

    env_updates: Mapping[str, str] = field(default_factory=dict)
    post_import_threads: Mapping[str, int] = field(default_factory=dict)
    cpu_affinity: tuple[int, ...] | None = None
    memory_limit: int | None = None


class FrameworkBootstrapAdapter(Protocol):
    """Minimal protocol for framework bootstrap adapters."""

    name: str

    def build_plan(self, runtime_spec: RuntimeContextSpec, allocation_view: RuntimeAllocationView | Any, visibility_plan: DeviceVisibilityPlan) -> FrameworkBootstrapResult:
        """Return framework-specific bootstrap effects."""

    def validate_before_import(self, result: FrameworkBootstrapResult) -> None:
        """Validate import ordering before framework import."""

    def apply_pre_import(self, result: FrameworkBootstrapResult, *, environ: dict[str, str] | None = None) -> None:
        """Apply environment updates before importing the framework."""

    def apply_post_import(self, result: FrameworkBootstrapResult) -> None:
        """Apply optional post-import configuration."""


class PlainBootstrapAdapter:
    """Adapter for process-global plain Python/BLAS resource controls."""

    name = "plain"
    thread_env_vars = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

    def build_plan(self, runtime_spec: RuntimeContextSpec, allocation_view: RuntimeAllocationView | Any, visibility_plan: DeviceVisibilityPlan) -> FrameworkBootstrapResult:
        config = runtime_spec.frameworks.get("plain", {})
        updates = {}
        threads = config.get("num_threads") or (len(getattr(allocation_view, "cpus", ())) or None)
        if threads:
            updates.update({name: str(int(threads)) for name in self.thread_env_vars})
        updates.update({str(key): str(value) for key, value in runtime_spec.env.items()})
        cpu_affinity = _cpu_affinity_from_config(config, allocation_view)
        memory_limit = _memory_limit_from_config(runtime_spec.limits, config)
        return FrameworkBootstrapResult(updates, cpu_affinity=cpu_affinity, memory_limit=memory_limit)

    def validate_before_import(self, result: FrameworkBootstrapResult) -> None:
        return None

    def apply_pre_import(self, result: FrameworkBootstrapResult, *, environ: dict[str, str] | None = None) -> None:
        target = os.environ if environ is None else environ
        target.update(result.env_updates)
        if result.cpu_affinity is not None and environ is None and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(result.cpu_affinity))
        if result.memory_limit is not None and environ is None:
            resource.setrlimit(resource.RLIMIT_AS, (result.memory_limit, result.memory_limit))

    def apply_post_import(self, result: FrameworkBootstrapResult) -> None:
        return None


class _LazyFrameworkAdapter:
    name = "framework"
    module_name = "framework"

    def build_plan(self, runtime_spec: RuntimeContextSpec, allocation_view: RuntimeAllocationView | Any, visibility_plan: DeviceVisibilityPlan) -> FrameworkBootstrapResult:
        config = runtime_spec.frameworks.get(self.name, {})
        threads = config.get("num_threads")
        return FrameworkBootstrapResult(post_import_threads={self.name: int(threads)} if threads else {})

    def validate_before_import(self, result: FrameworkBootstrapResult) -> None:
        if self.module_name in sys.modules:
            raise FrameworkImportSafetyError(
                "framework was already imported before runtime bootstrap",
                context={"framework": self.name, "fix": "apply runtime bootstrap before importing framework modules"},
            )

    def apply_pre_import(self, result: FrameworkBootstrapResult, *, environ: dict[str, str] | None = None) -> None:
        return None

    def apply_post_import(self, result: FrameworkBootstrapResult) -> None:
        module = sys.modules.get(self.module_name)
        if module is not None and self.name in result.post_import_threads and hasattr(module, "set_num_threads"):
            module.set_num_threads(result.post_import_threads[self.name])


class TorchBootstrapAdapter(_LazyFrameworkAdapter):
    """Minimal torch adapter that validates pre-import ordering."""

    name = "torch"
    module_name = "torch"


class TensorFlowBootstrapAdapter(_LazyFrameworkAdapter):
    """Minimal TensorFlow adapter that validates pre-import ordering."""


    name = "tensorflow"
    module_name = "tensorflow"


class JaxBootstrapAdapter(_LazyFrameworkAdapter):
    """Placeholder JAX adapter that validates pre-import ordering."""

    name = "jax"
    module_name = "jax"


def default_adapters() -> dict[str, FrameworkBootstrapAdapter]:
    """Return lightweight default bootstrap adapters."""

    adapters: list[FrameworkBootstrapAdapter] = [PlainBootstrapAdapter(), TorchBootstrapAdapter(), TensorFlowBootstrapAdapter(), JaxBootstrapAdapter()]
    return {adapter.name: adapter for adapter in adapters}


def _cpu_affinity_from_config(config: Mapping[str, Any], allocation_view: RuntimeAllocationView | Any) -> tuple[int, ...] | None:
    if "cpu_affinity" in config:
        affinity = config["cpu_affinity"]
        if affinity is None:
            return None
        return tuple(int(cpu) for cpu in affinity)
    if config.get("set_cpu_affinity"):
        cpus = tuple(int(cpu) for cpu in getattr(allocation_view, "cpus", ()))
        return cpus or None
    return None


def _memory_limit_from_config(limits: Mapping[str, Any], config: Mapping[str, Any]) -> int | None:
    value = config.get("memory_limit", limits.get("memory"))
    if value is None:
        return None
    return _parse_byte_size(value)


def _parse_byte_size(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("memory limit must not be bool")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("memory limit must be >= 0")
        return value
    if not isinstance(value, str):
        raise ValueError("memory limit must be int or string")
    for suffix, factor in (("GiB", 1024**3), ("MiB", 1024**2), ("B", 1)):
        if value.endswith(suffix):
            number = value[: -len(suffix)]
            if not number.isdigit():
                break
            return int(number) * factor
    raise ValueError("memory limit must use B, MiB, or GiB units")


__all__ = [
    "FrameworkBootstrapAdapter",
    "FrameworkBootstrapResult",
    "JaxBootstrapAdapter",
    "PlainBootstrapAdapter",
    "TensorFlowBootstrapAdapter",
    "TorchBootstrapAdapter",
    "default_adapters",
]
