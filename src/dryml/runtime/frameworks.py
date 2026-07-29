"""Lightweight framework bootstrap adapter protocols and adapters."""

from __future__ import annotations

import importlib
import os
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

from dryml._framework_imports import coordinator, finder

try:
    import resource
except ModuleNotFoundError:  # Windows does not provide POSIX resource limits.
    resource = None

from .allocation import RuntimeAllocationView
from .devices import DeviceVisibilityPlan
from .errors import FrameworkImportSafetyError, RuntimeSpecError
from .specs import RuntimeContextSpec


def _freeze_device_values(values: Mapping[str, Mapping[str | int, int]]) -> Mapping[str, Mapping[str, int]]:
    """Canonicalize device identifiers before adapters compare plan controls."""

    groups: dict[str, Mapping[str, int]] = {}
    for kind, device_values in values.items():
        normalized: dict[str, int] = {}
        for device, value in device_values.items():
            token = str(device)
            if token in normalized:
                raise ValueError(f"duplicate canonical device identifier {token!r} for {kind!r}")
            normalized[token] = int(value)
        groups[str(kind)] = MappingProxyType(normalized)
    return MappingProxyType(groups)


@dataclass(frozen=True, slots=True)
class FrameworkBootstrapResult:
    """Framework-specific environment and post-import actions."""

    env_updates: Mapping[str, str] = field(default_factory=dict)
    post_import_threads: Mapping[str, int] = field(default_factory=dict)
    post_import_interop_threads: Mapping[str, int] = field(default_factory=dict, kw_only=True)
    visible_devices: Mapping[str, tuple[str, ...]] = field(default_factory=dict, kw_only=True)
    accelerator_memory: Mapping[str, Mapping[str | int, int]] = field(default_factory=dict, kw_only=True)
    accelerator_capacity: Mapping[str, Mapping[str | int, int]] = field(default_factory=dict, kw_only=True)
    allocator_policy: str | None = field(default=None, kw_only=True)
    process_memory: int | None = field(default=None, kw_only=True)
    cpu_affinity: tuple[int, ...] | None = None
    memory_limit: int | None = None

    def __post_init__(self) -> None:
        """Freeze nested planning facts before loader callbacks retain them."""

        object.__setattr__(self, "env_updates", MappingProxyType({str(key): str(value) for key, value in self.env_updates.items()}))
        object.__setattr__(self, "post_import_threads", MappingProxyType({str(key): int(value) for key, value in self.post_import_threads.items()}))
        object.__setattr__(self, "post_import_interop_threads", MappingProxyType({str(key): int(value) for key, value in self.post_import_interop_threads.items()}))
        object.__setattr__(self, "visible_devices", MappingProxyType({str(key): tuple(str(device) for device in value) for key, value in self.visible_devices.items()}))
        object.__setattr__(self, "accelerator_memory", _freeze_device_values(self.accelerator_memory))
        object.__setattr__(self, "accelerator_capacity", _freeze_device_values(self.accelerator_capacity))


@dataclass(frozen=True, slots=True)
class FrameworkCapabilities:
    """Controls an adapter can honestly report without importing its framework."""

    visibility: str = "mandatory"
    threads: str = "best-effort"
    process_memory: str = "declarative"
    accelerator_memory: str = "best-effort"
    allocator: str = "best-effort"


@dataclass(frozen=True, slots=True)
class FrameworkImportPlan:
    """Immutable group plan carried across the wrapped loader callbacks."""

    group: str
    roots: tuple[str, ...]
    fingerprint: str
    capabilities: FrameworkCapabilities = field(default_factory=FrameworkCapabilities)


@dataclass(frozen=True, slots=True)
class FrameworkPostResult:
    """Module-aware post-import statuses keyed by control name."""

    module: str
    statuses: Mapping[str, str] = field(default_factory=dict)
    diagnostics: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject unverifiable status vocabulary before it reaches publication."""

        allowed = {"pending-import", "visibility-enforced", "framework-configured", "declarative", "unsupported", "failed"}
        statuses = {str(control): str(status) for control, status in self.statuses.items()}
        unknown = set(statuses.values()) - allowed
        if unknown:
            raise ValueError(f"unknown framework control status: {sorted(unknown)!r}")
        object.__setattr__(self, "statuses", MappingProxyType(statuses))
        object.__setattr__(self, "diagnostics", MappingProxyType({str(key): str(value) for key, value in self.diagnostics.items()}))


@dataclass(frozen=True, slots=True)
class FrameworkRegistration:
    """One logical adapter group and its lightweight factory import path."""

    name: str
    roots: tuple[str, ...]
    factory: str | Any
    capabilities: FrameworkCapabilities = field(default_factory=FrameworkCapabilities)


def _roots_overlap(left: str, right: str) -> bool:
    """Return whether two module roots share an ancestor/descendant relation."""

    return left == right or left.startswith(right + ".") or right.startswith(left + ".")


class FrameworkRegistry:
    """Linearizable registry of framework groups and watched roots."""

    def __init__(self) -> None:
        """Initialize an unfrozen registry with no framework groups.

        Returns:
            None.
        """
        self._registrations: dict[str, FrameworkRegistration] = {}
        self._revision = 0
        self._frozen = False
        self._lock = threading.RLock()

    @property
    def revision(self) -> int:
        """Return the monotonic revision for registered framework metadata.

        Returns:
            Current registry revision number.
        """
        with self._lock:
            return self._revision

    @property
    def frozen(self) -> bool:
        """Return whether controlled planning has frozen registration."""

        with self._lock:
            return self._frozen

    def state(self) -> tuple[int, bool]:
        """Return revision and freeze state from one registry snapshot."""

        with self._lock:
            return self._revision, self._frozen

    def registrations(self) -> Mapping[str, FrameworkRegistration]:
        """Return a snapshot of all registered framework groups.

        Returns:
            Mapping from group name to immutable registration metadata.
        """
        with self._lock:
            return dict(self._registrations)

    def register(self, registration: FrameworkRegistration, *, builtin: bool = False) -> None:
        """Register roots before activation or reject unsafe registry mutation."""

        if not registration.roots or len(set(registration.roots)) != len(registration.roots):
            raise ValueError("framework registration requires unique roots")
        if callable(registration.factory):
            raise ValueError("framework factories must be lazy import paths or prebuilt adapters")
        with coordinator.writer():
            with self._lock:
                if self._frozen:
                    raise RuntimeError("framework registry is frozen")
                if registration.name in self._registrations:
                    raise ValueError(f"framework group {registration.name!r} is already registered")
                for existing in self._registrations.values():
                    if any(_roots_overlap(root, other) for root in registration.roots for other in existing.roots):
                        raise ValueError("framework registrations may not overlap roots")
                # ``register`` performs the check and root update under one finder
                # mutex, closing a find-spec-to-module-cache registration race.
                if builtin:
                    if not set(registration.roots).issubset(finder.roots()):
                        raise RuntimeError("built-in framework roots were not registered before interception")
                    finder.can_register(registration.roots, allow_existing=True)
                else:
                    finder.register(registration.roots)
                self._registrations[registration.name] = registration
                self._revision += 1

    def freeze(self) -> bool:
        """Prevent future registration once controlled planning is active."""

        # Registry freezing is a process mutation just like registration.  It
        # must not leapfrog an active loader callback or an observation.
        if coordinator.writer_owner == threading.get_ident():
            with self._lock:
                changed = not self._frozen
                self._frozen = True
            return changed
        with coordinator.writer():
            with self._lock:
                changed = not self._frozen
                self._frozen = True
                return changed

    def unfreeze(self) -> None:
        """Undo this transition's provisional freeze after failed publication."""

        if coordinator.writer_owner == threading.get_ident():
            with self._lock:
                self._frozen = False
            return
        with coordinator.writer():
            with self._lock:
                self._frozen = False

    def registration_for(self, module: str) -> FrameworkRegistration | None:
        """Resolve the registration that owns a module name.

        Args:
            module: Imported module or submodule name.

        Returns:
            Matching registration, or ``None`` when the module is unwatched.
        """
        return self.resolve(module)[0]

    def resolve(self, module: str) -> tuple[FrameworkRegistration | None, int]:
        """Return one registration and its revision from the same lock snapshot."""

        with self._lock:
            matches = [
                registration
                for registration in self._registrations.values()
                if any(module == root or module.startswith(root + ".") for root in registration.roots)
            ]
            if not matches:
                return None, self._revision
            return max(matches, key=lambda registration: max(len(root) for root in registration.roots)), self._revision

    def adapter_for(self, registration: FrameworkRegistration) -> Any:
        """Resolve a lightweight optional adapter only when a root is imported."""

        factory = registration.factory
        if isinstance(factory, str):
            module_name, _, attribute = factory.partition(":")
            if not attribute:
                raise ValueError("framework factory paths require 'module:attribute'")
            factory = getattr(importlib.import_module(module_name), attribute)
        return factory() if callable(factory) else factory


framework_registry = FrameworkRegistry()
for _registration in (
    FrameworkRegistration("tensorflow", ("tensorflow",), "dryml.tf.runtime:adapter"),
    FrameworkRegistration("torch", ("torch",), "dryml.torch.runtime:adapter"),
    FrameworkRegistration("jax", ("jax", "jaxlib"), "dryml.jax.runtime:adapter"),
):
    framework_registry.register(_registration, builtin=True)


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
            if resource is None:
                raise RuntimeSpecError(
                    "process memory limits are unsupported on this platform",
                    context={"memory_limit": result.memory_limit, "platform": os.name},
                )
            resource.setrlimit(resource.RLIMIT_AS, (result.memory_limit, result.memory_limit))

    def apply_post_import(self, result: FrameworkBootstrapResult) -> None:
        return None


class _LazyFrameworkAdapter:
    """Build generic framework controls without importing optional runtimes."""

    name = "framework"
    module_name = "framework"

    def build_plan(self, runtime_spec: RuntimeContextSpec, allocation_view: RuntimeAllocationView | Any, visibility_plan: DeviceVisibilityPlan) -> FrameworkBootstrapResult:
        """Build adapter-local controls from runtime and allocation facts.

        Args:
            runtime_spec: Declared runtime framework configuration.
            allocation_view: Effective process resource assignment.
            visibility_plan: Precomputed process device-visibility policy.

        Returns:
            Immutable controls for this adapter's import lifecycle.
        """
        config = runtime_spec.frameworks.get(self.name, {})
        threads = config.get("num_threads") or (len(getattr(allocation_view, "cpus", ())) or None)
        interop_threads = config.get("num_interop_threads")
        accelerator_memory = getattr(allocation_view, "accelerator_memory", {})
        capacity = getattr(allocation_view, "metadata", {}).get("accelerator_memory_capacity", {})
        return FrameworkBootstrapResult(
            post_import_threads={self.name: int(threads)} if threads else {},
            post_import_interop_threads={self.name: int(interop_threads)} if interop_threads else {},
            visible_devices=visibility_plan.visible_devices,
            accelerator_memory=accelerator_memory,
            accelerator_capacity=capacity,
            allocator_policy=config.get("allocator"),
            process_memory=getattr(allocation_view, "memory", None),
        )

    def validate_before_import(self, result: FrameworkBootstrapResult) -> None:
        """Reject a framework already loaded before controlled import.

        Args:
            result: Planned adapter controls retained for this import.

        Returns:
            None.
        """
        if self.module_name in sys.modules:
            raise FrameworkImportSafetyError(
                "framework was already imported before runtime bootstrap",
                context={"framework": self.name, "fix": "apply runtime bootstrap before importing framework modules"},
            )

    def apply_pre_import(self, result: FrameworkBootstrapResult, *, environ: dict[str, str] | None = None) -> None:
        """Apply generic pre-import controls when this adapter has any.

        Args:
            result: Planned adapter controls.
            environ: Optional environment mapping receiving reversible updates.

        Returns:
            None.
        """
        return None

    def apply_post_import(self, result: FrameworkBootstrapResult) -> None:
        """Apply optional generic thread controls after framework import.

        Args:
            result: Planned adapter controls.

        Returns:
            None.
        """
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

    # The optional-package parents install their DType/TensorSpec and backend
    # hooks here. Their runtime leaves only use the standard library, so this
    # planning path cannot import TensorFlow, PyTorch, JAX, or native runtimes.
    from dryml.jax.runtime import adapter as jax_adapter
    from dryml.tf.runtime import adapter as tensorflow_adapter
    from dryml.torch.runtime import adapter as torch_adapter

    adapters: list[FrameworkBootstrapAdapter] = [PlainBootstrapAdapter(), torch_adapter(), tensorflow_adapter(), jax_adapter()]
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
    "FrameworkCapabilities",
    "FrameworkBootstrapAdapter",
    "FrameworkBootstrapResult",
    "FrameworkImportPlan",
    "FrameworkPostResult",
    "FrameworkRegistration",
    "FrameworkRegistry",
    "JaxBootstrapAdapter",
    "PlainBootstrapAdapter",
    "TensorFlowBootstrapAdapter",
    "TorchBootstrapAdapter",
    "default_adapters",
    "framework_registry",
]
