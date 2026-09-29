"""Dependency-light adapter registration for watched optional frameworks."""

from __future__ import annotations

import importlib
import os
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from dryml._framework_imports import coordinator, finder

from .context import publication
from .enforcement import ControlStatus


@dataclass(frozen=True, slots=True)
class FrameworkCapabilities:
    """Declare adapter controls and whether changed observed controls are safe.

    The defaults require mandatory visibility and conservatively reject control
    changes after a root has been observed.
    """

    visibility: str = "mandatory"
    threading: str = "best-effort"
    allocator: str = "best-effort"
    process_memory: str = "declarative"
    accelerator_memory: str = "best-effort"
    safe_after_observation: bool = False


@dataclass(frozen=True, slots=True)
class FrameworkRegistration:
    """Describe one logical framework group without importing its dependency.

    Args:
        name: Unique logical group name.
        roots: Non-empty exact import roots owned by the group.
        factory: ``"module:attribute"`` adapter path or a dependency-light
            prebuilt adapter. Direct callable factories are rejected.
        capabilities: Immutable adapter lifecycle declarations.
    """

    name: str
    roots: tuple[str, ...]
    factory: str | Any
    capabilities: FrameworkCapabilities = field(default_factory=FrameworkCapabilities)

    def __post_init__(self) -> None:
        """Normalize roots and reject malformed public registration values."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("framework registration name must be non-empty")
        roots = tuple(self.roots)
        if not roots or len(roots) != len(set(roots)):
            raise ValueError("framework registration requires non-empty unique roots")
        if callable(self.factory):
            raise ValueError("framework registration rejects direct callable factories")
        object.__setattr__(self, "roots", roots)


@dataclass(frozen=True, slots=True)
class FrameworkImportPlan:
    """Immutable adapter controls retained through post-import finalization."""

    env_updates: Mapping[str, str] = field(default_factory=dict)
    visible_devices: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    threads: int | None = None
    interop_threads: int | None = None
    accelerator_memory: Mapping[str, Mapping[str | int, int]] = field(default_factory=dict)
    accelerator_capacity: Mapping[str, Mapping[str | int, int]] = field(default_factory=dict)
    allocator_policy: str | None = None
    process_memory: int | None = None

    def __post_init__(self) -> None:
        """Validate and deeply freeze controls before loader callbacks run."""
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in self.env_updates.items()):
            raise ValueError("framework plan environment updates must be strings")
        if any(value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0) for value in (self.threads, self.interop_threads)):
            raise ValueError("framework thread controls must be positive integers")
        if self.process_memory is not None and (isinstance(self.process_memory, bool) or not isinstance(self.process_memory, int) or self.process_memory < 0):
            raise ValueError("framework process memory must be a non-negative integer")
        object.__setattr__(self, "env_updates", MappingProxyType(dict(self.env_updates)))
        object.__setattr__(self, "visible_devices", MappingProxyType({str(kind): tuple(str(device) for device in devices) for kind, devices in self.visible_devices.items()}))
        object.__setattr__(self, "accelerator_memory", _freeze_device_values(self.accelerator_memory))
        object.__setattr__(self, "accelerator_capacity", _freeze_device_values(self.accelerator_capacity))


@dataclass(frozen=True, slots=True)
class FrameworkPostResult:
    """Independent post-import status outcomes reported by one adapter."""

    statuses: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the U5 closed status vocabulary and freeze the projection."""
        values = {str(key): getattr(value, "value", value) for key, value in self.statuses.items()}
        allowed = {item.value for item in ControlStatus}
        if any(value not in allowed for value in values.values()):
            raise ValueError("framework post statuses use the closed runtime vocabulary")
        object.__setattr__(self, "statuses", MappingProxyType(values))


def _freeze_device_values(values: Mapping[str, Mapping[str | int, int]]) -> Mapping[str, Mapping[str, int]]:
    """Canonicalize and freeze per-device integer values for plan comparison."""
    groups: dict[str, Mapping[str, int]] = {}
    for kind, device_values in values.items():
        normalized: dict[str, int] = {}
        for device, value in device_values.items():
            token = str(device)
            if token in normalized or isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("framework device values require unique positive integer entries")
            normalized[token] = value
        groups[str(kind)] = MappingProxyType(normalized)
    return MappingProxyType(groups)


def _overlap(left: str, right: str) -> bool:
    """Return whether module roots are equal or in an ancestor relation."""
    return left == right or left.startswith(right + ".") or right.startswith(left + ".")


class FrameworkRegistry:
    """PID-safe linearizable registry of watched framework adapter groups."""

    def __init__(self, *, finder_instance=None, finder=None) -> None:
        """Create an empty registry bound to the supplied passive finder.

        Args:
            finder_instance: Optional finder injection for deterministic tests.
        """
        if finder_instance is not None and finder is not None:
            raise TypeError("supply only one framework finder")
        self._finder = globals()["finder"] if finder_instance is None and finder is None else (finder_instance if finder_instance is not None else finder)
        self._registrations: dict[str, FrameworkRegistration] = {}
        self._frozen = False
        self._lock = threading.RLock()
        self._pid = os.getpid()

    def _check_pid(self) -> None:
        """Run U5's PID check before acquiring registry state locks."""
        publication._check_pid()
        if os.getpid() != self._pid:
            raise RuntimeError("framework registry state was inherited after fork; use spawn or a fresh interpreter")

    @property
    def frozen(self) -> bool:
        """Return whether registrations are permanently closed."""
        self._check_pid()
        with self._lock:
            return self._frozen

    def registrations(self) -> Mapping[str, FrameworkRegistration]:
        """Return a detached immutable mapping of registered groups."""
        self._check_pid()
        with self._lock:
            return MappingProxyType(dict(self._registrations))

    def register(self, registration: FrameworkRegistration, *, builtin: bool = False) -> None:
        """Register an adapter group before observation, loading, or freeze.

        Args:
            registration: Immutable group metadata to reserve.
            builtin: Permit roots reserved by base ``dryml`` startup.

        Raises:
            ValueError: For malformed, duplicate, overlapping, or callable data.
            RuntimeError: If the import contract is observed, loaded, or frozen.
        """
        if not isinstance(registration, FrameworkRegistration):
            raise TypeError("framework registration must be FrameworkRegistration")
        self._check_pid()
        with coordinator.writer():
            self._check_pid()
            with self._lock:
                if self._frozen:
                    raise RuntimeError("framework registry is frozen")
                if registration.name in self._registrations:
                    raise ValueError(f"framework group {registration.name!r} is already registered")
                if any(_overlap(root, old) for item in self._registrations.values() for root in registration.roots for old in item.roots):
                    raise ValueError("framework registrations may not overlap roots")
                if builtin:
                    self._finder.can_register(registration.roots, allow_existing=True)
                else:
                    self._finder.register(registration.roots)
                self._registrations[registration.name] = registration

    def freeze(self) -> None:
        """Close registration once a managed import contract is published."""
        self._check_pid()
        with coordinator.writer():
            self._check_pid()
            with self._lock:
                self._frozen = True

    @contextmanager
    def _publication_guard(self):
        """Freeze registration across one active publication transaction."""
        self._check_pid()
        with coordinator.writer():
            self._check_pid()
            with self._lock:
                changed = not self._frozen
                self._frozen = True
            try:
                yield
            except BaseException:
                if changed:
                    with self._lock:
                        self._frozen = False
                raise

    def observe_root(self, fullname: str) -> None:
        """Freeze registration after a watched root is observed by the finder."""
        del fullname
        self._check_pid()
        with self._lock:
            self._frozen = True

    def resolve(self, fullname: str) -> FrameworkRegistration | None:
        """Return the group owning *fullname*, if any, without importing it."""
        self._check_pid()
        with self._lock:
            matches = [item for item in self._registrations.values() if any(fullname == root or fullname.startswith(root + ".") for root in item.roots)]
            return max(matches, key=lambda item: max(map(len, item.roots))) if matches else None

    def adapter_for(self, registration: FrameworkRegistration) -> Any:
        """Resolve an adapter only after an intercepted root begins importing."""
        self._check_pid()
        factory = registration.factory
        if isinstance(factory, str):
            module, separator, attribute = factory.partition(":")
            if not separator or not module or not attribute:
                raise ValueError("framework factory paths require 'module:attribute'")
            factory = getattr(importlib.import_module(module), attribute)
            return factory() if callable(factory) else factory
        return factory


framework_registry = FrameworkRegistry()
for _registration in (
    FrameworkRegistration("tensorflow", ("tensorflow",), "dryml.tf.runtime:adapter"),
    FrameworkRegistration("torch", ("torch",), "dryml.torch.runtime:adapter"),
    FrameworkRegistration("jax", ("jax", "jaxlib"), "dryml.jax.runtime:adapter"),
):
    framework_registry.register(_registration, builtin=True)
finder.set_observation_callback(framework_registry.observe_root)


__all__ = ["FrameworkCapabilities", "FrameworkImportPlan", "FrameworkPostResult", "FrameworkRegistration", "FrameworkRegistry", "framework_registry"]
