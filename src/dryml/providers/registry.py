"""Import-safe provider registry."""

from __future__ import annotations

import importlib
from typing import Any

from .errors import ProviderRegistryError, ProviderValidationError
from .identity import ProviderIdentity, ProviderRef


class ProviderRegistry:
    """Registry for provider refs and lightweight in-process test providers."""

    def __init__(self) -> None:
        self._refs: dict[str, ProviderRef] = {}
        self._instances: dict[str, Any] = {}

    def register_ref(self, ref: ProviderRef) -> ProviderRef:
        """Register an import-path provider ref without importing it."""

        if not isinstance(ref, ProviderRef):
            raise ProviderRegistryError("register_ref requires ProviderRef", context={"type": type(ref).__name__})
        self._reserve_name(ref.name)
        self._refs[ref.name] = ref
        return ref

    def register_instance(self, provider: Any) -> ProviderIdentity:
        """Register an already-created provider for direct unit tests."""

        identity = getattr(provider, "identity", None)
        if not isinstance(identity, ProviderIdentity):
            raise ProviderRegistryError("provider instance must expose ProviderIdentity identity")
        self._reserve_name(identity.name)
        self._instances[identity.name] = provider
        self._refs[identity.name] = ProviderRef(
            identity.name,
            identity.module or provider.__class__.__module__,
            identity.qualname or provider.__class__.__qualname__,
            identity.version,
            identity.capabilities,
            identity.metadata,
        )
        return identity

    def get_ref(self, name: str) -> ProviderRef:
        """Return a registered provider ref by name."""

        try:
            return self._refs[name]
        except KeyError as exc:
            raise ProviderRegistryError("unknown provider", context={"name": name}) from exc

    def list_refs(self) -> tuple[ProviderRef, ...]:
        """Return provider refs in deterministic name order."""

        return tuple(self._refs[name] for name in sorted(self._refs))

    def get_instance(self, name: str) -> Any:
        """Return an in-process test provider instance."""

        try:
            return self._instances[name]
        except KeyError as exc:
            raise ProviderRegistryError("provider instance is not registered", context={"name": name}) from exc

    def load(self, name: str) -> Any:
        """Import and instantiate a provider ref; intended for probe workers."""

        if name in self._instances:
            return self._instances[name]
        return load_provider_ref(self.get_ref(name))

    def _reserve_name(self, name: str) -> None:
        if name in self._refs or name in self._instances:
            raise ProviderRegistryError("provider name is already registered", context={"name": name})


def load_provider_ref(ref: ProviderRef) -> Any:
    """Import and instantiate a provider from a ref."""

    try:
        module = importlib.import_module(ref.module)
        target: Any = module
        for part in ref.qualname.split("."):
            target = getattr(target, part)
        provider = target() if isinstance(target, type) or callable(target) else target
    except Exception as exc:
        raise ProviderRegistryError("provider could not be loaded", context={"name": ref.name, "module": ref.module, "qualname": ref.qualname, "error": str(exc)}) from exc
    identity = getattr(provider, "identity", None)
    if not isinstance(identity, ProviderIdentity):
        raise ProviderValidationError("loaded provider must expose ProviderIdentity identity", context={"name": ref.name})
    return provider


__all__ = ["ProviderRegistry", "load_provider_ref"]
