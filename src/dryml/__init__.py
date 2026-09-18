"""Dependency-light root exports for DRYML declarations, core identities, and packages.

Core conveniences and public package modules, including :mod:`dryml.methods`,
resolve lazily through ``__getattr__`` so importing :mod:`dryml` does not load
their implementation, runtime state, or optional frameworks.
"""

import importlib

from ._framework_imports import install_builtin_roots, install_passive_finder

install_builtin_roots()
install_passive_finder()

__version__ = "0.3.0.dev2"

_SUBMODULE_EXPORTS = {
    "context": "dryml.context",
    "core": "dryml.core",
    "artifacts": "dryml.artifacts",
    "execute": "dryml.execute",
    "dispatch": "dryml.dispatch",
    "env": "dryml.environments",
    "environments": "dryml.environments",
    "requirements": "dryml.requirements",
    "worlds": "dryml.worlds",
    "runtime": "dryml.runtime",
    "world": "dryml.worlds",
    "session": "dryml.session",
    "annotations": "dryml.annotations",
    "methods": "dryml.methods",
    "managed": "dryml.managed",
    "locking": "dryml.locking",
}

_CORE_EXPORTS = {
    "config",
    "configure",
    "reset_config",
    "status",
    "freeze",
    "Definition",
    "ConcreteDefinition",
    "Object",
    "Serializable",
    "Repo",
    "Ref",
    "Mat",
    "AutoRef",
    "normalize_args",
    "normalize_return",
    "signature_context",
    "function",
    "SignatureError",
    "ObjectId",
    "ObjectRef",
    "StateRef",
    "StateSelectorRef",
    "StoreReport",
    "load_object",
    "load_state_ref",
    "save_object",
    "object_namespace",
    "Selector",
    "SelectorSpec",
    "QuotedDef",
    "Par",
    "Present",
    "Missing",
    "AnyValue",
    "Exact",
    "Choice",
    "IntRange",
    "SubclassOf",
    "Satisfies",
    "UniformIntRange",
    "UniformFromSet",
    "SearchSpace",
    "SKIP_ARGS",
    "definition_mode",
    "selector_mode",
    "space_mode",
}


def __getattr__(name: str) -> object:
    """Resolve one documented lazy root export.

    Args:
        name: A name listed in :data:`__all__`.

    Returns:
        The requested public package module or core-owned convenience value.

    Raises:
        AttributeError: If ``name`` is not a documented root export.

    Side Effects:
        Imports the owning lightweight package or core module on first access and
        caches the resolved value in this module. Importing :mod:`dryml` alone
        does not resolve these exports.
    """

    if name in _SUBMODULE_EXPORTS:
        module = importlib.import_module(_SUBMODULE_EXPORTS[name])
        globals()[name] = module
        return module
    if name in _CORE_EXPORTS:
        if name in {"config", "configure", "reset_config", "status"}:
            module = importlib.import_module("dryml.core.session")
        else:
            module = importlib.import_module("dryml.core")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'dryml' has no attribute {name!r}")

__all__ = [
    "context",
    "core",
    "artifacts",
    "execute",
    "dispatch",
    "env",
    "environments",
    "requirements",
    "worlds",
    "runtime",
    "world",
    "session",
    "annotations",
    "methods",
    "managed",
    "locking",
    "config",
    "configure",
    "reset_config",
    "status",
    "freeze",
    "Definition",
    "ConcreteDefinition",
    "Object",
    "Serializable",
    "Repo",
    "Ref",
    "Mat",
    "AutoRef",
    "normalize_args",
    "normalize_return",
    "signature_context",
    "function",
    "SignatureError",
    "ObjectId",
    "ObjectRef",
    "StateRef",
    "StateSelectorRef",
    "StoreReport",
    "load_object",
    "load_state_ref",
    "save_object",
    "object_namespace",
    "Selector",
    "SelectorSpec",
    "QuotedDef",
    "Par",
    "Present",
    "Missing",
    "AnyValue",
    "Exact",
    "Choice",
    "IntRange",
    "SubclassOf",
    "Satisfies",
    "UniformIntRange",
    "UniformFromSet",
    "SearchSpace",
    "SKIP_ARGS",
    "definition_mode",
    "selector_mode",
    "space_mode",
]
