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
    "env": "dryml.environments",
    "environments": "dryml.environments",
    "requirements": "dryml.requirements",
    "worlds": "dryml.worlds",
    "runtime": "dryml.runtime",
    "world": "dryml.worlds",
    "session": "dryml.session",
    "annotations": "dryml.annotations",
    "methods": "dryml.methods",
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
    "ObjectId",
    "ObjectRef",
    "StateRef",
    "StateSelectorRef",
    "StoreReport",
    "load_object",
    "load_state_ref",
    "save_object",
    "object_namespace",
    "RefCDef",
    "RefCDefArg",
    "Selector",
    "SelectorArg",
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


def __getattr__(name):
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
    "env",
    "environments",
    "requirements",
    "worlds",
    "runtime",
    "world",
    "session",
    "annotations",
    "methods",
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
    "ObjectId",
    "ObjectRef",
    "StateRef",
    "StateSelectorRef",
    "StoreReport",
    "load_object",
    "load_state_ref",
    "save_object",
    "object_namespace",
    "RefCDef",
    "RefCDefArg",
    "Selector",
    "SelectorArg",
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
