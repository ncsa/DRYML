import importlib

from ._framework_imports import install_builtin_roots, install_passive_finder


install_builtin_roots()
install_passive_finder()

__version__ = "0.3.0.dev0"

_SUBMODULE_EXPORTS = {
    "annotations": "dryml.annotations",
    "core": "dryml.core",
    "dispatch": "dryml.dispatch",
    "artifacts": "dryml.artifacts",
    "env": "dryml.env",
    "environments": "dryml.environments",
    "formats": "dryml.formats",
    "managed": "dryml.managed",
    "metrics": "dryml.metrics",
    "operations": "dryml.operations",
    "providers": "dryml.providers",
    "reporting": "dryml.reporting",
    "records": "dryml.records",
    "runtime": "dryml.runtime",
    "session": "dryml.session",
    "world": "dryml.world",
    "worlds": "dryml.worlds",
}

_CORE_EXPORTS = {
    "config",
    "configure",
    "reset_config",
    "status",
    "freeze",
    "Definition",
    "ConcreteDefinition",
    "Ref",
    "Mat",
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
    "definition_mode",
    "selector_mode",
    "space_mode",
    "save_definition",
    "load_definition",
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
    "annotations",
    "core",
    "dispatch",
    "artifacts",
    "env",
    "environments",
    "formats",
    "managed",
    "metrics",
    "operations",
    "providers",
    "reporting",
    "records",
    "runtime",
    "session",
    "world",
    "worlds",
    "config",
    "configure",
    "reset_config",
    "status",
    "freeze",
    "Definition",
    "ConcreteDefinition",
    "Ref",
    "Mat",
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
    "definition_mode",
    "selector_mode",
    "space_mode",
    "save_definition",
    "load_definition",
]
