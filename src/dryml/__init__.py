import importlib

__version__ = "0.3.0-dev"

_SUBMODULE_EXPORTS = {
    "annotations": "dryml.annotations",
    "context": "dryml.context",
    "core2": "dryml.core2",
    "dispatch": "dryml.dispatch",
    "artifacts": "dryml.artifacts",
    "env": "dryml.env",
    "execute": "dryml.execute",
    "environments": "dryml.environments",
    "formats": "dryml.formats",
    "operations": "dryml.operations",
    "providers": "dryml.providers",
    "reporting": "dryml.reporting",
    "records": "dryml.records",
    "runtime": "dryml.runtime",
    "world": "dryml.world",
    "worlds": "dryml.worlds",
}

_CORE2_EXPORTS = {
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
}


def __getattr__(name):
    if name in _SUBMODULE_EXPORTS:
        module = importlib.import_module(_SUBMODULE_EXPORTS[name])
        globals()[name] = module
        return module
    if name in _CORE2_EXPORTS:
        if name in {"config", "configure", "reset_config", "status"}:
            module = importlib.import_module("dryml.core2.session")
        else:
            module = importlib.import_module("dryml.core2")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'dryml' has no attribute {name!r}")

__all__ = [
    "annotations",
    "context",
    "core2",
    "dispatch",
    "artifacts",
    "env",
    "execute",
    "environments",
    "formats",
    "operations",
    "providers",
    "reporting",
    "records",
    "runtime",
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
]
