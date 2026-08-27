import importlib

__version__ = "0.3.0-dev"

_SUBMODULE_EXPORTS = {
    "context": "dryml.context",
    "core": "dryml.core",
    "artifacts": "dryml.artifacts",
    "execute": "dryml.execute",
    "environments": "dryml.environments",
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
    "environments",
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
    "SKIP_ARGS",
    "definition_mode",
    "selector_mode",
    "space_mode",
]
