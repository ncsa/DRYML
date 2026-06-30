import importlib

__version__ = "0.3.0-dev"

_SUBMODULE_EXPORTS = {
    "context": "dryml.context",
    "core2": "dryml.core2",
    "artifacts": "dryml.artifacts",
    "execute": "dryml.execute",
    "environments": "dryml.environments",
}

_CORE2_EXPORTS = {
    "config",
    "configure",
    "reset_config",
    "status",
    "freeze",
    "FrozenCDef",
    "FrozenDef",
    "FrozenCDefArg",
    "FrozenDefArg",
    "FrozenConcreteDefinition",
    "FrozenDefinition",
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
    "context",
    "core2",
    "artifacts",
    "execute",
    "environments",
    "config",
    "configure",
    "reset_config",
    "status",
    "freeze",
    "FrozenCDef",
    "FrozenDef",
    "FrozenCDefArg",
    "FrozenDefArg",
    "FrozenConcreteDefinition",
    "FrozenDefinition",
]
