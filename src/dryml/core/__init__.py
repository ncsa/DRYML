"""Lazy public facade for core definitions, persistence, and query APIs."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType


_EXPORT_MODULES = {
    "Object": "dryml.core.object",
    "Serializable": "dryml.core.object",
    "UniqueID": "dryml.core.object",
    "Metadata": "dryml.core.object",
    "Compute": "dryml.core.object",
    "definition_mode": "dryml.core.object",
    "selector_mode": "dryml.core.object",
    "space_mode": "dryml.core.object",
    "ConcreteDefinition": "dryml.core.definition",
    "Definition": "dryml.core.definition",
    "SKIP_ARGS": "dryml.core.definition",
    "freeze": "dryml.core.definition",
    "ArgRole": "dryml.core.arg_roles",
    "MaterializeArg": "dryml.core.arg_roles",
    "RefCDef": "dryml.core.arg_roles",
    "RefCDefArg": "dryml.core.arg_roles",
    "SelectorArg": "dryml.core.arg_roles",
    "ValueArg": "dryml.core.arg_roles",
    "DefLink": "dryml.core.links",
    "Mat": "dryml.core.links",
    "Ref": "dryml.core.links",
    "ObjectId": "dryml.core.reference_values",
    "ObjectRef": "dryml.core.reference_values",
    "StateRef": "dryml.core.reference_values",
    "StateSelectorRef": "dryml.core.reference_values",
    "object_namespace": "dryml.core.reference_values",
    "AnyValue": "dryml.core.params",
    "Choice": "dryml.core.params",
    "Exact": "dryml.core.params",
    "IntRange": "dryml.core.params",
    "Missing": "dryml.core.params",
    "Par": "dryml.core.params",
    "Present": "dryml.core.params",
    "Satisfies": "dryml.core.params",
    "SubclassOf": "dryml.core.params",
    "UniformFromSet": "dryml.core.params",
    "UniformIntRange": "dryml.core.params",
    "QuotedDef": "dryml.core.quoted",
    "SelectorSpec": "dryml.core.quoted",
    "SearchSpace": "dryml.core.search_space",
    "Selector": "dryml.core.selector",
    "selector": "dryml.core.selector",
    "Repo": "dryml.core.repo",
    "load_object": "dryml.core.repo",
    "load_state_ref": "dryml.core.repo",
    "save_object": "dryml.core.repo",
    "LiveReusePolicy": "dryml.core.policies",
    "StoreReport": "dryml.core.repo_plan",
    "dtype": "dryml.core.dtype",
    "DType": "dryml.core.dtype",
    "SpecHint": "dryml.core.tensor_spec",
    "TensorSpec": "dryml.core.tensor_spec",
    "as_tensor_spec": "dryml.core.tensor_spec",
    "CONFIG_MISSING": "dryml.core.config",
    "ConfigError": "dryml.core.config",
    "ConfigRef": "dryml.core.config",
    "FactorySpec": "dryml.core.factory",
    "configure": "dryml.core.session",
    "reset_config": "dryml.core.session",
    "status": "dryml.core.session",
    "ImportRef": "dryml.core.symbol",
    "SourceSpec": "dryml.core.symbol",
    "resolve_symbol": "dryml.core.symbol",
    "symbol_ref": "dryml.core.symbol",
    "CDefEdge": "dryml.core.cdef_graph",
    "CDefNode": "dryml.core.cdef_graph",
    "CDefOccurrence": "dryml.core.cdef_graph",
    "ConcreteDefinitionGraph": "dryml.core.cdef_graph",
    "ConcreteDefinitionGraphCycleError": "dryml.core.cdef_graph",
    "ConcreteDefinitionGraphError": "dryml.core.cdef_graph",
    "EdgeKind": "dryml.core.cdef_graph",
    "iter_direct_cdef_edges": "dryml.core.cdef_graph",
    "Arg": "dryml.core.query",
    "DefinitionPath": "dryml.core.query",
    "DefinitionQuery": "dryml.core.query",
    "DefinitionResultSet": "dryml.core.query",
    "GraphPathError": "dryml.core.query",
    "Index": "dryml.core.query",
    "Key": "dryml.core.query",
    "Kwarg": "dryml.core.query",
    "Parameter": "dryml.core.query",
    "ObjectResultSet": "dryml.core.query",
    "ObjectRefResultSet": "dryml.core.query",
    "OccurrenceResultSet": "dryml.core.query",
    "ReferenceOccurrence": "dryml.core.query",
    "ReferenceQuery": "dryml.core.query",
    "ReferenceResultSet": "dryml.core.query",
    "QueryCardinalityError": "dryml.core.query",
    "QueryDomainError": "dryml.core.query",
    "QueryError": "dryml.core.query",
    "QueryExplanation": "dryml.core.query",
    "QueryIndexError": "dryml.core.query",
    "QueryPathError": "dryml.core.query",
    "SetMember": "dryml.core.query",
    "StateRefResultSet": "dryml.core.query",
}


def __getattr__(name: str) -> object:
    """Return one documented core export, loading and caching it on first access.

    Args:
        name: Public export name listed in :data:`__all__`.

    Returns:
        The identically owned object from its defining core module.

    Raises:
        AttributeError: If ``name`` is not a documented core export.

    Side Effects:
        Imports the export's owner module and stores the resolved object in this
        module's namespace.
    """

    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value

__all__ = [
    "load_object",
    "save_object",
    "load_state_ref",
    "LiveReusePolicy",
    "StoreReport",
    "Object",
    "Serializable",
    "UniqueID",
    "Metadata",
    "Compute",
    "Definition",
    "ConcreteDefinition",
    "DefLink",
    "Ref",
    "Mat",
    "ObjectId",
    "ObjectRef",
    "StateRef",
    "StateSelectorRef",
    "object_namespace",
    "freeze",
    "ArgRole",
    "RefCDef",
    "RefCDefArg",
    "SelectorArg",
    "MaterializeArg",
    "ValueArg",
    "QuotedDef",
    "SelectorSpec",
    "Selector",
    "selector",
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
    "Repo",
    "configure",
    "reset_config",
    "status",
    "definition_mode",
    "selector_mode",
    "space_mode",
    "dtype",
    "DType",
    "ConfigRef",
    "FactorySpec",
    "ConfigError",
    "CONFIG_MISSING",
    "as_tensor_spec",
    "SpecHint",
    "TensorSpec",
    "ImportRef",
    "SourceSpec",
    "symbol_ref",
    "resolve_symbol",
    "CDefEdge",
    "CDefNode",
    "CDefOccurrence",
    "ConcreteDefinitionGraph",
    "ConcreteDefinitionGraphCycleError",
    "ConcreteDefinitionGraphError",
    "EdgeKind",
    "iter_direct_cdef_edges",
    "Arg",
    "DefinitionPath",
    "DefinitionQuery",
    "DefinitionResultSet",
    "GraphPathError",
    "Index",
    "Key",
    "Kwarg",
    "Parameter",
    "ObjectResultSet",
    "ObjectRefResultSet",
    "OccurrenceResultSet",
    "ReferenceOccurrence",
    "ReferenceQuery",
    "ReferenceResultSet",
    "QueryCardinalityError",
    "QueryDomainError",
    "QueryError",
    "QueryExplanation",
    "QueryIndexError",
    "QueryPathError",
    "SetMember",
    "StateRefResultSet",
]


class _CoreModule(ModuleType):
    """Protect public names that collide with importable core submodules."""

    def __getattribute__(self, name: str) -> object:
        if name in _EXPORT_MODULES:
            value = vars(self).get(name)
            if value is None or isinstance(value, ModuleType):
                return __getattr__(name)
        return super().__getattribute__(name)


sys.modules[__name__].__class__ = _CoreModule
