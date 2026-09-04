"""Verify that core's public facade defers heavyweight implementation modules."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest


_EXPECTED_CORE_EXPORTS = (
    "load_object", "save_object", "load_state_ref", "LiveReusePolicy",
    "StoreReport", "Object", "Serializable", "UniqueID", "Metadata",
    "Compute", "Definition", "ConcreteDefinition", "DefLink", "Ref", "Mat",
    "ObjectId", "ObjectRef", "StateRef", "StateSelectorRef", "object_namespace",
    "freeze", "ArgRole", "RefCDef", "RefCDefArg", "SelectorArg",
    "MaterializeArg", "ValueArg", "QuotedDef", "SelectorSpec", "Selector",
    "selector", "Par", "Present", "Missing", "AnyValue", "Exact", "Choice",
    "IntRange", "SubclassOf", "Satisfies", "UniformIntRange", "UniformFromSet",
    "SearchSpace", "SKIP_ARGS", "Repo", "configure", "reset_config", "status",
    "definition_mode", "selector_mode", "space_mode", "dtype", "DType",
    "ConfigRef", "FactorySpec", "ConfigError", "CONFIG_MISSING", "as_tensor_spec",
    "SpecHint", "TensorSpec", "ImportRef", "SourceSpec", "symbol_ref",
    "resolve_symbol", "CDefEdge", "CDefNode", "CDefOccurrence",
    "ConcreteDefinitionGraph", "ConcreteDefinitionGraphCycleError",
    "ConcreteDefinitionGraphError", "EdgeKind", "iter_direct_cdef_edges", "Arg",
    "DefinitionPath", "DefinitionQuery", "DefinitionResultSet", "GraphPathError",
    "Index", "Key", "Kwarg", "Parameter", "ObjectResultSet",
    "ObjectRefResultSet", "OccurrenceResultSet", "ReferenceOccurrence",
    "ReferenceQuery", "ReferenceResultSet", "QueryCardinalityError", "QueryDomainError",
    "QueryError", "QueryExplanation", "QueryIndexError", "QueryPathError", "SetMember",
    "StateRefResultSet",
)

_EXPORT_MODULES = {
    **dict.fromkeys(("Object", "Serializable", "UniqueID", "Metadata", "Compute", "definition_mode", "selector_mode", "space_mode"), "dryml.core.object"),
    **dict.fromkeys(("ConcreteDefinition", "Definition", "SKIP_ARGS", "freeze"), "dryml.core.definition"),
    **dict.fromkeys(("ArgRole", "MaterializeArg", "RefCDef", "RefCDefArg", "SelectorArg", "ValueArg"), "dryml.core.arg_roles"),
    **dict.fromkeys(("DefLink", "Mat", "Ref"), "dryml.core.links"),
    **dict.fromkeys(("ObjectId", "ObjectRef", "StateRef", "StateSelectorRef", "object_namespace"), "dryml.core.reference_values"),
    **dict.fromkeys(("AnyValue", "Choice", "Exact", "IntRange", "Missing", "Par", "Present", "Satisfies", "SubclassOf", "UniformFromSet", "UniformIntRange"), "dryml.core.params"),
    **dict.fromkeys(("QuotedDef", "SelectorSpec"), "dryml.core.quoted"),
    "SearchSpace": "dryml.core.search_space",
    **dict.fromkeys(("Selector", "selector"), "dryml.core.selector"),
    **dict.fromkeys(("Repo", "load_object", "load_state_ref", "save_object"), "dryml.core.repo"),
    "LiveReusePolicy": "dryml.core.policies",
    "StoreReport": "dryml.core.repo_plan",
    **dict.fromkeys(("dtype", "DType"), "dryml.core.dtype"),
    **dict.fromkeys(("SpecHint", "TensorSpec", "as_tensor_spec"), "dryml.core.tensor_spec"),
    **dict.fromkeys(("CONFIG_MISSING", "ConfigError", "ConfigRef"), "dryml.core.config"),
    "FactorySpec": "dryml.core.factory",
    **dict.fromkeys(("configure", "reset_config", "status"), "dryml.core.session"),
    **dict.fromkeys(("ImportRef", "SourceSpec", "resolve_symbol", "symbol_ref"), "dryml.core.symbol"),
    **dict.fromkeys(("CDefEdge", "CDefNode", "CDefOccurrence", "ConcreteDefinitionGraph", "ConcreteDefinitionGraphCycleError", "ConcreteDefinitionGraphError", "EdgeKind", "iter_direct_cdef_edges"), "dryml.core.cdef_graph"),
    **dict.fromkeys(("Arg", "DefinitionPath", "DefinitionQuery", "DefinitionResultSet", "GraphPathError", "Index", "Key", "Kwarg", "Parameter", "ObjectResultSet", "ObjectRefResultSet", "OccurrenceResultSet", "ReferenceOccurrence", "ReferenceQuery", "ReferenceResultSet", "QueryCardinalityError", "QueryDomainError", "QueryError", "QueryExplanation", "QueryIndexError", "QueryPathError", "SetMember", "StateRefResultSet"), "dryml.core.query"),
}


def _run_import_probe(code: str) -> subprocess.CompletedProcess[str]:
    """Run an isolated source-tree import probe in a fresh Python process."""

    source = Path(__file__).resolve().parents[2] / "src"
    environment = os.environ | {"PYTHONPATH": os.pathsep.join((str(source), os.environ.get("PYTHONPATH", "")))}
    return subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=environment,
        text=True,
        capture_output=True,
    )


@pytest.mark.parametrize("module_name", ("dryml.core.object", "dryml.core.tensor_spec"))
def test_narrow_core_module_imports_do_not_load_heavy_packages(module_name: str) -> None:
    """Narrow core modules load without persistence, runtime, consumer, or backend imports."""

    _run_import_probe(
        f"""
import importlib
import sys

importlib.import_module({module_name!r})
forbidden_prefixes = (
    "dryml.core.query", "dryml.core.repo", "dryml.core.repo_plan", "dryml.core.session",
    "dryml.core.store", "dryml.artifacts", "dryml.code", "dryml.data", "dryml.dispatch",
    "dryml.environments", "dryml.execute", "dryml.managed", "dryml.models", "dryml.runtime",
    "dryml.session", "dryml.worlds", "tensorflow", "torch", "jax", "jaxlib", "ray",
)
loaded = [name for name in sys.modules if name.startswith(forbidden_prefixes)]
assert not loaded, loaded
"""
    )


def test_core_exports_resolve_from_owning_modules_and_cache() -> None:
    """Every supported facade export retains direct-import identity and is cached."""

    core = importlib.import_module("dryml.core")
    assert tuple(core.__all__) == _EXPECTED_CORE_EXPORTS
    assert set(_EXPORT_MODULES) == set(_EXPECTED_CORE_EXPORTS)

    for name in _EXPECTED_CORE_EXPORTS:
        direct = getattr(importlib.import_module(_EXPORT_MODULES[name]), name)
        resolved = getattr(core, name)
        assert resolved is direct
        assert core.__dict__[name] is direct


def test_core_missing_names_and_star_import_preserve_python_import_behavior() -> None:
    """Unknown names fail normally and star imports expose exactly the public manifest."""

    core = importlib.import_module("dryml.core")
    with pytest.raises(AttributeError, match="has no attribute 'missing_core_export'"):
        getattr(core, "missing_core_export")
    with pytest.raises(ImportError, match="cannot import name 'missing_core_export'"):
        exec("from dryml.core import missing_core_export", {})

    namespace: dict[str, object] = {}
    exec("from dryml.core import *", namespace)
    assert {name for name in namespace if name != "__builtins__"} == set(_EXPECTED_CORE_EXPORTS)


def test_core_and_symbol_imports_keep_code_lazy_until_source_capture() -> None:
    """Core imports stay passive; only source capture loads the lexical leaf."""

    _run_import_probe(
        """
import sys
import dryml.core
import dryml.core.symbol
assert not any(name == "dryml.code" or name.startswith("dryml.code.") for name in sys.modules)
"""
    )

    symbol_source = (Path(__file__).resolve().parents[2] / "src" / "dryml" / "core" / "symbol.py").read_text(encoding="utf-8")
    assert "from dryml.code.targets" not in symbol_source
    assert "_collect_source_dependencies" in symbol_source
    _run_import_probe(
        """
import sys
import dryml.core.symbol as symbol

symbol._collect_source_imports(object(), "def local(value):\\n    return value + 1", None)
loaded = {name for name in sys.modules if name.startswith("dryml.")}
allowed = {"dryml", "dryml._framework_imports", "dryml.core", "dryml.core.symbol"}
assert all(name in allowed or name.startswith("dryml.code") for name in loaded), loaded
assert "dryml.code.algorithms.lexical_dependencies" in loaded
forbidden = ("dryml.data", "dryml.environments", "dryml.execute", "dryml.runtime", "dryml.worlds", "tensorflow", "torch", "jax", "ray")
assert not [name for name in sys.modules if name.startswith(forbidden)]
"""
    )
