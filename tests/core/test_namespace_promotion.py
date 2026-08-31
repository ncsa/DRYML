"""Characterize the public core namespace after the promotion."""

import importlib
import os
from pathlib import Path
import subprocess
import sys


def _run_fresh_process(code: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` with this checkout's source tree first on ``PYTHONPATH``."""

    src = Path(__file__).resolve().parents[2] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(src), env.get("PYTHONPATH", "")))
    return subprocess.run(
        (sys.executable, "-c", code),
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )


def test_root_exports_core_lazily_with_the_existing_public_symbols():
    """The lazy root router exposes the promoted package and its existing symbols."""

    _run_fresh_process(
        """
import sys
import dryml

assert "core" in dryml.__all__
assert "dryml.core" not in sys.modules
core = dryml.core
assert core is importlib.import_module("dryml.core")
assert dryml.Definition is core.Definition
assert dryml.definition_mode is core.definition_mode
assert "dryml.core" + "2" not in sys.modules
""".replace("import sys", "import importlib\nimport sys")
    )


def test_promoted_package_keeps_the_destination_export_manifest():
    """The package export manifest remains the destination manifest after promotion."""

    core = importlib.import_module("dryml.core")

    assert core.__all__ == [
        "load_object", "save_object", "load_state_ref", "LiveReusePolicy", "StoreReport", "Object", "Serializable", "UniqueID",
        "Metadata", "Compute", "Definition", "ConcreteDefinition", "DefLink", "Ref", "Mat",
        "ObjectId", "ObjectRef", "StateRef", "StateSelectorRef", "object_namespace", "freeze",
        "ArgRole", "RefCDef", "RefCDefArg", "SelectorArg", "MaterializeArg",
        "ValueArg", "QuotedDef", "SelectorSpec", "Selector", "selector", "Par", "Present",
        "Missing", "AnyValue", "Exact", "Choice", "IntRange", "SubclassOf", "Satisfies",
        "UniformIntRange", "UniformFromSet", "SearchSpace", "SKIP_ARGS", "Repo", "configure",
        "reset_config", "status", "definition_mode", "selector_mode", "space_mode", "dtype",
        "DType", "ConfigRef", "FactorySpec", "ConfigError", "CONFIG_MISSING", "as_tensor_spec",
        "SpecHint", "TensorSpec", "ImportRef", "SourceSpec", "symbol_ref", "resolve_symbol",
        "CDefEdge", "CDefNode", "CDefOccurrence", "ConcreteDefinitionGraph",
        "ConcreteDefinitionGraphCycleError", "ConcreteDefinitionGraphError", "EdgeKind",
        "iter_direct_cdef_edges", "Arg", "DefinitionPath", "DefinitionQuery", "DefinitionResultSet",
        "GraphPathError", "Index", "Key", "Kwarg", "Parameter", "ObjectResultSet",
        "ObjectRefResultSet", "OccurrenceResultSet", "ReferenceOccurrence", "ReferenceQuery", "ReferenceResultSet", "QueryCardinalityError", "QueryDomainError", "QueryError",
        "QueryExplanation", "QueryIndexError", "QueryPathError", "SetMember", "StateRefResultSet",
    ]
