"""Fresh-process public import contracts documented by Sprint 10."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


HEAVY_FRAMEWORK_TOP_LEVEL_MODULES = frozenset(
    {"jax", "ray", "tensorflow", "torch"}
)
RETIRED_CORE_PACKAGE = "core" + "2"

EXPECTED_DRYML_ALL = (
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
)

EXPECTED_CORE_ALL = (
    "load_object",
    "load_alias",
    "save_object",
    "save_definition",
    "load_definition",
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
    "BatchMode",
    "CompilerInfo",
    "Method",
    "Traits",
    "traits",
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
    "ObjectResultSet",
    "OccurrenceResultSet",
    "QueryCardinalityError",
    "QueryDomainError",
    "QueryError",
    "QueryExplanation",
    "QueryIndexError",
    "QueryPathError",
    "SetMember",
)

EXPECTED_CODE_ALL = (
    "ASTAccessFact",
    "AnnotationFact",
    "CallSiteFact",
    "CallableInfo",
    "CallableFact",
    "CodeAnalysisContext",
    "CodeAnalysisError",
    "CodeAnalysisResult",
    "CodeFact",
    "CodeProbeRequest",
    "CodeProbeResult",
    "CodeTarget",
    "CodeTargetSpec",
    "CompilerInfo",
    "DiagnosticFact",
    "DynamicCallFact",
    "DynamicTracePolicy",
    "DynamicTraceProxyError",
    "FunctionAnalyzer",
    "Method",
    "MethodContractFact",
    "RequirementFact",
    "ShapeFact",
    "SourceFact",
    "SourceInfo",
    "StaticCallFact",
    "SymbolFact",
    "Traits",
    "analyze",
    "analyze_callable",
    "available_analyzers",
    "func_source_extract",
    "get_analyzer",
    "get_source_info",
    "normalize_target",
    "probe_target",
    "register_analyzer",
    "run_probe_request",
    "target_from_callable",
    "target_from_class_attribute",
    "target_from_definition_method",
    "target_from_import_path",
    "target_from_method",
    "traits",
    "trace",
)


def _run_fresh_python(source):
    completed = subprocess.run(
        [sys.executable, "-c", source],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _assert_no_retired_source_directory(source_root):
    retired_path = Path(source_root) / "dryml" / RETIRED_CORE_PACKAGE
    assert not retired_path.exists(), (
        "retired editable-package directory remains on disk: "
        f"{retired_path}"
    )


def test_editable_source_tree_has_no_retired_package_directory():
    _assert_no_retired_source_directory(Path(__file__).resolve().parents[2] / "src")


def test_editable_source_preflight_rejects_ignored_cache_only_directory(
    tmp_path,
):
    retired_path = tmp_path / "dryml" / RETIRED_CORE_PACKAGE / "__pycache__"
    retired_path.mkdir(parents=True)

    with pytest.raises(AssertionError, match="remains on disk"):
        _assert_no_retired_source_directory(tmp_path)


def test_top_level_public_all_is_frozen_and_code_remains_explicit():
    """The top-level manifest stays lazy and does not advertise ``code``."""

    _run_fresh_python(
        "import dryml, sys; "
        f"assert tuple(dryml.__all__) == {EXPECTED_DRYML_ALL!r}; "
        "assert 'code' not in dryml.__all__; "
        "assert 'code' not in dryml.__dict__; "
        "assert 'dryml.code' not in sys.modules; "
        "assert 'dryml.dispatch' not in sys.modules; "
        f"assert not {{name.split('.', 1)[0] for name in sys.modules}} "
        f"& {HEAVY_FRAMEWORK_TOP_LEVEL_MODULES!r}"
    )


def test_session_facade_is_lazy_and_keeps_legacy_core_session_distinct():
    """Loading the facade does not eagerly load dispatch or framework runtimes."""

    _run_fresh_python(
        "import dryml, sys; "
        "assert 'dryml.session' not in sys.modules; "
        "facade = dryml.session; "
        "assert facade is not dryml.configure; "
        "assert callable(facade.current); "
        "assert 'dryml.dispatch' not in sys.modules; "
        f"assert not {{name.split('.', 1)[0] for name in sys.modules}} "
        f"& {HEAVY_FRAMEWORK_TOP_LEVEL_MODULES!r}"
    )


def test_core_route_is_lazy_stable_and_has_no_obsolete_alias():
    """The permanent package is unique and the removed route stays absent."""

    _run_fresh_python(
        "import importlib, importlib.util, dryml, sys\n"
        f"obsolete = {RETIRED_CORE_PACKAGE!r}\n"
        "obsolete_path = 'dryml.' + obsolete\n"
        "assert 'dryml.core' not in sys.modules\n"
        "assert obsolete not in dryml.__dict__\n"
        "assert importlib.util.find_spec(obsolete_path) is None\n"
        "core = importlib.import_module('dryml.core')\n"
        "assert dryml.core is core\n"
        "assert dryml.Definition is core.Definition\n"
        f"assert tuple(core.__all__) == {EXPECTED_CORE_ALL!r}\n"
        "for name in core.__all__:\n"
        "    getattr(core, name)\n"
        "try:\n"
        "    importlib.import_module(obsolete_path)\n"
        "except ModuleNotFoundError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError(\n"
        "        'obsolete core package remained importable'\n"
        "    )\n"
        "assert obsolete_path not in sys.modules\n"
    )


def test_explicit_code_import_has_public_all_and_no_dispatch_coupling():
    """The explicit import remains lightweight and self-describing."""

    _run_fresh_python(
        "import dryml, sys; "
        "assert 'dryml.code' not in sys.modules; "
        "import dryml.code as code; "
        f"assert tuple(code.__all__) == {EXPECTED_CODE_ALL!r}; "
        "assert 'dryml.dispatch' not in sys.modules; "
        f"assert not {{name.split('.', 1)[0] for name in sys.modules}} "
        f"& {HEAVY_FRAMEWORK_TOP_LEVEL_MODULES!r}"
    )


def test_retired_ray_tune_plugin_import_is_lightweight():
    """The shipped package exports its plugin without importing Ray."""

    _run_fresh_python(
        "import sys\n"
        "import dryml.ray as ray_plugin\n"
        "assert tuple(ray_plugin.__all__) == ('tune',)\n"
        "from dryml.ray import *\n"
        "assert tune is ray_plugin.tune\n"
        "assert 'ray' not in sys.modules\n"
        "try:\n"
        "    tune.Tune2Trainer()\n"
        "except NotImplementedError as exc:\n"
        "    assert 'dryml.SearchSpace' in str(exc)\n"
        "else:\n"
        "    raise AssertionError('retired Ray Tune adapter was constructible')\n"
    )


@pytest.mark.parametrize("package", ("tf", "torch", "jax"))
def test_optional_adapter_leaves_keep_parent_installation_lightweight(package):
    """Adapter leaves preserve optional-parent registration without heavy imports."""

    _run_fresh_python(
        "import importlib, sys\n"
        "from dryml.core.backend import Backend, backend_existence_testers, backend_testers\n"
        "from dryml.core.dtype import DType\n"
        "from dryml.core.tensor_spec import TensorSpec\n"
        f"package = {package!r}\n"
        "leaf = importlib.import_module('dryml.' + package + '.runtime')\n"
        "parent = importlib.import_module('dryml.' + package)\n"
        "assert leaf.adapter().name == {'tf': 'tensorflow', 'torch': 'torch', 'jax': 'jax'}[package]\n"
        "assert hasattr(DType, package) and hasattr(TensorSpec, package)\n"
        "backend = getattr(Backend, package)\n"
        "assert backend in backend_testers and backend in backend_existence_testers\n"
        "assert importlib.import_module('dryml.' + package + '.runtime') is leaf\n"
        f"assert not {{name.split('.', 1)[0] for name in sys.modules}} & {HEAVY_FRAMEWORK_TOP_LEVEL_MODULES!r}\n"
    )
