"""Fresh-process public import contracts documented by Sprint 10."""

from __future__ import annotations

import subprocess
import sys


HEAVY_FRAMEWORK_TOP_LEVEL_MODULES = frozenset(
    {"jax", "ray", "tensorflow", "torch"}
)

EXPECTED_DRYML_ALL = (
    "annotations",
    "core2",
    "dispatch",
    "artifacts",
    "env",
    "environments",
    "formats",
    "managed",
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
    "save_definition",
    "load_definition",
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
