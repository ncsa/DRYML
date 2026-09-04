"""Tests for the exact dependency-light code-analysis package manifest."""

from __future__ import annotations

import subprocess
import sys


_EXPECTED_ROOT_EXPORTS = [
    "AccessCollection",
    "AnalysisErrorCode",
    "AnalysisKernel",
    "AnalysisResult",
    "CallableInfo",
    "CodeAnalysisError",
    "CodeFact",
    "CodeFacts",
    "CodeTarget",
    "CodeTargetInput",
    "Diagnostic",
    "DescriptorTarget",
    "FactRecord",
    "ImportTarget",
    "InvalidKernelError",
    "InvalidTargetError",
    "InvocationOutcome",
    "KernelCall",
    "KernelDependencyError",
    "KernelExecutionError",
    "KernelOutcome",
    "MissingOutputError",
    "ProgramGraph",
    "SourceInfo",
    "SourceTarget",
    "SourceUnavailableError",
    "TargetInfo",
    "TraversalKernel",
    "analyze",
    "analyze_callable",
    "extract_source",
    "get_source_info",
    "probe",
    "trace",
]


def test_code_exports_implemented_analysis_apis() -> None:
    """The source-tree root manifest is the exact documented Stage 3 surface."""

    import dryml.code as code

    assert code.__all__ == _EXPECTED_ROOT_EXPORTS
    for name in (
        "AttrAccess", "MethodCall", "collect_accesses_from_source", "parse_source",
        "ProgramNode", "ProgramEdge", "ProgramNodeKind", "ProgramEdgeKind",
        "build_program_graph", "TargetKind", "DescriptorKind", "normalize_target",
        "KernelContext", "KernelMode", "FactScalar", "FactValue", "SourceLocation",
        "LexicalDependency", "LexicalDependencies", "LexicalDependencyKernel",
        "collect_lexical_dependencies", "func_source_extract", "CompilerInfo",
        "Method", "Traits", "traits", "register_analyzer", "get_analyzer",
        "available_analyzers", "CodeProbeRequest", "CodeProbeResult",
    ):
        assert not hasattr(code, name)


def test_fresh_code_import_loads_no_product_or_optional_packages() -> None:
    """Importing the public package performs no analysis or product imports."""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import dryml.code; "
            "print(','.join(sorted(name for name in sys.modules if name.startswith('dryml.'))))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    loaded = set(filter(None, result.stdout.strip().split(",")))
    assert loaded <= {
        "dryml.code",
        "dryml.code.ast_tools",
        "dryml.code.callable_info",
        "dryml.code.errors",
        "dryml.code.facts",
        "dryml.code.graph",
        "dryml.code.analysis",
        "dryml.code.kernels",
        "dryml.code.probe",
        "dryml.code.source",
        "dryml.code.targets",
        "dryml.code.trace",
        "dryml._framework_imports",
    }
    assert not any(name.startswith(("dryml.core", "dryml.environments", "dryml.worlds")) for name in loaded)
