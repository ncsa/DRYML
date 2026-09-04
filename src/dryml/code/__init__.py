"""Dependency-light generic static code-analysis primitives.

The package exposes dependency-light target, graph, kernel, analysis, bounded
current-thread trace, and in-process probe contracts. It does not establish
product policy, import DRYML product packages, transform, or transport.
"""

from .ast_tools import AccessCollection, AttrAccess, MethodCall, collect_accesses_from_source, parse_source
from .callable_info import CallableInfo, analyze_callable
from .analysis import AnalysisResult, InvocationOutcome, analyze
from .errors import (
    AnalysisErrorCode,
    CodeAnalysisError,
    InvalidKernelError,
    InvalidTargetError,
    KernelDependencyError,
    KernelExecutionError,
    MissingOutputError,
    SourceUnavailableError,
)
from .facts import CodeFact, CodeFacts, Diagnostic, FactRecord, FactScalar, FactValue, SourceLocation
from .graph import ProgramGraph
from .kernels import AnalysisKernel, KernelCall, KernelOutcome, TraversalKernel
from .probe import probe
from .trace import trace
from .source import SourceInfo, extract_source, get_source_info
from .targets import (
    CodeTarget,
    CodeTargetInput,
    DescriptorKind,
    DescriptorTarget,
    ImportTarget,
    SourceTarget,
    TargetInfo,
    TargetKind,
    normalize_target,
)

__all__ = [
    "AccessCollection",
    "AnalysisErrorCode",
    "AnalysisKernel",
    "AnalysisResult",
    "AttrAccess",
    "CallableInfo",
    "CodeAnalysisError",
    "CodeFact",
    "CodeFacts",
    "CodeTarget",
    "CodeTargetInput",
    "DescriptorKind",
    "DescriptorTarget",
    "Diagnostic",
    "FactRecord",
    "FactScalar",
    "FactValue",
    "ImportTarget",
    "InvalidKernelError",
    "InvalidTargetError",
    "InvocationOutcome",
    "KernelCall",
    "KernelDependencyError",
    "KernelExecutionError",
    "KernelOutcome",
    "MethodCall",
    "MissingOutputError",
    "ProgramGraph",
    "SourceInfo",
    "SourceLocation",
    "SourceTarget",
    "SourceUnavailableError",
    "TargetInfo",
    "TargetKind",
    "TraversalKernel",
    "analyze",
    "analyze_callable",
    "collect_accesses_from_source",
    "extract_source",
    "get_source_info",
    "normalize_target",
    "parse_source",
    "probe",
    "trace",
]
