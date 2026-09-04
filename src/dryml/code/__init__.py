"""Dependency-light generic static code-analysis primitives.

The package exposes dependency-light target, graph, kernel, analysis, bounded
current-thread trace, and in-process probe contracts. It does not establish
product policy, import DRYML product packages, transform, or transport.
"""

from .analysis import AnalysisResult, InvocationOutcome, analyze
from .ast_tools import AccessCollection
from .callable_info import CallableInfo, analyze_callable
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
from .facts import CodeFact, CodeFacts, Diagnostic, FactRecord
from .graph import ProgramGraph
from .kernels import AnalysisKernel, KernelCall, KernelOutcome, TraversalKernel
from .probe import probe
from .trace import trace
from .source import SourceInfo, extract_source, get_source_info
from .targets import (
    CodeTarget,
    CodeTargetInput,
    DescriptorTarget,
    ImportTarget,
    SourceTarget,
    TargetInfo,
)

__all__ = [
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
