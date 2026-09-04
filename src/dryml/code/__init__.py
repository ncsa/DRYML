"""Dependency-light generic static code-analysis primitives.

The package exposes U1 target, source, AST, fact, error, and foundational graph
contracts. It does not execute targets, establish product policy, import DRYML
product packages, or expose future kernel, trace, transformation, or transport
APIs.
"""

from .ast_tools import AccessCollection, AttrAccess, MethodCall, collect_accesses_from_source, parse_source
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
from .facts import CodeFact, CodeFacts, Diagnostic, FactRecord, FactScalar, FactValue, SourceLocation
from .graph import ProgramGraph
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
    "KernelDependencyError",
    "KernelExecutionError",
    "MethodCall",
    "MissingOutputError",
    "ProgramGraph",
    "SourceInfo",
    "SourceLocation",
    "SourceTarget",
    "SourceUnavailableError",
    "TargetInfo",
    "TargetKind",
    "analyze_callable",
    "collect_accesses_from_source",
    "extract_source",
    "get_source_info",
    "normalize_target",
    "parse_source",
]
