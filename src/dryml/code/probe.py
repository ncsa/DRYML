"""In-process static convenience entry point for generic code analysis."""

from __future__ import annotations

from typing import Any, Iterable

from .analysis import AnalysisResult, analyze
from .kernels import KernelCall
from .targets import CodeTargetInput


def probe(target: CodeTargetInput, calls: Iterable[KernelCall[Any, Any]]) -> AnalysisResult:
    """Run the ordinary static analysis operation in the current process.

    Args:
        target: Supported static code target or target wrapper.
        calls: Per-request kernel calls, materialized exactly once by analysis.

    Returns:
        The same structured result and semantics as :func:`dryml.code.analyze`.

    Raises:
        CodeAnalysisError: Propagates static declaration, target, source, graph,
            dependency, and input-admission failures from :func:`analyze`.

    Side Effects:
        May read source or perform an explicit target import. It launches no
        subprocess, transport, environment, runtime, or tracing machinery.
    """

    return analyze(target, calls)


__all__ = ["probe"]
