"""Tests for public kernel contracts and declaration validation."""

from __future__ import annotations

import pytest

from dryml.code import AnalysisKernel, KernelCall, SourceTarget, TraversalKernel, analyze
from dryml.code.errors import InvalidKernelError, KernelDependencyError


class CountNodes(TraversalKernel[None, int, int]):
    """Count graph nodes through the standard traversal template."""

    input_type = type(None)
    output_type = int

    def begin(self, value: None, context: object) -> int:
        """Start the count at zero."""

        return 0

    def visit(self, node: object, state: int, context: object) -> int:
        """Count one visited graph node."""

        return state + 1

    def finish(self, state: int, context: object) -> int:
        """Return the completed count."""

        return state


class InvalidTypes(AnalysisKernel[None, None]):
    """Declare an invalid parameterized runtime input type."""

    input_type = list[int]
    output_type = type(None)

    def run(self, graph: object, value: None, context: object) -> None:
        """Provide an unreachable implementation."""


class EmptyTargetKinds(AnalysisKernel[None, None]):
    """Declare an invalid empty target-kind set."""

    input_type = type(None)
    output_type = type(None)
    target_kinds = frozenset()

    def run(self, graph: object, value: None, context: object) -> None:
        """Provide an unreachable implementation."""


class TraceProducer(AnalysisKernel[None, None]):
    """Declare a trace-only producer for dependency validation."""

    input_type = type(None)
    output_type = type(None)
    mode = "trace"

    def run(self, graph: object, value: None, context: object) -> None:
        """Provide an unreachable implementation."""


class StaticDependent(AnalysisKernel[None, None]):
    """Declare an illegal static dependency on a trace kernel."""

    input_type = type(None)
    output_type = type(None)
    requires = (TraceProducer,)

    def run(self, graph: object, value: None, context: object) -> None:
        """Provide an unreachable implementation."""


class MissingProducer(AnalysisKernel[None, None]):
    """Provide a missing required producer declaration."""

    input_type = type(None)
    output_type = type(None)
    requires = (CountNodes,)

    def run(self, graph: object, value: None, context: object) -> None:
        """Provide an unreachable implementation."""


def test_traversal_kernel_uses_the_unfused_standard_template() -> None:
    """Traversal kernels visit canonical graph nodes exactly once in order."""

    result = analyze(
        SourceTarget("def subject():\n    return value\n", name="subject"),
        (KernelCall(CountNodes(), None),),
    )

    assert result.require(CountNodes) == len(result.graph.nodes)


@pytest.mark.parametrize(
    ("call", "error_type"),
    [
        (KernelCall(InvalidTypes(), None), InvalidKernelError),
        (KernelCall(EmptyTargetKinds(), None), InvalidKernelError),
        (KernelCall(MissingProducer(), None), KernelDependencyError),
        (KernelCall(TraceProducer(), None), InvalidKernelError),
        (KernelCall(StaticDependent(), None), KernelDependencyError),
    ],
)
def test_invalid_declarations_fail_before_target_resolution(
    call: KernelCall[object, object], error_type: type[Exception],
) -> None:
    """Invalid declarations reject a request before target normalization runs."""

    with pytest.raises(error_type):
        analyze(object(), (call,))


def test_duplicate_and_cyclic_kernel_types_fail_before_execution() -> None:
    """Duplicate producers and cycles are rejected before any consumer runs."""

    ran: list[str] = []

    class First(AnalysisKernel[None, int]):
        input_type = type(None)
        output_type = int

        def run(self, graph: object, value: None, context: object) -> int:
            ran.append("first")
            return 1

    class Second(AnalysisKernel[None, int]):
        input_type = type(None)
        output_type = int
        requires = (First,)

        def run(self, graph: object, value: None, context: object) -> int:
            ran.append("second")
            return 2

    First.requires = (Second,)
    with pytest.raises(KernelDependencyError):
        analyze(object(), (KernelCall(First(), None), KernelCall(Second(), None)))
    with pytest.raises(InvalidKernelError):
        analyze(object(), (KernelCall(CountNodes(), None), KernelCall(CountNodes(), None)))

    assert ran == []
