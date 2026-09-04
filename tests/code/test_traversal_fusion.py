"""Differential tests for optional conservative traversal fusion."""

from __future__ import annotations

from dryml.code import AnalysisKernel, CodeFact, CodeFacts, KernelCall, SourceTarget, TraversalKernel
from dryml.code.algorithms import LexicalDependencyKernel
from dryml.code.analysis import _analyze
from dryml.code.graph import ProgramGraph, ProgramNode
from dryml.code.kernels import KernelContext


_EVENTS: list[str] = []


class NodeFacts(TraversalKernel[None, int, CodeFacts]):
    """Return one exact fact describing each canonical node traversal."""

    input_type = type(None)
    output_type = CodeFacts
    fusion_safe = True

    def begin(self, value: None, context: KernelContext) -> int:
        """Start a private canonical-node count."""

        _EVENTS.append("facts:begin")
        return 0

    def visit(self, node: ProgramNode, state: int, context: KernelContext) -> int:
        """Record one visited node and advance the private count."""

        _EVENTS.append(f"facts:{node.id}")
        return state + 1

    def finish(self, state: int, context: KernelContext) -> CodeFacts:
        """Publish an immutable count fact after traversal completes."""

        _EVENTS.append("facts:finish")
        return CodeFacts((CodeFact("node-count", state),))


class LoggingLexicalDependencyKernel(LexicalDependencyKernel):
    """Expose the inherited lexical traversal's callback order for this test."""

    def begin(self, value: None, context: KernelContext) -> object:
        """Record the lexical traversal start before delegating its private state."""

        _EVENTS.append("lexical:begin")
        return super().begin(value, context)

    def visit(self, node: ProgramNode, state: object, context: KernelContext) -> object:
        """Record one lexical node callback before preserving lexical semantics."""

        _EVENTS.append(f"lexical:{node.id}")
        return super().visit(node, state, context)  # type: ignore[arg-type]

    def finish(self, state: object, context: KernelContext) -> object:
        """Record lexical completion while returning its ordinary public output."""

        _EVENTS.append("lexical:finish")
        return super().finish(state, context)  # type: ignore[arg-type]


class DependencyFact(AnalysisKernel[None, CodeFact]):
    """Provide an immutable dependency artifact for a fused traversal."""

    input_type = type(None)
    output_type = CodeFact

    def run(self, graph: ProgramGraph, value: None, context: KernelContext) -> CodeFact:
        """Return framework-owned immutable evidence available to dependents."""

        return CodeFact("dependency", graph.digest)


class DependentNodeFacts(TraversalKernel[None, int, CodeFacts]):
    """Traverse after reading the successful immutable direct dependency."""

    input_type = type(None)
    output_type = CodeFacts
    requires = (DependencyFact,)
    fusion_safe = True

    def begin(self, value: None, context: KernelContext) -> int:
        """Read the direct immutable artifact before recording nodes."""

        assert context.require(DependencyFact).kind == "dependency"
        _EVENTS.append("dependent:begin")
        return 0

    def visit(self, node: ProgramNode, state: int, context: KernelContext) -> int:
        """Record one node in this traversal's private state."""

        _EVENTS.append(f"dependent:{node.id}")
        return state + 1

    def finish(self, state: int, context: KernelContext) -> CodeFacts:
        """Return the private traversal count as an exact fact wrapper."""

        _EVENTS.append("dependent:finish")
        return CodeFacts((CodeFact("dependent-node-count", state),))


class FailingTraversal(TraversalKernel[None, None, int]):
    """Fail during one visit without affecting a compatible fused peer."""

    input_type = type(None)
    output_type = int
    fusion_safe = True

    def begin(self, value: None, context: KernelContext) -> None:
        """Start this intentionally failing private traversal."""

        _EVENTS.append("failing:begin")

    def visit(self, node: ProgramNode, state: None, context: KernelContext) -> None:
        """Fail on the first callback without exposing error content."""

        _EVENTS.append(f"failing:{node.id}")
        raise RuntimeError("private failure")

    def finish(self, state: None, context: KernelContext) -> int:
        """Provide an unreachable nominal output."""

        raise AssertionError("failed traversal must not finish")


class FailingDependent(AnalysisKernel[None, int]):
    """Require the failing peer so normal dependency skipping is observable."""

    input_type = type(None)
    output_type = int
    requires = (FailingTraversal,)

    def run(self, graph: ProgramGraph, value: None, context: KernelContext) -> int:
        """Provide an implementation that must not run after peer failure."""

        raise AssertionError("failed peer dependent must be skipped")


class UnsafeTraversal(NodeFacts):
    """Inherit the template but explicitly decline traversal fusion."""

    fusion_safe = False


class UnflaggedTraversal(TraversalKernel[None, int, int]):
    """Use the standard template without declaring the fusion-safe contract."""

    input_type = type(None)
    output_type = int

    def begin(self, value: None, context: KernelContext) -> int:
        """Start an ordinary private node count."""

        return 0

    def visit(self, node: ProgramNode, state: int, context: KernelContext) -> int:
        """Advance the ordinary private node count."""

        return state + 1

    def finish(self, state: int, context: KernelContext) -> int:
        """Return the completed count."""

        return state


class OrdinaryKernel(AnalysisKernel[None, int]):
    """Provide a non-traversal peer that must use the normal run path."""

    input_type = type(None)
    output_type = int

    def run(self, graph: ProgramGraph, value: None, context: KernelContext) -> int:
        """Return the graph size without entering a visitor callback sequence."""

        _EVENTS.append("ordinary:run")
        return len(graph.nodes)


class StatefulTraversal(NodeFacts):
    """Carry mutable instance state that makes fusion conservatively unsafe."""

    def __init__(self) -> None:
        """Create state outside the callback-local traversal contract."""

        self.seen = 0

    def visit(self, node: ProgramNode, state: int, context: KernelContext) -> int:
        """Update instance state while preserving the inherited result behavior."""

        self.seen += 1
        return super().visit(node, state, context)


class RunOverrideTraversal(NodeFacts):
    """Override the template, making the traversal ineligible for fusion."""

    def run(self, graph: ProgramGraph, value: None, context: KernelContext) -> CodeFacts:
        """Use the inherited traversal mechanics through a nonconforming override."""

        _EVENTS.append("override:run")
        return super().run(graph, value, context)


class IntermediateRunOverride(NodeFacts):
    """Provide an intermediate inherited override for decline coverage."""

    def run(self, graph: ProgramGraph, value: None, context: KernelContext) -> CodeFacts:
        """Delegate through a nonconforming inherited template override."""

        _EVENTS.append("intermediate:run")
        return super().run(graph, value, context)


class IndirectRunOverrideTraversal(IntermediateRunOverride):
    """Inherit a nonconforming run override without defining one directly."""


class TraversalDependent(NodeFacts):
    """Create a dependency path that must prevent a shared traversal batch."""

    requires = (NodeFacts,)

    def begin(self, value: None, context: KernelContext) -> int:
        """Read the predecessor before beginning this private traversal."""

        assert context.require(NodeFacts).values[0].kind == "node-count"
        return super().begin(value, context)


def _target() -> SourceTarget:
    """Provide deterministic source with several graph nodes for traversal tests."""

    return SourceTarget("def subject(value):\n    return external(value)\n", name="subject")


def _run(*calls: KernelCall[object, object], fused: bool) -> tuple[object, tuple[str, ...]]:
    """Run one fixture request and retain its framework-visible callback trace."""

    _EVENTS.clear()
    result = _analyze(_target(), calls, fuse_traversals=fused)
    return result, tuple(_EVENTS)


def _kernel_events(events: tuple[str, ...], prefix: str) -> tuple[str, ...]:
    """Return the callback subsequence belonging to one fixture kernel."""

    return tuple(event for event in events if event.startswith(prefix))


def test_eligible_lexical_and_fact_traversals_share_one_canonical_walk() -> None:
    """Eligible peers interleave each node while preserving unfused artifacts."""

    calls = (
        KernelCall(LexicalDependencyKernel(), None),
        KernelCall(LoggingLexicalDependencyKernel(), None),
        KernelCall(NodeFacts(), None),
    )
    fused, events = _run(*calls, fused=True)
    unfused, _ = _run(*calls, fused=False)

    assert fused.outcomes == unfused.outcomes
    assert fused.facts == unfused.facts
    assert fused.diagnostics == unfused.diagnostics
    node_ids = tuple(node.id for node in fused.graph.nodes)
    assert events == (
        "lexical:begin",
        "facts:begin",
        *(item for node_id in node_ids for item in (f"lexical:{node_id}", f"facts:{node_id}")),
        "lexical:finish",
        "facts:finish",
    )


def test_fusion_preserves_dependency_facts_and_isolates_visitor_failure() -> None:
    """Fused peers retain private contexts, facts, and failure attribution."""

    calls = (
        KernelCall(DependencyFact(), None),
        KernelCall(DependentNodeFacts(), None),
        KernelCall(NodeFacts(), None),
        KernelCall(FailingTraversal(), None),
        KernelCall(FailingDependent(), None),
    )
    fused, fused_events = _run(*calls, fused=True)
    unfused, unfused_events = _run(*calls, fused=False)

    assert fused.outcomes == unfused.outcomes
    assert fused.facts == unfused.facts
    assert fused.diagnostics == unfused.diagnostics
    assert tuple(outcome.status for outcome in fused.outcomes) == ("succeeded", "succeeded", "succeeded", "failed", "skipped")
    assert fused.outcomes[-1].skipped_for == (FailingTraversal,)
    assert _kernel_events(fused_events, "dependent:") == _kernel_events(unfused_events, "dependent:")
    assert _kernel_events(fused_events, "facts:") == _kernel_events(unfused_events, "facts:")


def test_nonconforming_or_dependent_traversals_decline_fusion() -> None:
    """Unsafe, overridden, ordinary, and dependency-path kernels stay unfused."""

    calls = (
        KernelCall(TraversalDependent(), None),
        KernelCall(UnsafeTraversal(), None),
        KernelCall(UnflaggedTraversal(), None),
        KernelCall(OrdinaryKernel(), None),
        KernelCall(StatefulTraversal(), None),
        KernelCall(RunOverrideTraversal(), None),
        KernelCall(IndirectRunOverrideTraversal(), None),
        KernelCall(NodeFacts(), None),
    )
    fused, events = _run(*calls, fused=True)
    unfused, unfused_events = _run(*calls, fused=False)

    assert fused.outcomes == unfused.outcomes
    assert fused.facts == unfused.facts
    assert fused.diagnostics == unfused.diagnostics
    assert "override:run" in events
    assert "intermediate:run" in events
    assert "ordinary:run" in events
    assert events == unfused_events
