"""Tests for stable static analysis scheduling and partial results."""

from __future__ import annotations

import gc
import weakref

import pytest

from dryml.code import (
    AnalysisKernel,
    CodeFact,
    CodeFacts,
    KernelCall,
    MissingOutputError,
    SourceTarget,
    analyze,
)
from dryml.code.errors import InvalidKernelError


_RUNS: list[str] = []


class Seed(AnalysisKernel[int, list[object]]):
    """Produce one mutable shared artifact."""

    input_type = int
    output_type = list

    def run(self, graph: object, value: int, context: object) -> list[object]:
        """Produce one list that dependents can identify."""

        _RUNS.append("seed")
        return [value]


class Left(AnalysisKernel[None, int]):
    """Read the shared producer artifact."""

    input_type = type(None)
    output_type = int
    requires = (Seed,)

    def run(self, graph: object, value: None, context: object) -> int:
        """Return the identity of the required shared artifact."""

        _RUNS.append("left")
        return id(context.require(Seed))  # type: ignore[union-attr]


class Right(AnalysisKernel[None, int]):
    """Read the shared producer artifact independently."""

    input_type = type(None)
    output_type = int
    requires = (Seed,)

    def run(self, graph: object, value: None, context: object) -> int:
        """Return the identity of the required shared artifact."""

        _RUNS.append("right")
        return id(context.require(Seed))  # type: ignore[union-attr]


class Join(AnalysisKernel[None, tuple[int, int]]):
    """Read both branches of the dependency diamond."""

    input_type = type(None)
    output_type = tuple
    requires = (Left, Right)

    def run(self, graph: object, value: None, context: object) -> tuple[int, int]:
        """Return both branch values."""

        _RUNS.append("join")
        return (context.require(Left), context.require(Right))  # type: ignore[union-attr]


class Fails(AnalysisKernel[None, int]):
    """Raise a secret-bearing error that must be redacted."""

    input_type = type(None)
    output_type = int

    def run(self, graph: object, value: None, context: object) -> int:
        """Fail without exposing the exception payload."""

        _RUNS.append("fails")
        raise RuntimeError("/private/path sentinel-secret")


class Skipped(AnalysisKernel[None, int]):
    """Depend directly on a failed kernel."""

    input_type = type(None)
    output_type = int
    requires = (Fails,)

    def run(self, graph: object, value: None, context: object) -> int:
        """Provide an implementation that must not execute."""

        _RUNS.append("skipped")
        return 1


class TransitiveSkip(AnalysisKernel[None, int]):
    """Depend transitively on a failed kernel."""

    input_type = type(None)
    output_type = int
    requires = (Skipped,)

    def run(self, graph: object, value: None, context: object) -> int:
        """Provide an implementation that must not execute."""

        _RUNS.append("transitive")
        return 1


class Independent(AnalysisKernel[None, int]):
    """Continue independently after another kernel fails."""

    input_type = type(None)
    output_type = int

    def run(self, graph: object, value: None, context: object) -> int:
        """Produce an independent value."""

        _RUNS.append("independent")
        return 7


class WrongOutput(AnalysisKernel[None, int]):
    """Produce a value outside its nominal declaration."""

    input_type = type(None)
    output_type = int

    def run(self, graph: object, value: None, context: object) -> int:
        """Return the wrong runtime value intentionally."""

        return "wrong"  # type: ignore[return-value]


class NoneOutput(AnalysisKernel[None, None]):
    """Produce a successful nominal ``None`` value."""

    input_type = type(None)
    output_type = type(None)

    def run(self, graph: object, value: None, context: object) -> None:
        """Return the declared ``None`` value."""


class NoneAsObject(AnalysisKernel[None, object]):
    """Return invalid ``None`` through a broad declaration."""

    input_type = type(None)
    output_type = object

    def run(self, graph: object, value: None, context: object) -> object:
        """Return a value only the exact None declaration may admit."""


class FactAsObject(AnalysisKernel[None, object]):
    """Return a fact-shaped value through a non-fact declaration."""

    input_type = type(None)
    output_type = object

    def run(self, graph: object, value: None, context: object) -> object:
        """Return an exact fact that remains an ordinary opaque output."""

        return CodeFact("hidden", 1)


class SingleFact(AnalysisKernel[None, CodeFact]):
    """Produce one exact fact wrapper."""

    input_type = type(None)
    output_type = CodeFact

    def run(self, graph: object, value: None, context: object) -> CodeFact:
        """Produce one generic fact."""

        return CodeFact("single", "value")


class ManyFacts(AnalysisKernel[None, CodeFacts]):
    """Produce exact aggregate fact wrappers."""

    input_type = type(None)
    output_type = CodeFacts

    def run(self, graph: object, value: None, context: object) -> CodeFacts:
        """Produce two generic facts."""

        return CodeFacts((CodeFact("many", 1), CodeFact("many", 2)))


class OrdinaryTuple(AnalysisKernel[None, tuple[CodeFact]]):
    """Produce a generic tuple that must not become facts."""

    input_type = type(None)
    output_type = tuple

    def run(self, graph: object, value: None, context: object) -> tuple[CodeFact]:
        """Return an ordinary container of facts."""

        return (CodeFact("ordinary", 1),)


class ContextFacts(AnalysisKernel[None, int]):
    """Observe only facts from its declared dependency closure."""

    input_type = type(None)
    output_type = int
    requires = (SingleFact, ManyFacts)

    def run(self, graph: object, value: None, context: object) -> int:
        """Count declared dependency facts."""

        return len(context.facts())  # type: ignore[union-attr]


class FunctionMetadata(AnalysisKernel[None, str]):
    """Read metadata-only graph provenance without retaining the target."""

    input_type = type(None)
    output_type = str

    def run(self, graph: object, value: None, context: object) -> str:
        """Return the normalized target kind."""

        return graph.target.kind  # type: ignore[union-attr]


class Receiver:
    """Provide bound-method and callable-instance target fixtures."""

    def method(self) -> None:
        """Provide a normal bound Python method."""

    def __call__(self) -> None:
        """Provide a normal callable instance target."""


def _target() -> SourceTarget:
    """Provide a source-only target for deterministic analysis tests."""

    return SourceTarget("def subject():\n    return value\n", name="subject", filename="subject.py")


def test_analysis_executes_diamond_once_in_stable_order() -> None:
    """A shared artifact runs once while result order remains submission order."""

    _RUNS.clear()
    result = analyze(_target(), (KernelCall(Join(), None), KernelCall(Right(), None), KernelCall(Seed(), 3), KernelCall(Left(), None)))

    assert _RUNS == ["seed", "right", "left", "join"]
    assert tuple(outcome.kernel for outcome in result.outcomes) == (Join, Right, Seed, Left)
    assert result.require(Join)[0] == result.require(Join)[1]
    assert result.base_graph is result.graph


def test_failure_skips_dependents_and_continues_independent_work() -> None:
    """Failures are structured, redacted, and do not stop independent kernels."""

    _RUNS.clear()
    result = analyze(_target(), (KernelCall(TransitiveSkip(), None), KernelCall(Independent(), None), KernelCall(Skipped(), None), KernelCall(Fails(), None)))

    assert _RUNS == ["independent", "fails"]
    assert tuple(outcome.status for outcome in result.outcomes) == ("skipped", "succeeded", "skipped", "failed")
    assert result.outcomes[0].skipped_for == (Skipped,)
    assert result.outcomes[2].skipped_for == (Fails,)
    assert not result.complete
    assert result.require(Independent) == 7
    with pytest.raises(MissingOutputError):
        result.require(Fails)
    diagnostic = result.outcomes[-1].diagnostics[0]
    assert diagnostic.code == "kernel.execution"
    assert "secret" not in diagnostic.message
    assert "/" not in diagnostic.message


def test_output_validation_and_missing_output_semantics() -> None:
    """Wrong values fail while a declared successful ``None`` remains available."""

    result = analyze(_target(), (KernelCall(WrongOutput(), None), KernelCall(NoneOutput(), None), KernelCall(NoneAsObject(), None)))

    assert result.outcomes[0].status == "failed"
    assert result.outcomes[0].diagnostics[0].code == "kernel.output_type"
    assert result.outcomes[2].status == "failed"
    assert result.output(WrongOutput) is None
    assert result.output(NoneOutput) is None
    assert result.require(NoneOutput) is None
    with pytest.raises(MissingOutputError):
        result.require(WrongOutput)


def test_fact_aggregation_is_exact_and_dependency_scoped() -> None:
    """Only exact fact wrappers aggregate in deterministic producer order."""

    result = analyze(
        _target(),
        (KernelCall(ContextFacts(), None), KernelCall(OrdinaryTuple(), None), KernelCall(ManyFacts(), None), KernelCall(SingleFact(), None), KernelCall(FactAsObject(), None)),
    )

    assert result.require(ContextFacts) == 3
    assert tuple(record.fact.kind for record in result.facts) == ("many", "many", "single")
    assert tuple(record.producer for record in result.facts) == (ManyFacts, ManyFacts, SingleFact)
    assert tuple(record.graph_digest for record in result.facts) == (result.graph.digest,) * 3
    assert tuple(record.origin for record in result.facts) == ("static",) * 3


def test_target_and_input_rejection_happen_before_kernel_execution() -> None:
    """Target-kind and input admission run before any submitted consumer code."""

    class FunctionOnly(AnalysisKernel[None, int]):
        input_type = type(None)
        output_type = int
        target_kinds = frozenset({"function"})

        def run(self, graph: object, value: None, context: object) -> int:
            raise AssertionError("kernel executed")

    with pytest.raises(InvalidKernelError):
        analyze(SourceTarget("def subject():\n    return 1\n", name="subject"), (KernelCall(FunctionOnly(), None),))
    with pytest.raises(InvalidKernelError):
        analyze(_target(), (KernelCall(Independent(), "wrong"),))


def test_calls_are_materialized_once_and_results_do_not_retain_live_targets() -> None:
    """One-shot call iterables are accepted and framework results keep no target."""

    yielded = 0

    def calls() -> object:
        nonlocal yielded
        yielded += 1
        yield KernelCall(FunctionMetadata(), None)

    def local_target() -> None:
        return None

    reference = weakref.ref(local_target)
    result = analyze(local_target, calls())
    del local_target
    gc.collect()

    assert yielded == 1
    assert result.require(FunctionMetadata) == "function"
    assert reference() is None


def test_bound_and_callable_targets_expose_metadata_without_receiver_retention() -> None:
    """Graphs retain normalized receiver provenance but not live receivers."""

    receiver = Receiver()
    reference = weakref.ref(receiver)
    bound = analyze(receiver.method, (KernelCall(FunctionMetadata(), None),))
    callable_result = analyze(receiver, (KernelCall(FunctionMetadata(), None),))
    del receiver
    gc.collect()

    assert bound.require(FunctionMetadata) == "bound_method"
    assert callable_result.require(FunctionMetadata) == "callable_instance"
    assert bound.target.owner_qualname == "Receiver"
    assert callable_result.target.owner_qualname == "Receiver"
    assert reference() is None
